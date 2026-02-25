use anyhow::{Context, Result};
use fastembed::{RerankInitOptions, RerankerModel, TextRerank};
use ort::ep;

/// Cross-encoder reranker using fastembed-rs
pub struct FastembedReranker {
    model: TextRerank,
}

impl FastembedReranker {
    /// Create a new reranker with the specified model and hardware acceleration
    pub fn new(model: RerankerModel, show_download_progress: bool) -> Result<Self> {
        let execution_providers = Self::configure_execution_providers();

        let options = RerankInitOptions::new(model)
            .with_show_download_progress(show_download_progress)
            .with_execution_providers(execution_providers);

        let model =
            TextRerank::try_new(options).context("Failed to initialize fastembed reranker")?;

        Ok(Self { model })
    }

    /// Configure execution providers for hardware acceleration.
    ///
    /// CoreML is gated behind `SAGE_COREML=1` (same as the embedder) because
    /// CoreML allocates ~10 GB of persistent ANE/compilation buffers on first
    /// inference, which OOMs machines when the daemon is running. CPU-only
    /// execution uses ~50–200 MB and is safe for always-on daemons.
    #[allow(clippy::vec_init_then_push)] // conditional pushes based on platform/feature flags
    fn configure_execution_providers() -> Vec<ort::ep::ExecutionProviderDispatch> {
        let mut providers = Vec::new();

        #[cfg(target_os = "macos")]
        if std::env::var("SAGE_COREML").is_ok() {
            tracing::debug!("Reranker: CoreML execution provider enabled via SAGE_COREML");
            providers.push(ep::CoreML::default().build());
        } else {
            tracing::debug!(
                "Reranker: CoreML disabled by default (set SAGE_COREML=1 to enable). \
                 CPU-only reranking uses ~50-200 MB vs ~10 GB with CoreML."
            );
        }

        #[cfg(feature = "cuda")]
        {
            providers.push(ep::CUDA::default().build());
        }

        providers.push(ep::CPU::default().build());
        providers
    }

    /// Rerank documents by relevance to the query.
    /// Returns (original_index, score) pairs sorted by score descending.
    pub fn rerank(
        &mut self,
        query: &str,
        documents: &[&str],
        top_k: usize,
    ) -> Result<Vec<(usize, f32)>> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }

        // Rerank with fastembed - return_documents=false, use default batch size
        let results = self
            .model
            .rerank(query, documents, false, None)
            .context("Fastembed reranking failed")?;

        // Convert to (index, score) pairs and sort by score descending
        let mut scored: Vec<(usize, f32)> = results.iter().map(|r| (r.index, r.score)).collect();

        // Sort by score descending
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Return top-k
        scored.truncate(top_k);
        Ok(scored)
    }
}
