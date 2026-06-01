//! Generic ONNX cross-encoder reranker.
//!
//! fastembed 5.9's `RerankerModel` enum cannot express newer checkpoints
//! (e.g. Qwen3-Reranker-0.6B), so this module loads any HuggingFace
//! cross-encoder's ONNX model + `tokenizer.json` directly through `ort` +
//! `tokenizers`, mirroring how `embedding/colbert.rs` pins the CPU execution
//! provider. It supports two ways to turn model output into a relevance score:
//!
//! - [`ScoreStrategy::ClassifierLogit`] — sequence-classification head
//!   (bge-style): the model emits a single relevance logit; the score is that
//!   logit.
//! - [`ScoreStrategy::YesNoLogit`] — generative reranker (Qwen3-Reranker-style):
//!   the model emits vocabulary logits; the score is the softmax-normalized
//!   probability mass on the "yes" token vs the "no" token at the final
//!   position, per the values carried by S8's `ModelRegistry` reranker spec
//!   (recorded in `docs/superpowers/plans/2026-05-31-research-notes.md`).
//!
//! All concrete model ids, prompt strings, and token ids come from the
//! reranker spec (`crate::model::RerankerSpec` via `RerankerChoice::from_spec`)
//! — nothing repo-specific is hardcoded here.

use anyhow::{Context, Result};
use ort::session::Session;
use ort::value::Tensor;
use parking_lot::Mutex;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use tokenizers::Tokenizer;

/// Prompt wrapper for a generative (yes/no) reranker. Strings come from the
/// model's reranker spec (S8 registry / recorded spike), never hardcoded per
/// corpus.
#[derive(Debug, Clone)]
pub struct PromptTemplate {
    /// Text emitted before the query (e.g. the instruction header).
    pub prefix: String,
    /// Text emitted between the query and the document.
    pub middle: String,
    /// Text emitted after the document (terminates the prompt so the next-token
    /// logit is the relevance judgment).
    pub suffix: String,
}

impl PromptTemplate {
    /// Render the full prompt string for a (query, document) pair.
    #[must_use]
    pub fn render(&self, query: &str, document: &str) -> String {
        format!(
            "{}{}{}{}{}",
            self.prefix, query, self.middle, document, self.suffix
        )
    }
}

/// How to convert model output logits into a single relevance score.
#[derive(Debug, Clone)]
pub enum ScoreStrategy {
    /// Sequence-classification head: a single relevance logit per pair.
    ClassifierLogit,
    /// Generative reranker: compare the "yes" vs "no" token logits at the
    /// final position. `yes_id`/`no_id` come from the reranker spec.
    YesNoLogit {
        yes_id: usize,
        no_id: usize,
        prompt: PromptTemplate,
    },
}

/// Score for a classifier-head model: the relevance logit is the last element
/// of the flattened output (shape `[batch=1, num_labels]`, num_labels==1 for
/// bge rerankers). Returns the raw logit (monotonic in relevance; the caller
/// only needs ordering).
pub(crate) fn classifier_score_from_logits(logits: &[f32]) -> Result<f32> {
    logits
        .last()
        .copied()
        .context("classifier reranker produced an empty logits tensor")
}

/// Score for a yes/no generative reranker: take the logits at the final
/// sequence position (`final_pos_logits`, length == vocab size), then return
/// the softmax probability of "yes" over {yes, no}:
/// `exp(l_yes) / (exp(l_yes) + exp(l_no))`. Computed in a numerically stable
/// way by subtracting the max of the two logits.
pub(crate) fn yes_no_score_from_logits(
    final_pos_logits: &[f32],
    yes_id: usize,
    no_id: usize,
) -> Result<f32> {
    let l_yes = *final_pos_logits.get(yes_id).with_context(|| {
        format!("yes_id {yes_id} out of range (vocab {})", final_pos_logits.len())
    })?;
    let l_no = *final_pos_logits.get(no_id).with_context(|| {
        format!("no_id {no_id} out of range (vocab {})", final_pos_logits.len())
    })?;
    let m = l_yes.max(l_no);
    let e_yes = (l_yes - m).exp();
    let e_no = (l_no - m).exp();
    Ok(e_yes / (e_yes + e_no))
}

/// A generic ONNX cross-encoder reranker. Thread-safe: the inner `ort::Session`
/// is built lazily on first scoring and guarded by a `Mutex` (single-threaded
/// inference, like `ColbertEmbedder`). Construction is cheap.
pub struct OnnxReranker {
    /// Directory containing the ONNX model file and `tokenizer.json`.
    model_dir: PathBuf,
    /// ONNX session filename within `model_dir` (e.g. `model_int8.onnx` for the
    /// bge classifier, `model.onnx` for the fp16 Qwen3 export). Passed in as
    /// DATA from the reranker spec — never hardcoded, since exports differ.
    session_file: String,
    /// Loaded eagerly (small, validates the dir): the HF tokenizer.
    tokenizer: Tokenizer,
    /// How to turn model output into a relevance score.
    strategy: ScoreStrategy,
    /// ORT intra-op threads (from `SEMANTEX_ORT_THREADS`, default 4 — set by caller).
    threads: usize,
    /// Opt into CoreML on macOS (gated by `SEMANTEX_COREML`).
    use_coreml: bool,
    /// Lazily-built ONNX session.
    session: OnceLock<Mutex<Session>>,
    /// Serializes the build path so concurrent first-callers don't build twice.
    build_lock: std::sync::Mutex<()>,
}

impl OnnxReranker {
    /// Construct a reranker from a model directory. Reads `tokenizer.json`
    /// eagerly (cheap, and validates the directory); defers the heavy ONNX
    /// session (`session_file`) to the first `score_pair`/`rerank` call.
    ///
    /// `session_file` is the ONNX model leaf within `model_dir`. It is carried
    /// as data because exports differ (bge ships `model_int8.onnx`; the hosted
    /// Qwen3-Reranker export is fp16 `model.onnx`).
    ///
    /// # Errors
    /// Errors if the directory or `tokenizer.json` is missing/unreadable.
    pub fn new(
        model_dir: &Path,
        session_file: &str,
        strategy: ScoreStrategy,
        threads: usize,
        use_coreml: bool,
    ) -> Result<Self> {
        let tok_path = model_dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tok_path)
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer {}: {e}", tok_path.display()))?;
        Ok(Self {
            model_dir: model_dir.to_path_buf(),
            session_file: session_file.to_string(),
            tokenizer,
            strategy,
            threads: threads.max(1),
            use_coreml,
            session: OnceLock::new(),
            build_lock: std::sync::Mutex::new(()),
        })
    }

    /// Execution providers: CoreML iff `use_coreml` on macOS, CUDA under the
    /// `cuda` feature, then always CPU. Mirrors `fastembed_reranker.rs` /
    /// `colbert.rs` so reranking never allocates the ~10 GB CoreML buffers
    /// unless explicitly opted in.
    #[allow(clippy::vec_init_then_push)] // conditional pushes based on platform/feature flags
    fn execution_providers(&self) -> Vec<ort::ep::ExecutionProviderDispatch> {
        let mut providers = Vec::new();
        #[cfg(target_os = "macos")]
        if self.use_coreml {
            providers.push(ort::ep::CoreML::default().build());
        }
        #[cfg(not(target_os = "macos"))]
        let _ = self.use_coreml;
        #[cfg(feature = "cuda")]
        {
            providers.push(ort::ep::CUDA::default().build());
        }
        providers.push(ort::ep::CPU::default().build());
        providers
    }

    /// Build the ONNX session from `model_dir/session_file`. Uses the verbatim
    /// `ort 2.0.0-rc.11` builder API.
    fn build_session(&self) -> Result<Session> {
        let model_path = self.model_dir.join(&self.session_file);
        let session = Session::builder()
            .context("ort Session::builder failed")?
            .with_execution_providers(self.execution_providers())
            .context("failed to set execution providers")?
            .with_intra_threads(self.threads)
            .context("failed to set intra-op threads")?
            .commit_from_file(&model_path)
            .with_context(|| format!("failed to load ONNX model {}", model_path.display()))?;
        Ok(session)
    }

    /// Get the session, building it once on first call (double-checked locking,
    /// same discipline as `ColbertEmbedder::encoder`).
    fn session(&self) -> Result<&Mutex<Session>> {
        if let Some(s) = self.session.get() {
            return Ok(s);
        }
        let _guard = self
            .build_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(s) = self.session.get() {
            return Ok(s);
        }
        let built = self.build_session()?;
        let _ = self.session.set(Mutex::new(built));
        Ok(self.session.get().expect("session set above"))
    }

    /// Tokenize one (query, document) pair into `(input_ids, attention_mask)`
    /// as `i64` vectors. For the classifier strategy the model sees the query
    /// and document as a sentence pair joined by the tokenizer's separator; for
    /// the yes/no strategy it sees the rendered prompt as a single sequence.
    fn encode_pair(&self, query: &str, document: &str) -> Result<(Vec<i64>, Vec<i64>)> {
        match &self.strategy {
            ScoreStrategy::ClassifierLogit => {
                // Cross-encoders score "query [SEP] document". The tokenizer's
                // post-processor inserts the separator when given a pair via
                // encode((a, b)); we pass the pair tuple.
                let enc = self
                    .tokenizer
                    .encode((query, document), true)
                    .map_err(|e| anyhow::anyhow!("tokenizer.encode(pair) failed: {e}"))?;
                Ok(to_i64_pair(&enc))
            }
            ScoreStrategy::YesNoLogit { prompt, .. } => {
                let text = prompt.render(query, document);
                let enc = self
                    .tokenizer
                    .encode(text, true)
                    .map_err(|e| anyhow::anyhow!("tokenizer.encode failed: {e}"))?;
                Ok(to_i64_pair(&enc))
            }
        }
    }

    /// Score a single (query, document) pair. Runs one ONNX forward pass.
    pub fn score_pair(&self, query: &str, document: &str) -> Result<f32> {
        let (ids, mask) = self.encode_pair(query, document)?;
        let seq = ids.len();
        let shape = vec![1_i64, seq as i64];
        let id_tensor = Tensor::from_array((shape.clone(), ids))
            .context("failed to build input_ids tensor")?;
        let mask_tensor = Tensor::from_array((shape.clone(), mask))
            .context("failed to build attention_mask tensor")?;

        // The generative (YesNoLogit) export (Qwen3-Reranker) additionally
        // requires a `position_ids` input (0..seq); the classifier export
        // (bge / XLM-RoBERTa) does NOT accept one. Build inputs accordingly so
        // each model gets exactly the tensors it declares.
        let mut inputs = ort::inputs![
            "input_ids" => id_tensor,
            "attention_mask" => mask_tensor,
        ];
        if matches!(self.strategy, ScoreStrategy::YesNoLogit { .. }) {
            let position_ids: Vec<i64> = (0..seq as i64).collect();
            let pos_tensor = Tensor::from_array((shape, position_ids))
                .context("failed to build position_ids tensor")?;
            inputs.push((
                std::borrow::Cow::Borrowed("position_ids"),
                pos_tensor.into(),
            ));
        }

        let session = self.session()?;
        let mut guard = session.lock();
        let outputs = guard
            .run(inputs)
            .context("ONNX reranker forward pass failed")?;

        // First (only) output is the logits tensor; index by position 0 to be
        // robust to the model's output name.
        let (out_shape, logits) = outputs[0]
            .try_extract_tensor::<f32>()
            .context("failed to extract f32 logits from reranker output")?;

        match &self.strategy {
            ScoreStrategy::ClassifierLogit => classifier_score_from_logits(logits),
            ScoreStrategy::YesNoLogit { yes_id, no_id, .. } => {
                // Generative output shape is [batch=1, seq, vocab]; the
                // judgment is the final position's vocab logits.
                let vocab = out_shape[out_shape.len() - 1] as usize;
                anyhow::ensure!(vocab > 0, "reranker output has zero-width vocab dim");
                let n = logits.len();
                anyhow::ensure!(n >= vocab, "reranker logits ({n}) shorter than vocab ({vocab})");
                let final_pos = &logits[n - vocab..];
                yes_no_score_from_logits(final_pos, *yes_id, *no_id)
            }
        }
    }

    /// Rerank `documents` by relevance to `query`. Returns `(original_index,
    /// score)` sorted by score descending, truncated to `top_k`.
    ///
    /// Identity pass-through (no session built, no inference) when
    /// `SEMANTEX_RERANKER` is not enabled — same contract as
    /// `FastembedReranker::rerank`, so the hybrid caller treats the stage as
    /// always-callable.
    ///
    /// Latency guard: the caller (`hybrid.rs`) already truncates the candidate
    /// list to `rerank_candidates` before this runs, so the per-document loop is
    /// bounded; a 0.6B model never runs on an unbounded list.
    pub fn rerank(
        &self,
        query: &str,
        documents: &[&str],
        top_k: usize,
    ) -> Result<Vec<(usize, f32)>> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }
        if !crate::search::fastembed_reranker::reranker_enabled() {
            tracing::debug!(
                "Reranker disabled (SEMANTEX_RERANKER!=on); returning identity ordering"
            );
            let n = documents.len().min(top_k);
            return Ok((0..n).map(|i| (i, 0.0_f32)).collect());
        }
        let mut scored: Vec<(usize, f32)> = Vec::with_capacity(documents.len());
        for (i, doc) in documents.iter().enumerate() {
            let s = self.score_pair(query, doc)?;
            scored.push((i, s));
        }
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        Ok(scored)
    }
}

/// Convert a tokenizers `Encoding` to `(input_ids, attention_mask)` as i64.
fn to_i64_pair(enc: &tokenizers::Encoding) -> (Vec<i64>, Vec<i64>) {
    let ids = enc.get_ids().iter().map(|&x| i64::from(x)).collect();
    let mask = enc
        .get_attention_mask()
        .iter()
        .map(|&x| i64::from(x))
        .collect();
    (ids, mask)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifier_score_reads_last_logit() {
        assert_eq!(classifier_score_from_logits(&[0.42]).unwrap(), 0.42);
        // Multi-label safety: still reads the last element.
        assert_eq!(classifier_score_from_logits(&[-1.0, 3.5]).unwrap(), 3.5);
    }

    #[test]
    fn classifier_score_errors_on_empty() {
        assert!(classifier_score_from_logits(&[]).is_err());
    }

    #[test]
    fn yes_no_score_is_monotonic_and_bounded() {
        // Vocab of 4; yes_id=2, no_id=3.
        let logits = [0.0, 0.0, 2.0, 1.0];
        let s = yes_no_score_from_logits(&logits, 2, 3).unwrap();
        // softmax(2,1) over {yes,no} = e^1/(e^1+e^0) = 0.7310586
        assert!((s - 0.731_058_6).abs() < 1e-5, "got {s}");
        // Strong yes -> near 1.0; strong no -> near 0.0.
        assert!(yes_no_score_from_logits(&[0.0, 0.0, 10.0, 0.0], 2, 3).unwrap() > 0.99);
        assert!(yes_no_score_from_logits(&[0.0, 0.0, 0.0, 10.0], 2, 3).unwrap() < 0.01);
    }

    #[test]
    fn yes_no_score_errors_on_oob_token() {
        assert!(yes_no_score_from_logits(&[0.1, 0.2], 5, 0).is_err());
    }

    #[test]
    fn prompt_template_renders_in_order() {
        let t = PromptTemplate {
            prefix: "<I>".into(),
            middle: "<M>".into(),
            suffix: "<S>".into(),
        };
        assert_eq!(t.render("Q", "D"), "<I>Q<M>D<S>");
    }

    /// Construction reads `tokenizer.json` eagerly (cheap, validates the dir)
    /// while the heavy ONNX session is deferred to the first scoring call. So an
    /// empty dir fails at construction (no tokenizer), and a missing dir fails
    /// too — mirroring ColbertEmbedder's "fail late for the model, fail early
    /// for the tokenizer/dir".
    #[test]
    fn new_rejects_missing_dir_but_is_lazy_for_files() {
        let tmp = tempfile::TempDir::new().unwrap();
        // Existing but empty dir: missing tokenizer.json -> construction fails.
        let r = OnnxReranker::new(
            tmp.path(),
            "model_int8.onnx",
            ScoreStrategy::ClassifierLogit,
            2,
            false,
        );
        assert!(r.is_err(), "missing tokenizer.json should fail at construction");

        let missing = std::path::Path::new("/no/such/reranker/dir");
        assert!(
            OnnxReranker::new(
                missing,
                "model_int8.onnx",
                ScoreStrategy::ClassifierLogit,
                2,
                false,
            )
            .is_err()
        );
    }

    /// rerank() must be an identity pass-through (no session built) when the
    /// master switch is off — exactly like FastembedReranker. We assert this
    /// without a model by checking the disabled branch returns ascending
    /// indices with score 0.0 and never touches the (absent) ONNX file.
    #[test]
    fn rerank_is_identity_when_disabled() {
        use crate::search::fastembed_reranker::ENV_ENABLE;
        with_env(&[(ENV_ENABLE, None)], || {
            let tmp = tempfile::TempDir::new().unwrap();
            // Write a minimal valid tokenizer.json so construction succeeds.
            std::fs::write(tmp.path().join("tokenizer.json"), MINIMAL_TOKENIZER_JSON).unwrap();
            let r = OnnxReranker::new(
                tmp.path(),
                "model_int8.onnx",
                ScoreStrategy::ClassifierLogit,
                2,
                false,
            )
            .expect("construct with tokenizer present");
            let docs = ["a", "b", "c"];
            let docs_ref: Vec<&str> = docs.iter().copied().collect();
            let out = r.rerank("q", &docs_ref, 2).expect("identity rerank");
            assert_eq!(out, vec![(0, 0.0_f32), (1, 0.0_f32)]);
        });
    }

    /// Smallest tokenizer.json the `tokenizers` crate will load: a WordLevel
    /// model with a minimal vocab and whitespace pre-tokenizer. Enough for
    /// `Tokenizer::from_file` to succeed in the disabled-path test (we never
    /// actually encode through it there).
    const MINIMAL_TOKENIZER_JSON: &str = r#"{
      "version": "1.0",
      "truncation": null,
      "padding": null,
      "added_tokens": [],
      "normalizer": null,
      "pre_tokenizer": {"type": "Whitespace"},
      "post_processor": null,
      "decoder": null,
      "model": {"type": "WordLevel", "vocab": {"[UNK]": 0}, "unk_token": "[UNK]"}
    }"#;

    /// Manual smoke: downloads the bge-v2-m3 ONNX classifier and verifies an
    /// on-topic doc outranks an off-topic one. Requires network on first run.
    /// Exercises the real ONNX classifier (ClassifierLogit) path end-to-end.
    ///
    ///   SEMANTEX_RERANKER=on cargo test -p semantex-core \
    ///     -- --ignored onnx_reranker::tests::onnx_classifier_ranks_on_topic
    #[test]
    #[ignore]
    fn onnx_classifier_ranks_on_topic() {
        use crate::config::SemantexConfig;
        use crate::search::fastembed_reranker::ENV_ENABLE;
        use crate::search::reranker_download::ensure_reranker_model;
        use crate::search::reranker_model::{select_reranker_choice_from_env, RerankerChoice};

        with_env(&[(ENV_ENABLE, Some("on"))], || {
            // Force the ONNX classifier alias.
            // SAFETY: guarded by the with_env mutex above.
            unsafe { std::env::set_var("SEMANTEX_RERANKER_MODEL", "bge-onnx") };
            let spec = match select_reranker_choice_from_env() {
                RerankerChoice::Onnx(s) => s,
                other => panic!("expected ONNX choice for bge-onnx, got {other:?}"),
            };
            let config = SemantexConfig::default();
            let dir = ensure_reranker_model(&config.models_dir(), &spec.files)
                .expect("download (offline?)");
            // The bge ONNX export ships `model.onnx` (carried in spec.session_file).
            let r = OnnxReranker::new(&dir, &spec.session_file, ScoreStrategy::ClassifierLogit, 4, false)
                .expect("construct");
            let docs = [
                "Pizza is a popular Italian dish.",
                "fn binary_search(a: &[i32], t: i32) -> Option<usize> { /* ... */ }",
            ];
            let docs_ref: Vec<&str> = docs.iter().copied().collect();
            let out = r
                .rerank("how does binary search work", &docs_ref, 2)
                .expect("rerank");
            assert_eq!(out[0].0, 1, "on-topic code doc should rank first");
        });
    }

    /// Env scrub/restore helper (copied verbatim from `fastembed_reranker.rs`
    /// test module — tests can't share a private fn across files).
    fn with_env<F: FnOnce()>(vars: &[(&str, Option<&str>)], f: F) {
        use std::sync::Mutex;
        static ENV_LOCK: Mutex<()> = Mutex::new(());
        let _guard = ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let prior: Vec<(String, Option<String>)> = vars
            .iter()
            .map(|(k, _)| ((*k).to_string(), std::env::var(*k).ok()))
            .collect();
        // SAFETY: env vars are guarded by ENV_LOCK above.
        unsafe {
            for (k, v) in vars {
                match v {
                    Some(val) => std::env::set_var(k, val),
                    None => std::env::remove_var(k),
                }
            }
        }
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
        // SAFETY: env vars are guarded by ENV_LOCK above.
        unsafe {
            for (k, v) in &prior {
                match v {
                    Some(val) => std::env::set_var(k, val),
                    None => std::env::remove_var(k),
                }
            }
        }
        if let Err(e) = result {
            std::panic::resume_unwind(e);
        }
    }
}
