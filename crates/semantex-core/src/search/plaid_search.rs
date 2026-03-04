//! PLAID-based ColBERT searcher using memory-mapped index.
//!
//! This module wraps the `next-plaid` crate to provide ColBERT late-interaction
//! search over a pre-built PLAID index. Each document in the PLAID index is
//! mapped back to a semantex `chunk_id` via a bincode-encoded `Vec<u64>` mapping file.

use crate::embedding::colbert::ColbertEmbedder;
use crate::types::ScoredChunkId;
use anyhow::Result;
use next_plaid::{MmapIndex, SearchParameters};
use std::path::Path;

/// PLAID-based ColBERT searcher using memory-mapped index.
///
/// Wraps `next_plaid::MmapIndex` and a doc-to-chunk ID mapping so that
/// PLAID passage IDs can be translated back to semantex's internal `chunk_id`.
pub struct PlaidSearcher {
    /// Memory-mapped PLAID index.
    index: MmapIndex,
    /// Maps PLAID doc_id (positional index) to semantex chunk_id (SQLite row ID).
    doc_to_chunk: Vec<u64>,
}

impl PlaidSearcher {
    /// Open an existing PLAID index and its chunk-ID mapping from disk.
    ///
    /// # Arguments
    ///
    /// * `index_dir`    - Directory containing the PLAID index files.
    /// * `mapping_path` - Path to the bincode-encoded `Vec<u64>` mapping file
    ///   (`plaid_mapping.bin`).
    ///
    /// # Errors
    ///
    /// Returns an error if the index cannot be opened or the mapping file
    /// cannot be read/decoded.
    pub fn open(index_dir: &Path, mapping_path: &Path) -> Result<Self> {
        let index = MmapIndex::load(&index_dir.to_string_lossy())?;

        let mapping_bytes = std::fs::read(mapping_path)?;
        let (doc_to_chunk, _): (Vec<u64>, _) =
            bincode::serde::decode_from_slice(&mapping_bytes, bincode::config::standard())?;

        Ok(Self {
            index,
            doc_to_chunk,
        })
    }

    /// Search using ColBERT MaxSim scoring.
    ///
    /// Encodes `query` via the provided `ColbertEmbedder`, searches the PLAID
    /// index, then maps passage IDs back to semantex chunk IDs.
    ///
    /// Returns `ScoredChunkId` items sorted by descending score.
    ///
    /// # Errors
    ///
    /// Returns an error if query encoding or PLAID search fails.
    pub fn search(
        &self,
        embedder: &ColbertEmbedder,
        query: &str,
        top_k: usize,
    ) -> Result<Vec<ScoredChunkId>> {
        let query_emb = embedder.encode_query(query)?;

        let params = SearchParameters {
            top_k,
            ..Default::default()
        };

        let results = self.index.search(&query_emb, &params, None)?;

        let mut scored: Vec<ScoredChunkId> = results
            .passage_ids
            .iter()
            .zip(results.scores.iter())
            .filter_map(|(&doc_id, &score)| {
                let doc_idx = doc_id as usize;
                self.doc_to_chunk
                    .get(doc_idx)
                    .map(|&chunk_id| ScoredChunkId::new(chunk_id, score))
            })
            .collect();

        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(scored)
    }
}
