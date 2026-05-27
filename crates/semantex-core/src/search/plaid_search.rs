//! PLAID-based ColBERT searcher using memory-mapped index.
//!
//! This module wraps the `next-plaid` crate to provide ColBERT late-interaction
//! search over a pre-built PLAID index. Each document in the PLAID index is
//! mapped back to a semantex `chunk_id` via a postcard-encoded `Vec<u64>` mapping file.

use crate::embedding::colbert::ColbertEmbedder;
use crate::types::ScoredChunkId;
use anyhow::Result;
use next_plaid::{MmapIndex, SearchParameters};
use std::path::Path;

/// Default number of IVF centroid clusters probed during PLAID approximate search.
///
/// `next_plaid::SearchParameters::default()` uses `n_ivf_probe = 8`, but ColBERT
/// literature recommends 32–64 for high recall. With semantex's 48-dim ColBERT
/// model and typical index sizes the extra cost is sub-millisecond per query
/// while recall improves measurably for short queries (each query token gets
/// `n_ivf_probe` candidate cells). Overridable via `SEMANTEX_PLAID_PROBE`.
const DEFAULT_N_IVF_PROBE: usize = 32;

/// Default centroid score threshold for PLAID approximate search.
/// Overridable via `SEMANTEX_PLAID_CENTROID_THRESHOLD`.
const DEFAULT_CENTROID_THRESHOLD: f32 = 0.4;

/// Resolve `SEMANTEX_PLAID_PROBE` via a caller-provided env lookup function.
///
/// Factored out as a pure helper so the env-driven default can be unit-tested
/// without manipulating the process environment.
fn resolve_plaid_probe<F>(env: F) -> usize
where
    F: Fn(&str) -> Option<String>,
{
    env("SEMANTEX_PLAID_PROBE")
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_N_IVF_PROBE)
}

/// Resolve `SEMANTEX_PLAID_CENTROID_THRESHOLD` via a caller-provided env lookup.
///
/// Returns `Some(threshold)`: either the parsed env value or the built-in default.
/// The threshold is always set (never `None`) because the previous default of
/// `None` from `SearchParameters::default()` was a less aggressive prune.
fn resolve_plaid_centroid_threshold<F>(env: F) -> Option<f32>
where
    F: Fn(&str) -> Option<String>,
{
    env("SEMANTEX_PLAID_CENTROID_THRESHOLD")
        .and_then(|v| v.parse().ok())
        .map_or(Some(DEFAULT_CENTROID_THRESHOLD), Some)
}

/// Translate a chunk-ID subset into a PLAID doc-ID subset.
///
/// `doc_to_chunk` is the positional doc_id → chunk_id mapping persisted as
/// `plaid_mapping.bin`. We walk it once and emit the positional `i64` doc IDs
/// of entries whose chunk ID is present in `chunk_id_subset`. Lifted out of
/// `search_with_subset` so the translation is unit-testable without an actual
/// PLAID index.
fn translate_chunk_subset_to_doc_subset(doc_to_chunk: &[u64], chunk_id_subset: &[u64]) -> Vec<i64> {
    let chunk_set: std::collections::HashSet<u64> = chunk_id_subset.iter().copied().collect();
    doc_to_chunk
        .iter()
        .enumerate()
        .filter_map(|(doc_idx, &cid)| {
            if chunk_set.contains(&cid) {
                Some(doc_idx as i64)
            } else {
                None
            }
        })
        .collect()
}

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
    /// * `mapping_path` - Path to the postcard-encoded `Vec<u64>` mapping file
    ///   (`plaid_mapping.bin`).
    ///
    /// # Errors
    ///
    /// Returns an error if the index cannot be opened or the mapping file
    /// cannot be read/decoded.
    pub fn open(index_dir: &Path, mapping_path: &Path) -> Result<Self> {
        let index = MmapIndex::load(&index_dir.to_string_lossy())?;

        let mapping_bytes = std::fs::read(mapping_path)?;
        let doc_to_chunk: Vec<u64> = postcard::from_bytes(&mapping_bytes)?;

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

        let n_ivf_probe = resolve_plaid_probe(|k| std::env::var(k).ok());
        let centroid_score_threshold = resolve_plaid_centroid_threshold(|k| std::env::var(k).ok());

        let params = SearchParameters {
            top_k,
            n_ivf_probe,
            centroid_score_threshold,
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

    /// Search using ColBERT MaxSim scoring, optionally restricted to a subset
    /// of chunk IDs.
    ///
    /// When `chunk_id_subset` is `Some(&[..])`, the slice's chunk IDs are
    /// translated to PLAID doc IDs via `self.doc_to_chunk` and passed through
    /// `next_plaid::MmapIndex::search`'s `subset` parameter (next-plaid 1.3+).
    /// PLAID restricts scoring to those doc IDs and proportionally scales
    /// `n_ivf_probe` to compensate for the smaller candidate pool, so callers
    /// MAY skip post-filter pruning.
    ///
    /// When `chunk_id_subset` is `None`, this behaves identically to `search`.
    ///
    /// # Errors
    ///
    /// Returns an error if query encoding or PLAID search fails.
    pub fn search_with_subset(
        &self,
        embedder: &ColbertEmbedder,
        query: &str,
        top_k: usize,
        chunk_id_subset: Option<&[u64]>,
    ) -> Result<Vec<ScoredChunkId>> {
        let query_emb = embedder.encode_query(query)?;

        let n_ivf_probe = resolve_plaid_probe(|k| std::env::var(k).ok());
        let centroid_score_threshold = resolve_plaid_centroid_threshold(|k| std::env::var(k).ok());

        let params = SearchParameters {
            top_k,
            n_ivf_probe,
            centroid_score_threshold,
            ..Default::default()
        };

        // Translate chunk_id subset -> PLAID doc_id subset via positional
        // lookup against `self.doc_to_chunk` (matches the mapping built during
        // index construction). An empty subset short-circuits to an empty
        // result rather than handing PLAID an empty `&[]` (whose semantics —
        // scan all or scan none — vary by version).
        let plaid_subset: Option<Vec<i64>> = chunk_id_subset
            .map(|chunks| translate_chunk_subset_to_doc_subset(&self.doc_to_chunk, chunks));

        if matches!(plaid_subset.as_ref(), Some(s) if s.is_empty()) {
            return Ok(Vec::new());
        }

        let results = self
            .index
            .search(&query_emb, &params, plaid_subset.as_deref())?;

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

    /// Borrow the doc_id → chunk_id positional mapping.
    ///
    /// Exposed for callers (e.g. `HybridSearcher`) that need to enumerate the
    /// indexed chunk IDs to compute a subset for `search_with_subset` without
    /// re-reading `plaid_mapping.bin` from disk.
    pub fn doc_to_chunk(&self) -> &[u64] {
        &self.doc_to_chunk
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_plaid_probe_uses_default_when_env_unset() {
        let probe = resolve_plaid_probe(|_| None);
        assert_eq!(probe, DEFAULT_N_IVF_PROBE);
        assert_eq!(probe, 32, "DEFAULT_N_IVF_PROBE must be 32 per v0.4 spec");
    }

    #[test]
    fn resolve_plaid_probe_honors_env_override() {
        // Reproducing the spec acceptance: SEMANTEX_PLAID_PROBE=8 must restore
        // the pre-change next_plaid::SearchParameters::default() behavior.
        let probe = resolve_plaid_probe(|k| {
            if k == "SEMANTEX_PLAID_PROBE" {
                Some("8".to_string())
            } else {
                None
            }
        });
        assert_eq!(probe, 8);
    }

    #[test]
    fn resolve_plaid_probe_ignores_unparseable_env() {
        let probe = resolve_plaid_probe(|k| {
            if k == "SEMANTEX_PLAID_PROBE" {
                Some("not-a-number".to_string())
            } else {
                None
            }
        });
        // Unparseable value falls back to default rather than panicking.
        assert_eq!(probe, DEFAULT_N_IVF_PROBE);
    }

    #[test]
    fn resolve_plaid_centroid_threshold_uses_default_when_env_unset() {
        let thr = resolve_plaid_centroid_threshold(|_| None);
        assert_eq!(thr, Some(DEFAULT_CENTROID_THRESHOLD));
    }

    #[test]
    fn resolve_plaid_centroid_threshold_honors_env_override() {
        let thr = resolve_plaid_centroid_threshold(|k| {
            if k == "SEMANTEX_PLAID_CENTROID_THRESHOLD" {
                Some("0.25".to_string())
            } else {
                None
            }
        });
        assert!(matches!(thr, Some(v) if (v - 0.25).abs() < 1e-6));
    }

    #[test]
    fn translate_subset_emits_positional_doc_ids() {
        // doc_to_chunk: doc 0 -> chunk 100, doc 1 -> chunk 200, doc 2 -> chunk 300,
        //               doc 3 -> chunk 400, doc 4 -> chunk 500.
        let d2c: Vec<u64> = vec![100, 200, 300, 400, 500];
        // Subset selects chunks {200, 400, 500} -> docs {1, 3, 4}.
        let subset = [200u64, 400, 500];
        let docs = translate_chunk_subset_to_doc_subset(&d2c, &subset);
        assert_eq!(docs, vec![1i64, 3, 4]);
    }

    #[test]
    fn translate_subset_skips_unmapped_chunks() {
        // Chunk IDs not present in doc_to_chunk are silently dropped — they
        // simply aren't in the PLAID index, so there's no doc to point at.
        let d2c: Vec<u64> = vec![10, 20, 30];
        let subset = [20u64, 999, 30];
        let docs = translate_chunk_subset_to_doc_subset(&d2c, &subset);
        assert_eq!(docs, vec![1i64, 2]);
    }

    #[test]
    fn translate_subset_empty_chunks_yields_empty_docs() {
        let d2c: Vec<u64> = vec![10, 20, 30];
        let docs = translate_chunk_subset_to_doc_subset(&d2c, &[]);
        assert!(docs.is_empty());
    }

    #[test]
    fn translate_subset_preserves_doc_order_not_subset_order() {
        // Subset order is {500, 100}; doc order must be {0, 4} (positional).
        let d2c: Vec<u64> = vec![100, 200, 300, 400, 500];
        let subset = [500u64, 100];
        let docs = translate_chunk_subset_to_doc_subset(&d2c, &subset);
        assert_eq!(docs, vec![0i64, 4], "doc IDs must be ascending positional");
    }
}
