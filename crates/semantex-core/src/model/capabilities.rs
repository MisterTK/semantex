//! `ModelCapabilities` + capability→backend negotiation.
//!
//! The engine NEGOTIATES against these descriptors rather than branching on a
//! model id: `multi_vector=true` routes to the PLAID/MaxSim backend, `false` to
//! single-vector/HNSW. Adding a future capability = one field here (defaulted
//! off) + one handler; existing models keep working unchanged — the "new
//! capabilities ship without an engine refactor" guarantee (design §4 S8).

use crate::search::dense_backend::DenseBackendKind;
use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Engine-negotiable model capabilities. Every field defaults to the
/// conservative profile so a partial `models.toml` entry — or an older built-in
/// that predates a new capability — keeps working with the capability OFF.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ModelCapabilities {
    /// `true` → per-token vectors (ColBERT/PLAID MaxSim); `false` → single-vector.
    #[serde(default)]
    pub multi_vector: bool,
    /// Matryoshka truncation dims, if the model is MRL-trained (else `None` →
    /// fixed dim; do not truncate). CodeRankEmbed is NOT MRL (design §4 S2).
    #[serde(default)]
    pub matryoshka_dims: Option<Vec<usize>>,
    /// `true` → the model also emits a sparse signal (reserved; no built-in uses
    /// it yet — present so a future SPLADE-style model needs no struct change).
    #[serde(default)]
    pub produces_sparse: bool,
    /// `true` → the model is instruction-aware (e.g. takes a query prefix /
    /// reranker instruction). Informational; the prefix itself lives on the spec.
    #[serde(default)]
    pub instruction_aware: bool,
    /// Max batch the model can encode/score at once (`None` → engine default).
    #[serde(default)]
    pub max_batch: Option<usize>,
}

/// Which dense backend a model's capabilities select. S8's own enum; maps to
/// S1's `DenseBackendKind` via [`BackendKind::dense_kind`] (the single coupling
/// point between the registry and S1's seam).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    /// ColBERT late-interaction + PLAID (multi-vector).
    ColbertPlaid,
    /// CodeRankEmbed single-vector + HNSW.
    CoderankHnsw,
}

/// Negotiate the dense backend from capabilities alone (no model id branching).
/// `multi_vector=true` → PLAID/MaxSim; otherwise single-vector/HNSW.
#[must_use]
pub fn backend_for(caps: &ModelCapabilities) -> BackendKind {
    if caps.multi_vector {
        BackendKind::ColbertPlaid
    } else {
        BackendKind::CoderankHnsw
    }
}

impl BackendKind {
    /// Map to S1's on-disk/selection enum. The only coupling point between the
    /// registry and the `DenseBackend` seam.
    ///
    /// TOTAL (post-S2): both `BackendKind`s now map to a representable
    /// `DenseBackendKind`. (Pre-S2 this was fallible because S1's enum lacked the
    /// `CoderankHnsw` variant; S2 added it, so the conversion is infallible.) The
    /// `Result` return is kept for source stability with the existing call sites
    /// — it now never returns `Err`.
    pub fn dense_kind(self) -> Result<DenseBackendKind> {
        match self {
            BackendKind::ColbertPlaid => Ok(DenseBackendKind::ColbertPlaid),
            BackendKind::CoderankHnsw => Ok(DenseBackendKind::CoderankHnsw),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multi_vector_routes_to_plaid_single_to_hnsw() {
        let mv = ModelCapabilities {
            multi_vector: true,
            ..Default::default()
        };
        assert_eq!(backend_for(&mv), BackendKind::ColbertPlaid);

        let sv = ModelCapabilities {
            multi_vector: false,
            ..Default::default()
        };
        assert_eq!(backend_for(&sv), BackendKind::CoderankHnsw);
    }

    #[test]
    fn colbert_plaid_backend_kind_maps_to_s1_dense_kind() {
        assert_eq!(
            BackendKind::ColbertPlaid.dense_kind().unwrap(),
            DenseBackendKind::ColbertPlaid
        );
    }

    #[test]
    fn coderank_hnsw_dense_kind_maps_to_s1_dense_kind() {
        // Post-S2: the single-vector backend now maps totally to the S1 enum.
        assert_eq!(
            BackendKind::CoderankHnsw.dense_kind().unwrap(),
            DenseBackendKind::CoderankHnsw
        );
    }

    #[test]
    fn capabilities_default_is_single_vector_profile() {
        let c = ModelCapabilities::default();
        assert!(!c.multi_vector);
        assert!(c.matryoshka_dims.is_none());
        assert!(!c.produces_sparse);
        assert!(c.max_batch.is_none());
    }

    #[test]
    fn partial_toml_keeps_unset_capabilities_off() {
        // A manifest that only sets multi_vector must default the rest off.
        let c: ModelCapabilities = toml::from_str("multi_vector = true\n").unwrap();
        assert!(c.multi_vector);
        assert!(!c.instruction_aware);
        assert!(c.matryoshka_dims.is_none());
    }
}
