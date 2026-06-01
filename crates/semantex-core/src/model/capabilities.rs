//! `ModelCapabilities` + capability→backend negotiation.
//!
//! The engine NEGOTIATES against these descriptors rather than branching on a
//! model id: a single-vector model routes to the HNSW backend. Adding a future
//! capability = one field here (defaulted off) + one handler; existing models
//! keep working unchanged — the "new capabilities ship without an engine
//! refactor" guarantee (design §4 S8).
//!
//! The `multi_vector` field is RETAINED (a future multi-vector backend slots in
//! behind the `DenseBackend` seam), but no built-in backend serves it today:
//! `backend_for` errors on a multi-vector model rather than silently
//! mis-routing it to the single-vector path.

use crate::search::dense_backend::DenseBackendKind;
use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Engine-negotiable model capabilities. Every field defaults to the
/// conservative profile so a partial `models.toml` entry — or an older built-in
/// that predates a new capability — keeps working with the capability OFF.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ModelCapabilities {
    /// `true` → per-token vectors (late-interaction / MaxSim); `false` →
    /// single-vector. RETAINED for a future multi-vector backend; no built-in
    /// backend serves `true` today (see [`backend_for`]).
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
///
/// Single-variant today (D4: coderank-hnsw is the sole built-in dense backend),
/// but kept an enum so a future backend slots in alongside it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    /// CodeRankEmbed single-vector + HNSW.
    CoderankHnsw,
}

/// Negotiate the dense backend from capabilities alone (no model id branching).
/// A single-vector model routes to the HNSW backend. A `multi_vector=true` model
/// has NO built-in backend (D4 removed the ColBERT/PLAID path) — `Err`, so the
/// caller surfaces a clear "no multi-vector backend" message rather than
/// silently mis-routing it to the single-vector path.
pub fn backend_for(caps: &ModelCapabilities) -> Result<BackendKind> {
    if caps.multi_vector {
        anyhow::bail!(
            "model declares multi_vector=true but no multi-vector dense backend \
             is available (the coderank-hnsw backend is single-vector). Use a \
             single-vector embedder, or add a multi-vector backend behind the \
             DenseBackend seam."
        )
    }
    Ok(BackendKind::CoderankHnsw)
}

impl BackendKind {
    /// Map to S1's on-disk/selection enum. The only coupling point between the
    /// registry and the `DenseBackend` seam. The `Result` return is kept for
    /// source stability with existing call sites — it never returns `Err` today.
    pub fn dense_kind(self) -> Result<DenseBackendKind> {
        match self {
            BackendKind::CoderankHnsw => Ok(DenseBackendKind::CoderankHnsw),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_vector_routes_to_hnsw() {
        let sv = ModelCapabilities {
            multi_vector: false,
            ..Default::default()
        };
        assert_eq!(backend_for(&sv).unwrap(), BackendKind::CoderankHnsw);
    }

    #[test]
    fn multi_vector_has_no_backend() {
        // D4 removed the ColBERT/PLAID path: a multi-vector model no longer
        // resolves to a built-in backend and must error (not mis-route).
        let mv = ModelCapabilities {
            multi_vector: true,
            ..Default::default()
        };
        let err = backend_for(&mv).expect_err("multi-vector must have no backend");
        assert!(err.to_string().contains("multi-vector"), "got: {err}");
    }

    #[test]
    fn coderank_hnsw_dense_kind_maps_to_s1_dense_kind() {
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
