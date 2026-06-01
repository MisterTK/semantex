//! Config-driven model registry (S8). Every model — embedder, reranker, LLM —
//! is declared as DATA (`ModelSpec` + `ModelCapabilities`) and resolved by role
//! through `ModelRegistry` from compiled-in permissive defaults plus an optional
//! user `models.toml`. Swapping a model is a config change, not a recompile.
//!
//! See `docs/superpowers/specs/2026-05-31-semantex-sota-overhaul-design.md` §4 S8.

pub mod capabilities;
pub mod manifest;
pub mod registry;
pub mod spec;

pub use capabilities::{BackendKind, ModelCapabilities, backend_for};
pub use registry::{ModelRegistry, dense_context_enabled};
pub use spec::{
    EmbedderFingerprint, EmbedderSpec, LlmSpec, ModelRole, ModelSource, ModelSpec, Pooling,
    QuantKind, RerankerSpec, RoleData, ScoreStrategyKind,
};
