//! `ModelSpec` — every model declared as DATA, never code.
//!
//! A spec carries the model's identity, where to fetch it, its capabilities,
//! and the role-specific nuance that would otherwise be hardcoded in engine
//! code (embedder dims/prefix/pooling/quant; reranker score strategy + prompt;
//! llm provider/model/endpoint). The engine reads these fields — it never
//! special-cases a model id. This is what keeps quality from being flattened to
//! a lowest common denominator. See the SOTA overhaul design spec §4 S8 / §2 D9.

use crate::model::capabilities::ModelCapabilities;
use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Which pipeline stage a model serves.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelRole {
    /// Dense embedder (the dense search channel).
    Embedder,
    /// Cross-encoder reranker (final precision stage).
    Reranker,
    /// LLM for query understanding / HyDE (feature `llm` only).
    Llm,
}

/// Where a model's files are fetched from. Carried as DATA so a new model needs
/// no code — only a manifest entry. The actual download is done by the existing
/// `embedding/*_model.rs` provisioners, fed these coordinates.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum ModelSource {
    /// HuggingFace repo (`<owner>/<repo>`); `files` are the resolve-path leaves.
    Hf { repo: String, files: Vec<String> },
    /// A pre-provisioned local directory (airgap / hand-placed weights).
    Local { dir: String },
    /// Arbitrary HTTPS base URL + file leaves (self-hosted mirror).
    Url { base: String, files: Vec<String> },
}

/// Token pooling for an embedder (how token vectors collapse to one vector).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Pooling {
    /// Mean over the attention mask.
    Mean,
    /// CLS token (position 0).
    Cls,
    /// No pooling — per-token vectors (ColBERT / PLAID MaxSim).
    LateInteraction,
}

/// On-disk vector quantization for an embedder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuantKind {
    /// Full-precision f32 vectors.
    None,
    /// Symmetric per-vector int8 (zero-point 0) — integration doc D-int8.
    Int8Symmetric,
}

/// How a reranker turns model output into a relevance score (mirrors S3's
/// `onnx_reranker::ScoreStrategy`, but as manifest data).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScoreStrategyKind {
    /// bge-style sequence-classification: single relevance logit.
    ClassifierHead,
    /// Qwen3-Reranker-style generative: logit of the "yes" token.
    YesNoLogit,
}

/// Embedder-specific nuance. EVERY field that would otherwise be a hardcoded
/// constant in `embedding/*.rs` lives here as data.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbedderSpec {
    /// Output embedding dimension (e.g. 768 for CodeRankEmbed, 48/token ColBERT).
    pub dims: usize,
    /// Max context tokens the encoder accepts (inputs truncate to this).
    pub max_context: usize,
    /// Prefix prepended to QUERIES (e.g. CodeRankEmbed's instruction). May be "".
    #[serde(default)]
    pub query_prefix: String,
    /// Prefix prepended to DOCUMENTS. Usually "".
    #[serde(default)]
    pub doc_prefix: String,
    /// Token pooling.
    pub pooling: Pooling,
    /// Whether to L2-normalize the pooled vector.
    #[serde(default = "default_true")]
    pub normalize: bool,
    /// On-disk vector quantization.
    pub quant: QuantKind,
}

/// Reranker-specific nuance.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RerankerSpec {
    /// Score-extraction strategy.
    pub score_strategy: ScoreStrategyKind,
    /// Prompt template for `YesNoLogit` rerankers (prefix/middle/suffix around
    /// the query+doc). Ignored for `ClassifierHead`.
    #[serde(default)]
    pub prompt_prefix: String,
    #[serde(default)]
    pub prompt_middle: String,
    #[serde(default)]
    pub prompt_suffix: String,
    /// Token id of "yes" for `YesNoLogit`. `None` for `ClassifierHead`.
    #[serde(default)]
    pub yes_token_id: Option<usize>,
    /// Token id of "no" for `YesNoLogit`.
    #[serde(default)]
    pub no_token_id: Option<usize>,
}

/// LLM-specific nuance (delegated to genai when the `llm` feature is on). Held
/// as plain strings so the default (no-`llm`) build still parses a manifest that
/// happens to contain an llm entry — it just never instantiates it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LlmSpec {
    /// genai provider (e.g. "anthropic", "ollama"). Maps to SEMANTEX_LLM_PROVIDER.
    #[serde(default)]
    pub provider: String,
    /// Model id passed to genai. Maps to SEMANTEX_LLM_MODEL.
    pub model: String,
    /// Optional endpoint override (Ollama / self-hosted). Maps to SEMANTEX_LLM_ENDPOINT.
    #[serde(default)]
    pub endpoint: String,
}

/// The role-specific payload of a [`ModelSpec`]. Exactly one variant matches
/// `ModelSpec::role`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum RoleData {
    Embedder(EmbedderSpec),
    Reranker(RerankerSpec),
    Llm(LlmSpec),
}

/// A fully-declared model. The umbrella type the registry stores and resolves.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelSpec {
    /// Stable logical id used by config selection (e.g. `"coderank-137m"`,
    /// `"bge-reranker-v2-m3"`). Distinct from the dense BACKEND name
    /// (`coderank-hnsw`/`colbert-plaid`) the embedder routes to via capabilities.
    pub id: String,
    /// Pipeline stage. MUST agree with `role_data`'s variant.
    pub role: ModelRole,
    /// Where to fetch the model files.
    pub source: ModelSource,
    /// Engine-negotiated capabilities (defaults fill missing manifest fields).
    #[serde(default)]
    pub capabilities: ModelCapabilities,
    /// Role-specific nuance (flattened so a manifest entry is one table).
    #[serde(flatten)]
    pub role_data: RoleData,
}

impl ModelSpec {
    /// Validate internal consistency: `role` agrees with `role_data`, ids are
    /// non-empty, sources name at least one file, and role-specific invariants
    /// hold (e.g. a `YesNoLogit` reranker carries both token ids). Errors NAME
    /// the offending field so a bad `models.toml` is actionable (risk row
    /// "model-manifest misconfiguration").
    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(!self.id.trim().is_empty(), "model spec has an empty `id`");
        let role_ok = matches!(
            (self.role, &self.role_data),
            (ModelRole::Embedder, RoleData::Embedder(_))
                | (ModelRole::Reranker, RoleData::Reranker(_))
                | (ModelRole::Llm, RoleData::Llm(_))
        );
        anyhow::ensure!(
            role_ok,
            "model `{}`: `role` {:?} disagrees with its role data",
            self.id,
            self.role
        );
        match &self.source {
            ModelSource::Hf { repo, files } => {
                anyhow::ensure!(
                    !repo.trim().is_empty(),
                    "model `{}`: empty hf `repo`",
                    self.id
                );
                anyhow::ensure!(
                    !files.is_empty(),
                    "model `{}`: hf source lists no `files`",
                    self.id
                );
            }
            ModelSource::Local { dir } => {
                anyhow::ensure!(
                    !dir.trim().is_empty(),
                    "model `{}`: empty local `dir`",
                    self.id
                );
            }
            ModelSource::Url { base, files } => {
                anyhow::ensure!(
                    !base.trim().is_empty(),
                    "model `{}`: empty url `base`",
                    self.id
                );
                anyhow::ensure!(
                    !files.is_empty(),
                    "model `{}`: url source lists no `files`",
                    self.id
                );
            }
        }
        if let RoleData::Embedder(e) = &self.role_data {
            anyhow::ensure!(e.dims > 0, "model `{}`: embedder `dims` must be > 0", self.id);
            anyhow::ensure!(
                e.max_context > 0,
                "model `{}`: embedder `max_context` must be > 0",
                self.id
            );
        }
        if let RoleData::Reranker(r) = &self.role_data
            && matches!(r.score_strategy, ScoreStrategyKind::YesNoLogit)
        {
            anyhow::ensure!(
                r.yes_token_id.is_some() && r.no_token_id.is_some(),
                "model `{}`: yes_no_logit reranker needs both `yes_token_id` and `no_token_id`",
                self.id
            );
        }
        Ok(())
    }
}

fn default_true() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn role_round_trips_through_toml() {
        for (role, s) in [
            (ModelRole::Embedder, "embedder"),
            (ModelRole::Reranker, "reranker"),
            (ModelRole::Llm, "llm"),
        ] {
            // serde rename = lowercase; round-trips via a tiny TOML doc.
            let doc = format!("role = \"{s}\"\n");
            let parsed: RoleHolder = toml::from_str(&doc).unwrap();
            assert_eq!(parsed.role, role);
        }
    }

    #[test]
    fn source_hf_round_trips() {
        let doc = r#"
            kind = "hf"
            repo = "owner/Model-onnx-int8"
            files = ["model_int8.onnx", "tokenizer.json"]
        "#;
        let src: ModelSource = toml::from_str(doc).unwrap();
        match src {
            ModelSource::Hf { repo, files } => {
                assert_eq!(repo, "owner/Model-onnx-int8");
                assert_eq!(files, vec!["model_int8.onnx", "tokenizer.json"]);
            }
            other => panic!("expected Hf, got {other:?}"),
        }
    }

    #[test]
    fn embedder_spec_carries_every_nuance() {
        let e = EmbedderSpec {
            dims: 768,
            max_context: 8192,
            query_prefix: "Represent this query for searching relevant code: ".to_string(),
            doc_prefix: String::new(),
            pooling: Pooling::Cls,
            normalize: true,
            quant: QuantKind::Int8Symmetric,
        };
        // The fields are data, not behavior — a spec is fully described here.
        assert_eq!(e.dims, 768);
        assert_eq!(e.pooling, Pooling::Cls);
        assert_eq!(e.quant, QuantKind::Int8Symmetric);
        assert!(e.query_prefix.ends_with(' '));
    }

    /// Helper struct so the role test can deserialize a bare `role = "…"`.
    #[derive(Deserialize)]
    struct RoleHolder {
        role: ModelRole,
    }
}
