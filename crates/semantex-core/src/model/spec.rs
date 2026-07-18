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
    /// No pooling — per-token vectors (for a future multi-vector /
    /// late-interaction MaxSim backend; no built-in backend serves it today).
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
    /// Output embedding dimension (e.g. 768 for CodeRankEmbed; for a
    /// multi-vector model this is the per-token dimension).
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
    /// Optional distilled static token table artifact (Ember Tier-0 doc-side
    /// encoder), as a bare filename resolved inside the model directory. Not
    /// downloaded from the model source — produced offline by
    /// `semantex distill-static-table`. `None` for models without one.
    #[serde(default)]
    pub static_token_table: Option<String>,
    /// Optional frozen universal PLAID centroids artifact (Ember Plan B), as
    /// a bare filename resolved inside the model directory. Produced offline
    /// by `semantex distill-centroids`. `None` for models without one.
    #[serde(default)]
    pub frozen_centroids: Option<String>,
}

/// Reranker-specific nuance.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RerankerSpec {
    /// Score-extraction strategy.
    pub score_strategy: ScoreStrategyKind,
    /// Max sequence length (in tokens) fed to the cross-encoder. The tokenizer
    /// truncates each (query, document) pair / rendered prompt to this many
    /// tokens so a large chunk never blows past the model's trained positional
    /// range (bge @512 — past it the logits are garbage) or its CPU O(seq²)
    /// latency/memory cliff (Qwen3-0.6B). Mirrors `EmbedderSpec.max_context`.
    /// Defaults to 512 (bge's trained context) when absent from a manifest entry.
    #[serde(default = "default_reranker_max_context")]
    pub max_context: usize,
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
///
/// Externally tagged: in a `models.toml` entry the payload lives in a subtable
/// named for the role (`[model.embedder]` / `[model.reranker]` / `[model.llm]`),
/// alongside the independent `role = "…"` discriminator field on `ModelSpec`.
/// `ModelSpec` flattens this enum, so the subtable key sits at the same level as
/// `id`/`source`/`capabilities`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
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
    /// (e.g. `coderank-hnsw`) the embedder routes to via capabilities.
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
            anyhow::ensure!(
                e.dims > 0,
                "model `{}`: embedder `dims` must be > 0",
                self.id
            );
            anyhow::ensure!(
                e.max_context > 0,
                "model `{}`: embedder `max_context` must be > 0",
                self.id
            );
            for (field, value) in [
                ("static_token_table", &e.static_token_table),
                ("frozen_centroids", &e.frozen_centroids),
            ] {
                if let Some(name) = value {
                    anyhow::ensure!(
                        !name.is_empty()
                            && !name.contains('/')
                            && !name.contains('\\')
                            && !name.contains(".."),
                        "model `{}`: embedder `{field}` must be a bare filename, got {name:?}",
                        self.id
                    );
                }
            }
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

/// A stable content fingerprint of an embedder spec, used to stamp a dense index
/// and detect an embedder change at open time. Generalizes the schema /
/// stemmer-flag / dense-backend guards into one uniform "is this index's vector
/// space the one this embedder produces?" check.
///
/// The fingerprint covers exactly the fields that change the produced vectors:
/// the model id, dims, pooling, quantization, normalization, BOTH the query and
/// document prefixes, AND the runtime `dense_context` flag. (`doc_prefix` is
/// prepended to document text at index time, so it changes the stored vector
/// space — overriding it on the same id MUST invalidate the on-disk index.
/// `dense_context` likewise changes the EMBEDDED TEXT — annotation+code vs raw
/// code — so the two settings produce distinct vector spaces and MUST be distinct
/// indexes.) A reranker/LLM swap does NOT touch the dense vector space → not
/// fingerprinted → no reindex; that is the design's query-time-live-swap
/// guarantee.
pub struct EmbedderFingerprint;

impl EmbedderFingerprint {
    /// Compute the fingerprint string for `(id, spec, dense_context)`.
    /// Deterministic across runs/platforms (xxh64 of a canonical byte encoding).
    ///
    /// `dense_context` is the runtime `SEMANTEX_DENSE_CONTEXT` A/B flag: when on,
    /// documents are embedded as `annotation\ncode` instead of raw code, which
    /// changes the stored vectors. It is therefore part of the vector-space
    /// identity — toggling it MUST yield a different fingerprint so the index is
    /// re-embedded (not silently reused under the wrong embedded text). Pure: the
    /// caller resolves the flag (no env read here) so both the build-time write
    /// and the open-time expected-fingerprint computation pass the same value.
    #[must_use]
    pub fn compute(id: &str, spec: &EmbedderSpec, dense_context: bool) -> String {
        // Canonical, stable encoding of the vector-space-defining fields
        // (including BOTH prefixes — doc_prefix changes the stored doc vectors —
        // and the dense_context flag, which changes the embedded text). Avoid
        // serde here so a future non-fingerprinted field (e.g. max_context, which
        // does NOT change the vector space) can be added to EmbedderSpec without
        // silently invalidating every index.
        let pooling = match spec.pooling {
            Pooling::Mean => "mean",
            Pooling::Cls => "cls",
            Pooling::LateInteraction => "late_interaction",
        };
        let quant = match spec.quant {
            QuantKind::None => "none",
            QuantKind::Int8Symmetric => "int8_symmetric",
        };
        let ctx = u8::from(dense_context);
        let canonical = format!(
            "id={id};dims={};pooling={pooling};quant={quant};norm={};qpre={};dpre={};ctx={ctx}",
            spec.dims, spec.normalize, spec.query_prefix, spec.doc_prefix
        );
        let hash = xxhash_rust::xxh64::xxh64(canonical.as_bytes(), 0);
        format!("{hash:016x}")
    }
}

fn default_true() -> bool {
    true
}

/// Default reranker `max_context` (bge-reranker-v2-m3's trained context length).
fn default_reranker_max_context() -> usize {
    512
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
            static_token_table: None,
            frozen_centroids: None,
        };
        // The fields are data, not behavior — a spec is fully described here.
        assert_eq!(e.dims, 768);
        assert_eq!(e.pooling, Pooling::Cls);
        assert_eq!(e.quant, QuantKind::Int8Symmetric);
        assert!(e.query_prefix.ends_with(' '));
    }

    #[test]
    fn fingerprint_is_stable_and_sensitive() {
        let base = EmbedderSpec {
            dims: 768,
            max_context: 8192,
            query_prefix: "Q: ".to_string(),
            doc_prefix: String::new(),
            pooling: Pooling::Cls,
            normalize: true,
            quant: QuantKind::Int8Symmetric,
            static_token_table: None,
            frozen_centroids: None,
        };
        let fp1 = EmbedderFingerprint::compute("coderank-137m", &base, false);
        let fp2 = EmbedderFingerprint::compute("coderank-137m", &base, false);
        assert_eq!(fp1, fp2, "same id+spec → same fingerprint (deterministic)");
        assert!(!fp1.is_empty());

        // Changing dims changes the fingerprint (vector space differs).
        let mut diff_dims = base.clone();
        diff_dims.dims = 384;
        assert_ne!(
            fp1,
            EmbedderFingerprint::compute("coderank-137m", &diff_dims, false)
        );

        // Changing pooling changes it.
        let mut diff_pool = base.clone();
        diff_pool.pooling = Pooling::Mean;
        assert_ne!(
            fp1,
            EmbedderFingerprint::compute("coderank-137m", &diff_pool, false)
        );

        // Changing quant changes it.
        let mut diff_quant = base.clone();
        diff_quant.quant = QuantKind::None;
        assert_ne!(
            fp1,
            EmbedderFingerprint::compute("coderank-137m", &diff_quant, false)
        );

        // Changing the id changes it (different model, same shape).
        assert_ne!(
            fp1,
            EmbedderFingerprint::compute("other-embedder", &base, false)
        );
    }

    #[test]
    fn fingerprint_is_sensitive_to_dense_context() {
        // The `dense_context` flag changes the EMBEDDED TEXT (annotation+code vs
        // raw code) → it changes the stored vector space. ctx=on and ctx=off on
        // the SAME spec MUST get different fingerprints, so the engine never
        // silently reuses an index embedded under the other setting (the literal
        // bug F5 fixes). All other fields held equal.
        let base = EmbedderSpec {
            dims: 768,
            max_context: 8192,
            query_prefix: "Q: ".to_string(),
            doc_prefix: String::new(),
            pooling: Pooling::Cls,
            normalize: true,
            quant: QuantKind::Int8Symmetric,
            static_token_table: None,
            frozen_centroids: None,
        };
        let off = EmbedderFingerprint::compute("coderank-137m", &base, false);
        let on = EmbedderFingerprint::compute("coderank-137m", &base, true);
        assert_ne!(
            off, on,
            "dense_context must be part of the fingerprint (it changes embedded text)"
        );
        // Still deterministic per setting.
        assert_eq!(
            on,
            EmbedderFingerprint::compute("coderank-137m", &base, true)
        );
    }

    #[test]
    fn fingerprint_is_sensitive_to_doc_prefix() {
        // `doc_prefix` is applied to DOCUMENT text at index time → it changes the
        // stored vector space. Two specs differing ONLY in doc_prefix MUST get
        // different fingerprints, else the engine would silently reuse an index
        // embedded under the old prefix.
        let base = EmbedderSpec {
            dims: 768,
            max_context: 8192,
            query_prefix: "Q: ".to_string(),
            doc_prefix: String::new(),
            pooling: Pooling::Cls,
            normalize: true,
            quant: QuantKind::Int8Symmetric,
            static_token_table: None,
            frozen_centroids: None,
        };
        let mut diff_doc = base.clone();
        diff_doc.doc_prefix = "passage: ".to_string();
        assert_ne!(
            EmbedderFingerprint::compute("coderank-137m", &base, false),
            EmbedderFingerprint::compute("coderank-137m", &diff_doc, false),
            "doc_prefix must be part of the fingerprint"
        );
    }

    /// Helper struct so the role test can deserialize a bare `role = "…"`.
    #[derive(Deserialize)]
    struct RoleHolder {
        role: ModelRole,
    }

    #[test]
    fn embedder_aux_artifact_fields_default_to_none_and_roundtrip() {
        let s = crate::model::manifest::builtin_specs()
            .into_iter()
            .find(|s| s.id == "coderank-137m")
            .unwrap();
        let RoleData::Embedder(e) = &s.role_data else {
            panic!()
        };
        assert!(e.static_token_table.is_none());
        assert!(e.frozen_centroids.is_none());

        // Real TOML roundtrip: a minimal embedder table that omits both aux
        // artifact keys entirely must still deserialize with both as None.
        let doc = r#"
            dims = 768
            max_context = 8192
            pooling = "cls"
            quant = "int8_symmetric"
        "#;
        let parsed: EmbedderSpec = toml::from_str(doc).unwrap();
        assert!(parsed.static_token_table.is_none());
        assert!(parsed.frozen_centroids.is_none());
    }

    #[test]
    fn embedder_aux_artifact_fields_reject_path_traversal() {
        let mut s = crate::model::manifest::builtin_specs()
            .into_iter()
            .find(|s| s.id == "lateon-colbert")
            .unwrap();
        if let RoleData::Embedder(e) = &mut s.role_data {
            e.frozen_centroids = Some("../evil.npy".to_string());
        }
        let err = s.validate().unwrap_err();
        assert!(err.to_string().contains("frozen_centroids"), "got: {err}");
    }

    #[test]
    fn aux_artifact_fields_do_not_change_the_fingerprint() {
        let specs = crate::model::manifest::builtin_specs();
        let s = specs.iter().find(|s| s.id == "lateon-colbert").unwrap();
        let RoleData::Embedder(e) = &s.role_data else {
            panic!()
        };
        let mut stripped = e.clone();
        stripped.static_token_table = None;
        stripped.frozen_centroids = None;
        assert_eq!(
            EmbedderFingerprint::compute("lateon-colbert", e, false),
            EmbedderFingerprint::compute("lateon-colbert", &stripped, false),
            "aux artifacts are doc-side only; they must never invalidate an index"
        );
    }
}
