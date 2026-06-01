//! Built-in permissive model specs + user `models.toml` merge.
//!
//! Built-ins are MIT/Apache only (CLAUDE.md permissive-only rule). They encode
//! the model nuances S2/S3 recorded (CodeRankEmbed dim/prefix/pooling;
//! Qwen3-Reranker yes/no template) as DATA so the engine never special-cases an
//! id. A user `models.toml` (project `.semantex/` over `~/.semantex/`) adds or
//! overrides specs by id with no code change.

use crate::model::capabilities::ModelCapabilities;
use crate::model::spec::{
    EmbedderSpec, ModelRole, ModelSource, ModelSpec, Pooling, QuantKind, RerankerSpec, RoleData,
    ScoreStrategyKind,
};

/// The compiled-in permissive default specs. MIT/Apache only.
///
/// Embedder nuance (dim/prefix/pooling/quant) quotes S2 Spike 1's recorded
/// values; reranker template/token ids quote S3 Spike 1's recorded values. These
/// are semantex's own built-ins — they are NOT repo-specific tuning (CLAUDE.md
/// rule #2). LLM specs are appended feature-gated (Task 11).
pub fn builtin_specs() -> Vec<ModelSpec> {
    let mut specs = vec![
        // ── Embedders ───────────────────────────────────────────────────────
        // CodeRankEmbed 137M (MIT) — the single-vector candidate. The spec id is
        // model-descriptive (`coderank-137m`); its capabilities (single-vector)
        // route it to the `coderank-hnsw` BACKEND (Task 5 negotiation).
        ModelSpec {
            id: "coderank-137m".to_string(),
            role: ModelRole::Embedder,
            source: ModelSource::Hf {
                // RECORDED: hosted int8 export on the HF Hub. The `.onnx.data`
                // external-weights file ships alongside the graph + tokenizer.
                repo: "MisterTK/CodeRankEmbed-onnx-int8".to_string(),
                files: vec![
                    "model_int8.onnx".to_string(),
                    "model_int8.onnx.data".to_string(),
                    "tokenizer.json".to_string(),
                    "config.json".to_string(),
                ],
            },
            capabilities: ModelCapabilities {
                multi_vector: false,
            },
            role_data: RoleData::Embedder(EmbedderSpec {
                dims: 768,
                max_context: 8192,
                // RECORDED EXACT, trailing space included.
                query_prefix: "Represent this query for searching relevant code: ".to_string(),
                doc_prefix: String::new(),
                pooling: Pooling::Cls,
                normalize: true,
                quant: QuantKind::Int8Symmetric,
            }),
        },
        // LateOn-Code-edge ColBERT — today's late-interaction path + shipped
        // default (D4). The spec id is model-descriptive (`lateon-colbert`); its
        // capabilities (multi-vector) route it to the `colbert-plaid` BACKEND.
        ModelSpec {
            id: "lateon-colbert".to_string(),
            role: ModelRole::Embedder,
            source: ModelSource::Hf {
                repo: "lightonai/LateOn-Code-edge".to_string(),
                files: vec![
                    "model_int8.onnx".to_string(),
                    "tokenizer.json".to_string(),
                    "onnx_config.json".to_string(),
                ],
            },
            capabilities: ModelCapabilities {
                multi_vector: true,
            },
            role_data: RoleData::Embedder(EmbedderSpec {
                dims: 48,
                max_context: 512,
                query_prefix: String::new(),
                doc_prefix: String::new(),
                pooling: Pooling::LateInteraction,
                normalize: true,
                quant: QuantKind::Int8Symmetric,
            }),
        },
        // ── Rerankers ───────────────────────────────────────────────────────
        // bge-reranker-v2-m3 (permissive, already shipped) — classifier head.
        ModelSpec {
            id: "bge-reranker-v2-m3".to_string(),
            role: ModelRole::Reranker,
            source: ModelSource::Hf {
                repo: "BAAI/bge-reranker-v2-m3".to_string(),
                files: vec![
                    "model_int8.onnx".to_string(),
                    "tokenizer.json".to_string(),
                    "config.json".to_string(),
                ],
            },
            capabilities: ModelCapabilities::default(),
            role_data: RoleData::Reranker(RerankerSpec {
                score_strategy: ScoreStrategyKind::ClassifierHead,
                prompt_prefix: String::new(),
                prompt_middle: String::new(),
                prompt_suffix: String::new(),
                yes_token_id: None,
                no_token_id: None,
            }),
        },
        // Qwen3-Reranker-0.6B (Apache-2.0) — yes/no generative. The hosted export
        // is fp16 (NOT int8 — the reranker runs at higher precision); yes/no token
        // ids + template are RECORDED in research-notes (S3 Spike 1).
        ModelSpec {
            id: "qwen3-reranker-0.6b".to_string(),
            role: ModelRole::Reranker,
            source: ModelSource::Hf {
                // RECORDED: fp16 ONNX export hosted on the HF Hub.
                repo: "MisterTK/Qwen3-Reranker-0.6B-onnx".to_string(),
                files: vec![
                    "model.onnx".to_string(),
                    "tokenizer.json".to_string(),
                    "config.json".to_string(),
                ],
            },
            capabilities: ModelCapabilities::default(),
            role_data: RoleData::Reranker(RerankerSpec {
                score_strategy: ScoreStrategyKind::YesNoLogit,
                // RECORDED verbatim (S3 Spike 1, prompt template).
                prompt_prefix:
                    "<Instruct>: Given a code search query, judge whether the document is relevant.\n<Query>: "
                        .to_string(),
                prompt_middle: "\n<Document>: ".to_string(),
                prompt_suffix: "\n<Relevant>:".to_string(),
                // RECORDED yes/no token ids (S3 Spike 1).
                yes_token_id: Some(9693),
                no_token_id: Some(2152),
            }),
        },
    ];
    append_builtin_llm_specs(&mut specs);
    specs
}

/// Appends LLM-role built-ins. Inert (no-op) on the default build — LLM specs
/// only exist with the `llm` feature so the default build pulls zero LLM deps
/// (CLAUDE.md rule #8). Defined here so `builtin_specs` is feature-uniform.
#[cfg(not(feature = "llm"))]
fn append_builtin_llm_specs(_specs: &mut Vec<ModelSpec>) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtins_include_both_embedders_and_both_rerankers() {
        let specs = builtin_specs();
        let ids: Vec<&str> = specs.iter().map(|s| s.id.as_str()).collect();
        assert!(
            ids.contains(&"coderank-137m"),
            "missing coderank-137m: {ids:?}"
        );
        assert!(
            ids.contains(&"lateon-colbert"),
            "missing lateon-colbert: {ids:?}"
        );
        assert!(ids.contains(&"bge-reranker-v2-m3"), "missing bge: {ids:?}");
        assert!(ids.contains(&"qwen3-reranker-0.6b"), "missing qwen3: {ids:?}");
    }

    #[test]
    fn coderank_embedder_carries_recorded_nuance() {
        let s = builtin_specs()
            .into_iter()
            .find(|s| s.id == "coderank-137m")
            .unwrap();
        assert_eq!(s.role, ModelRole::Embedder);
        let RoleData::Embedder(e) = &s.role_data else {
            panic!("coderank-137m must be an embedder");
        };
        // S2 Spike 1 recorded values (arctic-embed-m-long base).
        assert_eq!(e.dims, 768);
        assert_eq!(e.pooling, Pooling::Cls);
        assert_eq!(e.quant, QuantKind::Int8Symmetric);
        assert!(
            e.query_prefix.ends_with(' '),
            "recorded prefix keeps trailing space"
        );
        assert!(e.doc_prefix.is_empty(), "documents get no prefix");
        // Single-vector → not multi_vector.
        assert!(!s.capabilities.multi_vector);
    }

    #[test]
    fn colbert_embedder_is_multi_vector_late_interaction() {
        let s = builtin_specs()
            .into_iter()
            .find(|s| s.id == "lateon-colbert")
            .unwrap();
        let RoleData::Embedder(e) = &s.role_data else {
            panic!("lateon-colbert must be an embedder");
        };
        assert_eq!(e.pooling, Pooling::LateInteraction);
        assert!(s.capabilities.multi_vector, "ColBERT is multi-vector");
    }

    #[test]
    fn qwen3_reranker_is_yes_no_with_template() {
        let s = builtin_specs()
            .into_iter()
            .find(|s| s.id == "qwen3-reranker-0.6b")
            .unwrap();
        assert_eq!(s.role, ModelRole::Reranker);
        let RoleData::Reranker(r) = &s.role_data else {
            panic!("qwen3 must be a reranker");
        };
        assert_eq!(r.score_strategy, ScoreStrategyKind::YesNoLogit);
        // YesNoLogit rerankers MUST carry yes/no token ids (filled from the spike).
        assert!(r.yes_token_id.is_some());
        assert!(r.no_token_id.is_some());
    }

    #[test]
    fn bge_reranker_is_classifier_head() {
        let s = builtin_specs()
            .into_iter()
            .find(|s| s.id == "bge-reranker-v2-m3")
            .unwrap();
        let RoleData::Reranker(r) = &s.role_data else {
            panic!("bge must be a reranker");
        };
        assert_eq!(r.score_strategy, ScoreStrategyKind::ClassifierHead);
    }

    #[test]
    fn all_builtins_validate() {
        for s in builtin_specs() {
            s.validate()
                .unwrap_or_else(|e| panic!("builtin {} invalid: {e}", s.id));
        }
    }
}
