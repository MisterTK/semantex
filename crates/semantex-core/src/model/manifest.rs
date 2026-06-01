//! Built-in permissive model specs + user `models.toml` merge.
//!
//! Built-ins are MIT/Apache only (CLAUDE.md permissive-only rule). They encode
//! the model nuances S2/S3 recorded (CodeRankEmbed dim/prefix/pooling;
//! Qwen3-Reranker yes/no template) as DATA so the engine never special-cases an
//! id. A user `models.toml` (project `.semantex/` over `~/.semantex/`) adds or
//! overrides specs by id with no code change.

use crate::config::SemantexConfig;
use crate::model::capabilities::ModelCapabilities;
use crate::model::spec::{
    EmbedderSpec, ModelRole, ModelSource, ModelSpec, Pooling, QuantKind, RerankerSpec, RoleData,
    ScoreStrategyKind,
};
use anyhow::Result;
use serde::Deserialize;
use std::path::{Path, PathBuf};

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
                ..ModelCapabilities::default()
            },
            role_data: RoleData::Embedder(EmbedderSpec {
                dims: 768,
                max_context: 8192,
                // RECORDED EXACT, trailing space included.
                query_prefix: "Represent this query for searching relevant code: ".to_string(),
                doc_prefix: String::new(),
                // RECORDED: mean (mask-weighted) pooling, NOT CLS — see
                // research-notes (S2 — CodeRankEmbed). The single_vector.rs
                // encoder mean-pools to match.
                pooling: Pooling::Mean,
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
                ..ModelCapabilities::default()
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
        // Qwen3-Embedding-0.6B (Apache-2.0) — the 2026-SOTA single-vector A/B
        // candidate (semantica review). Single-vector → capabilities route it to
        // the `coderank-hnsw` BACKEND exactly like coderank-137m (no S2 dep). The
        // base Qwen repo coordinates are carried as DATA (S8 downloads nothing).
        ModelSpec {
            id: "qwen3-embed-0.6b".to_string(),
            role: ModelRole::Embedder,
            source: ModelSource::Hf {
                repo: "Qwen/Qwen3-Embedding-0.6B".to_string(),
                files: vec![
                    "model_int8.onnx".to_string(),
                    "tokenizer.json".to_string(),
                    "config.json".to_string(),
                ],
            },
            capabilities: ModelCapabilities {
                multi_vector: false,
                instruction_aware: true,
                ..ModelCapabilities::default()
            },
            role_data: RoleData::Embedder(EmbedderSpec {
                dims: 1024,
                max_context: 32768,
                // Qwen3-Embedding uses an instruction-style query prefix; the
                // task instruction is tuned for code retrieval here. Documents
                // get NO prefix (Qwen3-Embedding convention).
                query_prefix:
                    "Instruct: Given a code search query, retrieve relevant code snippets.\nQuery: "
                        .to_string(),
                doc_prefix: String::new(),
                pooling: Pooling::Mean,
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
                // bge-reranker-v2-m3 is trained at 512 tokens; truncate there.
                max_context: 512,
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
                // Cap the prompt-wrapped total so the O(seq^2) 0.6B model stays
                // CPU-sane. Qwen3-Reranker's trained context is large, but for a
                // code-search reranker a 2048-token window (query + chunk +
                // chat-control wrapper) is ample and bounds latency/memory.
                max_context: 2048,
                // VERBATIM verified chat template (S3 Spike 1, CPU-verified
                // P(yes|relevant)=0.990; recorded in research-notes ## S3). The
                // <|im_start|>/<|im_end|> control tokens and the trailing
                // assistant/<think> terminator are load-bearing: omitting them
                // collapses P(yes) and silently breaks the YesNoLogit score.
                // prefix = system block + user header up to "<Query>: " (the
                // instruction is baked in since PromptTemplate has no separate
                // instruction slot); middle injects the document; suffix closes
                // the user turn and opens the assistant turn so the next-token
                // logit is the yes/no judgment.
                prompt_prefix: concat!(
                    "<|im_start|>system\n",
                    "Judge whether the Document meets the requirements based on the Query and the Instruct provided. ",
                    "Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n",
                    "<|im_start|>user\n",
                    "<Instruct>: Given a code search query, judge whether the document is relevant.\n",
                    "<Query>: "
                )
                .to_string(),
                prompt_middle: "\n<Document>: ".to_string(),
                prompt_suffix: concat!(
                    "<|im_end|>\n",
                    "<|im_start|>assistant\n",
                    "<think>\n\n</think>\n\n"
                )
                .to_string(),
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

/// LLM-role built-ins, compiled ONLY with the `llm` feature so the default
/// build pulls zero LLM deps. These are inert until `SEMANTEX_LLM_MODEL` selects
/// one by id — no LLM runs by default. Users override/add via `models.toml`.
///
/// CLAUDE.md rule #6 forbids hardcoded model names/providers/endpoints as a
/// *default selection*; here they are merely available manifest data (the user
/// opts in by id), never the active model unless explicitly selected.
#[cfg(feature = "llm")]
fn append_builtin_llm_specs(specs: &mut Vec<ModelSpec>) {
    use crate::model::spec::LlmSpec;
    // An airgap-friendly Ollama default (local endpoint). Provider/model/endpoint
    // are manifest DATA the user can override; nothing here forces a network LLM.
    specs.push(ModelSpec {
        id: "ollama-default".to_string(),
        role: ModelRole::Llm,
        // LLM weights are not fetched by semantex (genai/Ollama manage them);
        // a Local source with the conventional Ollama dir documents that.
        source: ModelSource::Local {
            dir: "ollama".to_string(),
        },
        capabilities: ModelCapabilities {
            instruction_aware: true,
            ..ModelCapabilities::default()
        },
        role_data: RoleData::Llm(LlmSpec {
            provider: "ollama".to_string(),
            model: "qwen2.5-coder:7b".to_string(),
            endpoint: String::new(),
        }),
    });
}

/// Wire shape of a `models.toml` document: a `[[model]]` array of specs.
#[derive(Debug, Deserialize)]
struct UserManifest {
    #[serde(default)]
    model: Vec<ModelSpec>,
}

/// Parse + validate a user `models.toml`. Each spec is validated; the first
/// invalid one aborts with an error naming the file and the offending field, so
/// a misconfiguration never silently mis-loads (risk row "model-manifest
/// misconfiguration").
pub fn load_user_manifest(path: &Path) -> Result<Vec<ModelSpec>> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("failed to read model manifest {}: {e}", path.display()))?;
    // Probe the RAW document first: serde's externally-tagged `RoleData` keeps
    // exactly one role subtable and silently discards any others, so a stray
    // second table would be a silent mis-load of user input. Reject != 1 role
    // table per `[[model]]` (naming the id + the tables found) before the typed
    // parse collapses the information.
    check_one_role_table_per_model(&text, path)?;
    let manifest: UserManifest = toml::from_str(&text)
        .map_err(|e| anyhow::anyhow!("failed to parse model manifest {}: {e}", path.display()))?;
    for spec in &manifest.model {
        spec.validate()
            .map_err(|e| anyhow::anyhow!("invalid model in {}: {e}", path.display()))?;
    }
    Ok(manifest.model)
}

/// The role-subtable keys an externally-tagged [`RoleData`] can present.
const ROLE_TABLE_KEYS: [&str; 3] = ["embedder", "reranker", "llm"];

/// Reject any `[[model]]` entry that carries more or fewer than one of the role
/// subtables {`embedder`, `reranker`, `llm`}. serde would keep one and drop the
/// rest — a silent mis-load — so we count them on the raw `toml::Value` and emit
/// a clear error naming the model id and the offending tables.
fn check_one_role_table_per_model(text: &str, path: &Path) -> Result<()> {
    let raw: toml::Value = toml::from_str(text)
        .map_err(|e| anyhow::anyhow!("failed to parse model manifest {}: {e}", path.display()))?;
    let Some(models) = raw.get("model").and_then(toml::Value::as_array) else {
        return Ok(()); // no `[[model]]` array → nothing to check
    };
    for (idx, model) in models.iter().enumerate() {
        let present: Vec<&str> = ROLE_TABLE_KEYS
            .iter()
            .copied()
            .filter(|k| model.get(*k).is_some())
            .collect();
        if present.len() != 1 {
            let id = model
                .get("id")
                .and_then(toml::Value::as_str)
                .map_or_else(|| format!("#{idx}"), |s| format!("`{s}`"));
            anyhow::bail!(
                "model {id} in {}: expected exactly one role table \
                 ({}), found {} ({:?})",
                path.display(),
                ROLE_TABLE_KEYS.join(", "),
                present.len(),
                present,
            );
        }
    }
    Ok(())
}

/// Locate the active user manifest: a project-local `<project>/.semantex/models.toml`
/// takes precedence over the global `~/.semantex/models.toml`. Returns `None`
/// when neither exists (the registry then runs on built-ins only).
pub fn user_manifest_path(project_path: Option<&Path>) -> Option<PathBuf> {
    if let Some(project) = project_path {
        let local = SemantexConfig::project_index_dir(project).join("models.toml");
        if local.exists() {
            return Some(local);
        }
    }
    let global = SemantexConfig::semantex_home().join("models.toml");
    if global.exists() {
        return Some(global);
    }
    None
}

/// Merge built-in and user specs, with user specs overriding built-ins **by id**
/// (the modularity guarantee: replace a built-in's data, or add a brand-new
/// model, without touching code). Order: built-ins first (in declaration
/// order), then any user ids not already present.
#[must_use]
pub fn merge(builtin: Vec<ModelSpec>, user: Vec<ModelSpec>) -> Vec<ModelSpec> {
    let mut out = builtin;
    for u in user {
        if let Some(existing) = out.iter_mut().find(|s| s.id == u.id) {
            *existing = u; // override by id
        } else {
            out.push(u); // new id → append
        }
    }
    out
}

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
        assert!(
            ids.contains(&"qwen3-reranker-0.6b"),
            "missing qwen3: {ids:?}"
        );
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
        // RECORDED: mean (mask-weighted) pooling, NOT CLS.
        assert_eq!(e.pooling, Pooling::Mean);
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
    fn qwen3_embed_resolves_validates_and_routes_to_hnsw() {
        use crate::model::capabilities::{BackendKind, backend_for};
        let s = builtin_specs()
            .into_iter()
            .find(|s| s.id == "qwen3-embed-0.6b")
            .expect("qwen3-embed-0.6b must be a built-in");
        assert_eq!(s.role, ModelRole::Embedder);
        s.validate().expect("qwen3-embed-0.6b must validate");
        let RoleData::Embedder(e) = &s.role_data else {
            panic!("qwen3-embed-0.6b must be an embedder");
        };
        assert_eq!(e.dims, 1024);
        assert_eq!(e.pooling, Pooling::Mean);
        assert!(e.normalize);
        assert_eq!(e.quant, QuantKind::Int8Symmetric);
        // Single-vector → routes to the coderank-hnsw backend (no S2 dep).
        assert!(!s.capabilities.multi_vector);
        assert!(s.capabilities.instruction_aware);
        assert_eq!(backend_for(&s.capabilities), BackendKind::CoderankHnsw);
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

    #[test]
    fn load_user_manifest_parses_a_second_embedder() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("models.toml");
        std::fs::write(
            &path,
            r#"
            [[model]]
            id = "gte-modernbert-hnsw"
            role = "embedder"
            [model.source]
            kind = "hf"
            repo = "Alibaba-NLP/gte-modernbert-base"
            files = ["model_int8.onnx", "tokenizer.json"]
            [model.capabilities]
            multi_vector = false
            [model.embedder]
            dims = 768
            max_context = 8192
            query_prefix = ""
            pooling = "cls"
            quant = "int8_symmetric"
            "#,
        )
        .unwrap();
        let specs = load_user_manifest(&path).unwrap();
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].id, "gte-modernbert-hnsw");
        assert_eq!(specs[0].role, ModelRole::Embedder);
        specs[0].validate().unwrap();
    }

    #[test]
    fn load_user_manifest_errors_clearly_on_bad_spec() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("models.toml");
        // Embedder with dims=0 → validate() must reject, naming the field.
        std::fs::write(
            &path,
            r#"
            [[model]]
            id = "broken"
            role = "embedder"
            [model.source]
            kind = "hf"
            repo = "x/y"
            files = ["model_int8.onnx"]
            [model.embedder]
            dims = 0
            max_context = 8192
            pooling = "mean"
            quant = "none"
            "#,
        )
        .unwrap();
        let err = load_user_manifest(&path).expect_err("dims=0 must error");
        let msg = err.to_string();
        assert!(msg.contains("broken") && msg.contains("dims"), "got: {msg}");
    }

    #[test]
    fn load_user_manifest_errors_on_multiple_role_tables() {
        // A `[[model]]` with role="embedder" but BOTH an [model.embedder] and a
        // stray [model.reranker] table: serde would silently keep one and drop
        // the other. The loader must reject it (naming the id + the tables), so
        // user input is never silently mis-loaded.
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("models.toml");
        std::fs::write(
            &path,
            r#"
            [[model]]
            id = "two-tables"
            role = "embedder"
            [model.source]
            kind = "hf"
            repo = "x/y"
            files = ["model_int8.onnx"]
            [model.embedder]
            dims = 768
            max_context = 8192
            pooling = "cls"
            quant = "int8_symmetric"
            [model.reranker]
            score_strategy = "classifier_head"
            "#,
        )
        .unwrap();
        let err = load_user_manifest(&path).expect_err("two role tables must error");
        let msg = err.to_string();
        assert!(msg.contains("two-tables"), "must name the model id: {msg}");
        assert!(
            msg.contains("embedder") && msg.contains("reranker"),
            "must name the conflicting tables: {msg}"
        );
    }

    #[test]
    fn merge_lets_user_override_a_builtin_by_id() {
        let builtin = builtin_specs();
        // A user spec re-using the `coderank-137m` id overrides the built-in.
        let mut overridden = builtin
            .iter()
            .find(|s| s.id == "coderank-137m")
            .cloned()
            .unwrap();
        if let RoleData::Embedder(e) = &mut overridden.role_data {
            e.max_context = 4096; // user shrinks the context window
        }
        let merged = merge(builtin.clone(), vec![overridden]);
        // Same count (override, not append).
        assert_eq!(merged.len(), builtin.len());
        let s = merged.iter().find(|s| s.id == "coderank-137m").unwrap();
        let RoleData::Embedder(e) = &s.role_data else {
            panic!()
        };
        assert_eq!(e.max_context, 4096, "user override must win");
    }

    #[test]
    fn merge_appends_a_new_user_id() {
        let builtin = builtin_specs();
        let mut newspec = builtin
            .iter()
            .find(|s| s.id == "coderank-137m")
            .cloned()
            .unwrap();
        newspec.id = "my-custom-embedder".to_string();
        let merged = merge(builtin.clone(), vec![newspec]);
        assert_eq!(merged.len(), builtin.len() + 1);
        assert!(merged.iter().any(|s| s.id == "my-custom-embedder"));
    }
}
