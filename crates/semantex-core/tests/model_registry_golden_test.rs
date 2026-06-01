//! S8 acceptance: capability routing, user-manifest 2nd-model end-to-end,
//! fingerprint stamping, and the clean-mismatch error arm.
//!
//! POST-S2 NOTE: S2 added the `CoderankHnsw` variant to `DenseBackendKind` and
//! totalized the capability→backend conversion, so `embedder_backend_kind()` now
//! resolves a multi-vector embedder to `Ok(ColbertPlaid)` and a single-vector
//! embedder to `Ok(CoderankHnsw)`. These golden tests assert that totalized
//! contract.

use semantex_core::config::SemantexConfig;
use semantex_core::model::ModelRegistry;
use semantex_core::model::capabilities::{BackendKind, backend_for};
use semantex_core::model::spec::RoleData;
use semantex_core::search::dense_backend::DenseBackendKind;
use std::fs;

/// Gate 4 + spec §6: the dense backend the registry resolves for the default
/// multi-vector built-in embedder is IDENTICAL to constructing the
/// `DenseBackendKind` directly from the model's known capability (no quality
/// regression from the indirection).
#[test]
fn registry_resolved_backend_equals_direct_for_default() {
    // lateon-colbert (multi-vector) → PLAID, both ways.
    let mut cfg = SemantexConfig::default();
    cfg.embedder = "lateon-colbert".to_string();
    let reg = ModelRegistry::from_config(&cfg, None).unwrap();
    let resolved = reg.embedder_backend_kind().unwrap();
    // Direct construction: a multi-vector capability maps to PLAID.
    let direct = backend_for(&reg.active_embedder().unwrap().capabilities)
        .dense_kind()
        .unwrap();
    assert_eq!(resolved, direct);
    assert_eq!(resolved, DenseBackendKind::ColbertPlaid);
    assert_eq!(
        BackendKind::ColbertPlaid.dense_kind().unwrap(),
        DenseBackendKind::ColbertPlaid
    );
}

/// Gate 4 (post-S2): a single-vector built-in (coderank-137m) RESOLVES from its
/// spec and capability-routes to the coderank-hnsw backend, now totally mapped
/// to S1's `DenseBackendKind`. Proves the negotiation is capability-driven (not
/// id-branching).
#[test]
fn single_vector_routes_to_hnsw_backendkind() {
    let mut cfg = SemantexConfig::default();
    cfg.embedder = "coderank-137m".to_string();
    let reg = ModelRegistry::from_config(&cfg, None).unwrap();
    // Capability negotiation picks the coderank-hnsw BackendKind from the spec.
    assert_eq!(
        backend_for(&reg.active_embedder().unwrap().capabilities),
        BackendKind::CoderankHnsw
    );
    // …and the S1 DenseBackendKind conversion is now total (post-S2).
    assert_eq!(
        reg.embedder_backend_kind().unwrap(),
        DenseBackendKind::CoderankHnsw
    );
}

/// Gate 3: a user `models.toml` adding a SECOND permissive embedder loads and
/// capability-routes end-to-end with NO code change. The 2nd model is multi-
/// vector so it routes to a representable `DenseBackendKind` pre-S2.
#[test]
fn user_manifest_second_model_loads_and_routes() {
    let tmp = tempfile::TempDir::new().unwrap();
    let project = tmp.path().join("proj");
    let semantex_dir = project.join(".semantex");
    fs::create_dir_all(&semantex_dir).unwrap();
    // A 2nd permissive multi-vector (late-interaction) embedder.
    fs::write(
        semantex_dir.join("models.toml"),
        r#"
        [[model]]
        id = "my-colbert-variant"
        role = "embedder"
        [model.source]
        kind = "hf"
        repo = "example/my-colbert-variant"
        files = ["model_int8.onnx", "tokenizer.json"]
        [model.capabilities]
        multi_vector = true
        [model.embedder]
        dims = 96
        max_context = 512
        query_prefix = ""
        pooling = "late_interaction"
        quant = "int8_symmetric"
        "#,
    )
    .unwrap();

    let mut cfg = SemantexConfig::default();
    cfg.embedder = "my-colbert-variant".to_string();
    let reg = ModelRegistry::from_config(&cfg, Some(&project)).unwrap();

    // It resolves…
    let spec = reg.active_embedder().unwrap();
    assert_eq!(spec.id, "my-colbert-variant");
    let RoleData::Embedder(_) = &spec.role_data else {
        panic!()
    };
    // …and capability-routes to PLAID from the spec alone (multi_vector=true).
    assert_eq!(
        reg.embedder_backend_kind().unwrap(),
        DenseBackendKind::ColbertPlaid
    );
}

/// Gate 3 (single-vector arm): a user `models.toml` single-vector embedder also
/// LOADS and capability-routes (to coderank-hnsw) from the manifest alone, and
/// (post-S2) the S1 backend-kind conversion is total.
#[test]
fn user_manifest_single_vector_model_capability_routes() {
    let tmp = tempfile::TempDir::new().unwrap();
    let project = tmp.path().join("proj");
    let semantex_dir = project.join(".semantex");
    fs::create_dir_all(&semantex_dir).unwrap();
    // A 2nd permissive single-vector embedder (gte-modernbert-base, Apache-2.0).
    fs::write(
        semantex_dir.join("models.toml"),
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

    let mut cfg = SemantexConfig::default();
    cfg.embedder = "gte-modernbert-hnsw".to_string();
    let reg = ModelRegistry::from_config(&cfg, Some(&project)).unwrap();

    let spec = reg.active_embedder().unwrap();
    assert_eq!(spec.id, "gte-modernbert-hnsw");
    // Capability-routes to the coderank-hnsw BackendKind from the manifest alone.
    assert_eq!(backend_for(&spec.capabilities), BackendKind::CoderankHnsw);
    // Post-S2: the DenseBackendKind conversion is total.
    assert_eq!(
        reg.embedder_backend_kind().unwrap(),
        DenseBackendKind::CoderankHnsw
    );
}

/// Gate 1 (config-only reranker/llm swap): swapping the reranker by config alone
/// changes the resolved spec with no recompile and no index touched.
#[test]
fn swapping_reranker_by_config_changes_resolved_spec() {
    use semantex_core::model::spec::ScoreStrategyKind;

    let mut cfg = SemantexConfig::default();
    // Default reranker → classifier head.
    let reg = ModelRegistry::from_config(&cfg, None).unwrap();
    let RoleData::Reranker(r) = &reg.active_reranker().unwrap().role_data else {
        panic!()
    };
    assert_eq!(r.score_strategy, ScoreStrategyKind::ClassifierHead);

    // Swap by config alone → yes/no generative reranker, different strategy.
    cfg.reranker_model = "qwen3-reranker-0.6b".to_string();
    let reg = ModelRegistry::from_config(&cfg, None).unwrap();
    let spec = reg.active_reranker().unwrap();
    assert_eq!(spec.id, "qwen3-reranker-0.6b");
    let RoleData::Reranker(r) = &spec.role_data else {
        panic!()
    };
    assert_eq!(r.score_strategy, ScoreStrategyKind::YesNoLogit);
}

/// Gate 2 (error arm): a mismatched on-disk fingerprint errors cleanly.
#[test]
fn mismatched_index_errors_cleanly() {
    use semantex_core::search::dense_backend::verify_persisted_fingerprint_matches;
    use semantex_core::types::IndexMeta;
    let tmp = tempfile::TempDir::new().unwrap();
    let index_dir = tmp.path().join(".semantex");
    fs::create_dir_all(&index_dir).unwrap();
    let meta = IndexMeta {
        schema_version: IndexMeta::CURRENT_SCHEMA_VERSION,
        project_path: index_dir.clone(),
        created_at: "0".to_string(),
        updated_at: "0".to_string(),
        file_count: 0,
        chunk_count: 0,
        embedding_model: "lateon-colbert".to_string(),
        embedding_dim: 48,
        use_bm25_stemmer: true,
        dense_backend: "colbert-plaid".to_string(),
        embedder_fingerprint: "BUILT_WITH_THIS".to_string(),
    };
    fs::write(
        index_dir.join("meta.json"),
        serde_json::to_string(&meta).unwrap(),
    )
    .unwrap();
    let err = verify_persisted_fingerprint_matches(&index_dir, "ACTIVE_IS_DIFFERENT")
        .expect_err("a different active embedder must error");
    assert!(
        err.to_string().contains("--rebuild"),
        "error must guide the user"
    );
}

/// Write a tiny synthetic repo so a real index can be built without network
/// (the lateon-colbert path needs the ColBERT model; gate the model-dependent
/// assertion with `#[ignore]`). This test only inspects meta.json shape.
fn write_tiny_repo(dir: &std::path::Path) {
    fs::create_dir_all(dir).unwrap();
    fs::write(
        dir.join("lib.rs"),
        "pub fn add(a: i32, b: i32) -> i32 { a + b }\n",
    )
    .unwrap();
}

#[test]
#[ignore = "builds a real dense index (needs the embedder model download)"]
fn build_stamps_a_nonempty_embedder_fingerprint() {
    use semantex_core::index::builder::IndexBuilder;
    use semantex_core::types::IndexMeta;

    let tmp = tempfile::TempDir::new().unwrap();
    let project = tmp.path().join("proj");
    write_tiny_repo(&project);
    // Build with the shipped DEFAULT embedder (lateon-colbert / colbert-plaid),
    // the only path representable pre-S2.
    let cfg = SemantexConfig::default();
    IndexBuilder::new(&cfg).unwrap().build(&project).unwrap();

    let meta_str = fs::read_to_string(project.join(".semantex").join("meta.json")).unwrap();
    let meta: IndexMeta = serde_json::from_str(&meta_str).unwrap();
    assert!(
        !meta.embedder_fingerprint.is_empty(),
        "builder must stamp the embedder fingerprint"
    );
}
