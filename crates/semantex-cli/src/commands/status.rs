use anyhow::{Context, Result};
use colored::Colorize;
use semantex_core::config::SemantexConfig;
use semantex_core::embedding::{model_manager, single_vector_model};
use semantex_core::index::layout;
use semantex_core::index::registry;
use semantex_core::index::storage::ChunkStore;
use semantex_core::model::registry::ModelRegistry;
use semantex_core::search::dense_backend::DenseBackendKind;
use semantex_core::types::IndexMeta;
use std::path::Path;

/// Human-readable dense-backend name + late-interaction/single-vector
/// descriptor, matching the strings `IndexBuilder` stamps into `meta.json`
/// (`crates/semantex-core/src/index/builder.rs`) and the model registry's
/// built-in specs (`crates/semantex-core/src/model/manifest.rs`).
fn dense_backend_display(kind: DenseBackendKind) -> (&'static str, &'static str) {
    match kind {
        DenseBackendKind::ColbertPlaid => ("LateOn-Code-edge", "late-interaction"),
        DenseBackendKind::CoderankHnsw => ("CodeRankEmbed", "single-vector"),
    }
}

/// Whether `models_dir` has every file the named dense model needs, per the
/// model-specific provisioners (`model_manager::is_colbert_downloaded`,
/// `single_vector_model::is_coderank_downloaded`). Unrecognized names (e.g. a
/// project `models.toml` embedder not in the two shipped built-ins) report
/// `None` — status can't say downloaded/not without a matching provisioner.
fn dense_model_downloaded(name: &str, models_dir: &Path) -> Option<bool> {
    match name {
        "LateOn-Code-edge" => Some(model_manager::is_colbert_downloaded(models_dir)),
        "CodeRankEmbed" => Some(single_vector_model::is_coderank_downloaded(models_dir)),
        _ => None,
    }
}

/// Resolve which dense backend is ACTIVE and its (name, dim, pooling
/// descriptor), preferring the on-disk index's `meta.json` (the ground truth
/// for what THIS project was actually built with) and falling back to what
/// the current config would select for a fresh build, resolved via
/// [`ModelRegistry`] (never hardcoded — this affects only what gets printed,
/// not S8 embedder selection itself).
fn resolve_active_dense(
    meta: Option<&IndexMeta>,
    config: &SemantexConfig,
    project_path: &Path,
) -> (String, u32, &'static str) {
    match meta {
        Some(m) => {
            let descriptor = match m.dense_backend.as_str() {
                "colbert-plaid" => "late-interaction",
                "coderank-hnsw" => "single-vector",
                _ => "unknown pooling",
            };
            (m.embedding_model.clone(), m.embedding_dim, descriptor)
        }
        None => match ModelRegistry::from_config(config, Some(project_path))
            .ok()
            .and_then(|reg| reg.embedder_backend_kind().ok())
        {
            Some(kind) => {
                let (name, descriptor) = dense_backend_display(kind);
                let dim = match kind {
                    DenseBackendKind::ColbertPlaid => 48,
                    DenseBackendKind::CoderankHnsw => {
                        semantex_core::embedding::single_vector::SingleVectorEmbedder::embedding_dim(
                        ) as u32
                    }
                };
                (name.to_string(), dim, descriptor)
            }
            None => ("unknown (embedder config error)".to_string(), 0, ""),
        },
    }
}

/// Render the `Models:` section: ONE unambiguous line naming the ACTIVE dense
/// backend (the model this project's queries/index actually use), plus a
/// note for any OTHER known dense model that happens to be downloaded but is
/// NOT the active one — so "downloaded" never gets mistaken for "in use".
/// Returned as a string (rather than printed directly) so the line-selection
/// logic is unit-testable without capturing stdout.
fn format_models_section(
    meta: Option<&IndexMeta>,
    config: &SemantexConfig,
    project_path: &Path,
) -> String {
    use std::fmt::Write as _;

    let models_dir = config.models_dir();
    let (active_name, active_dim, descriptor) = resolve_active_dense(meta, config, project_path);

    let mut out = format!("{}\n", "Models:".bold());
    let status = match dense_model_downloaded(&active_name, &models_dir) {
        Some(true) => "downloaded".green().to_string(),
        Some(false) => "not downloaded".red().to_string(),
        None => "status unknown".yellow().to_string(),
    };
    if active_dim > 0 {
        let _ = writeln!(
            out,
            "  Dense: {active_name} ({active_dim}-dim {descriptor}): {status}"
        );
    } else {
        let _ = writeln!(out, "  Dense: {active_name}");
    }

    // Only mention a non-active model when it's actually downloaded — an
    // undownloaded, unused model is not worth a line.
    for other_name in ["LateOn-Code-edge", "CodeRankEmbed"] {
        if other_name == active_name {
            continue;
        }
        if dense_model_downloaded(other_name, &models_dir) == Some(true) {
            let _ = writeln!(
                out,
                "  {other_name} (dense, not active for this project): downloaded"
            );
        }
    }
    out
}

pub fn run(path: &Path, config: &SemantexConfig) -> Result<()> {
    let project_path = path
        .canonicalize()
        .with_context(|| format!("Invalid path: {}", path.display()))?;

    println!("{}", "semantex status".bold());
    println!();

    let index_dir = SemantexConfig::project_index_dir(&project_path);
    let meta_path = index_dir.join("meta.json");

    // Read once up front so both the `Models:` section (needs the active
    // backend, if an index already recorded one) and the `Index:` section
    // below share the same parse — no risk of the two disagreeing.
    let meta_read: Option<(String, Option<IndexMeta>)> = if meta_path.exists() {
        let content = std::fs::read_to_string(&meta_path)?;
        let parsed = serde_json::from_str::<IndexMeta>(&content).ok();
        Some((content, parsed))
    } else {
        None
    };
    let meta_opt: Option<&IndexMeta> = meta_read.as_ref().and_then(|(_, m)| m.as_ref());

    print!("{}", format_models_section(meta_opt, config, &project_path));
    println!();

    // Index status
    println!("{}", "Index:".bold());

    if let Some((content, parsed)) = meta_read {
        let Some(meta) = parsed else {
            let stale_version = serde_json::from_str::<serde_json::Value>(&content)
                .ok()
                .and_then(|v| v.get("schema_version").and_then(serde_json::Value::as_u64));
            let version_label =
                stale_version.map_or_else(|| "unreadable".to_string(), |v| format!("v{v}"));
            println!(
                "  {} Index at {} is stale (current schema: v{}).",
                "!".yellow(),
                version_label,
                IndexMeta::CURRENT_SCHEMA_VERSION
            );
            println!(
                "  Run 'semantex index {}' to rebuild (auto-rebuild also triggers on first search).",
                path.display()
            );
            println!();
            return Ok(());
        };

        // Read live counts from SQLite (meta.json may be stale after incremental indexing)
        let db_path = index_dir.join("chunks.db");
        let (file_count, chunk_count) = if db_path.exists() {
            let store = ChunkStore::open(&db_path)?;
            (
                store.file_count().unwrap_or(meta.file_count),
                store.chunk_count().unwrap_or(meta.chunk_count),
            )
        } else {
            (meta.file_count, meta.chunk_count)
        };

        println!("  Project:    {}", meta.project_path.display());
        println!("  Files:      {}", file_count.to_string().cyan());
        println!("  Chunks:     {}", chunk_count.to_string().cyan());
        println!("  Model:      {}", meta.embedding_model);
        println!("  Dimensions: {}", meta.embedding_dim);
        println!("  Updated:    {}", meta.updated_at);
        println!("  Index path: {}", index_dir.display().to_string().dimmed());

        // Check component files
        let has_dense = index_dir.join("dense").exists();
        let has_sparse = index_dir.join("sparse").exists();
        let has_chunks = index_dir.join("chunks.db").exists();
        println!();
        println!("  Components:");
        println!(
            "    Dense (HNSW): {}",
            if has_dense {
                "present".green()
            } else {
                "missing".red()
            }
        );
        println!(
            "    Sparse (BM25): {}",
            if has_sparse {
                "present".green()
            } else {
                "missing".red()
            }
        );
        println!(
            "    Chunk store:   {}",
            if has_chunks {
                "present".green()
            } else {
                "missing".red()
            }
        );
    } else {
        println!(
            "  {} No index found for {}",
            "!".yellow(),
            project_path.display()
        );
        println!("  Run 'semantex index {}' to build one.", path.display());
    }

    print_branches_section(&project_path);

    println!();
    println!("{}", "Config:".bold());
    println!("  Max results:    {}", config.max_count);
    println!("  Max file size:  {} bytes", config.max_file_size);
    println!("  Max file count: {}", config.max_file_count);
    println!("  Chunk size:     {} tokens", config.chunk_size);
    println!("  Chunk overlap:  {} tokens", config.chunk_overlap);
    println!(
        "  Reranking:      {}",
        if config.rerank {
            "enabled".green()
        } else {
            "disabled".yellow()
        }
    );

    Ok(())
}

/// Wave 2: list the branches the registry has tracked as indexed for this
/// project (`branch`, `branch_key`, last-indexed age, current-or-not).
/// Additive-only — printed after the existing `Index:` block so this stays
/// backward compatible with anything scraping the earlier output.
fn print_branches_section(project_path: &Path) {
    let Some(entry) = registry::read_all_v2()
        .projects
        .into_iter()
        .find(|p| p.path == project_path)
    else {
        return;
    };
    if entry.branches.is_empty() {
        return;
    }

    let current_key = layout::current_branch_key(project_path);
    let mut branches = entry.branches;
    branches.sort_by_key(|b| std::cmp::Reverse(b.last_indexed_ts));

    println!();
    println!("{}", "Branches:".bold());
    for b in &branches {
        let marker = if b.branch_key == current_key {
            " (current)".green().to_string()
        } else {
            String::new()
        };
        let age = format_unix_ts_age(b.last_indexed_ts);
        let commit = b
            .head_commit
            .as_deref()
            .map_or_else(|| "unknown".to_string(), |c| c.chars().take(8).collect());
        println!(
            "  {}{}: indexed {} ago, commit {}",
            b.branch, marker, age, commit
        );
    }
}

/// Render "N seconds/minutes/hours/days ago" from a Unix-seconds timestamp,
/// reusing the same coarse buckets `state::index_age_secs`/`format_age`
/// (semantex-mcp) already display elsewhere in this CLI's output.
fn format_unix_ts_age(unix_ts: i64) -> String {
    if unix_ts <= 0 {
        return "unknown".to_string();
    }
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs() as i64);
    let secs = (now - unix_ts).max(0);
    if secs < 60 {
        format!("{secs}s")
    } else if secs < 3600 {
        format!("{}m", secs / 60)
    } else if secs < 86400 {
        format!("{}h", secs / 3600)
    } else {
        format!("{}d", secs / 86400)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use semantex_core::types::IndexMeta;
    use std::path::PathBuf;

    fn meta_with(embedding_model: &str, embedding_dim: u32, dense_backend: &str) -> IndexMeta {
        IndexMeta {
            schema_version: IndexMeta::CURRENT_SCHEMA_VERSION,
            project_path: PathBuf::from("/tmp/project"),
            created_at: "2026-01-01T00:00:00Z".to_string(),
            updated_at: "2026-01-01T00:00:00Z".to_string(),
            file_count: 1,
            chunk_count: 1,
            embedding_model: embedding_model.to_string(),
            embedding_dim,
            use_bm25_stemmer: true,
            dense_backend: dense_backend.to_string(),
            embedder_fingerprint: "test".to_string(),
        }
    }

    #[test]
    fn dense_backend_display_names_match_meta_json_convention() {
        // These strings must match what `IndexBuilder` stamps into meta.json
        // (crates/semantex-core/src/index/builder.rs) so the on-disk-index
        // path and the no-index-yet fallback path agree on naming.
        assert_eq!(
            dense_backend_display(DenseBackendKind::ColbertPlaid),
            ("LateOn-Code-edge", "late-interaction")
        );
        assert_eq!(
            dense_backend_display(DenseBackendKind::CoderankHnsw),
            ("CodeRankEmbed", "single-vector")
        );
    }

    #[test]
    fn resolve_active_dense_prefers_index_meta_over_config() {
        // Even if the runtime config's default embedder is lateon-colbert, an
        // existing index built with CodeRankEmbed must report CodeRankEmbed —
        // meta.json is the ground truth for what THIS project actually has.
        let meta = meta_with("CodeRankEmbed", 768, "coderank-hnsw");
        let config = SemantexConfig::default();
        let tmp = tempfile::tempdir().unwrap();
        let (name, dim, descriptor) = resolve_active_dense(Some(&meta), &config, tmp.path());
        assert_eq!(name, "CodeRankEmbed");
        assert_eq!(dim, 768);
        assert_eq!(descriptor, "single-vector");
    }

    #[test]
    fn resolve_active_dense_falls_back_to_registry_without_index() {
        // No index yet: fall back to what the config would build (the shipped
        // default, lateon-colbert -> LateOn-Code-edge / 48-dim / late-interaction).
        let config = SemantexConfig::default();
        let tmp = tempfile::tempdir().unwrap();
        let (name, dim, descriptor) = resolve_active_dense(None, &config, tmp.path());
        assert_eq!(name, "LateOn-Code-edge");
        assert_eq!(dim, 48);
        assert_eq!(descriptor, "late-interaction");
    }

    #[test]
    fn format_models_section_names_active_backend_unambiguously() {
        // The core regression this fixes: a single, unambiguous "Dense:" line
        // naming the model actually in use for this project, not a bare
        // "CodeRankEmbed (dense): downloaded" line that says nothing about
        // whether CodeRankEmbed is the active embedder.
        let meta = meta_with("LateOn-Code-edge", 48, "colbert-plaid");
        let config = SemantexConfig::default();
        let tmp = tempfile::tempdir().unwrap();
        let out = format_models_section(Some(&meta), &config, tmp.path());
        assert!(
            out.contains("Dense: LateOn-Code-edge (48-dim late-interaction):"),
            "got: {out}"
        );
    }

    #[test]
    fn format_models_section_notes_inactive_but_downloaded_model() {
        // If CodeRankEmbed's files happen to be present in models_dir (e.g. a
        // prior opt-in) but this project's active backend is LateOn-Code-edge,
        // the section must call CodeRankEmbed out as downloaded-but-inactive,
        // never implying it's the model in use.
        let meta = meta_with("LateOn-Code-edge", 48, "colbert-plaid");
        let mut config = SemantexConfig::default();
        let tmp = tempfile::tempdir().unwrap();
        config.model_dir = Some(tmp.path().to_path_buf());

        // Stand up just enough of a CodeRankEmbed download to satisfy
        // is_coderank_downloaded's file-presence check.
        let coderank_dir = tmp.path().join("CodeRankEmbed");
        std::fs::create_dir_all(&coderank_dir).unwrap();
        for f in [
            "model_int8.onnx",
            "model_int8.onnx.data",
            "tokenizer.json",
            "config.json",
        ] {
            std::fs::write(coderank_dir.join(f), b"stub").unwrap();
        }

        let out = format_models_section(Some(&meta), &config, tmp.path());
        assert!(
            out.contains("Dense: LateOn-Code-edge (48-dim late-interaction):"),
            "got: {out}"
        );
        assert!(
            out.contains("CodeRankEmbed (dense, not active for this project): downloaded"),
            "got: {out}"
        );
    }

    #[test]
    fn dense_model_downloaded_unknown_name_reports_none() {
        let tmp = tempfile::tempdir().unwrap();
        assert_eq!(
            dense_model_downloaded("some-custom-embedder", tmp.path()),
            None
        );
    }
}
