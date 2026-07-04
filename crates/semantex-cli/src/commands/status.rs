use anyhow::{Context, Result};
use colored::Colorize;
use semantex_core::config::SemantexConfig;
use semantex_core::embedding::single_vector_model;
use semantex_core::index::layout;
use semantex_core::index::registry;
use semantex_core::index::storage::ChunkStore;
use semantex_core::types::IndexMeta;
use std::path::Path;

pub fn run(path: &Path, config: &SemantexConfig) -> Result<()> {
    let project_path = path
        .canonicalize()
        .with_context(|| format!("Invalid path: {}", path.display()))?;

    println!("{}", "semantex status".bold());
    println!();

    // Model status
    println!("{}", "Models:".bold());
    let models_dir = config.models_dir();

    let embedder_status = if single_vector_model::is_coderank_downloaded(&models_dir) {
        "downloaded".green().to_string()
    } else {
        "not downloaded".red().to_string()
    };
    println!("  CodeRankEmbed (dense): {embedder_status}");
    println!();

    // Index status
    println!("{}", "Index:".bold());
    let index_dir = SemantexConfig::project_index_dir(&project_path);
    let meta_path = index_dir.join("meta.json");

    if meta_path.exists() {
        let content = std::fs::read_to_string(&meta_path)?;
        let Ok(meta) = serde_json::from_str::<IndexMeta>(&content) else {
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
    branches.sort_by(|a, b| b.last_indexed_ts.cmp(&a.last_indexed_ts));

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
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
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
