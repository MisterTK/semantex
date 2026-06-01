use anyhow::{Context, Result};
use colored::Colorize;
use semantex_core::config::SemantexConfig;
use semantex_core::embedding::single_vector_model;
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
