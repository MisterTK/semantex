use anyhow::{Context, Result};
use colored::Colorize;
use sage_core::config::SageConfig;
use sage_core::embedding::model_manager;
use sage_core::index::storage::ChunkStore;
use sage_core::types::IndexMeta;
use std::path::Path;

pub fn run(path: &Path, config: &SageConfig) -> Result<()> {
    let project_path = path
        .canonicalize()
        .with_context(|| format!("Invalid path: {}", path.display()))?;

    println!("{}", "sage status".bold());
    println!();

    // Model status
    println!("{}", "Models:".bold());
    let models_dir = config.models_dir();

    let colbert_status = if model_manager::is_colbert_downloaded(&models_dir) {
        "downloaded".green().to_string()
    } else {
        "not downloaded".red().to_string()
    };
    println!("  LateOn-Code-edge (ColBERT): {colbert_status}");
    println!();

    // Index status
    println!("{}", "Index:".bold());
    let index_dir = SageConfig::project_index_dir(&project_path);
    let meta_path = index_dir.join("meta.json");

    if meta_path.exists() {
        let content = std::fs::read_to_string(&meta_path)?;
        let meta: IndexMeta = serde_json::from_str(&content)?;

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
        let has_plaid = index_dir.join("plaid").exists();
        let has_sparse = index_dir.join("sparse").exists();
        let has_chunks = index_dir.join("chunks.db").exists();
        println!();
        println!("  Components:");
        println!(
            "    Dense (PLAID): {}",
            if has_plaid {
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
        println!("  Run 'sage index {}' to build one.", path.display());
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
