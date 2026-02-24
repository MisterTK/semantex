use anyhow::{Context, Result};
use colored::Colorize;
use sage_core::config::SageConfig;
use sage_core::index::builder::IndexBuilder;
use std::path::Path;

pub fn run(path: &Path, config: &SageConfig) -> Result<()> {
    let project_path = path
        .canonicalize()
        .with_context(|| format!("Invalid path: {}", path.display()))?;

    println!("{} {}", "Indexing".green().bold(), project_path.display());

    let builder = IndexBuilder::new(config)?;
    let stats = builder.build(&project_path)?;

    println!();
    println!("{}", "Index complete!".green().bold());
    println!(
        "  Files scanned:  {}",
        stats.files_scanned.to_string().cyan()
    );
    println!(
        "  Files indexed:  {}",
        stats.files_indexed.to_string().cyan()
    );
    println!(
        "  Files skipped:  {}",
        stats.files_skipped.to_string().yellow()
    );
    if stats.files_deleted > 0 {
        println!(
            "  Files deleted:  {}",
            stats.files_deleted.to_string().yellow()
        );
    }
    println!(
        "  Chunks created: {}",
        stats.chunks_created.to_string().cyan()
    );
    if stats.chunks_removed > 0 {
        println!(
            "  Chunks removed: {}",
            stats.chunks_removed.to_string().yellow()
        );
    }
    println!("  Duration:       {:.1}s", stats.duration.as_secs_f64());

    let index_dir = SageConfig::project_index_dir(&project_path);
    println!(
        "  Index path:     {}",
        index_dir.display().to_string().dimmed()
    );

    Ok(())
}
