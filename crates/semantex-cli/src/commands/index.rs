use anyhow::{Context, Result};
use colored::Colorize;
use semantex_core::config::SemantexConfig;
use semantex_core::index::branches;
use semantex_core::index::builder::IndexBuilder;
use semantex_core::index::registry;
use std::path::Path;

pub fn run(path: &Path, config: &SemantexConfig, force: bool) -> Result<()> {
    let project_path = path
        .canonicalize()
        .with_context(|| format!("Invalid path: {}", path.display()))?;

    if registry::is_system_temp_root(&project_path) {
        anyhow::bail!(
            "Refusing to index {} — it's the system temp directory, not a project. \
             Every tool on the machine writes here, so it can never stay indexed. \
             Index a specific subdirectory instead.",
            project_path.display()
        );
    }

    if !force && registry::is_likely_multi_repo_container(&project_path) {
        anyhow::bail!(
            "Refusing to index {} — it looks like a workspace containing several \
             independent repos (no .git here, but multiple nested repos found), not \
             a single project. Index each repo individually, or re-run with --force \
             to index it anyway as one combined project.",
            project_path.display()
        );
    }

    println!("{} {}", "Indexing".green().bold(), project_path.display());

    // Wave 2: reconcile a branch switch (restore an existing snapshot, or
    // snapshot the outgoing branch) BEFORE building, so the incremental
    // build below only has to catch drift since the branch was last
    // indexed instead of re-embedding the whole tree.
    match branches::detect_and_handle_branch_switch(&project_path) {
        Ok(branches::BranchSwitchAction::Restored {
            from_branch_key,
            to_branch_key,
        }) => {
            println!(
                "{} branch switch {} -> {}: restored existing index snapshot",
                "Detected".cyan(),
                from_branch_key,
                to_branch_key
            );
        }
        Ok(branches::BranchSwitchAction::SnapshottedOutgoing {
            from_branch_key,
            to_branch_key,
        }) => {
            println!(
                "{} branch switch {} -> {}: snapshotted outgoing branch",
                "Detected".cyan(),
                from_branch_key,
                to_branch_key
            );
        }
        Ok(_) => {}
        Err(e) => {
            eprintln!(
                "  {} Branch switch check failed: {}",
                "Warning:".yellow(),
                e
            );
        }
    }

    let builder = IndexBuilder::new(config)?;
    let stats = builder.build(&project_path)?;

    // Register project in global registry for cross-repo status tracking.
    registry::register(&project_path);
    // Wave 2: record this branch as indexed (registry `branches[]`) and
    // enforce the per-project snapshot retention cap.
    branches::record_branch_indexed(&project_path);

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

    let index_dir = SemantexConfig::project_index_dir(&project_path);
    println!(
        "  Index path:     {}",
        index_dir.display().to_string().dimmed()
    );

    Ok(())
}
