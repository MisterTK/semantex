use anyhow::{Context, Result};
use colored::Colorize;
use semantex_core::config::SemantexConfig;
use semantex_core::file::watcher::FileWatcher;
use semantex_core::index::branches;
use semantex_core::index::layout;
use semantex_core::index::updater::IndexUpdater;
use semantex_core::search::hybrid::HybridSearcher;
use semantex_core::server::listener::Listener;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

/// Run branch-switch detection/reconciliation, printing a note when a switch
/// was found. Cheap no-op when the branch hasn't changed (`Unchanged`) or
/// there's no prior recorded branch (`FirstBuild`) — safe to call before
/// every re-index in the watch loop, not just the initial one.
fn reconcile_branch_switch(project_path: &Path) {
    match branches::detect_and_handle_branch_switch(project_path) {
        Ok(branches::BranchSwitchAction::Restored {
            from_branch_key,
            to_branch_key,
        }) => {
            println!(
                "{} branch switch {} -> {}: restored existing index snapshot",
                "->".cyan(),
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
                "->".cyan(),
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
}

pub fn run(path: &Path, config: &SemantexConfig) -> Result<()> {
    let project_path = path
        .canonicalize()
        .with_context(|| format!("Invalid path: {}", path.display()))?;

    println!(
        "{} {} for changes...",
        "Watching".green().bold(),
        project_path.display()
    );
    println!("Press Ctrl+C to stop.");
    println!();

    // Do an initial index
    println!("{}", "Running initial index...".dimmed());
    reconcile_branch_switch(&project_path);
    match IndexUpdater::update(&project_path, config) {
        Ok(stats) => {
            branches::record_branch_indexed(&project_path);
            println!(
                "  Initial index: {} files, {} chunks",
                stats.files_indexed, stats.chunks_created
            );
        }
        Err(e) => {
            eprintln!("  {} Initial index failed: {}", "Warning:".yellow(), e);
        }
    }
    println!();

    // Start the search daemon in a background thread
    let index_dir = SemantexConfig::project_index_dir(&project_path);
    let port_file = index_dir.join("semantex.port");
    let shutdown = Arc::new(AtomicBool::new(false));

    // Open searcher for daemon
    match HybridSearcher::open(&index_dir, config) {
        Ok(searcher) => {
            let shutdown_clone = shutdown.clone();
            let port_file_clone = port_file.clone();
            let project_path_clone = project_path.clone();
            // Spec L Item 1.4: mirror the LLM init from `semantex serve` so
            // `semantex watch`'s daemon also routes via the LLM classifier
            // when SEMANTEX_LLM_* env vars are set. Errors degrade silently
            // to the keyword classifier — never block the watch loop.
            #[cfg(feature = "llm")]
            let llm_backend = semantex_core::llm::LlmBackend::from_env()
                .unwrap_or_else(|e| {
                    tracing::warn!("LLM backend init failed: {e}; disabling LLM features");
                    None
                })
                .map(|b| {
                    let cap = b.into_arc();
                    tracing::info!("LLM enabled: {}", cap.label());
                    cap
                });
            std::thread::spawn(move || {
                match Listener::bind(
                    &port_file_clone,
                    searcher,
                    project_path_clone,
                    Duration::from_hours(24), // 24h timeout (watch keeps it alive)
                    shutdown_clone,
                ) {
                    Ok(listener) => {
                        #[cfg(feature = "llm")]
                        let listener = listener.with_llm(llm_backend);
                        println!(
                            "  {} Search daemon started on 127.0.0.1:{}",
                            "OK".green(),
                            listener.port()
                        );
                        if let Err(e) = listener.run() {
                            tracing::error!("Daemon listener error: {}", e);
                        }
                    }
                    Err(e) => {
                        tracing::error!("Failed to start daemon listener: {}", e);
                    }
                }
            });
        }
        Err(e) => {
            eprintln!(
                "  {} Could not start search daemon: {}",
                "Warning:".yellow(),
                e
            );
        }
    }
    println!();

    // Start watching. Also watch the directory CONTAINING the resolved git
    // HEAD file (which for a linked worktree lives outside project_path, so
    // the recursive watch above wouldn't see it) so a branch switch that
    // touches no tracked file (e.g. `git switch -c` from an identical tree)
    // still triggers a wake-up — see `reconcile_branch_switch`.
    //
    // The PARENT directory, not the HEAD file itself: git replaces HEAD via
    // write-tmp-then-rename, and inotify watches the inode — a watch on the
    // file itself fires once for the first switch and is then auto-removed
    // with the old inode, so every later switch would go unseen. A
    // non-recursive directory watch survives renames of its children.
    let mut watcher = FileWatcher::new()?;
    let rx = watcher.watch(&project_path)?;
    if let Some(head_file) = layout::git_head_file(&project_path)
        && let Some(git_dir) = head_file.parent()
        && let Err(e) = watcher.watch_additional(git_dir, false)
    {
        tracing::debug!("Could not watch git dir ({}): {e}", git_dir.display());
    }

    while let Ok(changed_paths) = rx.recv() {
        if shutdown.load(Ordering::Relaxed) {
            break;
        }

        let count = changed_paths.len();
        println!("{} {} file(s) changed, re-indexing...", "->".cyan(), count);
        for p in &changed_paths {
            println!("   {}", p.display().to_string().dimmed());
        }

        // Cheap on every iteration; only does real work when HEAD has
        // actually moved to a different branch since the root was last
        // synced (see module doc / `branches::detect_and_handle_branch_switch`).
        reconcile_branch_switch(&project_path);

        match IndexUpdater::update(&project_path, config) {
            Ok(stats) => {
                branches::record_branch_indexed(&project_path);
                println!(
                    "   {} Re-indexed {} files, {} chunks in {:.1}s",
                    "OK".green(),
                    stats.files_indexed,
                    stats.chunks_created,
                    stats.duration.as_secs_f64()
                );

                // Notify the daemon to reload (the daemon's Tantivy reader needs a reload)
                // We send a health check which doesn't reload, but the searcher's
                // sparse index will pick up changes on the next reader reload.
                // For now, the watch loop and daemon share the same index files,
                // and the searcher will re-read on next query via Tantivy's mmap.
            }
            Err(e) => {
                eprintln!("   {} Re-index failed: {}", "Error:".red(), e);
            }
        }
        println!();
    }

    // Clean up daemon
    shutdown.store(true, Ordering::Relaxed);
    let _ = std::fs::remove_file(&port_file);

    Ok(())
}
