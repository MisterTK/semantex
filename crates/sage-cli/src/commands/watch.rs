use anyhow::{Context, Result};
use colored::Colorize;
use sage_core::config::SageConfig;
use sage_core::file::watcher::FileWatcher;
use sage_core::index::updater::IndexUpdater;
use sage_core::search::hybrid::HybridSearcher;
use sage_core::server::listener::Listener;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

pub fn run(path: &Path, config: &SageConfig) -> Result<()> {
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
    match IndexUpdater::update(&project_path, config) {
        Ok(stats) => {
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
    let index_dir = SageConfig::project_index_dir(&project_path);
    let socket_path = index_dir.join("sage.sock");
    let shutdown = Arc::new(AtomicBool::new(false));

    // Open searcher for daemon
    match HybridSearcher::open(&index_dir, config) {
        Ok(searcher) => {
            let shutdown_clone = shutdown.clone();
            let socket_path_clone = socket_path.clone();
            std::thread::spawn(move || {
                match Listener::bind(
                    &socket_path_clone,
                    searcher,
                    Duration::from_secs(86400), // 24h timeout (watch keeps it alive)
                    shutdown_clone,
                ) {
                    Ok(listener) => {
                        if let Err(e) = listener.run() {
                            tracing::error!("Daemon listener error: {}", e);
                        }
                    }
                    Err(e) => {
                        tracing::error!("Failed to start daemon listener: {}", e);
                    }
                }
            });

            println!(
                "  {} Search daemon started on {}",
                "OK".green(),
                socket_path.display()
            );
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

    // Start watching
    let mut watcher = FileWatcher::new()?;
    let rx = watcher.watch(&project_path)?;

    while let Ok(changed_paths) = rx.recv() {
        if shutdown.load(Ordering::Relaxed) {
            break;
        }

        let count = changed_paths.len();
        println!("{} {} file(s) changed, re-indexing...", "->".cyan(), count);
        for p in &changed_paths {
            println!("   {}", p.display().to_string().dimmed());
        }

        match IndexUpdater::update(&project_path, config) {
            Ok(stats) => {
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
    let _ = std::fs::remove_file(&socket_path);

    Ok(())
}
