use anyhow::{Context, Result};
use colored::Colorize;
use semantex_core::config::SemantexConfig;
use semantex_core::file::watcher::FileWatcher;
use semantex_core::index::updater::IndexUpdater;
use semantex_core::search::hybrid::HybridSearcher;
use semantex_core::server::listener::Listener;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

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
    let index_dir = SemantexConfig::project_index_dir(&project_path);
    let port_file = index_dir.join("semantex.port");
    let shutdown = Arc::new(AtomicBool::new(false));

    // Open searcher for daemon
    match HybridSearcher::open(&index_dir, config) {
        Ok(searcher) => {
            let shutdown_clone = shutdown.clone();
            let port_file_clone = port_file.clone();
            let project_path_clone = project_path.clone();
            std::thread::spawn(move || {
                match Listener::bind(
                    &port_file_clone,
                    searcher,
                    project_path_clone,
                    Duration::from_hours(24), // 24h timeout (watch keeps it alive)
                    shutdown_clone,
                ) {
                    Ok(listener) => {
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
    let _ = std::fs::remove_file(&port_file);

    Ok(())
}
