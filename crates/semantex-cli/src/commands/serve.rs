use anyhow::{Context, Result};
use colored::Colorize;
use semantex_core::config::SemantexConfig;
use semantex_core::index::branches;
use semantex_core::server::SemantexServer;
use std::path::Path;

pub fn run(path: &Path, config: &SemantexConfig) -> Result<()> {
    let project_path = path
        .canonicalize()
        .with_context(|| format!("Invalid path: {}", path.display()))?;

    // Prevent redundant daemons: check if one is fully up or still starting.
    // Multiple subagents/sessions racing to start a daemon will each call
    // `semantex serve`; the second+ arrivals should exit immediately.
    let already_healthy = semantex_core::server::daemon_healthy(&project_path);
    let already_starting =
        !already_healthy && semantex_core::server::daemon_starting(&project_path);
    if already_healthy || already_starting {
        eprintln!(
            "{} Daemon already {} for {}",
            "Note:".yellow(),
            if already_starting {
                "starting"
            } else {
                "running"
            },
            project_path.display()
        );
        return Ok(());
    }

    // Wave 2 — KNOWN LIMITATION: branch-switch reconciliation happens at
    // daemon STARTUP ONLY. The daemon opens one `HybridSearcher` and serves
    // it for its whole lifetime (up to 24h); `Listener::run`'s accept loop
    // has no searcher-reopen hook, and mutating the index files under a live
    // open searcher is exactly the torn-read/WAL-replay hazard the
    // branch-switch handling locks against — so a `git switch` made while
    // the daemon is running is NOT picked up by the daemon itself. It IS
    // picked up by every other entry point (`semantex index`/`watch`, any
    // MCP session's initialize + per-search checks), each of which
    // reconciles under the exclusive index lock; this daemon sees the new
    // content only after a restart. If `Listener` ever grows a searcher
    // reload mechanism, wire `branches::detect_and_handle_branch_switch`
    // into it.
    //
    // If HEAD moved since the root was last synced, restore/snapshot as
    // usual, then run an incremental update BEFORE opening the searcher —
    // leaving the root's `chunks.db` written for the wrong branch would mean
    // the daemon serves mismatched results for its entire lifetime.
    // Unchanged/first-build is a no-op — `serve` still requires
    // `semantex index` to have run at least once (unchanged behavior).
    match branches::detect_and_handle_branch_switch(&project_path) {
        Ok(action) if action.switched() => {
            eprintln!(
                "{} Branch switch detected — reconciling index before starting daemon...",
                "Note:".cyan()
            );
            match semantex_core::index::updater::IndexUpdater::update(&project_path, config) {
                Ok(_) => branches::record_branch_indexed(&project_path),
                Err(e) => eprintln!(
                    "  {} Incremental re-index after branch switch failed: {}",
                    "Warning:".yellow(),
                    e
                ),
            }
        }
        Ok(_) => {}
        Err(e) => eprintln!(
            "  {} Branch switch check failed: {}",
            "Warning:".yellow(),
            e
        ),
    }

    let server = SemantexServer::new(&project_path, config);

    eprintln!(
        "{} search daemon for {}",
        "Starting".green().bold(),
        project_path.display()
    );
    eprintln!("  Port file: {}", server.port_file_path().display());
    eprintln!("  PID:    {}", std::process::id());
    eprintln!("Press Ctrl+C to stop.");
    eprintln!();

    server.run()
}
