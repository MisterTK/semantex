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

    // Wave 2 batch 2: this is now only the STARTUP-time reconcile — it still
    // runs a full incremental update before the daemon ever opens its first
    // searcher, so day-one startup after a branch switch immediately serves
    // fresh results (leaving the root's `chunks.db` written for the wrong
    // branch would mean the daemon serves mismatched results until the
    // per-request hook below happens to trigger). The daemon no longer needs
    // a RESTART to notice a `git switch` made mid-session, though: `Listener`
    // now resolves its searcher per-request through a
    // `server::cache::SearcherCache`, which runs the SAME
    // `branches::branch_switch_pending` / `detect_and_handle_branch_switch`
    // check on every request (see that module's docs for the full flow,
    // including the residual "SnapshottedOutgoing" narrow window it
    // documents) and separately reloads when the index is rebuilt beneath it
    // (e.g. `semantex index` run again in another terminal) — the daemon no
    // longer pins one `HybridSearcher` for its whole lifetime.
    //
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
