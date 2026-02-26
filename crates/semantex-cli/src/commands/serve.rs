use anyhow::{Context, Result};
use colored::Colorize;
use semantex_core::config::SemantexConfig;
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
