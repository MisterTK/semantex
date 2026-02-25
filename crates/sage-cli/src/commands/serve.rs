use anyhow::{Context, Result};
use colored::Colorize;
use sage_core::config::SageConfig;
use sage_core::server::SageServer;
use std::path::Path;

pub fn run(path: &Path, config: &SageConfig) -> Result<()> {
    let project_path = path
        .canonicalize()
        .with_context(|| format!("Invalid path: {}", path.display()))?;

    // Check if daemon is already running
    if sage_core::server::daemon_healthy(&project_path) {
        eprintln!(
            "{} Daemon already running for {}",
            "Note:".yellow(),
            project_path.display()
        );
        return Ok(());
    }

    let server = SageServer::new(&project_path, config);

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
