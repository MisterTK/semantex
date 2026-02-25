use anyhow::{Context, Result};
use std::path::Path;

use crate::client;

/// Start a persistent client connected to the daemon for the given project.
///
/// The persistent client maintains a long-lived TCP connection using the
/// binary protocol, eliminating per-query connection and serialization overhead.
///
/// If a client is already running, this prints its PID and exits.
pub fn run(project_path: &Path) -> Result<()> {
    let project_path = project_path
        .canonicalize()
        .with_context(|| format!("Invalid path: {}", project_path.display()))?;

    // Check if already running
    if let Some(pid) = client::client_alive() {
        eprintln!("Persistent client already running (PID {pid})");
        return Ok(());
    }

    let port = sage_core::server::read_daemon_port(&project_path).context(format!(
        "No daemon running for {}. Start one with 'sage serve {}'.",
        project_path.display(),
        project_path.display()
    ))?;

    // Verify the daemon is reachable via binary protocol
    let mut client_conn =
        client::PersistentClient::connect(port).context("Failed to connect to daemon")?;

    match client_conn.health() {
        Ok(true) => {}
        Ok(false) => anyhow::bail!("Daemon health check returned unhealthy"),
        Err(e) => anyhow::bail!("Daemon health check failed: {e}"),
    }

    // Write PID file
    client::write_client_pid()?;

    eprintln!(
        "Persistent client connected (PID {}, port: {})",
        std::process::id(),
        port
    );
    eprintln!("Binary protocol active. Searches will use fast path.");
    eprintln!("Use 'sage disconnect' to stop.");

    Ok(())
}
