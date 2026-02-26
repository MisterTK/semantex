use anyhow::{Context, Result};
use colored::Colorize;
use std::path::Path;

pub fn run(path: &Path) -> Result<()> {
    let project_path = path
        .canonicalize()
        .with_context(|| format!("Invalid path: {}", path.display()))?;

    if semantex_core::server::stop_daemon(&project_path)? {
        eprintln!(
            "{} daemon for {}",
            "Stopped".green().bold(),
            project_path.display()
        );
    } else {
        eprintln!(
            "{} No daemon running for {}",
            "Note:".yellow(),
            project_path.display()
        );
    }

    Ok(())
}
