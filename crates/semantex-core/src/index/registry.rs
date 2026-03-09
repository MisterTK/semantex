//! Global project registry — tracks all repos that have been indexed.
//!
//! Stored at `~/.semantex/projects.json` as a JSON array of canonical absolute paths.
//! Both the CLI session hook and the MCP server read this to discover repos that may
//! have drifted (index age > threshold) without waiting for a user to open them.

use std::path::{Path, PathBuf};

fn registry_path() -> Option<PathBuf> {
    dirs::home_dir().map(|h| h.join(".semantex").join("projects.json"))
}

/// Read all registered project paths from the registry.
pub fn read_all() -> Vec<PathBuf> {
    let Some(path) = registry_path() else {
        return vec![];
    };
    let Ok(content) = std::fs::read_to_string(&path) else {
        return vec![];
    };
    serde_json::from_str::<Vec<String>>(&content)
        .unwrap_or_default()
        .into_iter()
        .map(PathBuf::from)
        .collect()
}

/// Register a project in the global registry (upsert — no duplicates).
pub fn register(canonical: &Path) {
    let Some(path) = registry_path() else {
        return;
    };
    let canonical_str = canonical.to_string_lossy().to_string();
    let mut entries: Vec<String> = {
        std::fs::read_to_string(&path)
            .ok()
            .and_then(|c| serde_json::from_str(&c).ok())
            .unwrap_or_default()
    };
    if entries.contains(&canonical_str) {
        return; // already registered
    }
    entries.push(canonical_str);
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(json) = serde_json::to_string(&entries) {
        let _ = std::fs::write(&path, json);
    }
}
