//! `semantex skills-generate` — produces per-platform skill files from one
//! canonical tool registry.
//!
//! See `crates/semantex-cli/src/skills/` for the canonical metadata and
//! per-platform formatters.

use crate::skills::{Platform, tools};
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

/// Default output directory: `./.semantex-skills/` under the cwd.
pub const DEFAULT_OUT_DIR: &str = "./.semantex-skills";

/// Run `semantex skills-generate`.
///
/// `platform_filter` is `None` (or `"all"`) for every supported platform, or
/// a platform id like `"claude-code"` for a single target.
///
/// Returns `Ok(())` on full success, `Err(...)` on partial failure (per-file
/// errors are also written to stderr so the operator can see every miss).
pub fn run(out: Option<PathBuf>, platform_filter: Option<&str>, force: bool) -> Result<()> {
    let out_dir = out.unwrap_or_else(|| PathBuf::from(DEFAULT_OUT_DIR));

    let platforms: Vec<Platform> = match platform_filter {
        None | Some("all") => Platform::ALL.to_vec(),
        Some(id) => {
            let p = Platform::from_id(id).with_context(|| {
                format!(
                    "Unknown platform `{}`. Known platforms: {}",
                    id,
                    Platform::ALL
                        .iter()
                        .map(|p| p.id())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            })?;
            vec![p]
        }
    };

    let tool_set = tools::all_tools();

    std::fs::create_dir_all(&out_dir)
        .with_context(|| format!("Failed to create output directory `{}`", out_dir.display()))?;

    let mut failures: Vec<String> = Vec::new();
    let mut wrote: Vec<PathBuf> = Vec::new();

    for platform in platforms {
        let rel = platform.relative_path();
        let target = out_dir.join(rel);
        match generate_one(platform, &tool_set, &target, force) {
            Ok(()) => wrote.push(target),
            Err(e) => {
                let msg = format!("[{}] {}: {e:#}", platform.id(), target.display());
                eprintln!("semantex skills-generate: {msg}");
                failures.push(msg);
            }
        }
    }

    println!(
        "semantex skills-generate: wrote {} file{} under {}",
        wrote.len(),
        if wrote.len() == 1 { "" } else { "s" },
        out_dir.display()
    );
    for path in &wrote {
        println!("  - {}", path.display());
    }

    if !failures.is_empty() {
        anyhow::bail!(
            "{} platform target{} failed; see stderr above",
            failures.len(),
            if failures.len() == 1 { "" } else { "s" }
        );
    }

    Ok(())
}

fn generate_one(
    platform: Platform,
    tool_set: &[tools::ToolMetadata],
    target: &Path,
    force: bool,
) -> Result<()> {
    if let Some(parent) = target.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create parent directory `{}`", parent.display()))?;
    }
    if target.exists() && !force {
        anyhow::bail!(
            "Target file already exists (use --force to overwrite): {}",
            target.display()
        );
    }
    let body = platform.render(tool_set);
    std::fs::write(target, body)
        .with_context(|| format!("Failed to write `{}`", target.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn generates_all_platforms_into_tempdir() {
        let dir = tempdir().expect("tempdir");
        let out = dir.path().join("skills");
        run(Some(out.clone()), None, false).expect("generation should succeed");
        for platform in Platform::ALL {
            let path = out.join(platform.relative_path());
            assert!(
                path.exists(),
                "expected output file `{}` for platform {}",
                path.display(),
                platform.id()
            );
            let meta = std::fs::metadata(&path).expect("metadata");
            assert!(
                meta.len() > 0,
                "file `{}` should be non-empty",
                path.display()
            );
        }
    }

    #[test]
    fn single_platform_filter() {
        let dir = tempdir().expect("tempdir");
        let out = dir.path().join("skills");
        run(Some(out.clone()), Some("claude-code"), false)
            .expect("single platform generation should succeed");
        let target = out.join(Platform::ClaudeCode.relative_path());
        assert!(target.exists());
        // Cursor was not requested.
        let cursor = out.join(Platform::Cursor.relative_path());
        assert!(!cursor.exists());
    }

    #[test]
    fn refuses_overwrite_without_force() {
        let dir = tempdir().expect("tempdir");
        let out = dir.path().join("skills");
        run(Some(out.clone()), Some("claude-code"), false).expect("first run");
        let err = run(Some(out.clone()), Some("claude-code"), false)
            .expect_err("second run without --force should fail");
        let msg = format!("{err:#}");
        assert!(msg.contains("failed"), "got: {msg}");
    }

    #[test]
    fn force_overwrites() {
        let dir = tempdir().expect("tempdir");
        let out = dir.path().join("skills");
        run(Some(out.clone()), Some("claude-code"), false).expect("first run");
        run(Some(out.clone()), Some("claude-code"), true).expect("force should overwrite");
    }

    #[test]
    fn unknown_platform_errors() {
        let dir = tempdir().expect("tempdir");
        let out = dir.path().join("skills");
        let err = run(Some(out), Some("nonexistent-platform"), false)
            .expect_err("unknown platform should error");
        let msg = format!("{err:#}");
        assert!(msg.contains("Unknown platform"), "got: {msg}");
    }
}
