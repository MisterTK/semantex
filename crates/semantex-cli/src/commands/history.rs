//! `semantex history <file>` — v13 Wave 2. A small, additive CLI surface
//! over `semantex_core::index::history`'s query helpers: a compact table of
//! the commits that touched a file (sha8, age, author, subject), most
//! recent first.
//!
//! Read-only against whatever `history.db` the last `semantex index` build
//! left behind — this command never triggers a build itself. An absent or
//! empty `history.db` is reported plainly rather than treated as an error
//! (a fresh project, a non-git directory, or an index built before history
//! population landed are all ordinary, not broken, states).

use anyhow::{Context, Result};
use colored::Colorize;
use semantex_core::config::SemantexConfig;
use semantex_core::index::history;
use std::path::{Path, PathBuf};

pub fn run(file: &str, path: &Path, limit: usize) -> Result<()> {
    let project_path = path
        .canonicalize()
        .with_context(|| format!("Invalid path: {}", path.display()))?;

    let db_path = SemantexConfig::project_index_dir(&project_path).join("history.db");
    if !db_path.exists() {
        println!(
            "No history.db yet for {}. Run 'semantex index' — history is populated automatically \
             during indexing (git repos only).",
            project_path.display()
        );
        return Ok(());
    }

    let conn = history::open(&project_path)
        .with_context(|| format!("Failed to open {}", db_path.display()))?;

    // Repo-relative path, matching how `file_commits.path` is stored (the
    // indexer strips the project root — see `index::builder::IndexBuilder::build`).
    let rel_path = normalize_query_path(&project_path, file);

    let commits = history::commits_touching_file(&conn, &rel_path, limit)
        .with_context(|| format!("Failed to query history for `{rel_path}`"))?;

    if commits.is_empty() {
        println!("No recorded history for `{rel_path}`.");
        return Ok(());
    }

    println!("{}", format!("History: {rel_path}").bold());
    println!(
        "{:<8}  {:<10}  {:<20}  {}",
        "SHA".dimmed(),
        "AGE".dimmed(),
        "AUTHOR".dimmed(),
        "SUBJECT".dimmed()
    );
    for c in &commits {
        let sha8 = c.hash.chars().take(8).collect::<String>();
        let age = history::humanize_age(c.ts);
        println!(
            "{:<8}  {:<10}  {:<20}  {}",
            sha8.cyan(),
            age,
            truncate(&c.author, 20),
            c.subject
        );
    }
    Ok(())
}

/// Best-effort normalization: if `file` is absolute (or relative to CWD but
/// outside the project root), rewrite it relative to `project_path` when
/// possible; otherwise pass it through unchanged (an already repo-relative
/// path — the common case — needs no rewriting, beyond stripping any `./`
/// prefixes: the index stores plain relative paths, so `./src/lib.rs` from
/// shell tab-completion would otherwise never match).
fn normalize_query_path(project_path: &Path, file: &str) -> String {
    let candidate = PathBuf::from(file);
    if candidate.is_relative() {
        let mut normalized = file.replace('\\', "/");
        while let Some(rest) = normalized.strip_prefix("./") {
            normalized = rest.to_string();
        }
        return normalized;
    }
    candidate
        .canonicalize()
        .ok()
        .and_then(|abs| abs.strip_prefix(project_path).ok().map(Path::to_path_buf))
        .map_or_else(|| file.replace('\\', "/"), |rel| rel.display().to_string())
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let truncated: String = s.chars().take(max.saturating_sub(1)).collect();
    format!("{truncated}…")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_query_path_passes_through_relative_paths() {
        let project = Path::new("/does/not/matter");
        assert_eq!(
            normalize_query_path(project, "src/lib.rs"),
            "src/lib.rs".to_string()
        );
    }

    #[test]
    fn normalize_query_path_strips_dot_slash_prefix() {
        let project = Path::new("/does/not/matter");
        assert_eq!(
            normalize_query_path(project, "./src/lib.rs"),
            "src/lib.rs".to_string()
        );
        assert_eq!(
            normalize_query_path(project, "././a.rs"),
            "a.rs".to_string()
        );
    }

    #[test]
    fn truncate_leaves_short_strings_untouched() {
        assert_eq!(truncate("short", 20), "short");
    }

    #[test]
    fn truncate_shortens_long_strings_with_ellipsis() {
        let long = "a".repeat(30);
        let out = truncate(&long, 10);
        assert_eq!(out.chars().count(), 10);
        assert!(out.ends_with('…'));
    }

    #[test]
    fn run_on_non_git_project_reports_no_history_without_erroring() {
        let tmp = tempfile::TempDir::new().unwrap();
        run("src/lib.rs", tmp.path(), 10).unwrap();
    }

    #[test]
    fn run_with_populated_history_succeeds() {
        let tmp = tempfile::TempDir::new().unwrap();
        let index_dir = tmp.path().join(".semantex");
        std::fs::create_dir_all(&index_dir).unwrap();
        let conn =
            semantex_core::index::layout::open_history_db(&index_dir.join("history.db")).unwrap();
        conn.execute(
            "INSERT INTO commits (hash, author, ts, message) VALUES ('deadbeefcafe', 'Ada', 1700000000, 'Fix bug')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO file_commits (path, hash) VALUES ('src/lib.rs', 'deadbeefcafe')",
            [],
        )
        .unwrap();
        drop(conn);

        run("src/lib.rs", tmp.path(), 10).unwrap();
    }
}
