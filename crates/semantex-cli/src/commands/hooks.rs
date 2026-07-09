//! Claude Code hook commands — full lifecycle.
//!
//! SessionStart   → detect index, pre-warm daemon, output context
//! PreToolUse     → Grep|Glob nudge or deny, Bash search nudge or deny, Read nudge or deny (--deny flag)
//! SubagentStart  → inject semantex context into Explore/general-purpose agents
//! SessionEnd     → stop daemon
//!
//! Output format follows the Claude Code hooks spec:
//!   { "hookSpecificOutput": { "hookEventName": "...", "additionalContext": "..." } }

use anyhow::Result;
use semantex_core::config::SemantexConfig;
use semantex_core::index::registry;
use semantex_core::index::state::{self, IndexState};
use std::path::PathBuf;
use std::process::Stdio;

/// Read-hook nudge text — points the agent at the `semantex_agent` MCP tool and
/// keeps the duplicate-re-read framing. Extracted as a const so tests can assert
/// it references `semantex_agent` without touching control flow.
const READ_HOOK_NUDGE: &str = concat!(
    "Use the `semantex_agent` tool (semantic code search) instead of Read for exploring code — ",
    "one call returns a complete answer with file:line + code; it covers regex and file-glob too. ",
    "You may already have this file's content from a prior `semantex_agent` result; ",
    "re-reading is duplicate work.",
);

/// Check if a semantex index directory (`.semantex/meta.json`) exists for the current
/// directory or any parent. Returns the `.semantex` directory path if found.
fn find_index_dir() -> Option<PathBuf> {
    let mut dir = std::env::current_dir().ok()?;
    loop {
        let semantex_dir = dir.join(".semantex");
        if semantex_dir.join("meta.json").exists() {
            return Some(semantex_dir);
        }
        if !dir.pop() {
            return None;
        }
    }
}

/// SessionStart hook — called via `semantex --session-hook`.
///
/// If no index or stale: auto-start background indexing, output context about fallback behavior.
/// If index found & ready: pre-warm daemon + fire-and-forget incremental reindex, output full search instructions.
pub fn cmd_session_hook() -> Result<()> {
    let Some(index_dir) = find_index_dir() else {
        // No index at all — start building one in the background
        if let Ok(cwd) = std::env::current_dir() {
            super::spawn_background_index(&cwd);
        }

        let output = serde_json::json!({
            "hookSpecificOutput": {
                "hookEventName": "SessionStart",
                "additionalContext": concat!(
                    "semantex (semantic code search) is building an index for this project in the background.\n",
                    "Use Grep/Glob for now. Semantic search will be available shortly.\n",
                    "semantex searches will automatically fall back to ripgrep keyword results until the index is ready.",
                ),
            }
        });
        println!("{}", serde_json::to_string(&output)?);
        return Ok(());
    };

    let project_dir = index_dir.parent().unwrap_or(&index_dir);

    // Use state::detect_for_config() to catch stale indexes — schema mismatch AND
    // (S8) an embedder-fingerprint change, so a model/embedding swap auto-rebuilds.
    // The session hook runs once per session, so loading the config here is cheap.
    let idx_state = match SemantexConfig::load(Some(project_dir)) {
        Ok(cfg) => state::detect_for_config(project_dir, &cfg),
        // Config load failed → fall back to schema-only detect (never block on a
        // transient config problem).
        Err(_) => state::detect(project_dir),
    };
    if idx_state == IndexState::Stale {
        super::spawn_background_index(project_dir);

        let output = serde_json::json!({
            "hookSpecificOutput": {
                "hookEventName": "SessionStart",
                "additionalContext": concat!(
                    "semantex (semantic code search) has a stale index — rebuilding in the background.\n",
                    "Use Grep/Glob for now. Semantic search will be available shortly.\n",
                    "semantex searches will automatically fall back to ripgrep keyword results until the index is ready.",
                ),
            }
        });
        println!("{}", serde_json::to_string(&output)?);
        return Ok(());
    }

    // Pre-warm daemon (fire-and-forget, non-blocking)
    let _ = std::process::Command::new(super::self_exe())
        .arg("serve")
        .arg(project_dir)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn(); // semantex serve handles "already running" gracefully

    // Fire-and-forget incremental reindex to pick up changes since last session
    super::spawn_background_index(project_dir);

    // Refresh any other registered repos whose index is older than SEMANTEX_REFRESH_SECS.
    refresh_stale_registry_repos(project_dir);

    let additional_context = format!(
        "semantex semantic code search is available (index: {}).\n\
         Use the `semantex_agent` tool (semantic code search) instead of Grep/Glob/Read — one call \
         returns a complete answer with file:line + code; it covers regex and file-glob too. \
         Trust the result and skip follow-up Reads.\n\
         Fallback: Grep/Glob only if the `semantex_agent` tool is unavailable.",
        index_dir.display()
    );

    let output = serde_json::json!({
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": additional_context,
        }
    });

    println!("{}", serde_json::to_string(&output)?);
    Ok(())
}

/// PreToolUse Grep|Glob hook — called via `semantex --grep-hook [--deny]`.
///
/// Default (no --deny): soft nudge via additionalContext — Grep/Glob still runs.
/// With --deny: hard block via permissionDecision — forces Claude to use semantex.
///
/// If no index: output `{}` (silent — let Grep/Glob work normally)
pub fn cmd_grep_hook(deny: bool) -> Result<()> {
    if find_index_dir().is_none() {
        println!("{{}}");
        return Ok(());
    }

    let nudge = concat!(
        "Use the `semantex_agent` tool (semantic code search) instead of Grep/Glob — ",
        "one call returns a complete answer with file:line + code; it covers regex and ",
        "file-glob too. Use the result directly instead of running follow-up searches.",
    );

    let output = if deny {
        serde_json::json!({
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "additionalContext": nudge,
                "permissionDecision": "deny",
                "permissionDecisionReason": nudge,
            }
        })
    } else {
        serde_json::json!({
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "additionalContext": nudge,
            }
        })
    };

    println!("{}", serde_json::to_string(&output)?);
    Ok(())
}

/// SubagentStart hook — called via `semantex --subagent-hook`.
///
/// Injects semantex instructions directly into newly spawned subagents.
/// Unlike PreToolUse:Agent (which sends context to the parent), SubagentStart
/// fires inside the subagent's own conversation, so the instructions actually
/// reach the agent that will do the searching.
///
/// If no index: output `{}` (silent)
/// If index found: output SubagentStart context
pub fn cmd_subagent_hook() -> Result<()> {
    if find_index_dir().is_none() {
        println!("{{}}");
        return Ok(());
    }

    let output = serde_json::json!({
        "hookSpecificOutput": {
            "hookEventName": "SubagentStart",
            "additionalContext": concat!(
                "IMPORTANT: the `semantex_agent` tool (semantic code search) is available. ",
                "Use it instead of Grep/Glob/Read for EVERY code search — one call returns a ",
                "complete answer with file:line + code, and it covers regex and file-glob too. ",
                "Use the result directly, no follow-up Reads.\n\n",
                "Fall back to Grep/Glob only if the `semantex_agent` tool is unavailable.",
            ),
        }
    });

    println!("{}", serde_json::to_string(&output)?);
    Ok(())
}

/// PreToolUse Bash hook — called via `semantex --bash-hook [--deny]`.
///
/// Inspects `tool_input.command` from stdin. If the command is a search operation
/// (grep, rg, find, fd, etc.), redirects to semantex. Otherwise, silent pass-through.
///
/// Default (no --deny): soft nudge via additionalContext.
/// With --deny: hard block via permissionDecision.
pub fn cmd_bash_hook(deny: bool) -> Result<()> {
    if find_index_dir().is_none() {
        println!("{{}}");
        return Ok(());
    }

    // Read tool_input from stdin
    let mut input = String::new();
    std::io::Read::read_to_string(&mut std::io::stdin(), &mut input)?;

    let is_search = if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&input) {
        parsed
            .get("tool_input")
            .and_then(|ti| ti.get("command"))
            .and_then(|c| c.as_str())
            .is_some_and(is_search_command)
    } else {
        false
    };

    if !is_search {
        println!("{{}}");
        return Ok(());
    }

    let nudge = concat!(
        "Use the `semantex_agent` tool (semantic code search) instead of grep/rg/find — ",
        "one call returns a complete answer with file:line + code; it covers regex and ",
        "file-glob too. Use the result directly instead of running follow-up searches.",
    );

    let output = if deny {
        serde_json::json!({
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "additionalContext": nudge,
                "permissionDecision": "deny",
                "permissionDecisionReason": nudge,
            }
        })
    } else {
        serde_json::json!({
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "additionalContext": nudge,
            }
        })
    };

    println!("{}", serde_json::to_string(&output)?);
    Ok(())
}

/// Read the global project registry and trigger a background reindex for any repo
/// whose index is older than `SEMANTEX_REFRESH_SECS` (default: 3600 s), skipping
/// `current_dir` which has already been handled by the caller.
fn refresh_stale_registry_repos(current_dir: &std::path::Path) {
    let threshold: u64 = std::env::var("SEMANTEX_REFRESH_SECS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3600);
    // Bounds how many background builds one hook invocation can kick off.
    // Defaults to the build-slot gate's own concurrency limit — spawning more
    // than that just queues raw OS processes for no benefit; the rest get
    // picked up on a later session-hook run since staleness persists.
    let max_spawns: usize = std::env::var("SEMANTEX_REFRESH_MAX_SPAWNS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(semantex_core::index::gate::max_concurrent_builds);

    // Self-heal: a registry entry whose directory is gone (deleted repo,
    // expired tmp dir from a benchmark/test run) can never become non-stale,
    // so it would be refreshed — and spawn a doomed indexer — forever.
    let removed = registry::retain(|p| p.path.is_dir());
    if removed > 0 {
        tracing::debug!(removed, "Session hook: pruned missing registry paths");
    }

    let mut repos = registry::read_all();
    // Shortest path first, so a repo root is always considered "kept" before
    // any of its own subdirectories — needed for the nesting check below.
    repos.sort_by_key(|p| p.as_os_str().len());

    let mut kept_roots: Vec<PathBuf> = Vec::new();
    let mut spawned = 0usize;

    for repo in repos {
        let canonical = repo.canonicalize().unwrap_or(repo);
        if canonical == current_dir {
            continue; // already handled
        }
        // Skip paths nested inside an already-kept registered project: a
        // monorepo where every sub-crate got auto-registered (each `cd` into
        // a subdirectory self-registers on first use) would otherwise fan out
        // one background build per subdirectory for what the ancestor's
        // index already covers.
        if kept_roots.iter().any(|root| canonical.starts_with(root)) {
            continue;
        }
        kept_roots.push(canonical.clone());

        if spawned >= max_spawns {
            continue; // deferred to a later session-hook run
        }

        // S8: detect_for_config so an embedder change in any registered repo also
        // triggers a refresh. One config load per registered repo on the session
        // hook (not a hot loop); fall back to schema-only on a config error.
        let idx_state = match SemantexConfig::load(Some(&canonical)) {
            Ok(cfg) => state::detect_for_config(&canonical, &cfg),
            Err(_) => state::detect(&canonical),
        };
        if idx_state == IndexState::Building {
            continue; // already in progress
        }
        let age = state::index_age_secs(&canonical).unwrap_or(0);
        if idx_state == IndexState::NotIndexed || idx_state == IndexState::Stale || age >= threshold
        {
            tracing::debug!(
                path = %canonical.display(),
                age_secs = age,
                "Session hook: refreshing stale registry repo"
            );
            super::spawn_background_index(&canonical);
            spawned += 1;
        }
    }
}

/// Check if a shell command is a search operation that semantex should replace.
fn is_search_command(cmd: &str) -> bool {
    let first_token = cmd.split_whitespace().next().unwrap_or("");
    matches!(
        first_token,
        "grep" | "rg" | "ag" | "ack" | "find" | "fd" | "fgrep" | "egrep"
    ) || (first_token == "git" && cmd.contains("grep"))
}

/// Decide whether reading `file_path` is worth a soft semantex nudge.
///
/// Mirrors graphify's path-awareness: nudge for a source/doc file the agent is
/// exploring, but stay silent (return `false`) on semantex's own output, VCS /
/// dependency / build dirs, lock files, and binary/data/media files — anything
/// where a "use semantex instead" nudge would be noise or risk a loop.
fn should_nudge_read_path(file_path: &str) -> bool {
    // Lock files — match by exact name (case-sensitive, as these are well-known).
    const LOCK_FILES: [&str; 10] = [
        "Cargo.lock",
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "uv.lock",
        "poetry.lock",
        "Gemfile.lock",
        "composer.lock",
        "go.sum",
        "flake.lock",
    ];
    // Binary / data / media extensions — case-insensitive. The compound suffix
    // `*.min.js` is handled separately below.
    const BINARY_EXTS: &[&str] = &[
        "lock", "bin", "wasm", "so", "dylib", "a", "o", "exe", "dll", "png", "jpg", "jpeg", "gif",
        "svg", "ico", "webp", "pdf", "zip", "tar", "gz", "tgz", "bz2", "7z", "woff", "woff2",
        "ttf", "eot", "otf", "mp4", "mp3", "mov", "wav", "db", "sqlite", "sqlite3", "parquet",
        "pyc", "class", "jar", "map",
    ];

    // Normalize separators so the checks below are OS-agnostic.
    let path = file_path.replace('\\', "/");

    // semantex's own output — never nudge (avoid loops / re-reads of our results).
    if path.contains("/.semantex/") || path.starts_with(".semantex/") {
        return false;
    }

    // VCS / dependency / build directories — not source the agent is exploring.
    for marker in ["/.git/", "/node_modules/", "/target/", "/dist/", "/build/"] {
        if path.contains(marker) {
            return false;
        }
    }
    // Also catch these as a leading path component (no leading slash).
    for prefix in [".git/", "node_modules/", "target/", "dist/", "build/"] {
        if path.starts_with(prefix) {
            return false;
        }
    }

    // File name (last path component).
    let name = path.rsplit('/').next().unwrap_or(&path);

    if LOCK_FILES.contains(&name) {
        return false;
    }

    let name_lower = name.to_ascii_lowercase();
    // Compound suffix: `*.min.js` is a built/minified artifact.
    if name_lower.ends_with(".min.js") {
        return false;
    }
    if let Some((_, ext)) = name_lower.rsplit_once('.')
        && BINARY_EXTS.contains(&ext)
    {
        return false;
    }

    // Otherwise: a source/doc file worth nudging.
    true
}

/// PreToolUse Read hook — called via `semantex --read-hook [--deny]`.
///
/// Catches the measured failure mode (the substitution probe): the agent doing
/// DUPLICATE re-`Read`s of files semantex already returned. Graphify parity —
/// it hooks the native Read tool the Bash hook can't see.
///
/// Default (no --deny): soft nudge via additionalContext — the Read still runs.
/// With --deny: hard block via permissionDecision.
///
/// Fails open everywhere: no index, unparseable stdin, missing `file_path`, or a
/// path not worth nudging → `{}` (silent pass-through).
pub fn cmd_read_hook(deny: bool) -> Result<()> {
    if find_index_dir().is_none() {
        println!("{{}}");
        return Ok(());
    }

    // Read tool_input from stdin.
    let mut input = String::new();
    if std::io::Read::read_to_string(&mut std::io::stdin(), &mut input).is_err() {
        // Fail open on a stdin read error.
        println!("{{}}");
        return Ok(());
    }

    // Parse JSON + extract tool_input.file_path. Any failure → fail open.
    let file_path = serde_json::from_str::<serde_json::Value>(&input)
        .ok()
        .and_then(|v| {
            v.get("tool_input")
                .and_then(|ti| ti.get("file_path"))
                .and_then(|p| p.as_str())
                .map(str::to_owned)
        });

    let Some(file_path) = file_path else {
        println!("{{}}");
        return Ok(());
    };

    if !should_nudge_read_path(&file_path) {
        println!("{{}}");
        return Ok(());
    }

    let nudge = READ_HOOK_NUDGE;

    let output = if deny {
        serde_json::json!({
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "additionalContext": nudge,
                "permissionDecision": "deny",
                "permissionDecisionReason": nudge,
            }
        })
    } else {
        serde_json::json!({
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "additionalContext": nudge,
            }
        })
    };

    println!("{}", serde_json::to_string(&output)?);
    Ok(())
}

/// SessionEnd hook — called via `semantex --session-end-hook`.
///
/// Stops the daemon if one is running. No JSON output needed
/// (SessionEnd hooks have no control over the session).
pub fn cmd_session_end_hook() -> Result<()> {
    let Some(index_dir) = find_index_dir() else {
        return Ok(());
    };

    let project_dir = index_dir.parent().unwrap_or(&index_dir);

    // `stop` connects to the per-project daemon via its port file, so binary
    // identity doesn't affect routing — self_exe() just keeps it consistent.
    let _ = std::process::Command::new(super::self_exe())
        .arg("stop")
        .arg(project_dir)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status(); // wait for completion (fast — socket send)

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_nudge_read_path_true_for_source() {
        for p in [
            "src/main.rs",
            "crates/x/lib.py",
            "README.md",
            "docs/design.md",
            "a/b/Component.tsx",
        ] {
            assert!(should_nudge_read_path(p), "expected nudge for {p}");
        }
    }

    #[test]
    fn should_nudge_read_path_false_for_own_output_and_vcs() {
        for p in [
            ".semantex/meta.json",
            "proj/.semantex/chunks.db",
            ".git/config",
            "node_modules/x/y.js",
            "target/debug/foo",
        ] {
            assert!(!should_nudge_read_path(p), "expected silent for {p}");
        }
    }

    #[test]
    fn read_hook_nudge_points_at_semantex_agent_tool() {
        assert!(
            READ_HOOK_NUDGE.contains("semantex_agent"),
            "read-hook nudge must point at the semantex_agent MCP tool"
        );
        // Keeps the duplicate-re-read framing.
        assert!(READ_HOOK_NUDGE.contains("duplicate work"));
    }

    #[test]
    fn should_nudge_read_path_false_for_lock_and_binary() {
        for p in [
            "Cargo.lock",
            "package-lock.json",
            "go.sum",
            "assets/logo.png",
            "data.bin",
            "x.wasm",
            "f.min.js",
        ] {
            assert!(!should_nudge_read_path(p), "expected silent for {p}");
        }
    }
}
