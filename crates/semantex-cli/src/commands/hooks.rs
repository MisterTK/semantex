//! Claude Code hook commands — full lifecycle.
//!
//! SessionStart   → detect index, pre-warm daemon, output context
//! PreToolUse     → Grep|Glob nudge or deny, Bash search nudge or deny (--deny flag)
//! SubagentStart  → inject semantex context into Explore/general-purpose agents
//! SessionEnd     → stop daemon
//!
//! Output format follows the Claude Code hooks spec:
//!   { "hookSpecificOutput": { "hookEventName": "...", "additionalContext": "..." } }

use anyhow::Result;
use semantex_core::index::state::{self, IndexState};
use std::path::PathBuf;
use std::process::Stdio;

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

    // Use state::detect() to catch stale indexes (schema mismatch)
    let idx_state = state::detect(project_dir);
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
    let _ = std::process::Command::new("semantex")
        .arg("serve")
        .arg(project_dir)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn(); // semantex serve handles "already running" gracefully

    // Fire-and-forget incremental reindex to pick up changes since last session
    super::spawn_background_index(project_dir);

    let additional_context = format!(
        "semantex semantic code search is available (index: {}).\n\
         - `semantex --deep \"question\"` — one call, complete answer. Use for how/why/connect questions. Trust the answer, skip follow-up Reads.\n\
         - `semantex --refs \"query\"` — compact refs for simple lookups (find definition, list references).\n\
         - `semantex --grep \"literal\"` — exact match.\n\
         Fallback: Grep for regex patterns, Glob for file names.",
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
        "STOP: Use semantex via Bash instead of Grep/Glob — semantex finds code by meaning.\n",
        "- Complex questions (how/why/explain): `semantex --deep \"question\"` — one call replaces 5-10 grep+read iterations. Use the answer directly.\n",
        "- Find references: `semantex --refs \"query\"` or `semantex --refs -e \"pattern\" \"query\"`\n",
        "- Exact string: `semantex --grep \"literal\"`",
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
                "IMPORTANT: semantex (semantic code search) is available. Use it via Bash instead of Grep/Glob.\n\n",
                "Start with semantex for EVERY code search:\n",
                "- How/why/explain questions → `semantex --deep \"question\"` — one call, complete answer. Use it directly, no follow-up Reads.\n",
                "- Find/where/what questions → `semantex --refs \"query\"`\n",
                "- Exact string → `semantex --grep \"literal\"`\n",
                "- Regex + semantic → `semantex --refs -e \"regex\" \"query\"`\n\n",
                "Fall back to Grep ONLY for regex patterns, Glob ONLY for file name patterns.",
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
        "Use `semantex --refs \"query\"` or `semantex --grep \"literal\"` instead of grep/rg/find. ",
        "Use `semantex --deep \"question\"` for complex questions.",
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

/// Check if a shell command is a search operation that semantex should replace.
fn is_search_command(cmd: &str) -> bool {
    let first_token = cmd.split_whitespace().next().unwrap_or("");
    matches!(
        first_token,
        "grep" | "rg" | "ag" | "ack" | "find" | "fd" | "fgrep" | "egrep"
    ) || (first_token == "git" && cmd.contains("grep"))
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

    let _ = std::process::Command::new("semantex")
        .arg("stop")
        .arg(project_dir)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status(); // wait for completion (fast — socket send)

    Ok(())
}
