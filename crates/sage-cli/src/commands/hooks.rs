//! Claude Code hook commands — full lifecycle.
//!
//! SessionStart  → detect index, pre-warm daemon, output context
//! PreToolUse    → Grep|Glob nudge, Task sub-agent injection
//! SessionEnd    → stop daemon
//!
//! Output format follows the Claude Code hooks spec:
//!   { "hookSpecificOutput": { "hookEventName": "...", "additionalContext": "..." } }

use anyhow::Result;
use sage_core::index::state::{self, IndexState};
use std::path::PathBuf;
use std::process::Stdio;

/// Check if a sage index directory (`.sage/meta.json`) exists for the current
/// directory or any parent. Returns the `.sage` directory path if found.
fn find_index_dir() -> Option<PathBuf> {
    let mut dir = std::env::current_dir().ok()?;
    loop {
        let sage_dir = dir.join(".sage");
        if sage_dir.join("meta.json").exists() {
            return Some(sage_dir);
        }
        if !dir.pop() {
            return None;
        }
    }
}

/// SessionStart hook — called via `sage --session-hook`.
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
                    "sage (semantic code search) is building an index for this project in the background.\n",
                    "Use Grep/Glob for now. Semantic search will be available shortly.\n",
                    "sage searches will automatically fall back to ripgrep keyword results until the index is ready.",
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
                    "sage (semantic code search) has a stale index — rebuilding in the background.\n",
                    "Use Grep/Glob for now. Semantic search will be available shortly.\n",
                    "sage searches will automatically fall back to ripgrep keyword results until the index is ready.",
                ),
            }
        });
        println!("{}", serde_json::to_string(&output)?);
        return Ok(());
    }

    // Pre-warm daemon (fire-and-forget, non-blocking)
    match std::env::current_exe() {
        Ok(exe) => {
            let _ = std::process::Command::new(exe)
                .arg("serve")
                .arg(project_dir)
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn(); // sage serve handles "already running" gracefully
        }
        Err(e) => eprintln!("sage: cannot determine executable path: {e}"),
    }

    // Fire-and-forget incremental reindex to pick up changes since last session
    super::spawn_background_index(project_dir);

    let additional_context = format!(
        concat!(
            "sage (semantic code search) is available for this project (index: {}).\n\n",
            "Use sage as your PRIMARY search tool instead of Grep/Glob for code exploration.\n\n",
            "## When to Use What\n",
            "| Need | Tool | Example |\n",
            "|------|------|---------|\n",
            "| Conceptual/semantic search | `sage \"query\"` | `sage \"authentication flow\" src/` |\n",
            "| Regex + semantic hybrid | `sage -e \"pattern\" \"query\"` | `sage -e \"login\\|auth\" \"auth flow\"` |\n",
            "| Exact string/BM25 | `sage --grep \"token\"` | `sage --grep \"ConnectionFactory\"` |\n",
            "| Find file by exact name | Glob | `Glob pattern=\"**/auth*.rs\"` |\n",
            "| Exact regex on file content | Grep | `Grep pattern=\"specific_function_name\"` |\n\n",
            "## Tips\n",
            "- First query may take 30-90s (model loading). Subsequent queries: <100ms via daemon.\n",
            "- Use `-c` to show code snippets. Use `--code-only` to exclude docs/config.\n",
            "- Use `-m N` to control result count (default 10).\n",
            "- Sub-agents should also use sage via Bash for all code searches.",
        ),
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

/// PreToolUse Grep|Glob hook — called via `sage --grep-hook`.
///
/// Soft interception: Grep/Glob still runs, but additionalContext nudges Claude
/// to prefer sage for subsequent searches.
///
/// If no index: output `{}` (silent — let Grep/Glob work normally)
/// If index found: output PreToolUse nudge
pub fn cmd_grep_hook() -> Result<()> {
    if find_index_dir().is_none() {
        println!("{{}}");
        return Ok(());
    }

    let output = serde_json::json!({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "additionalContext": concat!(
                "Use sage via Bash instead of Grep/Glob — sage finds code by meaning and replaces grep for code search.\n",
                "Commands: `sage \"query\" .`, `sage -e \"pattern\" \"query\"`, `sage --grep \"literal\"`",
            ),
        }
    });

    println!("{}", serde_json::to_string(&output)?);
    Ok(())
}

/// PreToolUse Task hook — called via `sage --task-hook`.
///
/// Injects sage instructions into sub-agent prompts so spawned agents
/// also use sage instead of Grep/Glob.
///
/// If no index: output `{}` (silent)
/// If index found: output directive instructions for the sub-agent
pub fn cmd_task_hook() -> Result<()> {
    if find_index_dir().is_none() {
        println!("{{}}");
        return Ok(());
    }

    let output = serde_json::json!({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "additionalContext": concat!(
                "IMPORTANT: sage (semantic code search) is available in this project. ",
                "Use sage via Bash for ALL code searches instead of Grep or Glob tools.\n\n",
                "Commands:\n",
                "- `sage \"natural language query\" .` — semantic search\n",
                "- `sage -e \"regex\" \"query\"` — regex + semantic hybrid\n",
                "- `sage --grep \"literal\"` — fast exact/BM25 search\n",
                "- `sage -c \"query\" .` — include code snippets\n",
                "- `sage --code-only \"query\" .` — exclude docs/config\n\n",
                "Only fall back to Grep for exact regex on file content, or Glob for finding files by name pattern.",
            ),
        }
    });

    println!("{}", serde_json::to_string(&output)?);
    Ok(())
}

/// SessionEnd hook — called via `sage --session-end-hook`.
///
/// Stops the daemon if one is running. No JSON output needed
/// (SessionEnd hooks have no control over the session).
pub fn cmd_session_end_hook() -> Result<()> {
    let Some(index_dir) = find_index_dir() else {
        return Ok(());
    };

    let project_dir = index_dir.parent().unwrap_or(&index_dir);

    match std::env::current_exe() {
        Ok(exe) => {
            let _ = std::process::Command::new(exe)
                .arg("stop")
                .arg(project_dir)
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status(); // wait for completion (fast — socket send)
        }
        Err(e) => eprintln!("sage: cannot determine executable path: {e}"),
    }

    Ok(())
}
