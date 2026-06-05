use anyhow::{Context, Result};
use colored::Colorize;
use std::io::Write;
use std::path::{Path, PathBuf};

/// Installation scope — mirrors Claude Code's own settings hierarchy.
#[derive(Debug, Clone, PartialEq, clap::ValueEnum)]
pub enum InstallScope {
    /// ~/.claude/settings.json — hooks fire for all your projects
    User,
    /// .claude/settings.json in CWD — shared with team, committed to git
    Project,
    /// .claude/settings.local.json in CWD — this repo only, gitignored
    Local,
}

/// Binary name for hook commands — always bare so it resolves via PATH.
fn semantex_binary_path() -> &'static str {
    "semantex"
}

fn dirs_home() -> PathBuf {
    dirs::home_dir().unwrap_or_else(std::env::temp_dir)
}

/// Embedded SKILL.md — single source of truth lives in plugin/skills/semantex/SKILL.md.
const SKILL_MD: &str = include_str!("../../../../plugin/skills/semantex/SKILL.md");

/// Prompt the user to choose an install scope, or fall back to user-scope in non-interactive
/// environments (CI, piped stdin).
fn prompt_scope() -> Result<InstallScope> {
    use std::io::{self, IsTerminal, Write};

    if !io::stdin().is_terminal() {
        eprintln!(
            "{} Non-interactive mode — defaulting to user scope.",
            "note:".dimmed()
        );
        return Ok(InstallScope::User);
    }

    eprintln!("Where should semantex hooks be installed?\n");
    eprintln!(
        "  {}  {}  all your projects (you only)",
        "user   ".bold(),
        "~/.claude/settings.json         ".dimmed(),
    );
    eprintln!(
        "  {}  {}  this repo, shared with team via git",
        "project".bold(),
        ".claude/settings.json           ".dimmed(),
    );
    eprintln!(
        "  {}  {}  this repo, you only (gitignored)",
        "local  ".bold(),
        ".claude/settings.local.json     ".dimmed(),
    );
    eprintln!();

    loop {
        eprint!("Scope [user]: ");
        io::stderr().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim().to_lowercase();

        match input.as_str() {
            "" | "user" | "u" => return Ok(InstallScope::User),
            "project" | "proj" | "p" => return Ok(InstallScope::Project),
            "local" | "loc" | "l" => return Ok(InstallScope::Local),
            _ => eprintln!("Please enter user, project, or local."),
        }
    }
}

/// The three semantex MCP search tools the agent calls — pre-approved so a fresh
/// session never gets a per-tool permission prompt for them.
const SEMANTEX_MCP_TOOLS: [&str; 3] = [
    "mcp__semantex__semantex_agent",
    "mcp__semantex__semantex_search",
    "mcp__semantex__semantex_deep",
];

/// Build the semantex MCP server config object (matches `plugin/.mcp.json`):
/// `{"type": "stdio", "command": <binary>, "args": ["mcp"]}`.
fn semantex_mcp_server(binary: &str) -> serde_json::Value {
    serde_json::json!({
        "type": "stdio",
        "command": binary,
        "args": ["mcp"],
    })
}

/// Ensure `root["mcpServers"]["semantex"]` is the semantex stdio server config,
/// creating the `mcpServers` object if absent. Preserves every other key and
/// every other server. Idempotent.
fn merge_mcp_server(root: &mut serde_json::Value, binary: &str) {
    // If root isn't an object (e.g. a fresh `{}` or a corrupted scalar we chose to
    // proceed on), coerce it to an empty object so we never panic.
    if !root.is_object() {
        *root = serde_json::json!({});
    }
    let obj = root.as_object_mut().expect("root coerced to object");
    let servers = obj
        .entry("mcpServers")
        .or_insert_with(|| serde_json::json!({}));
    if !servers.is_object() {
        *servers = serde_json::json!({});
    }
    servers
        .as_object_mut()
        .expect("mcpServers coerced to object")
        .insert("semantex".to_string(), semantex_mcp_server(binary));
}

/// Ensure `"semantex"` appears in `settings["enabledMcpjsonServers"]` (the exact
/// Claude Code field that pre-approves a project `.mcp.json` server). Creates the
/// array if absent; never duplicates; preserves existing entries.
fn ensure_enabled_mcp(settings: &mut serde_json::Value) {
    if !settings.is_object() {
        *settings = serde_json::json!({});
    }
    let obj = settings
        .as_object_mut()
        .expect("settings coerced to object");
    let list = obj
        .entry("enabledMcpjsonServers")
        .or_insert_with(|| serde_json::json!([]));
    if !list.is_array() {
        *list = serde_json::json!([]);
    }
    let arr = list.as_array_mut().expect("enabledMcpjsonServers is array");
    if !arr.iter().any(|v| v.as_str() == Some("semantex")) {
        arr.push(serde_json::json!("semantex"));
    }
}

/// Ensure the three `mcp__semantex__*` tool entries appear in
/// `settings["permissions"]["allow"]`, creating the nested `permissions` object
/// and `allow` array if absent. Never duplicates; preserves existing allow
/// entries.
fn ensure_tool_permissions(settings: &mut serde_json::Value) {
    if !settings.is_object() {
        *settings = serde_json::json!({});
    }
    let obj = settings
        .as_object_mut()
        .expect("settings coerced to object");
    let perms = obj
        .entry("permissions")
        .or_insert_with(|| serde_json::json!({}));
    if !perms.is_object() {
        *perms = serde_json::json!({});
    }
    let allow = perms
        .as_object_mut()
        .expect("permissions coerced to object")
        .entry("allow")
        .or_insert_with(|| serde_json::json!([]));
    if !allow.is_array() {
        *allow = serde_json::json!([]);
    }
    let arr = allow.as_array_mut().expect("allow is array");
    for tool in SEMANTEX_MCP_TOOLS {
        if !arr.iter().any(|v| v.as_str() == Some(tool)) {
            arr.push(serde_json::json!(tool));
        }
    }
}

/// Resolve the settings file path and base claude dir for the given scope.
/// Returns (claude_dir, settings_path).
fn scope_paths(scope: &InstallScope) -> Result<(PathBuf, PathBuf)> {
    match scope {
        InstallScope::User => {
            let claude_dir = dirs_home().join(".claude");
            let settings = claude_dir.join("settings.json");
            Ok((claude_dir, settings))
        }
        InstallScope::Project => {
            let cwd = std::env::current_dir()?;
            let claude_dir = cwd.join(".claude");
            let settings = claude_dir.join("settings.json");
            Ok((claude_dir, settings))
        }
        InstallScope::Local => {
            let cwd = std::env::current_dir()?;
            let claude_dir = cwd.join(".claude");
            let settings = claude_dir.join("settings.local.json");
            Ok((claude_dir, settings))
        }
    }
}

/// Path to the user's `~/.claude.json` (Claude Code's global config — holds
/// user/global `mcpServers` and per-project local `mcpServers`).
fn claude_json_path() -> PathBuf {
    dirs_home().join(".claude.json")
}

/// Read + parse `~/.claude.json`.
///
/// - Missing file → `Ok(Some(default))` (we create it fresh).
/// - Present + valid JSON → `Ok(Some(value))`.
/// - Present + UNPARSEABLE → `Ok(None)` — the caller MUST skip the write and warn;
///   we never clobber a file we couldn't parse (it's the user's most important
///   Claude file).
fn read_claude_json() -> Result<Option<serde_json::Value>> {
    let path = claude_json_path();
    if !path.exists() {
        return Ok(Some(serde_json::json!({})));
    }
    let content = std::fs::read_to_string(&path)?;
    match serde_json::from_str::<serde_json::Value>(&content) {
        Ok(v) => Ok(Some(v)),
        Err(_) => Ok(None),
    }
}

/// Atomically write `value` to `~/.claude.json`, backing up any existing file to
/// `~/.claude.json.semantex-bak` first. Write goes to a temp file in the same dir
/// then `rename`s into place (atomic on the same filesystem) so a crash mid-write
/// can never leave a truncated config.
fn write_claude_json_atomic(value: &serde_json::Value) -> Result<()> {
    let path = claude_json_path();

    // Back up an existing file before we touch it.
    if path.exists() {
        let backup = dirs_home().join(".claude.json.semantex-bak");
        std::fs::copy(&path, &backup)
            .with_context(|| format!("backing up {} before write", path.display()))?;
    }

    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent)?;
    let tmp = path.with_extension("json.semantex-tmp");
    std::fs::write(&tmp, serde_json::to_string_pretty(value)?)
        .with_context(|| format!("writing temp file {}", tmp.display()))?;
    std::fs::rename(&tmp, &path)
        .with_context(|| format!("atomically renaming into {}", path.display()))?;
    Ok(())
}

/// Print the "could not parse ~/.claude.json — skipping" warning + the manual
/// recovery command, so the user can register the server by hand.
fn warn_unparseable_claude_json(binary: &str) {
    let path = claude_json_path();
    eprintln!(
        "  {} {} is not valid JSON — leaving it untouched (never clobbering it).",
        "warning:".yellow().bold(),
        path.display()
    );
    eprintln!(
        "           Register the server manually with: claude mcp add semantex -- {binary} mcp"
    );
}

/// FIX #1 — register + pre-approve the semantex MCP server for the given scope.
///
/// Robustness contract for `~/.claude.json` (User/Local scopes): parse-or-skip
/// (never clobber on a parse error), back up to `.semantex-bak`, atomic rename.
fn register_mcp_server(
    scope: &InstallScope,
    binary: &str,
    settings: &mut serde_json::Value,
    settings_path: &Path,
) -> Result<()> {
    match scope {
        InstallScope::Project => {
            // Project servers live in a repo-root `.mcp.json`. repo_root is the dir
            // CONTAINING `.claude` — i.e. settings_path's grandparent.
            let repo_root = settings_path
                .parent() // .claude
                .and_then(|p| p.parent()) // repo root
                .map_or_else(|| PathBuf::from("."), Path::to_path_buf);
            let mcp_path = repo_root.join(".mcp.json");

            let mut mcp_json: serde_json::Value = if mcp_path.exists() {
                let content = std::fs::read_to_string(&mcp_path)?;
                serde_json::from_str(&content).unwrap_or_else(|_| serde_json::json!({}))
            } else {
                serde_json::json!({})
            };
            merge_mcp_server(&mut mcp_json, binary);
            std::fs::write(&mcp_path, serde_json::to_string_pretty(&mcp_json)?)?;

            // Pre-approve the project .mcp.json server + the tool calls in settings.json.
            ensure_enabled_mcp(settings);
            ensure_tool_permissions(settings);

            eprintln!(
                "  {} MCP server → {}",
                "✓".green().bold(),
                mcp_path.display()
            );
        }
        InstallScope::Local => {
            // Local servers live under ~/.claude.json projects["<abs repo>"].mcpServers.
            let repo_root = std::env::current_dir()?;
            let repo_key = repo_root.to_string_lossy().to_string();

            match read_claude_json()? {
                Some(mut root) => {
                    if !root.is_object() {
                        root = serde_json::json!({});
                    }
                    let projects = root
                        .as_object_mut()
                        .expect("root is object")
                        .entry("projects")
                        .or_insert_with(|| serde_json::json!({}));
                    if !projects.is_object() {
                        *projects = serde_json::json!({});
                    }
                    let project_entry = projects
                        .as_object_mut()
                        .expect("projects is object")
                        .entry(repo_key)
                        .or_insert_with(|| serde_json::json!({}));
                    // Reuse merge_mcp_server on the per-project sub-object.
                    merge_mcp_server(project_entry, binary);
                    write_claude_json_atomic(&root)?;

                    // Local project servers in ~/.claude.json are auto-loaded — no
                    // enabledMcpjsonServers needed. Just pre-approve the tool calls.
                    ensure_tool_permissions(settings);

                    eprintln!(
                        "  {} MCP server → {}",
                        "✓".green().bold(),
                        claude_json_path().display()
                    );
                }
                None => warn_unparseable_claude_json(binary),
            }
        }
        InstallScope::User => {
            // User/global servers live in ~/.claude.json TOP-LEVEL mcpServers
            // (auto-approved — no enabledMcpjsonServers needed).
            match read_claude_json()? {
                Some(mut root) => {
                    merge_mcp_server(&mut root, binary);
                    write_claude_json_atomic(&root)?;
                    // Pre-approve the tool calls in ~/.claude/settings.json.
                    ensure_tool_permissions(settings);
                    eprintln!(
                        "  {} MCP server → {}",
                        "✓".green().bold(),
                        claude_json_path().display()
                    );
                }
                None => warn_unparseable_claude_json(binary),
            }
        }
    }
    Ok(())
}

/// Install semantex hooks + skill into Claude Code.
///
/// If `scope` is None, prompts the user interactively.
pub fn install_claude_code(scope: Option<InstallScope>) -> Result<()> {
    let scope = match scope {
        Some(s) => s,
        None => prompt_scope()?,
    };

    let binary = semantex_binary_path();
    let (claude_dir, settings_path) = scope_paths(&scope)?;

    // --- 1. Install hooks into the appropriate settings file ---
    let mut settings: serde_json::Value = if settings_path.exists() {
        let content = std::fs::read_to_string(&settings_path)?;
        serde_json::from_str(&content)?
    } else {
        serde_json::json!({})
    };

    let hooks = settings
        .as_object_mut()
        .context("settings.json is not an object")?
        .entry("hooks")
        .or_insert_with(|| serde_json::json!({}));

    let hooks_obj = hooks.as_object_mut().context("hooks is not an object")?;

    // PreToolUse hooks — nudge Grep/Glob, Bash search commands, and native Read
    // (the Bash hook can't see a file read through the native Read tool) toward semantex
    hooks_obj.insert(
        "PreToolUse".to_string(),
        serde_json::json!([
            {
                "matcher": "Grep|Glob",
                "hooks": [{
                    "type": "command",
                    "command": format!("{binary} --grep-hook"),
                    "timeout": 5,
                }]
            },
            {
                "matcher": "Bash",
                "hooks": [{
                    "type": "command",
                    "command": format!("{binary} --bash-hook"),
                    "timeout": 5,
                }]
            },
            {
                "matcher": "Read",
                "hooks": [{
                    "type": "command",
                    "command": format!("{binary} --read-hook"),
                    "timeout": 5,
                }]
            }
        ]),
    );

    // SubagentStart hooks — inject semantex context into Explore and general-purpose subagents
    hooks_obj.insert(
        "SubagentStart".to_string(),
        serde_json::json!([
            {
                "matcher": "Explore",
                "hooks": [{
                    "type": "command",
                    "command": format!("{binary} --subagent-hook"),
                    "timeout": 5,
                }]
            },
            {
                "matcher": "general-purpose",
                "hooks": [{
                    "type": "command",
                    "command": format!("{binary} --subagent-hook"),
                    "timeout": 5,
                }]
            }
        ]),
    );

    // SessionStart hook — injects semantex context + pre-warms daemon
    hooks_obj.insert(
        "SessionStart".to_string(),
        serde_json::json!([{
            "matcher": "startup|resume",
            "hooks": [{
                "type": "command",
                "command": format!("{binary} --session-hook"),
                "timeout": 15,
            }]
        }]),
    );

    // SessionEnd hook — stops daemon
    hooks_obj.insert(
        "SessionEnd".to_string(),
        serde_json::json!([{
            "hooks": [{
                "type": "command",
                "command": format!("{binary} --session-end-hook"),
                "timeout": 10,
            }]
        }]),
    );

    // --- 1b. FIX #1: register + pre-approve the semantex MCP server ---
    // This mutates `settings` (adds enabledMcpjsonServers / permissions.allow) and
    // writes the MCP server config to its scope-appropriate location, so it must run
    // BEFORE we persist settings.json below.
    register_mcp_server(&scope, binary, &mut settings, &settings_path)?;

    std::fs::create_dir_all(settings_path.parent().expect("settings path has parent"))?;
    std::fs::write(&settings_path, serde_json::to_string_pretty(&settings)?)?;

    eprintln!(
        "  {} hooks → {}",
        "✓".green().bold(),
        settings_path.display()
    );

    // For local-scope installs, ensure settings.local.json is gitignored.
    if scope == InstallScope::Local {
        ensure_gitignored(&claude_dir, "settings.local.json");
    }

    // --- 2. Install skill ---
    // Skill always goes into the same claude_dir as the hooks so each scope is self-contained.
    let skill_dir = claude_dir.join("skills").join("semantex");
    std::fs::create_dir_all(&skill_dir)?;
    std::fs::write(skill_dir.join("SKILL.md"), SKILL_MD)?;

    eprintln!(
        "  {} skill → {}",
        "✓".green().bold(),
        skill_dir.join("SKILL.md").display()
    );

    // Project-scope: remind the user to commit the settings file.
    if scope == InstallScope::Project {
        eprintln!(
            "\n  {} Commit {} to share semantex with your team.",
            "tip:".cyan().bold(),
            settings_path.display()
        );
    }

    eprintln!(
        "\n{} semantex installed for Claude Code. Restart Claude Code to activate.",
        "Done!".green().bold(),
    );
    Ok(())
}

/// Append `entry` to the `.gitignore` inside `dir` if not already present.
fn ensure_gitignored(dir: &Path, entry: &str) {
    // Walk up to find the nearest .gitignore (repo root or dir itself)
    let candidates = [dir.join(".gitignore"), dir.join("../.gitignore")];
    for gitignore_path in &candidates {
        if let Ok(p) = gitignore_path.canonicalize() {
            let existing = std::fs::read_to_string(&p).unwrap_or_default();
            if existing.lines().any(|l| l.trim() == entry) {
                return;
            }
            if let Ok(mut f) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&p)
            {
                if !existing.ends_with('\n') && !existing.is_empty() {
                    let _ = writeln!(f);
                }
                let _ = writeln!(f, "{entry}");
            }
            return;
        }
    }
}

/// Uninstall semantex hooks and skill from Claude Code.
/// Removes from all scopes where hooks are found.
pub fn uninstall_claude_code() -> Result<()> {
    let home = dirs_home();
    let cwd = std::env::current_dir().unwrap_or_else(|_| home.clone());

    let candidates = [
        home.join(".claude").join("settings.json"),
        cwd.join(".claude").join("settings.json"),
        cwd.join(".claude").join("settings.local.json"),
    ];

    let mut any_found = false;
    for settings_path in &candidates {
        if !settings_path.exists() {
            continue;
        }
        let content = std::fs::read_to_string(settings_path)?;
        let mut settings: serde_json::Value = serde_json::from_str(&content)?;

        let changed = if let Some(hooks) = settings.get_mut("hooks").and_then(|h| h.as_object_mut())
        {
            let before = hooks.len();
            hooks.remove("SessionStart");
            hooks.remove("PreToolUse");
            hooks.remove("SessionEnd");
            hooks.remove("SubagentStart");
            hooks.len() < before
        } else {
            false
        };

        if changed {
            std::fs::write(settings_path, serde_json::to_string_pretty(&settings)?)?;
            eprintln!(
                "  {} hooks removed from {}",
                "✓".green().bold(),
                settings_path.display()
            );
            any_found = true;
        }
    }

    // Remove skill directories from all scopes
    let skill_candidates = [
        home.join(".claude").join("skills").join("semantex"),
        cwd.join(".claude").join("skills").join("semantex"),
    ];
    for skill_dir in &skill_candidates {
        if skill_dir.exists() {
            std::fs::remove_dir_all(skill_dir)?;
            eprintln!(
                "  {} skill removed from {}",
                "✓".green().bold(),
                skill_dir.display()
            );
            any_found = true;
        }
    }

    if !any_found {
        eprintln!("No semantex hooks or skills found to remove.");
    }

    Ok(())
}

/// Install for OpenCode.
pub fn install_opencode() -> Result<()> {
    println!("{}", "Installing OpenCode integration...".green().bold());

    println!();
    println!("Add to your OpenCode configuration:");
    println!();
    println!("  semantex_command: \"semantex\"");
    println!();
    println!(
        "{}",
        "See semantex documentation for full OpenCode integration details.".dimmed()
    );

    Ok(())
}

/// Install for Codex.
pub fn install_codex() -> Result<()> {
    println!("{}", "Installing Codex integration...".green().bold());

    println!();
    println!("Add to your Codex configuration:");
    println!();
    println!("  semantex_command: \"semantex\"");
    println!();
    println!(
        "{}",
        "See semantex documentation for full Codex integration details.".dimmed()
    );

    Ok(())
}

/// Remove all semantex hook registrations.
pub fn uninstall_all() -> Result<()> {
    uninstall_claude_code()?;
    eprintln!("Removed all semantex hook registrations.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn merge_mcp_server_adds_preserving_other_servers() {
        // Pre-existing config with an unrelated server and an unrelated top-level key.
        let mut root = json!({
            "mcpServers": {
                "other": { "type": "stdio", "command": "other-bin", "args": [] }
            },
            "someTopLevelKey": 42
        });

        merge_mcp_server(&mut root, "semantex");

        let servers = root.get("mcpServers").and_then(|v| v.as_object()).unwrap();
        // semantex added with the expected shape.
        assert_eq!(
            servers.get("semantex").unwrap(),
            &json!({ "type": "stdio", "command": "semantex", "args": ["mcp"] })
        );
        // The other server is preserved untouched.
        assert_eq!(
            servers.get("other").unwrap(),
            &json!({ "type": "stdio", "command": "other-bin", "args": [] })
        );
        // Unrelated top-level key preserved.
        assert_eq!(root.get("someTopLevelKey").unwrap(), &json!(42));
    }

    #[test]
    fn merge_mcp_server_creates_servers_object_when_absent() {
        let mut root = json!({ "projects": {} });
        merge_mcp_server(&mut root, "/abs/semantex");
        assert_eq!(
            root["mcpServers"]["semantex"],
            json!({ "type": "stdio", "command": "/abs/semantex", "args": ["mcp"] })
        );
        // Untouched sibling preserved.
        assert_eq!(root["projects"], json!({}));
    }

    #[test]
    fn merge_mcp_server_is_idempotent() {
        let mut root = json!({});
        merge_mcp_server(&mut root, "semantex");
        let after_first = root.clone();
        merge_mcp_server(&mut root, "semantex");
        // Re-running produces no change (single semantex entry, identical value).
        assert_eq!(root, after_first);
        assert_eq!(root["mcpServers"].as_object().unwrap().len(), 1);
    }

    #[test]
    fn ensure_enabled_mcp_adds_when_absent() {
        let mut settings = json!({});
        ensure_enabled_mcp(&mut settings);
        assert_eq!(settings["enabledMcpjsonServers"], json!(["semantex"]));
    }

    #[test]
    fn ensure_enabled_mcp_no_dup_and_preserves_existing() {
        let mut settings = json!({ "enabledMcpjsonServers": ["existing-server"] });
        ensure_enabled_mcp(&mut settings);
        // existing-server preserved, semantex appended once.
        assert_eq!(
            settings["enabledMcpjsonServers"],
            json!(["existing-server", "semantex"])
        );
        // Idempotent: re-running does not duplicate.
        ensure_enabled_mcp(&mut settings);
        assert_eq!(
            settings["enabledMcpjsonServers"],
            json!(["existing-server", "semantex"])
        );
    }

    #[test]
    fn ensure_tool_permissions_creates_allow_with_three_tools() {
        let mut settings = json!({});
        ensure_tool_permissions(&mut settings);
        let allow = settings["permissions"]["allow"].as_array().unwrap();
        assert_eq!(allow.len(), 3);
        for tool in SEMANTEX_MCP_TOOLS {
            assert!(
                allow.iter().any(|v| v.as_str() == Some(tool)),
                "missing {tool}"
            );
        }
    }

    #[test]
    fn ensure_tool_permissions_merges_into_existing_allow_without_loss_or_dup() {
        let mut settings = json!({
            "permissions": {
                "allow": ["Bash(ls:*)", "mcp__semantex__semantex_agent"],
                "deny": ["Bash(rm:*)"]
            }
        });
        ensure_tool_permissions(&mut settings);
        let allow = settings["permissions"]["allow"].as_array().unwrap();
        // Existing non-semantex entry preserved.
        assert!(allow.iter().any(|v| v.as_str() == Some("Bash(ls:*)")));
        // The pre-existing semantex_agent entry is not duplicated.
        assert_eq!(
            allow
                .iter()
                .filter(|v| v.as_str() == Some("mcp__semantex__semantex_agent"))
                .count(),
            1
        );
        // All three tools now present.
        for tool in SEMANTEX_MCP_TOOLS {
            assert!(
                allow.iter().any(|v| v.as_str() == Some(tool)),
                "missing {tool}"
            );
        }
        // Total = 1 existing (Bash) + 3 semantex (one was already there) = 4.
        assert_eq!(allow.len(), 4);
        // Sibling `deny` key untouched.
        assert_eq!(settings["permissions"]["deny"], json!(["Bash(rm:*)"]));
        // Idempotent.
        let snapshot = settings.clone();
        ensure_tool_permissions(&mut settings);
        assert_eq!(settings, snapshot);
    }
}
