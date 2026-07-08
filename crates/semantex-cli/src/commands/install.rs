use anyhow::{Context, Result};
use colored::Colorize;
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::skills::templates;
use crate::skills::tools::all_tools;

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

// ─────────────────────────────────────────────────────────────────────────
// Shared helpers for the direct-install platforms below (Cursor, Devin
// Desktop, Aider, Trae, Copilot, Gemini). Unlike `install_claude_code`,
// these platforms have no hook system — each installer just registers the
// MCP server (where the platform supports MCP at all) and writes an
// always-on instructions/rules file to the platform's real, documented
// config path. All writes are idempotent: JSON/YAML configs are
// parsed-merged-rewritten (never clobbering unrelated keys), and
// semantex-owned sections inside shared markdown files are replaced in
// place by an exact-match heading marker, never appended twice.
// ─────────────────────────────────────────────────────────────────────────

/// Read a JSON file, defaulting to `{}` if missing or unparseable. Never
/// errors — an unparseable file is treated the same as an absent one so a
/// broken third-party config can't block installation (the write below
/// still only touches the `semantex` key, so nothing else is lost unless
/// the file was already corrupt).
fn read_json_or_default(path: &Path) -> serde_json::Value {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|c| serde_json::from_str(&c).ok())
        .unwrap_or_else(|| serde_json::json!({}))
}

fn write_json_pretty(path: &Path, value: &serde_json::Value) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, serde_json::to_string_pretty(value)?)
        .with_context(|| format!("writing {}", path.display()))
}

fn write_text_file(path: &Path, content: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, content).with_context(|| format!("writing {}", path.display()))
}

/// Merge `{"type":"stdio","command":binary,"args":["mcp"]}` into
/// `root[key].semantex`, creating `key` (and `root`) if absent. Generic
/// over the top-level key name since platforms disagree on it — most use
/// `mcpServers`, VS Code's `.vscode/mcp.json` uses `servers`.
fn merge_mcp_entry(root: &mut serde_json::Value, key: &str, binary: &str) {
    if !root.is_object() {
        *root = serde_json::json!({});
    }
    let obj = root.as_object_mut().expect("root coerced to object");
    let servers = obj.entry(key).or_insert_with(|| serde_json::json!({}));
    if !servers.is_object() {
        *servers = serde_json::json!({});
    }
    servers
        .as_object_mut()
        .expect("servers coerced to object")
        .insert("semantex".to_string(), semantex_mcp_server(binary));
}

/// Inverse of [`merge_mcp_entry`]: drop the `semantex` key, leaving every
/// other server registration untouched. Returns whether a write happened.
fn remove_mcp_entry(path: &Path, key: &str) -> Result<bool> {
    if !path.exists() {
        return Ok(false);
    }
    let mut root = read_json_or_default(path);
    let removed = root
        .as_object_mut()
        .and_then(|o| o.get_mut(key))
        .and_then(|s| s.as_object_mut())
        .is_some_and(|servers| servers.remove("semantex").is_some());
    if removed {
        write_json_pretty(path, &root)?;
    }
    Ok(removed)
}

/// Sentinel comments delimiting a semantex-owned section inside a shared
/// markdown file. Deliberately NOT heading-based (e.g. matching the next
/// `## ` line): the rendered body itself contains a nested `## Tools`
/// heading, which a heading-level boundary scan would mistake for the
/// start of the *next* section, truncating our own content and orphaning
/// the rest back into the file on every re-run. Explicit sentinels are
/// immune to whatever heading structure the body happens to use.
const SEMANTEX_SECTION_BEGIN: &str = "<!-- semantex:begin -->";
const SEMANTEX_SECTION_END: &str = "<!-- semantex:end -->";

/// Idempotently replace or append the semantex-owned section (delimited by
/// [`SEMANTEX_SECTION_BEGIN`]/[`SEMANTEX_SECTION_END`]) in a shared
/// markdown file. Never touches content outside the sentinels, so it's
/// safe to run against a file the user also hand-edits.
fn replace_or_append_section(content: &str, new_section: &str) -> String {
    let wrapped = format!(
        "{SEMANTEX_SECTION_BEGIN}\n{}\n{SEMANTEX_SECTION_END}",
        new_section.trim()
    );

    if let Some(start) = content.find(SEMANTEX_SECTION_BEGIN)
        && let Some(end_rel) = content[start..].find(SEMANTEX_SECTION_END)
    {
        let end = start + end_rel + SEMANTEX_SECTION_END.len();
        let head = content[..start].trim_end();
        let tail = content[end..].trim_start();
        let mut parts: Vec<&str> = Vec::new();
        if !head.is_empty() {
            parts.push(head);
        }
        parts.push(&wrapped);
        if !tail.is_empty() {
            parts.push(tail);
        }
        let mut out = parts.join("\n\n");
        if !out.ends_with('\n') {
            out.push('\n');
        }
        return out;
    }

    if content.trim().is_empty() {
        format!("{wrapped}\n")
    } else {
        format!("{}\n\n{wrapped}\n", content.trim_end())
    }
}

/// Remove the semantex-owned section from a shared markdown file (inverse
/// of [`replace_or_append_section`]). Deletes the file if that was the
/// only content; leaves the file untouched if the sentinels aren't present
/// (or malformed — begin with no matching end).
fn remove_owned_section(path: &Path) -> Result<bool> {
    if !path.exists() {
        return Ok(false);
    }
    let content = std::fs::read_to_string(path)?;
    let Some(start) = content.find(SEMANTEX_SECTION_BEGIN) else {
        return Ok(false);
    };
    let Some(end_rel) = content[start..].find(SEMANTEX_SECTION_END) else {
        return Ok(false);
    };
    let end = start + end_rel + SEMANTEX_SECTION_END.len();
    let head = content[..start].trim_end();
    let tail = content[end..].trim_start();
    let cleaned = [head, tail]
        .iter()
        .filter(|s| !s.is_empty())
        .copied()
        .collect::<Vec<_>>()
        .join("\n\n");
    if cleaned.trim().is_empty() {
        std::fs::remove_file(path)?;
    } else {
        let mut out = cleaned;
        if !out.ends_with('\n') {
            out.push('\n');
        }
        std::fs::write(path, out)?;
    }
    Ok(true)
}

fn remove_file_if_exists(path: &Path) -> Result<bool> {
    if path.exists() {
        std::fs::remove_file(path)?;
        Ok(true)
    } else {
        Ok(false)
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Cursor — `.cursor/mcp.json` (`~/.cursor/mcp.json` with `--global`), key
// `mcpServers`; rules at `.cursor/rules/semantex.mdc` (Cursor requires the
// `.mdc` extension — a plain `.md` in that directory is silently ignored).
// Verified against https://cursor.com/docs/mcp and
// https://cursor.com/docs/context/rules (2026-07-08).
// ─────────────────────────────────────────────────────────────────────────

fn cursor_mcp_path(global: bool) -> PathBuf {
    if global {
        dirs_home().join(".cursor").join("mcp.json")
    } else {
        PathBuf::from(".cursor").join("mcp.json")
    }
}

fn cursor_rules_path() -> PathBuf {
    PathBuf::from(".cursor").join("rules").join("semantex.mdc")
}

/// Install semantex for Cursor.
pub fn install_cursor(global: bool) -> Result<()> {
    let binary = semantex_binary_path();

    let mcp_path = cursor_mcp_path(global);
    let mut mcp_json = read_json_or_default(&mcp_path);
    merge_mcp_entry(&mut mcp_json, "mcpServers", binary);
    write_json_pretty(&mcp_path, &mcp_json)?;
    eprintln!(
        "  {} MCP server → {}",
        "✓".green().bold(),
        mcp_path.display()
    );

    let rules_path = cursor_rules_path();
    write_text_file(&rules_path, &templates::cursor::render(&all_tools()))?;
    eprintln!("  {} rule → {}", "✓".green().bold(), rules_path.display());

    eprintln!(
        "\n{} semantex installed for Cursor. Restart Cursor to activate.",
        "Done!".green().bold()
    );
    Ok(())
}

pub fn uninstall_cursor(global: bool) -> Result<()> {
    let mut any = remove_mcp_entry(&cursor_mcp_path(global), "mcpServers")?;
    any |= remove_file_if_exists(&cursor_rules_path())?;
    if any {
        eprintln!("{} semantex removed from Cursor.", "✓".green().bold());
    } else {
        eprintln!("No semantex Cursor install found to remove.");
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────
// Devin Desktop (formerly Windsurf) — MCP is global-only:
// `~/.codeium/windsurf/mcp_config.json`, key `mcpServers` (path carried
// over unchanged through the 2026-06-02 rebrand). Rules are written to
// both the new preferred `.devin/rules/semantex.md` and the legacy
// `.windsurf/rules/semantex.md` fallback — both are read, `.devin/` wins
// on conflict — so this works whether the user is still on classic
// Windsurf or has moved to Devin Desktop.
// ─────────────────────────────────────────────────────────────────────────

fn devin_desktop_mcp_path() -> PathBuf {
    dirs_home()
        .join(".codeium")
        .join("windsurf")
        .join("mcp_config.json")
}

fn devin_desktop_rules_paths() -> [PathBuf; 2] {
    [
        PathBuf::from(".devin").join("rules").join("semantex.md"),
        PathBuf::from(".windsurf").join("rules").join("semantex.md"),
    ]
}

/// Install semantex for Devin Desktop / Windsurf.
pub fn install_devin_desktop() -> Result<()> {
    let binary = semantex_binary_path();

    let mcp_path = devin_desktop_mcp_path();
    let mut mcp_json = read_json_or_default(&mcp_path);
    merge_mcp_entry(&mut mcp_json, "mcpServers", binary);
    write_json_pretty(&mcp_path, &mcp_json)?;
    eprintln!(
        "  {} MCP server → {}",
        "✓".green().bold(),
        mcp_path.display()
    );

    let body = templates::devin_desktop::render(&all_tools());
    for rules_path in devin_desktop_rules_paths() {
        write_text_file(&rules_path, &body)?;
        eprintln!("  {} rule → {}", "✓".green().bold(), rules_path.display());
    }

    eprintln!(
        "\n{} semantex installed for Devin Desktop / Windsurf. Restart to activate.",
        "Done!".green().bold()
    );
    Ok(())
}

pub fn uninstall_devin_desktop() -> Result<()> {
    let mut any = remove_mcp_entry(&devin_desktop_mcp_path(), "mcpServers")?;
    for rules_path in devin_desktop_rules_paths() {
        any |= remove_file_if_exists(&rules_path)?;
    }
    if any {
        eprintln!(
            "{} semantex removed from Devin Desktop / Windsurf.",
            "✓".green().bold()
        );
    } else {
        eprintln!("No semantex Devin Desktop / Windsurf install found to remove.");
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────
// Aider — no native MCP support exists (confirmed against current docs:
// an open RFC, an unmerged PR, nothing shipped). The only integration
// lever is `.aider.conf.yml`'s `read:` key, auto-discovered with no
// flags, pointing at a markdown file that gets injected as read-only
// context every session.
// ─────────────────────────────────────────────────────────────────────────

const AIDER_CONTEXT_PATH: &str = ".aider/semantex.md";

fn aider_conf_path() -> PathBuf {
    PathBuf::from(".aider.conf.yml")
}

/// Install semantex for Aider.
pub fn install_aider() -> Result<()> {
    let rules_path = PathBuf::from(AIDER_CONTEXT_PATH);
    write_text_file(&rules_path, &templates::aider::render_body_md(&all_tools()))?;
    eprintln!(
        "  {} context → {}",
        "✓".green().bold(),
        rules_path.display()
    );

    let conf_path = aider_conf_path();
    let mut conf: serde_yml::Value = if conf_path.exists() {
        let content = std::fs::read_to_string(&conf_path)?;
        serde_yml::from_str(&content)
            .unwrap_or_else(|_| serde_yml::Value::Mapping(serde_yml::Mapping::default()))
    } else {
        serde_yml::Value::Mapping(serde_yml::Mapping::default())
    };
    if !conf.is_mapping() {
        conf = serde_yml::Value::Mapping(serde_yml::Mapping::default());
    }
    let map = conf.as_mapping_mut().expect("coerced to mapping above");
    let read_key = serde_yml::Value::String("read".to_string());
    let entry = serde_yml::Value::String(AIDER_CONTEXT_PATH.to_string());
    match map.get_mut(&read_key) {
        Some(serde_yml::Value::Sequence(seq)) => {
            if !seq.iter().any(|v| v.as_str() == Some(AIDER_CONTEXT_PATH)) {
                seq.push(entry);
            }
        }
        Some(existing) if existing.as_str() != Some(AIDER_CONTEXT_PATH) => {
            let prev = existing.clone();
            *existing = serde_yml::Value::Sequence(vec![prev, entry]);
        }
        Some(_) => {}
        None => {
            map.insert(read_key, serde_yml::Value::Sequence(vec![entry]));
        }
    }
    std::fs::write(&conf_path, serde_yml::to_string(&conf)?)?;
    eprintln!(
        "  {} config → {} (`read:` updated)",
        "✓".green().bold(),
        conf_path.display()
    );

    eprintln!(
        "\n{} semantex installed for Aider. Note: Aider has no native MCP support (confirmed as \
         of 2026-07 — an open RFC only), so this wires semantex's tool guide into every \
         session's context instead of registering a live tool.",
        "Done!".green().bold()
    );
    Ok(())
}

pub fn uninstall_aider() -> Result<()> {
    let mut any = remove_file_if_exists(&PathBuf::from(AIDER_CONTEXT_PATH))?;
    let conf_path = aider_conf_path();
    if conf_path.exists() {
        let content = std::fs::read_to_string(&conf_path)?;
        if let Ok(mut conf) = serde_yml::from_str::<serde_yml::Value>(&content)
            && let Some(map) = conf.as_mapping_mut()
        {
            let read_key = serde_yml::Value::String("read".to_string());
            if let Some(serde_yml::Value::Sequence(seq)) = map.get_mut(&read_key) {
                let before = seq.len();
                seq.retain(|v| v.as_str() != Some(AIDER_CONTEXT_PATH));
                if seq.len() != before {
                    any = true;
                    if seq.is_empty() {
                        map.remove(&read_key);
                    }
                    std::fs::write(&conf_path, serde_yml::to_string(&conf)?)?;
                }
            }
        }
    }
    if any {
        eprintln!("{} semantex removed from Aider.", "✓".green().bold());
    } else {
        eprintln!("No semantex Aider install found to remove.");
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────
// Trae — MCP at `.trae/mcp.json` (key `mcpServers`); skill file at
// `.trae/skills/semantex/SKILL.md`, the same discovery convention as
// Claude Code's `.claude/skills/` (verified against docs.trae.ai
// 2026-07-08 — graphify already uses this exact skill path).
// ─────────────────────────────────────────────────────────────────────────

fn trae_mcp_path() -> PathBuf {
    PathBuf::from(".trae").join("mcp.json")
}

fn trae_skill_path() -> PathBuf {
    PathBuf::from(".trae")
        .join("skills")
        .join("semantex")
        .join("SKILL.md")
}

/// Install semantex for Trae.
pub fn install_trae() -> Result<()> {
    let binary = semantex_binary_path();

    let mcp_path = trae_mcp_path();
    let mut mcp_json = read_json_or_default(&mcp_path);
    merge_mcp_entry(&mut mcp_json, "mcpServers", binary);
    write_json_pretty(&mcp_path, &mcp_json)?;
    eprintln!(
        "  {} MCP server → {}",
        "✓".green().bold(),
        mcp_path.display()
    );

    let skill_path = trae_skill_path();
    write_text_file(&skill_path, &templates::trae::render(&all_tools()))?;
    eprintln!("  {} skill → {}", "✓".green().bold(), skill_path.display());

    eprintln!(
        "\n{} semantex installed for Trae. Restart Trae to activate.",
        "Done!".green().bold()
    );
    Ok(())
}

pub fn uninstall_trae() -> Result<()> {
    let mut any = remove_mcp_entry(&trae_mcp_path(), "mcpServers")?;
    any |= remove_file_if_exists(&trae_skill_path())?;
    if any {
        eprintln!("{} semantex removed from Trae.", "✓".green().bold());
    } else {
        eprintln!("No semantex Trae install found to remove.");
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────
// GitHub Copilot — two DISTINCT MCP surfaces with different formats,
// both registered: VS Code Copilot Chat's `.vscode/mcp.json` (top-level
// key `servers`, not `mcpServers`) and the standalone Copilot CLI's
// `~/.copilot/mcp-config.json` (top-level key `mcpServers`). Instructions
// go to the shared `.github/copilot-instructions.md`. Verified against
// code.visualstudio.com/docs and docs.github.com (2026-07-08).
// ─────────────────────────────────────────────────────────────────────────

fn copilot_vscode_mcp_path() -> PathBuf {
    PathBuf::from(".vscode").join("mcp.json")
}

fn copilot_cli_mcp_path() -> PathBuf {
    dirs_home().join(".copilot").join("mcp-config.json")
}

fn copilot_instructions_path() -> PathBuf {
    PathBuf::from(".github").join("copilot-instructions.md")
}

/// Install semantex for GitHub Copilot (VS Code Copilot Chat + Copilot CLI).
pub fn install_copilot() -> Result<()> {
    let binary = semantex_binary_path();

    let vscode_path = copilot_vscode_mcp_path();
    let mut vscode_json = read_json_or_default(&vscode_path);
    merge_mcp_entry(&mut vscode_json, "servers", binary);
    write_json_pretty(&vscode_path, &vscode_json)?;
    eprintln!(
        "  {} MCP server (VS Code) → {}",
        "✓".green().bold(),
        vscode_path.display()
    );

    let cli_path = copilot_cli_mcp_path();
    let mut cli_json = read_json_or_default(&cli_path);
    merge_mcp_entry(&mut cli_json, "mcpServers", binary);
    write_json_pretty(&cli_path, &cli_json)?;
    eprintln!(
        "  {} MCP server (Copilot CLI) → {}",
        "✓".green().bold(),
        cli_path.display()
    );

    let instructions_path = copilot_instructions_path();
    let existing = std::fs::read_to_string(&instructions_path).unwrap_or_default();
    let updated =
        replace_or_append_section(&existing, &templates::copilot::render_body_md(&all_tools()));
    write_text_file(&instructions_path, &updated)?;
    eprintln!(
        "  {} instructions → {}",
        "✓".green().bold(),
        instructions_path.display()
    );

    eprintln!(
        "\n{} semantex installed for GitHub Copilot. Restart VS Code / Copilot CLI to activate.",
        "Done!".green().bold()
    );
    Ok(())
}

pub fn uninstall_copilot() -> Result<()> {
    let mut any = remove_mcp_entry(&copilot_vscode_mcp_path(), "servers")?;
    any |= remove_mcp_entry(&copilot_cli_mcp_path(), "mcpServers")?;
    any |= remove_owned_section(&copilot_instructions_path())?;
    if any {
        eprintln!(
            "{} semantex removed from GitHub Copilot.",
            "✓".green().bold()
        );
    } else {
        eprintln!("No semantex GitHub Copilot install found to remove.");
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────
// Gemini CLI — `.gemini/settings.json` (`~/.gemini/settings.json` with
// `--global`), key `mcpServers`; context in the hierarchical `GEMINI.md`.
// Verified against geminicli.com/docs (2026-07-08).
// ─────────────────────────────────────────────────────────────────────────

fn gemini_settings_path(global: bool) -> PathBuf {
    if global {
        dirs_home().join(".gemini").join("settings.json")
    } else {
        PathBuf::from(".gemini").join("settings.json")
    }
}

fn gemini_context_path() -> PathBuf {
    PathBuf::from("GEMINI.md")
}

/// Install semantex for Gemini CLI.
pub fn install_gemini(global: bool) -> Result<()> {
    let binary = semantex_binary_path();

    let settings_path = gemini_settings_path(global);
    let mut settings = read_json_or_default(&settings_path);
    merge_mcp_entry(&mut settings, "mcpServers", binary);
    write_json_pretty(&settings_path, &settings)?;
    eprintln!(
        "  {} MCP server → {}",
        "✓".green().bold(),
        settings_path.display()
    );

    let context_path = gemini_context_path();
    let existing = std::fs::read_to_string(&context_path).unwrap_or_default();
    let updated =
        replace_or_append_section(&existing, &templates::gemini::render_body_md(&all_tools()));
    write_text_file(&context_path, &updated)?;
    eprintln!(
        "  {} context → {}",
        "✓".green().bold(),
        context_path.display()
    );

    eprintln!(
        "\n{} semantex installed for Gemini CLI. Restart to activate.",
        "Done!".green().bold()
    );
    Ok(())
}

pub fn uninstall_gemini(global: bool) -> Result<()> {
    let mut any = remove_mcp_entry(&gemini_settings_path(global), "mcpServers")?;
    any |= remove_owned_section(&gemini_context_path())?;
    if any {
        eprintln!("{} semantex removed from Gemini CLI.", "✓".green().bold());
    } else {
        eprintln!("No semantex Gemini CLI install found to remove.");
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────
// OpenCode — `opencode.json` (project root; global `~/.config/opencode/
// opencode.json`), top-level key `mcp` (NOT `mcpServers`), entries shaped
// `{"type":"local","command":[...],"enabled":true}` (command+args as ONE
// array, not separate fields). Instructions go through the `instructions`
// array (a list of markdown file paths merged with, not replacing,
// AGENTS.md) pointing at `.opencode/semantex.md` — this is the
// non-destructive option since it doesn't require owning AGENTS.md.
// `opencode mcp add` is interactive-only (no scriptable flags), so direct
// file edits are the only automatable path. Verified against
// opencode.ai/docs (2026-07-08).
// ─────────────────────────────────────────────────────────────────────────

const OPENCODE_INSTRUCTIONS_PATH: &str = ".opencode/semantex.md";

fn opencode_config_path(global: bool) -> PathBuf {
    if global {
        dirs_home().join(".config").join("opencode").join("opencode.json")
    } else {
        PathBuf::from("opencode.json")
    }
}

fn opencode_mcp_entry(binary: &str) -> serde_json::Value {
    serde_json::json!({
        "type": "local",
        "command": [binary, "mcp"],
        "enabled": true,
    })
}

/// Install semantex for OpenCode.
pub fn install_opencode(global: bool) -> Result<()> {
    let binary = semantex_binary_path();

    let config_path = opencode_config_path(global);
    let mut config = read_json_or_default(&config_path);
    if !config.is_object() {
        config = serde_json::json!({});
    }
    let obj = config.as_object_mut().expect("config coerced to object");
    obj.entry("$schema")
        .or_insert_with(|| serde_json::json!("https://opencode.ai/config.json"));
    let mcp = obj.entry("mcp").or_insert_with(|| serde_json::json!({}));
    if !mcp.is_object() {
        *mcp = serde_json::json!({});
    }
    mcp.as_object_mut()
        .expect("mcp coerced to object")
        .insert("semantex".to_string(), opencode_mcp_entry(binary));

    let instructions = obj
        .entry("instructions")
        .or_insert_with(|| serde_json::json!([]));
    if !instructions.is_array() {
        *instructions = serde_json::json!([]);
    }
    let arr = instructions.as_array_mut().expect("instructions is array");
    if !arr
        .iter()
        .any(|v| v.as_str() == Some(OPENCODE_INSTRUCTIONS_PATH))
    {
        arr.push(serde_json::json!(OPENCODE_INSTRUCTIONS_PATH));
    }

    write_json_pretty(&config_path, &config)?;
    eprintln!("  {} MCP server → {}", "✓".green().bold(), config_path.display());

    let instructions_path = PathBuf::from(OPENCODE_INSTRUCTIONS_PATH);
    write_text_file(&instructions_path, &templates::opencode::render_body_md(&all_tools()))?;
    eprintln!(
        "  {} instructions → {}",
        "✓".green().bold(),
        instructions_path.display()
    );

    eprintln!(
        "\n{} semantex installed for OpenCode. Restart to activate.",
        "Done!".green().bold()
    );
    Ok(())
}

pub fn uninstall_opencode(global: bool) -> Result<()> {
    let config_path = opencode_config_path(global);
    let mut any = false;
    if config_path.exists() {
        let mut config = read_json_or_default(&config_path);
        if let Some(obj) = config.as_object_mut() {
            if let Some(mcp) = obj.get_mut("mcp").and_then(|v| v.as_object_mut())
                && mcp.remove("semantex").is_some()
            {
                any = true;
            }
            if let Some(instructions) = obj.get_mut("instructions").and_then(|v| v.as_array_mut())
            {
                let before = instructions.len();
                instructions.retain(|v| v.as_str() != Some(OPENCODE_INSTRUCTIONS_PATH));
                if instructions.len() != before {
                    any = true;
                }
            }
            if any {
                write_json_pretty(&config_path, &config)?;
            }
        }
    }
    any |= remove_file_if_exists(&PathBuf::from(OPENCODE_INSTRUCTIONS_PATH))?;
    if any {
        eprintln!("{} semantex removed from OpenCode.", "✓".green().bold());
    } else {
        eprintln!("No semantex OpenCode install found to remove.");
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────
// Codex CLI (OpenAI's `codex`, github.com/openai/codex — NOT GitHub
// Copilot) — MCP config is TOML (not JSON): `~/.codex/config.toml` under
// `[mcp_servers.semantex]`. Project-scoped `.codex/config.toml` exists but
// is unreliable per multiple open upstream bugs (openai/codex#13025,
// #9676), so the installer targets the global file, matching what
// `codex mcp add` itself writes. Instructions go to the project-root
// `AGENTS.md` (auto-loaded with no config — this is the tool AGENTS.md
// originated from). Verified against developers.openai.com/codex/mcp and
// developers.openai.com/codex/guides/agents-md (2026-07-08).
// ─────────────────────────────────────────────────────────────────────────

fn codex_config_path() -> PathBuf {
    dirs_home().join(".codex").join("config.toml")
}

fn codex_agents_path() -> PathBuf {
    PathBuf::from("AGENTS.md")
}

fn read_toml_or_default(path: &Path) -> toml::Value {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|c| c.parse::<toml::Value>().ok())
        .unwrap_or_else(|| toml::Value::Table(toml::value::Table::new()))
}

fn write_toml_pretty(path: &Path, value: &toml::Value) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, toml::to_string_pretty(value)?)
        .with_context(|| format!("writing {}", path.display()))
}

fn merge_toml_mcp_entry(root: &mut toml::Value, binary: &str) {
    if !root.is_table() {
        *root = toml::Value::Table(toml::value::Table::new());
    }
    let table = root.as_table_mut().expect("root coerced to table");
    let servers = table
        .entry("mcp_servers".to_string())
        .or_insert_with(|| toml::Value::Table(toml::value::Table::new()));
    if !servers.is_table() {
        *servers = toml::Value::Table(toml::value::Table::new());
    }
    let mut entry = toml::value::Table::new();
    entry.insert("command".to_string(), toml::Value::String(binary.to_string()));
    entry.insert(
        "args".to_string(),
        toml::Value::Array(vec![toml::Value::String("mcp".to_string())]),
    );
    servers
        .as_table_mut()
        .expect("mcp_servers coerced to table")
        .insert("semantex".to_string(), toml::Value::Table(entry));
}

fn remove_toml_mcp_entry(path: &Path) -> Result<bool> {
    if !path.exists() {
        return Ok(false);
    }
    let mut root = read_toml_or_default(path);
    let removed = root
        .as_table_mut()
        .and_then(|t| t.get_mut("mcp_servers"))
        .and_then(|s| s.as_table_mut())
        .is_some_and(|servers| servers.remove("semantex").is_some());
    if removed {
        write_toml_pretty(path, &root)?;
    }
    Ok(removed)
}

/// Install semantex for Codex CLI.
pub fn install_codex() -> Result<()> {
    let binary = semantex_binary_path();

    let config_path = codex_config_path();
    let mut config = read_toml_or_default(&config_path);
    merge_toml_mcp_entry(&mut config, binary);
    write_toml_pretty(&config_path, &config)?;
    eprintln!("  {} MCP server → {}", "✓".green().bold(), config_path.display());

    let agents_path = codex_agents_path();
    let existing = std::fs::read_to_string(&agents_path).unwrap_or_default();
    let updated = replace_or_append_section(&existing, &templates::codex::render_body_md(&all_tools()));
    write_text_file(&agents_path, &updated)?;
    eprintln!("  {} instructions → {}", "✓".green().bold(), agents_path.display());

    eprintln!(
        "\n{} semantex installed for Codex CLI. Restart to activate.",
        "Done!".green().bold()
    );
    Ok(())
}

pub fn uninstall_codex() -> Result<()> {
    let mut any = remove_toml_mcp_entry(&codex_config_path())?;
    any |= remove_owned_section(&codex_agents_path())?;
    if any {
        eprintln!("{} semantex removed from Codex CLI.", "✓".green().bold());
    } else {
        eprintln!("No semantex Codex CLI install found to remove.");
    }
    Ok(())
}

/// Remove all semantex hook registrations across every platform this CLI
/// knows how to install for. Each platform's uninstaller no-ops (with a
/// "nothing to remove" message) if it was never installed, so this is safe
/// to run unconditionally.
pub fn uninstall_all() -> Result<()> {
    uninstall_claude_code()?;
    uninstall_cursor(false)?;
    uninstall_cursor(true)?;
    uninstall_devin_desktop()?;
    uninstall_aider()?;
    uninstall_trae()?;
    uninstall_copilot()?;
    uninstall_gemini(false)?;
    uninstall_gemini(true)?;
    uninstall_codex()?;
    uninstall_opencode(false)?;
    uninstall_opencode(true)?;
    eprintln!("Removed all semantex hook registrations.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn replace_or_append_section_survives_nested_h2_headings_in_body() {
        // Regression test: an earlier implementation matched the section
        // boundary against the next `## `-prefixed line, which broke as
        // soon as the rendered body contained its OWN `## Tools` heading —
        // it truncated the owned section right there and orphaned
        // everything after it back into the file on every re-run. Sentinel
        // comments must be immune to whatever heading structure the body
        // uses internally.
        let user_content =
            "# My repo\n\nSome hand-written notes.\n\n## other-tool\n\nUnrelated section.\n";
        let body = "## semantex\n\nintro text\n\n## Tools\n\n### tool_one\n\ndetails\n";

        let installed = replace_or_append_section(user_content, body);
        assert!(installed.contains("Some hand-written notes."));
        assert!(installed.contains("## other-tool"));
        assert!(installed.contains("Unrelated section."));
        assert!(installed.contains("## Tools"));
        assert!(installed.contains("### tool_one"));

        // Re-running install must not duplicate the section.
        let reinstalled = replace_or_append_section(&installed, body);
        assert_eq!(reinstalled.matches(SEMANTEX_SECTION_BEGIN).count(), 1);
        assert_eq!(reinstalled.matches("## Tools").count(), 1);

        // Uninstall must remove the ENTIRE owned section (including the
        // nested "## Tools" sub-heading that used to get orphaned) while
        // leaving unrelated user content untouched.
        let dir = std::env::temp_dir().join(format!(
            "semantex-install-test-{:?}",
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("shared.md");
        std::fs::write(&path, &reinstalled).unwrap();
        let removed = remove_owned_section(&path).unwrap();
        assert!(removed);
        let after = std::fs::read_to_string(&path).unwrap();
        assert!(after.contains("Some hand-written notes."));
        assert!(after.contains("## other-tool"));
        assert!(
            !after.contains("## Tools"),
            "orphaned nested heading survived uninstall: {after}"
        );
        assert!(!after.contains("semantex:begin"));
        std::fs::remove_dir_all(&dir).ok();
    }

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
