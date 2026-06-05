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
