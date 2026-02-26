use anyhow::{Context, Result};
use colored::Colorize;
use std::path::PathBuf;

/// Get the semantex binary path for hook configuration.
fn semantex_binary_path() -> Result<String> {
    std::env::current_exe()
        .map(|p| p.to_string_lossy().to_string())
        .context("Failed to determine semantex binary path")
}

fn dirs_home() -> PathBuf {
    dirs::home_dir().unwrap_or_else(std::env::temp_dir)
}

/// Embedded SKILL.md — single source of truth lives in plugin/skills/semantex/SKILL.md.
const SKILL_MD: &str = include_str!("../../../../plugin/skills/semantex/SKILL.md");

/// Install semantex hooks + skill into Claude Code.
/// Writes hooks to ~/.claude/settings.json and skill to ~/.claude/skills/semantex/SKILL.md.
pub fn install_claude_code() -> Result<()> {
    let binary = semantex_binary_path()?;
    let home = dirs_home();
    let claude_dir = home.join(".claude");
    let settings_path = claude_dir.join("settings.json");

    // --- 1. Install hooks into settings.json ---
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

    // PreToolUse hooks — Grep|Glob nudge + Task sub-agent injection
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
                "matcher": "Task",
                "hooks": [{
                    "type": "command",
                    "command": format!("{binary} --task-hook"),
                    "timeout": 5,
                }]
            }
        ]),
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

    // --- 2. Install skill to ~/.claude/skills/semantex/SKILL.md ---
    let skill_dir = claude_dir.join("skills").join("semantex");
    std::fs::create_dir_all(&skill_dir)?;
    std::fs::write(skill_dir.join("SKILL.md"), SKILL_MD)?;

    eprintln!(
        "  {} skill → {}",
        "✓".green().bold(),
        skill_dir.join("SKILL.md").display()
    );

    eprintln!(
        "\n{} semantex installed for Claude Code. Restart Claude Code to activate.",
        "Done!".green().bold(),
    );
    Ok(())
}

/// Uninstall semantex hooks and skill from Claude Code.
pub fn uninstall_claude_code() -> Result<()> {
    let home = dirs_home();
    let claude_dir = home.join(".claude");
    let settings_path = claude_dir.join("settings.json");

    // Remove hooks from settings.json
    if settings_path.exists() {
        let content = std::fs::read_to_string(&settings_path)?;
        let mut settings: serde_json::Value = serde_json::from_str(&content)?;

        if let Some(hooks) = settings.get_mut("hooks").and_then(|h| h.as_object_mut()) {
            hooks.remove("SessionStart");
            hooks.remove("PreToolUse");
            hooks.remove("SessionEnd");
        }

        std::fs::write(&settings_path, serde_json::to_string_pretty(&settings)?)?;
        eprintln!("Removed semantex hooks from {}", settings_path.display());
    } else {
        eprintln!("No Claude Code settings found.");
    }

    // Remove skill directory
    let skill_dir = claude_dir.join("skills").join("semantex");
    if skill_dir.exists() {
        std::fs::remove_dir_all(&skill_dir)?;
        eprintln!("Removed semantex skill from {}", skill_dir.display());
    }

    Ok(())
}

/// Install for OpenCode.
pub fn install_opencode() -> Result<()> {
    println!("{}", "Installing OpenCode integration...".green().bold());

    let semantex_path = std::env::current_exe()?;
    let semantex_path_str = semantex_path.display();

    println!();
    println!("Add to your OpenCode configuration:");
    println!();
    println!("  semantex_command: \"{semantex_path_str}\"");
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

    let semantex_path = std::env::current_exe()?;
    let semantex_path_str = semantex_path.display();

    println!();
    println!("Add to your Codex configuration:");
    println!();
    println!("  semantex_command: \"{semantex_path_str}\"");
    println!();
    println!(
        "{}",
        "See semantex documentation for full Codex integration details.".dimmed()
    );

    Ok(())
}

/// Remove all semantex hook registrations.
pub fn uninstall_all() -> Result<()> {
    let _ = uninstall_claude_code();
    eprintln!("Removed all semantex hook registrations.");
    Ok(())
}
