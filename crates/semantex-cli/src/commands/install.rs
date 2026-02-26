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

/// Install semantex hooks into Claude Code's settings.
/// Writes to ~/.claude/settings.json
pub fn install_claude_code() -> Result<()> {
    let binary = semantex_binary_path()?;
    let settings_path = dirs_home().join(".claude").join("settings.json");

    let mut settings: serde_json::Value = if settings_path.exists() {
        let content = std::fs::read_to_string(&settings_path)?;
        serde_json::from_str(&content)?
    } else {
        serde_json::json!({})
    };

    // Add hooks configuration
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
        "{} Installed semantex hooks into {}",
        "Done!".green().bold(),
        settings_path.display()
    );
    eprintln!("Restart Claude Code to activate.");
    Ok(())
}

/// Uninstall semantex hooks from Claude Code's settings.
pub fn uninstall_claude_code() -> Result<()> {
    let settings_path = dirs_home().join(".claude").join("settings.json");
    if !settings_path.exists() {
        eprintln!("No Claude Code settings found.");
        return Ok(());
    }

    let content = std::fs::read_to_string(&settings_path)?;
    let mut settings: serde_json::Value = serde_json::from_str(&content)?;

    if let Some(hooks) = settings.get_mut("hooks").and_then(|h| h.as_object_mut()) {
        hooks.remove("SessionStart");
        hooks.remove("PreToolUse");
        hooks.remove("SessionEnd");
    }

    std::fs::write(&settings_path, serde_json::to_string_pretty(&settings)?)?;
    eprintln!("Removed semantex hooks from {}", settings_path.display());
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
