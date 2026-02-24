use anyhow::{Context, Result};
use colored::Colorize;
use std::path::PathBuf;

/// Get the sage binary path for hook configuration.
fn sage_binary_path() -> Result<String> {
    std::env::current_exe()
        .map(|p| p.to_string_lossy().to_string())
        .context("Failed to determine sage binary path")
}

fn dirs_home() -> PathBuf {
    std::env::var("HOME").map_or_else(|_| PathBuf::from("/tmp"), PathBuf::from)
}

/// Install sage hooks into Claude Code's settings.
/// Writes to ~/.claude/settings.json
pub fn install_claude_code() -> Result<()> {
    let binary = sage_binary_path()?;
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

    // SessionStart hook — injects sage context + pre-warms daemon
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
        "{} Installed sage hooks into {}",
        "Done!".green().bold(),
        settings_path.display()
    );
    eprintln!("Restart Claude Code to activate.");
    Ok(())
}

/// Uninstall sage hooks from Claude Code's settings.
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
    eprintln!("Removed sage hooks from {}", settings_path.display());
    Ok(())
}

/// Install for OpenCode.
pub fn install_opencode() -> Result<()> {
    println!("{}", "Installing OpenCode integration...".green().bold());

    let sage_path = std::env::current_exe()?;
    let sage_path_str = sage_path.display();

    println!();
    println!("Add to your OpenCode configuration:");
    println!();
    println!("  sage_command: \"{sage_path_str}\"");
    println!();
    println!(
        "{}",
        "See sage documentation for full OpenCode integration details.".dimmed()
    );

    Ok(())
}

/// Install for Codex.
pub fn install_codex() -> Result<()> {
    println!("{}", "Installing Codex integration...".green().bold());

    let sage_path = std::env::current_exe()?;
    let sage_path_str = sage_path.display();

    println!();
    println!("Add to your Codex configuration:");
    println!();
    println!("  sage_command: \"{sage_path_str}\"");
    println!();
    println!(
        "{}",
        "See sage documentation for full Codex integration details.".dimmed()
    );

    Ok(())
}

/// Remove all sage hook registrations.
pub fn uninstall_all() -> Result<()> {
    let _ = uninstall_claude_code();
    eprintln!("Removed all sage hook registrations.");
    Ok(())
}
