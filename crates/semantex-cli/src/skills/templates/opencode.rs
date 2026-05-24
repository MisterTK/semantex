//! OpenCode `.opencode.json` formatter.
//!
//! OpenCode's project configuration lives in `.opencode.json`. The exact
//! schema for skills/rules is still evolving, so we emit a defensible
//! shape: an `mcpServers` block registering semantex, plus a `rules` field
//! containing a markdown body that documents the tools. The format is
//! best-effort and may need updating once OpenCode publishes a stable
//! schema.

use crate::skills::tools::ToolMetadata;
use crate::skills::{SKILL_INTRO, SKILL_TAGLINE};

pub fn render(tools: &[ToolMetadata]) -> String {
    let rules_md = render_rules_md(tools);
    let config = serde_json::json!({
        "$schema": "https://opencode.ai/config.json",
        "mcp": {
            "semantex": {
                "type": "local",
                "command": ["semantex", "mcp"],
                "enabled": true,
                "description": SKILL_TAGLINE,
            }
        },
        "rules": rules_md,
        "instructions": [SKILL_TAGLINE],
    });
    serde_json::to_string_pretty(&config).expect("OpenCode config serialises")
}

fn render_rules_md(tools: &[ToolMetadata]) -> String {
    let mut out = String::new();
    out.push_str("# semantex — semantic code search\n\n");
    out.push_str(SKILL_INTRO);
    out.push_str("\n\n");
    out.push_str(concat!(
        "Prefer `semantex_agent` over chained grep + read. One call replaces 5–10 ",
        "iterations.\n\n",
    ));
    out.push_str("## Tools\n\n");
    out.push_str(&super::render_tools_md(tools));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skills::tools::all_tools;

    #[test]
    fn parses_as_valid_json() {
        let tools = all_tools();
        let out = render(&tools);
        let value: serde_json::Value =
            serde_json::from_str(&out).expect("OpenCode config must be valid JSON");
        assert!(value.get("mcp").is_some());
        let rules = value
            .get("rules")
            .and_then(|v| v.as_str())
            .expect("rules present");
        for tool in &tools {
            assert!(rules.contains(tool.name), "rules missing `{}`", tool.name);
        }
    }
}
