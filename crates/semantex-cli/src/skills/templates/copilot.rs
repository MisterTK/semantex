//! GitHub Copilot CLI settings snippet (`settings.json.snippet`).
//!
//! Copilot CLI reads MCP server configuration from a `settings.json` file.
//! We emit a snippet the operator can merge into their existing settings:
//! it registers the semantex MCP server and stores a markdown rules body
//! under a private key so an operator can extract it.

use crate::skills::tools::ToolMetadata;
use crate::skills::{SKILL_INTRO, SKILL_TAGLINE};

pub fn render(tools: &[ToolMetadata]) -> String {
    let rules_md = render_rules_md(tools);
    let tool_list = tools
        .iter()
        .map(|t| {
            serde_json::json!({
                "name": t.name,
                "description": t.description,
                "live": t.live,
            })
        })
        .collect::<Vec<_>>();
    let snippet = serde_json::json!({
        "mcpServers": {
            "semantex": {
                "type": "stdio",
                "command": "semantex",
                "args": ["mcp"],
                "description": SKILL_TAGLINE,
            },
        },
        "_semantex_rules": rules_md,
        "_semantex_tools": tool_list,
    });
    serde_json::to_string_pretty(&snippet).expect("Copilot snippet serialises")
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
            serde_json::from_str(&out).expect("Copilot snippet must be valid JSON");
        assert!(value.get("mcpServers").is_some());
        let rules = value
            .get("_semantex_rules")
            .and_then(|v| v.as_str())
            .expect("rules body present");
        for tool in &tools {
            assert!(rules.contains(tool.name), "rules missing `{}`", tool.name);
        }
    }
}
