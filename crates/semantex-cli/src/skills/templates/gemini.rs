//! Gemini CLI extension manifest (`extension.json`).
//!
//! Gemini CLI extensions ship as a JSON manifest pointing at an MCP server
//! and a context file. We emit the manifest with a `contextFileName`
//! sibling that the operator should write alongside (the markdown body is
//! also included inside the JSON under `_context_md` for convenience).

use crate::skills::tools::ToolMetadata;
use crate::skills::{SKILL_INTRO, SKILL_TAGLINE};

pub fn render(tools: &[ToolMetadata]) -> String {
    let context_md = render_context_md(tools);
    let tool_list = tools
        .iter()
        .map(|t| serde_json::json!({ "name": t.name, "description": t.description }))
        .collect::<Vec<_>>();
    let manifest = serde_json::json!({
        "name": "semantex",
        "version": "0.3.0",
        "description": SKILL_TAGLINE,
        "mcpServers": {
            "semantex": {
                "command": "semantex",
                "args": ["mcp"],
            },
        },
        "contextFileName": "GEMINI.md",
        "_context_md": context_md,
        "_tools": tool_list,
    });
    serde_json::to_string_pretty(&manifest).expect("Gemini manifest serialises")
}

/// The section written into `GEMINI.md` by `install-gemini` — an `## `-level
/// heading so it nests as one owned section inside the hierarchical
/// `GEMINI.md` file, which may have other hand-written content.
pub(crate) fn render_body_md(tools: &[ToolMetadata]) -> String {
    render_context_md(tools)
}

fn render_context_md(tools: &[ToolMetadata]) -> String {
    let mut out = String::new();
    out.push_str("## semantex\n\n");
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
            serde_json::from_str(&out).expect("Gemini manifest must be valid JSON");
        assert_eq!(value.get("name").and_then(|v| v.as_str()), Some("semantex"));
        assert!(value.get("mcpServers").is_some());
        let context = value
            .get("_context_md")
            .and_then(|v| v.as_str())
            .expect("_context_md present");
        for tool in tools.iter().filter(|t| t.live) {
            assert!(
                context.contains(tool.name),
                "context missing `{}`",
                tool.name
            );
        }
    }
}
