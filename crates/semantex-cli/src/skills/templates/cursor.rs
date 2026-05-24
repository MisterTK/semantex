//! Cursor `.cursor/rules/semantex.mdc` formatter.
//!
//! Output: markdown with a small front-matter-style header (`description`,
//! `globs`, `alwaysApply`) per Cursor's project-rules convention. The body
//! documents semantex's MCP tools so the Cursor agent picks them over
//! built-in grep.

use crate::skills::tools::ToolMetadata;
use crate::skills::{SKILL_INTRO, SKILL_TAGLINE};
use std::fmt::Write;

pub fn render(tools: &[ToolMetadata]) -> String {
    let mut out = String::new();
    // Cursor rule front matter is YAML between `---` fences.
    out.push_str("---\n");
    let _ = writeln!(out, "description: \"{}\"", escape_yaml(SKILL_TAGLINE));
    out.push_str("globs: [\"**/*\"]\n");
    out.push_str("alwaysApply: true\n");
    out.push_str("---\n\n");

    out.push_str("# semantex — Semantic Code Search\n\n");
    out.push_str(SKILL_INTRO);
    out.push_str("\n\n");
    out.push_str(concat!(
        "Whenever you would otherwise run a series of grep + read calls, prefer the ",
        "`semantex_agent` MCP tool. One call replaces 5–10 grep + read iterations.\n\n",
    ));

    out.push_str("## Tools\n\n");
    out.push_str(&super::render_tools_md(tools));

    out.push_str("## Fallbacks\n\n");
    out.push_str(
        "Only fall back to Cursor's native search when you need an exact regex match or a glob lookup.\n",
    );
    out
}

fn escape_yaml(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skills::tools::all_tools;

    #[test]
    fn parses_as_markdown_with_yaml_header() {
        let tools = all_tools();
        let out = render(&tools);
        assert!(out.starts_with("---\n"));
        let body_start = out
            .find("\n---\n")
            .expect("front matter should close with `---`");
        let header = &out[4..body_start];
        let _: serde_yml::Value =
            serde_yml::from_str(header).expect("Cursor header must be valid YAML");
        assert!(out.contains("semantex_agent"));
        // Markdown body must mention every tool.
        for tool in &tools {
            assert!(out.contains(tool.name), "missing tool `{}`", tool.name);
        }
    }
}
