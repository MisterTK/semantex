//! Claude Code SKILL.md formatter.
//!
//! Output: YAML front matter with `name` + `description` followed by a
//! markdown body documenting every MCP tool. Mirrors the conventions of
//! `plugin/skills/semantex/SKILL.md`.

use crate::skills::tools::ToolMetadata;
use crate::skills::{SKILL_INTRO, SKILL_TAGLINE};
use std::fmt::Write;

pub fn render(tools: &[ToolMetadata]) -> String {
    let mut out = String::new();
    out.push_str("---\n");
    out.push_str("name: semantex\n");
    // YAML scalar — already a plain ASCII sentence in our case.
    let _ = writeln!(out, "description: \"{}\"", escape_yaml(SKILL_TAGLINE));
    out.push_str("license: Apache-2.0\n");
    out.push_str("allowed-tools: Bash\n");
    out.push_str("---\n\n");

    out.push_str("# semantex — Intelligent Code Search\n\n");
    out.push_str(SKILL_INTRO);
    out.push_str("\n\n");

    out.push_str("Prefer `semantex_agent` for any code search question. ");
    out.push_str(
        "It auto-routes to the right strategy and returns a complete answer in one call.\n\n",
    );

    out.push_str("## Tools\n\n");
    out.push_str(&super::render_tools_md(tools));

    out.push_str("## Fallbacks\n\n");
    out.push_str(concat!(
        "- `Grep`: exact regex on file content only — no semantic understanding.\n",
        "- `Glob`: find files by name/path pattern only.\n",
    ));
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
    fn renders_yaml_frontmatter_and_markdown() {
        let tools = all_tools();
        let out = render(&tools);
        assert!(out.starts_with("---\n"), "should open with YAML fence");
        // Find the closing fence on its own line.
        let body_start = out
            .find("\n---\n")
            .expect("YAML front matter should close with `---`");
        let frontmatter = &out[4..body_start];
        let body = &out[body_start + 5..];

        // Parse front matter as YAML mapping.
        let parsed: serde_yml::Value =
            serde_yml::from_str(frontmatter).expect("front matter must be valid YAML");
        assert!(parsed.get("name").is_some(), "front matter must have name");
        assert!(
            parsed.get("description").is_some(),
            "front matter must have description"
        );

        // Body should be non-empty markdown and mention every tool.
        assert!(!body.trim().is_empty(), "markdown body must be non-empty");
        for tool in tools.iter().filter(|t| t.live) {
            assert!(
                body.contains(tool.name),
                "body should mention tool `{}`",
                tool.name
            );
        }
    }
}
