//! Per-platform skill formatters.
//!
//! Each module owns one output target. Formatters take a slice of
//! [`crate::skills::tools::ToolMetadata`] and return a fully-rendered file
//! body as a `String`. They never write to disk — that's the CLI command's
//! job.

pub mod aider;
pub mod claude_code;
pub mod codex;
pub mod copilot;
pub mod cursor;
pub mod gemini;
pub mod opencode;
pub mod trae;
pub mod windsurf;

use crate::skills::tools::ToolMetadata;
use std::fmt::Write;

/// Helper: render a markdown bullet list for a single tool's arguments.
pub(crate) fn render_args_md(tool: &ToolMetadata) -> String {
    if tool.args.is_empty() {
        return "- *(no arguments)*\n".into();
    }
    let mut out = String::new();
    for arg in tool.args {
        let req = if arg.required { " (required)" } else { "" };
        // Writing to a `String` cannot fail.
        let _ = writeln!(
            out,
            "- `{}` *({})* {}: {}",
            arg.name, arg.ty, req, arg.description
        );
    }
    out
}

/// Helper: render markdown examples for a single tool.
pub(crate) fn render_examples_md(tool: &ToolMetadata) -> String {
    if tool.examples.is_empty() {
        return String::new();
    }
    let mut out = String::from("\nExamples:\n\n");
    for ex in tool.examples {
        let _ = writeln!(out, "- {}: `{}`", ex.label, ex.args_json);
    }
    out
}

/// Helper: render a markdown section per tool.
pub(crate) fn render_tools_md(tools: &[ToolMetadata]) -> String {
    let mut out = String::new();
    for tool in tools {
        let status_tag = if tool.live {
            ""
        } else {
            " *(planned in v0.3, not yet live)*"
        };
        let _ = writeln!(out, "### `{}`{}\n", tool.name, status_tag);
        out.push_str(tool.description);
        out.push_str("\n\n");
        if !tool.when_to_use.is_empty() {
            out.push_str("When to use:\n\n");
            for w in tool.when_to_use {
                let _ = writeln!(out, "- {w}");
            }
            out.push('\n');
        }
        out.push_str("Arguments:\n\n");
        out.push_str(&render_args_md(tool));
        out.push_str(&render_examples_md(tool));
        out.push('\n');
    }
    out
}
