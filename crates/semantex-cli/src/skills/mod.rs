//! Skill generation for `semantex skills-generate`.
//!
//! One canonical tool registry (`tools.rs`) → N platform-specific output
//! formatters (`templates/`). Each formatter produces a single file destined
//! for the target platform's agent-instruction conventions.

pub mod templates;
pub mod tools;

use crate::skills::tools::ToolMetadata;

/// The set of platforms we know how to generate for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Platform {
    ClaudeCode,
    Cursor,
    Codex,
    Aider,
    Gemini,
    Copilot,
    OpenCode,
    /// Cognition rebranded Windsurf to "Devin Desktop" on 2026-06-02 and
    /// EOL'd the old Cascade agent on 2026-07-01; `id()` now leads with the
    /// current product name but `from_id` still accepts the legacy
    /// `"windsurf"` alias so old docs/muscle-memory keep working.
    DevinDesktop,
    Trae,
}

impl Platform {
    pub const ALL: &'static [Platform] = &[
        Platform::ClaudeCode,
        Platform::Cursor,
        Platform::Codex,
        Platform::Aider,
        Platform::Gemini,
        Platform::Copilot,
        Platform::OpenCode,
        Platform::DevinDesktop,
        Platform::Trae,
    ];

    /// Stable, lower-case identifier used on the CLI (`--platform <id>`).
    pub fn id(self) -> &'static str {
        match self {
            Platform::ClaudeCode => "claude-code",
            Platform::Cursor => "cursor",
            Platform::Codex => "codex",
            Platform::Aider => "aider",
            Platform::Gemini => "gemini",
            Platform::Copilot => "copilot",
            Platform::OpenCode => "opencode",
            Platform::DevinDesktop => "devin-desktop",
            Platform::Trae => "trae",
        }
    }

    /// Path of the generated file *relative to the output directory*.
    pub fn relative_path(self) -> &'static str {
        match self {
            Platform::ClaudeCode => "claude-code/SKILL.md",
            Platform::Cursor => "cursor/.cursor/rules/semantex.mdc",
            Platform::Codex => "codex/AGENTS.md",
            Platform::Aider => "aider/.aider.conf.yml.snippet",
            Platform::Gemini => "gemini/extension.json",
            Platform::Copilot => "copilot/settings.json.snippet",
            Platform::OpenCode => "opencode/.opencode.json",
            Platform::DevinDesktop => "devin-desktop/semantex.md",
            Platform::Trae => "trae/trae-rules.md",
        }
    }

    /// Resolve a `--platform <id>` argument back to a [`Platform`].
    /// `"windsurf"` is accepted as a deprecated alias for `"devin-desktop"`.
    pub fn from_id(id: &str) -> Option<Platform> {
        if id == "windsurf" {
            return Some(Platform::DevinDesktop);
        }
        Self::ALL.iter().copied().find(|p| p.id() == id)
    }

    /// Generate the platform's skill file as a string.
    pub fn render(self, tools: &[ToolMetadata]) -> String {
        match self {
            Platform::ClaudeCode => templates::claude_code::render(tools),
            Platform::Cursor => templates::cursor::render(tools),
            Platform::Codex => templates::codex::render(tools),
            Platform::Aider => templates::aider::render(tools),
            Platform::Gemini => templates::gemini::render(tools),
            Platform::Copilot => templates::copilot::render(tools),
            Platform::OpenCode => templates::opencode::render(tools),
            Platform::DevinDesktop => templates::devin_desktop::render(tools),
            Platform::Trae => templates::trae::render(tools),
        }
    }
}

/// Shared introduction every platform's skill should expose to the agent.
pub(crate) const SKILL_INTRO: &str = concat!(
    "semantex is a fully local semantic code search engine. It combines dense ",
    "CodeRankEmbed embeddings (HNSW), BM25 sparse retrieval and structural code graphs into a daemon ",
    "queried over MCP. Use it instead of running grep + read loops — every additional ",
    "tool call resends the accumulated context (O(N^2) cost), and semantex collapses ",
    "those loops into one call."
);

/// Short tagline used in front-matter / metadata fields.
pub(crate) const SKILL_TAGLINE: &str = concat!(
    "Semantic code search via the semantex MCP server. One call replaces 5–10 grep + ",
    "read iterations. 25+ languages, fully local."
);
