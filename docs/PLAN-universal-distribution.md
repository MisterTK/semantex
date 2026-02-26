# Universal Plugin & Distribution System — Phase 2 + 3

Status: **Planned** (Phase 1 complete)

## Phase 2: Unified Multi-Agent Install

### 2.1 Refactor install.rs

Replace the three separate functions (`install_claude_code`, `install_opencode`, `install_codex`) with a unified system.

**File:** `crates/semantex-cli/src/commands/install.rs`

```rust
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum AgentKind {
    ClaudeCode,
    Codex,
    Cursor,
    Windsurf,
    OpenCode,
    VsCode,
    Gemini,
}

impl AgentKind {
    pub fn display_name(&self) -> &str {
        match self {
            Self::ClaudeCode => "Claude Code",
            Self::Codex => "Codex",
            Self::Cursor => "Cursor",
            Self::Windsurf => "Windsurf",
            Self::OpenCode => "OpenCode",
            Self::VsCode => "VS Code",
            Self::Gemini => "Gemini CLI",
        }
    }

    /// Check if this agent is installed by probing its config directory
    pub fn is_installed(&self) -> bool {
        self.config_dir().map_or(false, |d| d.exists())
    }

    pub fn config_dir(&self) -> Option<PathBuf> {
        let home = dirs::home_dir()?;
        Some(match self {
            Self::ClaudeCode => home.join(".claude"),
            Self::Codex => home.join(".codex"),
            Self::Cursor => home.join(".cursor"),
            Self::Windsurf => home.join(".codeium").join("windsurf"),
            Self::OpenCode => home.join(".config").join("opencode"),
            Self::VsCode => home.join(".vscode"), // or workspace .vscode/
            Self::Gemini => home.join(".gemini"),
        })
    }

    pub fn supports_hooks(&self) -> bool {
        matches!(self, Self::ClaudeCode | Self::Gemini)
    }
}

pub fn install_agent(kind: AgentKind) -> anyhow::Result<()> {
    let binary = semantex_binary_path()?;
    register_mcp(kind, &binary)?;
    install_skill(kind)?;
    if kind.supports_hooks() {
        register_hooks(kind, &binary)?;
    }
    Ok(())
}

pub fn uninstall_agent(kind: AgentKind) -> anyhow::Result<()> {
    remove_mcp(kind)?;
    remove_skill(kind)?;
    if kind.supports_hooks() {
        remove_hooks(kind)?;
    }
    Ok(())
}
```

### 2.2 Per-Agent MCP Config Formats

| Agent | Config File | Format |
|-------|-----------|--------|
| Claude Code | `~/.claude/settings.json` | JSON: `{ "mcpServers": { "semantex": { "command": "semantex", "args": ["mcp"] } } }` |
| Codex | `~/.codex/config.toml` | TOML: `[mcp.semantex]\ncommand = "semantex"\nargs = ["mcp"]` |
| Cursor | `~/.cursor/mcp.json` | JSON: same structure as Claude Code |
| Windsurf | `~/.codeium/windsurf/mcp_config.json` | JSON: same structure as Claude Code |
| OpenCode | `~/.config/opencode/opencode.json` | JSON: `{ "mcp": { "semantex": { "command": "semantex", "args": ["mcp"] } } }` |
| VS Code | `.vscode/mcp.json` (workspace) | JSON: same structure as Claude Code |
| Gemini | `~/.gemini/settings.json` | JSON: `{ "mcpServers": { "semantex": { "command": "semantex", "args": ["mcp"] } } }` |

### 2.3 SKILL.md Installation

Copy the bundled SKILL.md to each agent's skills directory:

| Agent | Skills Directory |
|-------|-----------------|
| Claude Code | `~/.claude/skills/semantex/SKILL.md` |
| Codex | `~/.agents/skills/semantex/SKILL.md` |
| Cursor | `~/.cursor/skills/semantex/SKILL.md` |
| Windsurf | `~/.codeium/windsurf/skills/semantex/SKILL.md` |
| OpenCode | `~/.config/opencode/skills/semantex/SKILL.md` |
| Gemini | `~/.gemini/skills/semantex/SKILL.md` |

The SKILL.md content is embedded in the binary at compile time via `include_str!`.

### 2.4 Hook Registration (Claude Code + Gemini)

Only Claude Code and Gemini CLI support hooks. The hooks intercept:
- **SessionStart**: Start semantex daemon, report index status
- **PreToolUse (Grep|Glob)**: Suggest semantex alternative for semantic queries
- **PreToolUse (Task)**: Inject semantex awareness into sub-agents
- **SessionEnd**: Clean up daemon if idle

Hook registration merges into the agent's existing hook config, avoiding duplicates.

### 2.5 Auto-Detect + Interactive Install

When `semantex install` is run without arguments:

```
semantex install

Detected agents:
  [x] Claude Code (~/.claude/)
  [x] Cursor (~/.cursor/)
  [ ] VS Code (not found)
  [ ] Codex (not found)
  ...

Install for selected agents? [Y/n]
```

Uses `dialoguer` crate for the interactive checklist (already a common Rust TUI crate).

### 2.6 CLI Subcommand Changes

**File:** `crates/semantex-cli/src/main.rs`

Replace:
```rust
// OLD — remove these
InstallClaudeCode,
InstallCodex,
InstallOpenCode,
// And remove --install-claude-code, --install-opencode, --install-codex flags
```

With:
```rust
/// Install semantex integration for coding agents
Install {
    /// Agent to install for (auto-detects if omitted)
    #[arg(value_enum)]
    agent: Option<AgentKind>,
    /// Install for all detected agents
    #[arg(long)]
    all: bool,
},
/// Uninstall semantex integration
Uninstall {
    #[arg(value_enum)]
    agent: Option<AgentKind>,
    /// Uninstall from all agents
    #[arg(long)]
    all: bool,
},
```

### 2.7 New Dependency

Add to `crates/semantex-cli/Cargo.toml`:
```toml
toml_edit = "0.22"    # For Codex TOML config manipulation
dialoguer = "0.11"    # Interactive agent selection
```

---

## Phase 3: Marketplace & Package Managers

### 3.1 Claude Code Plugin Marketplace

Ensure the `plugin/` directory passes validation:
```bash
cd plugin && claude plugin validate .
```

Users install via:
```bash
claude plugin install tk/semantex
```

The plugin manifest (`plugin.json`) already includes skills, hooks, and MCP references. The marketplace handles distribution — no binary bundling needed since the plugin assumes `semantex` is on PATH.

### 3.2 Homebrew Tap

Create new repo `tk/homebrew-semantex` with formula:

**File:** `Formula/semantex.rb`
```ruby
class Semantex < Formula
  desc "Semantic code search — hybrid ColBERT + BM25 retrieval engine"
  homepage "https://github.com/tk/semantex"
  version "0.1.0"
  license "Apache-2.0"

  on_macos do
    on_arm do
      url "https://github.com/tk/semantex/releases/download/v#{version}/semantex-v#{version}-aarch64-apple-darwin.tar.gz"
      # sha256 "..." — fill from release checksums
    end
    on_intel do
      url "https://github.com/tk/semantex/releases/download/v#{version}/semantex-v#{version}-x86_64-apple-darwin.tar.gz"
    end
  end

  on_linux do
    on_arm do
      url "https://github.com/tk/semantex/releases/download/v#{version}/semantex-v#{version}-aarch64-unknown-linux-gnu.tar.gz"
    end
    on_intel do
      url "https://github.com/tk/semantex/releases/download/v#{version}/semantex-v#{version}-x86_64-unknown-linux-gnu.tar.gz"
    end
  end

  def install
    bin.install "semantex"
    # Install ONNX Runtime dylib alongside binary
    lib.install Dir["libonnxruntime*"]
  end

  test do
    system "#{bin}/semantex", "--version"
  end
end
```

Install: `brew install tk/semantex/semantex`

### 3.3 npm Wrapper (Optional)

For agents that configure MCP via npm (Cursor, some VS Code setups):

**Package:** `semantex-search` on npm

Structure using platform-specific `optionalDependencies`:
```
semantex-search/
  package.json
  bin/semantex-search.js       # Thin wrapper that finds + execs native binary
@semantex-search/
  darwin-arm64/
    package.json
    bin/semantex               # Native binary
  darwin-x64/
  linux-x64/
  linux-arm64/
  win32-x64/
```

**`package.json`:**
```json
{
  "name": "semantex-search",
  "version": "0.1.0",
  "description": "Semantic code search — hybrid ColBER + BM25",
  "bin": { "semantex-search": "bin/semantex-search.js" },
  "optionalDependencies": {
    "@semantex-search/darwin-arm64": "0.1.0",
    "@semantex-search/darwin-x64": "0.1.0",
    "@semantex-search/linux-x64": "0.1.0",
    "@semantex-search/linux-arm64": "0.1.0",
    "@semantex-search/win32-x64": "0.1.0"
  }
}
```

Usage: `npx semantex-search mcp` for MCP server, `npx semantex-search "query"` for search.

This approach follows the esbuild/turbo pattern and works well with npm-based tool registries.

### 3.4 Cursor Marketplace (MCP Registry)

Cursor sources MCP servers from a registry. Submit `semantex` to the Cursor MCP directory once the npm wrapper is available, enabling one-click install from Cursor settings.

### 3.5 VS Code Extension (Future)

A minimal VS Code extension that:
1. Installs the semantex binary (or uses one on PATH)
2. Registers as MCP provider
3. Adds "Search with semantex" command to command palette

Lower priority — VS Code users can use the MCP config directly.

---

## Implementation Order

1. **Phase 2.1-2.3**: Core refactor (install.rs + CLI) — ~2-3 hours
2. **Phase 2.4**: Hook registration — ~1 hour (mostly porting existing Claude Code logic)
3. **Phase 2.5-2.6**: Interactive install + CLI cleanup — ~1 hour
4. **Phase 3.1**: Plugin validation — ~30 min (fix any issues found)
5. **Phase 3.2**: Homebrew formula — ~1 hour (after first GitHub Release)
6. **Phase 3.3**: npm wrapper — ~2 hours (after Homebrew)
7. **Phase 3.4-3.5**: Marketplace submissions — dependent on releases
