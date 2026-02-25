# Universal Plugin & Distribution System — Phase 2 + 3

Status: **Planned** (Phase 1 complete)

## Phase 2: Unified Multi-Agent Install

### 2.1 Refactor install.rs

Replace the three separate functions (`install_claude_code`, `install_opencode`, `install_codex`) with a unified system.

**File:** `crates/sage-cli/src/commands/install.rs`

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
    let binary = sage_binary_path()?;
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
| Claude Code | `~/.claude/settings.json` | JSON: `{ "mcpServers": { "sage": { "command": "sage", "args": ["mcp"] } } }` |
| Codex | `~/.codex/config.toml` | TOML: `[mcp.sage]\ncommand = "sage"\nargs = ["mcp"]` |
| Cursor | `~/.cursor/mcp.json` | JSON: same structure as Claude Code |
| Windsurf | `~/.codeium/windsurf/mcp_config.json` | JSON: same structure as Claude Code |
| OpenCode | `~/.config/opencode/opencode.json` | JSON: `{ "mcp": { "sage": { "command": "sage", "args": ["mcp"] } } }` |
| VS Code | `.vscode/mcp.json` (workspace) | JSON: same structure as Claude Code |
| Gemini | `~/.gemini/settings.json` | JSON: `{ "mcpServers": { "sage": { "command": "sage", "args": ["mcp"] } } }` |

### 2.3 SKILL.md Installation

Copy the bundled SKILL.md to each agent's skills directory:

| Agent | Skills Directory |
|-------|-----------------|
| Claude Code | `~/.claude/skills/sage/SKILL.md` |
| Codex | `~/.agents/skills/sage/SKILL.md` |
| Cursor | `~/.cursor/skills/sage/SKILL.md` |
| Windsurf | `~/.codeium/windsurf/skills/sage/SKILL.md` |
| OpenCode | `~/.config/opencode/skills/sage/SKILL.md` |
| Gemini | `~/.gemini/skills/sage/SKILL.md` |

The SKILL.md content is embedded in the binary at compile time via `include_str!`.

### 2.4 Hook Registration (Claude Code + Gemini)

Only Claude Code and Gemini CLI support hooks. The hooks intercept:
- **SessionStart**: Start sage daemon, report index status
- **PreToolUse (Grep|Glob)**: Suggest sage alternative for semantic queries
- **PreToolUse (Task)**: Inject sage awareness into sub-agents
- **SessionEnd**: Clean up daemon if idle

Hook registration merges into the agent's existing hook config, avoiding duplicates.

### 2.5 Auto-Detect + Interactive Install

When `sage install` is run without arguments:

```
sage install

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

**File:** `crates/sage-cli/src/main.rs`

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
/// Install sage integration for coding agents
Install {
    /// Agent to install for (auto-detects if omitted)
    #[arg(value_enum)]
    agent: Option<AgentKind>,
    /// Install for all detected agents
    #[arg(long)]
    all: bool,
},
/// Uninstall sage integration
Uninstall {
    #[arg(value_enum)]
    agent: Option<AgentKind>,
    /// Uninstall from all agents
    #[arg(long)]
    all: bool,
},
```

### 2.7 New Dependency

Add to `crates/sage-cli/Cargo.toml`:
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
claude plugin install tk/sage
```

The plugin manifest (`plugin.json`) already includes skills, hooks, and MCP references. The marketplace handles distribution — no binary bundling needed since the plugin assumes `sage` is on PATH.

### 3.2 Homebrew Tap

Create new repo `tk/homebrew-sage` with formula:

**File:** `Formula/sage.rb`
```ruby
class Sage < Formula
  desc "Semantic code search — hybrid ColBERT + BM25 retrieval engine"
  homepage "https://github.com/tk/sage"
  version "0.1.0"
  license "Apache-2.0"

  on_macos do
    on_arm do
      url "https://github.com/tk/sage/releases/download/v#{version}/sage-v#{version}-aarch64-apple-darwin.tar.gz"
      # sha256 "..." — fill from release checksums
    end
    on_intel do
      url "https://github.com/tk/sage/releases/download/v#{version}/sage-v#{version}-x86_64-apple-darwin.tar.gz"
    end
  end

  on_linux do
    on_arm do
      url "https://github.com/tk/sage/releases/download/v#{version}/sage-v#{version}-aarch64-unknown-linux-gnu.tar.gz"
    end
    on_intel do
      url "https://github.com/tk/sage/releases/download/v#{version}/sage-v#{version}-x86_64-unknown-linux-gnu.tar.gz"
    end
  end

  def install
    bin.install "sage"
    # Install ONNX Runtime dylib alongside binary
    lib.install Dir["libonnxruntime*"]
  end

  test do
    system "#{bin}/sage", "--version"
  end
end
```

Install: `brew install tk/sage/sage`

### 3.3 npm Wrapper (Optional)

For agents that configure MCP via npm (Cursor, some VS Code setups):

**Package:** `sage-search` on npm

Structure using platform-specific `optionalDependencies`:
```
sage-search/
  package.json
  bin/sage-search.js       # Thin wrapper that finds + execs native binary
@sage-search/
  darwin-arm64/
    package.json
    bin/sage               # Native binary
  darwin-x64/
  linux-x64/
  linux-arm64/
  win32-x64/
```

**`package.json`:**
```json
{
  "name": "sage-search",
  "version": "0.1.0",
  "description": "Semantic code search — hybrid ColBER + BM25",
  "bin": { "sage-search": "bin/sage-search.js" },
  "optionalDependencies": {
    "@sage-search/darwin-arm64": "0.1.0",
    "@sage-search/darwin-x64": "0.1.0",
    "@sage-search/linux-x64": "0.1.0",
    "@sage-search/linux-arm64": "0.1.0",
    "@sage-search/win32-x64": "0.1.0"
  }
}
```

Usage: `npx sage-search mcp` for MCP server, `npx sage-search "query"` for search.

This approach follows the esbuild/turbo pattern and works well with npm-based tool registries.

### 3.4 Cursor Marketplace (MCP Registry)

Cursor sources MCP servers from a registry. Submit `sage` to the Cursor MCP directory once the npm wrapper is available, enabling one-click install from Cursor settings.

### 3.5 VS Code Extension (Future)

A minimal VS Code extension that:
1. Installs the sage binary (or uses one on PATH)
2. Registers as MCP provider
3. Adds "Search with sage" command to command palette

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
