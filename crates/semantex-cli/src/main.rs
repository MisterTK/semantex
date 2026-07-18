#![allow(clippy::missing_errors_doc)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::struct_excessive_bools)]
#![allow(clippy::unnecessary_wraps)]

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

mod client;
mod commands;
mod skills;
mod telemetry;

use anyhow::{Context, Result};

unsafe extern "C" {
    /// `mi_collect(force: bool)` — force mimalloc to return freed pages to the OS.
    /// Linked automatically because mimalloc is the global allocator.
    safe fn mi_collect(force: bool);
}

/// Force mimalloc to return freed pages to the OS.
fn mimalloc_purge() {
    mi_collect(true);
}
use clap::{Parser, Subcommand};
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(
    name = "semantex",
    about = "Semantic grep — fully local code search with dense+sparse hybrid retrieval",
    version,
    after_help = "Examples:\n  semantex \"function that handles authentication\" .\n  semantex index .\n  semantex --content --max-count 10 \"database connection pool\"\n  semantex --json --rerank \"error handling\"\n  semantex -G --grep \"ConnectionFactory\" .  # grep-like exhaustive search"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Search queries (multiple allowed with --refs for batch mode)
    #[arg(value_name = "QUERY")]
    queries: Vec<String>,

    /// Path to search/index (defaults to current directory)
    #[arg(short = 'p', long, value_name = "PATH")]
    path: Option<PathBuf>,

    /// Maximum number of results
    #[arg(short = 'm', long, default_value = "10", env = "SEMANTEX_MAX_COUNT")]
    max_count: usize,

    /// Show matching content snippets (alias for --verbose)
    #[arg(short = 'c', long, env = "SEMANTEX_CONTENT")]
    content: bool,

    /// Verbose output: show full content with colors (default: compact 3-line snippets)
    #[arg(short = 'v', long)]
    verbose: bool,

    /// Lines of context around match
    #[arg(short = 'C', long, default_value = "0")]
    context: usize,

    /// Show line numbers (default: on)
    #[arg(short = 'n', long)]
    line_number: bool,

    /// Hide line numbers
    #[arg(long = "no-line-number", conflicts_with = "line_number")]
    no_line_number: bool,

    /// Enable cross-encoder reranking (slower but may improve ranking)
    #[arg(long)]
    rerank: bool,

    /// Dense HNSW search only
    #[arg(long)]
    dense_only: bool,

    /// BM25 search only
    #[arg(long)]
    sparse_only: bool,

    /// JSON output (for agents)
    #[arg(long)]
    json: bool,

    /// Omit content from JSON output (paths + metadata only)
    #[arg(long, requires = "json")]
    no_content: bool,

    /// Trim content to relevant snippet in JSON output
    #[arg(long, requires = "json", conflicts_with = "no_content")]
    snippet: bool,

    /// Grep-like output: file:line: preview (one line per result)
    #[arg(short = 'g', long, conflicts_with_all = ["json", "content", "context"])]
    grep: bool,

    /// Grep parity mode: exact+BM25 only, no dense/rerank, exhaustive (50 results)
    /// Matches grep's recall with better ranking.
    #[arg(short = 'G', long)]
    grep_mode: bool,

    /// Include only files with this extension (repeatable, e.g. -t rs -t py)
    #[arg(
        short = 't',
        long = "type",
        value_name = "EXT",
        env = "SEMANTEX_INCLUDE_TYPES",
        value_delimiter = ','
    )]
    include_type: Vec<String>,

    /// Exclude files with this extension (repeatable)
    #[arg(
        long = "exclude-type",
        value_name = "EXT",
        env = "SEMANTEX_EXCLUDE_TYPES",
        value_delimiter = ','
    )]
    exclude_type: Vec<String>,

    /// Exclude non-code files (md, json, yaml, toml, txt, log, cfg, ini, env, pdf, ipynb, lock)
    #[arg(long)]
    code_only: bool,

    /// Maximum file size to index (bytes)
    #[arg(long, env = "SEMANTEX_MAX_FILE_SIZE")]
    max_file_size: Option<u64>,

    /// Maximum number of files to index
    #[arg(long)]
    max_file_count: Option<usize>,

    /// Internal: Claude Code session hook (outputs JSON context)
    #[arg(long, hide = true)]
    session_hook: bool,

    /// Internal: Claude Code PreToolUse hook for Grep|Glob interception
    #[arg(long, hide = true)]
    grep_hook: bool,

    /// Internal: Claude Code SessionEnd hook for daemon cleanup
    #[arg(long, hide = true)]
    session_end_hook: bool,

    /// Internal: Claude Code SubagentStart hook for subagent context injection
    #[arg(long, hide = true)]
    subagent_hook: bool,

    /// Internal: Claude Code PreToolUse hook for Bash search interception
    #[arg(long, hide = true)]
    bash_hook: bool,

    /// Internal: Claude Code PreToolUse hook for Read interception
    #[arg(long, hide = true)]
    read_hook: bool,

    /// Hard-deny mode for hooks: block the tool call instead of just nudging.
    /// Use with --grep-hook or --bash-hook.
    #[arg(long, hide = true)]
    deny: bool,

    /// Uninstall semantex hooks from Claude Code
    #[arg(long)]
    uninstall_claude_code: bool,

    /// Remove all semantex hook registrations
    #[arg(long)]
    uninstall: bool,

    /// Install semantex hooks into Claude Code's settings.json
    #[arg(long)]
    install_claude_code: bool,

    /// Install semantex hooks into OpenCode
    #[arg(long)]
    install_opencode: bool,

    /// Install semantex hooks into Codex
    #[arg(long)]
    install_codex: bool,

    /// Compact references only: file:start-end name (no snippets, no scores).
    /// Ideal for agent consumption — minimises tokens, agent uses Read for content.
    #[arg(long, conflicts_with_all = ["json", "grep", "content", "verbose"])]
    refs: bool,

    /// Peek mode: refs + first 5 lines of code per result.
    /// More detail than --refs, less than full content.
    #[arg(long, conflicts_with_all = ["json", "grep", "content", "verbose", "refs"])]
    peek: bool,

    /// Show callers, callees, and type references for a named symbol.
    /// Standalone mode — skips normal search. Example: semantex --around my_function
    #[arg(long, value_name = "SYMBOL")]
    around: Option<String>,

    /// Regex pattern to combine with semantic search (hybrid mode).
    /// Example: semantex -e "async fn" "database pool"
    #[arg(short = 'e', long = "pattern")]
    pattern: Option<String>,

    /// Deep mode: search, read, and summarize internally.
    /// Returns a curated prose answer instead of raw results.
    #[arg(long, conflicts_with_all = ["json", "grep", "refs", "peek", "around", "content"])]
    deep: bool,

    /// Which repo(s) to search: `repo` (default, just this project), `all`
    /// (every project registered in ~/.semantex/projects.json), or a
    /// comma-separated list of project display names/paths — any string
    /// other than `repo`/`all` is treated as names; names matching no
    /// registered project are reported as skipped. Non-`repo` scopes always
    /// run a hybrid dense+sparse search fanned out across targets and
    /// RRF-fused, with results prefixed `[project]`; they honor only
    /// --max-count/--rerank/--grep-mode/--json — other single-repo flags
    /// (--refs, --peek, --deep, --around, --dense-only, --sparse-only, ...)
    /// are ignored.
    #[arg(long, default_value = "repo")]
    scope: String,
}

#[derive(Subcommand)]
enum Commands {
    /// Build or update the search index
    Index {
        /// Path to index (defaults to current directory)
        #[arg(default_value = ".")]
        path: PathBuf,
        /// Index anyway if `path` looks like a multi-repo workspace
        /// container (several sibling repos, no `.git` of its own)
        #[arg(long)]
        force: bool,
    },
    /// Watch for file changes and auto-reindex
    Watch {
        /// Path to watch (defaults to current directory)
        #[arg(default_value = ".")]
        path: PathBuf,
    },
    /// Download required ONNX models
    DownloadModels,
    /// Show index and model status
    Status {
        /// Path to check (defaults to current directory)
        #[arg(default_value = ".")]
        path: PathBuf,
    },
    /// Start search daemon (keeps models loaded for fast queries)
    Serve {
        /// Path to serve (defaults to current directory)
        #[arg(default_value = ".")]
        path: PathBuf,
    },
    /// Stop a running search daemon
    Stop {
        /// Path to stop (defaults to current directory)
        #[arg(default_value = ".")]
        path: PathBuf,
    },
    /// Run as MCP server (stdio by default; `--http` for HTTP transport)
    Mcp {
        /// Serve over HTTP instead of stdio. Bound to 127.0.0.1 by default.
        #[arg(long)]
        http: bool,

        /// HTTP port (default: 5050). Only used with --http.
        #[arg(long, default_value_t = 5050, requires = "http")]
        port: u16,

        /// Allow binding to 0.0.0.0 (any interface). Off by default. Bearer-token
        /// auth is REQUIRED when this is set (see --auth-token / SEMANTEX_HTTP_TOKEN) —
        /// the server refuses to start remote without a resolvable token.
        #[arg(long, requires = "http")]
        allow_remote: bool,

        /// Named toolset bundle to expose: `core` (3 tools: search, deep, agent),
        /// `structural` (5 tools: symbol, callers, callees, implementations,
        /// architecture), or `all` (10 tools, default).
        #[arg(long, value_name = "NAME", default_value = "all")]
        toolset: String,

        /// Bearer token required to call `/mcp/*` over HTTP. Takes precedence
        /// over `SEMANTEX_HTTP_TOKEN`; if neither is set, a token is
        /// auto-generated and persisted at `~/.semantex/http_token` (0600).
        /// Only used with --http.
        #[arg(long, value_name = "TOKEN", requires = "http")]
        auth_token: Option<String>,

        /// Require bearer-token auth even on loopback (127.0.0.1). Auth is
        /// always required when --allow-remote is set, regardless of this
        /// flag. Only used with --http.
        #[arg(long, requires = "http")]
        require_auth: bool,
    },
    /// Install Claude Code integration
    InstallClaudeCode {
        /// Installation scope: user (all projects), project (shared via git), or local (this repo, gitignored)
        #[arg(long, value_enum)]
        scope: Option<commands::install::InstallScope>,
    },
    /// Uninstall Claude Code integration (hooks, skill, and MCP server
    /// registration across every scope install-claude-code may have used)
    UninstallClaudeCode,
    /// Install Codex integration (`~/.codex/config.toml` + project `AGENTS.md`)
    InstallCodex,
    /// Uninstall Codex integration
    UninstallCodex,
    /// Install OpenCode integration (`opencode.json` + `.opencode/semantex.md`)
    InstallOpenCode {
        /// Write to `~/.config/opencode/opencode.json` instead of the project-scoped `opencode.json`
        #[arg(long)]
        global: bool,
    },
    /// Uninstall OpenCode integration
    UninstallOpenCode {
        #[arg(long)]
        global: bool,
    },
    /// Install Cursor integration (`.cursor/mcp.json` + `.cursor/rules/semantex.mdc`)
    InstallCursor {
        /// Write to `~/.cursor/mcp.json` instead of the project-scoped `.cursor/mcp.json`
        #[arg(long)]
        global: bool,
    },
    /// Uninstall Cursor integration
    UninstallCursor {
        #[arg(long)]
        global: bool,
    },
    /// Install Devin Desktop / Windsurf integration. Windsurf was rebranded
    /// "Devin Desktop" by Cognition on 2026-06-02; `install-windsurf` is
    /// kept as a deprecated alias.
    #[command(alias = "install-windsurf")]
    InstallDevinDesktop,
    /// Uninstall Devin Desktop / Windsurf integration
    #[command(alias = "uninstall-windsurf")]
    UninstallDevinDesktop,
    /// Install Aider integration (`.aider.conf.yml` + `.aider/semantex.md`;
    /// Aider has no native MCP support)
    InstallAider,
    /// Uninstall Aider integration
    UninstallAider,
    /// Install Trae integration (`.trae/mcp.json` + `.trae/skills/semantex/SKILL.md`)
    InstallTrae,
    /// Uninstall Trae integration
    UninstallTrae,
    /// Install GitHub Copilot integration (VS Code Copilot Chat + Copilot CLI)
    InstallCopilot,
    /// Uninstall GitHub Copilot integration
    UninstallCopilot,
    /// Install Gemini CLI integration (`.gemini/settings.json` + `GEMINI.md`)
    InstallGemini {
        /// Write to `~/.gemini/settings.json` instead of the project-scoped file
        #[arg(long)]
        global: bool,
    },
    /// Uninstall Gemini CLI integration
    UninstallGemini {
        #[arg(long)]
        global: bool,
    },
    /// Start persistent client (binary protocol, lower latency)
    Connect {
        /// Path to project (defaults to current directory)
        #[arg(default_value = ".")]
        path: PathBuf,
    },
    /// Stop persistent client
    Disconnect,
    /// Validate index consistency (meta-DB sync, stale files, dense/sparse integrity, graph)
    Validate {
        /// Path to validate (defaults to current directory)
        #[arg(default_value = ".")]
        path: PathBuf,
    },
    /// Show git commit history for a file, as recorded in the index's
    /// `history.db` (v13 Wave 2). Read-only against whatever the last
    /// `semantex index` build populated — run `semantex index` first if
    /// history.db doesn't exist yet.
    History {
        /// File to show history for (repo-relative, matching the index).
        file: String,

        /// Project path (default: current directory)
        #[arg(short, long, default_value = ".")]
        path: PathBuf,

        /// Max commits to show (default: 20)
        #[arg(long, default_value_t = 20)]
        limit: usize,
    },
    /// Generate per-platform skill files describing semantex's MCP tools.
    ///
    /// Emits one file per supported agent platform (Claude Code, Cursor, Codex,
    /// Aider, Gemini CLI, Copilot CLI, OpenCode, Devin Desktop/Windsurf, Trae) from a single
    /// canonical tool registry.
    SkillsGenerate {
        /// Output directory (default: ./.semantex-skills/)
        #[arg(long, value_name = "DIR")]
        out: Option<PathBuf>,

        /// Restrict to a single platform id, e.g. `claude-code`, `cursor`, ...
        /// Use `all` (the default) for every supported platform.
        #[arg(long, value_name = "NAME", default_value = "all")]
        platform: String,

        /// Overwrite existing files in the output directory.
        #[arg(long)]
        force: bool,
    },
    /// Intelligent code search with automatic query routing and formatting.
    Agent {
        /// The search query
        query: String,

        /// Path to the project (default: current directory)
        #[arg(short, long, default_value = ".")]
        path: PathBuf,

        /// Override query classification: semantic, deep, exact_symbol, structural, regex, analytical, exhaustive, file_pattern, architecture, feature_planning
        #[arg(long)]
        route: Option<String>,

        /// Include full source code blocks
        #[arg(long)]
        full: bool,

        /// Response budget in bytes (default: 12000)
        #[arg(long)]
        budget: Option<usize>,

        /// Output raw JSON instead of formatted text
        #[arg(long)]
        json: bool,

        /// Output the structured ranked hits (a JSON array of
        /// SearchResultItem, same shape as `search --json`) instead of the
        /// prose answer. Combine with `--route <name>` to measure a SPECIFIC
        /// retrieval route — each hit's `file` is repo-relative. Synthesis
        /// routes (deep / architecture / feature_planning) emit `[]`.
        #[arg(long, conflicts_with = "json")]
        json_hits: bool,

        /// Which repo(s) to search: `repo` (default, just `path`, via the
        /// daemon exactly as before), `all` (every project registered in
        /// ~/.semantex/projects.json), or a comma-separated list of project
        /// display names/paths — any string other than `repo`/`all` is
        /// treated as names; names matching no registered project are
        /// reported as skipped. Non-`repo` scopes run entirely in-process
        /// (no daemon): a hybrid dense+sparse search fanned out across
        /// targets and RRF-fused — `--route` and `--full` are ignored, since
        /// route classification (structural walks, deep synthesis, ...) is a
        /// single-repo concept. Results are prefixed `[project]`.
        #[arg(long, default_value = "repo")]
        scope: String,
    },
    /// Show LLM backend status and run a 1-token health-check classify call.
    ///
    /// Prints the configured backend label (or "no LLM configured"), the
    /// provider and endpoint (if Genai), and the result of a classify call
    /// for a standard probe query.
    ///
    /// Only available when built with `--features llm`.
    #[cfg(feature = "llm")]
    LlmStatus,
    /// Distill a static per-token embedding table from the ColBERT document
    /// encoder (Ember Plan A, Task 4 — see
    /// `docs/superpowers/plans/2026-07-17-ember-plan-a-gate1.md`). Internal
    /// dev/eval tooling for the Gate-1 quality decision, not a stable
    /// user-facing command.
    #[command(hide = true)]
    DistillStaticTable {
        /// Corpus directory to walk (repeatable: pass `--corpus` multiple
        /// times to concatenate several directories into one corpus).
        #[arg(long = "corpus", value_name = "DIR", required = true)]
        corpus: Vec<PathBuf>,

        /// Output path for the distilled `static_token_table.bin`.
        #[arg(long, value_name = "PATH")]
        out: PathBuf,

        /// After saving, reload the table via `StaticTokenTable::load` and
        /// print its dims (sanity-checks the save/load round trip).
        #[arg(long)]
        verify: bool,
    },
}

/// Resolve the ONNX Runtime shared library for `ort`'s load-dynamic mode.
///
/// semantex-core declares `ort` with the `load-dynamic` feature, which puts
/// `ort` in pure dynamic-loading mode on every platform, so a real
/// `libonnxruntime` must be located at runtime. Resolution order:
///   1. A copy semantex previously provisioned under `~/.semantex/runtime/`.
///   2. For commands that load ONNX Runtime in-process, download + cache the
///      official Microsoft release matching `ort`'s required version.
///   3. A system-installed library, as an offline fallback.
///
/// Returns the absolute path to set as `ORT_DYLIB_PATH`, or `None` to let `ort`
/// fall back to the OS loader search.
fn resolve_ort_dylib() -> Option<PathBuf> {
    use semantex_core::embedding::runtime_manager;

    // Only commands that load ONNX Runtime in-process need a dylib; lightweight
    // read/admin subcommands never touch it (and `download-models` provisions
    // itself), so don't set ORT_DYLIB_PATH for them.
    if !command_wants_onnxruntime(std::env::args().skip(1)) {
        return None;
    }

    let runtime_root = semantex_core::config::SemantexConfig::semantex_home().join("runtime");
    if let Some(path) = runtime_manager::find_onnxruntime(&runtime_root) {
        return Some(path);
    }
    // Prefer the known-good managed runtime over an unknown system library: the
    // latter may be older than the version `ort` requires. The system path is an
    // offline fallback only.
    match runtime_manager::ensure_onnxruntime(&runtime_root) {
        Ok(path) => Some(path),
        Err(e) => {
            tracing::warn!(
                "Could not auto-provision ONNX Runtime ({e}); \
                 falling back to a system-installed library. \
                 Install ONNX Runtime {} or newer, or set ORT_DYLIB_PATH manually.",
                runtime_manager::ONNXRUNTIME_VERSION
            );
            find_ort_dylib().map(PathBuf::from)
        }
    }
}

/// Whether the requested subcommand loads ONNX Runtime in-process and should
/// therefore trigger auto-provisioning of the runtime library on first use.
/// Mirrors [`memory_constrained_from_args`]' arg-peek style: lightweight
/// read/admin subcommands are excluded; the default (search) plus
/// index/watch/serve/mcp load ONNX Runtime.
fn command_wants_onnxruntime<I, S>(args: I) -> bool
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    for arg in args {
        let arg = arg.as_ref();
        if arg == "--" {
            return true; // a positional search query follows
        }
        if !arg.starts_with('-') {
            return !matches!(
                arg,
                "status"
                    | "stop"
                    | "validate"
                    | "download-models"
                    | "connect"
                    | "disconnect"
                    | "install-claude-code"
                    | "uninstall-claude-code"
                    | "install-codex"
                    | "uninstall-codex"
                    | "install-open-code"
                    | "uninstall-open-code"
                    | "install-cursor"
                    | "uninstall-cursor"
                    | "install-devin-desktop"
                    | "install-windsurf"
                    | "uninstall-devin-desktop"
                    | "uninstall-windsurf"
                    | "install-aider"
                    | "uninstall-aider"
                    | "install-trae"
                    | "uninstall-trae"
                    | "install-copilot"
                    | "uninstall-copilot"
                    | "install-gemini"
                    | "uninstall-gemini"
                    | "skills-generate"
                    | "llm-status"
                    | "history"
            );
        }
    }
    false
}

/// Find ONNX Runtime shared library on platform-specific paths.
fn find_ort_dylib() -> Option<&'static str> {
    let candidates: &[&str] = if cfg!(target_os = "macos") {
        &[
            "/opt/homebrew/lib/libonnxruntime.dylib",
            "/usr/local/lib/libonnxruntime.dylib",
        ]
    } else if cfg!(target_os = "windows") {
        &[
            r"C:\Program Files\onnxruntime\lib\onnxruntime.dll",
            r"C:\Program Files\Microsoft\ONNX Runtime\onnxruntime.dll",
        ]
    } else {
        // Linux (x86_64, aarch64, etc.)
        &[
            "/usr/lib/libonnxruntime.so",
            "/usr/local/lib/libonnxruntime.so",
            "/usr/lib/x86_64-linux-gnu/libonnxruntime.so",
            "/usr/lib/aarch64-linux-gnu/libonnxruntime.so",
            "/usr/lib64/libonnxruntime.so",
        ]
    };
    candidates
        .iter()
        .find(|p| std::path::Path::new(p).exists())
        .copied()
}

/// Peek at `std::env::args()` to detect whether the user is invoking the
/// `mcp` subcommand. Used to set `SEMANTEX_ORT_THREADS=1` BEFORE threads
/// spawn (i.e. before the tracing subscriber and clap parsing).
///
/// MCP processes should use minimal ONNX threads — queries are short and
/// multiple sessions may run in parallel; 1 thread keeps the per-process
/// inference memory footprint low. The old `McpServer::with_toolset` set
/// this env var, but by then tracing + mimalloc threads were already
/// running, which is real UB on platforms with strict env-var thread rules.
/// Returns true when the requested subcommand is memory-heavy enough that
/// we want to cap ORT threads aggressively. Currently: `mcp` (many parallel
/// query sessions, each loading the dense embedder), `index` (see note
/// below — largely vestigial today), `watch` (continuous incremental
/// encoding, mostly query-shaped batches).
///
/// # This does NOT bound a full index build's dominant thread/memory cost
///
/// `SEMANTEX_ORT_THREADS` (forced to `1` here) only feeds the QUERY-tuned
/// embedder constructors (`SingleVectorEmbedder::new` /
/// `ColbertEmbedder::new`, see `embedding::single_vector::query_threads`).
/// A full `index` build uses the INDEXING-tuned constructors
/// (`SingleVectorEmbedder::for_indexing` / `ColbertEmbedder::for_indexing`,
/// see `embedding::colbert::index_ort_threads`), which read the
/// independent `SEMANTEX_INDEX_ORT_THREADS` knob — unaffected by this
/// function's env vars — and default to `embedding::colbert::default_index_threads`
/// (real ORT intra-op threads: half the cores, or ALL of them when
/// `index::gate::max_concurrent_builds()` says only one full build can run
/// at a time) regardless of what this function decides. Measured live on a
/// 4-core box (one build slot, so the "use all cores" branch applies): a
/// full `index` build shows 4 actively-running ORT threads throughout the
/// encode phase, NOT 1.
///
/// An EARLIER version of this comment claimed "1 thread keeps the encoding
/// spike under 1 GB" for `index` specifically, and cited a "10 GB spike with
/// 4 threads on a 2k-chunk repo" — both were measurements from before the
/// `index::gate` build-slot design (which introduced the separate
/// `for_indexing`/`SEMANTEX_INDEX_ORT_THREADS` path) and no longer describe
/// what `index` actually does; ORT thread count is not the dominant memory
/// driver for a full build. The dominant driver (E2E-verified, 344 files /
/// 4665 chunks: ~9.5 GB peak RSS) is the dense backend's single-call
/// index-build pass (see `crate::memory::kernel_rss_cap_bytes` for the
/// address-space-vs-RSS accounting around that peak), not the number of ORT
/// intra-op threads. `index` stays in this list because forcing
/// `OMP_NUM_THREADS`/`MKL_NUM_THREADS`/etc. to 1 is still a reasonable
/// default for the process as a whole (BLAS-backed CPU kernels a future
/// dependency might introduce), it just isn't the thing actually keeping a
/// build's memory down today.
fn memory_constrained_mode_requested() -> bool {
    memory_constrained_from_args(std::env::args().skip(1))
}

fn memory_constrained_from_args<I, S>(args: I) -> bool
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    for arg in args {
        let arg = arg.as_ref();
        if arg == "--" {
            return false;
        }
        if !arg.starts_with('-') {
            return matches!(arg, "mcp" | "index" | "watch" | "serve");
        }
    }
    false
}

/// True for subcommands that are PURE background indexing — safe to lower the
/// whole process to background CPU priority so it yields to the developer's
/// foreground work. Deliberately excludes `mcp`/`serve`: those also answer
/// search queries, and niceing them would slow query responses under load.
fn wants_background_priority() -> bool {
    wants_background_priority_from_args(std::env::args().skip(1))
}

fn wants_background_priority_from_args<I, S>(args: I) -> bool
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    for arg in args {
        let arg = arg.as_ref();
        if arg == "--" {
            return false;
        }
        if !arg.starts_with('-') {
            return matches!(arg, "index" | "watch");
        }
    }
    false
}

fn main() -> Result<()> {
    // ── HARDEST POSSIBLE FAILSAFE: memory caps installed at process start ──
    //
    // MUST be the first thing main() does. On Linux this installs
    // setrlimit(RLIMIT_AS, …) so any allocation past the cap returns ENOMEM
    // from the kernel (allocator panics → process exits cleanly). On macOS
    // there is NO equivalent userspace API — the soft RSS cap (polled by
    // `semantex_core::memory::check_rss_or_abort`) is the only failsafe, and
    // it hard-aborts the process after a few consecutive overshoots.
    //
    // The cap is derived from system RAM (50% soft, 75% kernel, clamped to
    // sane bounds). Override via `SEMANTEX_MAX_RSS_MB`. Set
    // `SEMANTEX_NO_RLIMIT=1` to skip the kernel cap (containers with cgroup
    // limits). Set `SEMANTEX_QUIET_LIMITS=1` to suppress the startup line.
    {
        use semantex_core::memory::{
            KernelCapResult, install_kernel_rss_cap, soft_rss_limit_mb, system_ram_mb,
        };
        let kernel = install_kernel_rss_cap();
        let soft_mb = soft_rss_limit_mb();
        if std::env::var("SEMANTEX_QUIET_LIMITS").is_err() {
            let ram = system_ram_mb().map_or_else(|| "unknown".to_string(), |n| format!("{n} MB"));
            let kernel_desc = match kernel {
                KernelCapResult::Installed(b) => {
                    format!("kernel hard={} MB", b / (1024 * 1024))
                }
                KernelCapResult::UnsupportedPlatform => {
                    "kernel hard=N/A (macOS/Windows: no userspace API; soft cap is the only guard)"
                        .to_string()
                }
                KernelCapResult::Disabled => {
                    "kernel hard=disabled (SEMANTEX_NO_RLIMIT=1)".to_string()
                }
                KernelCapResult::Failed(e) => {
                    format!("kernel hard=install failed (errno={e})")
                }
            };
            eprintln!(
                "[semantex] memory caps: app soft={soft_mb} MB, {kernel_desc}; system RAM={ram}. \
                 Override via SEMANTEX_MAX_RSS_MB; suppress via SEMANTEX_QUIET_LIMITS=1."
            );
        }
    }

    // Register mimalloc purge so memory module can force page return to OS.
    semantex_core::memory::register_purge_fn(mimalloc_purge);

    // Out-of-band RSS watchdog: catches a single huge allocation/library call
    // with no `check_rss_or_abort` checkpoint inside it (the gap that let one
    // real-world index build reach ~30GB RSS on a 48GB Mac before it was
    // killed by hand). Independent of the main thread's call stack, so it
    // closes the gap regardless of which code path causes it. Must be spawned
    // after `register_purge_fn` (it purges on overshoot) and before any
    // memory-heavy work starts.
    semantex_core::memory::spawn_rss_watchdog();

    // Pure-indexing subcommands (`index`, `watch`) run as a good background
    // citizen: lower the whole process to background CPU priority so it yields
    // to the developer's foreground work. Done before ORT/rayon threads spawn
    // so they inherit the lowered priority (Linux). Skipped for `mcp`/`serve`,
    // which answer queries. Only matters under contention — idle builds stay
    // full speed.
    if wants_background_priority() {
        semantex_core::priority::lower_process_to_background();
    }

    // Enable ANSI escape codes on Windows Terminal / PowerShell
    #[cfg(windows)]
    {
        let _ = colored::control::set_virtual_terminal(true);
    }

    // ── Hard memory limits: must be set before ORT/OpenMP loads ───────────
    //
    // fastembed hardcodes intra_op_num_threads = available_parallelism() (all
    // CPU cores). On a 12-core machine, ORT allocates ~1 GB of inference
    // workspace per thread on first search → 12 GB spike, which OOMs the
    // system when two daemons are running.
    //
    // Capping at 4 threads reduces the first-inference spike from ~12 GB to
    // ~4 GB total. Queries remain fast — inference is compute-bound, not
    // memory-bandwidth-bound, so 4 threads still saturate the model.
    //
    // SAFETY: Called at program startup before any threads are spawned.
    // OMP_NUM_THREADS must be set before the OpenMP runtime initialises
    // (which happens on the first parallel region, i.e. first ORT inference).
    {
        let ort_threads = std::env::var("SEMANTEX_ORT_THREADS")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(4)
            .to_string();
        unsafe {
            for var in &[
                "OMP_NUM_THREADS",        // OpenMP (used by ORT on macOS)
                "MKL_NUM_THREADS",        // Intel MKL (alternative BLAS)
                "OPENBLAS_NUM_THREADS",   // OpenBLAS
                "VECLIB_MAXIMUM_THREADS", // Apple Accelerate / vecLib
            ] {
                // Don't clobber an explicit user setting
                if std::env::var(var).is_err() {
                    std::env::set_var(var, &ort_threads);
                }
            }

            // SEMANTEX_ORT_THREADS=1 for memory-constrained subcommands.
            //
            // Previously set inside `McpServer::with_toolset` (server.rs), but
            // by the time that constructor ran, mimalloc + tracing threads had
            // already been spawned — which is real UB per Rust's strict
            // env-var thread rules on Linux/Apple. Moving it here guarantees
            // the env var is set BEFORE any thread (mimalloc init, tracing
            // subscriber, tokio runtime, ORT session) reads it.
            //
            // Applied to: mcp (many parallel sessions, each ~1 GB ORT scratch),
            // index (verified 10 GB spike with 4 threads on a 2 k-chunk repo
            // due to dense rebuild + ORT scratch), watch (continuous encoding),
            // serve (similar memory profile). One-off CLI search keeps the
            // higher default (4) since it pays the cost once.
            if memory_constrained_mode_requested() && std::env::var("SEMANTEX_ORT_THREADS").is_err()
            {
                std::env::set_var("SEMANTEX_ORT_THREADS", "1");
                // The OMP/MKL/etc. vars set above must follow suit so we don't
                // get 4-thread BLAS underneath single-thread ORT.
                for var in &[
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                ] {
                    std::env::set_var(var, "1");
                }
            }
        }
    }

    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::WARN.into()),
        )
        .with_target(false)
        .init();

    let cli = Cli::parse();

    // Hook flags — fast-path, no config needed, must complete in <50ms
    if cli.session_hook {
        return commands::hooks::cmd_session_hook();
    }
    if cli.grep_hook {
        return commands::hooks::cmd_grep_hook(cli.deny);
    }
    if cli.session_end_hook {
        return commands::hooks::cmd_session_end_hook();
    }
    if cli.subagent_hook {
        return commands::hooks::cmd_subagent_hook();
    }
    if cli.bash_hook {
        return commands::hooks::cmd_bash_hook(cli.deny);
    }
    if cli.read_hook {
        return commands::hooks::cmd_read_hook(cli.deny);
    }
    if cli.uninstall_claude_code {
        return commands::install::uninstall_claude_code();
    }
    if cli.uninstall {
        return commands::install::uninstall_all();
    }
    if cli.install_claude_code {
        return commands::install::install_claude_code(None);
    }
    if cli.install_opencode {
        return commands::install::install_opencode(false);
    }
    if cli.install_codex {
        return commands::install::install_codex();
    }

    // Fast daemon path: skip config loading + ORT detection when daemon port file exists.
    // Saves ~5-7ms by avoiding YAML config I/O + ORT dylib stat() calls.
    // Skipped entirely for non-CurrentRepo scopes: the daemon only ever
    // knows about the ONE project it was started for, so a cross-repo query
    // falls through to the slower (config-loading) path below, where the
    // federated branch lives. Gated on the PARSED scope (not a string
    // compare) so values that parse to CurrentRepo — e.g. `--scope ""` —
    // stay on the fast single-repo path.
    #[allow(clippy::collapsible_if)]
    if cli.command.is_none()
        && commands::federated::parse_scope(&cli.scope)
            == semantex_core::search::federation::SearchScope::CurrentRepo
    {
        // Fast path for --around: graph walk via daemon
        if let Some(ref symbol) = cli.around {
            let path = cli.path.as_deref().unwrap_or(Path::new("."));
            if let Ok(project_path) = path.canonicalize() {
                if let Ok(port) = semantex_core::server::read_daemon_port(&project_path) {
                    if commands::search::run_around_daemon(symbol, port).is_ok() {
                        return Ok(());
                    }
                }
            }
        }

        // Fast path for --deep: deep search via daemon
        if cli.deep && !cli.queries.is_empty() {
            let path = cli.path.as_deref().unwrap_or(Path::new("."));
            if let Ok(project_path) = path.canonicalize() {
                if let Ok(port) = semantex_core::server::read_daemon_port(&project_path) {
                    if commands::search::run_deep_via_binary_daemon(
                        cli.queries.first().map_or("", String::as_str),
                        port,
                        cli.verbose,
                    )
                    .is_ok()
                    {
                        return Ok(());
                    }
                }
            }
        }

        if !cli.queries.is_empty() {
            let path = cli.path.as_deref().unwrap_or(Path::new("."));
            if let Ok(project_path) = path.canonicalize() {
                if let Ok(port) = semantex_core::server::read_daemon_port(&project_path) {
                    // Batch mode: multiple queries with --refs
                    if cli.queries.len() > 1 && !cli.refs {
                        eprintln!(
                            "[warning] multiple queries require --refs for batch mode; using first query only"
                        );
                    }
                    if cli.queries.len() > 1 && cli.refs {
                        if commands::search::run_batch_via_binary_daemon(
                            &cli.queries,
                            cli.max_count,
                            cli.code_only,
                            &cli.include_type,
                            &cli.exclude_type,
                            port,
                        )
                        .is_ok()
                        {
                            return Ok(());
                        }
                    } else {
                        let search_opts = commands::search::SearchOpts {
                            queries: cli.queries.clone(),
                            path: cli.path.clone().unwrap_or_else(|| PathBuf::from(".")),
                            max_count: cli.max_count,
                            verbose: cli.content || cli.verbose,
                            context: cli.context,
                            line_number: !cli.no_line_number,
                            rerank: cli.rerank,
                            dense_only: cli.dense_only,
                            sparse_only: cli.sparse_only,
                            json: cli.json,
                            no_content: cli.no_content,
                            snippet: cli.snippet,
                            grep: cli.grep,
                            grep_mode: cli.grep_mode,
                            include_type: cli.include_type.clone(),
                            exclude_type: cli.exclude_type.clone(),
                            code_only: cli.code_only,
                            refs: cli.refs,
                            peek: cli.peek,
                            pattern: cli.pattern.clone(),
                            deep: cli.deep,
                        };
                        if commands::search::run_via_binary_daemon(&search_opts, port).is_ok() {
                            return Ok(());
                        }
                        // Fall through to normal path if daemon connection fails
                    }
                }
            }
        }
    }

    // Resolve the ONNX Runtime shared library for `ort`'s load-dynamic mode and
    // expose it via ORT_DYLIB_PATH. Deferred to after the daemon fast-path
    // (daemon queries never load ORT in-process), but kept before any threads
    // spawn so the set_var below is sound.
    if std::env::var_os("ORT_DYLIB_PATH").is_none_or(|v| v.is_empty())
        && let Some(path) = resolve_ort_dylib()
    {
        // SAFETY: Called at program startup before any threads are spawned.
        unsafe { std::env::set_var("ORT_DYLIB_PATH", &path) };
    }

    // Build config with CLI overrides
    let path = cli.path.clone().unwrap_or_else(|| PathBuf::from("."));
    let mut config = semantex_core::SemantexConfig::load(Some(&path))?;

    if let Some(max_file_size) = cli.max_file_size {
        config.max_file_size = max_file_size;
    }
    if let Some(max_file_count) = cli.max_file_count {
        config.max_file_count = max_file_count;
    }
    if cli.rerank {
        config.rerank = true;
    }
    config.max_count = cli.max_count;
    config.content = cli.content;
    config.context_lines = cli.context;

    match cli.command {
        Some(Commands::Index { path, force }) => {
            telemetry::track("index");
            commands::index::run(&path, &config, force)
        }
        Some(Commands::Watch { path }) => {
            telemetry::track("watch");
            commands::watch::run(&path, &config)
        }
        Some(Commands::Serve { path }) => {
            telemetry::track("serve");
            commands::serve::run(&path, &config)
        }
        Some(Commands::Stop { path }) => commands::stop::run(&path),
        Some(Commands::DownloadModels) => {
            telemetry::track("download_models");
            commands::download::run(&config)
        }
        Some(Commands::Status { path }) => commands::status::run(&path, &config),
        Some(Commands::Mcp {
            http,
            port,
            allow_remote,
            toolset,
            auth_token,
            require_auth,
        }) => {
            telemetry::track("mcp");
            commands::mcp::run(
                &config,
                http,
                port,
                allow_remote,
                &toolset,
                auth_token,
                require_auth,
            )
        }
        Some(Commands::InstallClaudeCode { scope }) => {
            telemetry::track("install_claude_code");
            commands::install::install_claude_code(scope)
        }
        Some(Commands::UninstallClaudeCode) => {
            telemetry::track("uninstall_claude_code");
            commands::install::uninstall_claude_code()
        }
        Some(Commands::InstallCodex) => {
            telemetry::track("install_codex");
            commands::install::install_codex()
        }
        Some(Commands::UninstallCodex) => {
            telemetry::track("uninstall_codex");
            commands::install::uninstall_codex()
        }
        Some(Commands::InstallOpenCode { global }) => {
            telemetry::track("install_opencode");
            commands::install::install_opencode(global)
        }
        Some(Commands::UninstallOpenCode { global }) => {
            telemetry::track("uninstall_opencode");
            commands::install::uninstall_opencode(global)
        }
        Some(Commands::InstallCursor { global }) => {
            telemetry::track("install_cursor");
            commands::install::install_cursor(global)
        }
        Some(Commands::UninstallCursor { global }) => {
            telemetry::track("uninstall_cursor");
            commands::install::uninstall_cursor(global)
        }
        Some(Commands::InstallDevinDesktop) => {
            telemetry::track("install_devin_desktop");
            commands::install::install_devin_desktop()
        }
        Some(Commands::UninstallDevinDesktop) => {
            telemetry::track("uninstall_devin_desktop");
            commands::install::uninstall_devin_desktop()
        }
        Some(Commands::InstallAider) => {
            telemetry::track("install_aider");
            commands::install::install_aider()
        }
        Some(Commands::UninstallAider) => {
            telemetry::track("uninstall_aider");
            commands::install::uninstall_aider()
        }
        Some(Commands::InstallTrae) => {
            telemetry::track("install_trae");
            commands::install::install_trae()
        }
        Some(Commands::UninstallTrae) => {
            telemetry::track("uninstall_trae");
            commands::install::uninstall_trae()
        }
        Some(Commands::InstallCopilot) => {
            telemetry::track("install_copilot");
            commands::install::install_copilot()
        }
        Some(Commands::UninstallCopilot) => {
            telemetry::track("uninstall_copilot");
            commands::install::uninstall_copilot()
        }
        Some(Commands::InstallGemini { global }) => {
            telemetry::track("install_gemini");
            commands::install::install_gemini(global)
        }
        Some(Commands::UninstallGemini { global }) => {
            telemetry::track("uninstall_gemini");
            commands::install::uninstall_gemini(global)
        }
        Some(Commands::Connect { path }) => commands::connect::run(&path),
        Some(Commands::Disconnect) => commands::disconnect::run(),
        Some(Commands::Validate { path }) => commands::validate::run(&path),
        Some(Commands::History { file, path, limit }) => {
            telemetry::track("history");
            commands::history::run(&file, &path, limit)
        }
        Some(Commands::SkillsGenerate {
            out,
            platform,
            force,
        }) => {
            telemetry::track("skills_generate");
            let platform_filter = if platform == "all" {
                None
            } else {
                Some(platform.as_str())
            };
            commands::skills_generate::run(out, platform_filter, force)
        }
        #[cfg(feature = "llm")]
        Some(Commands::LlmStatus) => commands::llm_status::run(),
        Some(Commands::Agent {
            query,
            path,
            route,
            full,
            budget,
            json,
            json_hits,
            scope,
        }) => {
            use semantex_core::search::agent_classifier::AgentRoute;
            use semantex_core::server::protocol::AgentRequest;
            telemetry::track("agent");

            // `--scope` != `repo` runs entirely in-process (no daemon
            // required) and never falls through to the daemon-based path
            // below — see `commands::federated::run_agent`'s doc comment for
            // why route classification is skipped for cross-repo queries.
            // This keeps `--scope repo` (the default) byte-identical to
            // every pre-federation invocation.
            let parsed_scope = commands::federated::parse_scope(&scope);
            if parsed_scope != semantex_core::search::federation::SearchScope::CurrentRepo {
                let project_path = path
                    .canonicalize()
                    .with_context(|| format!("Invalid path: {}", path.display()))?;
                let (formatted, hits) = commands::federated::run_agent(
                    &parsed_scope,
                    &project_path,
                    &query,
                    budget.unwrap_or(0),
                    &config,
                );
                if json_hits {
                    println!("{}", serde_json::to_string_pretty(&hits)?);
                } else if json {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&serde_json::json!({
                            "formatted": formatted,
                            "results": hits,
                        }))?
                    );
                } else {
                    println!("{formatted}");
                }
                return Ok(());
            }

            // Accept any AgentRoute variant via FromStr (covers snake_case
            // and no-separator forms). Two convenience aliases stay: "search"
            // for semantic, "exact" for exact_symbol, "files" for file_pattern.
            let route_override = match route.as_deref() {
                None => None,
                Some(r) => Some(match r {
                    "search" => AgentRoute::Semantic,
                    "exact" => AgentRoute::ExactSymbol,
                    "files" => AgentRoute::FilePattern,
                    other => other.parse::<AgentRoute>().map_err(|_| {
                        anyhow::anyhow!(
                            "Unknown route: {other}. Valid: file_pattern (files), regex, \
                             exact_symbol (exact), structural, deep, analytical, exhaustive, \
                             semantic (search), architecture, feature_planning"
                        )
                    })?,
                }),
            };

            let project_path = path
                .canonicalize()
                .with_context(|| format!("Invalid path: {}", path.display()))?;
            let port = semantex_core::server::read_daemon_port(&project_path)
                .context("Daemon not running. Start it with: semantex serve <path>")?;

            let request = AgentRequest {
                query,
                route: route_override,
                budget,
                full_code: full,
            };

            if json_hits {
                // Forced-route measurement seam: return the engine's structured
                // ranked hits (Vec<SearchResultItem>) as a JSON array in the
                // SAME shape as `search --json`, so the relevance harness's
                // file-level matcher consumes it unchanged. Each `file` is
                // repo-relative.
                let response = semantex_core::server::daemon_agent_hits_binary(port, request)
                    .context("Failed to send agent-hits request to daemon")?;
                println!("{}", serde_json::to_string_pretty(&response.hits)?);
                return Ok(());
            }

            let response = semantex_core::server::daemon_agent_binary(port, request)
                .context("Failed to send agent request to daemon")?;

            if json {
                println!("{}", serde_json::to_string_pretty(&response)?);
            } else {
                println!("{}", response.formatted);
            }
            Ok(())
        }
        Some(Commands::DistillStaticTable {
            corpus,
            out,
            verify,
        }) => commands::distill_static_table::run(&corpus, &out, verify),
        None => {
            // Default: search
            if let Some(ref symbol) = cli.around {
                let path = cli.path.clone().unwrap_or_else(|| PathBuf::from("."));
                return commands::search::run_around(symbol, &path, &config);
            }

            if cli.queries.is_empty() {
                // No query, no subcommand — print help
                use clap::CommandFactory;
                Cli::command().print_help()?;
                println!();
                Ok(())
            } else if commands::federated::parse_scope(&cli.scope)
                != semantex_core::search::federation::SearchScope::CurrentRepo
            {
                // Cross-repo fan-out — see `commands::federated::run_search`'s
                // doc comment. Gated on the PARSED scope so any value that
                // parses to CurrentRepo (`repo`, empty) takes the unchanged
                // single-repo branch below. NOTE: federated mode honors only
                // query/--max-count/--rerank/--grep-mode/--json; single-repo
                // flags like --refs/--peek/--deep/--around/--dense-only/
                // --sparse-only are ignored on this path (see --scope's help
                // text).
                telemetry::track("search_federated");
                let parsed_scope = commands::federated::parse_scope(&cli.scope);
                let path = cli.path.unwrap_or_else(|| PathBuf::from("."));
                let project_path = path
                    .canonicalize()
                    .with_context(|| format!("Invalid path: {}", path.display()))?;
                commands::federated::run_search(
                    &parsed_scope,
                    &project_path,
                    cli.queries.first().map_or("", String::as_str),
                    cli.max_count,
                    cli.rerank,
                    cli.grep_mode,
                    cli.json,
                    &config,
                )
            } else {
                telemetry::track("search");
                let search_opts = commands::search::SearchOpts {
                    queries: cli.queries,
                    path: cli.path.unwrap_or_else(|| PathBuf::from(".")),
                    max_count: cli.max_count,
                    verbose: cli.content || cli.verbose,
                    context: cli.context,
                    line_number: !cli.no_line_number,
                    rerank: cli.rerank,
                    dense_only: cli.dense_only,
                    sparse_only: cli.sparse_only,
                    json: cli.json,
                    no_content: cli.no_content,
                    snippet: cli.snippet,
                    grep: cli.grep,
                    grep_mode: cli.grep_mode,
                    include_type: cli.include_type,
                    exclude_type: cli.exclude_type,
                    code_only: cli.code_only,
                    refs: cli.refs,
                    peek: cli.peek,
                    pattern: cli.pattern,
                    deep: cli.deep,
                };
                commands::search::run(&search_opts, &config)
            }
        }
    }
}

#[cfg(test)]
mod arg_peek_tests {
    use super::{
        command_wants_onnxruntime, memory_constrained_from_args,
        wants_background_priority_from_args,
    };

    #[test]
    fn background_priority_only_for_pure_index_paths() {
        // index/watch are pure background indexing — safe to nice the process.
        assert!(wants_background_priority_from_args(["index", "."]));
        assert!(wants_background_priority_from_args(["watch", "/p"]));
        // mcp/serve answer queries — must NOT be niced.
        assert!(!wants_background_priority_from_args(["mcp"]));
        assert!(!wants_background_priority_from_args(["serve"]));
        // unrelated subcommands stay normal priority.
        assert!(!wants_background_priority_from_args(["search", "foo"]));
    }

    #[test]
    fn detects_mcp_subcommand_plain() {
        assert!(memory_constrained_from_args(["mcp"]));
    }

    #[test]
    fn detects_index_subcommand() {
        // Indexing is the most memory-heavy operation — must lower ORT threads.
        assert!(memory_constrained_from_args(["index", "."]));
    }

    #[test]
    fn detects_watch_subcommand() {
        assert!(memory_constrained_from_args(["watch", "/some/path"]));
    }

    #[test]
    fn detects_serve_subcommand() {
        assert!(memory_constrained_from_args(["serve"]));
    }

    #[test]
    fn detects_mcp_subcommand_with_flags() {
        assert!(memory_constrained_from_args([
            "mcp", "--http", "--port", "5050"
        ]));
    }

    #[test]
    fn detects_mcp_with_global_flag_before() {
        // global flags (start with `-`) are skipped to find the subcommand
        assert!(memory_constrained_from_args(["-v", "mcp"]));
    }

    #[test]
    fn rejects_status_and_other_lightweight_subcommands() {
        // status/health/validate are cheap reads — keep the higher thread default.
        assert!(!memory_constrained_from_args(["status"]));
        assert!(!memory_constrained_from_args(["validate"]));
        assert!(!memory_constrained_from_args(["download-models"]));
    }

    #[test]
    fn rejects_no_subcommand() {
        assert!(!memory_constrained_from_args(Vec::<&str>::new()));
        // Search-mode invocation: positional query, not a memory-constrained mode.
        assert!(!memory_constrained_from_args(["how does auth work"]));
    }

    #[test]
    fn stops_at_double_dash() {
        // Anything after `--` is a positional arg to the previous command —
        // not a subcommand.
        assert!(!memory_constrained_from_args(["--", "mcp"]));
    }

    #[test]
    fn handles_mcp_after_pure_flags() {
        // Pure flags (no flag-takes-value form) before the subcommand work.
        assert!(memory_constrained_from_args(["--json", "mcp", "--http"]));
    }

    #[test]
    fn known_limitation_flag_with_value_misclassifies() {
        // Documented limitation: a flag that takes a value followed by an
        // arg looks indistinguishable from "subcommand here". Acceptable
        // because the worst outcome is SEMANTEX_ORT_THREADS=4 instead of 1.
        assert!(!memory_constrained_from_args(["--max-count", "5", "mcp"]));
    }

    #[test]
    fn ort_provision_for_search_and_heavy_subcommands() {
        // Default search (positional query) loads ONNX Runtime in-process.
        assert!(command_wants_onnxruntime(["how does auth work"]));
        assert!(command_wants_onnxruntime(["index", "."]));
        assert!(command_wants_onnxruntime(["watch", "/p"]));
        assert!(command_wants_onnxruntime(["serve"]));
        assert!(command_wants_onnxruntime(["mcp", "--http"]));
        assert!(command_wants_onnxruntime(["-v", "mcp"]));
        // A positional query after `--` is still a search.
        assert!(command_wants_onnxruntime(["--", "status"]));
    }

    #[test]
    fn ort_no_provision_for_lightweight_subcommands() {
        // These never load ONNX Runtime in-process, so don't trigger a download.
        assert!(!command_wants_onnxruntime(["status"]));
        assert!(!command_wants_onnxruntime(["stop"]));
        assert!(!command_wants_onnxruntime(["validate"]));
        assert!(!command_wants_onnxruntime(["download-models"])); // provisions itself
        assert!(!command_wants_onnxruntime(["install-claude-code"]));
        assert!(!command_wants_onnxruntime(["install-cursor"]));
        assert!(!command_wants_onnxruntime(["uninstall-cursor"]));
        assert!(!command_wants_onnxruntime(["install-devin-desktop"]));
        assert!(!command_wants_onnxruntime(["install-windsurf"])); // deprecated alias
        assert!(!command_wants_onnxruntime(["install-aider"]));
        assert!(!command_wants_onnxruntime(["install-trae"]));
        assert!(!command_wants_onnxruntime(["install-copilot"]));
        assert!(!command_wants_onnxruntime(["install-gemini"]));
        assert!(!command_wants_onnxruntime(["uninstall-codex"]));
        assert!(!command_wants_onnxruntime(["install-open-code"]));
        assert!(!command_wants_onnxruntime(["uninstall-open-code"]));
        assert!(!command_wants_onnxruntime(["connect"]));
        // v13 Wave 2: `history` is a read-only query over history.db — no
        // embedder needed.
        assert!(!command_wants_onnxruntime(["history"]));
        // Bare invocation / pure flags (help/version): no download.
        assert!(!command_wants_onnxruntime(Vec::<&str>::new()));
        assert!(!command_wants_onnxruntime(["--version"]));
    }
}

/// Clap-level tests for the `mcp` subcommand's HTTP auth flag plumbing
/// (`--auth-token`, `--require-auth`). See `commands::mcp::run` for the
/// resolution logic these flags feed into.
#[cfg(test)]
mod mcp_auth_arg_tests {
    use super::{Cli, Commands};
    use clap::Parser;

    #[test]
    fn auth_token_and_require_auth_parse_with_http() {
        let cli = Cli::try_parse_from([
            "semantex",
            "mcp",
            "--http",
            "--auth-token",
            "s3cr3t",
            "--require-auth",
        ])
        .expect("clap should accept --auth-token and --require-auth with --http");
        let Some(Commands::Mcp {
            http,
            allow_remote,
            auth_token,
            require_auth,
            ..
        }) = cli.command
        else {
            panic!("expected Commands::Mcp");
        };
        assert!(http);
        assert!(!allow_remote);
        assert_eq!(auth_token.as_deref(), Some("s3cr3t"));
        assert!(require_auth);
    }

    #[test]
    fn auth_flags_default_to_none_and_off() {
        let cli = Cli::try_parse_from(["semantex", "mcp", "--http"])
            .expect("clap should accept --http alone");
        let Some(Commands::Mcp {
            auth_token,
            require_auth,
            ..
        }) = cli.command
        else {
            panic!("expected Commands::Mcp");
        };
        assert_eq!(auth_token, None);
        assert!(!require_auth);
    }

    #[test]
    fn allow_remote_and_auth_token_compose() {
        let cli = Cli::try_parse_from([
            "semantex",
            "mcp",
            "--http",
            "--allow-remote",
            "--auth-token",
            "tok",
        ])
        .expect("clap should accept --allow-remote with --auth-token");
        let Some(Commands::Mcp {
            allow_remote,
            auth_token,
            ..
        }) = cli.command
        else {
            panic!("expected Commands::Mcp");
        };
        assert!(allow_remote);
        assert_eq!(auth_token.as_deref(), Some("tok"));
    }

    #[test]
    fn auth_token_without_http_is_rejected() {
        // `requires = "http"` on --auth-token: stdio mode has no HTTP auth surface.
        let result = Cli::try_parse_from(["semantex", "mcp", "--auth-token", "tok"]);
        assert!(
            result.is_err(),
            "--auth-token without --http should fail to parse"
        );
    }

    #[test]
    fn require_auth_without_http_is_rejected() {
        let result = Cli::try_parse_from(["semantex", "mcp", "--require-auth"]);
        assert!(
            result.is_err(),
            "--require-auth without --http should fail to parse"
        );
    }
}
