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
}

#[derive(Subcommand)]
enum Commands {
    /// Build or update the search index
    Index {
        /// Path to index (defaults to current directory)
        #[arg(default_value = ".")]
        path: PathBuf,
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

        /// Allow binding to 0.0.0.0 (any interface). Off by default — only
        /// loopback is safe without auth. Prints a stderr warning when enabled.
        #[arg(long, requires = "http")]
        allow_remote: bool,

        /// Named toolset bundle to expose: `core` (4 tools), `structural` (5 tools),
        /// or `all` (13 tools, default).
        #[arg(long, value_name = "NAME", default_value = "all")]
        toolset: String,
    },
    /// Install Claude Code integration
    InstallClaudeCode {
        /// Installation scope: user (all projects), project (shared via git), or local (this repo, gitignored)
        #[arg(long, value_enum)]
        scope: Option<commands::install::InstallScope>,
    },
    /// Install Codex integration
    InstallCodex,
    /// Install OpenCode integration
    InstallOpenCode,
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
    /// Generate per-platform skill files describing semantex's MCP tools.
    ///
    /// Emits one file per supported agent platform (Claude Code, Cursor, Codex,
    /// Aider, Gemini CLI, Copilot CLI, OpenCode, Windsurf, Trae) from a single
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

        /// Override query classification: semantic, deep, exact_symbol, structural, regex, analytical, exhaustive, file_pattern, architecture, exhaustive_structural, deep_with_examples, feature_planning
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
/// sessions, each loading ColBERT), `index` (large batch encoding under
/// PLAID rebuild — verified to spike to 10 GB on a 2 k-chunk repo with
/// 4 threads), `watch` (continuous incremental encoding).
///
/// On a 12-core machine, fastembed's default of "all cores" allocates
/// ~1 GB of ORT inference workspace per thread on first encode. 1 thread
/// keeps the encoding spike under 1 GB without meaningfully slowing
/// large batch operations (encoding is bandwidth-bound, not compute-bound).
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

#[cfg(test)]
mod arg_peek_tests {
    use super::memory_constrained_from_args;

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
            // due to PLAID rebuild + ORT scratch), watch (continuous encoding),
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
        return commands::install::install_opencode();
    }
    if cli.install_codex {
        return commands::install::install_codex();
    }

    // Fast daemon path: skip config loading + ORT detection when daemon port file exists.
    // Saves ~5-7ms by avoiding YAML config I/O + ORT dylib stat() calls.
    #[allow(clippy::collapsible_if)]
    if cli.command.is_none() {
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

    // Auto-detect ONNX Runtime library if not explicitly set.
    // Deferred to after the daemon fast-path: daemon queries never load ORT in-process.
    if std::env::var("ORT_DYLIB_PATH").is_err()
        && let Some(path) = find_ort_dylib()
    {
        // SAFETY: Called at program startup before any threads are spawned.
        unsafe { std::env::set_var("ORT_DYLIB_PATH", path) };
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
        Some(Commands::Index { path }) => {
            telemetry::track("index");
            commands::index::run(&path, &config)
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
        }) => {
            telemetry::track("mcp");
            commands::mcp::run(&config, http, port, allow_remote, &toolset)
        }
        Some(Commands::InstallClaudeCode { scope }) => {
            telemetry::track("install_claude_code");
            commands::install::install_claude_code(scope)
        }
        Some(Commands::InstallCodex) => {
            telemetry::track("install_codex");
            commands::install::install_codex()
        }
        Some(Commands::InstallOpenCode) => {
            telemetry::track("install_opencode");
            commands::install::install_opencode()
        }
        Some(Commands::Connect { path }) => commands::connect::run(&path),
        Some(Commands::Disconnect) => commands::disconnect::run(),
        Some(Commands::Validate { path }) => commands::validate::run(&path),
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
        }) => {
            use semantex_core::search::agent_classifier::AgentRoute;
            use semantex_core::server::protocol::AgentRequest;
            telemetry::track("agent");

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
                             semantic (search), architecture, exhaustive_structural, \
                             deep_with_examples, feature_planning"
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

            let response = semantex_core::server::daemon_agent_binary(port, request)
                .context("Failed to send agent request to daemon")?;

            if json {
                println!("{}", serde_json::to_string_pretty(&response)?);
            } else {
                println!("{}", response.formatted);
            }
            Ok(())
        }
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
