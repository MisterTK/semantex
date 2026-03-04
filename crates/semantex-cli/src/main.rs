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

use anyhow::{Context, Result};
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
    /// Run as MCP server (stdio transport)
    Mcp,
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
    /// Intelligent code search with automatic query routing and formatting.
    Agent {
        /// The search query
        query: String,

        /// Path to the project (default: current directory)
        #[arg(short, long, default_value = ".")]
        path: PathBuf,

        /// Override query classification: semantic, deep, exact_symbol, structural, regex, analytical, file_pattern
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

fn main() -> Result<()> {
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
        Some(Commands::Index { path }) => commands::index::run(&path, &config),
        Some(Commands::Watch { path }) => commands::watch::run(&path, &config),
        Some(Commands::Serve { path }) => commands::serve::run(&path, &config),
        Some(Commands::Stop { path }) => commands::stop::run(&path),
        Some(Commands::DownloadModels) => commands::download::run(&config),
        Some(Commands::Status { path }) => commands::status::run(&path, &config),
        Some(Commands::Mcp) => commands::mcp::run(&config),
        Some(Commands::InstallClaudeCode { scope }) => {
            commands::install::install_claude_code(scope)
        }
        Some(Commands::InstallCodex) => commands::install::install_codex(),
        Some(Commands::InstallOpenCode) => commands::install::install_opencode(),
        Some(Commands::Connect { path }) => commands::connect::run(&path),
        Some(Commands::Disconnect) => commands::disconnect::run(),
        Some(Commands::Validate { path }) => commands::validate::run(&path),
        Some(Commands::Agent { query, path, route, full, budget, json }) => {
            use semantex_core::server::protocol::AgentRequest;
            use semantex_core::search::agent_classifier::AgentRoute;

            let route_override = match route.as_deref() {
                None => None,
                Some(r) => Some(match r {
                    "search" | "semantic" => AgentRoute::Semantic,
                    "deep" => AgentRoute::Deep,
                    "exact_symbol" | "exact" => AgentRoute::ExactSymbol,
                    "structural" => AgentRoute::Structural,
                    "regex" => AgentRoute::Regex,
                    "analytical" => AgentRoute::Analytical,
                    "file_pattern" | "files" => AgentRoute::FilePattern,
                    other => anyhow::bail!("Unknown route: {other}. Valid: semantic, deep, exact_symbol, structural, regex, analytical, file_pattern"),
                }),
            };

            let project_path = path.canonicalize()
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
