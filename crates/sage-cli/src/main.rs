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

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(
    name = "sage",
    about = "Semantic grep — fully local code search with dense+sparse hybrid retrieval",
    version,
    after_help = "Examples:\n  sage \"function that handles authentication\" .\n  sage index .\n  sage --content --max-count 10 \"database connection pool\"\n  sage --json --rerank \"error handling\"\n  sage -G --grep \"ConnectionFactory\" .  # grep-like exhaustive search"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Search query (when no subcommand is given)
    #[arg(value_name = "QUERY")]
    query: Option<String>,

    /// Path to search/index (defaults to current directory)
    #[arg(value_name = "PATH")]
    path: Option<PathBuf>,

    /// Maximum number of results
    #[arg(short = 'm', long, default_value = "10", env = "SAGE_MAX_COUNT")]
    max_count: usize,

    /// Show matching content snippets
    #[arg(short = 'c', long, env = "SAGE_CONTENT")]
    content: bool,

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
        env = "SAGE_INCLUDE_TYPES",
        value_delimiter = ','
    )]
    include_type: Vec<String>,

    /// Exclude files with this extension (repeatable)
    #[arg(
        long = "exclude-type",
        value_name = "EXT",
        env = "SAGE_EXCLUDE_TYPES",
        value_delimiter = ','
    )]
    exclude_type: Vec<String>,

    /// Exclude non-code files (md, json, yaml, toml, txt, log, cfg, ini, env, pdf, ipynb, lock)
    #[arg(long)]
    code_only: bool,

    /// Maximum file size to index (bytes)
    #[arg(long, env = "SAGE_MAX_FILE_SIZE")]
    max_file_size: Option<u64>,

    /// Maximum number of files to index
    #[arg(long)]
    max_file_count: Option<usize>,

    /// Internal: Claude Code session hook (outputs JSON context)
    #[arg(long, hide = true)]
    session_hook: bool,

    /// Internal: Claude Code task hook (outputs JSON for agent prompts)
    #[arg(long, hide = true)]
    task_hook: bool,

    /// Internal: Claude Code PreToolUse hook for Grep|Glob interception
    #[arg(long, hide = true)]
    grep_hook: bool,

    /// Internal: Claude Code SessionEnd hook for daemon cleanup
    #[arg(long, hide = true)]
    session_end_hook: bool,

    /// Uninstall sage hooks from Claude Code
    #[arg(long)]
    uninstall_claude_code: bool,

    /// Remove all sage hook registrations
    #[arg(long)]
    uninstall: bool,

    /// Install sage hooks into Claude Code's settings.json
    #[arg(long)]
    install_claude_code: bool,

    /// Install sage hooks into OpenCode
    #[arg(long)]
    install_opencode: bool,

    /// Install sage hooks into Codex
    #[arg(long)]
    install_codex: bool,

    /// Regex pattern to combine with semantic search (hybrid mode).
    /// Example: sage -e "async fn" "database pool"
    #[arg(short = 'e', long = "pattern")]
    pattern: Option<String>,
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
    InstallClaudeCode,
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
        let ort_threads = std::env::var("SAGE_ORT_THREADS")
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
    if cli.task_hook {
        return commands::hooks::cmd_task_hook();
    }
    if cli.grep_hook {
        return commands::hooks::cmd_grep_hook();
    }
    if cli.session_end_hook {
        return commands::hooks::cmd_session_end_hook();
    }
    if cli.uninstall_claude_code {
        return commands::install::uninstall_claude_code();
    }
    if cli.uninstall {
        return commands::install::uninstall_all();
    }
    if cli.install_claude_code {
        return commands::install::install_claude_code();
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
        if let Some(ref query) = cli.query {
            let path = cli.path.as_deref().unwrap_or(Path::new("."));
            if let Ok(project_path) = path.canonicalize() {
                if let Ok(port) = sage_core::server::read_daemon_port(&project_path) {
                    let search_opts = commands::search::SearchOpts {
                        query: query.clone(),
                        path: cli.path.clone().unwrap_or_else(|| PathBuf::from(".")),
                        max_count: cli.max_count,
                        content: cli.content,
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
                        pattern: cli.pattern.clone(),
                    };
                    if commands::search::run_via_binary_daemon(&search_opts, port).is_ok() {
                        return Ok(());
                    }
                    // Fall through to normal path if daemon connection fails
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
    let mut config = sage_core::SageConfig::load(Some(&path))?;

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
        Some(Commands::InstallClaudeCode) => commands::install::install_claude_code(),
        Some(Commands::InstallCodex) => commands::install::install_codex(),
        Some(Commands::InstallOpenCode) => commands::install::install_opencode(),
        Some(Commands::Connect { path }) => commands::connect::run(&path),
        Some(Commands::Disconnect) => commands::disconnect::run(),
        None => {
            // Default: search
            if let Some(query) = cli.query {
                let search_opts = commands::search::SearchOpts {
                    query,
                    path: cli.path.unwrap_or_else(|| PathBuf::from(".")),
                    max_count: cli.max_count,
                    content: cli.content,
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
                    pattern: cli.pattern,
                };
                commands::search::run(&search_opts, &config)
            } else {
                // No query, no subcommand — print help
                use clap::CommandFactory;
                Cli::command().print_help()?;
                println!();
                Ok(())
            }
        }
    }
}
