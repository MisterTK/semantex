use anyhow::{Context, Result};
use colored::Colorize;
use semantex_core::config::SemantexConfig;
use semantex_core::index::state::{self, IndexState};
use semantex_core::search::SearchQuery;
use semantex_core::search::hybrid::HybridSearcher;
use semantex_core::search::ripgrep_fallback;
use semantex_core::server::protocol::{SearchRequest, SearchResultItem};
use semantex_core::types::{ChunkType, SearchResult};
use std::path::{Path, PathBuf};

pub struct SearchOpts {
    pub query: String,
    pub path: PathBuf,
    pub max_count: usize,
    pub verbose: bool,
    pub context: usize,
    pub line_number: bool,
    pub rerank: bool,
    pub dense_only: bool,
    pub sparse_only: bool,
    pub json: bool,
    pub no_content: bool,
    pub snippet: bool,
    pub grep: bool,
    pub grep_mode: bool,
    pub include_type: Vec<String>,
    pub exclude_type: Vec<String>,
    pub code_only: bool,
    pub pattern: Option<String>,
}

pub fn run(opts: &SearchOpts, config: &SemantexConfig) -> Result<()> {
    let target_path = opts
        .path
        .canonicalize()
        .with_context(|| format!("Invalid path: {}", opts.path.display()))?;

    // Walk up from target_path to find the nearest project root with a .semantex index.
    // This lets `semantex "query" src/` work when the index lives at the repo root.
    let project_path = find_project_root(&target_path).unwrap_or_else(|| target_path.clone());

    let index_dir = SemantexConfig::project_index_dir(&project_path);

    match state::detect(&project_path) {
        IndexState::NotIndexed | IndexState::Stale => {
            eprintln!(
                "No index found for {}. Building in background...",
                project_path.display()
            );
            super::spawn_background_index(&project_path);
            // Fallback search scoped to what the user actually asked for
            return run_ripgrep_fallback(opts, &target_path);
        }
        IndexState::Building => {
            eprintln!("Index is building. Showing keyword results...");
            return run_ripgrep_fallback(opts, &target_path);
        }
        IndexState::Ready => { /* continue to normal search path */ }
    }

    // Try binary protocol to daemon first (fastest path)
    if let Ok(port) = semantex_core::server::read_daemon_port(&project_path) {
        if run_via_binary_daemon(opts, port).is_ok() {
            return Ok(());
        }
    }

    // Try JSON daemon as fallback
    if semantex_core::server::daemon_healthy(&project_path) {
        return run_via_daemon(opts, &project_path);
    }

    // Fall back to direct search
    run_direct(opts, config, &project_path, &index_dir)
}

/// Use ripgrep as a keyword-level fallback when the semantex index is unavailable.
fn run_ripgrep_fallback(opts: &SearchOpts, project_path: &std::path::Path) -> Result<()> {
    if !ripgrep_fallback::is_rg_available() {
        anyhow::bail!(
            "No semantex index and ripgrep (rg) not found.\n\
             Either wait for the index to finish building, or install ripgrep: https://github.com/BurntSushi/ripgrep"
        );
    }

    eprintln!(
        "{}",
        "[fallback: keyword search — semantic index building]"
            .yellow()
            .dimmed()
    );

    let max = if opts.grep_mode || opts.grep {
        50
    } else {
        opts.max_count
    };

    let results = ripgrep_fallback::search(&opts.query, project_path, max)?;

    if results.is_empty() {
        if opts.grep {
            // grep mode: no output for no results
        } else if opts.json {
            println!("[]");
        } else {
            eprintln!("No results found.");
        }
        return Ok(());
    }

    if opts.json {
        let json = ripgrep_fallback::format_as_json(&results);
        println!("{json}");
    } else if opts.grep {
        for r in &results {
            let preview: String = r.content.chars().take(120).collect();
            println!("{}:{}: {}", r.file.display(), r.line_number, preview.trim());
        }
    } else {
        for r in &results {
            println!(
                "{}:{} {} [{}]",
                r.file.display().to_string().green(),
                r.line_number.to_string().yellow(),
                "0.0000".cyan(),
                "ripgrep".dimmed(),
            );
            if opts.verbose {
                println!("  {}", r.content);
                println!();
            }
        }
        eprintln!("\n{} results [ripgrep fallback]", results.len());
    }

    Ok(())
}

pub fn run_via_binary_daemon(opts: &SearchOpts, port: u16) -> Result<()> {
    let request = SearchRequest {
        query: opts.query.clone(),
        max_results: if opts.grep_mode || opts.grep {
            50
        } else {
            opts.max_count
        },
        use_dense: if opts.grep_mode {
            false
        } else {
            !opts.sparse_only
        },
        use_sparse: true,
        use_rerank: if opts.grep_mode { false } else { opts.rerank },
        include_types: opts.include_type.clone(),
        exclude_types: opts.exclude_type.clone(),
        code_only: opts.code_only,
        include_content: opts.verbose
            || opts.grep
            || (opts.json && !opts.no_content)
            || (!opts.json && !opts.grep),
        snippet: !opts.verbose && !opts.json && !opts.grep,
        grep_mode: opts.grep_mode,
        regex_pattern: opts.pattern.clone(),
    };

    let response = semantex_core::server::daemon_search_binary(port, request)?;

    if response.results.is_empty() {
        if opts.grep {
            // grep mode: no output for no results
        } else if opts.json {
            println!("[]");
        } else {
            eprintln!("No results found.");
        }
        return Ok(());
    }

    if opts.grep {
        print_grep_daemon(&response.results)?;
    } else if opts.json {
        print_json_daemon(&response.results)?;
    } else {
        print_results_daemon(&response.results, opts)?;
    }

    if let Some(ref m) = response.metrics {
        print_metrics_stderr(m);
    }

    Ok(())
}

fn run_via_daemon(opts: &SearchOpts, project_path: &std::path::Path) -> Result<()> {
    let request = SearchRequest {
        query: opts.query.clone(),
        max_results: if opts.grep_mode || opts.grep {
            50
        } else {
            opts.max_count
        },
        use_dense: if opts.grep_mode {
            false
        } else {
            !opts.sparse_only
        },
        use_sparse: true,
        use_rerank: if opts.grep_mode { false } else { opts.rerank },
        include_types: opts.include_type.clone(),
        exclude_types: opts.exclude_type.clone(),
        code_only: opts.code_only,
        include_content: opts.verbose
            || opts.grep
            || (opts.json && !opts.no_content)
            || (!opts.json && !opts.grep),
        snippet: !opts.verbose && !opts.json && !opts.grep,
        grep_mode: opts.grep_mode,
        regex_pattern: opts.pattern.clone(),
    };

    let response = semantex_core::server::daemon_search(project_path, &request)?;

    if response.results.is_empty() {
        if opts.grep {
            // grep mode: no output for no results
        } else if opts.json {
            println!("[]");
        } else {
            eprintln!("No results found.");
        }
        return Ok(());
    }

    if opts.grep {
        print_grep_daemon(&response.results)?;
    } else if opts.json {
        print_json_daemon(&response.results)?;
    } else {
        print_results_daemon(&response.results, opts)?;
    }

    if let Some(ref m) = response.metrics {
        print_metrics_stderr(m);
    }

    Ok(())
}

fn run_direct(
    opts: &SearchOpts,
    config: &SemantexConfig,
    project_path: &std::path::Path,
    index_dir: &std::path::Path,
) -> Result<()> {
    // Grep/sparse-only: lightweight path — no ONNX model, returns in <1 s.
    // The dense embedder (~16 MB int8 ONNX) lives only in the daemon, so
    // sparse-only mode is safe to run in-process.
    if opts.grep_mode || opts.sparse_only {
        let searcher = HybridSearcher::open_sparse_only(index_dir, config)
            .context("Failed to open sparse search index")?;
        return run_with_searcher(opts, project_path, &searcher);
    }

    // Dense/hybrid search: route through the daemon so the ONNX model lives in
    // one dedicated process (shared across all concurrent callers).
    // If the daemon is not yet ready we fall back to BM25 sparse results (<1 s)
    // and keep the daemon booting in the background.
    spawn_daemon_and_search_sparse(opts, config, project_path, index_dir)
}

/// Execute a search against an already-opened searcher and print results.
fn run_with_searcher(
    opts: &SearchOpts,
    project_path: &std::path::Path,
    searcher: &HybridSearcher,
) -> Result<()> {
    let effective_max = if opts.grep_mode || opts.grep {
        50
    } else {
        opts.max_count
    };
    let mut query = SearchQuery::new(&opts.query).max_results(effective_max);
    if opts.grep_mode {
        query = query.grep_mode();
    } else {
        if !opts.rerank {
            query = query.no_rerank();
        }
        if opts.dense_only {
            query = query.dense_only();
        }
        if opts.sparse_only {
            query = query.sparse_only();
        }
    }
    if opts.code_only {
        query = query.code_only();
    }
    if !opts.include_type.is_empty() {
        query = query.include_types(opts.include_type.clone());
    }
    if !opts.exclude_type.is_empty() {
        query = query.exclude_types(opts.exclude_type.clone());
    }
    if opts.pattern.is_some() {
        query = query.regex_pattern(opts.pattern.clone());
    }

    let output = searcher.search(&query)?;
    let results = output.results;

    if results.is_empty() {
        if opts.grep {
            // grep mode: no output for no results
        } else if opts.json {
            println!("[]");
        } else {
            eprintln!("No results found.");
        }
        return Ok(());
    }

    if opts.grep {
        print_grep(&results)?;
    } else if opts.json && opts.no_content {
        print_json_no_content(&results)?;
    } else if opts.json && opts.snippet {
        print_json_snippet(&results)?;
    } else if opts.json {
        print_json(&results)?;
    } else {
        print_results(&results, opts, project_path)?;
    }

    print_metrics_stderr(&output.metrics);

    Ok(())
}

/// Spawn the daemon if it is not already running or starting.
///
/// This is fire-and-forget: the daemon starts in a detached background process
/// and this function returns immediately. Multiple callers (e.g. parallel
/// subagents) will each check before spawning, so only one daemon is started.
fn spawn_daemon_if_needed(project_path: &std::path::Path) {
    // Already fully ready — nothing to do.
    if semantex_core::server::daemon_healthy(project_path) {
        return;
    }
    // PID file present and process alive — daemon is loading models, don't pile on.
    if semantex_core::server::daemon_starting(project_path) {
        return;
    }

    let Ok(exe) = std::env::current_exe() else {
        return;
    };
    let _ = std::process::Command::new(&exe)
        .arg("serve")
        .arg(project_path)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn();
}

/// Non-blocking dense search path.
///
/// Spawns the daemon (if absent), waits up to 2 s for it to respond, then falls
/// back to BM25 sparse results if it is still loading. The daemon continues
/// warming in the background so the next search gets full hybrid quality.
fn spawn_daemon_and_search_sparse(
    opts: &SearchOpts,
    config: &SemantexConfig,
    project_path: &std::path::Path,
    index_dir: &std::path::Path,
) -> Result<()> {
    spawn_daemon_if_needed(project_path);

    // Quick poll: if the daemon was already warm (or loads very fast), catch it
    // within 2 s instead of immediately falling back.
    for _ in 0..4 {
        std::thread::sleep(std::time::Duration::from_millis(500));
        if let Ok(port) = semantex_core::server::read_daemon_port(project_path) {
            if run_via_binary_daemon(opts, port).is_ok() {
                return Ok(());
            }
        }
    }

    // Daemon is still loading — serve BM25 results now, full hybrid on next call.
    eprintln!(
        "{}",
        "[daemon warming up — showing keyword results]"
            .yellow()
            .dimmed()
    );
    let searcher = HybridSearcher::open_sparse_only(index_dir, config)
        .context("Failed to open sparse search index")?;
    run_with_searcher(opts, project_path, &searcher)
}

fn print_results(
    results: &[SearchResult],
    opts: &SearchOpts,
    project_path: &std::path::Path,
) -> Result<()> {
    if opts.verbose {
        // Verbose mode: colorized full-content output (old behavior)
        for result in results {
            let file_display = result.chunk.file_path.display().to_string();
            let score = format!("{:.4}", result.score);
            let source = format!("{:?}", result.source);

            if opts.line_number {
                println!(
                    "{}:{}:{} {} [{}]",
                    file_display.green(),
                    result.chunk.start_line.to_string().yellow(),
                    result.chunk.end_line.to_string().yellow(),
                    score.cyan(),
                    source.dimmed(),
                );
            } else {
                println!(
                    "{} {} [{}]",
                    file_display.green(),
                    score.cyan(),
                    source.dimmed(),
                );
            }

            let content = &result.chunk.content;
            let display_content = if content.len() > 2000 {
                format!("{}...", &content[..content.floor_char_boundary(2000)])
            } else {
                content.clone()
            };
            for line in display_content.lines() {
                println!("  {line}");
            }
            println!();

            if opts.context > 0 {
                let full_path = project_path.join(&result.chunk.file_path);
                if let Ok(file_content) = std::fs::read_to_string(&full_path) {
                    let lines: Vec<&str> = file_content.lines().collect();
                    let start = result
                        .chunk
                        .start_line
                        .saturating_sub(opts.context as u32 + 1)
                        as usize;
                    let end = ((result.chunk.end_line as usize) + opts.context).min(lines.len());

                    for (idx, line) in lines[start..end].iter().enumerate() {
                        let line_num = start + idx + 1;
                        let is_match = line_num >= result.chunk.start_line as usize
                            && line_num <= result.chunk.end_line as usize;
                        if is_match {
                            println!("  {} {}", format!("{line_num:>4}").yellow(), line);
                        } else {
                            println!("  {} {}", format!("{line_num:>4}").dimmed(), line.dimmed());
                        }
                    }
                    println!();
                }
            }
        }
    } else {
        // Compact mode: plain-text, 3-line snippets, no ANSI colors
        for result in results {
            let file_display = result.chunk.file_path.display().to_string();

            // Extract name and language from chunk_type
            let (name, lang) = match &result.chunk.chunk_type {
                ChunkType::AstNode { name, language, .. } => {
                    (Some(name.as_str()), Some(language.as_str()))
                }
                _ => (None, None),
            };

            // Header line: file:start-end name (lang) [score]
            let mut header = format!(
                "{}:{}-{}",
                file_display, result.chunk.start_line, result.chunk.end_line
            );
            if let Some(name) = name {
                header.push(' ');
                header.push_str(name);
            }
            if let Some(lang) = lang {
                header.push_str(&format!(" ({lang})"));
            }
            header.push_str(&format!(" [{:.2}]", result.score));
            println!("{header}");

            // Content lines: 2-space indented, max 3 lines
            let snippet = make_snippet(&result.chunk.content, &result.chunk.chunk_type);
            for line in snippet.lines() {
                if line == "..." {
                    println!("  ...");
                } else {
                    println!("  {line}");
                }
            }

            // Blank line between results
            println!();
        }
    }

    Ok(())
}

fn print_json(results: &[SearchResult]) -> Result<()> {
    let json_results: Vec<serde_json::Value> = results
        .iter()
        .map(|r| {
            serde_json::json!({
                "file": r.chunk.file_path.display().to_string(),
                "start_line": r.chunk.start_line,
                "end_line": r.chunk.end_line,
                "score": r.score,
                "source": format!("{:?}", r.source),
                "content": r.chunk.content,
                "chunk_type": r.chunk.chunk_type,
            })
        })
        .collect();

    println!("{}", serde_json::to_string_pretty(&json_results)?);
    Ok(())
}

/// JSON output with content omitted — paths and metadata only.
fn print_json_no_content(results: &[SearchResult]) -> Result<()> {
    let json_results: Vec<serde_json::Value> = results
        .iter()
        .map(|r| {
            serde_json::json!({
                "file": r.chunk.file_path.display().to_string(),
                "start_line": r.chunk.start_line,
                "end_line": r.chunk.end_line,
                "score": r.score,
                "source": format!("{:?}", r.source),
                "chunk_type": r.chunk.chunk_type,
            })
        })
        .collect();

    println!("{}", serde_json::to_string_pretty(&json_results)?);
    Ok(())
}

/// JSON output with content trimmed to a short snippet.
fn print_json_snippet(results: &[SearchResult]) -> Result<()> {
    let json_results: Vec<serde_json::Value> = results
        .iter()
        .map(|r| {
            let snippet = make_snippet(&r.chunk.content, &r.chunk.chunk_type);
            serde_json::json!({
                "file": r.chunk.file_path.display().to_string(),
                "start_line": r.chunk.start_line,
                "end_line": r.chunk.end_line,
                "score": r.score,
                "source": format!("{:?}", r.source),
                "content": snippet,
                "chunk_type": r.chunk.chunk_type,
            })
        })
        .collect();

    println!("{}", serde_json::to_string_pretty(&json_results)?);
    Ok(())
}

/// Grep-like output: one line per result as `file:line: preview`.
fn print_grep(results: &[SearchResult]) -> Result<()> {
    for r in results {
        let preview = r
            .chunk
            .content
            .lines()
            .next()
            .unwrap_or("")
            .chars()
            .take(120)
            .collect::<String>();
        println!(
            "{}:{}: {}",
            r.chunk.file_path.display(),
            r.chunk.start_line,
            preview.trim(),
        );
    }
    Ok(())
}

// --- Metrics printer (shared by direct + daemon paths) ---

fn print_metrics_stderr(m: &semantex_core::search::SearchMetrics) {
    let mut parts = vec![format!("{}ms total", m.total_ms)];
    if let Some(ms) = m.dense_ms {
        parts.push(format!("{ms}ms dense"));
    }
    if let Some(ms) = m.sparse_ms {
        parts.push(format!("{ms}ms sparse"));
    }
    if let Some(ms) = m.rerank_ms {
        parts.push(format!("{ms}ms rerank"));
    }
    parts.push(format!("{} results", m.result_count));
    parts.push(format!("[{}]", m.query_type));
    eprintln!("{}", parts.join(" | "));
}

// --- Daemon response printers ---

fn print_results_daemon(results: &[SearchResultItem], opts: &SearchOpts) -> Result<()> {
    if opts.verbose {
        // Verbose mode: colorized full-content output (old behavior)
        for r in results {
            let file_display = &r.file;
            let score = format!("{:.4}", r.score);
            let source = &r.source;

            if opts.line_number {
                println!(
                    "{}:{}:{} {} [{}]",
                    file_display.green(),
                    r.start_line.to_string().yellow(),
                    r.end_line.to_string().yellow(),
                    score.cyan(),
                    source.dimmed(),
                );
            } else {
                println!(
                    "{} {} [{}]",
                    file_display.green(),
                    score.cyan(),
                    source.dimmed(),
                );
            }

            if let Some(ref content) = r.content {
                let display_content = if content.len() > 2000 {
                    format!("{}...", &content[..content.floor_char_boundary(2000)])
                } else {
                    content.clone()
                };
                for line in display_content.lines() {
                    println!("  {line}");
                }
                println!();
            }
        }
    } else {
        // Compact mode: plain-text, 3-line snippets, no ANSI colors
        for r in results {
            // Header line: file:start-end name (lang) [score]
            let mut header = format!("{}:{}-{}", r.file, r.start_line, r.end_line);
            if let Some(ref name) = r.name {
                header.push(' ');
                header.push_str(name);
            }
            if let Some(ref lang) = r.language {
                header.push_str(&format!(" ({lang})"));
            }
            header.push_str(&format!(" [{:.2}]", r.score));
            println!("{header}");

            // Content lines: 2-space indented, max 3 lines
            if let Some(ref content) = r.content {
                let lines: Vec<&str> = content.lines().take(4).collect();
                let show = if lines.len() > 3 { 3 } else { lines.len() };
                for line in &lines[..show] {
                    println!("  {line}");
                }
                if lines.len() > 3 {
                    println!("  ...");
                }
            }

            // Blank line between results
            println!();
        }
    }

    Ok(())
}

fn print_json_daemon(results: &[SearchResultItem]) -> Result<()> {
    use std::io::{BufWriter, Write};
    let json_results: Vec<serde_json::Value> = results
        .iter()
        .map(|r| {
            let mut val = serde_json::json!({
                "file": r.file,
                "start_line": r.start_line,
                "end_line": r.end_line,
                "score": r.score,
                "source": r.source,
                "chunk_type": r.chunk_type,
            });
            if let Some(ref content) = r.content {
                val["content"] = serde_json::Value::String(content.clone());
            }
            if let Some(ref name) = r.name {
                val["name"] = serde_json::Value::String(name.clone());
            }
            if let Some(ref lang) = r.language {
                val["language"] = serde_json::Value::String(lang.clone());
            }
            val
        })
        .collect();

    let stdout = std::io::stdout();
    let mut out = BufWriter::new(stdout.lock());
    serde_json::to_writer_pretty(&mut out, &json_results)?;
    writeln!(out)?;
    out.flush()?;
    Ok(())
}

fn print_grep_daemon(results: &[SearchResultItem]) -> Result<()> {
    use std::io::{BufWriter, Write};
    let stdout = std::io::stdout();
    let mut out = BufWriter::new(stdout.lock());
    for r in results {
        let preview = r
            .content
            .as_deref()
            .unwrap_or("")
            .lines()
            .next()
            .unwrap_or("")
            .chars()
            .take(120)
            .collect::<String>();
        writeln!(out, "{}:{}: {}", r.file, r.start_line, preview.trim())?;
    }
    out.flush()?;
    Ok(())
}

/// Create a short snippet from chunk content based on chunk type.
fn make_snippet(content: &str, chunk_type: &ChunkType) -> String {
    match chunk_type {
        ChunkType::AstNode { .. } => truncate_lines(content, 3),
        ChunkType::TextWindow { .. } => truncate_lines(content, 3),
        ChunkType::PdfPage { .. } => {
            if content.len() > 100 {
                format!("{}...", &content[..content.floor_char_boundary(100)])
            } else {
                content.to_string()
            }
        }
    }
}

/// Take the first `n` lines; append "..." if truncated. Single pass.
fn truncate_lines(content: &str, n: usize) -> String {
    let lines: Vec<&str> = content.lines().take(n + 1).collect();
    if lines.len() > n {
        let snippet = lines[..n].join("\n");
        format!("{snippet}\n...")
    } else {
        lines.join("\n")
    }
}

/// Walk up from `start` to find the nearest project root.
///
/// First pass: look for an existing `.semantex/meta.json` (existing index).
/// Second pass: look for project root markers (`.git`, `Cargo.toml`, `package.json`, etc.)
/// so that `semantex "query" src/` indexes the project root, not `src/`.
///
/// Returns `None` only if no index and no project root marker is found.
fn find_project_root(start: &Path) -> Option<PathBuf> {
    let base = if start.is_dir() {
        start.to_path_buf()
    } else {
        start.parent()?.to_path_buf()
    };

    // Pass 1: existing index
    let mut dir = base.clone();
    loop {
        if dir.join(".semantex").join("meta.json").exists() {
            return Some(dir);
        }
        if !dir.pop() {
            break;
        }
    }

    // Pass 2: project root markers
    const MARKERS: &[&str] = &[
        ".git",
        "Cargo.toml",
        "package.json",
        "go.mod",
        "pyproject.toml",
        "setup.py",
        "pom.xml",
        "build.gradle",
        "CMakeLists.txt",
        "Makefile",
        ".hg",
    ];
    let mut dir = base;
    loop {
        if MARKERS.iter().any(|m| dir.join(m).exists()) {
            return Some(dir);
        }
        if !dir.pop() {
            break;
        }
    }

    None
}
