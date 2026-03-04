use anyhow::{Context, Result};
use colored::Colorize;
use semantex_core::config::SemantexConfig;
use semantex_core::index::state::{self, IndexState};
use semantex_core::search::SearchQuery;
use semantex_core::search::hybrid::HybridSearcher;
use semantex_core::search::ripgrep_fallback;
use semantex_core::server::protocol::{
    MultiSearchRequest, SearchRequest, SearchResponse, SearchResultItem,
};
use semantex_core::types::{ChunkType, SearchResult};
use std::collections::HashSet;
use std::fmt::Write as _;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

pub struct SearchOpts {
    /// All search queries. len()==1 for single mode; len()>1 enables batch mode (--refs only).
    pub queries: Vec<String>,
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
    pub refs: bool,
    pub peek: bool,
    pub pattern: Option<String>,
    pub deep: bool,
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
    if let Ok(port) = semantex_core::server::read_daemon_port(&project_path)
        && run_via_binary_daemon(opts, port).is_ok()
    {
        return Ok(());
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

    let results = ripgrep_fallback::search(
        opts.queries.first().map_or("", String::as_str),
        project_path,
        max,
    )?;

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
    // Deep mode: send DeepSearchRequest instead of SearchRequest
    if opts.deep {
        let deep_req = semantex_core::server::protocol::DeepSearchRequest {
            query: opts.queries.first().cloned().unwrap_or_default(),
            max_results: 20,
            use_graph: true,
        };
        let response = semantex_core::server::daemon_deep_search_binary(port, deep_req)?;
        print_deep(&response)?;
        if opts.verbose {
            let m = &response.metrics;
            eprintln!(
                "Deep: {}ms total | {}ms search | {}ms triage | {}ms graph | {}ms read | {}ms summarize | {} chunks read",
                m.total_ms,
                m.search_ms,
                m.triage_ms,
                m.graph_ms,
                m.read_ms,
                m.summarize_ms,
                m.chunks_read
            );
        }
        return Ok(());
    }

    let request = SearchRequest {
        query: opts.queries.first().cloned().unwrap_or_default(),
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
        include_content: opts.peek
            || (!opts.no_content || !opts.json || opts.grep || opts.verbose) && !opts.refs,
        snippet: !opts.peek && !opts.refs && !opts.verbose && !opts.json && !opts.grep,
        grep_mode: opts.grep_mode,
        regex_pattern: opts.pattern.clone(),
        // Auto-peek top result when in --refs mode (saves a --peek roundtrip)
        auto_peek_top: opts.refs,
    };

    let t0 = std::time::Instant::now();
    let response = semantex_core::server::daemon_search_binary(port, request)?;
    let t_roundtrip = t0.elapsed();

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

    let t1 = std::time::Instant::now();
    if opts.refs {
        print_refs_daemon(&response)?;
    } else if opts.peek {
        print_peek_daemon(&response)?;
    } else if opts.grep {
        print_grep_daemon(&response.results)?;
    } else if opts.json {
        print_json_daemon(&response.results)?;
    } else {
        print_results_daemon(&response.results, opts)?;
    }
    let t_output = t1.elapsed();

    if let Some(ref m) = response.metrics {
        print_metrics_stderr(m);
    }

    if opts.verbose {
        eprintln!(
            "CLI timing: roundtrip={:?} output={:?} total={:?}",
            t_roundtrip,
            t_output,
            t0.elapsed()
        );
    }

    Ok(())
}

/// Batch mode: run multiple --refs queries in one daemon round trip.
/// Each query is run as a separate SearchRequest inside a MultiSearchRequest.
/// Output: one "## Query:" section per query, each with auto-peek + confidence hint,
/// followed by a single combined hint footer.
pub fn run_batch_via_binary_daemon(
    queries: &[String],
    max_results: usize,
    code_only: bool,
    include_types: &[String],
    exclude_types: &[String],
    port: u16,
) -> Result<()> {
    let batch_requests: Vec<SearchRequest> = queries
        .iter()
        .map(|q| SearchRequest {
            query: q.clone(),
            max_results,
            use_dense: true,
            use_sparse: true,
            use_rerank: false,
            include_types: include_types.to_vec(),
            exclude_types: exclude_types.to_vec(),
            code_only,
            include_content: false, // refs mode
            snippet: false,
            grep_mode: false,
            regex_pattern: None,
            auto_peek_top: true, // auto-peek top result per query
        })
        .collect();

    let multi_req = MultiSearchRequest {
        queries: batch_requests,
    };
    let multi_resp = semantex_core::server::daemon_multi_search_binary(port, multi_req)?;

    let stdout = std::io::stdout();
    let mut out = BufWriter::new(stdout.lock());

    for (query, response) in queries.iter().zip(multi_resp.responses.iter()) {
        writeln!(out, "## Query: \"{query}\"")?;
        // Print refs inline (reuse grouping logic without creating a BufWriter)
        let results = &response.results;
        let mut groups: Vec<(String, Vec<&SearchResultItem>)> = Vec::new();
        for r in results {
            if let Some(g) = groups.iter_mut().find(|(f, _)| f == &r.file) {
                g.1.push(r);
            } else {
                groups.push((r.file.clone(), vec![r]));
            }
        }
        let files: Vec<&str> = groups.iter().map(|(f, _)| f.as_str()).collect();
        let prefix = common_path_prefix(&files);
        let mut auto_peeked = false;

        for (file, items) in &groups {
            let display_file = if prefix.is_empty() {
                file.as_str()
            } else {
                file.strip_prefix(&prefix).unwrap_or(file.as_str())
            };
            writeln!(out, "## {display_file}")?;
            for r in items {
                write!(out, "  :{}-{}", r.start_line, r.end_line)?;
                if let Some(ref name) = r.name {
                    write!(out, " {name}")?;
                }
                if let Some(ref kind) = r.kind {
                    write!(out, " [{kind}]")?;
                }
                writeln!(out)?;
                if let Some(ref summary) = r.summary {
                    for line in summary.lines() {
                        writeln!(out, "    {line}")?;
                    }
                }
                if !auto_peeked && let Some(ref content) = r.content {
                    let lines: Vec<&str> = content.lines().take(6).collect();
                    let show = lines.len().min(5);
                    for line in &lines[..show] {
                        writeln!(out, "    \u{2502} {line}")?;
                    }
                    if content.lines().count() > 5 {
                        writeln!(out, "    \u{2502} ...")?;
                    }
                    auto_peeked = true;
                }
            }
            writeln!(out)?;
        }
        if results.is_empty() {
            writeln!(out, "  (no results)")?;
            writeln!(out)?;
        }
    }

    writeln!(
        out,
        "# Hint: use --peek for code snippets, --around <name> for call graph, or Read file:line for full content"
    )?;
    out.flush()?;
    Ok(())
}

fn run_via_daemon(opts: &SearchOpts, project_path: &std::path::Path) -> Result<()> {
    let request = SearchRequest {
        query: opts.queries.first().cloned().unwrap_or_default(),
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
        include_content: opts.peek
            || (!opts.no_content || !opts.json || opts.grep || opts.verbose) && !opts.refs,
        snippet: !opts.peek && !opts.refs && !opts.verbose && !opts.json && !opts.grep,
        grep_mode: opts.grep_mode,
        regex_pattern: opts.pattern.clone(),
        auto_peek_top: opts.refs,
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

    if opts.refs {
        print_refs_daemon(&response)?;
    } else if opts.peek {
        print_peek_daemon(&response)?;
    } else if opts.grep {
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
    // Deep mode: requires the full hybrid searcher (dense + sparse).
    // Route through the daemon so the ONNX model stays in one process.
    // If no daemon is available, fall back to a sparse-only deep search.
    if opts.deep {
        let searcher = HybridSearcher::open_sparse_only(index_dir, config)
            .context("Failed to open search index for deep search")?;
        let result = semantex_core::search::deep::deep_search(
            &searcher,
            opts.queries.first().map_or("", String::as_str),
            20,
            true,
        )?;
        let response = semantex_core::server::protocol::DeepSearchResponse {
            answer: result.answer,
            sources: result
                .sources
                .into_iter()
                .map(|s| semantex_core::server::protocol::DeepSearchSource {
                    file: s.file,
                    start_line: s.start_line,
                    end_line: s.end_line,
                    name: s.name,
                    kind: s.kind,
                })
                .collect(),
            metrics: semantex_core::server::protocol::DeepResponseMetrics {
                search_ms: result.metrics.search_ms,
                triage_ms: result.metrics.triage_ms,
                graph_ms: result.metrics.graph_ms,
                read_ms: result.metrics.read_ms,
                summarize_ms: result.metrics.summarize_ms,
                total_ms: result.metrics.total_ms,
                chunks_searched: result.metrics.chunks_searched,
                chunks_read: result.metrics.chunks_read,
            },
        };
        if opts.verbose {
            let m = &response.metrics;
            eprintln!(
                "Deep: {}ms total | {}ms search | {}ms triage | {}ms graph | {}ms read | {}ms summarize | {} chunks read",
                m.total_ms,
                m.search_ms,
                m.triage_ms,
                m.graph_ms,
                m.read_ms,
                m.summarize_ms,
                m.chunks_read
            );
        }
        return print_deep(&response);
    }

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
    let primary_query = opts.queries.first().map_or("", String::as_str);
    let mut query = SearchQuery::new(primary_query).max_results(effective_max);
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

    if opts.refs {
        print_refs(&results)?;
    } else if opts.peek {
        print_peek(&results)?;
    } else if opts.grep {
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

    let _ = std::process::Command::new("semantex")
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
        if let Ok(port) = semantex_core::server::read_daemon_port(project_path)
            && run_via_binary_daemon(opts, port).is_ok()
        {
            return Ok(());
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
                let _ = write!(header, " ({lang})");
            }
            let _ = write!(header, " [{:.2}]", result.score);
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

/// Refs output: grouped by file, with kind/summary, path prefix stripped, hint appended.
fn print_refs(results: &[SearchResult]) -> Result<()> {
    let stdout = std::io::stdout();
    let mut out = BufWriter::new(stdout.lock());

    // Group by file, preserving relevance order
    let mut groups: Vec<(String, Vec<&SearchResult>)> = Vec::new();
    for r in results {
        let file = r.chunk.file_path.display().to_string();
        if let Some(g) = groups.iter_mut().find(|(f, _)| f == &file) {
            g.1.push(r);
        } else {
            groups.push((file, vec![r]));
        }
    }

    // Compute common prefix to strip
    let files: Vec<&str> = groups.iter().map(|(f, _)| f.as_str()).collect();
    let prefix = common_path_prefix(&files);

    for (file, items) in &groups {
        let display_file = if prefix.is_empty() {
            file.as_str()
        } else {
            file.strip_prefix(&prefix).unwrap_or(file.as_str())
        };
        writeln!(out, "## {display_file}")?;
        for r in items {
            write!(out, "  :{}-{}", r.chunk.start_line, r.chunk.end_line)?;
            if let ChunkType::AstNode {
                ref name,
                ref kind,
                ref structured_meta,
                ..
            } = r.chunk.chunk_type
            {
                write!(out, " {name} [{kind}]")?;
                writeln!(out)?;
                if let Some(meta) = structured_meta {
                    let summary = meta.display_summary();
                    for line in summary.lines() {
                        writeln!(out, "    {line}")?;
                    }
                }
            } else {
                writeln!(out)?;
            }
        }
        writeln!(out)?;
    }

    writeln!(
        out,
        "# Hint: use --peek for code snippets, --around <name> for call graph, or Read file:line for full content"
    )?;
    out.flush()?;
    Ok(())
}

/// Refs output from daemon results: grouped by file, with kind/summary, path prefix stripped.
/// Top result gets a 5-line auto-peek when `auto_peek_top` was set on the request.
/// Confidence hint appended to footer.
fn print_refs_daemon(response: &SearchResponse) -> Result<()> {
    let results = &response.results;
    let stdout = std::io::stdout();
    let mut out = BufWriter::new(stdout.lock());

    // Group by file, preserving relevance order
    let mut groups: Vec<(String, Vec<&SearchResultItem>)> = Vec::new();
    for r in results {
        if let Some(g) = groups.iter_mut().find(|(f, _)| f == &r.file) {
            g.1.push(r);
        } else {
            groups.push((r.file.clone(), vec![r]));
        }
    }

    // Compute common prefix to strip
    let files: Vec<&str> = groups.iter().map(|(f, _)| f.as_str()).collect();
    let prefix = common_path_prefix(&files);

    // Track whether we've printed the auto-peeked top result yet
    let mut auto_peeked = false;

    for (file, items) in &groups {
        let display_file = if prefix.is_empty() {
            file.as_str()
        } else {
            file.strip_prefix(&prefix).unwrap_or(file.as_str())
        };
        writeln!(out, "## {display_file}")?;
        for r in items {
            write!(out, "  :{}-{}", r.start_line, r.end_line)?;
            if let Some(ref name) = r.name {
                write!(out, " {name}")?;
            }
            if let Some(ref kind) = r.kind {
                write!(out, " [{kind}]")?;
            }
            writeln!(out)?;
            if let Some(ref summary) = r.summary {
                for line in summary.lines() {
                    writeln!(out, "    {line}")?;
                }
            }
            // Auto-peek: print code preview for top-1 result only
            if !auto_peeked && let Some(ref content) = r.content {
                let lines: Vec<&str> = content.lines().take(6).collect();
                let show = lines.len().min(5);
                for line in &lines[..show] {
                    writeln!(out, "    \u{2502} {line}")?;
                }
                if content.lines().count() > 5 {
                    writeln!(out, "    \u{2502} ...")?;
                }
                auto_peeked = true;
            }
        }
        writeln!(out)?;
    }

    // Confidence hint footer
    let hint = match response.confidence.as_deref() {
        Some("high") => {
            "# [high confidence] Summaries above are sufficient — Read file:start-end only if you need exact code"
        }
        Some("medium") => {
            "# [medium confidence] Use --around <name> for call graph, --deep \"question\" for auto-summarized answer"
        }
        Some("low") => {
            "# [low confidence] Rephrase query, try --deep \"question\", or use Grep for exact match"
        }
        _ => {
            "# Hint: --around <name> for call graph, --deep \"question\" for auto-summarized answer, Read file:line for exact code"
        }
    };
    writeln!(out, "{hint}")?;
    out.flush()?;
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
                let _ = write!(header, " ({lang})");
            }
            let _ = write!(header, " [{:.2}]", r.score);
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
        ChunkType::AstNode { .. } | ChunkType::TextWindow { .. } => truncate_lines(content, 3),
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

/// Compute the longest common directory prefix across a list of file paths.
/// Only strips if the prefix has ≥2 path components (avoids stripping "src/").
/// Returns empty string if no suitable prefix found.
fn common_path_prefix(files: &[&str]) -> String {
    if files.len() <= 1 {
        return String::new();
    }

    // Split each file into directory components (exclude filename)
    let dir_parts: Vec<Vec<&str>> = files
        .iter()
        .map(|f| {
            let parts: Vec<&str> = f.split('/').collect();
            // All except the last component (filename)
            if parts.len() > 1 {
                parts[..parts.len() - 1].to_vec()
            } else {
                vec![]
            }
        })
        .collect();

    // Find common prefix length
    let min_len = dir_parts.iter().map(Vec::len).min().unwrap_or(0);
    let mut common_len = 0;
    for i in 0..min_len {
        if dir_parts.iter().all(|p| p[i] == dir_parts[0][i]) {
            common_len += 1;
        } else {
            break;
        }
    }

    // Only strip if ≥2 components
    if common_len < 2 {
        return String::new();
    }

    let prefix = dir_parts[0][..common_len].join("/");
    format!("{prefix}/")
}

/// Peek output: refs grouped by file + first 5 lines of code per result.
fn print_peek(results: &[SearchResult]) -> Result<()> {
    let stdout = std::io::stdout();
    let mut out = BufWriter::new(stdout.lock());

    // Group by file, preserving relevance order
    let mut groups: Vec<(String, Vec<&SearchResult>)> = Vec::new();
    for r in results {
        let file = r.chunk.file_path.display().to_string();
        if let Some(g) = groups.iter_mut().find(|(f, _)| f == &file) {
            g.1.push(r);
        } else {
            groups.push((file, vec![r]));
        }
    }

    let files: Vec<&str> = groups.iter().map(|(f, _)| f.as_str()).collect();
    let prefix = common_path_prefix(&files);

    for (file, items) in &groups {
        let display_file = if prefix.is_empty() {
            file.as_str()
        } else {
            file.strip_prefix(&prefix).unwrap_or(file.as_str())
        };
        writeln!(out, "## {display_file}")?;
        for r in items {
            // Refs line
            write!(out, "  :{}-{}", r.chunk.start_line, r.chunk.end_line)?;
            if let ChunkType::AstNode {
                ref name,
                ref kind,
                ref structured_meta,
                ..
            } = r.chunk.chunk_type
            {
                write!(out, " {name} [{kind}]")?;
                writeln!(out)?;
                if let Some(meta) = structured_meta {
                    let summary = meta.display_summary();
                    for line in summary.lines() {
                        writeln!(out, "    {line}")?;
                    }
                }
            } else {
                writeln!(out)?;
            }
            // Code preview: first 5 lines with "    │ " prefix
            let lines: Vec<&str> = r.chunk.content.lines().take(6).collect();
            let show = lines.len().min(5);
            for line in &lines[..show] {
                writeln!(out, "    \u{2502} {line}")?;
            }
            if r.chunk.content.lines().count() > 5 {
                writeln!(out, "    \u{2502} ...")?;
            }
        }
        writeln!(out)?;
    }

    writeln!(
        out,
        "# Hint: use --around <name> for call graph, or Read file:line for full content"
    )?;
    out.flush()?;
    Ok(())
}

/// Peek output from daemon results: refs + first 5 lines of code per result.
/// Confidence hint shown only when low (high/medium: show --around hint instead).
fn print_peek_daemon(response: &SearchResponse) -> Result<()> {
    let results = &response.results;
    let stdout = std::io::stdout();
    let mut out = BufWriter::new(stdout.lock());

    // Group by file, preserving relevance order
    let mut groups: Vec<(String, Vec<&SearchResultItem>)> = Vec::new();
    for r in results {
        if let Some(g) = groups.iter_mut().find(|(f, _)| f == &r.file) {
            g.1.push(r);
        } else {
            groups.push((r.file.clone(), vec![r]));
        }
    }

    let files: Vec<&str> = groups.iter().map(|(f, _)| f.as_str()).collect();
    let prefix = common_path_prefix(&files);

    for (file, items) in &groups {
        let display_file = if prefix.is_empty() {
            file.as_str()
        } else {
            file.strip_prefix(&prefix).unwrap_or(file.as_str())
        };
        writeln!(out, "## {display_file}")?;
        for r in items {
            // Refs line
            write!(out, "  :{}-{}", r.start_line, r.end_line)?;
            if let Some(ref name) = r.name {
                write!(out, " {name}")?;
            }
            if let Some(ref kind) = r.kind {
                write!(out, " [{kind}]")?;
            }
            writeln!(out)?;
            if let Some(ref summary) = r.summary {
                for line in summary.lines() {
                    writeln!(out, "    {line}")?;
                }
            }
            // Code preview: first 5 lines with "    │ " prefix
            if let Some(ref content) = r.content {
                let lines: Vec<&str> = content.lines().take(6).collect();
                let show = lines.len().min(5);
                for line in &lines[..show] {
                    writeln!(out, "    \u{2502} {line}")?;
                }
                if content.lines().count() > 5 {
                    writeln!(out, "    \u{2502} ...")?;
                }
            }
        }
        writeln!(out)?;
    }

    // Show low-confidence warning; otherwise show --around hint
    let hint = if response.confidence.as_deref() == Some("low") {
        "# [low confidence] Results may not match your intent — try rephrasing or use Grep"
    } else {
        "# Hint: use --around <name> for call graph, or Read file:line for full content"
    };
    writeln!(out, "{hint}")?;
    out.flush()?;
    Ok(())
}

/// Format and print a DeepSearchResponse.
fn print_deep(resp: &semantex_core::server::protocol::DeepSearchResponse) -> Result<()> {
    let stdout = std::io::stdout();
    let mut out = BufWriter::new(stdout.lock());
    writeln!(out, "<answer>")?;
    writeln!(out, "{}", resp.answer)?;
    writeln!(out, "</answer>")?;
    writeln!(out)?;
    writeln!(out, "Sources:")?;
    let mut unique_files: HashSet<&str> = HashSet::new();
    for s in &resp.sources {
        write!(out, "  {}:{}-{}", s.file, s.start_line, s.end_line)?;
        if let Some(ref name) = s.name {
            write!(out, " {name}")?;
        }
        if let Some(ref kind) = s.kind {
            write!(out, " [{kind}]")?;
        }
        writeln!(out)?;
        unique_files.insert(&s.file);
    }
    writeln!(out)?;
    writeln!(
        out,
        "[Complete: {} chunks read across {} files — no further Read calls needed]",
        resp.metrics.chunks_read,
        unique_files.len()
    )?;
    out.flush()?;
    Ok(())
}

/// Run a deep search via daemon (called from the fast path in main.rs).
pub fn run_deep_via_binary_daemon(query: &str, port: u16, verbose: bool) -> Result<()> {
    let request = semantex_core::server::protocol::DeepSearchRequest {
        query: query.to_string(),
        max_results: 20,
        use_graph: true,
    };
    let response = semantex_core::server::daemon_deep_search_binary(port, request)?;
    print_deep(&response)?;
    if verbose {
        let m = &response.metrics;
        eprintln!(
            "Deep: {}ms total | {}ms search | {}ms triage | {}ms graph | {}ms read | {}ms summarize | {} chunks read",
            m.total_ms,
            m.search_ms,
            m.triage_ms,
            m.graph_ms,
            m.read_ms,
            m.summarize_ms,
            m.chunks_read
        );
    }
    Ok(())
}

/// Run graph walk via a running daemon (called from the fast path in main.rs).
pub fn run_around_daemon(symbol: &str, port: u16) -> Result<()> {
    let request = semantex_core::server::protocol::GraphWalkRequest {
        symbol: symbol.to_string(),
    };
    let response = semantex_core::server::daemon_graph_walk_binary(port, request)?;
    print_around_response(symbol, &response)
}

/// Run graph walk — tries daemon first, falls back to direct SQLite.
pub fn run_around(symbol: &str, path: &Path, config: &SemantexConfig) -> Result<()> {
    let target_path = path
        .canonicalize()
        .with_context(|| format!("Invalid path: {}", path.display()))?;
    let project_path = find_project_root(&target_path).unwrap_or_else(|| target_path.clone());

    // Try daemon first
    if let Ok(port) = semantex_core::server::read_daemon_port(&project_path) {
        let request = semantex_core::server::protocol::GraphWalkRequest {
            symbol: symbol.to_string(),
        };
        if let Ok(response) = semantex_core::server::daemon_graph_walk_binary(port, request) {
            return print_around_response(symbol, &response);
        }
    }

    // Fallback: direct SQLite (no daemon needed)
    let index_dir = SemantexConfig::project_index_dir(&project_path);
    let response = semantex_core::server::graph_walk_direct(symbol, &index_dir, config)?;
    print_around_response(symbol, &response)
}

/// Write a single graph walk item, returning true if it was new (not a duplicate).
fn write_graph_item(
    out: &mut BufWriter<std::io::StdoutLock<'_>>,
    r: &SearchResultItem,
    seen: &mut HashSet<(String, u32, u32)>,
) -> std::io::Result<bool> {
    let key = (r.file.clone(), r.start_line, r.end_line);
    if !seen.insert(key) {
        return Ok(false);
    }
    write!(out, "  {}:{}-{}", r.file, r.start_line, r.end_line)?;
    if let Some(ref name) = r.name {
        write!(out, " {name}")?;
    }
    if let Some(ref kind) = r.kind {
        write!(out, " [{kind}]")?;
    }
    writeln!(out)?;
    if let Some(ref summary) = r.summary {
        for line in summary.lines() {
            writeln!(out, "    {line}")?;
        }
    }
    Ok(true)
}

/// Write a graph walk section, skipping duplicates already in `seen`.
fn write_graph_section(
    out: &mut BufWriter<std::io::StdoutLock<'_>>,
    heading: &str,
    items: &[SearchResultItem],
    seen: &mut HashSet<(String, u32, u32)>,
) -> std::io::Result<()> {
    if items.is_empty() {
        return Ok(());
    }
    // Pre-scan to count unique items and collect their output
    let mut item_outputs: Vec<String> = Vec::new();
    for r in items {
        let key = (r.file.clone(), r.start_line, r.end_line);
        if !seen.insert(key) {
            continue;
        }
        let mut item_str = format!("  {}:{}-{}", r.file, r.start_line, r.end_line);
        if let Some(ref name) = r.name {
            let _ = write!(item_str, " {name}");
        }
        if let Some(ref kind) = r.kind {
            let _ = write!(item_str, " [{kind}]");
        }
        if let Some(ref summary) = r.summary {
            for line in summary.lines() {
                let _ = write!(item_str, "\n    {line}");
            }
        }
        item_outputs.push(item_str);
    }
    if !item_outputs.is_empty() {
        writeln!(out, "{heading} ({})", item_outputs.len())?;
        for item in &item_outputs {
            writeln!(out, "{item}")?;
        }
        writeln!(out)?;
    }
    Ok(())
}

/// Format and print a GraphWalkResponse with deduplication across sections.
fn print_around_response(
    symbol: &str,
    resp: &semantex_core::server::protocol::GraphWalkResponse,
) -> Result<()> {
    let stdout = std::io::stdout();
    let mut out = BufWriter::new(stdout.lock());

    if resp.target.is_empty() {
        writeln!(out, "# Symbol '{symbol}' not found in index")?;
        out.flush()?;
        return Ok(());
    }

    // Deduplicate across all sections by (file, start_line, end_line)
    let mut seen: HashSet<(String, u32, u32)> = HashSet::new();

    // Target section (always shown, seeds the seen set)
    writeln!(out, "## Target: {symbol}")?;
    for r in &resp.target {
        write_graph_item(&mut out, r, &mut seen)?;
    }
    writeln!(out)?;

    // Remaining sections — duplicates silently skipped
    write_graph_section(&mut out, "## Callers", &resp.callers, &mut seen)?;
    write_graph_section(&mut out, "## Callees", &resp.callees, &mut seen)?;
    write_graph_section(&mut out, "## Type References", &resp.type_refs, &mut seen)?;
    write_graph_section(&mut out, "## Hierarchy", &resp.hierarchy, &mut seen)?;

    writeln!(
        out,
        "# Hint: use --peek for code snippets, or Read file:line for full content"
    )?;
    out.flush()?;
    Ok(())
}

/// Walk up from `start` to find the nearest project root.
///
/// First pass: look for an existing `.semantex/meta.json` (existing index).
/// Second pass: look for project root markers (`.git`, `Cargo.toml`, `package.json`, etc.)
/// so that `semantex "query" src/` indexes the project root, not `src/`.
///
/// Returns `None` only if no index and no project root marker is found.
fn find_project_root(start: &Path) -> Option<PathBuf> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_common_path_prefix_same_dir() {
        let prefix = common_path_prefix(&["a/b/c.rs", "a/b/d.rs"]);
        assert_eq!(prefix, "a/b/");
    }

    #[test]
    fn test_common_path_prefix_no_common() {
        let prefix = common_path_prefix(&["a/b.rs", "c/d.rs"]);
        assert_eq!(prefix, "");
    }

    #[test]
    fn test_common_path_prefix_single_file() {
        let prefix = common_path_prefix(&["a/b/c.rs"]);
        assert_eq!(prefix, "");
    }

    #[test]
    fn test_common_path_prefix_requires_two_components() {
        // Only one dir component in common — don't strip "src/"
        let prefix = common_path_prefix(&["src/main.rs", "src/lib.rs"]);
        assert_eq!(prefix, "");
    }

    #[test]
    fn test_common_path_prefix_deep_common() {
        let prefix = common_path_prefix(&[
            "crates/semantex-core/src/a.rs",
            "crates/semantex-core/src/b.rs",
            "crates/semantex-core/src/c.rs",
        ]);
        assert_eq!(prefix, "crates/semantex-core/src/");
    }

    #[test]
    fn test_common_path_prefix_partial_match() {
        // Three files where only top-level dir matches — but that's only 1 component
        let prefix = common_path_prefix(&["crates/foo/src/a.rs", "crates/bar/src/b.rs"]);
        // "crates/" is only 1 component — should not be stripped
        assert_eq!(prefix, "");
    }
}
