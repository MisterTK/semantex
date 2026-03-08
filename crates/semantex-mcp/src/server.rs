use crate::protocol::{
    InitializeResult, JsonRpcRequest, JsonRpcResponse, LogLevel, PROTOCOL_VERSION,
    ServerCapabilities, ServerInfo, Tool, ToolAnnotations, ToolCallResult, ToolContent,
    ToolsCapability,
};
use anyhow::{Context, Result};
use parking_lot::Mutex;
use semantex_core::config::SemantexConfig;
use semantex_core::index::builder::IndexBuilder;
use semantex_core::index::state::{self, IndexState};
use semantex_core::search::SearchQuery;
use semantex_core::search::deep as deep_search;
use semantex_core::search::hybrid::HybridSearcher;
use semantex_core::search::ripgrep_fallback;
use std::collections::HashMap;
use std::io::{BufRead, BufWriter, Write as _};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

// =============================================================================
// Notification writer — sends JSON-RPC notifications to stdout during execution
// =============================================================================

/// Writes JSON-RPC 2.0 notifications to the shared stdout.
///
/// Used for progress notifications during long-running tool calls and for
/// logging messages. Thread-safe (wraps `Arc<Mutex<BufWriter<Stdout>>>`).
#[derive(Clone)]
struct NotificationWriter {
    stdout: Arc<Mutex<BufWriter<std::io::Stdout>>>,
}

impl NotificationWriter {
    /// Send a `notifications/progress` to the client.
    fn send_progress(
        &self,
        token: &serde_json::Value,
        progress: f64,
        total: Option<f64>,
        message: Option<&str>,
    ) {
        let mut params = serde_json::json!({
            "progressToken": token,
            "progress": progress,
        });
        if let Some(t) = total {
            params["total"] = serde_json::json!(t);
        }
        if let Some(m) = message {
            params["message"] = serde_json::json!(m);
        }
        self.send_notification("notifications/progress", &params);
    }

    /// Send a `notifications/message` (MCP logging) to the client.
    fn send_log(&self, level: LogLevel, logger: &str, data: &serde_json::Value) {
        let params = serde_json::json!({
            "level": level,
            "logger": logger,
            "data": data,
        });
        self.send_notification("notifications/message", &params);
    }

    fn send_notification(&self, method: &str, params: &serde_json::Value) {
        let notification = serde_json::json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        });
        if let Ok(json) = serde_json::to_string(&notification) {
            let mut out = self.stdout.lock();
            let _ = writeln!(out, "{json}");
            let _ = out.flush();
        }
    }
}

// =============================================================================
// Searcher cache
// =============================================================================

/// Cached searcher entry with LRU tracking
struct CachedSearcher {
    searcher: HybridSearcher,
    last_used: Mutex<Instant>,
}

/// Tracks the indexing status for a project path.
#[derive(Debug, Clone)]
#[allow(dead_code)] // Ready and Failed are set by background threads, read for double-spawn guard
enum IndexingStatus {
    Building,
    Ready,
    Failed(String),
}

/// Idle eviction timeout: entries unused for this duration are dropped
const IDLE_TIMEOUT_SECS: u64 = 600; // 10 minutes

// =============================================================================
// MCP Server
// =============================================================================

pub struct McpServer {
    config: SemantexConfig,
    /// LRU cache of `HybridSearcher` instances keyed by canonical index directory.
    cache: Mutex<HashMap<PathBuf, Arc<CachedSearcher>>>,
    max_cached: usize,
    /// Per-project indexing status — prevents double-spawning background index builds.
    index_states: Arc<Mutex<HashMap<PathBuf, IndexingStatus>>>,
    /// Current MCP logging level — only messages at or above this level are sent.
    log_level: Mutex<LogLevel>,
}

impl McpServer {
    pub fn new(config: SemantexConfig) -> Self {
        let max_cached = std::env::var("SEMANTEX_MCP_CACHE_SIZE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(5);
        Self {
            config,
            cache: Mutex::new(HashMap::new()),
            max_cached,
            index_states: Arc::new(Mutex::new(HashMap::new())),
            log_level: Mutex::new(LogLevel::default()),
        }
    }

    // -------------------------------------------------------------------------
    // Searcher cache
    // -------------------------------------------------------------------------

    fn get_searcher(&self, index_dir: &std::path::Path) -> Result<Arc<CachedSearcher>> {
        let mut cache = self.cache.lock();
        let now = Instant::now();
        cache.retain(|_, entry| {
            now.duration_since(*entry.last_used.lock()).as_secs() < IDLE_TIMEOUT_SECS
        });

        if let Some(entry) = cache.get(index_dir) {
            *entry.last_used.lock() = now;
            return Ok(Arc::clone(entry));
        }

        let searcher = HybridSearcher::open(index_dir, &self.config)?;
        let entry = Arc::new(CachedSearcher {
            searcher,
            last_used: Mutex::new(now),
        });

        if cache.len() >= self.max_cached
            && let Some(lru_key) = cache
                .iter()
                .min_by_key(|(_, v)| *v.last_used.lock())
                .map(|(k, _)| k.clone())
        {
            cache.remove(&lru_key);
        }

        cache.insert(index_dir.to_path_buf(), Arc::clone(&entry));
        Ok(entry)
    }

    fn invalidate_cache(&self, index_dir: &std::path::Path) {
        self.cache.lock().remove(index_dir);
    }

    // -------------------------------------------------------------------------
    // Main I/O loop
    // -------------------------------------------------------------------------

    /// Run the MCP server on stdin/stdout.
    pub fn run(&self) -> Result<()> {
        let stdin = std::io::stdin();
        let stdout = Arc::new(Mutex::new(BufWriter::new(std::io::stdout())));
        let writer = NotificationWriter {
            stdout: Arc::clone(&stdout),
        };

        for line in stdin.lock().lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }

            let request: JsonRpcRequest = match serde_json::from_str(&line) {
                Ok(req) => req,
                Err(e) => {
                    let resp = JsonRpcResponse::error(None, -32700, format!("Parse error: {e}"));
                    let mut out = stdout.lock();
                    writeln!(out, "{}", serde_json::to_string(&resp)?)?;
                    out.flush()?;
                    continue;
                }
            };

            // JSON-RPC 2.0: Notifications (no id) MUST NOT receive a response.
            if request.is_notification() {
                self.handle_notification(&request);
                continue;
            }

            let response = self.handle_request(&request, &writer);
            let mut out = stdout.lock();
            writeln!(out, "{}", serde_json::to_string(&response)?)?;
            out.flush()?;
        }

        Ok(())
    }

    // -------------------------------------------------------------------------
    // Notification handling (no response)
    // -------------------------------------------------------------------------

    #[allow(clippy::unused_self)]
    fn handle_notification(&self, req: &JsonRpcRequest) {
        match req.method.as_str() {
            "notifications/initialized" => {
                tracing::debug!("Client initialized");
            }
            "notifications/cancelled" => {
                // We don't support cancellation yet, but silently acknowledge.
                if let Some(request_id) = req.params.get("requestId") {
                    tracing::debug!(?request_id, "Client cancelled request (not supported)");
                }
            }
            "notifications/roots/list_changed" => {
                tracing::debug!("Client roots changed (not used)");
            }
            _ => {
                // Per JSON-RPC 2.0: unknown notifications are silently ignored.
                tracing::trace!(method = %req.method, "Ignoring unknown notification");
            }
        }
    }

    // -------------------------------------------------------------------------
    // Request dispatch
    // -------------------------------------------------------------------------

    fn handle_request(&self, req: &JsonRpcRequest, writer: &NotificationWriter) -> JsonRpcResponse {
        match req.method.as_str() {
            "initialize" => self.handle_initialize(req.id.clone()),
            "ping" => JsonRpcResponse::success(req.id.clone(), serde_json::json!({})),
            "tools/list" => self.handle_tools_list(req.id.clone()),
            "tools/call" => self.handle_tool_call(req.id.clone(), &req.params, writer),
            "logging/setLevel" => self.handle_set_log_level(req.id.clone(), &req.params),
            _ => JsonRpcResponse::error(
                req.id.clone(),
                -32601,
                format!("Method not found: {}", req.method),
            ),
        }
    }

    // -------------------------------------------------------------------------
    // initialize
    // -------------------------------------------------------------------------

    fn handle_initialize(&self, id: Option<serde_json::Value>) -> JsonRpcResponse {
        // Auto-index cwd if no index exists and it looks like a project directory
        if let Ok(cwd) = std::env::current_dir() {
            let looks_like_project = [
                ".git",
                "Cargo.toml",
                "package.json",
                "go.mod",
                "pyproject.toml",
                "setup.py",
                "Makefile",
                "CMakeLists.txt",
                "pom.xml",
                "build.gradle",
            ]
            .iter()
            .any(|marker| cwd.join(marker).exists());

            if looks_like_project {
                let idx_state = state::detect(&cwd);
                if idx_state == IndexState::NotIndexed || idx_state == IndexState::Stale {
                    self.spawn_background_index(&cwd);
                }
            }
        }

        let result = InitializeResult {
            protocol_version: PROTOCOL_VERSION.into(),
            capabilities: ServerCapabilities {
                tools: Some(ToolsCapability {
                    list_changed: false,
                }),
                logging: Some(serde_json::json!({})),
            },
            server_info: ServerInfo {
                name: "semantex".into(),
                version: env!("CARGO_PKG_VERSION").into(),
            },
            instructions: Some(concat!(
                "IMPORTANT: Use semantex_agent as the DEFAULT tool for all code search queries. ",
                "It auto-classifies your query and selects the optimal search strategy with fallbacks. ",
                "One call in, one answer out.\n\n",
                "semantex_agent — default for all queries. Auto-routes to the best strategy ",
                "(semantic search, deep search, graph walk, exact symbol, regex, or file glob). ",
                "Returns pre-formatted text. Use this unless you need structured JSON output.\n\n",
                "semantex_search — structured JSON results only. Use when you need raw SearchResultItem ",
                "data for programmatic processing. For human-readable results, use semantex_agent instead.\n\n",
                "semantex_deep — structured JSON results only. Use when you need raw DeepSearchResponse ",
                "data. For human-readable results, use semantex_agent instead.\n\n",
                "Fall back to Grep ONLY for regex patterns, Glob ONLY for file name patterns.\n\n",
                "All tools are read-only and safe without user confirmation. ",
                "Auto-indexes on first use; returns keyword results while index builds."
            ).into()),
        };
        JsonRpcResponse::success(
            id,
            serde_json::to_value(result).expect("InitializeResult serialization"),
        )
    }

    // -------------------------------------------------------------------------
    // logging/setLevel
    // -------------------------------------------------------------------------

    fn handle_set_log_level(
        &self,
        id: Option<serde_json::Value>,
        params: &serde_json::Value,
    ) -> JsonRpcResponse {
        if let Some(level_str) = params.get("level").and_then(|v| v.as_str())
            && let Ok(level) =
                serde_json::from_value::<LogLevel>(serde_json::Value::String(level_str.to_string()))
        {
            *self.log_level.lock() = level;
            tracing::info!(?level, "MCP log level updated");
            return JsonRpcResponse::success(id, serde_json::json!({}));
        }
        JsonRpcResponse::error(id, -32602, "Invalid log level".into())
    }

    // -------------------------------------------------------------------------
    // Background indexing
    // -------------------------------------------------------------------------

    fn spawn_background_index(&self, path: &std::path::Path) {
        let canonical = match path.canonicalize() {
            Ok(p) => p,
            Err(_) => path.to_path_buf(),
        };

        {
            let mut states = self.index_states.lock();
            if matches!(states.get(&canonical), Some(IndexingStatus::Building)) {
                return;
            }
            states.insert(canonical.clone(), IndexingStatus::Building);
        }

        let config = self.config.clone();
        let states = Arc::clone(&self.index_states);
        std::thread::spawn(move || {
            tracing::info!(path = %canonical.display(), "Background indexing started");
            match IndexBuilder::new(&config).and_then(|b| b.build(&canonical)) {
                Ok(stats) => {
                    tracing::info!(
                        files = stats.files_indexed,
                        chunks = stats.chunks_created,
                        secs = stats.duration.as_secs_f64(),
                        "Background indexing completed"
                    );
                    states.lock().insert(canonical, IndexingStatus::Ready);
                }
                Err(e) => {
                    tracing::error!(err = %e, "Background indexing failed");
                    states
                        .lock()
                        .insert(canonical, IndexingStatus::Failed(e.to_string()));
                }
            }
        });
    }

    // -------------------------------------------------------------------------
    // tools/list — all tool definitions with annotations + output schemas
    // -------------------------------------------------------------------------

    #[allow(clippy::unused_self, clippy::too_many_lines)]
    fn handle_tools_list(&self, id: Option<serde_json::Value>) -> JsonRpcResponse {
        let tools = vec![
            Tool {
                name: "semantex_agent".into(),
                title: Some("Intelligent Code Search".into()),
                description: concat!(
                    "Intelligent code search with automatic query classification. ",
                    "Analyzes your query, selects the best search strategy (semantic, exact symbol, ",
                    "graph walk, deep search, regex, file pattern), executes with fallbacks, and returns ",
                    "a formatted answer. This is the recommended default — use it for all code search queries. ",
                    "Returns pre-formatted text ready for direct use."
                ).into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language question, code symbol, regex pattern, or glob pattern. Use instead of 'queries' for a single query."
                        },
                        "queries": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Multiple queries to run in one call. Results are merged. Use instead of 'query' when you need to search for 2-3 related concepts at once."
                        },
                        "path": {
                            "type": "string",
                            "description": "Project path (default: current directory)"
                        },
                        "full_code": {
                            "type": "boolean",
                            "description": "Include full source code blocks in response (default: false)"
                        },
                        "budget": {
                            "type": "integer",
                            "description": "Response size budget in bytes (default: 12000, ~3K tokens)"
                        },
                        "depth": {
                            "type": "string",
                            "enum": ["quick", "search", "deep"],
                            "description": "Search depth. 'quick'=symbol lookup (~50ms), 'search'=hybrid with snippets (~100ms), 'deep'=full implementations with call graphs (~200ms). Omit to auto-detect."
                        },
                        "focus": {
                            "type": "string",
                            "enum": ["implementation", "callers", "signatures", "patterns"],
                            "description": "What to emphasize in results. 'implementation'=full code bodies, 'callers'=who calls these functions, 'signatures'=function signatures only, 'patterns'=usage examples."
                        }
                    }
                }),
                output_schema: None,
                annotations: Some(ToolAnnotations::read_only("Intelligent Code Search")),
            },
            Tool {
                name: "semantex_search".into(),
                title: Some("Semantic Code Search".into()),
                description: concat!(
                    "For simple lookups: find definitions, list references, locate files. ",
                    "Finds code by meaning or exact match (grep_mode=true). ",
                    "Returns ranked file chunks with paths, lines, scores. 25+ languages supported. ",
                    "**Prefer semantex_agent for most queries** — it auto-classifies and handles fallbacks. ",
                    "Use semantex_search directly only when you need structured JSON results or specific ",
                    "search parameters (max_results, rerank, grep_mode)."
                ).into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "query": { "type": "string", "description": "Natural language search query" },
                        "path": { "type": "string", "description": "Project path to search (defaults to cwd)" },
                        "max_results": { "type": "integer", "description": "Maximum results to return", "default": 10 },
                        "rerank": { "type": "boolean", "description": "Enable cross-encoder reranking (slower but may improve ranking)", "default": false },
                        "grep_mode": { "type": "boolean", "description": "Fast grep-like search using only sparse+exact matching (no embedding model needed)", "default": false }
                    },
                    "required": ["query"]
                }),
                output_schema: Some(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "results": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "file": { "type": "string" },
                                    "lines": { "type": "string" },
                                    "score": { "type": "number" },
                                    "snippet": { "type": "string" },
                                    "name": { "type": "string" },
                                    "lang": { "type": "string" }
                                },
                                "required": ["file", "lines", "score", "snippet"]
                            }
                        },
                        "metrics": {
                            "type": "object",
                            "properties": {
                                "total_ms": { "type": "integer" },
                                "result_count": { "type": "integer" },
                                "query_type": { "type": "string" }
                            }
                        }
                    },
                    "required": ["results"]
                })),
                annotations: Some(ToolAnnotations::read_only("Semantic Code Search")),
            },
            Tool {
                name: "semantex_index".into(),
                title: Some("Build Search Index".into()),
                description: concat!(
                    "Build or update the semantex search index. Usually NOT needed — semantex auto-indexes on first search. ",
                    "Call only to force a rebuild after major changes."
                ).into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Project path to index" }
                    },
                    "required": ["path"]
                }),
                output_schema: None,
                annotations: Some(ToolAnnotations::local_mutation("Build Search Index")),
            },
            Tool {
                name: "semantex_status".into(),
                title: Some("Index Status".into()),
                description: "Check semantex index status: whether it exists, file count, chunk count, freshness. Use to verify indexing is complete.".into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Project path to check" }
                    }
                }),
                output_schema: None,
                annotations: Some(ToolAnnotations::read_only("Index Status")),
            },
            Tool {
                name: "semantex_health".into(),
                title: Some("Health Check".into()),
                description: "Health check for the semantex system, including model availability and configuration.".into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {},
                    "additionalProperties": false
                }),
                output_schema: None,
                annotations: Some(ToolAnnotations::read_only("Health Check")),
            },
            Tool {
                name: "semantex_validate".into(),
                title: Some("Validate Index".into()),
                description: "Run consistency checks on a semantex index: meta-DB sync, stale files, dense/sparse index integrity, graph consistency. Returns per-check pass/fail with details.".into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Project path (defaults to cwd)" }
                    }
                }),
                output_schema: None,
                annotations: Some(ToolAnnotations::read_only("Validate Index")),
            },
            Tool {
                name: "semantex_deep".into(),
                title: Some("Deep Code Search".into()),
                description: concat!(
                    "One call replaces 5-10 grep+read iterations. ",
                    "Searches, reads, graph-expands, and summarizes into a prose answer with sources. ",
                    "**Prefer semantex_agent for most queries** — it auto-routes to deep search when appropriate. ",
                    "Use semantex_deep directly only when you need structured JSON results or explicit control ",
                    "over the deep search pipeline. ",
                    "The answer is authoritative — do not re-read source files listed in the response."
                ).into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language question about the code"
                        },
                        "path": {
                            "type": "string",
                            "description": "Project path (defaults to current working directory)"
                        }
                    },
                    "required": ["query"]
                }),
                output_schema: Some(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "answer": { "type": "string" },
                        "sources": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "file": { "type": "string" },
                                    "start_line": { "type": "integer" },
                                    "end_line": { "type": "integer" },
                                    "name": { "type": "string" },
                                    "kind": { "type": "string" }
                                },
                                "required": ["file", "start_line", "end_line"]
                            }
                        },
                        "metrics": { "type": "object" }
                    },
                    "required": ["answer", "sources"]
                })),
                annotations: Some(ToolAnnotations::read_only("Deep Code Search")),
            },
        ];

        JsonRpcResponse::success(
            id,
            serde_json::json!({ "tools": serde_json::to_value(&tools).expect("tools serialization") }),
        )
    }

    // -------------------------------------------------------------------------
    // tools/call — dispatch + structured output
    // -------------------------------------------------------------------------

    fn handle_tool_call(
        &self,
        id: Option<serde_json::Value>,
        params: &serde_json::Value,
        writer: &NotificationWriter,
    ) -> JsonRpcResponse {
        let tool_name = params.get("name").and_then(|v| v.as_str()).unwrap_or("");
        let arguments = params
            .get("arguments")
            .cloned()
            .unwrap_or(serde_json::json!({}));

        // Extract _meta.progressToken if the client sent one.
        let progress_token = params
            .get("_meta")
            .and_then(|m| m.get("progressToken"))
            .cloned();

        let result = match tool_name {
            "semantex_agent" => self.tool_agent(&arguments),
            "semantex_search" => self.tool_search(&arguments, writer, progress_token.as_ref()),
            "semantex_index" => self.tool_index(&arguments).map(ToolOutput::text),
            "semantex_status" => self.tool_status(&arguments).map(ToolOutput::text),
            "semantex_health" => self.tool_health(&arguments).map(ToolOutput::text),
            "semantex_validate" => self.tool_validate(&arguments).map(ToolOutput::text),
            "semantex_deep" => self.tool_deep_search(&arguments, writer, progress_token.as_ref()),
            _ => Err(anyhow::anyhow!("Unknown tool: {tool_name}")),
        };

        match result {
            Ok(output) => {
                // Log tool completion at info level
                self.maybe_log(
                    writer,
                    LogLevel::Info,
                    "tools",
                    &serde_json::json!({
                        "tool": tool_name,
                        "status": "ok",
                    }),
                );

                JsonRpcResponse::success(
                    id,
                    serde_json::to_value(ToolCallResult {
                        content: vec![ToolContent {
                            content_type: "text".into(),
                            text: output.text,
                        }],
                        is_error: None,
                        structured_content: output.structured,
                    })
                    .expect("ToolCallResult serialization"),
                )
            }
            Err(e) => {
                self.maybe_log(
                    writer,
                    LogLevel::Error,
                    "tools",
                    &serde_json::json!({
                        "tool": tool_name,
                        "error": format!("{e}"),
                    }),
                );

                JsonRpcResponse::success(
                    id,
                    serde_json::to_value(ToolCallResult {
                        content: vec![ToolContent {
                            content_type: "text".into(),
                            text: format!("Error: {e}"),
                        }],
                        is_error: Some(true),
                        structured_content: None,
                    })
                    .expect("ToolCallResult serialization"),
                )
            }
        }
    }

    /// Send a log notification if the level meets the current threshold.
    fn maybe_log(
        &self,
        writer: &NotificationWriter,
        level: LogLevel,
        logger: &str,
        data: &serde_json::Value,
    ) {
        if level >= *self.log_level.lock() {
            writer.send_log(level, logger, data);
        }
    }

    // -------------------------------------------------------------------------
    // Tool: semantex_agent
    // -------------------------------------------------------------------------

    #[tracing::instrument(skip(self, args), fields(tool = "agent"))]
    fn tool_agent(&self, args: &serde_json::Value) -> Result<ToolOutput> {
        use semantex_core::search::agent::AgentPipeline;
        use semantex_core::search::agent_classifier::AgentRoute;
        use semantex_core::server::protocol::AgentRequest;

        // Support both single `query` and batch `queries` parameters.
        let queries: Vec<String> = if let Some(arr) = args.get("queries").and_then(|v| v.as_array()) {
            let qs: Vec<String> = arr
                .iter()
                .filter_map(|v| v.as_str())
                .map(std::string::ToString::to_string)
                .collect();
            if qs.is_empty() {
                return Err(anyhow::anyhow!("'queries' array is empty"));
            }
            qs
        } else {
            let q = args
                .get("query")
                .and_then(|v| v.as_str())
                .ok_or_else(|| anyhow::anyhow!("Missing 'query' or 'queries' parameter"))?;
            vec![q.to_string()]
        };

        let is_batch = queries.len() > 1;

        let path = args.get("path").and_then(|v| v.as_str()).map_or_else(
            || std::env::current_dir().context("Failed to determine current directory"),
            |p| Ok(PathBuf::from(p)),
        )?;

        let full_code = args
            .get("full_code")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);

        // Parse the optional `depth` parameter and map it to an AgentRoute override.
        let depth_route: Option<AgentRoute> = match args
            .get("depth")
            .and_then(|v| v.as_str())
        {
            Some("quick") => Some(AgentRoute::ExactSymbol),
            Some("search") => Some(AgentRoute::Semantic),
            Some("deep") => Some(AgentRoute::Deep),
            _ => None,
        };

        // Parse the optional `focus` parameter.
        let focus: Option<&str> = args.get("focus").and_then(|v| v.as_str());

        tracing::info!(
            queries = ?queries,
            path = %path.display(),
            full_code,
            ?depth_route,
            focus,
            "MCP agent search"
        );

        let path = path.canonicalize().unwrap_or_else(|_| path.clone());

        let idx_state = state::detect(&path);
        match idx_state {
            IndexState::NotIndexed | IndexState::Stale => {
                self.spawn_background_index(&path);
                // For batch queries, fall back using the first query.
                return Self::do_ripgrep_fallback(&queries[0], &path, 10);
            }
            IndexState::Building => {
                return Self::do_ripgrep_fallback(&queries[0], &path, 10);
            }
            IndexState::Ready => {}
        }

        let index_dir = SemantexConfig::project_index_dir(&path);
        let cached = self.get_searcher(&index_dir)?;
        let pipeline = AgentPipeline::new(&cached.searcher, path.clone());

        let budget = args
            .get("budget")
            .and_then(serde_json::Value::as_u64)
            .map_or(12_000, |v| v as usize);

        // Run each query and collect formatted results.
        let mut parts: Vec<String> = Vec::with_capacity(queries.len());
        for q in &queries {
            let request = AgentRequest {
                query: q.clone(),
                route: depth_route,
                budget: Some(budget),
                full_code,
            };
            let response = pipeline.handle(&request);
            parts.push(response.formatted);
        }

        // Combine results.
        let mut combined = if is_batch {
            let n = parts.len();
            format!(
                "[Batch results for {} queries — merged]\n\n{}",
                n,
                parts.join("\n\n---\n\n")
            )
        } else {
            parts.remove(0)
        };

        // Apply focus formatting to the result.
        combined = apply_focus(combined, focus);

        Ok(ToolOutput::text(combined))
    }

    // -------------------------------------------------------------------------
    // Tool: semantex_search
    // -------------------------------------------------------------------------

    #[tracing::instrument(skip(self, args, _writer, _progress_token), fields(tool = "search"))]
    fn tool_search(
        &self,
        args: &serde_json::Value,
        _writer: &NotificationWriter,
        _progress_token: Option<&serde_json::Value>,
    ) -> Result<ToolOutput> {
        let query = args
            .get("query")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing 'query' parameter"))?;
        let path = args.get("path").and_then(|v| v.as_str()).map_or_else(
            || std::env::current_dir().context("Failed to determine current directory"),
            |p| Ok(PathBuf::from(p)),
        )?;
        let max_results = args
            .get("max_results")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(10) as usize;
        let rerank = args
            .get("rerank")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        let grep_mode = args
            .get("grep_mode")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);

        tracing::info!(query, path = %path.display(), max_results, rerank, grep_mode, "MCP search");

        let idx_state = state::detect(&path);
        match idx_state {
            IndexState::Ready => {
                self.do_semantex_search(query, &path, max_results, rerank, grep_mode)
            }
            IndexState::NotIndexed | IndexState::Stale => {
                self.spawn_background_index(&path);
                Self::do_ripgrep_fallback(query, &path, max_results)
            }
            IndexState::Building => Self::do_ripgrep_fallback(query, &path, max_results),
        }
    }

    fn do_semantex_search(
        &self,
        query: &str,
        path: &std::path::Path,
        max_results: usize,
        rerank: bool,
        grep_mode: bool,
    ) -> Result<ToolOutput> {
        let index_dir = SemantexConfig::project_index_dir(path);

        let search_output = if grep_mode {
            let searcher = HybridSearcher::open_sparse_only(&index_dir, &self.config)?;
            let sq = SearchQuery::new(query).grep_mode();
            searcher.search(&sq)?
        } else {
            let cached = self.get_searcher(&index_dir)?;
            let mut sq = SearchQuery::new(query).max_results(max_results);
            if !rerank {
                sq = sq.no_rerank();
            }
            cached.searcher.search(&sq)?
        };

        let json_results: Vec<serde_json::Value> = search_output
            .results
            .iter()
            .map(|r| {
                let snippet = make_snippet_mcp(&r.chunk.content, &r.chunk.chunk_type);
                let score = (r.score * 100.0_f32).round() / 100.0_f32;
                let mut val = serde_json::json!({
                    "file": r.chunk.file_path.display().to_string(),
                    "lines": format!("{}-{}", r.chunk.start_line, r.chunk.end_line),
                    "score": score,
                    "snippet": snippet,
                });
                if let semantex_core::types::ChunkType::AstNode { name, language, .. } =
                    &r.chunk.chunk_type
                {
                    val["name"] = serde_json::Value::String(name.clone());
                    val["lang"] = serde_json::Value::String(language.clone());
                }
                val
            })
            .collect();

        let metrics = &search_output.metrics;
        let metrics_json = serde_json::json!({
            "total_ms": metrics.total_ms,
            "result_count": metrics.result_count,
            "query_type": metrics.query_type,
        });

        // Structured content (machine-readable, for clients that support outputSchema)
        let structured = serde_json::json!({
            "results": json_results,
            "metrics": metrics_json,
        });

        // Text content (human/LLM-readable fallback)
        let json_text = serde_json::to_string(&json_results)?;
        let response_bytes = json_text.len();
        let footer = format_metrics_footer(metrics, response_bytes);
        let text = format!("{json_text}\n\n{footer}");

        Ok(ToolOutput {
            text,
            structured: Some(structured),
        })
    }

    fn do_ripgrep_fallback(
        query: &str,
        path: &std::path::Path,
        max_results: usize,
    ) -> Result<ToolOutput> {
        if !ripgrep_fallback::is_rg_available() {
            return Ok(ToolOutput::text(
                "Note: Index building. ripgrep not available for keyword fallback.\n\n[]"
                    .to_string(),
            ));
        }

        let results = ripgrep_fallback::search(query, path, max_results)?;
        let json = ripgrep_fallback::format_as_json(&results);
        Ok(ToolOutput::text(format!(
            "Note: Index building. Showing keyword (ripgrep) results.\n\n{json}"
        )))
    }

    // -------------------------------------------------------------------------
    // Tool: semantex_index
    // -------------------------------------------------------------------------

    #[tracing::instrument(skip(self, args), fields(tool = "index"))]
    fn tool_index(&self, args: &serde_json::Value) -> Result<String> {
        let path = args
            .get("path")
            .and_then(|v| v.as_str())
            .map(PathBuf::from)
            .ok_or_else(|| anyhow::anyhow!("Missing 'path' parameter"))?;

        tracing::info!(path = %path.display(), "MCP index request");

        let index_dir = SemantexConfig::project_index_dir(&path);
        self.invalidate_cache(&index_dir);
        self.spawn_background_index(&path);

        Ok(format!(
            "Indexing started for {}. Use semantex_status to check progress.",
            path.display()
        ))
    }

    // -------------------------------------------------------------------------
    // Tool: semantex_status
    // -------------------------------------------------------------------------

    #[allow(clippy::unused_self)]
    #[tracing::instrument(skip(self, args), fields(tool = "status"))]
    fn tool_status(&self, args: &serde_json::Value) -> Result<String> {
        let path = args.get("path").and_then(|v| v.as_str()).map_or_else(
            || std::env::current_dir().context("Failed to determine current directory"),
            |p| Ok(PathBuf::from(p)),
        )?;

        let index_dir = SemantexConfig::project_index_dir(&path);
        let meta_path = index_dir.join("meta.json");

        if meta_path.exists() {
            let content = std::fs::read_to_string(&meta_path)?;
            let meta: semantex_core::types::IndexMeta = serde_json::from_str(&content)?;
            Ok(format!(
                "Index exists:\n  Files: {}\n  Chunks: {}\n  Model: {}\n  Updated: {}",
                meta.file_count, meta.chunk_count, meta.embedding_model, meta.updated_at
            ))
        } else {
            Ok("No index found for this project. Run 'semantex index' first.".into())
        }
    }

    // -------------------------------------------------------------------------
    // Tool: semantex_health
    // -------------------------------------------------------------------------

    #[tracing::instrument(skip(self, _args), fields(tool = "health"))]
    fn tool_health(&self, _args: &serde_json::Value) -> Result<String> {
        use semantex_core::embedding::model_manager;

        let mut status = Vec::new();
        status.push(format!(
            "Semantex Health Check v{}",
            env!("CARGO_PKG_VERSION")
        ));
        status.push("=".repeat(50));

        let models_dir = self.config.models_dir();
        status.push(format!("\nModels Directory: {}", models_dir.display()));

        if model_manager::is_colbert_downloaded(&models_dir) {
            let model_dir = models_dir.join("LateOn-Code-edge");
            let model_path = model_dir.join("model_int8.onnx");
            let tokenizer_path = model_dir.join("tokenizer.json");
            status.push("\nColBERT Model: OK".to_string());
            status.push("  Name: LateOn-Code-edge".to_string());
            status.push(format!("  Path: {}", model_dir.display()));
            status.push(format!(
                "  Model file: {}",
                if model_path.exists() {
                    "present"
                } else {
                    "MISSING"
                }
            ));
            status.push(format!(
                "  Tokenizer: {}",
                if tokenizer_path.exists() {
                    "present"
                } else {
                    "MISSING"
                }
            ));
        } else {
            status.push("\nColBERT Model: NOT DOWNLOADED".to_string());
            status.push("  Run 'semantex download-models' to download".to_string());
        }

        if self.config.rerank {
            status.push("\nReranker: ENABLED (fastembed JINA Reranker v1 Turbo)".to_string());
        } else {
            status.push("\nReranker: DISABLED".to_string());
        }

        status.push("\nConfiguration:".to_string());
        status.push(format!("  Chunk size: {}", self.config.chunk_size));
        status.push(format!("  Chunk overlap: {}", self.config.chunk_overlap));
        status.push(format!(
            "  Reranking: {}",
            if self.config.rerank {
                "enabled"
            } else {
                "disabled"
            }
        ));
        status.push(format!(
            "  Rerank candidates: {}",
            self.config.rerank_candidates
        ));

        let cache = self.cache.lock();
        status.push(format!(
            "\nSearcher Cache: {}/{} entries",
            cache.len(),
            self.max_cached
        ));
        let now = Instant::now();
        for (path, entry) in cache.iter() {
            let age = now.duration_since(*entry.last_used.lock()).as_secs();
            status.push(format!("  {} (idle {}s)", path.display(), age));
        }

        Ok(status.join("\n"))
    }

    // -------------------------------------------------------------------------
    // Tool: semantex_validate
    // -------------------------------------------------------------------------

    #[allow(clippy::unused_self)]
    #[tracing::instrument(skip(self, args), fields(tool = "validate"))]
    fn tool_validate(&self, args: &serde_json::Value) -> Result<String> {
        let path = args.get("path").and_then(|v| v.as_str()).map_or_else(
            || std::env::current_dir().context("Failed to determine current directory"),
            |p| Ok(PathBuf::from(p)),
        )?;

        let report = semantex_core::index::validate::validate(&path)?;

        let mut lines = Vec::new();
        for check in &report.checks {
            let icon = if check.passed { "PASS" } else { "FAIL" };
            lines.push(format!("[{icon}] {}: {}", check.name, check.message));
        }
        lines.push(String::new());
        lines.push(report.summary());

        if !report.all_passed() {
            lines.push("Index may need rebuilding: run `semantex index <path>`.".to_string());
        }

        Ok(lines.join("\n"))
    }

    // -------------------------------------------------------------------------
    // Tool: semantex_deep (with progress notifications)
    // -------------------------------------------------------------------------

    #[allow(clippy::too_many_lines)]
    #[tracing::instrument(skip(self, args, writer, progress_token), fields(tool = "deep_search"))]
    fn tool_deep_search(
        &self,
        args: &serde_json::Value,
        writer: &NotificationWriter,
        progress_token: Option<&serde_json::Value>,
    ) -> Result<ToolOutput> {
        let query = args
            .get("query")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing required parameter: query"))?;

        let path = args.get("path").and_then(|v| v.as_str()).map_or_else(
            || std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            PathBuf::from,
        );

        let path = path.canonicalize().unwrap_or_else(|_| path.clone());

        tracing::info!(query, path = %path.display(), "MCP deep search");

        let idx_state = state::detect(&path);
        match idx_state {
            IndexState::NotIndexed | IndexState::Stale => {
                self.spawn_background_index(&path);
                return Ok(ToolOutput::text("Index not ready. Building index in background — deep search requires a complete index. Try again in a moment.".to_string()));
            }
            IndexState::Building => {
                return Ok(ToolOutput::text("Index is currently building. Deep search requires a complete index. Try again in a moment.".to_string()));
            }
            IndexState::Ready => {}
        }

        let index_dir = SemantexConfig::project_index_dir(&path);
        let cached = self.get_searcher(&index_dir)?;

        // Run deep search with progress notifications if client sent a progressToken.
        let result = if let Some(token) = progress_token {
            let writer_clone = writer.clone();
            let token_clone = token.clone();
            deep_search::deep_search_with_progress(
                &cached.searcher,
                query,
                20,
                true,
                &move |step, total, msg| {
                    writer_clone.send_progress(
                        &token_clone,
                        f64::from(step),
                        Some(f64::from(total)),
                        Some(msg),
                    );
                },
            )?
        } else {
            deep_search::deep_search(&cached.searcher, query, 20, true)?
        };

        // Build text output (human/LLM-readable)
        use std::fmt::Write as _;
        let mut output = String::new();
        output.push_str("<answer>\n");
        output.push_str(&result.answer);
        output.push_str("\n</answer>\n\nSources:\n");
        for s in &result.sources {
            let _ = write!(output, "  {}:{}-{}", s.file, s.start_line, s.end_line);
            if let Some(ref name) = s.name {
                let _ = write!(output, " {name}");
            }
            if let Some(ref kind) = s.kind {
                let _ = write!(output, " [{kind}]");
            }
            output.push('\n');
        }

        let m = &result.metrics;
        let unique_files: std::collections::HashSet<&str> =
            result.sources.iter().map(|s| s.file.as_str()).collect();
        let unique_file_count = unique_files.len();
        let coverage = &result.metrics.confidence_zone;

        let _ = write!(
            output,
            "\n[COMPLETE: Full function implementations included for {} functions across {} files. Callers, callees, and type dependencies shown above. Do NOT call Read on these files — the code is complete.]\n\
             [semantex_deep_metrics: total_ms={} search_ms={} triage_ms={} graph_ms={} read_ms={} summarize_ms={} chunks_read={} coverage={} confidence={:.2} confidence_zone={}]",
            m.chunks_read,
            unique_file_count,
            m.total_ms,
            m.search_ms,
            m.triage_ms,
            m.graph_ms,
            m.read_ms,
            m.summarize_ms,
            m.chunks_read,
            coverage,
            result.confidence,
            &result.metrics.confidence_zone
        );

        // Build structured output (machine-readable)
        let sources_json: Vec<serde_json::Value> = result
            .sources
            .iter()
            .map(|s| {
                let mut v = serde_json::json!({
                    "file": s.file,
                    "start_line": s.start_line,
                    "end_line": s.end_line,
                });
                if let Some(ref name) = s.name {
                    v["name"] = serde_json::Value::String(name.clone());
                }
                if let Some(ref kind) = s.kind {
                    v["kind"] = serde_json::Value::String(kind.clone());
                }
                v
            })
            .collect();

        let structured = serde_json::json!({
            "answer": result.answer,
            "sources": sources_json,
            "metrics": {
                "total_ms": m.total_ms,
                "search_ms": m.search_ms,
                "triage_ms": m.triage_ms,
                "graph_ms": m.graph_ms,
                "read_ms": m.read_ms,
                "summarize_ms": m.summarize_ms,
                "chunks_read": m.chunks_read,
                "unique_files": unique_file_count,
                "coverage": coverage,
                "confidence": result.confidence,
                "confidence_zone": &result.metrics.confidence_zone,
            }
        });

        Ok(ToolOutput {
            text: output,
            structured: Some(structured),
        })
    }
}

// =============================================================================
// Tool output helper
// =============================================================================

/// Carries both text content and optional structured content from a tool call.
struct ToolOutput {
    text: String,
    structured: Option<serde_json::Value>,
}

impl ToolOutput {
    fn text(text: String) -> Self {
        Self {
            text,
            structured: None,
        }
    }
}

// =============================================================================
// Formatting helpers
// =============================================================================

fn format_metrics_footer(
    metrics: &semantex_core::search::SearchMetrics,
    response_bytes: usize,
) -> String {
    let mut parts = vec![format!("total_ms={}", metrics.total_ms)];
    if let Some(ms) = metrics.dense_ms {
        parts.push(format!("dense_ms={ms}"));
    }
    if let Some(ms) = metrics.sparse_ms {
        parts.push(format!("sparse_ms={ms}"));
    }
    if let Some(ms) = metrics.exact_ms {
        parts.push(format!("exact_ms={ms}"));
    }
    if let Some(ms) = metrics.fusion_ms {
        parts.push(format!("fusion_ms={ms}"));
    }
    if let Some(ms) = metrics.rerank_ms {
        parts.push(format!("rerank_ms={ms}"));
    }
    parts.push(format!("results={}", metrics.result_count));
    parts.push(format!("query_type={}", metrics.query_type));
    parts.push(format!("response_bytes={response_bytes}"));
    format!("[semantex_metrics: {}]", parts.join(" "))
}

fn make_snippet_mcp(content: &str, chunk_type: &semantex_core::types::ChunkType) -> String {
    match chunk_type {
        semantex_core::types::ChunkType::AstNode { .. } => truncate_lines_mcp(content, 3),
        semantex_core::types::ChunkType::TextWindow { .. } => truncate_lines_mcp(content, 2),
        semantex_core::types::ChunkType::PdfPage { .. } => {
            if content.len() > 100 {
                format!("{}...", &content[..content.floor_char_boundary(100)])
            } else {
                content.to_string()
            }
        }
    }
}

fn truncate_lines_mcp(content: &str, n: usize) -> String {
    let lines: Vec<&str> = content.lines().take(n + 1).collect();
    if lines.len() > n {
        format!("{}...", lines[..n].join("\n"))
    } else {
        lines.join("\n")
    }
}

/// Apply `focus` formatting to a pre-formatted agent result string.
///
/// - `"signatures"`: Strip code block bodies, keeping only the first content line (signature) of
///   each fenced code block. Prepends `[focus: signatures]`.
/// - `"callers"`: Prepends `[focus: callers]` and appends a note about call graph edges.
/// - `"implementation"`: Prepends `[focus: implementation — full code included]`.
/// - `"patterns"`: Prepends `[focus: patterns]`.
/// - `None` or unrecognised value: returns text unchanged.
fn apply_focus(text: String, focus: Option<&str>) -> String {
    match focus {
        Some("signatures") => {
            // Collapse each fenced code block to only its first (signature) line.
            // State: outside block, inside block before first content, inside block after first.
            #[derive(PartialEq)]
            enum State {
                Outside,
                InsideBeforeFirst,
                InsideAfterFirst,
            }

            let mut out = String::from("[focus: signatures]\n\n");
            let mut state = State::Outside;

            for line in text.lines() {
                match state {
                    State::Outside => {
                        out.push_str(line);
                        out.push('\n');
                        if line.starts_with("```") {
                            state = State::InsideBeforeFirst;
                        }
                    }
                    State::InsideBeforeFirst => {
                        if line.starts_with("```") {
                            // Empty block — closing fence immediately.
                            out.push_str(line);
                            out.push('\n');
                            state = State::Outside;
                        } else {
                            // First content line = the function signature.
                            out.push_str(line);
                            out.push('\n');
                            state = State::InsideAfterFirst;
                        }
                    }
                    State::InsideAfterFirst => {
                        if line.starts_with("```") {
                            // Closing fence — emit it and return outside.
                            out.push_str(line);
                            out.push('\n');
                            state = State::Outside;
                        }
                        // All other body lines are skipped.
                    }
                }
            }
            out
        }
        Some("callers") => {
            format!(
                "[focus: callers]\n\n{text}\n\n[focus: callers — check the call graph edges shown above]"
            )
        }
        Some("implementation") => {
            format!("[focus: implementation — full code included]\n\n{text}")
        }
        Some("patterns") => {
            format!("[focus: patterns]\n\n{text}")
        }
        _ => text,
    }
}

#[cfg(test)]
mod tests {
    use super::apply_focus;

    #[test]
    fn test_apply_focus_none_passthrough() {
        let text = "some result".to_string();
        assert_eq!(apply_focus(text.clone(), None), text);
    }

    #[test]
    fn test_apply_focus_unknown_passthrough() {
        let text = "some result".to_string();
        assert_eq!(apply_focus(text.clone(), Some("unknown")), text);
    }

    #[test]
    fn test_apply_focus_implementation_prepends_marker() {
        let out = apply_focus("body".to_string(), Some("implementation"));
        assert!(out.starts_with("[focus: implementation — full code included]"));
        assert!(out.contains("body"));
    }

    #[test]
    fn test_apply_focus_callers_wraps() {
        let out = apply_focus("body".to_string(), Some("callers"));
        assert!(out.starts_with("[focus: callers]"));
        assert!(out.contains("body"));
        assert!(out.contains("call graph"));
    }

    #[test]
    fn test_apply_focus_patterns_prepends_marker() {
        let out = apply_focus("body".to_string(), Some("patterns"));
        assert!(out.starts_with("[focus: patterns]"));
        assert!(out.contains("body"));
    }

    #[test]
    fn test_apply_focus_signatures_strips_body() {
        let input = "header\n```rust\nfn foo(x: i32) -> i32 {\n    x + 1\n}\n```\nfooter\n";
        let out = apply_focus(input.to_string(), Some("signatures"));
        assert!(out.starts_with("[focus: signatures]"));
        // Signature line kept
        assert!(out.contains("fn foo(x: i32) -> i32 {"));
        // Body line stripped
        assert!(!out.contains("x + 1"));
        // Closing fence and footer kept
        assert!(out.contains("footer"));
    }

    #[test]
    fn test_apply_focus_signatures_empty_block() {
        let input = "before\n```\n```\nafter\n";
        let out = apply_focus(input.to_string(), Some("signatures"));
        assert!(out.contains("before"));
        assert!(out.contains("after"));
    }

    #[test]
    fn test_apply_focus_signatures_multiple_blocks() {
        let input = "```\nfn a() {}\n    body_a\n```\n```\nfn b() {}\n    body_b\n```\n";
        let out = apply_focus(input.to_string(), Some("signatures"));
        assert!(out.contains("fn a() {}"));
        assert!(!out.contains("body_a"));
        assert!(out.contains("fn b() {}"));
        assert!(!out.contains("body_b"));
    }
}
