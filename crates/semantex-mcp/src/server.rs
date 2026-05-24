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
use semantex_core::index::storage::ChunkStore;
use semantex_core::search::SearchQuery;
use semantex_core::search::deep as deep_search;
use semantex_core::search::hybrid::HybridSearcher;
use semantex_core::search::ripgrep_fallback;
use semantex_core::server::listener::warm_state_ready;
use std::collections::{HashMap, HashSet};
use std::io::{BufRead, BufWriter, Write as _};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use semantex_core::index::registry;

/// Default toolset bundle: exposes the full surface (all 13 tools).
pub const DEFAULT_TOOLSET: &str = "all";

/// Valid toolset names: `core`, `structural`, `all`.
pub const TOOLSETS: &[&str] = &["core", "structural", "all"];

/// Format seconds into a human-readable age string.
fn format_age(secs: u64) -> String {
    if secs < 60 {
        format!("{secs}s")
    } else if secs < 3600 {
        format!("{}m {}s", secs / 60, secs % 60)
    } else if secs < 86400 {
        format!("{}h {}m", secs / 3600, (secs % 3600) / 60)
    } else {
        format!("{}d {}h", secs / 86400, (secs % 86400) / 3600)
    }
}

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

/// Idle eviction timeout: entries unused for this duration are dropped.
/// Kept short (2 min) so that MCP processes sitting idle between queries
/// release searcher memory quickly — critical when multiple sessions run.
const IDLE_TIMEOUT_SECS: u64 = 120; // 2 minutes

// =============================================================================
// MCP Server
// =============================================================================

pub struct McpServer {
    config: SemantexConfig,
    /// LRU cache of `HybridSearcher` instances keyed by canonical index directory.
    cache: Mutex<HashMap<PathBuf, Arc<CachedSearcher>>>,
    max_cached: usize,
    /// RSS limit in MB. When exceeded, all cached searchers are evicted.
    rss_limit_mb: u64,
    /// Per-project indexing status — prevents double-spawning background index builds.
    index_states: Arc<Mutex<HashMap<PathBuf, IndexingStatus>>>,
    /// Current MCP logging level — only messages at or above this level are sent.
    log_level: Mutex<LogLevel>,
    /// Active toolset bundle name (`core`, `structural`, or `all`).
    /// Controls which tools are visible to `tools/list` and callable via
    /// `tools/call`. Defaults to `all`.
    toolset: String,
}

impl McpServer {
    pub fn new(config: SemantexConfig) -> Self {
        Self::with_toolset(config, DEFAULT_TOOLSET)
    }

    /// Construct an MCP server restricted to a specific toolset bundle.
    /// Unknown toolset names fall back to `all` (the full surface) so that
    /// callers don't accidentally lock themselves out of the tool list.
    pub fn with_toolset(config: SemantexConfig, toolset: &str) -> Self {
        // Note: `SEMANTEX_ORT_THREADS` (and the derived OMP/MKL/BLAS thread
        // caps) is set in `semantex-cli`'s `main()` before any threads spawn —
        // the only safe time to mutate the environment. Setting it here would
        // be UB because mimalloc and tracing have already started workers, so
        // the previous `unsafe { set_var(...) }` block was removed (Finding 5a).
        // Tests that bypass `main` get the fastembed default, which is fine.

        let max_cached = std::env::var("SEMANTEX_MCP_CACHE_SIZE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1);
        let rss_limit_mb = std::env::var("SEMANTEX_MAX_RSS_MB")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(512);
        let toolset = if TOOLSETS.contains(&toolset) {
            toolset.to_string()
        } else {
            tracing::warn!(
                requested = toolset,
                "Unknown toolset name; falling back to 'all'"
            );
            DEFAULT_TOOLSET.to_string()
        };
        Self {
            config,
            cache: Mutex::new(HashMap::new()),
            max_cached,
            rss_limit_mb,
            index_states: Arc::new(Mutex::new(HashMap::new())),
            log_level: Mutex::new(LogLevel::default()),
            toolset,
        }
    }

    /// Returns the active toolset name.
    pub fn toolset(&self) -> &str {
        &self.toolset
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

        // Evict BEFORE allocating the new searcher to keep peak memory bounded.
        while cache.len() >= self.max_cached {
            if let Some(lru_key) = cache
                .iter()
                .min_by_key(|(_, v)| *v.last_used.lock())
                .map(|(k, _)| k.clone())
            {
                tracing::info!(evicted = %lru_key.display(), "Evicting cached searcher (cache full)");
                cache.remove(&lru_key);
            } else {
                break;
            }
        }

        // Drop the lock while opening the searcher (I/O + model load is slow).
        drop(cache);
        let searcher = HybridSearcher::open(index_dir, &self.config)?;
        let entry = Arc::new(CachedSearcher {
            searcher,
            last_used: Mutex::new(now),
        });

        let mut cache = self.cache.lock();
        cache.insert(index_dir.to_path_buf(), Arc::clone(&entry));
        Ok(entry)
    }

    fn invalidate_cache(&self, index_dir: &std::path::Path) {
        self.cache.lock().remove(index_dir);
    }

    /// Check process RSS and apply graduated memory pressure relief.
    /// Called after each tool invocation to prevent runaway memory growth.
    ///
    /// - Over 75% of limit: evict all cached searchers + purge allocator
    /// - Over limit: same + log a hard warning
    fn check_rss_and_evict(&self) {
        if self.rss_limit_mb == 0 {
            return;
        }
        let Some(rss_mb) = semantex_core::memory::current_rss_mb() else {
            return;
        };
        let threshold_75 = self.rss_limit_mb * 3 / 4;
        if rss_mb > threshold_75 {
            let mut cache = self.cache.lock();
            let evicted = cache.len();
            cache.clear();
            drop(cache);
            // Also clear the index_states map (small but unbounded)
            self.index_states
                .lock()
                .retain(|_, s| matches!(s, IndexingStatus::Building));
            // Force mimalloc to return freed pages to the OS
            semantex_core::memory::purge_allocator();

            if rss_mb > self.rss_limit_mb {
                tracing::warn!(
                    rss_mb,
                    limit_mb = self.rss_limit_mb,
                    evicted,
                    "RSS limit exceeded — evicted all cached searchers and purged allocator"
                );
            } else {
                tracing::info!(
                    rss_mb,
                    threshold_mb = threshold_75,
                    evicted,
                    "RSS at 75% of limit — proactively evicted cached searchers"
                );
            }
        }
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

        // Skip background indexing if RSS is already over 50% of the limit —
        // indexing is very memory-intensive and could push us over the edge.
        if self.rss_limit_mb > 0
            && let Some(rss_mb) = semantex_core::memory::current_rss_mb()
            && rss_mb > self.rss_limit_mb / 2
        {
            tracing::warn!(
                rss_mb,
                limit_mb = self.rss_limit_mb,
                "Skipping background index — RSS already at {}% of limit",
                rss_mb * 100 / self.rss_limit_mb
            );
            self.index_states
                .lock()
                .insert(canonical, IndexingStatus::Failed("Memory pressure".into()));
            return;
        }

        // Register this project in the global registry so tool_status can list all repos.
        registry::register(&canonical);

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

    /// Detect the index state, using the warm-state sentinel as a fast-path
    /// (E8(d)). The fast path is the common case: warm daemon, no concurrent
    /// rebuild, current schema. To preserve correctness it still validates the
    /// two invariants that `state::detect` would catch beyond plain presence:
    ///
    /// 1. **`.semantex.lock`** — held by an in-progress indexer. If we returned
    ///    `Ready` while the lock is held, callers would query a half-built
    ///    `chunks.db`. A single `flock` syscall is sub-microsecond on warm
    ///    cache (Finding 1).
    /// 2. **Schema staleness** — `meta.json::schema_version` vs.
    ///    `IndexMeta::CURRENT_SCHEMA_VERSION`. One bounded JSON parse
    ///    (~100 µs) — still vastly cheaper than full `state::detect` which
    ///    additionally stat's siblings and parses chunk_count, etc.
    ///
    /// Performance-vs-correctness tradeoff: we keep the sentinel-presence
    /// short-circuit (avoiding ~700 µs of `detect()` work in the warm-and-clean
    /// case), but add ~100 µs of lock+schema re-validation so the fast path
    /// cannot return `Ready` under a concurrent reindex or after a schema bump.
    ///
    /// Callers that need a fully detailed state (e.g. `tool_status` for
    /// reporting) should still use `state::detect` directly; this helper is
    /// intended only for the search-path tool handlers (`tool_agent`,
    /// `tool_search`, `tool_deep_search`, and the M1-M6 structural tools).
    fn detect_state_fast(path: &std::path::Path) -> IndexState {
        let index_dir = path.join(".semantex");
        if warm_state_ready(&index_dir) {
            // Even with a hot sentinel, an indexer can be rebuilding right
            // now (rebuild-while-running). One flock syscall catches that.
            let lock_path = index_dir.join(".semantex.lock");
            if state::is_locked(&lock_path) {
                return IndexState::Building;
            }
            // A schema bump invalidates the sentinel-implied "Ready" claim.
            let meta_path = index_dir.join("meta.json");
            if state::is_stale(&meta_path) {
                return IndexState::Stale;
            }
            IndexState::Ready
        } else {
            state::detect(path)
        }
    }

    /// Trigger a background refresh if the index is older than `SEMANTEX_REFRESH_SECS`
    /// (default: 3600 s). Returns `true` if a refresh was actually triggered.
    fn maybe_trigger_refresh(&self, path: &std::path::Path) -> bool {
        let threshold: u64 = std::env::var("SEMANTEX_REFRESH_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(3600);

        let Some(age) = state::index_age_secs(path) else {
            return false;
        };
        if age < threshold {
            return false;
        }

        let canonical = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
        if matches!(
            self.index_states.lock().get(&canonical),
            Some(IndexingStatus::Building)
        ) {
            return false; // already in progress
        }

        tracing::info!(path = %path.display(), age_secs = age, "Triggering background refresh (index stale)");
        self.spawn_background_index(path);
        true
    }

    // -------------------------------------------------------------------------
    // tools/list — toolset-filtered tool definitions with annotations + output
    // schemas. The full set is constructed in `all_tools()` and filtered
    // through `tools_for_toolset()` according to the active bundle.
    // -------------------------------------------------------------------------

    fn handle_tools_list(&self, id: Option<serde_json::Value>) -> JsonRpcResponse {
        let tools = self.tools_for_toolset(&self.toolset);
        JsonRpcResponse::success(
            id,
            serde_json::json!({ "tools": serde_json::to_value(&tools).expect("tools serialization") }),
        )
    }

    /// Return the list of tool definitions for the given toolset bundle.
    ///
    /// Bundles (per spec I3):
    /// - `core`: 4 tools — `semantex_search`, `semantex_deep`, `semantex_agent`, `semantex_symbol`.
    /// - `structural`: 5 tools — `semantex_symbol`, `semantex_callers`, `semantex_callees`,
    ///   `semantex_implementations`, `semantex_architecture`.
    /// - `all` (default): every tool registered with the server (13 total).
    ///
    /// Unknown toolset names fall back to `all` so callers (e.g. W7's HTTP
    /// transport) cannot accidentally lock themselves out of the surface.
    /// Exposed publicly so HTTP transports can call it without going through
    /// the JSON-RPC layer.
    #[must_use]
    pub fn tools_for_toolset(&self, toolset: &str) -> Vec<Tool> {
        let all = Self::all_tools();
        let allow: &[&str] = match toolset {
            "core" => &[
                "semantex_search",
                "semantex_deep",
                "semantex_agent",
                "semantex_symbol",
            ],
            "structural" => &[
                "semantex_symbol",
                "semantex_callers",
                "semantex_callees",
                "semantex_implementations",
                "semantex_architecture",
            ],
            _ => return all, // `all` or unknown — return the full surface
        };
        let allow_set: HashSet<&str> = allow.iter().copied().collect();
        all.into_iter()
            .filter(|t| allow_set.contains(t.name.as_str()))
            .collect()
    }

    /// Build the full tool catalog (all 13 tools). Single source of truth used
    /// by both `tools/list` (filtered by toolset) and `tool_call` (dispatch).
    #[allow(clippy::too_many_lines)]
    fn all_tools() -> Vec<Tool> {
        vec![
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
            // -------------------------------------------------------------
            // M1 — semantex_symbol
            // -------------------------------------------------------------
            Tool {
                name: "semantex_symbol".into(),
                title: Some("Exact Symbol Lookup".into()),
                description: concat!(
                    "Exact symbol lookup backed by the indexed symbol table. ",
                    "One call returns the symbol's location, signature, docstring, ",
                    "semantic role, and the count of callers/callees. ",
                    "Replaces 3-5 grep+read iterations for a single named symbol. ",
                    "Use when you know the symbol name; use semantex_search for ",
                    "fuzzy / natural-language queries."
                )
                .into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "name": { "type": "string", "description": "Symbol name to look up (case-sensitive)" },
                        "kind": { "type": "string", "description": "Optional kind filter (function, method, class, struct, enum, interface, trait)" },
                        "path": { "type": "string", "description": "Project path (defaults to cwd)" }
                    },
                    "required": ["name"]
                }),
                output_schema: Some(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "matches": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "location": { "type": "object" },
                                    "signature": { "type": "string" },
                                    "docstring": { "type": "string" },
                                    "semantic_role": { "type": "string" },
                                    "callers_count": { "type": "integer" },
                                    "callees_count": { "type": "integer" },
                                    "confidence": { "type": "string" }
                                },
                                "required": ["location"]
                            }
                        }
                    },
                    "required": ["matches"]
                })),
                annotations: Some(ToolAnnotations::read_only("Exact Symbol Lookup")),
            },
            // -------------------------------------------------------------
            // M2 — semantex_callers
            // -------------------------------------------------------------
            Tool {
                name: "semantex_callers".into(),
                title: Some("Reverse Call Graph".into()),
                description: concat!(
                    "Reverse call-graph walk: list all chunks that call a given symbol. ",
                    "Default depth=1 (direct callers); depth=2 also includes callers-of-callers. ",
                    "Replaces 5-15 grep iterations when finding usages of an API. ",
                    "Returns one entry per caller with location, signature, and edge_kind."
                )
                .into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "symbol": { "type": "string", "description": "Symbol name to find callers of" },
                        "depth": { "type": "integer", "description": "Walk depth: 1 (direct) or 2 (transitive). Default 1.", "default": 1 },
                        "path": { "type": "string", "description": "Project path (defaults to cwd)" }
                    },
                    "required": ["symbol"]
                }),
                output_schema: Some(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "callers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "caller_location": { "type": "object" },
                                    "caller_signature": { "type": "string" },
                                    "edge_kind": { "type": "string" }
                                },
                                "required": ["caller_location", "edge_kind"]
                            }
                        }
                    },
                    "required": ["callers"]
                })),
                annotations: Some(ToolAnnotations::read_only("Reverse Call Graph")),
            },
            // -------------------------------------------------------------
            // M3 — semantex_callees
            // -------------------------------------------------------------
            Tool {
                name: "semantex_callees".into(),
                title: Some("Forward Call Graph".into()),
                description: concat!(
                    "Forward call-graph walk: list all symbols invoked by a given function. ",
                    "Default depth=1 (direct callees); depth=2 also includes callees-of-callees. ",
                    "Use when tracing what a function does. Same shape as semantex_callers ",
                    "but outbound edges."
                )
                .into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "symbol": { "type": "string", "description": "Symbol name to find callees of" },
                        "depth": { "type": "integer", "description": "Walk depth: 1 (direct) or 2 (transitive). Default 1.", "default": 1 },
                        "path": { "type": "string", "description": "Project path (defaults to cwd)" }
                    },
                    "required": ["symbol"]
                }),
                output_schema: Some(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "callees": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "callee_location": { "type": "object" },
                                    "callee_signature": { "type": "string" },
                                    "edge_kind": { "type": "string" }
                                },
                                "required": ["callee_location", "edge_kind"]
                            }
                        }
                    },
                    "required": ["callees"]
                })),
                annotations: Some(ToolAnnotations::read_only("Forward Call Graph")),
            },
            // -------------------------------------------------------------
            // M4 — semantex_implementations
            // -------------------------------------------------------------
            Tool {
                name: "semantex_implementations".into(),
                title: Some("Trait/Interface Implementations".into()),
                description: concat!(
                    "Find all implementations of a trait, interface, or protocol. ",
                    "Backed by the indexed type-hierarchy edges. ",
                    "Returns one entry per impl with the impl location, concrete type, ",
                    "and the list of method names physically defined inside the impl block ",
                    "(`methods_defined_in_impl`). This is a strict subset of the trait's ",
                    "methods — to compute true overrides, intersect this list against the ",
                    "trait declaration via `semantex_symbol`."
                )
                .into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "trait_or_interface": { "type": "string", "description": "Trait, interface, or abstract base class name" },
                        "path": { "type": "string", "description": "Project path (defaults to cwd)" }
                    },
                    "required": ["trait_or_interface"]
                }),
                output_schema: Some(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "implementations": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "impl_location": { "type": "object" },
                                    "type_name": { "type": "string" },
                                    "methods_defined_in_impl": {
                                        "type": "array",
                                        "items": { "type": "string" },
                                        "description": "Method names physically declared inside this impl block (queried from symbol_defs). NOT the trait's declared method set — intersect against the trait declaration to compute true overrides."
                                    }
                                },
                                "required": ["impl_location", "type_name"]
                            }
                        }
                    },
                    "required": ["implementations"]
                })),
                annotations: Some(ToolAnnotations::read_only("Trait/Interface Implementations")),
            },
            // -------------------------------------------------------------
            // M5 — semantex_examples
            // -------------------------------------------------------------
            Tool {
                name: "semantex_examples".into(),
                title: Some("Pattern Exemplars".into()),
                description: concat!(
                    "Find structurally-confirmed exemplars of a programming pattern ",
                    "(e.g. 'rust.drop_impl', 'rust.tokio_spawn', 'ts.try_catch'). ",
                    "Returns 3 pre-curated examples instead of 10 grep results. ",
                    "Backed by the pattern catalog mined at index time."
                )
                .into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "pattern": { "type": "string", "description": "Pattern name (e.g. 'rust.drop_impl'). Use semantex_search to discover available patterns." },
                        "language": { "type": "string", "description": "Optional language filter (rust, typescript)" },
                        "max": { "type": "integer", "description": "Max exemplars to return (default 3)", "default": 3 },
                        "path": { "type": "string", "description": "Project path (defaults to cwd)" }
                    },
                    "required": ["pattern"]
                }),
                output_schema: Some(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "examples": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "location": { "type": "object" },
                                    "snippet": { "type": "string" },
                                    "pattern": { "type": "string" },
                                    "language": { "type": "string" }
                                },
                                "required": ["location", "snippet"]
                            }
                        }
                    },
                    "required": ["examples"]
                })),
                annotations: Some(ToolAnnotations::read_only("Pattern Exemplars")),
            },
            // -------------------------------------------------------------
            // M6 — semantex_architecture
            // -------------------------------------------------------------
            Tool {
                name: "semantex_architecture".into(),
                title: Some("Architectural Primer".into()),
                description: concat!(
                    "Session-start architectural primer for a codebase. ",
                    "Returns a compact LLM-optimized JSON document with: ",
                    "(1) god_nodes — the top symbols by PageRank centrality, ",
                    "(2) communities — clusters of related chunks with entry points, ",
                    "(3) boundaries — directory-level coupling counts. ",
                    "One call gives an architectural overview without exploring the tree manually. ",
                    "This is the single biggest context-window win for unfamiliar codebases."
                )
                .into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "focus": {
                            "type": "string",
                            "enum": ["god_nodes", "communities", "boundaries"],
                            "description": "Restrict output to one section. Omit for all three."
                        },
                        "path": { "type": "string", "description": "Project path (defaults to cwd)" }
                    }
                }),
                output_schema: Some(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "god_nodes": { "type": "array" },
                        "communities": { "type": "array" },
                        "boundaries": { "type": "array" }
                    }
                })),
                annotations: Some(ToolAnnotations::read_only("Architectural Primer")),
            },
        ]
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

        let rss_before = semantex_core::memory::current_rss_mb();

        // Enforce toolset gating: tools outside the active bundle return
        // a clear error rather than running silently, so HTTP transports
        // (W7) and the CLI's `--toolset` flag give identical behaviour.
        let visible: HashSet<String> = self
            .tools_for_toolset(&self.toolset)
            .into_iter()
            .map(|t| t.name)
            .collect();
        if !visible.contains(tool_name) {
            return JsonRpcResponse::success(
                id,
                serde_json::to_value(ToolCallResult {
                    content: vec![ToolContent {
                        content_type: "text".into(),
                        text: format!(
                            "Tool '{tool_name}' is not available in toolset '{}'. \
                             Available tools: {}",
                            self.toolset,
                            visible.iter().cloned().collect::<Vec<_>>().join(", ")
                        ),
                    }],
                    is_error: Some(true),
                    structured_content: None,
                })
                .expect("ToolCallResult serialization"),
            );
        }

        let result = match tool_name {
            "semantex_agent" => self.tool_agent(&arguments),
            "semantex_search" => self.tool_search(&arguments, writer, progress_token.as_ref()),
            "semantex_index" => self.tool_index(&arguments).map(ToolOutput::text),
            "semantex_status" => self.tool_status(&arguments).map(ToolOutput::text),
            "semantex_health" => self.tool_health(&arguments).map(ToolOutput::text),
            "semantex_validate" => self.tool_validate(&arguments).map(ToolOutput::text),
            "semantex_deep" => self.tool_deep_search(&arguments, writer, progress_token.as_ref()),
            "semantex_symbol" => self.tool_symbol(&arguments),
            "semantex_callers" => self.tool_callers(&arguments),
            "semantex_callees" => self.tool_callees(&arguments),
            "semantex_implementations" => self.tool_implementations(&arguments),
            "semantex_examples" => self.tool_examples(&arguments),
            "semantex_architecture" => self.tool_architecture(&arguments),
            _ => Err(anyhow::anyhow!("Unknown tool: {tool_name}")),
        };

        // Check RSS after every tool call and evict caches if over limit.
        self.check_rss_and_evict();

        if let (Some(before), Some(after)) = (rss_before, semantex_core::memory::current_rss_mb()) {
            let cached = self.cache.lock().len();
            tracing::debug!(
                tool = tool_name,
                rss_before_mb = before,
                rss_after_mb = after,
                cached_searchers = cached,
                limit_mb = self.rss_limit_mb,
                "Memory after tool call"
            );
        }

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
        let queries: Vec<String> = if let Some(arr) = args.get("queries").and_then(|v| v.as_array())
        {
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
        let depth_route: Option<AgentRoute> = match args.get("depth").and_then(|v| v.as_str()) {
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

        let idx_state = Self::detect_state_fast(&path);
        match idx_state {
            IndexState::NotIndexed | IndexState::Stale => {
                self.spawn_background_index(&path);
                // For batch queries, fall back using the first query.
                return Self::do_ripgrep_fallback(&queries[0], &path, 10);
            }
            IndexState::Building => {
                return Self::do_ripgrep_fallback(&queries[0], &path, 10);
            }
            IndexState::Ready => {
                // Silently refresh if index is older than threshold.
                self.maybe_trigger_refresh(&path);
            }
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

        let idx_state = Self::detect_state_fast(&path);
        match idx_state {
            IndexState::Ready => {
                self.maybe_trigger_refresh(&path);
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

    #[tracing::instrument(skip(self, args), fields(tool = "status"))]
    fn tool_status(&self, args: &serde_json::Value) -> Result<String> {
        let explicit_path = args.get("path").and_then(|v| v.as_str()).map(PathBuf::from);

        if let Some(path) = explicit_path {
            // Single-repo detailed status
            Ok(self.repo_status_detail(&path))
        } else {
            // All-repos overview from registry + current directory
            let mut repos: Vec<PathBuf> = registry::read_all();
            // Always include CWD
            if let Ok(cwd) = std::env::current_dir() {
                let cwd = cwd.canonicalize().unwrap_or(cwd);
                if !repos.contains(&cwd) {
                    repos.push(cwd);
                }
            }
            if repos.is_empty() {
                return Ok(
                    "No indexed projects found. Run 'semantex_index' on a project first.".into(),
                );
            }

            let mut lines = vec![
                "Index Status — All Tracked Repos".to_string(),
                "=".repeat(50),
            ];

            for repo in &repos {
                lines.push(String::new());
                lines.push(self.repo_status_detail(repo));
            }

            Ok(lines.join("\n"))
        }
    }

    /// Build a human-readable status block for a single repository.
    fn repo_status_detail(&self, path: &Path) -> String {
        let canonical = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
        let display = canonical.display();

        let idx_state = state::detect(&canonical);
        let state_label = match idx_state {
            IndexState::NotIndexed => "NOT INDEXED",
            IndexState::Building => "BUILDING",
            IndexState::Stale => "STALE (schema mismatch)",
            IndexState::Ready => "READY",
        };

        // Check if our in-process tracker says building (covers the window where
        // the lock file hasn't been created yet)
        let in_process_building = matches!(
            self.index_states.lock().get(&canonical),
            Some(IndexingStatus::Building)
        );
        let state_label = if in_process_building && idx_state != IndexState::Building {
            "BUILDING"
        } else {
            state_label
        };

        let index_dir = SemantexConfig::project_index_dir(&canonical);
        let meta_path = index_dir.join("meta.json");

        if !meta_path.exists() {
            // Trigger indexing if truly not indexed
            if idx_state == IndexState::NotIndexed && !in_process_building {
                self.spawn_background_index(&canonical);
                return format!(
                    "{display}\n  State: {state_label} — indexing started automatically"
                );
            }
            return format!("{display}\n  State: {state_label}");
        }

        let Ok(content) = std::fs::read_to_string(&meta_path) else {
            return format!("{display}\n  State: unreadable meta.json");
        };
        let Ok(meta) = serde_json::from_str::<semantex_core::types::IndexMeta>(&content) else {
            return format!("{display}\n  State: corrupted meta.json — re-indexing");
        };

        let age_str =
            state::index_age_secs(&canonical).map_or_else(|| "unknown".into(), format_age);

        // Trigger background refresh if index is old and ready
        let refresh_note = if idx_state == IndexState::Ready && !in_process_building {
            if self.maybe_trigger_refresh(&canonical) {
                "  (background refresh triggered)\n"
            } else {
                ""
            }
        } else {
            ""
        };

        format!(
            "{display}\n  State: {state_label}{refresh_note}\n  Age:   {age_str}\n  Files: {}\n  Chunks: {}\n  Model: {}",
            meta.file_count, meta.chunk_count, meta.embedding_model
        )
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

        let idx_state = Self::detect_state_fast(&path);
        match idx_state {
            IndexState::NotIndexed | IndexState::Stale => {
                self.spawn_background_index(&path);
                return Ok(ToolOutput::text("Index not ready. Building index in background — deep search requires a complete index. Try again in a moment.".to_string()));
            }
            IndexState::Building => {
                return Ok(ToolOutput::text("Index is currently building. Deep search requires a complete index. Try again in a moment.".to_string()));
            }
            IndexState::Ready => {
                self.maybe_trigger_refresh(&path);
            }
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

    // -------------------------------------------------------------------------
    // M1 — semantex_symbol
    //
    // Exact symbol lookup backed by `symbol_defs` + chunk fan-in/out counts.
    // Returns one entry per matching symbol with location, signature, docstring,
    // semantic role and the count of callers/callees. Optional `kind` argument
    // filters symbol_kind (function, method, class, struct, enum, trait, …).
    // -------------------------------------------------------------------------

    #[tracing::instrument(skip(self, args), fields(tool = "symbol"))]
    fn tool_symbol(&self, args: &serde_json::Value) -> Result<ToolOutput> {
        let name = args
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing required parameter: name"))?;
        let kind_filter = args
            .get("kind")
            .and_then(|v| v.as_str())
            .map(std::string::ToString::to_string);
        let path = resolve_project_path(args)?;

        if !require_index_ready(&path) {
            self.spawn_background_index(&path);
            return Ok(ToolOutput::text(
                "Index not ready. Building in background — retry shortly.".to_string(),
            ));
        }

        let index_dir = SemantexConfig::project_index_dir(&path);
        let store = ChunkStore::open_for_search(&index_dir.join("chunks.db"))
            .context("Failed to open chunk store for semantex_symbol")?;

        let mut matches = store.lookup_symbol_exact(name)?;
        if let Some(ref k) = kind_filter {
            matches.retain(|(_, _, sk)| sk == k);
        }

        if matches.is_empty() {
            let structured = serde_json::json!({ "matches": [] });
            return Ok(ToolOutput {
                text: format!("No symbol named '{name}' found."),
                structured: Some(structured),
            });
        }

        let mut out_matches: Vec<serde_json::Value> = Vec::with_capacity(matches.len());
        for (chunk_id_i64, _file_path, sym_kind) in &matches {
            let chunk_id = *chunk_id_i64 as u64;
            let chunk = store.get_chunk(chunk_id)?;
            let meta = store.get_structured_meta(chunk_id)?;

            let location = serde_json::json!({
                "file": chunk.file_path.display().to_string(),
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "symbol_kind": sym_kind,
            });

            let signature = meta
                .as_ref()
                .and_then(|m| m.signature.clone())
                .unwrap_or_default();
            let docstring = meta
                .as_ref()
                .and_then(|m| m.docstring.clone())
                .unwrap_or_default();
            let semantic_role = meta
                .as_ref()
                .and_then(|m| m.semantic_role.as_ref().map(|r| r.as_label().to_string()))
                .unwrap_or_default();

            let callers_count = store.get_call_edges_to(&[chunk_id])?.len();
            let callees_count = store.get_call_edges_from(&[chunk_id])?.len();

            // Confidence: Extracted when we got both signature and at least one
            // graph edge; Inferred otherwise. Single-match results promote to
            // Extracted so callers can trust an unambiguous lookup.
            let conf = if matches.len() == 1
                || (!signature.is_empty() && (callers_count + callees_count) > 0)
            {
                semantex_core::types::Confidence::Extracted
            } else {
                semantex_core::types::Confidence::Inferred
            };

            out_matches.push(serde_json::json!({
                "location": location,
                "signature": signature,
                "docstring": docstring,
                "semantic_role": semantic_role,
                "callers_count": callers_count,
                "callees_count": callees_count,
                "confidence": conf.label(),
            }));
        }

        let structured = serde_json::json!({ "matches": out_matches });
        let text = serde_json::to_string_pretty(&structured)?;
        Ok(ToolOutput {
            text,
            structured: Some(structured),
        })
    }

    // -------------------------------------------------------------------------
    // M2 — semantex_callers
    //
    // Reverse call-graph walk over `call_graph` edges. Depth 1 = direct callers;
    // depth 2 = callers-of-callers (depth capped at 2 to bound cost).
    // -------------------------------------------------------------------------

    #[tracing::instrument(skip(self, args), fields(tool = "callers"))]
    fn tool_callers(&self, args: &serde_json::Value) -> Result<ToolOutput> {
        let symbol = args
            .get("symbol")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing required parameter: symbol"))?;
        let depth = args
            .get("depth")
            .and_then(serde_json::Value::as_u64)
            .map_or(1u8, |d| d.clamp(1, 2) as u8);
        let path = resolve_project_path(args)?;

        if !require_index_ready(&path) {
            self.spawn_background_index(&path);
            return Ok(ToolOutput::text(
                "Index not ready. Building in background — retry shortly.".to_string(),
            ));
        }

        let index_dir = SemantexConfig::project_index_dir(&path);
        let store = ChunkStore::open_for_search(&index_dir.join("chunks.db"))
            .context("Failed to open chunk store for semantex_callers")?;

        let seeds = store.lookup_symbol_exact(symbol)?;
        if seeds.is_empty() {
            return Ok(ToolOutput {
                text: format!("No symbol named '{symbol}' found."),
                structured: Some(serde_json::json!({ "callers": [] })),
            });
        }
        let seed_ids: Vec<u64> = seeds.iter().map(|(id, _, _)| *id as u64).collect();

        // Walk one hop.
        let mut caller_ids: HashSet<u64> = HashSet::new();
        let hop1 = store.get_call_edges_to(&seed_ids)?;
        for (_, caller_id) in &hop1 {
            caller_ids.insert(*caller_id);
        }
        // Optional second hop.
        if depth >= 2 {
            let hop1_vec: Vec<u64> = caller_ids.iter().copied().collect();
            if !hop1_vec.is_empty() {
                let hop2 = store.get_call_edges_to(&hop1_vec)?;
                for (_, caller_id) in &hop2 {
                    // Don't include the seed symbols themselves.
                    if !seed_ids.contains(caller_id) {
                        caller_ids.insert(*caller_id);
                    }
                }
            }
        }
        // Don't include the seed itself.
        for sid in &seed_ids {
            caller_ids.remove(sid);
        }
        let ids_vec: Vec<u64> = caller_ids.into_iter().collect();
        let chunks = store.get_chunks(&ids_vec)?;
        let callers = chunks_to_graph_entries(
            &store,
            &chunks,
            "caller_location",
            "caller_signature",
            "call",
        )?;
        let structured = serde_json::json!({ "callers": callers });
        let text = serde_json::to_string_pretty(&structured)?;
        Ok(ToolOutput {
            text,
            structured: Some(structured),
        })
    }

    // -------------------------------------------------------------------------
    // M3 — semantex_callees
    //
    // Forward call-graph walk over `call_graph` edges. Mirrors `tool_callers`.
    // -------------------------------------------------------------------------

    #[tracing::instrument(skip(self, args), fields(tool = "callees"))]
    fn tool_callees(&self, args: &serde_json::Value) -> Result<ToolOutput> {
        let symbol = args
            .get("symbol")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing required parameter: symbol"))?;
        let depth = args
            .get("depth")
            .and_then(serde_json::Value::as_u64)
            .map_or(1u8, |d| d.clamp(1, 2) as u8);
        let path = resolve_project_path(args)?;

        if !require_index_ready(&path) {
            self.spawn_background_index(&path);
            return Ok(ToolOutput::text(
                "Index not ready. Building in background — retry shortly.".to_string(),
            ));
        }

        let index_dir = SemantexConfig::project_index_dir(&path);
        let store = ChunkStore::open_for_search(&index_dir.join("chunks.db"))
            .context("Failed to open chunk store for semantex_callees")?;

        let seeds = store.lookup_symbol_exact(symbol)?;
        if seeds.is_empty() {
            return Ok(ToolOutput {
                text: format!("No symbol named '{symbol}' found."),
                structured: Some(serde_json::json!({ "callees": [] })),
            });
        }
        let seed_ids: Vec<u64> = seeds.iter().map(|(id, _, _)| *id as u64).collect();

        let mut callee_ids: HashSet<u64> = HashSet::new();
        let hop1 = store.get_call_edges_from(&seed_ids)?;
        for (_, callee_id) in &hop1 {
            callee_ids.insert(*callee_id);
        }
        if depth >= 2 {
            let hop1_vec: Vec<u64> = callee_ids.iter().copied().collect();
            if !hop1_vec.is_empty() {
                let hop2 = store.get_call_edges_from(&hop1_vec)?;
                for (_, callee_id) in &hop2 {
                    if !seed_ids.contains(callee_id) {
                        callee_ids.insert(*callee_id);
                    }
                }
            }
        }
        for sid in &seed_ids {
            callee_ids.remove(sid);
        }
        let ids_vec: Vec<u64> = callee_ids.into_iter().collect();
        let chunks = store.get_chunks(&ids_vec)?;
        let callees = chunks_to_graph_entries(
            &store,
            &chunks,
            "callee_location",
            "callee_signature",
            "call",
        )?;
        let structured = serde_json::json!({ "callees": callees });
        let text = serde_json::to_string_pretty(&structured)?;
        Ok(ToolOutput {
            text,
            structured: Some(structured),
        })
    }

    // -------------------------------------------------------------------------
    // M4 — semantex_implementations
    //
    // Find all impls of a trait/interface/protocol via `type_hierarchy`.
    // A trait's children (impls) are stored as `child_chunk -> parent_chunk`
    // edges; we resolve all chunks whose `parent_name` matches the trait.
    // -------------------------------------------------------------------------

    #[tracing::instrument(skip(self, args), fields(tool = "implementations"))]
    fn tool_implementations(&self, args: &serde_json::Value) -> Result<ToolOutput> {
        let trait_name = args
            .get("trait_or_interface")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing required parameter: trait_or_interface"))?;
        let path = resolve_project_path(args)?;

        if !require_index_ready(&path) {
            self.spawn_background_index(&path);
            return Ok(ToolOutput::text(
                "Index not ready. Building in background — retry shortly.".to_string(),
            ));
        }

        let index_dir = SemantexConfig::project_index_dir(&path);
        let db_path = index_dir.join("chunks.db");
        let store = ChunkStore::open_for_search(&db_path)
            .context("Failed to open chunk store for semantex_implementations")?;

        // First locate trait definition chunks to anchor the hierarchy walk.
        let trait_defs = store.lookup_symbol_exact(trait_name)?;
        let trait_chunk_ids: Vec<u64> = trait_defs.iter().map(|(id, _, _)| *id as u64).collect();

        // Direct SQL on `type_hierarchy`: pull impls by parent_name OR by
        // parent_chunk membership in trait_chunk_ids (covers post-resolve case).
        let impl_chunk_ids = query_implementations(&db_path, trait_name, &trait_chunk_ids)?;

        if impl_chunk_ids.is_empty() {
            return Ok(ToolOutput {
                text: format!("No implementations of '{trait_name}' found."),
                structured: Some(serde_json::json!({ "implementations": [] })),
            });
        }

        let chunks = store.get_chunks(&impl_chunk_ids)?;
        let mut entries: Vec<serde_json::Value> = Vec::with_capacity(chunks.len());
        for chunk in &chunks {
            let meta = store.get_structured_meta(chunk.id)?;
            let type_name = meta
                .as_ref()
                .and_then(|m| m.name.clone())
                .unwrap_or_default();

            // Methods physically defined inside this impl block. Derived from
            // `symbol_defs` rows whose chunk falls within the impl chunk's
            // line range in the same file. This is narrower than "trait
            // method overrides" — pairing this list against the trait's
            // declared methods (to compute true overrides) requires loading
            // the trait declaration, which is left to a follow-up. The
            // previous implementation used `meta.calls` which is the set of
            // *outgoing call targets* inside the impl (helpers, log macros,
            // …) — semantically unrelated to overridden methods (Finding 11).
            let methods_defined_in_impl = query_methods_in_impl(
                &db_path,
                &chunk.file_path.to_string_lossy(),
                chunk.start_line,
                chunk.end_line,
            )
            .unwrap_or_default();

            entries.push(serde_json::json!({
                "impl_location": {
                    "file": chunk.file_path.display().to_string(),
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                },
                "type_name": type_name,
                "methods_defined_in_impl": methods_defined_in_impl,
            }));
        }

        let structured = serde_json::json!({ "implementations": entries });
        let text = serde_json::to_string_pretty(&structured)?;
        Ok(ToolOutput {
            text,
            structured: Some(structured),
        })
    }

    // -------------------------------------------------------------------------
    // M5 — semantex_examples
    //
    // Pattern-catalog-backed exemplar finder. Queries the `pattern_matches`
    // table for chunks that fired the given pattern; returns up to `max` with
    // chunk snippets. The catalog is mined at index time so results are
    // structurally confirmed, not raw grep hits.
    // -------------------------------------------------------------------------

    #[tracing::instrument(skip(self, args), fields(tool = "examples"))]
    fn tool_examples(&self, args: &serde_json::Value) -> Result<ToolOutput> {
        let pattern = args
            .get("pattern")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing required parameter: pattern"))?;
        let language = args
            .get("language")
            .and_then(|v| v.as_str())
            .map(std::string::ToString::to_string);
        let max = args
            .get("max")
            .and_then(serde_json::Value::as_u64)
            .map_or(3usize, |m| m.clamp(1, 20) as usize);
        let path = resolve_project_path(args)?;

        if !require_index_ready(&path) {
            self.spawn_background_index(&path);
            return Ok(ToolOutput::text(
                "Index not ready. Building in background — retry shortly.".to_string(),
            ));
        }

        let index_dir = SemantexConfig::project_index_dir(&path);
        let db_path = index_dir.join("chunks.db");
        let store = ChunkStore::open_for_search(&db_path)
            .context("Failed to open chunk store for semantex_examples")?;

        let example_rows = query_pattern_examples(&db_path, pattern, language.as_deref(), max)?;
        if example_rows.is_empty() {
            return Ok(ToolOutput {
                text: format!("No exemplars for pattern '{pattern}' found."),
                structured: Some(serde_json::json!({ "examples": [] })),
            });
        }

        let chunk_ids: Vec<u64> = example_rows.iter().map(|(id, _, _)| *id).collect();
        let chunks = store.get_chunks(&chunk_ids)?;
        let chunk_by_id: HashMap<u64, &semantex_core::types::Chunk> =
            chunks.iter().map(|c| (c.id, c)).collect();

        let mut entries: Vec<serde_json::Value> = Vec::with_capacity(example_rows.len());
        for (cid, pat_name, lang) in &example_rows {
            let Some(chunk) = chunk_by_id.get(cid) else {
                continue;
            };
            let snippet = make_snippet_mcp(&chunk.content, &chunk.chunk_type);
            entries.push(serde_json::json!({
                "location": {
                    "file": chunk.file_path.display().to_string(),
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                },
                "snippet": snippet,
                "pattern": pat_name,
                "language": lang,
            }));
        }

        let structured = serde_json::json!({ "examples": entries });
        let text = serde_json::to_string_pretty(&structured)?;
        Ok(ToolOutput {
            text,
            structured: Some(structured),
        })
    }

    // -------------------------------------------------------------------------
    // M6 — semantex_architecture
    //
    // Compact LLM-optimized architectural primer combining PageRank, simple
    // connected-component community detection, and directory-level boundary
    // edge counts. One call replaces the manual exploration loop at the
    // start of an unfamiliar codebase. Optional `focus` arg restricts output
    // to one section: god_nodes | communities | boundaries.
    // -------------------------------------------------------------------------

    #[tracing::instrument(skip(self, args), fields(tool = "architecture"))]
    fn tool_architecture(&self, args: &serde_json::Value) -> Result<ToolOutput> {
        let focus = args
            .get("focus")
            .and_then(|v| v.as_str())
            .map(str::to_string);
        let path = resolve_project_path(args)?;

        if !require_index_ready(&path) {
            self.spawn_background_index(&path);
            return Ok(ToolOutput::text(
                "Index not ready. Building in background — retry shortly.".to_string(),
            ));
        }

        let index_dir = SemantexConfig::project_index_dir(&path);
        let db_path = index_dir.join("chunks.db");
        let store = ChunkStore::open_for_search(&db_path)
            .context("Failed to open chunk store for semantex_architecture")?;

        // 1. God nodes — top-N chunks by PageRank centrality.
        let god_nodes = if focus.as_deref().is_none_or(|f| f == "god_nodes") {
            let top = query_top_centrality(&db_path, 10)?;
            let mut entries: Vec<serde_json::Value> = Vec::new();
            for (chunk_id, centrality) in &top {
                let Ok(chunk) = store.get_chunk(*chunk_id) else {
                    continue;
                };
                let meta = store.get_structured_meta(*chunk_id)?;
                let (symbol, role) = if let Some(ref m) = meta {
                    let name = m
                        .name
                        .clone()
                        .unwrap_or_else(|| chunk.file_path.display().to_string());
                    let role_str = m.semantic_role.as_ref().map_or_else(
                        || m.kind.clone().unwrap_or_default(),
                        |r| r.as_label().to_string(),
                    );
                    (name, role_str)
                } else {
                    (chunk.file_path.display().to_string(), "module".to_string())
                };
                entries.push(serde_json::json!({
                    "symbol": symbol,
                    "centrality": centrality,
                    "role": role,
                    "location": {
                        "file": chunk.file_path.display().to_string(),
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                    },
                }));
            }
            entries
        } else {
            Vec::new()
        };

        // 2. Communities — connected components over call + import edges.
        let communities = if focus.as_deref().is_none_or(|f| f == "communities") {
            compute_communities(&store, &db_path)?
        } else {
            Vec::new()
        };

        // 3. Boundaries — directory-level coupling: count cross-dir edges.
        let boundaries = if focus.as_deref().is_none_or(|f| f == "boundaries") {
            compute_boundaries_from_path(&db_path)?
        } else {
            Vec::new()
        };

        let mut structured = serde_json::Map::new();
        if focus.as_deref().is_none_or(|f| f == "god_nodes") {
            structured.insert("god_nodes".into(), serde_json::Value::Array(god_nodes));
        }
        if focus.as_deref().is_none_or(|f| f == "communities") {
            structured.insert("communities".into(), serde_json::Value::Array(communities));
        }
        if focus.as_deref().is_none_or(|f| f == "boundaries") {
            structured.insert("boundaries".into(), serde_json::Value::Array(boundaries));
        }
        let structured = serde_json::Value::Object(structured);
        let text = serde_json::to_string_pretty(&structured)?;
        Ok(ToolOutput {
            text,
            structured: Some(structured),
        })
    }
}

// =============================================================================
// Helpers for the new M1-M6 tools
// =============================================================================

/// Resolve `path` from a JSON args object, defaulting to the current directory
/// and canonicalizing where possible.
fn resolve_project_path(args: &serde_json::Value) -> Result<PathBuf> {
    let raw = args.get("path").and_then(|v| v.as_str()).map_or_else(
        || std::env::current_dir().context("Failed to determine current directory"),
        |p| Ok(PathBuf::from(p)),
    )?;
    Ok(raw.canonicalize().unwrap_or(raw))
}

/// Gate every M1-M6 handler behind a warm-state-aware readiness check.
/// Returns true iff the index is Ready (warm or detected).
fn require_index_ready(path: &Path) -> bool {
    McpServer::detect_state_fast(path) == IndexState::Ready
}

/// Convert a list of chunks into a uniform graph-entry shape used by M2/M3.
fn chunks_to_graph_entries(
    store: &ChunkStore,
    chunks: &[semantex_core::types::Chunk],
    location_key: &str,
    signature_key: &str,
    edge_kind: &str,
) -> Result<Vec<serde_json::Value>> {
    let mut out = Vec::with_capacity(chunks.len());
    for chunk in chunks {
        let meta = store.get_structured_meta(chunk.id)?;
        let signature = meta
            .as_ref()
            .and_then(|m| m.signature.clone())
            .unwrap_or_default();
        let location = serde_json::json!({
            "file": chunk.file_path.display().to_string(),
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
        });
        out.push(serde_json::json!({
            location_key: location,
            signature_key: signature,
            "edge_kind": edge_kind,
        }));
    }
    Ok(out)
}

/// Pull implementation chunk IDs for a trait/interface from `type_hierarchy`.
///
/// Two paths:
/// 1. By parent_name (string match) — works even when hierarchy resolution
///    hasn't connected `parent_chunk` yet.
/// 2. By parent_chunk membership — works post-resolve.
fn query_implementations(
    db_path: &Path,
    trait_name: &str,
    trait_chunk_ids: &[u64],
) -> Result<Vec<u64>> {
    use rusqlite::Connection;
    let conn = Connection::open_with_flags(db_path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
        .with_context(|| format!("Failed to open {} read-only", db_path.display()))?;

    let mut ids: HashSet<u64> = HashSet::new();

    // Path 1: parent_name match.
    {
        let mut stmt = conn.prepare(
            "SELECT child_chunk FROM type_hierarchy
             WHERE parent_name = ?1 AND child_chunk IS NOT NULL",
        )?;
        let rows = stmt.query_map([trait_name], |row| row.get::<_, i64>(0))?;
        for r in rows {
            ids.insert(r? as u64);
        }
    }
    // Path 2: parent_chunk membership.
    if !trait_chunk_ids.is_empty() {
        let placeholders: String = trait_chunk_ids
            .iter()
            .map(|_| "?")
            .collect::<Vec<_>>()
            .join(",");
        let sql = format!(
            "SELECT child_chunk FROM type_hierarchy \
             WHERE parent_chunk IN ({placeholders}) AND child_chunk IS NOT NULL"
        );
        let mut stmt = conn.prepare(&sql)?;
        let id_params: Vec<rusqlite::types::Value> = trait_chunk_ids
            .iter()
            .map(|id| rusqlite::types::Value::Integer(*id as i64))
            .collect();
        let param_refs: Vec<&dyn rusqlite::types::ToSql> = id_params
            .iter()
            .map(|v| v as &dyn rusqlite::types::ToSql)
            .collect();
        let rows = stmt.query_map(param_refs.as_slice(), |row| row.get::<_, i64>(0))?;
        for r in rows {
            ids.insert(r? as u64);
        }
    }

    Ok(ids.into_iter().collect())
}

/// Return the names of methods physically defined inside an impl block.
///
/// Backs Finding 11's accurate replacement for the old `method_overrides`
/// field. Strategy: join `symbol_defs` with `chunks` and select rows whose
/// chunk lives in the same file as the impl and whose `[start_line, end_line]`
/// range is fully contained within the impl chunk's range. The `symbol_kind`
/// filter (`'method'`) excludes nested helper functions / type aliases / etc.
///
/// We intentionally return method names only (not a fully-qualified path) —
/// callers seeking the method's source can issue `semantex_symbol(name)` and
/// scope to the impl's file.
fn query_methods_in_impl(
    db_path: &Path,
    impl_file_path: &str,
    impl_start_line: u32,
    impl_end_line: u32,
) -> Result<Vec<String>> {
    use rusqlite::Connection;
    let conn = Connection::open_with_flags(db_path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
        .with_context(|| format!("Failed to open {} read-only", db_path.display()))?;

    let mut stmt = conn.prepare(
        "SELECT DISTINCT sd.symbol_name
         FROM symbol_defs sd
         JOIN chunks c ON c.id = sd.chunk_id
         WHERE sd.symbol_kind = 'method'
           AND c.file_path = ?1
           AND c.start_line >= ?2
           AND c.end_line <= ?3
         ORDER BY c.start_line ASC",
    )?;
    let rows = stmt.query_map(
        rusqlite::params![
            impl_file_path,
            i64::from(impl_start_line),
            i64::from(impl_end_line)
        ],
        |row| row.get::<_, String>(0),
    )?;

    let mut out: Vec<String> = Vec::new();
    for r in rows {
        out.push(r?);
    }
    Ok(out)
}

/// Pull (chunk_id, pattern_name, language) rows from `pattern_matches`.
/// Optional `language` filter scopes results to one catalog.
fn query_pattern_examples(
    db_path: &Path,
    pattern: &str,
    language: Option<&str>,
    max: usize,
) -> Result<Vec<(u64, String, String)>> {
    use rusqlite::Connection;
    let conn = Connection::open_with_flags(db_path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
        .with_context(|| format!("Failed to open {} read-only", db_path.display()))?;

    let mut out: Vec<(u64, String, String)> = Vec::new();
    if let Some(lang) = language {
        let mut stmt = conn.prepare(
            "SELECT chunk_id, pattern_name, language FROM pattern_matches \
             WHERE pattern_name = ?1 AND language = ?2 LIMIT ?3",
        )?;
        let rows = stmt.query_map(rusqlite::params![pattern, lang, max as i64], |row| {
            Ok((
                row.get::<_, i64>(0)? as u64,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
            ))
        })?;
        for r in rows {
            out.push(r?);
        }
    } else {
        let mut stmt = conn.prepare(
            "SELECT chunk_id, pattern_name, language FROM pattern_matches \
             WHERE pattern_name = ?1 LIMIT ?2",
        )?;
        let rows = stmt.query_map(rusqlite::params![pattern, max as i64], |row| {
            Ok((
                row.get::<_, i64>(0)? as u64,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
            ))
        })?;
        for r in rows {
            out.push(r?);
        }
    }
    Ok(out)
}

/// Top-N chunks ordered by `structural_centrality` desc.
fn query_top_centrality(db_path: &Path, n: usize) -> Result<Vec<(u64, f64)>> {
    use rusqlite::Connection;
    let conn = Connection::open_with_flags(db_path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
        .with_context(|| format!("Failed to open {} read-only", db_path.display()))?;
    let mut stmt = conn.prepare(
        "SELECT chunk_id, structural_centrality FROM chunk_centrality \
         ORDER BY structural_centrality DESC LIMIT ?1",
    )?;
    let rows = stmt.query_map([n as i64], |row| {
        Ok((row.get::<_, i64>(0)? as u64, row.get::<_, f64>(1)?))
    })?;
    let mut out = Vec::with_capacity(n);
    for r in rows {
        out.push(r?);
    }
    Ok(out)
}

/// Detect simple communities via BFS over call edges from the top-centrality
/// seed set. Returns up to 5 largest components, each as
/// `{label, size, members, entry_points}`. Members are file-path samples.
fn compute_communities(store: &ChunkStore, db_path: &Path) -> Result<Vec<serde_json::Value>> {
    // Seeds: top 200 chunks by PageRank centrality. Bounded so we don't walk
    // the whole graph in giant repos.
    let seeds = match query_top_centrality(db_path, 200) {
        Ok(s) if !s.is_empty() => s,
        _ => return Ok(Vec::new()),
    };
    let seed_ids: Vec<u64> = seeds.iter().map(|(id, _)| *id).collect();

    // Pull edges incident to seed_ids.
    let call_out = store.get_call_edges_from(&seed_ids)?;
    let call_in = store.get_call_edges_to(&seed_ids)?;
    let mut adj: HashMap<u64, HashSet<u64>> = HashMap::new();
    let add_edge = |a: u64, b: u64, adj: &mut HashMap<u64, HashSet<u64>>| {
        adj.entry(a).or_default().insert(b);
        adj.entry(b).or_default().insert(a);
    };
    for sid in &seed_ids {
        adj.entry(*sid).or_default();
    }
    for (a, b) in &call_out {
        add_edge(*a, *b, &mut adj);
    }
    for (a, b) in &call_in {
        add_edge(*a, *b, &mut adj);
    }

    // BFS to enumerate components.
    let mut visited: HashSet<u64> = HashSet::new();
    let mut components: Vec<Vec<u64>> = Vec::new();
    for &node in adj.keys() {
        if visited.contains(&node) {
            continue;
        }
        let mut component = Vec::new();
        let mut stack = vec![node];
        while let Some(n) = stack.pop() {
            if !visited.insert(n) {
                continue;
            }
            component.push(n);
            if let Some(neighbors) = adj.get(&n) {
                for &nb in neighbors {
                    if !visited.contains(&nb) {
                        stack.push(nb);
                    }
                }
            }
        }
        components.push(component);
    }

    // Sort by size, keep top 5.
    components.sort_by_key(|c| std::cmp::Reverse(c.len()));
    components.truncate(5);

    // Score map for picking entry points (top-centrality member).
    let score_by_id: HashMap<u64, f64> = seeds.iter().copied().collect();

    let mut out = Vec::with_capacity(components.len());
    for (idx, component) in components.iter().enumerate() {
        if component.is_empty() {
            continue;
        }
        // Sample up to 8 members for the response (file paths).
        let mut member_paths: Vec<String> = Vec::new();
        let mut member_seen: HashSet<String> = HashSet::new();
        let mut entry_points: Vec<serde_json::Value> = Vec::new();

        // Sort component members by centrality desc; pick top 3 as entry points.
        let mut sorted: Vec<u64> = component.clone();
        sorted.sort_by(|a, b| {
            score_by_id
                .get(b)
                .partial_cmp(&score_by_id.get(a))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for &cid in sorted.iter().take(component.len().min(20)) {
            if let Ok(chunk) = store.get_chunk(cid) {
                let p = chunk.file_path.display().to_string();
                if member_seen.insert(p.clone()) {
                    member_paths.push(p);
                }
                if entry_points.len() < 3 {
                    let meta = store.get_structured_meta(cid).ok().flatten();
                    let name = meta
                        .and_then(|m| m.name)
                        .unwrap_or_else(|| chunk.file_path.display().to_string());
                    entry_points.push(serde_json::json!({
                        "symbol": name,
                        "location": {
                            "file": chunk.file_path.display().to_string(),
                            "start_line": chunk.start_line,
                            "end_line": chunk.end_line,
                        },
                    }));
                }
            }
            if member_paths.len() >= 8 {
                break;
            }
        }

        out.push(serde_json::json!({
            "label": format!("community-{}", idx + 1),
            "size": component.len(),
            "members": member_paths,
            "entry_points": entry_points,
        }));
    }

    Ok(out)
}

/// Count edges that cross top-level directory boundaries. Returns up to 25
/// `{from, to, edge_count}` entries sorted by edge_count desc.
fn compute_boundaries_from_path(db_path: &Path) -> Result<Vec<serde_json::Value>> {
    use rusqlite::Connection;
    let conn = Connection::open_with_flags(db_path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
        .with_context(|| format!("Failed to open {} read-only", db_path.display()))?;

    let mut counts: HashMap<(String, String), u64> = HashMap::new();
    {
        let mut stmt = conn.prepare("SELECT importer_path, imported_path FROM module_edges")?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?;
        for r in rows {
            let (a, b) = r?;
            let da = top_level_dir(&a);
            let db = top_level_dir(&b);
            if da == db {
                continue; // intra-directory edge
            }
            *counts.entry((da, db)).or_insert(0) += 1;
        }
    }

    let mut sorted: Vec<((String, String), u64)> = counts.into_iter().collect();
    sorted.sort_by_key(|entry| std::cmp::Reverse(entry.1));
    sorted.truncate(25);

    Ok(sorted
        .into_iter()
        .map(|((from, to), n)| {
            serde_json::json!({
                "from": from,
                "to": to,
                "edge_count": n,
            })
        })
        .collect())
}

/// First path component (top-level directory). For `crates/foo/bar.rs` → `crates`.
/// Empty for files at the root.
fn top_level_dir(p: &str) -> String {
    let path = Path::new(p);
    path.components()
        .next()
        .and_then(|c| c.as_os_str().to_str())
        .unwrap_or("")
        .to_string()
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
    use super::{DEFAULT_TOOLSET, McpServer, TOOLSETS, apply_focus, top_level_dir};
    use semantex_core::config::SemantexConfig;
    use semantex_core::index::state::IndexState;

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

    // ─────────────────────────────────────────────────────────────────────
    // I3 — toolset bundle tests
    // ─────────────────────────────────────────────────────────────────────

    fn make_server(toolset: &str) -> McpServer {
        let config = SemantexConfig::default();
        McpServer::with_toolset(config, toolset)
    }

    #[test]
    fn toolsets_constants_present() {
        assert_eq!(DEFAULT_TOOLSET, "all");
        assert!(TOOLSETS.contains(&"core"));
        assert!(TOOLSETS.contains(&"structural"));
        assert!(TOOLSETS.contains(&"all"));
        assert_eq!(TOOLSETS.len(), 3);
    }

    #[test]
    fn toolset_all_exposes_thirteen_tools() {
        let server = make_server("all");
        let tools = server.tools_for_toolset("all");
        assert_eq!(
            tools.len(),
            13,
            "toolset 'all' must expose 13 tools, got {}",
            tools.len()
        );
    }

    #[test]
    fn toolset_core_exposes_four_tools() {
        let server = make_server("core");
        let tools = server.tools_for_toolset("core");
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(tools.len(), 4, "core bundle must have 4 tools: {names:?}");
        assert!(names.contains(&"semantex_search"));
        assert!(names.contains(&"semantex_deep"));
        assert!(names.contains(&"semantex_agent"));
        assert!(names.contains(&"semantex_symbol"));
    }

    #[test]
    fn toolset_structural_exposes_five_tools() {
        let server = make_server("structural");
        let tools = server.tools_for_toolset("structural");
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(
            tools.len(),
            5,
            "structural bundle must have 5 tools: {names:?}"
        );
        for required in &[
            "semantex_symbol",
            "semantex_callers",
            "semantex_callees",
            "semantex_implementations",
            "semantex_architecture",
        ] {
            assert!(
                names.contains(required),
                "structural bundle missing {required}"
            );
        }
    }

    #[test]
    fn unknown_toolset_falls_back_to_all() {
        let server = make_server("nonsense");
        assert_eq!(server.toolset(), "all");
        let tools = server.tools_for_toolset("nonsense");
        // Filter falls through to `all` (which returns the full surface).
        assert_eq!(tools.len(), 13);
    }

    #[test]
    fn new_tools_registered_with_required_schemas() {
        let server = make_server("all");
        let tools = server.tools_for_toolset("all");
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        for required in &[
            "semantex_symbol",
            "semantex_callers",
            "semantex_callees",
            "semantex_implementations",
            "semantex_examples",
            "semantex_architecture",
        ] {
            assert!(
                names.contains(required),
                "M1-M6 tool {required} missing from registry"
            );
            let tool = tools.iter().find(|t| t.name == *required).unwrap();
            // Each new tool must declare an input schema with `type: object`.
            assert_eq!(
                tool.input_schema.get("type").and_then(|v| v.as_str()),
                Some("object"),
                "{required} should have type:object input schema"
            );
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // M6 helper test — top_level_dir
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn top_level_dir_extracts_first_component() {
        assert_eq!(top_level_dir("crates/semantex-mcp/src/server.rs"), "crates");
        assert_eq!(top_level_dir("src/main.rs"), "src");
        assert_eq!(top_level_dir("README.md"), "README.md");
        assert_eq!(top_level_dir(""), "");
    }

    // ─────────────────────────────────────────────────────────────────────
    // M1-M5 — handler behaviour against an in-memory test index
    // ─────────────────────────────────────────────────────────────────────

    use semantex_core::index::storage::ChunkStore;
    use semantex_core::types::{AstNodeKind, Chunk, ChunkType};
    use std::path::PathBuf;

    fn make_test_chunk(id: u64, file: &str, name: &str, kind: AstNodeKind, content: &str) -> Chunk {
        let _ = id;
        Chunk {
            id: 0,
            file_path: PathBuf::from(file),
            start_line: 1,
            end_line: content.lines().count() as u32,
            content: content.to_string(),
            chunk_type: ChunkType::AstNode {
                name: name.to_string(),
                kind,
                language: "rust".to_string(),
                structured_meta: None,
            },
        }
    }

    fn build_minimal_index() -> (tempfile::TempDir, PathBuf) {
        use rusqlite::Connection;
        let dir = tempfile::tempdir().expect("tempdir");
        let semantex_dir = dir.path().join(".semantex");
        std::fs::create_dir_all(&semantex_dir).unwrap();
        let db_path = semantex_dir.join("chunks.db");

        // Open via ChunkStore::open to run init_schema, then close.
        {
            let store = ChunkStore::open(&db_path).expect("open store");
            let _ = store
                .insert_chunk(
                    &make_test_chunk(1, "src/a.rs", "foo", AstNodeKind::Function, "fn foo() {}"),
                    1234,
                    0,
                )
                .expect("insert foo");
            let _ = store
                .insert_chunk(
                    &make_test_chunk(
                        2,
                        "src/b.rs",
                        "bar",
                        AstNodeKind::Function,
                        "fn bar() { foo(); }",
                    ),
                    5678,
                    0,
                )
                .expect("insert bar");
            store
                .insert_symbol_def(1, "foo", "function", "src/a.rs")
                .expect("symbol_defs foo");
            store
                .insert_symbol_def(2, "bar", "function", "src/b.rs")
                .expect("symbol_defs bar");
            store
                .store_call_graph_edge(2, "foo", Some(1))
                .expect("call_graph bar→foo");
        }

        // Required aux tables for M5/M6 — create them manually.
        {
            let conn = Connection::open(&db_path).unwrap();
            conn.execute_batch(
                "CREATE TABLE IF NOT EXISTS pattern_matches (
                    chunk_id INTEGER NOT NULL,
                    pattern_name TEXT NOT NULL,
                    language TEXT NOT NULL,
                    description TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    PRIMARY KEY (chunk_id, pattern_name)
                );
                CREATE TABLE IF NOT EXISTS chunk_centrality (
                    chunk_id INTEGER PRIMARY KEY,
                    structural_centrality REAL NOT NULL
                );
                INSERT INTO pattern_matches(chunk_id, pattern_name, language, description, file_path)
                    VALUES (1, 'rust.toy_pattern', 'rust', 'toy pattern for test', 'src/a.rs');
                INSERT INTO chunk_centrality(chunk_id, structural_centrality)
                    VALUES (1, 0.9), (2, 0.1);
                ",
            )
            .unwrap();
        }

        // Write a valid meta.json so state::detect returns Ready.
        let now_secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let meta = semantex_core::types::IndexMeta {
            schema_version: semantex_core::types::IndexMeta::CURRENT_SCHEMA_VERSION,
            project_path: dir.path().to_path_buf(),
            created_at: now_secs.to_string(),
            updated_at: now_secs.to_string(),
            file_count: 2,
            chunk_count: 2,
            embedding_model: "test".to_string(),
            embedding_dim: 96,
        };
        std::fs::write(
            semantex_dir.join("meta.json"),
            serde_json::to_string(&meta).unwrap(),
        )
        .unwrap();

        let project = dir.path().to_path_buf();
        (dir, project)
    }

    #[test]
    fn tool_symbol_missing_argument_errors() {
        let server = make_server("all");
        let result = server.tool_symbol(&serde_json::json!({}));
        assert!(result.is_err(), "missing 'name' should error");
    }

    #[test]
    fn tool_symbol_not_found_returns_empty_matches() {
        let (_dir, project) = build_minimal_index();
        let server = make_server("all");
        let out = server
            .tool_symbol(&serde_json::json!({
                "name": "no_such_symbol_anywhere",
                "path": project.display().to_string(),
            }))
            .expect("call");
        let s = out.structured.expect("structured");
        let matches = s["matches"].as_array().expect("matches array");
        assert!(matches.is_empty(), "should have no matches: {matches:?}");
    }

    #[test]
    fn tool_symbol_finds_known_symbol() {
        let (_dir, project) = build_minimal_index();
        let server = make_server("all");
        let out = server
            .tool_symbol(&serde_json::json!({
                "name": "foo",
                "path": project.display().to_string(),
            }))
            .expect("call");
        let s = out.structured.expect("structured");
        let matches = s["matches"].as_array().expect("matches array");
        assert_eq!(matches.len(), 1, "should find foo once: {matches:?}");
        let m = &matches[0];
        assert_eq!(m["location"]["file"].as_str(), Some("src/a.rs"));
        // bar calls foo, so callers_count should be >= 1
        assert!(m["callers_count"].as_u64().unwrap_or(0) >= 1);
    }

    #[test]
    fn tool_callers_finds_caller() {
        let (_dir, project) = build_minimal_index();
        let server = make_server("all");
        let out = server
            .tool_callers(&serde_json::json!({
                "symbol": "foo",
                "path": project.display().to_string(),
            }))
            .expect("call");
        let s = out.structured.expect("structured");
        let callers = s["callers"].as_array().expect("callers array");
        assert!(
            !callers.is_empty(),
            "bar should appear as a caller of foo: {callers:?}"
        );
    }

    #[test]
    fn tool_callees_finds_callee() {
        let (_dir, project) = build_minimal_index();
        let server = make_server("all");
        let out = server
            .tool_callees(&serde_json::json!({
                "symbol": "bar",
                "path": project.display().to_string(),
            }))
            .expect("call");
        let s = out.structured.expect("structured");
        let callees = s["callees"].as_array().expect("callees array");
        assert!(
            !callees.is_empty(),
            "foo should appear as a callee of bar: {callees:?}"
        );
    }

    #[test]
    fn tool_examples_returns_pattern_match() {
        let (_dir, project) = build_minimal_index();
        let server = make_server("all");
        let out = server
            .tool_examples(&serde_json::json!({
                "pattern": "rust.toy_pattern",
                "path": project.display().to_string(),
            }))
            .expect("call");
        let s = out.structured.expect("structured");
        let examples = s["examples"].as_array().expect("examples array");
        assert!(
            !examples.is_empty(),
            "toy_pattern should yield examples: {examples:?}"
        );
        assert_eq!(examples[0]["pattern"].as_str(), Some("rust.toy_pattern"));
    }

    #[test]
    fn tool_examples_unknown_pattern_returns_empty() {
        let (_dir, project) = build_minimal_index();
        let server = make_server("all");
        let out = server
            .tool_examples(&serde_json::json!({
                "pattern": "rust.no_such_pattern_at_all",
                "path": project.display().to_string(),
            }))
            .expect("call");
        let s = out.structured.expect("structured");
        let examples = s["examples"].as_array().expect("examples array");
        assert!(examples.is_empty());
    }

    #[test]
    fn tool_architecture_full_response_has_three_sections() {
        let (_dir, project) = build_minimal_index();
        let server = make_server("all");
        let out = server
            .tool_architecture(&serde_json::json!({
                "path": project.display().to_string(),
            }))
            .expect("call");
        let s = out.structured.expect("structured");
        // With no focus, every section must be present.
        assert!(s.get("god_nodes").is_some(), "god_nodes missing: {s}");
        assert!(s.get("communities").is_some(), "communities missing: {s}");
        assert!(s.get("boundaries").is_some(), "boundaries missing: {s}");
    }

    #[test]
    fn tool_architecture_focus_limits_to_one_section() {
        let (_dir, project) = build_minimal_index();
        let server = make_server("all");
        let out = server
            .tool_architecture(&serde_json::json!({
                "path": project.display().to_string(),
                "focus": "god_nodes",
            }))
            .expect("call");
        let s = out.structured.expect("structured");
        assert!(s.get("god_nodes").is_some());
        assert!(
            s.get("communities").is_none(),
            "communities should be filtered out when focus=god_nodes"
        );
        assert!(s.get("boundaries").is_none());
    }

    #[test]
    fn tool_architecture_god_nodes_ranked_by_centrality() {
        let (_dir, project) = build_minimal_index();
        let server = make_server("all");
        let out = server
            .tool_architecture(&serde_json::json!({
                "path": project.display().to_string(),
                "focus": "god_nodes",
            }))
            .expect("call");
        let s = out.structured.expect("structured");
        let gods = s["god_nodes"].as_array().unwrap();
        assert!(
            !gods.is_empty(),
            "should have god nodes when centrality table populated"
        );
        // foo (centrality 0.9) should come before bar (centrality 0.1).
        // The test fixture has no structured_meta, so god_node.symbol falls back
        // to the file path — chunk 1 (foo) lives at src/a.rs, chunk 2 at src/b.rs.
        let first_centrality = gods[0]["centrality"].as_f64().unwrap_or(0.0);
        let last_centrality = gods[gods.len() - 1]["centrality"].as_f64().unwrap_or(1.0);
        assert!(
            first_centrality >= last_centrality,
            "god_nodes must be sorted by centrality desc: first={first_centrality}, last={last_centrality}"
        );
        // First entry should be the chunk that maps to centrality 0.9 (chunk 1 → src/a.rs).
        assert_eq!(
            gods[0]["location"]["file"].as_str(),
            Some("src/a.rs"),
            "highest centrality node should map to src/a.rs"
        );
    }

    // ─────────────────────────────────────────────────────────────────────
    // Finding 1 — detect_state_fast must respect concurrent rebuild lock
    // ─────────────────────────────────────────────────────────────────────

    /// When a warm-state sentinel is present but `.semantex.lock` is held by
    /// an indexer, the fast path must return `Building` (not `Ready`) so the
    /// MCP query handlers fall back instead of racing with a half-built DB.
    #[test]
    fn detect_state_fast_returns_building_when_lock_held() {
        let (dir, project) = build_minimal_index();
        let semantex_dir = project.join(".semantex");

        // Plant a warm-state sentinel pointing at the current process so
        // `warm_state_ready` short-circuits to the fast path.
        let sentinel = semantex_dir.join("warm_state.lock");
        std::fs::write(&sentinel, std::process::id().to_string()).expect("write sentinel");

        // Create and exclusively lock the build-coordination file. While the
        // lock is held the index is mid-rebuild and is NOT safe to query.
        let lock_path = semantex_dir.join(".semantex.lock");
        let lock_file = std::fs::File::create(&lock_path).expect("create .semantex.lock");
        lock_file.lock().expect("acquire exclusive flock");

        let state = McpServer::detect_state_fast(&project);
        assert_eq!(
            state,
            IndexState::Building,
            "fast path returned {state:?} while .semantex.lock was held — would race with indexer"
        );

        // Lock released on drop.
        drop(lock_file);
        drop(dir);
    }

    /// The fast path must still return `Ready` for the common case (sentinel
    /// present, no concurrent rebuild, current schema). This guards against
    /// over-tightening the validation in `detect_state_fast`.
    #[test]
    fn detect_state_fast_returns_ready_when_warm_and_clean() {
        let (_dir, project) = build_minimal_index();
        let semantex_dir = project.join(".semantex");
        let sentinel = semantex_dir.join("warm_state.lock");
        std::fs::write(&sentinel, std::process::id().to_string()).expect("write sentinel");

        let state = McpServer::detect_state_fast(&project);
        assert_eq!(
            state,
            IndexState::Ready,
            "fast path should return Ready when sentinel is live and no rebuild lock is held"
        );
    }

    // ─────────────────────────────────────────────────────────────────────
    // Finding 11 — tool_implementations returns methods, NOT call edges
    // ─────────────────────────────────────────────────────────────────────

    /// Build an in-memory index that simulates:
    ///
    /// ```ignore
    /// trait Foo { fn bar(&self); fn baz(&self); }
    /// impl Foo for X {                        // chunk 10: lines 1..30
    ///     fn bar(&self) { self.helper(); }    // chunk 11: lines 5..10
    ///     fn baz(&self) { log!("..."); }      // chunk 12: lines 15..20
    /// }
    /// fn helper(...) { ... }                  // chunk 13: lines 35..40
    /// ```
    ///
    /// The impl chunk's `meta.calls` contains the OUTGOING call targets
    /// (`self.helper`, `log`) — the bug had us reporting those as the impl's
    /// "method overrides", which is the wrong concept entirely.
    fn build_index_with_impl_block() -> (tempfile::TempDir, PathBuf) {
        use rusqlite::Connection;
        let dir = tempfile::tempdir().expect("tempdir");
        let semantex_dir = dir.path().join(".semantex");
        std::fs::create_dir_all(&semantex_dir).unwrap();
        let db_path = semantex_dir.join("chunks.db");

        let impl_file = "src/x.rs";
        let trait_file = "src/foo.rs";

        // 1) Insert chunks. Important: AUTOINCREMENT assigns sequential IDs
        //    starting at 1 in insertion order — we rely on that to know
        //    which chunk_id refers to which entity below.
        let trait_id = {
            let store = ChunkStore::open(&db_path).expect("open store");
            // chunk 1: the trait declaration itself
            let trait_chunk = Chunk {
                id: 0,
                file_path: PathBuf::from(trait_file),
                start_line: 1,
                end_line: 4,
                content: "trait Foo { fn bar(&self); fn baz(&self); }".to_string(),
                chunk_type: ChunkType::AstNode {
                    name: "Foo".to_string(),
                    kind: AstNodeKind::Other("trait".to_string()),
                    language: "rust".to_string(),
                    structured_meta: None,
                },
            };
            let trait_id = store
                .insert_chunk(&trait_chunk, 100, 0)
                .expect("insert trait");
            // chunk 2: the impl block (covers lines 1..30 in x.rs)
            let impl_chunk = Chunk {
                id: 0,
                file_path: PathBuf::from(impl_file),
                start_line: 1,
                end_line: 30,
                content: "impl Foo for X { ... }".to_string(),
                chunk_type: ChunkType::AstNode {
                    name: "X".to_string(),
                    kind: AstNodeKind::Other("impl".to_string()),
                    language: "rust".to_string(),
                    structured_meta: None,
                },
            };
            let impl_id = store
                .insert_chunk(&impl_chunk, 200, 0)
                .expect("insert impl");
            // chunk 3: fn bar(&self) inside the impl (lines 5..10)
            let bar_chunk = Chunk {
                id: 0,
                file_path: PathBuf::from(impl_file),
                start_line: 5,
                end_line: 10,
                content: "fn bar(&self) { self.helper(); }".to_string(),
                chunk_type: ChunkType::AstNode {
                    name: "bar".to_string(),
                    kind: AstNodeKind::Method,
                    language: "rust".to_string(),
                    structured_meta: None,
                },
            };
            let bar_id = store.insert_chunk(&bar_chunk, 200, 0).expect("insert bar");
            // chunk 4: fn baz(&self) inside the impl (lines 15..20)
            let baz_chunk = Chunk {
                id: 0,
                file_path: PathBuf::from(impl_file),
                start_line: 15,
                end_line: 20,
                content: "fn baz(&self) { log!(\"...\"); }".to_string(),
                chunk_type: ChunkType::AstNode {
                    name: "baz".to_string(),
                    kind: AstNodeKind::Method,
                    language: "rust".to_string(),
                    structured_meta: None,
                },
            };
            let baz_id = store.insert_chunk(&baz_chunk, 200, 0).expect("insert baz");
            // chunk 5: a free helper function OUTSIDE the impl block (lines 35..40)
            let helper_chunk = Chunk {
                id: 0,
                file_path: PathBuf::from(impl_file),
                start_line: 35,
                end_line: 40,
                content: "fn helper() -> () { /* outside impl */ }".to_string(),
                chunk_type: ChunkType::AstNode {
                    name: "helper".to_string(),
                    kind: AstNodeKind::Function,
                    language: "rust".to_string(),
                    structured_meta: None,
                },
            };
            let helper_id = store
                .insert_chunk(&helper_chunk, 200, 0)
                .expect("insert helper");

            // Symbol defs — kind matters for the query
            store
                .insert_symbol_def(trait_id, "Foo", "trait", trait_file)
                .expect("symbol_defs Foo");
            store
                .insert_symbol_def(impl_id, "X", "impl", impl_file)
                .expect("symbol_defs X");
            store
                .insert_symbol_def(bar_id, "bar", "method", impl_file)
                .expect("symbol_defs bar");
            store
                .insert_symbol_def(baz_id, "baz", "method", impl_file)
                .expect("symbol_defs baz");
            // helper is a free fn — must NOT appear in methods_defined_in_impl
            store
                .insert_symbol_def(helper_id, "helper", "function", impl_file)
                .expect("symbol_defs helper");

            // Outgoing call edges from the impl block itself. The OLD
            // (Finding 11) behaviour returned THIS list. We assert below
            // that the new field does NOT contain these tokens.
            store
                .store_call_graph_edge(impl_id, "self.helper", Some(helper_id))
                .expect("call edge self.helper");
            store
                .store_call_graph_edge(impl_id, "log", None)
                .expect("call edge log");
            trait_id
        };

        // 2) type_hierarchy: impl X → trait Foo. We need child_chunk and
        //    parent_chunk set so `query_implementations` can find them via
        //    parent_chunk membership (the chunk-ID path).
        {
            let conn = Connection::open(&db_path).unwrap();
            // Required aux tables (per build_minimal_index)
            conn.execute_batch(
                "CREATE TABLE IF NOT EXISTS pattern_matches (
                    chunk_id INTEGER NOT NULL,
                    pattern_name TEXT NOT NULL,
                    language TEXT NOT NULL,
                    description TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    PRIMARY KEY (chunk_id, pattern_name)
                );
                CREATE TABLE IF NOT EXISTS chunk_centrality (
                    chunk_id INTEGER PRIMARY KEY,
                    structural_centrality REAL NOT NULL
                );",
            )
            .unwrap();
            // The impl is chunk 2 (impl_id); the trait is chunk 1 (trait_id).
            conn.execute(
                "INSERT INTO type_hierarchy (child_name, parent_name, relation, child_chunk, parent_chunk)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                rusqlite::params!["X", "Foo", "implements", 2_i64, trait_id as i64],
            ).unwrap();
        }

        // 3) Write meta.json so detect_state_fast returns Ready.
        let now_secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let meta = semantex_core::types::IndexMeta {
            schema_version: semantex_core::types::IndexMeta::CURRENT_SCHEMA_VERSION,
            project_path: dir.path().to_path_buf(),
            created_at: now_secs.to_string(),
            updated_at: now_secs.to_string(),
            file_count: 2,
            chunk_count: 5,
            embedding_model: "test".to_string(),
            embedding_dim: 96,
        };
        std::fs::write(
            semantex_dir.join("meta.json"),
            serde_json::to_string(&meta).unwrap(),
        )
        .unwrap();

        let project = dir.path().to_path_buf();
        (dir, project)
    }

    /// `semantex_implementations` must return the methods physically defined
    /// inside the impl block — NOT the outgoing call edges (which was the
    /// pre-Finding-11 bug, where helpers and macro names leaked into the
    /// "method overrides" field).
    #[test]
    fn tool_implementations_returns_methods_not_call_edges() {
        let (_dir, project) = build_index_with_impl_block();
        let server = make_server("all");
        let out = server
            .tool_implementations(&serde_json::json!({
                "trait_or_interface": "Foo",
                "path": project.display().to_string(),
            }))
            .expect("call");
        let s = out.structured.expect("structured");
        let impls = s["implementations"].as_array().expect("implementations");
        assert_eq!(impls.len(), 1, "expected one impl of Foo, got {impls:?}");
        let entry = &impls[0];

        // Old (broken) field name must be gone — schema migration sanity check.
        assert!(
            entry.get("method_overrides").is_none(),
            "old `method_overrides` field must be removed: {entry}"
        );

        let methods = entry["methods_defined_in_impl"]
            .as_array()
            .expect("methods_defined_in_impl array");
        let names: Vec<&str> = methods.iter().filter_map(|v| v.as_str()).collect();

        assert!(
            names.contains(&"bar"),
            "bar should be a method defined in the impl: {names:?}"
        );
        assert!(
            names.contains(&"baz"),
            "baz should be a method defined in the impl: {names:?}"
        );
        // The free helper fn lives OUTSIDE the impl's [1..30] line range and
        // has kind=function — must not appear.
        assert!(
            !names.contains(&"helper"),
            "helper is a free fn outside the impl, not a method: {names:?}"
        );
        // Outgoing call targets must NOT leak in — this was the Finding 11 bug.
        assert!(
            !names.iter().any(|n| n.contains("self.helper")),
            "outgoing call target self.helper leaked into methods list: {names:?}"
        );
        assert!(
            !names.contains(&"log"),
            "outgoing macro/call target `log` leaked into methods list: {names:?}"
        );
    }
}
