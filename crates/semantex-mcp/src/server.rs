use crate::protocol::{
    InitializeResult, JsonRpcRequest, JsonRpcResponse, LogLevel, PROTOCOL_VERSION,
    ServerCapabilities, ServerInfo, Tool, ToolAnnotations, ToolCallResult, ToolContent,
    ToolsCapability,
};
use anyhow::{Context, Result};
use parking_lot::Mutex;
use semantex_core::config::SemantexConfig;
use semantex_core::index::branches;
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

/// Default toolset bundle: exposes the full `all_tools()` surface (10 tools).
/// `structural` is a separate 5-tool bundle, not a subset of `all` — see
/// [`McpServer::tools_for_toolset`].
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
    /// `Arc` so the background-index thread can drop a project's entry when its
    /// rebuild completes (S1: a cached searcher must not outlive a
    /// branch-switch reconcile — it would keep serving the old branch's index).
    cache: Arc<Mutex<HashMap<PathBuf, Arc<CachedSearcher>>>>,
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
    /// Server-side `semantex_agent` output defaults (budget/full_code/depth),
    /// read once from env at construction. Per-call MCP args override these.
    mcp_defaults: McpAgentDefaults,
    /// Optional LLM backend, initialised once at MCP server construction via
    /// `LlmBackend::from_env()` (Spec L §4 Item 1.4). When `Some`, every
    /// `AgentPipeline::new` call chains `.with_llm(...)`, enabling the LLM
    /// classifier override and HyDE retrieval inside `tool_agent`.
    #[cfg(feature = "llm")]
    llm: Option<Arc<dyn semantex_core::llm::LlmCapability>>,
    /// Shared current-thread Tokio runtime, built once at construction and
    /// reused by every `tool_agent` call so HyDE / LLM-classify do NOT build a
    /// per-request runtime (closing the v0.7.1 MCP wiring gap). Mirrors
    /// `Listener::bind`'s `llm_runtime` on the daemon path.
    #[cfg(feature = "llm")]
    llm_runtime: Option<Arc<tokio::runtime::Runtime>>,
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
        // Initialise the LLM backend once at MCP startup. The MCP server runs
        // independently of the daemon (it's the in-process search path used by
        // Claude Code, Cursor, etc.), so it needs its own backend instance to
        // make AgentPipeline's LLM hooks reachable.
        #[cfg(feature = "llm")]
        let llm: Option<Arc<dyn semantex_core::llm::LlmCapability>> =
            match semantex_core::llm::LlmBackend::from_env() {
                Ok(Some(backend)) => {
                    let cap = backend.into_arc();
                    tracing::info!("MCP LLM enabled: {}", cap.label());
                    Some(cap)
                }
                Ok(None) => None,
                Err(e) => {
                    tracing::warn!("MCP LLM backend init failed: {e}; disabling LLM features");
                    None
                }
            };

        // Build the shared runtime once (mirrors Listener::bind). A
        // current_thread build failure is treated like "no runtime": leave it
        // None and let AgentPipeline fall back to its per-call path rather than
        // refuse to start the MCP server.
        #[cfg(feature = "llm")]
        let llm_runtime: Option<Arc<tokio::runtime::Runtime>> = if llm.is_some() {
            match tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            {
                Ok(rt) => Some(Arc::new(rt)),
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        "Failed to build shared Tokio runtime for the MCP server; \
                         HyDE/classify will use a per-call runtime fallback."
                    );
                    None
                }
            }
        } else {
            None
        };

        Self {
            config,
            cache: Arc::new(Mutex::new(HashMap::new())),
            max_cached,
            rss_limit_mb,
            index_states: Arc::new(Mutex::new(HashMap::new())),
            log_level: Mutex::new(LogLevel::default()),
            toolset,
            mcp_defaults: McpAgentDefaults::from_env(),
            #[cfg(feature = "llm")]
            llm,
            #[cfg(feature = "llm")]
            llm_runtime,
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

    // -------------------------------------------------------------------------
    // Cross-repo federation (Wave 2 §B): `scope` param on semantex_agent /
    // semantex_search
    // -------------------------------------------------------------------------

    /// Parse the optional `scope` tool argument into a
    /// [`SearchScope`](semantex_core::search::federation::SearchScope).
    ///
    /// String values go through the SAME grammar as the CLI's `--scope`
    /// flag ([`federation::parse_scope_str`]): `"repo"`/empty → CurrentRepo,
    /// `"all"` → All, and any OTHER string is treated as comma-separated
    /// project names (`Named`) — never silently mapped to CurrentRepo, so a
    /// client sending `scope: "frontend"` (or a typo like `"All"`) gets the
    /// Named path, where unmatched names are surfaced in `skipped` instead
    /// of quietly returning wrong-scope results. Absent scope or a
    /// non-string/non-array JSON type is the safe default CurrentRepo.
    fn parse_scope(args: &serde_json::Value) -> semantex_core::search::federation::SearchScope {
        use semantex_core::search::federation::{SearchScope, parse_scope_str};
        match args.get("scope") {
            Some(serde_json::Value::String(s)) => parse_scope_str(s),
            Some(serde_json::Value::Array(arr)) => {
                let names: Vec<String> = arr
                    .iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect();
                if names.is_empty() {
                    SearchScope::CurrentRepo
                } else {
                    SearchScope::Named(names)
                }
            }
            _ => SearchScope::CurrentRepo,
        }
    }

    /// `IndexSearcher` impl backed by the server's own LRU searcher cache
    /// ([`McpServer::get_searcher`]) — the searcher-opening machinery
    /// `semantex_search`/`semantex_agent` already use for the single-repo
    /// path. Federating N targets through this just calls `get_searcher`
    /// once per target: with the default `max_cached=1`, each new target
    /// evicts the previous one BEFORE opening (see `get_searcher`), so a
    /// federated query still only ever holds one dense index resident at a
    /// time — "open sequentially, search, release" falls out of the
    /// existing eviction policy for free. Raising `SEMANTEX_MCP_CACHE_SIZE`
    /// lets an operator keep more targets warm across federated calls, at
    /// the RSS cost of doing so (still bounded by `check_rss_and_evict`).
    fn get_searcher_for_target(
        &self,
        target: &semantex_core::search::federation::IndexTarget,
    ) -> Result<Arc<CachedSearcher>> {
        let index_dir = semantex_core::search::federation::target_index_dir(target);
        self.get_searcher(&index_dir)
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
                // S8: detect_for_config so an embedder/embedding-text change also
                // marks the index Stale and kicks off a background rebuild here.
                // initialize runs once per MCP session — not a hot path — so the
                // one registry resolution is affordable (the per-search
                // `detect_state_fast` stays schema-only to keep the warm path
                // sub-microsecond; this is its session-level backstop).
                let idx_state = state::detect_for_config(&cwd, &self.config);
                // Wave 2: `detect_for_config` is schema/embedder-only — it has no
                // opinion on branches, so a `git switch` since the index was last
                // built looks like `Ready` to it even though the root now needs
                // reconciling. `branch_switch_pending` is a cheap read-only check
                // (two small file reads); force the background index in that case
                // too, so `spawn_background_index` gets a chance to run the
                // restore-or-snapshot handling before rebuilding.
                let branch_switched = branches::branch_switch_pending(&cwd);
                if idx_state == IndexState::NotIndexed
                    || idx_state == IndexState::Stale
                    || branch_switched
                {
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
                "## Tool choice: there is only one tool\n\n",
                "Use `semantex_agent` for every code-search question. It auto-classifies ",
                "the query and dispatches to the right strategy internally (semantic search, ",
                "exact symbol lookup, callers/callees/imports graph walk, deep multi-source ",
                "synthesis, regex, file-glob). One call in, one complete answer out.\n\n",
                "Do NOT chain multiple semantex_* calls to assemble an answer. If the agent ",
                "answer is incomplete, refine your QUESTION and call `semantex_agent` again. ",
                "Don't call `semantex_search` first to 'explore' before calling `semantex_agent` ",
                "— that double-spends turns. Don't grep first either; `semantex_agent` already ",
                "covers regex queries.\n\n",
                "## The other tools (rarely needed)\n\n",
                "- `semantex_search` — structured JSON only, for programmatic SearchResultItem ",
                "  consumption. Most agents should NOT use this; use `semantex_agent`.\n",
                "- `semantex_deep` — structured JSON only, for programmatic DeepSearchResponse. ",
                "  Most agents should NOT use this; use `semantex_agent`.\n",
                "- `semantex_status`, `semantex_health`, `semantex_validate`, `semantex_index` ",
                "  — diagnostics / one-shot ops. Not for searching.\n\n",
                "Fall back to Grep ONLY for tiny mechanical regex sweeps, Glob ONLY for ",
                "file-name patterns. All semantex tools are read-only and safe without ",
                "user confirmation. The index is auto-built on first use; queries during ",
                "the build return keyword results.",
                "\n\n## Git history\n\n",
                "Use `semantex_history` for commit-history questions: recent changes, ",
                "commits since a tag/sha/date, commits touching a file, commit-message ",
                "search, and per-commit diffs (`commits: [shas]`). Cross-repo via 'scope'. ",
                "It refreshes from git on every call. It is NOT a code-search tool — code ",
                "content questions still go to `semantex_agent`."
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

        // Every caller of this method is an unattended auto-index trigger
        // (first tool use against a not-yet-indexed cwd) — never an explicit
        // request. Refuse anything under a system temp root at any depth, so
        // a session that happens to have opened in a throwaway scratch
        // directory doesn't get it permanently tracked as a project.
        if registry::is_under_system_temp_root(&canonical) {
            self.index_states.lock().insert(
                canonical,
                IndexingStatus::Failed("Path is under the system temp directory".into()),
            );
            return;
        }

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

        // Refuse to auto-index a directory that looks like a multi-repo
        // workspace container (e.g. `~/dev` holding dozens of sibling
        // checkouts) rather than a single project — the walker has no repo
        // boundary, so this would silently flatten every nested repo into
        // one undifferentiated build. An explicit `semantex index --force`
        // remains available for anyone who genuinely wants that.
        if registry::is_likely_multi_repo_container(&canonical) {
            tracing::warn!(
                path = %canonical.display(),
                "Skipping auto-index — looks like a multi-repo workspace container, not a single project"
            );
            self.index_states.lock().insert(
                canonical,
                IndexingStatus::Failed("Looks like a multi-repo workspace container".into()),
            );
            return;
        }

        // Register this project in the global registry so tool_status can list all repos.
        registry::register(&canonical);

        let config = self.config.clone();
        let states = Arc::clone(&self.index_states);
        let cache = Arc::clone(&self.cache);
        std::thread::spawn(move || {
            // Wave 2: reconcile a branch switch BEFORE building — restores an
            // existing snapshot for the new branch, or snapshots the outgoing
            // branch, so the build below only re-indexes actual drift instead
            // of the whole tree. Acquires the exclusive index lock internally
            // for any mutation. Best-effort: log-and-continue on failure, the
            // build itself is still safe to attempt either way.
            if let Err(e) = branches::detect_and_handle_branch_switch(&canonical) {
                tracing::warn!(err = %e, "Branch switch check failed (continuing with build)");
            }

            tracing::info!(path = %canonical.display(), "Background indexing started");
            match IndexBuilder::new(&config).and_then(|b| b.build(&canonical)) {
                Ok(stats) => {
                    tracing::info!(
                        files = stats.files_indexed,
                        chunks = stats.chunks_created,
                        secs = stats.duration.as_secs_f64(),
                        "Background indexing completed"
                    );
                    branches::record_branch_indexed(&canonical);
                    // S1: drop any cached searcher for this project so the
                    // next search reopens against the just-rebuilt (possibly
                    // branch-switched) index instead of serving the old
                    // branch's content from a stale open handle.
                    cache
                        .lock()
                        .remove(&SemantexConfig::project_index_dir(&canonical));
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
    ///
    /// S8 note: this path is DELIBERATELY schema-only and does NOT fold in the
    /// embedder fingerprint (`detect_for_config`).
    /// It runs on every search call, and resolving the model registry (merge
    /// built-ins + optional `models.toml` + validate every spec) is not free;
    /// adding it here would defeat the sub-microsecond warm fast path.
    /// The embedder-change auto-rebuild trigger is handled once per session in
    /// `handle_initialize` (and surfaced in `tool_status`), both of which call
    /// `state::detect_for_config`.
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

    /// Per-search state check for the search-path tool handlers: exactly
    /// [`Self::detect_state_fast`], plus the cheap (two small file reads)
    /// [`branches::branch_switch_pending`] probe. Without it, a `git switch`
    /// mid-session would go unnoticed until the next `initialize` or the
    /// 1-hour age refresh — an actively-used cached searcher would keep
    /// serving the OLD branch's results indefinitely. When a switch is
    /// pending we kick off the background reconcile+rebuild, drop the stale
    /// cached searcher, and report `Building` so the caller falls back to
    /// ripgrep instead of answering from the wrong branch's index.
    fn search_path_state(&self, path: &std::path::Path) -> IndexState {
        let state = Self::detect_state_fast(path);
        if state == IndexState::Ready && branches::branch_switch_pending(path) {
            tracing::info!(path = %path.display(), "Branch switch pending — triggering reconcile before serving searches");
            self.invalidate_cache(&SemantexConfig::project_index_dir(path));
            self.spawn_background_index(path);
            return IndexState::Building;
        }
        state
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
    /// Bundles (measured against the current 10-tool `all_tools()` catalog):
    /// - `core`: 3 tools — `semantex_search`, `semantex_deep`, `semantex_agent`
    ///   (the minimum chat-bot surface).
    /// - `structural`: 5 tools — `semantex_symbol`, `semantex_callers`, `semantex_callees`,
    ///   `semantex_implementations`, `semantex_architecture` (the M1-M6 graph-navigation
    ///   tools hidden from `all` post-Phase-3; see [`Self::structural_tools`]). NOT a
    ///   subset of `all` — these tool defs live nowhere in `all_tools()`, so `structural`
    ///   is its own bundle, not a filter over `all`.
    /// - `all` (default): every tool registered with the server (10 total).
    ///
    /// Unknown toolset names fall back to `all` so callers (e.g. W7's HTTP
    /// transport) cannot accidentally lock themselves out of the surface.
    /// Exposed publicly so HTTP transports can call it without going through
    /// the JSON-RPC layer.
    #[must_use]
    pub fn tools_for_toolset(&self, toolset: &str) -> Vec<Tool> {
        match toolset {
            "core" => {
                let allow_set: HashSet<&str> =
                    ["semantex_search", "semantex_deep", "semantex_agent"]
                        .into_iter()
                        .collect();
                Self::all_tools()
                    .into_iter()
                    .filter(|t| allow_set.contains(t.name.as_str()))
                    .collect()
            }
            // Phase 3 hid M1-M6 from the default `all` surface (measured
            // +76pp CCB regression from agents chaining them additively —
            // see the NOTE above `semantex_docs_context` registration below).
            // That hide was about the DEFAULT surface, not about deleting the
            // tools: a caller that explicitly opts into `--toolset structural`
            // is asking for exactly this bundle, not stumbling into it, so it
            // stays exempt from the regression this hide guards against.
            "structural" => Self::structural_tools(),
            _ => Self::all_tools(),
        }
    }

    /// Build the full tool catalog for the default `all` bundle (10 tools).
    /// Single source of truth used by both `tools/list` (via
    /// `tools_for_toolset("all")` / any unrecognized name) and `tool_call`
    /// dispatch. Does NOT include the `structural` bundle's 5 tools — see
    /// [`Self::structural_tools`] and [`Self::tools_for_toolset`].
    #[allow(clippy::too_many_lines)]
    fn all_tools() -> Vec<Tool> {
        vec![
            Tool {
                name: "semantex_agent".into(),
                title: Some("Intelligent Code Search".into()),
                description: concat!(
                    "Intelligent code search: auto-classifies the query (semantic, exact symbol, ",
                    "graph walk, deep search, regex, file glob), runs the best strategy with fallbacks, ",
                    "and returns a ready-to-use answer. THE recommended tool for ALL code-search needs — ",
                    "one call in, one complete answer out. Do NOT chain semantex_* calls to assemble an ",
                    "answer; if the result is incomplete, refine the QUESTION and call semantex_agent again. ",
                    "Accepts a natural-language question, a symbol, a regex, or a glob. ",
                    "Use 'mode' ONLY when you know the retrieval shape: 'lexical' for exact symbol/regex, ",
                    "'structural' for callers/callees/imports, 'deep' for multi-source explanation. ",
                    "Leave 'mode' unset (auto) for everything else — the classifier is usually right."
                ).into(),
                input_schema: serde_json::json!({
                    "$schema": "https://json-schema.org/draft/2020-12/schema",
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
                            "description": "Include full source code blocks (default: server SEMANTEX_MCP_FULL_CODE, normally false)"
                        },
                        "budget": {
                            "type": "integer",
                            "description": "Response size budget in bytes (default: server SEMANTEX_MCP_BUDGET, normally 12000 ≈3K tokens)"
                        },
                        "mode": {
                            "type": "string",
                            "enum": ["auto", "lexical", "search", "structural", "deep"],
                            "description": "Retrieval strategy. 'auto' (default) = keyword classifier decides; 'lexical' = exact symbol or regex lookup; 'search' = hybrid dense+sparse; 'structural' = callers/callees/imports graph walk; 'deep' = multi-source synthesis. Overrides 'depth'. Leave unset unless you know the shape."
                        },
                        "depth": {
                            "type": "string",
                            "enum": ["quick", "search", "deep"],
                            "description": "Search depth. 'quick'=symbol lookup (~50ms), 'search'=hybrid with snippets (~100ms), 'deep'=full implementations with call graphs (~200ms). Omit to auto-detect. Overridden by 'mode' when both are set."
                        },
                        "focus": {
                            "type": "string",
                            "enum": ["implementation", "callers", "signatures", "patterns"],
                            "description": "What to emphasize in results. 'implementation'=full code bodies, 'callers'=who calls these functions, 'signatures'=function signatures only, 'patterns'=usage examples."
                        },
                        "response_format": {
                            "type": "string",
                            "enum": ["concise", "detailed"],
                            "description": "Output verbosity. 'concise' returns ~1/3 the context (paths + minimal snippets); 'detailed' returns fuller results. Omit for the server default. Ignored if an explicit 'budget' is set."
                        },
                        "scope": {
                            "description": "Which repo(s) to search. 'repo' (default) = only the current project at 'path' — identical to omitting this field. 'all' = every project registered in the local semantex registry (~/.semantex/projects.json), fanned out and RRF-fused with per-hit provenance. Any OTHER string is treated as comma-separated project display names/paths, and an array of names does the same (e.g. [\"frontend\", \"backend\"]) — names matching no registered project are reported in the response's skipped list rather than silently ignored. Cross-repo scopes always run a hybrid search: 'mode'/'depth'/'focus' route classification is skipped and those fields are ignored. Results carry a 'project' field and formatted paths are prefixed '[project] '.",
                            "oneOf": [
                                {"type": "string"},
                                {"type": "array", "items": {"type": "string"}}
                            ]
                        },
                        "include_history": {
                            "type": "boolean",
                            "description": "Default false. When true, appends a compact 'Recent changes' section (author, age, commit subject) for hit files with recorded git history, on 'deep'/'analytical' routes only. Requires the index's history.db to be populated (git repos only)."
                        }
                    }
                }),
                // NOTE: name/lang are nullable (the agent always emits these keys,
                // null when absent) — intentional divergence from semantex_search,
                // which omits them. `results` is required: it's always present when
                // structuredContent is emitted (only set when hits exist). `project`
                // is only present on hits from a cross-repo ('scope' != 'repo') call.
                output_schema: Some(serde_json::json!({
                    "$schema": "https://json-schema.org/draft/2020-12/schema",
                    "type": "object",
                    "properties": {
                        "results": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "file": {"type":"string"},
                                    "lines": {"type":"string"},
                                    "score": {"type":"number"},
                                    "name": {"type":["string","null"]},
                                    "snippet": {"type":"string"},
                                    "lang": {"type":["string","null"]},
                                    "project": {"type":"string"}
                                },
                                "required": ["file","lines","score","snippet"]
                            }
                        }
                    },
                    "required": ["results"]
                })),
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
                    "$schema": "https://json-schema.org/draft/2020-12/schema",
                    "type": "object",
                    "properties": {
                        "query": { "type": "string", "description": "Natural language search query" },
                        "path": { "type": "string", "description": "Project path to search (defaults to cwd)" },
                        "max_results": { "type": "integer", "description": "Maximum results to return", "default": 10 },
                        "rerank": { "type": "boolean", "description": "Enable cross-encoder reranking (slower but may improve ranking)", "default": false },
                        "grep_mode": { "type": "boolean", "description": "Fast grep-like search using only sparse+exact matching (no embedding model needed)", "default": false },
                        "scope": {
                            "description": "Which repo(s) to search. 'repo' (default) = only 'path' — identical to omitting this field. 'all' = every project in the local semantex registry, RRF-fused with provenance. Any OTHER string is treated as comma-separated project display names/paths, and an array of names does the same — names matching no registered project are reported in the response's 'skipped' list rather than silently ignored.",
                            "oneOf": [
                                {"type": "string"},
                                {"type": "array", "items": {"type": "string"}}
                            ]
                        }
                    },
                    "required": ["query"]
                }),
                output_schema: Some(serde_json::json!({
                    "$schema": "https://json-schema.org/draft/2020-12/schema",
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
                                    "lang": { "type": "string" },
                                    "project": { "type": "string" }
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
                        },
                        "skipped": {
                            "type": "array",
                            "description": "Cross-repo ('scope' != 'repo') calls only: targets left out of the federated query — projects whose index wasn't ready, and scope names that matched no registered project.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "project": { "type": "string" },
                                    "path": { "type": "string" },
                                    "reason": { "type": "string" }
                                },
                                "required": ["project", "path", "reason"]
                            }
                        },
                        "note": {
                            "type": "string",
                            "description": "Cross-repo calls only: set when scope resolution itself produced nothing to query (e.g. 'all' with an empty registry)."
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
                    "$schema": "https://json-schema.org/draft/2020-12/schema",
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
                    "$schema": "https://json-schema.org/draft/2020-12/schema",
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
                    "$schema": "https://json-schema.org/draft/2020-12/schema",
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
                    "$schema": "https://json-schema.org/draft/2020-12/schema",
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
                    "$schema": "https://json-schema.org/draft/2020-12/schema",
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
                    "$schema": "https://json-schema.org/draft/2020-12/schema",
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
            // NOTE — Phase-3 surface restriction (2026-05-25), updated
            // 2026-07-06 (E2E Wave 3) to match actual dispatch behavior.
            //
            // The six M1-M6 structural tools (semantex_symbol, _callers,
            // _callees, _implementations, _examples, _architecture) shipped
            // in v0.3 as visible MCP tools and triggered a measured
            // **+76pp regression in agent CCB** vs v0.2 (run22 −56% →
            // run23 +20%). Investigation in
            // `docs/BENCHMARK-v0.3-REGRESSION-ANALYSIS.md` traced this
            // to agents using M1-M6 *additively* — calling 4-6 structural
            // tools in sequence to gather what semantex_deep returned in
            // one call in v0.2.
            //
            // Fix: M1-M6 (minus `semantex_examples`) are dropped from
            // `all_tools()` / `tools/list`'s default `all` bundle and moved to
            // `Self::structural_tools()`, surfaced only under the explicit
            // `--toolset structural` bundle (see `tools_for_toolset`). Their
            // handler bodies never moved off `McpServer`.
            //
            // IMPORTANT — this is NOT "still reachable regardless of
            // toolset" back-compat: `handle_tool_call` gates every dispatch
            // on `tools_for_toolset(&self.toolset)` membership (added after
            // this note was first written), so calling e.g. `semantex_symbol`
            // while running under the default `all` toolset is rejected with
            // "not available in toolset 'all'". A v0.3.0 client that learned
            // these names needs `--toolset structural` to reach them again.
            // `semantex_agent`'s internal `Structural` route (see
            // `search/agent.rs::handle_structural`) remains the
            // always-available path that covers callers/callees/imports/
            // type-refs in a single graph-walk call, with no toolset
            // dependency.
            //
            // Default (`all`) surface: the 10 tools currently registered
            // below, with `semantex_agent` as the unconditional entry point.
            // -------------------------------------------------------------
            // -------------------------------------------------------------
            // v13 Wave 2 — semantex_docs_context.
            //
            // Registered ADDITIVELY at the end of this list (and the
            // dispatch match in `handle_tool_call` below) per the Wave 0/2
            // integration plan, to minimize merge conflicts with other
            // teams editing tool registration this batch.
            //
            // Deterministic documentation-scaffold tool: zero LLM wiring
            // (Wave 0 contract §F maintainer decision). Emits a
            // structurally-complete JSON scaffold — symbol inventory,
            // call-graph edges, import edges, existing doc-comment text,
            // file:line provenance — that the *host* agent (the user's own
            // LLM) turns into prose under `semantex_docs/` in the user's
            // repo, guided by the `semantex-docs` Skill. See
            // `semantex_core::index::docs_scaffold` and
            // `crate::docs_context` for the implementation.
            // -------------------------------------------------------------
            Tool {
                name: "semantex_docs_context".into(),
                title: Some("Documentation Context Scaffold".into()),
                description: concat!(
                    "Deterministic documentation scaffold — NOT an LLM call, does not write prose. ",
                    "Returns structurally-complete data (symbol inventory, call-graph edges, import ",
                    "edges, existing doc-comment text, file:line provenance) for you, the calling agent, ",
                    "to turn into maintained markdown docs. Use scope=\"overview\" once per repo for the ",
                    "architectural primer (god nodes / communities / boundaries) plus a module inventory ",
                    "and language stats — write this to `semantex_docs/README.md`. Use ",
                    "scope={\"module\": \"<path>\"} per file/directory you want to document in depth — ",
                    "write each to `semantex_docs/<module>.md`. Every scaffold item carries file:line ",
                    "provenance: cite it, and verify claims against the source before writing prose. ",
                    "On repeat runs, diff the scaffold against the existing doc and update only drifted ",
                    "sections. See the `semantex-docs` Skill for the full workflow."
                ).into(),
                input_schema: serde_json::json!({
                    "$schema": "https://json-schema.org/draft/2020-12/schema",
                    "type": "object",
                    "properties": {
                        "scope": {
                            "description": "\"overview\" for the repo-wide architecture + module inventory scaffold, or {\"module\": \"<path>\"} for one file's symbol/call/import scaffold. <path> must match the file path as indexed (project-relative).",
                            "oneOf": [
                                { "const": "overview" },
                                {
                                    "type": "object",
                                    "properties": { "module": { "type": "string" } },
                                    "required": ["module"],
                                    "additionalProperties": false
                                }
                            ]
                        },
                        "path": {
                            "type": "string",
                            "description": "Project path (defaults to current working directory)"
                        },
                        "budget": {
                            "type": "integer",
                            "description": "Approximate token budget for the returned scaffold (default 6000, clamped to [500, 100000]). Oversized scaffolds are trimmed deterministically — highest-signal items (most-referenced symbols, most-connected modules) survive."
                        }
                    },
                    "required": ["scope"]
                }),
                output_schema: None,
                annotations: Some(ToolAnnotations::read_only("Documentation Context Scaffold")),
            },
            // -------------------------------------------------------------
            // v13 Wave 2 — semantex_memory_save / semantex_memory_recall.
            //
            // Registered ADDITIVELY at the end of this list (and the
            // dispatch match in `handle_tool_call` below), same convention
            // as `semantex_docs_context` above, to minimize merge conflicts
            // with other teams editing tool registration this batch.
            //
            // Project memory: durable, project-scoped notes an agent saves
            // across sessions in `<project>/.semantex/memory.db` (schema +
            // CRUD in `semantex_core::index::memory`; param parsing +
            // rendering in `crate::memory_tools`). Independent of the code
            // index — no readiness gate, no `ChunkStore` open.
            // -------------------------------------------------------------
            Tool {
                name: "semantex_memory_save".into(),
                title: Some("Save Project Memory Note".into()),
                description: concat!(
                    "Save a short note to durable project memory — persists across sessions in ",
                    "`.semantex/memory.db`, independent of the code index. USE for things you had to ",
                    "figure out that aren't recoverable by searching the code: a design decision and its ",
                    "rationale, a gotcha/footgun you hit, a convention the team follows that isn't ",
                    "obviously stated anywhere, or a TODO/follow-up tied to a specific scope. Do NOT use ",
                    "for anything derivable by reading or searching the code (e.g. \"function foo calls ",
                    "bar\") — use semantex_search / semantex_agent for that instead, and don't save notes ",
                    "reflexively on every turn. `scope` groups related notes for later recall; suggested ",
                    "conventions: \"global\" (project-wide), \"file:<rel_path>\", \"module:<dir>\", ",
                    "\"task:<slug>\" — but any freeform string works. Notes are capped per project ",
                    "(oldest evicted first) so this is for durable, high-signal notes, not a scratch log."
                ).into(),
                input_schema: serde_json::json!({
                    "$schema": "https://json-schema.org/draft/2020-12/schema",
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The note text. Keep it short and self-contained (a sentence or two) — this is a durable memory entry, not a log line."
                        },
                        "scope": {
                            "type": "string",
                            "description": "Freeform scope key to group this note under (default \"global\"). Suggested conventions: \"global\", \"file:<rel_path>\", \"module:<dir>\", \"task:<slug>\"."
                        },
                        "tags": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "Optional freeform tags for this note (e.g. [\"gotcha\", \"perf\"])."
                        },
                        "path": {
                            "type": "string",
                            "description": "Project path (defaults to current working directory)"
                        }
                    },
                    "required": ["content"]
                }),
                output_schema: None,
                annotations: Some(ToolAnnotations::local_mutation("Save Project Memory Note")),
            },
            Tool {
                name: "semantex_memory_recall".into(),
                title: Some("Recall Project Memory Notes".into()),
                description: concat!(
                    "Recall notes previously saved with semantex_memory_save, ranked best-match-first. ",
                    "USE at the start of a task (or before making a decision that might already have prior ",
                    "context) to check whether a relevant decision, gotcha, or convention was already ",
                    "recorded for this project — cheaper than rediscovering it. Do NOT use as a substitute ",
                    "for code search: if nothing relevant was ever explicitly saved, this returns nothing ",
                    "useful and you should fall back to semantex_search / semantex_agent. Omit `query` to ",
                    "list the most recent notes instead of ranking by relevance. Filter with `scope` to ",
                    "match semantex_memory_save's scope conventions (e.g. \"file:<rel_path>\")."
                ).into(),
                input_schema: serde_json::json!({
                    "$schema": "https://json-schema.org/draft/2020-12/schema",
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Free text to rank notes by relevance against. Omit to just list the most recent notes."
                        },
                        "scope": {
                            "type": "string",
                            "description": "Restrict to notes saved under this exact scope key."
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max notes to return (default 5, clamped to [1, 50])."
                        },
                        "path": {
                            "type": "string",
                            "description": "Project path (defaults to current working directory)"
                        }
                    }
                }),
                output_schema: None,
                annotations: Some(ToolAnnotations::read_only("Recall Project Memory Notes")),
            },
            Tool {
                name: "semantex_history".into(),
                title: Some("Git History".into()),
                description: concat!(
                    "Query the project's indexed git history: recent commits, commits since a ",
                    "tag/sha/date, commits touching a file, full-text search over commit messages, ",
                    "and per-commit detail with diffs. History refreshes incrementally from git on ",
                    "every call, so results always reflect THIS LOCAL CLONE's current HEAD (no ",
                    "reindex needed after a pull) — but an un-pulled clone looks falsely idle; run ",
                    "`git pull` first when 'latest'/'recent' must mean upstream-current, not just ",
                    "locally-current. Use for release notes / changelogs ('since': tag or ",
                    "YYYY-MM-DD), 'what changed recently?', and cross-repo dependency change ",
                    "tracking (scope='all' or named projects). List mode returns one compact line ",
                    "per commit; pass commits=[shas] for detail mode with --stat and a ",
                    "budget-bounded patch (max 10 shas/call). ",
                    "Pass other_branches=true to also list other active local/remote branches — ",
                    "useful when a feature isn't in the current branch yet. ",
                    "NOT for searching code content — ",
                    "use semantex_agent for that."
                )
                .into(),
                input_schema: serde_json::json!({
                    "$schema": "https://json-schema.org/draft/2020-12/schema",
                    "type": "object",
                    "properties": {
                        "since": {
                            "type": "string",
                            "description": "Only commits AFTER this point (exclusive). Accepts a tag, sha, any git rev (e.g. HEAD~5), or a UTC date YYYY-MM-DD."
                        },
                        "query": {
                            "type": "string",
                            "description": "Full-text match over commit messages (FTS5 when available)."
                        },
                        "file": {
                            "type": "string",
                            "description": "Repo-relative path — only commits that touched it."
                        },
                        "author": {
                            "type": "string",
                            "description": "Author-name substring filter."
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max commits per project (default 20).",
                            "default": 20
                        },
                        "commits": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "Detail mode: expand these shas (7-40 hex chars each) with full message, files, --stat, and a budget-bounded patch. Max 10 per call; excess reported as skipped. When present, the list-mode filters above are ignored."
                        },
                        "budget": {
                            "type": "integer",
                            "description": "Detail mode only: total patch-byte budget split across requested shas (default SEMANTEX_MCP_BUDGET, normally 12000); each commit is guaranteed at least ~1000 bytes of patch, so the total may exceed this budget when many commits are requested."
                        },
                        "path": {
                            "type": "string",
                            "description": "Project path (defaults to current working directory)."
                        },
                        "scope": {
                            "description": "Which repo(s). 'repo' (default) = only 'path'. 'all' = every project in the local semantex registry. Any other string is treated as comma-separated registered project names, and an array of names does the same — unmatched names are reported in 'skipped'. Cross-repo output is one time-ordered section per project (no fusion).",
                            "oneOf": [
                                {"type": "string"},
                                {"type": "array", "items": {"type": "string"}}
                            ]
                        },
                        "other_branches": {
                            "type": "boolean",
                            "description": "List mode only: also list other local/remote branches (excluding the current one and its own upstream), most recently active first, capped at 10. Not computed unless requested."
                        }
                    }
                }),
                output_schema: Some(serde_json::json!({
                    "$schema": "https://json-schema.org/draft/2020-12/schema",
                    "type": "object",
                    "properties": {
                        "commits": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "hash": { "type": "string" },
                                    "author": { "type": "string" },
                                    "ts": { "type": "integer" },
                                    "age": { "type": "string" },
                                    "subject": { "type": "string" },
                                    "files": { "type": "array", "items": { "type": "string" } },
                                    "body": { "type": "string" },
                                    "stat": { "type": "string" },
                                    "patch": { "type": "string" },
                                    "patch_truncated": { "type": "boolean" }
                                },
                                "required": ["hash", "author", "ts", "subject"]
                            }
                        },
                        "projects": {
                            "type": "array",
                            "description": "Cross-repo calls only: one entry per project section.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "project": { "type": "string" },
                                    "commits": { "type": "array" },
                                    "errors": { "type": "array" },
                                    "upstream": { "type": ["object", "null"], "description": "This project's ahead/behind status vs. its upstream tracking branch." },
                                    "other_branches": { "type": "array", "description": "Other local/remote branches for this project." }
                                }
                            }
                        },
                        "errors": {
                            "type": "array",
                            "description": "Detail mode only: one entry per requested sha that failed \
                                (not found, ambiguous, etc.) — distinct from skipped_shas, which is \
                                capacity overflow, not a lookup failure.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "sha": { "type": "string" },
                                    "error": { "type": "string" }
                                },
                                "required": ["sha", "error"]
                            }
                        },
                        "skipped": { "type": "array", "items": { "type": "string" } },
                        "skipped_shas": { "type": "array", "items": { "type": "string" } },
                        "upstream": {
                            "type": ["object", "null"],
                            "description": "List mode only: this local clone's ahead/behind status vs. its upstream tracking branch, or null if none is configured.",
                            "properties": {
                                "ref": { "type": "string" },
                                "ahead": { "type": "integer" },
                                "behind": { "type": "integer" },
                                "fetch_ts": { "type": ["integer", "null"] },
                                "fetch_age": { "type": ["string", "null"] }
                            }
                        },
                        "other_branches": {
                            "type": "array",
                            "description": "Other local/remote branches (excluding current + its upstream), most recent first. Empty unless the other_branches param was true.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": { "type": "string" },
                                    "is_remote": { "type": "boolean" },
                                    "ts": { "type": "integer" },
                                    "age": { "type": "string" },
                                    "subject": { "type": "string" }
                                }
                            }
                        }
                    }
                })),
                annotations: Some(ToolAnnotations::read_only("Git History")),
            },
        ]
    }

    /// Build the 5-tool `structural` bundle: `semantex_symbol`, `_callers`,
    /// `_callees`, `_implementations`, `_architecture` (M1-M4 + M6; M5
    /// `semantex_examples` was never part of the historical `structural`
    /// allow-list and stays unlisted under any toolset name).
    ///
    /// These are the same Tool definitions `all_tools()` carried pre-Phase-3
    /// (restored verbatim from git history — their handler bodies never left
    /// `McpServer`, only the `tools/list` advertisement did), now surfaced
    /// ONLY when a caller explicitly requests `--toolset structural` rather
    /// than unconditionally in `all`. That's the distinction that matters for
    /// the CCB regression Phase 3 fixed: it was agents encountering these
    /// tools by default and chaining them additively, not a caller
    /// deliberately opting into the structural-navigation bundle.
    #[allow(clippy::too_many_lines)]
    fn structural_tools() -> Vec<Tool> {
        vec![
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
            // v13 Wave 2 — additive, see the Tool registration above.
            "semantex_docs_context" => self.tool_docs_context(&arguments),
            "semantex_memory_save" => self.tool_memory_save(&arguments),
            "semantex_memory_recall" => self.tool_memory_recall(&arguments),
            "semantex_history" => self.tool_history(&arguments),
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

        // `scope` != CurrentRepo bypasses the single-repo route-classification
        // pipeline below entirely (see `tool_agent_federated`'s doc comment for
        // why): this is what keeps `scope: "repo"` (the default) byte-identical
        // to every pre-federation caller.
        let scope = Self::parse_scope(args);
        if scope != semantex_core::search::federation::SearchScope::CurrentRepo {
            return Ok(self.tool_agent_federated(args, &queries, &path, &scope));
        }

        let full_code = args
            .get("full_code")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(self.mcp_defaults.full_code);

        // Parse the optional `mode` parameter (stable public vocabulary).
        // Precedence: explicit `mode` arg > `depth` arg > SEMANTEX_MCP_DEPTH env > auto.
        let mode_arg = args.get("mode").and_then(|v| v.as_str());
        let mode_query: &str = queries.first().map_or("", String::as_str);
        let mode_route: Option<AgentRoute> = mode_to_route(mode_arg, mode_query);

        // Parse the optional `depth` parameter and map it to an AgentRoute override.
        // `mode` takes precedence; depth only matters when mode is absent/auto.
        let depth_arg = args
            .get("depth")
            .and_then(|v| v.as_str())
            .or(self.mcp_defaults.depth.as_deref());
        let depth_route: Option<AgentRoute> = if mode_route.is_some() {
            // `mode` wins — depth is ignored.
            mode_route
        } else {
            match depth_arg {
                Some("quick") => Some(AgentRoute::ExactSymbol),
                Some("search") => Some(AgentRoute::Semantic),
                Some("deep") => Some(AgentRoute::Deep),
                _ => None,
            }
        };

        // Parse the optional `focus` parameter.
        let focus: Option<&str> = args.get("focus").and_then(|v| v.as_str());

        tracing::info!(
            queries = ?queries,
            path = %path.display(),
            full_code,
            mode = mode_arg.unwrap_or("auto"),
            ?depth_route,
            focus,
            "MCP agent search"
        );

        let path = path.canonicalize().unwrap_or_else(|_| path.clone());

        let idx_state = self.search_path_state(&path);
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
        // Spec L §4 Item 1.4 + S5: inject BOTH the LLM backend and the shared
        // runtime so the classifier override and HyDE retrieval reuse one
        // runtime instead of building a per-call one (no more per-request
        // "no shared runtime" warning on the MCP path).
        #[cfg(feature = "llm")]
        let pipeline = pipeline
            .with_llm(self.llm.clone())
            .with_runtime(self.llm_runtime.clone());

        let budget = args
            .get("budget")
            .and_then(serde_json::Value::as_u64)
            .map_or(self.mcp_defaults.budget, |v| v as usize);

        let response_format = args.get("response_format").and_then(|v| v.as_str());
        let budget = if args.get("budget").is_some() {
            budget // explicit byte budget wins
        } else {
            budget_for_format(response_format, budget)
        };

        // v13 Wave 2 (history) — additive, default false: appends a compact
        // "recent changes" line per unique hit file to Deep/Analytical
        // responses when `history.db` has commits for that file. Off by
        // default so existing callers see byte-identical output. On the Deep
        // route `hits` is the synthesis's own cited sources (see
        // `AgentPipeline::handle_deep` / `deep_source_to_hit`), not a generic
        // search-hit list — Deep's answer is otherwise prose-only, so without
        // that wiring this addendum could never fire there.
        let include_history = args
            .get("include_history")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);

        // Run each query and collect formatted results.
        let mut parts: Vec<String> = Vec::with_capacity(queries.len());
        let mut all_hits: Vec<serde_json::Value> = Vec::new();
        for q in &queries {
            let request = AgentRequest {
                query: q.clone(),
                route: depth_route,
                budget: Some(budget),
                full_code,
            };
            let (response, hits) = pipeline.handle_with_hits(&request);
            for h in &hits {
                all_hits.push(serde_json::json!({
                    "file": h.file,
                    "lines": format!("{}-{}", h.start_line, h.end_line),
                    "score": h.score,
                    "name": h.name,
                    "snippet": h.summary.clone().or_else(|| h.content.clone()).unwrap_or_default(),
                    "lang": h.language,
                }));
            }
            let mut formatted = response.formatted;
            if include_history
                && matches!(response.route, AgentRoute::Deep | AgentRoute::Analytical)
                && let Some(addendum) = history_addendum_for_hits(&path, &hits)
            {
                formatted.push_str(&addendum);
            }
            parts.push(formatted);
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

        if all_hits.is_empty() {
            Ok(ToolOutput::text(combined))
        } else {
            Ok(ToolOutput::text_with_structured(
                combined,
                serde_json::json!({ "results": all_hits }),
            ))
        }
    }

    /// `semantex_agent` with `scope` != `repo`.
    ///
    /// Cross-repo federation intentionally does NOT run the single-repo route
    /// classifier (`AgentRoute::{Structural,Deep,Architecture,FeaturePlanning}`
    /// are graph-walk/synthesis routes tied to one repo's call graph — "callers
    /// of X across every registered repo" isn't a coherent single operation).
    /// Instead every federated query always runs a hybrid dense+sparse search
    /// per target ([`McpFederatedSearcher`]), fused with RRF
    /// ([`federation::run_federated_search`]) exactly like `tool_search_federated`.
    /// Formatted output is rendered with the existing
    /// [`format_search_results`] (so it reads the same as any other search
    /// route's answer), with each hit's path prefixed `[project] `; the
    /// `structuredContent`/hits JSON keeps `file` clean and adds a sibling
    /// `project` field instead (so existing single-repo JSON consumers that
    /// might one day read a `scope: "all"` response aren't surprised by a
    /// mutated `file`).
    fn tool_agent_federated(
        &self,
        args: &serde_json::Value,
        queries: &[String],
        path: &std::path::Path,
        scope: &semantex_core::search::federation::SearchScope,
    ) -> ToolOutput {
        use semantex_core::search::agent_formatter::{
            DEFAULT_BUDGET, FormatStyle, format_search_results,
        };
        use semantex_core::search::federation::{self, project_display_name};
        use semantex_core::server::protocol::{SearchResponse, SearchResultItem};
        use std::fmt::Write as _;

        let budget = args
            .get("budget")
            .and_then(serde_json::Value::as_u64)
            .map_or(self.mcp_defaults.budget, |v| v as usize);
        let response_format = args.get("response_format").and_then(|v| v.as_str());
        let budget = if args.get("budget").is_some() {
            budget
        } else {
            budget_for_format(response_format, budget)
        };
        let budget = if budget == 0 { DEFAULT_BUDGET } else { budget };

        let cwd = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
        let registry_v2 = registry::read_all_v2();
        let searcher = McpFederatedSearcher {
            server: self,
            rerank: true,
            grep_mode: false,
        };

        tracing::info!(?queries, ?scope, path = %cwd.display(), "MCP federated agent search");

        let mut sections: Vec<String> = Vec::with_capacity(queries.len());
        let mut all_hits: Vec<serde_json::Value> = Vec::new();
        let mut skipped_projects: Vec<String> = Vec::new();
        // Scope-level diagnosis (e.g. `all` with an empty registry) — the
        // same for every query in the batch, so keep the first one seen.
        let mut federation_note: Option<String> = None;

        for q in queries {
            let outcome =
                federation::run_federated_search(scope, &registry_v2, &cwd, q, 20, &searcher);

            let items: Vec<SearchResultItem> = outcome
                .hits
                .iter()
                .map(|h| {
                    let project = project_display_name(&registry_v2, &h.target.project_root);
                    let mut item = federated_hit_to_item(h);
                    item.file = format!("[{project}] {}", item.file);
                    item
                })
                .collect();

            for h in &outcome.hits {
                let project = project_display_name(&registry_v2, &h.target.project_root);
                let item = federated_hit_to_item(h);
                all_hits.push(serde_json::json!({
                    "file": item.file,
                    "lines": format!("{}-{}", item.start_line, item.end_line),
                    "score": item.score,
                    "name": item.name,
                    "snippet": item.summary.clone().unwrap_or_default(),
                    "lang": item.language,
                    "project": project,
                }));
            }

            for s in &outcome.skipped {
                skipped_projects.push(format!(
                    "{} ({})",
                    project_display_name(&registry_v2, &s.target.project_root),
                    s.reason
                ));
            }
            if federation_note.is_none() {
                federation_note.clone_from(&outcome.note);
            }

            let resp = SearchResponse {
                results: items,
                duration_ms: 0,
                dense_count: 0,
                sparse_count: 0,
                fused_count: outcome.hits.len(),
                metrics: None,
                confidence: None,
                disambiguation: None,
            };
            sections.push(format_search_results(&resp, FormatStyle::Default, budget));
        }

        let mut combined = if sections.len() > 1 {
            format!(
                "[Batch results for {} queries — merged]\n\n{}",
                sections.len(),
                sections.join("\n\n---\n\n")
            )
        } else {
            sections.remove(0)
        };

        if let Some(note) = &federation_note {
            let _ = write!(combined, "\n\n[federation: {note}]");
        }

        if !skipped_projects.is_empty() {
            skipped_projects.sort();
            skipped_projects.dedup();
            let _ = write!(
                combined,
                "\n\n[federation: skipped {} target(s): {}]",
                skipped_projects.len(),
                skipped_projects.join(", ")
            );
        }

        if all_hits.is_empty() {
            ToolOutput::text(combined)
        } else {
            ToolOutput::text_with_structured(combined, serde_json::json!({ "results": all_hits }))
        }
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

        // `scope` != CurrentRepo takes a completely separate path
        // (`tool_search_federated`) that never touches the single-repo
        // `idx_state`/background-index machinery below — this is what keeps
        // `scope: "repo"` (the default, and every pre-federation caller)
        // byte-identical to today.
        let scope = Self::parse_scope(args);
        if scope != semantex_core::search::federation::SearchScope::CurrentRepo {
            tracing::info!(
                query,
                ?scope,
                max_results,
                rerank,
                grep_mode,
                "MCP federated search"
            );
            return self.tool_search_federated(
                query,
                &path,
                max_results,
                rerank,
                grep_mode,
                &scope,
            );
        }

        tracing::info!(query, path = %path.display(), max_results, rerank, grep_mode, "MCP search");

        let idx_state = self.search_path_state(&path);
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

    /// `semantex_search` with `scope` != `repo`: fan the query out across
    /// every target `scope` resolves to (via [`McpFederatedSearcher`], which
    /// reuses the same LRU searcher cache as the single-repo path), fuse
    /// with RRF ([`federation::run_federated_search`]), and shape the
    /// output like `do_semantex_search`'s JSON but with an added `project`
    /// field per hit and a `skipped` list for targets that weren't ready.
    fn tool_search_federated(
        &self,
        query: &str,
        path: &std::path::Path,
        max_results: usize,
        rerank: bool,
        grep_mode: bool,
        scope: &semantex_core::search::federation::SearchScope,
    ) -> Result<ToolOutput> {
        use semantex_core::search::federation::{self, project_display_name};

        let cwd = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
        let registry_v2 = registry::read_all_v2();
        let searcher = McpFederatedSearcher {
            server: self,
            rerank,
            grep_mode,
        };
        let outcome = federation::run_federated_search(
            scope,
            &registry_v2,
            &cwd,
            query,
            max_results,
            &searcher,
        );

        let json_results: Vec<serde_json::Value> = outcome
            .hits
            .iter()
            .map(|h| {
                let snippet = make_snippet_mcp(&h.result.chunk.content, &h.result.chunk.chunk_type);
                // Federated scores are RRF values (~1/(60+rank), so ranks
                // differ only in the 4th-5th decimal) — round to 5 decimals,
                // NOT the single-repo path's 2, which would collapse most of
                // the ranking into identical values.
                let score = (h.result.score * 100_000.0_f32).round() / 100_000.0_f32;
                let project = project_display_name(&registry_v2, &h.target.project_root);
                let mut val = serde_json::json!({
                    "file": h.result.chunk.file_path.display().to_string(),
                    "lines": format!("{}-{}", h.result.chunk.start_line, h.result.chunk.end_line),
                    "score": score,
                    "snippet": snippet,
                    "project": project,
                });
                if let semantex_core::types::ChunkType::AstNode { name, language, .. } =
                    &h.result.chunk.chunk_type
                {
                    val["name"] = serde_json::Value::String(name.clone());
                    val["lang"] = serde_json::Value::String(language.clone());
                }
                val
            })
            .collect();

        let skipped_json: Vec<serde_json::Value> = outcome
            .skipped
            .iter()
            .map(|s| {
                serde_json::json!({
                    "project": project_display_name(&registry_v2, &s.target.project_root),
                    "path": s.target.project_root.display().to_string(),
                    "reason": s.reason,
                })
            })
            .collect();

        let mut structured = serde_json::json!({
            "results": json_results,
            "skipped": skipped_json,
        });
        if let Some(note) = &outcome.note {
            structured["note"] = serde_json::Value::String(note.clone());
        }

        let json_text = serde_json::to_string(&json_results)?;
        let mut footer = format!(
            "[semantex_federation: hits={} skipped={}]",
            json_results.len(),
            skipped_json.len()
        );
        if let Some(note) = &outcome.note {
            use std::fmt::Write as _;
            let _ = write!(footer, "\n[federation: {note}]");
        }
        let text = format!("{json_text}\n\n{footer}");

        Ok(ToolOutput {
            text,
            structured: Some(structured),
        })
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

        // S8: detect_for_config so STALE in the status report also reflects an
        // embedder-fingerprint change (model/embedding-text swap), not only a
        // schema mismatch. Status is a diagnostic path, not a hot loop.
        let idx_state = state::detect_for_config(&canonical, &self.config);
        let state_label = match idx_state {
            IndexState::NotIndexed => "NOT INDEXED",
            IndexState::Building => "BUILDING",
            IndexState::Stale => "STALE (schema or embedder change)",
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

        let branches_str = Self::branches_detail(&canonical);

        format!(
            "{display}\n  State: {state_label}{refresh_note}\n  Age:   {age_str}\n  Files: {}\n  Chunks: {}\n  Model: {}{branches_str}",
            meta.file_count, meta.chunk_count, meta.embedding_model
        )
    }

    /// Wave 2: render the registry's tracked branches for `canonical`, if
    /// any — appended (not restructuring) to `repo_status_detail`'s existing
    /// output so anything scraping the prior fields stays unaffected.
    fn branches_detail(canonical: &Path) -> String {
        use std::fmt::Write as _;

        let Some(entry) = registry::read_all_v2()
            .projects
            .into_iter()
            .find(|p| p.path == canonical)
        else {
            return String::new();
        };
        if entry.branches.is_empty() {
            return String::new();
        }

        let current_key = semantex_core::index::layout::current_branch_key(canonical);
        let mut branches = entry.branches;
        branches.sort_by(|a, b| b.last_indexed_ts.cmp(&a.last_indexed_ts));

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);

        let mut out = String::from("\n  Branches:");
        for b in &branches {
            let marker = if b.branch_key == current_key {
                " (current)"
            } else {
                ""
            };
            let age = if b.last_indexed_ts > 0 {
                format_age((now - b.last_indexed_ts).max(0) as u64)
            } else {
                "unknown".to_string()
            };
            let _ = write!(out, "\n    {}{marker}: indexed {age} ago", b.branch);
        }
        out
    }

    // -------------------------------------------------------------------------
    // Tool: semantex_health
    // -------------------------------------------------------------------------

    #[tracing::instrument(skip(self, _args), fields(tool = "health"))]
    fn tool_health(&self, _args: &serde_json::Value) -> Result<String> {
        use semantex_core::embedding::single_vector_model;

        let mut status = Vec::new();
        status.push(format!(
            "Semantex Health Check v{}",
            env!("CARGO_PKG_VERSION")
        ));
        status.push("=".repeat(50));

        let models_dir = self.config.models_dir();
        status.push(format!("\nModels Directory: {}", models_dir.display()));

        if single_vector_model::is_coderank_downloaded(&models_dir) {
            let model_dir = models_dir.join("CodeRankEmbed");
            let model_path = model_dir.join("model_int8.onnx");
            let tokenizer_path = model_dir.join("tokenizer.json");
            status.push("\nDense Embedder Model: OK".to_string());
            status.push("  Name: CodeRankEmbed".to_string());
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
            status.push("\nDense Embedder Model: NOT DOWNLOADED".to_string());
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
            "  Rerank scoring window: {} (SEMANTEX_RERANK_CANDIDATES)",
            self.config.rerank_top_n
        ));
        status.push(format!(
            "  Retrieval pool: {}",
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

        let idx_state = self.search_path_state(&path);
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
        use semantex_core::index::architecture::{
            ARCH_TINY_REPO_THRESHOLD, ArchBudget, budget_for_chunk_count, build_arch_overview,
        };

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

        // Derive an adaptive output budget from index size. Mirrors the
        // agent path's `handle_architecture` so both surfaces emit the same
        // section sizes for the same index. Tiny repos collapse to god_nodes
        // only (no communities / boundaries) to avoid empty-section noise.
        let chunk_count = store.chunk_count().unwrap_or(0);
        let arch_budget = if chunk_count < ARCH_TINY_REPO_THRESHOLD {
            ArchBudget {
                god_nodes: 5,
                communities: 0,
                boundaries: 0,
                deep_examples_max: 0,
                exhaustive_max: 0,
            }
        } else {
            budget_for_chunk_count(chunk_count as usize)
        };

        let overview = build_arch_overview(&store, &db_path, Some(arch_budget))
            .context("Failed to build architectural overview")?;

        // Adapter: convert ArchOverview into the legacy tool_architecture JSON
        // shape. We preserve the v0.3 Phase 4 surface:
        //   god_nodes: [{symbol, centrality, role, location:{file,start_line,end_line}}]
        //   communities: [{label, size, members:[file_path], entry_points:[{symbol, location:{...}}]}]
        //   boundaries: [{from, to, edge_count}]
        // The `role` field needs the structured_meta `kind` fallback that the
        // typed GodNode struct doesn't carry, so we re-read meta per god_node
        // here. With MEDIUM budget (10 god_nodes) that's at most 10 single-row
        // queries — negligible at the tool boundary.
        let god_nodes_json: Vec<serde_json::Value> =
            if focus.as_deref().is_none_or(|f| f == "god_nodes") {
                let mut entries: Vec<serde_json::Value> =
                    Vec::with_capacity(overview.god_nodes.len());
                // Batch-fetch chunks for all god_node ids so symbol-name reads
                // go through Chunk::symbol_name() (single source of truth — see
                // Defect #15). This is one batched query for the full list.
                let ids: Vec<u64> = overview.god_nodes.iter().map(|g| g.chunk_id).collect();
                let chunks = store.get_chunks(&ids).unwrap_or_default();
                let chunk_by_id: HashMap<u64, semantex_core::types::Chunk> =
                    chunks.into_iter().map(|c| (c.id, c)).collect();
                for g in &overview.god_nodes {
                    let meta = store.get_structured_meta(g.chunk_id).ok().flatten();
                    // Role precedence (preserves legacy shape):
                    //   1. structured_meta.semantic_role.as_label()
                    //   2. structured_meta.kind
                    //   3. "module"  (no meta at all)
                    //   4. ""        (meta present but no role / kind)
                    let role = match meta.as_ref() {
                        Some(m) => m.semantic_role.as_ref().map_or_else(
                            || m.kind.clone().unwrap_or_default(),
                            |r| r.as_label().to_string(),
                        ),
                        None => "module".to_string(),
                    };
                    // Defect #15: symbol-name read goes through Chunk::symbol_name
                    // (matches hybrid.rs Phase 3d), falling back to file path
                    // for non-AstNode chunks. Architecture overview is
                    // intentionally AstNode-centric (god nodes are functions /
                    // classes / methods), so dropping non-AstNode names here is
                    // acceptable.
                    let symbol = chunk_by_id
                        .get(&g.chunk_id)
                        .and_then(|c| c.symbol_name().map(str::to_string))
                        .unwrap_or_else(|| g.file.clone());
                    entries.push(serde_json::json!({
                        "symbol": symbol,
                        "centrality": g.centrality,
                        "role": role,
                        "location": {
                            "file": g.file,
                            "start_line": g.start_line,
                            "end_line": g.end_line,
                        },
                    }));
                }
                entries
            } else {
                Vec::new()
            };

        let communities_json: Vec<serde_json::Value> =
            if focus.as_deref().is_none_or(|f| f == "communities") {
                overview
                    .communities
                    .iter()
                    .map(|c| {
                        let entry_points: Vec<serde_json::Value> = c
                            .entry_points
                            .iter()
                            .map(|ep| {
                                serde_json::json!({
                                    "symbol": ep.symbol,
                                    "location": {
                                        "file": ep.file,
                                        "start_line": ep.start_line,
                                        "end_line": ep.end_line,
                                    },
                                })
                            })
                            .collect();
                        serde_json::json!({
                            "label": c.label,
                            "size": c.size,
                            "members": c.member_files,
                            "entry_points": entry_points,
                        })
                    })
                    .collect()
            } else {
                Vec::new()
            };

        let boundaries_json: Vec<serde_json::Value> =
            if focus.as_deref().is_none_or(|f| f == "boundaries") {
                overview
                    .boundaries
                    .iter()
                    .map(|b| {
                        serde_json::json!({
                            "from": b.from,
                            "to": b.to,
                            "edge_count": b.edge_count,
                        })
                    })
                    .collect()
            } else {
                Vec::new()
            };

        let mut structured = serde_json::Map::new();
        if focus.as_deref().is_none_or(|f| f == "god_nodes") {
            structured.insert("god_nodes".into(), serde_json::Value::Array(god_nodes_json));
        }
        if focus.as_deref().is_none_or(|f| f == "communities") {
            structured.insert(
                "communities".into(),
                serde_json::Value::Array(communities_json),
            );
        }
        if focus.as_deref().is_none_or(|f| f == "boundaries") {
            structured.insert(
                "boundaries".into(),
                serde_json::Value::Array(boundaries_json),
            );
        }
        let structured = serde_json::Value::Object(structured);
        let text = serde_json::to_string_pretty(&structured)?;
        Ok(ToolOutput {
            text,
            structured: Some(structured),
        })
    }

    // -------------------------------------------------------------------------
    // v13 Wave 2 — semantex_docs_context
    //
    // Deterministic documentation scaffold. Param parsing, scaffold
    // construction and text rendering live in `crate::docs_context` so this
    // handler stays a thin wrapper matching the shape of every other tool
    // handler above (resolve path → gate on index readiness → open store →
    // delegate).
    // -------------------------------------------------------------------------

    #[tracing::instrument(skip(self, args), fields(tool = "docs_context"))]
    fn tool_docs_context(&self, args: &serde_json::Value) -> Result<ToolOutput> {
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
            .context("Failed to open chunk store for semantex_docs_context")?;

        let result = crate::docs_context::run(&store, &db_path, args)?;
        Ok(ToolOutput {
            text: result.text,
            structured: Some(result.structured),
        })
    }

    // -------------------------------------------------------------------------
    // v13 Wave 2 — semantex_memory_save / semantex_memory_recall
    //
    // Project memory lives in its own `memory.db`, independent of the code
    // index — no index-readiness gate, no `ChunkStore`. Param parsing, CRUD
    // dispatch, and text rendering live in `crate::memory_tools` /
    // `semantex_core::index::memory`; these handlers just resolve the
    // project path and delegate, matching every other tool handler's shape.
    // -------------------------------------------------------------------------

    // `&self` is unused in both handlers below (memory notes need neither
    // the RSS-cache/searcher state nor the index-readiness gate other
    // handlers use `self` for) but is kept so both fit the `self.tool_x(...)`
    // dispatch shape every other arm in `handle_tool_call` uses.
    #[allow(clippy::unused_self)]
    #[tracing::instrument(skip(self, args), fields(tool = "memory_save"))]
    fn tool_memory_save(&self, args: &serde_json::Value) -> Result<ToolOutput> {
        let path = resolve_project_path(args)?;
        let result = crate::memory_tools::run_save(&path, args)?;
        Ok(ToolOutput {
            text: result.text,
            structured: Some(result.structured),
        })
    }

    #[allow(clippy::unused_self)]
    #[tracing::instrument(skip(self, args), fields(tool = "memory_recall"))]
    fn tool_memory_recall(&self, args: &serde_json::Value) -> Result<ToolOutput> {
        let path = resolve_project_path(args)?;
        let result = crate::memory_tools::run_recall(&path, args)?;
        Ok(ToolOutput {
            text: result.text,
            structured: Some(result.structured),
        })
    }

    // -------------------------------------------------------------------------
    // v13 Wave 2 — semantex_history
    //
    // First-class git-history queries (list + detail), single-repo here;
    // `scope != repo` delegates to the federated path (Task 5).
    // -------------------------------------------------------------------------

    /// `semantex_history` — first-class git-history queries (list + detail),
    /// single-repo here; `scope != repo` delegates to the federated path.
    /// Works with or without a search index: history.db is populated on
    /// demand and git is the source of truth.
    #[allow(clippy::unused_self)]
    #[tracing::instrument(skip(self, args), fields(tool = "history"))]
    fn tool_history(&self, args: &serde_json::Value) -> Result<ToolOutput> {
        let scope = Self::parse_scope(args);
        let budget = args
            .get("budget")
            .and_then(serde_json::Value::as_u64)
            .map_or(self.mcp_defaults.budget, |v| v as usize);
        if scope != semantex_core::search::federation::SearchScope::CurrentRepo {
            let registry_v2 = registry::read_all_v2();
            let (text, structured) = history_federated(&registry_v2, &scope, args, budget)?;
            return Ok(ToolOutput::text_with_structured(text, structured));
        }
        let path = resolve_project_path(args)?;
        let (text, structured) = history_for_project(&path, None, args, budget)?;
        Ok(ToolOutput::text_with_structured(text, structured))
    }
}

/// v13 Wave 2 (history): `semantex_agent`'s `include_history` addendum.
///
/// For up to 5 distinct hit files (in hit order), looks up the single most
/// recent commit touching that file and renders one compact line —
/// `- <file>: <subject> (<author>, <age>)`. Returns `None` (no addendum,
/// not an empty one) when `history.db` doesn't exist yet or no hit's file
/// has any recorded history — a silent, best-effort enrichment, never a
/// reason to fail or alter a `semantex_agent` call.
fn history_addendum_for_hits(
    project_path: &Path,
    hits: &[semantex_core::server::protocol::SearchResultItem],
) -> Option<String> {
    use semantex_core::index::{history, layout};

    const MAX_FILES: usize = 5;
    let db_path = layout::history_db_path(project_path);
    if !db_path.exists() {
        return None;
    }
    let conn = rusqlite::Connection::open(&db_path).ok()?;
    // Wait briefly instead of failing with SQLITE_BUSY if a concurrent
    // build is populating history.db right now.
    let _ = conn.busy_timeout(std::time::Duration::from_secs(5));

    let mut seen = std::collections::HashSet::new();
    let mut lines = Vec::new();
    for h in hits {
        if lines.len() >= MAX_FILES {
            break;
        }
        if !seen.insert(h.file.clone()) {
            continue;
        }
        let Ok(commits) = history::commits_touching_file(&conn, &h.file, 1) else {
            continue;
        };
        let Some(c) = commits.first() else { continue };
        lines.push(format!(
            "- {}: {} ({}, {})",
            h.file,
            c.subject,
            c.author,
            history::humanize_age(c.ts)
        ));
    }
    if lines.is_empty() {
        return None;
    }
    Some(format!("\n\n## Recent changes\n{}\n", lines.join("\n")))
}

/// One project's `semantex_history` answer: (markdown text, structured JSON).
/// `label` prefixes a `## [label]` section header on federated calls.
///
/// Refreshes history from git first (spec §2): `populate_history` is
/// incremental and idempotent — a caught-up repo costs one `git rev-parse`.
/// Deliberately does NOT take `.semantex.lock` (that would read as "index
/// building" elsewhere); concurrent writers are safe via SQLite tx +
/// INSERT OR IGNORE + the 5s busy timeout.
fn history_for_project(
    project_root: &std::path::Path,
    label: Option<&str>,
    args: &serde_json::Value,
    budget: usize,
) -> Result<(String, serde_json::Value)> {
    use semantex_core::index::history;
    use std::fmt::Write as _;

    if let Err(e) = history::populate_history(project_root) {
        tracing::debug!(project = %project_root.display(), "history refresh skipped: {e}");
    }
    let header = label.map_or(String::new(), |l| format!("## [{l}]\n\n"));
    if !history::has_history(project_root) {
        let msg = format!(
            "{header}No git history recorded for this project \
             (not a git repository, zero commits, or `git` unavailable)."
        );
        return Ok((msg, serde_json::json!({ "commits": [] })));
    }
    let conn = history::open(project_root)?;

    // ── Detail mode: `commits` param present ──────────────────────────
    if let Some(shas) = args.get("commits").and_then(|v| v.as_array()) {
        let requested: Vec<String> = shas
            .iter()
            .filter_map(|v| v.as_str())
            .map(str::to_string)
            .collect();
        if requested.is_empty() {
            anyhow::bail!("'commits' must be a non-empty array of sha strings");
        }
        let (take, skipped_shas) = if requested.len() > history::MAX_DETAIL_COMMITS {
            requested.split_at(history::MAX_DETAIL_COMMITS)
        } else {
            (&requested[..], &[][..])
        };
        let per_commit_budget = (budget / take.len()).max(1_000);
        let mut text = header;
        let mut commits_json = Vec::new();
        let mut errors_json = Vec::new();
        for sha in take {
            match history::commit_detail(project_root, &conn, sha, per_commit_budget) {
                Ok(d) => {
                    let short: String = d.hash.chars().take(8).collect();
                    let _ = writeln!(
                        text,
                        "### {short} — {}\n{} · {}",
                        d.subject,
                        d.author,
                        history::humanize_age(d.ts)
                    );
                    if !d.body.is_empty() {
                        let _ = writeln!(text, "\n{}", d.body);
                    }
                    if !d.stat.is_empty() {
                        let _ = writeln!(text, "\n```\n{}\n```", d.stat);
                    }
                    if !d.patch.is_empty() {
                        let _ = writeln!(text, "\n```diff\n{}\n```", d.patch);
                        if d.patch_truncated {
                            text.push_str("[diff truncated]\n");
                        }
                    }
                    text.push('\n');
                    commits_json.push(serde_json::json!({
                        "hash": d.hash,
                        "author": d.author,
                        "ts": d.ts,
                        "age": history::humanize_age(d.ts),
                        "subject": d.subject,
                        "body": d.body,
                        "files": d.files,
                        "stat": d.stat,
                        "patch": d.patch,
                        "patch_truncated": d.patch_truncated,
                    }));
                }
                Err(e) => {
                    let _ = writeln!(text, "### {sha}\n[error: {e}]\n");
                    errors_json.push(serde_json::json!({ "sha": sha, "error": e.to_string() }));
                }
            }
        }
        if !skipped_shas.is_empty() {
            let _ = writeln!(
                text,
                "[{} sha(s) beyond the per-call cap of {} skipped: {}]",
                skipped_shas.len(),
                history::MAX_DETAIL_COMMITS,
                skipped_shas.join(", ")
            );
        }
        return Ok((
            text.trim_end().to_string(),
            serde_json::json!({
                "commits": commits_json,
                "errors": errors_json,
                "skipped_shas": skipped_shas
            }),
        ));
    }

    // ── List mode ──────────────────────────────────────────────────────
    let mut filter = history::HistoryFilter {
        since_ts: None,
        author: args
            .get("author")
            .and_then(|v| v.as_str())
            .map(str::to_string),
        file: args
            .get("file")
            .and_then(|v| v.as_str())
            .map(str::to_string),
        message_query: args
            .get("query")
            .and_then(|v| v.as_str())
            .map(str::to_string),
        limit: args
            .get("limit")
            .and_then(serde_json::Value::as_u64)
            .map_or(20, |v| v as usize),
    };
    let mut window_note: Option<String> = None;
    if let Some(since) = args.get("since").and_then(|v| v.as_str()) {
        let Some(ts) = history::resolve_since_to_ts(project_root, since) else {
            anyhow::bail!(
                "could not resolve since='{since}' \
                 (accepted: YYYY-MM-DD, a tag, a sha, or any git rev)"
            );
        };
        filter.since_ts = Some(ts);
        if let Some(oldest) = history::oldest_commit_ts(&conn)
            && ts < oldest
        {
            window_note = Some(format!(
                "[note: the captured history window starts {} — commits between \
                 '{since}' and that point are not recorded; raise \
                 SEMANTEX_HISTORY_COMMITS (default 500) and reindex to widen it]",
                history::humanize_age(oldest)
            ));
        }
    }
    let commits = history::filter_commits(&conn, &filter)?;
    let mut text = header;
    let mut commits_json = Vec::new();
    if commits.is_empty() {
        text.push_str("No commits matched.");
    } else {
        for c in &commits {
            let files = history::files_for_commit(&conn, &c.hash).unwrap_or_default();
            let short: String = c.hash.chars().take(8).collect();
            let _ = writeln!(
                text,
                "{short}  {:>8}  {} — {} ({} file{})",
                history::humanize_age(c.ts),
                c.author,
                c.subject,
                files.len(),
                if files.len() == 1 { "" } else { "s" }
            );
            commits_json.push(serde_json::json!({
                "hash": c.hash,
                "author": c.author,
                "ts": c.ts,
                "age": history::humanize_age(c.ts),
                "subject": c.subject,
                "files": files,
            }));
        }
    }
    if let Some(note) = window_note {
        let _ = write!(text, "\n{note}");
    }

    // ── Upstream staleness (always computed; note only when behind > 0) ──
    let upstream = history::upstream_status(project_root);
    let upstream_json = upstream.as_ref().map_or(serde_json::Value::Null, |u| {
        serde_json::json!({
            "ref": u.upstream_ref,
            "ahead": u.ahead,
            "behind": u.behind,
            "fetch_ts": u.fetch_ts,
            "fetch_age": u.fetch_ts.map(history::humanize_age),
        })
    });
    if let Some(u) = &upstream
        && u.behind > 0
    {
        let _ = write!(
            text,
            "\n[local clone is {} commit{} behind {} ({}) — `git pull` for upstream-current state]",
            u.behind,
            if u.behind == 1 { "" } else { "s" },
            u.upstream_ref,
            u.fetch_ts
                .map_or("fetch time unknown".to_string(), |ts| format!(
                    "fetched {}",
                    history::humanize_age(ts)
                )),
        );
    }

    // ── Other-branch visibility (opt-in via `other_branches: true`) ──────
    let mut other_branches_json = Vec::new();
    if args
        .get("other_branches")
        .and_then(serde_json::Value::as_bool)
        == Some(true)
    {
        let current = history::current_branch(project_root);
        let upstream_ref = upstream.as_ref().map(|u| u.upstream_ref.as_str());
        let branches = history::other_branches(
            project_root,
            current.as_deref(),
            upstream_ref,
            history::MAX_OTHER_BRANCHES,
        );
        if !branches.is_empty() {
            text.push_str("\n\nOther active branches:");
            for b in &branches {
                let _ = write!(
                    text,
                    "\n  {}{} ({}) — {}",
                    b.name,
                    if b.is_remote { " [remote]" } else { "" },
                    history::humanize_age(b.ts),
                    b.subject
                );
                other_branches_json.push(serde_json::json!({
                    "name": b.name,
                    "is_remote": b.is_remote,
                    "ts": b.ts,
                    "age": history::humanize_age(b.ts),
                    "subject": b.subject,
                }));
            }
        }
    }

    Ok((
        text.trim_end().to_string(),
        serde_json::json!({
            "commits": commits_json,
            "upstream": upstream_json,
            "other_branches": other_branches_json
        }),
    ))
}

/// Cross-repo `semantex_history`: one `## [project]` section per selected
/// registry project, commits time-ordered WITHIN each section. No RRF —
/// history is naturally per-repo, and interleaving unrelated repos' commit
/// streams would only obscure them. Unmatched names and per-project errors
/// land in `skipped` (mirrors `run_federated_search`'s skip semantics).
/// `args` passes through unchanged, so detail mode (`commits`) fans out
/// too: the repo that owns a sha expands it, and every other repo renders
/// a per-commit `[error: ...]` line instead of failing the whole call.
fn history_federated(
    registry: &semantex_core::index::registry::RegistryV2,
    scope: &semantex_core::search::federation::SearchScope,
    args: &serde_json::Value,
    budget: usize,
) -> Result<(String, serde_json::Value)> {
    use semantex_core::search::federation::SearchScope;
    use std::fmt::Write as _;

    let mut selected: Vec<&semantex_core::index::registry::ProjectEntry> = Vec::new();
    let mut skipped: Vec<String> = Vec::new();
    match scope {
        SearchScope::All => selected.extend(registry.projects.iter()),
        SearchScope::Named(names) => {
            for name in names {
                match registry
                    .projects
                    .iter()
                    .find(|p| history_project_matches(p, name))
                {
                    Some(p) => selected.push(p),
                    None => skipped.push(format!("no registered project matched '{name}'")),
                }
            }
        }
        SearchScope::CurrentRepo => {
            anyhow::bail!("history_federated must not be called with scope=repo")
        }
    }
    // Two names can match one project — history it once.
    {
        let mut seen = std::collections::HashSet::new();
        selected.retain(|p| seen.insert(p.path.clone()));
    }
    // scope=all only: a worktree's git history duplicates its parent repo's
    // commit DAG, and a registered subdirectory of an already-selected repo
    // duplicates its commits too — both would just render the same section
    // twice under a different display name. Named scope is unaffected:
    // naming a worktree or nested subdirectory explicitly is deliberate
    // intent. The path-prefix dedup mirrors the technique
    // `refresh_stale_registry_repos` (hooks.rs) already uses when deciding
    // which repos to reindex — cheaper than a git-toplevel subprocess call.
    if matches!(scope, SearchScope::All) {
        selected.retain(|p| !p.is_worktree);
        selected.sort_by_key(|p| p.path.as_os_str().len());
        let mut kept_roots: Vec<std::path::PathBuf> = Vec::new();
        selected.retain(|p| {
            if kept_roots.iter().any(|root| p.path.starts_with(root)) {
                false
            } else {
                kept_roots.push(p.path.clone());
                true
            }
        });
    }
    if selected.is_empty() {
        let text = if skipped.is_empty() {
            "registry has no projects — nothing to federate \
             (build an index in a repo to register it)"
                .to_string()
        } else {
            format!("no projects matched; skipped: {}", skipped.join("; "))
        };
        return Ok((
            text,
            serde_json::json!({ "projects": [], "skipped": skipped }),
        ));
    }

    let mut sections: Vec<String> = Vec::new();
    let mut projects_json: Vec<serde_json::Value> = Vec::new();
    for p in selected {
        match history_for_project(&p.path, Some(&p.display_name), args, budget) {
            Ok((text, structured)) => {
                sections.push(text);
                projects_json.push(serde_json::json!({
                    "project": p.display_name,
                    "commits": structured.get("commits").cloned()
                        .unwrap_or_else(|| serde_json::json!([])),
                    "errors": structured.get("errors").cloned()
                        .unwrap_or_else(|| serde_json::json!([])),
                    "upstream": structured.get("upstream").cloned()
                        .unwrap_or(serde_json::Value::Null),
                    "other_branches": structured.get("other_branches").cloned()
                        .unwrap_or_else(|| serde_json::json!([])),
                }));
            }
            Err(e) => skipped.push(format!("{} ({e})", p.display_name)),
        }
    }
    let mut text = sections.join("\n\n");
    if !skipped.is_empty() {
        let _ = write!(
            text,
            "\n\n[federation: skipped {}: {}]",
            skipped.len(),
            skipped.join(", ")
        );
    }
    Ok((
        text,
        serde_json::json!({ "projects": projects_json, "skipped": skipped }),
    ))
}

/// Name matching for history federation: display name, full path, or the
/// path's final component. (Local rather than reusing federation's private
/// matcher — semantics kept deliberately identical; if federation's
/// `project_matches_name` is ever made pub, swap this for it.)
fn history_project_matches(p: &semantex_core::index::registry::ProjectEntry, name: &str) -> bool {
    p.display_name == name
        || p.path == std::path::Path::new(name)
        || p.path
            .file_name()
            .is_some_and(|f| f == std::ffi::OsStr::new(name))
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
/// Returns true iff the index is Ready (warm or detected) AND the live root
/// still belongs to the current git branch (S1: after a mid-session
/// `git switch`, the structural tools must not answer from the old branch's
/// index — every caller reacts to `false` by spawning the background
/// reconcile+rebuild and returning a "building" message).
fn require_index_ready(path: &Path) -> bool {
    McpServer::detect_state_fast(path) == IndexState::Ready
        && !branches::branch_switch_pending(path)
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

// NOTE: the local `query_top_centrality`, `compute_communities`,
// `compute_boundaries_from_path`, and `top_level_dir` helpers used by the
// previous `tool_architecture` body were removed when the handler was rerouted
// through `semantex_core::index::architecture::build_arch_overview` (v0.4.1
// W-MCP defect #14). The shared implementations now live in
// `crates/semantex-core/src/index/architecture.rs` and are used by both the
// agent path (`AgentRoute::Architecture`) and this MCP handler.

// =============================================================================
// MCP agent output defaults
// =============================================================================

/// Server-side defaults for the `semantex_agent` output shape, read ONCE from
/// the environment at MCP server construction (the in-process config model).
/// A per-call MCP arg always overrides these. `depth == None` means auto-classify.
#[derive(Debug, Clone)]
struct McpAgentDefaults {
    budget: usize,
    full_code: bool,
    depth: Option<String>,
}

impl McpAgentDefaults {
    fn from_env() -> Self {
        let budget = std::env::var("SEMANTEX_MCP_BUDGET")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|&b| b > 0)
            .unwrap_or(12_000);
        let full_code = std::env::var("SEMANTEX_MCP_FULL_CODE")
            .is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"));
        let depth = match std::env::var("SEMANTEX_MCP_DEPTH")
            .ok()
            .map(|v| v.trim().to_ascii_lowercase())
            .as_deref()
        {
            Some("quick") => Some("quick".to_string()),
            Some("search") => Some("search".to_string()),
            Some("deep") => Some("deep".to_string()),
            _ => None, // "auto", empty, unset, or unknown -> auto-classify
        };
        Self {
            budget,
            full_code,
            depth,
        }
    }
}

// =============================================================================
// Cross-repo federation: IndexSearcher wired through the LRU searcher cache
// =============================================================================

/// [`IndexSearcher`](semantex_core::search::federation::IndexSearcher) impl
/// used by the `scope != repo` paths of `semantex_search`/`semantex_agent`.
/// Opens each target through [`McpServer::get_searcher_for_target`] (the
/// existing LRU cache with idle-timeout + RSS-pressure eviction), so a
/// federated query reuses the exact same searcher-opening machinery as a
/// single-repo call rather than a parallel ad hoc one.
struct McpFederatedSearcher<'a> {
    server: &'a McpServer,
    rerank: bool,
    grep_mode: bool,
}

impl semantex_core::search::federation::IndexSearcher for McpFederatedSearcher<'_> {
    fn search_target(
        &self,
        target: &semantex_core::search::federation::IndexTarget,
        query: &str,
        limit: usize,
    ) -> Result<Vec<semantex_core::types::SearchResult>> {
        if self.grep_mode {
            // Sparse-only: skip the ONNX/dense load entirely, matching the
            // single-repo `do_semantex_search` grep_mode path. Not cached
            // (mirrors that path too) — grep-mode opens are cheap.
            let index_dir = semantex_core::search::federation::target_index_dir(target);
            let searcher = HybridSearcher::open_sparse_only(&index_dir, &self.server.config)?;
            let sq = SearchQuery::new(query).grep_mode().max_results(limit);
            return Ok(searcher.search(&sq)?.results);
        }
        let cached = self.server.get_searcher_for_target(target)?;
        let mut sq = SearchQuery::new(query).max_results(limit);
        if !self.rerank {
            sq = sq.no_rerank();
        }
        Ok(cached.searcher.search(&sq)?.results)
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

    fn text_with_structured(text: String, structured: serde_json::Value) -> Self {
        Self {
            text,
            structured: Some(structured),
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

/// Convert a federated hit into a plain [`SearchResultItem`] with `file` left
/// bare (repo-relative, un-prefixed). `tool_agent_federated` prefixes `[project]
/// ` onto a CLONE of `.file` only for the human-formatted text path;
/// JSON/structuredContent builders read this bare form and carry `project` as
/// a separate sibling field instead — see that function's doc comment.
fn federated_hit_to_item(
    hit: &semantex_core::search::federation::FederatedHit,
) -> semantex_core::server::protocol::SearchResultItem {
    use semantex_core::server::protocol::SearchResultItem;
    use semantex_core::types::ChunkType;

    let (name, language) = match &hit.result.chunk.chunk_type {
        ChunkType::AstNode { name, language, .. } => (Some(name.clone()), Some(language.clone())),
        _ => (None, None),
    };
    let summary = make_snippet_mcp(&hit.result.chunk.content, &hit.result.chunk.chunk_type);

    SearchResultItem {
        file: hit.result.chunk.file_path.display().to_string(),
        start_line: hit.result.chunk.start_line,
        end_line: hit.result.chunk.end_line,
        score: hit.result.score,
        source: format!("{:?}", hit.result.source),
        chunk_type: format!("{:?}", hit.result.chunk.chunk_type),
        name,
        language,
        content: Some(hit.result.chunk.content.clone()),
        kind: None,
        summary: Some(summary),
    }
}

/// Resolve the stable public `mode` vocabulary to an `AgentRoute` override.
///
/// Pure (no I/O, no `self`) so the mapping and precedence are unit-testable
/// without a live index. Returns `None` to mean "no override — let the
/// keyword classifier (or `depth`/env, per the caller's precedence) decide".
///
/// Mapping:
///   - `auto` / `None` / unknown → `None` (auto-classify; unknown warns)
///   - `lexical` → `ExactSymbol`, promoted to `Regex` when `query` contains
///     regex metacharacters (reuses `agent_classifier::is_regex`, the single
///     source of truth)
///   - `search` → `Semantic`
///   - `structural` → `Structural`
///   - `deep` → `Deep`
fn mode_to_route(
    mode: Option<&str>,
    query: &str,
) -> Option<semantex_core::search::agent_classifier::AgentRoute> {
    use semantex_core::search::agent_classifier::{AgentRoute, is_regex};
    match mode {
        Some("auto") | None => None, // explicit auto or absent → let depth/env/classifier decide
        Some("lexical") => {
            // Exact symbol lookup; promote to Regex if the query contains
            // regex metacharacters (backslash-class, alternation, bracket
            // class, anchors, or quantifier group).
            if is_regex(query) {
                Some(AgentRoute::Regex)
            } else {
                Some(AgentRoute::ExactSymbol)
            }
        }
        Some("search") => Some(AgentRoute::Semantic),
        Some("structural") => Some(AgentRoute::Structural),
        Some("deep") => Some(AgentRoute::Deep),
        Some(other) => {
            tracing::warn!(
                mode = other,
                "unknown 'mode' value — ignoring, falling back to depth/auto"
            );
            None
        }
    }
}

/// Map an optional `response_format` to an effective budget given the base budget.
/// `concise` ≈ ⅓ tokens (best-practice "return less by default"); `detailed`/None = base.
fn budget_for_format(response_format: Option<&str>, base_budget: usize) -> usize {
    match response_format {
        Some("concise") => (base_budget / 3).max(1),
        _ => base_budget, // "detailed", None, or unknown -> base
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
    use super::{
        DEFAULT_TOOLSET, McpAgentDefaults, McpServer, TOOLSETS, apply_focus, budget_for_format,
        history_addendum_for_hits, history_federated, mode_to_route, require_index_ready,
    };
    use semantex_core::config::SemantexConfig;
    use semantex_core::index::architecture::top_level_dir;
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
    // response_format — budget tier mapping
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn response_format_maps_to_budget_tier() {
        // concise ~= 1/3, detailed = base, default(None) = base.
        assert_eq!(budget_for_format(Some("concise"), 12_000), 4_000);
        assert_eq!(budget_for_format(Some("detailed"), 12_000), 12_000);
        assert_eq!(budget_for_format(None, 12_000), 12_000);
        assert_eq!(budget_for_format(Some("bogus"), 12_000), 12_000);
    }

    // ─────────────────────────────────────────────────────────────────────
    // McpAgentDefaults — env-tunable output defaults
    // ─────────────────────────────────────────────────────────────────────

    /// Serializes the `SEMANTEX_MCP_*` env mutation in this test binary so the
    /// agent-defaults test never races other tests on process env. Mirrors
    /// `MCP_LLM_ENV_LOCK` (in `llm_runtime_wiring_tests`); kept module-local
    /// here so it works in the default (no-`llm`) build too.
    static MCP_AGENT_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn mcp_agent_defaults_from_env_overrides_and_falls_back() {
        let _guard = MCP_AGENT_ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // SAFETY: guarded by MCP_AGENT_ENV_LOCK; restored before the lock drops.
        // All env work + reads happen first; the prior values are restored
        // BEFORE any assert, so a failing assert can never leak the keys.
        let prev_budget = std::env::var("SEMANTEX_MCP_BUDGET").ok();
        let prev_full_code = std::env::var("SEMANTEX_MCP_FULL_CODE").ok();
        let prev_depth = std::env::var("SEMANTEX_MCP_DEPTH").ok();

        // Case 1: unset -> fallback defaults.
        unsafe {
            std::env::remove_var("SEMANTEX_MCP_BUDGET");
            std::env::remove_var("SEMANTEX_MCP_FULL_CODE");
            std::env::remove_var("SEMANTEX_MCP_DEPTH");
        }
        let unset = McpAgentDefaults::from_env();

        // Case 2: explicit overrides.
        unsafe {
            std::env::set_var("SEMANTEX_MCP_BUDGET", "6000");
            std::env::set_var("SEMANTEX_MCP_FULL_CODE", "1");
            std::env::set_var("SEMANTEX_MCP_DEPTH", "deep");
        }
        let overridden = McpAgentDefaults::from_env();

        // Case 3: depth "auto" collapses to None (auto-classify).
        unsafe {
            std::env::set_var("SEMANTEX_MCP_DEPTH", "auto");
        }
        let auto = McpAgentDefaults::from_env();

        // SAFETY: guarded by MCP_AGENT_ENV_LOCK; restore prior values before
        // the lock drops (and before the asserts, so a panic can't leak).
        unsafe {
            match prev_budget {
                Some(v) => std::env::set_var("SEMANTEX_MCP_BUDGET", v),
                None => std::env::remove_var("SEMANTEX_MCP_BUDGET"),
            }
            match prev_full_code {
                Some(v) => std::env::set_var("SEMANTEX_MCP_FULL_CODE", v),
                None => std::env::remove_var("SEMANTEX_MCP_FULL_CODE"),
            }
            match prev_depth {
                Some(v) => std::env::set_var("SEMANTEX_MCP_DEPTH", v),
                None => std::env::remove_var("SEMANTEX_MCP_DEPTH"),
            }
        }

        assert_eq!(unset.budget, 12_000);
        assert!(!unset.full_code);
        assert_eq!(unset.depth, None); // None == auto-classify

        assert_eq!(overridden.budget, 6_000);
        assert!(overridden.full_code);
        assert_eq!(overridden.depth.as_deref(), Some("deep"));

        assert_eq!(auto.depth, None);
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
    fn toolset_all_exposes_eleven_visible_tools() {
        // Phase 3 surface restriction: the M1-M6 structural tools that
        // shipped in v0.3 caused a measured +76pp regression in agent CCB
        // vs v0.2. They're hidden from the *default* `all` bundle's
        // tools/list (moved to their own `structural` bundle — see
        // `toolset_structural_exposes_five_m1_m6_tools` below).
        //
        // v13 Wave 2 adds `semantex_docs_context` as a genuinely new visible
        // tool (not one of the moved M1-M6), plus `semantex_memory_save`
        // / `semantex_memory_recall` (also new, also visible), plus
        // `semantex_history` (also new, also visible), so the count is now 11.
        let server = make_server("all");
        let tools = server.tools_for_toolset("all");
        assert_eq!(
            tools.len(),
            11,
            "toolset 'all' must expose the v0.2 set of 7 visible tools plus \
             v13's semantex_docs_context, semantex_memory_save/recall, and semantex_history, got {}",
            tools.len()
        );
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        for required in &[
            "semantex_agent",
            "semantex_search",
            "semantex_deep",
            "semantex_status",
            "semantex_health",
            "semantex_validate",
            "semantex_index",
            "semantex_docs_context",
            "semantex_memory_save",
            "semantex_memory_recall",
            "semantex_history",
        ] {
            assert!(names.contains(required), "missing visible tool {required}");
        }
        // M1-M6 must NOT be visible under the default `all` bundle.
        for hidden in &[
            "semantex_symbol",
            "semantex_callers",
            "semantex_callees",
            "semantex_implementations",
            "semantex_examples",
            "semantex_architecture",
        ] {
            assert!(
                !names.contains(hidden),
                "M1-M6 tool {hidden} should not be in the default 'all' bundle post-Phase-3"
            );
        }
    }

    #[test]
    fn toolset_core_exposes_three_search_tools() {
        // Post-Phase-3 `core` is the minimum chat-bot surface: agent + the
        // two structured-JSON variants. Chat-only clients pick this; richer
        // clients pick `all` (10) for the diagnostic tools.
        let server = make_server("core");
        let tools = server.tools_for_toolset("core");
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(tools.len(), 3, "core bundle must have 3 tools: {names:?}");
        assert!(names.contains(&"semantex_search"));
        assert!(names.contains(&"semantex_deep"));
        assert!(names.contains(&"semantex_agent"));
    }

    #[test]
    fn toolset_structural_exposes_five_m1_m6_tools() {
        // `structural` is its own bundle (M1-M4 + M6; M5 `semantex_examples`
        // was never in the historical allow-list), NOT a filter over `all` —
        // these 5 Tool defs live in `structural_tools()`, not `all_tools()`.
        // A caller must explicitly opt into `--toolset structural` to see
        // them; the default `all` bundle never does (that's what Phase 3
        // fixed — see `toolset_all_exposes_eleven_visible_tools`).
        let server = make_server("structural");
        let tools = server.tools_for_toolset("structural");
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(
            tools.len(),
            5,
            "structural bundle must have exactly 5 tools: {names:?}"
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
        assert!(
            !names.contains(&"semantex_examples"),
            "semantex_examples was never part of the structural bundle"
        );
        // Distinct from `all` — not an alias, an entirely separate bundle.
        assert_ne!(tools.len(), server.tools_for_toolset("all").len());
    }

    #[test]
    fn unknown_toolset_falls_back_to_all() {
        let server = make_server("nonsense");
        assert_eq!(server.toolset(), "all");
        let tools = server.tools_for_toolset("nonsense");
        // Filter falls through to `all` (Phase 3: 7 visible tools + v13's
        // semantex_docs_context + memory_save/recall + semantex_history = 11).
        assert_eq!(tools.len(), 11);
    }

    #[test]
    fn m1_m6_tool_call_rejected_under_all_toolset_but_dispatchable_under_structural() {
        // Reconciles the Phase-3 NOTE (above `semantex_docs_context`
        // registration) with actual `handle_tool_call` behavior: the
        // toolset-gating check added after that note was written means
        // M1-M6 tool calls are NOT reachable "regardless of toolset" — they
        // 404-equivalent under `all`/`core`, and only work under the
        // explicit `structural` bundle.
        let all_server = make_server("all");
        let visible_all: std::collections::HashSet<String> = all_server
            .tools_for_toolset("all")
            .into_iter()
            .map(|t| t.name)
            .collect();
        assert!(
            !visible_all.contains("semantex_symbol"),
            "semantex_symbol must not be dispatchable-by-discovery under 'all'"
        );

        let structural_server = make_server("structural");
        let visible_structural: std::collections::HashSet<String> = structural_server
            .tools_for_toolset("structural")
            .into_iter()
            .map(|t| t.name)
            .collect();
        assert!(
            visible_structural.contains("semantex_symbol"),
            "semantex_symbol must be dispatchable under the explicit 'structural' toolset"
        );
    }

    // (structural-toolset dispatch is proven end-to-end by the existing
    // `tool_symbol_finds_known_symbol`, `tool_callers_finds_caller`, etc.
    // tests below — they call the McpServer methods directly, the same path
    // `handle_tool_call` takes via its match arms once the toolset gate
    // above admits the tool name.)

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
            // v0.4.1 W-Index #4: IndexMeta now persists use_bm25_stemmer so
            // open-time mismatch detection can refuse to load a mis-tuned
            // index. Default-true matches the v0.4 legacy build behaviour.
            use_bm25_stemmer: true,
            dense_backend: "coderank-hnsw".to_string(),
            embedder_fingerprint: "test".to_string(),
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
    fn tool_symbol_dispatch_via_handle_tool_call_gated_by_toolset() {
        // End-to-end proof (through `handle_tool_call`, the real JSON-RPC
        // dispatch path — not the direct `tool_symbol` method call above)
        // that `semantex_symbol` is reachable ONLY under the explicit
        // `structural` toolset, and rejected under the default `all`.
        let (_dir, project) = build_minimal_index();
        let writer = super::NotificationWriter {
            stdout: std::sync::Arc::new(parking_lot::Mutex::new(std::io::BufWriter::new(
                std::io::stdout(),
            ))),
        };
        let args = serde_json::json!({
            "name": "semantex_symbol",
            "arguments": {
                "name": "foo",
                "path": project.display().to_string(),
            }
        });

        let structural_server = make_server("structural");
        let ok_response =
            structural_server.handle_tool_call(Some(serde_json::json!(1)), &args, &writer);
        let ok_json = serde_json::to_value(&ok_response).unwrap();
        let ok_result = &ok_json["result"];
        assert_ne!(
            ok_result["isError"].as_bool(),
            Some(true),
            "semantex_symbol must dispatch successfully under --toolset structural: {ok_result:?}"
        );

        let all_server = make_server("all");
        let rejected_response =
            all_server.handle_tool_call(Some(serde_json::json!(2)), &args, &writer);
        let rejected_json = serde_json::to_value(&rejected_response).unwrap();
        let rejected_result = &rejected_json["result"];
        assert_eq!(
            rejected_result["isError"].as_bool(),
            Some(true),
            "semantex_symbol must be rejected under the default --toolset all: {rejected_result:?}"
        );
        let text = rejected_result["content"][0]["text"].as_str().unwrap_or("");
        assert!(
            text.contains("not available in toolset 'all'"),
            "rejection message should name the active toolset: {text}"
        );
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

    /// Defect #14: `tool_architecture` must route through
    /// `build_arch_overview` with a size-adaptive budget. On a tiny repo
    /// (chunk_count < `ARCH_TINY_REPO_THRESHOLD` = 500), the budget collapses
    /// to god_nodes only — communities and boundaries must be empty arrays.
    /// This mirrors `handle_architecture` on the agent path.
    #[test]
    fn tool_architecture_tiny_repo_collapses_to_god_nodes_only() {
        // `build_minimal_index` produces a 2-chunk fixture. 2 < 500 → tiny
        // tier → communities=0 / boundaries=0 in the budget override.
        let (_dir, project) = build_minimal_index();
        let server = make_server("all");
        let out = server
            .tool_architecture(&serde_json::json!({
                "path": project.display().to_string(),
            }))
            .expect("call");
        let s = out.structured.expect("structured");
        // Every section key must still be present (focus=None keeps all
        // three keys in the response shape, even if their arrays are empty).
        assert!(s.get("god_nodes").is_some(), "god_nodes missing: {s}");
        assert!(s.get("communities").is_some(), "communities missing: {s}");
        assert!(s.get("boundaries").is_some(), "boundaries missing: {s}");
        // Communities and boundaries must be empty on a tiny repo — the
        // budget override forces communities=0 / boundaries=0 before passing
        // through to build_arch_overview.
        let communities = s["communities"].as_array().expect("communities array");
        assert!(
            communities.is_empty(),
            "tiny-repo budget must produce zero communities; got: {communities:?}"
        );
        let boundaries = s["boundaries"].as_array().expect("boundaries array");
        assert!(
            boundaries.is_empty(),
            "tiny-repo budget must produce zero boundaries; got: {boundaries:?}"
        );
        // God_nodes are populated from the fixture's chunk_centrality rows
        // (chunks 1 + 2). Cap on tiny-repo override is 5, so we get at most 2
        // for this fixture.
        let gods = s["god_nodes"].as_array().expect("god_nodes array");
        assert!(
            !gods.is_empty(),
            "tiny-repo must still expose god_nodes when centrality table is populated"
        );
        assert!(
            gods.len() <= 5,
            "tiny-repo override caps god_nodes at 5; got {}",
            gods.len()
        );
    }

    /// Defect #15: `tool_architecture`'s symbol-name reads must use
    /// `Chunk::symbol_name()` (writer-side truth in `chunks.chunk_type`), NOT
    /// `StructuredChunkMeta.name`. The two writers can drift; this reader
    /// must agree with the v0.4 hybrid.rs Phase 3d definition-boost signal.
    ///
    /// We plant a synthetic chunk where the two name sources disagree on
    /// purpose, then assert the MCP response surfaces the AstNode.name.
    #[test]
    fn tool_architecture_god_nodes_use_chunk_symbol_name_not_structured_meta() {
        use rusqlite::Connection;
        use semantex_core::chunking::structured_meta::StructuredChunkMeta;

        let dir = tempfile::tempdir().expect("tempdir");
        let semantex_dir = dir.path().join(".semantex");
        std::fs::create_dir_all(&semantex_dir).unwrap();
        let db_path = semantex_dir.join("chunks.db");

        // Chunk with mismatched name fields:
        //   AstNode.name              = "FooBar"      (Phase 3d source)
        //   structured_meta.name      = "WrongName"
        // If tool_architecture ever drifts back to reading
        // structured_meta.name, this test catches it.
        {
            let store = ChunkStore::open(&db_path).expect("open store");
            let meta = StructuredChunkMeta {
                name: Some("WrongName".to_string()),
                kind: Some("function".to_string()),
                ..StructuredChunkMeta::default()
            };
            let chunk = Chunk {
                id: 0,
                file_path: PathBuf::from("src/lib.rs"),
                start_line: 1,
                end_line: 5,
                content: "fn FooBar() {}".to_string(),
                chunk_type: ChunkType::AstNode {
                    name: "FooBar".to_string(),
                    kind: AstNodeKind::Function,
                    language: "rust".to_string(),
                    structured_meta: Some(Box::new(meta)),
                },
            };
            let chunk_id = store.insert_chunk(&chunk, 0xfeed_face, 0).unwrap();
            // Plant chunk_centrality so the chunk is selected as a god_node.
            let conn = Connection::open(&db_path).unwrap();
            conn.execute_batch(
                "CREATE TABLE IF NOT EXISTS chunk_centrality (
                    chunk_id INTEGER PRIMARY KEY,
                    structural_centrality REAL NOT NULL
                );",
            )
            .unwrap();
            conn.execute(
                "INSERT INTO chunk_centrality(chunk_id, structural_centrality) VALUES (?1, 0.9)",
                rusqlite::params![chunk_id as i64],
            )
            .unwrap();
        }

        // Write a valid meta.json so detect_state_fast returns Ready.
        let now_secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let meta = semantex_core::types::IndexMeta {
            schema_version: semantex_core::types::IndexMeta::CURRENT_SCHEMA_VERSION,
            project_path: dir.path().to_path_buf(),
            created_at: now_secs.to_string(),
            updated_at: now_secs.to_string(),
            file_count: 1,
            chunk_count: 1,
            embedding_model: "test".to_string(),
            embedding_dim: 96,
            // v0.4.1 W-Index #4: IndexMeta now persists use_bm25_stemmer so
            // open-time mismatch detection can refuse to load a mis-tuned
            // index. Default-true matches the v0.4 legacy build behaviour.
            use_bm25_stemmer: true,
            dense_backend: "coderank-hnsw".to_string(),
            embedder_fingerprint: "test".to_string(),
        };
        std::fs::write(
            semantex_dir.join("meta.json"),
            serde_json::to_string(&meta).unwrap(),
        )
        .unwrap();

        let project = dir.path().to_path_buf();
        let server = make_server("all");
        let out = server
            .tool_architecture(&serde_json::json!({
                "path": project.display().to_string(),
                "focus": "god_nodes",
            }))
            .expect("call");
        let s = out.structured.expect("structured");
        let gods = s["god_nodes"].as_array().expect("god_nodes array");
        assert_eq!(gods.len(), 1, "expected exactly one god node");
        let symbol = gods[0]["symbol"].as_str().unwrap_or("");
        assert_eq!(
            symbol, "FooBar",
            "tool_architecture must read symbol via Chunk::symbol_name() \
             (got {symbol:?}; expected `FooBar` from AstNode.name, NOT \
             `WrongName` from structured_meta.name)"
        );
        assert_ne!(
            symbol, "WrongName",
            "regression: tool_architecture leaked structured_meta.name \
             instead of Chunk::symbol_name()"
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

    /// S1 (Wave 2): a mid-session `git switch` must gate the structural
    /// tools — `require_index_ready` returns false while the live root still
    /// belongs to another branch, even though `detect_state_fast` says Ready
    /// (its schema/lock checks have no opinion on branches).
    #[test]
    fn require_index_ready_is_false_while_branch_switch_pending() {
        let (_dir, project) = build_minimal_index();

        // Fake a git repo on branch "a" and record "a" as the root's branch.
        let git = project.join(".git");
        std::fs::create_dir_all(git.join("refs").join("heads")).unwrap();
        std::fs::write(git.join("HEAD"), "ref: refs/heads/a\n").unwrap();
        std::fs::write(git.join("refs").join("heads").join("a"), "deadbeef\n").unwrap();
        let root_meta = semantex_core::index::layout::BranchMeta {
            branch: "a".to_string(),
            branch_key: semantex_core::index::layout::branch_key_for_branch("a"),
            head_commit: Some("deadbeef".to_string()),
        };
        std::fs::write(
            project.join(".semantex").join("branch.json"),
            serde_json::to_string(&root_meta).unwrap(),
        )
        .unwrap();

        assert!(
            require_index_ready(&project),
            "no switch pending: index is Ready and branch matches"
        );

        // HEAD moves to "b" — the root still holds "a"'s index, so the
        // structural tools must refuse until the reconcile lands.
        std::fs::write(git.join("HEAD"), "ref: refs/heads/b\n").unwrap();
        std::fs::write(git.join("refs").join("heads").join("b"), "cafebabe\n").unwrap();
        assert!(
            !require_index_ready(&project),
            "branch switch pending: must not answer from the old branch's index"
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
            // v0.4.1 W-Index #4: IndexMeta now persists use_bm25_stemmer so
            // open-time mismatch detection can refuse to load a mis-tuned
            // index. Default-true matches the v0.4 legacy build behaviour.
            use_bm25_stemmer: true,
            dense_backend: "coderank-hnsw".to_string(),
            embedder_fingerprint: "test".to_string(),
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

    #[test]
    fn agent_tool_description_is_self_describing() {
        let agent = McpServer::all_tools()
            .into_iter()
            .find(|t| t.name == "semantex_agent")
            .expect("semantex_agent present");
        let d = agent.description.to_lowercase();
        assert!(d.contains("one call"), "must state one-call contract: {d}");
        assert!(
            d.contains("do not chain") || d.contains("don't chain"),
            "must say don't chain: {d}"
        );
        assert!(
            d.contains("refine"),
            "must tell the agent to refine the query: {d}"
        );
    }

    #[test]
    fn agent_input_schema_declares_2020_12_dialect() {
        let agent = McpServer::all_tools()
            .into_iter()
            .find(|t| t.name == "semantex_agent")
            .unwrap();
        assert_eq!(
            agent.input_schema.get("$schema").and_then(|v| v.as_str()),
            Some("https://json-schema.org/draft/2020-12/schema")
        );
    }

    #[test]
    fn agent_input_schema_declares_scope_param() {
        let agent = McpServer::all_tools()
            .into_iter()
            .find(|t| t.name == "semantex_agent")
            .unwrap();
        let scope = &agent.input_schema["properties"]["scope"];
        assert!(
            scope.is_object(),
            "semantex_agent schema must declare a 'scope' property"
        );
        let one_of = scope["oneOf"]
            .as_array()
            .expect("scope is oneOf[string,array]");
        assert_eq!(one_of.len(), 2);
        // The string branch must NOT be an enum: any non-'repo'/'all' string
        // is valid input (treated as comma-separated project names — the
        // same grammar as the CLI's --scope), so an enum would reject it.
        let string_variant = one_of
            .iter()
            .find(|v| v["type"] == "string")
            .expect("one oneOf branch is a string");
        assert!(
            string_variant.get("enum").is_none(),
            "scope string branch must accept arbitrary project names, not just repo/all"
        );
        let array_variant = one_of
            .iter()
            .find(|v| v["type"] == "array")
            .expect("one oneOf branch is an array of names");
        assert_eq!(array_variant["items"]["type"], "string");
        // The description must document the names-grammar and skipped
        // reporting so clients aren't surprised by the non-enum string.
        let desc = scope["description"].as_str().unwrap_or_default();
        assert!(
            desc.contains("skipped"),
            "scope description documents skipped reporting: {desc}"
        );
    }

    #[test]
    fn search_input_schema_declares_scope_param() {
        let search = McpServer::all_tools()
            .into_iter()
            .find(|t| t.name == "semantex_search")
            .unwrap();
        assert!(
            search.input_schema["properties"]["scope"].is_object(),
            "semantex_search schema must declare a 'scope' property"
        );
    }

    #[test]
    fn agent_output_schema_declares_project_field() {
        let agent = McpServer::all_tools()
            .into_iter()
            .find(|t| t.name == "semantex_agent")
            .unwrap();
        let output = agent
            .output_schema
            .expect("semantex_agent has an output schema");
        assert_eq!(
            output["properties"]["results"]["items"]["properties"]["project"]["type"],
            "string"
        );
    }

    #[test]
    fn search_output_schema_declares_project_field() {
        let search = McpServer::all_tools()
            .into_iter()
            .find(|t| t.name == "semantex_search")
            .unwrap();
        let output = search
            .output_schema
            .expect("semantex_search has an output schema");
        assert_eq!(
            output["properties"]["results"]["items"]["properties"]["project"]["type"],
            "string"
        );
        // Federated calls also emit `skipped` and `note` — both declared.
        let skipped = &output["properties"]["skipped"];
        assert_eq!(skipped["type"], "array");
        for field in ["project", "path", "reason"] {
            assert_eq!(skipped["items"]["properties"][field]["type"], "string");
        }
        assert_eq!(output["properties"]["note"]["type"], "string");
    }

    #[test]
    fn parse_scope_defaults_to_current_repo() {
        use semantex_core::search::federation::SearchScope;
        assert_eq!(
            McpServer::parse_scope(&serde_json::json!({})),
            SearchScope::CurrentRepo
        );
        assert_eq!(
            McpServer::parse_scope(&serde_json::json!({"scope": "repo"})),
            SearchScope::CurrentRepo
        );
        // A malformed (wrong-JSON-type) value degrades to the safe
        // single-repo default rather than erroring.
        assert_eq!(
            McpServer::parse_scope(&serde_json::json!({"scope": 42})),
            SearchScope::CurrentRepo
        );
    }

    /// Scope-grammar parity with the CLI: a non-'repo'/'all' string is
    /// Named (comma-split), NOT silently CurrentRepo — `scope: "frontend"`
    /// must never quietly return current-repo results, and a typo like
    /// "All" resolves to zero targets and is surfaced via `skipped`.
    #[test]
    fn parse_scope_treats_unknown_strings_as_named() {
        use semantex_core::search::federation::SearchScope;
        assert_eq!(
            McpServer::parse_scope(&serde_json::json!({"scope": "frontend"})),
            SearchScope::Named(vec!["frontend".to_string()])
        );
        assert_eq!(
            McpServer::parse_scope(&serde_json::json!({"scope": "frontend,backend"})),
            SearchScope::Named(vec!["frontend".to_string(), "backend".to_string()])
        );
        assert_eq!(
            McpServer::parse_scope(&serde_json::json!({"scope": "All"})),
            SearchScope::Named(vec!["All".to_string()])
        );
    }

    #[test]
    fn parse_scope_all_and_named() {
        use semantex_core::search::federation::SearchScope;
        assert_eq!(
            McpServer::parse_scope(&serde_json::json!({"scope": "all"})),
            SearchScope::All
        );
        assert_eq!(
            McpServer::parse_scope(&serde_json::json!({"scope": ["frontend", "backend"]})),
            SearchScope::Named(vec!["frontend".to_string(), "backend".to_string()])
        );
        // An empty array is not a valid Named scope — falls back to CurrentRepo.
        assert_eq!(
            McpServer::parse_scope(&serde_json::json!({"scope": []})),
            SearchScope::CurrentRepo
        );
    }

    #[test]
    fn agent_missing_query_errors_actionably() {
        let srv = McpServer::new(SemantexConfig::default());
        match srv.tool_agent(&serde_json::json!({})) {
            Err(err) => assert!(err.to_string().contains("query"), "actionable: {err}"),
            Ok(_) => panic!("expected Err for missing query, got Ok"),
        }
    }

    #[test]
    fn agent_output_has_structured_and_text() {
        let srv = McpServer::new(SemantexConfig::default());
        let cwd = std::env::current_dir().unwrap();
        // Only meaningful when this repo has a built index; skip cleanly otherwise.
        if !cwd.join(".semantex/meta.json").exists() {
            return;
        }
        let out = srv
            .tool_agent(&serde_json::json!({"query":"dense backend","depth":"search"}))
            .unwrap();
        // search route over a built index is expected to surface item hits.
        let s = out
            .structured
            .expect("search route over a built index must emit structuredContent");
        assert!(
            s.get("results").is_some(),
            "structuredContent must have results[]"
        );
        assert!(!out.text.is_empty(), "prose summary must stay in content[]");
    }

    #[test]
    fn agent_synthesis_route_is_text_only_no_structured() {
        let srv = McpServer::new(SemantexConfig::default());
        let cwd = std::env::current_dir().unwrap();
        if !cwd.join(".semantex/meta.json").exists() {
            return;
        }
        // 'deep' routes to the synthesis path, which yields no item-list hits ->
        // structuredContent must be None, but the prose answer must still ride in content[].
        let out = srv
            .tool_agent(&serde_json::json!({"query":"how does indexing work","depth":"deep"}))
            .unwrap();
        assert!(
            !out.text.is_empty(),
            "synthesis route must still return prose in content[]"
        );
        assert!(
            out.structured.is_none(),
            "synthesis route (no item hits) must not emit structuredContent"
        );
    }

    // ─────────────────────────────────────────────────────────────────────
    // v13 Wave 2 (history) — `include_history` addendum plumbing
    // ─────────────────────────────────────────────────────────────────────

    fn sample_hit(file: &str) -> semantex_core::server::protocol::SearchResultItem {
        semantex_core::server::protocol::SearchResultItem {
            file: file.to_string(),
            start_line: 1,
            end_line: 5,
            score: 0.9,
            source: "dense".to_string(),
            chunk_type: "ast_node".to_string(),
            name: None,
            language: None,
            content: None,
            kind: None,
            summary: None,
        }
    }

    #[test]
    fn history_addendum_none_when_history_db_missing() {
        let tmp = tempfile::TempDir::new().unwrap();
        let hits = vec![sample_hit("src/lib.rs")];
        assert!(history_addendum_for_hits(tmp.path(), &hits).is_none());
    }

    #[test]
    fn history_addendum_none_when_no_hit_file_has_history() {
        let tmp = tempfile::TempDir::new().unwrap();
        let db_path = semantex_core::index::layout::history_db_path(tmp.path());
        let conn = semantex_core::index::layout::open_history_db(&db_path).unwrap();
        conn.execute(
            "INSERT INTO commits (hash, author, ts, message) VALUES ('h1', 'a', 1, 'm')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO file_commits (path, hash) VALUES ('src/other.rs', 'h1')",
            [],
        )
        .unwrap();
        drop(conn);

        let hits = vec![sample_hit("src/lib.rs")];
        assert!(history_addendum_for_hits(tmp.path(), &hits).is_none());
    }

    #[test]
    fn history_addendum_renders_compact_line_per_unique_file() {
        let tmp = tempfile::TempDir::new().unwrap();
        let db_path = semantex_core::index::layout::history_db_path(tmp.path());
        let conn = semantex_core::index::layout::open_history_db(&db_path).unwrap();
        conn.execute(
            "INSERT INTO commits (hash, author, ts, message) VALUES ('h1', 'Ada', 1700000000, 'Fix parser bug')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO file_commits (path, hash) VALUES ('src/lib.rs', 'h1')",
            [],
        )
        .unwrap();
        drop(conn);

        // Two hits from the same file must collapse into one line.
        let hits = vec![sample_hit("src/lib.rs"), sample_hit("src/lib.rs")];
        let addendum = history_addendum_for_hits(tmp.path(), &hits).expect("must render");
        assert!(addendum.contains("## Recent changes"));
        assert!(addendum.contains("src/lib.rs"));
        assert!(addendum.contains("Fix parser bug"));
        assert!(addendum.contains("Ada"));
        assert_eq!(
            addendum.matches("src/lib.rs").count(),
            1,
            "one hit line per unique file, not per hit"
        );
    }

    #[test]
    fn agent_include_history_default_false_and_schema_documented() {
        // Default parsing: absent `include_history` must behave as `false`
        // (parity with the same `unwrap_or(false)` pattern the schema-shape
        // test below checks is exposed).
        let args = serde_json::json!({"query": "x"});
        assert!(
            !args
                .get("include_history")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false)
        );
        let args_true = serde_json::json!({"query": "x", "include_history": true});
        assert!(
            args_true
                .get("include_history")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false)
        );
    }

    #[test]
    fn agent_tool_schema_shape_is_stable() {
        let t = McpServer::all_tools()
            .into_iter()
            .find(|x| x.name == "semantex_agent")
            .unwrap();
        let props = t.input_schema["properties"].as_object().unwrap();
        for k in [
            "query",
            "queries",
            "path",
            "full_code",
            "budget",
            "mode",
            "depth",
            "focus",
            "response_format",
            "include_history",
        ] {
            assert!(props.contains_key(k), "agent must expose '{k}'");
        }
        assert!(
            t.annotations.is_some(),
            "agent must keep read-only annotations"
        );
    }

    // ─────────────────────────────────────────────────────────────────────
    // mode parameter — JSON Schema shape and description copy
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn mode_schema_has_correct_enum_values() {
        let t = McpServer::all_tools()
            .into_iter()
            .find(|x| x.name == "semantex_agent")
            .unwrap();
        let mode = &t.input_schema["properties"]["mode"];
        let variants = mode["enum"]
            .as_array()
            .expect("mode must have an enum array");
        let names: Vec<&str> = variants.iter().filter_map(|v| v.as_str()).collect();
        for expected in &["auto", "lexical", "search", "structural", "deep"] {
            assert!(
                names.contains(expected),
                "mode enum must contain '{expected}'; got {names:?}"
            );
        }
        assert_eq!(
            names.len(),
            5,
            "mode enum must have exactly 5 values; got {names:?}"
        );
    }

    #[test]
    fn mode_description_mentions_auto_default() {
        let t = McpServer::all_tools()
            .into_iter()
            .find(|x| x.name == "semantex_agent")
            .unwrap();
        let desc = t.input_schema["properties"]["mode"]["description"]
            .as_str()
            .expect("mode must have a description");
        assert!(
            desc.to_lowercase().contains("auto"),
            "mode description must mention 'auto': {desc}"
        );
        assert!(
            desc.to_lowercase().contains("default") || desc.to_lowercase().contains("classifier"),
            "mode description must explain what auto does: {desc}"
        );
    }

    #[test]
    fn agent_description_mentions_mode_guidance() {
        let t = McpServer::all_tools()
            .into_iter()
            .find(|x| x.name == "semantex_agent")
            .unwrap();
        let d = t.description.to_lowercase();
        assert!(d.contains("mode"), "description must mention 'mode': {d}");
        assert!(
            d.contains("auto") || d.contains("classifier"),
            "description must tell the agent that auto is the default: {d}"
        );
    }

    // ─────────────────────────────────────────────────────────────────────
    // mode → route mapping and precedence (unit tests — no index needed)
    // ─────────────────────────────────────────────────────────────────────

    /// Each stable `mode` value maps to the documented `AgentRoute` override.
    /// `mode_to_route` is pure, so this exercises the full resolution table
    /// directly — no live index required (closing the gap where route
    /// dispatch was previously only testable through a built index).
    #[test]
    fn mode_to_route_maps_each_stable_value() {
        use semantex_core::search::agent_classifier::AgentRoute;
        // search / structural / deep map to fixed routes (query is irrelevant).
        assert_eq!(
            mode_to_route(Some("search"), "anything"),
            Some(AgentRoute::Semantic)
        );
        assert_eq!(
            mode_to_route(Some("structural"), "who calls foo"),
            Some(AgentRoute::Structural)
        );
        assert_eq!(
            mode_to_route(Some("deep"), "how does X work"),
            Some(AgentRoute::Deep)
        );
    }

    #[test]
    fn mode_to_route_lexical_plain_symbol_is_exact() {
        use semantex_core::search::agent_classifier::AgentRoute;
        assert_eq!(
            mode_to_route(Some("lexical"), "AgentRoute"),
            Some(AgentRoute::ExactSymbol),
            "plain camel-case symbol → ExactSymbol"
        );
        assert_eq!(
            mode_to_route(Some("lexical"), "tool_agent"),
            Some(AgentRoute::ExactSymbol),
            "plain snake_case symbol → ExactSymbol"
        );
    }

    #[test]
    fn mode_to_route_lexical_regex_promotes_to_regex() {
        use semantex_core::search::agent_classifier::AgentRoute;
        for pat in [
            r"auth\w+Handler", // backslash-class
            "foo|bar",         // unquoted alternation
            "[A-Z]+",          // bracket class
            "(foo|bar)",       // quantifier group
            "^start",          // anchor
            "end$",            // anchor
        ] {
            assert_eq!(
                mode_to_route(Some("lexical"), pat),
                Some(AgentRoute::Regex),
                "regex metacharacters in {pat:?} must promote lexical → Regex"
            );
        }
    }

    #[test]
    fn mode_to_route_auto_and_none_yield_no_override() {
        // Both explicit `auto` and an absent `mode` mean "let the classifier
        // (or depth/env per the caller's precedence) decide" → None.
        assert_eq!(mode_to_route(Some("auto"), "anything"), None);
        assert_eq!(mode_to_route(None, "anything"), None);
    }

    #[test]
    fn mode_to_route_unknown_value_falls_back_to_none() {
        // Unknown values warn and fall back to None (auto) rather than error,
        // so a stray/typo'd mode never breaks the call.
        assert_eq!(mode_to_route(Some("bogus"), "anything"), None);
        assert_eq!(mode_to_route(Some(""), "anything"), None);
    }

    /// `mode` schema shape: 2020-12 dialect + plain string type.
    #[test]
    fn mode_schema_2020_12_dialect() {
        let t = McpServer::all_tools()
            .into_iter()
            .find(|x| x.name == "semantex_agent")
            .unwrap();
        assert_eq!(
            t.input_schema.get("$schema").and_then(|v| v.as_str()),
            Some("https://json-schema.org/draft/2020-12/schema"),
            "input_schema must declare 2020-12 dialect"
        );
        // mode is a sibling of the existing depth/response_format enums —
        // both use plain string type.
        let mode = &t.input_schema["properties"]["mode"];
        assert_eq!(
            mode["type"].as_str(),
            Some("string"),
            "mode must be type string"
        );
    }

    // ─────────────────────────────────────────────────────────────────────
    // v13 Wave 2 — semantex_docs_context: schema + dispatch tests
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn docs_context_tool_is_registered_read_only_and_additive() {
        let tools = McpServer::all_tools();
        let tool = tools
            .iter()
            .find(|t| t.name == "semantex_docs_context")
            .expect("semantex_docs_context must be registered");
        // Registered additively, after the original v0.2 set. It's no longer
        // literally the LAST entry — the MEMORY workstream (same v13 Wave 2
        // batch) appended `semantex_memory_save`/`semantex_memory_recall`
        // after it, following the same additive-registration convention —
        // but it must still come before those newer entries.
        let docs_context_idx = tools
            .iter()
            .position(|t| t.name == "semantex_docs_context")
            .unwrap();
        for later in ["semantex_memory_save", "semantex_memory_recall"] {
            let idx = tools.iter().position(|t| t.name == later);
            if let Some(idx) = idx {
                assert!(
                    docs_context_idx < idx,
                    "semantex_docs_context must precede later additive registrations like {later}"
                );
            }
        }
        assert_eq!(
            tool.annotations.as_ref().and_then(|a| a.read_only_hint),
            Some(true),
            "semantex_docs_context must be readOnlyHint: true"
        );
        assert_eq!(
            tool.input_schema["required"],
            serde_json::json!(["scope"]),
            "scope must be required"
        );
        // Visible in the default "all" toolset (no allowlist entry needed —
        // tools_for_toolset returns all_tools() unfiltered for "all" and any
        // unrecognized name; only "core" and "structural" narrow it).
        let server = make_server("all");
        let visible = server.tools_for_toolset("all");
        assert!(
            visible.iter().any(|t| t.name == "semantex_docs_context"),
            "semantex_docs_context must be visible in the 'all' toolset"
        );
    }

    #[test]
    fn tool_docs_context_missing_scope_errors() {
        let (_dir, project) = build_minimal_index();
        let server = make_server("all");
        let result = server.tool_docs_context(&serde_json::json!({
            "path": project.display().to_string(),
        }));
        assert!(result.is_err(), "missing 'scope' should error");
    }

    #[test]
    fn tool_docs_context_overview_dispatches_and_returns_structured_scaffold() {
        let (_dir, project) = build_minimal_index();
        let server = make_server("all");
        let out = server
            .tool_docs_context(&serde_json::json!({
                "scope": "overview",
                "path": project.display().to_string(),
            }))
            .expect("call");
        let s = out.structured.expect("structured");
        assert!(s.get("god_nodes").is_some(), "god_nodes missing: {s}");
        assert!(
            s.get("module_inventory").is_some(),
            "module_inventory missing: {s}"
        );
        assert!(
            s.get("language_stats").is_some(),
            "language_stats missing: {s}"
        );
        assert!(
            out.text.contains("Overview scaffold"),
            "text rendering must be human-readable: {}",
            out.text
        );
    }

    #[test]
    fn tool_docs_context_module_scope_dispatches_and_finds_symbol() {
        let (_dir, project) = build_minimal_index();
        let server = make_server("all");
        let out = server
            .tool_docs_context(&serde_json::json!({
                "scope": { "module": "src/a.rs" },
                "path": project.display().to_string(),
            }))
            .expect("call");
        let s = out.structured.expect("structured");
        assert_eq!(s["path"].as_str(), Some("src/a.rs"));
        let symbols = s["symbols"].as_array().expect("symbols array");
        assert_eq!(symbols.len(), 1, "src/a.rs has exactly one symbol: foo");
        assert_eq!(symbols[0]["name"].as_str(), Some("foo"));
        assert_eq!(symbols[0]["provenance"]["file"].as_str(), Some("src/a.rs"));
        assert!(
            out.text.contains("Module scaffold: src/a.rs"),
            "text rendering must name the module: {}",
            out.text
        );
    }

    #[test]
    fn tool_docs_context_module_scope_resolves_call_edge() {
        // Fixture: bar (src/b.rs) calls foo (src/a.rs). Querying src/b.rs's
        // module scaffold must show a resolved outgoing call edge to foo;
        // querying src/a.rs must show a resolved incoming call edge from bar.
        let (_dir, project) = build_minimal_index();
        let server = make_server("all");

        let out_b = server
            .tool_docs_context(&serde_json::json!({
                "scope": { "module": "src/b.rs" },
                "path": project.display().to_string(),
            }))
            .expect("call");
        let s_b = out_b.structured.expect("structured");
        let calls_out = s_b["calls_out"].as_array().expect("calls_out array");
        assert_eq!(calls_out.len(), 1);
        assert_eq!(calls_out[0]["callee_name"].as_str(), Some("foo"));
        assert_eq!(
            calls_out[0]["resolved"]["file"].as_str(),
            Some("src/a.rs"),
            "call edge to foo must resolve to src/a.rs"
        );

        let out_a = server
            .tool_docs_context(&serde_json::json!({
                "scope": { "module": "src/a.rs" },
                "path": project.display().to_string(),
            }))
            .expect("call");
        let s_a = out_a.structured.expect("structured");
        let calls_in = s_a["calls_in"].as_array().expect("calls_in array");
        assert_eq!(calls_in.len(), 1);
        assert_eq!(calls_in[0]["caller_symbol"].as_str(), Some("bar"));
        assert_eq!(calls_in[0]["provenance"]["file"].as_str(), Some("src/b.rs"));
    }

    #[test]
    fn tool_docs_context_unknown_scope_shape_errors() {
        let (_dir, project) = build_minimal_index();
        let server = make_server("all");
        let result = server.tool_docs_context(&serde_json::json!({
            "scope": "bogus",
            "path": project.display().to_string(),
        }));
        assert!(result.is_err(), "unknown scope string should error");
    }

    // ─────────────────────────────────────────────────────────────────────
    // v13 Wave 2 — semantex_memory_save / semantex_memory_recall: schema +
    // dispatch tests (mirrors the semantex_docs_context test pattern above).
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn memory_tools_are_registered_additive_and_correctly_annotated() {
        let tools = McpServer::all_tools();

        let save = tools
            .iter()
            .find(|t| t.name == "semantex_memory_save")
            .expect("semantex_memory_save must be registered");
        assert_eq!(
            save.annotations.as_ref().and_then(|a| a.read_only_hint),
            Some(false),
            "semantex_memory_save mutates state — must NOT be readOnlyHint: true"
        );
        assert_eq!(
            save.input_schema["required"],
            serde_json::json!(["content"]),
            "content must be the only required field"
        );

        let recall = tools
            .iter()
            .find(|t| t.name == "semantex_memory_recall")
            .expect("semantex_memory_recall must be registered");
        assert_eq!(
            recall.annotations.as_ref().and_then(|a| a.read_only_hint),
            Some(true),
            "semantex_memory_recall must be readOnlyHint: true"
        );

        // Additive: both come after semantex_docs_context in the list.
        // semantex_history was added after the memory tools, so memory_recall
        // is now second-to-last and memory_save is third-to-last.
        let save_idx = tools
            .iter()
            .position(|t| t.name == "semantex_memory_save")
            .unwrap();
        let recall_idx = tools
            .iter()
            .position(|t| t.name == "semantex_memory_recall")
            .unwrap();
        assert_eq!(
            recall_idx,
            tools.len() - 2,
            "semantex_memory_recall must be second-to-last (before semantex_history)"
        );
        assert_eq!(
            save_idx,
            tools.len() - 3,
            "semantex_memory_save must immediately precede semantex_memory_recall"
        );

        // Visible in the default "all" toolset.
        let server = make_server("all");
        let visible = server.tools_for_toolset("all");
        assert!(visible.iter().any(|t| t.name == "semantex_memory_save"));
        assert!(visible.iter().any(|t| t.name == "semantex_memory_recall"));
    }

    #[test]
    fn tool_memory_save_missing_content_errors() {
        let dir = tempfile::tempdir().unwrap();
        let server = make_server("all");
        let result = server.tool_memory_save(&serde_json::json!({
            "path": dir.path().display().to_string(),
        }));
        assert!(result.is_err(), "missing 'content' should error");
    }

    #[test]
    fn tool_memory_save_then_recall_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let server = make_server("all");

        let save_out = server
            .tool_memory_save(&serde_json::json!({
                "content": "always run migrations before enforcing the note cap",
                "scope": "module:index",
                "tags": ["gotcha"],
                "path": dir.path().display().to_string(),
            }))
            .expect("save should succeed");
        let structured = save_out.structured.expect("structured content");
        let id = structured["id"].as_i64().expect("id present");
        assert!(id > 0);
        assert!(save_out.text.contains("Saved note"));

        let recall_out = server
            .tool_memory_recall(&serde_json::json!({
                "query": "migrations note cap",
                "scope": "module:index",
                "path": dir.path().display().to_string(),
            }))
            .expect("recall should succeed");
        let structured = recall_out.structured.expect("structured content");
        let notes = structured["notes"].as_array().expect("notes array");
        assert_eq!(notes.len(), 1);
        assert_eq!(notes[0]["id"].as_i64(), Some(id));
        assert_eq!(notes[0]["scope"].as_str(), Some("module:index"));
    }

    #[test]
    fn tool_memory_recall_defaults_limit_and_handles_empty_store() {
        let dir = tempfile::tempdir().unwrap();
        let server = make_server("all");
        let out = server
            .tool_memory_recall(&serde_json::json!({
                "path": dir.path().display().to_string(),
            }))
            .expect("recall on empty store should succeed, not error");
        let structured = out.structured.expect("structured content");
        assert_eq!(structured["count"].as_u64(), Some(0));
    }

    #[test]
    fn tool_memory_dispatch_reachable_via_handle_tool_call() {
        // Proves the `handle_tool_call` match arms (not just the direct
        // method calls above) reach the memory handlers, same pattern the
        // M1-M6 backward-compat comment above references.
        let dir = tempfile::tempdir().unwrap();
        let server = make_server("all");
        let writer = super::NotificationWriter {
            stdout: std::sync::Arc::new(parking_lot::Mutex::new(std::io::BufWriter::new(
                std::io::stdout(),
            ))),
        };

        let save_response = server.handle_tool_call(
            Some(serde_json::json!(1)),
            &serde_json::json!({
                "name": "semantex_memory_save",
                "arguments": {
                    "content": "dispatch reachability check",
                    "path": dir.path().display().to_string(),
                }
            }),
            &writer,
        );
        let save_json = serde_json::to_value(&save_response).unwrap();
        assert_ne!(
            save_json["result"]["isError"],
            serde_json::json!(true),
            "save dispatch should not error: {save_json}"
        );

        let recall_response = server.handle_tool_call(
            Some(serde_json::json!(2)),
            &serde_json::json!({
                "name": "semantex_memory_recall",
                "arguments": {
                    "query": "dispatch reachability",
                    "path": dir.path().display().to_string(),
                }
            }),
            &writer,
        );
        let recall_json = serde_json::to_value(&recall_response).unwrap();
        assert_ne!(
            recall_json["result"]["isError"],
            serde_json::json!(true),
            "recall dispatch should not error: {recall_json}"
        );
        let text = recall_json["result"]["content"][0]["text"]
            .as_str()
            .unwrap_or_default();
        assert!(
            text.contains("dispatch reachability check") || text.contains("memory note"),
            "recall text should reflect the saved note: {text}"
        );
    }

    // ─────────────────────────────────────────────────────────────────────
    // v13 Wave 2 (history) — semantex_history tool tests
    // ─────────────────────────────────────────────────────────────────────

    /// Scripted temp git repo for history-tool tests: `commits` = (file, message).
    /// No semantex index is built — semantex_history must work without one.
    fn scripted_git_repo(commits: &[(&str, &str)]) -> tempfile::TempDir {
        let dir = tempfile::TempDir::new().unwrap();
        let git = |args: &[&str]| {
            let status = std::process::Command::new("git")
                .arg("-C")
                .arg(dir.path())
                .args(args)
                .status()
                .expect("git must be on PATH");
            assert!(status.success(), "git {args:?} failed");
        };
        git(&["init", "-q"]);
        git(&["config", "user.email", "test@example.com"]);
        git(&["config", "user.name", "Test User"]);
        for (file, msg) in commits {
            std::fs::write(dir.path().join(file), *msg).unwrap();
            git(&["add", "."]);
            git(&["commit", "-q", "-m", msg]);
        }
        dir
    }

    #[test]
    fn tool_history_lists_recent_commits_without_an_index() {
        let repo = scripted_git_repo(&[("a.rs", "add a"), ("b.rs", "add b"), ("c.rs", "add c")]);
        let server = make_server("all");
        let out = server
            .tool_history(&serde_json::json!({ "path": repo.path().display().to_string() }))
            .expect("tool_history");
        let text = &out.text;
        assert!(text.contains("add c"), "most recent commit missing: {text}");
        assert!(text.contains("add a"));
        let commits = out.structured.expect("structured")["commits"]
            .as_array()
            .expect("commits array")
            .len();
        assert_eq!(commits, 3);
    }

    #[test]
    fn tool_history_non_git_dir_is_graceful() {
        let dir = tempfile::TempDir::new().unwrap();
        let server = make_server("all");
        let out = server
            .tool_history(&serde_json::json!({ "path": dir.path().display().to_string() }))
            .expect("non-git dir must not error");
        assert!(out.text.contains("No git history"), "got: {}", out.text);
    }

    #[test]
    fn tool_history_since_tag_excludes_tagged_commit() {
        // Commit timestamps are pinned a day apart via GIT_COMMITTER_DATE
        // (instead of sleeping past %ct's 1-second resolution): `since`
        // filters strictly-greater on the committer timestamp.
        let repo = tempfile::TempDir::new().unwrap();
        let git = |args: &[&str], date: &str| {
            let status = std::process::Command::new("git")
                .arg("-C")
                .arg(repo.path())
                .args(args)
                .env("GIT_AUTHOR_DATE", date)
                .env("GIT_COMMITTER_DATE", date)
                .status()
                .expect("git must be on PATH");
            assert!(status.success(), "git {args:?} failed");
        };
        let tagged = "2026-01-01T00:00:00 +0000";
        let after = "2026-01-02T00:00:00 +0000";
        git(&["init", "-q"], tagged);
        git(&["config", "user.email", "test@example.com"], tagged);
        git(&["config", "user.name", "Test User"], tagged);
        std::fs::write(repo.path().join("a.rs"), "before tag").unwrap();
        git(&["add", "."], tagged);
        git(&["commit", "-q", "-m", "before tag"], tagged);
        git(&["tag", "v1"], tagged);
        std::fs::write(repo.path().join("b.rs"), "x").unwrap();
        git(&["add", "."], after);
        git(&["commit", "-q", "-m", "after tag"], after);

        let server = make_server("all");
        let out = server
            .tool_history(&serde_json::json!({
                "path": repo.path().display().to_string(),
                "since": "v1",
            }))
            .expect("tool_history");
        let text = &out.text;
        assert!(text.contains("after tag"), "got: {text}");
        assert!(
            !text.contains("before tag"),
            "tagged commit must be excluded: {text}"
        );
    }

    #[test]
    fn tool_history_bad_since_is_an_error_with_accepted_forms() {
        let repo = scripted_git_repo(&[("a.rs", "one")]);
        let server = make_server("all");
        match server.tool_history(&serde_json::json!({
            "path": repo.path().display().to_string(),
            "since": "definitely-not-a-ref",
        })) {
            Err(err) => assert!(
                err.to_string().contains("could not resolve since"),
                "actionable: {err}"
            ),
            Ok(_) => panic!("expected Err for unresolvable since, got Ok"),
        }
    }

    #[test]
    fn tool_history_detail_mode_returns_bounded_patch() {
        let repo = scripted_git_repo(&[(
            "a.rs",
            "fn main() { /* a reasonably long line of content here */ }",
        )]);
        let server = make_server("all");
        // Fetch the sha via list mode's structured output.
        let list = server
            .tool_history(&serde_json::json!({ "path": repo.path().display().to_string() }))
            .unwrap();
        let sha = list.structured.unwrap()["commits"][0]["hash"]
            .as_str()
            .unwrap()
            .to_string();

        let out = server
            .tool_history(&serde_json::json!({
                "path": repo.path().display().to_string(),
                "commits": [sha],
                "budget": 1000,
            }))
            .expect("detail mode");
        let s = out.structured.expect("structured");
        let d = &s["commits"][0];
        assert!(d["stat"].as_str().unwrap().contains("a.rs"));
        assert!(d["patch"].as_str().unwrap().len() <= 1000);
        assert!(out.text.contains("a.rs"));
    }

    #[test]
    fn tool_history_no_upstream_note_when_no_remote_configured() {
        let repo = scripted_git_repo(&[("a.rs", "one")]);
        let server = make_server("all");
        let out = server
            .tool_history(&serde_json::json!({ "path": repo.path().display().to_string() }))
            .unwrap();
        assert!(out.structured.unwrap()["upstream"].is_null());
        assert!(!out.text.contains("behind"));
    }

    #[test]
    fn tool_history_behind_note_appears_when_stale() {
        let upstream = scripted_git_repo(&[("a.rs", "one"), ("b.rs", "two")]);
        std::process::Command::new("git")
            .arg("-C")
            .arg(upstream.path())
            .args(["branch", "-M", "main"])
            .status()
            .unwrap();
        let local = tempfile::TempDir::new().unwrap();
        std::process::Command::new("git")
            .args([
                "clone",
                "-q",
                upstream.path().to_str().unwrap(),
                local.path().to_str().unwrap(),
            ])
            .status()
            .unwrap();

        std::fs::write(upstream.path().join("c.rs"), "three").unwrap();
        std::process::Command::new("git")
            .arg("-C")
            .arg(upstream.path())
            .args(["add", "."])
            .status()
            .unwrap();
        std::process::Command::new("git")
            .arg("-C")
            .arg(upstream.path())
            .args(["commit", "-q", "-m", "three"])
            .status()
            .unwrap();
        std::process::Command::new("git")
            .arg("-C")
            .arg(local.path())
            .args(["fetch", "-q", "origin"])
            .status()
            .unwrap();

        let server = make_server("all");
        let out = server
            .tool_history(&serde_json::json!({ "path": local.path().display().to_string() }))
            .unwrap();
        let s = out.structured.unwrap();
        assert_eq!(s["upstream"]["behind"].as_u64(), Some(1));
        assert!(out.text.contains("behind origin/main"), "got: {}", out.text);
    }

    #[test]
    fn tool_history_other_branches_is_opt_in() {
        let repo = scripted_git_repo(&[("a.rs", "one")]);
        std::process::Command::new("git")
            .arg("-C")
            .arg(repo.path())
            .args(["branch", "-M", "main"])
            .status()
            .unwrap();
        std::process::Command::new("git")
            .arg("-C")
            .arg(repo.path())
            .args(["checkout", "-q", "-b", "feature-a"])
            .status()
            .unwrap();
        std::fs::write(repo.path().join("b.rs"), "two").unwrap();
        std::process::Command::new("git")
            .arg("-C")
            .arg(repo.path())
            .args(["add", "."])
            .status()
            .unwrap();
        std::process::Command::new("git")
            .arg("-C")
            .arg(repo.path())
            .args(["commit", "-q", "-m", "feature commit"])
            .status()
            .unwrap();
        std::process::Command::new("git")
            .arg("-C")
            .arg(repo.path())
            .args(["checkout", "-q", "main"])
            .status()
            .unwrap();

        let server = make_server("all");
        let default_out = server
            .tool_history(&serde_json::json!({ "path": repo.path().display().to_string() }))
            .unwrap();
        assert_eq!(
            default_out.structured.unwrap()["other_branches"]
                .as_array()
                .unwrap()
                .len(),
            0,
            "other_branches must be empty unless explicitly requested"
        );

        let opt_in_out = server
            .tool_history(&serde_json::json!({
                "path": repo.path().display().to_string(),
                "other_branches": true,
            }))
            .unwrap();
        let branches = opt_in_out.structured.unwrap()["other_branches"]
            .as_array()
            .unwrap()
            .clone();
        assert_eq!(branches.len(), 1, "got: {branches:?}");
        assert_eq!(branches[0]["name"].as_str(), Some("feature-a"));
    }

    /// A sha that doesn't resolve must surface in structured `errors`, not
    /// just the human-readable text — otherwise a programmatic consumer
    /// reading only `structured` can't tell "not found" from "no commits".
    #[test]
    fn tool_history_detail_mode_invalid_sha_reports_structured_error() {
        let repo = scripted_git_repo(&[("a.rs", "one")]);
        let server = make_server("all");
        let bogus_sha = "0000000000000000000000000000000000dead";

        let out = server
            .tool_history(&serde_json::json!({
                "path": repo.path().display().to_string(),
                "commits": [bogus_sha],
            }))
            .expect("detail mode should not fail the whole call");
        let s = out.structured.expect("structured");
        assert!(
            s["commits"].as_array().unwrap().is_empty(),
            "no commit entry for an unresolved sha: {s}"
        );
        let errors = s["errors"].as_array().expect("errors array");
        assert_eq!(errors.len(), 1, "got: {errors:?}");
        assert_eq!(errors[0]["sha"].as_str().unwrap(), bogus_sha);
        assert!(!errors[0]["error"].as_str().unwrap().is_empty());
        assert!(out.text.contains(bogus_sha) && out.text.contains("[error:"));
    }

    /// In-memory registry pointing at real temp git repos — no SEMANTEX_HOME
    /// mutation (which would race parallel unit tests; the code-search
    /// federation suite needs a whole separate process for that).
    fn registry_of(
        entries: &[(&str, &std::path::Path)],
    ) -> semantex_core::index::registry::RegistryV2 {
        let mut reg = semantex_core::index::registry::RegistryV2::default();
        for (name, path) in entries {
            reg.projects
                .push(semantex_core::index::registry::ProjectEntry {
                    path: path.to_path_buf(),
                    project_id: (*name).to_string(),
                    display_name: (*name).to_string(),
                    branches: vec![],
                    embedder_fingerprint: String::new(),
                    is_worktree: false,
                });
        }
        reg
    }

    #[test]
    fn history_federated_named_scope_sections_and_skipped() {
        let repo_a = scripted_git_repo(&[("a.rs", "alpha commit")]);
        let repo_b = scripted_git_repo(&[("b.rs", "beta commit")]);
        let reg = registry_of(&[("proj-a", repo_a.path()), ("proj-b", repo_b.path())]);
        let scope = semantex_core::search::federation::SearchScope::Named(vec![
            "proj-a".to_string(),
            "ghost".to_string(),
        ]);

        let (text, structured) =
            history_federated(&reg, &scope, &serde_json::json!({}), 12_000).unwrap();
        assert!(text.contains("## [proj-a]"), "got: {text}");
        assert!(text.contains("alpha commit"));
        assert!(
            !text.contains("beta commit"),
            "unnamed project must not appear"
        );
        assert!(
            text.contains("ghost"),
            "skipped name must be surfaced: {text}"
        );
        assert_eq!(structured["projects"].as_array().unwrap().len(), 1);
        assert_eq!(structured["skipped"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn history_federated_all_scope_covers_every_project() {
        let repo_a = scripted_git_repo(&[("a.rs", "alpha commit")]);
        let repo_b = scripted_git_repo(&[("b.rs", "beta commit")]);
        let reg = registry_of(&[("proj-a", repo_a.path()), ("proj-b", repo_b.path())]);

        let (text, structured) = history_federated(
            &reg,
            &semantex_core::search::federation::SearchScope::All,
            &serde_json::json!({}),
            12_000,
        )
        .unwrap();
        assert!(text.contains("alpha commit") && text.contains("beta commit"));
        assert_eq!(structured["projects"].as_array().unwrap().len(), 2);
    }

    /// Detail mode fans out across repos: only one repo owns the sha, so
    /// every other project must render a per-commit `[error: ...]` line —
    /// not fail the whole call. Locks the passthrough in as intentional.
    #[test]
    fn history_federated_detail_mode_degrades_gracefully_across_repos() {
        let repo_a = scripted_git_repo(&[("a.rs", "alpha commit")]);
        let repo_b = scripted_git_repo(&[("b.rs", "beta commit")]);
        let reg = registry_of(&[("proj-a", repo_a.path()), ("proj-b", repo_b.path())]);
        let out = std::process::Command::new("git")
            .arg("-C")
            .arg(repo_a.path())
            .args(["rev-parse", "HEAD"])
            .output()
            .unwrap();
        let sha = String::from_utf8(out.stdout).unwrap().trim().to_string();

        let (text, structured) = history_federated(
            &reg,
            &semantex_core::search::federation::SearchScope::All,
            &serde_json::json!({ "commits": [sha] }),
            12_000,
        )
        .unwrap();
        assert!(
            text.contains("alpha commit"),
            "owning repo must expand the sha: {text}"
        );
        assert!(
            text.contains("[error:"),
            "non-owning repo must degrade to a per-commit error line: {text}"
        );
        let projects = structured["projects"].as_array().unwrap();
        assert_eq!(projects.len(), 2, "both projects report a section");
        assert!(
            structured["skipped"].as_array().unwrap().is_empty(),
            "a missing sha is not a project-level skip"
        );
        assert!(
            projects[0]["errors"].as_array().unwrap().is_empty(),
            "owning repo (proj-a) has no error: {projects:?}"
        );
        let non_owner_errors = projects[1]["errors"].as_array().unwrap();
        assert_eq!(
            non_owner_errors.len(),
            1,
            "non-owning repo (proj-b) must report the sha in structured errors: {projects:?}"
        );
        assert_eq!(non_owner_errors[0]["sha"].as_str().unwrap(), sha);
    }

    /// Two scope names resolving to the same project (display name + full
    /// path) must produce exactly one section, not a duplicate.
    #[test]
    fn history_federated_dedups_names_resolving_to_one_project() {
        let repo_a = scripted_git_repo(&[("a.rs", "alpha commit")]);
        let reg = registry_of(&[("proj-a", repo_a.path())]);
        let scope = semantex_core::search::federation::SearchScope::Named(vec![
            "proj-a".to_string(),
            repo_a.path().display().to_string(),
        ]);

        let (text, structured) =
            history_federated(&reg, &scope, &serde_json::json!({}), 12_000).unwrap();
        assert_eq!(
            text.matches("## [proj-a]").count(),
            1,
            "duplicate section: {text}"
        );
        assert_eq!(structured["projects"].as_array().unwrap().len(), 1);
        assert!(structured["skipped"].as_array().unwrap().is_empty());
    }

    #[test]
    fn history_federated_all_scope_excludes_worktrees() {
        let repo_a = scripted_git_repo(&[("a.rs", "alpha commit")]);
        let repo_b = scripted_git_repo(&[("b.rs", "beta commit")]);
        let mut reg = registry_of(&[("proj-a", repo_a.path()), ("proj-b", repo_b.path())]);
        reg.projects[1].is_worktree = true; // proj-b simulates a worktree registration

        let (text, structured) = history_federated(
            &reg,
            &semantex_core::search::federation::SearchScope::All,
            &serde_json::json!({}),
            12_000,
        )
        .unwrap();
        assert!(text.contains("alpha commit"));
        assert!(
            !text.contains("beta commit"),
            "worktree project must be excluded from scope=all: {text}"
        );
        assert_eq!(structured["projects"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn history_federated_named_scope_still_finds_a_worktree() {
        let repo_a = scripted_git_repo(&[("a.rs", "alpha commit")]);
        let mut reg = registry_of(&[("proj-a", repo_a.path())]);
        reg.projects[0].is_worktree = true;

        let (text, _structured) = history_federated(
            &reg,
            &semantex_core::search::federation::SearchScope::Named(vec!["proj-a".to_string()]),
            &serde_json::json!({}),
            12_000,
        )
        .unwrap();
        assert!(
            text.contains("alpha commit"),
            "naming a worktree project explicitly must still resolve it: {text}"
        );
    }

    #[test]
    fn history_federated_all_scope_dedups_nested_registry_paths() {
        let outer = scripted_git_repo(&[("a.rs", "outer commit")]);
        let inner_path = outer.path().join("nested-crate");
        std::fs::create_dir_all(&inner_path).unwrap();
        let reg = registry_of(&[
            ("outer-repo", outer.path()),
            ("inner-crate", inner_path.as_path()),
        ]);

        let (text, structured) = history_federated(
            &reg,
            &semantex_core::search::federation::SearchScope::All,
            &serde_json::json!({}),
            12_000,
        )
        .unwrap();
        let projects = structured["projects"].as_array().unwrap();
        assert_eq!(
            projects.len(),
            1,
            "nested registry path must be deduped in scope=all: {projects:?}"
        );
        assert_eq!(projects[0]["project"].as_str(), Some("outer-repo"));
        assert!(text.contains("outer commit"));
    }

    #[test]
    fn history_federated_propagates_upstream_and_other_branches_fields() {
        let repo_a = scripted_git_repo(&[("a.rs", "alpha commit")]);
        let reg = registry_of(&[("proj-a", repo_a.path())]);

        let (_text, structured) = history_federated(
            &reg,
            &semantex_core::search::federation::SearchScope::All,
            &serde_json::json!({}),
            12_000,
        )
        .unwrap();
        let projects = structured["projects"].as_array().unwrap();
        assert!(projects[0]["upstream"].is_null(), "no remote configured");
        assert_eq!(projects[0]["other_branches"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn history_federated_empty_registry_says_why() {
        let reg = semantex_core::index::registry::RegistryV2::default();
        let (text, structured) = history_federated(
            &reg,
            &semantex_core::search::federation::SearchScope::All,
            &serde_json::json!({}),
            12_000,
        )
        .unwrap();
        assert!(text.contains("registry has no projects"), "got: {text}");
        assert!(structured["projects"].as_array().unwrap().is_empty());
    }
}

#[cfg(all(feature = "llm", test))]
mod llm_runtime_wiring_tests {
    use super::*;
    use semantex_core::config::SemantexConfig;

    /// Serializes the `SEMANTEX_LLM_*` env mutation in this test binary so two
    /// LLM-env tests never race on process env. (`semantex-core`'s own
    /// `TEST_ENV_LOCK` is `pub(crate)` and not reachable across crates; this is
    /// the mcp-crate-local equivalent.)
    static MCP_LLM_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Regression for the v0.7.1 MCP wiring gap: `McpServer` must construct a
    /// shared Tokio runtime once, so `tool_agent` can chain `.with_runtime(...)`
    /// instead of falling back to AgentPipeline's per-call runtime branch.
    ///
    /// Building a `current_thread` runtime in a test environment always
    /// succeeds, so the field must be `Some` after construction (when an LLM
    /// backend is configured). With no LLM env configured, the backend is
    /// `None` and the runtime is intentionally left `None`, so we configure a
    /// minimal env-driven backend to exercise the runtime-build path.
    #[test]
    fn mcp_server_builds_shared_runtime() {
        let _guard = MCP_LLM_ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // SAFETY: guarded by MCP_LLM_ENV_LOCK; restored before the lock drops.
        // A model name is enough for GenaiBackend::from_env() to yield Some,
        // which makes McpServer build the shared runtime. No network is used.
        let prev_model = std::env::var("SEMANTEX_LLM_MODEL").ok();
        let prev_provider = std::env::var("SEMANTEX_LLM_PROVIDER").ok();
        unsafe {
            std::env::set_var("SEMANTEX_LLM_MODEL", "gpt-4o-mini");
            std::env::set_var("SEMANTEX_LLM_PROVIDER", "openai");
        }

        let server = McpServer::new(SemantexConfig::default());
        let has_runtime = server.llm_runtime.is_some();

        // SAFETY: guarded by TEST_ENV_LOCK.
        unsafe {
            match prev_model {
                Some(v) => std::env::set_var("SEMANTEX_LLM_MODEL", v),
                None => std::env::remove_var("SEMANTEX_LLM_MODEL"),
            }
            match prev_provider {
                Some(v) => std::env::set_var("SEMANTEX_LLM_PROVIDER", v),
                None => std::env::remove_var("SEMANTEX_LLM_PROVIDER"),
            }
        }

        assert!(
            server.llm.is_some(),
            "test precondition: an LLM backend must be configured from env"
        );
        assert!(
            has_runtime,
            "McpServer must hold a shared Tokio runtime for the MCP HyDE/classify path"
        );
    }
}
