use crate::protocol::{
    InitializeResult, JsonRpcRequest, JsonRpcResponse, ServerCapabilities, ServerInfo, Tool,
    ToolCallResult, ToolContent, ToolsCapability,
};
use anyhow::{Context, Result};
use parking_lot::Mutex;
use semantex_core::config::SemantexConfig;
use semantex_core::index::builder::IndexBuilder;
use semantex_core::index::state::{self, IndexState};
use semantex_core::search::SearchQuery;
use semantex_core::search::hybrid::HybridSearcher;
use semantex_core::search::ripgrep_fallback;
use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

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

pub struct McpServer {
    config: SemantexConfig,
    /// LRU cache of `HybridSearcher` instances keyed by canonical index directory.
    /// Avoids reloading the PLAID index, sparse index, and `SQLite` per request.
    cache: Mutex<HashMap<PathBuf, Arc<CachedSearcher>>>,
    max_cached: usize,
    /// Per-project indexing status — prevents double-spawning background index builds.
    index_states: Arc<Mutex<HashMap<PathBuf, IndexingStatus>>>,
}

/// Idle eviction timeout: entries unused for this duration are dropped
const IDLE_TIMEOUT_SECS: u64 = 600; // 10 minutes

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
        }
    }

    /// Get or create a cached searcher for the given index directory.
    fn get_searcher(&self, index_dir: &std::path::Path) -> Result<Arc<CachedSearcher>> {
        let mut cache = self.cache.lock();

        // Evict idle entries (unused for >10 minutes)
        let now = Instant::now();
        cache.retain(|_, entry| {
            now.duration_since(*entry.last_used.lock()).as_secs() < IDLE_TIMEOUT_SECS
        });

        // Check cache — update last_used on hit
        if let Some(entry) = cache.get(index_dir) {
            tracing::debug!(index_dir = %index_dir.display(), "Searcher cache hit");
            *entry.last_used.lock() = now;
            return Ok(Arc::clone(entry));
        }

        // Cache miss — create new searcher
        tracing::info!(index_dir = %index_dir.display(), "Searcher cache miss, opening new searcher");
        let searcher = HybridSearcher::open(index_dir, &self.config)?;

        let entry = Arc::new(CachedSearcher {
            searcher,
            last_used: Mutex::new(now),
        });

        // Evict LRU if at capacity
        if cache.len() >= self.max_cached
            && let Some(lru_key) = cache
                .iter()
                .min_by_key(|(_, v)| *v.last_used.lock())
                .map(|(k, _)| k.clone())
        {
            tracing::info!(evicted = %lru_key.display(), "Evicting LRU searcher from cache");
            cache.remove(&lru_key);
        }

        cache.insert(index_dir.to_path_buf(), Arc::clone(&entry));
        Ok(entry)
    }

    /// Invalidate cache entry for a path (e.g., after re-indexing)
    fn invalidate_cache(&self, index_dir: &std::path::Path) {
        let mut cache = self.cache.lock();
        if cache.remove(index_dir).is_some() {
            tracing::info!(index_dir = %index_dir.display(), "Invalidated searcher cache entry");
        }
    }

    /// Run the MCP server on stdin/stdout
    pub fn run(&self) -> Result<()> {
        let stdin = std::io::stdin();
        let mut stdout = std::io::stdout();

        for line in stdin.lock().lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }

            let request: JsonRpcRequest = match serde_json::from_str(&line) {
                Ok(req) => req,
                Err(e) => {
                    let resp = JsonRpcResponse::error(None, -32700, format!("Parse error: {e}"));
                    writeln!(stdout, "{}", serde_json::to_string(&resp)?)?;
                    stdout.flush()?;
                    continue;
                }
            };

            let response = self.handle_request(&request);
            writeln!(stdout, "{}", serde_json::to_string(&response)?)?;
            stdout.flush()?;
        }

        Ok(())
    }

    fn handle_request(&self, req: &JsonRpcRequest) -> JsonRpcResponse {
        match req.method.as_str() {
            "initialize" => self.handle_initialize(req.id.clone()),
            "notifications/initialized" => {
                // Notification, no response needed, but we return one anyway
                JsonRpcResponse::success(req.id.clone(), serde_json::json!({}))
            }
            "tools/list" => self.handle_tools_list(req.id.clone()),
            "tools/call" => self.handle_tool_call(req.id.clone(), &req.params),
            _ => JsonRpcResponse::error(
                req.id.clone(),
                -32601,
                format!("Method not found: {}", req.method),
            ),
        }
    }

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
            protocol_version: "2024-11-05".into(),
            capabilities: ServerCapabilities {
                tools: ToolsCapability {
                    list_changed: false,
                },
            },
            server_info: ServerInfo {
                name: "semantex".into(),
                version: env!("CARGO_PKG_VERSION").into(),
            },
        };
        JsonRpcResponse::success(
            id,
            serde_json::to_value(result).expect("InitializeResult serialization"),
        )
    }

    /// Spawn a background thread to build the index for `path`, guarded against double-spawn.
    fn spawn_background_index(&self, path: &std::path::Path) {
        let canonical = match path.canonicalize() {
            Ok(p) => p,
            Err(_) => path.to_path_buf(),
        };

        {
            let mut states = self.index_states.lock();
            if matches!(states.get(&canonical), Some(IndexingStatus::Building)) {
                return; // already building — don't double-spawn
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

    #[allow(clippy::unused_self)]
    fn handle_tools_list(&self, id: Option<serde_json::Value>) -> JsonRpcResponse {
        let tools = vec![
            Tool {
                name: "semantex_search".into(),
                description: concat!(
                    "Use semantex_search INSTEAD of grep/ripgrep for all code search. ",
                    "Finds code by meaning (\"authentication flow\") or exact match (grep_mode=true). ",
                    "Returns ranked file chunks with paths, lines, scores. 25+ languages supported. ",
                    "Auto-indexes on first use — returns keyword results while index builds. ",
                    "Tips: use 2-6 word natural language queries."
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
            },
            Tool {
                name: "semantex_index".into(),
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
            },
            Tool {
                name: "semantex_status".into(),
                description: "Check semantex index status: whether it exists, file count, chunk count, freshness. Use to verify indexing is complete.".into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Project path to check" }
                    }
                }),
            },
            Tool {
                name: "semantex_health".into(),
                description: "Health check for the semantex system, including model availability and configuration.".into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {}
                }),
            },
        ];

        JsonRpcResponse::success(
            id,
            serde_json::json!({ "tools": serde_json::to_value(&tools).expect("tools serialization") }),
        )
    }

    fn handle_tool_call(
        &self,
        id: Option<serde_json::Value>,
        params: &serde_json::Value,
    ) -> JsonRpcResponse {
        let tool_name = params.get("name").and_then(|v| v.as_str()).unwrap_or("");
        let arguments = params
            .get("arguments")
            .cloned()
            .unwrap_or(serde_json::json!({}));

        let result = match tool_name {
            "semantex_search" => self.tool_search(&arguments),
            "semantex_index" => self.tool_index(&arguments),
            "semantex_status" => self.tool_status(&arguments),
            "semantex_health" => self.tool_health(&arguments),
            _ => Err(anyhow::anyhow!("Unknown tool: {tool_name}")),
        };

        match result {
            Ok(text) => JsonRpcResponse::success(
                id,
                serde_json::to_value(ToolCallResult {
                    content: vec![ToolContent {
                        content_type: "text".into(),
                        text,
                    }],
                    is_error: None,
                })
                .expect("ToolCallResult serialization"),
            ),
            Err(e) => JsonRpcResponse::success(
                id,
                serde_json::to_value(ToolCallResult {
                    content: vec![ToolContent {
                        content_type: "text".into(),
                        text: format!("Error: {e}"),
                    }],
                    is_error: Some(true),
                })
                .expect("ToolCallResult serialization"),
            ),
        }
    }

    #[tracing::instrument(skip(self, args), fields(tool = "search"))]
    fn tool_search(&self, args: &serde_json::Value) -> Result<String> {
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

        tracing::info!(
            query,
            path = %path.display(),
            max_results,
            rerank,
            grep_mode,
            "MCP search request"
        );

        // Check index state — fall back to ripgrep if index not ready
        let idx_state = state::detect(&path);
        match idx_state {
            IndexState::Ready => self.do_semantex_search(query, &path, max_results, rerank, grep_mode),
            IndexState::NotIndexed | IndexState::Stale => {
                self.spawn_background_index(&path);
                Self::do_ripgrep_fallback(query, &path, max_results)
            }
            IndexState::Building => Self::do_ripgrep_fallback(query, &path, max_results),
        }
    }

    /// Perform a full semantex search using the index.
    fn do_semantex_search(
        &self,
        query: &str,
        path: &std::path::Path,
        max_results: usize,
        rerank: bool,
        grep_mode: bool,
    ) -> Result<String> {
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
                serde_json::json!({
                    "file": r.chunk.file_path.display().to_string(),
                    "start_line": r.chunk.start_line,
                    "end_line": r.chunk.end_line,
                    "score": r.score,
                    "content": r.chunk.content,
                })
            })
            .collect();

        let json_text = serde_json::to_string_pretty(&json_results)?;
        let response_bytes = json_text.len();
        let metrics = search_output.metrics;
        let footer = format_metrics_footer(&metrics, response_bytes);
        Ok(format!("{json_text}\n\n{footer}"))
    }

    /// Fall back to ripgrep for keyword search while the index builds.
    fn do_ripgrep_fallback(
        query: &str,
        path: &std::path::Path,
        max_results: usize,
    ) -> Result<String> {
        if !ripgrep_fallback::is_rg_available() {
            return Ok(
                "Note: Index building. ripgrep not available for keyword fallback.\n\n[]"
                    .to_string(),
            );
        }

        let results = ripgrep_fallback::search(query, path, max_results)?;
        let json = ripgrep_fallback::format_as_json(&results);
        Ok(format!(
            "Note: Index building. Showing keyword (ripgrep) results.\n\n{json}"
        ))
    }

    #[tracing::instrument(skip(self, args), fields(tool = "index"))]
    fn tool_index(&self, args: &serde_json::Value) -> Result<String> {
        let path = args
            .get("path")
            .and_then(|v| v.as_str())
            .map(PathBuf::from)
            .ok_or_else(|| anyhow::anyhow!("Missing 'path' parameter"))?;

        tracing::info!(path = %path.display(), "MCP index request");

        // Invalidate cached searcher for this project since index is being rebuilt
        let index_dir = SemantexConfig::project_index_dir(&path);
        self.invalidate_cache(&index_dir);

        // Run indexing in the background to avoid blocking the MCP event loop.
        // The caller can use semantex_status to check progress.
        self.spawn_background_index(&path);

        Ok(format!(
            "Indexing started for {}. Use semantex_status to check progress.",
            path.display()
        ))
    }

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

    #[tracing::instrument(skip(self, _args), fields(tool = "health"))]
    fn tool_health(&self, _args: &serde_json::Value) -> Result<String> {
        use semantex_core::embedding::model_manager;

        let mut status = Vec::new();
        status.push(format!("Semantex Health Check v{}", env!("CARGO_PKG_VERSION")));
        status.push("=".repeat(50));

        // Check models directory
        let models_dir = self.config.models_dir();
        status.push(format!("\nModels Directory: {}", models_dir.display()));

        // Check ColBERT model
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

        // Reranker status
        if self.config.rerank {
            status.push("\nReranker: ENABLED (fastembed JINA Reranker v1 Turbo)".to_string());
        } else {
            status.push("\nReranker: DISABLED".to_string());
        }

        // Configuration
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

        // Cache status
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
}

/// Format a compact metrics footer for MCP responses.
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
