pub mod handler;
pub mod listener;
pub mod protocol;
#[cfg(test)]
mod tests;

use crate::config::SemantexConfig;
use crate::search::hybrid::HybridSearcher;
use anyhow::{Context, Result};
use listener::Listener;
use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

/// Default idle timeout before the daemon auto-exits (30 minutes)
const DEFAULT_IDLE_TIMEOUT_SECS: u64 = 1800;

/// The semantex search daemon
pub struct SemantexServer {
    index_dir: PathBuf,
    config: SemantexConfig,
    idle_timeout: Duration,
    shutdown: Arc<AtomicBool>,
}

impl SemantexServer {
    /// Create a new server for the given project path
    pub fn new(project_path: &Path, config: &SemantexConfig) -> Self {
        let index_dir = SemantexConfig::project_index_dir(project_path);
        Self {
            index_dir,
            config: config.clone(),
            idle_timeout: Duration::from_secs(DEFAULT_IDLE_TIMEOUT_SECS),
            shutdown: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Set the idle timeout duration
    pub fn with_idle_timeout(mut self, timeout: Duration) -> Self {
        self.idle_timeout = timeout;
        self
    }

    /// Get the port file path for this server
    pub fn port_file_path(&self) -> PathBuf {
        self.index_dir.join("semantex.port")
    }

    /// Get the PID file path
    pub fn pid_path(&self) -> PathBuf {
        self.index_dir.join("semantex.pid")
    }

    /// Start the server (blocks until shutdown)
    pub fn run(&self) -> Result<()> {
        // Verify index exists
        if !self.index_dir.join("chunks.db").exists() {
            anyhow::bail!(
                "No index found at {}. Run 'semantex index' first.",
                self.index_dir.display()
            );
        }

        // Write PID file
        let pid = std::process::id();
        std::fs::write(self.pid_path(), pid.to_string())
            .with_context(|| format!("Failed to write PID file: {}", self.pid_path().display()))?;

        // Install signal handler
        install_signal_handler(self.shutdown.clone())?;

        // Open searcher (loads all models into memory)
        tracing::info!("Loading search index...");
        let searcher = HybridSearcher::open(&self.index_dir, &self.config)
            .context("Failed to open search index")?;
        tracing::info!("Search index loaded");

        // Start listening
        let listener = Listener::bind(
            &self.port_file_path(),
            searcher,
            self.idle_timeout,
            self.shutdown.clone(),
        )?;

        let result = listener.run();

        // Cleanup PID file
        let _ = std::fs::remove_file(self.pid_path());

        result
    }

    /// Get the shutdown flag for external control
    pub fn shutdown_flag(&self) -> Arc<AtomicBool> {
        self.shutdown.clone()
    }
}

/// Install Ctrl-C / SIGTERM handler that sets the shutdown flag (cross-platform)
#[allow(clippy::needless_pass_by_value)] // Arc cloned inside for ctrlc handler
fn install_signal_handler(shutdown: Arc<AtomicBool>) -> Result<()> {
    let flag = Arc::clone(&shutdown);
    ctrlc::set_handler(move || {
        flag.store(true, std::sync::atomic::Ordering::Relaxed);
    })
    .context("Failed to install Ctrl-C handler")?;
    Ok(())
}

/// Read the daemon port from the port file for a given project.
pub fn read_daemon_port(project_path: &Path) -> Result<u16> {
    let index_dir = SemantexConfig::project_index_dir(project_path);
    let port_file = index_dir.join("semantex.port");
    let content = std::fs::read_to_string(&port_file)
        .with_context(|| format!("Failed to read port file: {}", port_file.display()))?;
    content.trim().parse::<u16>().with_context(|| {
        format!(
            "Invalid port in {}: {}",
            port_file.display(),
            content.trim()
        )
    })
}

/// Check if a daemon is running and healthy for the given project
pub fn daemon_healthy(project_path: &Path) -> bool {
    let Ok(port) = read_daemon_port(project_path) else {
        return false;
    };

    if let Ok(response) = send_request_to_port(port, r#"{"type":"health"}"#) {
        response.contains("\"status\":\"ok\"")
    } else {
        // Stale port file, clean up
        let index_dir = SemantexConfig::project_index_dir(project_path);
        let _ = std::fs::remove_file(index_dir.join("semantex.port"));
        let _ = std::fs::remove_file(index_dir.join("semantex.pid"));
        false
    }
}

/// Send a raw JSON request to the daemon at the given port and return the response
fn send_request_to_port(port: u16, request_json: &str) -> Result<String> {
    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], port));
    let mut stream = TcpStream::connect_timeout(&addr, Duration::from_secs(5))
        .with_context(|| format!("Failed to connect to daemon at 127.0.0.1:{port}"))?;

    stream.set_read_timeout(Some(Duration::from_secs(30)))?;
    stream.set_write_timeout(Some(Duration::from_secs(5)))?;

    // Send request
    stream.write_all(request_json.as_bytes())?;
    stream.write_all(b"\n")?;
    stream.flush()?;

    // Read response
    let mut reader = BufReader::new(&stream);
    let mut response = String::new();
    reader.read_line(&mut response)?;

    Ok(response)
}

/// Send a search request to a running daemon
pub fn daemon_search(
    project_path: &Path,
    request: &protocol::SearchRequest,
) -> Result<protocol::SearchResponse> {
    let port = read_daemon_port(project_path)?;

    let request_json = serde_json::json!({
        "type": "search",
        "query": request.query,
        "max_results": request.max_results,
        "use_dense": request.use_dense,
        "use_sparse": request.use_sparse,
        "use_rerank": request.use_rerank,
        "include_types": request.include_types,
        "exclude_types": request.exclude_types,
        "code_only": request.code_only,
        "include_content": request.include_content,
        "snippet": request.snippet,
    });

    let response_str = send_request_to_port(port, &request_json.to_string())?;
    let response: serde_json::Value =
        serde_json::from_str(&response_str).context("Failed to parse daemon response")?;

    if response.get("type").and_then(|t| t.as_str()) == Some("error") {
        let msg = response
            .get("message")
            .and_then(|m| m.as_str())
            .unwrap_or("Unknown error");
        anyhow::bail!("Daemon error: {msg}");
    }

    // Parse the search response
    let results: Vec<protocol::SearchResultItem> = response
        .get("results")
        .and_then(|r| serde_json::from_value(r.clone()).ok())
        .unwrap_or_default();

    Ok(protocol::SearchResponse {
        results,
        duration_ms: response
            .get("duration_ms")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0),
        dense_count: response
            .get("dense_count")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0) as usize,
        sparse_count: response
            .get("sparse_count")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0) as usize,
        fused_count: response
            .get("fused_count")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0) as usize,
        metrics: response
            .get("metrics")
            .and_then(|m| serde_json::from_value(m.clone()).ok()),
        confidence: response
            .get("confidence")
            .and_then(|v| v.as_str())
            .map(std::string::ToString::to_string),
    })
}

/// Send a binary (bincode) search request to the daemon at the given port.
/// Returns the SearchResponse or an error. Much faster than JSON.
pub fn daemon_search_binary(
    port: u16,
    request: protocol::SearchRequest,
) -> Result<protocol::SearchResponse> {
    use std::io::Read;

    let bin_req = protocol::BinaryRequest::Search(request);

    let frame = protocol::encode_binary_request(&bin_req);

    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], port));
    let mut stream = TcpStream::connect_timeout(&addr, Duration::from_secs(5))
        .with_context(|| format!("Failed to connect to daemon at 127.0.0.1:{port}"))?;

    stream.set_read_timeout(Some(Duration::from_secs(30)))?;
    stream.set_write_timeout(Some(Duration::from_secs(5)))?;

    // Send binary frame
    stream.write_all(&frame)?;
    stream.flush()?;

    // Read binary response: [magic:1][len:4 LE][payload]
    let mut magic = [0u8; 1];
    stream.read_exact(&mut magic)?;
    if magic[0] != protocol::BINARY_MAGIC {
        anyhow::bail!("Expected binary response, got 0x{:02x}", magic[0]);
    }

    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf)?;
    let len = u32::from_le_bytes(len_buf) as usize;

    let mut payload = vec![0u8; len];
    stream.read_exact(&mut payload)?;

    let bin_resp = protocol::decode_binary_response(&payload)
        .map_err(|e| anyhow::anyhow!("Failed to decode binary response: {e}"))?;

    match bin_resp {
        protocol::BinaryResponse::Search(sr) => Ok(sr),
        protocol::BinaryResponse::Error(e) => {
            anyhow::bail!("Daemon error: {}", e.message)
        }
        other => anyhow::bail!("Unexpected response type: {other:?}"),
    }
}

/// Send a binary graph walk request to the daemon at the given port.
pub fn daemon_graph_walk_binary(
    port: u16,
    request: protocol::GraphWalkRequest,
) -> Result<protocol::GraphWalkResponse> {
    use std::io::Read;

    let bin_req = protocol::BinaryRequest::GraphWalk(request);
    let frame = protocol::encode_binary_request(&bin_req);

    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], port));
    let mut stream = TcpStream::connect_timeout(&addr, Duration::from_secs(5))
        .with_context(|| format!("Failed to connect to daemon at 127.0.0.1:{port}"))?;

    stream.set_read_timeout(Some(Duration::from_secs(30)))?;
    stream.set_write_timeout(Some(Duration::from_secs(5)))?;

    stream.write_all(&frame)?;
    stream.flush()?;

    let mut magic = [0u8; 1];
    stream.read_exact(&mut magic)?;
    if magic[0] != protocol::BINARY_MAGIC {
        anyhow::bail!("Expected binary response, got 0x{:02x}", magic[0]);
    }

    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf)?;
    let len = u32::from_le_bytes(len_buf) as usize;

    let mut payload = vec![0u8; len];
    stream.read_exact(&mut payload)?;

    let bin_resp = protocol::decode_binary_response(&payload)
        .map_err(|e| anyhow::anyhow!("Failed to decode binary response: {e}"))?;

    match bin_resp {
        protocol::BinaryResponse::GraphWalk(gr) => Ok(gr),
        protocol::BinaryResponse::Error(e) => {
            anyhow::bail!("Daemon error: {}", e.message)
        }
        other => anyhow::bail!("Unexpected response type: {other:?}"),
    }
}

/// Send a binary multi-search request to the daemon at the given port.
/// Returns all SearchResponses (one per query) or an error.
pub fn daemon_multi_search_binary(
    port: u16,
    request: protocol::MultiSearchRequest,
) -> Result<protocol::MultiSearchResponse> {
    use std::io::Read;

    let bin_req = protocol::BinaryRequest::MultiSearch(request);
    let frame = protocol::encode_binary_request(&bin_req);

    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], port));
    let mut stream = TcpStream::connect_timeout(&addr, Duration::from_secs(5))
        .with_context(|| format!("Failed to connect to daemon at 127.0.0.1:{port}"))?;

    stream.set_read_timeout(Some(Duration::from_secs(30)))?;
    stream.set_write_timeout(Some(Duration::from_secs(5)))?;

    stream.write_all(&frame)?;
    stream.flush()?;

    let mut magic = [0u8; 1];
    stream.read_exact(&mut magic)?;
    if magic[0] != protocol::BINARY_MAGIC {
        anyhow::bail!("Expected binary response, got 0x{:02x}", magic[0]);
    }

    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf)?;
    let len = u32::from_le_bytes(len_buf) as usize;

    let mut payload = vec![0u8; len];
    stream.read_exact(&mut payload)?;

    let bin_resp = protocol::decode_binary_response(&payload)
        .map_err(|e| anyhow::anyhow!("Failed to decode binary response: {e}"))?;

    match bin_resp {
        protocol::BinaryResponse::MultiSearch(mr) => Ok(mr),
        protocol::BinaryResponse::Error(e) => {
            anyhow::bail!("Daemon error: {}", e.message)
        }
        other => anyhow::bail!("Unexpected response type: {other:?}"),
    }
}

/// Send a binary deep search request to the daemon at the given port.
pub fn daemon_deep_search_binary(
    port: u16,
    request: protocol::DeepSearchRequest,
) -> Result<protocol::DeepSearchResponse> {
    use std::io::Read;

    let bin_req = protocol::BinaryRequest::DeepSearch(request);
    let frame = protocol::encode_binary_request(&bin_req);

    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], port));
    let mut stream = TcpStream::connect_timeout(&addr, Duration::from_secs(5))
        .with_context(|| format!("Failed to connect to daemon at 127.0.0.1:{port}"))?;

    stream.set_read_timeout(Some(Duration::from_secs(60)))?; // deep takes longer
    stream.set_write_timeout(Some(Duration::from_secs(5)))?;

    stream.write_all(&frame)?;
    stream.flush()?;

    let mut magic = [0u8; 1];
    stream.read_exact(&mut magic)?;
    if magic[0] != protocol::BINARY_MAGIC {
        anyhow::bail!("Expected binary response, got 0x{:02x}", magic[0]);
    }

    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf)?;
    let len = u32::from_le_bytes(len_buf) as usize;

    let mut payload = vec![0u8; len];
    stream.read_exact(&mut payload)?;

    let bin_resp = protocol::decode_binary_response(&payload)
        .map_err(|e| anyhow::anyhow!("Failed to decode binary response: {e}"))?;

    match bin_resp {
        protocol::BinaryResponse::DeepSearch(dr) => Ok(dr),
        protocol::BinaryResponse::Error(e) => {
            anyhow::bail!("Daemon error: {}", e.message)
        }
        other => anyhow::bail!("Unexpected response type: {other:?}"),
    }
}

/// Perform a graph walk directly against the index without a running daemon.
/// Used as a fallback when the daemon is unavailable.
pub fn graph_walk_direct(
    symbol: &str,
    index_dir: &std::path::Path,
    config: &SemantexConfig,
) -> Result<protocol::GraphWalkResponse> {
    let searcher = HybridSearcher::open_sparse_only(index_dir, config)
        .context("Failed to open search index for graph walk")?;
    searcher.with_store(|store| handler::graph_walk_from_store(store, symbol))
}

/// Send a binary health check to the daemon at the given port.
pub fn daemon_healthy_binary(port: u16) -> bool {
    use std::io::Read;

    let bin_req = protocol::BinaryRequest::Health;
    let frame = protocol::encode_binary_request(&bin_req);

    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], port));
    let Ok(mut stream) = TcpStream::connect_timeout(&addr, Duration::from_secs(2)) else {
        return false;
    };

    if stream
        .set_read_timeout(Some(Duration::from_secs(2)))
        .is_err()
    {
        return false;
    }

    if stream.write_all(&frame).is_err() || stream.flush().is_err() {
        return false;
    }

    let mut magic = [0u8; 1];
    if stream.read_exact(&mut magic).is_err() || magic[0] != protocol::BINARY_MAGIC {
        return false;
    }

    let mut len_buf = [0u8; 4];
    if stream.read_exact(&mut len_buf).is_err() {
        return false;
    }
    let len = u32::from_le_bytes(len_buf) as usize;

    let mut payload = vec![0u8; len];
    if stream.read_exact(&mut payload).is_err() {
        return false;
    }

    matches!(
        protocol::decode_binary_response(&payload),
        Ok(protocol::BinaryResponse::Health(
            protocol::HealthResponse { .. }
        ))
    )
}

/// Check if a daemon has been spawned and is loading models (PID file exists,
/// process alive, but port file not yet written).
///
/// Used to prevent N concurrent processes from each spawning a daemon when
/// the first one is still initialising — without this guard, a cluster of
/// subagents waking up simultaneously would race to each call `semantex serve`,
/// causing N redundant heavy daemon processes.
pub fn daemon_starting(project_path: &Path) -> bool {
    let index_dir = SemantexConfig::project_index_dir(project_path);
    let port_path = index_dir.join("semantex.port");
    let pid_path = index_dir.join("semantex.pid");

    // If the port file exists the daemon is already fully ready — not "starting".
    if port_path.exists() {
        return false;
    }

    let Ok(pid_str) = std::fs::read_to_string(&pid_path) else {
        return false;
    };
    let Ok(pid) = pid_str.trim().parse::<u32>() else {
        // Malformed PID file from a previous crash — remove it.
        let _ = std::fs::remove_file(&pid_path);
        return false;
    };

    if is_process_alive(pid) {
        true
    } else {
        // Stale PID file left by a crashed/OOM-killed daemon.
        let _ = std::fs::remove_file(&pid_path);
        false
    }
}

/// Check if a process is alive using POSIX signal 0 (existence check, no signal sent).
#[cfg(unix)]
fn is_process_alive(pid: u32) -> bool {
    // SAFETY: kill(pid, 0) is a read-only probe — it sends no signal and only
    // checks whether the process exists and we have permission to signal it.
    unsafe { libc::kill(pid as libc::pid_t, 0) == 0 }
}

/// On non-Unix platforms return `true` as a conservative fallback to avoid
/// spurious double-spawns (if the daemon didn't start, search falls back to sparse).
#[cfg(not(unix))]
fn is_process_alive(_pid: u32) -> bool {
    true
}

/// Stop a running daemon for the given project
pub fn stop_daemon(project_path: &Path) -> Result<bool> {
    let index_dir = SemantexConfig::project_index_dir(project_path);
    let port_file = index_dir.join("semantex.port");
    let pid_path = index_dir.join("semantex.pid");

    let Ok(port) = read_daemon_port(project_path) else {
        return Ok(false);
    };

    // Try graceful shutdown
    if send_request_to_port(port, r#"{"type":"shutdown"}"#).is_ok() {
        // Wait briefly for cleanup
        std::thread::sleep(Duration::from_millis(200));
    }

    // Clean up stale files
    let _ = std::fs::remove_file(&port_file);
    let _ = std::fs::remove_file(&pid_path);

    Ok(true)
}
