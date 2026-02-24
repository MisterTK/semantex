pub mod handler;
pub mod listener;
pub mod protocol;
#[cfg(test)]
mod tests;

use crate::config::SageConfig;
use crate::search::hybrid::HybridSearcher;
use anyhow::{Context, Result};
use listener::Listener;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

/// Default idle timeout before the daemon auto-exits (30 minutes)
const DEFAULT_IDLE_TIMEOUT_SECS: u64 = 1800;

/// The sage search daemon
pub struct SageServer {
    index_dir: PathBuf,
    config: SageConfig,
    idle_timeout: Duration,
    shutdown: Arc<AtomicBool>,
}

impl SageServer {
    /// Create a new server for the given project path
    pub fn new(project_path: &Path, config: &SageConfig) -> Self {
        let index_dir = SageConfig::project_index_dir(project_path);
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

    /// Get the socket path for this server
    pub fn socket_path(&self) -> PathBuf {
        self.index_dir.join("sage.sock")
    }

    /// Get the PID file path
    pub fn pid_path(&self) -> PathBuf {
        self.index_dir.join("sage.pid")
    }

    /// Start the server (blocks until shutdown)
    pub fn run(&self) -> Result<()> {
        // Verify index exists
        if !self.index_dir.join("chunks.db").exists() {
            anyhow::bail!(
                "No index found at {}. Run 'sage index' first.",
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
            &self.socket_path(),
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

/// Install SIGINT/SIGTERM handler that sets the shutdown flag
#[allow(clippy::needless_pass_by_value)] // Arc cloned inside for signal_hook::flag::register
fn install_signal_handler(shutdown: Arc<AtomicBool>) -> Result<()> {
    signal_hook::flag::register(signal_hook::consts::SIGINT, Arc::clone(&shutdown))?;
    signal_hook::flag::register(signal_hook::consts::SIGTERM, Arc::clone(&shutdown))?;
    Ok(())
}

/// Check if a daemon is running and healthy for the given project
pub fn daemon_healthy(project_path: &Path) -> bool {
    let index_dir = SageConfig::project_index_dir(project_path);
    let socket_path = index_dir.join("sage.sock");

    if !socket_path.exists() {
        return false;
    }

    // Try a health check
    if let Ok(response) = send_request(&socket_path, r#"{"type":"health"}"#) {
        response.contains("\"status\":\"ok\"")
    } else {
        // Stale socket, clean up
        let _ = std::fs::remove_file(&socket_path);
        let _ = std::fs::remove_file(index_dir.join("sage.pid"));
        false
    }
}

/// Send a raw JSON request to the daemon and return the response
pub fn send_request(socket_path: &Path, request_json: &str) -> Result<String> {
    let mut stream = UnixStream::connect(socket_path)
        .with_context(|| format!("Failed to connect to daemon at {}", socket_path.display()))?;

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
    let index_dir = SageConfig::project_index_dir(project_path);
    let socket_path = index_dir.join("sage.sock");

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

    let response_str = send_request(&socket_path, &request_json.to_string())?;
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
    })
}

/// Send a binary (bincode) search request to the daemon socket.
/// Returns the SearchResponse or an error. Much faster than JSON.
pub fn daemon_search_binary(
    socket_path: &Path,
    request: protocol::SearchRequest,
) -> Result<protocol::SearchResponse> {
    use std::io::Read;

    let bin_req = protocol::BinaryRequest::Search(request);

    let frame = protocol::encode_binary_request(&bin_req);

    let mut stream = UnixStream::connect(socket_path)
        .with_context(|| format!("Failed to connect to daemon at {}", socket_path.display()))?;

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

/// Send a binary health check to the daemon socket.
pub fn daemon_healthy_binary(socket_path: &Path) -> bool {
    use std::io::Read;

    let bin_req = protocol::BinaryRequest::Health;
    let frame = protocol::encode_binary_request(&bin_req);

    let Ok(mut stream) = UnixStream::connect(socket_path) else {
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

/// Stop a running daemon for the given project
pub fn stop_daemon(project_path: &Path) -> Result<bool> {
    let index_dir = SageConfig::project_index_dir(project_path);
    let socket_path = index_dir.join("sage.sock");
    let pid_path = index_dir.join("sage.pid");

    if !socket_path.exists() {
        return Ok(false);
    }

    // Try graceful shutdown
    if send_request(&socket_path, r#"{"type":"shutdown"}"#).is_ok() {
        // Wait briefly for cleanup
        std::thread::sleep(Duration::from_millis(200));
    }

    // Clean up stale files
    let _ = std::fs::remove_file(&socket_path);
    let _ = std::fs::remove_file(&pid_path);

    Ok(true)
}
