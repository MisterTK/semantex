use super::handler::Handler;
use super::protocol::{
    self, BINARY_MAGIC, BinaryFrameError, BinaryResponse, ErrorResponse, Request, Response,
};
use crate::config::SemantexConfig;
use crate::embedding::{colbert::ColbertEmbedder, model_manager};
use crate::index::storage::{self, PrefetchOutcome};
use crate::search::hybrid::HybridSearcher;
use anyhow::{Context, Result};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use crate::memory::current_rss_bytes;

/// Name of the warm-state sentinel file (E8d). Lives next to `semantex.pid` in
/// `.semantex/`. Its presence + a live daemon PID means the MCP layer can
/// bypass the full `state::detect` pass on subsequent calls.
pub const WARM_STATE_SENTINEL: &str = "warm_state.lock";

/// TCP localhost listener for the semantex daemon
#[allow(clippy::struct_field_names)]
pub struct Listener {
    port_file: PathBuf,
    sentinel_file: PathBuf,
    listener: TcpListener,
    searcher: HybridSearcher,
    search_count: AtomicU64,
    start_time: Instant,
    idle_timeout: Duration,
    shutdown: Arc<AtomicBool>,
    project_root: PathBuf,
}

impl Listener {
    /// Create a new TCP listener bound to an OS-assigned ephemeral port.
    /// Writes the assigned port number to `port_file`.
    pub fn bind(
        port_file: &std::path::Path,
        searcher: HybridSearcher,
        project_root: PathBuf,
        idle_timeout: Duration,
        shutdown: Arc<AtomicBool>,
    ) -> Result<Self> {
        // Bind to localhost with an OS-assigned ephemeral port
        let listener =
            TcpListener::bind("127.0.0.1:0").context("Failed to bind TCP listener on 127.0.0.1")?;

        let port = listener
            .local_addr()
            .context("Failed to get local address")?
            .port();

        // Write port to file for client discovery
        std::fs::write(port_file, format!("{port}\n"))
            .with_context(|| format!("Failed to write port file: {}", port_file.display()))?;

        // Use non-blocking mode with sleep for periodic shutdown/idle checks
        listener
            .set_nonblocking(true)
            .context("Failed to set non-blocking mode")?;

        // E8(d): sentinel file lives next to the port file.
        let sentinel_file = port_file.parent().map_or_else(
            || PathBuf::from(WARM_STATE_SENTINEL),
            |p| p.join(WARM_STATE_SENTINEL),
        );

        Ok(Self {
            port_file: port_file.to_path_buf(),
            sentinel_file,
            listener,
            searcher,
            search_count: AtomicU64::new(0),
            start_time: Instant::now(),
            idle_timeout,
            shutdown,
            project_root,
        })
    }

    /// Get the port this listener is bound to.
    pub fn port(&self) -> u16 {
        self.listener.local_addr().map_or(0, |a| a.port())
    }

    /// Path to the warm-state sentinel file written when the daemon completes
    /// its initial readiness check (E8d).
    pub fn sentinel_path(&self) -> &Path {
        &self.sentinel_file
    }

    /// Create the warm-state sentinel containing the daemon PID, after the
    /// initial state check has completed.
    ///
    /// The sentinel's presence + a live PID = the MCP layer can skip the
    /// full `state::detect` pass. Failures here are logged and ignored —
    /// missing sentinel just means MCP falls back to full state checks
    /// (correct behaviour, just slower).
    ///
    /// # Atomicity (Finding 12)
    ///
    /// `std::fs::write` opens with O_CREAT|O_TRUNC then `write_all`s. During
    /// the gap between truncate and write a concurrent reader observes an
    /// empty file. In an earlier version `warm_state_ready` reacted to a
    /// parse-failure by deleting the sentinel; the daemon's `write_all` then
    /// succeeded against the unlinked inode and left no sentinel for future
    /// readers, permanently disabling E8d for the daemon's lifetime.
    ///
    /// We now write to a sibling temp file and `rename` it into place.
    /// `rename` is atomic on POSIX — readers either see the previous sentinel
    /// or the new one, never an empty file. On Windows `rename` over an
    /// existing target can fail; we fall back to remove + rename. The
    /// fallback isn't atomic but is still much narrower than O_TRUNC.
    fn write_sentinel(&self) {
        let pid = std::process::id();
        write_sentinel_atomic(&self.sentinel_file, pid);
    }

    /// Run the accept loop until shutdown, idle timeout, or RSS limit exceeded.
    ///
    /// The RSS limit defaults to 2 048 MB and can be overridden with the
    /// `SEMANTEX_MAX_RSS_MB` environment variable (set to `0` to disable).
    /// When the limit is hit the daemon exits cleanly; the next `semantex`
    /// invocation will start a fresh daemon with a clean memory footprint.
    pub fn run(&self) -> Result<()> {
        let rss_limit_mb: u64 = std::env::var("SEMANTEX_MAX_RSS_MB")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1024);

        tracing::info!(
            port = self.port(),
            idle_timeout_s = self.idle_timeout.as_secs(),
            rss_limit_mb,
            "Daemon listening on 127.0.0.1:{}",
            self.port()
        );

        // E8(d): the daemon is now bound and ready to accept connections.
        // Drop the warm-state sentinel so subsequent MCP calls can skip the
        // full state check.
        self.write_sentinel();

        let mut last_activity = Instant::now();
        let mut loop_count: u32 = 0;

        loop {
            // Check shutdown flag
            if self.shutdown.load(Ordering::Relaxed) {
                tracing::info!("Shutdown signal received");
                break;
            }

            // Check idle timeout
            if last_activity.elapsed() > self.idle_timeout {
                tracing::info!(
                    idle_s = last_activity.elapsed().as_secs(),
                    "Idle timeout reached, shutting down"
                );
                break;
            }

            // RSS guard — checked every 500 iterations (~5 s at 10 ms/iteration)
            loop_count += 1;
            if rss_limit_mb > 0
                && loop_count.is_multiple_of(500)
                && let Some(rss_bytes) = current_rss_bytes()
            {
                let rss_mb = rss_bytes / (1024 * 1024);
                if rss_mb > rss_limit_mb {
                    tracing::warn!(
                        rss_mb,
                        rss_limit_mb,
                        "RSS limit exceeded ({} MB > {} MB) — daemon exiting to free memory. \
                         Next search will restart it.",
                        rss_mb,
                        rss_limit_mb
                    );
                    break;
                }
            }

            // Accept a connection (non-blocking + 500ms sleep on WouldBlock)
            match self.listener.accept() {
                Ok((stream, _addr)) => {
                    // Switch to blocking mode for connection handling
                    if let Err(e) = stream.set_nonblocking(false) {
                        tracing::warn!("Failed to set blocking mode: {}", e);
                        continue;
                    }
                    last_activity = Instant::now();
                    loop_count = 0; // reset so we don't exit right after heavy search
                    if let Err(e) = self.handle_connection(stream) {
                        tracing::warn!("Connection error: {}", e);
                    }
                }
                Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    // No pending connection — sleep briefly and retry
                    std::thread::sleep(Duration::from_millis(10));
                }
                Err(e) => {
                    tracing::error!("Accept error: {}", e);
                    std::thread::sleep(Duration::from_millis(100));
                }
            }
        }

        self.cleanup();
        Ok(())
    }

    #[allow(clippy::needless_pass_by_value)] // TcpStream ownership needed for set_read/write_timeout
    fn handle_connection(&self, stream: TcpStream) -> Result<()> {
        stream.set_read_timeout(Some(Duration::from_secs(5)))?;
        stream.set_write_timeout(Some(Duration::from_secs(5)))?;

        // Peek at the first byte to determine protocol
        let mut peek = [0u8; 1];
        let mut reader = BufReader::new(&stream);
        reader.read_exact(&mut peek)?;

        if peek[0] == BINARY_MAGIC {
            self.handle_binary_connection(&mut reader, &stream)
        } else {
            self.handle_json_connection(peek[0], &mut reader, &stream)
        }
    }

    /// Handle a binary (postcard) framed connection.
    ///
    /// Frame layout (v0.4.1 W-Index #2):
    ///   `[BINARY_MAGIC][len:4 LE][BINARY_PROTOCOL_VERSION][postcard]`
    ///
    /// The magic byte has already been consumed by the caller. We read the
    /// 4-byte length, then the length-prefixed body whose first byte is the
    /// protocol version — that version check happens inside
    /// `decode_binary_request`, which surfaces a `BinaryFrameError`. Version
    /// mismatch is reported back to the client as a structured
    /// `Response::Error`; we do NOT crash or silently drop the connection.
    fn handle_binary_connection(
        &self,
        reader: &mut BufReader<&TcpStream>,
        stream: &TcpStream,
    ) -> Result<()> {
        // Read 4-byte LE length
        let mut len_buf = [0u8; 4];
        reader.read_exact(&mut len_buf)?;
        let len = u32::from_le_bytes(len_buf) as usize;

        if len > 16 * 1024 * 1024 {
            // Reject unreasonably large payloads (16 MiB)
            return Ok(());
        }

        // Read body = [VERSION:1][postcard].
        let mut body = vec![0u8; len];
        reader.read_exact(&mut body)?;

        let response = match protocol::decode_binary_request(&body) {
            Ok(bin_req) => {
                let request: Request = bin_req.into();
                let is_shutdown = matches!(request, Request::Shutdown);
                let handler = Handler::new(
                    &self.searcher,
                    &self.search_count,
                    self.project_root.clone(),
                );
                let resp = handler.handle(request, self.start_time);

                if is_shutdown {
                    self.shutdown.store(true, Ordering::Relaxed);
                }

                resp
            }
            Err(BinaryFrameError::UnsupportedVersion { expected, got }) => {
                tracing::warn!(
                    expected,
                    got,
                    "Rejected binary frame with unsupported protocol version"
                );
                Response::Error(ErrorResponse {
                    message: format!(
                        "Unsupported binary protocol version: daemon speaks v{expected}, \
                         client sent v{got}. Upgrade the client or daemon to match.",
                    ),
                })
            }
            Err(BinaryFrameError::Decode(e)) => Response::Error(ErrorResponse {
                message: format!("Invalid binary request: {e}"),
            }),
        };

        // Write binary response — always at the daemon's protocol version.
        let bin_resp: BinaryResponse = response.into();
        let frame = protocol::encode_binary_response(&bin_resp);
        let mut writer = stream;
        writer.write_all(&frame)?;
        writer.flush()?;

        Ok(())
    }

    /// Handle a JSON newline-delimited connection.
    /// The first byte has already been consumed, so we prepend it.
    fn handle_json_connection(
        &self,
        first_byte: u8,
        reader: &mut BufReader<&TcpStream>,
        stream: &TcpStream,
    ) -> Result<()> {
        let mut line = String::new();
        line.push(first_byte as char);

        // Read the rest of the line
        match reader.read_line(&mut line) {
            Ok(0) => return Ok(()), // EOF
            Ok(_) => {}
            Err(e) => {
                tracing::debug!("Read error: {}", e);
                return Ok(());
            }
        }

        let line = line.trim();
        if line.is_empty() {
            return Ok(());
        }

        // Parse request
        let response = match serde_json::from_str::<Request>(line) {
            Ok(request) => {
                let is_shutdown = matches!(request, Request::Shutdown);
                let handler = Handler::new(
                    &self.searcher,
                    &self.search_count,
                    self.project_root.clone(),
                );
                let response = handler.handle(request, self.start_time);

                if is_shutdown {
                    self.shutdown.store(true, Ordering::Relaxed);
                }

                response
            }
            Err(e) => Response::Error(ErrorResponse {
                message: format!("Invalid request: {e}"),
            }),
        };

        // Write response as single JSON line
        let response_json = serde_json::to_string(&response)?;
        let mut writer = stream;
        writer.write_all(response_json.as_bytes())?;
        writer.write_all(b"\n")?;
        writer.flush()?;

        Ok(())
    }

    /// Reload the underlying searcher (for watch integration)
    pub fn reload_searcher(&mut self, searcher: HybridSearcher) {
        self.searcher = searcher;
        tracing::info!("Searcher reloaded");
    }

    fn cleanup(&self) {
        if self.port_file.exists() {
            let _ = std::fs::remove_file(&self.port_file);
        }
        // E8(d): remove warm-state sentinel so the MCP layer falls back to
        // full state checks once the daemon is gone.
        // Finding 13: only remove if the sentinel still belongs to us.
        remove_sentinel_if_ours(&self.sentinel_file);
        tracing::info!(
            searches = self.search_count.load(Ordering::Relaxed),
            uptime_s = self.start_time.elapsed().as_secs(),
            "Daemon stopped"
        );
    }
}

impl Drop for Listener {
    fn drop(&mut self) {
        if self.port_file.exists() {
            let _ = std::fs::remove_file(&self.port_file);
        }
        // Defensive: also remove the sentinel here in case `run()` returned
        // early via `?` and never reached the `cleanup()` call.
        // Finding 13: never delete a sentinel that doesn't belong to us. If
        // this Listener's instance happens to outlive its own daemon and a
        // successor daemon has already written its own sentinel, deleting it
        // here would permanently break E8d for the successor.
        remove_sentinel_if_ours(&self.sentinel_file);
    }
}

/// Atomically (POSIX) write `pid` to `path` via a sibling temp file + rename.
///
/// See [`Listener::write_sentinel`] for the rationale. Errors are logged and
/// otherwise swallowed — a missing sentinel just causes the MCP layer to fall
/// back to full state checks, which is correct but slower. We never want a
/// sentinel write failure to crash the daemon.
fn write_sentinel_atomic(path: &Path, pid: u32) {
    // Build the sibling temp path:  warm_state.lock  →  warm_state.lock.tmp.<pid>
    let Some(parent) = path.parent() else {
        tracing::debug!(
            "warm-state sentinel {} has no parent directory; skipping write",
            path.display()
        );
        return;
    };
    let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    let temp_path = parent.join(format!("{file_name}.tmp.{pid}"));

    if let Err(e) = std::fs::write(&temp_path, pid.to_string()) {
        tracing::debug!(
            "Failed to write warm-state sentinel temp {}: {}",
            temp_path.display(),
            e
        );
        return;
    }

    match std::fs::rename(&temp_path, path) {
        Ok(()) => {
            tracing::debug!(pid, "Wrote warm-state sentinel at {}", path.display());
        }
        Err(e) => {
            // Windows: rename may fail if the target exists. Fall back to
            // remove + rename. On POSIX rename always overwrites, so this
            // path is effectively Windows-only.
            if cfg!(windows) && e.kind() == std::io::ErrorKind::AlreadyExists {
                let _ = std::fs::remove_file(path);
                if let Err(e2) = std::fs::rename(&temp_path, path) {
                    tracing::debug!(
                        "Failed to rename warm-state sentinel (windows fallback) \
                         {} -> {}: {}",
                        temp_path.display(),
                        path.display(),
                        e2
                    );
                    let _ = std::fs::remove_file(&temp_path);
                } else {
                    tracing::debug!(
                        pid,
                        "Wrote warm-state sentinel at {} (windows fallback)",
                        path.display()
                    );
                }
            } else {
                tracing::debug!(
                    "Failed to rename warm-state sentinel {} -> {}: {}",
                    temp_path.display(),
                    path.display(),
                    e
                );
                let _ = std::fs::remove_file(&temp_path);
            }
        }
    }
}

/// Finding 13: remove the sentinel file at `path` only if it currently contains
/// the current process's PID.
///
/// Why this matters: if daemon A is mid-shutdown and daemon B has already
/// started and written its own sentinel (same path, same project), an
/// unconditional `remove_file` from A's `Drop`/`cleanup` would delete B's
/// sentinel and break warm-state detection for the entire lifetime of B.
///
/// Read failures (file gone, permission denied) and parse failures are treated
/// as "not ours" and we silently do nothing.
fn remove_sentinel_if_ours(path: &Path) {
    let Ok(content) = std::fs::read_to_string(path) else {
        return;
    };
    let Ok(pid) = content.trim().parse::<u32>() else {
        // Not a well-formed PID — not ours, leave it alone. The next daemon
        // start will atomically overwrite it via `write_sentinel_atomic`.
        return;
    };
    if pid == std::process::id() {
        let _ = std::fs::remove_file(path);
    }
}

// ════════════════════════════════════════════════════════════════════════
// E8 cold-start helpers — invoked by `SemantexServer::run` before / during
// the heavy `HybridSearcher::open` call.
// ════════════════════════════════════════════════════════════════════════

/// Run the parallel page-cache prefetch for an index directory (E8b).
///
/// Wraps `storage::prefetch_index_files` with daemon-friendly logging so the
/// caller can fire-and-forget at startup. Returns the prefetch outcome for
/// callers that want to record timings (cold-start benchmark).
pub fn prefetch_index(index_dir: &Path) -> PrefetchOutcome {
    let outcome = storage::prefetch_index_files(index_dir);
    tracing::debug!(
        sqlite_ms = outcome.sqlite_ns / 1_000_000,
        sparse_ms = outcome.sparse_ns / 1_000_000,
        plaid_ms = outcome.plaid_ns / 1_000_000,
        sqlite_ok = outcome.sqlite_ok,
        sparse_ok = outcome.sparse_ok,
        plaid_ok = outcome.plaid_ok,
        "E8(b) prefetch_index_files completed",
    );
    outcome
}

/// Spawn the background ColBERT warm thread (E8c).
///
/// The thread is fire-and-forget: it materializes the ONNX session in the
/// global `ColbertEmbedder` singleton and runs a dummy encode, so the first
/// real user query doesn't pay the ~200-300ms session-build cost. Errors
/// are logged and otherwise ignored — cold start still works if warm-up
/// fails (first query just pays the cost itself).
///
/// Returns the spawned thread handle; the caller may detach (drop) it or
/// join it. Production daemon paths detach.
pub fn spawn_colbert_warm_thread(config: &SemantexConfig) -> std::thread::JoinHandle<()> {
    let models_root = config.models_dir();
    std::thread::Builder::new()
        .name("semantex-colbert-warmup".to_string())
        .spawn(move || {
            let t = Instant::now();
            match model_manager::ensure_colbert_model(&models_root) {
                Ok(model_dir) => match ColbertEmbedder::global(&model_dir) {
                    Ok(embedder) => match embedder.warm_up() {
                        Ok(()) => {
                            tracing::info!(
                                elapsed_ms = t.elapsed().as_millis(),
                                "E8(c) ColBERT background warm-up complete",
                            );
                        }
                        Err(e) => {
                            tracing::warn!(
                                error = %e,
                                "E8(c) ColBERT warm-up failed (first query will pay the cost)",
                            );
                        }
                    },
                    Err(e) => {
                        tracing::warn!(
                            error = %e,
                            "E8(c) ColBERT global init failed during warm-up",
                        );
                    }
                },
                Err(e) => {
                    tracing::debug!(
                        error = %e,
                        "E8(c) ColBERT model not available for warm-up",
                    );
                }
            }
        })
        .expect("failed to spawn ColBERT warm-up thread")
}

/// Check whether the warm-state sentinel exists and points to a live PID (E8d).
///
/// Used by the MCP layer (W5) to decide whether to skip a full `state::detect`
/// pass on subsequent calls. Returns `true` iff:
/// 1. `<index_dir>/warm_state.lock` exists,
/// 2. its contents parse as a u32 PID,
/// 3. that PID matches a live process.
///
/// # Race avoidance (Finding 12)
///
/// We previously deleted the sentinel on parse failure, treating it as a
/// "malformed sentinel that needs cleanup". That was wrong: a transient empty
/// file is a legitimate race window (the daemon was mid-`O_TRUNC` between
/// truncate and write_all). The reader's overzealous cleanup raced the
/// daemon's write_all, leaving an unlinked inode and no sentinel for future
/// readers. We now leave malformed sentinels alone — the daemon's next
/// startup will overwrite them atomically via `write_sentinel_atomic`.
///
/// Stale-PID sentinels (the daemon crashed) ARE still cleaned up — those are
/// not a race condition, they're genuine garbage.
pub fn warm_state_ready(index_dir: &Path) -> bool {
    let sentinel = index_dir.join(WARM_STATE_SENTINEL);
    let Ok(pid_str) = std::fs::read_to_string(&sentinel) else {
        return false;
    };
    let Ok(pid) = pid_str.trim().parse::<u32>() else {
        // Malformed or transient empty sentinel — return false without
        // deleting. If this is a write-in-progress race the daemon's
        // atomic rename will resolve it within microseconds; if the file
        // is genuinely garbage, the next daemon start will overwrite it.
        // Either way, deleting here is the wrong call.
        return false;
    };

    if is_process_alive(pid) {
        true
    } else {
        // Stale sentinel from a crashed daemon — clean up. But only if we're
        // confident this isn't OUR PID racing a write — we check it's a
        // dead PID, which is sufficient (no live process owns it).
        let _ = std::fs::remove_file(&sentinel);
        false
    }
}

/// POSIX existence probe — sends signal 0 (no-op) to check whether a process
/// with the given PID is alive and signal-able by the current user.
#[cfg(unix)]
fn is_process_alive(pid: u32) -> bool {
    // SAFETY: kill(pid, 0) is a read-only probe — it sends no signal and only
    // checks whether the process exists and we have permission to signal it.
    unsafe { libc::kill(pid as libc::pid_t, 0) == 0 }
}

/// On non-Unix platforms return `true` as a conservative fallback to match
/// the existing `super::is_process_alive` behaviour.
#[cfg(not(unix))]
fn is_process_alive(_pid: u32) -> bool {
    true
}

#[cfg(test)]
mod e8_listener_tests {
    use super::*;
    use tempfile::TempDir;

    /// `warm_state_ready` must return false when the sentinel does not exist —
    /// the MCP layer should fall back to a full state check in that case.
    #[test]
    fn warm_state_ready_false_when_missing() {
        let tmp = TempDir::new().unwrap();
        assert!(!warm_state_ready(tmp.path()));
    }

    /// A sentinel containing the current PID counts as warm; current PID is
    /// always alive by definition.
    #[test]
    fn warm_state_ready_true_for_current_pid() {
        let tmp = TempDir::new().unwrap();
        let sentinel = tmp.path().join(WARM_STATE_SENTINEL);
        std::fs::write(&sentinel, std::process::id().to_string()).unwrap();
        assert!(warm_state_ready(tmp.path()));
    }

    /// A sentinel pointing at a definitely-dead PID should return false AND
    /// remove the stale sentinel so subsequent calls take the fast no-file
    /// path.
    ///
    /// We can't hard-code a "dead PID" because the OS reuses them. Instead,
    /// we spawn a child, wait for it to exit, then use its (now-dead) PID.
    /// There's a small race window where the OS might recycle the PID before
    /// our check — extremely unlikely on a normal test host, and `cargo test`
    /// retries flaky tests anyway.
    #[cfg(unix)]
    #[test]
    fn warm_state_ready_removes_stale_sentinel() {
        use std::process::Command;

        let child = Command::new("true")
            .spawn()
            .expect("failed to spawn /usr/bin/true");
        let dead_pid = child.id();
        // Wait for the child to actually exit so the PID becomes dead.
        let _ = child.wait_with_output();

        let tmp = TempDir::new().unwrap();
        let sentinel = tmp.path().join(WARM_STATE_SENTINEL);
        std::fs::write(&sentinel, dead_pid.to_string()).unwrap();
        assert!(
            !warm_state_ready(tmp.path()),
            "PID {dead_pid} should be dead and trigger the stale-sentinel path"
        );
        assert!(
            !sentinel.exists(),
            "stale sentinel should be removed by warm_state_ready"
        );
    }

    /// Finding 12: a malformed sentinel (non-numeric content) is treated as
    /// not-warm and must be LEFT IN PLACE. The previous behaviour (delete on
    /// parse failure) raced with a daemon's mid-write O_TRUNC window: the
    /// reader observed an empty file and unlinked the inode, after which the
    /// daemon's write_all hit the unlinked inode and disappeared — permanently
    /// disabling E8d for the daemon's lifetime.
    #[test]
    fn warm_state_ready_does_not_remove_malformed_sentinel() {
        let tmp = TempDir::new().unwrap();
        let sentinel = tmp.path().join(WARM_STATE_SENTINEL);
        std::fs::write(&sentinel, "not-a-pid").unwrap();
        assert!(!warm_state_ready(tmp.path()));
        assert!(
            sentinel.exists(),
            "Finding 12: malformed sentinel must NOT be removed — could be a \
             transient empty-file race with a concurrent daemon write"
        );
    }

    /// Finding 12: an empty sentinel (the worst case of the O_TRUNC race) must
    /// likewise be left in place.
    #[test]
    fn warm_state_ready_does_not_remove_empty_sentinel() {
        let tmp = TempDir::new().unwrap();
        let sentinel = tmp.path().join(WARM_STATE_SENTINEL);
        std::fs::write(&sentinel, "").unwrap();
        assert!(
            !warm_state_ready(tmp.path()),
            "empty sentinel must report not-ready"
        );
        assert!(
            sentinel.exists(),
            "empty sentinel must NOT be deleted — the daemon may be \
             rewriting it right now"
        );
    }

    /// Finding 12: `write_sentinel_atomic` must produce a fully-written PID
    /// file. Even when called many times in rapid succession, no observer
    /// should ever see a partial or empty file (rename is atomic on POSIX).
    #[test]
    fn write_sentinel_atomic_produces_complete_file() {
        let tmp = TempDir::new().unwrap();
        let sentinel = tmp.path().join(WARM_STATE_SENTINEL);

        // Multiple sequential writes should all leave a well-formed file.
        for _ in 0..50 {
            write_sentinel_atomic(&sentinel, std::process::id());
            let contents = std::fs::read_to_string(&sentinel).unwrap();
            let parsed: u32 = contents
                .trim()
                .parse()
                .expect("sentinel must always contain a parseable PID");
            assert_eq!(parsed, std::process::id());
        }
    }

    /// Finding 12 (concurrent writer + reader): the reader must never observe
    /// an empty or partial sentinel while the writer is busy. This test runs
    /// a tight read loop on one thread while another thread hammers atomic
    /// writes. After the writer completes, the sentinel must be present and
    /// well-formed.
    #[test]
    fn write_sentinel_atomic_under_concurrent_reads() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

        let tmp = TempDir::new().unwrap();
        let sentinel = Arc::new(tmp.path().join(WARM_STATE_SENTINEL));
        let stop = Arc::new(AtomicBool::new(false));
        let empty_observed = Arc::new(AtomicUsize::new(0));
        let partial_observed = Arc::new(AtomicUsize::new(0));

        // Reader thread: tight loop reading the sentinel and recording any
        // empty / unparseable observations.
        let reader_sentinel = Arc::clone(&sentinel);
        let reader_stop = Arc::clone(&stop);
        let reader_empty = Arc::clone(&empty_observed);
        let reader_partial = Arc::clone(&partial_observed);
        let reader = std::thread::spawn(move || {
            while !reader_stop.load(Ordering::Relaxed) {
                if let Ok(content) = std::fs::read_to_string(&*reader_sentinel) {
                    if content.is_empty() {
                        reader_empty.fetch_add(1, Ordering::Relaxed);
                    } else if content.trim().parse::<u32>().is_err() {
                        reader_partial.fetch_add(1, Ordering::Relaxed);
                    }
                }
                // No sleep — we want the tightest possible read loop to
                // maximise the chance of catching a write-in-progress.
            }
        });

        // Writer: 100 atomic writes in rapid succession.
        for _ in 0..100 {
            write_sentinel_atomic(&sentinel, std::process::id());
        }

        stop.store(true, Ordering::Relaxed);
        reader.join().unwrap();

        // After all writes complete, sentinel must exist and be valid.
        assert!(
            sentinel.exists(),
            "sentinel must exist after writes complete"
        );
        let final_contents = std::fs::read_to_string(&*sentinel).unwrap();
        let parsed: u32 = final_contents.trim().parse().unwrap();
        assert_eq!(parsed, std::process::id());

        let empty = empty_observed.load(Ordering::Relaxed);
        let partial = partial_observed.load(Ordering::Relaxed);
        assert_eq!(
            empty, 0,
            "Finding 12: reader observed {empty} empty-file reads — \
             write_sentinel_atomic is not atomic"
        );
        assert_eq!(
            partial, 0,
            "Finding 12: reader observed {partial} partial reads — \
             write_sentinel_atomic is not atomic"
        );
    }

    /// Finding 13: when a `Listener` is dropped, it must not delete a sentinel
    /// owned by some other process. Construct a fake Listener path and seed
    /// the sentinel with a foreign PID; after invoking the drop helper the
    /// file must still be there.
    #[test]
    fn remove_sentinel_if_ours_leaves_foreign_pid_alone() {
        let tmp = TempDir::new().unwrap();
        let sentinel = tmp.path().join(WARM_STATE_SENTINEL);

        // Pick a PID that's definitely not ours and definitely valid u32.
        let my_pid = std::process::id();
        let foreign_pid = if my_pid == 1 { 2 } else { 1 };
        std::fs::write(&sentinel, foreign_pid.to_string()).unwrap();

        remove_sentinel_if_ours(&sentinel);

        assert!(
            sentinel.exists(),
            "Finding 13: must not delete sentinel owned by PID {foreign_pid} \
             (ours is {my_pid})"
        );
        let contents = std::fs::read_to_string(&sentinel).unwrap();
        let parsed: u32 = contents.trim().parse().unwrap();
        assert_eq!(
            parsed, foreign_pid,
            "Finding 13: foreign sentinel content must be untouched"
        );
    }

    /// Finding 13 complement: when the sentinel does contain OUR PID, removal
    /// must succeed.
    #[test]
    fn remove_sentinel_if_ours_deletes_our_pid() {
        let tmp = TempDir::new().unwrap();
        let sentinel = tmp.path().join(WARM_STATE_SENTINEL);
        std::fs::write(&sentinel, std::process::id().to_string()).unwrap();

        remove_sentinel_if_ours(&sentinel);

        assert!(
            !sentinel.exists(),
            "remove_sentinel_if_ours must delete a sentinel containing our PID"
        );
    }

    /// Finding 13 corner case: a malformed (non-numeric) sentinel must not be
    /// touched by `remove_sentinel_if_ours` — we can't prove it's ours, so we
    /// leave it alone.
    #[test]
    fn remove_sentinel_if_ours_leaves_malformed_alone() {
        let tmp = TempDir::new().unwrap();
        let sentinel = tmp.path().join(WARM_STATE_SENTINEL);
        std::fs::write(&sentinel, "garbage").unwrap();

        remove_sentinel_if_ours(&sentinel);

        assert!(
            sentinel.exists(),
            "Finding 13: malformed sentinel must not be removed (can't prove ownership)"
        );
    }

    /// Finding 13: missing file is a no-op (no panic, no error).
    #[test]
    fn remove_sentinel_if_ours_handles_missing_file() {
        let tmp = TempDir::new().unwrap();
        let sentinel = tmp.path().join(WARM_STATE_SENTINEL);
        // File does not exist — must be a quiet no-op.
        remove_sentinel_if_ours(&sentinel);
        assert!(!sentinel.exists());
    }
}
