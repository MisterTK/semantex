use super::handler::Handler;
use super::protocol::{self, BINARY_MAGIC, BinaryResponse, ErrorResponse, Request, Response};
use crate::search::hybrid::HybridSearcher;
use anyhow::{Context, Result};
use std::io::{BufRead, BufReader, Read, Write};
use std::os::unix::net::UnixListener;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Unix domain socket listener for the sage daemon
#[allow(clippy::struct_field_names)]
pub struct Listener {
    socket_path: PathBuf,
    listener: UnixListener,
    searcher: HybridSearcher,
    search_count: AtomicU64,
    start_time: Instant,
    idle_timeout: Duration,
    shutdown: Arc<AtomicBool>,
}

impl Listener {
    /// Create a new UDS listener bound to the given socket path
    pub fn bind(
        socket_path: &Path,
        searcher: HybridSearcher,
        idle_timeout: Duration,
        shutdown: Arc<AtomicBool>,
    ) -> Result<Self> {
        // Remove stale socket file if it exists
        if socket_path.exists() {
            std::fs::remove_file(socket_path).with_context(|| {
                format!("Failed to remove stale socket: {}", socket_path.display())
            })?;
        }

        let listener = UnixListener::bind(socket_path)
            .with_context(|| format!("Failed to bind UDS at {}", socket_path.display()))?;

        // Blocking mode with 500ms SO_RCVTIMEO for periodic shutdown/idle checks.
        // This replaces the 50ms non-blocking poll loop, reducing avg latency from 25ms to ~0ms.
        {
            use std::os::unix::io::AsRawFd;
            let timeout = libc::timeval {
                tv_sec: 0,
                tv_usec: 500_000, // 500ms
            };
            // SAFETY: Setting socket option on a valid file descriptor before any concurrent access.
            unsafe {
                libc::setsockopt(
                    listener.as_raw_fd(),
                    libc::SOL_SOCKET,
                    libc::SO_RCVTIMEO,
                    (&raw const timeout).cast::<libc::c_void>(),
                    std::mem::size_of::<libc::timeval>() as libc::socklen_t,
                );
            }
        }

        Ok(Self {
            socket_path: socket_path.to_path_buf(),
            listener,
            searcher,
            search_count: AtomicU64::new(0),
            start_time: Instant::now(),
            idle_timeout,
            shutdown,
        })
    }

    /// Run the accept loop until shutdown or idle timeout
    pub fn run(&self) -> Result<()> {
        tracing::info!(
            socket = %self.socket_path.display(),
            idle_timeout_s = self.idle_timeout.as_secs(),
            "Daemon listening"
        );

        let mut last_activity = Instant::now();

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

            // Accept a connection (blocks up to 500ms per SO_RCVTIMEO)
            match self.listener.accept() {
                Ok((stream, _addr)) => {
                    last_activity = Instant::now();
                    if let Err(e) = self.handle_connection(stream) {
                        tracing::warn!("Connection error: {}", e);
                    }
                }
                Err(ref e)
                    if e.kind() == std::io::ErrorKind::WouldBlock
                        || e.kind() == std::io::ErrorKind::TimedOut =>
                {
                    // SO_RCVTIMEO expired — loop back to check shutdown/idle
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

    #[allow(clippy::needless_pass_by_value)] // UnixStream ownership needed for set_read/write_timeout
    fn handle_connection(&self, stream: std::os::unix::net::UnixStream) -> Result<()> {
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

    /// Handle a binary (bincode) framed connection.
    fn handle_binary_connection(
        &self,
        reader: &mut BufReader<&std::os::unix::net::UnixStream>,
        stream: &std::os::unix::net::UnixStream,
    ) -> Result<()> {
        // Read 4-byte LE length
        let mut len_buf = [0u8; 4];
        reader.read_exact(&mut len_buf)?;
        let len = u32::from_le_bytes(len_buf) as usize;

        if len > 16 * 1024 * 1024 {
            // Reject unreasonably large payloads (16 MiB)
            return Ok(());
        }

        // Read payload
        let mut payload = vec![0u8; len];
        reader.read_exact(&mut payload)?;

        let response = match protocol::decode_binary_request(&payload) {
            Ok(bin_req) => {
                let request: Request = bin_req.into();
                let is_shutdown = matches!(request, Request::Shutdown);
                let handler = Handler::new(&self.searcher, &self.search_count);
                let resp = handler.handle(request, self.start_time);

                if is_shutdown {
                    self.shutdown.store(true, Ordering::Relaxed);
                }

                resp
            }
            Err(e) => Response::Error(ErrorResponse {
                message: format!("Invalid binary request: {e}"),
            }),
        };

        // Write binary response
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
        reader: &mut BufReader<&std::os::unix::net::UnixStream>,
        stream: &std::os::unix::net::UnixStream,
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
                let handler = Handler::new(&self.searcher, &self.search_count);
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
        if self.socket_path.exists() {
            let _ = std::fs::remove_file(&self.socket_path);
        }
        tracing::info!(
            searches = self.search_count.load(Ordering::Relaxed),
            uptime_s = self.start_time.elapsed().as_secs(),
            "Daemon stopped"
        );
    }
}

impl Drop for Listener {
    fn drop(&mut self) {
        if self.socket_path.exists() {
            let _ = std::fs::remove_file(&self.socket_path);
        }
    }
}
