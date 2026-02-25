use anyhow::{Context, Result};
use sage_core::server::protocol::{self, BINARY_MAGIC, BinaryRequest, BinaryResponse};
use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::time::Duration;

/// PID file name for the persistent client process.
const CLIENT_PID_FILE: &str = "client.pid";

/// A persistent client that maintains a long-lived TCP connection to the
/// sage daemon. Eliminates per-query connection overhead and uses the
/// binary (bincode) protocol for minimal serialization cost.
pub struct PersistentClient {
    stream: TcpStream,
    /// Pre-allocated read buffer
    read_buf: Vec<u8>,
}

impl PersistentClient {
    /// Connect to the daemon at the given TCP port.
    pub fn connect(port: u16) -> Result<Self> {
        let addr = std::net::SocketAddr::from(([127, 0, 0, 1], port));
        let stream = TcpStream::connect_timeout(&addr, Duration::from_secs(5))
            .with_context(|| format!("Failed to connect to daemon at 127.0.0.1:{port}"))?;

        stream.set_read_timeout(Some(Duration::from_secs(30)))?;
        stream.set_write_timeout(Some(Duration::from_secs(5)))?;

        // Pre-allocate 64 KiB read buffer
        let read_buf = vec![0u8; 64 * 1024];

        Ok(Self { stream, read_buf })
    }

    /// Send a binary health check.
    pub fn health(&mut self) -> Result<bool> {
        let frame = protocol::encode_binary_request(&BinaryRequest::Health);
        self.stream.write_all(&frame)?;
        self.stream.flush()?;

        let bin_resp = self.read_binary_response()?;
        Ok(matches!(bin_resp, BinaryResponse::Health(_)))
    }

    /// Read a binary-framed response from the stream.
    fn read_binary_response(&mut self) -> Result<BinaryResponse> {
        let mut magic = [0u8; 1];
        self.stream.read_exact(&mut magic)?;
        if magic[0] != BINARY_MAGIC {
            anyhow::bail!("Expected binary response, got 0x{:02x}", magic[0]);
        }

        let mut len_buf = [0u8; 4];
        self.stream.read_exact(&mut len_buf)?;
        let len = u32::from_le_bytes(len_buf) as usize;

        // Grow buffer if needed
        if self.read_buf.len() < len {
            self.read_buf.resize(len, 0);
        }

        self.stream.read_exact(&mut self.read_buf[..len])?;

        protocol::decode_binary_response(&self.read_buf[..len])
            .map_err(|e| anyhow::anyhow!("Failed to decode binary response: {e}"))
    }
}

// --- PID file management ---

/// Path to the sage state directory: ~/.sage/ (cross-platform)
pub fn sage_home() -> Result<PathBuf> {
    Ok(sage_core::config::SageConfig::sage_home())
}

/// Path to the persistent client PID file: ~/.sage/client.pid
pub fn client_pid_path() -> Result<PathBuf> {
    Ok(sage_home()?.join(CLIENT_PID_FILE))
}

/// Write the current process PID to the client PID file.
pub fn write_client_pid() -> Result<()> {
    let dir = sage_home()?;
    std::fs::create_dir_all(&dir)?;
    std::fs::write(client_pid_path()?, std::process::id().to_string())?;
    Ok(())
}

/// Check if a persistent client process is running. Returns the PID if the
/// PID file exists and the process appears to be alive.
pub fn client_alive() -> Option<u32> {
    let path = client_pid_path().ok()?;
    let pid_str = std::fs::read_to_string(&path).ok()?;
    let pid: u32 = pid_str.trim().parse().ok()?;

    if process_exists(pid) {
        Some(pid)
    } else {
        // Stale PID file, clean up
        let _ = std::fs::remove_file(&path);
        None
    }
}

/// Check whether a process with the given PID exists (cross-platform).
fn process_exists(pid: u32) -> bool {
    #[cfg(unix)]
    {
        // SAFETY: signal 0 is a read-only existence check.
        // The PID comes from our own PID file, not external input.
        unsafe { libc::kill(pid as libc::pid_t, 0) == 0 }
    }
    #[cfg(windows)]
    {
        // Attempt to open the process with PROCESS_QUERY_LIMITED_INFORMATION
        use std::process::Command;
        Command::new("tasklist")
            .args(["/FI", &format!("PID eq {pid}"), "/NH"])
            .output()
            .is_ok_and(|o| String::from_utf8_lossy(&o.stdout).contains(&pid.to_string()))
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = pid;
        // Cannot check process existence on this platform — assume alive
        true
    }
}

/// Stop the persistent client by cleaning up the PID file.
/// The daemon itself is stopped via `sage stop`.
pub fn stop_client() -> Result<bool> {
    let path = client_pid_path()?;
    if !path.exists() {
        return Ok(false);
    }
    let _ = std::fs::remove_file(&path);
    Ok(true)
}
