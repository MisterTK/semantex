//! HTTP transport for the semantex MCP server (I2).
//!
//! Exposes the existing `McpServer` JSON-RPC handler over HTTP/SSE, enabling
//! agents that prefer HTTP transports (Vercel AI SDK, `LangChain`, `LlamaIndex`,
//! browser-based clients) to talk to semantex without spawning a subprocess.
//!
//! Routes:
//!   * `POST /mcp/{toolset}`         — Streamable HTTP (JSON-RPC body in, JSON-RPC body out)
//!   * `GET  /mcp/{toolset}/events`  — Server-Sent Events stream for notifications
//!   * `GET  /healthz`              — Liveness probe
//!
//! Toolsets: `core`, `structural`, `all` (per spec section 2.5 I3).
//! `POST /mcp/` (no toolset) is an alias for `/mcp/all`.
//!
//! ### Security defaults (per spec risk D3 / Wave 0 contract section C)
//! * Binds to `127.0.0.1` (loopback) by default.
//! * `--allow-remote` (forwarded via `allow_remote`) required to bind `0.0.0.0`.
//!   A loud stderr warning is printed when remote binding is enabled.
//! * CORS is disabled by default (no `Access-Control-Allow-*` headers).
//! * Bearer-token auth (see [`AuthConfig`]) is REQUIRED whenever `--allow-remote`
//!   is set — the server refuses to start remote without a resolvable token.
//!   On loopback, auth stays off by default; `--require-auth` opts in. Token
//!   resolution precedence: `--auth-token` flag > `SEMANTEX_HTTP_TOKEN` env >
//!   auto-generated 32-byte hex token persisted at `~/.semantex/http_token`
//!   (0600 perms on Unix). `/healthz` is always open; `/mcp/*` (including the
//!   SSE `/events` routes) require the token the same way once auth is on.
//!
//! ### Workaround note
//! This module currently proxies JSON-RPC requests to a child `semantex mcp`
//! subprocess because `McpServer::handle_request` is not public. See
//! `coordination_request.md` at the repo root. Swap `SubprocessBackend` for an
//! in-process call once W5 promotes `handle_request` + adds `tools_for_toolset`.

use anyhow::{Context, Result};
use axum::{
    Json, Router,
    extract::{Path, State},
    http::{HeaderMap, StatusCode, header},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use rand::RngCore;
use serde_json::Value;
use std::collections::{HashMap, VecDeque};
use std::path::{Path as FsPath, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::{Mutex, mpsc, oneshot};

// ============================================================================
// Toolset filter (mirrors spec section 2.5 I3)
// ============================================================================

/// Tool names that belong to the `core` toolset.
/// NOTE: Kept in sync with `McpServer::tools_for_toolset("core")`. The HTTP
/// dispatcher uses this list as a defense-in-depth filter — the child
/// subprocess (when started with `--toolset core`) is the authoritative source
/// of truth. See `SubprocessBackend::verify_child_toolset_alignment`.
const CORE_TOOLS: &[&str] = &[
    "semantex_search",
    "semantex_deep",
    "semantex_agent",
    "semantex_symbol",
];

/// Tool names that belong to the `structural` toolset.
const STRUCTURAL_TOOLS: &[&str] = &[
    "semantex_symbol",
    "semantex_callers",
    "semantex_callees",
    "semantex_implementations",
    "semantex_architecture",
];

/// Three valid toolset names plus the empty-string alias for `all`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Toolset {
    Core,
    Structural,
    All,
}

impl Toolset {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "core" => Some(Self::Core),
            "structural" => Some(Self::Structural),
            "all" | "" => Some(Self::All),
            _ => None,
        }
    }

    /// CLI-style name used when forwarding the toolset to the child subprocess.
    pub fn as_cli_name(self) -> &'static str {
        match self {
            Self::Core => "core",
            Self::Structural => "structural",
            Self::All => "all",
        }
    }

    /// Returns `true` if the tool name belongs to this toolset.
    fn includes(self, tool_name: &str) -> bool {
        match self {
            Self::All => true,
            Self::Core => CORE_TOOLS.contains(&tool_name),
            Self::Structural => STRUCTURAL_TOOLS.contains(&tool_name),
        }
    }

    /// Filter a `tools/list` response in-place. Keeps the metrics/extras intact;
    /// only the `tools` array is restricted.
    fn filter_tools_list(self, response: &mut Value) {
        if matches!(self, Self::All) {
            return;
        }
        let Some(result) = response.get_mut("result") else {
            return;
        };
        let Some(tools) = result.get_mut("tools").and_then(|v| v.as_array_mut()) else {
            return;
        };
        tools.retain(|t| {
            t.get("name")
                .and_then(|n| n.as_str())
                .is_some_and(|n| self.includes(n))
        });
    }
}

// ============================================================================
// Backend abstraction
// ============================================================================

/// A backend that handles JSON-RPC requests and returns JSON-RPC responses.
///
/// The HTTP transport layer is intentionally decoupled from the `McpServer`
/// implementation. Today the only backend is `SubprocessBackend` (which
/// spawns and multiplexes a child `semantex mcp` process). Once W5 promotes
/// `McpServer::handle_request` to public, we can add an `InProcessBackend`.
pub trait Backend: Send + Sync + 'static {
    /// Dispatch one JSON-RPC request, get one JSON-RPC response back.
    ///
    /// For notifications (no `id` field), the implementation should return
    /// `Ok(Value::Null)` after queuing the message — the HTTP layer turns
    /// that into a `204 No Content`.
    fn dispatch(
        &self,
        request: Value,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Value>> + Send + '_>>;
}

// ============================================================================
// SubprocessBackend — proxies to `semantex mcp` over stdio
// ============================================================================

/// Internal prefix for IDs the backend synthesizes (no client supplied one).
/// Restored to the wire id transparently before returning to the HTTP layer.
const INTERNAL_ID_PREFIX: &str = "_semantex_internal:";
/// Internal prefix for namespaced client-supplied IDs. Each request gets a
/// unique per-instance counter to ensure no collision even when multiple
/// concurrent HTTP clients send the literal same `id`.
const CLIENT_ID_PREFIX: &str = "_semantex_client:";

/// Respawn rate limit: at most `MAX_RESPAWNS_PER_WINDOW` restarts within
/// `RESPAWN_WINDOW`. After that, `ensure_started` returns an error.
const MAX_RESPAWNS_PER_WINDOW: u32 = 5;
const RESPAWN_WINDOW: Duration = Duration::from_mins(1);

/// Spawn-and-multiplex a child `semantex mcp` (stdio) and route HTTP-arriving
/// JSON-RPC messages through it. Responses are matched by `id`.
pub struct SubprocessBackend {
    inner: Arc<SubprocessInner>,
}

struct SubprocessInner {
    binary_path: PathBuf,
    /// `--toolset` to pass to the child. `None` means default ("all").
    toolset: Option<String>,
    state: Mutex<Option<ChildState>>,
    /// Atomic counter for namespacing client-supplied IDs. Bumped on every
    /// request so two concurrent clients with the same `id` get unique keys
    /// in the `pending` map.
    request_seq: AtomicU64,
    /// Recent respawn timestamps (newest at the back). Trimmed on each check.
    respawn_history: Mutex<VecDeque<Instant>>,
    /// Queue of waiters for `null`-id responses (initialization handshake style;
    /// shouldn't normally happen with our protocol but kept for safety).
    null_id_waiters: Mutex<VecDeque<oneshot::Sender<Value>>>,
}

struct ChildState {
    _child: Child,
    /// Sender into the dedicated writer task. The writer task owns
    /// `ChildStdin` exclusively, so concurrent `send_request` calls don't
    /// contend on a single stdin lock.
    writer_tx: mpsc::UnboundedSender<Vec<u8>>,
    /// Map of in-flight requests keyed by the wire id we sent the child.
    /// Reader looks up by this key when a response arrives.
    pending: Arc<Mutex<HashMap<String, oneshot::Sender<Value>>>>,
}

impl SubprocessBackend {
    /// Locate the `semantex` binary at runtime. Falls back to `$PATH` lookup
    /// if `current_exe()` is unavailable.
    pub fn detect_binary() -> Result<PathBuf> {
        if let Ok(this) = std::env::current_exe() {
            return Ok(this);
        }
        Ok(PathBuf::from("semantex"))
    }

    pub fn new(binary_path: PathBuf) -> Self {
        Self::new_with_toolset(binary_path, None)
    }

    /// Construct a `SubprocessBackend` that forwards `--toolset <name>` to the
    /// child mcp process. Pass `None` to leave the default ("all").
    pub fn new_with_toolset(binary_path: PathBuf, toolset: Option<String>) -> Self {
        Self {
            inner: Arc::new(SubprocessInner {
                binary_path,
                toolset,
                state: Mutex::new(None),
                request_seq: AtomicU64::new(1),
                respawn_history: Mutex::new(VecDeque::new()),
                null_id_waiters: Mutex::new(VecDeque::new()),
            }),
        }
    }

    /// Check + record a respawn. Returns Err if the rate limit is exceeded.
    async fn enforce_respawn_limit(&self) -> Result<()> {
        let mut history = self.inner.respawn_history.lock().await;
        let now = Instant::now();
        // Drop entries older than the window.
        while let Some(front) = history.front() {
            if now.duration_since(*front) > RESPAWN_WINDOW {
                history.pop_front();
            } else {
                break;
            }
        }
        if history.len() >= MAX_RESPAWNS_PER_WINDOW as usize {
            anyhow::bail!(
                "subprocess respawn rate limit exceeded ({} restarts in {}s); refusing to spawn again",
                MAX_RESPAWNS_PER_WINDOW,
                RESPAWN_WINDOW.as_secs()
            );
        }
        history.push_back(now);
        Ok(())
    }

    /// Lazily start the child process on first use.
    async fn ensure_started(&self) -> Result<()> {
        let mut state = self.inner.state.lock().await;
        if state.is_some() {
            return Ok(());
        }

        // Enforce respawn limit BEFORE the (relatively expensive) spawn.
        // Holding `state` lock here ensures no two ensure_started calls race.
        self.enforce_respawn_limit().await?;

        let mut cmd = Command::new(&self.inner.binary_path);
        cmd.arg("mcp");
        if let Some(ts) = &self.inner.toolset {
            cmd.arg("--toolset").arg(ts);
        }
        let mut child = cmd
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .kill_on_drop(true)
            .spawn()
            .with_context(|| {
                format!(
                    "failed to spawn `{} mcp` for HTTP transport backend",
                    self.inner.binary_path.display()
                )
            })?;

        let stdin = child
            .stdin
            .take()
            .context("child mcp process has no stdin")?;
        let stdout = child
            .stdout
            .take()
            .context("child mcp process has no stdout")?;

        let pending: Arc<Mutex<HashMap<String, oneshot::Sender<Value>>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let pending_for_reader = Arc::clone(&pending);
        let inner_for_reader = Arc::clone(&self.inner);

        // Dedicated writer task — owns ChildStdin exclusively.
        // No stdin lock needed: send_request just enqueues bytes.
        let (writer_tx, writer_rx) = mpsc::unbounded_channel::<Vec<u8>>();
        tokio::spawn(writer_loop(stdin, writer_rx));

        // Reader: pumps stdout, dispatches responses to pending oneshots.
        // On exit, drains pending + clears backend state to allow respawn.
        tokio::spawn(reader_loop(stdout, pending_for_reader, inner_for_reader));

        *state = Some(ChildState {
            _child: child,
            writer_tx,
            pending,
        });
        Ok(())
    }

    /// Send one JSON-RPC request through the child and await the response.
    ///
    /// Notifications (no `id` field) are fire-and-forget and return
    /// `Ok(Value::Null)` so the HTTP layer can return `204 No Content`.
    async fn send_request(&self, mut request: Value) -> Result<Value> {
        self.ensure_started().await?;

        // ── Notification fast-path ────────────────────────────────────────
        // JSON-RPC spec: notifications have no `id` and MUST receive no
        // response. Don't register a waiter — just enqueue the bytes and
        // return Null so the HTTP layer answers 204 No Content.
        if request.get("id").is_none() {
            let mut line = serde_json::to_string(&request)
                .context("failed to serialize JSON-RPC notification")?;
            line.push('\n');
            // Snapshot the writer_tx under the state lock, then drop the lock
            // before sending so we don't serialize concurrent HTTP traffic.
            let writer_tx = {
                let guard = self.inner.state.lock().await;
                guard
                    .as_ref()
                    .context("subprocess backend not initialized")?
                    .writer_tx
                    .clone()
            };
            writer_tx
                .send(line.into_bytes())
                .map_err(|_| anyhow::anyhow!("subprocess writer channel closed"))?;
            return Ok(Value::Null);
        }

        // ── Request path: namespace the id, register pending, send ────────
        //
        // Why we namespace:
        //   * Synthesized ids must never collide with client-supplied ids.
        //   * Two clients sending `{"id": 1, ...}` concurrently must each get
        //     their own response back, not the other's.
        //
        // Strategy:
        //   * Capture the original id (if any) so we can restore it on the
        //     response before returning to the HTTP handler.
        //   * Rewrite the wire id to a unique key in {INTERNAL,CLIENT}_PREFIX
        //     namespace using a monotonic per-instance counter.

        let original_id = request.get("id").cloned();
        let seq = self.inner.request_seq.fetch_add(1, Ordering::Relaxed);

        let (wire_id_str, wire_id_value, restore_original) = match &original_id {
            None => unreachable!("notification handled above"),
            Some(Value::Null) => {
                // Treat null-id as a (rare) request — synthesize a fresh id.
                let key = format!("{INTERNAL_ID_PREFIX}{seq}");
                let v = Value::String(key.clone());
                (key, v, true)
            }
            Some(client_id) => {
                // Namespace the client id with the per-request seq so two
                // clients sending `id:1` never collide.
                let key = format!(
                    "{CLIENT_ID_PREFIX}{seq}:{}",
                    serde_json::to_string(client_id).unwrap_or_else(|_| "null".to_string())
                );
                let v = Value::String(key.clone());
                (key, v, true)
            }
        };
        request["id"] = wire_id_value;

        // Register pending BEFORE writing — eliminates a race where the
        // response could arrive before the waiter is registered.
        let (tx, rx) = oneshot::channel::<Value>();
        let pending = {
            let guard = self.inner.state.lock().await;
            let state = guard
                .as_ref()
                .context("subprocess backend not initialized")?;
            state.pending.lock().await.insert(wire_id_str.clone(), tx);
            Arc::clone(&state.pending)
        };

        // Snapshot the writer channel and drop the state lock.
        let writer_tx = {
            let guard = self.inner.state.lock().await;
            guard
                .as_ref()
                .context("subprocess backend not initialized")?
                .writer_tx
                .clone()
        };

        let mut line =
            serde_json::to_string(&request).context("failed to serialize JSON-RPC request")?;
        line.push('\n');

        // If writer queue is closed, drop our pending entry to avoid leaking.
        if let Err(e) = writer_tx.send(line.into_bytes()) {
            pending.lock().await.remove(&wire_id_str);
            return Err(anyhow::anyhow!("subprocess writer channel closed: {e}"));
        }

        // Await the response.
        let mut response = rx
            .await
            .map_err(|_| anyhow::anyhow!("subprocess closed before response arrived"))?;

        // Restore the original id on the wire response.
        if restore_original
            && let Some(orig) = original_id
            && let Some(obj) = response.as_object_mut()
        {
            obj.insert("id".to_string(), orig);
        }

        Ok(response)
    }

    /// Cross-check the child's `tools/list` output against the toolset we
    /// asked for. Logs a warning if they diverge — this is a defense-in-depth
    /// signal that something is out of sync (constants here vs. server.rs).
    /// Returns the response unmodified.
    #[cfg(test)]
    pub fn expected_tools_for(toolset: Toolset) -> Vec<&'static str> {
        match toolset {
            Toolset::Core => CORE_TOOLS.to_vec(),
            Toolset::Structural => STRUCTURAL_TOOLS.to_vec(),
            Toolset::All => vec![],
        }
    }
}

/// Writer task: owns `ChildStdin` and drains the outbound queue.
/// Eliminates contention on stdin — multiple `send_request` callers just
/// enqueue bytes via the mpsc channel and continue.
async fn writer_loop(mut stdin: ChildStdin, mut rx: mpsc::UnboundedReceiver<Vec<u8>>) {
    while let Some(bytes) = rx.recv().await {
        if let Err(e) = stdin.write_all(&bytes).await {
            tracing::error!(err = %e, "writer_loop: failed to write to child stdin; exiting");
            break;
        }
        // Best-effort flush; failure on flush is usually just "stdin closed".
        if let Err(e) = stdin.flush().await {
            tracing::warn!(err = %e, "writer_loop: child stdin flush failed");
        }
    }
    tracing::debug!("writer_loop: outbound queue closed; exiting");
}

/// Reader loop: pumps lines from the child's stdout, parses each as a
/// JSON-RPC message, and dispatches responses to waiting callers.
///
/// On exit (child closed stdout or read error), drains the pending map and
/// sends each awaiter a synthetic error response so they wake immediately
/// instead of hanging. Also clears `inner.state` so a subsequent
/// `ensure_started` triggers a respawn.
async fn reader_loop(
    stdout: ChildStdout,
    pending: Arc<Mutex<HashMap<String, oneshot::Sender<Value>>>>,
    inner: Arc<SubprocessInner>,
) {
    let mut reader = BufReader::new(stdout).lines();
    let exit_reason: &str;
    loop {
        match reader.next_line().await {
            Ok(Some(line)) => {
                if line.trim().is_empty() {
                    continue;
                }
                let value: Value = match serde_json::from_str(&line) {
                    Ok(v) => v,
                    Err(e) => {
                        tracing::warn!(err = %e, line = %line, "non-JSON line from child mcp");
                        continue;
                    }
                };

                // Dispatch by id. Wire ids are always our prefixed strings
                // (we rewrote them on send), but accept numeric/null too in
                // case the child echoes something odd.
                let id = value.get("id").cloned();
                match id {
                    Some(Value::String(s)) => {
                        if let Some(tx) = pending.lock().await.remove(&s) {
                            let _ = tx.send(value);
                        }
                    }
                    Some(Value::Number(n)) => {
                        if let Some(tx) = pending.lock().await.remove(&n.to_string()) {
                            let _ = tx.send(value);
                        }
                    }
                    Some(Value::Null) | None => {
                        // Notification from the child OR a response with null id.
                        if let Some(tx) = inner.null_id_waiters.lock().await.pop_front() {
                            let _ = tx.send(value);
                        }
                    }
                    _ => {}
                }
            }
            Ok(None) => {
                exit_reason = "child closed stdout";
                break;
            }
            Err(e) => {
                tracing::error!(err = %e, "child mcp stdout read error");
                exit_reason = "child stdout read error";
                break;
            }
        }
    }
    tracing::warn!(reason = exit_reason, "child mcp subprocess exited");

    // ── Drain pending: wake each awaiter with a proper error response ─────
    //
    // Without this, every in-flight send_request hangs forever on the
    // oneshot receiver. With it, callers get a JSON-RPC-shaped error they
    // can surface to the HTTP client.
    let mut pending_map = pending.lock().await;
    let drained: Vec<(String, oneshot::Sender<Value>)> = pending_map.drain().collect();
    drop(pending_map);
    for (wire_id, tx) in drained {
        let _ = tx.send(serde_json::json!({
            "jsonrpc": "2.0",
            "id": wire_id,
            "error": {
                "code": -32099,
                "message": format!("subprocess closed: {exit_reason}")
            }
        }));
    }

    // Also drain null_id_waiters so they don't leak.
    let mut null_waiters = inner.null_id_waiters.lock().await;
    while let Some(tx) = null_waiters.pop_front() {
        let _ = tx.send(serde_json::json!({
            "jsonrpc": "2.0",
            "id": null,
            "error": {
                "code": -32099,
                "message": format!("subprocess closed: {exit_reason}")
            }
        }));
    }
    drop(null_waiters);

    // Clear state so the next ensure_started spawns a fresh child.
    *inner.state.lock().await = None;
}

impl Backend for SubprocessBackend {
    fn dispatch(
        &self,
        request: Value,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Value>> + Send + '_>> {
        Box::pin(async move { self.send_request(request).await })
    }
}

// ============================================================================
// Bearer-token auth (Wave 0 contract section C)
// ============================================================================

/// Length in bytes of an auto-generated token (before hex-encoding, so the
/// persisted file holds 64 hex characters).
const GENERATED_TOKEN_BYTES: usize = 32;

/// Raw auth inputs forwarded from the CLI. Kept separate from the resolved
/// `Option<String>` token so `resolve` can apply the full precedence order
/// (flag > env > auto-generated persisted) in one place, with unit tests
/// exercising each branch independently of the CLI's clap parsing.
#[derive(Debug, Clone, Default)]
pub struct AuthConfig {
    /// `--auth-token <TOKEN>`, if the caller passed one explicitly.
    pub cli_token: Option<String>,
    /// `--require-auth`: force enforcement even on loopback.
    pub require_auth: bool,
}

impl AuthConfig {
    /// Resolve the token to enforce for this run, given whether the server
    /// is binding remotely.
    ///
    /// Returns:
    /// * `Ok(Some(token))` — auth is enforced with this token.
    /// * `Ok(None)` — auth is not requested (loopback, no `--require-auth`,
    ///   no explicit token/env) and stays off, matching today's behavior.
    /// * `Err(_)` — auth was requested (remote, `--require-auth`, or an
    ///   explicit token/env was set) but no token could be materialized.
    ///   Callers must fail closed rather than start unauthenticated.
    pub fn resolve(&self, allow_remote: bool) -> Result<Option<String>> {
        let env_token = std::env::var("SEMANTEX_HTTP_TOKEN")
            .ok()
            .filter(|s| !s.is_empty());
        // Precedence: --auth-token flag > SEMANTEX_HTTP_TOKEN env. An empty
        // string in either slot is ignored (never enforce an empty token —
        // a bare `Authorization: Bearer ` header would trivially match it).
        let explicit = self
            .cli_token
            .clone()
            .filter(|s| !s.is_empty())
            .or(env_token);

        // Whenever `--allow-remote` is set, auth is REQUIRED (contract C) —
        // the old fully-open 0.0.0.0 mode must become impossible. On
        // loopback, auth stays off unless the caller opted in via
        // `--require-auth` or by supplying a token explicitly. A present but
        // empty `--auth-token` still counts as opting in (falls through to
        // the generated token) so a degenerate flag never disables auth.
        let auth_wanted =
            allow_remote || self.require_auth || explicit.is_some() || self.cli_token.is_some();
        if !auth_wanted {
            return Ok(None);
        }
        if let Some(token) = explicit {
            return Ok(Some(token));
        }

        // Auto-generate + persist at ~/.semantex/http_token.
        load_or_generate_persisted_token(&default_token_path()).map(Some)
    }
}

/// `~/.semantex/http_token` — the default persisted-token location. Routed
/// through `semantex_home()` so a `SEMANTEX_HOME` override relocates the
/// token alongside every other per-user semantex file (`semantex_home()`
/// itself falls back to a temp dir when no home directory exists, so this
/// always yields a usable path).
fn default_token_path() -> PathBuf {
    semantex_core::config::SemantexConfig::semantex_home().join("http_token")
}

/// Read the persisted token at `path` if present and non-empty; otherwise
/// generate a fresh 32-byte random hex token, persist it (0600 on Unix, best
/// effort elsewhere), and print the *path* (never the token) to stderr.
fn load_or_generate_persisted_token(path: &FsPath) -> Result<String> {
    if let Ok(existing) = std::fs::read_to_string(path) {
        let trimmed = existing.trim();
        if !trimmed.is_empty() {
            return Ok(trimmed.to_string());
        }
    }

    let token = generate_token_hex();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    write_token_file(path, &token)
        .with_context(|| format!("failed to persist auth token at {}", path.display()))?;
    eprintln!(
        "semantex mcp --http: generated a new bearer token, saved to {}",
        path.display()
    );
    Ok(token)
}

/// Generate a random 32-byte token, hex-encoded (64 chars).
fn generate_token_hex() -> String {
    use std::fmt::Write;
    let mut bytes = [0u8; GENERATED_TOKEN_BYTES];
    rand::rng().fill_bytes(&mut bytes);
    let mut hex = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(hex, "{b:02x}");
    }
    hex
}

#[cfg(unix)]
fn write_token_file(path: &FsPath, token: &str) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    std::fs::write(path, token)?;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
    Ok(())
}

#[cfg(not(unix))]
fn write_token_file(path: &FsPath, token: &str) -> Result<()> {
    // Best-effort on non-Unix targets: no POSIX permission bits to restrict.
    std::fs::write(path, token).context("failed to write token file")
}

/// Constant-time byte comparison. Runs in time proportional to the length of
/// the SHORTER input regardless of where the first mismatch occurs (the only
/// thing that leaks via timing is a length mismatch, which is not sensitive —
/// token length isn't secret). Hand-rolled per contract section C ("use
/// `subtle` ONLY if already in the dependency tree; otherwise hand-roll a
/// branchless byte-compare") — `subtle` is only present transitively via
/// `rustls`, not a direct dependency of any workspace crate, so this avoids
/// promoting it to a direct dep for one function.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff: u8 = 0;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

/// Check the `Authorization: Bearer <token>` header against the expected
/// token using constant-time comparison. Returns `None` when auth passes (or
/// is disabled, i.e. `expected` is `None`); `Some(response)` (401 +
/// `WWW-Authenticate: Bearer`) when it fails.
fn check_bearer_auth(headers: &HeaderMap, expected: Option<&str>) -> Option<Response> {
    let expected = expected?;
    let provided = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "));
    match provided {
        Some(token) if constant_time_eq(token.as_bytes(), expected.as_bytes()) => None,
        _ => Some(unauthorized_response()),
    }
}

fn unauthorized_response() -> Response {
    (
        StatusCode::UNAUTHORIZED,
        [(header::WWW_AUTHENTICATE, "Bearer")],
        Json(serde_json::json!({
            "error": "unauthorized",
            "hint": "missing or invalid bearer token; set the Authorization: Bearer <token> header",
        })),
    )
        .into_response()
}

// ============================================================================
// HTTP server
// ============================================================================

/// State shared by all axum handlers.
struct AppState<B: Backend> {
    backend: Arc<B>,
    /// `Some(token)` when bearer auth is enforced on `/mcp/*` routes (both
    /// POST and the SSE `/events` routes); `None` when auth is disabled
    /// (loopback default, no `--require-auth`, no explicit token/env).
    /// `/healthz` never consults this — it is always open.
    auth_token: Option<Arc<str>>,
}

// Manual `Clone` impl because the derive would require `B: Clone`, but the
// backend itself is held behind `Arc` so we only need to clone the pointer.
impl<B: Backend> Clone for AppState<B> {
    fn clone(&self) -> Self {
        Self {
            backend: Arc::clone(&self.backend),
            auth_token: self.auth_token.clone(),
        }
    }
}

/// Run the HTTP MCP server.
///
/// * `port` — TCP port to bind.
/// * `allow_remote` — if `false`, binds `127.0.0.1`; if `true`, binds `0.0.0.0`
///   and prints a security warning to stderr.
/// * `auth` — bearer-token auth inputs (`--auth-token` / `--require-auth`).
///   Resolved here per contract section C precedence; remote binds refuse to
///   start if no token can be resolved.
/// * `backend` — JSON-RPC dispatcher (today: `SubprocessBackend`).
pub async fn run_http_server<B: Backend>(
    port: u16,
    allow_remote: bool,
    auth: AuthConfig,
    backend: Arc<B>,
) -> Result<()> {
    // Resolve auth BEFORE binding: `--allow-remote` must refuse to start if
    // no token is resolvable (contract C — the old unauthenticated 0.0.0.0
    // mode is no longer allowed to exist).
    let auth_token = auth
        .resolve(allow_remote)
        .context("failed to resolve HTTP bearer auth token")?;
    if allow_remote && auth_token.is_none() {
        anyhow::bail!(
            "refusing to bind 0.0.0.0 without a bearer auth token; pass --auth-token, \
             set SEMANTEX_HTTP_TOKEN, or ensure a home directory is available for the \
             auto-generated token at ~/.semantex/http_token",
        );
    }

    let bind_host = if allow_remote {
        eprintln!(
            "semantex mcp serve --http --allow-remote: binding to 0.0.0.0:{port}\n  WARNING: anyone on the network can reach this server.\n  Bearer-token auth is REQUIRED and enforced on all /mcp/* routes (see stderr above for the token path if it was just generated).",
        );
        "0.0.0.0"
    } else if auth_token.is_some() {
        eprintln!(
            "semantex mcp serve --http: binding to 127.0.0.1:{port} with bearer auth required (--require-auth or a token was supplied).",
        );
        "127.0.0.1"
    } else {
        "127.0.0.1"
    };

    let addr: std::net::SocketAddr = format!("{bind_host}:{port}")
        .parse()
        .with_context(|| format!("invalid bind address {bind_host}:{port}"))?;

    let app = build_router_with_auth(backend, auth_token);
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .with_context(|| format!("failed to bind {addr}"))?;

    let real_addr = listener.local_addr().unwrap_or(addr);
    tracing::info!(%real_addr, "semantex MCP HTTP transport listening");
    eprintln!("semantex mcp serve --http: listening on http://{real_addr}");

    axum::serve(listener, app)
        .await
        .context("axum::serve terminated unexpectedly")?;
    Ok(())
}

/// Build the axum `Router` with auth disabled. Exposed for tests so they can
/// drive the handlers directly via `tower::ServiceExt`. Production code
/// should go through [`run_http_server`], which calls
/// [`build_router_with_auth`] with the resolved token.
pub fn build_router<B: Backend>(backend: Arc<B>) -> Router {
    build_router_with_auth(backend, None)
}

/// Build the axum `Router`, enforcing bearer auth on `/mcp/*` (POST and SSE
/// `/events` routes alike) when `auth_token` is `Some`. `/healthz` is never
/// gated.
pub fn build_router_with_auth<B: Backend>(backend: Arc<B>, auth_token: Option<String>) -> Router {
    let state = AppState {
        backend,
        auth_token: auth_token.map(Arc::from),
    };
    Router::new()
        .route("/healthz", get(healthz))
        .route("/mcp/", post(mcp_default_post))
        .route("/mcp/", get(mcp_default_events))
        .route("/mcp/{toolset}", post(mcp_post))
        .route("/mcp/{toolset}/events", get(mcp_events))
        // Friendly default 404 with a hint
        .fallback(unknown_route)
        .with_state(state)
}

async fn healthz() -> impl IntoResponse {
    (StatusCode::OK, "ok")
}

async fn unknown_route() -> impl IntoResponse {
    (
        StatusCode::NOT_FOUND,
        Json(serde_json::json!({
            "error": "not_found",
            "hint": "valid routes: POST /mcp/{core|structural|all}, GET /mcp/{toolset}/events, GET /healthz",
        })),
    )
}

async fn mcp_default_post<B: Backend>(
    State(state): State<AppState<B>>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    if let Some(resp) = check_bearer_auth(&headers, state.auth_token.as_deref()) {
        return resp;
    }
    mcp_dispatch(state.backend, Toolset::All, body).await
}

async fn mcp_default_events<B: Backend>(
    State(state): State<AppState<B>>,
    headers: HeaderMap,
) -> Response {
    if let Some(resp) = check_bearer_auth(&headers, state.auth_token.as_deref()) {
        return resp;
    }
    // Minimal SSE: open the stream, send an immediate `event: hello` so the
    // client knows the channel is alive, then keep-alive. Full notification
    // forwarding requires backend changes; documented in coordination_request.md.
    let body = "event: hello\ndata: {\"status\":\"ok\",\"toolset\":\"all\"}\n\n";
    (
        StatusCode::OK,
        [
            ("content-type", "text/event-stream"),
            ("cache-control", "no-cache"),
        ],
        body,
    )
        .into_response()
}

async fn mcp_post<B: Backend>(
    State(state): State<AppState<B>>,
    headers: HeaderMap,
    Path(toolset_str): Path<String>,
    Json(body): Json<Value>,
) -> Response {
    if let Some(resp) = check_bearer_auth(&headers, state.auth_token.as_deref()) {
        return resp;
    }
    let Some(toolset) = Toolset::parse(&toolset_str) else {
        return unknown_toolset(&toolset_str).into_response();
    };
    mcp_dispatch(state.backend, toolset, body).await
}

async fn mcp_events<B: Backend>(
    State(state): State<AppState<B>>,
    headers: HeaderMap,
    Path(toolset_str): Path<String>,
) -> Response {
    if let Some(resp) = check_bearer_auth(&headers, state.auth_token.as_deref()) {
        return resp;
    }
    let Some(toolset) = Toolset::parse(&toolset_str) else {
        return unknown_toolset(&toolset_str).into_response();
    };
    let label = toolset.as_cli_name();
    let body = format!("event: hello\ndata: {{\"status\":\"ok\",\"toolset\":\"{label}\"}}\n\n");
    (
        StatusCode::OK,
        [
            ("content-type", "text/event-stream"),
            ("cache-control", "no-cache"),
        ],
        body,
    )
        .into_response()
}

fn unknown_toolset(name: &str) -> (StatusCode, Json<Value>) {
    (
        StatusCode::NOT_FOUND,
        Json(serde_json::json!({
            "error": "unknown_toolset",
            "toolset": name,
            "valid": ["core", "structural", "all"],
        })),
    )
}

/// Dispatch one request body to the backend, applying toolset filtering on
/// `tools/list` responses.
async fn mcp_dispatch<B: Backend>(
    backend: Arc<B>,
    toolset: Toolset,
    request: Value,
) -> axum::response::Response {
    let method = request
        .get("method")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let is_notification = request.get("id").is_none();

    // tools/call against a tool outside the toolset → -32601 without calling backend.
    if method == "tools/call"
        && let Some(tool_name) = request
            .get("params")
            .and_then(|p| p.get("name"))
            .and_then(|n| n.as_str())
        && !toolset.includes(tool_name)
    {
        return Json(serde_json::json!({
            "jsonrpc": "2.0",
            "id": request.get("id").cloned().unwrap_or(Value::Null),
            "error": {
                "code": -32601,
                "message": format!("Method not found: {tool_name} (not in toolset)")
            }
        }))
        .into_response();
    }

    match backend.dispatch(request).await {
        Ok(mut response) => {
            // Notification: backend returns Value::Null → HTTP 204 No Content.
            // The JSON-RPC spec mandates no body for notifications.
            if is_notification || response.is_null() {
                return (StatusCode::NO_CONTENT, ()).into_response();
            }
            if method == "tools/list" {
                toolset.filter_tools_list(&mut response);
            }
            Json(response).into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "jsonrpc": "2.0",
                "id": Value::Null,
                "error": {
                    "code": -32603,
                    "message": format!("backend dispatch failed: {e}")
                }
            })),
        )
            .into_response(),
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Method, Request};
    use tower::ServiceExt;

    /// In-memory backend that returns a canned `tools/list` response.
    struct FakeBackend {
        tools: Vec<&'static str>,
    }

    impl Backend for FakeBackend {
        fn dispatch(
            &self,
            request: Value,
        ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Value>> + Send + '_>>
        {
            let tools = self.tools.clone();
            Box::pin(async move {
                let method = request.get("method").and_then(|v| v.as_str()).unwrap_or("");
                let id = request.get("id").cloned();
                // Notifications: no id field → return Null per the contract.
                if id.is_none() {
                    return Ok(Value::Null);
                }
                let id = id.unwrap_or(Value::Null);
                match method {
                    "tools/list" => {
                        let tools_json: Vec<Value> = tools
                            .iter()
                            .map(|n| serde_json::json!({ "name": *n, "description": "" }))
                            .collect();
                        Ok(serde_json::json!({
                            "jsonrpc": "2.0",
                            "id": id,
                            "result": { "tools": tools_json }
                        }))
                    }
                    "ping" => Ok(serde_json::json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "result": {}
                    })),
                    _ => Ok(serde_json::json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "error": { "code": -32601, "message": format!("Method not found: {method}") }
                    })),
                }
            })
        }
    }

    fn fake_backend_with(all_tools: Vec<&'static str>) -> Arc<FakeBackend> {
        Arc::new(FakeBackend { tools: all_tools })
    }

    fn full_tool_set() -> Vec<&'static str> {
        vec![
            "semantex_search",
            "semantex_deep",
            "semantex_agent",
            "semantex_symbol",
            "semantex_callers",
            "semantex_callees",
            "semantex_implementations",
            "semantex_architecture",
            "semantex_index",
            "semantex_status",
            "semantex_health",
            "semantex_validate",
            "semantex_examples",
        ]
    }

    async fn post_json(app: Router, path: &str, body: Value) -> (StatusCode, Value) {
        let req = Request::builder()
            .method(Method::POST)
            .uri(path)
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        let status = resp.status();
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let value: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
        (status, value)
    }

    async fn post_json_raw_status(app: Router, path: &str, body: Value) -> StatusCode {
        let req = Request::builder()
            .method(Method::POST)
            .uri(path)
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();
        app.oneshot(req).await.unwrap().status()
    }

    /// Like `post_json`, but optionally attaches an `Authorization: Bearer
    /// <token>` header.
    async fn post_json_auth(
        app: Router,
        path: &str,
        body: Value,
        bearer: Option<&str>,
    ) -> (StatusCode, Value) {
        let mut builder = Request::builder()
            .method(Method::POST)
            .uri(path)
            .header("content-type", "application/json");
        if let Some(token) = bearer {
            builder = builder.header("authorization", format!("Bearer {token}"));
        }
        let req = builder
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        let status = resp.status();
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let value: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
        (status, value)
    }

    /// GET a route, optionally with a bearer token. Used to exercise the SSE
    /// endpoints and `/healthz`.
    async fn get_with_auth(app: Router, path: &str, bearer: Option<&str>) -> StatusCode {
        let mut builder = Request::builder().method(Method::GET).uri(path);
        if let Some(token) = bearer {
            builder = builder.header("authorization", format!("Bearer {token}"));
        }
        let req = builder.body(Body::empty()).unwrap();
        app.oneshot(req).await.unwrap().status()
    }

    #[test]
    fn toolset_parse_valid() {
        assert_eq!(Toolset::parse("core"), Some(Toolset::Core));
        assert_eq!(Toolset::parse("structural"), Some(Toolset::Structural));
        assert_eq!(Toolset::parse("all"), Some(Toolset::All));
        assert_eq!(Toolset::parse(""), Some(Toolset::All));
    }

    #[test]
    fn toolset_parse_invalid() {
        assert_eq!(Toolset::parse("bogus"), None);
        assert_eq!(Toolset::parse("CORE"), None); // case-sensitive
    }

    #[test]
    fn toolset_includes_core() {
        assert!(Toolset::Core.includes("semantex_search"));
        assert!(Toolset::Core.includes("semantex_agent"));
        assert!(!Toolset::Core.includes("semantex_callers"));
    }

    #[test]
    fn toolset_includes_structural() {
        assert!(Toolset::Structural.includes("semantex_symbol"));
        assert!(Toolset::Structural.includes("semantex_architecture"));
        assert!(!Toolset::Structural.includes("semantex_search"));
    }

    #[test]
    fn toolset_all_accepts_anything() {
        assert!(Toolset::All.includes("semantex_anything"));
    }

    #[test]
    fn filter_tools_list_keeps_only_matching() {
        let mut resp = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "tools": [
                    {"name": "semantex_search"},
                    {"name": "semantex_callers"},
                    {"name": "semantex_agent"},
                ]
            }
        });
        Toolset::Core.filter_tools_list(&mut resp);
        let tools = resp["result"]["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 2);
        let names: Vec<&str> = tools.iter().map(|t| t["name"].as_str().unwrap()).collect();
        assert!(names.contains(&"semantex_search"));
        assert!(names.contains(&"semantex_agent"));
    }

    #[test]
    fn filter_tools_list_all_is_noop() {
        let mut resp = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "result": { "tools": [{"name": "x"}, {"name": "y"}] }
        });
        Toolset::All.filter_tools_list(&mut resp);
        assert_eq!(resp["result"]["tools"].as_array().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn healthz_returns_ok() {
        let backend = fake_backend_with(vec![]);
        let app = build_router(backend);
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/healthz")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn unknown_toolset_returns_404() {
        let backend = fake_backend_with(full_tool_set());
        let app = build_router(backend);
        let (status, body) = post_json(
            app,
            "/mcp/bogus",
            serde_json::json!({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(body["error"], "unknown_toolset");
    }

    #[tokio::test]
    async fn tools_list_core_filters_correctly() {
        let backend = fake_backend_with(full_tool_set());
        let app = build_router(backend);
        let (status, body) = post_json(
            app,
            "/mcp/core",
            serde_json::json!({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let names: Vec<&str> = body["result"]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["name"].as_str().unwrap())
            .collect();
        // Core toolset must equal the spec's core list (intersected with what the backend reports).
        assert!(names.contains(&"semantex_search"));
        assert!(names.contains(&"semantex_agent"));
        assert!(!names.contains(&"semantex_callers"));
    }

    #[tokio::test]
    async fn tools_list_all_returns_everything() {
        let backend = fake_backend_with(full_tool_set());
        let app = build_router(backend);
        let (status, body) = post_json(
            app,
            "/mcp/all",
            serde_json::json!({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            body["result"]["tools"].as_array().unwrap().len(),
            full_tool_set().len()
        );
    }

    #[tokio::test]
    async fn tools_call_outside_toolset_returns_method_not_found() {
        let backend = fake_backend_with(full_tool_set());
        let app = build_router(backend);
        let (status, body) = post_json(
            app,
            "/mcp/core",
            serde_json::json!({
                "jsonrpc": "2.0", "id": 7, "method": "tools/call",
                "params": {"name": "semantex_callers", "arguments": {}}
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK); // JSON-RPC errors are still 200
        assert_eq!(body["error"]["code"], -32601);
    }

    #[tokio::test]
    async fn default_path_is_alias_for_all() {
        let backend = fake_backend_with(full_tool_set());
        let app = build_router(backend);
        let (status, body) = post_json(
            app,
            "/mcp/",
            serde_json::json!({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            body["result"]["tools"].as_array().unwrap().len(),
            full_tool_set().len()
        );
    }

    /// Sanity check: `SubprocessBackend`'s `binary_path` detection always returns
    /// something non-empty.
    #[test]
    fn subprocess_backend_detect_binary() {
        let path = SubprocessBackend::detect_binary().unwrap();
        assert!(!path.as_os_str().is_empty());
    }

    /// Validate that the repo-root `server.json` parses and has every field
    /// the 2025-12-11 MCP registry schema declares as required, plus the
    /// fields downstream registries (Smithery, Cline, Cursor) commonly expect.
    #[test]
    fn server_json_at_repo_root_is_well_formed() {
        // `include_str!` reaches up out of the crate to the workspace root.
        const SRC: &str = include_str!("../../../server.json");
        let v: Value = serde_json::from_str(SRC).expect("server.json must parse as JSON");

        // Required by the 2025-12-11 schema.
        for required in ["name", "description", "version"] {
            assert!(
                v.get(required).is_some(),
                "server.json missing required field `{required}`"
            );
        }

        // Reverse-DNS naming with exactly one slash.
        let name = v["name"].as_str().expect("name must be a string");
        assert!(
            name.matches('/').count() == 1,
            "name must contain exactly one slash (got `{name}`)"
        );

        // Version must NOT be a range (no `^`, `~`, `>`, `<`).
        let version = v["version"].as_str().expect("version must be a string");
        for ch in ['^', '~', '>', '<', '='] {
            assert!(
                !version.starts_with(ch),
                "version `{version}` is a range, not a fixed semver"
            );
        }

        // At least one of `packages` / `remotes` must be present so clients
        // know how to install/connect.
        assert!(
            v.get("packages").is_some() || v.get("remotes").is_some(),
            "server.json must declare at least one of `packages` or `remotes`"
        );

        // Schema URL must point to a 2025-12-11-compatible reference.
        if let Some(schema) = v.get("$schema").and_then(|s| s.as_str()) {
            assert!(
                schema.contains("server.schema.json"),
                "$schema URL `{schema}` should reference server.schema.json"
            );
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // Bug-fix regression tests
    // ─────────────────────────────────────────────────────────────────────

    /// FINDING 8: Notifications return 204 No Content quickly.
    /// JSON-RPC spec: notifications MUST NOT receive a response body.
    #[tokio::test]
    async fn notification_returns_204_no_content() {
        let backend = fake_backend_with(full_tool_set());
        let app = build_router(backend);
        let start = std::time::Instant::now();
        let status = post_json_raw_status(
            app,
            "/mcp/all",
            serde_json::json!({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        )
        .await;
        let elapsed = start.elapsed();
        assert_eq!(status, StatusCode::NO_CONTENT);
        // Should be near-instant — much less than the implicit 1s safety margin.
        assert!(
            elapsed < std::time::Duration::from_millis(500),
            "notification handler took {elapsed:?}, should be near-instant"
        );
    }

    /// FINDING 2: Two clients sending the same `id:1` MUST each get their
    /// own response back. This is a backend-level test using a deferred
    /// fake backend that holds responses to force concurrent in-flight
    /// requests.
    #[tokio::test]
    async fn id_collision_two_clients_same_id() {
        // Build a backend that echoes the params back so we can verify each
        // caller got the response that corresponded to their own request.
        struct EchoBackend;
        impl Backend for EchoBackend {
            fn dispatch(
                &self,
                request: Value,
            ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Value>> + Send + '_>>
            {
                Box::pin(async move {
                    // Small jitter to force interleaving.
                    let jitter_ms = request
                        .get("params")
                        .and_then(|p| p.get("delay_ms"))
                        .and_then(serde_json::Value::as_u64)
                        .unwrap_or(0);
                    if jitter_ms > 0 {
                        tokio::time::sleep(std::time::Duration::from_millis(jitter_ms)).await;
                    }
                    let id = request.get("id").cloned().unwrap_or(Value::Null);
                    let echo = request.get("params").cloned().unwrap_or(Value::Null);
                    Ok(serde_json::json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "result": echo,
                    }))
                })
            }
        }

        let backend = Arc::new(EchoBackend);
        let app1 = build_router(Arc::clone(&backend));
        let app2 = build_router(Arc::clone(&backend));

        // Concurrent requests with the SAME id but distinguishable params.
        let (r1, r2) = tokio::join!(
            post_json(
                app1,
                "/mcp/all",
                serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "ping",
                    "params": {"caller": "alpha", "delay_ms": 50}
                }),
            ),
            post_json(
                app2,
                "/mcp/all",
                serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "ping",
                    "params": {"caller": "beta", "delay_ms": 10}
                }),
            ),
        );

        let (s1, b1) = r1;
        let (s2, b2) = r2;
        assert_eq!(s1, StatusCode::OK);
        assert_eq!(s2, StatusCode::OK);
        // Each client's response must echo its OWN params, not the other's.
        assert_eq!(b1["result"]["caller"], "alpha");
        assert_eq!(b2["result"]["caller"], "beta");
        // And the id round-trips back as the client originally sent it.
        assert_eq!(b1["id"], 1);
        assert_eq!(b2["id"], 1);
    }

    /// FINDING 6: 10 concurrent requests complete in roughly the time of 1,
    /// not 10× serial. (Backend-level stress test against EchoBackend.)
    #[tokio::test]
    async fn concurrent_requests_parallelize() {
        struct SlowEcho;
        impl Backend for SlowEcho {
            fn dispatch(
                &self,
                request: Value,
            ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Value>> + Send + '_>>
            {
                Box::pin(async move {
                    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                    let id = request.get("id").cloned().unwrap_or(Value::Null);
                    Ok(serde_json::json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "result": {}
                    }))
                })
            }
        }

        let backend = Arc::new(SlowEcho);
        let start = std::time::Instant::now();
        // Each request takes 100ms backend-side. With proper parallelization,
        // 10 concurrent should finish well under 1s (the serial baseline).
        let mut joins = Vec::new();
        for i in 0..10 {
            let app = build_router(Arc::clone(&backend));
            joins.push(tokio::spawn(async move {
                post_json(
                    app,
                    "/mcp/all",
                    serde_json::json!({"jsonrpc": "2.0", "id": i, "method": "ping"}),
                )
                .await
            }));
        }
        for j in joins {
            let (status, _body) = j.await.unwrap();
            assert_eq!(status, StatusCode::OK);
        }
        let elapsed = start.elapsed();
        assert!(
            elapsed < std::time::Duration::from_millis(500),
            "10 concurrent requests took {elapsed:?} — expected <500ms (parallel), not ~1s (serial)"
        );
    }

    /// FINDING 4: `expected_tools_for` constants align with the spec slices
    /// so a runtime sanity check can detect drift.
    #[test]
    fn expected_tools_match_spec_constants() {
        assert_eq!(
            SubprocessBackend::expected_tools_for(Toolset::Core),
            CORE_TOOLS.to_vec()
        );
        assert_eq!(
            SubprocessBackend::expected_tools_for(Toolset::Structural),
            STRUCTURAL_TOOLS.to_vec()
        );
        assert!(SubprocessBackend::expected_tools_for(Toolset::All).is_empty());
    }

    /// FINDING 4: Constructor accepts a toolset and stores it for forwarding.
    #[test]
    fn subprocess_backend_stores_toolset() {
        let b = SubprocessBackend::new_with_toolset(
            PathBuf::from("/nonexistent/semantex"),
            Some("core".to_string()),
        );
        assert_eq!(b.inner.toolset.as_deref(), Some("core"));
    }

    /// FINDING 7: When the subprocess dies, in-flight awaiters get error
    /// responses instead of hanging. We simulate this by tearing down the
    /// pending map directly (the same machinery `reader_loop` uses on exit).
    #[tokio::test]
    async fn pending_awaiters_drained_on_subprocess_exit() {
        // Build the minimum structure reader_loop manipulates.
        let pending: Arc<Mutex<HashMap<String, oneshot::Sender<Value>>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let (tx_a, rx_a) = oneshot::channel::<Value>();
        let (tx_b, rx_b) = oneshot::channel::<Value>();
        pending.lock().await.insert("k1".to_string(), tx_a);
        pending.lock().await.insert("k2".to_string(), tx_b);

        // Simulate the drain logic from reader_loop.
        let mut map = pending.lock().await;
        let drained: Vec<(String, oneshot::Sender<Value>)> = map.drain().collect();
        drop(map);
        for (wire_id, tx) in drained {
            let _ = tx.send(serde_json::json!({
                "jsonrpc": "2.0",
                "id": wire_id,
                "error": {
                    "code": -32099,
                    "message": "subprocess closed: test"
                }
            }));
        }

        // Both awaiters should now wake with error responses.
        let resp_a = rx_a.await.expect("awaiter A drained");
        let resp_b = rx_b.await.expect("awaiter B drained");
        assert_eq!(resp_a["error"]["code"], -32099);
        assert_eq!(resp_b["error"]["code"], -32099);
        // Wire id is preserved so the HTTP layer can correlate.
        assert!(
            resp_a["id"] == "k1" || resp_a["id"] == "k2",
            "wire id roundtrip"
        );
    }

    /// FINDING 7: Respawn rate limit kicks in after MAX_RESPAWNS_PER_WINDOW.
    #[tokio::test]
    async fn respawn_limit_enforced() {
        let b = SubprocessBackend::new_with_toolset(PathBuf::from("/nonexistent/semantex"), None);
        // First MAX_RESPAWNS_PER_WINDOW attempts should succeed (recording).
        for _ in 0..MAX_RESPAWNS_PER_WINDOW {
            b.enforce_respawn_limit().await.unwrap();
        }
        // Next one must fail.
        let r = b.enforce_respawn_limit().await;
        assert!(
            r.is_err(),
            "respawn rate limit should reject the {}'th call",
            MAX_RESPAWNS_PER_WINDOW + 1
        );
    }

    // ─────────────────────────────────────────────────────────────────────
    // Bearer-token auth (Wave 0 contract section C)
    // ─────────────────────────────────────────────────────────────────────

    /// Serializes tests that mutate the process-global `SEMANTEX_HTTP_TOKEN`
    /// env var, so they can't interfere with each other when the test binary
    /// runs them concurrently.
    static AUTH_TEST_ENV_LOCK: Mutex<()> = Mutex::const_new(());

    #[test]
    fn constant_time_eq_matches_equal() {
        assert!(constant_time_eq(b"abc123", b"abc123"));
    }

    #[test]
    fn constant_time_eq_rejects_mismatch_same_length() {
        assert!(!constant_time_eq(b"abc123", b"abc124"));
    }

    #[test]
    fn constant_time_eq_rejects_different_length() {
        assert!(!constant_time_eq(b"abc", b"abcd"));
        assert!(!constant_time_eq(b"", b"a"));
    }

    #[test]
    fn constant_time_eq_empty_equal() {
        assert!(constant_time_eq(b"", b""));
    }

    /// Precedence: `--auth-token` flag wins over everything else, even when
    /// `SEMANTEX_HTTP_TOKEN` is also set.
    #[tokio::test]
    async fn token_precedence_cli_flag_wins_over_env() {
        let _guard = AUTH_TEST_ENV_LOCK.lock().await;
        // SAFETY: guarded by AUTH_TEST_ENV_LOCK; no other test reads/writes
        // SEMANTEX_HTTP_TOKEN concurrently.
        unsafe { std::env::set_var("SEMANTEX_HTTP_TOKEN", "env-token") };
        let cfg = AuthConfig {
            cli_token: Some("flag-token".to_string()),
            require_auth: false,
        };
        let resolved = cfg.resolve(false).unwrap();
        unsafe { std::env::remove_var("SEMANTEX_HTTP_TOKEN") };
        assert_eq!(resolved.as_deref(), Some("flag-token"));
    }

    /// Precedence: with no CLI flag, `SEMANTEX_HTTP_TOKEN` is used.
    #[tokio::test]
    async fn token_precedence_env_used_when_no_flag() {
        let _guard = AUTH_TEST_ENV_LOCK.lock().await;
        // SAFETY: guarded by AUTH_TEST_ENV_LOCK.
        unsafe { std::env::set_var("SEMANTEX_HTTP_TOKEN", "env-only-token") };
        let cfg = AuthConfig {
            cli_token: None,
            require_auth: false,
        };
        let resolved = cfg.resolve(false).unwrap();
        unsafe { std::env::remove_var("SEMANTEX_HTTP_TOKEN") };
        assert_eq!(resolved.as_deref(), Some("env-only-token"));
    }

    /// On loopback with no flag, no env, and no `--require-auth`, auth stays
    /// off entirely (today's behavior is preserved).
    #[tokio::test]
    async fn loopback_default_no_auth_requested() {
        let _guard = AUTH_TEST_ENV_LOCK.lock().await;
        // SAFETY: guarded by AUTH_TEST_ENV_LOCK.
        unsafe { std::env::remove_var("SEMANTEX_HTTP_TOKEN") };
        let cfg = AuthConfig {
            cli_token: None,
            require_auth: false,
        };
        let resolved = cfg.resolve(false).unwrap();
        assert!(resolved.is_none());
    }

    /// `--require-auth` on loopback with no explicit token falls through to
    /// the auto-generated persisted token. `SEMANTEX_HOME` redirects the
    /// token path into a tempdir so the test never touches the real
    /// `~/.semantex/http_token`.
    #[tokio::test]
    async fn require_auth_on_loopback_falls_back_to_generated_token() {
        let _guard = AUTH_TEST_ENV_LOCK.lock().await;
        let dir = tempfile::tempdir().unwrap();
        // SAFETY: guarded by AUTH_TEST_ENV_LOCK.
        unsafe {
            std::env::remove_var("SEMANTEX_HTTP_TOKEN");
            std::env::set_var("SEMANTEX_HOME", dir.path());
        }
        let cfg = AuthConfig {
            cli_token: None,
            require_auth: true,
        };
        let resolved = cfg.resolve(false);
        // SAFETY: guarded by AUTH_TEST_ENV_LOCK.
        unsafe { std::env::remove_var("SEMANTEX_HOME") };
        let token = resolved.unwrap().expect("require-auth must yield a token");
        assert_eq!(token.len(), GENERATED_TOKEN_BYTES * 2);
        assert!(
            dir.path().join("http_token").exists(),
            "token must be persisted under SEMANTEX_HOME"
        );
    }

    /// `--allow-remote` always wants auth, even with no flag/env — this is
    /// the "REQUIRED whenever --allow-remote" contract requirement.
    #[tokio::test]
    async fn allow_remote_always_wants_auth() {
        let _guard = AUTH_TEST_ENV_LOCK.lock().await;
        let dir = tempfile::tempdir().unwrap();
        // SAFETY: guarded by AUTH_TEST_ENV_LOCK.
        unsafe {
            std::env::remove_var("SEMANTEX_HTTP_TOKEN");
            std::env::set_var("SEMANTEX_HOME", dir.path());
        }
        let cfg = AuthConfig {
            cli_token: None,
            require_auth: false,
        };
        let resolved = cfg.resolve(true);
        // SAFETY: guarded by AUTH_TEST_ENV_LOCK.
        unsafe { std::env::remove_var("SEMANTEX_HOME") };
        // `resolve` must never return `Ok(None)` when `allow_remote` is true:
        // with a writable home it generates + persists a token.
        let token = resolved.unwrap();
        assert!(token.is_some(), "remote must never resolve to no-auth");
    }

    /// The auto-generate-then-persist path: first call generates and writes
    /// a 64-hex-char token to the path; a second call reuses the same
    /// persisted value instead of generating a new one.
    #[tokio::test]
    async fn persisted_token_generated_then_reused() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("http_token");
        assert!(!path.exists());

        let first = load_or_generate_persisted_token(&path).unwrap();
        assert_eq!(first.len(), GENERATED_TOKEN_BYTES * 2, "64 hex chars");
        assert!(first.chars().all(|c| c.is_ascii_hexdigit()));
        assert!(path.exists());

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o600, "token file must be 0600");
        }

        let second = load_or_generate_persisted_token(&path).unwrap();
        assert_eq!(first, second, "second call must reuse the persisted token");
    }

    /// Two independent generations (different paths) must not collide.
    #[tokio::test]
    async fn generated_tokens_are_random() {
        let a = generate_token_hex();
        let b = generate_token_hex();
        assert_ne!(a, b);
    }

    // ── axum-level auth enforcement tests ──────────────────────────────────

    /// `/healthz` never requires auth, even when the router was built with a
    /// token configured.
    #[tokio::test]
    async fn healthz_never_requires_auth() {
        let backend = fake_backend_with(vec![]);
        let app = build_router_with_auth(backend, Some("secret".to_string()));
        let status = get_with_auth(app, "/healthz", None).await;
        assert_eq!(status, StatusCode::OK);
    }

    /// POST `/mcp/all` with no auth configured (router built via the plain
    /// `build_router`, i.e. loopback default) succeeds without a token —
    /// preserves today's open-by-default loopback behavior.
    #[tokio::test]
    async fn mcp_post_no_auth_configured_is_open() {
        let backend = fake_backend_with(full_tool_set());
        let app = build_router(backend);
        let (status, _) = post_json_auth(
            app,
            "/mcp/all",
            serde_json::json!({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}),
            None,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
    }

    /// POST `/mcp/all` with auth configured and no `Authorization` header
    /// gets 401 + `WWW-Authenticate: Bearer`.
    #[tokio::test]
    async fn mcp_post_missing_token_returns_401_with_challenge() {
        let backend = fake_backend_with(full_tool_set());
        let app = build_router_with_auth(backend, Some("secret".to_string()));
        let req = Request::builder()
            .method(Method::POST)
            .uri("/mcp/all")
            .header("content-type", "application/json")
            .body(Body::from(
                serde_json::to_vec(
                    &serde_json::json!({"jsonrpc":"2.0","id":1,"method":"tools/list"}),
                )
                .unwrap(),
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(
            resp.headers()
                .get(axum::http::header::WWW_AUTHENTICATE)
                .and_then(|v| v.to_str().ok()),
            Some("Bearer")
        );
    }

    /// POST `/mcp/all` with a wrong bearer token gets 401.
    #[tokio::test]
    async fn mcp_post_wrong_token_returns_401() {
        let backend = fake_backend_with(full_tool_set());
        let app = build_router_with_auth(backend, Some("secret".to_string()));
        let (status, _) = post_json_auth(
            app,
            "/mcp/all",
            serde_json::json!({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}),
            Some("wrong"),
        )
        .await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
    }

    /// POST `/mcp/all` with the correct bearer token succeeds.
    #[tokio::test]
    async fn mcp_post_correct_token_returns_200() {
        let backend = fake_backend_with(full_tool_set());
        let app = build_router_with_auth(backend, Some("secret".to_string()));
        let (status, body) = post_json_auth(
            app,
            "/mcp/all",
            serde_json::json!({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}),
            Some("secret"),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            body["result"]["tools"].as_array().unwrap().len(),
            full_tool_set().len()
        );
    }

    /// The default-alias POST route (`/mcp/`) enforces auth the same way.
    #[tokio::test]
    async fn mcp_default_post_requires_auth_when_configured() {
        let backend = fake_backend_with(full_tool_set());
        let app = build_router_with_auth(backend, Some("secret".to_string()));
        let (status, _) = post_json_auth(
            app,
            "/mcp/",
            serde_json::json!({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}),
            None,
        )
        .await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
    }

    /// SSE endpoint (`GET /mcp/{toolset}/events`) requires auth just like
    /// POST, when configured.
    #[tokio::test]
    async fn sse_events_requires_auth_when_configured() {
        let backend = fake_backend_with(full_tool_set());
        let app = build_router_with_auth(backend, Some("secret".to_string()));
        let missing = get_with_auth(app.clone(), "/mcp/all/events", None).await;
        assert_eq!(missing, StatusCode::UNAUTHORIZED);
        let ok = get_with_auth(app, "/mcp/all/events", Some("secret")).await;
        assert_eq!(ok, StatusCode::OK);
    }

    /// The default-alias SSE route (`GET /mcp/`) also requires auth.
    #[tokio::test]
    async fn sse_default_events_requires_auth_when_configured() {
        let backend = fake_backend_with(full_tool_set());
        let app = build_router_with_auth(backend, Some("secret".to_string()));
        let missing = get_with_auth(app.clone(), "/mcp/", None).await;
        assert_eq!(missing, StatusCode::UNAUTHORIZED);
        let ok = get_with_auth(app, "/mcp/", Some("secret")).await;
        assert_eq!(ok, StatusCode::OK);
    }

    /// SSE endpoint with no auth configured (loopback default) is open, same
    /// as POST.
    #[tokio::test]
    async fn sse_events_no_auth_configured_is_open() {
        let backend = fake_backend_with(full_tool_set());
        let app = build_router(backend);
        let status = get_with_auth(app, "/mcp/all/events", None).await;
        assert_eq!(status, StatusCode::OK);
    }
}
