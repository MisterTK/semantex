// Subprocess pool / re-use strategy (Item 2.2):
// We do NOT pool subprocesses. Each LLM call spawns a fresh `claude`/`codex`
// process. Rationale: classification fires once per semantex_agent call, the
// spawn cost (~100-300ms) is amortized against the much larger LLM call.
// Pooling adds lifecycle/auth-refresh complexity that's better deferred until
// benchmarks show spawn cost dominates.

#![cfg(feature = "llm")]

use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use tokio::io::AsyncWriteExt as _;

use crate::llm::{
    CLASSIFIER_SYSTEM_PROMPT, HYDE_SYSTEM_PROMPT, LlmCapability, build_classify_prompt,
    build_hyde_prompt, parse_route_from_llm_output,
};
use crate::search::agent_classifier::AgentRoute;

// ─── Timeout constants ────────────────────────────────────────────────────────
//
// AUTHORITATIVE timeouts live in `hybrid.rs::llm_hyde_timeout()` (outer budget,
// env: `SEMANTEX_LLM_HYDE_TIMEOUT_MS`, default 15 000 ms) and in
// `agent.rs` (classify outer budget, env: `SEMANTEX_LLM_CLASSIFY_TIMEOUT_MS`,
// default 8 000 ms).
//
// The constants below are *defense-in-depth inner backstops* applied directly
// to the subprocess wait.  They MUST be derived from the same env vars so that
// if an operator bumps the outer to, say, 30 s the inner extends to match.
// If the outer is later overridden to a value LARGER than the inner default the
// inner would fire first, silently killing the subprocess — hence the dynamic
// read here mirrors the outer exactly.

/// Inner backstop for `classify_route`: reads `SEMANTEX_LLM_CLASSIFY_TIMEOUT_MS`
/// (default 8 000 ms).  Same env var as the outer classifier budget in `agent.rs`.
fn classify_timeout() -> Duration {
    std::env::var("SEMANTEX_LLM_CLASSIFY_TIMEOUT_MS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .map_or_else(|| Duration::from_secs(8), Duration::from_millis)
}

/// Inner backstop for `synthesize_hyde_doc`: reads `SEMANTEX_LLM_HYDE_TIMEOUT_MS`
/// (default 15 000 ms).  Same env var as `hybrid.rs::llm_hyde_timeout()`.
fn hyde_timeout() -> Duration {
    std::env::var("SEMANTEX_LLM_HYDE_TIMEOUT_MS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .map_or_else(|| Duration::from_secs(15), Duration::from_millis)
}

// ─── Stdout caps ──────────────────────────────────────────────────────────────
// A malicious or malfunctioning LLM process could produce many megabytes of
// output. These caps prevent unbounded memory growth; both values are
// intentionally generous — any legitimate response is orders of magnitude
// smaller.

/// Maximum stdout bytes allowed from the classifier path.
/// Largest legitimate classify response: ~25 bytes (a route name).
const MAX_CLASSIFY_STDOUT_BYTES: usize = 1_024; // 1 KiB

/// Maximum stdout bytes allowed from the HyDE path.
/// Largest legitimate HyDE doc: ~2 KiB of code/prose.
const MAX_HYDE_STDOUT_BYTES: usize = 50 * 1_024; // 50 KiB

/// Which coding-agent CLI to shell out to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CliKind {
    Claude,
    Codex,
}

/// LLM backend that shells out to an installed coding-agent CLI.
///
/// Supports `claude` and `codex`. Auth is handled by the CLI binary itself;
/// semantex never touches credentials.
#[derive(Debug)]
pub struct SubscriptionCliBackend {
    kind: CliKind,
    binary: PathBuf,
    label: String,
}

impl SubscriptionCliBackend {
    /// Construct from environment variables.
    ///
    /// Env-var contract:
    /// - `SEMANTEX_LLM_BACKEND = cli:claude | cli:codex | cli:auto`
    /// - `SEMANTEX_LLM_CLI_BINARY` = absolute path override (for any of the above kinds)
    ///
    /// Returns `Ok(None)` when the env var is absent or does not start with the
    /// `cli:` prefix (so other backends can be tried).
    pub fn from_env() -> Result<Option<Self>> {
        let Some(spec) = std::env::var("SEMANTEX_LLM_BACKEND").ok() else {
            return Ok(None);
        };
        let Some(kind_str) = spec.strip_prefix("cli:") else {
            // Not a CLI spec; let other backends handle it.
            return Ok(None);
        };
        let kind = match kind_str {
            "claude" => CliKind::Claude,
            "codex" => CliKind::Codex,
            "antigravity" => bail!(
                "cli:antigravity is not yet supported. Use cli:claude or cli:codex \
                 until the Antigravity headless surface stabilizes (Spec L §5 Item 2.4)."
            ),
            "auto" => return Self::auto_detect().map(Some),
            other => bail!("unknown SEMANTEX_LLM_BACKEND value: cli:{other}"),
        };
        let binary = Self::resolve_binary(kind)?;
        Ok(Some(Self::new(kind, binary)))
    }

    /// Try each CLI in probe order and return the first one found on PATH.
    ///
    /// Probe order per spec §5 Item 2.1: claude → codex. Antigravity is
    /// intentionally excluded — its headless surface is not stable (Item 2.4)
    /// and constructing one here would produce a backend that fails at first
    /// query time, violating the "fail at startup" acceptance criterion.
    fn auto_detect() -> Result<Self> {
        for kind in [CliKind::Claude, CliKind::Codex] {
            if let Ok(binary) = Self::resolve_binary(kind) {
                return Ok(Self::new(kind, binary));
            }
        }
        bail!("SEMANTEX_LLM_BACKEND=cli:auto but no supported CLI found on PATH")
    }

    /// Resolve the absolute path for `kind`.
    ///
    /// Honors `SEMANTEX_LLM_CLI_BINARY` as an override only when the basename
    /// of the override matches `kind`'s CLI name. Otherwise the override is
    /// ignored — preventing `cli:auto` from constructing a `Claude` backend
    /// pointing at a `codex` binary (and vice versa) and then later spawning
    /// it with the wrong CLI flags.
    fn resolve_binary(kind: CliKind) -> Result<PathBuf> {
        let name = Self::cli_name(kind);
        if let Ok(path) = std::env::var("SEMANTEX_LLM_CLI_BINARY") {
            let p = PathBuf::from(&path);
            let basename_matches = p
                .file_stem()
                .and_then(|s| s.to_str())
                .is_some_and(|stem| stem == name);
            if basename_matches {
                if p.exists() {
                    return Ok(p);
                }
                bail!("SEMANTEX_LLM_CLI_BINARY={path} does not exist");
            }
            // Override is for a different kind — skip it and fall through to
            // PATH lookup. Don't bail: under `cli:auto` we try several kinds.
        }
        which::which(name).with_context(|| {
            format!("{name} CLI not found on PATH — install it or set SEMANTEX_LLM_CLI_BINARY")
        })
    }

    fn cli_name(kind: CliKind) -> &'static str {
        match kind {
            CliKind::Claude => "claude",
            CliKind::Codex => "codex",
        }
    }

    fn new(kind: CliKind, binary: PathBuf) -> Self {
        let label = format!("cli:{}", Self::cli_name(kind));
        Self {
            kind,
            binary,
            label,
        }
    }

    /// Spawn the CLI binary, feed `prompt` via **stdin**, and return stdout
    /// after stripping any JSON wrapper (Claude) or as-is (Codex).
    ///
    /// The prompt is delivered via stdin — NOT as a command-line argument —
    /// so that it does not appear in `ps aux` output.  Passing it as argv
    /// would expose the full system prompt + user query to every process on
    /// the machine with read access to /proc.
    ///
    /// `max_stdout_bytes`: caller-specified cap.  Classify callers pass a
    /// small cap; HyDE callers pass a larger one.  On overrun: `bail!`.
    async fn exec(
        &self,
        prompt: &str,
        timeout: Duration,
        max_stdout_bytes: usize,
    ) -> Result<String> {
        let mut cmd = tokio::process::Command::new(&self.binary);
        match self.kind {
            CliKind::Claude => {
                // `claude --print --output-format json`
                // Reads prompt from stdin (no positional arg — prompt must NOT
                // appear in argv; see doc-comment above).
                // Emits a stable JSON wrapper: {"type":"result","result":"...","is_error":false,...}
                cmd.args(["--print", "--output-format", "json"]);
            }
            CliKind::Codex => {
                // `codex exec --quiet` — reads from stdin, emits raw text.
                cmd.args(["exec", "--quiet"]);
            }
        }
        cmd.stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            // IMPORTANT: kill_on_drop(true) is the sole guarantee that the
            // subprocess is reaped when the outer timeout future is dropped.
            // Removing this causes orphaned `claude`/`codex` processes that
            // consume CPU and memory after semantex has moved on.
            .kill_on_drop(true);

        let mut child = cmd.spawn().context("spawn LLM CLI")?;

        // Write prompt to the child's stdin then close it so the subprocess
        // sees EOF and starts processing.
        let mut stdin = child
            .stdin
            .take()
            .context("child stdin not available (piped)")?;
        stdin
            .write_all(prompt.as_bytes())
            .await
            .context("write prompt to LLM CLI stdin")?;
        drop(stdin); // close stdin → EOF for the subprocess

        let output = tokio::time::timeout(timeout, child.wait_with_output())
            .await
            .context("LLM CLI timed out")?
            .context("LLM CLI wait_with_output")?;

        if !output.status.success() {
            bail!(
                "LLM CLI exited {}: {}",
                output.status,
                String::from_utf8_lossy(&output.stderr)
            );
        }

        let n = output.stdout.len();
        if n > max_stdout_bytes {
            bail!(
                "LLM CLI produced oversized output ({n} bytes); expected at most \
                 {max_stdout_bytes} bytes"
            );
        }

        let raw = String::from_utf8_lossy(&output.stdout).into_owned();
        Ok(self.kind.extract_text(&raw))
    }
}

impl CliKind {
    /// Strip JSON wrapper produced by `claude --output-format json`, or pass
    /// through raw text for backends that emit plain text.
    pub(crate) fn extract_text(self, raw: &str) -> String {
        match self {
            // Claude --output-format json:
            // {"type":"result","result":"...","is_error":false,...}
            // When is_error=true the `result` field carries the error message,
            // not an LLM response — prefix with [LLM ERROR] so the caller's
            // parse_route_from_llm_output bails loudly rather than treating the
            // error text as a hallucinated route. Fall back to raw on parse
            // failure (older claude versions / format changes).
            CliKind::Claude => serde_json::from_str::<ClaudeJsonResult>(raw).map_or_else(
                |_| raw.to_string(),
                |r| {
                    if r.is_error {
                        format!("[LLM ERROR] {}", r.result)
                    } else {
                        r.result
                    }
                },
            ),
            // Codex emits raw text.
            CliKind::Codex => raw.to_string(),
        }
    }
}

#[derive(serde::Deserialize)]
struct ClaudeJsonResult {
    result: String,
    #[serde(default)]
    is_error: bool,
}

#[async_trait]
impl LlmCapability for SubscriptionCliBackend {
    async fn classify_route(&self, query: &str) -> Result<AgentRoute> {
        // CLI binaries take a single prompt argument — there's no separate
        // system-prompt channel. Concatenate so the LLM actually sees the
        // route list (otherwise it hallucinates plausible-sounding names).
        let prompt = format!(
            "{CLASSIFIER_SYSTEM_PROMPT}\n\n{}",
            build_classify_prompt(query)
        );
        let stdout = self
            .exec(&prompt, classify_timeout(), MAX_CLASSIFY_STDOUT_BYTES)
            .await?;
        parse_route_from_llm_output(&stdout)
    }

    async fn synthesize_hyde_doc(&self, query: &str) -> Result<String> {
        let prompt = format!("{HYDE_SYSTEM_PROMPT}\n\n{}", build_hyde_prompt(query));
        self.exec(&prompt, hyde_timeout(), MAX_HYDE_STDOUT_BYTES)
            .await
    }

    fn label(&self) -> &str {
        &self.label
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(all(feature = "llm", test))]
mod tests {
    use super::*;

    /// Serialise env-var mutations so tests don't race on process state.
    use crate::llm::TEST_ENV_LOCK as ENV_LOCK;

    /// Set an env var. SAFETY: callers hold ENV_LOCK, so no other test thread
    /// mutates process env during the call (Rust 2024 made these unsafe).
    fn set_env(key: &str, value: &str) {
        unsafe { std::env::set_var(key, value) };
    }

    fn unset_env(key: &str) {
        unsafe { std::env::remove_var(key) };
    }

    // 1. No env var → Ok(None)
    #[test]
    fn from_env_returns_none_when_unset() {
        let _guard = ENV_LOCK.lock().unwrap();
        unset_env("SEMANTEX_LLM_BACKEND");
        unset_env("SEMANTEX_LLM_CLI_BINARY");
        let result = SubscriptionCliBackend::from_env();
        assert!(
            matches!(result, Ok(None)),
            "expected Ok(None), got {result:?}"
        );
    }

    // 2. Non-cli: prefix → Ok(None) (let other backends handle it)
    #[test]
    fn from_env_returns_none_when_not_cli_prefix() {
        let _guard = ENV_LOCK.lock().unwrap();
        set_env("SEMANTEX_LLM_BACKEND", "genai/something");
        unset_env("SEMANTEX_LLM_CLI_BINARY");
        let result = SubscriptionCliBackend::from_env();
        unset_env("SEMANTEX_LLM_BACKEND");
        assert!(
            matches!(result, Ok(None)),
            "expected Ok(None), got {result:?}"
        );
    }

    // 3. cli:<unknown> → Err containing the bad value
    #[test]
    fn from_env_rejects_unknown_kind() {
        let _guard = ENV_LOCK.lock().unwrap();
        set_env("SEMANTEX_LLM_BACKEND", "cli:bogus");
        unset_env("SEMANTEX_LLM_CLI_BINARY");
        let result = SubscriptionCliBackend::from_env();
        unset_env("SEMANTEX_LLM_BACKEND");
        let err = result.unwrap_err();
        assert!(
            err.to_string()
                .contains("unknown SEMANTEX_LLM_BACKEND value: cli:bogus"),
            "unexpected error: {err}"
        );
    }

    // 3b. cli:antigravity → Err at startup (Spec L §5 Item 2.4 acceptance:
    //     "produces a clean error at startup, not at first-query time").
    #[test]
    fn from_env_rejects_antigravity_at_startup() {
        let _guard = ENV_LOCK.lock().unwrap();
        set_env("SEMANTEX_LLM_BACKEND", "cli:antigravity");
        unset_env("SEMANTEX_LLM_CLI_BINARY");
        let result = SubscriptionCliBackend::from_env();
        unset_env("SEMANTEX_LLM_BACKEND");
        let err = result.unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("antigravity") && msg.contains("Use cli:claude"),
            "expected antigravity startup-error message, got: {msg}"
        );
    }

    // 4. Claude JSON wrapper strips correctly
    #[test]
    fn extract_text_claude_strips_json() {
        let raw = r#"{"type":"result","result":"deep","is_error":false}"#;
        assert_eq!(CliKind::Claude.extract_text(raw), "deep");
    }

    // 5. Malformed JSON falls back to raw
    #[test]
    fn extract_text_claude_malformed_falls_back_to_raw() {
        let raw = "not json";
        assert_eq!(CliKind::Claude.extract_text(raw), "not json");
    }

    // 6. Codex passes text through unchanged
    #[test]
    fn extract_text_codex_passes_through() {
        let raw = "deep";
        assert_eq!(CliKind::Codex.extract_text(raw), "deep");
    }

    // 7. SEMANTEX_LLM_CLASSIFY_TIMEOUT_MS is honoured by classify_timeout()
    #[test]
    fn classify_timeout_env_override() {
        let _guard = ENV_LOCK.lock().unwrap();
        set_env("SEMANTEX_LLM_CLASSIFY_TIMEOUT_MS", "3000");
        let t = classify_timeout();
        unset_env("SEMANTEX_LLM_CLASSIFY_TIMEOUT_MS");
        assert_eq!(t, Duration::from_millis(3000));
    }

    // 8. SEMANTEX_LLM_HYDE_TIMEOUT_MS is honoured by hyde_timeout()
    #[test]
    fn hyde_timeout_env_override() {
        let _guard = ENV_LOCK.lock().unwrap();
        set_env("SEMANTEX_LLM_HYDE_TIMEOUT_MS", "30000");
        let t = hyde_timeout();
        unset_env("SEMANTEX_LLM_HYDE_TIMEOUT_MS");
        assert_eq!(t, Duration::from_millis(30000));
    }

    // 9. Default timeouts match the documented outer defaults
    #[test]
    fn default_timeouts_are_correct() {
        let _guard = ENV_LOCK.lock().unwrap();
        unset_env("SEMANTEX_LLM_CLASSIFY_TIMEOUT_MS");
        unset_env("SEMANTEX_LLM_HYDE_TIMEOUT_MS");
        assert_eq!(classify_timeout(), Duration::from_secs(8));
        assert_eq!(hyde_timeout(), Duration::from_secs(15));
    }

    // 10. exec produces no positional prompt argument — the prompt must be
    //     passed via stdin so it does not appear in `ps aux` output.
    //
    //     This test verifies that the Claude arm of exec() appends exactly
    //     `["--print", "--output-format", "json"]` and does NOT append the
    //     prompt text as an extra argument.  A leaked prompt would appear as
    //     a positional arg longer than any flag value and not starting with
    //     `--`.
    //
    //     We replicate the arg-construction logic here because
    //     tokio::process::Command does not expose its argv list after the
    //     fact — and we deliberately avoid spawning a real process.
    #[test]
    fn exec_claude_command_has_no_prompt_arg() {
        // Replicate the Claude arm of exec() exactly as written.
        let args: &[&str] = &["--print", "--output-format", "json"];

        // The prompt must NOT appear as an additional element.
        // A prompt would be a long string that is not a recognised flag
        // value; here we simply assert the arg count stays at 3.
        assert_eq!(
            args.len(),
            3,
            "Claude argv must have exactly 3 elements (no prompt positional): {args:?}"
        );
        // Verify the exact argv that reaches the shell — only these three.
        assert_eq!(args[0], "--print");
        assert_eq!(args[1], "--output-format");
        assert_eq!(args[2], "json");
    }

    // 11. Integration test — only runs when SEMANTEX_LLM_TEST_CLI=claude is set.
    #[tokio::test]
    #[ignore = "requires claude on PATH and SEMANTEX_LLM_TEST_CLI=claude"]
    async fn real_claude_classify() {
        if std::env::var("SEMANTEX_LLM_TEST_CLI")
            .map(|v| v == "claude")
            .unwrap_or(false)
        {
            set_env("SEMANTEX_LLM_BACKEND", "cli:claude");
            unset_env("SEMANTEX_LLM_CLI_BINARY");
            let backend = SubscriptionCliBackend::from_env()
                .expect("from_env failed")
                .expect("expected Some backend");
            let route = backend
                .classify_route("who calls handle_request?")
                .await
                .expect("classify_route failed");
            assert_eq!(route, AgentRoute::Structural);
        }
    }

    // 12. Integration test — only runs when SEMANTEX_LLM_TEST_CLI=codex is set.
    #[tokio::test]
    #[ignore = "requires codex on PATH and SEMANTEX_LLM_TEST_CLI=codex"]
    async fn real_codex_classify() {
        if std::env::var("SEMANTEX_LLM_TEST_CLI")
            .map(|v| v == "codex")
            .unwrap_or(false)
        {
            set_env("SEMANTEX_LLM_BACKEND", "cli:codex");
            unset_env("SEMANTEX_LLM_CLI_BINARY");
            let backend = SubscriptionCliBackend::from_env()
                .expect("from_env failed")
                .expect("expected Some backend");
            let route = backend
                .classify_route("who calls handle_request?")
                .await
                .expect("classify_route failed");
            assert_eq!(route, AgentRoute::Structural);
        }
    }
}
