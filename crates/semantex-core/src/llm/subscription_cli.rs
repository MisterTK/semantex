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

use crate::llm::{
    CLASSIFIER_SYSTEM_PROMPT, HYDE_SYSTEM_PROMPT, LlmCapability, build_classify_prompt,
    build_hyde_prompt, parse_route_from_llm_output,
};
use crate::search::agent_classifier::AgentRoute;

const CLASSIFY_TIMEOUT: Duration = Duration::from_secs(8);
const HYDE_TIMEOUT: Duration = Duration::from_secs(15);

/// Which coding-agent CLI to shell out to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CliKind {
    Claude,
    Codex,
    Antigravity,
}

/// LLM backend that shells out to an installed coding-agent CLI.
///
/// Supports `claude`, `codex`, and (stubbed) `antigravity`.  Auth is handled
/// by the CLI binary itself; semantex never touches credentials.
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
    /// - `SEMANTEX_LLM_BACKEND = cli:claude | cli:codex | cli:antigravity | cli:auto`
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
            CliKind::Antigravity => "antigravity",
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

    /// Spawn the CLI binary, feed `prompt` as the sole argument, and return
    /// stdout after stripping any JSON wrapper (Claude) or as-is (Codex).
    async fn exec(&self, prompt: &str, timeout: Duration) -> Result<String> {
        let mut cmd = tokio::process::Command::new(&self.binary);
        match self.kind {
            CliKind::Claude => {
                // `claude --print --output-format json <PROMPT>`
                // Emits a stable JSON wrapper: {"type":"result","result":"...","is_error":false,...}
                cmd.args(["--print", "--output-format", "json"]).arg(prompt);
            }
            CliKind::Codex => {
                // `codex exec --quiet <PROMPT>` — emits raw text.
                cmd.args(["exec", "--quiet"]).arg(prompt);
            }
            CliKind::Antigravity => {
                // Item 2.4: defense-in-depth — `from_env` already rejects this
                // kind at startup so an `Antigravity` variant should never
                // reach `exec`. Kept so any future internal construction path
                // still produces the documented error rather than spawning a
                // non-existent CLI.
                bail!(
                    "cli:antigravity is not yet supported. Use cli:claude or cli:codex \
                     until the Antigravity headless surface stabilizes (Spec L §5 Item 2.4)."
                );
            }
        }
        cmd.stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true);

        let child = cmd.spawn().context("spawn LLM CLI")?;
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
            // Codex emits raw text; Antigravity is rejected at from_env so
            // this arm is unreachable in practice but keeps the match total.
            CliKind::Codex | CliKind::Antigravity => raw.to_string(),
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
        let stdout = self.exec(&prompt, CLASSIFY_TIMEOUT).await?;
        parse_route_from_llm_output(&stdout)
    }

    async fn synthesize_hyde_doc(&self, query: &str) -> Result<String> {
        let prompt = format!("{HYDE_SYSTEM_PROMPT}\n\n{}", build_hyde_prompt(query));
        self.exec(&prompt, HYDE_TIMEOUT).await
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

    // 7. Integration test — only runs when SEMANTEX_LLM_TEST_CLI=claude is set.
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

    // 8. Integration test — only runs when SEMANTEX_LLM_TEST_CLI=codex is set.
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
