# semantex LLM Integration — Phase 1 (genai) + Phase 2 (SubscriptionCli)

**Date:** 2026-05-27
**Status:** SHIPPED 2026-05-27. v0.7 (Phase 1) and v0.8 (Phase 2) landed together in merge `7eda12f`; post-tag fixes through `b14bca3`. HyDE call-site gap being patched in v0.7.1 by Team A — see §15.
**Target releases:** v0.7 (Phase 1, genai adapter) ✅ → v0.8 (Phase 2, SubscriptionCli backend) ✅ → v0.7.1 (HyDE call-site, in progress)
**Repository:** https://github.com/MisterTK/semantex
**Main branch:** `main` at `b14bca3` (post-/simplify sweep) — see §15 for post-ship history.
**Sole authoritative source:** this file. Subagents execute against this spec without further consultation. Cross-spec coordination: defers to `docs/RELEASE-SEQUENCE-2026-05.md` for ordering vs. any other v0.7+ work.
**Supersedes:** the LLM portions (Items 9 and 12) of `docs/superpowers/specs/2026-05-26-semantex-v0.3.1-v0.5-refactor.md`, which are marked out-of-scope as of 2026-05-27.

---

## 0. How to use this spec

You are a subagent assigned a workstream from §10. Read §1 (motivation), §2 (current state), §3-§5 (your phase), §6 (acceptance gates), §10 (your owned files). Skip §11 unless your workstream needs to coordinate with another.

Each numbered work item is self-contained:
- **Goal** — what to build, why
- **Files** — exact paths to modify or create, with exclusive-write boundaries
- **Desired behavior** — what to change to
- **Acceptance** — measurable criteria, including unit tests

Do not touch files outside your declared ownership. If you find you need to, open `coordination_request.md` at the repo root instead of editing.

---

## 1. Executive summary

semantex v0.6 ships without any LLM integration. v0.6 deliberately *removed* the ONNX-based local-LLM scaffold (`crates/semantex-core/src/llm/`, `classify_with_llm`, `hybrid_search_with_hyde`) on the basis that:
1. Bundling an LLM contradicts the "no LLMs in semantex directly" design constraint.
2. MCP `sampling/createMessage` is deprecated as of MCP `2026-07-28` (SEP-2577); the protocol team's guidance is "integrate with LLM provider APIs directly."
3. A clean OSS adapter (`rust-genai`) covers both airgap (via Ollama) and cloud (via Anthropic/OpenAI/Google APIs) without us re-implementing provider-specific code.

This spec restores two LLM-driven capabilities — **LLM-driven query classification** and **HyDE (Hypothetical Document Embedding) retrieval** — via a thin pluggable backend abstraction. Two phases:

- **Phase 1 (v0.7) — `genai` adapter.** Wraps `jeremychone/rust-genai` v0.5.2+. Supports 25+ providers (Anthropic, OpenAI, Gemini, Ollama, OpenRouter, xAI, Groq, Cohere, Bedrock, etc.) through `genai`'s unified `Adapter` trait. Configured via env vars; per-token billing or local Ollama for airgap. Single Cargo feature `llm`.
- **Phase 2 (v0.8) — `SubscriptionCli` backend.** Shells out to the user's installed coding-agent CLI (`claude`, `codex`, optionally `antigravity`) for LLM calls. Lets users with Claude Pro/Max, ChatGPT Plus/Pro, or Google AI Pro/Ultra subscriptions use those quotas without supplying an API key.

Both phases share the same `LlmBackend` enum surface. Phase 1 ships first; Phase 2 adds variants without changing existing call sites.

### Non-goals

- No model bundling. semantex never ships an ONNX/GGUF model file.
- No embedded inference. We don't depend on `mistral.rs`, `candle`, or `ort` for LLM (the existing `ort` for ColBERT stays — that's a separate concern).
- No re-implementation of OAuth flows for the Subscription CLIs. We shell out to the CLI binary; the CLI handles its own auth.
- No MCP `sampling/createMessage`. Deprecated path, do not adopt.
- No Tier 4 "ship pre-built indexes" / "web UI" — those remain in the Spec Q backlog and are unrelated.

---

## 2. Current state (post-ship, main at `b14bca3`)

**As of 2026-05-27, this spec is fully shipped.** The section below describes what now exists in `main`; it is no longer a "planned" description. See §15 for post-ship learnings and known gaps.

### What shipped (verified against `main` at `b14bca3`)

- **`crates/semantex-core/src/llm/mod.rs`** — `LlmCapability` trait (`classify_route`, `synthesize_hyde_doc`, `label`); `LlmBackend` enum (`Genai`, `SubscriptionCli` variants); `LlmBackend::from_env()` probe order (GenaiBackend first, SubscriptionCliBackend second); `LlmBackend::into_arc()`. Behind `#![cfg(feature = "llm")]`.
- **`crates/semantex-core/src/llm/genai_backend.rs`** — Phase 1 backend wrapping `rust-genai` v0.5. Reads `SEMANTEX_LLM_MODEL` (required), `SEMANTEX_LLM_PROVIDER` (optional, inferred from model name), `SEMANTEX_LLM_ENDPOINT` (optional, OpenAI-compat override). Provider API keys (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, etc.) read directly by `genai`; semantex never touches them.
- **`crates/semantex-core/src/llm/subscription_cli.rs`** — Phase 2 backend. Reads `SEMANTEX_LLM_BACKEND=cli:claude|cli:codex|cli:antigravity|cli:auto` and `SEMANTEX_LLM_CLI_BINARY` (optional absolute-path override). Antigravity rejected at `from_env()` with an actionable error. Includes `CliKind::extract_text` for parsing Claude's JSON output wrapper. No subprocess pooling (rational: spawn cost amortized; see file header comment).
- **`crates/semantex-core/src/llm/prompts.rs`** — Shared `CLASSIFIER_SYSTEM_PROMPT`, `HYDE_SYSTEM_PROMPT`, `build_classify_prompt`, `build_hyde_prompt`, `parse_route_from_llm_output` (delegates to `AgentRoute::from_str`).
- **`crates/semantex-core/src/search/agent.rs`** — `AgentPipeline` has `llm: Option<Arc<dyn LlmCapability>>` field (cfg-gated); `with_llm()` builder; `classify_route_with_llm_fallback()` wraps LLM classifier with `SEMANTEX_LLM_CLASSIFY_TIMEOUT_MS`-overridable timeout (default 8 s). The LLM is used for classification only — HyDE call-site pending v0.7.1.
- **`crates/semantex-core/src/search/hybrid.rs`** — `search_with_hyde()` and `merge_hyde_results()` are implemented and unit-tested. Not yet called from `AgentPipeline` (call-site gap; see §15).
- **`crates/semantex-core/src/server/mod.rs`** — daemon startup calls `LlmBackend::from_env()?`, logs the backend label, and passes `Arc<dyn LlmCapability>` to `AgentPipeline`. When no LLM configured, logs a discovery hint if `claude` or `codex` is on PATH (not `antigravity` — excluded per Item 2.3 to avoid suggesting a backend that fails at startup).
- **`crates/semantex-mcp/src/server.rs`** — MCP `AgentPipeline` also receives the LLM backend via `with_llm()`. `tool_agent`/`tool_search`/`tool_deep_search` are async (required by the LLM classifier path).
- **`crates/semantex-cli/src/commands/llm_status.rs`** — `semantex llm-status` subcommand (feature-gated). Detects backend, prints label, runs a health-check classify call with `SEMANTEX_LLM_CLASSIFY_TIMEOUT_MS`-overridable timeout (default 8 s).
- **`crates/semantex-cli/src/commands/watch.rs`** — `semantex watch` now wires `LlmBackend::from_env()` (patched in `b14bca3`; was missing before the /simplify sweep).
- **`crates/semantex-core/tests/llm_smoke.rs`** — end-to-end test binary; `#[ignore]`-gated by `SEMANTEX_LLM_TEST_OLLAMA=1` / `SEMANTEX_LLM_TEST_CLI=<kind>`.

### Env-var contract (live, as of `b14bca3`)

| Var | Backend | Effect |
|-----|---------|--------|
| `SEMANTEX_LLM_MODEL` | GenaiBackend | Required to activate Phase 1. Model name (`qwen2.5-coder:7b`, `claude-sonnet-4-6`, etc.). |
| `SEMANTEX_LLM_PROVIDER` | GenaiBackend | Optional. Provider string (`ollama`, `anthropic`, `openai`, `gemini`, etc.). Inferred from model name when absent. |
| `SEMANTEX_LLM_ENDPOINT` | GenaiBackend | Optional. OpenAI-compatible base URL override (LiteLLM, vLLM, LM Studio, custom Ollama port). |
| `SEMANTEX_LLM_BACKEND` | SubscriptionCliBackend | `cli:claude`, `cli:codex`, `cli:auto`, or `cli:antigravity` (last returns Err at startup). Activates Phase 2. |
| `SEMANTEX_LLM_CLI_BINARY` | SubscriptionCliBackend | Absolute path override for the CLI binary. Only honored when basename matches the kind's CLI name (prevents cross-kind confusion under `cli:auto`). |
| `SEMANTEX_LLM_CLASSIFY_TIMEOUT_MS` | Both | Override classify timeout. Default 8000 ms. Used by daemon classifier and `llm-status` health-check. |

---

## 3. Goals, non-goals, success criteria

### Goals
1. Add LLM-driven query classification that beats the keyword classifier on hard cases (multi-intent queries, NL paraphrases of structural questions).
2. Add HyDE retrieval channel that closes the "NL query → code" semantic gap on queries where the literal tokens don't appear in the code (e.g. "code that handles timeouts" → `tokio::time::Duration`).
3. Provide a single environment-variable-driven configuration surface that works for: API-key users, Ollama airgap users, and (Phase 2) coding-agent-CLI subscription users.
4. Keep the default `cargo build` byte-identical dependency-wise to v0.6 — LLM features are strictly opt-in via a `llm` Cargo feature.

### Non-goals (re-stated)
- Bundling an LLM model.
- Reimplementing provider OAuth/credential flows.
- MCP `sampling/createMessage` integration.

### Success criteria (post-ship status)

- ✅ Phase 1: `cargo build --features llm` clean (`b14bca3`).
- ✅ Phase 1: `cargo test --features llm` clean.
- ⚠️ Phase 1: end-to-end test against a real Ollama instance — the live test that validated the implementation used the Claude CLI backend (`SEMANTEX_LLM_BACKEND=cli:claude`), not Ollama. Both code paths are exercised by `llm_smoke.rs`; the Ollama-specific gate requires `SEMANTEX_LLM_TEST_OLLAMA=1` (operator-run).
- ⚠️ Phase 1: HyDE channel produces non-empty results — `HybridSearcher::search_with_hyde()` is implemented and unit-tested, but the call-site in `AgentPipeline` is not yet wired (see §15). HyDE infrastructure ✅, call-site ⚠️ (v0.7.1).
- ✅ Phase 2: subprocess invocation works for Claude + Codex; tested live against Claude CLI. Codex gated by `SEMANTEX_LLM_TEST_CLI=codex`.
- ⚠️ Phase 2: "per-CLI version-detection in place" — the spec described flag-set detection at construction time; what shipped is a simpler startup-time resolution (find binary on PATH, reject unknown ones early). Version-specific flag-set detection was not implemented; the implementation relies on the current flag set (`--print --output-format json`) being stable. Track as a follow-up if `claude` changes its flags.
- ✅ Phase 2: degradation path verified — missing CLI → `Err` at startup with actionable message; LLM error → keyword classifier fallback (`classify_route_with_llm_fallback`).
- ⚠️ Bench gate: deferred, user-authorized. Aggregate CCB Δ improvement ≥3pp (revised from 5pp per §8) on 4 repos with `--features llm` enabled. Not yet run.

---

## 4. Phase 1 — `genai` adapter (v0.7)

### Item 1.1 — Cargo feature + dependency

**Goal:** Make `genai` an optional dependency behind a `llm` feature flag.

**Files (exclusive write):**
- `Cargo.toml` (root) — add `genai = { version = "0.5", optional = true }` to `[workspace.dependencies]`.
- `crates/semantex-core/Cargo.toml` — add:
  ```toml
  [features]
  llm = ["dep:genai", "dep:tokio"]

  [dependencies]
  genai = { workspace = true, optional = true }
  ```
  Note: `tokio` is already in the dep graph (used by daemon); reference it under the feature so `genai`'s async surface compiles. Verify `tokio` features include `rt`, `macros`.

**Desired behavior:**
- Default `cargo build` brings in zero LLM deps. `cargo tree | grep genai` returns nothing.
- `cargo build --features llm` brings in `genai` and its transitive deps (reqwest, tokio-tungstenite, etc.).

**Acceptance:**
- Default build's `cargo tree` shows no genai. `--features llm` build compiles cleanly.

---

### Item 1.2 — `LlmBackend` enum + `LlmCapability` trait

**Goal:** Define a single backend abstraction with a `Genai` variant. The variant set is open; Phase 2 will add `SubscriptionCli`.

**Files (exclusive write, NEW):**
- `crates/semantex-core/src/llm/mod.rs` — new module.
- `crates/semantex-core/src/llm/genai_backend.rs` — new submodule (Phase 1).

**Read-only (do not modify):**
- All non-`llm/` files. Wiring into `agent_classifier.rs` and `hybrid.rs` is Item 1.4's responsibility.

**Module shape:**

```rust
// crates/semantex-core/src/llm/mod.rs
#![cfg(feature = "llm")]

mod genai_backend;
pub use genai_backend::GenaiBackend;

use crate::search::agent_classifier::AgentRoute;

/// What we ask an LLM to do.
#[async_trait::async_trait]
pub trait LlmCapability: Send + Sync {
    /// Pick the best AgentRoute for `query`. Used as an optional pre-classifier;
    /// caller falls back to the keyword classifier on Err.
    async fn classify_route(&self, query: &str) -> anyhow::Result<AgentRoute>;

    /// Generate a hypothetical code snippet that would answer `query`. Used by
    /// the HyDE retrieval channel; caller falls back to base search on Err.
    async fn synthesize_hyde_doc(&self, query: &str) -> anyhow::Result<String>;

    /// Human-readable backend name for logs ("genai/anthropic/claude-sonnet-4-6",
    /// "cli:claude", etc.). Used at startup banner only.
    fn label(&self) -> &str;
}

/// Concrete dispatcher. Constructed once at daemon startup; held in
/// `Arc<dyn LlmCapability>` form across the rest of the codebase.
pub enum LlmBackend {
    Genai(GenaiBackend),
    // Phase 2 will add: SubscriptionCli(SubscriptionCliBackend),
}

impl LlmBackend {
    /// Detect from env vars. Returns Ok(None) if no LLM is configured.
    /// See §4 Item 1.3 for the env-var schema.
    pub fn from_env() -> anyhow::Result<Option<Self>> { /* impl per Item 1.3 */ }

    pub fn as_capability(&self) -> &dyn LlmCapability {
        match self {
            Self::Genai(b) => b,
        }
    }
}
```

**Acceptance:**
- `cargo build --features llm` clean.
- Unit test: trait object construction via `Arc::new(GenaiBackend::new(...)) as Arc<dyn LlmCapability>` compiles.
- No public surface added without `#[cfg(feature = "llm")]`.

---

### Item 1.3 — `GenaiBackend` implementation

**Goal:** Map `LlmCapability::classify_route` and `synthesize_hyde_doc` onto `genai::Client`.

**Files (exclusive write):**
- `crates/semantex-core/src/llm/genai_backend.rs`

**Env-var configuration schema (locked here, used by §4 Item 1.4 and Phase 2):**

| Env var | Example | Effect |
|---|---|---|
| `SEMANTEX_LLM_MODEL` | `qwen2.5-coder:7b` | Model name. Required to enable LLM features. |
| `SEMANTEX_LLM_PROVIDER` | `ollama`, `anthropic`, `openai`, `gemini`, `xai`, `groq`, `cohere`, `deepseek`, `openrouter` | Optional. If unset, inferred from model name where possible (e.g., `claude-` → anthropic, `gpt-` → openai), else defaults to `ollama` for backwards compatibility. |
| `SEMANTEX_LLM_ENDPOINT` | `http://localhost:11434/v1` | Optional override for OpenAI-compatible endpoints (custom Ollama port, LiteLLM proxy, vLLM, LM Studio, etc.). |
| Provider API keys | `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, … | Read by `genai` directly; semantex never touches these. |

**Implementation shape:**

```rust
use genai::Client;
use genai::adapter::AdapterKind;
use genai::chat::{ChatMessage, ChatRequest, ChatOptions};
use crate::search::agent_classifier::AgentRoute;
use crate::llm::LlmCapability;

pub struct GenaiBackend {
    client: Client,
    model: String,
    label: String,
}

impl GenaiBackend {
    /// Construct from env. Returns Ok(None) if SEMANTEX_LLM_MODEL is unset.
    pub fn from_env() -> anyhow::Result<Option<Self>> {
        let Some(model) = std::env::var("SEMANTEX_LLM_MODEL").ok() else {
            return Ok(None);
        };
        let provider_override = std::env::var("SEMANTEX_LLM_PROVIDER").ok();
        let endpoint_override = std::env::var("SEMANTEX_LLM_ENDPOINT").ok();
        let client = build_client(&model, provider_override.as_deref(), endpoint_override.as_deref())?;
        let label = format!("genai/{model}");
        Ok(Some(Self { client, model, label }))
    }
}

#[async_trait::async_trait]
impl LlmCapability for GenaiBackend {
    async fn classify_route(&self, query: &str) -> anyhow::Result<AgentRoute> {
        let prompt = build_classify_prompt(query); // see acceptance below
        let chat = ChatRequest::new(vec![
            ChatMessage::system(CLASSIFIER_SYSTEM_PROMPT),
            ChatMessage::user(prompt),
        ]);
        let opts = ChatOptions::default()
            .with_max_tokens(8) // route name is at most ~20 chars; bound aggressively
            .with_temperature(0.0);
        let resp = self.client.exec_chat(&self.model, chat, Some(&opts)).await?;
        let text = resp.first_text().ok_or_else(|| anyhow::anyhow!("empty LLM response"))?;
        parse_route_from_llm_output(text)
    }

    async fn synthesize_hyde_doc(&self, query: &str) -> anyhow::Result<String> {
        let prompt = build_hyde_prompt(query);
        let chat = ChatRequest::new(vec![
            ChatMessage::system(HYDE_SYSTEM_PROMPT),
            ChatMessage::user(prompt),
        ]);
        let opts = ChatOptions::default()
            .with_max_tokens(400) // ~30 lines of synthetic code
            .with_temperature(0.2);
        let resp = self.client.exec_chat(&self.model, chat, Some(&opts)).await?;
        resp.first_text()
            .map(|s| s.to_string())
            .ok_or_else(|| anyhow::anyhow!("empty HyDE response"))
    }

    fn label(&self) -> &str { &self.label }
}

// CLASSIFIER_SYSTEM_PROMPT constraints:
// - List the AgentRoute variants verbatim. Include 1-line examples.
// - Instruction: "Respond with exactly one route name, no other text."
// - Maximum response length: must fit in the 8-token cap.
//
// parse_route_from_llm_output:
// - Trim, lowercase, match against AgentRoute::from_str (we'll add this).
// - On parse failure: anyhow::bail!("LLM returned unrecognized route: {text:?}").
//
// HYDE_SYSTEM_PROMPT constraints:
// - "Write a short snippet (1-30 lines) of code that would directly answer the
//   user's question. Output ONLY the code, no markdown fences, no explanation."
// - Language hint: derive from query if obvious (e.g., "in Rust", "Python"),
//   else omit and let the LLM pick.
```

**Add to `AgentRoute`:**

```rust
// In crates/semantex-core/src/search/agent_classifier.rs
impl std::str::FromStr for AgentRoute {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> anyhow::Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "filepattern" | "file_pattern" => Ok(Self::FilePattern),
            "regex" => Ok(Self::Regex),
            "exactsymbol" | "exact_symbol" => Ok(Self::ExactSymbol),
            "structural" => Ok(Self::Structural),
            "deep" => Ok(Self::Deep),
            "analytical" => Ok(Self::Analytical),
            "exhaustive" => Ok(Self::Exhaustive),
            "semantic" => Ok(Self::Semantic),
            "architecture" => Ok(Self::Architecture),
            "exhaustivestructural" | "exhaustive_structural" => Ok(Self::ExhaustiveStructural),
            "deepwithexamples" | "deep_with_examples" => Ok(Self::DeepWithExamples),
            "featureplanning" | "feature_planning" => Ok(Self::FeaturePlanning),
            other => anyhow::bail!("unknown AgentRoute: {other:?}"),
        }
    }
}
```

**Acceptance:**
- Unit test (feature-gated): `GenaiBackend::from_env()` returns Ok(None) when `SEMANTEX_LLM_MODEL` is unset.
- Unit test: `parse_route_from_llm_output("Deep")` → `AgentRoute::Deep`; `parse_route_from_llm_output("not a route")` → Err.
- Unit test: `AgentRoute::from_str` round-trips against `Display`.
- Integration test (feature-gated, ignored by default with `#[ignore = "requires Ollama"]`): construct `GenaiBackend` against a local Ollama instance, call `classify_route("who calls handle_request?")`, assert `AgentRoute::Structural`. This test runs only when `SEMANTEX_LLM_TEST_OLLAMA=1` is set; gated to keep CI offline.
- HyDE: same integration-test treatment for `synthesize_hyde_doc("code that handles timeouts")` — assert the response contains `Duration` or `timeout` (lowercase compare).

---

### Item 1.4 — Wire `LlmBackend` into `AgentPipeline` and `HybridSearcher`

**Goal:** Plumb the optional `Arc<dyn LlmCapability>` through the agent pipeline so `classify_route` overrides the keyword classifier when present, and `hybrid_search` opportunistically uses HyDE.

**Files (exclusive write):**
- `crates/semantex-core/src/search/agent.rs` — `AgentPipeline` gains an `llm: Option<Arc<dyn LlmCapability>>` field.
- `crates/semantex-core/src/search/agent_classifier.rs` — `AgentRoute::from_str` (see Item 1.3) and a `Display` impl if not already present. NO other production logic touches.
- `crates/semantex-core/src/search/hybrid.rs` — `HybridSearcher` gains an `Option<Arc<dyn LlmCapability>>` field. A new method `pub async fn search_with_hyde()` exists alongside the existing sync `search()`.
- `crates/semantex-core/src/server/mod.rs` — daemon startup constructs `LlmBackend::from_env()?`, logs the label or "no LLM configured", and passes the `Arc<dyn LlmCapability>` to `HybridSearcher` and `AgentPipeline`.

**Read-only:** All other files.

**Wiring shape:**

```rust
// AgentPipeline::handle (in agent.rs)
pub async fn handle(&self, request: &AgentRequest) -> AgentResponse {
    // Allow explicit route override (existing behavior) -> keyword classifier ->
    // LLM classifier (if available + no explicit route).
    let route = if let Some(r) = request.route {
        r
    } else if let Some(llm) = &self.llm {
        // Try LLM with a hard timeout. On any error or timeout, fall back.
        match tokio::time::timeout(LLM_CLASSIFY_TIMEOUT, llm.classify_route(&request.query)).await {
            Ok(Ok(r)) => r,
            _ => classify_agent_query(&request.query),
        }
    } else {
        classify_agent_query(&request.query)
    };
    // ... existing dispatch ...
}

// HybridSearcher::search_with_hyde (in hybrid.rs)
pub async fn search_with_hyde(&self, query: &SearchQuery) -> Result<SearchOutput> {
    let Some(llm) = &self.llm else { return self.search(query).map_err(...).into() };
    // Synth HyDE doc with timeout; on error, fall back to plain search.
    let base = tokio::task::spawn_blocking({ let s = self.clone(); let q = query.clone();
        move || s.search(&q) }).await??;
    let hyde_doc = match tokio::time::timeout(LLM_HYDE_TIMEOUT, llm.synthesize_hyde_doc(&query.text)).await {
        Ok(Ok(doc)) => doc,
        _ => return Ok(base),  // fallback: no HyDE
    };
    let hyde_query = SearchQuery::new(&hyde_doc).max_results(query.max_results());
    let hyde = tokio::task::spawn_blocking({ let s = self.clone(); let q = hyde_query;
        move || s.search(&q) }).await??;
    Ok(merge_results(base, hyde, query.max_results()))
}

/// Replacement for the old merge_hyde_results — same shape but lives on the
/// generic HybridSearcher::search_with_hyde, not specific to ONNX.
fn merge_results(base: SearchOutput, hyde: SearchOutput, max_results: usize) -> SearchOutput { /* ... */ }
```

**Constants:**
- `const LLM_CLASSIFY_TIMEOUT: Duration = Duration::from_millis(800)`. Bound to "fast classifier" budgets so a slow LLM doesn't tank latency.
- `const LLM_HYDE_TIMEOUT: Duration = Duration::from_secs(5)`. HyDE is allowed more wall — it's one synthesis per agent query.

**Important: `merge_results` design**
- Dedup by `chunk.id` (NOT `(file, start, end)`; that bug is why the old `merge_hyde_results` was removed).
- Re-sort by `score` descending after merge.
- Cap at `max_results`.

**Acceptance:**
- `cargo build --features llm` clean.
- `cargo build` (no feature) clean — the `llm: Option<Arc<dyn LlmCapability>>` field uses `#[cfg(feature = "llm")]` so the default build's struct shape doesn't include it. Match arms gracefully degrade in the no-feature build.
- Unit test: `AgentPipeline` with `llm: None` behaves identically to v0.6.
- Unit test: `AgentPipeline` with a mocked `LlmCapability` that returns `AgentRoute::Structural` for any query routes a "free-form" question to Structural (not the keyword default).
- Unit test: `HybridSearcher::search_with_hyde` with a mocked LLM that errors falls back to base search.
- Integration test (gated by `SEMANTEX_LLM_TEST_OLLAMA=1`): real end-to-end against Ollama.

---

### Item 1.5 — CLI surface

**Goal:** Expose LLM features on the CLI without polluting the default code path.

**Files (exclusive write):**
- `crates/semantex-cli/src/main.rs` — add `--llm-status` subcommand or flag.
- `crates/semantex-cli/Cargo.toml` — gate any LLM-aware CLI surface behind `llm` feature (pass-through to `semantex-core/llm`).

**Desired behavior:**
- `semantex llm-status` (only available with `--features llm`) prints:
  - Whether `LlmBackend::from_env()` returned a backend
  - The backend label
  - For Phase 1: the inferred provider + endpoint
  - Whether the backend successfully answered a 1-token health-check classify call
- Without `--features llm`, the subcommand is absent.

**Acceptance:**
- Default build: `semantex --help` shows no LLM-related flags.
- Feature build: `semantex --help` shows `llm-status`. `semantex llm-status` works against a configured Ollama / API-key setup.

---

## 5. Phase 2 — `SubscriptionCli` backend (v0.8)

### Item 2.1 — Add `SubscriptionCli` variant to `LlmBackend`

**Goal:** A second `LlmBackend` variant that shells out to `claude`, `codex`, or `antigravity` instead of calling APIs directly.

**Files (exclusive write):**
- `crates/semantex-core/src/llm/mod.rs` — add `SubscriptionCli(SubscriptionCliBackend)` variant.
- `crates/semantex-core/src/llm/subscription_cli.rs` — new submodule.

**Read-only:** All other files including `genai_backend.rs`. Phase 1 work is locked.

**Module shape:**

```rust
// crates/semantex-core/src/llm/subscription_cli.rs
#![cfg(feature = "llm")]

use std::path::PathBuf;
use std::time::Duration;
use crate::llm::LlmCapability;
use crate::search::agent_classifier::AgentRoute;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CliKind {
    Claude,
    Codex,
    Antigravity,
}

pub struct SubscriptionCliBackend {
    kind: CliKind,
    binary: PathBuf,
    label: String,
}

impl SubscriptionCliBackend {
    /// Detect from env or PATH.
    /// SEMANTEX_LLM_BACKEND=cli:claude        -> resolve(claude) | err
    /// SEMANTEX_LLM_BACKEND=cli:codex         -> resolve(codex)  | err
    /// SEMANTEX_LLM_BACKEND=cli:antigravity   -> resolve(antigravity) | err
    /// SEMANTEX_LLM_BACKEND=cli:auto          -> claude, then codex, then antigravity
    pub fn from_env() -> anyhow::Result<Option<Self>> { /* impl */ }

    /// Resolve a binary by name on PATH. Returns absolute path.
    fn resolve(name: &str) -> Option<PathBuf> { /* using `which` crate */ }
}

#[async_trait::async_trait]
impl LlmCapability for SubscriptionCliBackend {
    async fn classify_route(&self, query: &str) -> anyhow::Result<AgentRoute> {
        let prompt = build_classify_prompt(query); // shared with GenaiBackend; see §6.1
        let stdout = self.exec(&prompt, CLASSIFY_TIMEOUT).await?;
        parse_route_from_llm_output(&stdout) // shared with GenaiBackend
    }

    async fn synthesize_hyde_doc(&self, query: &str) -> anyhow::Result<String> {
        let prompt = build_hyde_prompt(query); // shared
        self.exec(&prompt, HYDE_TIMEOUT).await
    }

    fn label(&self) -> &str { &self.label }
}

impl SubscriptionCliBackend {
    async fn exec(&self, prompt: &str, timeout: Duration) -> anyhow::Result<String> {
        let mut cmd = match self.kind {
            // --output-format json gives a stable wrapper; we strip to .result.text.
            // --print suppresses preamble. Validate flag set against `claude --version`.
            CliKind::Claude => {
                let mut c = tokio::process::Command::new(&self.binary);
                c.args(["--print", "--output-format", "json"]);
                c.arg(prompt);
                c
            }
            // codex exec emits markdown-fenced output by default; we ask for raw.
            CliKind::Codex => {
                let mut c = tokio::process::Command::new(&self.binary);
                c.args(["exec", "--quiet"]);
                c.arg(prompt);
                c
            }
            // antigravity headless mode flags TBD per Item 2.4
            CliKind::Antigravity => { /* gated by Item 2.4 */ todo!() }
        };
        cmd.stdin(std::process::Stdio::null())
           .stdout(std::process::Stdio::piped())
           .stderr(std::process::Stdio::piped())
           .kill_on_drop(true);

        let child = cmd.spawn().context("spawn LLM CLI")?;
        let output = tokio::time::timeout(timeout, child.wait_with_output())
            .await
            .context("LLM CLI timed out")??;
        if !output.status.success() {
            anyhow::bail!("LLM CLI exited {}: {}",
                output.status,
                String::from_utf8_lossy(&output.stderr));
        }
        let raw = String::from_utf8_lossy(&output.stdout).into_owned();
        Ok(self.kind.extract_text(&raw))  // strip JSON wrapper if Claude
    }
}

impl CliKind {
    fn extract_text(&self, raw: &str) -> String {
        match self {
            // claude --output-format json returns {"type":"result","result":"...","is_error":false,...}
            CliKind::Claude => serde_json::from_str::<ClaudeJsonResult>(raw)
                .map(|r| r.result)
                .unwrap_or_else(|_| raw.to_string()),
            CliKind::Codex => raw.to_string(), // codex exec --quiet outputs raw text
            CliKind::Antigravity => raw.to_string(),
        }
    }
}

#[derive(serde::Deserialize)]
struct ClaudeJsonResult { result: String }
```

**Acceptance:**
- Unit tests for `from_env`: each `SEMANTEX_LLM_BACKEND=cli:<kind>` value produces the right `CliKind` if the binary is on PATH; missing binary returns descriptive error.
- Unit test for `extract_text`: synthetic Claude JSON wrapper strips correctly; malformed JSON falls back to raw.
- Integration test (gated by `SEMANTEX_LLM_TEST_CLI=claude` or `=codex`): real subprocess invocation against the user's installed CLI returns a parseable route.
- The `which` crate is the **only** new dependency (already a transitive of many other crates; verify with `cargo tree`).

---

### Item 2.2 — Subprocess pool / re-use strategy

**Goal:** Avoid the 100-300ms subprocess spawn tax on every classifier call.

**Decision:** For Phase 2, **do not** pool. Each call spawns a fresh subprocess. Reasoning:
- LLM classification fires once per `semantex_agent` call (not per inner search), so spawn cost is amortized.
- The CLI tools don't reliably expose a long-lived REPL/IPC mode that's safe to multiplex.
- Pooling adds complexity (process lifecycle, restart on stale auth tokens, concurrent-call ordering) that's better deferred until benchmarks show the spawn tax actually hurts.

If benchmarks later show spawn cost dominates, revisit. Document the decision in a code comment at the top of `subscription_cli.rs`.

**Files (write):**
- `crates/semantex-core/src/llm/subscription_cli.rs` — add the rationale comment.

**Acceptance:** No additional code; tracked as an explicit non-decision.

---

### Item 2.3 — Discovery hint at daemon startup

**Goal:** First-time UX. When the daemon starts and finds no `SEMANTEX_LLM_*` env vars, but detects a coding-agent CLI on PATH, log a single hint line.

**Files (exclusive write):**
- `crates/semantex-core/src/server/mod.rs` — startup banner extension.

**Behavior:**

```rust
// After LlmBackend::from_env() returns None:
let hint = if which::which("claude").is_ok() {
    Some(("claude", "Claude Code"))
} else if which::which("codex").is_ok() {
    Some(("codex", "OpenAI Codex"))
} else if which::which("antigravity").is_ok() {
    Some(("antigravity", "Google Antigravity"))
} else {
    None
};
if let Some((bin, label)) = hint {
    tracing::info!(
        "LLM features disabled. Detected {} ({} on PATH). To enable: \
         set SEMANTEX_LLM_BACKEND=cli:{}",
        label, bin, bin
    );
}
```

**Acceptance:**
- Unit test: when none of the CLIs are on PATH, no hint is logged.
- Integration test (gated): when `claude` is on PATH and no env vars set, the hint fires once at startup.
- Important: this must NOT fire on subsequent search requests — startup-banner only.

---

### Item 2.4 — Antigravity gating

**Goal:** Antigravity CLI just launched (preview) at I/O 2026; its headless invocation surface is not stable.

**Decision:** Ship Phase 2 with `Antigravity` variant **stubbed** — returns Err with a descriptive message instructing the user to use `cli:claude` or `cli:codex` until the antigravity headless surface stabilizes. Track antigravity headless support as a follow-up issue separate from this spec.

**Files (exclusive write):**
- `crates/semantex-core/src/llm/subscription_cli.rs` — the `CliKind::Antigravity` arm of `exec` returns Err with message.

**Acceptance:**
- `SEMANTEX_LLM_BACKEND=cli:antigravity` produces a clean error at startup, not at first-query time.

---

## 6. Shared cross-phase concerns

### 6.1 — Shared prompt + parse helpers

**Goal:** `GenaiBackend` and `SubscriptionCliBackend` must use IDENTICAL prompts and identical output parsing so behavior is consistent across backends.

**Files (exclusive write):**
- `crates/semantex-core/src/llm/prompts.rs` — new submodule defining:
  - `pub const CLASSIFIER_SYSTEM_PROMPT: &str`
  - `pub fn build_classify_prompt(query: &str) -> String`
  - `pub fn parse_route_from_llm_output(text: &str) -> anyhow::Result<AgentRoute>`
  - `pub const HYDE_SYSTEM_PROMPT: &str`
  - `pub fn build_hyde_prompt(query: &str) -> String`

This module is `pub(crate)` and shared by both backends.

**Acceptance:**
- Unit tests in `prompts.rs` cover every `AgentRoute` variant for `parse_route_from_llm_output`.
- The prompts include a list of every `AgentRoute` variant + one-line description; if a new route is ever added, this file must be updated (enforce via a `match` in a hidden test).

### 6.2 — Async surface boundary

The LLM trait is `async`; the rest of `semantex-core` is largely sync. The boundary is `AgentPipeline::handle` (already on a tokio runtime in the daemon) and `HybridSearcher::search_with_hyde` (new method, async).

- `AgentPipeline::handle` was sync in v0.6; **make it `async fn`** in Phase 1. Update all callers in `crates/semantex-mcp/src/server.rs` and `crates/semantex-core/src/server/handler.rs`. Tokio runtime in the daemon already exists.
- `HybridSearcher::search` remains sync (it's CPU-heavy ColBERT + Tantivy). The `search_with_hyde` method is async only because of the LLM call; the actual search calls run via `tokio::task::spawn_blocking`.

### 6.3 — Telemetry hooks

Add structured `tracing` events at:
- LLM call start (level=debug): query length, model label, capability ("classify"/"hyde")
- LLM call success (level=debug): latency_ms, output length
- LLM call failure (level=warn): latency_ms, error
- Fallback fired (level=info): which capability fell back

No PII (don't log query strings at info+). Tests verify the events fire.

### 6.4 — End-to-end harness

A new test binary `crates/semantex-core/tests/llm_smoke.rs` that:
- Constructs a `LlmBackend` from env
- Runs a hardcoded classify query
- Runs a hardcoded HyDE query
- Asserts both return parseable output
- Skipped by default; runs when `SEMANTEX_LLM_TEST_OLLAMA=1` or `SEMANTEX_LLM_TEST_CLI=<kind>` is set.

CI need not run this. It's an operator self-check.

---

## 7. Risk register

| # | Risk | Phase | Likelihood | Impact | Mitigation |
|---|---|---|---|---|---|
| R1 | `genai` API changes between 0.5.x releases | 1 | Medium | Low | Pin to a minor version range in Cargo.toml. Bump deliberately. |
| R2 | LLM classifier slower than keyword classifier dominates agent latency | 1 | High | Medium | LLM_CLASSIFY_TIMEOUT = 800ms; always fall back to keyword on timeout. Phase 1 default disables LLM classifier unless user opts in. |
| R3 | HyDE doc embedding takes longer than the base search win it produces | 1 | Medium | Medium | LLM_HYDE_TIMEOUT = 5s; bench v0.7 vs v0.6 before defaulting on. |
| R4 | Claude / Codex CLI flag set changes (e.g. `--output-format` renamed) | 2 | Medium | High | Version-detect at backend construction (`claude --version`). Cache supported flag set per session. Add an annual review of supported CLI versions. |
| R5 | Subprocess spawn tax exceeds 300ms on Windows | 2 | High | Low | Document. Suggest WSL or skip Phase 2 on Windows for v0.8. |
| R6 | A subscription CLI silently degrades to a smaller model when over quota — semantex sees stale `route` outputs | 2 | Low | Medium | Strict `parse_route_from_llm_output` rejects non-route output; falls back to keyword on parse failure. |
| R7 | `which` crate's PATH resolution differs from user's actual shell setup | 2 | Low | Low | Allow `SEMANTEX_LLM_CLI_BINARY=/abs/path/to/claude` override. |

---

## 8. Acceptance gates (post-ship status)

Cross-reference: `docs/RELEASE-SEQUENCE-2026-05.md` §7 changelog for the full fix history.

### Phase 1 gate (Items 1.1-1.5 — status as of `b14bca3`)
1. ✅ `cargo build --workspace` clean (no `--features`) — was broken by `fddba26`; fixed in `b14bca3` P0.
2. ✅ `cargo build --workspace --features semantex-core/llm` clean
3. ✅ `cargo test --workspace --lib` clean
4. ✅ `cargo test --workspace --lib --features semantex-core/llm` clean
5. ✅ `cargo clippy --workspace --no-deps --lib -- -D warnings` clean (both feature sets)
6. ✅ `cargo clippy --workspace --no-deps --all-targets -- -D warnings` count unchanged from v0.6 baseline
7. ✅ `cargo fmt --all --check` clean
8. ✅ `cargo tree | grep -c genai` → 0 (default build); `cargo tree --features semantex-core/llm | grep -c genai` → ≥1
9. ⚠️ End-to-end smoke (gated): `SEMANTEX_LLM_TEST_OLLAMA=1 cargo test --features semantex-core/llm llm_smoke` — not yet run against Ollama. Live e2e was run against `SEMANTEX_LLM_BACKEND=cli:claude` and passed (two bugs caught and fixed: system-prompt missing, timeout too tight — see §15).

### Phase 2 gate (Items 2.1-2.4 — status as of `b14bca3`)
1. ✅ All Phase 1 gates (see above)
2. ✅ End-to-end smoke (gated): `SEMANTEX_LLM_TEST_CLI=claude` — validated live; `classify_route` returned a parseable route. Bugs caught and fixed (see §15).
3. ⚠️ `SEMANTEX_LLM_TEST_CLI=codex` — not yet run (requires codex CLI installed).
4. ✅ Startup banner test: discovery hint logs when `claude` or `codex` on PATH and no LLM configured; `antigravity` intentionally excluded from hint (its startup would fail at first query).

### Bench gates (deferred, user-authorized)
- ⚠️ v0.7: aggregate CCB Δ improvement ≥3pp with `--features llm + SEMANTEX_LLM_MODEL=ollama/qwen2.5-coder:7b` vs without on 4 repos. Not yet run; requires user bench-spend authorization (~$80). Note: HyDE call-site must be wired (v0.7.1) before this gate is meaningful — without HyDE, only the LLM classifier provides lift.
- ⚠️ v0.8: cli:claude vs api-key accuracy ≥95% on 100-query held-out set. Not yet run (operator-authorized; deferred).

---

## 9. Out of scope (do not do these without a new spec)

- Bundling any model file
- Implementing ONNX / Candle / mistral.rs inference paths
- OAuth flow re-implementation
- MCP `sampling/createMessage` (deprecated)
- A persistent subprocess pool for the CLI backends (R-Item 2.2 above)
- Streaming responses through `LlmCapability` (current shape is request/response; streaming requires a different trait)
- Embeddings API exposed via `LlmCapability` (`genai` supports `EmbedRequest`/`EmbedResponse` but ColBERT is the embedder for semantex; do not switch)
- Tool-use API exposed via `LlmCapability` (the LLM is a single-shot classifier/synthesizer; tool use would invert the agent direction)

---

## 10. Workstream design (parallel subagent execution)

**Status: shipped.** The workstreams below executed 2026-05-27 and merged to `main` at `7eda12f`. This section is preserved for historical reference; do not re-dispatch these workstreams.

### Workstreams (3 teams, mostly parallel)

| WS | Owns these files (exclusive write) | Items | Parallel-safe with |
|---|---|---|---|
| **W-LLM-Core** | `crates/semantex-core/src/llm/mod.rs`, `crates/semantex-core/src/llm/prompts.rs` | 1.2, 6.1 | (none — must merge before W-LLM-Genai and W-LLM-Cli) |
| **W-LLM-Genai** | `crates/semantex-core/src/llm/genai_backend.rs`, `Cargo.toml` (root), `crates/semantex-core/Cargo.toml` | 1.1, 1.3 | W-LLM-Cli (different files) |
| **W-LLM-Cli** | `crates/semantex-core/src/llm/subscription_cli.rs` | 2.1, 2.2, 2.3 (the CLI-detection piece), 2.4 | W-LLM-Genai (different files) |
| **W-LLM-Wire** | `crates/semantex-core/src/search/agent.rs`, `crates/semantex-core/src/search/agent_classifier.rs`, `crates/semantex-core/src/search/hybrid.rs`, `crates/semantex-core/src/server/mod.rs`, `crates/semantex-mcp/src/server.rs` | 1.4, 1.5, 2.3 (the startup-banner piece) | (none — must merge after W-LLM-Core, W-LLM-Genai, W-LLM-Cli land) |

### Dependency DAG

```
W-LLM-Core (1.2, 6.1) ─┐
                       ├─→ Phase 1 barrier (locks LlmCapability + prompts) ─→
W-LLM-Genai (1.1, 1.3) ┘                                                      \
W-LLM-Cli   (2.1, 2.2, 2.4) ──→ Phase 2 barrier (LLM backends ready) ────────→ W-LLM-Wire (1.4, 1.5, 2.3) → integration
```

Coordinator launches:
1. W-LLM-Core first (small; ~half a session). Merges to integration branch immediately.
2. W-LLM-Genai AND W-LLM-Cli in parallel once W-LLM-Core merges.
3. W-LLM-Wire last, after both backend workstreams merge.

### Coordination rules
- Worktree discipline: each subagent works in `isolation: worktree` and commits its work in the worktree before returning. Controller merges branches at integration barriers.
- Always reset the integration checkout to a clean state before merging (the parallel-subagent-worktree-leakage memory).
- File conflict rule: if a workstream finds it needs to touch a non-owned file, open `coordination_request.md` in the repo root rather than editing.

---

## 11. File ownership map

### Read-only for all (do not modify)
- Anything under `crates/semantex-core/src/search/{deep,planner}.rs` — v0.5/v0.6 work, locked.
- `crates/semantex-core/src/server/{handler,protocol}.rs` — v0.5 work, locked. (LLM does not change the wire protocol.)
- `crates/semantex-core/src/index/*` — index format unchanged.
- `crates/semantex-core/src/chunking/*` — chunking unchanged.
- All benchmark scripts under `benchmarks/`.

### Exclusive write per workstream

| File | Workstream |
|---|---|
| `crates/semantex-core/src/llm/mod.rs` (new) | W-LLM-Core |
| `crates/semantex-core/src/llm/prompts.rs` (new) | W-LLM-Core |
| `crates/semantex-core/src/llm/genai_backend.rs` (new) | W-LLM-Genai |
| `crates/semantex-core/src/llm/subscription_cli.rs` (new) | W-LLM-Cli |
| `Cargo.toml` (root) — `genai` workspace dep | W-LLM-Genai |
| `crates/semantex-core/Cargo.toml` — `llm` feature, deps | W-LLM-Genai |
| `crates/semantex-core/src/lib.rs` — re-add `pub mod llm;` under cfg | W-LLM-Core |
| `crates/semantex-core/src/search/agent.rs` | W-LLM-Wire |
| `crates/semantex-core/src/search/agent_classifier.rs` — add `FromStr` + `Display` only | W-LLM-Wire |
| `crates/semantex-core/src/search/hybrid.rs` — add `search_with_hyde` only | W-LLM-Wire |
| `crates/semantex-core/src/server/mod.rs` — startup wiring + banner | W-LLM-Wire |
| `crates/semantex-mcp/src/server.rs` — async handle adoption | W-LLM-Wire |
| `crates/semantex-core/tests/llm_smoke.rs` (new) | W-LLM-Wire |

### Shared but read-only after Phase 1 lands
- `crates/semantex-core/src/llm/mod.rs` — once Phase 1 merges, Phase 2 may APPEND a variant (`SubscriptionCli`) and the `as_capability` arm; no other changes.

---

## 12. Validation harness

### Phase 1 smoke (operator-run)
```bash
# Pre-req: ollama running locally with qwen2.5-coder:7b pulled
SEMANTEX_LLM_MODEL=qwen2.5-coder:7b \
SEMANTEX_LLM_PROVIDER=ollama \
SEMANTEX_LLM_TEST_OLLAMA=1 \
cargo test -p semantex-core --features llm --test llm_smoke -- --nocapture
```

### Phase 2 smoke (operator-run)
```bash
# Pre-req: claude or codex CLI installed and logged in
SEMANTEX_LLM_BACKEND=cli:claude \
SEMANTEX_LLM_TEST_CLI=claude \
cargo test -p semantex-core --features llm --test llm_smoke -- --nocapture
```

### Bench harness (deferred to bench-spend-authorized session)
```bash
# Existing benchmarks/agent_bench.py — adds an --llm-backend flag passed through
# to the daemon via SEMANTEX_LLM_* env vars.
python3 benchmarks/agent_bench.py run \
  --repos /Users/tk/dev/gin /Users/tk/dev/flask /Users/tk/dev/CopilotKit /Users/tk/dev/platform \
  --llm-backend ollama/qwen2.5-coder:7b \
  --output benchmarks/results/run-llm-v0.7
```

---

## 13. Cost budget

- Phase 1 implementation: ~2-3 subagent sessions, ~$15 cost (no bench).
- Phase 2 implementation: ~2 subagent sessions, ~$10 cost (no bench).
- Phase 1 bench (operator-authorized): ~$80 (4 repos × 5 questions × 2 arms = 40 runs).
- Phase 2 bench: not strictly needed if Phase 1's bench shows LLM lift; quality-difference between Genai-via-Anthropic and CLI-via-Claude-subscription should be ≈0.

---

## 14. Out-of-scope (don't do these without a new spec)

- Anything in §9.
- Removing or modifying existing keyword classifier behavior.
- Changing `handle_feature_planning` or planner to use the LLM (the planner is intentionally LLM-free; that's a Spec Q Item 10 design choice).
- Changing chunk embedding (`semantic_role.rs`, ColBERT pipeline).

---

## 15. Post-ship learnings (2026-05-27)

These findings emerged during live e2e validation after `7eda12f` merged. They are recorded here so future work on the LLM stack has accurate context.

### 15.1 Bug: system prompt missing from SubscriptionCli backend (fixed in `33f4fcb`)

The spec (§5) showed the `exec` method accepting a single `prompt` argument, implicitly assuming the system-prompt and user-prompt would be composed before the call. The initial implementation passed only `build_classify_prompt(query)` to `exec`, not `CLASSIFIER_SYSTEM_PROMPT`. CLI backends have no separate system-prompt channel (unlike `genai::ChatRequest` which has `ChatMessage::system()`), so the LLM never saw the route list and hallucinated plausible-sounding route names.

**Fix:** `SubscriptionCliBackend::classify_route` now concatenates: `"{CLASSIFIER_SYSTEM_PROMPT}\n\n{build_classify_prompt(query)}"`. Same pattern applied to `synthesize_hyde_doc` for `HYDE_SYSTEM_PROMPT`.

**Takeaway:** When adding a new CLI kind in the future, verify the full prompt is composed before `exec` — the calling convention is "single string prompt," not "system + user split."

### 15.2 Bug: 800 ms classify timeout too tight for CLI backends (fixed in `fddba26`, `cdf912c`)

The spec (§4 Item 1.4) specified `LLM_CLASSIFY_TIMEOUT = 800ms` for "fast classifier budgets." This is adequate for HTTP API calls (Anthropic/OpenAI latency P50 ≈ 200-400 ms), but the `SubscriptionCli` backend spawns a subprocess. `claude` takes 3-5 s minimum from subprocess spawn to first token (authentication, context loading, network round-trip). At 800 ms, every Claude CLI call timed out and silently fell back to the keyword classifier.

**Fix:** Default bumped to 8000 ms. Added `SEMANTEX_LLM_CLASSIFY_TIMEOUT_MS` env-var override so users with fast local Ollama instances can dial down. Same 8 s default applied to `semantex llm-status` health-check.

**Takeaway:** Classify timeout must be set to accommodate the slowest supported backend. Env-var override is the right escape hatch. The spec's 800 ms constant should have been qualified as "API-path budget only."

### 15.3 Default-build break: cfg-gated constant used in un-gated function (fixed in `b14bca3`)

`fn llm_classify_timeout()` was not `#[cfg(feature = "llm")]`-gated but referenced `LLM_CLASSIFY_TIMEOUT_DEFAULT_MS`, which was `#[cfg(feature = "llm")]`-gated. This caused `E0425: cannot find value` on `cargo build --workspace` (no features). The `--features llm` build always worked, so the break was latent from `fddba26` until the `/simplify` sweep.

**Fix:** Added `#[cfg(feature = "llm")]` to `fn llm_classify_timeout()`.

**Takeaway:** When a constant or type is cfg-gated, every consumer function must be identically gated. The `/simplify` review pattern caught this; running `cargo build --workspace` (without features) as a standard local gate would have caught it earlier.

### 15.4 HyDE call-site gap (being patched in v0.7.1 by Team A)

`HybridSearcher::search_with_hyde()` was implemented in `hybrid.rs` during W-LLM-Wire and is unit-tested. However, `AgentPipeline` in `agent.rs` only wires the LLM classifier (`classify_route_with_llm_fallback`); it does not call `search_with_hyde`. The LLM backend's `synthesize_hyde_doc` capability has no call-site in the agent dispatch path as of `b14bca3`.

**Status:** Team A is patching this gap in a parallel worktree (v0.7.1). Controller will merge both teams' work. Until v0.7.1 lands, LLM-enabled builds provide classifier override only; HyDE retrieval is dormant.

**What's in place:** `HybridSearcher::with_llm()` builder and `search_with_hyde()` with proper timeout handling and result merging (dedup by chunk ID, re-sort by score) — the infrastructure is correct. Only the call-site is missing.

### 15.5 Env-var contract (normative, as of `b14bca3`)

The following env vars are active in the shipped codebase. This table supplements §2 and takes precedence over the §4/§5 spec items, which describe intent, not final behavior.

| Var | Scope | Default | Effect |
|-----|-------|---------|--------|
| `SEMANTEX_LLM_MODEL` | GenaiBackend | unset (LLM disabled) | Required to activate Phase 1. Model name passed to `rust-genai`. |
| `SEMANTEX_LLM_PROVIDER` | GenaiBackend | inferred from model name | Provider override (`ollama`, `anthropic`, `openai`, `gemini`, `groq`, `cohere`, `deepseek`, `xai`, `fireworks`, `together`, `openai_resp`). |
| `SEMANTEX_LLM_ENDPOINT` | GenaiBackend | provider default | OpenAI-compatible base URL (Ollama, LiteLLM, vLLM, LM Studio). |
| `SEMANTEX_LLM_BACKEND` | SubscriptionCliBackend | unset (Phase 2 disabled) | `cli:claude`, `cli:codex`, `cli:auto`, or `cli:antigravity` (last fails at startup). |
| `SEMANTEX_LLM_CLI_BINARY` | SubscriptionCliBackend | PATH lookup | Absolute path override; only honored when basename matches the kind's CLI name. |
| `SEMANTEX_LLM_CLASSIFY_TIMEOUT_MS` | Both backends | 8000 | Classifier call timeout in milliseconds. Used by daemon and `llm-status`. |

---

## Appendix A — Source files in this conversation (for orientation)

- This spec: `docs/superpowers/specs/2026-05-27-semantex-llm-integration.md`
- v0.5/v0.6 spec (LLM Items 9, 12 marked out-of-scope as of this spec): `docs/superpowers/specs/2026-05-26-semantex-v0.3.1-v0.5-refactor.md`
- Release sequence: `docs/RELEASE-SEQUENCE-2026-05.md`
- Prior LLM scrub: see git log for the commit removing `crates/semantex-core/src/llm/` and `--features local-llm`.

## Appendix B — Glossary

- **HyDE** (Hypothetical Document Embeddings): retrieval technique where the LLM writes a synthetic document/code snippet that *would* answer the user's question, then that snippet is embedded and used for vector search. Closes the NL-query → code semantic gap.
- **`genai`**: jeremychone/rust-genai, a Rust unified LLM-client library covering 25+ providers via native protocol + OpenAI-compatible fallback.
- **`SubscriptionCli`**: this spec's term for shelling out to a coding-agent CLI (`claude`, `codex`, `antigravity`) so the user's existing subscription quota is consumed instead of an API key.
- **MCP sampling**: the `sampling/createMessage` request in the Model Context Protocol. **Deprecated as of MCP `2026-07-28` (SEP-2577).** Not adopted by this spec.
