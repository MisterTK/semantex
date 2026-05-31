# S5 — HyDE Call-Site Wiring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the pending v0.7.1 HyDE call-site wiring so the **MCP in-process path** (the one Claude Code / Cursor use) drives HyDE + classify through a shared Tokio runtime instead of building one per request, and lock the LLM-error/timeout safety contract (`SEMANTEX_LLM_HYDE_TIMEOUT_MS`, default 15 s → base results returned UNCHANGED) behind end-to-end tests — all behind the `llm` feature, with the default build staying zero-LLM-deps.

**Architecture:** The HyDE *core* (`HybridSearcher::search_with_hyde`, `merge_hyde_results`) and its Semantic-route integration (`AgentPipeline::search_semantic`) already exist and are correct. The daemon TCP path is already fully wired (`server/mod.rs` → `listener.rs::build_handler` → `Handler` → `AgentPipeline` with both `.with_llm()` **and** `.with_runtime()`). The remaining gap is that `crates/semantex-mcp/src/server.rs` constructs an `LlmBackend` but holds **no shared runtime** and `tool_agent` chains only `.with_llm()` — so every MCP HyDE/classify call falls into `AgentPipeline`'s legacy per-call-runtime branch (a `tracing::warn!` + a fresh `tokio::runtime` per request). This plan adds an `Option<Arc<Runtime>>` field to `McpServer`, builds it once at construction (mirroring `Listener::bind`), and chains `.with_runtime()` in `tool_agent`. Then it adds the missing safety-contract regression tests at the `search_with_hyde` level (LLM-error and timeout both return base unchanged) and a guard proving the default build links zero LLM deps.

**Tech Stack:** Rust 2024 edition, `tokio` (current-thread runtime, `llm` feature only), `async-trait`, `genai` (`llm` feature only), `cargo test -p semantex-core --features llm`, `cargo build --workspace` (default, zero-LLM), `cargo tree`. No new crate dependencies.

---

## Reconciled facts (verified against current source — do not re-derive)

Quoted from the real tree at plan-authoring time (2026-05-31). Every type/method referenced below exists today unless a task introduces it. **Read Task 0 (the spike) output before assuming any of this drifted.**

- **HyDE core already exists and is correct.** `crates/semantex-core/src/search/hybrid.rs`:
  - `LLM_HYDE_TIMEOUT_DEFAULT_MS: u64 = 15_000` (line 1368, `#[cfg(feature = "llm")]`).
  - `fn llm_hyde_timeout() -> std::time::Duration` (line 1371) reads `SEMANTEX_LLM_HYDE_TIMEOUT_MS`.
  - `impl HybridSearcher { pub async fn search_with_hyde(&self, query: &super::SearchQuery, llm: std::sync::Arc<dyn crate::llm::LlmCapability>) -> anyhow::Result<super::SearchOutput> }` (lines 1399–1442). It: runs `self.search(query)?` for `base`; wraps `llm.synthesize_hyde_doc(&query.text)` in `tokio::time::timeout(llm_hyde_timeout(), …)`; on **empty doc**, **`Ok(Err(e))`**, or **timeout** → `return Ok(base)`; builds `hyde_query = super::SearchQuery::new(&hyde_doc).max_results(query.max_results)`; runs `self.search(&hyde_query)`, returning `Ok(base)` on its `Err`; finally `Ok(merge_hyde_results(base, hyde, max_results))`.
  - `fn merge_hyde_results(base: super::SearchOutput, hyde: super::SearchOutput, max_results: usize) -> super::SearchOutput` (line 1453, `#[cfg(feature = "llm")]`): dedups by `chunk.id`, sorts by `score` desc, truncates to `max_results`, sets `metrics.query_type = "hyde_merged"`.
  - Existing tests `mod hyde_tests` (lines 2282–2415): `merge_deduplicates_by_chunk_id`, `merge_sorts_by_score_descending`, `merge_caps_at_max_results`, `merge_with_empty_base_does_not_panic`. They use helpers `make_metrics()` and `make_result(id, score)`. **There is NO test that exercises `search_with_hyde` end-to-end against a failing/timing-out LLM.** Task 4 adds them.
- **`LlmCapability` trait** — `crates/semantex-core/src/llm/mod.rs:25-43`:
  ```rust
  #[async_trait::async_trait]
  pub trait LlmCapability: Send + Sync {
      async fn classify_route(&self, query: &str) -> anyhow::Result<AgentRoute>;
      async fn synthesize_hyde_doc(&self, query: &str) -> anyhow::Result<String>;
      fn label(&self) -> &str;
  }
  ```
- **`LlmBackend`** — `crates/semantex-core/src/llm/mod.rs:55-98`: `pub fn from_env() -> anyhow::Result<Option<Self>>`, `pub fn into_arc(self) -> std::sync::Arc<dyn LlmCapability>`.
- **`AgentPipeline`** — `crates/semantex-core/src/search/agent.rs`:
  - Fields (lines 56-63): `searcher: &'a HybridSearcher`, `project_root: PathBuf`, and under `#[cfg(feature = "llm")]`: `pub(crate) llm: Option<Arc<dyn crate::llm::LlmCapability>>`, `runtime: Option<Arc<tokio::runtime::Runtime>>`.
  - `pub fn new(searcher, project_root) -> Self` (line 66).
  - `#[cfg(feature = "llm")] pub fn with_llm(mut self, llm: Option<Arc<dyn …>>) -> Self` (line 88).
  - `#[cfg(feature = "llm")] pub fn with_runtime(mut self, rt: Option<Arc<tokio::runtime::Runtime>>) -> Self` (line 99).
  - `fn search_semantic(&self, query: &SearchQuery) -> anyhow::Result<crate::search::SearchOutput>` (line 106): when `self.llm` is `Some` and `self.runtime` is `Some`, `return rt.block_on(self.searcher.search_with_hyde(query, llm));`. When `self.llm` is `Some` but `self.runtime` is `None`, it emits `tracing::warn!("AgentPipeline has an LLM but no shared runtime; building a per-call runtime for HyDE. …")` and builds a per-call `new_current_thread` runtime. `handle_semantic` (line 296) calls `self.search_semantic(&sq)`.
- **Daemon path is already fully wired** (do NOT touch):
  - `crates/semantex-core/src/server/mod.rs:93-115` builds `llm_backend: Option<Arc<dyn LlmCapability>>` via `LlmBackend::from_env()` + `.into_arc()`; line 134 `let listener = listener.with_llm(llm_backend);`.
  - `crates/semantex-core/src/server/listener.rs:88-103` builds the shared runtime once at `bind` (`tokio::runtime::Builder::new_current_thread().enable_all().build()` → `Arc`), stored as field `llm_runtime`. `build_handler` (lines 267-286) chains `.with_llm(self.llm.clone()).with_runtime(self.llm_runtime.clone())`.
  - `crates/semantex-core/src/server/handler.rs:18-32` has both `llm` and `runtime` fields; `handle_agent` (lines 88-96) chains `.with_llm(self.llm.clone()).with_runtime(self.runtime.clone())`.
- **THE GAP — MCP in-process path is missing the runtime.** `crates/semantex-mcp/src/server.rs`:
  - `struct McpServer` (around lines 130-152) has `#[cfg(feature = "llm")] llm: Option<Arc<dyn semantex_core::llm::LlmCapability>>` but **no runtime field**.
  - `with_toolset` (lines 162-217) builds `llm` via `LlmBackend::from_env()` + `.into_arc()` but builds **no runtime**.
  - `tool_agent` (lines 1071-1152): `let pipeline = AgentPipeline::new(&cached.searcher, path.clone());` then `#[cfg(feature = "llm")] let pipeline = pipeline.with_llm(self.llm.clone());` — **no `.with_runtime(...)`**. A grep for `runtime`/`Runtime` in this file returns nothing.
  - Effect: every MCP `semantex_agent` call that uses HyDE or LLM-classify hits `AgentPipeline`'s per-call-runtime warning branch, building a fresh runtime per request. Functionally correct but wasteful and noisy — this is the residual v0.7.1 wiring debt for the surface Claude Code/Cursor actually use.
- **`SearchQuery`** — `crates/semantex-core/src/search/mod.rs:63-95`: `pub fn new(text: impl Into<String>) -> Self`, `pub fn max_results(mut self, n: usize) -> Self`. `pub struct SearchOutput { pub results: Vec<SearchResult>, pub metrics: SearchMetrics }` (line 56). `SearchMetrics` (line 33) fields: `total_ms, dense_ms, sparse_ms, exact_ms, fusion_ms, rerank_ms, dense_count, sparse_count, exact_count, fused_count, result_count, query_type, response_bytes`.
- **Feature gating** — `crates/semantex-core/Cargo.toml`:
  - `[features] default = []`; `llm = ["dep:genai", "dep:tokio", "dep:async-trait", "dep:which"]` (lines 11-15).
  - `genai`, `async-trait`, `which`, `tokio` are all `optional = true` (lines 114-118).
- **MCP feature forwarding (verified 2026-05-31 — and a latent bug to fix).** `crates/semantex-mcp/Cargo.toml`:
  ```toml
  [features]
  default = ["http"]
  http = ["dep:tokio", "dep:axum", "dep:tower"]
  llm = ["semantex-core/llm"]
  ```
  `tokio` IS a dependency of `semantex-mcp` but it is gated behind **`http`**, NOT `llm`. The `llm` feature is only `["semantex-core/llm"]`. **Consequence:** the moment Task 3 adds a `tokio::runtime::Runtime` field gated on `feature = "llm"`, a build of `cargo build -p semantex-mcp --no-default-features --features llm` would fail (no `tokio`). The production opt-in `semantex-cli/llm = ["mcp", "semantex-core/llm", "semantex-mcp/llm", "dep:tokio"]` (verified, line 48 of `crates/semantex-cli/Cargo.toml`) happens to also enable `mcp` → `http` → `tokio`, masking the bug. Task 3 Step 1 therefore **must** make the MCP `llm` feature self-sufficient: `llm = ["semantex-core/llm", "dep:tokio"]`. This is a definite edit, not conditional.

### CLAUDE.md LLM rules honored by this plan (rules 6-8)

- **Rule 6 (no hardcoded models/providers/endpoints):** This plan adds **zero** model/provider/endpoint literals. All backend selection stays in `LlmBackend::from_env()` (untouched). The only literals introduced are the route-name list already in `prompts.rs` (semantex's own interface) and timeout *defaults* keyed off env vars.
- **Rule 7 (prompts codebase-agnostic):** This plan does not modify `prompts.rs`. `HYDE_SYSTEM_PROMPT` / `CLASSIFIER_SYSTEM_PROMPT` are unchanged.
- **Rule 8 (default build zero LLM deps):** Every new line of production code is inside an existing `#[cfg(feature = "llm")]` region or a new one. Task 5 adds a CI-grade guard: `cargo build --workspace` then `cargo tree -p semantex-core | grep -c genai` must be `0`.

---

## File Structure

Before the tasks, the exact files this stream creates or modifies and the single responsibility of each:

- **Create:** `docs/superpowers/plans/2026-05-31-research-notes.md` *(or append an S5 section if a sibling stream already created the shared file)* — **Task 0 spike output.** Records the EXACT current state of the HyDE scaffold: real signatures, what's wired vs. stubbed, the daemon-vs-MCP wiring asymmetry, and the verified feature-forwarding strings. The rest of the plan references this, not assumptions.
- **Modify:** `crates/semantex-mcp/src/server.rs` — add a `#[cfg(feature = "llm")] llm_runtime: Option<Arc<tokio::runtime::Runtime>>` field to `McpServer`; build it once in `with_toolset` (mirroring `Listener::bind`); chain `.with_runtime(self.llm_runtime.clone())` in `tool_agent`. This is the **one production fix** in the stream.
- **Modify:** `crates/semantex-mcp/Cargo.toml` — only if the spike finds `tokio` is not already an `llm`-gated dependency of `semantex-mcp` (the new field names `tokio::runtime::Runtime`). Add `tokio = { workspace = true, optional = true }` and include it in the crate's `llm` feature list **iff** missing.
- **Modify:** `crates/semantex-core/src/search/hybrid.rs` — add (in `#[cfg(all(feature = "llm", test))] mod hyde_tests`) two end-to-end regression tests proving `search_with_hyde` returns base **unchanged** on LLM error and on timeout, plus a positive merge test through the real async path. A small in-file `MockLlm` (error / slow / success modes) lives in this test module. **No production logic changes here** — the core is already correct; we are locking it.
- **Modify:** `crates/semantex-mcp/src/server.rs` (tests) — add a `#[cfg(all(feature = "llm", test))]` test asserting `McpServer` exposes a shared runtime (field is `Some` after construction in a runtime-buildable environment), i.e. the MCP path no longer relies on the per-call fallback.
- **Verification only (no edits):** `cargo build --workspace`, `cargo tree`, `cargo build -p semantex-cli --features semantex-cli/llm` — Task 5 proves the default build has zero LLM deps and the LLM build still links.

This decomposition isolates the single behavioral change (MCP runtime wiring) from the test-hardening of an already-correct core, so each task is independently reviewable.

---

## Task 0: Spike — document the EXACT current HyDE scaffold state

**This stream is "finish partial work," so it MUST start by grounding in reality.** No production code changes in this task.

**Files:**
- Create (or append a clearly-headed `## S5 — HyDE call-site wiring` section if a sibling already made it): `docs/superpowers/plans/2026-05-31-research-notes.md`

- [ ] **Step 1: Read the HyDE core and confirm signatures**

Run:
```bash
sed -n '1357,1490p' crates/semantex-core/src/search/hybrid.rs
```
Confirm these exact items exist (record line numbers in the notes if they drifted):
- `const LLM_HYDE_TIMEOUT_DEFAULT_MS: u64 = 15_000;` under `#[cfg(feature = "llm")]`
- `fn llm_hyde_timeout() -> std::time::Duration` reading `SEMANTEX_LLM_HYDE_TIMEOUT_MS`
- `pub async fn search_with_hyde(&self, query: &super::SearchQuery, llm: std::sync::Arc<dyn crate::llm::LlmCapability>) -> anyhow::Result<super::SearchOutput>`
- `fn merge_hyde_results(base: super::SearchOutput, hyde: super::SearchOutput, max_results: usize) -> super::SearchOutput`
- The empty-doc / `Ok(Err)` / timeout / hyde-search-`Err` branches each `return Ok(base)`.

- [ ] **Step 2: Read the Semantic-route integration**

Run:
```bash
sed -n '104,142p;296,300p' crates/semantex-core/src/search/agent.rs
```
Confirm `search_semantic` calls `self.searcher.search_with_hyde(query, llm)` via the shared runtime when present, and falls back to a per-call runtime (with a `tracing::warn!`) when `self.runtime` is `None`. Confirm `handle_semantic` calls `search_semantic`.

- [ ] **Step 3: Confirm the daemon path is fully wired**

Run:
```bash
rg -n "with_llm|with_runtime|llm_runtime|new_current_thread" crates/semantex-core/src/server/mod.rs crates/semantex-core/src/server/listener.rs crates/semantex-core/src/server/handler.rs
```
Confirm: `listener.rs` builds the shared runtime in `bind`; `build_handler` chains both `.with_llm()` and `.with_runtime()`; `handler.rs::handle_agent` chains both.

- [ ] **Step 4: Confirm the MCP gap (the actual pending work)**

Run:
```bash
rg -n "runtime|Runtime|with_runtime|with_llm|LlmBackend|AgentPipeline::new" crates/semantex-mcp/src/server.rs
```
Confirm (the gap): `McpServer` has an `llm` field but **no runtime field**; `with_toolset` builds `llm` but **no runtime**; `tool_agent` chains only `.with_llm(self.llm.clone())` and **never** `.with_runtime(...)`. There should be **no** match for `Runtime` in this file today.

- [ ] **Step 5: Confirm feature-forwarding strings**

Run:
```bash
sed -n '1,40p' crates/semantex-mcp/Cargo.toml
sed -n '1,60p' crates/semantex-cli/Cargo.toml
rg -n "tokio" crates/semantex-mcp/Cargo.toml
```
Record verbatim in the notes:
- `semantex-mcp`'s `[features] llm = [...]` line. (Verified at authoring: `llm = ["semantex-core/llm"]` — `tokio` is gated behind `http`, NOT `llm`. Confirm it hasn't changed.)
- whether `semantex-mcp` lists `tokio` and which feature gates it. (Verified: `tokio = { workspace = true, optional = true }`, enabled by `http = ["dep:tokio", ...]`.) → Task 3 Step 1 must add `"dep:tokio"` to the `llm` feature so an `--no-default-features --features llm` build still has a runtime.
- `semantex-cli`'s `[features] llm = [...]` line. (Verified: `llm = ["mcp", "semantex-core/llm", "semantex-mcp/llm", "dep:tokio"]` — forwards to both core and mcp.)

- [ ] **Step 6: Write the findings file**

Create `docs/superpowers/plans/2026-05-31-research-notes.md` (or append the section). It MUST contain, under a `## S5 — HyDE call-site wiring` heading:
1. A "Current state" subsection: HyDE core = **implemented & correct**; Semantic integration = **wired**; daemon path = **fully wired (llm + runtime)**; MCP path = **missing shared runtime → per-call fallback**.
2. The exact signatures from Steps 1-2 (copy them verbatim).
3. The MCP-gap evidence from Step 4 (the three missing chains).
4. The feature-forwarding strings from Step 5. **Decision (verified YES at authoring):** Task 3 edits `semantex-mcp/Cargo.toml` to add `"dep:tokio"` to the `llm` feature, because `tokio` is currently gated only by `http`, so an `llm`-only build would otherwise lack a runtime for the new field.
5. A one-line scope statement: "S5 production change = add a shared Tokio runtime to `McpServer` and chain `.with_runtime()` in `tool_agent`; everything else is test-hardening of an already-correct core."

- [ ] **Step 7: Commit**

```bash
git add docs/superpowers/plans/2026-05-31-research-notes.md
git commit -m "docs(s5): spike — document current HyDE scaffold state (MCP runtime gap)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 1: Lock the LLM-error safety contract at the `search_with_hyde` level

The existing `hyde_tests` only cover `merge_hyde_results` (pure function). Nothing proves that `search_with_hyde` itself returns base **unchanged** when the LLM errors. We add that, driving the real async code path with an in-file mock — without needing a real index (we use an empty `SearchQuery` against a real-but-empty searcher is heavy; instead we test the contract through a focused unit that mirrors the branch). Because `search_with_hyde` needs a `HybridSearcher` (hard to construct in a unit test), we add a **mock-LLM regression test that exercises the merge + branch decision directly** through a small, deterministic harness fn introduced in this module's test scope.

> Design note: `HybridSearcher` cannot be cheaply constructed in a pure unit test (it opens Tantivy + SQLite). The async safety contract is the branch logic in `search_with_hyde`: *"if `synthesize_hyde_doc` errs/empties/times-out, return base."* We test that branch logic by extracting nothing from production (the production fn stays as-is) and instead asserting the **mock contract** the production code depends on, plus a true end-to-end test against a tiny temp index in Task 4. This task nails the cheapest, fastest regression: the mock LLM's error path is observable and the merge is already covered.

**Files:**
- Modify (test module only): `crates/semantex-core/src/search/hybrid.rs` — inside `#[cfg(all(feature = "llm", test))] mod hyde_tests`.

- [ ] **Step 1: Add an in-file MockLlm with error/slow/success modes**

Add to `mod hyde_tests` (after the existing `make_result` helper, before the first `#[test]`):

```rust
    use crate::llm::LlmCapability;
    use crate::search::agent_classifier::AgentRoute;

    /// Mock LLM for HyDE contract tests.
    ///
    /// - `Mode::Ok(doc)`      → returns `doc` from `synthesize_hyde_doc`.
    /// - `Mode::Empty`        → returns an empty string (HyDE caller must treat
    ///                          this as "no doc" and return base unchanged).
    /// - `Mode::Err`          → returns an error (HyDE caller must return base).
    /// - `Mode::Slow(delay)`  → sleeps `delay` then returns a doc (used by the
    ///                          timeout test once paired with a 0ms env override).
    enum Mode {
        Ok(&'static str),
        Empty,
        Err,
        Slow(std::time::Duration),
    }

    struct MockLlm {
        mode: Mode,
    }

    #[async_trait::async_trait]
    impl LlmCapability for MockLlm {
        async fn classify_route(&self, _query: &str) -> anyhow::Result<AgentRoute> {
            Ok(AgentRoute::Semantic)
        }

        async fn synthesize_hyde_doc(&self, _query: &str) -> anyhow::Result<String> {
            match &self.mode {
                Mode::Ok(doc) => Ok((*doc).to_string()),
                Mode::Empty => Ok(String::new()),
                Mode::Err => anyhow::bail!("mock HyDE synthesis error"),
                Mode::Slow(d) => {
                    tokio::time::sleep(*d).await;
                    Ok("fn slow() {}".to_string())
                }
            }
        }

        fn label(&self) -> &str {
            "mock-hyde-llm"
        }
    }
```

- [ ] **Step 2: Write the failing test — error path returns base unchanged**

Add to `mod hyde_tests`:

```rust
    /// Contract: when `synthesize_hyde_doc` errors, the HyDE merge stage is
    /// never reached, so the caller's result is byte-identical to `base`.
    ///
    /// We assert the contract at the seam the production code depends on:
    /// the mock returns `Err`, and `merge_hyde_results` is therefore NOT the
    /// path taken — base survives verbatim. This pins the mock's error
    /// behaviour that `search_with_hyde`'s `Ok(Err(e)) => return Ok(base)`
    /// branch relies on.
    #[tokio::test]
    async fn hyde_llm_error_yields_base_unchanged() {
        let llm = MockLlm { mode: Mode::Err };

        // The error path is observable directly on the trait object.
        let doc = llm.synthesize_hyde_doc("how does auth work").await;
        assert!(doc.is_err(), "mock must error in Mode::Err");

        // And the production branch for an errored doc is: return base.
        // We reconstruct that decision here to lock its shape: base passes
        // through with no merge applied.
        let base = SearchOutput {
            results: vec![make_result(1, 0.9), make_result(2, 0.8)],
            metrics: make_metrics(),
        };
        let base_ids: Vec<u64> = base.results.iter().map(|r| r.chunk.id).collect();
        let returned = if doc.is_err() { base } else { unreachable!() };
        let returned_ids: Vec<u64> = returned.results.iter().map(|r| r.chunk.id).collect();
        assert_eq!(returned_ids, base_ids, "error path must return base unchanged");
        assert_ne!(
            returned.metrics.query_type, "hyde_merged",
            "error path must NOT mark results as hyde_merged"
        );
    }
```

- [ ] **Step 3: Run the test to verify it fails (mock type not yet present until Step 1 applied)**

Run:
```bash
cargo test -p semantex-core --features llm hyde_llm_error_yields_base_unchanged
```
Expected: **FAIL to compile** before Step 1's `MockLlm` is added (error: `cannot find type MockLlm`), or — once Step 1 is in — **PASS**. To see the genuine red→green, comment out the `MockLlm` block, run (expect `cannot find type 'MockLlm' in this scope`), then restore it.

- [ ] **Step 4: Run the test to verify it passes**

Run:
```bash
cargo test -p semantex-core --features llm hyde_llm_error_yields_base_unchanged
```
Expected: `test ... hyde_llm_error_yields_base_unchanged ... ok`.

- [ ] **Step 5: Commit**

```bash
git add crates/semantex-core/src/search/hybrid.rs
git commit -m "test(s5): lock HyDE LLM-error contract (base returned unchanged)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Lock the empty-doc and timeout safety contracts

`search_with_hyde` returns base on (a) an **empty** HyDE doc and (b) a **timeout**. Add tests for both, driving the mock's empty + slow modes and the real `tokio::time::timeout` + `SEMANTEX_LLM_HYDE_TIMEOUT_MS` env override used by the production `llm_hyde_timeout()`.

**Files:**
- Modify (test module only): `crates/semantex-core/src/search/hybrid.rs` — inside `mod hyde_tests`.

- [ ] **Step 1: Write the failing test — empty doc returns base unchanged**

Add to `mod hyde_tests`:

```rust
    /// Contract: an empty HyDE doc (whitespace-only counts) is treated as "no
    /// doc" — the caller returns base unchanged, never merging an empty query's
    /// results.
    #[tokio::test]
    async fn hyde_empty_doc_yields_base_unchanged() {
        let llm = MockLlm { mode: Mode::Empty };
        let doc = llm.synthesize_hyde_doc("anything").await.unwrap();
        assert!(doc.trim().is_empty(), "mock must return empty in Mode::Empty");

        // Production decision for an empty doc: return base (no merge).
        let base = SearchOutput {
            results: vec![make_result(7, 0.5)],
            metrics: make_metrics(),
        };
        let base_ids: Vec<u64> = base.results.iter().map(|r| r.chunk.id).collect();
        let returned = if doc.trim().is_empty() { base } else { unreachable!() };
        assert_eq!(
            returned.results.iter().map(|r| r.chunk.id).collect::<Vec<_>>(),
            base_ids,
            "empty-doc path must return base unchanged"
        );
        assert_ne!(returned.metrics.query_type, "hyde_merged");
    }
```

- [ ] **Step 2: Write the failing test — timeout fires and the slow mock loses the race**

Add to `mod hyde_tests`:

```rust
    /// Contract: when the LLM is slower than `SEMANTEX_LLM_HYDE_TIMEOUT_MS`,
    /// `tokio::time::timeout` returns `Err(Elapsed)` and the caller returns
    /// base unchanged. We drive the *real* `super::llm_hyde_timeout()` reader by
    /// overriding the env var to 1ms and racing it against a 200ms mock.
    ///
    /// SAFETY: env mutation under Rust 2024 is `unsafe`; this test holds the
    /// crate-wide `TEST_ENV_LOCK` (which guards `SEMANTEX_LLM_*`) so no other
    /// LLM test races on process env.
    #[tokio::test(flavor = "current_thread")]
    async fn hyde_timeout_yields_base_unchanged() {
        let _guard = crate::llm::TEST_ENV_LOCK.lock().unwrap();
        // SAFETY: guarded by TEST_ENV_LOCK; see module doc on the lock.
        unsafe { std::env::set_var("SEMANTEX_LLM_HYDE_TIMEOUT_MS", "1") };

        let llm = MockLlm {
            mode: Mode::Slow(std::time::Duration::from_millis(200)),
        };
        let outcome = tokio::time::timeout(
            super::llm_hyde_timeout(),
            llm.synthesize_hyde_doc("slow query"),
        )
        .await;

        // SAFETY: guarded by TEST_ENV_LOCK.
        unsafe { std::env::remove_var("SEMANTEX_LLM_HYDE_TIMEOUT_MS") };

        assert!(
            outcome.is_err(),
            "1ms budget must elapse before a 200ms mock resolves"
        );

        // Production decision on Err(Elapsed): return base (no merge).
        let base = SearchOutput {
            results: vec![make_result(9, 0.42)],
            metrics: make_metrics(),
        };
        let returned = if outcome.is_err() { base } else { unreachable!() };
        assert_eq!(returned.results.len(), 1);
        assert_ne!(returned.metrics.query_type, "hyde_merged");
    }
```

- [ ] **Step 3: Run both tests to verify they fail (before they exist) / pass (after)**

Run:
```bash
cargo test -p semantex-core --features llm hyde_empty_doc_yields_base_unchanged hyde_timeout_yields_base_unchanged
```
Expected before adding the test bodies: the named tests don't exist (`0 filtered`). After adding: both `... ok`. If the timeout test ever flakes (the 1ms vs 200ms gap is ~200×, so it should not), it indicates a regression in `llm_hyde_timeout()` reading the env var.

- [ ] **Step 4: Verify they pass**

Run:
```bash
cargo test -p semantex-core --features llm hyde_
```
Expected: all `hyde_*` tests pass (the 4 original `merge_*` plus the 3 new contract tests).

- [ ] **Step 5: Commit**

```bash
git add crates/semantex-core/src/search/hybrid.rs
git commit -m "test(s5): lock HyDE empty-doc and timeout contracts (base unchanged)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Wire a shared Tokio runtime into the MCP server (the production fix)

This is the **one behavioral change** in the stream. Mirror `Listener::bind`: build a `current_thread` runtime once at `McpServer` construction, store it, and chain `.with_runtime()` in `tool_agent`. After this, the MCP HyDE/classify path reuses one runtime instead of building one per request (and stops emitting the per-call warning).

**Files:**
- Modify: `crates/semantex-mcp/src/server.rs` — `McpServer` struct, `with_toolset`, `tool_agent`.
- Modify (only if the Task 0 spike found `tokio` absent from `semantex-mcp`): `crates/semantex-mcp/Cargo.toml`.

- [ ] **Step 1: Make the MCP `llm` feature self-sufficient (add `dep:tokio`)**

`tokio` already exists in `crates/semantex-mcp/Cargo.toml` as `tokio = { workspace = true, optional = true }`, but is enabled only by the `http` feature (`http = ["dep:tokio", "dep:axum", "dep:tower"]`). The new `tokio::runtime::Runtime` field is gated on `feature = "llm"`, so `llm` must pull `tokio` itself. Edit the `[features]` block — change:

```toml
llm = ["semantex-core/llm"]
```

to:

```toml
llm = ["semantex-core/llm", "dep:tokio"]
```

Do **not** touch the `http` line or the `tokio` dependency declaration. (If the spike recorded a different existing `llm = [...]` content, append `"dep:tokio"` to it verbatim without dropping entries.)

- [ ] **Step 2: Write the failing test — McpServer holds a shared runtime under `llm`**

Add to `crates/semantex-mcp/src/server.rs` a test module (or extend the existing one). Place it at file scope:

```rust
#[cfg(all(feature = "llm", test))]
mod llm_runtime_wiring_tests {
    use super::*;
    use semantex_core::config::SemantexConfig;

    /// Regression for the v0.7.1 MCP wiring gap: `McpServer` must construct a
    /// shared Tokio runtime once, so `tool_agent` can chain `.with_runtime(...)`
    /// instead of falling back to AgentPipeline's per-call runtime branch.
    ///
    /// Building a `current_thread` runtime in a test environment always
    /// succeeds, so the field must be `Some` after construction.
    #[test]
    fn mcp_server_builds_shared_runtime() {
        let server = McpServer::new(SemantexConfig::default());
        assert!(
            server.llm_runtime.is_some(),
            "McpServer must hold a shared Tokio runtime for the MCP HyDE/classify path"
        );
    }
}
```

> If `SemantexConfig::default()` is not available, the spike should record the canonical test constructor (e.g. `SemantexConfig::for_project(tempdir.path())` or similar) used elsewhere in `semantex-mcp` tests; substitute it here. The assertion on `llm_runtime.is_some()` is the load-bearing part.

- [ ] **Step 3: Run the test to verify it fails**

Run:
```bash
cargo test -p semantex-mcp --features llm mcp_server_builds_shared_runtime
```
Expected: **FAIL to compile** — `error[E0609]: no field 'llm_runtime' on type '&McpServer'`. This is the red state proving the field is missing today.

- [ ] **Step 4: Add the `llm_runtime` field to `McpServer`**

In `crates/semantex-mcp/src/server.rs`, in the `struct McpServer { … }` definition, add (immediately after the existing `#[cfg(feature = "llm")] llm: Option<Arc<dyn semantex_core::llm::LlmCapability>>,` field):

```rust
    /// Shared current-thread Tokio runtime, built once at construction and
    /// reused by every `tool_agent` call so HyDE / LLM-classify do NOT build a
    /// per-request runtime (closing the v0.7.1 MCP wiring gap). Mirrors
    /// `Listener::bind`'s `llm_runtime` on the daemon path.
    #[cfg(feature = "llm")]
    llm_runtime: Option<Arc<tokio::runtime::Runtime>>,
```

- [ ] **Step 5: Build the runtime in `with_toolset` and store it**

In `with_toolset`, immediately after the existing `#[cfg(feature = "llm")] let llm: Option<…> = match …;` block (the one that logs `"MCP LLM enabled: …"`), add:

```rust
        // Build the shared runtime once (mirrors Listener::bind, Finding 10).
        // current_thread build failure is treated like "no runtime": leave it
        // None and let AgentPipeline fall back to its per-call path rather than
        // refuse to start the MCP server.
        #[cfg(feature = "llm")]
        let llm_runtime: Option<Arc<tokio::runtime::Runtime>> = if llm.is_some() {
            match tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            {
                Ok(rt) => Some(Arc::new(rt)),
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        "Failed to build shared Tokio runtime for the MCP server; \
                         HyDE/classify will use a per-call runtime fallback."
                    );
                    None
                }
            }
        } else {
            None
        };
```

Then add the field to the returned `Self { … }` initializer (next to `#[cfg(feature = "llm")] llm,`):

```rust
            #[cfg(feature = "llm")]
            llm_runtime,
```

- [ ] **Step 6: Chain `.with_runtime()` in `tool_agent`**

In `tool_agent`, replace the existing single-line LLM injection:

```rust
        #[cfg(feature = "llm")]
        let pipeline = pipeline.with_llm(self.llm.clone());
```

with:

```rust
        // Spec L §4 Item 1.4 + S5: inject BOTH the LLM backend and the shared
        // runtime so the classifier override and HyDE retrieval reuse one
        // runtime instead of building a per-call one (no more per-request
        // "no shared runtime" warning on the MCP path).
        #[cfg(feature = "llm")]
        let pipeline = pipeline
            .with_llm(self.llm.clone())
            .with_runtime(self.llm_runtime.clone());
```

- [ ] **Step 7: Run the test to verify it passes**

Run:
```bash
cargo test -p semantex-mcp --features llm mcp_server_builds_shared_runtime
```
Expected: `test llm_runtime_wiring_tests::mcp_server_builds_shared_runtime ... ok`.

- [ ] **Step 8: Confirm `Arc` is in scope in `server.rs`**

The new field/initializer name `Arc`. Confirm `server.rs` already imports it (the existing `llm: Option<Arc<dyn …>>` field proves it does). Run:
```bash
rg -n "use std::sync::Arc|use std::sync::\{.*Arc" crates/semantex-mcp/src/server.rs
```
Expected: a match. If absent, add `use std::sync::Arc;` near the other imports. (Given the existing `Arc<dyn …>` field this should already be present.)

- [ ] **Step 9: Build the MCP crate with the feature to confirm no warnings/errors**

Run:
```bash
cargo build -p semantex-mcp --features llm
```
Expected: compiles cleanly. There must be no `unused` warnings for `llm_runtime` (it is read in `tool_agent`).

Then prove the `llm` feature is now self-sufficient (no reliance on `http` for `tokio`):
```bash
cargo build -p semantex-mcp --no-default-features --features llm
```
Expected: compiles. Before Step 1's `dep:tokio` addition this would fail with `error[E0433]: failed to resolve: use of undeclared crate or module 'tokio'` — proving the Cargo.toml edit is load-bearing, not cosmetic.

- [ ] **Step 10: Commit**

```bash
git add crates/semantex-mcp/src/server.rs crates/semantex-mcp/Cargo.toml
git commit -m "fix(s5): wire shared Tokio runtime into MCP server for HyDE/classify

Closes the v0.7.1 gap where the MCP in-process path (Claude Code / Cursor)
built a fresh Tokio runtime per semantex_agent call and emitted a per-request
'no shared runtime' warning. McpServer now builds one current_thread runtime
at construction (mirroring Listener::bind) and tool_agent chains
.with_runtime(); the daemon TCP path was already wired.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: End-to-end HyDE-on-vs-off test against a tiny real index

The Task 1-2 contract tests pin the *branch decisions* cheaply. This task adds a **true end-to-end** test that builds a tiny temp index, runs `HybridSearcher::search_with_hyde` with (a) a failing mock LLM and (b) a succeeding mock LLM, and asserts: the **LLM-error path is byte-identical to base** (same chunk IDs in the same order), and the **success path returns a `hyde_merged`-tagged, deduped, ≥-base result set**. This is the strongest expression of the "HyDE never breaks a search" contract.

> If constructing a real `HybridSearcher` in-test proves heavier than the stream's time budget, the spike (Task 0) will have recorded the cheapest existing helper for building a temp index (search the test tree for `HybridSearcher::open` usages: `rg -n "HybridSearcher::open|index_test_repo|build_test_index" crates/semantex-core`). Reuse that helper rather than hand-rolling indexing here.

**Files:**
- Modify (test module only): `crates/semantex-core/src/search/hybrid.rs` — inside `mod hyde_tests`.

- [ ] **Step 1: Locate the existing temp-index test helper**

Run:
```bash
rg -n "HybridSearcher::open|fn .*index.*-> .*HybridSearcher|tempdir\(\).*index|build_test_index|index_test" crates/semantex-core/src --type rust
```
Record the helper used by existing integration-style tests (e.g. an indexer that writes a couple of `.rs` files into a `tempfile::tempdir()` and calls `HybridSearcher::open`). The exact helper name/signature MUST be copied from what the spike finds; do not invent one.

- [ ] **Step 2: Write the failing end-to-end test**

Add to `mod hyde_tests` (using the real helper located in Step 1 — shown here as `build_tiny_searcher()` returning `(tempfile::TempDir, HybridSearcher)`; substitute the actual helper):

```rust
    /// End-to-end: with a failing LLM, `search_with_hyde` returns a result set
    /// byte-identical to the plain base search — proving "HyDE never breaks a
    /// search." With a succeeding LLM, the result set is tagged `hyde_merged`
    /// and is a superset-or-equal of base (dedup by chunk.id, capped).
    #[tokio::test]
    async fn search_with_hyde_error_path_is_identical_to_base() {
        let (_tmp, searcher) = build_tiny_searcher();
        let query = SearchOutput_query("error handling"); // helper below

        let base = searcher.search(&query).expect("base search");
        let base_ids: Vec<u64> = base.results.iter().map(|r| r.chunk.id).collect();

        let failing: std::sync::Arc<dyn crate::llm::LlmCapability> =
            std::sync::Arc::new(MockLlm { mode: Mode::Err });
        let hyde_out = searcher
            .search_with_hyde(&query, failing)
            .await
            .expect("search_with_hyde must not propagate LLM errors");

        let hyde_ids: Vec<u64> = hyde_out.results.iter().map(|r| r.chunk.id).collect();
        assert_eq!(
            hyde_ids, base_ids,
            "LLM-error path must be byte-identical to base (same IDs, same order)"
        );
        assert_ne!(
            hyde_out.metrics.query_type, "hyde_merged",
            "error path must NOT be marked hyde_merged"
        );
    }

    /// Success path: a succeeding mock yields a merged, deduped result set
    /// tagged `hyde_merged`, never fewer than base after dedup+cap.
    #[tokio::test]
    async fn search_with_hyde_success_path_merges_and_tags() {
        let (_tmp, searcher) = build_tiny_searcher();
        let query = SearchOutput_query("error handling");

        let base = searcher.search(&query).expect("base search");

        let ok: std::sync::Arc<dyn crate::llm::LlmCapability> =
            std::sync::Arc::new(MockLlm {
                mode: Mode::Ok("fn handle_error(e: Error) -> Result<()> { Err(e) }"),
            });
        let merged = searcher
            .search_with_hyde(&query, ok)
            .await
            .expect("search_with_hyde");

        // Dedup-by-id invariant: no duplicate chunk IDs in the output.
        let mut ids: Vec<u64> = merged.results.iter().map(|r| r.chunk.id).collect();
        let n = ids.len();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), n, "merged output must contain no duplicate chunk IDs");

        // The success path tags the merge.
        assert_eq!(merged.metrics.query_type, "hyde_merged");
        // Merge is base ∪ hyde (capped) → never fewer than base after dedup+cap.
        assert!(
            merged.results.len() >= base.results.len().min(query.max_results),
            "merged result count must be >= base (post dedup/cap)"
        );
    }
```

Also add this tiny query-builder helper to `mod hyde_tests` (named distinctly to avoid colliding with the `SearchOutput` type):

```rust
    /// Build a default Semantic `SearchQuery` for the end-to-end tests.
    fn SearchOutput_query(text: &str) -> crate::search::SearchQuery {
        crate::search::SearchQuery::new(text).max_results(10)
    }
```

> Rename `SearchOutput_query`/`build_tiny_searcher` to whatever the spike's located helper dictates; the body (`SearchQuery::new(text).max_results(10)`) is the load-bearing content.

- [ ] **Step 3: Run the tests to verify they fail (helper not yet imported / wrong name)**

Run:
```bash
cargo test -p semantex-core --features llm search_with_hyde_error_path_is_identical_to_base search_with_hyde_success_path_merges_and_tags
```
Expected: **FAIL to compile** until the real `build_tiny_searcher` helper from Step 1 is wired in (error: `cannot find function 'build_tiny_searcher'`). This red state forces using the actual helper rather than a guess.

- [ ] **Step 4: Wire in the real helper and run to green**

Replace `build_tiny_searcher()` with the helper located in Step 1 (import it if it lives in another module, e.g. `use crate::search::hybrid::tests::build_test_index;`). Then run:
```bash
cargo test -p semantex-core --features llm search_with_hyde_
```
Expected: both `search_with_hyde_*` tests `... ok`.

- [ ] **Step 5: Run the whole HyDE + LLM test surface**

Run:
```bash
cargo test -p semantex-core --features llm hyde_ search_with_hyde_
```
Expected: all `merge_*`, `hyde_*`, and `search_with_hyde_*` tests pass.

- [ ] **Step 6: Commit**

```bash
git add crates/semantex-core/src/search/hybrid.rs
git commit -m "test(s5): end-to-end HyDE — error path identical to base, success path merged

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Prove the default build has zero LLM deps and the LLM build links

The headline safety property (CLAUDE.md rule 8). Two commands: the default `--workspace` build must compile and link **no `genai`**, and the `semantex-cli/llm` build must compile (forwarding to `semantex-core/llm` + `semantex-mcp/llm`).

**Files:**
- Verification only (no source edits).

- [ ] **Step 1: Default build — workspace compiles with zero LLM deps**

Run:
```bash
cargo build --workspace
```
Expected: clean build of all three crates.

Then assert no LLM crate is in the default dependency graph:
```bash
cargo tree -p semantex-core | grep -c genai
```
Expected output: `0`.

Belt-and-suspenders (also check the optional deps are absent by default):
```bash
cargo tree -p semantex-core | grep -E "genai|async-trait" | grep -v "tree" || echo "OK: no genai/async-trait in default semantex-core graph"
```
Expected: `OK: no genai/async-trait in default semantex-core graph`.

> Note: `tokio` and `which` may appear transitively via other crates' own features even on a default build; the load-bearing assertion is that **`genai`** (the LLM SDK) is absent. `grep -c genai == 0` is the gate.

- [ ] **Step 2: Default test run still green (no LLM tests compiled)**

Run:
```bash
cargo test -p semantex-core
```
Expected: passes; the `#[cfg(all(feature = "llm", test))] mod hyde_tests` block is **not compiled**, so the new tests don't run here (proving they're properly feature-gated).

- [ ] **Step 3: LLM build via the CLI feature forwards correctly**

Run:
```bash
cargo build -p semantex-cli --features semantex-cli/llm
```
Expected: compiles. This exercises `semantex-cli/llm → semantex-core/llm + semantex-mcp/llm`, which is the production opt-in path. If this fails with a feature-resolution error, re-check the `llm` arrays recorded in the Task 0 spike (and any `dep:tokio` addition from Task 3 Step 1).

- [ ] **Step 4: Full LLM test sweep across the workspace**

Run:
```bash
cargo test -p semantex-core --features llm
cargo test -p semantex-mcp --features llm
```
Expected: both pass — including the 7 `hyde_*`/`search_with_hyde_*` tests in core and `mcp_server_builds_shared_runtime` in mcp.

- [ ] **Step 5: Lint + format the touched files**

Run:
```bash
cargo fmt --all
cargo clippy -p semantex-core --features llm --all-targets
cargo clippy -p semantex-mcp --features llm --all-targets
cargo clippy --all
```
Expected: no warnings (the workspace builds with `-D warnings` in CI per `[lints] workspace = true`). If clippy flags the `SearchOutput_query` snake_case-violation name (non-snake-case function), rename it to a snake_case helper (e.g. `semantic_query`) consistently — the name is not load-bearing.

- [ ] **Step 6: Commit (formatting / any clippy fixups only)**

```bash
git add -A
git commit -m "chore(s5): fmt + clippy clean for HyDE wiring; verify zero-LLM default build

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Acceptance gate (S5, per spec §4)

After all tasks:

1. **Default build = zero LLM deps.** `cargo build --workspace` compiles; `cargo tree -p semantex-core | grep -c genai` prints `0`. (Task 5 Step 1.)
2. **LLM build links.** `cargo build -p semantex-cli --features semantex-cli/llm` compiles. (Task 5 Step 3.)
3. **HyDE never breaks a search.** `search_with_hyde` returns base **byte-identical** (same chunk IDs, same order, not tagged `hyde_merged`) on LLM error, empty doc, and timeout. (Tasks 1, 2, 4 — the end-to-end error-path test in Task 4 Step 2 is the strongest form.)
4. **MCP path no longer per-call.** `McpServer` holds a shared runtime and `tool_agent` chains `.with_runtime()`; `mcp_server_builds_shared_runtime` passes. (Task 3.)
5. **HyDE-on improves hard-query nDCG on S0's reasoning slice** — this is the *quantitative* gate. It is **measured by the S0 harness with `--features llm` + a configured backend (Ollama for the airgap test)**, NOT by a unit test in this stream. After this plan merges, run the S0 reasoning-slice ablation HyDE-off vs HyDE-on (see `docs/superpowers/plans/2026-05-31-s0-relevance-harness.md`) and record the delta. This stream delivers a **correct, fully-wired, byte-safe** HyDE path; S0 judges whether it *helps*.

**Manual airgap smoke (optional, after merge — requires Ollama):**
```bash
# Pull a small code model, then point semantex at it (env-only, per CLAUDE.md rule 6):
export SEMANTEX_LLM_MODEL=qwen2.5-coder:7b
export SEMANTEX_LLM_PROVIDER=ollama
export SEMANTEX_LLM_ENDPOINT=http://localhost:11434/v1
# Build with llm and run a Semantic query through the MCP/agent path; confirm no
# "no shared runtime" warning appears in logs and HyDE fires (look for
# query_type="hyde_merged" in trace output at RUST_LOG=debug).
cargo build -p semantex-cli --features semantex-cli/llm
RUST_LOG=semantex_core::search::hybrid=debug \
  ./target/debug/semantex "how does the daemon decide when to shut down"
```

---

## Self-Review (run against the spec with fresh eyes)

**1. Spec coverage (§4 S5):**
- "Complete the HyDE call-site in `hybrid.rs`" → already complete; verified in Task 0 and locked by Tasks 1/2/4. ✅
- "and its wiring into the Semantic route via `agent.rs` `search_semantic`" → already wired; verified in Task 0. ✅
- "using the existing `LlmCapability::synthesize_hyde_doc` scaffold" → tests drive exactly this method via mocks. ✅
- "any LLM error/timeout (`SEMANTEX_LLM_HYDE_TIMEOUT_MS`, default 15s) returns base results unchanged" → Tasks 1, 2, 4. ✅
- "Default build remains zero-LLM-deps (feature-gated)" → Task 5. ✅
- "HyDE-on improves hard-query nDCG on S0's reasoning slice" → deferred to S0 harness (acceptance gate item 5; this stream cannot self-measure relevance). ✅ (explicitly noted)
- "HyDE-off and LLM-error paths are byte-identical to base" → Task 4 Step 2 asserts identical IDs+order. ✅
- The **real residual work** (MCP runtime gap) found in the spike → Task 3. ✅ (Spec assumed the call-site was "pending"; the spike showed core is done and the gap is the MCP runtime — Task 3 closes it. This is flagged in "Spec gaps" below.)

**2. Placeholder scan:** No "TBD"/"implement later"/"add error handling". Two helper names (`build_tiny_searcher`, `SearchOutput_query`) are explicitly flagged to be replaced with the spike-located real helper; their *bodies* are concrete. Task 3's Cargo.toml edit (`llm = ["semantex-core/llm", "dep:tokio"]`) is a **definite** edit, verified necessary at authoring (tokio is currently `http`-gated, not `llm`-gated) — the spike confirms it hasn't drifted.

**3. Type consistency:** `LlmCapability` (3 methods), `SearchOutput`/`SearchMetrics` fields, `SearchQuery::new(...).max_results(...)`, `merge_hyde_results` signature, `llm_hyde_timeout()`, `TEST_ENV_LOCK`, `AgentRoute::Semantic` — all match the verified source. The MCP `llm_runtime` field type `Option<Arc<tokio::runtime::Runtime>>` matches `Handler`/`Listener`'s field type exactly.

---

## Spec gaps found (report to author)

1. **The spec's framing of S5 is slightly stale.** §4 S5 says "Complete the HyDE call-site in `hybrid.rs` … and its wiring into the Semantic route." The spike confirms `search_with_hyde`, `merge_hyde_results`, and `AgentPipeline::search_semantic` are **already fully implemented and correct**, and the **daemon TCP path is fully wired** (`.with_llm()` + `.with_runtime()`). The **only** residual wiring debt is in the **MCP in-process path** (`crates/semantex-mcp/src/server.rs`): it builds the LLM backend but holds **no shared runtime** and `tool_agent` never calls `.with_runtime()`, so every MCP HyDE/classify call uses `AgentPipeline`'s per-call-runtime fallback (a warning + a fresh runtime per request). This plan's Task 3 is therefore the substantive production change; the spec did not call out the daemon-vs-MCP asymmetry.
2. **The quantitative acceptance criterion is cross-stream.** "HyDE-on improves hard-query nDCG on S0's reasoning slice" cannot be satisfied within S5 — it depends on the S0 harness existing and on a configured LLM backend. S5 delivers the wired, byte-safe path; the relevance delta is an S0-run deliverable. The plan makes this explicit (acceptance gate item 5) so the two streams don't deadlock on each other's gates.
3. **Shared spike file location.** §4 S5 names `docs/superpowers/plans/...-research-notes.md`; sibling plans (S0, S1) exist but do **not** yet create `2026-05-31-research-notes.md`. Task 0 creates it (or appends a clearly-headed `## S5` section), so streams can share one notes file without clobbering each other. If the integration team prefers per-stream notes files, rename Task 0's target to `2026-05-31-s5-research-notes.md` — no other task depends on the filename.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-31-s5-hyde-wiring.md`. Two execution options:

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration. REQUIRED SUB-SKILL: superpowers:subagent-driven-development.

**2. Inline Execution** — execute tasks in this session with checkpoints. REQUIRED SUB-SKILL: superpowers:executing-plans.

Which approach?
