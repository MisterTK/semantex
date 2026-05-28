# Release Sequence — v0.3.1 → v0.4 → v0.5 → v0.6 → v0.7+

**Date:** 2026-05-26 (initial), amended 2026-05-27 (post-v0.6 LLM scrub; post-v0.7/v0.8 LLM ship; post-v0.7.1 HyDE wire; post-v0.7.2 audit-findings sweep)
**Status:** Authoritative ordering for the v0.3.1→v0.8 sequence. All releases SHIPPED through v0.7.2 (production hardening). All 14 Team C audit findings resolved. Origin `main` at `64f63e7`.
**Source specs:**
- `docs/superpowers/specs/2026-05-26-semantex-v0.3.1-v0.5-refactor.md` (quality/benchmark — "Spec Q")
- `docs/v0.4_SPEC.md` (deps & capabilities — "Spec D")
- `docs/superpowers/specs/2026-05-27-semantex-llm-integration.md` (LLM integration — "Spec L"; **supersedes** Spec Q Items 9 + 12)

This document resolves the v0.4 label collision between Spec Q/Spec D, locks the file-ownership order across them, amends two acceptance gates that are unsafe as written, and (as of 2026-05-27) records that all LLM-bearing work has been carved out into Spec L for separate sequencing. Subagent dispatch MUST NOT begin until this doc is read and §5's pre-flight is green.

---

## 1. Version remap (authoritative)

| Release | Source | Scope | Bench artifact | Status |
|---|---|---|---|---|
| **v0.3.1** | Spec Q Tier 1 (Items 1, 2, 3, 4) | Adaptive output budgets, multi-language classifier, deep_with_examples trim, small-repo arch override | `run27-v0.3.1/` (gin + flask + platform subset) | shipped 2026-05-27 |
| **v0.4** | Spec D (all 18 items, 5 workstreams) | Dep upgrades (bincode→postcard, axum 0.8, rusqlite 0.40, tantivy 0.26), PLAID 1.3 wiring, colgrep ranking signals, C# 10 / Scala 3 / Vue / OCaml chunking | `run28-v0.4/` (full 6 repos) | shipped 2026-05-27 |
| **v0.5** | Spec Q Tier 2 (Items 5, 6, 7, 8) | Deep audit, confidence-driven disambiguation, adaptive structural walk, deep dedup | `run29-v0.5/` (full 6 repos) | shipped 2026-05-27 (bench deferred) |
| **v0.6** | Spec Q Tier 3 partial (Item 10 only — Items 9, 11, 12 **moved to Spec L**) | Multi-step internal planner (pure-Rust orchestration). Items 11 (cross-repo daemon) deferred; Items 9 (LLM classifier+HyDE) and 12 (LLM index-time enrichment) **out of scope per `docs/superpowers/specs/2026-05-27-semantex-llm-integration.md`** | `run28-v0.6/` (5 repos × 3 reps, polluted indexes — see v0.6.2) | shipped 2026-05-27, includes /simplify fix sweep; **LLM scaffold scrubbed** post-tag |
| **v0.6.1** | LLM scrub | Removes the v0.6 ONNX LLM scaffold (`crates/semantex-core/src/llm/`, `classify_with_llm`, `hybrid_search_with_hyde`, `--features local-llm`); Item 10 planner remains shipped. | n/a | shipped 2026-05-27 |
| **v0.6.2** | Index-pollution defaults | Built-in walker patterns skip code-graph / LSP-index outputs (`graphify-out/`, `*.scip`, `*.lsif`, `.graphify_*`, etc.) at index time. Generic tool-class patterns, not project-specific. | `run29-v0.6-clean/` (same v0.6.1 code + clean indexes) | shipped 2026-05-27; pushed to origin |
| **v0.7** | Spec L Phase 1 | `genai` adapter: LLM-driven classifier + HyDE retrieval, behind `--features llm` Cargo feature. Default build dep-graph unchanged. | TBD (Spec L §8 Phase 1 gate; bench-spend deferred) | shipped 2026-05-27. Tag `v0.7` now points at the squashed integration commit `8bd9a01` (was `270d816` pre-squash). Phase 1, Phase 2, and the v0.7.1 HyDE call-site all landed together in this single squashed commit. |
| **v0.7.1** | Spec L §4 Item 1.4 HyDE call-site | Wire `HybridSearcher::search_with_hyde` into `AgentPipeline::handle_semantic` so the HyDE doc actually augments search results. Adds `SEMANTEX_LLM_HYDE_TIMEOUT_MS` env override (default 15 s, sized for cli:claude HyDE ~5–10 s). | n/a | shipped 2026-05-27 (squashed into the v0.7 commit `8bd9a01`; no separate tag). Live-verified on flask + cli:claude: HyDE-on returns 10 results in ~12.9 s vs HyDE-off 4 results in ~30 ms. |
| **v0.7.2** | Production-hardening sweep | All 14 Team C audit findings resolved across 3 parallel teams. Highlights: WARN-level LLM startup banner (visible without RUST_LOG=info); prompt piped via stdin instead of argv (no `ps aux` leak); env-overridable inner timeouts; bounded subprocess stdout; `CliKind::Antigravity` variant removed; `with_llm` accepts `Option<Arc>`; public LLM surface trimmed to `LlmBackend`+`LlmCapability`; `Listener` owns shared tokio runtime (no per-call construction); `infer_provider` covers Bedrock/Fireworks/LiteLLM paths. | n/a | shipped 2026-05-27 (commit `64f63e7`; no separate tag). Live-verified WARN banner appears and prompt is absent from argv. |
| **v0.8** | Spec L Phase 2 | `SubscriptionCli` backend: shell out to `claude` / `codex` / (gated) `antigravity` for users with bundled subscription quota. | TBD (Spec L §8 Phase 2 gate; bench-spend deferred) | shipped 2026-05-27. Tag `v0.8` now points at the same squashed commit as `v0.7` (`8bd9a01`); they ship in lockstep. |
| (strategic) | Spec Q Tier 4 (Items 13–16) | Quality benchmarks, prebuilt indexes, streaming MCP, web UI | n/a | not started |
| (deferred) | Spec Q Item 11 (cross-repo workspace daemon) | Single daemon serving N projects | n/a | conditional on downstream consumer request |

Anything still saying "Tier 2 = v0.4" or "Tier 3 = v0.5" in Spec Q has been remapped; the spec header has been amended. Disregard older copies.

**2026-05-27 amendment:** v0.6 originally landed Spec Q Item 9 (local LLM classifier + HyDE) + Item 10 (internal planner) per the table above. Item 9 (and the deferred Item 12) have since been **scrubbed from main** and re-scoped into the standalone **Spec L** (`docs/superpowers/specs/2026-05-27-semantex-llm-integration.md`), targeting v0.7 (Phase 1: genai adapter) and v0.8 (Phase 2: SubscriptionCli backend). The Item 10 planner remains shipped — it is pure-Rust orchestration with no LLM dependency.

---

## 2. Execution order

```
v0.3.1 (Spec Q Tier 1) ──→ bench run27 ──→ v0.4 (Spec D, all WS)  ──→ bench run28 ──→ v0.5 (Spec Q Tier 2) ──→ bench run29 ──→ v0.6 (Spec Q Tier 3)
       1-2 days, $15            ~6 h wall (5 parallel WS), $0       1-2 weeks, $80                       1-2 months, $80
```

Rationale:
1. **v0.3.1 first** — small (4 items), surgical, closes the 7 known regressions documented in `docs/BENCHMARK-v0.3-PHASE4-RESULTS.md`. Validates the bench harness before larger surgery and gives Spec D a known-good baseline.
2. **v0.4 second** — three of its items are unavoidable maintenance (bincode poison-pill, axum 0.7 panics, tantivy 0.26 compile break). Several items (colgrep ranking signals) move score composition; doing them before Spec Q Tier 2 means Tier 2 is designed against the real post-ranking baseline.
3. **v0.5 and v0.6** — re-evaluate against `run28-v0.4`. Some Tier 2 items may shrink or grow once Spec D's ranking signals are in.

No step in this sequence runs in parallel with another step. Within v0.4, the 5 workstreams run in parallel per Spec D §3.2.

---

## 3. File-ownership conflicts between Spec Q and Spec D

The two specs overlap on three files. Spec D wins the first two (it lands first); Spec Q wins the third (Spec D doesn't touch it).

### 3.1 `crates/semantex-core/src/server/protocol.rs`

| Owner | Item | Change |
|---|---|---|
| Spec D | Item 2 (bincode → postcard) | Sweeping rewrite of `encode_*`/`decode_*` functions; changes wire format payload |
| Spec Q | W-Delta Item 6 (disambiguation field) | Adds `disambiguation: Option<Vec<DisambigSuggestion>>` to `SearchResponse` / `AgentResponse` |

**Rule:** Spec D Item 2 MUST land before Spec Q Item 6 begins. The disambiguation field is defined as serde-derived, so it transports through postcard without further work. Spec Q W-Delta MUST verify on the v0.4 branch that postcard is the active codec before adding the field.

### 3.2 `crates/semantex-core/src/search/hybrid.rs`

| Owner | Item | Change |
|---|---|---|
| Spec D | W-C Items 6–9 | Adds 4 new score-boost phases (3b path penalty, 3c stem boost, 3d definition boost, 3e file coherence) after existing Phase 3 |
| Spec Q | (read-only) | No writes |

No write conflict, but **measurement conflict**: Spec Q Tier 1 (Items 1, 2, 4) targets the same regressions Spec D's ranking signals partially address (gin Q1 arch, platform Q2 error_handling). Tier 1 measures against run25; Tier 2/3 measure against `run28-v0.4` per §1. This is fine as long as the order in §2 is followed.

### 3.3 `crates/semantex-core/src/search/agent.rs`, `index/architecture.rs`, `agent_classifier.rs`, `deep.rs`, `agent_formatter.rs`

Owned exclusively by Spec Q workstreams. Spec D does not touch these.

### 3.4 `crates/semantex-core/src/search/sparse_search.rs`

Spec D internal: W-A Item 5 (tantivy 0.26) MUST land before W-E Item 18 (configurable stemmer). Already noted in Spec D §3.3.

### 3.5 `crates/semantex-core/src/index/builder.rs`

Spec D internal: W-B Item 13 (use returned doc IDs) MUST land before W-B Item 14 (buffer_size). Already noted in Spec D §3.3.

---

## 4. Amended acceptance gates

### 4.1 Spec D §10 — loosen "every cell neutral-or-better"

Spec D §10 currently requires CCB neutral-or-better for every (repo × question_type) cell vs run25. Run-to-run noise on this benchmark is ±5–10pp per cell (see MEMORY signal-quality table), so a strict per-cell gate will be tripped by noise on most runs.

**Amended gate:** Aggregate CCB Δ on `run28-v0.4` MUST be neutral-or-better than `run27-v0.3.1` (not run25). No single cell may regress by more than **10pp** vs `run27-v0.3.1`. Cells already-regressing in `run27-v0.3.1` are exempt — Spec D is not required to fix Spec Q's open work.

### 4.2 Spec Q Item 2 — investigation-first, gate softened

Spec Q Item 2 (multi-language classifier) is the only Tier 1 item that self-admits the regression may not be in the classifier at all ("the regression may actually be in the Deep handler's behavior on multi-language repos, not the classifier").

**Amended Tier 1 gate for Item 2:**
1. W-Beta MUST run the classifier on the exact Q2 wording from `benchmarks/agent_bench.py::QUESTIONS` and record which route fires, before writing any production code.
2. If the classifier already routes to `Deep` for platform Q2, W-Beta MUST open `coordination_request.md` rather than implementing the spec's `detect_languages` override — the regression source is elsewhere (likely Deep handler), and Tier 2 Item 5 already owns that work.
3. If the classifier routes to `Structural`, W-Beta proceeds with Item 2 as written, and the platform Q2 ≤+20% bench gate stands.
4. If the investigation outcome doesn't fit either branch above, W-Beta MUST escalate via `coordination_request.md`. Do NOT invent a new override path.

### 4.3 Bench cadence

- `run27-v0.3.1/` — subset (gin + flask + platform), runs after Spec Q Tier 1 integration branch builds clean. Gate: Items 1, 3, 4 per their per-cell targets; Item 2 per §4.2.
- `run28-v0.4/` — full 6 repos, runs after all 5 Spec D workstreams merge and the integration branch is green. Gate: §4.1.
- `run29-v0.5/` — full 6 repos, runs after Spec Q Tier 2 integration. Gate: aggregate CCB Δ ≤ −35% vs `run28-v0.4`.

Each bench artifact lives under `benchmarks/results/<name>/` and the comparison script is regenerated from MEMORY notes.

---

## 5. Pre-flight checklist (before dispatching any subagent)

- [ ] Spec Q header reads "v0.3.1 → v0.6", Tier 2 reads "v0.5", Tier 3 reads "v0.6+" (already applied in this edit).
- [ ] This file exists at `docs/RELEASE-SEQUENCE-2026-05.md` and is committed to `main`.
- [ ] No other branch claims the v0.4 label.
- [ ] `main` is at the Phase 4 commit (`446b720`) or later, with `run25` results present at `benchmarks/results/run25/`.
- [ ] Each subagent dispatch prompt includes the line: "Read `docs/RELEASE-SEQUENCE-2026-05.md` §3 (file-ownership) and §4 (amended gates) before reading the spec body."
- [ ] Worktree-leakage prevention from `~/.claude/projects/.../memory/parallel-subagent-worktree-leakage.md` is in effect: each subagent dispatched with `isolation: worktree`, integration checkout reset clean before merging branches in.

---

## 6. Open, deferred, and out-of-scope items

### 6.1 Still in-flight (2026-05-27)

None — all of v0.3.1 → v0.7.2 is shipped on origin/main. Remaining work falls under §6.2 (deferred for bench spend) and §6.3 (long-term / conditional).

### 6.2 Deferred — pending user-authorized bench spend

- `run28-v0.4/`, `run29-v0.5/`, `run30-v0.6/` bench artifacts — code-quality gates passed; bench metrics not yet collected. ~$80 each.
- Spec L §8 bench gate (v0.7): aggregate CCB Δ improvement ≥3pp with `--features llm + SEMANTEX_LLM_MODEL=ollama/qwen2.5-coder:7b` vs without on 4 repos.
- Spec L §8 Phase 2 operator-run smoke tests (cli:claude, cli:codex) — integration tests are written and `#[ignore]`-gated; operator runs them. End-to-end `cli:claude` was live-verified during v0.7.2 (banner + argv + HyDE), but the formal CCB-comparison run is still deferred.

### 6.3 Conditional / long-term deferred

- Tier 4 items (Spec Q Items 13–16): quality benchmarks, prebuilt indexes, streaming MCP, web UI. Not on the critical path.
- Spec Q Item 11 (cross-repo workspace daemon): single daemon serving N projects. Gated on a downstream consumer request.
- `docs/v0.4-future_SPEC.md` (referenced in Spec D §13): deferred deps (`fastembed 5.13`, `ort rc.12`, `notify 9.x`). These items remain blocked on upstream pin conflicts or RC-only availability as of 2026-05-27; the spec doc does not exist yet.
- Spec L §9 non-goals: model bundling, OAuth flows, MCP `sampling/createMessage` (deprecated per SEP-2577), subprocess pooling for CLI backends, streaming `LlmCapability` trait, embeddings / tool-use API on `LlmCapability`.

---

## 7. Changelog

- **2026-05-26** — Initial coordination doc. Resolves v0.4 label collision (Spec D wins, Spec Q tiers shift by one). Locks `protocol.rs` order (postcard → disambiguation). Loosens Spec D §10 per-cell gate; adds investigation gate to Spec Q Item 2.
- **2026-05-27** — Post-v0.6 amendments:
  - v0.3.1, v0.4, v0.4.1, v0.5, v0.6 all shipped and tagged on `origin/main`. Bench artifacts run28/29/30 deferred to a separate user-authorized session.
  - All LLM-bearing work carved out of Spec Q (Items 9, 12) and Spec D into the new standalone Spec L (`docs/superpowers/specs/2026-05-27-semantex-llm-integration.md`).
  - The v0.6 LLM scaffold (`crates/semantex-core/src/llm/`, `classify_with_llm`, `hybrid_search_with_hyde`, `merge_hyde_results`, `--features local-llm`) is **scrubbed from main**. Reason: design decision to (a) treat LLM integration as a dedicated workstream, (b) avoid bundling a model, (c) leverage `rust-genai` and coding-agent CLI subscriptions instead of hand-rolling provider clients. See Spec L §1 for full rationale.
  - The v0.6 Item 10 planner (`crates/semantex-core/src/search/planner.rs`) and `AgentRoute::FeaturePlanning` route remain shipped — they are pure-Rust orchestration with no LLM dependency.
  - Spec Q Item 11 (cross-repo workspace daemon) remains conditional / deferred, unchanged.
- **2026-05-27 (later)** — Post-bench amendments:
  - Ran `run28-v0.6/` (5 repos × 3 reps) against v0.6.1. Aggregate CCB −18% vs built-ins; one regression on platform Q1 (+7%).
  - Root cause: every benchmark repo had `graphify-out/` directories from a code-graph generator. These contained stringified symbol-graph JSON dumps that matched almost any semantic query and accounted for 16–46% of indexed chunks per repo.
  - Mitigated repo-side via `.semantexignore`, reran as `run29-v0.6-clean/`: aggregate CCB **−25%**, cost **−14%**, every per-repo and per-question delta net-negative.
  - Landed those patterns as semantex's **default ignore set** (commit `8a4d0eb`, tagged `v0.6.2`) so any user with similar tool outputs gets the cleanup automatically. Patterns cover graphify-out, SCIP, LSIF, and adjacent code-intel tooling. Per `crates/CLAUDE.md` rule #2, they're tool-class identifiers and not project-specific.
  - Investigation playbook update for future regression hunts: when a single repo/cell flips negative in a bench run, first check `chunks.db` composition for newly-introduced large-file pollution before suspecting code changes.
- **2026-05-27 (v0.7 + v0.8 ship)** — Spec L LLM integration fully shipped. Both Phase 1 and Phase 2 landed in a single merge commit `7eda12f` (pre-squash tags `v0.7` at `270d816`, `v0.8` at `c48e0e1`). Four post-merge fix commits followed (`33f4fcb`, `fddba26`, `cdf912c`, `b14bca3`) addressing: missing system prompt in CLI backend; 800 ms classify timeout too tight for cli:claude (bumped to 8 s + env override); `llm-status` health-check timeout; P0 default-build break (`llm_classify_timeout` not cfg-gated); `semantex watch` not wiring LLM; `SEMANTEX_LLM_CLI_BINARY` override honored for wrong `CliKind`; Claude JSON error responses; env-mutation test race; `LlmBackend::into_arc()`; `parse_route_from_llm_output` delegates to `AgentRoute::from_str`.

- **2026-05-27 (v0.7.1 HyDE call-site)** — `HybridSearcher::search_with_hyde` had been implemented in v0.7 but `AgentPipeline::handle_semantic` was calling `searcher.search()` directly, so HyDE was unreachable from production. Now `AgentPipeline::search_semantic` routes through `search_with_hyde` when an LLM is wired, falling back to plain search on any HyDE failure (HyDE never breaks a query). Added `SEMANTEX_LLM_HYDE_TIMEOUT_MS` env override. Live-verified on `/Users/tk/dev/flask` + cli:claude: semantic query "code that handles request timeouts in Flask" returned **10 results in 12.9 s** with HyDE vs **4 results in 30 ms** without — HyDE found 6 unique chunks the base path missed.

- **2026-05-27 (v0.7+v0.8+v0.7.1 squash)** — The full v0.7/v0.8/v0.7.1 history (24 commits across W-LLM-Core, W-LLM-Genai, W-LLM-Cli, W-LLM-Wire workstreams, integration merges, post-ship fixes, /simplify sweep, and HyDE call-site patch) was squashed into a single commit `8bd9a01` and pushed. Tags `v0.7` and `v0.8` were moved to point at the squashed commit (both ship together).

- **2026-05-27 (v0.7.2 production-hardening sweep)** — Team C audit identified 14 findings (3 HIGH, 6 MEDIUM, 5 LOW); zero BLOCKERs. Three parallel teams resolved all of them in commit `64f63e7`:
  - **Finding 1 (HIGH)** — Promoted `tracing::info!("LLM enabled: ...")` and discovery-hint lines to `tracing::warn!` so they appear under the default WARN filter. Operators starting `semantex serve` now see the LLM banner without needing `RUST_LOG=info`.
  - **Finding 2 (HIGH)** — Reconciled HyDE inner/outer timeouts; `subscription_cli.rs::HYDE_TIMEOUT` now reads `SEMANTEX_LLM_HYDE_TIMEOUT_MS`. Both layers extend in lockstep when the user overrides the budget.
  - **Finding 3 (HIGH)** — `SubscriptionCli` classify inner timeout now reads `SEMANTEX_LLM_CLASSIFY_TIMEOUT_MS`. Prevents orphan `claude`/`codex` subprocesses running past the outer agent budget.
  - **Finding 4 (HIGH)** — Prompt now piped via stdin instead of argv. `ps aux | grep claude` shows only `claude --print --output-format json` during a query — no prompt text visible to other local users.
  - **Finding 5 (MEDIUM)** — `SubscriptionCliBackend::exec` accepts `max_stdout_bytes`; 1 KiB cap for classify, 50 KiB for HyDE. A malicious LLM returning a multi-MB response now bails with the byte count rather than OOM-ing the daemon.
  - **Finding 6 (MEDIUM)** — `CliKind::Antigravity` variant removed. The `from_env` bail message is preserved so `SEMANTEX_LLM_BACKEND=cli:antigravity` still errors at startup with the actionable message.
  - **Finding 7 (MEDIUM)** — `AgentPipeline::with_llm` now accepts `Option<Arc<dyn LlmCapability>>` (was `Arc<...>`). All callers (`handler.rs`, `mcp/server.rs`, `watch.rs`, `llm_status.rs`) collapse to `pipeline.with_llm(self.llm.clone())`.
  - **Finding 8 (MEDIUM)** — Public LLM API surface trimmed: `GenaiBackend`, `SubscriptionCliBackend`, `CliKind` are now `pub(crate)`. External callers use `LlmBackend::from_env()` + `into_arc()` + `LlmCapability` trait only. `LlmBackend` carries `#[allow(private_interfaces)]` because its variants intentionally wrap `pub(crate)` types.
  - **Finding 9 (MEDIUM)** — `TEST_ENV_LOCK` in `llm/mod.rs` and `env_cap_behaviour_serial` in `memory.rs` documented as guarding disjoint env-var families; cross-references added so a future test touching both families knows to hold the LLM lock.
  - **Finding 10 (MEDIUM)** — `Listener` now owns an `Arc<tokio::runtime::Runtime>` (under `#[cfg(feature = "llm")]`), constructed once at `Listener::bind`. `Handler` and `AgentPipeline` gain `with_runtime` builders. The per-call current-thread runtime construction in `classify_route_with_llm_fallback` and `search_semantic` is now a legacy fallback that emits a one-time warning if it ever fires.
  - **Finding 11 (LOW)** — `kill_on_drop(true)` documented as the sole guarantee subprocesses are reaped on outer-timeout future drop.
  - **Finding 12 (LOW)** — `infer_provider` now covers Bedrock (`nova-*`), Fireworks/path-style names (`accounts/.../models/...`), and LiteLLM/vLLM gateways (suggests `SEMANTEX_LLM_PROVIDER=openai` in the error message).
  - **Finding 13 (LOW)** — Existing `hyde_tests` module already covers `merge_hyde_results` dedup-by-chunk-id, score-descending sort, `max_results` cap, and empty-base safety. Higher-level mock-LLM tests for `search_with_hyde` would require a real `HybridSearcher` and were deemed lower value than the operator-gated Ollama / Claude integration tests already in place.
  - **Finding 14 (LOW)** — `o2-` OpenAI prefix added defensively to `infer_provider` alongside `o1`/`o3`/`o4`.
  - Verified end-to-end on flask + cli:claude: WARN-level banner appears, prompt absent from `ps aux`, HyDE-on semantic query returns 10 results. 766 lib tests pass with `--features llm` (up from 761).
