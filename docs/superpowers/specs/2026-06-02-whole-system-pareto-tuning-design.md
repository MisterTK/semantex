# Whole-System Pareto-Tuning of semantex (MCP-first) — Design v2

- **Date:** 2026-06-02
- **Status:** approved design (brainstorming complete; next = writing-plans)
- **Supersedes:** v1 (config-arm sweep). v2 incorporates the user directive — **MCP is the primary product distribution and must run optimally for ANY agent client with ZERO CLAUDE.md crutch**; the output-shaping defaults are OURS to own and analyze, not a prompt nudge.
- **Foundation:** two verification workflows against live code + the June-2026 MCP ecosystem (`wf_1f1121b5-4a1` config-arms/daemon/index; `wf_70db4f51-8c9` MCP spec + best-practices + current-surface critique). Key sources: MCP spec 2025-11-25 (`server/tools`, SEP-1613/1303), Anthropic "writing effective tools for agents" + "Code execution with MCP", Sourcegraph/claude-context/coa code-search patterns, Claude Code issue #55677 (structuredContent drops content[]).

## 1. Problem & objective

Every retrieval A/B to date optimized **single-gold nDCG on CoIR** — a component proxy often anti-correlated with system value (adaptive pruning is nDCG-bad but the −18% CCB feature; more recall is nDCG-good, CCB-bad). And the prior agent benchmarks ran *with* a CLAUDE.md/`SEMANTEX_MD` nudge — measuring "semantex + a prompt crutch," not the product a generic MCP client gets.

**Optimize semantex AS A SYSTEM, measured as a bare MCP server any agent consumes.** Objective = a 3-way **Pareto**: (i) answer quality (tool-blind judge), (ii) **CCB** (real cumulative attended context, O(N²) in turns), (iii) latency (time-to-answer). MCP is the product surface, so its tool descriptions, output format, and **defaults** are first-class optimization targets — "are OUR defaults optimal?" must be measurable, hence tunable.

## 2. Verified architecture facts (confirmed against live code — do not re-litigate)

- **`semantex_agent` (the tool the benchmark drives) runs IN-PROCESS** in `semantex mcp`: `McpServer::new` builds config from env **once per process**; `claude` spawns a fresh `semantex mcp` per call. ⇒ **per-arm `SEMANTEX_*` env on the MCP server is the clean isolation seam** (no daemon mediates the agent path). CAVEAT: stop any pre-existing `semantex serve` daemon on bench repos (`main.rs:802` fast path could short-circuit).
- **Output-volume defaults are HARDCODED at the agent call-site**, not env-tunable: `budget` `map_or(12_000…)` (`server.rs:1208`), `full_code` `unwrap_or(false)` (`server.rs:1153`), `depth` classifier-auto with no default-on override (`server.rs:1156`). `SEMANTEX_MAX_COUNT`/`_CONTENT` do NOT reach this path. ⇒ to answer "are our defaults optimal?" we MUST make these env-readable (Workstream A).
- **The MCP surface is fairly mature** (`server.rs:677-932`): `ToolAnnotations` (read_only / local_mutation) set; `outputSchema` + `structuredContent` on `semantex_search`/`semantex_deep`; `isError` used (`:1008/:1091`). **Gaps:** primary tool `semantex_agent` has `output_schema: None` (`:774`, prose-only `content[]`); no `concise|detailed` enum (only raw byte `budget`); `structuredContent` null for `semantex_agent`.
- **Footgun (MCP 2025-11-25):** `structuredContent` is client-side and NOT guaranteed in the model's context; some clients (Claude Code #55677) DROP `content[]` when `structuredContent` is present. ⇒ any structured output MUST also serialize a summary into a `content[]` TextContent block.
- **Index confound:** the two embedders write different dense stores; `detect_for_config` returns **Stale** if the running config's fingerprint ≠ the repo's ACTIVE pointer → agent path falls back to ripgrep + background rebuild (`server.rs ~1179-1183`). ⇒ pre-index each (repo × embedder), verify `Ready` before paid reps.
- **`claude_bench` already captures latency** (`duration_ms` `:256`, `elapsed` `:320`) and real CCB; it has NO per-arm env seam (`ARMS` hardcoded; `mcp_config_for` has no `env`); and it injects a `SEMANTEX_MD` CLAUDE.md nudge (to be removed).

## 3. Methodology: bare-MCP, no CLAUDE.md (the core change)

The benchmark must measure the product as a generic MCP client gets it: **neutral/empty system prompt, only the MCP server-instructions block + the tool schemas — no CLAUDE.md, no `SEMANTEX_MD` nudge.** The tool name/description/inputSchema/error-text is the only cross-client guidance channel; if the bare tool underperforms, that is a *product* defect to fix in the descriptions, not paper over with a prompt. All quality/CCB/latency is thus attributed to **server behavior + defaults**, not prompt scaffolding.

## 4. Workstream A — MCP-foundations PR (product improvement; prerequisite for the sweep)

Make the MCP surface best-practice-compliant AND its defaults tunable. TDD in `crates/semantex-mcp/` (+ `protocol.rs`). Each item small + independently testable; lands as one gated PR (or a tight series).

1. **Env-tunable output defaults** (unblocks the sweep + lets ops tune): `SEMANTEX_MCP_BUDGET` (bytes; fallback 12000 at `server.rs:1208`), `SEMANTEX_MCP_FULL_CODE` (0/1; fallback false at `:1153`), `SEMANTEX_MCP_DEPTH` (quick|search|deep|auto; fallback auto at `:1156`). Env is read once at `McpServer::new` (matches the in-process model); the per-call MCP arg still overrides env when the client passes one.
2. **`response_format` enum** (`concise|detailed`) on `semantex_agent`, mapping to budget tiers (concise ≈ ⅓ tokens) — best-practice "return less by default" + a clean discrete sweep axis. Default tier chosen by the sweep.
3. **`output_schema` + `structuredContent` for `semantex_agent`** mirroring `semantex_search` (array of {path, start_line, end_line, score, name, snippet}), AND **keep the prose summary in `content[]`** (the #55677 footgun fix). Declare schemas in JSON Schema **2020-12** dialect (SEP-1613).
4. **Tighten `semantex_agent` description to be self-describing** (no reliance on server-instructions/CLAUDE.md): restate in-band "one call in → one complete answer out; do NOT chain semantex_* calls; if incomplete, refine the QUESTION and call again." Trim schema verbosity (definitions tax every request).
5. **Error hygiene:** bad-param/empty results → `isError:true` Tool Execution Errors with specific, actionable next-step text (so the model self-corrects), not protocol errors (SEP-1303). Verify for `semantex_agent` inputs.
6. **Schema/output snapshot tests** so tool schemas + a sample agent output stay self-contained + regression-safe.

(Forward-compat note, NOT a task: the 2026-07-28 RC lifts `structuredContent` to any JSON value + adds W3C Trace Context in `_meta`; item 3 should not assume object-only.)

## 5. Workstream B — bare-MCP harness

Extend `benchmarks/claude_bench.py` (the arbiter):
1. **Bare-MCP mode:** neutral system prompt; DROP the `SEMANTEX_MD` injection + run with NO project CLAUDE.md (hermetic empty config dir already exists — ensure it carries no skill/CLAUDE.md). The semantex arm = only `--mcp-config` + the server's own instructions/schemas.
2. **Per-arm env seam:** `SX_CONFIG_ARMS: {arm → {SEMANTEX_* env}}` threaded into `mcp_config_for`'s `mcpServers["semantex"]["env"]`; generalize `--arms` to accept config-arm names (+ keep `builtin`).
3. **3-way Pareto report:** surface the already-captured latency alongside CCB + quality; per-arm medians; Pareto ranking.
4. **Setup discipline:** stop any daemon on bench repos; pre-index each (repo × embedder); a FREE dry-run assert (`Ready`, no ripgrep fallback) per (repo × embedder) before paid reps.

## 6. Workstream C — the fractional sweep

Two sub-sweeps, all bare-MCP, ≥4 reps, multi-repo. The full grid (embedder×graph×adaptive×budget×full_code×depth ≈ 96) is intractable → fractional: tune RETRIEVAL first, then SERVER-DEFAULTS on the winning retrieval config (each server-default arm reuses the index — server-time only, no re-index).

**Sub-sweep 1 — retrieval frontier (default server settings; 2 embedder builds/repo):**
`builtin` (floor) · `sx-lateon` (default) · `sx-coderank` (validate/maybe-REVERSE the flip) · `sx-graph2hop` (`GRAPH_HOPS=2`+`GRAPH_CENTRALITY_WEIGHT=0.2`) · `sx-adaptive-off` (`ADAPTIVE_SIZING=0`) · `sx-stacked` (graph2hop + adaptive-off). Phase 2 arms run only if `sx-lateon ≥ builtin`; they're query-time env on the lateon index (near-zero extra indexing).

**Sub-sweep 2 — server-default frontier (on the winning retrieval config from #1; server-time env only, same index):**
from the winner, vary one factor at a time + two combos: `budget-low` (~6K) · `budget-high` (~24K) · `full_code-on` · `depth-deep` · `lean` (budget-low + concise) · `rich` (budget-high + full_code-on + depth-deep). (Default budget/full_code/depth was already measured as the winner in #1.)

**Conditional `sx-rerank`** (`RERANK=1`+`RERANKER=on`+`RERANKER_MODEL=qwen3-reranker-0.6b`+`RERANK_CANDIDATES=25`): a precondition gate FIRST (does qwen3-reranker load on this CPU? warm latency tolerable? note `MAX_RSS_MB` may not reach the in-process path). Include only if it passes; else document "excluded: <reason>".

## 7. Pareto reporting + decision rules

Per arm: quality (judge mean 1–10), CCB (mean + CAF + peak), latency (median time-to-answer), turns. An arm **Pareto-dominates** if ≥ on all three axes, > on one.
- `sx-coderank` Pareto-dominates `sx-lateon` → **flag the lateon flip for reversal** (its own gated PR — Phase 1 is a real re-litigation of the shipped decision).
- Any retrieval lever (graph2hop / adaptive-off / stacked) that Pareto-improves → gated default-flip follow-up PR.
- Sub-sweep 2 picks the **optimal server defaults** (budget / full_code / depth / response_format tier) → set them as the shipped MCP defaults in a gated PR. This is the direct answer to "are our defaults optimal?"
- Ties/mixed → keep current defaults; document the frontier + the tradeoff (e.g. "budget-high buys +X quality at +Y CCB").

## 8. Repos, reps, questions

- **Repos** (verify exist + indexable): gin (Go), flask (Python), pub (Dart) for Phase 1; add CopilotKit (large TS — lateon's known under-retrieval failure mode) if its per-embedder index cost is affordable.
- **Reps ≥ 4** (memory: Q1/Q2/Q4 flip sign on single runs; Q3 high-confidence; Q5 medium). The existing 5 questions.

## 9. Risks / open items

- **Bare-MCP may score semantex LOWER than the prior nudged runs** — that's the point (honest measurement); if it does, the fix is better tool descriptions/output (Workstream A item 4), measured again.
- `sx-rerank` likely unrunnable on CPU → precondition gate; sweep proceeds without it.
- CopilotKit index cost (large, two embedders) → restrict to small repos for Phase 1 if needed.
- Latency conflates model-think + tool time — it's the *system* latency (correct for the Pareto); note it.
- Judge cost/variance → reuse the existing tool-blind judge + neutral-cwd discipline; 4 reps damps variance.
- **Doc-sync:** verify `dense_backend.rs` module-doc + CLAUDE.md state lateon-colbert as the live default (updated in `1801341`); fix if a stale read remains.

## 10. Out of scope (v2)

- HyDE (needs `--features llm`; breaks zero-LLM default).
- MMR / weighted-RRF / fusion arms (measured-dead on relevance; not worth reps unless a result reopens them).
- New MCP tools / resources / prompts beyond the existing tool set (consolidation is already good — single `semantex_agent` entry).
- A cheap deterministic agent-turn proxy (medium-sweep budget chosen).

## 11. Outputs & sequencing

1. **Workstream A** (MCP-foundations PR) — gated, TDD, lands first; improves the product regardless of the sweep.
2. **Workstream B** (bare-MCP harness) — benchmark-only PR.
3. **Workstream C** (the sweep) — run + a results doc `docs/superpowers/plans/2026-06-02-system-tuning-results.md` + memory entry.
4. Each Pareto-winning change (retrieval default, server default, or a flip reversal) → its own gated cutover PR, same discipline as the lateon flip.
