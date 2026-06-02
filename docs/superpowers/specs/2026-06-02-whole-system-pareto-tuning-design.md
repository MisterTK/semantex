# Whole-System Pareto-Tuning of semantex via claude_bench config-arms — Design

- **Date:** 2026-06-02
- **Status:** approved design (brainstorming complete; next = writing-plans)
- **Author context:** continues the CPU-optimization arc after the lateon-colbert default flip (`1801341`). Foundation verified by an 8-agent workflow (run `wf_1f1121b5-4a1`, 4 mappers + adversarial critique) against the live code.

## 1. Problem & objective

Every retrieval A/B to date (and the entire "why no feature uplift" analysis) optimized **single-gold nDCG on CoIR** — a component proxy that is often anti-correlated with system value: adaptive pruning is nDCG-bad but is the −18% CCB feature; MMR/graph "hurt" single-gold but the system rewards diversity + related-code-in-one-shot (fewer turns); more recall is nDCG-good but CCB-bad. The directive: **tune semantex as one system, measured by what actually matters to a coding agent.**

**Objective = a 3-way Pareto:** (i) **answer quality** (tool-blind LLM judge), (ii) **CCB** (real cumulative attended context, O(N²) in turns), (iii) **latency** (time-to-answer). No single scalar; rank arms on the Pareto frontier.

**Arbiter = `benchmarks/claude_bench.py`** — real Claude agents answer fixed conceptual code questions; the only difference between arms is the search tool/config. It already computes real CCB (sum of per-turn `usage.input + cache_read + cache_creation`), peak, CAF, num_turns, AND latency (`duration_ms` server.rs result_stats @ claude_bench.py:256; `elapsed` @ :320). Hermetic per-arm `CLAUDE_CONFIG_DIR` + tool-blind judge run in a neutral cwd (the cwd-contamination gotcha is already handled).

## 2. Verified architecture facts (do not re-litigate — confirmed against live code)

- **The `semantex_agent` MCP tool (what claude_bench drives) runs IN-PROCESS** in the `semantex mcp` server: `McpServer::new` builds the config **from env, once**, → `get_searcher` → `HybridSearcher::open(index_dir, &config)` → `AgentPipeline`. `claude` spawns a **fresh `semantex mcp` per invocation**. ⇒ **Per-arm `SEMANTEX_*` env on the MCP server process is the correct, clean isolation seam** — no separate daemon mediates the agent path, so there is no stale-daemon-config trap *for the agent path*.
- **CAVEAT (fast-daemon short-circuit):** `main.rs:802` has a fast path; a pre-existing long-lived `semantex serve` daemon on a bench repo could short-circuit the in-process config. ⇒ setup MUST stop any daemon on each bench repo before the run.
- **`claude_bench` has NO env seam today:** `ARMS=("builtin","graphify","semantex")` hardcoded; `--arms choices=ARMS`; `mcp_config_for(arm)` returns `{"mcpServers":{"semantex":{"command":SEMANTEX_BIN,"args":["mcp"]}}}` with **no `env` block**. Claude Code forwards an `env` dict on an mcpServers entry to the MCP subprocess — that is the injection point.
- **Output volume is NOT env-controllable on the agent path.** It is driven by the per-call MCP **`budget` arg (default 12000 bytes ≈ 3K tokens)** + `budget_for_chunk_count`, chosen by the *calling agent*. `SEMANTEX_MAX_COUNT`/`SEMANTEX_CONTENT` only affect the CLI `--max-count`/`--content` + the `semantex_search` JSON tool, NOT `semantex_agent`. The real output-shaping levers are MCP **args**: `depth` (quick|search|deep; auto-detect if omitted → `AgentRoute::Deep` @ server.rs:1156-1159) and `full_code` (server.rs:1150). ⇒ "terse/wide" CCB-shaping is a **prompt arm** (nudge `depth`/`full_code` via the per-arm CLAUDE.md the harness already injects), NOT a config-env arm. **Out of scope for v1** (see §7).
- **Index confound:** the two embedders write different dense stores (`dense/colbert-plaid/<fp>/` vs `dense/coderank-hnsw/<fp>/`) — they coexist on disk. But `state::detect_for_config` returns **Stale** when the running config's embedder fingerprint ≠ the repo's ACTIVE pointer for that backend, and the agent path then **falls back to ripgrep + spawns a background rebuild** (server.rs ~1179-1183). ⇒ each (repo × embedder) MUST be pre-indexed and verified `Ready` before reps, or measurements are polluted.

## 3. Exact env knobs (verified present in `crates/`)

`SEMANTEX_EMBEDDER` (lateon-colbert | coderank-137m), `SEMANTEX_ADAPTIVE_SIZING` (0/1), `SEMANTEX_GRAPH_HOPS`, `SEMANTEX_GRAPH_CENTRALITY_WEIGHT`, `SEMANTEX_GRAPH_MODULE_DECAY`, `SEMANTEX_GRAPH_DISABLE`, `SEMANTEX_FUSION`, `SEMANTEX_MMR_LAMBDA`, `SEMANTEX_RERANK`, `SEMANTEX_RERANKER`, `SEMANTEX_RERANKER_MODEL`, `SEMANTEX_RERANK_CANDIDATES`, `SEMANTEX_MAX_RSS_MB`. Graph is **ON by default** (opt-out via `SEMANTEX_GRAPH_DISABLE`); default `SEMANTEX_GRAPH_HOPS` is 1.

## 4. Infra changes to `claude_bench.py`

1. **Per-arm env registry + seam.** Add `SX_CONFIG_ARMS: dict[str, dict[str,str]]` (arm-name → SEMANTEX_* env). In `mcp_config_for(arm)`, when `arm` is a semantex config-arm, emit `{"mcpServers":{"semantex":{"command":SEMANTEX_BIN,"args":["mcp"],"env":{...arm env...}}}}`. Generalize `--arms` to accept the new names (and keep `builtin`). All config-arms reuse the existing semantex `SEMANTEX_MD` nudge.
2. **Surface latency in `report`.** CCB + quality are already reported; add the already-captured per-question latency (`elapsed`/`duration_ms`) as a first-class column and compute per-arm medians for the 3-way Pareto. (No new capture needed — verify the field threads from `run` → results jsonl → `report`.)
3. **Setup discipline (new `setup` responsibilities / a preflight subcommand):**
   - `semantex stop <repo>` on every bench repo (kill any daemon that could short-circuit in-process config).
   - **Per-embedder pre-index:** for each repo, build BOTH dense stores — `SEMANTEX_EMBEDDER=lateon-colbert semantex index <repo>` and `SEMANTEX_EMBEDDER=coderank-137m semantex index <repo>` — so both versioned dirs + ACTIVE pointers exist.
   - **Free dry-run assert:** for each (repo × embedder arm), one `semantex_status`/`validate` (or a single agent-free search) under the arm's env; assert state `Ready` (not `Stale`) and NO ripgrep fallback, BEFORE spending paid reps. Abort the arm if it can't be made Ready.

## 5. The config-arm set (two-phase)

Each arm = `builtin` (floor) or semantex + an env dict. Exact env per arm (centrality weight value TBD at implementation — start ~0.2; cross-check the S6 overhaul design's suggested range):

**Phase 1 — decision-reversing, cheap (2 embedder builds/repo). Maps onto the existing 3-arm shape.**
| arm | env | question it answers |
|---|---|---|
| `builtin` | (no MCP) | the floor — does semantex beat native Grep/Read/Glob on the Pareto at all? |
| `sx-lateon` | `{SEMANTEX_EMBEDDER: lateon-colbert}` (the shipped default) | the baseline every sx arm is measured against |
| `sx-coderank` | `{SEMANTEX_EMBEDDER: coderank-137m}` | **validate (or REVERSE) the default flip at the system level** — the one arm that can change a shipped decision |

**Phase 2 — run ONLY if `sx-lateon` Pareto-≥ `builtin`. All query-time env on the already-built lateon index → near-zero extra indexing.**
| arm | env | lever |
|---|---|---|
| `sx-graph2hop` | `{SEMANTEX_GRAPH_HOPS: 2, SEMANTEX_GRAPH_CENTRALITY_WEIGHT: 0.2}` | graph-maximal: does surfacing 2-hop related code + centrality help the agent (fewer turns / better answers) despite its single-gold nDCG null? |
| `sx-adaptive-off` | `{SEMANTEX_ADAPTIVE_SIZING: 0}` | the +45%-recall lever on the agent path — does more recall help answers or just bloat CCB? |
| `sx-stacked` | `{SEMANTEX_GRAPH_HOPS: 2, SEMANTEX_GRAPH_CENTRALITY_WEIGHT: 0.2, SEMANTEX_ADAPTIVE_SIZING: 0}` | do the two winning query-time levers COMPOSE, or interfere? |

**Conditional — `sx-rerank`** `{SEMANTEX_RERANK:1, SEMANTEX_RERANKER:on, SEMANTEX_RERANKER_MODEL:qwen3-reranker-0.6b, SEMANTEX_RERANK_CANDIDATES:25, SEMANTEX_MAX_RSS_MB:8192}`. **Precondition gate** (do FIRST, free-ish): a one-shot `semantex --rerank` confirms qwen3-reranker-0.6b loads on this CPU AND warm latency is tolerable (prior measurements: 47–120 s/query — likely a latency-axis loser). Note `SEMANTEX_MAX_RSS_MB` is read in CLI `main()`, NOT the in-process MCP constructor, so the RSS lift may not apply on this path — verify. Include the arm ONLY if the gate passes; otherwise document "rerank excluded: <reason>" and skip.

**Dropped:** `sx-terse` (output volume not env-expressible — see §2). `sx-wide`'s "more results" half (same reason; salvaged as `sx-adaptive-off`).

## 6. Repos, reps, questions, decision rule

- **Repos** (verify exist + indexable): gin (Go), flask (Python), pub (Dart), + CopilotKit (large TS — where lateon's known under-retrieval failure mode on broad large-repo Qs would surface). Start with the 3 small ones for Phase 1; add CopilotKit if its per-embedder index cost is affordable (lateon indexes faster than coderank; budget the build).
- **Reps ≥ 4** (project memory: Q1 architecture / Q2 error_handling / Q4 exhaustive flip sign on single runs; Q3 deep_technical is high-confidence; Q5 feature_planning medium). The existing 5 questions, unchanged.
- **Pareto report** per arm: quality (judge mean, 1–10), CCB (mean + CAF + peak), latency (median time-to-answer), turns. **Decision rule:** an arm *Pareto-dominates* if it is ≥ on all three axes and > on one. (1) If `sx-coderank` Pareto-dominates `sx-lateon` → **flag the flip for reversal** (its own gated PR). (2) Any Phase-2 lever that Pareto-improves `sx-lateon` → a gated default-flip follow-up PR (same discipline as the lateon flip). (3) Ties / mixed → keep current defaults, document the frontier.

## 7. Out of scope (v1)

- **Output-shaping arms** (`depth`/`full_code` prompt-nudge arms) — the real CCB lever, but it's a *prompt* arm (per-arm CLAUDE.md), a different mechanism; defer to v2 once the config-arm frontier is known.
- **HyDE** — needs `--features llm`; breaks the zero-LLM default; separate effort.
- **MMR / weighted-RRF / fusion arms** — already measured-dead on relevance; not worth agent-CCB reps until/unless a config-arm result suggests otherwise.
- **A cheap deterministic proxy** for the inner loop — deferred (medium-sweep budget chosen).

## 8. Risks / open items

- **`sx-rerank` may be unrunnable** (model load / CPU latency / RSS-path) → the precondition gate handles it; the sweep proceeds without it if it fails.
- **CopilotKit index cost** on CPU (large repo, two embedders) → may restrict Phase 1 to the 3 small repos + add CopilotKit only for the cheaper Phase-2 (lateon-only) arms.
- **Latency capture fidelity** — `claude_bench` captures wall-clock incl. agent-thinking, not just semantex's time; it's the *system* latency (correct for the system Pareto) but conflates model + tool time. Acceptable (the agent IS the system); note it.
- **Judge cost/variance** — the blind judge is an API cost; reuse the existing judge + neutral-cwd discipline; 4 reps damps variance.
- **Doc-sync (pre-existing):** confirm `dense_backend.rs` module-doc + CLAUDE.md already state lateon-colbert as the live default (updated in `1801341`); the critique flagged a possible stale read — verify, fix if needed (cheap, not blocking).

## 9. Outputs

- `claude_bench` config-arm seam + latency-in-report (its own small infra PR, gated).
- A results doc `docs/superpowers/plans/2026-06-02-system-tuning-results.md` + a memory entry.
- Each Pareto-winning lever → its own gated follow-up cutover PR. Phase 1 could itself trigger a *revert* of the lateon flip if coderank system-dominates.
