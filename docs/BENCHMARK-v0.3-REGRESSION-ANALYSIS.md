# v0.3 Agent-CCB Regression Analysis & Recovery Plan

**Date:** 2026-05-25
**Status:** SOTA claim BLOCKED by spec §4.7 honest-cap; ship-decision pending
**Run compared:** `benchmarks/results/run22` (v0.2, 2026-03-08) vs
`benchmarks/results/run23` (v0.3, 2026-05-25)
**Cost of measurement:** $81.78 USD, 2h 39m wall-clock

---

## 0. Cap-condition status (spec §4.7)

> "B3 cap (agent CCB): average CCB reduction is ≥−50%. If <−50% → claim is
> capped at 'competitive with Claude Code built-ins, wins on specific question
> types'."

| Metric | v0.2 (run22) | v0.3 (run23) | Δ (pp) | Verdict |
|---|---|---|---|---|
| Aggregate turns Δ | **−50%** | **+16%** | **+66pp** | regressed |
| Aggregate tool_calls Δ | **−60%** | **+17%** | **+77pp** | regressed |
| Aggregate tokens / CCB Δ | **−56%** | **+20%** | **+76pp** | regressed |
| Aggregate cost Δ | −11% | +7% | +18pp | regressed |
| Aggregate CAF (caching gain) | −21% | +1% | +22pp | regressed |

**The B3 cap is decisively tripped. The "−70% CCB SOTA" headline cannot ship.**
Per spec, the honest claim is capped at *"competitive with built-ins, wins on
specific question types"* — and even that is generous given the data.

---

## 1. Per-question-type aggregate (mean of treatment Δ across 6 repos)

The headline. Every question type regressed. The strongest v0.2 wins regressed
hardest in v0.3:

| Question type | v0.2 turns Δ | **v0.3 turns Δ** | v0.2 CCB Δ | **v0.3 CCB Δ** | regression (pp) |
|---|---|---|---|---|---|
| architecture | −35% | **+7%** | −41% | **+16%** | **+57pp** |
| error_handling | **−69%** ⭐ | **+4%** | **−78%** ⭐⭐ | **+2%** | **+79pp** |
| deep_technical | −12% | **+21%** | −12% | **+25%** | **+37pp** |
| exhaustive | −35% | **+33%** | −37% | **+55%** | **+93pp** ⚠️ worst |
| feature_planning | −47% | **+20%** | −51% | **+29%** | **+80pp** |

⭐ = strongest v0.2 win, now flat or negative in v0.3.
⚠️ exhaustive (Q4) was already noisy in v0.2; the v0.3 regression is real but
the v0.2 number was low-confidence.

---

## 2. The 12 worst per-(repo, question) regressions

These are the smoking guns — same question, same prompt, dramatically more
turns in v0.3 than in v0.2:

| Repo | Q | Type | v0.2 turns | v0.3 turns | turns Δ | v0.2 CCB | v0.3 CCB | CCB Δ |
|---|---|---|---|---|---|---|---|---|
| flask | Q2 | error_ha | **6** | **56** | **+833%** | 116K | 3.2M | **+2676%** |
| gin | Q2 | error_ha | 11 | 59 | +436% | 232K | 2.6M | +1026% |
| qgrep | Q2 | error_ha | 11 | 51 | +364% | 281K | 2.8M | +896% |
| pub | Q3 | deep_tec | 8 | 38 | +375% | 157K | 1.5M | +873% |
| platform | Q3 | deep_tec | 9 | 45 | +400% | 195K | 1.8M | +841% |
| gin | Q3 | deep_tec | 8 | 32 | +300% | 138K | 1.2M | +790% |
| CopilotKit | Q1 | architec | 23 | 84 | +265% | 615K | 5.3M | +769% |
| CopilotKit | Q5 | feature_ | 18 | 107 | +494% | 614K | 5.3M | +758% |
| platform | Q1 | architec | 16 | 77 | +381% | 383K | 3.1M | +710% |
| pub | Q4 | exhausti | 19 | 75 | +295% | 697K | 5.4M | +669% |
| CopilotKit | Q2 | error_ha | 16 | 69 | +331% | 471K | 3.5M | +649% |
| gin | Q1 | architec | 10 | 28 | +180% | 176K | 1.2M | +570% |

**Pattern:** every regression is the agent going **substantially deeper** —
3-9× more turns. This is not a search-quality issue. The agent is *not* failing
to find answers; it's choosing to explore more.

---

## 3. Where v0.3 still wins (only 6 of 30 question-runs net positive)

| Repo | Q | Type | BL turns | TX turns | BL CCB | TX CCB | TX Δ |
|---|---|---|---|---|---|---|---|
| qgrep | Q3 | deep_tec | 23 | 13 | 981K | 413K | **−58%** ⭐ |
| qgrep | Q1 | architec | 49 | 30 | 3.3M | 1.6M | **−53%** ⭐ |
| flask | Q1 | architec | 42 | 20 | 1.8M | 1.2M | −35% |
| pub | Q2 | error_ha | 55 | 44 | 2.8M | 1.9M | −30% |
| CopilotKit | Q2 | error_ha | 65 | 69 | 4.3M | 3.5M | −19% |
| platform | Q5 | feature_ | 67 | 54 | 2.3M | 2.0M | −14% |

**Insight from the wins:** When v0.3 *does* win, it wins as hard as v0.2 ever
did (qgrep Q3 at −58%, qgrep Q1 at −53%). The mechanism still works — it just
fires inconsistently.

---

## 4. Root-cause analysis: what changed between v0.2 and v0.3?

The benchmark conditions are otherwise identical: same questions, same
Claude model class (Sonnet), same repos, same query construction. The
*only* thing that changed is the **MCP tool surface**:

| Aspect | v0.2 (run22) | v0.3 (run23) |
|---|---|---|
| MCP tools | 7 (agent, search, deep, index, status, health, validate) | **13** (7 above + M1-M6 structural) |
| Default toolset | implicit "all 7" | explicit `all` (13) |
| Tool descriptions | terse | per-tool with `when_to_use`, `examples` |
| Indexes | BM25 + dense (older PLAID) | BM25 + dense (PLAID 1.3) + E3 annotations + E5 PageRank + E7 patterns |

**Hypothesis (high confidence given the magnitude/uniformity of the regression):**

> v0.3 added 6 specialized structural tools (M1-M6) designed to *replace*
> 3-5 grep+read iterations each. The agent uses them **additively** rather
> than *alternatively*. It calls `semantex_symbol`, then `semantex_callers`,
> then `semantex_implementations`, then `semantex_examples`, then
> `semantex_architecture` — burning 10-30 turns to gather what
> `semantex_deep` returned in one turn in v0.2.

Evidence supporting this:
1. The wins in §3 are all v0.3 runs where the agent picked `semantex_agent` /
   `semantex_deep` first — v0.2 behavior preserved → v0.2-class win.
2. The losses in §2 show consistent 3-9× turn inflation — exactly what
   "use 4-6 sequential structural tools instead of 1 deep call" looks like.
3. Error_handling regressed worst (+79pp) because v0.2 one-shotted it with
   `semantex_deep`. The v0.3 agent likely tries `semantex_examples` (the new
   pattern catalog tool), finds no `error_handling` pattern (the catalog has
   things like `rust.tokio_spawn`, `rust.drop_impl` — no error_handling),
   then explores via the other M-tools.

---

## 5. Recovery plan (3 phases, increasing effort)

### Phase 1 — Behavioral fix via tool descriptions (1-2 hours, ~$5 to validate)

**Change:** Strengthen the `instructions` block in `McpServer::handle_initialize`
and each M1-M6 tool's `description` field to push the v0.2 behavior:

1. Make `semantex_agent` the **only** unconditional recommendation.
2. Mark M1-M6 each as "ADVANCED — call only AFTER `semantex_agent` answers and a
   *specific* detail is missing". List explicitly: "do not call multiple of
   these in sequence; if you find yourself doing that, the question should
   have gone to `semantex_agent` or `semantex_deep`."
3. Remove the `when_to_use` bullet lists from M1-M6 (they invite browsing).
4. Update SKILL.md (W6's templates) to lead with: "If you don't know which
   tool, the answer is `semantex_agent`."

**Files:** `crates/semantex-mcp/src/server.rs` (handle_initialize +
handle_tools_list); `crates/semantex-cli/src/skills/tools.rs` (template
descriptions). Regenerate skills.

**Validation:** Re-run `agent_bench` on `gin` + `flask` + `qgrep` only
(15 runs, ~$5). If aggregate CCB Δ recovers to within ±10% of baseline,
ship Phase 1 and continue measuring.

**Expected outcome:** Recovers ~50% of the regression (probably gets us to
−10% to −30% CCB, not yet at the v0.2 −56% mark but enough to claim
"competitive on agent CCB").

---

### Phase 2 — Make `core` the default toolset (1-2 days)

**Change:** Switch the MCP server's default from `all` (13 tools) to `core`
(4 tools: `semantex_agent`, `semantex_deep`, `semantex_search`,
`semantex_symbol`). M1-M6 remain available behind `--toolset structural` or
`--toolset all` for power users.

**Rationale:** The data shows every problem we have is "too many tools
visible to the agent". Default behavior should match v0.2's surface; opt-in
is the right shape for the new tools.

**Files:** `server.rs::DEFAULT_TOOLSET` from `"all"` to `"core"`. Update
SKILL.md generated by skills-generate. Update README.

**Validation:** Re-run the full agent benchmark (60 runs, ~$80) one more
time with the new default. **Expected: CCB Δ returns to v0.2-class
(−50% to −60%) because we've removed the exploration temptation entirely.**

---

### Phase 3 — Hide M1-M6 behind semantex_agent's auto-router (1-2 weeks)

**Change:** The external MCP surface is **5 tools max** (`semantex_agent`,
`semantex_deep`, `semantex_search`, `semantex_health`, `semantex_status`).
M1-M6 become **internal routes** inside `semantex_agent`'s query
classifier — when the user asks "who calls X?", `semantex_agent` internally
dispatches to the callers logic without exposing it as a separate tool.

**Rationale:** Aligns with graphify's design (the user's reference
competitor). The agent doesn't need to *know* about callers/callees as
tools; it needs to *ask the right question* and have one entry point that
routes correctly.

**Files:** Move M1-M6 handler logic into `search/agent.rs::route_query`.
Delete the tool registrations from `handle_tools_list`. Update skill files.

**Validation:** Full benchmark. **Target: −70% CCB Δ vs baseline (the
original spec target, now reachable because the agent has zero
exploration surface — it gets one good answer or asks again).**

**Risk:** Internal routing may misclassify queries that would have benefited
from explicit structural calls. Mitigate by exposing a `force_route`
parameter on `semantex_agent` for programmatic users.

---

## 6. Honest update to BENCHMARK-v0.3.md

The §0 cap-condition checklist must be updated **before** the v0.3 release.
Mark:

- [x] **B3 cap (agent CCB):** CAPPED — measured **+20%** vs ≥−50% target.
      Cap fired at run23 (2026-05-25). Claim downgraded to **"v0.3 is competitive
      with Claude Code built-ins on most question types but does NOT improve
      agent CCB at the default 13-tool toolset"** until Phase 1/2/3 above lands.

§5 "Honest losses" section gets the per-question-type table from §1 of this
document. The 12-row "worst regressions" table from §2 of this document is
the receipt — losses are tied to specific (repo, question) pairs anyone can
re-run.

---

## 7. Recommended immediate action

**Stop the v0.3 release as a SOTA claim.** Ship it as "v0.3-tools — adds the
M1-M6 structural tools as opt-in via `--toolset structural`; default
behavior unchanged from v0.2." Then execute Phase 2 fix and re-tag as
v0.3.1 with the CCB recovery measured.

The memory-safety + correctness wins from this session (the 15 confirmed
findings + the next-plaid upgrade) ship cleanly under that smaller claim —
they're real improvements regardless of the M1-M6 surface decision.
