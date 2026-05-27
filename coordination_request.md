# v0.3 Coordination Requests

This file collects cross-workstream coordination notes raised during Phase 1/2
parallel execution. All marked status reflects state as of Barrier 1.

---

## Request 1 — W4 → W5: warm-state sentinel fast-path
**Status:** Resolved by W5.

W5's M1-M6 handlers and the existing tool_agent/tool_search/tool_deep_search handlers all consume `warm_state_ready` via a `detect_state_fast` helper. `tool_status` deliberately bypasses the fast-path.

---

## Request 2 — W1 → Integration: test-fixture field additions
**Status:** Resolved at W1 merge.

W1 added `confidence` and `confidence_score` to test fixtures in `adaptive.rs` and `deep.rs` (outside owned set) — three sites total. Mechanical only.

---

## Request 3 — W3 → Integration: hybrid.rs reranker constructor swap
**Status:** Resolved in Barrier 1 prep (commit `2a0a0c8`).

Swapped `FastembedReranker::new(RerankerModel::JINARerankerV1TurboEn, false)` → `FastembedReranker::new_default(false)` so `SEMANTEX_RERANKER=on` uses BGE Reranker v2 M3. Default-off behaviour unchanged.

---

## Request 4 — W7 → W5: promote handler + use `tools_for_toolset`
**Status:** Partially open.

W5 made `McpServer::tools_for_toolset(&self, &str) -> Vec<Tool>` public (server.rs:597). W7's HTTP layer should use it instead of its hardcoded `CORE_TOOLS` / `STRUCTURAL_TOOLS` constants. **Outstanding follow-up.**

W5's `handle_request` is NOT public (server.rs:362). W7 worked around this by spawning a child stdio process per HTTP server lifetime (multiplexed). To eliminate the subprocess hop, promote `handle_request` to `pub` and refactor `http_transport.rs` to call it directly. **Tracked as a v0.3.x cleanup; not required for the v0.3 ship.**

---

## Request 5 — W6 → W5: skills-generate canonical tool registry
**Status:** Open follow-up.

`crates/semantex-cli/src/skills/tools.rs` re-declares all 13 tools' metadata. Source of truth divergence — long-term, factor a shared `pub fn tool_metadata() -> Vec<ToolMetadata>` in `semantex-mcp` and re-export. Acceptable for v0.3 with the comment on the duplication.

---

## Request 6 — W-Beta (v0.3.1 Item 2) → W-Gamma / Tier 2 Item 5
**Date:** 2026-05-26
**Branch:** `v0.3.1/w-beta`
**Status:** Investigation-only outcome — `DONE_WITH_CONCERNS`. Release-sequence `docs/RELEASE-SEQUENCE-2026-05.md` §4.2 branch (a).

### Decision

**Do NOT implement the proposed `detect_languages` override in this workstream.**
**Defer the platform Q2 +69% CCB regression to Tier 2 Item 5 (deep audit).**

### Rationale

Spec Q Item 2 hypothesized that `Structural` over-matches the multi-language platform repo on generic Q2 wording, motivating a `detect_languages` helper + classifier override that routes to `Deep` when 2+ languages are present AND the query contains `"handle"` / `"deal with"` / `"support"` without a specific symbol.

§4.2 mandates that the W-Beta subagent first run the classifier on the EXACT Q2 wording from `benchmarks/agent_bench.py::QUESTIONS` before writing any production code. If the classifier already routes to `Deep`, the regression source is downstream (Deep handler on multi-language repos), and Tier 2 Item 5 already owns that work.

### Investigation result

Q2 verbatim wording from `benchmarks/agent_bench.py::QUESTIONS`:

> "How does this project handle errors? What patterns are used for error propagation, reporting, and recovery?"

Test added (see commit on `v0.3.1/w-beta`):
`crates/semantex-core/src/search/agent_classifier.rs::tests::q2_already_routes_to_deep_so_no_classifier_fix_needed` — passes green on `cargo test -p semantex-core --lib`.

The classifier returns `AgentRoute::Deep`. The Deep prefix `"how "` matches at step 5 of `classify_agent_query`, before any of the structural keywords are even checked. None of the structural keywords (callers / callees / who calls / used by / uses / depends on / references / imports / etc.) appear in the Q2 wording — the original Item 2 motivation note in the spec ("the current classifier matches 'handle' / 'uses' / 'references' in `structural_keywords` too aggressively") was an inaccurate recollection of the structural keyword list; `"handle"` is not in it.

This matches §4.2 branch (a) exactly.

### Recommendation

1. **Mark Tier 1 Item 2 done-with-concerns (no code change).** The Item 2 acceptance gate (`platform Q2 TX CCB Δ ≤ +20%`) cannot be satisfied by a classifier change because the classifier already routes correctly; any movement on that number must come from the Deep handler.
2. **Defer the platform Q2 +69% regression to Tier 2 Item 5 (deep audit).** Item 5 already owns `search/deep.rs`, `search/agent_formatter.rs`, `search/agent.rs::handle_deep`, and its investigation plan (spec §5 Item 5 steps 1–3) explicitly diffs v0.2 vs Phase 4 deep transcripts. Multi-language handler behavior fits naturally inside that audit.
3. **No `detect_languages` helper is added in this workstream.** If Item 5 finds that the deep handler legitimately needs per-language behavior, Item 5 may add it under its own ownership of `deep.rs` (and may consult W-Beta if the helper would live in `index/storage.rs`).

### Side observations (informational only)

- The spec's proposed schema for `detect_languages` assumed a `language_name` column directly on the `chunks` table. The actual layout stores language inside the JSON-encoded `chunk_type` column (variant `AstNode { language: String, ... }` in `crates/semantex-core/src/types.rs`). Any future implementation will need to extract language via `json_extract(chunk_type, '$.AstNode.language')` or by deserializing `chunk_type`, not by selecting `language_name` directly. Does not change the §4.2 branch (a) decision.
- TextWindow and PdfPage chunks have no language field, so a future helper would need to either filter them out or return only AstNode languages.

### Files touched

- `crates/semantex-core/src/search/agent_classifier.rs` — added one `#[cfg(test)]` test `q2_already_routes_to_deep_so_no_classifier_fix_needed` plus a docblock pointing at §4.2. No production code modified.
- `coordination_request.md` (this entry).

### Verification

```
cargo build  -p semantex-core --release            # clean
cargo test   -p semantex-core --lib                # 589 passed, 1 ignored, 0 failed
cargo clippy -p semantex-core -- -D warnings       # clean
cargo fmt    --all --check                         # clean
```
