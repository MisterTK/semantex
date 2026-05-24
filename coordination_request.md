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
