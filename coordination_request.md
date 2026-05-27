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
---

## Request 7 — v0.4 WS-A Item 4 (rusqlite 0.38 → 0.40): BLOCKED by next-plaid 1.3
**Status:** OPEN — needs cross-spec decision.

`crates/semantex-core/Cargo.toml` and `crates/semantex-mcp/Cargo.toml` were bumped from `rusqlite = "0.38"` to `"0.40"` per Spec D §5.3. Build fails with:

```
package `libsqlite3-sys` links to the native library `sqlite3`, but it
conflicts with a previous package which links to `sqlite3` as well:
  package `libsqlite3-sys v0.38.0`
    ... rusqlite v0.40.0 (semantex-core, semantex-mcp)
  vs
  package `libsqlite3-sys v0.36.0`
    ... rusqlite v0.38.0
    ... next-plaid v1.3.0 (workspace.dependencies)
```

`libsqlite3-sys` is a `links = "sqlite3"` crate; the Cargo resolver can't unify two majors of it because each tries to link the native library.

`next-plaid 1.3.1` (latest published on crates.io) pins `rusqlite = "^0.38"`. There is no `next-plaid 1.4` yet. Until next-plaid bumps its own rusqlite, semantex-core cannot move past 0.38.

**Options:**
1. **Defer Item 4 to v0.5 / v0.6** (recommended). Mark Item 4 BLOCKED in the Spec D acceptance matrix. WAL perf gain (3.51.3 → 3.53.1) is a "nice-to-have" — semantex doesn't bottleneck on SQLite right now.
2. **File an upstream PR against next-plaid** to bump its rusqlite. Wait for 1.4 release. Outside this spec's 6h budget.
3. **Vendor a forked next-plaid** with rusqlite bumped. Anti-pattern; rejected per CLAUDE.md OSS rules.

Reverted Item 4 in this workstream commit; Items 2 / 3 / 5 ship as planned. Coordinator should pick option (1) and document in Spec D §5.3 that Item 4 is deferred.

---

## Request 8 — v0.4 WS-A Item 5: Spec-prescribed `IndexingTerm` is `pub(crate)` in tantivy 0.26.1
**Status:** Resolved in-band (spec text appears to be incorrect).

Spec D §5.4.2 directs:

> change `tantivy::Term::from_field_u64(self.chunk_id_field, id)` to
> `tantivy::IndexingTerm::from_field_u64(self.chunk_id_field, id)`.
> The surrounding `self.writer.delete_term(term)` accepts the new
> `IndexingTerm` type.

In the published tantivy 0.26.1 source, `IndexingTerm` is `pub(crate)` (see `src/indexer/indexing_term.rs:16`) and is not re-exported at the crate root. `Term::from_field_u64` is still public and `IndexWriter::delete_term` still takes `Term` (`src/indexer/index_writer.rs:680`). The intended migration appears to be a no-op for this call site.

**Action taken:** kept the existing `tantivy::Term::from_field_u64` call. WS-E item 18 should verify against tantivy docs whether a future tantivy 0.27 exposes `IndexingTerm` publicly and adjust the call there.

A second tantivy 0.26 API change DID require a one-line fix (also in `sparse_search.rs`): `TopDocs::with_limit(k)` is no longer itself a `Collector`; you now chain `.order_by_score()` to obtain a Collector with the same `Vec<(Score, DocAddress)>` fruit type. This is the lazy-scorer split mentioned in Spec D §5.4.1; the fix is mechanical and additive (no behavior change for the equivalent old call).

---

## Request 9 — W-Delta (v0.5 Item 6): postcard does NOT honor `#[serde(default)]` for trailing fields
**Date:** 2026-05-26
**Branch:** `worktree-agent-a0fd7166b902fbff0`
**Status:** Resolved — Option A landed in commit `HEAD` on `worktree-agent-a0fd7166b902fbff0`
(the commit closing this request is titled `v0.5 Item 6: confidence-driven
disambiguation suggestions (protocol v1→v2)`; once pushed, the SHA can be
substituted here).

### Resolution summary (added on close)

- `BINARY_PROTOCOL_VERSION` bumped from `1` to `2` in `protocol.rs`.
- Added `pub struct DisambigSuggestion { name: String, path: String, line: usize }`
  with `Serialize, Deserialize, Debug, Clone, PartialEq` derives.
- Added `disambiguation: Option<Vec<DisambigSuggestion>>` to `SearchResponse`
  and `AgentResponse`. **Used `#[serde(default)]` only** — `skip_serializing_if`
  was attempted per the coordinator's hint but empirically breaks postcard
  decode (`DeserializeUnexpectedEnd` on roundtrip); the postcard wire format
  requires a 1-byte tag for `None` and absent fields are NOT defaulted on the
  read side. Comment on each field records the constraint so a future cleanup
  pass doesn't re-introduce `skip_serializing_if`.
- Backward-decode test `decode_v1_response_as_v2_rejected_cleanly` (plus
  symmetric request-side variant) asserts a v=1 frame surfaces as
  `BinaryFrameError::UnsupportedVersion { expected: 2, got: 1 }` — not a
  silent postcard decode error. A second test asserts `BINARY_PROTOCOL_VERSION == 2`
  so an accidental revert lights up immediately.
- `disambiguation_from_results(&[SearchResult])` in `search/agent.rs` populates
  up to 3 entries when top result confidence is `Confidence::Ambiguous`; skips
  duplicate names (vs the top result and amongst itself) and non-AST chunks
  (TextWindow/PdfPage). Called from every `handle_*` path that produces
  fresh `SearchResult`s (semantic / exact_symbol / analytical / exhaustive /
  exhaustive_structural) and propagated through `HandlerResult.disambiguation`
  into `AgentResponse.disambiguation`. Also wired into `Handler::handle_search`
  in `server/handler.rs` so the programmatic `semantex_search` path emits the
  structured field even when not going through `semantex_agent`.
- `agent_formatter::append_disambiguation_block` (new helper, scoped to
  Item 6 — does NOT touch `format_search_results`/`format_deep_results`/
  `format_graph_results`/`format_code_blocks`) renders the spec §5 format
  with the actual suggestion count substituted for the illustrative "4 distinct
  concepts" wording.
- Mechanical cascade: 13 `disambiguation: None,` site updates in non-owned
  files (`server/mod.rs` JSON decoder — 1 production site;
  `server/tests.rs` — 8 test fixtures; `server/protocol.rs` — 1 existing test
  fixture; `search/agent_formatter.rs` — 4 test fixtures). The cascade is the
  unavoidable cost of adding a public field that the coordinator approved.

Original investigation text retained below for the audit trail.

### Decision needed

Spec Q §5 Item 6's dispatch prompt and Item 6 itself both assume that adding a trailing `Option<T>` field with `#[serde(default)]` to a postcard-serialized struct is wire-format-backward-compatible. The prompt instructs:

> "Bump `BINARY_PROTOCOL_VERSION` only if removing fields; adding optional fields is backward-compatible — verify with a unit test that an older payload (without the new field) still decodes."
> "adding an `Option<T>` with `#[serde(default)]` typically does NOT require a version bump"

**This premise is empirically false for postcard 1.x.** Postcard is a non-self-describing format; the deserializer reads exactly the bytes the schema expects. `#[serde(default)]` is honored only when a containing enum-variant or struct is absent — not when a *trailing field* is absent from the byte stream. Adding `disambiguation: Option<Vec<DisambigSuggestion>>` at the end of `SearchResponse` / `AgentResponse` causes `postcard::Error::DeserializeUnexpectedEnd` on old payloads.

### Repro (verified locally on this worktree's host, postcard 1.1.3 + serde 1)

Minimal Rust program:

```rust
#[derive(Serialize, Deserialize)] struct V1 { a: u32, b: String }
#[derive(Serialize, Deserialize)] struct V2 {
    a: u32, b: String,
    #[serde(default)] c: Option<Vec<String>>,
}

let v1 = V1 { a: 42, b: "hi".into() };
let bytes = postcard::to_stdvec(&v1).unwrap();
// v1 bytes: [42, 2, 104, 105]   — no trailing byte for `c`
let v2: Result<V2, _> = postcard::from_bytes(&bytes);
// v2 from v1: Err(DeserializeUnexpectedEnd)
```

Actual run output:

```
v1 bytes: [42, 2, 104, 105]
v2 decoded from v1: Err(DeserializeUnexpectedEnd)
```

The "unit test that confirms backward decode" the prompt asks for **would fail** on the new code, blocking the Item 6 gate.

### Why this matters

`SearchResponse` and `AgentResponse` are part of `BinaryResponse`. They're serialized by the daemon to clients (CLI + MCP) via postcard, and the framing layer already carries `BINARY_PROTOCOL_VERSION` (currently 1) precisely to surface schema drift as `BinaryFrameError::UnsupportedVersion` rather than a silent mis-decode. The mechanism was added in v0.4.1 W-Index #2 for exactly this kind of change.

### Three options the controller can choose

**Option A — bump `BINARY_PROTOCOL_VERSION` from 1 to 2.**
- Old daemon ↔ new client (or vice versa) cleanly errors with `UnsupportedVersion`. User restarts daemon (one command, daemon auto-respawns). Same UX as the v0.4 postcard cutover.
- Backward-decode test becomes: "v2 client rejects v1 payload with `UnsupportedVersion`" — clean and writable.
- Overrides spec language "bump only on field removal", which is based on the empirically false postcard premise.

**Option B — defer the wire field; ship the formatting-only portion of Item 6.**
- Append the disambiguation block to `AgentResponse.formatted` (string-only) when top confidence is Ambiguous.
- Pros: zero wire-format risk, no version bump, no client coordination.
- Cons: programmatic consumers of `SearchResponse` / `AgentResponse` can't read structured suggestions. Spec §5 Item 6's "synthetic SearchResult with Ambiguous confidence produces 3 disambiguation entries" acceptance test checks the formatted string, not the struct.

**Option C — add a new `BinaryResponse::SearchV2` variant rather than mutating `SearchResponse`.**
- Postcard reads enum tags as varint; old clients reading a new `SearchV2` payload still fail at the tag level, so the protocol version bump in Option A is still effectively required for safety. Doubles surface area for little benefit. Not recommended.

**W-Delta recommendation:** Option A. One-line constant bump, plus an updated test, and the framing-layer error path is already wired. The v0.4 → v0.5 boundary is the least surprising place to require a daemon restart.

### What W-Delta is doing while you decide

- **Item 6:** STOPPED writing. The `DisambigSuggestion` struct, field additions, populate-on-Ambiguous logic, formatter helper, and tests are NOT yet written. No edits to `protocol.rs`, `agent.rs`, or `agent_formatter.rs` for Item 6.
- **Item 7 (adaptive structural walk):** wire-format-neutral (only changes call shapes inside the daemon; `GraphWalkResponse` struct is unchanged). Proceeded and committed in this branch as a separate commit so the controller can integrate Item 7 independently of the Item 6 decision.

### Resume plan once decided

- **Option A:** bump `BINARY_PROTOCOL_VERSION` 1 → 2, add `DisambigSuggestion` + the new field, write per-spec tests + a `decode_v1_response_as_v2_rejected_cleanly` test asserting `UnsupportedVersion`. ETA: ~30 min.
- **Option B:** add `format_disambiguation_block` helper, call from agent handlers when top confidence is Ambiguous, no protocol change. ETA: ~20 min.
- **Option C:** scope new variant carefully — escalate again with concrete patch sketch before writing.
