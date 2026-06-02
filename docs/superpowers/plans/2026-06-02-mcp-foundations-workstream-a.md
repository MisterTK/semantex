# MCP Foundations (Workstream A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Make semantex's MCP server best-practice-compliant and its output defaults env-tunable, so (a) the bare-MCP system sweep (Workstream B/C) can vary "our defaults" server-side, and (b) the product is better for any MCP client. From spec `docs/superpowers/specs/2026-06-02-whole-system-pareto-tuning-design.md` §4.

**Architecture:** Two gated PRs. **A1** (Tasks 1–5) — localized to `crates/semantex-mcp/src/server.rs`: env-tunable `budget`/`full_code`/`depth` (the sweep blocker), a `concise|detailed` `response_format` enum, a self-describing `semantex_agent` description, `$schema` 2020-12 dialect, and error/snapshot hygiene. **A2** (Tasks 6–7) — threads structured hits through the agent pipeline (`crates/semantex-core/src/search/agent.rs`) so `semantex_agent` can emit `output_schema`+`structuredContent` while keeping the prose summary in `content[]` (the Claude Code #55677 footgun fix).

**Tech Stack:** Rust (semantex-mcp + semantex-core). MCP spec 2025-11-25. Build: `cargo build -p semantex-mcp`; test: `cargo test -p semantex-mcp` / `-p semantex-core`. macOS e2e: `ORT_DYLIB_PATH=/opt/homebrew/lib/libonnxruntime.dylib`. Full gate before each PR lands: `cargo build --workspace && cargo test --workspace && cargo clippy --all && cargo fmt --all --check` + `actually verify_change`; zero-LLM default preserved (`cargo tree | grep genai` empty).

## Verified anchors (current code)
- `crates/semantex-mcp/src/server.rs`: `McpServer` struct `131`; `new` `161`/`with_toolset` `168`; `all_tools` `~728`; `semantex_agent` tool def `730-776` (description `733-739`, input_schema `740-773`, `output_schema: None` `774`); `tool_agent` `1118-1239` (full_code default `1150-1153`, depth parse `1156-1161`, budget default `1205-1208`, response build `1238` `ToolOutput::text`); dispatch builds `content[]`+`structured_content` from `output.text`/`output.structured` `1060-1069`; `ToolOutput` struct+`text()` `2600-2613`; inline tests `2740`.
- `crates/semantex-core/src/search/agent.rs`: `HandlerResult` `43-53`; `AgentRequest`→`handle` `144-198` (`AgentResponse{route,formatted,metrics,disambiguation}` `182-197`); handlers building `Vec<SearchResultItem>` then discarding it at `300,395,428,530,590,628,810,977`.
- `crates/semantex-core/src/server/protocol.rs`: `SearchResultItem` `239-256` (`file,start_line,end_line,score,source,chunk_type,name?,language?,content?,kind?,summary?`).
- Env helper to MIRROR (do NOT reuse — it's `pub(crate)` in semantex-core): `config::env_usize` `301`, `env_string` `316`.

---

## Task 1: Env-tunable MCP output defaults (the sweep blocker)

Add `SEMANTEX_MCP_BUDGET` (usize bytes), `SEMANTEX_MCP_FULL_CODE` (0/1), `SEMANTEX_MCP_DEPTH` (quick|search|deep|auto). Read ONCE at construction (matches the in-process model). A per-call MCP arg still overrides the env default.

**Files:**
- Modify: `crates/semantex-mcp/src/server.rs` (new `McpAgentDefaults` near `ToolOutput` ~2600; `McpServer` field ~131; `with_toolset` ~168; `tool_agent` `1150-1208`; tests `2740`)

- [ ] **Step 1: Write the failing test** (in `server.rs` `mod tests`, ~2740)

```rust
    #[test]
    fn mcp_agent_defaults_from_env_overrides_and_falls_back() {
        // No env -> shipped defaults.
        // SAFETY: tests are single-threaded per process here; serialize env via the
        // dedicated key names so no other test touches them.
        unsafe {
            std::env::remove_var("SEMANTEX_MCP_BUDGET");
            std::env::remove_var("SEMANTEX_MCP_FULL_CODE");
            std::env::remove_var("SEMANTEX_MCP_DEPTH");
        }
        let d = McpAgentDefaults::from_env();
        assert_eq!(d.budget, 12_000);
        assert!(!d.full_code);
        assert_eq!(d.depth, None); // None == auto-classify

        unsafe {
            std::env::set_var("SEMANTEX_MCP_BUDGET", "6000");
            std::env::set_var("SEMANTEX_MCP_FULL_CODE", "1");
            std::env::set_var("SEMANTEX_MCP_DEPTH", "deep");
        }
        let d = McpAgentDefaults::from_env();
        assert_eq!(d.budget, 6_000);
        assert!(d.full_code);
        assert_eq!(d.depth.as_deref(), Some("deep"));

        // "auto" is treated as no override.
        unsafe { std::env::set_var("SEMANTEX_MCP_DEPTH", "auto"); }
        assert_eq!(McpAgentDefaults::from_env().depth, None);
        unsafe {
            std::env::remove_var("SEMANTEX_MCP_BUDGET");
            std::env::remove_var("SEMANTEX_MCP_FULL_CODE");
            std::env::remove_var("SEMANTEX_MCP_DEPTH");
        }
    }
```

- [ ] **Step 2: Run it; expect FAIL** (`McpAgentDefaults` undefined)

Run: `cargo test -p semantex-mcp mcp_agent_defaults_from_env -- --nocapture`
Expected: FAIL — `cannot find type/struct McpAgentDefaults`.

- [ ] **Step 3: Implement `McpAgentDefaults`** (add near `ToolOutput`, ~`server.rs:2598`)

```rust
/// Server-side defaults for the `semantex_agent` output shape, read ONCE from
/// the environment at MCP server construction (the in-process config model).
/// A per-call MCP arg always overrides these. `depth == None` means auto-classify.
#[derive(Debug, Clone)]
struct McpAgentDefaults {
    budget: usize,
    full_code: bool,
    depth: Option<String>,
}

impl McpAgentDefaults {
    fn from_env() -> Self {
        let budget = std::env::var("SEMANTEX_MCP_BUDGET")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|&b| b > 0)
            .unwrap_or(12_000);
        let full_code = std::env::var("SEMANTEX_MCP_FULL_CODE")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let depth = match std::env::var("SEMANTEX_MCP_DEPTH")
            .ok()
            .map(|v| v.trim().to_ascii_lowercase())
            .as_deref()
        {
            Some("quick") => Some("quick".to_string()),
            Some("search") => Some("search".to_string()),
            Some("deep") => Some("deep".to_string()),
            _ => None, // "auto", empty, unset, or unknown -> auto-classify
        };
        Self { budget, full_code, depth }
    }
}
```

- [ ] **Step 4: Run it; expect PASS**

Run: `cargo test -p semantex-mcp mcp_agent_defaults_from_env`
Expected: PASS.

- [ ] **Step 5: Wire `McpAgentDefaults` into `McpServer`**

Add a struct field after `toolset: String,` (~`server.rs:145`):
```rust
    /// Server-side `semantex_agent` output defaults (budget/full_code/depth),
    /// read once from env at construction. Per-call MCP args override these.
    mcp_defaults: McpAgentDefaults,
```
In `with_toolset` (the real constructor, ~`168`), set the field in the returned `Self { ... }` literal:
```rust
            mcp_defaults: McpAgentDefaults::from_env(),
```

- [ ] **Step 6: Apply the defaults in `tool_agent`** (`server.rs`)

`full_code` (replace `1150-1153`):
```rust
        let full_code = args
            .get("full_code")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(self.mcp_defaults.full_code);
```
`depth_route` (replace `1156-1161` — fall back to the env default when the arg is absent):
```rust
        let depth_arg = args
            .get("depth")
            .and_then(|v| v.as_str())
            .or(self.mcp_defaults.depth.as_deref());
        let depth_route: Option<AgentRoute> = match depth_arg {
            Some("quick") => Some(AgentRoute::ExactSymbol),
            Some("search") => Some(AgentRoute::Semantic),
            Some("deep") => Some(AgentRoute::Deep),
            _ => None,
        };
```
`budget` (replace `1205-1208`):
```rust
        let budget = args
            .get("budget")
            .and_then(serde_json::Value::as_u64)
            .map_or(self.mcp_defaults.budget, |v| v as usize);
```

- [ ] **Step 7: Update the input_schema descriptions** (`server.rs:756,760`) so they don't lie about hardcoded defaults:
  - `full_code` desc → `"Include full source code blocks (default: server SEMANTEX_MCP_FULL_CODE, normally false)"`
  - `budget` desc → `"Response size budget in bytes (default: server SEMANTEX_MCP_BUDGET, normally 12000 ≈3K tokens)"`

- [ ] **Step 8: Build + full test + clippy/fmt**

Run: `cargo build -p semantex-mcp && cargo test -p semantex-mcp && cargo clippy -p semantex-mcp && cargo fmt --all`
Expected: green; no warnings.

- [ ] **Step 9: Commit**

```bash
git add crates/semantex-mcp/src/server.rs
git commit -m "feat(mcp): env-tunable semantex_agent defaults (SEMANTEX_MCP_BUDGET/_FULL_CODE/_DEPTH)"
```

---

## Task 2: `response_format` enum (concise|detailed) → budget tiers

A discrete output-volume knob (best practice "return less by default") + a clean sweep axis. Maps to a budget multiplier applied to the effective budget when the caller did NOT pass an explicit `budget`.

**Files:** Modify `crates/semantex-mcp/src/server.rs` (input_schema ~772; `tool_agent` budget block ~1205; tests 2740)

- [ ] **Step 1: Failing test**

```rust
    #[test]
    fn response_format_maps_to_budget_tier() {
        // concise ~= 1/3, detailed = base, default(None) = base.
        assert_eq!(budget_for_format(Some("concise"), 12_000), 4_000);
        assert_eq!(budget_for_format(Some("detailed"), 12_000), 12_000);
        assert_eq!(budget_for_format(None, 12_000), 12_000);
        assert_eq!(budget_for_format(Some("bogus"), 12_000), 12_000);
    }
```

- [ ] **Step 2: Run; expect FAIL** (`budget_for_format` undefined). `cargo test -p semantex-mcp response_format_maps`

- [ ] **Step 3: Implement** the pure helper (near `apply_focus`, free fn):

```rust
/// Map an optional `response_format` to an effective budget given the base budget.
/// `concise` ≈ ⅓ tokens (best-practice "return less by default"); `detailed`/None = base.
fn budget_for_format(response_format: Option<&str>, base_budget: usize) -> usize {
    match response_format {
        Some("concise") => (base_budget / 3).max(1),
        _ => base_budget, // "detailed", None, or unknown -> base
    }
}
```

- [ ] **Step 4: Run; expect PASS.** `cargo test -p semantex-mcp response_format_maps`

- [ ] **Step 5: Wire into `tool_agent`** — after the `budget` block (~`1208`), apply the format ONLY when the caller didn't pin a raw `budget`:

```rust
        let response_format = args.get("response_format").and_then(|v| v.as_str());
        let budget = if args.get("budget").is_some() {
            budget // explicit byte budget wins
        } else {
            budget_for_format(response_format, budget)
        };
```

- [ ] **Step 6: Add to input_schema** (after the `focus` property, ~`771`):

```rust
                        "response_format": {
                            "type": "string",
                            "enum": ["concise", "detailed"],
                            "description": "Output verbosity. 'concise' returns ~1/3 the context (paths + minimal snippets); 'detailed' returns fuller results. Omit for the server default. Ignored if an explicit 'budget' is set."
                        }
```

- [ ] **Step 7: Build + test + clippy/fmt** (as Task 1 Step 8).

- [ ] **Step 8: Commit** — `git commit -am "feat(mcp): response_format enum (concise|detailed) -> budget tiers on semantex_agent"`

---

## Task 3: Self-describing `semantex_agent` description (no CLAUDE.md reliance)

Restate the usage contract IN-BAND so any client (no CLAUDE.md/skill) uses the tool correctly.

**Files:** Modify `crates/semantex-mcp/src/server.rs:733-739`

- [ ] **Step 1: Failing test** (assert the description carries the contract)

```rust
    #[test]
    fn agent_tool_description_is_self_describing() {
        let agent = McpServer::all_tools()
            .into_iter()
            .find(|t| t.name == "semantex_agent")
            .expect("semantex_agent present");
        let d = agent.description.to_lowercase();
        assert!(d.contains("one call"), "must state one-call contract: {d}");
        assert!(d.contains("do not chain") || d.contains("don't chain"), "must say don't chain: {d}");
        assert!(d.contains("refine"), "must tell the agent to refine the query: {d}");
    }
```
(If `all_tools` is private, make it `pub(crate)` — it has no callers outside the impl; add `pub(crate)` to its `fn`.)

- [ ] **Step 2: Run; expect FAIL** (current description lacks these phrases). `cargo test -p semantex-mcp agent_tool_description_is_self`

- [ ] **Step 3: Replace the description** (`server.rs:733-739`):

```rust
                description: concat!(
                    "Intelligent code search: auto-classifies the query (semantic, exact symbol, ",
                    "graph walk, deep search, regex, file glob), runs the best strategy with fallbacks, ",
                    "and returns a ready-to-use answer. THE recommended tool for ALL code-search needs — ",
                    "one call in, one complete answer out. Do NOT chain semantex_* calls to assemble an ",
                    "answer; if the result is incomplete, refine the QUESTION and call semantex_agent again. ",
                    "Accepts a natural-language question, a symbol, a regex, or a glob."
                ).into(),
```

- [ ] **Step 4: Run; expect PASS.** `cargo test -p semantex-mcp agent_tool_description_is_self`

- [ ] **Step 5: Build + test + clippy/fmt.**

- [ ] **Step 6: Commit** — `git commit -am "feat(mcp): make semantex_agent description self-describing (no CLAUDE.md reliance)"`

---

## Task 4: JSON Schema 2020-12 dialect + error hygiene

Declare the schema dialect (SEP-1613) and confirm bad-input errors surface as Tool Execution Errors (`isError:true`) so the model self-corrects.

**Files:** Modify `crates/semantex-mcp/src/server.rs` (each tool's `input_schema` top object; `tool_agent` empty-query path ~1131; tests)

- [ ] **Step 1: Failing test** — assert the agent inputSchema declares the 2020-12 dialect:

```rust
    #[test]
    fn agent_input_schema_declares_2020_12_dialect() {
        let agent = McpServer::all_tools().into_iter().find(|t| t.name == "semantex_agent").unwrap();
        assert_eq!(
            agent.input_schema.get("$schema").and_then(|v| v.as_str()),
            Some("https://json-schema.org/draft/2020-12/schema")
        );
    }
```

- [ ] **Step 2: Run; expect FAIL.** `cargo test -p semantex-mcp agent_input_schema_declares`

- [ ] **Step 3: Add `"$schema"` to the agent input_schema** (top of the json! at `740`):

```rust
                input_schema: serde_json::json!({
                    "$schema": "https://json-schema.org/draft/2020-12/schema",
                    "type": "object",
                    "properties": {
```
(Apply the same `"$schema"` line to the other tools' `input_schema` and `output_schema` objects for consistency — `semantex_search` `788`/`799`, `semantex_index` `837`, and the rest in `all_tools`.)

- [ ] **Step 4: Run; expect PASS.**

- [ ] **Step 5: Error-hygiene test** — the existing empty-`queries` and missing-`query` paths already `return Err(...)`, and the dispatch maps `Err` → `is_error: Some(true)` (`server.rs:1091`). Add a regression test that a bad agent call yields an `isError` result with actionable text (drive `tool_call` for `semantex_agent` with `{}` and assert the `ToolCallResult.is_error == Some(true)` and the text names the missing param). Use the existing dispatch entry (find the `tool_call`/`dispatch` fn that returns `JsonRpcResponse`); assert on its `ToolCallResult`. If driving the full JSON-RPC is heavy, instead assert `tool_agent(&json!({}))` returns `Err` whose message contains `"query"`:

```rust
    #[test]
    fn agent_missing_query_errors_actionably() {
        let srv = McpServer::new(SemantexConfig::default());
        let err = srv.tool_agent(&serde_json::json!({})).unwrap_err();
        assert!(err.to_string().contains("query"), "actionable: {err}");
    }
```

- [ ] **Step 6: Run; expect PASS** (behavior already exists; this pins it). Build + test + clippy/fmt.

- [ ] **Step 7: Commit** — `git commit -am "feat(mcp): declare JSON Schema 2020-12 dialect + pin actionable input-error hygiene"`

---

## Task 5: Schema/output snapshot test (regression-safety) — closes A1

Pin the agent tool's schema shape so descriptions/schemas stay self-contained and don't silently drift.

**Files:** Modify `crates/semantex-mcp/src/server.rs` tests

- [ ] **Step 1: Add the snapshot-style test**

```rust
    #[test]
    fn agent_tool_schema_shape_is_stable() {
        let t = McpServer::all_tools().into_iter().find(|x| x.name == "semantex_agent").unwrap();
        let props = t.input_schema["properties"].as_object().unwrap();
        for k in ["query","queries","path","full_code","budget","depth","focus","response_format"] {
            assert!(props.contains_key(k), "agent must expose '{k}'");
        }
        assert!(t.annotations.is_some(), "agent must keep read-only annotations");
    }
```

- [ ] **Step 2: Run; expect PASS** (all present after Tasks 1–4). `cargo test -p semantex-mcp agent_tool_schema_shape`

- [ ] **Step 3: A1 full gate + commit**

```bash
cargo build --workspace && cargo test --workspace && cargo clippy --all && cargo fmt --all --check
cargo tree | grep -i genai && echo FAIL || echo "zero-LLM ok"
git commit -am "test(mcp): pin semantex_agent schema shape (A1 complete)"
```
Then `actually verify_change`; land A1 as PR `feat/mcp-foundations-a1` (per-PR branch, rebase-merge after green) — this UNBLOCKS the Workstream B/C sweep (env-tunable defaults).

---

## Task 6 (A2): Thread structured hits through the agent pipeline

`semantex_agent` returns prose only; to emit `structuredContent` we surface the `Vec<SearchResultItem>` the handlers already build (then discard). Synthesis routes (deep/architecture) that have no clean item list leave it empty (their answer still rides in `content[]`).

**Files:** Modify `crates/semantex-core/src/search/agent.rs` (`HandlerResult` 43; the item-list handlers; `AgentResponse` 182), and `crates/semantex-core/src/search/agent.rs` callers of `AgentResponse`.

- [ ] **Step 1: Failing test** (in `agent.rs` tests — assert a semantic query yields non-empty `hits`)

```rust
    #[test]
    fn agent_response_exposes_structured_hits_for_item_routes() {
        // Build a tiny in-memory index (reuse the existing agent-test fixture helper
        // in this module — see the other handle_* tests) and run a Semantic route.
        let pipeline = test_pipeline_with_fixture(); // existing helper in agent.rs tests
        let resp = pipeline.handle(&AgentRequest {
            query: "authenticate user".into(),
            route: Some(AgentRoute::Semantic),
            budget: Some(12_000),
            full_code: false,
        });
        assert!(!resp.hits.is_empty(), "Semantic route must surface structured hits");
        assert!(resp.hits.iter().all(|h| !h.file.is_empty()));
    }
```
(If no in-module fixture helper exists, add a minimal one mirroring the existing `handle_*` unit tests in `agent.rs`.)

- [ ] **Step 2: Run; expect FAIL** (`AgentResponse` has no `hits`). `cargo test -p semantex-core agent_response_exposes_structured`

- [ ] **Step 3: Add `hits` to `HandlerResult`** (`agent.rs:43`):

```rust
pub(crate) struct HandlerResult {
    formatted: String,
    fallback_used: bool,
    result_count: usize,
    /// Structured hits for the item-list routes (semantic/exact/structural/
    /// exhaustive/regex/file_pattern/analytical). Empty for synthesis routes
    /// (deep/architecture) whose answer is prose-only. Surfaced to MCP
    /// `structuredContent`; the prose answer always rides in `content[]`.
    hits: Vec<crate::server::protocol::SearchResultItem>,
    disambiguation: Option<Vec<crate::server::protocol::DisambigSuggestion>>,
}
```

- [ ] **Step 4: Populate `hits` in every `HandlerResult { ... }` constructor.** For the item-list handlers (`300,395,428,530,590,628,810` + the analytical/exhaustive sites), the local `items: Vec<SearchResultItem>` is moved into a `SearchResponse`. Clone it for `hits` BEFORE the move, i.e. change `results: items,` to keep a copy:
```rust
                let count = items.len();
                let hits = items.clone();
                let resp = SearchResponse { results: items, /* ...unchanged... */ };
                // ...
                HandlerResult { formatted, fallback_used: is_fallback, result_count: count, hits, disambiguation: disambig }
```
For the **no-result / synthesis / deep-fallback** `HandlerResult { ... }` constructors (e.g. the `_ =>` empties, `deep_search` paths, `handle_deep`, `handle_architecture`), add `hits: Vec::new(),`. (Compiler will flag every `HandlerResult` literal missing the field — fix each; ~12 sites.)

- [ ] **Step 5: Add `hits` to `AgentResponse`** (`agent.rs:182`) and populate it from `result.hits`:
```rust
        AgentResponse {
            route,
            formatted,
            hits: result.hits,
            metrics: AgentMetrics { /* unchanged */ },
            disambiguation: result.disambiguation,
        }
```
Add the field to the `AgentResponse` struct definition (find it via `grep -n "pub struct AgentResponse" crates/semantex-core/src/...`):
```rust
    /// Structured hits for item-list routes (empty for synthesis routes).
    pub hits: Vec<crate::server::protocol::SearchResultItem>,
```

- [ ] **Step 6: Run; expect PASS.** `cargo test -p semantex-core agent_response_exposes_structured`. Then `cargo build -p semantex-core && cargo test -p semantex-core` (fix any other `AgentResponse`/`HandlerResult` literal the compiler flags). Commit:
```bash
git commit -am "feat(core): surface structured hits on AgentResponse for MCP structuredContent (A2 step 1)"
```

---

## Task 7 (A2): Emit `output_schema` + `structuredContent` on `semantex_agent`

**Files:** Modify `crates/semantex-mcp/src/server.rs` (`ToolOutput` 2600; agent tool def `output_schema` 774; `tool_agent` response build 1210-1238; tests)

- [ ] **Step 1: Failing test** — `tool_agent` output carries structured results AND keeps text:

```rust
    #[test]
    fn agent_output_has_structured_and_text() {
        let srv = McpServer::new(SemantexConfig::default());
        // Drive against this repo's own .semantex if present, else skip cleanly.
        let cwd = std::env::current_dir().unwrap();
        if !cwd.join(".semantex/meta.json").exists() { return; }
        let out = srv.tool_agent(&serde_json::json!({"query":"dense backend","depth":"search"})).unwrap();
        assert!(!out.text.is_empty(), "prose summary must stay in content[]");
        // structured may be None on a ripgrep fallback; when present it's an object with `results`.
        if let Some(s) = &out.structured {
            assert!(s.get("results").is_some(), "structuredContent has results[]");
        }
    }
```

- [ ] **Step 2: Run; expect FAIL/observe** (`out.structured` always None today). `cargo test -p semantex-mcp agent_output_has_structured -- --nocapture`

- [ ] **Step 3: Add a `ToolOutput` constructor** (`server.rs:2605`):
```rust
    fn text_with_structured(text: String, structured: serde_json::Value) -> Self {
        Self { text, structured: Some(structured) }
    }
```

- [ ] **Step 4: Build structuredContent in `tool_agent`** — collect hits across queries; serialize to the search-mirroring shape; keep `combined` text. Replace the final `Ok(ToolOutput::text(combined))` (`1238`):
```rust
        // Collect structured hits across all queries (item-list routes only).
        let mut all_hits: Vec<serde_json::Value> = Vec::new();
        for q in &queries {
            // (re-run is avoided: collect inside the existing per-query loop instead — see Step 4b)
        }
        // ...see Step 4b: build `all_hits` inside the loop at 1211-1221...
        if all_hits.is_empty() {
            Ok(ToolOutput::text(combined))
        } else {
            let structured = serde_json::json!({ "results": all_hits });
            Ok(ToolOutput::text_with_structured(combined, structured))
        }
```

- [ ] **Step 4b: Capture hits in the per-query loop** (`1211-1221`) — push each response's hits:
```rust
        let mut parts: Vec<String> = Vec::with_capacity(queries.len());
        let mut all_hits: Vec<serde_json::Value> = Vec::new();
        for q in &queries {
            let request = AgentRequest { query: q.clone(), route: depth_route, budget: Some(budget), full_code };
            let response = pipeline.handle(&request);
            for h in &response.hits {
                all_hits.push(serde_json::json!({
                    "file": h.file,
                    "lines": format!("{}-{}", h.start_line, h.end_line),
                    "score": h.score,
                    "name": h.name,
                    "snippet": h.summary.clone().or_else(|| h.content.clone()).unwrap_or_default(),
                    "lang": h.language,
                }));
            }
            parts.push(response.formatted);
        }
```
(Remove the duplicate `all_hits`/loop sketch from Step 4; keep this single loop.)

- [ ] **Step 5: Set the agent `output_schema`** (`server.rs:774`, mirroring `semantex_search` `799-827`):
```rust
                output_schema: Some(serde_json::json!({
                    "$schema": "https://json-schema.org/draft/2020-12/schema",
                    "type": "object",
                    "properties": {
                        "results": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "file": {"type":"string"},
                                    "lines": {"type":"string"},
                                    "score": {"type":"number"},
                                    "name": {"type":["string","null"]},
                                    "snippet": {"type":"string"},
                                    "lang": {"type":["string","null"]}
                                },
                                "required": ["file","lines","score","snippet"]
                            }
                        }
                    }
                })),
```

- [ ] **Step 6: Run; expect PASS** (`cargo test -p semantex-mcp agent_output_has_structured`). The dispatch (`server.rs:1060-1069`) already puts `output.text` into `content[]` AND `output.structured` into `structured_content` — so the #55677 footgun (content[] dropped) is avoided: text is always present.

- [ ] **Step 7: A2 full gate + land**
```bash
cargo build --workspace && cargo test --workspace && cargo clippy --all && cargo fmt --all --check
cargo tree | grep -i genai && echo FAIL || echo "zero-LLM ok"
```
`actually verify_change`; commit + land A2 as PR `feat/mcp-foundations-a2`:
```bash
git commit -am "feat(mcp): semantex_agent emits output_schema + structuredContent (keeps content[] prose; #55677-safe)"
```

---

## Self-review notes
- **Spec coverage:** §4.1 env defaults → Task 1; §4.2 response_format → Task 2; §4.3 output_schema/structuredContent + footgun → Tasks 6–7; §4.4 self-describing description → Task 3; §4.5 error hygiene → Task 4; §4.6 schema dialect + snapshot → Tasks 4–5. All §4 items covered.
- **Sequencing:** Task 1 is the Workstream-B/C sweep blocker → land A1 (Tasks 1–5) FIRST. A2 (6–7) is product quality, not a sweep blocker.
- **Type consistency:** `McpAgentDefaults{budget:usize,full_code:bool,depth:Option<String>}`, `budget_for_format(Option<&str>,usize)->usize`, `HandlerResult.hits:Vec<SearchResultItem>`, `AgentResponse.hits:Vec<SearchResultItem>`, `ToolOutput::text_with_structured(String,Value)` — names used identically across tasks.
- **Risk:** Task 6 touches ~12 `HandlerResult` literals (compiler-enforced exhaustive) + the `AgentResponse` definition + any other constructor; build after Step 4 catches them all. If a synthesis route (deep/architecture) is awkward, `hits: Vec::new()` is always valid.
- **Out of scope (this plan):** Workstream B (bare-MCP harness) and C (the sweep) get their own plans after A lands.
