# Benchmark Comparison Set — semantex v0.3

This document pins exact versions and invocation patterns for every comparison
tool referenced by the v0.3 SOTA benchmark suite (spec §4.2). It is the
contract `benchmarks/run_public.py` honors when it decides whether to run a
given competitor or skip with a "not installed" note.

**Honesty rule.** When a tool can't be verified at this exact pin from the
W8 scaffolding stage, the row says `verification: not verified`. The human
running the real sweep will replace that with the actual install transcript.

---

## 1. semantex v0.3 (this project)

| Field | Value |
|---|---|
| Version pin | Release tag `v0.3.0` (target). Today: built from `v0.3-integration` branch. |
| Install | `cargo build --release -p semantex-cli` then copy `target/release/semantex` to `$PATH` |
| Version cmd | `semantex --version` |
| Invoke (warm) | `semantex --json --max-count 10 -p <repo> "<query>"` |
| Invoke (refs only) | `semantex --refs --max-count 10 -p <repo> "<query>"` |
| MCP | `semantex mcp` (stdio) or `semantex mcp --http --port 5050` |
| Configs to test | default + `SEMANTEX_RERANKER=on` (W3 cross-encoder) + `SEMANTEX_FUSION=cc` (legacy fusion, W1) |

---

## 2. ripgrep 14.x

| Field | Value |
|---|---|
| Version pin | `>=14.0.0,<15.0.0` — `cargo install ripgrep --version "^14"` or `brew install ripgrep` |
| Install (macOS) | `brew install ripgrep` |
| Install (Linux, Cargo) | `cargo install ripgrep --version "^14.1.0"` |
| Version cmd | `rg --version | head -1` |
| Verification | Run `rg --version | grep -E "^ripgrep 14"` — exit 0 if pinned. |
| Invoke (B1/B2 lexical baseline) | `rg --json --max-count 20 --hidden -L "<pattern>" <repo>` |
| Notes | Pure regex baseline. For NL→code (B2) queries, fall back to a tokenization heuristic — split the NL query into stemmed alternation: `(error\|fail\|panic).*handl(e\|ing\|ed)`. **The query-rewrite logic is owned by `benchmarks/v0_3/competitor_ripgrep.py`; it must be deterministic and reviewed for fairness.** |

---

## 3. graphify 0.8.x

| Field | Value |
|---|---|
| Version pin | `>=0.8.0,<0.9.0` |
| Install | `npm i -g graphify@^0.8.0` (graphify ships via NPM as of 2026) |
| Version cmd | `graphify --version` |
| Verification | not verified — install URL has not been hit from this scaffolding stage |
| Invoke (query mode) | `graphify query --json --repo <repo> "<query>"` |
| Invoke (MCP mode) | `graphify mcp` (stdio) — exposes `graphify_query`, `graphify_graph`, etc. |
| Notes | Direct KG competitor. Spec §4.2 wants both query and MCP modes measured. If MCP exposes a different tool-set than query mode, prefer MCP for B3 (agent CCB) so we measure the realistic agent surface. Graphify ships an HTML graph and `GRAPH_REPORT.md` — irrelevant to B1/B2 retrieval but contributes context if pasted into B3 transcripts. |

---

## 4. Claude Code built-in tools (Grep/Glob/Read)

| Field | Value |
|---|---|
| Version pin | `claude --version` ≥ the version used for the v0.2.x baseline (CCB run17). Document the exact build at sweep time. |
| Install | `npm i -g @anthropic-ai/claude-code` or the installer at https://claude.ai/download |
| Version cmd | `claude --version` |
| Verification | `claude` is installed locally (per agent_bench.py's `shutil.which("claude")`). Specific build is not pinned at scaffolding time. |
| Invoke | `claude --headless --tools "Grep,Glob,Read,Bash" -p "<prompt>"` (no MCP). Compared as the agent_bench.py "baseline" arm. |
| Notes | This is the dominant agent stack we benchmark against. The CCB delta vs. Claude+semantex MCP is the primary B3 / B5 number. Existing `benchmarks/agent_bench.py` already drives this. |

---

## 5. Cursor — index search

| Field | Value |
|---|---|
| Version pin | Latest Cursor release at sweep time |
| Install | https://cursor.sh/download |
| Version cmd | (Cursor reports version in app About dialog) |
| Verification | not verified — Cursor's index API surface is private as of 2026-05. |
| Invoke | If Cursor exposes its index via MCP at sweep time, register the MCP server and treat as alternative tool. Otherwise: **exclude from B1/B2/B5 with explicit "excluded — no public retrieval API" note in `BENCHMARK-v0.3.md`**. |
| Notes | Per spec §4.2: "Via MCP if exposed; otherwise excluded with note." We do not paste Cursor results obtained interactively — that would be uncontrolled and unreproducible. |

---

## 6. GitHub Copilot — symbol search

| Field | Value |
|---|---|
| Version pin | `gh copilot --version` ≥ 1.x at sweep time |
| Install | `brew install gh` then `gh extension install github/gh-copilot` |
| Version cmd | `gh --version && gh copilot --version` |
| Verification | not verified |
| Invoke | If Copilot CLI exposes a symbol-search subcommand, use it. Otherwise: **exclude from retrieval benchmarks with note**, include in B3 only when the agent stack is `claude` + Copilot. |
| Notes | Spec §4.2: "API if available; otherwise excluded." |

---

## 7. lat.md `lat search`

| Field | Value |
|---|---|
| Version pin | Latest stable from https://github.com/1st1/lat.md at sweep time |
| Install | Follow lat.md README — current as of dump SHA in spec Appendix A |
| Version cmd | `lat --version` |
| Verification | not verified |
| Invoke | `lat search --json --repo <repo> "<query>"` (exact flag set TBD at sweep time — check `lat search --help`) |
| Notes | "Semantic-only" competitor per spec. Different shape from semantex (agent-maintained markdown KG vs. embedding-based retrieval) — useful as a "different category but same end-user goal" comparator. |

---

## 8. Plain dense baselines

Single-vector dense retrieval over the same corpus, no hybrid. These exist
to isolate "what does ColBERT MaxSim buy you over plain HNSW".

| Model | Install | Notes |
|---|---|---|
| `jinaai/jina-embeddings-v2-base-code` | `pip install sentence-transformers fastembed && python -m fastembed download jinaai/jina-embeddings-v2-base-code` | Code-tuned, 137M params. |
| `voyage-code-3` | Voyage AI API (requires key) — `pip install voyageai` | Closed-source SOTA dense baseline. Set `VOYAGE_API_KEY`. |
| `text-embedding-3-large` | OpenAI API — `pip install openai`, set `OPENAI_API_KEY` | Generic strong dense baseline. |
| ColBERT-stock | Use `colbert-ir/colbertv2.0` directly via PyLate — does NOT use semantex's int8 ONNX variant | Verifies semantex's PLAID indexing implementation is not regressing the underlying ColBERT signal. |

| Field | Value |
|---|---|
| Install (HNSW driver) | `pip install hnswlib` |
| Reference indexer | `benchmarks/v0_3/baseline_dense_index.py` — TBD by human at sweep time |
| Verification | not verified |

The W8 scaffolding does NOT build the dense-baseline indices itself — they
take hours on the B1 dataset. The reproducer script will skip them with a
clear `BASELINE_DENSE not built — skipping` stderr message.

---

## 9. Plain sparse baseline — Tantivy BM25

| Field | Value |
|---|---|
| Version | Tantivy 0.22.x (matches semantex's own dependency pin) |
| Install | Built as a tiny Rust binary at `benchmarks/v0_3/bm25_only/`. The human builds with `cargo build --release` in that subdir. |
| Verification | not verified — directory is created lazily by the reproducer |
| Invoke | `bm25_only --repo <repo> --query "<q>" --top 10` |
| Notes | Stock BM25 with the default English stemmer. No language-aware tokenization. Establishes "what does the hybrid stack buy over BM25 alone." |

---

## 10. Hybrid baseline — bge-large + bm25 + RRF

| Field | Value |
|---|---|
| Models | `BAAI/bge-large-en-v1.5` (dense) + Tantivy stock BM25 (sparse) |
| Install | `pip install fastembed tantivy` then small driver script |
| Reference driver | `benchmarks/v0_3/baseline_hybrid.py` — TBD by human at sweep time |
| Verification | not verified |
| Fusion | Reciprocal Rank Fusion (k=60) — same fusion as semantex v0.3 |
| Notes | The "generic strong hybrid" baseline. If semantex v0.3 doesn't beat this, the engine improvements aren't paying off. |

---

## Reproducibility checklist (per spec §4.5)

- [ ] Every row above has a `verified-on: <date>` line filled in by the human at sweep time.
- [ ] Each tool's invocation command is exercised on a sample query before the full sweep starts.
- [ ] Versions captured in `benchmarks/results/v0.3/COMPETITOR_VERSIONS.txt` at sweep time (one line per tool: `tool=version`).
- [ ] Tools excluded for verification failure are listed under "Excluded" in `docs/BENCHMARK-v0.3.md` with the reason.

## How `benchmarks/run_public.py` uses this file

The reproducer reads tool names from this document via simple section parsing
and validates each is on `$PATH` (or, for API tools, that the relevant env
var is set). Tools that fail the precondition check are skipped with a
**stderr message, not a hard failure**, so a partial sweep can still produce
results. The summary table marks skipped tools `n/a`.

If you add a new tool here, the section header must follow `## <N>. <toolname>`
so the parser picks it up. See `benchmarks/v0_3/competitor_registry.py`.
