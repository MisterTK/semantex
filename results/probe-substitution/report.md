# Substitution-vs-recall probe — gin / LateOn-colbert (2026-06-04)

model: claude-sonnet-4-6 · repo: gin · backend: colbert-plaid (LateOn-Code-edge) · SEMANTEX_ADAPTIVE_SIZING=0

## The question

A coding agent calls `semantex_agent` ~1/task and THEN still greps/reads natively. WHY?
- **DUPLICATE** (re-reads files semantex ALREADY returned) → TRUST/SUBSTITUTION problem → fix = tool-output/adoption work.
- **NEW** (reads files semantex did NOT return) → RETRIEVAL RECALL problem → fix = the EMBEDDER.

The duplicate-vs-new ratio of post-semantex native file-accesses is the answer.

## VERDICT

**MOSTLY-DUPLICATE (100%)** → the agent re-reads what semantex already returned. This is a TRUST/SUBSTITUTION problem; the lever is tool-output / adoption work (make the agent trust + stop re-verifying), NOT the embedder.

Classification n (combined bare + steered): **16 runs called semantex ≥1×**, yielding **37 post-sx native file-accesses** (37 DUP / 0 NEW; HIGH-conf subset 36).


## BIG CAVEAT — adoption gap dominates the bare arm

In the **bare-MCP arm (no CLAUDE.md, the shipped regime)**, only **1/15 runs called semantex at all** — 14/15 ignored the MCP tool and went straight to native grep/Read. So BEFORE the duplicate-vs-new question even applies, the agent mostly doesn't adopt the tool. The **steered arm** (SEMANTEX_MD nudge = 'use semantex as PRIMARY') is added purely to manufacture enough sx-using runs to answer the duplicate-vs-new question; its substitution ratio is the load-bearing number, the bare arm's near-zero adoption is the louder finding.

> Methodological note: `claude -p` (non-interactive) DENIES MCP tools unless `--allowedTools mcp__semantex__*` is passed. The original `claude_bench.py` does NOT pass it, so its bare-MCP semantex calls were being permission-denied (verified). This probe grants the tool explicitly; the harness gap is flagged for follow-up.


## Arm A — bare-MCP (no nudge; the shipped regime)

- Runs captured: **15**; called semantex ≥1×: **1**; called semantex 0×: **14**
- Post-semantex native file-accesses classified (in ≥1-sx runs): **4** (mean 4.0/run)
- **DUPLICATE vs NEW (all conf): 4/4 = 100% DUP · 0/4 = 0% NEW**
- DUPLICATE vs NEW (HIGH-conf only — Read.file_path/Grep.path): 3/3 = 100% DUP · 0/3 = 0% NEW

| Q | type | runs w/ sx | runs 0-sx | post-sx | DUP | NEW | %DUP |
|---|---|---|---|---|---|---|---|
| Q1 | architecture | 0 | 3 | 0 | 0 | 0 | 0% |
| Q2 | error_handling | 0 | 3 | 0 | 0 | 0 | 0% |
| Q3 | deep_technical | 1 | 2 | 4 | 4 | 0 | 100% |
| Q4 | exhaustive | 0 | 3 | 0 | 0 | 0 | 0% |
| Q5 | feature_planning | 0 | 3 | 0 | 0 | 0 | 0% |

## Arm B — steered (SEMANTEX_MD nudge; manufactures sx usage)

- Runs captured: **15**; called semantex ≥1×: **15**; called semantex 0×: **0**
- Post-semantex native file-accesses classified (in ≥1-sx runs): **33** (mean 2.2/run)
- **DUPLICATE vs NEW (all conf): 33/33 = 100% DUP · 0/33 = 0% NEW**
- DUPLICATE vs NEW (HIGH-conf only — Read.file_path/Grep.path): 33/33 = 100% DUP · 0/33 = 0% NEW

| Q | type | runs w/ sx | runs 0-sx | post-sx | DUP | NEW | %DUP |
|---|---|---|---|---|---|---|---|
| Q1 | architecture | 3 | 0 | 0 | 0 | 0 | 0% |
| Q2 | error_handling | 3 | 0 | 5 | 5 | 0 | 100% |
| Q3 | deep_technical | 3 | 0 | 7 | 7 | 0 | 100% |
| Q4 | exhaustive | 3 | 0 | 19 | 19 | 0 | 100% |
| Q5 | feature_planning | 3 | 0 | 2 | 2 | 0 | 100% |

## Combined (both arms — the substitution-ratio sample)

- Runs captured: **30**; called semantex ≥1×: **16**; called semantex 0×: **14**
- Post-semantex native file-accesses classified (in ≥1-sx runs): **37** (mean 2.3/run)
- **DUPLICATE vs NEW (all conf): 37/37 = 100% DUP · 0/37 = 0% NEW**
- DUPLICATE vs NEW (HIGH-conf only — Read.file_path/Grep.path): 36/36 = 100% DUP · 0/36 = 0% NEW

| Q | type | runs w/ sx | runs 0-sx | post-sx | DUP | NEW | %DUP |
|---|---|---|---|---|---|---|---|
| Q1 | architecture | 3 | 3 | 0 | 0 | 0 | 0% |
| Q2 | error_handling | 3 | 3 | 5 | 5 | 0 | 100% |
| Q3 | deep_technical | 4 | 2 | 11 | 11 | 0 | 100% |
| Q4 | exhaustive | 3 | 3 | 19 | 19 | 0 | 100% |
| Q5 | feature_planning | 3 | 3 | 2 | 2 | 0 | 100% |

## Per-run detail (all runs)

| arm | run | sx calls | sx files ret | post-sx | DUP | NEW |
|---|---|---|---|---|---|---|
| bare | gin_Q1_r1 | 0 | 0 | 0 | 0 | 0 |
| bare | gin_Q1_r2 | 0 | 0 | 0 | 0 | 0 |
| bare | gin_Q1_r3 | 0 | 0 | 0 | 0 | 0 |
| bare | gin_Q2_r1 | 0 | 0 | 0 | 0 | 0 |
| bare | gin_Q2_r2 | 0 | 0 | 0 | 0 | 0 |
| bare | gin_Q2_r3 | 0 | 0 | 0 | 0 | 0 |
| bare | gin_Q3_r1 | 0 | 0 | 0 | 0 | 0 |
| bare | gin_Q3_r2 | 2 | 8 | 4 | 4 | 0 |
| bare | gin_Q3_r3 | 0 | 0 | 0 | 0 | 0 |
| bare | gin_Q4_r1 | 0 | 0 | 0 | 0 | 0 |
| bare | gin_Q4_r2 | 0 | 0 | 0 | 0 | 0 |
| bare | gin_Q4_r3 | 0 | 0 | 0 | 0 | 0 |
| bare | gin_Q5_r1 | 0 | 0 | 0 | 0 | 0 |
| bare | gin_Q5_r2 | 0 | 0 | 0 | 0 | 0 |
| bare | gin_Q5_r3 | 0 | 0 | 0 | 0 | 0 |
| steered | gin_Q1_r1 | 9 | 35 | 0 | 0 | 0 |
| steered | gin_Q1_r2 | 6 | 29 | 0 | 0 | 0 |
| steered | gin_Q1_r3 | 6 | 24 | 0 | 0 | 0 |
| steered | gin_Q2_r1 | 2 | 20 | 0 | 0 | 0 |
| steered | gin_Q2_r2 | 5 | 15 | 0 | 0 | 0 |
| steered | gin_Q2_r3 | 2 | 17 | 5 | 5 | 0 |
| steered | gin_Q3_r1 | 3 | 10 | 2 | 2 | 0 |
| steered | gin_Q3_r2 | 5 | 13 | 0 | 0 | 0 |
| steered | gin_Q3_r3 | 4 | 15 | 5 | 5 | 0 |
| steered | gin_Q4_r1 | 8 | 32 | 9 | 9 | 0 |
| steered | gin_Q4_r2 | 6 | 28 | 4 | 4 | 0 |
| steered | gin_Q4_r3 | 6 | 30 | 6 | 6 | 0 |
| steered | gin_Q5_r1 | 6 | 18 | 2 | 2 | 0 |
| steered | gin_Q5_r2 | 5 | 22 | 0 | 0 | 0 |
| steered | gin_Q5_r3 | 3 | 19 | 0 | 0 | 0 |

## Concrete post-semantex access examples

- `[bare] gin_Q3_r2` — Read `tree.go` → **DUPLICATE** (Read.file_path, conf=high); YES — semantex DID return it
- `[bare] gin_Q3_r2` — Bash `tree.go` → **DUPLICATE** (Bash(grep/cat/sed), conf=low); YES — semantex DID return it
- `[bare] gin_Q3_r2` — Read `tree.go` → **DUPLICATE** (Read.file_path, conf=high); YES — semantex DID return it
- `[bare] gin_Q3_r2` — Read `tree.go` → **DUPLICATE** (Read.file_path, conf=high); YES — semantex DID return it
- `[steered] gin_Q2_r3` — Read `errors.go` → **DUPLICATE** (Read.file_path, conf=high); YES — semantex DID return it
- `[steered] gin_Q2_r3` — Read `recovery.go` → **DUPLICATE** (Read.file_path, conf=high); YES — semantex DID return it
- `[steered] gin_Q2_r3` — Read `logger.go` → **DUPLICATE** (Read.file_path, conf=high); YES — semantex DID return it
- `[steered] gin_Q2_r3` — Read `context.go` → **DUPLICATE** (Read.file_path, conf=high); YES — semantex DID return it
- `[steered] gin_Q2_r3` — Read `logger.go` → **DUPLICATE** (Read.file_path, conf=high); YES — semantex DID return it
- `[steered] gin_Q3_r1` — Read `tree.go` → **DUPLICATE** (Read.file_path, conf=high); YES — semantex DID return it
- `[steered] gin_Q3_r1` — Read `tree.go` → **DUPLICATE** (Read.file_path, conf=high); YES — semantex DID return it
- `[steered] gin_Q3_r3` — Grep `tree.go` → **DUPLICATE** (Grep.path(file), conf=high); YES — semantex DID return it
- `[steered] gin_Q3_r3` — Read `tree.go` → **DUPLICATE** (Read.file_path, conf=high); YES — semantex DID return it
- `[steered] gin_Q3_r3` — Read `tree.go` → **DUPLICATE** (Read.file_path, conf=high); YES — semantex DID return it

## Parsing confidence & limitations

- **HIGH confidence (exact)**: `Read.file_path` is an exact path; `Grep` with a `path` arg resolving to a real file is exact. The HIGH-conf ratio is the trustworthy number; Read dominates the sample.

- **LOW confidence (best-effort, fuzzy)**: `Bash` grep/cat/sed/awk targets are extracted heuristically from the command string (last file-looking token, preferring one that exists in-repo); `Glob` patterns naming a literal file. May mis-attribute.

- **sx_returned_files** = the union of every `file:line` / `file:start-end` header AND inline ref in the semantex_agent tool_result (including callers/callees mentions). This is GENEROUS to DUPLICATE (a file merely named counts as 'returned'), so it UNDER-counts NEW — a NEW-heavy result would be conservative/strong, and a DUPLICATE-heavy result is the easier one to produce.

- Directory-/repo-wide Grep and Bash `ls`/`find` are NOT counted (no single target) — exploration, not a specific-file read.

- The steered arm's nudge changes WHETHER the agent calls semantex, not WHAT it reads afterward, so its post-sx accesses are valid for the duplicate-vs-new question.

