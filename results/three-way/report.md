# Three-way: semantex vs graphify vs serena (2026-06-05)

Low-cost apples-to-apples on **CopilotKit** (large TS monorepo). **Equal adoption**: all
three tools get the IDENTICAL soft PreToolUse nudge hook (each pointing at its own tool),
so this isolates TOOL QUALITY from adoption-mechanism quality. 4 arms × 3 broad questions
(Q1 architecture, Q3 deep-technical, Q5 feature-planning) × 1 rep, blind-judged quality.
~$6 total. Harness: `benchmarks/three_way.py` + `three_way_nudge.py`.

## Result — semantex clean sweep

| arm | quality (1-5) | CCB | native calls/run | $ |
|---|---|---|---|---|
| **semantex** | **5.0** | **1.18M** | **11.7** | **$0.32** |
| graphify | 4.67 | 1.63M | 36.7 | $0.36 |
| serena | 4.33 | 2.94M | 73.7 | $0.54 |
| builtin | 4.0 | 2.15M | 62.7 | $0.47 |

**semantex won every axis** — best quality, lowest context cost (−45% vs builtin, 2.5× lighter
than serena), cheapest.

## Why
- **The broad architecture question (Q1) is the differentiator.** builtin scored **2/5** (native
  grep/read can't assemble a large monorepo's big picture); semantex + graphify nailed it 5/5.
  This is the "one call, complete answer" value on broad large-repo questions.
- **semantex actually DISPLACES native exploration** (11.7 native calls/run vs builtin's 63) →
  CCB collapses. graphify partially displaces (37); **serena doesn't at all** — it greps 74×/run
  ON TOP OF ~63 symbol calls (additive, worst of both), so its fine-grained LSP approach pushes
  CCB *above* builtin.
- **serena is weakest here, with an asterisk**: these were broad SYNTHESIS questions, which favor
  retrieval-synthesis (semantex/graphify) over symbol-navigation. Serena's real strength —
  precise "where is X / who calls Y" — was NOT tested.

## Caveats
- n=3, 1 rep → **CCB ordering is robust** (2.5× gaps, not noise); **quality scores are close**
  (5.0 / 4.67 / 4.33) — semantex's perfect score is clear, graphify-vs-serena is within noise.
- One repo, broad-synthesis question mix. To firm up: add reps + a 2nd large repo + symbol-precise
  questions (so serena gets a fair shot at its strength).

**Bottom line:** at equal adoption, on broad large-repo questions, semantex stacks up very well —
a clean sweep on quality AND efficiency, with the win concentrated exactly where the session's
thesis predicted (broad questions native tools can't assemble cheaply).
