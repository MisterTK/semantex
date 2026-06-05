# Full-body trust-bundle A/B — NEGATIVE / lever falsified (2026-06-04)

**Question:** does returning the COMPLETE code body inline (`SEMANTEX_MCP_FULL_CODE=1`) + an
honest completeness marker make the agent stop RE-READING files semantex already cited?
(Substitution probe baseline: 37/37 post-sx reads were DUPLICATES of returned files.)

**Design:** gin, STEERED (constant SEMANTEX_MD nudge → reliable adoption), 2 arms differing only
in full_code. `sx-lateon` (off = 200-char previews) vs `sx-full-code` (on = full bodies + marker).
5 questions × 3 reps × 2 arms = 30 runs. Harness: `benchmarks/probe_fullbody_ab.py`.

## Headline numbers

| metric | off (previews) | on (full body) | delta |
|---|---|---|---|
| adoption | 15/15 | 15/15 | — |
| sx calls/run | 5.2 | 4.73 | — |
| **DUPLICATE re-reads/run** | **1.2** | **2.33** | **+94%** |
| NEW (recall) reads total | 1 | 5 | — |
| **CCB mean** | **603,466** | **614,307** | **+2%** |
| turns mean | 9.53 | 9.53 | **0%** |
| native calls/run | 3.33 | 3.8 | — |

## Why the A/B is INVALID as a test of the change — and the real findings

1. **The change was never exercised.** Across all 30 runs the agent's `semantex_agent` calls routed
   to **deep (177×) + structural (5×); item-list routes 0×**. The `full_code` change only affects the
   item-list routes (Semantic/Analytical/Exhaustive/ExactSymbol). Realistic agent questions classify to
   **deep**, so full_code never fired — 0/30 transcripts contain the `[COMPLETE: full bodies …]` marker.
   The +94% dup delta is noise between two arms running identical deep behavior.

2. **Full bodies do NOT stop re-reads.** The deep route ALREADY returns complete fenced bodies (both
   arms show full `func` implementations) — yet the agent still re-read 1–2 cited sources/run in BOTH
   arms. Handing the agent the body does not stop the re-read (the adversarial critic's "maybe → no";
   a Sonnet-class agent re-reads for verification/full-file-context regardless of body completeness).

3. **DECISIVE — re-reads are nearly free.** ON had ~2× the duplicate reads of OFF (35 vs 18), yet CCB
   moved +2% and turns were identical (9.53 = 9.53). Even a perfect "eliminate all re-reads" fix would
   save ~0–2% CCB and zero turns. The substitution behavior the probe found is real but **not a
   meaningful cost** → the trust/full-body lever has a near-zero CCB ceiling.

## Conclusion

The full-body trust-bundle is correct + reviewed + tested (branch `feat/trust-complete-body`,
`7e052c7`+`c5bf438`; CLI `--full` verified to serve bodies + an honest marker), but as an agent lever
it FAILS three ways: it targets routes the agent doesn't use, full bodies don't suppress re-reads where
they ARE served (deep), and re-reads cost ~nothing anyway. **Do NOT flip the `SEMANTEX_MCP_FULL_CODE`
default. Do NOT merge as a win.** The capability (full_code on item routes + honest marker) is harmless
opt-in if kept, but it does not move agent CCB/turns.

**Caveat:** STEERED inflates sx-call counts (~5/run vs ~1/run bare). The re-read findings are robust to
this; the "real CCB drivers" reading (deep-response size, sx over-calling) is steered-regime and not a
shipped-behavior claim.

## What the data redirects toward (not measured here)
- CCB is dominated by deep-synthesis response size + the number of sx calls, NOT re-reads.
- The substitution/trust lever (output changes to stop re-reads) is a dead end for CCB. The probe's
  "100% duplicate" was a true-but-immaterial finding.
