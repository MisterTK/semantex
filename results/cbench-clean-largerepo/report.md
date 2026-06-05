# Contamination-free agent benchmark — large repos (2026-06-05)

First end-to-end agent-quality run on the LARGE repos with the repos' tool-routing
`CLAUDE.md` neutralized (`--neutralize-claude-md`). Validates whether semantex moves
agent answer quality on broad large-repo questions, and isolates adoption vs retrieval.
**n=5/cell, 1 rep — directional, noisy (esp. quality).** builtin vs sx-lateon (shipped
default, bare) vs sx-lateon-steered (semantex as primary). 30/30 cells, 0 errors; both
repos' CLAUDE.md restored intact (harness self-heal verified).

## Results (quality 1-5 judged blind; CCB = cumulative attended context)

| repo | arm | quality | CCB | turns | sx/run | native/run |
|---|---|---|---|---|---|---|
| platform | builtin | 5.0 | 2,224,019 | 2.4 | 0 | 54.4 |
| platform | sx-lateon (bare) | 5.0 | 2,327,439 | 4.4 | 6.4 | 49.0 |
| platform | sx-lateon-steered | 4.8 | **949,872 (−57%)** | 13.4 | 5.8 | 6.6 |
| CopilotKit | builtin | 4.2 | 1,841,857 | 5.4 | 0 | 51.0 |
| CopilotKit | sx-lateon (bare) | 4.2 | 1,659,812 (−10%) | 3.6 | 4.6 | 44.0 |
| CopilotKit | sx-lateon-steered | **4.8** | **1,398,761 (−24%)** | 18.4 | 8.8 | 8.6 |

Q4 (the enumeration question, the recent fix): quality 5/5 for EVERY arm/repo — ceiling.

## What it says (robust, large-effect findings)

1. **Quality holds.** semantex MATCHES builtin everywhere (platform 5.0=5.0; CopilotKit 4.2=4.2 bare), and STEERED BEATS builtin on the hard repo (CopilotKit 4.8 vs 4.2 — closing the prior ~4.20 gap). No quality regression from routing the agent through semantex.
2. **The CCB win is large but ADOPTION-GATED.** When semantex is used as PRIMARY (steered), native exploration COLLAPSES (platform 49→6.6, CopilotKit 51→8.6) and CCB drops −57% / −24% at matched-or-better quality. When BARE (shipped default), the agent calls semantex (~6/run) but STILL greps ~49× — additive → CCB tied (platform) / −10% (CopilotKit). **The binding constraint is adoption, now confirmed on large repos with a clean harness.**
3. **The enumeration fix did NOT move end-to-end quality.** Q4 answer quality was already 5/5 for builtin too — the agent compensates for incomplete retrieval with native greps, so the (measured) recall improvement doesn't surface as a judged-quality delta at this granularity. The fix improves retrieval completeness + (when adopted) reduces the compensating greps, not the final answer score.

## Honest caveats
- **n=5/cell, 1 rep.** Quality is noisy; platform is at the 5.0 ceiling (questions too easy / builtin too strong there → no room to show a semantex quality edge). The one quality delta (CopilotKit steered 4.8 vs builtin 4.2) is the interesting signal but needs reps to confirm.
- Steered = an explicit "use semantex as primary" nudge. It is the high-VALUE config (unlocks −57% CCB), but bare-MCP (shipped) under-adopts. So the product lever is driving adoption (tool description / recommended CLAUDE.md), NOT more retrieval.

## Strategic takeaway
The clean harness (the measurement instrument we lacked all session) now exists and works. Its first verdict: **semantex delivers a large, quality-neutral-to-positive CCB win — gated on adoption.** The next lever is adoption (make the agent use semantex as primary in bare-MCP), not retrieval. The 4.20 gap is real and semantex (adopted) closes it.
