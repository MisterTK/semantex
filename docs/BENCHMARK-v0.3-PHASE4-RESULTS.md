# v0.3 Phase 4 — Final Agent CCB Results

**Date:** 2026-05-26
**Status:** Recovery validated; v0.3 ships with the Phase 4 route refactor.
**Total measurement cost across phases:** $108 (run23) + $26 (run24) + $77 (run25) = **$211 USD**

## Headline

| Metric | v0.2 (run22) | v0.3 visible (run23) | **v0.3 Phase 4 (run25)** | Recovery vs run23 |
|---|---|---|---|---|
| num_turns Δ | −50% | +16% | **−14%** | **+30pp** |
| tool_calls Δ | −60% | +17% | **−19%** | **+36pp** |
| **CCB Δ** | **−56%** | **+20%** | **−18%** | **+38pp** |
| peak_context Δ | −26% | +2% | **−6%** | +8pp |
| duration Δ | −38% | −4% | **−15%** | +11pp |
| cost_usd Δ | −11% | +7% | **−13%** | **+20pp** |
| CAF Δ | −21% | +1% | **−6%** | +7pp |

The B3 spec §4.7 cap is no longer tripped catastrophically. v0.3 Phase 4 shows a real and broad CCB-reduction benefit (−18%) vs the visible-tools regression (+20%).

## Per-question-type recovery (mean Δ across 6 repos)

| Q | Type | v0.2 | v0.3 vis | **v0.3 Phase 4** | Phase 4 recovery |
|---|---|---|---|---|---|
| Q1 | architecture | −41% | +16% | **−3%** | +19pp |
| Q2 | error_handling | −78% | +2% | **−16%** | +18pp |
| Q3 | deep_technical | −12% | +25% | **−14%** | **+39pp** |
| Q4 | exhaustive | −37% | +55% | **−9%** | **+64pp** ⭐ |
| Q5 | feature_planning | −51% | +29% | **−15%** | +44pp |

## Per-repo recovery

| Repo | v0.2 | v0.3 visible | **v0.3 Phase 4** | Recovery |
|---|---|---|---|---|
| gin | −76% | +13% | **−34%** | +47pp |
| flask | −71% | +12% | **−11%** | +23pp |
| pub | −68% | +37% | **−29%** | +66pp |
| qgrep | −45% | −20% | **−6%** | −14pp (was already winning) |
| platform | −25% | +24% | **+7%** | +17pp (still slight loss) |
| CopilotKit | −59% | +43% | **−30%** | **+73pp** (largest swing) |

**23 of 30 (repo, question) cells are net positive.**

## Top-10 Phase 4 wins

| Repo | Q | Type | BL turns | TX turns | Δ CCB |
|---|---|---|---|---|---|
| gin | Q5 | feature_planning | 32 | 15 | **−67%** |
| gin | Q2 | error_handling | 66 | 26 | **−65%** |
| pub | Q1 | architecture | 43 | 27 | **−53%** |
| CopilotKit | Q2 | error_handling | 71 | 44 | **−51%** |
| pub | Q5 | feature_planning | 65 | 39 | **−49%** |
| gin | Q3 | deep_technical | 17 | 9 | **−47%** |
| CopilotKit | Q1 | architecture | 68 | 48 | −36% |
| platform | Q3 | deep_technical | 37 | 25 | −35% |
| flask | Q2 | error_handling | 57 | 28 | −34% |
| pub | Q3 | deep_technical | 38 | 27 | −32% |

## 7 remaining regressions (v0.3.1 punch list)

| Repo | Q | BL turns | TX turns | Δ CCB | Hypothesis |
|---|---|---|---|---|---|
| gin | Q1 | 18 | 47 | **+122%** | Architecture route fires but agent does follow-up exploration. gin is tiny — arch overview was less useful than direct grep. |
| platform | Q2 | 46 | 67 | +69% | Structural classifier may match too aggressively on multi-language repo. |
| flask | Q3 | 35 | 45 | +54% | DeepWithExamples may return too many examples for flask's narrow scope. |
| CopilotKit | Q5 | 34 | 48 | +34% | Feature_planning on largest repo — handler doesn't bound result count. |
| platform | Q4 | 61 | 71 | +29% | ExhaustiveStructural's max_results=30 too high on 7800-chunk repo. |
| qgrep | Q5 | 42 | 42 | +24% | Equal turn count, +24% tokens — output verbosity not turn count. |
| CopilotKit | Q3 | 32 | 34 | +8% | Marginal regression; within run-to-run noise. |

## Why we don't match v0.2's −56% (baseline drift)

Sum across 6 repos × 5 questions:

| | v0.2 (run22) | v0.3-Phase4 (run25) |
|---|---|---|
| baseline turns | 1,032 | 1,368 (+33%) |
| baseline tokens | 34.7M | **60.8M (+75%)** |
| baseline cost | $11.0 | $13.4 (+22%) |

Claude Code v2.1.150 + Sonnet 4.6 with extended thinking does dramatically more baseline work for the same questions than the older Sonnet in run22. Treatment numbers grew proportionally:

| | v0.2 TX | Phase 4 TX |
|---|---|---|
| treatment turns | 518 | 1,181 (+128%) |
| treatment tokens | 15.2M | 50.0M (+229%) |
| treatment cost | $9.7 | $11.7 (+20%) |

Phase 4 saves ~10M tokens per benchmark run vs v0.2's ~19M; proportional savings smaller because per-turn cost is smaller on the new model. The spec's −70% target is unreachable without changing Claude Code's baseline behavior itself.

## Ship recommendation

**Ship v0.3.0 with the Phase 4 internal-routing architecture.** Queue the 7 (repo, question) regressions for v0.3.1 as small targeted handler-tuning fixes (smaller arch overview budget on small repos; cap result count in handle_exhaustive_structural; trim pattern hit count in handle_deep_with_examples).
