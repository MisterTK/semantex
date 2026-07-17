# next-plaid (lateon-colbert) vs coderank-hnsw (coderank-137m) — quality head-to-head

## Track A — academic (CoIR + CSN)

| embedder | dataset | n_queries | mrr@10 | ndcg@10 | recall@1 | recall@5 | recall@10 | map | cold ms | warm ms |
|---|---|---|---|---|---|---|---|---|---|---|
| lateon-colbert | coir/codetrans-dl | 180 | 0.1633 | 0.2969 | 0.0278 | 0.2722 | 0.7556 | 0.1837 | 705.2 | 88.0 |
| coderank-137m | coir/codetrans-dl | 180 | 0.1403 | 0.2763 | 0.0056 | 0.2667 | 0.7444 | 0.1615 | 1466.4 | 930.2 |
| lateon-colbert | csn/python | 200 | 0.8725 | 0.8970 | 0.8050 | 0.9550 | 0.9700 | 0.8725 | 661.9 | 36.5 |
| lateon-colbert | csn/javascript | 200 | 0.4771 | 0.5566 | 0.3350 | 0.6750 | 0.8100 | 0.4771 | 646.5 | 26.2 |
| lateon-colbert | csn/go | 200 | 0.7000 | 0.7593 | 0.5800 | 0.8700 | 0.9450 | 0.7000 | 659.2 | 24.7 |
| coderank-137m | csn/python | 200 | 0.8815 | 0.9026 | 0.8200 | 0.9550 | 0.9650 | 0.8815 | 595.7 | 119.4 |
| coderank-137m | csn/javascript | 200 | 0.4876 | 0.5633 | 0.3500 | 0.6700 | 0.8050 | 0.4876 | 543.0 | 64.7 |
| coderank-137m | csn/go | 200 | 0.7010 | 0.7604 | 0.5750 | 0.8850 | 0.9450 | 0.7010 | 637.8 | 72.1 |

## Track B — real-world (claude_bench.py, 1-5 Claude-judged quality)

| repo | question_type | arm | n | mean quality | stdev |
|---|---|---|---|---|---|
| CopilotKit | architecture | sx-coderank | 3 | 4.67 | 0.58 |
| CopilotKit | architecture | sx-lateon | 3 | 4.33 | 0.58 |
| CopilotKit | deep_technical | sx-coderank | 2 | 5 | 0.0 |
| CopilotKit | deep_technical | sx-lateon | 3 | 5 | 0.0 |
| CopilotKit | error_handling | sx-coderank | 1 | 5 | 0.0 |
| CopilotKit | error_handling | sx-lateon | 3 | 5 | 0.0 |
| CopilotKit | exhaustive | sx-coderank | 2 | 4.5 | 0.71 |
| CopilotKit | exhaustive | sx-lateon | 2 | 4.5 | 0.71 |
| CopilotKit | feature_planning | sx-lateon | 3 | 4.33 | 0.58 |
| flask | architecture | sx-coderank | 3 | 4.67 | 0.58 |
| flask | architecture | sx-lateon | 3 | 4.33 | 1.15 |
| flask | deep_technical | sx-coderank | 3 | 4.67 | 0.58 |
| flask | deep_technical | sx-lateon | 3 | 3.67 | 1.53 |
| flask | error_handling | sx-coderank | 3 | 5 | 0.0 |
| flask | error_handling | sx-lateon | 3 | 5 | 0.0 |
| flask | exhaustive | sx-coderank | 3 | 5 | 0.0 |
| flask | exhaustive | sx-lateon | 3 | 5 | 0.0 |
| flask | feature_planning | sx-coderank | 3 | 4 | 1.0 |
| flask | feature_planning | sx-lateon | 3 | 4.33 | 0.58 |
| gin | architecture | sx-coderank | 3 | 5 | 0.0 |
| gin | architecture | sx-lateon | 3 | 5 | 0.0 |
| gin | deep_technical | sx-coderank | 3 | 5 | 0.0 |
| gin | deep_technical | sx-lateon | 3 | 5 | 0.0 |
| gin | error_handling | sx-coderank | 3 | 5 | 0.0 |
| gin | error_handling | sx-lateon | 3 | 5 | 0.0 |
| gin | exhaustive | sx-coderank | 3 | 5 | 0.0 |
| gin | exhaustive | sx-lateon | 3 | 5 | 0.0 |
| gin | feature_planning | sx-coderank | 3 | 4.33 | 0.58 |
| gin | feature_planning | sx-lateon | 3 | 4.33 | 0.58 |
| platform | architecture | sx-coderank | 3 | 5 | 0.0 |
| platform | architecture | sx-lateon | 3 | 5 | 0.0 |
| platform | deep_technical | sx-coderank | 3 | 5 | 0.0 |
| platform | deep_technical | sx-lateon | 3 | 5 | 0.0 |
| platform | error_handling | sx-coderank | 3 | 5 | 0.0 |
| platform | error_handling | sx-lateon | 3 | 5 | 0.0 |
| platform | exhaustive | sx-coderank | 3 | 5 | 0.0 |
| platform | exhaustive | sx-lateon | 3 | 5 | 0.0 |
| platform | feature_planning | sx-coderank | 3 | 4.67 | 0.58 |
| platform | feature_planning | sx-lateon | 3 | 4 | 0.0 |
| pub | architecture | sx-coderank | 3 | 5 | 0.0 |
| pub | architecture | sx-lateon | 3 | 5 | 0.0 |
| pub | deep_technical | sx-coderank | 3 | 5 | 0.0 |
| pub | deep_technical | sx-lateon | 3 | 5 | 0.0 |
| pub | error_handling | sx-coderank | 3 | 5 | 0.0 |
| pub | error_handling | sx-lateon | 3 | 5 | 0.0 |
| pub | exhaustive | sx-coderank | 3 | 5 | 0.0 |
| pub | exhaustive | sx-lateon | 3 | 5 | 0.0 |
| pub | feature_planning | sx-coderank | 3 | 4.67 | 0.58 |
| pub | feature_planning | sx-lateon | 3 | 4.33 | 0.58 |
| semantex | architecture | sx-coderank | 3 | 5 | 0.0 |
| semantex | architecture | sx-lateon | 3 | 5 | 0.0 |
| semantex | deep_technical | sx-coderank | 3 | 5 | 0.0 |
| semantex | deep_technical | sx-lateon | 3 | 4.67 | 0.58 |
| semantex | error_handling | sx-coderank | 3 | 5 | 0.0 |
| semantex | error_handling | sx-lateon | 3 | 5 | 0.0 |
| semantex | exhaustive | sx-coderank | 3 | 5 | 0.0 |
| semantex | exhaustive | sx-lateon | 3 | 5 | 0.0 |
| semantex | feature_planning | sx-coderank | 3 | 4 | 0.0 |
| semantex | feature_planning | sx-lateon | 3 | 4.33 | 0.58 |

## Ambiguous cells (mean ± stdev bands overlap between arms — 3 reps can't call a winner here)

- CopilotKit/architecture: ambiguous (sx-lateon 4.33±0.58 vs sx-coderank 4.67±0.58, n=3)
- CopilotKit/deep_technical: ambiguous (sx-lateon 5±0.0 vs sx-coderank 5±0.0, n=3)
- CopilotKit/error_handling: ambiguous (sx-lateon 5±0.0 vs sx-coderank 5±0.0, n=3)
- CopilotKit/exhaustive: ambiguous (sx-lateon 4.5±0.71 vs sx-coderank 4.5±0.71, n=2)
- flask/architecture: ambiguous (sx-lateon 4.33±1.15 vs sx-coderank 4.67±0.58, n=3)
- flask/deep_technical: ambiguous (sx-lateon 3.67±1.53 vs sx-coderank 4.67±0.58, n=3)
- flask/error_handling: ambiguous (sx-lateon 5±0.0 vs sx-coderank 5±0.0, n=3)
- flask/exhaustive: ambiguous (sx-lateon 5±0.0 vs sx-coderank 5±0.0, n=3)
- flask/feature_planning: ambiguous (sx-lateon 4.33±0.58 vs sx-coderank 4±1.0, n=3)
- gin/architecture: ambiguous (sx-lateon 5±0.0 vs sx-coderank 5±0.0, n=3)
- gin/deep_technical: ambiguous (sx-lateon 5±0.0 vs sx-coderank 5±0.0, n=3)
- gin/error_handling: ambiguous (sx-lateon 5±0.0 vs sx-coderank 5±0.0, n=3)
- gin/exhaustive: ambiguous (sx-lateon 5±0.0 vs sx-coderank 5±0.0, n=3)
- gin/feature_planning: ambiguous (sx-lateon 4.33±0.58 vs sx-coderank 4.33±0.58, n=3)
- platform/architecture: ambiguous (sx-lateon 5±0.0 vs sx-coderank 5±0.0, n=3)
- platform/deep_technical: ambiguous (sx-lateon 5±0.0 vs sx-coderank 5±0.0, n=3)
- platform/error_handling: ambiguous (sx-lateon 5±0.0 vs sx-coderank 5±0.0, n=3)
- platform/exhaustive: ambiguous (sx-lateon 5±0.0 vs sx-coderank 5±0.0, n=3)
- pub/architecture: ambiguous (sx-lateon 5±0.0 vs sx-coderank 5±0.0, n=3)
- pub/deep_technical: ambiguous (sx-lateon 5±0.0 vs sx-coderank 5±0.0, n=3)
- pub/error_handling: ambiguous (sx-lateon 5±0.0 vs sx-coderank 5±0.0, n=3)
- pub/exhaustive: ambiguous (sx-lateon 5±0.0 vs sx-coderank 5±0.0, n=3)
- pub/feature_planning: ambiguous (sx-lateon 4.33±0.58 vs sx-coderank 4.67±0.58, n=3)
- semantex/architecture: ambiguous (sx-lateon 5±0.0 vs sx-coderank 5±0.0, n=3)
- semantex/deep_technical: ambiguous (sx-lateon 4.67±0.58 vs sx-coderank 5±0.0, n=3)
- semantex/error_handling: ambiguous (sx-lateon 5±0.0 vs sx-coderank 5±0.0, n=3)
- semantex/exhaustive: ambiguous (sx-lateon 5±0.0 vs sx-coderank 5±0.0, n=3)
- semantex/feature_planning: ambiguous (sx-lateon 4.33±0.58 vs sx-coderank 4±0.0, n=3)

## Side-by-Side Comparison

### Track A — academic benchmarks

| Dataset | Metric | lateon-colbert | coderank-137m | Delta |
|---|---|---:|---:|---:|
| **CoIR** codetrans-dl (n=180) | ndcg@10 | 0.2969 | 0.2763 | +0.0206 (+7.5%) → lateon |
| | mrr@10 | 0.1633 | 0.1403 | +0.0230 → lateon |
| | recall@1 | 0.0278 | 0.0056 | +0.0222 → lateon |
| | recall@5 | 0.2722 | 0.2667 | +0.0056 → lateon |
| | recall@10 | 0.7556 | 0.7444 | +0.0111 → lateon |
| | map | 0.1837 | 0.1615 | +0.0222 → lateon |
| | cold latency | 705.2 ms | 1466.4 ms | lateon 2.1x faster |
| | warm latency | 88.0 ms | 930.2 ms | lateon 10.6x faster |
| **CSN** python (n=200) | ndcg@10 | 0.8970 | 0.9026 | -0.0056 → coderank |
| | warm latency | 36.5 ms | 119.4 ms | lateon 3.3x faster |
| **CSN** javascript (n=200) | ndcg@10 | 0.5566 | 0.5633 | -0.0068 → coderank |
| | warm latency | 26.2 ms | 64.7 ms | lateon 2.5x faster |
| **CSN** go (n=200) | ndcg@10 | 0.7593 | 0.7604 | -0.0011 → coderank |
| | warm latency | 24.7 ms | 72.1 ms | lateon 2.9x faster |

CoIR favors lateon-colbert by a real margin; CSN (larger, 600 queries) is a near-wash tilted marginally to coderank-137m. Warm-query latency favors lateon-colbert consistently, 2.5-10.6x, across every dataset.

### Track B — real-world agent quality (172 judged cells, 1-5 scale)

| | sx-lateon | sx-coderank |
|---|---:|---:|
| Cells run | 90 | 90 |
| Cells timed out (excluded) | 1 | 7 |
| Cells scored | 89 | 83 |
| Non-ambiguous point-estimate wins (not statistically significant) | 2 | 6 |
| Tied / ambiguous cells | 21 | 21 |
| **Statistically confident wins (3-rep mean±stdev, non-overlapping bands)** | **0** | **0** |

No cell in the 29-cell (repo x question-type) grid cleared the ambiguity bar — 3 reps isn't enough to call a winner anywhere. The only real asymmetry is operational, not quality: sx-coderank timed out 7x more often than sx-lateon, entirely on CopilotKit (159k chunks), losing all 3 reps of one cell outright (CopilotKit/feature_planning).

### Bottom line

| | Track A | Track B |
|---|---|---|
| Verdict | Split by dataset (CoIR → lateon, CSN → coderank, wash) | Null result — no statistically confident winner |
| Latency signal | lateon-colbert 2.5-10.6x faster warm | 7:1 timeout skew toward coderank on the largest repo (echoes Track A) |

## Recommendation

Track A splits by dataset rather than pointing one direction: CoIR codetrans-dl (n=180) shows a real quality edge for lateon-colbert on ndcg@10 (0.2969 vs 0.2763, a ~7.5% relative delta — well outside the few-thousandths noise band that characterized the original D4 finding), but CSN (n=200 per language, 600 queries total across python/javascript/go — the larger and more representative-of-day-to-day-code-search Track A dataset) is a near-wash tilted marginally toward coderank-137m on every language (ndcg@10 deltas of 0.0056/0.0067/0.0011, all consistent with "within noise"). Track A also surfaces a consistent warm-query-latency advantage for lateon-colbert across all four datasets (e.g. 88ms vs 930ms warm on coir; 24–37ms vs 65–119ms warm on CSN, a 2–10x gap), though cold latency is mixed (lateon much faster cold on coir, coderank marginally faster cold on CSN). Track B cannot adjudicate this split: zero of the 29 comparable (repo, question_type) cells clear the ambiguous-cell filter — every mean±stdev band overlaps between arms, so the 3-rep (2-rep for one CopilotKit cell) design has no statistical power to call a winner anywhere in the real-world agent-quality data, and raw point estimates (not a valid signal on their own, since none cross the ambiguity threshold) lean toward coderank-137m in more non-tied cells (6) than lateon-colbert (2), with the remaining 21 tied at or near the 5/5 quality ceiling. One cell, CopilotKit/feature_planning, has no sx-coderank data at all — all 3 reps timed out, part of a broader pattern where 7 of the run's 8 timeouts (600s each) landed on sx-coderank vs only 1 on sx-lateon, concentrated entirely on CopilotKit, the largest indexed repo (~159k chunks); this is a real qualitative observation worth flagging, not a rigorous latency claim (Track A's cold/warm ms figures remain the rigorous measurement), but it echoes Track A's warm-latency edge for lateon-colbert and suggests coderank-hnsw may be more timeout-prone at large-repo scale under a fixed budget. Net: this data does **not** justify flipping the D4 default away from `coderank-hnsw` — CSN, the larger and more representative Track A benchmark, is a wash tilted its way, and Track B produced no cell where lateon-colbert wins with any statistical confidence — so we recommend **confirming** the current default. That said, lateon-colbert's real CoIR-dataset quality edge, its consistent warm-latency advantage, and its lower real-world timeout rate on the largest repo are legitimate reasons to revisit this call specifically for very-large-repo or latency-sensitive deployments, where the sub-1% CSN quality difference may matter less than serving latency under load.
