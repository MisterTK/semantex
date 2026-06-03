# Route-stress oracle-regret evaluation

- git rev: `69d6457`
- generated: 2026-06-03 11:52:04
- config: `SEMANTEX_ADAPTIVE_SIZING=0` (canonical A/B lock)
- retrieval routes scored: file_pattern, regex, exact_symbol, structural, semantic
- synthesis routes (scored 0 for file-gold): analytical, architecture, deep, exhaustive, feature_planning

## Summary (per repo)

| repo | queries | overall regret | router acc | synthesis picks |
| --- | --- | --- | --- | --- |
| platform | 22 | 0.1964 | 31.8% | 6 |

## Pooled per-route oracle-win counts (all repos)
(total queries: 22)

| route | oracle-wins |
| --- | --- |
| file_pattern | 0 |
| regex | 11 |
| exact_symbol | 9 |
| structural | 8 |
| semantic | 9 |

## platform

- total queries: **22**
- overall regret (oracle nDCG@10 − router-picked nDCG@10): **0.1964**
- router accuracy (router choice == an oracle-best route): **31.8%** (7/22)
- router chose a SYNTHESIS route (no file hits, scored 0): **6**

### Per-route oracle-win counts
(a route 'wins' a query when it is — or ties — the best-nDCG route; ~0 wins ⇒ deletion candidate)

| route | oracle-wins | mean nDCG@10 |
| --- | --- | --- |
| file_pattern | 0 | 0.0000 |
| regex | 11 | 0.5177 |
| exact_symbol | 9 | 0.4230 |
| structural | 8 | 0.3456 |
| semantic | 9 | 0.4563 |

### Per-mechanism × per-route mean nDCG@10
(does each route actually win on its intended mechanism? does one route dominate everything?)

| mechanism | file_pattern | regex | exact_symbol | structural | semantic |
| --- | --- | --- | --- | --- | --- |
| glob | 0.000 | **0.415** | 0.104 | 0.151 | 0.180 |
| lexical | 0.000 | 0.926 | 0.926 | **1.000** | 0.926 |
| regex | 0.000 | **0.448** | 0.249 | 0.217 | 0.357 |
| semantic | 0.000 | 0.477 | **0.544** | 0.377 | 0.544 |
| structural | 0.000 | **0.625** | 0.544 | 0.000 | 0.544 |
| usage | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

### Regret per mechanism

| mechanism | mean regret | router accuracy |
| --- | --- | --- |
| glob | 0.3173 | 0.0% |
| lexical | 0.0738 | 80.0% |
| regex | 0.1099 | 25.0% |
| semantic | 0.5436 | 0.0% |
| structural | 0.2044 | 66.7% |
| usage | 0.0000 | 0.0% |

### Confusion matrix (rows = intended_mechanism, cols = router choice)

| mechanism ↓ / choice → | semantic | deep | feature_planning |
| --- | --- | --- | --- |
| glob | 4 | 0 | 0 |
| lexical | 5 | 0 | 0 |
| regex | 4 | 0 | 0 |
| semantic | 0 | 3 | 0 |
| structural | 3 | 0 | 0 |
| usage | 0 | 2 | 1 |

