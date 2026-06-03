# Route-stress oracle-regret evaluation

- git rev: `90c6601`
- generated: 2026-06-03 18:11:26
- config: `SEMANTEX_ADAPTIVE_SIZING=0` (canonical A/B lock)
- retrieval routes scored: file_pattern, regex, exact_symbol, structural, semantic
- synthesis routes (scored 0 for file-gold): analytical, architecture, deep, exhaustive, feature_planning

## Summary (per repo)

| repo | queries | overall regret | router acc | synthesis picks |
| --- | --- | --- | --- | --- |
| platform | 27 | 0.1360 | 51.9% | 6 |

## Pooled per-route oracle-win counts (all repos)
(total queries: 27)

| route | oracle-wins |
| --- | --- |
| file_pattern | 5 |
| regex | 12 |
| exact_symbol | 9 |
| structural | 9 |
| semantic | 11 |

## platform

- total queries: **27**
- overall regret (oracle nDCG@10 − router-picked nDCG@10): **0.1360**
- router accuracy (router choice == an oracle-best route): **51.9%** (14/27)
- router chose a SYNTHESIS route (no file hits, scored 0): **6**

### Per-route oracle-win counts
(a route 'wins' a query when it is — or ties — the best-nDCG route; ~0 wins ⇒ deletion candidate)

| route | oracle-wins | mean nDCG@10 |
| --- | --- | --- |
| file_pattern | 5 | 0.1852 |
| regex | 12 | 0.4218 |
| exact_symbol | 9 | 0.4096 |
| structural | 9 | 0.3823 |
| semantic | 11 | 0.4654 |

### Per-mechanism × per-route mean nDCG@10
(does each route actually win on its intended mechanism? does one route dominate everything?)

| mechanism | file_pattern | regex | exact_symbol | structural | semantic |
| --- | --- | --- | --- | --- | --- |
| glob | **0.556** | 0.185 | 0.202 | 0.051 | 0.270 |
| lexical | 0.000 | 0.926 | **1.000** | 1.000 | 1.000 |
| regex | 0.000 | 0.448 | 0.403 | 0.399 | **0.529** |
| semantic | 0.000 | **0.477** | 0.333 | 0.477 | 0.462 |
| structural | 0.000 | **0.625** | 0.544 | 0.506 | 0.544 |
| usage | 0.000 | 0.000 | 0.000 | **0.105** | 0.000 |

### Regret per mechanism

| mechanism | mean regret | router accuracy |
| --- | --- | --- |
| glob | 0.1173 | 66.7% |
| lexical | 0.0000 | 100.0% |
| regex | 0.1613 | 25.0% |
| semantic | 0.4769 | 0.0% |
| structural | 0.0754 | 66.7% |
| usage | 0.1052 | 0.0% |

### Confusion matrix (rows = intended_mechanism, cols = router choice)

| mechanism ↓ / choice → | file_pattern | exact_symbol | structural | semantic | deep | feature_planning |
| --- | --- | --- | --- | --- | --- | --- |
| glob | 5 | 1 | 0 | 3 | 0 | 0 |
| lexical | 0 | 5 | 0 | 0 | 0 | 0 |
| regex | 0 | 3 | 0 | 1 | 0 | 0 |
| semantic | 0 | 0 | 0 | 0 | 3 | 0 |
| structural | 0 | 2 | 1 | 0 | 0 | 0 |
| usage | 0 | 0 | 0 | 0 | 2 | 1 |

