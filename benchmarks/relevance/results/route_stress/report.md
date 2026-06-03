# Route-stress oracle-regret evaluation

- git rev: `6fbc82a`
- generated: 2026-06-03 11:48:10
- config: `SEMANTEX_ADAPTIVE_SIZING=0` (canonical A/B lock)
- retrieval routes scored: file_pattern, regex, exact_symbol, structural, semantic
- synthesis routes (scored 0 for file-gold): analytical, architecture, deep, exhaustive, feature_planning

## Summary (per repo)

| repo | queries | overall regret | router acc | synthesis picks |
| --- | --- | --- | --- | --- |
| gin | 30 | 0.4566 | 33.3% | 10 |

## Pooled per-route oracle-win counts (all repos)
(total queries: 30)

| route | oracle-wins |
| --- | --- |
| file_pattern | 0 |
| regex | 15 |
| exact_symbol | 16 |
| structural | 20 |
| semantic | 18 |

## gin

- total queries: **30**
- overall regret (oracle nDCG@10 − router-picked nDCG@10): **0.4566**
- router accuracy (router choice == an oracle-best route): **33.3%** (10/30)
- router chose a SYNTHESIS route (no file hits, scored 0): **10**

### Per-route oracle-win counts
(a route 'wins' a query when it is — or ties — the best-nDCG route; ~0 wins ⇒ deletion candidate)

| route | oracle-wins | mean nDCG@10 |
| --- | --- | --- |
| file_pattern | 0 | 0.0000 |
| regex | 15 | 0.7176 |
| exact_symbol | 16 | 0.7031 |
| structural | 20 | 0.7623 |
| semantic | 18 | 0.7324 |

### Per-mechanism × per-route mean nDCG@10
(does each route actually win on its intended mechanism? does one route dominate everything?)

| mechanism | file_pattern | regex | exact_symbol | structural | semantic |
| --- | --- | --- | --- | --- | --- |
| glob | 0.000 | **0.682** | 0.376 | 0.540 | 0.537 |
| lexical | 0.000 | **0.926** | 0.926 | 0.926 | 0.926 |
| regex | 0.000 | **0.590** | 0.361 | 0.566 | 0.376 |
| semantic | 0.000 | 0.565 | **0.970** | 0.921 | 0.970 |
| structural | 0.000 | **1.000** | 0.859 | 0.894 | 0.859 |
| usage | 0.000 | 0.542 | **0.726** | 0.726 | 0.726 |

### Regret per mechanism

| mechanism | mean regret | router accuracy |
| --- | --- | --- |
| glob | 0.3308 | 40.0% |
| lexical | 0.0738 | 80.0% |
| regex | 0.4586 | 0.0% |
| semantic | 0.9701 | 0.0% |
| structural | 0.0939 | 80.0% |
| usage | 0.8123 | 0.0% |

### Confusion matrix (rows = intended_mechanism, cols = router choice)

| mechanism ↓ / choice → | file_pattern | structural | semantic | deep |
| --- | --- | --- | --- | --- |
| glob | 1 | 0 | 4 | 0 |
| lexical | 0 | 0 | 5 | 0 |
| regex | 1 | 0 | 4 | 0 |
| semantic | 0 | 0 | 0 | 5 |
| structural | 0 | 4 | 1 | 0 |
| usage | 0 | 0 | 0 | 5 |

