# semantex Relevance Report

- **git rev:** `af3ad57`
- **dense backend:** `colbert-plaid (default)`
- **model:** `n/a-dense-path`
- **cutoff k:** 10

## Metrics

| dataset        | ablation    |   n_queries |   mrr_at_10 |   ndcg_at_10 |   recall_at_1 |   recall_at_5 |   recall_at_10 |    map |
|:---------------|:------------|------------:|------------:|-------------:|--------------:|--------------:|---------------:|-------:|
| csn/python     | sparse-only |         200 |      0.9124 |       0.9231 |        0.8850 |        0.9500 |         0.9550 | 0.9127 |
| csn/javascript | sparse-only |         200 |      0.3114 |       0.3509 |        0.2400 |        0.4350 |         0.4750 | 0.3118 |
| csn/go         | sparse-only |         200 |      0.5603 |       0.6222 |        0.4300 |        0.7250 |         0.8150 | 0.5603 |
| csn/python     | dense-only  |         200 |      0.8861 |       0.9071 |        0.8300 |        0.9550 |         0.9700 | 0.8864 |
| csn/javascript | dense-only  |         200 |      0.4680 |       0.5285 |        0.3350 |        0.6600 |         0.7150 | 0.4684 |
| csn/go         | dense-only  |         200 |      0.6940 |       0.7471 |        0.5750 |        0.8650 |         0.9100 | 0.6940 |
| csn/python     | hybrid      |         200 |      0.8861 |       0.9071 |        0.8300 |        0.9550 |         0.9700 | 0.8864 |
| csn/javascript | hybrid      |         200 |      0.4624 |       0.5244 |        0.3250 |        0.6600 |         0.7150 | 0.4628 |
| csn/go         | hybrid      |         200 |      0.6933 |       0.7465 |        0.5750 |        0.8650 |         0.9100 | 0.6933 |
| csn/python     | rerank      |         200 |      0.8861 |       0.9071 |        0.8300 |        0.9550 |         0.9700 | 0.8864 |
| csn/javascript | rerank      |         200 |      0.4709 |       0.5307 |        0.3400 |        0.6650 |         0.7150 | 0.4713 |
| csn/go         | rerank      |         200 |      0.6940 |       0.7471 |        0.5750 |        0.8650 |         0.9100 | 0.6940 |

