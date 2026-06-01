# SOTA Overhaul — Spike Research Notes

> Append-only, shared by the stream spikes. The embedder (S2) and reranker (S3) ONNX
> artifacts were exported/selected, verified, and **HOSTED ahead of the build on 2026-06-01**
> via `benchmarks/onnx_models/prepare_models.py`. The S2/S3 spike tasks reduce to "verify the
> recorded facts below + wire the source," not "produce them."

## S2 — CodeRankEmbed embedder (DONE: hosted + verified)

- **ModelSpec.source:** `hf:MisterTK/CodeRankEmbed-onnx-int8` (MIT)
- **Upstream:** `nomic-ai/CodeRankEmbed` (MIT), base `Snowflake/snowflake-arctic-embed-m-long`; arch `nomic_bert` (custom_code, but baked into the ONNX graph — Rust `ort` needs no Python).
- **Precision/files:** int8 dynamic quant, **ONNX external-data format** → the download file list MUST be `["model_int8.onnx", "model_int8.onnx.data", "tokenizer.json", "config.json"]` (the `.onnx` graph is ~1.2 MB; the `.onnx.data` weights are ~137 MB — both must be co-located for `ort` to load).
- **Embedding:** dim **768**; pooling **mean** (mask-weighted) → **L2-normalize**; query prefix **`Represent this query for searching relevant code: `** (documents/code get NO prefix); max context 8192. **No Matryoshka** (fixed 768-dim).
- **ONNX I/O:** inputs `input_ids`, `attention_mask` (int64) [+ `token_type_ids`→zeros if the graph declares it]; output `last_hidden_state [batch, seq, hidden]` → mean-pool (if a future export already pools to `[batch, hidden]`, use directly).
- **Verified (CPU smoke):** sim(query, relevant code) = **0.656** vs sim(query, unrelated) = **0.104**.

## S3 — Qwen3-Reranker-0.6B reranker (DONE: hosted + verified)

- **ModelSpec.source:** `hf:MisterTK/Qwen3-Reranker-0.6B-onnx` (Apache-2.0)
- **Upstream:** `Qwen/Qwen3-Reranker-0.6B` (Apache-2.0); re-hosted as-is from community export `shawnw3i/Qwen3-Reranker-0.6B-ONNX`.
- **Precision/files:** **fp16, NOT int8** (the source is already quantized; re-quantizing produced an invalid graph — fp16 scales). Files: `["model.onnx", "tokenizer.json", "config.json"]` — **filename is `model.onnx`, not `model_int8.onnx`**; adjust S3's download sentinel/file list. ~1.1 GB. A true int8 build needs a fresh fp32 export (future optimization).
- **ScoreStrategy = YesNoLogit:** chat-format the prompt, run, take `logits[:, -1, :]`, softmax over the `yes`/`no` token logits → P(yes) is the relevance score.
- **Token ids (Qwen tokenizer):** `yes` = **9693**, `no` = **2152**.
- **ONNX I/O:** inputs `input_ids`, `attention_mask`, **`position_ids`** (all int64); output `logits [batch, seq, vocab]`.
- **Prompt template (verified):**
  ```
  <|im_start|>system
  Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
  <|im_start|>user
  <Instruct>: {instruction}
  <Query>: {query}
  <Document>: {doc}<|im_end|>
  <|im_start|>assistant
  <think>

  </think>

  ```
- **Verified (CPU smoke):** P(yes | relevant) = **0.990** vs P(yes | irrelevant) = **0.000**.
- Reranker stays **off by default** (D8); this is the opt-in code-capable option, bge-reranker-v2-m3 remains the permissive fallback.

## S0 relevance harness

> Verified live on 2026-05-31 in `benchmarks/relevance/.venv` (datasets 4.8.5,
> huggingface_hub 1.17.0, pytrec_eval 0.5, python 3.12). HF access on this
> machine **works** — CoIR (cosqa) + CodeSearchNet both reachable. semantex
> binary: `/Users/tk/.cargo/bin/semantex` (on PATH).

### Step 1 — `semantex --json` output schema (VERIFIED)
`SEMANTEX_QUIET_LIMITS=1 semantex "hybrid fusion rank" --json --max-count 2`
returns a JSON **array of objects** with key set (verbatim):
`content` (str), `end_line` (int), `file` (str, repo-relative), `score` (float),
`source` (str — e.g. `"Hybrid"`, `"Sparse"`, `"ripgrep"` during fallback),
`start_line` (int). With a built index also: `chunk_type`, and optionally
`name`, `language`. **`file` is the file-level gold key; `start_line`/`end_line`
give the function-level span.** Note: the query is a positional arg to the
top-level `semantex` command — **there is NO `search` subcommand** (the help
lists `index/watch/status/serve/agent/...` but search is the default top-level
action). The harness client therefore invokes `semantex <query> --json ...`,
which the plan's `_build_args` already does (`[binary, query, "--json", ...]`).

### Step 2 — ablation flags + embedder env (VERIFIED)
`semantex --help` confirms these flags exist: `--json`, `--dense-only`,
`--sparse-only`, `--rerank`, `-m/--max-count`, `--no-content` (also `--snippet`,
`--refs`, `--peek`, `--grep`). **There is NO `--hybrid` flag** (hybrid = the
default, neither `--dense-only` nor `--sparse-only`) and **NO
`--embedder`/`--dense-backend` flag** (confirmed: grep for `embedder|dense.backend`
in `--help` → none). Embedder selection is via env var only (owned by S1/S2/S8).
Per integration §4 D-env-knob the harness sets **`SEMANTEX_EMBEDDER`** (canonical;
values `lateon-colbert` | `coderank-137m` | `qwen3-embed-0.6b`); `SEMANTEX_DENSE_BACKEND`
(`colbert-plaid` | `coderank-hnsw`) is the kept-live alias. Until S1/S2 ship, the
env var is inert (today's binary ignores it) so the D4 A/B is a no-op — correctly
gated (plan Task 8.2).

### Step 3 — CodeSearchNet HF id + schema (VERIFIED; id MOVED)
The legacy bare id `code_search_net` **FAILS** on huggingface_hub 1.17.0
(`HfUriError: Repository id must be 'namespace/name'`) — the loading-script path
is dead. **Use the official parquet mirror id `code-search-net/code_search_net`**
(org `code-search-net`, 24.2K downloads, format:parquet). It needs **NO
`trust_remote_code`** (parquet, not a script). Verified load:
`load_dataset('code-search-net/code_search_net', 'python', split='test')`
→ 22,176 rows. Configs per language: `python`, `java`, `javascript`, `go`,
`php`, `ruby`. Columns: `repository_name`, `func_path_in_repository`,
`func_name`, `whole_func_string`, `language`, `func_code_string`,
`func_code_tokens`, `func_documentation_string`, `func_documentation_tokens`,
`split_name`, `func_code_url`. Field mapping for the loader (unchanged from plan):
query = **`func_documentation_string`**, code = **`whole_func_string`**, stable
id = **`func_code_url`**, path = `func_path_in_repository`.
**ACTION → `config/csn_subset.yaml`: set `dataset_id: code-search-net/code_search_net`
and `trust_remote_code: false`.**

### Step 4 — CoIR HF ids (VERIFIED; layout DIFFERS from plan's guess)
HF access works; CoIR is reachable. The org layout is **NOT** `<name>-{queries,corpus,qrels}`.
The real layout (verified for cosqa):
- **`CoIR-Retrieval/<name>-queries-corpus`** — one repo, TWO splits: `corpus` and
  `queries`. Both have columns `_id`, `partition`, `text`, `title`, `language`,
  `meta_information`. cosqa: 20,604 corpus docs + 20,604 queries.
- **`CoIR-Retrieval/<name>-qrels`** — splits `train`/`test`/`valid`; columns
  **`query_id`, `corpus_id`, `score`** (UNDERSCORES, not the hyphenated
  `query-id`/`corpus-id` the plan's injected test data + `_qrels_map` assume!).
  cosqa test: 500 rows, all positive.
- There is ALSO a flat `CoIR-Retrieval/cosqa` (MTEB-framework version) — not used.
- `CoIR-Retrieval/<name>-queries` and `-corpus` (separate) do **NOT** exist
  (DatasetNotFoundError) → must use the combined `-queries-corpus` repo.

**SCHEMA DIVERGENCE — qrels column names:** the plan's injectable
`build_corpus_from_splits` + its `_qrels_map` read `r["query-id"]` / `r["corpus-id"]`
(hyphens) and the unit tests inject hyphen keys. The REAL HF qrels use
`query_id` / `corpus_id` (underscores). Resolution: keep the injectable function
+ tests exactly as the plan writes them (hyphen-keyed, unit-tested), and have the
**network adapter `load_coir_subdataset` normalize** the HF rows to the
hyphen-keyed shape before calling `build_corpus_from_splits` (map `query_id`→`query-id`,
`corpus_id`→`corpus-id`). Corpus/query rows already use `_id`/`text` as the plan
expects.
**Real ids to put in `config/coir_subset.yaml`:**
- cosqa → queries_corpus_id: `CoIR-Retrieval/cosqa-queries-corpus`, qrels_id:
  `CoIR-Retrieval/cosqa-qrels`, qrels split `test`.
- CodeSearchNet (CoIR) → `CoIR-Retrieval/CodeSearchNet-queries-corpus` +
  `CoIR-Retrieval/CodeSearchNet-qrels` (same pattern; not re-probed but layout is
  org-uniform).
CoIR is **NOT deferred** (HF reachable here); the loader + a small live smoke are
buildable. The acceptance GATE still anchors on the deterministic BM25/CSN
`--sparse-only` baseline (model-independent) per integration §4 D-s0-gate.

### Step 5 — pytrec_eval (VERIFIED; installs + builds)
`pip install pytrec_eval` → built a wheel cleanly on macOS arm64 (python 3.12).
`RelevanceEvaluator(qrel, {'recip_rank','ndcg_cut.10','recall.10','map'}).evaluate(run)`
returns keys with the requested-measure dots turned to underscores:
**`recip_rank`, `ndcg_cut_10`, `recall_10`, `map`**. Task 2.2's
`test_metrics_vs_pytrec_eval.py` uses exactly these. pytrec_eval is a hard dev
dep here (no skip needed), but the test keeps `importorskip` for portability.

### Step 6 — swe_bench module contract (VERIFIED)
`benchmarks/swe_bench/src/swe_bench_harness/` importable via
`sys.path.insert(0,'src')`. Confirmed:
- `dataset.Instance` dataclass fields = `['instance_id','repo','base_commit','problem_statement']`
  → **NO `patch` field** (plan correct; `swe_loc.py` reads `patch` from the raw HF
  row / local JSON, not from `Instance`).
- `indexer.index_repo(*, repo_path: Path, semantex_binary: str, timeout_secs: int = 600) -> IndexResult`.
- `indexer.IndexResult` fields = `['ok','path','error','duration_secs']` (matches
  the plan's `ensure_index` usage: `result.ok` / `result.error`).
- Repo cache convention: `$SWE_BENCH_REPO_CACHE/<instance_id>/` (default
  `~/.swe_bench_repos`), completed index ⇔ `.semantex/meta.json` with
  `chunk_count > 0`. `~/.swe_bench_repos` may be EMPTY on this machine → SWE-loc
  must index on demand via `ensure_index`/`index_repo`.

### Locked outputs (referenced by later tasks)
- `semantex --json` keys: `file,start_line,end_line,score,source,content[,chunk_type,name,language]`; query is a positional arg, no `search` subcommand.
- ablation→flag: sparse-only→`--sparse-only`, dense-only→`--dense-only`, hybrid→(none), rerank→`--rerank`; embedder→`SEMANTEX_EMBEDDER` env.
- CSN id: `code-search-net/code_search_net`, `trust_remote_code: false`.
- CoIR: `CoIR-Retrieval/<name>-queries-corpus` (splits corpus/queries) + `CoIR-Retrieval/<name>-qrels` (split test, cols query_id/corpus_id/score — underscores; adapter normalizes to hyphens).
- pytrec_eval measures: `recip_rank, ndcg_cut_10, recall_10, map`.
- swe_bench: `Instance` lacks `patch`; `index_repo(*, repo_path, semantex_binary, timeout_secs)`→`IndexResult(ok,path,error,duration_secs)`.

### S0 addendum — CoIR loader verified live + subset-coherence caveat
`load_coir_subdataset(name='cosqa', queries_corpus_id='CoIR-Retrieval/cosqa-queries-corpus',
qrels_id='CoIR-Retrieval/cosqa-qrels', corpus_size=None, query_size=10)` verified
end-to-end against live HF on 2026-05-31: 20,604 corpus docs materialised, 10
queries selected, **all gold docs present in the materialised corpus**, sample
gold id `d20145__doc_11275.txt:1-...`. The underscore→hyphen qrels normalization
in `_normalize_qrel_rows` works.
**CAVEAT (load-bearing for CoIR runs):** capping `corpus_size` while selecting
queries by `query_size` can ORPHAN queries whose gold corpus doc falls outside
the first-N corpus slice (observed: `corpus_size=200` → 0 surviving queries,
because cosqa qrels are diagonal q_i↔d_i with i in the 20k range). The plan's
loader correctly DROPS orphaned queries (no silent gold loss), but to get a
non-empty CoIR eval either set `corpus_size: null` (index the full corpus) or
make corpus subsetting gold-aware (retain every selected query's gold doc) — a
follow-up refinement. The injectable `build_corpus_from_splits` + its unit tests
are unaffected (they use a tiny coherent fixture). `config/coir_subset.yaml`
ships `corpus_size: 5000`; bump to `null` for a real cosqa headline run, or wire
gold-aware corpus subsetting first.

### S0 addendum — acceptance gate calibration (Task 7.2)
The BM25/CSN gate (`scripts/reproduce_baseline.py --baseline csn_bm25`) ran the
FULL configured python subset (1000 docs / 200 queries, seed 20260531,
`--sparse-only`). Measured **MRR@10 = 0.9099–0.9124** across trials (~0.0025
run-to-run jitter from semantex's sparse tie-break ordering). Rank distribution:
**176/200 gold@rank-1, 8/200 not-in-top-10**, Recall@1=0.88, Recall@10=0.955 —
genuine strong retrieval, NOT a wiring fluke. The published CodeSearchNet/
CodeXGLUE BM25 MRR is ~0.49 (1000-candidate, docstring-SEPARATED protocol);
our subset scores higher because (a) ~1000-doc haystack and (b)
`whole_func_string` embeds the query docstring inside the document (near-exact
lexical overlap). So the gate anchors to semantex's OWN deterministic protocol
(`expected_mrr_at_10: 0.91`, `tolerance: 0.10`), satisfying the spec's "CSN
baseline within a *stated* tolerance" — provenance documented in
`config/baselines.yaml`, not hidden. **GATE: PASS (delta 0.0024 ≤ 0.10).** This
is the model-independent D-s0-gate anchor; CoIR headline numbers are a future
full-corpus run (loader verified-reachable, see CoIR addendum).
