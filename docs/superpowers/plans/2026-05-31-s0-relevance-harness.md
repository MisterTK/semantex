# S0 — Relevance Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a leaderboard-grade, pure-retrieval relevance harness `benchmarks/relevance/` that measures semantex on CoIR + CodeSearchNet + SWE-bench localization with MRR@10 / nDCG@10 / Recall@{1,5,10} / MAP, supports the `--sparse-only / --dense-only / --hybrid / --rerank` ablations and the D4 `--dense-backend colbert-plaid|coderank-hnsw` A/B, and reproduces a published CSN/CoIR baseline within a stated tolerance.

**Architecture:** A Python package under `benchmarks/relevance/` (mirroring `benchmarks/swe_bench/`: `pyproject.toml`, `src/`, `tests/`, `fixtures/`, gitignored `results/`) orchestrates three pieces: (1) dataset loaders that emit a uniform `(corpus, queries, qrels)` triple with deterministic *logged* subsetting; (2) a runner that indexes each corpus with `semantex index`, calls `semantex search --json` per query under the requested ablation/backend, and records the rank of each gold target; (3) pure-function metrics + a report module that emits `report.md` + machine-readable JSON with a subset manifest and a git-rev/model stamp. The SWE-loc loader reuses `benchmarks/swe_bench/`'s `dataset.py` / `repo_checkout.py` / `indexer.py` so the Phase-A 100 pre-indexed instances are shared, not re-derived.

**Tech Stack:**
- Python 3.12 (matches `benchmarks/swe_bench/.python-version`), `pytest`, `click`, `pyyaml`, `numpy`, `pandas`
- `datasets` (HuggingFace) for CoIR + CodeSearchNet loading; HF cache under `$HF_HOME`
- `pytrec_eval` (PyPI, the official TREC `trec_eval` Python binding) used **only in tests** to validate our metric formulas against the reference implementation — the production metrics are dependency-free pure NumPy
- Existing semantex CLI — invoked as a subprocess via `semantex search --json`; never imported
- Reuses `benchmarks/swe_bench/src/swe_bench_harness/{dataset,repo_checkout,indexer}.py` for SWE-loc (added to `sys.path` in `conftest.py` + at runtime)

---

## File Structure

```
benchmarks/relevance/
├── README.md                          # operator-facing how-to
├── pyproject.toml                     # Python project metadata + deps
├── .python-version                    # 3.12
├── .gitignore                         # results/, caches
├── config/
│   ├── coir_subset.yaml               # CoIR datasets + per-dataset sample sizes + seed
│   ├── csn_subset.yaml                # CodeSearchNet languages + sample sizes + seed
│   └── baselines.yaml                 # published CSN/CoIR baseline numbers + tolerances
├── src/relevance_harness/
│   ├── __init__.py
│   ├── types.py                       # Query, Document, Qrels, EvalCorpus, RankedResult dataclasses
│   ├── subset.py                      # deterministic, logged subset selection + manifest
│   ├── metrics.py                     # MRR@k, nDCG@k, Recall@k, MAP (pure NumPy)
│   ├── semantex_client.py             # subprocess wrapper for `semantex search --json` + ablation/backend env
│   ├── indexing.py                    # locate-or-build a semantex index for a corpus dir
│   ├── runner.py                      # (corpus,queries,qrels) × ablation → per-query ranked target positions
│   ├── datasets/
│   │   ├── __init__.py
│   │   ├── coir.py                    # CoIR loader → EvalCorpus (+ logged subset)
│   │   ├── csn.py                     # CodeSearchNet loader → EvalCorpus (+ logged subset)
│   │   └── swe_loc.py                 # SWE-bench Verified gold-patch → file/function qrels
│   └── report.py                      # aggregate → report.md + report.json (manifest + git/model stamp)
├── scripts/
│   ├── __init__.py
│   ├── run.py                         # main entrypoint: --dataset --ablation --dense-backend
│   └── reproduce_baseline.py          # acceptance gate: CSN/CoIR baseline within tolerance
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_types.py
│   ├── test_subset.py
│   ├── test_metrics.py
│   ├── test_metrics_vs_pytrec_eval.py
│   ├── test_semantex_client.py
│   ├── test_indexing.py
│   ├── test_runner.py
│   ├── test_coir.py
│   ├── test_csn.py
│   ├── test_swe_loc.py
│   └── test_report.py
├── fixtures/
│   ├── tiny_corpus/                   # 6-file repo indexed in tests (real files)
│   │   ├── auth.py
│   │   ├── db.py
│   │   └── util.py
│   ├── tiny_eval_corpus.json          # 3 queries + qrels over tiny_corpus for runner tests
│   └── sample_verified_subset.json    # 3 SWE-bench Verified rows WITH `patch` field
└── results/                           # gitignored; runN/ subdirs
    └── .gitkeep
```

**Module responsibilities (one job each):**
- `types.py` — the uniform data model every loader emits; no I/O.
- `subset.py` — deterministic seeded subsetting + the manifest of what was kept/dropped. Never silently truncate.
- `metrics.py` — pure ranking metrics on `(ranked_relevances)`; no I/O, no semantex.
- `semantex_client.py` — exactly one job: run `semantex search --json` with the right flags/env and parse the JSON array.
- `indexing.py` — locate-or-build a `.semantex` index for a corpus directory (reuses swe_bench `indexer.py`).
- `runner.py` — drive client over a corpus's queries, return per-query target positions; no metric math.
- `datasets/coir.py`, `datasets/csn.py`, `datasets/swe_loc.py` — each loads its source into an `EvalCorpus`.
- `report.py` — aggregate per-dataset + overall; write `report.md` + `report.json` with manifest + stamps.

---

## Phase 0 — Research & Scaffolding

### Task 0.1: Verify external assumptions (research-only, no commit)

**Files:** none (write findings to `docs/superpowers/plans/2026-05-31-research-notes.md`)

These facts are genuinely external/uncertain. Record the REAL values; later tasks reference them by the recorded names.

- [ ] **Step 1: Confirm the `semantex search --json` output schema (VERIFY — do not assume)**

Run (from the semantex repo root, which already has a `.semantex` index):
```bash
SEMANTEX_QUIET_LIMITS=1 semantex "hybrid fusion rank" --json --max-count 2
```
Expected: a JSON array of objects. Record the exact key set. As of this plan it is:
`file` (str, repo-relative), `start_line` (int), `end_line` (int), `score` (float), `source` (str e.g. "Hybrid"), `content` (str), `chunk_type` (str/obj), and optionally `name`, `language`.
**The `file` field is the file-level gold key; `start_line`/`end_line` give function-level overlap.** Record verbatim in research notes.

- [ ] **Step 2: Confirm semantex ablation flags + dense-backend env (VERIFY)**

Run: `SEMANTEX_QUIET_LIMITS=1 semantex search --help`
Record that these flags exist: `--json`, `--dense-only`, `--sparse-only`, `--rerank`, `-m/--max-count`, `--no-content`.
**Record explicitly: there is NO `--hybrid` flag (hybrid is the default = neither `--dense-only` nor `--sparse-only`) and NO `--dense-backend` flag.** Dense-backend selection is via the `SEMANTEX_DENSE_BACKEND` env var (introduced by streams S1/S2). Record that the harness selects the backend by setting `SEMANTEX_DENSE_BACKEND=colbert-plaid|coderank-hnsw` in the subprocess env.

- [ ] **Step 3: Confirm CodeSearchNet HuggingFace dataset id + schema (VERIFY)**

Run:
```bash
cd benchmarks/relevance && .venv/bin/python -c "
from datasets import load_dataset
d = load_dataset('code_search_net', 'python', split='test', trust_remote_code=True)
print('rows:', len(d))
print('cols:', d.column_names)
r = d[0]
print({k: (v[:80] if isinstance(v,str) else v) for k,v in r.items() if k in ('func_name','whole_func_string','func_documentation_string','func_code_url','repository_name','func_path_in_repository')})
"
```
Expected columns include `func_name`, `whole_func_string` (the code), `func_documentation_string` (the NL docstring → the query), `func_code_url`, `func_path_in_repository`, `repository_name`, `language`.
Record: (a) the exact config name for each language (`python`,`java`,`javascript`,`go`,`php`,`ruby`), (b) the query field (`func_documentation_string`), (c) the code field (`whole_func_string`), (d) a stable per-document id field (`func_code_url` is unique). If `code_search_net` requires `trust_remote_code=True`, record that. If the canonical id has moved (e.g. `Nan-Do/code_search_net_*` or `sentence-transformers/codesearchnet`), record the actual id used.

- [ ] **Step 4: Confirm CoIR task suite HuggingFace ids (VERIFY — most uncertain)**

CoIR is published as a suite of sub-datasets. Determine the real HF ids for the CPU-feasible subset. Try:
```bash
cd benchmarks/relevance && .venv/bin/python -c "
from datasets import load_dataset
# CoIR sub-datasets are published under the CoIR-Retrieval org as <name>-{queries,corpus,qrels}
for name in ['CoIR-Retrieval/CodeSearchNet','CoIR-Retrieval/cosqa','CoIR-Retrieval/codetrans-dl']:
    try:
        q = load_dataset(name+'-queries', split='test')
        print(name, 'queries cols:', q.column_names, 'n:', len(q))
    except Exception as e:
        print(name, 'ERR', repr(e)[:120])
"
```
Record, for each CoIR sub-dataset chosen for the subset: (a) the exact queries / corpus / qrels HF ids, (b) the column names for query text, doc text, doc id, and the qrels mapping (query-id → corpus-id → relevance). If the CoIR org layout differs from `<name>-{queries,corpus,qrels}`, record the actual layout. **If CoIR cannot be loaded on this machine at all (network/gated), record that and mark CoIR as deferred to a later task that runs where HF access exists — CSN + SWE-loc remain the buildable headline for the acceptance gate.**

- [ ] **Step 5: Confirm `pytrec_eval` is installable + its measure names (VERIFY)**

Run:
```bash
cd benchmarks/relevance && .venv/bin/pip install pytrec_eval >/dev/null 2>&1 && .venv/bin/python -c "
import pytrec_eval
qrel = {'q1': {'d1': 1, 'd2': 0, 'd3': 1}}
run  = {'q1': {'d1': 0.9, 'd3': 0.8, 'd2': 0.1}}
ev = pytrec_eval.RelevanceEvaluator(qrel, {'recip_rank','ndcg_cut.10','recall.10','map'})
print(ev.evaluate(run)['q1'])
"
```
Expected: a dict with keys `recip_rank`, `ndcg_cut_10`, `recall_10`, `map`. Record the exact measure-name strings — `test_metrics_vs_pytrec_eval.py` (Task 2.2) asserts our pure-NumPy metrics agree with these to within `1e-9`. If `pytrec_eval` fails to build on this platform, record that and mark Task 2.2 as a dev-only optional test (skip if import fails) — the canonical-definition metrics in Task 2.1 stand on their own.

- [ ] **Step 6: Confirm the reusable swe_bench modules + their import path (VERIFY)**

Run:
```bash
cd benchmarks/swe_bench && .venv/bin/python -c "
import sys; sys.path.insert(0,'src')
from swe_bench_harness.dataset import Instance, load_verified, select_subset
from swe_bench_harness.repo_checkout import checkout
from swe_bench_harness.indexer import index_repo, IndexResult
print('OK', Instance, index_repo)
"
```
Expected: prints `OK ...`. Record: (a) `swe_bench_harness.dataset.Instance` has fields `instance_id, repo, base_commit, problem_statement` but **NO `patch` field** — so `swe_loc.py` must read the `patch` field directly from the HF row / local JSON, not from `Instance`. (b) The repo cache convention is `$SWE_BENCH_REPO_CACHE/<instance_id>/` (default `~/.swe_bench_repos`) with a completed index marked by `.semantex/meta.json` containing `chunk_count > 0`. Record that `~/.swe_bench_repos` may be EMPTY on a fresh machine, so `swe_loc.py` + the runner must index on demand (reuse `index_repo`).

- [ ] **Step 7: Write research notes (no commit — controller commits)**

Write all recorded values to `docs/superpowers/plans/2026-05-31-research-notes.md` under a `## S0 relevance harness` section. Leave version control to the controller.

**Outputs locked after this task:** the `semantex --json` key set, the ablation→flag/env mapping, the CSN dataset id + fields, the CoIR ids (or a deferral note), `pytrec_eval` measure names, and the swe_bench module contract. All later tasks reference these recorded values.

---

### Task 0.2: Scaffold the Python project

**Files:**
- Create: `benchmarks/relevance/pyproject.toml`
- Create: `benchmarks/relevance/.python-version`
- Create: `benchmarks/relevance/.gitignore`
- Create: `benchmarks/relevance/README.md`
- Create: `benchmarks/relevance/src/relevance_harness/__init__.py`
- Create: `benchmarks/relevance/scripts/__init__.py`
- Create: `benchmarks/relevance/src/relevance_harness/datasets/__init__.py`
- Create: `benchmarks/relevance/tests/__init__.py`
- Create: `benchmarks/relevance/tests/conftest.py`
- Create: `benchmarks/relevance/results/.gitkeep`

- [ ] **Step 1: Write `pyproject.toml`**

`benchmarks/relevance/pyproject.toml`:
```toml
[project]
name = "relevance-harness"
version = "0.1.0"
requires-python = ">=3.12,<3.14"
dependencies = [
  "click>=8.1",
  "datasets>=2.20",
  "numpy>=1.26",
  "pandas>=2.2",
  "pyyaml>=6.0",
  "tabulate>=0.9",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-cov>=5.0", "ruff>=0.6", "pytrec_eval>=0.5"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]

[tool.ruff]
line-length = 100
target-version = "py312"
```

- [ ] **Step 2: Write `.python-version`**

`benchmarks/relevance/.python-version`:
```
3.12
```

- [ ] **Step 3: Write `.gitignore`**

`benchmarks/relevance/.gitignore`:
```
results/
*.egg-info/
__pycache__/
.pytest_cache/
.venv/
```

- [ ] **Step 4: Write `README.md` (operator-facing)**

`benchmarks/relevance/README.md`:
```markdown
# semantex Relevance Harness

Pure-retrieval relevance measurement for semantex (no LLM confound).
Datasets: CoIR (headline), CodeSearchNet (external calibration), SWE-bench
localization (code-native). Metrics: MRR@10, nDCG@10, Recall@{1,5,10}, MAP.

## Setup
```
cd benchmarks/relevance
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Run
```
# CodeSearchNet, full hybrid, default dense backend
python -m scripts.run --dataset csn --ablation hybrid

# Ablation sweep (sparse vs dense vs hybrid vs hybrid+rerank)
python -m scripts.run --dataset csn --ablation sparse-only
python -m scripts.run --dataset csn --ablation dense-only
python -m scripts.run --dataset csn --ablation rerank

# D4 dense-backend A/B (env-selected backend)
python -m scripts.run --dataset csn --ablation hybrid --dense-backend colbert-plaid
python -m scripts.run --dataset csn --ablation hybrid --dense-backend coderank-hnsw

# SWE-bench localization (reuses benchmarks/swe_bench Phase-A instances)
python -m scripts.run --dataset swe-loc --ablation hybrid

# Acceptance gate: reproduce a published baseline within tolerance
python -m scripts.reproduce_baseline --dataset csn
```

All subsets are seeded and logged; every report records the exact datasets,
sample sizes, dropped items, git rev, and dense backend used.
```

- [ ] **Step 5: Write `__init__.py` files + `conftest.py` + `.gitkeep`**

`benchmarks/relevance/src/relevance_harness/__init__.py`:
```python
"""Pure-retrieval relevance harness for semantex."""
__version__ = "0.1.0"
```

`benchmarks/relevance/scripts/__init__.py`: empty file.

`benchmarks/relevance/src/relevance_harness/datasets/__init__.py`: empty file.

`benchmarks/relevance/tests/__init__.py`: empty file.

`benchmarks/relevance/results/.gitkeep`: empty file.

`benchmarks/relevance/tests/conftest.py`:
```python
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Make the sibling swe_bench harness importable for SWE-loc reuse.
SWE_BENCH_SRC = ROOT.parent / "swe_bench" / "src"
if SWE_BENCH_SRC.is_dir():
    sys.path.insert(0, str(SWE_BENCH_SRC))


@pytest.fixture
def fixtures_dir() -> Path:
    return ROOT / "fixtures"
```

- [ ] **Step 6: Verify install + empty test suite**

```bash
cd benchmarks/relevance
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -v
```
Expected: `pytest` runs and exits 0 (no tests collected yet).

- [ ] **Step 7: Commit**

```bash
git add benchmarks/relevance/
git commit -m "$(cat <<'EOF'
feat(relevance): scaffold relevance benchmark Python project

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 1 — Core data model + subsetting

### Task 1.1: Uniform data model (TDD)

**Files:**
- Create: `benchmarks/relevance/src/relevance_harness/types.py`
- Create: `benchmarks/relevance/tests/test_types.py`

- [ ] **Step 1: Write failing tests**

`benchmarks/relevance/tests/test_types.py`:
```python
import pytest

from relevance_harness.types import (
    Document, EvalCorpus, Query, RankedResult,
)


def test_query_and_document_round_trip():
    q = Query(query_id="q1", text="how is auth handled", gold_doc_ids=("d1", "d3"))
    d = Document(doc_id="d1", text="def login(): ...", file_path="auth.py",
                 start_line=10, end_line=20)
    assert q.gold_doc_ids == ("d1", "d3")
    assert d.file_path == "auth.py"
    assert d.start_line == 10


def test_eval_corpus_qrels_lookup():
    corpus = EvalCorpus(
        name="tiny",
        documents=(
            Document(doc_id="d1", text="a", file_path="a.py", start_line=1, end_line=2),
            Document(doc_id="d2", text="b", file_path="b.py", start_line=1, end_line=2),
        ),
        queries=(
            Query(query_id="q1", text="find a", gold_doc_ids=("d1",)),
        ),
        corpus_dir=None,
    )
    assert corpus.name == "tiny"
    assert len(corpus.documents) == 2
    assert corpus.queries[0].gold_doc_ids == ("d1",)
    # qrels() returns {query_id: {doc_id: 1}}
    assert corpus.qrels() == {"q1": {"d1": 1}}


def test_ranked_result_holds_ordered_doc_ids():
    rr = RankedResult(query_id="q1", ranked_doc_ids=("d3", "d1", "d2"))
    assert rr.ranked_doc_ids[0] == "d3"
    assert rr.rank_of("d1") == 2          # 1-based rank
    assert rr.rank_of("missing") is None


def test_ranked_result_file_level_rank():
    # When matching by file (SWE-loc / CSN), ranked entries carry file paths.
    rr = RankedResult(
        query_id="q1",
        ranked_doc_ids=("x.py:1-5", "auth.py:10-20", "z.py:1-9"),
        ranked_files=("x.py", "auth.py", "z.py"),
    )
    assert rr.rank_of_file("auth.py") == 2
    assert rr.rank_of_file("absent.py") is None
```

- [ ] **Step 2: Run — expect failure**

Run: `cd benchmarks/relevance && pytest tests/test_types.py -v`
Expected: `ImportError` — `relevance_harness.types` not found / `cannot import name 'Document'`.

- [ ] **Step 3: Implement `types.py`**

`benchmarks/relevance/src/relevance_harness/types.py`:
```python
"""Uniform data model emitted by every dataset loader. No I/O."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class Document:
    """A retrievable unit. `file_path` + line range let us match by file/function."""
    doc_id: str
    text: str
    file_path: str
    start_line: int = 0
    end_line: int = 0


@dataclass(frozen=True)
class Query:
    """A query with its set of relevant (gold) document ids."""
    query_id: str
    text: str
    gold_doc_ids: tuple[str, ...]


@dataclass(frozen=True)
class EvalCorpus:
    """A complete retrieval task: documents + queries + (implicit) qrels.

    `corpus_dir` is the on-disk directory that gets indexed by semantex. For
    synthetic/HF corpora it is materialised on disk by the loader; for SWE-loc
    it points at the already-checked-out repo. None only in pure-unit tests.
    """
    name: str
    documents: tuple[Document, ...]
    queries: tuple[Query, ...]
    corpus_dir: Optional[Path]

    def qrels(self) -> dict[str, dict[str, int]]:
        """{query_id: {gold_doc_id: 1}} — binary relevance."""
        return {
            q.query_id: {d: 1 for d in q.gold_doc_ids}
            for q in self.queries
        }


@dataclass(frozen=True)
class RankedResult:
    """semantex's ranked output for one query, normalised to doc ids (+ files)."""
    query_id: str
    ranked_doc_ids: tuple[str, ...]
    ranked_files: tuple[str, ...] = field(default_factory=tuple)

    def rank_of(self, doc_id: str) -> Optional[int]:
        """1-based rank of `doc_id`, or None if absent."""
        for i, d in enumerate(self.ranked_doc_ids, start=1):
            if d == doc_id:
                return i
        return None

    def rank_of_file(self, file_path: str) -> Optional[int]:
        """1-based rank of the first result whose file == file_path, or None."""
        for i, f in enumerate(self.ranked_files, start=1):
            if f == file_path:
                return i
        return None
```

- [ ] **Step 4: Run — expect pass**

Run: `cd benchmarks/relevance && pytest tests/test_types.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/relevance/src/relevance_harness/types.py \
        benchmarks/relevance/tests/test_types.py
git commit -m "$(cat <<'EOF'
feat(relevance): uniform Query/Document/EvalCorpus/RankedResult data model

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 1.2: Deterministic logged subsetting (TDD)

**Files:**
- Create: `benchmarks/relevance/src/relevance_harness/subset.py`
- Create: `benchmarks/relevance/tests/test_subset.py`

- [ ] **Step 1: Write failing tests**

`benchmarks/relevance/tests/test_subset.py`:
```python
from relevance_harness.subset import SubsetManifest, select_queries


def _queries(n: int) -> list[dict]:
    return [{"query_id": f"q{i}", "text": f"t{i}"} for i in range(n)]


def test_select_is_deterministic_for_same_seed():
    qs = _queries(20)
    a, ma = select_queries(qs, n=5, seed=42, dataset="csn")
    b, mb = select_queries(qs, n=5, seed=42, dataset="csn")
    assert [q["query_id"] for q in a] == [q["query_id"] for q in b]
    assert ma.kept_ids == mb.kept_ids


def test_different_seeds_differ():
    qs = _queries(20)
    a, _ = select_queries(qs, n=5, seed=1, dataset="csn")
    b, _ = select_queries(qs, n=5, seed=2, dataset="csn")
    assert {q["query_id"] for q in a} != {q["query_id"] for q in b}


def test_manifest_records_kept_and_dropped():
    qs = _queries(10)
    selected, manifest = select_queries(qs, n=3, seed=0, dataset="csn")
    assert isinstance(manifest, SubsetManifest)
    assert manifest.total == 10
    assert manifest.selected == 3
    assert len(manifest.kept_ids) == 3
    assert len(manifest.dropped_ids) == 7
    assert set(manifest.kept_ids) | set(manifest.dropped_ids) == {q["query_id"] for q in qs}
    assert manifest.seed == 0
    assert manifest.dataset == "csn"


def test_n_none_or_larger_than_pool_keeps_all_and_logs_no_drop():
    qs = _queries(4)
    selected, manifest = select_queries(qs, n=None, seed=0, dataset="csn")
    assert len(selected) == 4
    assert manifest.selected == 4
    assert manifest.dropped_ids == []


def test_select_sorts_by_id_for_canonical_order():
    qs = [{"query_id": f"q{i}", "text": "x"} for i in (5, 1, 3, 2, 4)]
    selected, _ = select_queries(qs, n=None, seed=0, dataset="csn")
    assert [q["query_id"] for q in selected] == ["q1", "q2", "q3", "q4", "q5"]
```

- [ ] **Step 2: Run — expect failure**

Run: `cd benchmarks/relevance && pytest tests/test_subset.py -v`
Expected: `ImportError` — `relevance_harness.subset` not found.

- [ ] **Step 3: Implement `subset.py`**

`benchmarks/relevance/src/relevance_harness/subset.py`:
```python
"""Deterministic, seeded, *logged* subsetting. Never silently truncate."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class SubsetManifest:
    """Exactly what a subset kept and dropped — recorded in every report."""
    dataset: str
    total: int
    selected: int
    seed: int
    kept_ids: list[str]
    dropped_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "total": self.total,
            "selected": self.selected,
            "seed": self.seed,
            "kept_ids": self.kept_ids,
            "dropped_ids": self.dropped_ids,
        }


def select_queries(
    queries: list[dict],
    *,
    n: Optional[int],
    seed: int,
    dataset: str,
    id_key: str = "query_id",
) -> tuple[list[dict], SubsetManifest]:
    """Pick a deterministic seeded subset of `queries` and record a manifest.

    `queries` is a list of dicts each carrying `id_key`. Canonical order is by
    id (stable across runs). If `n` is None or >= the pool size, keep ALL and
    record an empty dropped list. Selection is reproducible for a fixed seed.
    """
    pool = sorted(queries, key=lambda q: q[id_key])
    total = len(pool)
    if n is None or n >= total:
        kept = pool
    else:
        rng = random.Random(seed)
        kept = sorted(rng.sample(pool, n), key=lambda q: q[id_key])
    kept_ids = [q[id_key] for q in kept]
    kept_set = set(kept_ids)
    dropped_ids = [q[id_key] for q in pool if q[id_key] not in kept_set]
    manifest = SubsetManifest(
        dataset=dataset,
        total=total,
        selected=len(kept),
        seed=seed,
        kept_ids=kept_ids,
        dropped_ids=dropped_ids,
    )
    return kept, manifest
```

- [ ] **Step 4: Run — expect pass**

Run: `cd benchmarks/relevance && pytest tests/test_subset.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/relevance/src/relevance_harness/subset.py \
        benchmarks/relevance/tests/test_subset.py
git commit -m "$(cat <<'EOF'
feat(relevance): deterministic seeded subsetting with kept/dropped manifest

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 2 — Metrics (the reference-correct core)

### Task 2.1: Ranking metrics from canonical definitions (TDD)

**Files:**
- Create: `benchmarks/relevance/src/relevance_harness/metrics.py`
- Create: `benchmarks/relevance/tests/test_metrics.py`

The four metrics use standard textbook IR definitions (no oxirs source needed):
- **MRR@k**: mean over queries of `1/rank` of the first relevant doc within top-k (0 if none).
- **nDCG@k**: `DCG@k / IDCG@k` with `DCG = Σ rel_i / log2(i+1)` (i 1-based), binary rel.
- **Recall@k**: `|relevant ∩ top-k| / |relevant|`.
- **MAP**: mean over queries of average precision (precision at each relevant hit, averaged over all relevant).

- [ ] **Step 1: Write failing tests**

`benchmarks/relevance/tests/test_metrics.py`:
```python
import math

import pytest

from relevance_harness.metrics import (
    average_precision, mean_average_precision, mrr_at_k, ndcg_at_k, recall_at_k,
)


# relevances: a list per query of 0/1 in ranked order (1 = doc at that rank is relevant)
def test_mrr_first_relevant_at_rank_2():
    # first relevant is at rank 2 -> 1/2
    assert mrr_at_k([[0, 1, 0, 1]], k=10) == pytest.approx(0.5)


def test_mrr_no_relevant_in_top_k_is_zero():
    assert mrr_at_k([[0, 0, 0, 1]], k=3) == pytest.approx(0.0)


def test_mrr_averages_across_queries():
    # q1: first rel at rank 1 -> 1.0 ; q2: first rel at rank 4 -> 0.25
    assert mrr_at_k([[1, 0], [0, 0, 0, 1]], k=10) == pytest.approx((1.0 + 0.25) / 2)


def test_recall_at_k_counts_relevant_in_top_k():
    # 3 relevant total (n_relevant given separately); 2 of them in top-3
    assert recall_at_k([[1, 0, 1, 1]], k=3, n_relevant=[3]) == pytest.approx(2 / 3)


def test_recall_at_k_zero_relevant_is_zero():
    assert recall_at_k([[0, 0]], k=2, n_relevant=[0]) == pytest.approx(0.0)


def test_ndcg_perfect_ranking_is_one():
    # 2 relevant, both at the top -> DCG == IDCG -> 1.0
    assert ndcg_at_k([[1, 1, 0, 0]], k=4, n_relevant=[2]) == pytest.approx(1.0)


def test_ndcg_known_value():
    # rels at ranks 2 and 3; n_relevant=2
    # DCG = 1/log2(3) + 1/log2(4) = 0.6309298 + 0.5 = 1.1309298
    # IDCG = 1/log2(2) + 1/log2(3) = 1.0 + 0.6309298 = 1.6309298
    dcg = 1 / math.log2(3) + 1 / math.log2(4)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k([[0, 1, 1, 0]], k=4, n_relevant=[2]) == pytest.approx(dcg / idcg)


def test_average_precision_known_value():
    # rels at ranks 1 and 3, n_relevant=2
    # precision@1 = 1/1 = 1.0 ; precision@3 = 2/3
    # AP = (1.0 + 0.6667) / 2 = 0.8333
    assert average_precision([1, 0, 1, 0], n_relevant=2) == pytest.approx((1.0 + 2 / 3) / 2)


def test_map_averages_average_precision():
    ap1 = average_precision([1, 0], n_relevant=1)       # 1.0
    ap2 = average_precision([0, 0, 1], n_relevant=1)    # 1/3
    assert mean_average_precision([[1, 0], [0, 0, 1]], n_relevant=[1, 1]) == pytest.approx(
        (ap1 + ap2) / 2
    )
```

- [ ] **Step 2: Run — expect failure**

Run: `cd benchmarks/relevance && pytest tests/test_metrics.py -v`
Expected: `ImportError` — `relevance_harness.metrics` not found.

- [ ] **Step 3: Implement `metrics.py`**

`benchmarks/relevance/src/relevance_harness/metrics.py`:
```python
"""Ranking metrics from canonical IR definitions. Pure functions, NumPy only.

Each function takes `relevances`: a list (one per query) of 0/1 ints in ranked
order, where a 1 at position i means the doc ranked i-th (1-based) is relevant.
`n_relevant` is the per-query count of relevant docs in the full qrels (needed
for Recall, nDCG ideal-DCG, and MAP — it can exceed the number of 1s present in
the truncated/returned list).
"""
from __future__ import annotations

import math


def _reciprocal_rank(rels: list[int], k: int) -> float:
    for i, r in enumerate(rels[:k], start=1):
        if r:
            return 1.0 / i
    return 0.0


def mrr_at_k(relevances: list[list[int]], *, k: int) -> float:
    """Mean reciprocal rank of the first relevant doc within top-k."""
    if not relevances:
        return 0.0
    return sum(_reciprocal_rank(r, k) for r in relevances) / len(relevances)


def recall_at_k(relevances: list[list[int]], *, k: int, n_relevant: list[int]) -> float:
    """Mean over queries of (relevant retrieved in top-k) / (total relevant)."""
    if not relevances:
        return 0.0
    total = 0.0
    for rels, nrel in zip(relevances, n_relevant):
        if nrel <= 0:
            total += 0.0
            continue
        hits = sum(1 for r in rels[:k] if r)
        total += hits / nrel
    return total / len(relevances)


def _dcg(rels: list[int], k: int) -> float:
    return sum(r / math.log2(i + 1) for i, r in enumerate(rels[:k], start=1))


def ndcg_at_k(relevances: list[list[int]], *, k: int, n_relevant: list[int]) -> float:
    """Mean normalised DCG@k with binary relevance.

    IDCG uses the ideal ranking: min(n_relevant, k) ones at the top.
    """
    if not relevances:
        return 0.0
    total = 0.0
    for rels, nrel in zip(relevances, n_relevant):
        dcg = _dcg(rels, k)
        ideal = [1] * min(nrel, k)
        idcg = _dcg(ideal, k)
        total += (dcg / idcg) if idcg > 0 else 0.0
    return total / len(relevances)


def average_precision(rels: list[int], *, n_relevant: int) -> float:
    """Average precision for a single query.

    AP = (1/n_relevant) * Σ_over_relevant_hits precision@(rank of that hit).
    """
    if n_relevant <= 0:
        return 0.0
    hits = 0
    score = 0.0
    for i, r in enumerate(rels, start=1):
        if r:
            hits += 1
            score += hits / i
    return score / n_relevant


def mean_average_precision(
    relevances: list[list[int]], *, n_relevant: list[int]
) -> float:
    """Mean of per-query average precision."""
    if not relevances:
        return 0.0
    return sum(
        average_precision(rels, n_relevant=nrel)
        for rels, nrel in zip(relevances, n_relevant)
    ) / len(relevances)
```

- [ ] **Step 4: Run — expect pass**

Run: `cd benchmarks/relevance && pytest tests/test_metrics.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/relevance/src/relevance_harness/metrics.py \
        benchmarks/relevance/tests/test_metrics.py
git commit -m "$(cat <<'EOF'
feat(relevance): MRR@k / nDCG@k / Recall@k / MAP from canonical IR definitions

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2.2: Validate metrics against `pytrec_eval` reference (TDD)

**Files:**
- Create: `benchmarks/relevance/tests/test_metrics_vs_pytrec_eval.py`

This is the formula-correctness gate: our pure metrics must agree with the official `trec_eval` binding to `1e-9`. Uses the measure-name strings recorded in Task 0.1 Step 5. If `pytrec_eval` failed to build (Task 0.1), the test self-skips so it never blocks CI.

- [ ] **Step 1: Write the test (and run it as the failing step — it fails until metrics map correctly)**

`benchmarks/relevance/tests/test_metrics_vs_pytrec_eval.py`:
```python
import pytest

pytrec_eval = pytest.importorskip("pytrec_eval")

from relevance_harness.metrics import mrr_at_k, ndcg_at_k, recall_at_k, mean_average_precision


def _rels_and_nrel(qrel: dict, run: dict, query_id: str):
    """Convert a qrel/run pair into (ranked 0/1 list, n_relevant) for one query."""
    relevant = {d for d, g in qrel[query_id].items() if g > 0}
    ranked = sorted(run[query_id].items(), key=lambda kv: kv[1], reverse=True)
    rels = [1 if doc in relevant else 0 for doc, _ in ranked]
    return rels, len(relevant)


def test_matches_pytrec_eval_on_a_few_queries():
    qrel = {
        "q1": {"d1": 1, "d2": 0, "d3": 1, "d4": 0},
        "q2": {"d1": 0, "d2": 1, "d5": 1},
    }
    run = {
        "q1": {"d1": 0.9, "d2": 0.5, "d3": 0.8, "d4": 0.1},
        "q2": {"d2": 0.7, "d1": 0.6, "d5": 0.2},
    }
    measures = {"recip_rank", "ndcg_cut.10", "recall.10", "map"}
    evaluator = pytrec_eval.RelevanceEvaluator(qrel, measures)
    ref = evaluator.evaluate(run)

    ref_mrr = sum(ref[q]["recip_rank"] for q in qrel) / len(qrel)
    ref_ndcg = sum(ref[q]["ndcg_cut_10"] for q in qrel) / len(qrel)
    ref_recall = sum(ref[q]["recall_10"] for q in qrel) / len(qrel)
    ref_map = sum(ref[q]["map"] for q in qrel) / len(qrel)

    relevances, n_relevant = [], []
    for q in sorted(qrel):
        rels, nrel = _rels_and_nrel(qrel, run, q)
        relevances.append(rels)
        n_relevant.append(nrel)

    assert mrr_at_k(relevances, k=10) == pytest.approx(ref_mrr, abs=1e-9)
    assert ndcg_at_k(relevances, k=10, n_relevant=n_relevant) == pytest.approx(ref_ndcg, abs=1e-9)
    assert recall_at_k(relevances, k=10, n_relevant=n_relevant) == pytest.approx(ref_recall, abs=1e-9)
    assert mean_average_precision(relevances, n_relevant=n_relevant) == pytest.approx(ref_map, abs=1e-9)
```

- [ ] **Step 2: Run — expect pass (or skip if pytrec_eval unavailable)**

Run: `cd benchmarks/relevance && pytest tests/test_metrics_vs_pytrec_eval.py -v`
Expected: 1 passed. (If `pytrec_eval` could not be installed per Task 0.1, expected: 1 skipped with reason "could not import 'pytrec_eval'".)

If it FAILS (a real mismatch), the bug is in `metrics.py` — `pytrec_eval` is the source of truth; fix `metrics.py` and re-run both `test_metrics.py` and this test until green.

- [ ] **Step 3: Commit**

```bash
git add benchmarks/relevance/tests/test_metrics_vs_pytrec_eval.py
git commit -m "$(cat <<'EOF'
test(relevance): validate ranking metrics against pytrec_eval reference

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 3 — semantex client + indexing

### Task 3.1: semantex search client with ablation/backend env (TDD)

**Files:**
- Create: `benchmarks/relevance/src/relevance_harness/semantex_client.py`
- Create: `benchmarks/relevance/tests/test_semantex_client.py`

Maps ablations to the REAL flags/env from Task 0.1: `sparse-only`→`--sparse-only`; `dense-only`→`--dense-only`; `hybrid`→neither; `rerank`→hybrid + `--rerank`. Dense backend → `SEMANTEX_DENSE_BACKEND` env. All runs add `--json --no-content -m <k>`.

- [ ] **Step 1: Write failing tests**

`benchmarks/relevance/tests/test_semantex_client.py`:
```python
import json
import subprocess
from unittest.mock import patch

from relevance_harness.semantex_client import SemantexClient, parse_results


SAMPLE_JSON = json.dumps([
    {"file": "auth.py", "start_line": 10, "end_line": 20, "score": 0.42,
     "source": "Hybrid", "chunk_type": "AstNode", "name": "login", "language": "python"},
    {"file": "db.py", "start_line": 1, "end_line": 9, "score": 0.20,
     "source": "Sparse", "chunk_type": "TextWindow"},
])


def test_parse_results_extracts_files_and_doc_ids():
    rr = parse_results("q1", SAMPLE_JSON)
    assert rr.query_id == "q1"
    assert rr.ranked_files == ("auth.py", "db.py")
    # doc id is "file:start-end" — stable, matches how loaders mint ids
    assert rr.ranked_doc_ids == ("auth.py:10-20", "db.py:1-9")


def test_parse_results_empty_array():
    rr = parse_results("q1", "[]")
    assert rr.ranked_doc_ids == ()
    assert rr.ranked_files == ()


def test_search_builds_hybrid_command_and_env():
    client = SemantexClient(semantex_binary="semantex", corpus_dir="/tmp/corpus")
    with patch("subprocess.run") as mr:
        mr.return_value = subprocess.CompletedProcess(args=[], returncode=0,
                                                      stdout=SAMPLE_JSON, stderr="")
        rr = client.search("q1", "auth handler", ablation="hybrid", k=10)
    args = mr.call_args.args[0]
    assert args[0] == "semantex"
    assert "auth handler" in args
    assert "--json" in args and "--no-content" in args
    assert "-m" in args and "10" in args
    # hybrid uses NEITHER dense-only nor sparse-only
    assert "--dense-only" not in args and "--sparse-only" not in args
    assert "--rerank" not in args
    assert mr.call_args.kwargs["cwd"] == "/tmp/corpus"
    assert rr.ranked_files == ("auth.py", "db.py")


def test_search_sparse_only_flag():
    client = SemantexClient(semantex_binary="semantex", corpus_dir="/tmp/c")
    with patch("subprocess.run") as mr:
        mr.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="[]", stderr="")
        client.search("q1", "x", ablation="sparse-only", k=5)
    assert "--sparse-only" in mr.call_args.args[0]


def test_search_dense_only_flag():
    client = SemantexClient(semantex_binary="semantex", corpus_dir="/tmp/c")
    with patch("subprocess.run") as mr:
        mr.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="[]", stderr="")
        client.search("q1", "x", ablation="dense-only", k=5)
    assert "--dense-only" in mr.call_args.args[0]


def test_search_rerank_adds_flag_on_hybrid():
    client = SemantexClient(semantex_binary="semantex", corpus_dir="/tmp/c")
    with patch("subprocess.run") as mr:
        mr.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="[]", stderr="")
        client.search("q1", "x", ablation="rerank", k=5)
    args = mr.call_args.args[0]
    assert "--rerank" in args
    assert "--dense-only" not in args and "--sparse-only" not in args


def test_dense_backend_sets_env():
    client = SemantexClient(
        semantex_binary="semantex", corpus_dir="/tmp/c", dense_backend="coderank-hnsw"
    )
    with patch("subprocess.run") as mr:
        mr.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="[]", stderr="")
        client.search("q1", "x", ablation="hybrid", k=5)
    env = mr.call_args.kwargs["env"]
    assert env["SEMANTEX_DENSE_BACKEND"] == "coderank-hnsw"


def test_failed_search_raises_with_stderr():
    client = SemantexClient(semantex_binary="semantex", corpus_dir="/tmp/c")
    with patch("subprocess.run") as mr:
        mr.return_value = subprocess.CompletedProcess(args=[], returncode=3, stdout="", stderr="boom")
        try:
            client.search("q1", "x", ablation="hybrid", k=5)
            assert False, "expected RuntimeError"
        except RuntimeError as e:
            assert "boom" in str(e)
```

- [ ] **Step 2: Run — expect failure**

Run: `cd benchmarks/relevance && pytest tests/test_semantex_client.py -v`
Expected: `ImportError` — `relevance_harness.semantex_client` not found.

- [ ] **Step 3: Implement `semantex_client.py`**

`benchmarks/relevance/src/relevance_harness/semantex_client.py`:
```python
"""Subprocess wrapper around `semantex search --json`. One job: run a query
under a given ablation/backend and return a normalised RankedResult.

Ablation -> CLI mapping (verified in Task 0.1):
  sparse-only -> --sparse-only
  dense-only  -> --dense-only
  hybrid      -> (neither flag; hybrid is the default)
  rerank      -> hybrid + --rerank
Dense backend -> SEMANTEX_DENSE_BACKEND env var (colbert-plaid | coderank-hnsw).
All runs add: --json --no-content -m <k>.
"""
from __future__ import annotations

import json
import os
import subprocess
from typing import Optional

from .types import RankedResult


_ABLATIONS = {"sparse-only", "dense-only", "hybrid", "rerank"}


def _doc_id(item: dict) -> str:
    """Stable doc id minted from file + line range; loaders mint ids the same way."""
    return f"{item['file']}:{item.get('start_line', 0)}-{item.get('end_line', 0)}"


def parse_results(query_id: str, stdout: str) -> RankedResult:
    """Parse the `semantex search --json` array into a RankedResult."""
    data = json.loads(stdout) if stdout.strip() else []
    doc_ids = tuple(_doc_id(it) for it in data)
    files = tuple(it["file"] for it in data)
    return RankedResult(query_id=query_id, ranked_doc_ids=doc_ids, ranked_files=files)


class SemantexClient:
    def __init__(
        self,
        *,
        semantex_binary: str,
        corpus_dir: str,
        dense_backend: Optional[str] = None,
        timeout_secs: int = 120,
    ):
        self.binary = semantex_binary
        self.corpus_dir = corpus_dir
        self.dense_backend = dense_backend
        self.timeout_secs = timeout_secs

    def _build_args(self, query: str, *, ablation: str, k: int) -> list[str]:
        if ablation not in _ABLATIONS:
            raise ValueError(f"unknown ablation {ablation!r}; expected one of {_ABLATIONS}")
        args = [self.binary, query, "--json", "--no-content", "-m", str(k)]
        if ablation == "sparse-only":
            args.append("--sparse-only")
        elif ablation == "dense-only":
            args.append("--dense-only")
        elif ablation == "rerank":
            args.append("--rerank")
        # hybrid: no extra flag
        return args

    def _build_env(self) -> dict:
        env = os.environ.copy()
        env["SEMANTEX_QUIET_LIMITS"] = "1"
        if self.dense_backend:
            env["SEMANTEX_DENSE_BACKEND"] = self.dense_backend
        return env

    def search(self, query_id: str, query: str, *, ablation: str, k: int) -> RankedResult:
        args = self._build_args(query, ablation=ablation, k=k)
        proc = subprocess.run(
            args,
            cwd=self.corpus_dir,
            capture_output=True,
            text=True,
            timeout=self.timeout_secs,
            env=self._build_env(),
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"semantex search failed (rc={proc.returncode}): "
                f"{proc.stderr.strip() or proc.stdout.strip()}"
            )
        return parse_results(query_id, proc.stdout)
```

- [ ] **Step 4: Run — expect pass**

Run: `cd benchmarks/relevance && pytest tests/test_semantex_client.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/relevance/src/relevance_harness/semantex_client.py \
        benchmarks/relevance/tests/test_semantex_client.py
git commit -m "$(cat <<'EOF'
feat(relevance): semantex --json client with ablation flags + dense-backend env

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3.2: Index locate-or-build (TDD, reuses swe_bench indexer)

**Files:**
- Create: `benchmarks/relevance/src/relevance_harness/indexing.py`
- Create: `benchmarks/relevance/tests/test_indexing.py`

Reuses `swe_bench_harness.indexer.index_repo` (Task 0.1 Step 6 confirmed the contract: completed index ⇔ `.semantex/meta.json` with `chunk_count > 0`).

- [ ] **Step 1: Write failing tests**

`benchmarks/relevance/tests/test_indexing.py`:
```python
import json
from unittest.mock import patch

import pytest

from relevance_harness.indexing import ensure_index, index_is_complete


def _write_meta(corpus_dir, chunk_count):
    sx = corpus_dir / ".semantex"
    sx.mkdir(parents=True, exist_ok=True)
    (sx / "meta.json").write_text(json.dumps({"chunk_count": chunk_count}))


def test_index_is_complete_true_with_positive_chunk_count(tmp_path):
    _write_meta(tmp_path, 42)
    assert index_is_complete(tmp_path) is True


def test_index_is_complete_false_when_missing(tmp_path):
    assert index_is_complete(tmp_path) is False


def test_index_is_complete_false_with_zero_chunks(tmp_path):
    _write_meta(tmp_path, 0)
    assert index_is_complete(tmp_path) is False


def test_ensure_index_skips_when_already_complete(tmp_path):
    _write_meta(tmp_path, 10)
    with patch("relevance_harness.indexing.index_repo") as mr:
        ensure_index(corpus_dir=tmp_path, semantex_binary="semantex")
    mr.assert_not_called()


def test_ensure_index_builds_when_incomplete(tmp_path):
    from swe_bench_harness.indexer import IndexResult
    with patch("relevance_harness.indexing.index_repo") as mr:
        mr.return_value = IndexResult(ok=True, path=tmp_path / ".semantex")
        ensure_index(corpus_dir=tmp_path, semantex_binary="semantex")
    mr.assert_called_once()
    assert mr.call_args.kwargs["repo_path"] == tmp_path


def test_ensure_index_raises_on_failed_build(tmp_path):
    from swe_bench_harness.indexer import IndexResult
    with patch("relevance_harness.indexing.index_repo") as mr:
        mr.return_value = IndexResult(ok=False, path=tmp_path / ".semantex", error="kaboom")
        with pytest.raises(RuntimeError, match="kaboom"):
            ensure_index(corpus_dir=tmp_path, semantex_binary="semantex")
```

- [ ] **Step 2: Run — expect failure**

Run: `cd benchmarks/relevance && pytest tests/test_indexing.py -v`
Expected: `ImportError` — `relevance_harness.indexing` not found.

- [ ] **Step 3: Implement `indexing.py`**

`benchmarks/relevance/src/relevance_harness/indexing.py`:
```python
"""Locate-or-build a semantex index for a corpus directory.

Reuses swe_bench's indexer so the SWE-loc path shares the Phase-A index cache.
A completed index is marked by `.semantex/meta.json` with chunk_count > 0
(same convention as benchmarks/swe_bench/scripts/pre_index.py)."""
from __future__ import annotations

import json
from pathlib import Path

from swe_bench_harness.indexer import index_repo


def index_is_complete(corpus_dir: Path) -> bool:
    meta = Path(corpus_dir) / ".semantex" / "meta.json"
    if not meta.exists():
        return False
    try:
        data = json.loads(meta.read_text())
    except (ValueError, OSError):
        return False
    return int(data.get("chunk_count", 0)) > 0


def ensure_index(
    *, corpus_dir: Path, semantex_binary: str, timeout_secs: int = 7200
) -> Path:
    """Build a semantex index in `corpus_dir` if one isn't already complete.

    Returns the `.semantex` dir path. Raises RuntimeError on a failed build.
    """
    corpus_dir = Path(corpus_dir)
    sx = corpus_dir / ".semantex"
    if index_is_complete(corpus_dir):
        return sx
    result = index_repo(
        repo_path=corpus_dir, semantex_binary=semantex_binary, timeout_secs=timeout_secs
    )
    if not result.ok:
        raise RuntimeError(f"semantex index failed for {corpus_dir}: {result.error}")
    return sx
```

- [ ] **Step 4: Run — expect pass**

Run: `cd benchmarks/relevance && pytest tests/test_indexing.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/relevance/src/relevance_harness/indexing.py \
        benchmarks/relevance/tests/test_indexing.py
git commit -m "$(cat <<'EOF'
feat(relevance): locate-or-build index helper reusing swe_bench indexer

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 4 — Runner

### Task 4.1: Per-corpus runner → per-query ranked positions (TDD)

**Files:**
- Create: `benchmarks/relevance/src/relevance_harness/runner.py`
- Create: `benchmarks/relevance/tests/test_runner.py`
- Create: `benchmarks/relevance/fixtures/tiny_eval_corpus.json`

The runner takes an `EvalCorpus`, ensures it's indexed, runs each query, and produces the `relevances` + `n_relevant` arrays the metrics consume. It supports two matching modes: **doc-id** (exact `file:start-end`, used by CSN/CoIR) and **file** (file-level, used by SWE-loc function/file recall). No metric math here.

- [ ] **Step 1: Write the fixture**

`benchmarks/relevance/fixtures/tiny_eval_corpus.json`:
```json
{
  "name": "tiny",
  "match_mode": "file",
  "queries": [
    {"query_id": "q1", "text": "user login authentication", "gold_files": ["auth.py"]},
    {"query_id": "q2", "text": "database connection pool", "gold_files": ["db.py"]},
    {"query_id": "q3", "text": "string formatting helper", "gold_files": ["util.py"]}
  ]
}
```

- [ ] **Step 2: Write failing tests**

`benchmarks/relevance/tests/test_runner.py`:
```python
from unittest.mock import patch

from relevance_harness.runner import RunOutput, run_corpus
from relevance_harness.types import Document, EvalCorpus, Query, RankedResult


def _corpus(tmp_path, match_mode="doc_id"):
    cdir = tmp_path / "corpus"
    cdir.mkdir()
    return EvalCorpus(
        name="tiny",
        documents=(
            Document(doc_id="auth.py:10-20", text="def login", file_path="auth.py", start_line=10, end_line=20),
            Document(doc_id="db.py:1-9", text="pool", file_path="db.py", start_line=1, end_line=9),
        ),
        queries=(
            Query(query_id="q1", text="login", gold_doc_ids=("auth.py:10-20",)),
            Query(query_id="q2", text="db pool", gold_doc_ids=("db.py:1-9",)),
        ),
        corpus_dir=cdir,
    )


def test_run_corpus_doc_id_match_builds_relevances(tmp_path):
    corpus = _corpus(tmp_path)
    fake = {
        "q1": RankedResult("q1", ("auth.py:10-20", "db.py:1-9")),    # gold at rank 1
        "q2": RankedResult("q2", ("auth.py:10-20", "db.py:1-9")),    # gold at rank 2
    }

    def _search(self, qid, text, *, ablation, k):
        return fake[qid]

    with patch("relevance_harness.runner.SemantexClient.search", _search), \
         patch("relevance_harness.runner.ensure_index"):
        out = run_corpus(corpus, ablation="hybrid", k=10, semantex_binary="semantex")
    assert isinstance(out, RunOutput)
    # q1 gold at rank 1 -> [1, 0]; q2 gold at rank 2 -> [0, 1]
    assert out.relevances == [[1, 0], [0, 1]]
    assert out.n_relevant == [1, 1]


def test_run_corpus_file_match_mode(tmp_path):
    corpus = EvalCorpus(
        name="t",
        documents=(),
        queries=(Query(query_id="q1", text="login", gold_doc_ids=("auth.py",)),),
        corpus_dir=tmp_path,
    )
    fake = RankedResult("q1", ("x.py:1-2", "auth.py:10-20"),
                        ranked_files=("x.py", "auth.py"))
    with patch("relevance_harness.runner.SemantexClient.search", lambda self, q, t, *, ablation, k: fake), \
         patch("relevance_harness.runner.ensure_index"):
        out = run_corpus(corpus, ablation="hybrid", k=10, semantex_binary="semantex",
                         match_mode="file")
    # gold file auth.py first appears at rank 2 -> [0, 1]
    assert out.relevances == [[0, 1]]
    assert out.n_relevant == [1]


def test_run_corpus_calls_ensure_index_once(tmp_path):
    corpus = _corpus(tmp_path)
    with patch("relevance_harness.runner.SemantexClient.search",
               lambda self, q, t, *, ablation, k: RankedResult(q, ())), \
         patch("relevance_harness.runner.ensure_index") as mi:
        run_corpus(corpus, ablation="hybrid", k=10, semantex_binary="semantex")
    mi.assert_called_once()
```

- [ ] **Step 3: Run — expect failure**

Run: `cd benchmarks/relevance && pytest tests/test_runner.py -v`
Expected: `ImportError` — `relevance_harness.runner` not found.

- [ ] **Step 4: Implement `runner.py`**

`benchmarks/relevance/src/relevance_harness/runner.py`:
```python
"""Drive the semantex client over an EvalCorpus and emit the arrays the metrics
consume. No metric math here — just per-query ranked relevance vectors.

match_mode:
  "doc_id" — a result is relevant iff its "file:start-end" id is in gold_doc_ids
             (CSN / CoIR exact-target matching).
  "file"   — a result is relevant iff its file path is in gold_doc_ids
             (SWE-loc file-level / function-level localisation).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .indexing import ensure_index
from .semantex_client import SemantexClient
from .types import EvalCorpus, RankedResult


@dataclass
class RunOutput:
    """Everything metrics.py needs, plus raw per-query results for the report."""
    corpus_name: str
    ablation: str
    relevances: list[list[int]]
    n_relevant: list[int]
    per_query: list[RankedResult]


def _relevance_vector(rr: RankedResult, gold: set[str], *, match_mode: str) -> list[int]:
    if match_mode == "file":
        return [1 if f in gold else 0 for f in rr.ranked_files]
    return [1 if d in gold else 0 for d in rr.ranked_doc_ids]


def run_corpus(
    corpus: EvalCorpus,
    *,
    ablation: str,
    k: int,
    semantex_binary: str,
    dense_backend: Optional[str] = None,
    match_mode: str = "doc_id",
) -> RunOutput:
    if corpus.corpus_dir is None:
        raise ValueError("corpus.corpus_dir must be set to index + search")
    ensure_index(corpus_dir=corpus.corpus_dir, semantex_binary=semantex_binary)

    client = SemantexClient(
        semantex_binary=semantex_binary,
        corpus_dir=str(corpus.corpus_dir),
        dense_backend=dense_backend,
    )

    relevances: list[list[int]] = []
    n_relevant: list[int] = []
    per_query: list[RankedResult] = []
    for q in corpus.queries:
        rr = client.search(q.query_id, q.text, ablation=ablation, k=k)
        gold = set(q.gold_doc_ids)
        relevances.append(_relevance_vector(rr, gold, match_mode=match_mode))
        n_relevant.append(len(gold))
        per_query.append(rr)

    return RunOutput(
        corpus_name=corpus.name,
        ablation=ablation,
        relevances=relevances,
        n_relevant=n_relevant,
        per_query=per_query,
    )
```

- [ ] **Step 5: Run — expect pass**

Run: `cd benchmarks/relevance && pytest tests/test_runner.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add benchmarks/relevance/src/relevance_harness/runner.py \
        benchmarks/relevance/tests/test_runner.py \
        benchmarks/relevance/fixtures/tiny_eval_corpus.json
git commit -m "$(cat <<'EOF'
feat(relevance): corpus runner → per-query relevance vectors (doc-id + file modes)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4.2: End-to-end runner smoke against the tiny fixture corpus (real semantex)

**Files:**
- Create: `benchmarks/relevance/fixtures/tiny_corpus/auth.py`
- Create: `benchmarks/relevance/fixtures/tiny_corpus/db.py`
- Create: `benchmarks/relevance/fixtures/tiny_corpus/util.py`
- Create: `benchmarks/relevance/tests/test_runner_e2e.py`

This is the harness's "runs end-to-end on a tiny fixture in tests" acceptance requirement. It builds a real index on 3 tiny files and runs a real `semantex` search. Gated on the binary existing; self-skips otherwise so unit CI stays hermetic.

- [ ] **Step 1: Write the tiny corpus files**

`benchmarks/relevance/fixtures/tiny_corpus/auth.py`:
```python
def login(username, password):
    """Authenticate a user by username and password, returning a session token."""
    token = _issue_token(username)
    return token


def _issue_token(username):
    return f"token-for-{username}"
```

`benchmarks/relevance/fixtures/tiny_corpus/db.py`:
```python
class ConnectionPool:
    """A pool of reusable database connections."""

    def __init__(self, size):
        self.size = size
        self._connections = []

    def acquire(self):
        return self._connections.pop()
```

`benchmarks/relevance/fixtures/tiny_corpus/util.py`:
```python
def format_currency(amount, symbol="$"):
    """Format a numeric amount as a currency string with a leading symbol."""
    return f"{symbol}{amount:.2f}"
```

- [ ] **Step 2: Write the e2e test**

`benchmarks/relevance/tests/test_runner_e2e.py`:
```python
import shutil

import pytest

from relevance_harness.metrics import mrr_at_k, recall_at_k
from relevance_harness.runner import run_corpus
from relevance_harness.types import EvalCorpus, Query


@pytest.mark.skipif(shutil.which("semantex") is None, reason="semantex binary not on PATH")
def test_tiny_corpus_end_to_end(tmp_path, fixtures_dir):
    # copy the tiny corpus to a writable temp dir (index writes .semantex/ there)
    corpus_dir = tmp_path / "tiny_corpus"
    shutil.copytree(fixtures_dir / "tiny_corpus", corpus_dir)

    corpus = EvalCorpus(
        name="tiny",
        documents=(),
        queries=(
            Query(query_id="q1", text="authenticate a user with a password", gold_doc_ids=("auth.py",)),
            Query(query_id="q2", text="pool of database connections", gold_doc_ids=("db.py",)),
            Query(query_id="q3", text="format a number as currency", gold_doc_ids=("util.py",)),
        ),
        corpus_dir=corpus_dir,
    )
    out = run_corpus(
        corpus, ablation="hybrid", k=10, semantex_binary="semantex", match_mode="file"
    )
    # every query's gold file should appear somewhere in the top-10
    assert recall_at_k(out.relevances, k=10, n_relevant=out.n_relevant) > 0.0
    # and the protocol produces a finite MRR
    assert 0.0 <= mrr_at_k(out.relevances, k=10) <= 1.0
```

- [ ] **Step 3: Run — expect pass (or skip if no binary)**

Run: `cd benchmarks/relevance && source .venv/bin/activate && pytest tests/test_runner_e2e.py -v -s`
Expected: 1 passed (recall > 0 on a tiny well-separated corpus). First run builds the index, so allow up to ~60 s.

If recall is 0 (semantex returned no gold file in top-10 on a clean 3-file corpus), that indicates a real wiring bug in `semantex_client`/`runner` — debug with `superpowers:systematic-debugging` before proceeding (print `out.per_query`).

- [ ] **Step 4: Commit**

```bash
git add benchmarks/relevance/fixtures/tiny_corpus/ \
        benchmarks/relevance/tests/test_runner_e2e.py
git commit -m "$(cat <<'EOF'
test(relevance): end-to-end runner smoke on a tiny real-index corpus

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 5 — Dataset loaders

### Task 5.1: CodeSearchNet loader (TDD)

**Files:**
- Create: `benchmarks/relevance/src/relevance_harness/datasets/csn.py`
- Create: `benchmarks/relevance/config/csn_subset.yaml`
- Create: `benchmarks/relevance/tests/test_csn.py`

Uses the CSN dataset id + field names recorded in Task 0.1 Step 3 (query = `func_documentation_string`, code = `whole_func_string`, id = `func_code_url`). The loader writes each function's code to a file under `corpus_dir` so semantex can index it, mints the doc id as `<relpath>:<start>-<end>` to match the client, and builds queries from the docstrings. Tests inject rows (no network).

- [ ] **Step 1: Write `csn_subset.yaml`**

`benchmarks/relevance/config/csn_subset.yaml`:
```yaml
# CodeSearchNet subset for CPU-feasible relevance runs.
# Each language indexes `corpus_size` functions and evaluates `query_size`
# queries drawn from functions that HAVE a docstring. Seed is fixed; the report
# logs exactly which query ids were kept/dropped.
dataset_id: code_search_net      # confirm/replace per research notes Task 0.1 Step 3
trust_remote_code: true
seed: 20260531
languages:
  - python
  - javascript
  - go
corpus_size: 1000                # functions indexed per language
query_size: 200                  # queries evaluated per language (subset of corpus)
```

- [ ] **Step 2: Write failing tests**

`benchmarks/relevance/tests/test_csn.py`:
```python
from pathlib import Path

from relevance_harness.datasets.csn import build_corpus_from_rows


SAMPLE_ROWS = [
    {
        "func_code_url": "https://github.com/o/r/blob/sha/auth.py#L10-L20",
        "func_path_in_repository": "auth.py",
        "func_name": "login",
        "whole_func_string": "def login(u, p):\n    \"\"\"Authenticate a user.\"\"\"\n    return True\n",
        "func_documentation_string": "Authenticate a user.",
        "language": "python",
    },
    {
        "func_code_url": "https://github.com/o/r/blob/sha/db.py#L1-L9",
        "func_path_in_repository": "db.py",
        "func_name": "pool",
        "whole_func_string": "def pool(n):\n    \"\"\"Create a connection pool.\"\"\"\n    return []\n",
        "func_documentation_string": "Create a connection pool.",
        "language": "python",
    },
    {
        # no docstring -> excluded from queries, still indexable as a doc
        "func_code_url": "https://github.com/o/r/blob/sha/x.py#L1-L3",
        "func_path_in_repository": "x.py",
        "func_name": "x",
        "whole_func_string": "def x():\n    return 1\n",
        "func_documentation_string": "",
        "language": "python",
    },
]


def test_build_corpus_writes_files_and_builds_queries(tmp_path):
    corpus = build_corpus_from_rows(
        rows=SAMPLE_ROWS, language="python", corpus_dir=tmp_path,
        query_size=None, seed=0,
    )
    # 3 documents materialised on disk
    assert len(corpus.documents) == 3
    files = {d.file_path for d in corpus.documents}
    for f in files:
        assert (tmp_path / f).is_file()
    # only the 2 rows WITH docstrings become queries
    assert len(corpus.queries) == 2
    texts = {q.text for q in corpus.queries}
    assert "Authenticate a user." in texts
    assert "" not in texts


def test_query_gold_doc_id_matches_its_document(tmp_path):
    corpus = build_corpus_from_rows(
        rows=SAMPLE_ROWS, language="python", corpus_dir=tmp_path, query_size=None, seed=0,
    )
    doc_ids = {d.doc_id for d in corpus.documents}
    for q in corpus.queries:
        assert len(q.gold_doc_ids) == 1
        assert q.gold_doc_ids[0] in doc_ids


def test_query_subset_is_seeded_and_logged(tmp_path):
    corpus = build_corpus_from_rows(
        rows=SAMPLE_ROWS, language="python", corpus_dir=tmp_path, query_size=1, seed=42,
    )
    assert len(corpus.queries) == 1
```

- [ ] **Step 3: Run — expect failure**

Run: `cd benchmarks/relevance && pytest tests/test_csn.py -v`
Expected: `ImportError` — `relevance_harness.datasets.csn` not found.

- [ ] **Step 4: Implement `csn.py`**

`benchmarks/relevance/src/relevance_harness/datasets/csn.py`:
```python
"""CodeSearchNet loader → EvalCorpus.

Each function's code is written to a file under corpus_dir so semantex can index
it; the doc id is "<relpath>:1-<nlines>" to match SemantexClient's id minting.
Queries are the functions' docstrings (functions without a docstring are indexed
but not queried). Field names per research notes (Task 0.1 Step 3).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from ..subset import select_queries
from ..types import Document, EvalCorpus, Query


def _safe_relpath(func_code_url: str, fallback_path: str, idx: int) -> str:
    """A unique, filesystem-safe relative path for one function's file."""
    base = fallback_path or f"func_{idx}"
    base = re.sub(r"[^A-Za-z0-9_./-]", "_", base)
    # disambiguate by index so two funcs in the same source path don't collide
    stem, dot, ext = base.rpartition(".")
    if dot:
        return f"{stem}__{idx}.{ext}"
    return f"{base}__{idx}.txt"


def build_corpus_from_rows(
    *,
    rows: list[dict],
    language: str,
    corpus_dir: Path,
    query_size: Optional[int],
    seed: int,
) -> EvalCorpus:
    corpus_dir = Path(corpus_dir)
    corpus_dir.mkdir(parents=True, exist_ok=True)

    documents: list[Document] = []
    candidate_queries: list[dict] = []
    for idx, r in enumerate(rows):
        code = r["whole_func_string"]
        relpath = _safe_relpath(r.get("func_code_url", ""), r.get("func_path_in_repository", ""), idx)
        dest = corpus_dir / relpath
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(code)
        nlines = max(1, code.count("\n") + 1)
        doc_id = f"{relpath}:1-{nlines}"
        documents.append(
            Document(doc_id=doc_id, text=code, file_path=relpath, start_line=1, end_line=nlines)
        )
        doc = (r.get("func_documentation_string") or "").strip()
        if doc:
            candidate_queries.append(
                {"query_id": r["func_code_url"], "text": doc, "gold_doc_id": doc_id}
            )

    kept, _manifest = select_queries(
        candidate_queries, n=query_size, seed=seed, dataset=f"csn/{language}"
    )
    queries = tuple(
        Query(query_id=q["query_id"], text=q["text"], gold_doc_ids=(q["gold_doc_id"],))
        for q in kept
    )
    return EvalCorpus(
        name=f"csn/{language}",
        documents=tuple(documents),
        queries=queries,
        corpus_dir=corpus_dir,
    )


def load_csn_corpus(
    *,
    language: str,
    corpus_dir: Path,
    dataset_id: str,
    corpus_size: Optional[int],
    query_size: Optional[int],
    seed: int,
    trust_remote_code: bool = True,
) -> EvalCorpus:
    """Load CodeSearchNet for one language from HuggingFace, materialise, subset.

    `corpus_size` caps the number of indexed functions (None = all of the test
    split). Selection of the corpus slice is the first `corpus_size` rows after
    sorting by func_code_url (deterministic)."""
    from datasets import load_dataset

    ds = load_dataset(dataset_id, language, split="test", trust_remote_code=trust_remote_code)
    rows = sorted(list(ds), key=lambda r: r["func_code_url"])
    if corpus_size is not None:
        rows = rows[:corpus_size]
    return build_corpus_from_rows(
        rows=rows, language=language, corpus_dir=corpus_dir,
        query_size=query_size, seed=seed,
    )
```

- [ ] **Step 5: Run — expect pass**

Run: `cd benchmarks/relevance && pytest tests/test_csn.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add benchmarks/relevance/src/relevance_harness/datasets/csn.py \
        benchmarks/relevance/config/csn_subset.yaml \
        benchmarks/relevance/tests/test_csn.py
git commit -m "$(cat <<'EOF'
feat(relevance): CodeSearchNet loader (materialise funcs, docstring queries)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5.2: SWE-bench localization loader (TDD, reuses swe_bench dataset)

**Files:**
- Create: `benchmarks/relevance/src/relevance_harness/datasets/swe_loc.py`
- Create: `benchmarks/relevance/fixtures/sample_verified_subset.json`
- Create: `benchmarks/relevance/tests/test_swe_loc.py`

Derives gold files (and changed function/symbol hunks) from each Verified instance's `patch`. Per Task 0.1 Step 6, `swe_bench_harness.dataset.Instance` lacks `patch`, so this loader reads the raw HF/local rows directly. The corpus_dir is the checked-out repo at `$SWE_BENCH_REPO_CACHE/<instance_id>/` — reusing the Phase-A cache. Each instance becomes ONE query (the `problem_statement`) whose gold set is the patched files; matching is file-level (`match_mode="file"`).

- [ ] **Step 1: Write the fixture (3 Verified rows WITH `patch`)**

`benchmarks/relevance/fixtures/sample_verified_subset.json`:
```json
[
  {
    "instance_id": "demo__demo-1",
    "repo": "demo/demo",
    "base_commit": "0000000000000000000000000000000000000001",
    "problem_statement": "Login fails when the password contains a unicode character.",
    "patch": "diff --git a/src/auth.py b/src/auth.py\n--- a/src/auth.py\n+++ b/src/auth.py\n@@ -10,7 +10,7 @@ def login(username, password):\n-    return check(password)\n+    return check(password.encode('utf-8'))\n"
  },
  {
    "instance_id": "demo__demo-2",
    "repo": "demo/demo",
    "base_commit": "0000000000000000000000000000000000000002",
    "problem_statement": "Connection pool leaks connections on error.",
    "patch": "diff --git a/src/db/pool.py b/src/db/pool.py\n--- a/src/db/pool.py\n+++ b/src/db/pool.py\n@@ -20,6 +20,8 @@ class ConnectionPool:\n     def acquire(self):\n+        if not self._connections:\n+            raise PoolEmpty()\n         return self._connections.pop()\n"
  },
  {
    "instance_id": "demo__demo-3",
    "repo": "demo/demo",
    "base_commit": "0000000000000000000000000000000000000003",
    "problem_statement": "Two files change in this fix.",
    "patch": "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1,1 +1,1 @@\n-x = 1\n+x = 2\ndiff --git a/pkg/b.py b/pkg/b.py\n--- a/pkg/b.py\n+++ b/pkg/b.py\n@@ -1,1 +1,1 @@\n-y = 1\n+y = 2\n"
  }
]
```

- [ ] **Step 2: Write failing tests**

`benchmarks/relevance/tests/test_swe_loc.py`:
```python
import json
from pathlib import Path

from relevance_harness.datasets.swe_loc import (
    changed_files_from_patch, load_swe_loc_queries,
)


def test_changed_files_from_single_file_patch():
    patch = (
        "diff --git a/src/auth.py b/src/auth.py\n"
        "--- a/src/auth.py\n+++ b/src/auth.py\n"
        "@@ -10,7 +10,7 @@ def login(username, password):\n"
        "-    return check(password)\n"
        "+    return check(password.encode('utf-8'))\n"
    )
    assert changed_files_from_patch(patch) == ["src/auth.py"]


def test_changed_files_from_multi_file_patch():
    patch = (
        "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x\n+y\n"
        "diff --git a/pkg/b.py b/pkg/b.py\n--- a/pkg/b.py\n+++ b/pkg/b.py\n@@ -1 +1 @@\n-1\n+2\n"
    )
    assert changed_files_from_patch(patch) == ["a.py", "pkg/b.py"]


def test_changed_files_ignores_dev_null_for_new_files():
    # a newly-added file: the b/ side is the gold path, a/ side is /dev/null
    patch = (
        "diff --git a/new.py b/new.py\n--- /dev/null\n+++ b/new.py\n@@ -0,0 +1 @@\n+x = 1\n"
    )
    assert changed_files_from_patch(patch) == ["new.py"]


def test_load_swe_loc_queries_from_local_fixture(fixtures_dir):
    queries = load_swe_loc_queries(local_path=fixtures_dir / "sample_verified_subset.json")
    assert len(queries) == 3
    q1 = next(q for q in queries if q.query_id == "demo__demo-1")
    assert "unicode" in q1.text.lower()
    assert q1.gold_doc_ids == ("src/auth.py",)
    q3 = next(q for q in queries if q.query_id == "demo__demo-3")
    assert set(q3.gold_doc_ids) == {"a.py", "pkg/b.py"}
```

- [ ] **Step 3: Run — expect failure**

Run: `cd benchmarks/relevance && pytest tests/test_swe_loc.py -v`
Expected: `ImportError` — `relevance_harness.datasets.swe_loc` not found.

- [ ] **Step 4: Implement `swe_loc.py`**

`benchmarks/relevance/src/relevance_harness/datasets/swe_loc.py`:
```python
"""SWE-bench localization loader.

Gold = the set of files changed by an instance's gold `patch`. Each Verified
instance becomes one query (its problem_statement); matching is file-level.
The corpus is the checked-out repo at $SWE_BENCH_REPO_CACHE/<instance_id>/,
shared with benchmarks/swe_bench's Phase-A cache.

Note: swe_bench_harness.dataset.Instance carries no `patch` field, so this
loader reads the raw HF rows / local JSON directly (Task 0.1 Step 6).
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional

from ..indexing import ensure_index
from ..types import EvalCorpus, Query


# Matches the "+++ b/<path>" header line of each file in a unified diff.
_PLUS_FILE_RE = re.compile(r"^\+\+\+ (?:b/)?(.+?)\s*$", re.MULTILINE)


def changed_files_from_patch(patch: str) -> list[str]:
    """Files touched by a unified-diff patch, in first-seen order, deduped.

    Uses the "+++ " header (post-image path). Skips /dev/null (deletions);
    new files take their b/ path. Strips a leading 'b/'.
    """
    files: list[str] = []
    for m in _PLUS_FILE_RE.finditer(patch):
        path = m.group(1).strip()
        if path == "/dev/null":
            continue
        if path not in files:
            files.append(path)
    return files


def _repo_cache_dir() -> Path:
    return Path(os.environ.get("SWE_BENCH_REPO_CACHE", Path.home() / ".swe_bench_repos"))


def load_swe_loc_queries(
    *,
    local_path: Optional[Path] = None,
    instance_ids: Optional[set[str]] = None,
) -> list[Query]:
    """Load Verified rows (local JSON or HF) → one file-localisation Query each.

    `instance_ids`, if given, restricts to those ids (e.g. the Phase-A 100).
    """
    if local_path is not None:
        rows = json.loads(Path(local_path).read_text())
    else:
        from datasets import load_dataset
        ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
        rows = list(ds)

    queries: list[Query] = []
    for r in rows:
        if instance_ids is not None and r["instance_id"] not in instance_ids:
            continue
        gold = tuple(changed_files_from_patch(r["patch"]))
        if not gold:
            continue
        queries.append(
            Query(query_id=r["instance_id"], text=r["problem_statement"], gold_doc_ids=gold)
        )
    return queries


def build_swe_loc_corpus(
    *,
    instance_id: str,
    query: Query,
    semantex_binary: str,
) -> EvalCorpus:
    """Wrap one already-checked-out, indexed instance repo as a single-query corpus.

    The repo must already exist under $SWE_BENCH_REPO_CACHE/<instance_id>/ (run
    benchmarks/swe_bench/scripts/pre_index.py first, or ensure_index builds it).
    """
    corpus_dir = _repo_cache_dir() / instance_id
    if not corpus_dir.is_dir():
        raise FileNotFoundError(
            f"repo for {instance_id} not found at {corpus_dir}; "
            f"run benchmarks/swe_bench/scripts/pre_index.py first"
        )
    ensure_index(corpus_dir=corpus_dir, semantex_binary=semantex_binary)
    return EvalCorpus(
        name=f"swe-loc/{instance_id}",
        documents=(),
        queries=(query,),
        corpus_dir=corpus_dir,
    )
```

- [ ] **Step 5: Run — expect pass**

Run: `cd benchmarks/relevance && pytest tests/test_swe_loc.py -v`
Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add benchmarks/relevance/src/relevance_harness/datasets/swe_loc.py \
        benchmarks/relevance/fixtures/sample_verified_subset.json \
        benchmarks/relevance/tests/test_swe_loc.py
git commit -m "$(cat <<'EOF'
feat(relevance): SWE-loc loader — gold files from Verified patches (file-level)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5.3: CoIR loader (TDD)

**Files:**
- Create: `benchmarks/relevance/src/relevance_harness/datasets/coir.py`
- Create: `benchmarks/relevance/config/coir_subset.yaml`
- Create: `benchmarks/relevance/tests/test_coir.py`

CoIR ships as `(corpus, queries, qrels)` triples (Task 0.1 Step 4). The loader materialises each corpus document to a file, builds queries from the queries split, and wires gold doc ids from the qrels mapping. Tests inject the three splits as in-memory lists (no network). If Task 0.1 marked CoIR deferred (no HF access), still build the loader from the recorded schema and ship the injectable `build_corpus_from_splits` — the network path stays untested until run on a machine with HF access.

- [ ] **Step 1: Write `coir_subset.yaml`**

`benchmarks/relevance/config/coir_subset.yaml`:
```yaml
# CoIR (headline) subset — CPU-feasible sub-datasets + sample sizes.
# Replace the *_id values with the real HF ids recorded in research notes
# (Task 0.1 Step 4). Each report logs which query ids were kept/dropped.
seed: 20260531
query_size: 200          # queries evaluated per sub-dataset (subset)
corpus_size: 5000        # corpus docs materialised + indexed per sub-dataset
subdatasets:
  - name: codesearchnet
    queries_id: CoIR-Retrieval/CodeSearchNet-queries     # confirm in research notes
    corpus_id: CoIR-Retrieval/CodeSearchNet-corpus       # confirm in research notes
    qrels_id: CoIR-Retrieval/CodeSearchNet-qrels         # confirm in research notes
  - name: cosqa
    queries_id: CoIR-Retrieval/cosqa-queries             # confirm in research notes
    corpus_id: CoIR-Retrieval/cosqa-corpus               # confirm in research notes
    qrels_id: CoIR-Retrieval/cosqa-qrels                 # confirm in research notes
```

- [ ] **Step 2: Write failing tests**

`benchmarks/relevance/tests/test_coir.py`:
```python
from relevance_harness.datasets.coir import build_corpus_from_splits


CORPUS = [
    {"_id": "d1", "text": "def login(u, p): return authenticate(u, p)"},
    {"_id": "d2", "text": "class ConnectionPool: ..."},
    {"_id": "d3", "text": "def format_currency(x): ..."},
]
QUERIES = [
    {"_id": "q1", "text": "authenticate a user"},
    {"_id": "q2", "text": "database connection pool"},
]
# qrels: TREC-style rows mapping query-id -> corpus-id -> relevance
QRELS = [
    {"query-id": "q1", "corpus-id": "d1", "score": 1},
    {"query-id": "q2", "corpus-id": "d2", "score": 1},
]


def test_build_corpus_materialises_docs_and_wires_qrels(tmp_path):
    corpus = build_corpus_from_splits(
        name="cosqa", corpus_rows=CORPUS, query_rows=QUERIES, qrel_rows=QRELS,
        corpus_dir=tmp_path, corpus_size=None, query_size=None, seed=0,
    )
    assert corpus.name == "coir/cosqa"
    assert len(corpus.documents) == 3
    # every corpus doc is on disk
    for d in corpus.documents:
        assert (tmp_path / d.file_path).is_file()
    # 2 queries, each with the gold doc id taken from qrels
    qrels = corpus.qrels()
    assert "q1" in qrels and "q2" in qrels
    q1 = next(q for q in corpus.queries if q.query_id == "q1")
    # gold doc id is the materialised doc id whose source _id == d1
    d1 = next(d for d in corpus.documents if d.doc_id.startswith("d1"))
    assert q1.gold_doc_ids == (d1.doc_id,)


def test_query_subset_seeded(tmp_path):
    corpus = build_corpus_from_splits(
        name="cosqa", corpus_rows=CORPUS, query_rows=QUERIES, qrel_rows=QRELS,
        corpus_dir=tmp_path, corpus_size=None, query_size=1, seed=42,
    )
    assert len(corpus.queries) == 1


def test_queries_without_qrels_are_dropped(tmp_path):
    queries = QUERIES + [{"_id": "q_orphan", "text": "no gold for me"}]
    corpus = build_corpus_from_splits(
        name="cosqa", corpus_rows=CORPUS, query_rows=queries, qrel_rows=QRELS,
        corpus_dir=tmp_path, corpus_size=None, query_size=None, seed=0,
    )
    ids = {q.query_id for q in corpus.queries}
    assert "q_orphan" not in ids
```

- [ ] **Step 3: Run — expect failure**

Run: `cd benchmarks/relevance && pytest tests/test_coir.py -v`
Expected: `ImportError` — `relevance_harness.datasets.coir` not found.

- [ ] **Step 4: Implement `coir.py`**

`benchmarks/relevance/src/relevance_harness/datasets/coir.py`:
```python
"""CoIR loader → EvalCorpus.

CoIR sub-datasets ship as three splits: corpus (docs), queries, qrels (TREC-style
query-id/corpus-id/score rows). We materialise each corpus doc to a file under
corpus_dir, mint a doc id "<sourceid>__<relpath>:1-<nlines>", build queries from
the queries split, and attach gold doc ids via the qrels mapping. Queries with no
qrels entry are dropped (logged via the subset manifest size). HF column names
per research notes (Task 0.1 Step 4); injectable splits keep this unit-testable.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from ..subset import select_queries
from ..types import Document, EvalCorpus, Query


def _qrels_map(qrel_rows: list[dict]) -> dict[str, list[str]]:
    """{query-id: [corpus-id, ...]} for rows with positive relevance."""
    out: dict[str, list[str]] = {}
    for r in qrel_rows:
        if int(r.get("score", 0)) <= 0:
            continue
        out.setdefault(r["query-id"], []).append(r["corpus-id"])
    return out


def build_corpus_from_splits(
    *,
    name: str,
    corpus_rows: list[dict],
    query_rows: list[dict],
    qrel_rows: list[dict],
    corpus_dir: Path,
    corpus_size: Optional[int],
    query_size: Optional[int],
    seed: int,
) -> EvalCorpus:
    corpus_dir = Path(corpus_dir)
    corpus_dir.mkdir(parents=True, exist_ok=True)

    rows = sorted(corpus_rows, key=lambda r: r["_id"])
    if corpus_size is not None:
        rows = rows[:corpus_size]

    documents: list[Document] = []
    source_to_docid: dict[str, str] = {}
    for idx, r in enumerate(rows):
        text = r["text"]
        relpath = f"doc_{idx}.txt"
        (corpus_dir / relpath).write_text(text)
        nlines = max(1, text.count("\n") + 1)
        doc_id = f"{r['_id']}__{relpath}:1-{nlines}"
        source_to_docid[r["_id"]] = doc_id
        documents.append(
            Document(doc_id=doc_id, text=text, file_path=relpath, start_line=1, end_line=nlines)
        )

    qmap = _qrels_map(qrel_rows)
    candidate_queries: list[dict] = []
    for r in query_rows:
        gold_source_ids = qmap.get(r["_id"], [])
        gold_doc_ids = [source_to_docid[g] for g in gold_source_ids if g in source_to_docid]
        if not gold_doc_ids:
            continue
        candidate_queries.append(
            {"query_id": r["_id"], "text": r["text"], "gold_doc_ids": tuple(gold_doc_ids)}
        )

    kept, _manifest = select_queries(
        candidate_queries, n=query_size, seed=seed, dataset=f"coir/{name}"
    )
    queries = tuple(
        Query(query_id=q["query_id"], text=q["text"], gold_doc_ids=q["gold_doc_ids"])
        for q in kept
    )
    return EvalCorpus(
        name=f"coir/{name}",
        documents=tuple(documents),
        queries=queries,
        corpus_dir=corpus_dir,
    )


def load_coir_subdataset(
    *,
    name: str,
    queries_id: str,
    corpus_id: str,
    qrels_id: str,
    corpus_dir: Path,
    corpus_size: Optional[int],
    query_size: Optional[int],
    seed: int,
) -> EvalCorpus:
    """Load one CoIR sub-dataset from HuggingFace, materialise, subset.

    HF split/column names per research notes (Task 0.1 Step 4). If the org layout
    differs, adjust the three load_dataset calls and the row-key access here.
    """
    from datasets import load_dataset

    corpus_rows = list(load_dataset(corpus_id, split="corpus"))
    query_rows = list(load_dataset(queries_id, split="queries"))
    qrel_rows = list(load_dataset(qrels_id, split="test"))
    return build_corpus_from_splits(
        name=name, corpus_rows=corpus_rows, query_rows=query_rows, qrel_rows=qrel_rows,
        corpus_dir=corpus_dir, corpus_size=corpus_size, query_size=query_size, seed=seed,
    )
```

- [ ] **Step 5: Run — expect pass**

Run: `cd benchmarks/relevance && pytest tests/test_coir.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add benchmarks/relevance/src/relevance_harness/datasets/coir.py \
        benchmarks/relevance/config/coir_subset.yaml \
        benchmarks/relevance/tests/test_coir.py
git commit -m "$(cat <<'EOF'
feat(relevance): CoIR loader (corpus/queries/qrels → EvalCorpus, injectable)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 6 — Report

### Task 6.1: Report — metrics table + manifest + stamps (TDD)

**Files:**
- Create: `benchmarks/relevance/src/relevance_harness/report.py`
- Create: `benchmarks/relevance/tests/test_report.py`

Turns one-or-more `RunOutput`s into a metrics row per (dataset, ablation, backend), plus a `report.md` and `report.json`. The JSON includes the subset manifest(s) and a reproducibility stamp: git rev + dense backend + model id (from env) + k.

- [ ] **Step 1: Write failing tests**

`benchmarks/relevance/tests/test_report.py`:
```python
import json

from relevance_harness.report import (
    ReproStamp, compute_metrics_row, render_report_json, render_report_md,
)
from relevance_harness.runner import RunOutput


def _run(name="csn/python", ablation="hybrid"):
    # q1 gold at rank 1, q2 gold at rank 2
    return RunOutput(
        corpus_name=name, ablation=ablation,
        relevances=[[1, 0], [0, 1]], n_relevant=[1, 1], per_query=[],
    )


def test_compute_metrics_row_has_all_metrics():
    row = compute_metrics_row(_run(), k=10)
    assert row["dataset"] == "csn/python"
    assert row["ablation"] == "hybrid"
    assert row["n_queries"] == 2
    for key in ("mrr_at_10", "ndcg_at_10", "recall_at_1", "recall_at_5", "recall_at_10", "map"):
        assert key in row
    # mrr = (1/1 + 1/2)/2 = 0.75
    assert abs(row["mrr_at_10"] - 0.75) < 1e-9
    # recall@1 = (1 + 0)/2 = 0.5 ; recall@10 = 1.0
    assert abs(row["recall_at_1"] - 0.5) < 1e-9
    assert abs(row["recall_at_10"] - 1.0) < 1e-9


def test_report_json_includes_stamp_and_manifest(tmp_path):
    rows = [compute_metrics_row(_run(), k=10)]
    stamp = ReproStamp(git_rev="abc123", dense_backend="coderank-hnsw",
                       model_id="CodeRankEmbed", k=10)
    manifests = [{"dataset": "csn/python", "total": 1000, "selected": 200,
                  "seed": 20260531, "kept_ids": ["q1"], "dropped_ids": []}]
    out = render_report_json(rows=rows, stamp=stamp, manifests=manifests)
    data = json.loads(out)
    assert data["stamp"]["git_rev"] == "abc123"
    assert data["stamp"]["dense_backend"] == "coderank-hnsw"
    assert data["rows"][0]["dataset"] == "csn/python"
    assert data["manifests"][0]["selected"] == 200


def test_report_md_contains_table_and_backend():
    rows = [compute_metrics_row(_run(), k=10)]
    stamp = ReproStamp(git_rev="abc123", dense_backend="colbert-plaid",
                       model_id="colbert", k=10)
    md = render_report_md(rows=rows, stamp=stamp, manifests=[])
    assert "csn/python" in md
    assert "colbert-plaid" in md
    assert "mrr_at_10" in md or "MRR@10" in md
```

- [ ] **Step 2: Run — expect failure**

Run: `cd benchmarks/relevance && pytest tests/test_report.py -v`
Expected: `ImportError` — `relevance_harness.report` not found.

- [ ] **Step 3: Implement `report.py`**

`benchmarks/relevance/src/relevance_harness/report.py`:
```python
"""Aggregate RunOutputs into a metrics table + report.md / report.json.

Every report carries a reproducibility stamp (git rev, dense backend, model id,
k) and the subset manifest(s), so a number is never reported without provenance.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass

import pandas as pd

from .metrics import mean_average_precision, mrr_at_k, ndcg_at_k, recall_at_k
from .runner import RunOutput


@dataclass(frozen=True)
class ReproStamp:
    git_rev: str
    dense_backend: str
    model_id: str
    k: int


def current_git_rev() -> str:
    """Short git rev of the semantex repo, or 'unknown' if not a git checkout."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def compute_metrics_row(run: RunOutput, *, k: int) -> dict:
    rels = run.relevances
    nrel = run.n_relevant
    return {
        "dataset": run.corpus_name,
        "ablation": run.ablation,
        "n_queries": len(rels),
        "mrr_at_10": mrr_at_k(rels, k=10),
        "ndcg_at_10": ndcg_at_k(rels, k=10, n_relevant=nrel),
        "recall_at_1": recall_at_k(rels, k=1, n_relevant=nrel),
        "recall_at_5": recall_at_k(rels, k=5, n_relevant=nrel),
        "recall_at_10": recall_at_k(rels, k=10, n_relevant=nrel),
        "map": mean_average_precision(rels, n_relevant=nrel),
    }


def render_report_json(*, rows: list[dict], stamp: ReproStamp, manifests: list[dict]) -> str:
    return json.dumps(
        {"stamp": asdict(stamp), "rows": rows, "manifests": manifests}, indent=2
    )


def render_report_md(*, rows: list[dict], stamp: ReproStamp, manifests: list[dict]) -> str:
    lines: list[str] = []
    lines.append("# semantex Relevance Report\n\n")
    lines.append(
        f"- **git rev:** `{stamp.git_rev}`\n"
        f"- **dense backend:** `{stamp.dense_backend}`\n"
        f"- **model:** `{stamp.model_id}`\n"
        f"- **cutoff k:** {stamp.k}\n\n"
    )
    lines.append("## Metrics\n\n")
    df = pd.DataFrame(rows)
    lines.append(df.to_markdown(index=False, floatfmt=".4f") + "\n\n")
    if manifests:
        lines.append("## Subset manifest\n\n")
        for m in manifests:
            lines.append(
                f"- **{m['dataset']}**: kept {m['selected']}/{m['total']} "
                f"(seed {m['seed']}, dropped {len(m.get('dropped_ids', []))})\n"
            )
    return "".join(lines)
```

- [ ] **Step 4: Run — expect pass**

Run: `cd benchmarks/relevance && pytest tests/test_report.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/relevance/src/relevance_harness/report.py \
        benchmarks/relevance/tests/test_report.py
git commit -m "$(cat <<'EOF'
feat(relevance): report — metrics table + subset manifest + git/model stamp

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 7 — Entrypoints

### Task 7.1: `run.py` main entrypoint

**Files:**
- Create: `benchmarks/relevance/scripts/run.py`

Wires loaders → runner → report. Exposes `--dataset {csn,coir,swe-loc}`, `--ablation {sparse-only,dense-only,hybrid,rerank}`, `--dense-backend {colbert-plaid,coderank-hnsw}`, `--k`, `--run-id`. Writes `results/<run-id>/report.md` + `report.json`. No new test (it's glue over tested modules); smoke-tested in Step 2.

- [ ] **Step 1: Write the script**

`benchmarks/relevance/scripts/run.py`:
```python
"""Main relevance entrypoint: load a dataset, run an ablation, write a report.

Examples:
  python -m scripts.run --dataset csn --ablation hybrid
  python -m scripts.run --dataset csn --ablation dense-only --dense-backend coderank-hnsw
  python -m scripts.run --dataset swe-loc --ablation hybrid
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import click
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
# make the sibling swe_bench harness importable for SWE-loc reuse
SWE_BENCH_SRC = ROOT.parent / "swe_bench" / "src"
if SWE_BENCH_SRC.is_dir():
    sys.path.insert(0, str(SWE_BENCH_SRC))

from relevance_harness.datasets.csn import load_csn_corpus
from relevance_harness.datasets.swe_loc import build_swe_loc_corpus, load_swe_loc_queries
from relevance_harness.report import (
    ReproStamp, compute_metrics_row, current_git_rev, render_report_json, render_report_md,
)
from relevance_harness.runner import run_corpus

CONFIG = ROOT / "config"
RESULTS = ROOT / "results"


def _phase_a_ids() -> set[str] | None:
    f = ROOT.parent / "swe_bench" / "config" / "instances_phase_a.txt"
    if not f.is_file():
        return None
    return {ln.strip() for ln in f.read_text().splitlines() if ln.strip()}


@click.command()
@click.option("--dataset", type=click.Choice(["csn", "coir", "swe-loc"]), required=True)
@click.option("--ablation", type=click.Choice(["sparse-only", "dense-only", "hybrid", "rerank"]),
              default="hybrid", show_default=True)
@click.option("--dense-backend", default="", help="colbert-plaid | coderank-hnsw (env override)")
@click.option("--k", default=10, type=int, show_default=True)
@click.option("--run-id", default="", help="reuse a results/<run-id> dir")
@click.option("--semantex-bin", default=os.environ.get("SEMANTEX_BINARY", "semantex"))
def main(dataset, ablation, dense_backend, k, run_id, semantex_bin):
    backend = dense_backend or None
    if not run_id:
        run_id = time.strftime(f"%Y%m%d-%H%M%S-{dataset}-{ablation}")
    out_dir = RESULTS / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus_root = out_dir / "corpora"

    rows: list[dict] = []
    manifests: list[dict] = []
    match_mode = "doc_id"

    if dataset == "csn":
        cfg = yaml.safe_load((CONFIG / "csn_subset.yaml").read_text())
        for lang in cfg["languages"]:
            corpus = load_csn_corpus(
                language=lang, corpus_dir=corpus_root / f"csn_{lang}",
                dataset_id=cfg["dataset_id"], corpus_size=cfg["corpus_size"],
                query_size=cfg["query_size"], seed=cfg["seed"],
                trust_remote_code=cfg.get("trust_remote_code", True),
            )
            out = run_corpus(corpus, ablation=ablation, k=k, semantex_binary=semantex_bin,
                             dense_backend=backend, match_mode="doc_id")
            rows.append(compute_metrics_row(out, k=k))
    elif dataset == "swe-loc":
        match_mode = "file"
        queries = load_swe_loc_queries(instance_ids=_phase_a_ids())
        agg_rel: list[list[int]] = []
        agg_nrel: list[int] = []
        for q in queries:
            try:
                corpus = build_swe_loc_corpus(
                    instance_id=q.query_id, query=q, semantex_binary=semantex_bin
                )
            except FileNotFoundError as e:
                click.echo(f"skip {q.query_id}: {e}", err=True)
                continue
            out = run_corpus(corpus, ablation=ablation, k=k, semantex_binary=semantex_bin,
                             dense_backend=backend, match_mode="file")
            agg_rel.extend(out.relevances)
            agg_nrel.extend(out.n_relevant)
        from relevance_harness.runner import RunOutput
        rows.append(compute_metrics_row(
            RunOutput(corpus_name="swe-loc", ablation=ablation,
                      relevances=agg_rel, n_relevant=agg_nrel, per_query=[]),
            k=k,
        ))
    else:  # coir
        click.echo("CoIR run: fill config/coir_subset.yaml with real HF ids "
                   "(research notes Task 0.1 Step 4), then wire load_coir_subdataset here.",
                   err=True)
        raise SystemExit(2)

    stamp = ReproStamp(
        git_rev=current_git_rev(),
        dense_backend=backend or os.environ.get("SEMANTEX_DENSE_BACKEND", "default"),
        model_id=os.environ.get("SEMANTEX_LLM_MODEL", "n/a-dense-path"),
        k=k,
    )
    (out_dir / "report.json").write_text(render_report_json(rows=rows, stamp=stamp, manifests=manifests))
    (out_dir / "report.md").write_text(render_report_md(rows=rows, stamp=stamp, manifests=manifests))
    click.echo(f"Report: {out_dir / 'report.md'}")
    click.echo((out_dir / "report.md").read_text())


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test against CSN python with a tiny subset (real network + index)**

Temporarily shrink the subset to keep the smoke fast, run, then restore:
```bash
cd benchmarks/relevance && source .venv/bin/activate
cp config/csn_subset.yaml /tmp/csn_subset.bak
python - <<'PY'
import yaml, pathlib
p = pathlib.Path("config/csn_subset.yaml")
c = yaml.safe_load(p.read_text())
c["languages"] = ["python"]; c["corpus_size"] = 50; c["query_size"] = 10
p.write_text(yaml.safe_dump(c))
PY
python -m scripts.run --dataset csn --ablation hybrid --k 10
cp /tmp/csn_subset.bak config/csn_subset.yaml
```
Expected: prints a `report.md` table with one `csn/python` row, all six metric columns populated, `recall_at_10 > 0`. First run downloads the CSN python split and builds an index (allow a few minutes). If `code_search_net` needs `trust_remote_code` and errors, fix `dataset_id`/`trust_remote_code` in the yaml per research notes and re-run.

- [ ] **Step 3: Commit**

```bash
git add benchmarks/relevance/scripts/run.py
git commit -m "$(cat <<'EOF'
feat(relevance): run.py entrypoint (csn/swe-loc/coir, ablation + dense-backend)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 7.2: Acceptance gate — reproduce a published baseline within tolerance

**Files:**
- Create: `benchmarks/relevance/config/baselines.yaml`
- Create: `benchmarks/relevance/scripts/reproduce_baseline.py`

This is the spec's hard acceptance gate: "on a real subset it reproduces a published CSN/CoIR baseline within a stated tolerance." We pin a published BM25 CSN MRR number (BM25 is reproducible without any model — it's pure Tantivy via `--sparse-only`), run `--sparse-only` on the configured CSN subset, and assert our measured MRR is within tolerance. BM25 is the right anchor because it's deterministic and model-independent; published CodeBERT/UniXcoder numbers depend on a specific model we don't ship.

- [ ] **Step 1: Write `baselines.yaml`**

`benchmarks/relevance/config/baselines.yaml`:
```yaml
# Published baselines to reproduce as the protocol-correctness gate.
# Anchor: BM25 on CodeSearchNet (deterministic, model-independent — exactly what
# semantex's --sparse-only channel computes). Tolerance is generous because our
# subset size and chunking differ from the published full-corpus protocol; the
# point is to confirm the harness measures the RIGHT ballpark, not to match to 3
# decimals. Update `expected_mrr_at_10` + `source` with the figure recorded in
# research notes (Task 0.1) for the exact published BM25 CSN number you anchor to.
csn_bm25:
  ablation: sparse-only
  dataset: csn
  language: python
  metric: mrr_at_10
  expected_mrr_at_10: 0.50      # published BM25 CSN MRR ballpark; refine per research notes
  tolerance: 0.15               # |measured - expected| must be <= this
  source: "CodeSearchNet / CoIR BM25 baseline — cite exact figure in research notes"
```

- [ ] **Step 2: Write the gate script**

`benchmarks/relevance/scripts/reproduce_baseline.py`:
```python
"""Acceptance gate: reproduce a published baseline within a stated tolerance.

Runs the configured ablation/dataset, computes the target metric, and exits
non-zero if |measured - expected| > tolerance. Proves the protocol is correct
before any tuning relies on the harness.
"""
from __future__ import annotations

import sys
from pathlib import Path

import click
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
SWE_BENCH_SRC = ROOT.parent / "swe_bench" / "src"
if SWE_BENCH_SRC.is_dir():
    sys.path.insert(0, str(SWE_BENCH_SRC))

import os

from relevance_harness.datasets.csn import load_csn_corpus
from relevance_harness.metrics import mrr_at_k
from relevance_harness.runner import run_corpus

CONFIG = ROOT / "config"


@click.command()
@click.option("--baseline", default="csn_bm25", show_default=True)
@click.option("--semantex-bin", default=os.environ.get("SEMANTEX_BINARY", "semantex"))
def main(baseline: str, semantex_bin: str):
    baselines = yaml.safe_load((CONFIG / "baselines.yaml").read_text())
    b = baselines[baseline]
    csn_cfg = yaml.safe_load((CONFIG / "csn_subset.yaml").read_text())

    corpus = load_csn_corpus(
        language=b["language"],
        corpus_dir=ROOT / "results" / "_baseline" / f"csn_{b['language']}",
        dataset_id=csn_cfg["dataset_id"],
        corpus_size=csn_cfg["corpus_size"],
        query_size=csn_cfg["query_size"],
        seed=csn_cfg["seed"],
        trust_remote_code=csn_cfg.get("trust_remote_code", True),
    )
    out = run_corpus(corpus, ablation=b["ablation"], k=10, semantex_binary=semantex_bin,
                     match_mode="doc_id")
    measured = mrr_at_k(out.relevances, k=10)
    expected = float(b["expected_mrr_at_10"])
    tol = float(b["tolerance"])
    delta = abs(measured - expected)

    click.echo(f"baseline={baseline} measured_mrr@10={measured:.4f} "
               f"expected={expected:.4f} tol={tol:.4f} delta={delta:.4f}")
    click.echo(f"source: {b['source']}")
    if delta > tol:
        click.echo("FAIL: outside tolerance — protocol or wiring is wrong.", err=True)
        raise SystemExit(1)
    click.echo("PASS: within tolerance — protocol reproduces the baseline.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the gate (real, end-to-end)**

```bash
cd benchmarks/relevance && source .venv/bin/activate
python -m scripts.reproduce_baseline --baseline csn_bm25
```
Expected: prints `measured_mrr@10=...` and `PASS: within tolerance`. This run uses the full configured CSN subset (`corpus_size`/`query_size` from `csn_subset.yaml`) so it takes minutes (download + index). If it prints `FAIL`, either (a) the measured number is genuinely off → debug `semantex_client`/`csn` with `superpowers:systematic-debugging`, or (b) the published anchor in `baselines.yaml` was wrong → correct `expected_mrr_at_10`/`tolerance` to the figure cited in research notes and re-run. Do NOT widen tolerance just to pass.

- [ ] **Step 4: Commit**

```bash
git add benchmarks/relevance/config/baselines.yaml \
        benchmarks/relevance/scripts/reproduce_baseline.py
git commit -m "$(cat <<'EOF'
feat(relevance): acceptance gate — reproduce published BM25 CSN baseline

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 7.3: Full unit-suite green + lint

**Files:** none modified — verification task.

- [ ] **Step 1: Run the full unit suite**

```bash
cd benchmarks/relevance && source .venv/bin/activate
pytest -v -m "not e2e" 2>/dev/null || pytest -v --deselect tests/test_runner_e2e.py
```
Expected: all unit tests pass; `test_runner_e2e.py` and `test_metrics_vs_pytrec_eval.py` pass or skip cleanly depending on `semantex`/`pytrec_eval` availability. Target count: `test_types` (4) + `test_subset` (5) + `test_metrics` (9) + `test_semantex_client` (8) + `test_indexing` (6) + `test_runner` (3) + `test_csn` (3) + `test_swe_loc` (4) + `test_coir` (3) + `test_report` (3) = 48 passing unit tests.

- [ ] **Step 2: Lint**

```bash
cd benchmarks/relevance && source .venv/bin/activate
ruff check src tests scripts
```
Expected: no errors (fix any reported lint issues, re-run pytest, then commit).

- [ ] **Step 3: Commit any lint fixups**

```bash
git add benchmarks/relevance/
git commit -m "$(cat <<'EOF'
chore(relevance): lint clean + full unit suite green

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 8 — Ablation sweep + D4 A/B (real runs, gated on S1/S2)

### Task 8.1: Ablation sweep on CSN (real)

**Files:** none modified — execution. Produces `results/<run-id>/` artifacts (gitignored; commit only the `report.md`).

This exercises the four ablations end-to-end and is the deliverable that feeds S2–S7's validation. Uses the default dense backend (`colbert-plaid`) so it runs on TODAY's semantex, before S2 lands.

- [ ] **Step 1: Run the four ablations on CSN**

```bash
cd benchmarks/relevance && source .venv/bin/activate
export SEMANTEX_BINARY=$(which semantex)
for ABL in sparse-only dense-only hybrid rerank; do
  python -m scripts.run --dataset csn --ablation "$ABL" --run-id csn-sweep --k 10
done
cat results/csn-sweep/report.md
```
Expected: a `report.md` whose table has four rows (one per ablation) across the configured languages; hybrid should be ≥ each single modality on MRR@10/nDCG@10 (the Layer-1 success criterion from the eval plan). Note: re-running with the same `--run-id` reuses the materialised corpora + indexes, so only the search step repeats per ablation.

- [ ] **Step 2: Commit the report**

```bash
git add benchmarks/relevance/results/csn-sweep/report.md
git commit -m "$(cat <<'EOF'
data(relevance): CSN ablation sweep (sparse/dense/hybrid/rerank), colbert-plaid

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

(`results/` is gitignored; force-add the single report with `git add -f` if needed, or copy it under `docs/` — the controller decides the artifact-archival policy.)

---

### Task 8.2: D4 dense-backend A/B (real; gated on S1+S2 shipping `coderank-hnsw`)

**Files:** none modified — execution.

**Dependency gate:** this task can only RUN once S1 (the `DenseBackend` seam) and S2 (`coderank-hnsw`) have shipped, so `SEMANTEX_DENSE_BACKEND=coderank-hnsw` is honoured. Until then, both invocations resolve to the same backend and the A/B is a no-op. Mark this task blocked-on-S2 in the tracker.

- [ ] **Step 1: Run hybrid under both backends on CSN + SWE-loc**

```bash
cd benchmarks/relevance && source .venv/bin/activate
export SEMANTEX_BINARY=$(which semantex)
for BK in colbert-plaid coderank-hnsw; do
  python -m scripts.run --dataset csn     --ablation hybrid --dense-backend "$BK" --run-id ab-$BK --k 10
  python -m scripts.run --dataset swe-loc --ablation hybrid --dense-backend "$BK" --run-id ab-$BK --k 10
done
echo "=== colbert-plaid ==="; cat results/ab-colbert-plaid/report.md
echo "=== coderank-hnsw ==="; cat results/ab-coderank-hnsw/report.md
```
Expected: two reports, each stamped with its `dense_backend`. The D4 decision (spec §2): `coderank-hnsw` must **meet-or-beat** `colbert-plaid` on Recall@10/nDCG@10 (CoIR + CSN) before cutover. Record the side-by-side in the integration notes; the controller owns the cutover decision.

- [ ] **Step 2: Commit both reports**

```bash
git add -f benchmarks/relevance/results/ab-colbert-plaid/report.md \
           benchmarks/relevance/results/ab-coderank-hnsw/report.md
git commit -m "$(cat <<'EOF'
data(relevance): D4 dense-backend A/B (colbert-plaid vs coderank-hnsw)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review Notes

- **Coverage vs spec §4 S0:** CoIR (loader + seeded/logged subset config, Task 5.3), CodeSearchNet (Task 5.1), SWE-loc from Verified gold patches reusing the Phase-A cache (Task 5.2); metrics MRR@10/nDCG@10/Recall@{1,5,10}/MAP (Task 2.1, validated against `pytrec_eval` in Task 2.2); ablations `--sparse-only/--dense-only/--hybrid/--rerank` (Task 3.1) AND `--dense-backend colbert-plaid|coderank-hnsw` via env (Task 3.1, exercised Task 8.2); report with subset manifest + git-rev/model stamp (Task 6.1); acceptance gate reproducing a published baseline within tolerance (Task 7.2). The "runs end-to-end on a tiny fixture in tests" requirement is Task 4.2.
- **No hallucinated APIs:** the `semantex --json` schema, the ablation→flag mapping, and the absence of `--hybrid`/`--dense-backend` flags are VERIFIED live (Task 0.1 Steps 1–2). The CSN dataset id/fields, CoIR ids, `pytrec_eval` measure names, and the swe_bench module contract are recorded by the spike (Task 0.1 Steps 3–6) and referenced thereafter. Loaders are unit-tested against injected rows so the network path is the only unverified surface, isolated to the `load_*` functions.
- **CLAUDE.md compliance:** everything lives under `benchmarks/` (exempt from the no-hardcoded-paths rule); the only absolute path is the `$SWE_BENCH_REPO_CACHE` default `~/.swe_bench_repos`, which mirrors the existing swe_bench harness and is env-overridable. No semantex `crates/` code is touched. Deps are permissive (datasets/numpy/pandas/pyyaml/click/tabulate; `pytrec_eval` is dev-only).
- **CPU-feasibility / no silent truncation:** all subsetting goes through `subset.select_queries`, which records a `SubsetManifest` (kept + dropped ids) that the report emits; `n=None` keeps everything and logs zero drops.
- **Type consistency:** `Query`, `Document`, `EvalCorpus`, `RankedResult` (types.py), `RunOutput` (runner.py), `SubsetManifest` (subset.py), `ReproStamp` (report.py) defined once and reused. `query_id`/`doc_id`/`file_path` are the universal keys; doc id is `file:start-end` everywhere (client + loaders agree).

## Gaps / spec requirements not turned into a fully self-contained task

1. **CoIR exact HF ids + the published CoIR baseline figure are unverified on this machine** (no confirmed HF access at plan-time). Task 0.1 Step 4 records them; Task 5.3 ships the injectable, unit-tested loader, but the CoIR network path and a CoIR-specific acceptance baseline are deferred to a machine with HF access. The acceptance gate (Task 7.2) therefore anchors on **BM25/CSN** (deterministic, model-independent) rather than CoIR — satisfying the spec's "CSN **or** CoIR baseline within tolerance" wording, but CoIR-as-headline still needs a real run to populate `coir_subset.yaml` and wire `load_coir_subdataset` into `run.py` (currently a `SystemExit(2)` stub).
2. **The published baseline number itself (`expected_mrr_at_10`, `tolerance`) is a placeholder ballpark.** Task 0.1 must cite the exact figure; the gate's correctness depends on that citation. I set a generous default tolerance (0.15) and flagged "do not widen tolerance to pass" — but the precise anchor is external and must be filled from a cited source.
3. **Function-level (vs file-level) SWE-loc recall** is only partially realized: `swe_loc.py` extracts changed *files* (file-level Recall/MRR, which the spec lists first). The spec also mentions function-level recall ("gold files/functions", "changed files/symbols"). Mapping a patch hunk's `@@ ... def name` to a specific gold *symbol* and matching it against a result's `start_line/end_line` overlap is a natural extension of `RankedResult.rank_of_file` + the hunk parser, but is **not** built here — it would be a follow-up task adding `changed_symbols_from_patch` + a `match_mode="function"` overlap check in `runner.py`. File-level localization is the shipped metric.
4. **D4 A/B (Task 8.2) cannot run until S1+S2 ship.** The harness is backend-agnostic and ready, but the `coderank-hnsw` backend and the `SEMANTEX_DENSE_BACKEND` env honoring it are owned by S1/S2; this task is correctly gated, not self-contained within S0.
