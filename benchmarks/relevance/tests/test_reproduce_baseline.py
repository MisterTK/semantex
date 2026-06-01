"""Unit tests for the acceptance-gate protocol logic (no network).

The external CoIR gate's metric computation, file-level matching, chunk-dedup,
and pass/fail decision are exercised with the loader and runner mocked, so the
protocol logic is verified hermetically. A separate, opt-in live test
(RELEVANCE_LIVE_COIR=1) reproduces the published MTEB BM25 number against real
HuggingFace + a real semantex index.
"""
import os
import shutil
from unittest.mock import patch

import pytest

from scripts import reproduce_baseline as gate
from relevance_harness.runner import RunOutput
from relevance_harness.types import Document, EvalCorpus, Query, RankedResult


def _run_output(relevances, n_relevant, per_query=None):
    return RunOutput(corpus_name="coir/codetrans-dl", ablation="sparse-only",
                     relevances=relevances, n_relevant=n_relevant,
                     per_query=per_query or [])


def test_evaluate_metric_ndcg_perfect_is_one():
    out = _run_output([[1], [1]], [1, 1])
    assert gate.evaluate_metric(out, metric="ndcg_at_10") == pytest.approx(1.0)


def test_evaluate_metric_mrr_rank_two():
    out = _run_output([[0, 1]], [1])
    assert gate.evaluate_metric(out, metric="mrr_at_10") == pytest.approx(0.5)


def test_evaluate_metric_rejects_unknown():
    out = _run_output([[1]], [1])
    with pytest.raises(ValueError, match="unknown gate metric"):
        gate.evaluate_metric(out, metric="precision_at_3")


def test_within_tolerance_boundaries():
    assert gate.within_tolerance(0.34, 0.34418, 0.12) is True
    assert gate.within_tolerance(0.10, 0.34418, 0.12) is False
    # exactly on the boundary passes
    assert gate.within_tolerance(0.22418, 0.34418, 0.12) is True


def _coir_corpus(tmp_path):
    # one query whose gold is doc file "doc_597.txt" (minted id "c637__doc_597.txt:1-41")
    docs = (
        Document(doc_id="c637__doc_597.txt:1-41", text="x", file_path="doc_597.txt",
                 start_line=1, end_line=41),
        Document(doc_id="c1__doc_0.txt:1-5", text="y", file_path="doc_0.txt",
                 start_line=1, end_line=5),
    )
    queries = (Query(query_id="637", text="tf code", gold_doc_ids=("c637__doc_597.txt:1-41",)),)
    return EvalCorpus(name="coir/codetrans-dl", documents=docs, queries=queries,
                      corpus_dir=tmp_path, manifest=None)


def test_filewise_corpus_rewrites_gold_to_file_paths(tmp_path):
    fc = gate.filewise_corpus(_coir_corpus(tmp_path))
    assert fc.queries[0].gold_doc_ids == ("doc_597.txt",)


def test_dedup_relevances_collapses_chunks_per_file(tmp_path):
    fc = gate.filewise_corpus(_coir_corpus(tmp_path))
    # semantex returned 3 chunks of the gold file + a distractor file; dedup must
    # keep ONE entry per file (gold first seen at rank 2 here).
    rr = RankedResult(
        query_id="637",
        ranked_doc_ids=("doc_0.txt:1-5", "doc_597.txt:1-20", "doc_597.txt:21-41"),
        ranked_files=("doc_0.txt", "doc_597.txt", "doc_597.txt"),
    )
    out = _run_output([], [], per_query=[rr])
    deduped = gate.dedup_relevances_by_file(out, fc)
    # unique files in order: [doc_0.txt(0), doc_597.txt(1)] -> gold at rank 2
    assert deduped.relevances == [[0, 1]]
    assert deduped.n_relevant == [1]
    # nDCG@10 = (1/log2(3)) / 1 -> matches the single-gold-at-rank-2 case
    import math
    assert gate.evaluate_metric(deduped, metric="ndcg_at_10") == pytest.approx(
        (1 / math.log2(3)) / 1.0
    )


_BASELINE = {
    "type": "external_coir",
    "subdataset": "codetrans-dl",
    "queries_corpus_id": "X/codetrans-dl-queries-corpus",
    "qrels_id": "X/codetrans-dl-qrels",
    "qrels_split": "test",
    "ablation": "sparse-only",
    "metric": "ndcg_at_10",
    "corpus_size": None,
    "query_size": None,
    "seed": 20260531,
    "expected_ndcg_at_10": 0.34418,    # published MTEB BM25 (loose external bound)
    "self_baseline_ndcg_at_10": 0.1884,  # semantex's own measured (tight internal band)
    "internal_tolerance": 0.025,         # TIGHT: catches a ranking regression
    "tolerance": 0.18,                   # LOOSE external sanity bound
    "retrieve_k": 50,
    "source": "MTEB BM25 baseline CodeTransOceanDL nDCG@10=0.34418",
}


def _run_gate_with_measured(measured: float, tmp_path):
    """Drive _run_external_coir with the measured nDCG@10 pinned to `measured`,
    so the two-band decision logic is exercised in isolation."""
    out = _run_output([], [], per_query=[])
    with patch.object(gate, "load_coir_subdataset", return_value=_coir_corpus(tmp_path)), \
         patch.object(gate, "run_corpus", return_value=out), \
         patch.object(gate, "dedup_relevances_by_file", return_value=out), \
         patch.object(gate, "evaluate_metric", return_value=measured):
        return gate._run_external_coir(_BASELINE, "semantex")


def test_external_coir_gate_passes_both_bands_on_correct_run(tmp_path):
    # measured == semantex's own baseline 0.1884: internal delta 0 <= 0.025 AND
    # external delta |0.1884 - 0.34418| = 0.1558 <= 0.18 -> PASS both bands.
    assert _run_gate_with_measured(0.1884, tmp_path) == 0


def test_external_coir_gate_fails_tight_band_on_0_19_to_0_12_regression(tmp_path):
    # The owner's required case: a ranking regression from ~0.19 to 0.12.
    #   tight internal: |0.12 - 0.1884| = 0.0684 > 0.025  -> FAILS the tight band.
    # The new gate FAILS it (whereas a single loose 0.18 band would not reliably
    # catch sub-band drift — see the next test for the cleanly-isolated case).
    assert _run_gate_with_measured(0.12, tmp_path) == 1


def test_tight_band_catches_a_regression_the_old_loose_only_gate_would_miss(tmp_path):
    # measured 0.22 is INSIDE the old loose-only band around the published number
    # (|0.22 - 0.34418| = 0.124 <= 0.18 -> OLD single-band gate would PASS, silent)
    # but OUTSIDE the tight internal band around semantex's own 0.1884
    # (|0.22 - 0.1884| = 0.0316 > 0.025 -> NEW gate FAILS). This is the subtle
    # regression the split exists to catch.
    assert _run_gate_with_measured(0.22, tmp_path) == 1


def test_external_coir_gate_fails_external_bound_on_gross_collapse(tmp_path):
    # gross protocol break: measured collapses toward 0 -> external bound breached
    # (|0.0 - 0.34418| = 0.344 >> 0.18) -> FAIL. (Internal also fails; either trips.)
    assert _run_gate_with_measured(0.0, tmp_path) == 1


@pytest.mark.skipif(
    os.environ.get("RELEVANCE_LIVE_COIR") != "1" or shutil.which("semantex") is None,
    reason="opt-in live gate: set RELEVANCE_LIVE_COIR=1 and have semantex on PATH",
)
def test_external_coir_gate_live_reproduces_published_number():
    # Full live reproduction against real HF + a real semantex index. Two-band
    # check from baselines.yaml: tight internal (0.025 around semantex's own 0.1884)
    # AND loose external (0.18 around MTEB's published 0.34418).
    import yaml
    from pathlib import Path
    cfg_path = Path(__file__).parent.parent / "config" / "baselines.yaml"
    b = yaml.safe_load(cfg_path.read_text())["coir_codetrans_dl"]
    rc = gate._run_external_coir(b, shutil.which("semantex"))
    assert rc == 0
