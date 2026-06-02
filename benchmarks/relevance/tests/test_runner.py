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
         patch("relevance_harness.runner.SemantexClient.reset_daemon"), \
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
         patch("relevance_harness.runner.SemantexClient.reset_daemon"), \
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
         patch("relevance_harness.runner.SemantexClient.reset_daemon"), \
         patch("relevance_harness.runner.ensure_index") as mi:
        run_corpus(corpus, ablation="hybrid", k=10, semantex_binary="semantex")
    mi.assert_called_once()


def test_run_corpus_resets_daemon_before_searching(tmp_path):
    # Defeats stale-daemon reuse: a daemon caches adaptive_sizing at spawn time and
    # lives 30 min idle, so the run must stop any existing daemon before searching
    # to guarantee the canonical adaptive-OFF A/B config takes effect.
    corpus = _corpus(tmp_path)
    with patch("relevance_harness.runner.SemantexClient.search",
               lambda self, q, t, *, ablation, k: RankedResult(q, ())), \
         patch("relevance_harness.runner.SemantexClient.reset_daemon") as mreset, \
         patch("relevance_harness.runner.ensure_index"):
        run_corpus(corpus, ablation="hybrid", k=10, semantex_binary="semantex")
    mreset.assert_called_once()


def test_run_corpus_skips_daemon_reset_for_rerank(tmp_path):
    # The rerank ablation requires a manually pre-started daemon (rerank env + a
    # raised RSS cap); resetting it would kill that daemon and silently disable
    # reranking, so rerank runs leave any existing daemon alone.
    corpus = _corpus(tmp_path)
    with patch("relevance_harness.runner.SemantexClient.search",
               lambda self, q, t, *, ablation, k: RankedResult(q, ())), \
         patch("relevance_harness.runner.SemantexClient.reset_daemon") as mreset, \
         patch("relevance_harness.runner.ensure_index"):
        run_corpus(corpus, ablation="rerank", k=10, semantex_binary="semantex")
    mreset.assert_not_called()
