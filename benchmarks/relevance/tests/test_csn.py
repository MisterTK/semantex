
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
