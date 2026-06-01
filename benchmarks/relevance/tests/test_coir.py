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
