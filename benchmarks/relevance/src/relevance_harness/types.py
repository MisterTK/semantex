"""Uniform data model emitted by every dataset loader. No I/O."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .subset import SubsetManifest


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
    # The subset manifest (kept/dropped query ids) for this corpus, if the loader
    # subsetted its queries. Reports surface it so every number carries provenance.
    manifest: Optional["SubsetManifest"] = None

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
    # Raw per-hit dicts as parsed from `--json` (may or may not carry a
    # "content" field depending on --no-content/--snippet). Empty unless the
    # caller asked for content; see relevance_harness.tokens for the consumer.
    raw: tuple[dict, ...] = field(default_factory=tuple)

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
