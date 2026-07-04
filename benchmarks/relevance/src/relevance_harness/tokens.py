"""Rough token-count estimate for a query's retrieval payload.

No tokenizer dependency is added (keeps the harness's zero-heavy-deps
policy): estimates tokens as chars-returned / 4, a commonly-cited ballpark
for English prose and source code across BPE tokenizers. This is an ESTIMATE
for comparing how much context an arm would hand an LLM (e.g. "hybrid returns
2x the tokens sparse-only does for the same Acc@10") — it is NOT an exact
token count and should not be read as one.
"""
from __future__ import annotations

import math

from .types import RankedResult

#: chars-per-token heuristic; see module docstring.
CHARS_PER_TOKEN_ESTIMATE = 4


def estimate_tokens_returned(rr: RankedResult) -> int:
    """Estimate tokens implied by the "content" field of each raw hit.

    Zero when `rr.raw` is empty or hits carry no "content" key (e.g. the
    caller used the default `with_content=False`, which passes
    `--no-content` and returns no payload at all — itself a meaningful data
    point: that arm returned metadata only).
    """
    total_chars = sum(len(item.get("content", "") or "") for item in rr.raw)
    return math.ceil(total_chars / CHARS_PER_TOKEN_ESTIMATE)
