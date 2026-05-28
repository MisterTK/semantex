"""Pure functions: turn-level token records → metrics."""
from __future__ import annotations

from collections import Counter


Turn = dict  # {"input_tokens", "output_tokens", "cache_creation_input_tokens",
             #  "cache_read_input_tokens", "tool_calls": list[str]}


def num_turns(turns: list[Turn]) -> int:
    return len(turns)


def ccb(turns: list[Turn]) -> int:
    """Cumulative Context Burden: total tokens the model actually attended to.

    With Anthropic prompt caching, the attended context per turn is
    input + cache_creation + cache_read. Sum across turns."""
    return sum(
        t["input_tokens"] + t["cache_creation_input_tokens"] + t["cache_read_input_tokens"]
        for t in turns
    )


def cost_usd(turns: list[Turn], *, model: str, pricing: dict) -> float:
    p = pricing[model]
    cost = 0.0
    for t in turns:
        cost += (
            t["input_tokens"] * p["input_per_mtok"]
            + t["output_tokens"] * p["output_per_mtok"]
            + t["cache_creation_input_tokens"] * p["cache_write_per_mtok"]
            + t["cache_read_input_tokens"] * p["cache_read_per_mtok"]
        ) / 1_000_000
    return cost


def tool_distribution(turns: list[Turn]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for t in turns:
        counter.update(t["tool_calls"])
    return dict(counter)
