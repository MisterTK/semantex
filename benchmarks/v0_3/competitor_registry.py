#!/usr/bin/env python3
"""
Competitor registry — describes how to detect each comparison tool referenced
by `benchmarks/COMPETITORS.md`.

Centralizing this here keeps `run_public.py` short and gives the human one
place to flip a tool from "auto-detect" to "force on/off" when they run the
real sweep.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CompetitorSpec:
    """How to detect, version-check, and invoke a single competitor.

    `detect()` returns (available: bool, version_or_reason: str).
    A tool's runner module in `benchmarks/v0_3/` is expected to live at
    `benchmarks/v0_3/competitor_<id>.py`. The reproducer imports it lazily
    only when the competitor is requested AND available.
    """
    id: str            # short slug used on CLI: --tool=ripgrep
    display_name: str
    install_hint: str  # one-line install command, never used as a runtime default
    runner_module: Optional[str]  # python module path, None = no runner stub yet

    def detect(self) -> tuple:
        raise NotImplementedError


@dataclass(frozen=True)
class BinarySpec(CompetitorSpec):
    """Tool that exposes itself as a single binary on $PATH."""
    binary: str = ""
    version_args: tuple = ("--version",)

    def detect(self) -> tuple:
        path = shutil.which(self.binary)
        if not path:
            return False, f"binary '{self.binary}' not on $PATH"
        try:
            out = subprocess.check_output(
                [path, *self.version_args],
                stderr=subprocess.STDOUT,
                timeout=10,
            )
            return True, out.decode("utf-8", errors="replace").strip().splitlines()[0]
        except (subprocess.SubprocessError, OSError) as e:
            return False, f"{self.binary} version probe failed: {e}"


@dataclass(frozen=True)
class EnvSpec(CompetitorSpec):
    """Tool that requires an env var (API keys, etc.) plus an optional binary."""
    required_env: tuple = ()
    optional_binary: Optional[str] = None

    def detect(self) -> tuple:
        missing = [k for k in self.required_env if not os.environ.get(k)]
        if missing:
            return False, f"env var(s) not set: {', '.join(missing)}"
        if self.optional_binary and not shutil.which(self.optional_binary):
            return False, f"binary '{self.optional_binary}' not on $PATH"
        return True, "env-configured"


@dataclass(frozen=True)
class ManualSpec(CompetitorSpec):
    """Tool that must be wired by the human at sweep time; reproducer always skips."""

    def detect(self) -> tuple:
        return False, "manual — wire at sweep time per COMPETITORS.md"


# Note: semantex itself is detected via the binary the user passes on the
# command line; it's not listed as a competitor.
COMPETITORS = (
    BinarySpec(
        id="ripgrep",
        display_name="ripgrep 14.x",
        install_hint="brew install ripgrep  # or: cargo install ripgrep --version '^14'",
        runner_module=None,  # TODO(human): benchmarks/v0_3/competitor_ripgrep.py
        binary="rg",
    ),
    BinarySpec(
        id="graphify",
        display_name="graphify 0.8.x",
        install_hint="npm i -g graphify@^0.8.0",
        runner_module=None,  # TODO(human): benchmarks/v0_3/competitor_graphify.py
        binary="graphify",
    ),
    BinarySpec(
        id="claude",
        display_name="Claude Code built-ins",
        install_hint="npm i -g @anthropic-ai/claude-code",
        runner_module=None,  # B3 driver is the existing agent_bench.py — wrapped by run_b3.py
        binary="claude",
    ),
    BinarySpec(
        id="lat",
        display_name="lat.md `lat search`",
        install_hint="see https://github.com/1st1/lat.md",
        runner_module=None,
        binary="lat",
    ),
    EnvSpec(
        id="voyage",
        display_name="voyage-code-3 (API)",
        install_hint="pip install voyageai && export VOYAGE_API_KEY=...",
        runner_module=None,
        required_env=("VOYAGE_API_KEY",),
    ),
    EnvSpec(
        id="openai-embeddings",
        display_name="text-embedding-3-large (API)",
        install_hint="pip install openai && export OPENAI_API_KEY=...",
        runner_module=None,
        required_env=("OPENAI_API_KEY",),
    ),
    ManualSpec(
        id="cursor",
        display_name="Cursor index",
        install_hint="https://cursor.sh/download — MCP exposure check at sweep time",
        runner_module=None,
    ),
    ManualSpec(
        id="copilot",
        display_name="GitHub Copilot symbol search",
        install_hint="gh extension install github/gh-copilot",
        runner_module=None,
    ),
    ManualSpec(
        id="bm25-only",
        display_name="Tantivy BM25 baseline",
        install_hint="build benchmarks/v0_3/bm25_only/ as a release Rust binary",
        runner_module=None,
    ),
    ManualSpec(
        id="hybrid-baseline",
        display_name="bge-large + bm25 + RRF baseline",
        install_hint="benchmarks/v0_3/baseline_hybrid.py — build at sweep time",
        runner_module=None,
    ),
    ManualSpec(
        id="dense-jina",
        display_name="jina-embeddings-v2-base-code (dense baseline)",
        install_hint="python -m fastembed download jinaai/jina-embeddings-v2-base-code",
        runner_module=None,
    ),
)


def by_id(slug: str) -> Optional[CompetitorSpec]:
    for c in COMPETITORS:
        if c.id == slug:
            return c
    return None


def list_ids() -> list:
    return [c.id for c in COMPETITORS]


if __name__ == "__main__":
    # CLI helper: list competitors and their detection status.
    import json as _json
    rows = []
    for c in COMPETITORS:
        ok, msg = c.detect()
        rows.append({
            "id": c.id,
            "display_name": c.display_name,
            "available": ok,
            "status": msg,
            "install_hint": c.install_hint,
            "has_runner": c.runner_module is not None,
        })
    print(_json.dumps(rows, indent=2))
