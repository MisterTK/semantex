#!/usr/bin/env python3
"""Common soft PreToolUse nudge hook for the three-way benchmark.

ONE script, IDENTICAL logic for every tool-equipped arm (semantex / graphify /
serena). Only the environment differs, so the *adoption mechanism* is held
constant and the benchmark isolates **tool quality** from adoption-mechanism
quality. (The builtin baseline gets NO hook at all.)

The hook reads PreToolUse JSON from stdin and three env vars:

  TW_NUDGE     the nudge text to inject (additionalContext).
  TW_OWN_DIR   the tool's own-output dir name to stay silent on (e.g.
               `.semantex` / `graphify-out` / `.serena`) — reading the tool's
               own artifacts must never trigger a "use the tool" nudge.
  TW_SELF_CMD  the tool's own CLI command (e.g. `graphify` / `semantex`); a Bash
               invocation of the tool itself must NOT be nudged. May be empty.

Behavior mirrors the spirit of semantex's own `should_nudge_read_path` hook:

  * Read  : nudge IFF the path is a source/doc file. SILENT for the tool's own
            dir, VCS/build dirs, lock files, and binary/media extensions.
  * Grep / Glob : always nudge (text/file scans the tool replaces).
  * Bash  : nudge IFF the command's first token is a search command
            (grep/rg/ag/ack/find/fd/fgrep/egrep, or `git grep`) AND the command
            does not start with TW_SELF_CMD (don't nudge the tool's own call).
  * anything else : pass through silently.

Fail OPEN: any stdin/parse/lookup error prints `{}` and exits 0, so a hook bug
can never block the agent (and never silently bias an arm toward "no tools").
The hook NEVER denies — it only ever adds context.
"""
from __future__ import annotations

import json
import os
import sys

# Directories whose contents are never source the tool should re-search.
_SKIP_DIR_PREFIXES = (".git/", "node_modules/", "target/", "dist/", "build/")
# Path *segments* that mean "inside a build/VCS dir" anywhere in the path.
_SKIP_DIR_SEGMENTS = ("/.git/", "/node_modules/", "/target/", "/dist/", "/build/")

# Lock/manifest files — machine-generated, not worth a semantic search.
_LOCK_FILES = frozenset({
    "cargo.lock", "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "go.sum", "go.mod", "poetry.lock", "pipfile.lock", "composer.lock",
    "gemfile.lock", "bun.lockb",
})

# Binary / media / minified extensions — no point nudging a semantic tool at them.
_BINARY_EXTS = frozenset({
    "png", "jpg", "jpeg", "gif", "svg", "ico", "webp", "pdf", "zip", "gz",
    "tar", "woff", "woff2", "ttf", "eot", "bin", "wasm", "so", "dylib", "exe",
    "dll", "db", "sqlite", "map", "lock", "class", "jar", "o", "a", "lib",
})

# First-token search commands a code-search tool is meant to replace.
_SEARCH_CMDS = frozenset({"grep", "rg", "ag", "ack", "find", "fd", "fgrep", "egrep"})


def _basename(path: str) -> str:
    return path.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def _is_silent_read(path: str, own_dir: str) -> bool:
    """True if a Read of `path` should NOT be nudged (own-dir / build / lock / binary)."""
    if not path:
        return True
    norm = path.replace("\\", "/")
    low = norm.lower()

    # The tool's own output dir (segment OR leading) — reading its artifacts is fine.
    if own_dir:
        od = own_dir.strip("/")
        if f"/{od}/" in norm or norm.startswith(f"{od}/") or norm.startswith(f"/{od}/"):
            return True

    # VCS / dependency / build directories.
    for seg in _SKIP_DIR_SEGMENTS:
        if seg in norm:
            return True
    for pre in _SKIP_DIR_PREFIXES:
        if norm.startswith(pre) or norm.startswith("/" + pre):
            return True

    base = _basename(norm)
    if base.lower() in _LOCK_FILES:
        return True

    # `.min.js` is effectively binary/minified — treat as silent.
    if low.endswith(".min.js"):
        return True

    # Extension check (last dotted segment of the basename).
    if "." in base:
        ext = base.rsplit(".", 1)[-1].lower()
        if ext in _BINARY_EXTS:
            return True
    return False


def _bash_first_token(command: str) -> tuple[str, str]:
    """Return (first_token, second_token) of a shell command, stripped. Empty if absent.

    Splits on whitespace; tolerant of leading env assignments is NOT attempted (the
    spec keys off the literal first token), so `FOO=bar grep ...` is treated as a
    non-search command — acceptable and conservative (won't over-nudge)."""
    parts = command.strip().split()
    first = parts[0] if parts else ""
    second = parts[1] if len(parts) > 1 else ""
    return first, second


def decide(tool_name: str, tool_input: dict, own_dir: str, self_cmd: str) -> str | None:
    """Pure decision: return the nudge text key to emit, or None to stay silent.

    Returns the SENTINEL string "NUDGE" when the arm's nudge should be injected,
    and None otherwise. (The caller substitutes the real TW_NUDGE text — keeping
    `decide` text-agnostic makes it trivially unit-testable.)"""
    if not isinstance(tool_input, dict):
        return None

    if tool_name == "Read":
        path = tool_input.get("file_path") or ""
        if not isinstance(path, str):
            return None
        return None if _is_silent_read(path, own_dir) else "NUDGE"

    if tool_name in ("Grep", "Glob"):
        return "NUDGE"

    if tool_name == "Bash":
        command = tool_input.get("command") or ""
        if not isinstance(command, str) or not command.strip():
            return None
        first, second = _bash_first_token(command)
        # Don't nudge the tool's own invocation (e.g. `graphify query ...`).
        if self_cmd and first == self_cmd:
            return None
        is_search = first in _SEARCH_CMDS or (first == "git" and second == "grep")
        return "NUDGE" if is_search else None

    return None


def main() -> int:
    try:
        raw = sys.stdin.read()
        payload = json.loads(raw) if raw.strip() else {}
        tool_name = payload.get("tool_name", "")
        tool_input = payload.get("tool_input", {})
        own_dir = os.environ.get("TW_OWN_DIR", "")
        self_cmd = os.environ.get("TW_SELF_CMD", "")
        verdict = decide(tool_name, tool_input, own_dir, self_cmd)
    except Exception:  # noqa: BLE001 — fail OPEN, never block the agent.
        print("{}")
        return 0

    if verdict is None:
        print("{}")
        return 0

    nudge = os.environ.get("TW_NUDGE", "")
    out = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "additionalContext": nudge,
        }
    }
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
