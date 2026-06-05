#!/usr/bin/env python3
"""Three-way agent benchmark: semantex vs graphify vs serena (+ builtin baseline)
at EQUAL adoption.

The point: every tool-equipped arm gets the IDENTICAL soft PreToolUse nudge hook
(`three_way_nudge.py`), each pointing at its own tool via env. Holding the
adoption mechanism constant isolates **tool quality** from adoption-mechanism
quality — a fairer head-to-head than each tool bringing its own (unequal) skill /
CLAUDE.md steer. The builtin baseline gets NO hook (native floor).

Arms (each a hermetic CLAUDE_CONFIG_DIR under ~/.cbench/three_way/<arm>/):

  builtin   no MCP, no hook. Native Read/Grep/Glob/Bash only.
  semantex  semantex MCP (SEMANTEX_EMBEDDER=lateon-colbert) + nudge → semantex_agent.
  graphify  no MCP (graphify is a Bash CLI; graph prebuilt) + nudge → `graphify query`.
  serena    serena MCP (LSP symbol tools) + nudge → mcp__serena__*.

Repo: /Users/tk/dev/CopilotKit (large TS monorepo). 3 questions (Q1/Q3/Q5 from
claude_bench.QUESTIONS). 1 rep. Judged quality + CCB are computed via the SAME
parser (claude_bench.parse_claude_stream).

Usage:
    # ONE smoke cell (proves plumbing — ~$0.40):
    python3 benchmarks/three_way.py run --arm semantex --qid Q1
    # full matrix (4 arms × 3 Q × 1 rep) — the CONTROLLER runs this, not the build step:
    python3 benchmarks/three_way.py run
    # rebuild the parsed table from saved raw stream-json (free):
    python3 benchmarks/three_way.py parse
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import shutil
import time
from pathlib import Path

# Reuse the claude_bench helpers (do NOT modify that file).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import claude_bench as cb  # noqa: E402

# ── constants ────────────────────────────────────────────────────────────

# Re-resolve the semantex binary from PATH at runtime (NOT at claude_bench import
# time) so a `PATH=$PWD/target/release:$PATH …` prefix on the harness command
# selects the freshly built binary, falling back to whatever claude_bench found.
SEMANTEX_BIN = shutil.which("semantex") or cb.SEMANTEX_BIN

REPO = "/Users/tk/dev/CopilotKit"
QIDS = ("Q1", "Q3", "Q5")  # architecture / deep_technical / feature_planning
REPS = 1

THREE_WAY_HOME = Path.home() / ".cbench" / "three_way"
RESULTS_DIR = Path(__file__).resolve().parent / "results" / "three-way"
RAW_DIR = RESULTS_DIR / "raw"
ROWS_PATH = RESULTS_DIR / "rows.json"

NUDGE_SCRIPT = str(Path(__file__).resolve().parent / "three_way_nudge.py")

# ── arm definitions ──────────────────────────────────────────────────────
# Each non-builtin arm: a nudge env triple (TW_NUDGE / TW_OWN_DIR / TW_SELF_CMD),
# an MCP server map (or None), and the --allowedTools list. The nudge SCRIPT is
# identical for every arm — only this env differs.

SEMANTEX_NUDGE = (
    "Use the semantex_agent tool (semantic code search) instead of Grep/Glob/Read "
    "— one call returns a complete answer with file:line + code; it covers regex "
    "and file-glob too."
)
GRAPHIFY_NUDGE = (
    "Use graphify (knowledge-graph code search) instead of Grep/Glob/Read: run "
    '`graphify query "<your question>"` via Bash — it returns the relevant '
    'nodes/edges from the prebuilt graph. Also `graphify explain "<symbol>"`.'
)
SERENA_NUDGE = (
    "Use serena's LSP symbol tools (mcp__serena__get_symbols_overview, "
    "find_symbol, find_referencing_symbols, search_for_pattern) instead of "
    "Grep/Glob/Read — they navigate code by symbols/references, not text."
)

ARMS: dict[str, dict] = {
    "builtin": {
        "nudge": None,  # no hook
        "mcp": None,
        "allowed_tools": ["Read", "Grep", "Glob", "Bash"],
    },
    "semantex": {
        "nudge": {"TW_NUDGE": SEMANTEX_NUDGE, "TW_OWN_DIR": ".semantex",
                  "TW_SELF_CMD": "semantex"},
        "mcp": {
            "semantex": {
                "command": SEMANTEX_BIN,
                "args": ["mcp"],
                "env": {"SEMANTEX_EMBEDDER": "lateon-colbert"},
            }
        },
        "allowed_tools": ["mcp__semantex__*", "Read", "Grep", "Glob", "Bash"],
    },
    "graphify": {
        "nudge": {"TW_NUDGE": GRAPHIFY_NUDGE, "TW_OWN_DIR": "graphify-out",
                  "TW_SELF_CMD": "graphify"},
        "mcp": None,  # graphify is a Bash CLI; graph prebuilt at graphify-out/graph.json
        "allowed_tools": ["Read", "Grep", "Glob", "Bash"],
    },
    "serena": {
        "nudge": {"TW_NUDGE": SERENA_NUDGE, "TW_OWN_DIR": ".serena",
                  "TW_SELF_CMD": ""},
        "mcp": {
            "serena": {
                "command": "/opt/homebrew/bin/uvx",
                "args": [
                    "--from", "git+https://github.com/oraios/serena", "serena",
                    "start-mcp-server", "--context", "ide",
                    "--project", REPO, "--mode", "no-onboarding",
                    "--enable-web-dashboard", "False", "--transport", "stdio",
                ],
            }
        },
        "allowed_tools": ["mcp__serena__*", "Read", "Grep", "Glob", "Bash"],
    },
}

ARM_NAMES = tuple(ARMS)


def question_by_id(qid: str) -> dict:
    for q in cb.QUESTIONS:
        if q["id"] == qid:
            return q
    raise KeyError(f"unknown qid {qid!r}; available: {[q['id'] for q in cb.QUESTIONS]}")


# ── hermetic per-arm config ──────────────────────────────────────────────


def arm_dir(arm: str) -> Path:
    return THREE_WAY_HOME / arm


def _hook_command(arm: str) -> str:
    """Inline-env command string that runs the common nudge hook for `arm`.

    `TW_NUDGE='...' TW_OWN_DIR='...' TW_SELF_CMD='...' python3 <abs three_way_nudge.py>`
    — Claude Code runs this via `sh -c`, so single-quoting the env values is safe
    for our nudge text (no embedded single quotes)."""
    env = ARMS[arm]["nudge"]
    parts = [f"{k}='{v}'" for k, v in env.items()]
    parts += ["python3", NUDGE_SCRIPT]
    return " ".join(parts)


def settings_for(arm: str) -> dict:
    """settings.json for the arm. builtin = no hooks; every other arm = the SAME
    PreToolUse hook (matchers Read|Grep|Glob and Bash) running the common script
    with the arm's env."""
    if ARMS[arm]["nudge"] is None:
        return {"hooks": {}}
    cmd = _hook_command(arm)
    hook = {"type": "command", "command": cmd, "timeout": 5}
    return {
        "hooks": {
            "PreToolUse": [
                {"matcher": "Read|Grep|Glob", "hooks": [hook]},
                {"matcher": "Bash", "hooks": [hook]},
            ]
        }
    }


def mcp_config_for(arm: str) -> dict:
    servers = ARMS[arm]["mcp"]
    return {"mcpServers": dict(servers) if servers else {}}


def write_arm_config(arm: str) -> Path:
    cdir = arm_dir(arm)
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "settings.json").write_text(json.dumps(settings_for(arm), indent=2))
    (cdir / "mcp.json").write_text(json.dumps(mcp_config_for(arm), indent=2))
    return cdir


def tool_flags(arm: str) -> list[str]:
    """--allowedTools for the arm + Skill disallowed (no skill contamination).
    builtin also explicitly denies every MCP glob so it stays a native floor."""
    if arm == "builtin":
        return ["--allowedTools", *ARMS[arm]["allowed_tools"],
                "--disallowedTools", "Skill", "mcp__semantex__*", "mcp__serena__*"]
    return ["--allowedTools", *ARMS[arm]["allowed_tools"],
            "--disallowedTools", "Skill"]


# ── one cell ─────────────────────────────────────────────────────────────


def run_cell(arm: str, qid: str, api_key: str, timeout: int = 600) -> dict:
    """Run a single (arm, qid) cell. Saves raw stream-json, returns a parsed row."""
    q = question_by_id(qid)
    cdir = write_arm_config(arm)
    mcp_path = cdir / "mcp.json"
    prompt = cb.PROMPT_TEMPLATE.format(question=q["question"])

    cmd = [
        cb.CLAUDE_BIN, "-p", prompt,
        "--output-format", "stream-json", "--verbose",
        "--model", cb.args_model(),
        "--strict-mcp-config", "--mcp-config", str(mcp_path),
    ]
    cmd += tool_flags(arm)  # includes --allowedTools + --disallowedTools Skill

    env = {**os.environ, "ANTHROPIC_API_KEY": api_key, "CLAUDE_CONFIG_DIR": str(cdir)}

    cb.eprint(f"    [{arm:>8}] {qid} …", end="", flush=True)
    t0 = time.time()
    raw = ""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO,
                           timeout=timeout, env=env)
        raw = r.stdout or ""
        if r.returncode != 0 and not raw:
            cb.eprint(f" ERR rc={r.returncode} {r.stderr[:200]}")
    except subprocess.TimeoutExpired:
        cb.eprint(f" TIMEOUT({timeout}s)")
    elapsed = time.time() - t0

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_DIR / f"{arm}_{qid}.jsonl"
    raw_path.write_text(raw)

    if not raw:
        cb.eprint(f" empty ({elapsed:.0f}s)")
        return {"arm": arm, "qid": qid, "error": "empty", "wall_secs": round(elapsed, 1)}

    m = cb.parse_claude_stream(raw)
    row = {
        "arm": arm,
        "qid": qid,
        "question_type": q["type"],
        "ccb": m["ccb"],
        "num_turns": m["num_turns"],
        "tool_calls": m["tool_calls"],
        "tool_calls_by_name": m["tool_calls_by_name"],
        "sx_tool_calls": m["sx_tool_calls"],
        "native_tool_calls": m["native_tool_calls"],
        "cost_usd": m["cost_usd"],
        "wall_secs": round(elapsed, 1),
        "is_error": m["is_error"],
        "answer": m["answer"],
    }
    cb.eprint(f" CCB={m['ccb']:,} turns={m['num_turns']} tc={m['tool_calls']} "
              f"${m['cost_usd']:.3f} ({elapsed:.0f}s)")
    cb.eprint(f"              tools: {m['tool_calls_by_name']}")
    return row


# ── parse subcommand: rebuild rows.json from saved raw ────────────────────


def parse_raw_dir() -> list[dict]:
    """Re-derive every parsed row from the saved raw stream-json files (free)."""
    rows: list[dict] = []
    if not RAW_DIR.exists():
        return rows
    for raw_path in sorted(RAW_DIR.glob("*.jsonl")):
        stem = raw_path.stem  # "<arm>_<qid>"
        if "_" not in stem:
            continue
        arm, qid = stem.rsplit("_", 1)
        raw = raw_path.read_text()
        if not raw.strip():
            rows.append({"arm": arm, "qid": qid, "error": "empty"})
            continue
        m = cb.parse_claude_stream(raw)
        q = next((x for x in cb.QUESTIONS if x["id"] == qid), {})
        rows.append({
            "arm": arm, "qid": qid, "question_type": q.get("type", "?"),
            "ccb": m["ccb"], "num_turns": m["num_turns"], "tool_calls": m["tool_calls"],
            "tool_calls_by_name": m["tool_calls_by_name"],
            "sx_tool_calls": m["sx_tool_calls"], "native_tool_calls": m["native_tool_calls"],
            "cost_usd": m["cost_usd"], "is_error": m["is_error"], "answer": m["answer"],
        })
    return rows


# ── tool-usage reporting ─────────────────────────────────────────────────


def _own_tool_calls(arm: str, by_name: dict[str, int]) -> int:
    """Count of the arm's OWN tool being used (proving the nudge/grant worked)."""
    if arm == "semantex":
        return sum(v for k, v in by_name.items() if k.startswith("mcp__semantex__"))
    if arm == "serena":
        return sum(v for k, v in by_name.items() if k.startswith("mcp__serena__"))
    if arm == "graphify":
        # graphify is invoked via Bash; the parser can't see the command text, so we
        # count it from the raw stream in print_tool_usage (set there). Fallback 0.
        return by_name.get("__graphify_query__", 0)
    return 0  # builtin has no "own" tool


def _count_graphify_bash(raw: str) -> int:
    """Count Bash tool_use blocks whose command starts with `graphify` (its own CLI)."""
    n = 0
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") != "assistant":
            continue
        for block in ev.get("message", {}).get("content", []):
            if isinstance(block, dict) and block.get("type") == "tool_use" \
                    and block.get("name") == "Bash":
                command = (block.get("input", {}) or {}).get("command", "")
                if isinstance(command, str) and command.strip().startswith("graphify"):
                    n += 1
    return n


def print_tool_usage(rows: list[dict]) -> None:
    cb.eprint("\n" + "=" * 64)
    cb.eprint("  Tool usage per arm (did each arm USE its tool vs native?)")
    cb.eprint("=" * 64)
    for row in rows:
        if row.get("error"):
            cb.eprint(f"  [{row['arm']:>8}] {row['qid']}: ERROR ({row['error']})")
            continue
        arm, qid = row["arm"], row["qid"]
        by_name = row.get("tool_calls_by_name", {})
        own = _own_tool_calls(arm, by_name)
        if arm == "graphify":
            raw_path = RAW_DIR / f"{arm}_{qid}.jsonl"
            if raw_path.exists():
                own = _count_graphify_bash(raw_path.read_text())
        native = row.get("native_tool_calls", row.get("tool_calls", 0))
        label = {"semantex": "semantex_agent (mcp__semantex__*)",
                 "graphify": "graphify-query (Bash)",
                 "serena": "serena tools (mcp__serena__*)",
                 "builtin": "own-tool (n/a)"}.get(arm, "own-tool")
        cb.eprint(f"  [{arm:>8}] {qid}: {label}={own}  native(grep/read/...)={native}  "
                  f"all={by_name}")


# ── subcommands ──────────────────────────────────────────────────────────


def cmd_run(args):
    api_key = cb.load_api_key()
    if not api_key:
        cb.eprint("ERROR: ANTHROPIC_API_KEY not set (env or semantex/.env). Aborting.")
        sys.exit(1)

    arms = [args.arm] if args.arm else list(ARM_NAMES)
    qids = [args.qid] if args.qid else list(QIDS)
    for arm in arms:
        if arm not in ARMS:
            cb.eprint(f"ERROR: unknown arm {arm!r}; choices: {ARM_NAMES}"); sys.exit(2)
    for qid in qids:
        question_by_id(qid)  # validates

    cb.eprint("=" * 64)
    cb.eprint(f"  Three-way (equal-adoption soft-hook): {' vs '.join(arms)}")
    cb.eprint(f"  repo={REPO}  qids={qids}  reps={REPS}  model={cb.args_model()}")
    cb.eprint("=" * 64)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    # Load any prior rows so a single-cell run augments rather than clobbers.
    if ROWS_PATH.exists():
        try:
            rows = json.loads(ROWS_PATH.read_text())
        except (OSError, json.JSONDecodeError):
            rows = []

    def _upsert(new_row: dict):
        key = (new_row["arm"], new_row["qid"])
        for i, r in enumerate(rows):
            if (r.get("arm"), r.get("qid")) == key:
                rows[i] = new_row
                return
        rows.append(new_row)

    fresh: list[dict] = []
    for arm in arms:
        for qid in qids:
            row = run_cell(arm, qid, api_key)
            _upsert(row)
            fresh.append(row)
            ROWS_PATH.write_text(json.dumps(rows, indent=2))
            time.sleep(1)

    cb.eprint(f"\n  saved rows: {ROWS_PATH}")
    print_tool_usage(fresh)


def cmd_parse(args):
    rows = parse_raw_dir()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ROWS_PATH.write_text(json.dumps(rows, indent=2))
    cb.eprint(f"  rebuilt {len(rows)} rows from {RAW_DIR} -> {ROWS_PATH}")
    print_tool_usage(rows)
    # Compact summary table.
    cb.eprint("\n  arm/qid       ccb        turns  tc   own  native  $")
    for r in rows:
        if r.get("error"):
            cb.eprint(f"  {r['arm']:>8}/{r['qid']}  ERROR")
            continue
        own = _own_tool_calls(r["arm"], r.get("tool_calls_by_name", {}))
        cb.eprint(f"  {r['arm']:>8}/{r['qid']}  {r['ccb']:>9,}  "
                  f"{r['num_turns']:>4}  {r['tool_calls']:>3}  {own:>3}  "
                  f"{r['native_tool_calls']:>5}   ${r['cost_usd']:.3f}")


# ── CLI ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command")

    rp = sub.add_parser("run", help="run cells (COSTS API tokens)")
    rp.add_argument("--arm", choices=list(ARM_NAMES),
                    help="run only this arm (default: all 4)")
    rp.add_argument("--qid", choices=list(QIDS),
                    help="run only this question (default: Q1,Q3,Q5)")

    sub.add_parser("parse", help="rebuild rows.json from saved raw stream-json (free)")

    a = p.parse_args()
    {"run": cmd_run, "parse": cmd_parse}.get(
        a.command, lambda _: p.print_help())(a)
