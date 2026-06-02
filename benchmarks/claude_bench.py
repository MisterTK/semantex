#!/usr/bin/env python3
"""
Claude Code head-to-head: builtin tools vs graphify (skill) vs semantex (MCP).

Layer 2 of the eval plan (docs/superpowers/plans/2026-05-30-semantex-eval-plan.md):
measure **context efficiency (CCB)** to answer conceptual code questions, with
the same model (Claude) and the *only* difference being which code-search tool
the agent has. This is graphify's own claimed axis ("token tax") — so we beat it
on its own metric, head-to-head, with a tool-blind quality judge to keep it fair.

Why Claude (not Gemini): graphify ships as a Claude Code *skill*, semantex ships
a Claude Code skill + MCP, and Claude's stream-json exposes per-turn `usage`
(input + cache_read + cache_creation), so we can compute **real CCB** — the
cumulative attended context — not Gemini's billing-token approximation.

## Real CCB

Each `assistant` event reports the context the model attended to that turn:

    ctx_k = usage.input_tokens
          + usage.cache_read_input_tokens
          + usage.cache_creation_input_tokens

(The cached portion still occupies the context window the model attends over; it
is merely billed cheaper. Summing only `input_tokens` undercounts ~1000x once
the prompt is cached — see the probe in the eval plan.)

    CCB  = sum(ctx_k for each assistant turn)   # O(N^2) growth in turns
    peak = max(ctx_k)
    CAF  = CCB / (num_turns * ctx_1)            # 1.0 = no growth

## Hermetic isolation (critical)

The dev's global ~/.claude has the semantex hooks + dozens of skills installed,
which would inject semantex into EVERY arm. So each arm runs with a private
CLAUDE_CONFIG_DIR (clean settings, no hooks), `--strict-mcp-config`, and a
per-arm `--mcp-config`. A fresh config dir is unauthenticated, so an
ANTHROPIC_API_KEY is required (sourced from semantex/.env or the environment).

  - builtin : clean dir, no MCP, no skills  -> native Grep/Read/Glob only
  - semantex: clean dir + semantex MCP registered (its native integration)
  - graphify: clean dir + graphify skill installed + graph.json pre-built

## Usage

    # 0. one-time: ensure ANTHROPIC_API_KEY in env (or in semantex/.env)
    # 1. prepare arms (builds graphify graphs, installs skill, writes configs) -- FREE
    python3 benchmarks/claude_bench.py setup --repos ~/dev/gin ~/dev/flask
    # 2. run the benchmark (COSTS API tokens) -- gated, like Phase A
    python3 benchmarks/claude_bench.py run \
        --repos ~/dev/gin ~/dev/flask --reps 4 --output benchmarks/results/cbench1/
    # 3. blind-judge answer quality (COSTS API tokens)
    python3 benchmarks/claude_bench.py judge --input benchmarks/results/cbench1/
    # 4. report (FREE)
    python3 benchmarks/claude_bench.py report --input benchmarks/results/cbench1/
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

CLAUDE_BIN = shutil.which("claude") or "claude"
GRAPHIFY_BIN = shutil.which("graphify") or "graphify"
SEMANTEX_BIN = shutil.which("semantex") or "/usr/local/bin/semantex"

ARMS = ("builtin", "graphify", "semantex")

# Workspace for hermetic per-arm Claude config dirs + graphify skill.
BENCH_HOME = Path(os.environ.get("CBENCH_HOME", Path.home() / ".cbench"))

# Identical conceptual questions across arms (mirror gemini_bench.py so results
# are comparable). Tagged semantic vs structural for the Layer-3 split.
QUESTIONS = [
    {"id": "Q1", "type": "architecture", "bucket": "semantic",
     "question": "What are the main components of this project and how do they "
                 "interact? Trace the primary data flow from entry point through "
                 "the core logic."},
    {"id": "Q2", "type": "error_handling", "bucket": "semantic",
     "question": "How does this project handle errors? What patterns are used "
                 "for error propagation, reporting, and recovery?"},
    {"id": "Q3", "type": "deep_technical", "bucket": "semantic",
     "question": "Explain the most complex algorithm or data transformation in "
                 "this codebase step by step."},
    {"id": "Q4", "type": "exhaustive", "bucket": "structural",
     "question": "List all configuration options, environment variables, and CLI "
                 "flags this project supports and where they are defined."},
    {"id": "Q5", "type": "feature_planning", "bucket": "structural",
     "question": "If I wanted to add comprehensive request/operation logging to "
                 "this project, what files would need to change (callers and "
                 "callees included) and what would the implementation look like?"},
]

PROMPT_TEMPLATE = (
    "Analyze the codebase in the current working directory. Answer this question "
    "thoroughly with specific file paths and line references.\n\nQuestion: {question}"
)

# Per-arm CLAUDE.md nudges. Kept symmetric: each tool-equipped arm gets one
# paragraph telling it to prefer its tool; builtin gets none. (Fairness: same
# length/structure, no arm gets richer task hints than another.)
SEMANTEX_MD = """\
# Code Search
This project has the `semantex_agent` MCP tool. Use it as your PRIMARY tool for
all code search and understanding — not grep/glob/find. One `semantex_agent`
call replaces many grep+read iterations and returns full function bodies plus
callers/callees. Trust its answer; only read files if the answer is incomplete.
"""
GRAPHIFY_MD = """\
# Code Search
This project has a graphify knowledge graph (graphify-out/graph.json) and the
graphify skill. Use it as your PRIMARY tool for code navigation — not
grep/glob/find. Query the graph for symbols, callers, callees, and shortest
paths between symbols instead of scanning files manually.
"""

# Whole-system tuning config-arms: each is a semantex MCP arm + a dict of
# SEMANTEX_* env applied to the `semantex mcp` process (the in-process server
# reads them at construction). Measured BARE-MCP (no CLAUDE.md nudge). The
# server-default arms (budget/full_code/depth) rely on Workstream-A env knobs.
SX_CONFIG_ARMS: dict[str, dict[str, str]] = {
    "sx-lateon":       {"SEMANTEX_EMBEDDER": "lateon-colbert"},
    "sx-coderank":     {"SEMANTEX_EMBEDDER": "coderank-137m"},
    "sx-graph2hop":    {"SEMANTEX_EMBEDDER": "lateon-colbert",
                        "SEMANTEX_GRAPH_HOPS": "2", "SEMANTEX_GRAPH_CENTRALITY_WEIGHT": "0.2"},
    "sx-adaptive-off": {"SEMANTEX_EMBEDDER": "lateon-colbert", "SEMANTEX_ADAPTIVE_SIZING": "0"},
    "sx-stacked":      {"SEMANTEX_EMBEDDER": "lateon-colbert", "SEMANTEX_GRAPH_HOPS": "2",
                        "SEMANTEX_GRAPH_CENTRALITY_WEIGHT": "0.2", "SEMANTEX_ADAPTIVE_SIZING": "0"},
    "sx-budget-low":   {"SEMANTEX_EMBEDDER": "lateon-colbert", "SEMANTEX_MCP_BUDGET": "6000"},
    "sx-budget-high":  {"SEMANTEX_EMBEDDER": "lateon-colbert", "SEMANTEX_MCP_BUDGET": "24000"},
    "sx-full-code":    {"SEMANTEX_EMBEDDER": "lateon-colbert", "SEMANTEX_MCP_FULL_CODE": "1"},
    "sx-depth-deep":   {"SEMANTEX_EMBEDDER": "lateon-colbert", "SEMANTEX_MCP_DEPTH": "deep"},
}


def is_semantex_arm(arm: str) -> bool:
    """True for the plain `semantex` arm and every `sx-*` config-arm."""
    return arm == "semantex" or arm in SX_CONFIG_ARMS


def all_arm_names() -> list[str]:
    """builtin + graphify + semantex + every config-arm (for CLI validation)."""
    return list(ARMS) + list(SX_CONFIG_ARMS)


def nudge_for_arm(arm: str) -> str | None:
    """Repo CLAUDE.md nudge for an arm. BARE-MCP: config-arms get NONE — the tool
    descriptions carry their own weight. Only legacy `semantex`/`graphify` keep one."""
    return {"semantex": SEMANTEX_MD, "graphify": GRAPHIFY_MD}.get(arm)


def eprint(*a, **k):
    print(*a, file=sys.stderr, **k)


# ── env / auth ────────────────────────────────────────────────────────


def load_api_key() -> str | None:
    """ANTHROPIC_API_KEY from env, else from semantex/.env."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return os.environ["ANTHROPIC_API_KEY"]
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("ANTHROPIC_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


# ── per-arm hermetic config ─────────────────────────────────────────────


def arm_config_dir(arm: str) -> Path:
    return BENCH_HOME / "config" / arm


def mcp_config_for(arm: str) -> dict:
    """Strict MCP set per arm. builtin/graphify get none; semantex + every sx-*
    config-arm get the semantex MCP server, with the config-arm's SEMANTEX_* env
    forwarded to the `semantex mcp` process (read once at server construction)."""
    if is_semantex_arm(arm):
        server = {"command": SEMANTEX_BIN, "args": ["mcp"]}
        env = SX_CONFIG_ARMS.get(arm)
        if env:
            server["env"] = dict(env)
        return {"mcpServers": {"semantex": server}}
    return {"mcpServers": {}}


def write_arm_config(arm: str) -> Path:
    """Create a clean CLAUDE_CONFIG_DIR for `arm`: empty settings (no inherited
    hooks), and for graphify, the installed skill. Returns the config dir."""
    cdir = arm_config_dir(arm)
    cdir.mkdir(parents=True, exist_ok=True)
    # Clean settings — explicitly no hooks so the dev's global semantex/grep
    # hooks can't leak into any arm.
    (cdir / "settings.json").write_text(json.dumps({"hooks": {}}, indent=2))
    if arm == "graphify":
        # Install graphify's skill into THIS config dir (not the global one).
        env = {**os.environ, "CLAUDE_CONFIG_DIR": str(cdir)}
        r = subprocess.run([GRAPHIFY_BIN, "install", "--platform", "claude"],
                           capture_output=True, text=True, env=env)
        if r.returncode != 0:
            eprint(f"  WARN: `graphify install` rc={r.returncode}: {r.stderr[:200]}")
        skills = list((cdir / "skills").glob("**/SKILL.md")) if (cdir / "skills").exists() else []
        eprint(f"  graphify skill files in {cdir}/skills: {len(skills)}")
    return cdir


def build_graphify_graph(repo: str) -> bool:
    """Build/refresh graphify-out/graph.json for a repo (no LLM). Idempotent."""
    graph = Path(repo) / "graphify-out" / "graph.json"
    if graph.exists():
        eprint(f"  graphify graph cached: {graph}")
        return True
    eprint(f"  building graphify graph: {repo} ...", end="", flush=True)
    r = subprocess.run([GRAPHIFY_BIN, "update", repo], capture_output=True, text=True,
                       timeout=900)
    ok = graph.exists()
    eprint(" ok" if ok else f" FAILED rc={r.returncode} {r.stderr[:160]}")
    return ok


def cmd_setup(args):
    arms = getattr(args, "arms", list(ARMS))
    BENCH_HOME.mkdir(parents=True, exist_ok=True)
    eprint(f"Preparing hermetic arms {arms} (free, no API calls)…")
    for arm in arms:
        eprint(f"[{arm}] config dir:")
        write_arm_config(arm)
    if "graphify" in arms:
        eprint("\nBuilding graphify graphs per repo:")
        for repo in args.repos:
            build_graphify_graph(repo)
    key = load_api_key()
    eprint(f"\nANTHROPIC_API_KEY: {'FOUND' if key else 'MISSING — set it before `run`'}")
    eprint("Setup complete. `run` will cost API tokens (gated).")


# ── stream-json parser → real CCB ───────────────────────────────────────


def parse_claude_stream(raw: str) -> dict:
    events = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    ctx_per_turn: list[int] = []
    tool_calls = 0
    answer = ""
    result_stats: dict = {}
    for ev in events:
        t = ev.get("type")
        if t == "assistant":
            msg = ev.get("message", {})
            u = msg.get("usage", {})
            ctx = (u.get("input_tokens", 0)
                   + u.get("cache_read_input_tokens", 0)
                   + u.get("cache_creation_input_tokens", 0))
            if ctx > 0:
                ctx_per_turn.append(ctx)
            for block in msg.get("content", []):
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    tool_calls += 1
        elif t == "result":
            result_stats = ev
            answer = ev.get("result", "") or answer

    ccb = sum(ctx_per_turn)
    peak = max(ctx_per_turn) if ctx_per_turn else 0
    first = ctx_per_turn[0] if ctx_per_turn else 0
    num_turns = result_stats.get("num_turns", len(ctx_per_turn))
    caf = (ccb / (len(ctx_per_turn) * first)) if (ctx_per_turn and first) else 0.0

    return {
        "ccb": ccb,
        "peak_context": peak,
        "num_turns": num_turns,
        "tool_calls": tool_calls,
        "caf": round(caf, 3),
        "cost_usd": result_stats.get("total_cost_usd", 0.0),
        "duration_ms": result_stats.get("duration_ms", 0),
        "is_error": result_stats.get("is_error", False),
        "ctx_per_turn": ctx_per_turn,
        "answer": answer,
    }


# ── runner ──────────────────────────────────────────────────────────────


def run_claude(prompt: str, repo: str, arm: str, api_key: str, timeout: int = 600) -> str:
    cdir = arm_config_dir(arm)
    mcp_path = cdir / "mcp.json"
    mcp_path.write_text(json.dumps(mcp_config_for(arm)))

    # Per-arm CLAUDE.md nudge in the repo (cleaned up after).
    md_path = Path(repo) / "CLAUDE.md"
    injected = False
    nudge = nudge_for_arm(arm)
    if nudge and not md_path.exists():
        md_path.write_text(nudge)
        injected = True

    cmd = [
        CLAUDE_BIN, "-p", prompt,
        "--output-format", "stream-json", "--verbose",
        "--model", args_model(),
        "--strict-mcp-config", "--mcp-config", str(mcp_path),
    ]
    if arm == "builtin":
        # No skills, no MCP — native search only.
        cmd += ["--disallowedTools", "Skill", "mcp__semantex__*"]
    elif is_semantex_arm(arm):
        cmd += ["--disallowedTools", "Skill"]  # MCP only, no skill contamination
    # graphify: allow Skill (its skill lives in this hermetic config dir).

    env = {**os.environ,
           "ANTHROPIC_API_KEY": api_key,
           "CLAUDE_CONFIG_DIR": str(cdir)}
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=repo,
                           timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        eprint(f" TIMEOUT({timeout}s)")
        return ""
    finally:
        if injected and md_path.exists():
            md_path.unlink()
    if r.returncode != 0 and not r.stdout:
        eprint(f" ERR rc={r.returncode} {r.stderr[:160]}")
        return ""
    return r.stdout


_MODEL = "claude-sonnet-4-6"


def args_model() -> str:
    return _MODEL


def run_single(question: dict, repo: str, arm: str, rep: int, api_key: str) -> dict:
    prompt = PROMPT_TEMPLATE.format(question=question["question"])
    eprint(f"    [{arm:>8}] {question['id']} r{rep}…", end="", flush=True)
    t0 = time.time()
    raw = run_claude(prompt, repo, arm, api_key)
    elapsed = time.time() - t0
    if not raw:
        eprint(f" empty ({elapsed:.0f}s)")
        return {"error": "empty", "arm": arm, "rep": rep, "repo": repo, **question}
    m = parse_claude_stream(raw)
    m.update({"arm": arm, "rep": rep, "repo": repo,
              "question_id": question["id"], "question_type": question["type"],
              "bucket": question["bucket"], "wall_secs": elapsed})
    eprint(f" CCB={m['ccb']:,} turns={m['num_turns']} tc={m['tool_calls']} "
           f"${m['cost_usd']:.3f} ({elapsed:.0f}s)")
    return m


def cmd_run(args):
    global _MODEL
    _MODEL = args.model
    api_key = load_api_key()
    if not api_key:
        eprint("ERROR: ANTHROPIC_API_KEY not set (env or semantex/.env). Aborting.")
        sys.exit(1)
    arms = getattr(args, "arms", list(ARMS))
    out = Path(args.output)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    results = []
    eprint("=" * 64)
    eprint(f"  Claude head-to-head: {' vs '.join(arms)}  (model={args.model})")
    eprint(f"  repos={len(args.repos)} questions={len(QUESTIONS)} reps={args.reps}")
    eprint("=" * 64)
    for repo in args.repos:
        rn = Path(repo).name
        eprint(f"\n  repo: {rn}")
        for q in QUESTIONS:
            for arm in arms:
                for rep in range(1, args.reps + 1):
                    res = run_single(q, repo, arm, rep, api_key)
                    results.append(res)
                    (out / "raw" / f"{rn}_{q['id']}_{arm}_r{rep}.json").write_text(
                        json.dumps(res, indent=2))
                    time.sleep(1)
    (out / "all_results.json").write_text(json.dumps(results, indent=2))
    eprint(f"\n  saved: {out/'all_results.json'}")
    eprint("  next: `judge` then `report`")


# ── blind quality judge ──────────────────────────────────────────────────


JUDGE_PROMPT = """\
You are grading an answer to a question that was asked about SOME codebase (the
codebase the answer itself describes — NOT any project in your current working
directory; you have no project context and must not assume one). Score the
ANSWER on a 1-5 scale for how correct, specific, and complete it is *as an
answer to the question*: 5=excellent (specific files/lines, complete), 3=partial,
1=poor/empty. You are NOT told which tool produced it; judge only its substance.

QUESTION: {question}

ANSWER:
{answer}

Reply with ONLY a JSON object: {{"score": <1-5>, "reason": "<one sentence>"}}"""

# Tools blocked for the judge — it grades text only, never explores.
JUDGE_BLOCK_TOOLS = ["Skill", "Bash", "Grep", "Glob", "Read", "Edit", "Write",
                     "WebSearch", "WebFetch", "Task", "TodoWrite"]


def cmd_judge(args):
    api_key = load_api_key()
    if not api_key:
        eprint("ERROR: ANTHROPIC_API_KEY not set. Aborting.")
        sys.exit(1)
    inp = Path(args.input)
    results = json.loads((inp / "all_results.json").read_text())
    qmap = {q["id"]: q["question"] for q in QUESTIONS}
    # CRITICAL: run the judge in a NEUTRAL, project-free working directory.
    # Running it inside a repo (or this one) makes Claude Code inject that
    # project's CLAUDE.md, so the judge wrongly grades every answer against the
    # cwd's project ("this describes Gin, but the codebase is semantex" -> 1/5).
    neutral_cwd = BENCH_HOME / "judge_cwd"
    neutral_cwd.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "ANTHROPIC_API_KEY": api_key,
           "CLAUDE_CONFIG_DIR": str(arm_config_dir("builtin"))}
    for r in results:
        if r.get("error") or not r.get("answer"):
            continue
        prompt = JUDGE_PROMPT.format(question=qmap.get(r["question_id"], ""),
                                     answer=r["answer"][:6000])
        # Tool-blind, MCP-free, project-free judge call.
        proc = subprocess.run(
            [CLAUDE_BIN, "-p", prompt, "--output-format", "json",
             "--model", args.judge_model, "--strict-mcp-config",
             "--mcp-config", json.dumps({"mcpServers": {}}),
             "--disallowedTools", *JUDGE_BLOCK_TOOLS],
            capture_output=True, text=True, env=env, cwd=str(neutral_cwd), timeout=180)
        score, reason = None, ""
        try:
            res = json.loads(proc.stdout)
            txt = res.get("result", "") if isinstance(res, dict) else ""
            j = json.loads(txt[txt.find("{"): txt.rfind("}") + 1])
            score, reason = j.get("score"), j.get("reason", "")
        except Exception as e:  # noqa: BLE001
            reason = f"judge-parse-failed: {e}"
        r["quality"] = score
        r["quality_reason"] = reason
        eprint(f"  {Path(r['repo']).name} {r['question_id']} {r['arm']} r{r.get('rep')}: "
               f"quality={score}")
    (inp / "all_results.json").write_text(json.dumps(results, indent=2))
    eprint("  judge scores merged into all_results.json")


# ── report ────────────────────────────────────────────────────────────────


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else 0.0


def cmd_report(args):
    inp = Path(args.input)
    results = json.loads((inp / "all_results.json").read_text())
    valid = [r for r in results if not r.get("error")]

    def agg(arm, field, bucket=None):
        rs = [r for r in valid if r["arm"] == arm and (bucket is None or r.get("bucket") == bucket)]
        return _mean([r.get(field) for r in rs])

    lines = ["# Claude head-to-head: builtin vs graphify vs semantex\n"]
    lines.append(f"*model={args.input}*  arms={', '.join(ARMS)}\n")
    lines.append("## Aggregate (mean per question-run)\n")
    lines.append("| Metric | builtin | graphify | semantex | sx vs builtin | sx vs graphify |")
    lines.append("|---|---|---|---|---|---|")

    def row(label, field, fmt="{:.0f}"):
        b, g, s = agg("builtin", field), agg("graphify", field), agg("semantex", field)
        def d(ref):
            return "N/A" if not ref else f"{(s/ref-1)*100:+.0f}%"
        lines.append(f"| {label} | {fmt.format(b)} | {fmt.format(g)} | {fmt.format(s)} "
                     f"| {d(b)} | {d(g)} |")

    row("CCB (real)", "ccb", "{:,.0f}")
    row("peak context", "peak_context", "{:,.0f}")
    row("turns", "num_turns", "{:.1f}")
    row("tool calls", "tool_calls", "{:.1f}")
    row("cost USD", "cost_usd", "{:.3f}")
    row("CAF", "caf", "{:.2f}")
    row("quality (1-5)", "quality", "{:.2f}")
    row("wall secs", "wall_secs", "{:.0f}")

    # semantic vs structural split (Layer 3 preview)
    lines.append("\n## CCB by bucket (semantic vs structural)\n")
    lines.append("| bucket | builtin | graphify | semantex |")
    lines.append("|---|---|---|---|")
    for bucket in ("semantic", "structural"):
        b, g, s = (agg(a, "ccb", bucket) for a in ARMS)
        lines.append(f"| {bucket} | {b:,.0f} | {g:,.0f} | {s:,.0f} |")

    # quality gate note
    lines.append("\n> Compare CCB **only at equal-or-better quality**. If "
                 "semantex's quality ≥ the others', the CCB reduction is a clean win.\n")
    report = "\n".join(lines) + "\n"
    (inp / "report.md").write_text(report)
    print(report)


# ── CLI ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command")

    sp = sub.add_parser("setup", help="prepare hermetic arms + graphify graphs (free)")
    sp.add_argument("--repos", nargs="+", required=True)
    sp.add_argument("--arms", nargs="+", default=list(ARMS), choices=all_arm_names(),
                    help="which arms to prepare (default: all). graphify graphs are "
                         "only built when the graphify arm is selected.")

    rp = sub.add_parser("run", help="run the benchmark (COSTS API tokens)")
    rp.add_argument("--repos", nargs="+", required=True)
    rp.add_argument("--output", required=True)
    rp.add_argument("--reps", type=int, default=4)
    rp.add_argument("--model", default="claude-sonnet-4-6")
    rp.add_argument("--arms", nargs="+", default=list(ARMS), choices=all_arm_names(),
                    help="which arms to run (default: all 3). e.g. --arms semantex to "
                         "measure only semantex against a prior run's stored baseline.")

    jp = sub.add_parser("judge", help="blind-grade answer quality (COSTS API tokens)")
    jp.add_argument("--input", required=True)
    jp.add_argument("--judge-model", default="claude-sonnet-4-6")

    rep = sub.add_parser("report", help="generate report (free)")
    rep.add_argument("--input", required=True)

    a = p.parse_args()
    {"setup": cmd_setup, "run": cmd_run, "judge": cmd_judge, "report": cmd_report}.get(
        a.command, lambda _: p.print_help())(a)
