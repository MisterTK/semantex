#!/usr/bin/env python3
"""Substitution-vs-recall probe (2026-06-04).

Routes a major investment decision: when a coding agent (Claude) calls the
`semantex` MCP tool ~1/task and THEN still does native grep/Read, WHY?

  DUPLICATE  = it re-reads files semantex ALREADY returned  -> TRUST/SUBSTITUTION
               problem -> the fix is tool-output/adoption work.
  NEW        = it reads files semantex did NOT return        -> RETRIEVAL RECALL
               problem -> the fix is the EMBEDDER.

The duplicate-vs-new ratio of post-semantex native file-accesses is the answer.

Reuses benchmarks/claude_bench.py helpers verbatim (mcp_config_for, arm_config_dir,
QUESTIONS, PROMPT_TEMPLATE, CLAUDE_BIN, args_model) for the SAME bare-MCP command
the real benchmark uses — NO CLAUDE.md nudge. Captures full stream-json per run.

Usage:
    python3 benchmarks/probe_substitution.py run     # COSTS API tokens (~$5)
    python3 benchmarks/probe_substitution.py parse   # FREE — (re)build report from raw
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import claude_bench as cb  # noqa: E402

ARM = "sx-lateon"
REPO = "/path/to/gin"
REPS = 3
OUT = Path(__file__).resolve().parent.parent / "results" / "probe-substitution"
RAW = OUT / "raw"
# Steered supplement: the bare-MCP arm rarely calls semantex at all (adoption gap),
# so it yields too few sx-using runs to answer the duplicate-vs-new question. The
# steered arm injects the repo CLAUDE.md nudge (claude_bench SEMANTEX_MD) telling
# the agent to use semantex as PRIMARY — this is the regime the question assumes
# ("an agent that DOES call semantex ~1/task"). Same dense backend/env as sx-lateon;
# the ONLY difference is the nudge, so the post-sx reads we classify are unaffected
# by the steer (the steer changes WHETHER it calls sx, not WHAT it reads after).
STEERED_ARM = "sx-lateon-steered"
RAW_STEERED = OUT / "raw_steered"

SX_AGENT_TOOL = "mcp__semantex__semantex_agent"

# A semantex_agent result line header looks like:  "gin.go:764-779 serveError [fn]"
# or "ginS/gins.go:56-58 Handle [fn]" or "errors.go:82 Error". Repo-relative path,
# then ':', then line or line-range. Capture the path.
_SX_FILE_RE = re.compile(r"(?m)^\s*([\w./\-]+\.\w+):\d+(?:-\d+)?\b")
# Also catch inline `path:line` refs the agent route prose may emit.
_INLINE_FILE_RE = re.compile(r"\b([\w./\-]+\.\w+):\d+(?:-\d+)?\b")

# Bash file-targeting commands we can best-effort attribute to a file.
_BASH_FILE_CMDS = ("cat", "head", "tail", "sed", "awk", "grep", "rg", "less", "more", "wc")


def repo_rel(path: str, repo: str = REPO) -> str:
    """Normalize an absolute or relative path to repo-relative posix form."""
    if not path:
        return ""
    p = path.strip().strip('"').strip("'")
    repo_abs = str(Path(repo).resolve())
    if p.startswith(repo_abs):
        p = p[len(repo_abs):].lstrip("/")
    elif p.startswith("/"):
        # absolute but outside repo — keep basename-ish tail; still normalize
        try:
            p = str(Path(p).relative_to(repo_abs))
        except ValueError:
            pass
    p = p.lstrip("./")
    return p


def extract_sx_files(result_text: str) -> set[str]:
    """File paths a semantex_agent tool_result references (repo-relative)."""
    files: set[str] = set()
    for m in _SX_FILE_RE.finditer(result_text):
        files.add(repo_rel(m.group(1)))
    for m in _INLINE_FILE_RE.finditer(result_text):
        files.add(repo_rel(m.group(1)))
    return {f for f in files if f}


def _tool_result_text(block) -> str:
    c = block.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return " ".join(x.get("text", "") for x in c if isinstance(x, dict))
    return ""


def bash_target_file(command: str) -> str | None:
    """Best-effort: extract a single concrete file path a grep/cat/sed/... Bash
    command targets. Fuzzy by nature; returns None when no clear file token."""
    if not command:
        return None
    # take the LAST pipeline segment's leading utility (closest to the file arg),
    # but also scan the whole command for a .ext token that looks like a path.
    first_word = command.strip().split()[0] if command.strip() else ""
    base = os.path.basename(first_word)
    is_file_cmd = base in _BASH_FILE_CMDS or any(
        f" {c} " in f" {command} " or command.strip().startswith(c + " ")
        for c in _BASH_FILE_CMDS
    )
    if not is_file_cmd:
        return None
    # find file-looking tokens (have an extension, not a flag/glob/regex)
    cands = []
    for tok in re.split(r"[\s|;&><]+", command):
        tok = tok.strip().strip('"').strip("'")
        if not tok or tok.startswith("-"):
            continue
        if any(ch in tok for ch in "*?[]()"):  # glob/regex — not a single file
            continue
        if re.search(r"/?[\w.\-/]+\.\w+$", tok) and "." in os.path.basename(tok):
            # avoid matching the command itself or common non-files
            if os.path.basename(tok) in _BASH_FILE_CMDS:
                continue
            cands.append(tok)
    if not cands:
        return None
    # prefer a candidate that resolves to a real file in the repo
    for c in cands:
        rr = repo_rel(c)
        if (Path(REPO) / rr).exists():
            return rr
    return repo_rel(cands[-1])


def parse_run(raw: str) -> dict:
    """Walk a stream-json transcript in temporal order. Build the set of files
    semantex returned, then classify each post-semantex native file access."""
    events = []
    s = raw.strip()
    if s.startswith("["):  # JSON array form
        try:
            events = json.loads(s)
        except json.JSONDecodeError:
            events = []
    if not events:
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    # First pass: map tool_use_id -> (tool_name, target) and collect tool_results.
    # We walk events in order; assistant emits tool_use, the following user event
    # carries the matching tool_result(s) keyed by tool_use_id.
    timeline = []  # ordered list of ("use", id, name, input) and ("result", id, text)
    for ev in events:
        t = ev.get("type")
        if t == "assistant":
            for b in ev.get("message", {}).get("content", []):
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    timeline.append(("use", b.get("id"), b.get("name", "?"), b.get("input", {})))
        elif t == "user":
            cont = ev.get("message", {}).get("content")
            if isinstance(cont, list):
                for b in cont:
                    if isinstance(b, dict) and b.get("type") == "tool_result":
                        timeline.append(("result", b.get("tool_use_id"), _tool_result_text(b)))

    # Match results back to uses.
    use_by_id = {}
    for item in timeline:
        if item[0] == "use":
            use_by_id[item[1]] = {"name": item[2], "input": item[3], "result": ""}
    for item in timeline:
        if item[0] == "result" and item[1] in use_by_id:
            use_by_id[item[1]]["result"] = item[2]

    # Ordered list of tool_use entries (with their results), in emission order.
    ordered = [use_by_id[i] for (k, i, *_rest) in timeline if k == "use" and i in use_by_id]

    sx_calls = [e for e in ordered if e["name"] == SX_AGENT_TOOL]
    sx_returned_files: set[str] = set()
    first_sx_index = None
    for idx, e in enumerate(ordered):
        if e["name"] == SX_AGENT_TOOL:
            if first_sx_index is None:
                first_sx_index = idx
            sx_returned_files |= extract_sx_files(e["result"])

    accesses = []  # post-semantex native file accesses
    if first_sx_index is not None:
        for e in ordered[first_sx_index + 1:]:
            name = e["name"]
            inp = e["input"] or {}
            file_rel, conf, method = None, None, None
            if name == "Read":
                fp = inp.get("file_path")
                if fp:
                    file_rel, conf, method = repo_rel(fp), "high", "Read.file_path"
            elif name == "Grep":
                # Grep with a concrete `path` arg targeting a single file
                p = inp.get("path")
                if p and (Path(REPO) / repo_rel(p)).is_file():
                    file_rel, conf, method = repo_rel(p), "high", "Grep.path(file)"
                # else: directory/repo-wide grep — not a single-file access; skip
            elif name == "Glob":
                # Glob is a filename pattern, not a file read; skip unless it names a file
                pat = inp.get("pattern", "")
                if pat and not any(ch in pat for ch in "*?[]") and "." in os.path.basename(pat):
                    file_rel, conf, method = repo_rel(pat), "low", "Glob.pattern(file)"
            elif name == "Bash":
                tgt = bash_target_file(inp.get("command", ""))
                if tgt:
                    file_rel, conf, method = tgt, "low", "Bash(grep/cat/sed)"
            if file_rel:
                klass = "DUPLICATE" if file_rel in sx_returned_files else "NEW"
                accesses.append({
                    "tool": name, "file": file_rel, "class": klass,
                    "confidence": conf, "method": method,
                })

    return {
        "sx_call_count": len(sx_calls),
        "sx_returned_files": sorted(sx_returned_files),
        "post_sx_accesses": accesses,
        "n_post_sx": len(accesses),
        "n_dup": sum(1 for a in accesses if a["class"] == "DUPLICATE"),
        "n_new": sum(1 for a in accesses if a["class"] == "NEW"),
    }


def _run_arm(arm: str, raw_dir: Path, steered: bool):
    api_key = cb.load_api_key()
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY missing", file=sys.stderr)
        sys.exit(1)
    raw_dir.mkdir(parents=True, exist_ok=True)
    cdir = cb.arm_config_dir(arm)
    mcp_path = cdir / "mcp.json"
    mcp_path.write_text(json.dumps(cb.mcp_config_for(arm)))
    print(f"[{arm}] SEMANTEX_BIN={cb.SEMANTEX_BIN} steered={steered}", file=sys.stderr)
    print(f"[{arm}] mcp={mcp_path}: {mcp_path.read_text()}", file=sys.stderr)
    nudge = cb.nudge_for_arm(arm) if steered else None  # SEMANTEX_MD for steered arm
    for q in cb.QUESTIONS:
        for rep in range(1, REPS + 1):
            out_path = raw_dir / f"gin_{q['id']}_r{rep}.jsonl"
            if out_path.exists() and out_path.stat().st_size > 0:
                print(f"  [skip] {q['id']} r{rep} (exists)", file=sys.stderr)
                continue
            prompt = cb.PROMPT_TEMPLATE.format(question=q["question"])
            # Inject the repo CLAUDE.md nudge for the steered arm (cleaned up after),
            # mirroring claude_bench.run_claude. Only when the repo has no CLAUDE.md.
            md_path = Path(REPO) / "CLAUDE.md"
            injected = False
            if nudge and not md_path.exists():
                md_path.write_text(nudge)
                injected = True
            # NOTE: `claude -p` (non-interactive) DENIES MCP tools unless explicitly
            # allowed — verified: without this grant every semantex_agent call returns
            # "you haven't granted it yet" and the agent falls back to native reads
            # (confounding the substitution measurement entirely). We grant the
            # semantex MCP tools + native search tools so the agent can use semantex
            # AND still freely grep/read afterward (exactly the behavior we measure).
            cmd = [cb.CLAUDE_BIN, "-p", prompt,
                   "--output-format", "stream-json", "--verbose",
                   "--model", cb.args_model(),
                   "--strict-mcp-config", "--mcp-config", str(mcp_path),
                   "--disallowedTools", "Skill",
                   "--allowedTools",
                   "mcp__semantex__semantex_agent",
                   "mcp__semantex__semantex_search",
                   "mcp__semantex__semantex_deep",
                   "Read", "Grep", "Glob", "Bash"]
            env = {**os.environ, "ANTHROPIC_API_KEY": api_key,
                   "CLAUDE_CONFIG_DIR": str(cdir)}
            print(f"  [run] {q['id']} r{rep} …", end="", flush=True, file=sys.stderr)
            t0 = time.time()
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO,
                                   timeout=600, env=env)
            except subprocess.TimeoutExpired:
                print(" TIMEOUT", file=sys.stderr)
                if injected and md_path.exists():
                    md_path.unlink()
                continue
            finally:
                if injected and md_path.exists():
                    md_path.unlink()
            out_path.write_text(r.stdout)
            el = time.time() - t0
            p = parse_run(r.stdout)
            print(f" rc={r.returncode} {el:.0f}s sx={p['sx_call_count']} "
                  f"post_sx={p['n_post_sx']} dup={p['n_dup']} new={p['n_new']}",
                  file=sys.stderr)
            time.sleep(1)


def cmd_run():
    _run_arm(ARM, RAW, steered=False)
    cmd_parse()


def cmd_run_steered():
    _run_arm(STEERED_ARM, RAW_STEERED, steered=True)
    cmd_parse()


def _load_rows(raw_dir: Path, arm_label: str) -> list[dict]:
    rows = []
    for q in cb.QUESTIONS:
        for rep in range(1, REPS + 1):
            path = raw_dir / f"gin_{q['id']}_r{rep}.jsonl"
            if not path.exists() or path.stat().st_size == 0:
                continue
            p = parse_run(path.read_text())
            p.update({"q": q["id"], "type": q["type"], "rep": rep, "arm": arm_label})
            rows.append(p)
    return rows


def _pct(n, d):
    return (100.0 * n / d) if d else 0.0


def _agg(rows: list[dict]) -> dict:
    with_sx = [r for r in rows if r["sx_call_count"] >= 1]
    no_sx = [r for r in rows if r["sx_call_count"] == 0]
    tp = sum(r["n_post_sx"] for r in with_sx)
    td = sum(r["n_dup"] for r in with_sx)
    tn = sum(r["n_new"] for r in with_sx)
    hc_dup = hc_new = 0
    for r in with_sx:
        for a in r["post_sx_accesses"]:
            if a["confidence"] == "high":
                hc_dup += a["class"] == "DUPLICATE"
                hc_new += a["class"] == "NEW"
    return {"n": len(rows), "with_sx": len(with_sx), "no_sx": len(no_sx),
            "post": tp, "dup": td, "new": tn, "hc_dup": hc_dup, "hc_new": hc_new,
            "rows": rows, "with_sx_rows": with_sx}


def _section_for(report: list, title: str, a: dict):
    report.append(f"\n## {title}\n")
    report.append(f"- Runs captured: **{a['n']}**; called semantex ≥1×: **{a['with_sx']}**; "
                  f"called semantex 0×: **{a['no_sx']}**")
    mean_post = (a['post'] / a['with_sx']) if a['with_sx'] else 0.0
    report.append(f"- Post-semantex native file-accesses classified (in ≥1-sx runs): "
                  f"**{a['post']}** (mean {mean_post:.1f}/run)")
    report.append(f"- **DUPLICATE vs NEW (all conf): "
                  f"{a['dup']}/{a['post']} = {_pct(a['dup'], a['post']):.0f}% DUP · "
                  f"{a['new']}/{a['post']} = {_pct(a['new'], a['post']):.0f}% NEW**")
    hct = a['hc_dup'] + a['hc_new']
    report.append(f"- DUPLICATE vs NEW (HIGH-conf only — Read.file_path/Grep.path): "
                  f"{a['hc_dup']}/{hct} = {_pct(a['hc_dup'], hct):.0f}% DUP · "
                  f"{a['hc_new']}/{hct} = {_pct(a['hc_new'], hct):.0f}% NEW")
    report.append("\n| Q | type | runs w/ sx | runs 0-sx | post-sx | DUP | NEW | %DUP |")
    report.append("|---|---|---|---|---|---|---|---|")
    for q in cb.QUESTIONS:
        qr = [r for r in a["rows"] if r["q"] == q["id"]]
        qsx = [r for r in qr if r["sx_call_count"] >= 1]
        q0 = [r for r in qr if r["sx_call_count"] == 0]
        tp = sum(r["n_post_sx"] for r in qsx)
        td = sum(r["n_dup"] for r in qsx)
        tn = sum(r["n_new"] for r in qsx)
        report.append(f"| {q['id']} | {q['type']} | {len(qsx)} | {len(q0)} | {tp} | "
                      f"{td} | {tn} | {_pct(td, tp):.0f}% |")


def cmd_parse():
    bare = _load_rows(RAW, "bare")
    steered = _load_rows(RAW_STEERED, "steered")
    all_rows = bare + steered
    a_bare = _agg(bare)
    a_steer = _agg(steered)
    a_all = _agg(all_rows)
    # Combined sx-using-run substitution stats (the headline n)
    comb_post, comb_dup, comb_new = a_all["post"], a_all["dup"], a_all["new"]
    hct = a_all["hc_dup"] + a_all["hc_new"]

    report = []
    report.append("# Substitution-vs-recall probe — gin / LateOn-colbert (2026-06-04)\n")
    report.append(f"model: {cb.args_model()} · repo: gin · backend: colbert-plaid (LateOn-Code-edge) · "
                  f"SEMANTEX_ADAPTIVE_SIZING=0\n")
    report.append("## The question\n")
    report.append("A coding agent calls `semantex_agent` ~1/task and THEN still greps/reads natively. WHY?\n"
                  "- **DUPLICATE** (re-reads files semantex ALREADY returned) → TRUST/SUBSTITUTION problem "
                  "→ fix = tool-output/adoption work.\n"
                  "- **NEW** (reads files semantex did NOT return) → RETRIEVAL RECALL problem → fix = the "
                  "EMBEDDER.\n\nThe duplicate-vs-new ratio of post-semantex native file-accesses is the answer.\n")

    report.append("## VERDICT\n")
    if comb_post == 0:
        verdict = "INCONCLUSIVE — no post-semantex native file-accesses to classify."
    else:
        dpct = _pct(comb_dup, comb_post)
        if dpct >= 65:
            verdict = (f"**MOSTLY-DUPLICATE ({dpct:.0f}%)** → the agent re-reads what semantex already "
                       f"returned. This is a TRUST/SUBSTITUTION problem; the lever is tool-output / "
                       f"adoption work (make the agent trust + stop re-verifying), NOT the embedder.")
        elif dpct <= 35:
            verdict = (f"**MOSTLY-NEW ({100 - dpct:.0f}% NEW)** → the agent reads files semantex MISSED. "
                       f"This is a RETRIEVAL RECALL problem; the EMBEDDER is the bet.")
        else:
            verdict = (f"**MIXED ({dpct:.0f}% DUP / {100 - dpct:.0f}% NEW)** → both trust/substitution AND "
                       f"recall contribute; neither lever alone closes the gap.")
    report.append(verdict + "\n")
    report.append(f"Classification n (combined bare + steered): **{a_all['with_sx']} runs called "
                  f"semantex ≥1×**, yielding **{comb_post} post-sx native file-accesses** "
                  f"({comb_dup} DUP / {comb_new} NEW; HIGH-conf subset {hct}).\n")

    report.append("\n## BIG CAVEAT — adoption gap dominates the bare arm\n")
    report.append(f"In the **bare-MCP arm (no CLAUDE.md, the shipped regime)**, only "
                  f"**{a_bare['with_sx']}/{a_bare['n']} runs called semantex at all** — "
                  f"{a_bare['no_sx']}/{a_bare['n']} ignored the MCP tool and went straight to native "
                  f"grep/Read. So BEFORE the duplicate-vs-new question even applies, the agent mostly "
                  f"doesn't adopt the tool. The **steered arm** (SEMANTEX_MD nudge = 'use semantex as "
                  f"PRIMARY') is added purely to manufacture enough sx-using runs to answer the "
                  f"duplicate-vs-new question; its substitution ratio is the load-bearing number, the "
                  f"bare arm's near-zero adoption is the louder finding.\n")
    report.append("> Methodological note: `claude -p` (non-interactive) DENIES MCP tools unless "
                  "`--allowedTools mcp__semantex__*` is passed. The original `claude_bench.py` does NOT "
                  "pass it, so its bare-MCP semantex calls were being permission-denied (verified). This "
                  "probe grants the tool explicitly; the harness gap is flagged for follow-up.\n")

    _section_for(report, "Arm A — bare-MCP (no nudge; the shipped regime)", a_bare)
    _section_for(report, "Arm B — steered (SEMANTEX_MD nudge; manufactures sx usage)", a_steer)
    _section_for(report, "Combined (both arms — the substitution-ratio sample)", a_all)

    # per-run detail (both arms)
    report.append("\n## Per-run detail (all runs)\n")
    report.append("| arm | run | sx calls | sx files ret | post-sx | DUP | NEW |")
    report.append("|---|---|---|---|---|---|---|")
    for r in all_rows:
        report.append(f"| {r['arm']} | gin_{r['q']}_r{r['rep']} | {r['sx_call_count']} | "
                      f"{len(r['sx_returned_files'])} | {r['n_post_sx']} | "
                      f"{r['n_dup']} | {r['n_new']} |")

    # concrete examples
    report.append("\n## Concrete post-semantex access examples\n")
    ex = 0
    for r in a_all["with_sx_rows"]:
        for acc in r["post_sx_accesses"]:
            if ex >= 14:
                break
            in_set = ("YES — semantex DID return it" if acc["class"] == "DUPLICATE"
                      else "NO — semantex did NOT return it")
            report.append(f"- `[{r['arm']}] gin_{r['q']}_r{r['rep']}` — {acc['tool']} "
                          f"`{acc['file']}` → **{acc['class']}** "
                          f"({acc['method']}, conf={acc['confidence']}); {in_set}")
            ex += 1

    # parsing limitations
    report.append("\n## Parsing confidence & limitations\n")
    report.append("- **HIGH confidence (exact)**: `Read.file_path` is an exact path; `Grep` with a "
                  "`path` arg resolving to a real file is exact. The HIGH-conf ratio is the trustworthy "
                  "number; Read dominates the sample.\n")
    report.append("- **LOW confidence (best-effort, fuzzy)**: `Bash` grep/cat/sed/awk targets are "
                  "extracted heuristically from the command string (last file-looking token, preferring "
                  "one that exists in-repo); `Glob` patterns naming a literal file. May mis-attribute.\n")
    report.append("- **sx_returned_files** = the union of every `file:line` / `file:start-end` header "
                  "AND inline ref in the semantex_agent tool_result (including callers/callees mentions). "
                  "This is GENEROUS to DUPLICATE (a file merely named counts as 'returned'), so it "
                  "UNDER-counts NEW — a NEW-heavy result would be conservative/strong, and a "
                  "DUPLICATE-heavy result is the easier one to produce.\n")
    report.append("- Directory-/repo-wide Grep and Bash `ls`/`find` are NOT counted (no single target) "
                  "— exploration, not a specific-file read.\n")
    report.append("- The steered arm's nudge changes WHETHER the agent calls semantex, not WHAT it reads "
                  "afterward, so its post-sx accesses are valid for the duplicate-vs-new question.\n")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.md").write_text("\n".join(report) + "\n")
    (OUT / "parsed.json").write_text(json.dumps(all_rows, indent=2))
    print("\n".join(report))
    print(f"\n[written] {OUT/'report.md'}", file=sys.stderr)


if __name__ == "__main__":
    {"run": cmd_run, "run-steered": cmd_run_steered, "parse": cmd_parse}.get(
        sys.argv[1] if len(sys.argv) > 1 else "parse", cmd_parse)()
