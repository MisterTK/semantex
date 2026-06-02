# Bare-MCP Harness (Workstream B) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Extend `benchmarks/claude_bench.py` so the whole-system Pareto sweep (Workstream C) can run: per-arm `SEMANTEX_*` config-arms (env on the MCP server), measured **bare-MCP** (no CLAUDE.md/`SEMANTEX_MD` nudge — the product any client gets), an **arm-agnostic 3-way Pareto report** (CCB × quality × latency), and **setup discipline** (per-embedder pre-index + daemon-stop + Ready-assert). From spec `docs/superpowers/specs/2026-06-02-whole-system-pareto-tuning-design.md` §5. (Workstream A — the env-tunable `SEMANTEX_MCP_*` defaults the server-default arms need — is already merged.)

**Architecture:** Benchmark-only, one file `benchmarks/claude_bench.py` (stdlib-only, run via `python3`). Three tasks, each a self-contained change + tests. New tests live in `benchmarks/tests/test_claude_bench.py` (pytest 8.3.5 on system `python3`; mirrors `benchmarks/swe_bench/tests/` style — import the module, test pure helpers). Lands as one PR `feat/bare-mcp-harness` (benchmark-only; no `cargo` gate — but DO run `python3 -m pytest benchmarks/tests/test_claude_bench.py` + `python3 benchmarks/claude_bench.py --help`).

**Tech Stack:** Python 3.12 stdlib (argparse/json/subprocess/statistics). The MCP config JSON forwards an `env` dict to the spawned `semantex mcp` process (Claude Code MCP feature). `semantex` CLI for index/stop.

## Verified anchors (current `benchmarks/claude_bench.py`)
- `ARMS = ("builtin","graphify","semantex")` (line 73). `SEMANTEX_BIN` (71). `BENCH_HOME` (76). 5 `QUESTIONS` (80). `SEMANTEX_MD`/`GRAPHIFY_MD` nudges (108/115).
- `mcp_config_for(arm)` (151): semantex → `{"mcpServers":{"semantex":{"command":SEMANTEX_BIN,"args":["mcp"]}}}` (NO env). `write_arm_config` (158): clean `{"hooks":{}}` config dir. `cmd_setup` (192): write_arm_config + graphify graphs.
- `run_claude` (266): writes mcp.json from `mcp_config_for`; **injects the CLAUDE.md nudge** into the repo (272-277, `nudge = {"semantex":SEMANTEX_MD,"graphify":GRAPHIFY_MD}.get(arm)`, cleaned up in `finally` 302-303); arm-specific `--disallowedTools` (`builtin`→Skill+mcp; `semantex`→Skill).
- `parse_claude_stream` (211): real CCB (`ccb`), `peak_context`, `num_turns`, `caf`, `cost_usd`, `duration_ms`, `is_error`. `run_single` (317) adds `wall_secs` (latency) + bucket/question. `cmd_run` (335): repo×question×arm×rep loop.
- `cmd_judge` writes `r["quality"]` (1-5) into `all_results.json` (425). `cmd_report` (441): `agg(arm,field,bucket)` + `row()` HARDCODE `builtin/graphify/semantex` (457/477) — has CCB/quality/wall_secs rows already, but only 3 fixed arms.
- argparse (490): `setup`/`run` have `--arms ... choices=ARMS` (496/505) — must accept config-arm names.

---

## Task 1: Config-arm registry + per-arm env seam + bare-MCP

**Files:** Modify `benchmarks/claude_bench.py`. Create `benchmarks/tests/test_claude_bench.py`.

- [ ] **Step 1: Write failing tests** — create `benchmarks/tests/test_claude_bench.py`:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # import claude_bench
import claude_bench as cb


def test_sx_config_arms_registry_has_core_arms():
    for arm in ("sx-lateon", "sx-coderank", "sx-graph2hop", "sx-adaptive-off", "sx-stacked"):
        assert arm in cb.SX_CONFIG_ARMS, f"missing arm {arm}"
    assert cb.SX_CONFIG_ARMS["sx-coderank"]["SEMANTEX_EMBEDDER"] == "coderank-137m"
    assert cb.SX_CONFIG_ARMS["sx-adaptive-off"]["SEMANTEX_ADAPTIVE_SIZING"] == "0"
    assert cb.SX_CONFIG_ARMS["sx-graph2hop"]["SEMANTEX_GRAPH_HOPS"] == "2"


def test_is_semantex_arm():
    assert cb.is_semantex_arm("semantex")
    assert cb.is_semantex_arm("sx-lateon")
    assert cb.is_semantex_arm("sx-coderank")
    assert not cb.is_semantex_arm("builtin")
    assert not cb.is_semantex_arm("graphify")


def test_mcp_config_for_config_arm_emits_env():
    cfg = cb.mcp_config_for("sx-coderank")
    sx = cfg["mcpServers"]["semantex"]
    assert sx["command"] == cb.SEMANTEX_BIN and sx["args"] == ["mcp"]
    assert sx["env"]["SEMANTEX_EMBEDDER"] == "coderank-137m"


def test_mcp_config_for_plain_semantex_and_builtin():
    assert cb.mcp_config_for("semantex")["mcpServers"]["semantex"]["args"] == ["mcp"]
    assert cb.mcp_config_for("builtin")["mcpServers"] == {}


def test_nudge_for_arm_is_bare_for_config_arms():
    # bare-MCP: config-arms get NO CLAUDE.md nudge (the product as any client gets it).
    assert cb.nudge_for_arm("sx-lateon") is None
    assert cb.nudge_for_arm("sx-coderank") is None
    assert cb.nudge_for_arm("graphify") == cb.GRAPHIFY_MD  # graphify arm unchanged
```

- [ ] **Step 2: Run; expect FAIL** (`SX_CONFIG_ARMS`/`is_semantex_arm`/`nudge_for_arm` undefined):

Run: `cd /Users/tk/dev/qgrep/semantex && python3 -m pytest benchmarks/tests/test_claude_bench.py -q`
Expected: errors — `module 'claude_bench' has no attribute 'SX_CONFIG_ARMS'`.

- [ ] **Step 3: Add the registry + helpers** (after `ARMS = ...`, ~line 74):

```python
# Whole-system tuning config-arms: each is a semantex MCP arm + a dict of
# SEMANTEX_* env applied to the `semantex mcp` process (the in-process server
# reads them at construction). Measured BARE-MCP (no CLAUDE.md nudge). The
# server-default arms (budget/full_code/depth) rely on Workstream-A env knobs.
# All set SEMANTEX_EMBEDDER explicitly (lateon-colbert is the shipped default;
# coderank-137m is the opt-in). graph/adaptive knobs are query-time.
SX_CONFIG_ARMS: dict[str, dict[str, str]] = {
    "sx-lateon":       {"SEMANTEX_EMBEDDER": "lateon-colbert"},
    "sx-coderank":     {"SEMANTEX_EMBEDDER": "coderank-137m"},
    "sx-graph2hop":    {"SEMANTEX_EMBEDDER": "lateon-colbert",
                        "SEMANTEX_GRAPH_HOPS": "2", "SEMANTEX_GRAPH_CENTRALITY_WEIGHT": "0.2"},
    "sx-adaptive-off": {"SEMANTEX_EMBEDDER": "lateon-colbert", "SEMANTEX_ADAPTIVE_SIZING": "0"},
    "sx-stacked":      {"SEMANTEX_EMBEDDER": "lateon-colbert", "SEMANTEX_GRAPH_HOPS": "2",
                        "SEMANTEX_GRAPH_CENTRALITY_WEIGHT": "0.2", "SEMANTEX_ADAPTIVE_SIZING": "0"},
    # server-default arms (Workstream C sub-sweep 2; all on the lateon index):
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
    """The repo CLAUDE.md nudge for an arm. BARE-MCP: config-arms get NONE — the
    tool descriptions must carry their own weight. Only the legacy `semantex`/
    `graphify` arms keep a nudge (back-compat with the old head-to-head)."""
    return {"semantex": SEMANTEX_MD, "graphify": GRAPHIFY_MD}.get(arm)
```

- [ ] **Step 4: Wire `env` into `mcp_config_for`** (replace lines 151-155):

```python
def mcp_config_for(arm: str) -> dict:
    """Strict MCP set per arm. builtin/graphify get none; semantex + every sx-*
    config-arm get the semantex MCP server, with the config-arm's SEMANTEX_* env
    forwarded to the `semantex mcp` process (the in-process server reads it once
    at construction)."""
    if is_semantex_arm(arm):
        server = {"command": SEMANTEX_BIN, "args": ["mcp"]}
        env = SX_CONFIG_ARMS.get(arm)
        if env:
            server["env"] = dict(env)
        return {"mcpServers": {"semantex": server}}
    return {"mcpServers": {}}
```

- [ ] **Step 5: Use the helpers in `run_claude`** — (a) the nudge (replace line 274 `nudge = {...}.get(arm)`) with `nudge = nudge_for_arm(arm)`; (b) the disallowedTools (replace the `elif arm == "semantex":` at line 288) with `elif is_semantex_arm(arm):`. Leave `write_arm_config` as-is (config-arms reuse the clean hermetic dir; a config-arm's `arm_config_dir(arm)` is created in setup).

- [ ] **Step 6: Generalize `--arms` validation** — in argparse, change `setup` and `run` `--arms` from `choices=ARMS` to accept config-arm names. Replace `choices=ARMS` (lines 496 + 505) with `choices=all_arm_names()` and update the default for `run` if desired (keep `default=list(ARMS)`).

- [ ] **Step 7: Run tests + smoke**

Run: `python3 -m pytest benchmarks/tests/test_claude_bench.py -q && python3 benchmarks/claude_bench.py --help`
Expected: all tests PASS; `--help` shows the subcommands without error.

- [ ] **Step 8: Commit**

```bash
git add benchmarks/claude_bench.py benchmarks/tests/test_claude_bench.py
git commit -m "feat(bench): config-arm env seam + bare-MCP (no CLAUDE.md nudge for sx-* arms)"
```

---

## Task 2: Arm-agnostic 3-way Pareto report

`cmd_report` hardcodes 3 arms; generalize it to report EVERY arm present in the data on the 3 Pareto axes (CCB, quality, latency=wall_secs), + a Pareto-ranking note. Keep the existing builtin baseline % for reference.

**Files:** Modify `benchmarks/claude_bench.py` (`cmd_report` ~441-485). Test in `test_claude_bench.py`.

- [ ] **Step 1: Failing test** (append to `test_claude_bench.py`):

```python
def test_report_aggregates_all_arms_three_axes(tmp_path):
    # synthetic all_results.json with config-arms; report must aggregate each arm
    # on ccb / quality / wall_secs (the 3 Pareto axes), not just the fixed 3 arms.
    import json
    rows = []
    for arm, ccb, q, secs in [("builtin", 100000, 4.0, 90), ("sx-lateon", 60000, 4.2, 45),
                              ("sx-coderank", 65000, 4.1, 80)]:
        for rep in range(2):
            rows.append({"arm": arm, "ccb": ccb, "quality": q, "wall_secs": secs,
                         "num_turns": 5, "peak_context": ccb, "tool_calls": 3,
                         "cost_usd": 0.1, "caf": 1.5, "bucket": "semantic",
                         "question_id": "Q1", "question_type": "architecture"})
    (tmp_path / "all_results.json").write_text(json.dumps(rows))
    table = cb.pareto_table(rows)  # pure fn: {arm: {ccb, quality, wall_secs, n}}
    assert set(table) == {"builtin", "sx-lateon", "sx-coderank"}
    assert table["sx-lateon"]["ccb"] == 60000
    assert table["sx-lateon"]["quality"] == 4.2
    assert table["sx-lateon"]["wall_secs"] == 45
    assert table["sx-lateon"]["n"] == 2
```

- [ ] **Step 2: Run; expect FAIL** (`pareto_table` undefined). `python3 -m pytest benchmarks/tests/test_claude_bench.py::test_report_aggregates_all_arms_three_axes -q`

- [ ] **Step 3: Add the pure `pareto_table` helper** (near `_mean`, ~line 436):

```python
def pareto_table(results: list[dict]) -> dict[str, dict]:
    """Per-arm means of the 3 Pareto axes (CCB, quality, latency) + count, over
    valid (non-error) rows. Arm-agnostic — covers builtin + every sx-* arm."""
    valid = [r for r in results if not r.get("error")]
    arms = []
    for r in valid:
        if r["arm"] not in arms:
            arms.append(r["arm"])
    out = {}
    for arm in arms:
        rs = [r for r in valid if r["arm"] == arm]
        out[arm] = {
            "ccb": round(_mean([r.get("ccb") for r in rs])),
            "quality": round(_mean([r.get("quality") for r in rs]), 2),
            "wall_secs": round(_mean([r.get("wall_secs") for r in rs])),
            "turns": round(_mean([r.get("num_turns") for r in rs]), 1),
            "n": len(rs),
        }
    return out
```

- [ ] **Step 4: Run; expect PASS.** `python3 -m pytest benchmarks/tests/test_claude_bench.py -q`

- [ ] **Step 5: Rewrite `cmd_report` to use `pareto_table`** (replace the body 441-485). Emit a dynamic per-arm Pareto table (one row per arm: CCB / quality / latency / turns / n), with the first arm (builtin if present, else the first) as the % reference, plus the existing bucket split made arm-agnostic:

```python
def cmd_report(args):
    inp = Path(args.input)
    results = json.loads((inp / "all_results.json").read_text())
    table = pareto_table(results)
    if not table:
        print("no valid results"); return
    arms = list(table)
    ref = "builtin" if "builtin" in table else arms[0]

    lines = ["# Bare-MCP system Pareto: " + " vs ".join(arms) + "\n",
             "## 3-way Pareto (mean per question-run; lower CCB/latency better, higher quality better)\n",
             "| arm | quality (1-5) | CCB | latency s | turns | n | CCB vs " + ref + " |",
             "|---|---|---|---|---|---|---|"]
    for arm in arms:
        t = table[arm]
        rc = table[ref]["ccb"]
        d = "ref" if arm == ref else ("N/A" if not rc else f"{(t['ccb']/rc - 1)*100:+.0f}%")
        lines.append(f"| {arm} | {t['quality']:.2f} | {t['ccb']:,} | {t['wall_secs']} "
                     f"| {t['turns']} | {t['n']} | {d} |")

    # CCB by bucket, arm-agnostic
    valid = [r for r in results if not r.get("error")]
    lines += ["\n## CCB by bucket (semantic vs structural)\n",
              "| bucket | " + " | ".join(arms) + " |",
              "|---|" + "|".join("---" for _ in arms) + "|"]
    for bucket in ("semantic", "structural"):
        cells = []
        for arm in arms:
            rs = [r for r in valid if r["arm"] == arm and r.get("bucket") == bucket]
            cells.append(f"{round(_mean([r.get('ccb') for r in rs])):,}")
        lines.append(f"| {bucket} | " + " | ".join(cells) + " |")

    lines.append("\n> Pareto: an arm dominates if it is >= on quality AND <= on CCB AND "
                 "<= on latency. Read the table for the frontier; ties keep the current default.\n")
    report = "\n".join(lines) + "\n"
    (inp / "report.md").write_text(report)
    print(report)
```

- [ ] **Step 6: Run tests + a report smoke** on a synthetic dir:

```bash
python3 -m pytest benchmarks/tests/test_claude_bench.py -q
mkdir -p /tmp/cbtest && python3 -c "import json,sys; sys.path.insert(0,'benchmarks'); \
json.dump([{'arm':'builtin','ccb':100000,'quality':4.0,'wall_secs':90,'num_turns':5,'bucket':'semantic'},\
{'arm':'sx-lateon','ccb':60000,'quality':4.2,'wall_secs':45,'num_turns':4,'bucket':'semantic'}], open('/tmp/cbtest/all_results.json','w'))"
python3 benchmarks/claude_bench.py report --input /tmp/cbtest
```
Expected: a markdown table with both arms, the 3 axes, CCB-vs-builtin %.

- [ ] **Step 7: Commit** — `git commit -am "feat(bench): arm-agnostic 3-way Pareto report (CCB x quality x latency)"`

---

## Task 3: Setup-discipline preflight (per-embedder index + daemon-stop + Ready-assert)

The sweep is invalid unless each (repo × embedder) is pre-indexed and `Ready` (else the agent path falls to ripgrep + a background rebuild, polluting CCB/quality). Add a `preflight` that: stops any daemon per repo; derives the embedders from the selected arms; pre-indexes each repo per embedder; asserts each (repo × embedder) is Ready (no ripgrep fallback) via a free `semantex --dense-only --json` probe.

**Files:** Modify `benchmarks/claude_bench.py` (helpers + a `preflight` subcommand). Test the pure embedder-derivation in `test_claude_bench.py`.

- [ ] **Step 1: Failing test:**

```python
def test_embedders_for_arms():
    # builtin/graphify contribute nothing; sx-* contribute their SEMANTEX_EMBEDDER.
    embs = cb.embedders_for_arms(["builtin", "sx-lateon", "sx-coderank", "sx-stacked"])
    assert embs == ["lateon-colbert", "coderank-137m"]  # dedup, order-stable
    assert cb.embedders_for_arms(["builtin"]) == []
```

- [ ] **Step 2: Run; expect FAIL.** `python3 -m pytest benchmarks/tests/test_claude_bench.py::test_embedders_for_arms -q`

- [ ] **Step 3: Add `embedders_for_arms` + the preflight helpers + subcommand:**

```python
def embedders_for_arms(arms: list[str]) -> list[str]:
    """Distinct SEMANTEX_EMBEDDER values the given config-arms need pre-indexed
    (order-stable, deduped). builtin/graphify/plain-semantex contribute nothing
    (plain `semantex` uses whatever index already exists)."""
    out: list[str] = []
    for arm in arms:
        emb = SX_CONFIG_ARMS.get(arm, {}).get("SEMANTEX_EMBEDDER")
        if emb and emb not in out:
            out.append(emb)
    return out


def _stop_daemon(repo: str) -> None:
    subprocess.run([SEMANTEX_BIN, "stop", "."], cwd=repo, capture_output=True, text=True)


def _index_repo(repo: str, embedder: str, timeout: int = 7200) -> bool:
    env = {**os.environ, "SEMANTEX_EMBEDDER": embedder, "SEMANTEX_QUIET_LIMITS": "1"}
    eprint(f"  index {Path(repo).name} [{embedder}] …", end="", flush=True)
    r = subprocess.run([SEMANTEX_BIN, "index", "."], cwd=repo, capture_output=True,
                       text=True, env=env, timeout=timeout)
    ok = r.returncode == 0
    eprint(" ok" if ok else f" FAILED rc={r.returncode} {r.stderr[:160]}")
    return ok


def _assert_ready(repo: str, embedder: str) -> bool:
    """Free dry-run: a dense-only search under the arm's embedder env must return
    JSON results (Ready), NOT ripgrep fallback. Returns True if the index is live."""
    env = {**os.environ, "SEMANTEX_EMBEDDER": embedder, "SEMANTEX_QUIET_LIMITS": "1"}
    r = subprocess.run([SEMANTEX_BIN, "--dense-only", "--json", "--no-content", "-m", "3",
                        "--", "function"], cwd=repo, capture_output=True, text=True, env=env)
    try:
        data = json.loads(r.stdout) if r.stdout.strip() else []
    except json.JSONDecodeError:
        data = None
    ready = isinstance(data, list)  # dense-only JSON array == the dense index served
    eprint(f"  ready[{embedder}] {Path(repo).name}: {'YES' if ready else 'NO (rebuild/fallback)'}")
    return ready


def cmd_preflight(args):
    arms = getattr(args, "arms", list(SX_CONFIG_ARMS))
    embs = embedders_for_arms(arms)
    if not embs:
        eprint("No sx-* arms selected — nothing to pre-index."); return
    eprint(f"Preflight: embedders {embs} on {len(args.repos)} repos (free).")
    all_ok = True
    for repo in args.repos:
        _stop_daemon(repo)  # so the MCP in-process config isn't shadowed by a stale daemon
        for emb in embs:
            if not _index_repo(repo, emb):
                all_ok = False; continue
            if not _assert_ready(repo, emb):
                all_ok = False
        _stop_daemon(repo)  # leave no daemon running into the bench
    eprint("Preflight " + ("OK — all (repo x embedder) Ready." if all_ok
                           else "INCOMPLETE — some indexes not Ready; fix before `run`."))
    if not all_ok:
        sys.exit(2)
```

- [ ] **Step 4: Wire the `preflight` subcommand** into argparse (after the `setup` parser, ~line 498):

```python
    pf = sub.add_parser("preflight", help="stop daemons + pre-index each repo per embedder + assert Ready (free)")
    pf.add_argument("--repos", nargs="+", required=True)
    pf.add_argument("--arms", nargs="+", default=list(SX_CONFIG_ARMS), choices=all_arm_names())
```
and add `"preflight": cmd_preflight` to the dispatch dict (line 517).

- [ ] **Step 5: Run tests + a `--help`/dry smoke** (do NOT actually index a big repo in the test — just confirm the command parses + the pure fn works):

```bash
python3 -m pytest benchmarks/tests/test_claude_bench.py -q
python3 benchmarks/claude_bench.py preflight --help
```
Expected: tests green; `preflight --help` shows `--repos`/`--arms`.

- [ ] **Step 6: Commit** — `git commit -am "feat(bench): preflight — per-embedder pre-index + daemon-stop + Ready-assert"`

---

## Task 4: Full gate + land Workstream B

- [ ] **Step 1: Full benchmark-test gate**
```bash
python3 -m pytest benchmarks/tests/test_claude_bench.py -v
python3 benchmarks/claude_bench.py --help
python3 benchmarks/claude_bench.py setup --help && python3 benchmarks/claude_bench.py run --help && python3 benchmarks/claude_bench.py preflight --help
# sanity: a config-arm produces an env block + bare nudge
python3 -c "import sys;sys.path.insert(0,'benchmarks');import claude_bench as c; \
print(c.mcp_config_for('sx-coderank')['mcpServers']['semantex']['env']); \
print('nudge', c.nudge_for_arm('sx-lateon'))"
```
Expected: all green; the config-arm prints `{'SEMANTEX_EMBEDDER': 'coderank-137m'}` and `nudge None`.

- [ ] **Step 2: `actually verify_change`** (benchmark-only diff; expect verified/attention, no broken).

- [ ] **Step 3: Land** — PR `feat/bare-mcp-harness`, rebase-merge after green. This UNBLOCKS Workstream C (the sweep).

---

## Self-review notes
- **Spec §5 coverage:** §5.1 bare-MCP (drop nudge) → Task 1 (`nudge_for_arm` returns None for config-arms); §5.2 per-arm env seam → Task 1 (`SX_CONFIG_ARMS` + `mcp_config_for` env); §5.3 3-way Pareto report → Task 2; §5.4 setup discipline → Task 3 (`preflight`). All covered.
- **Type consistency:** `SX_CONFIG_ARMS: dict[str,dict[str,str]]`, `is_semantex_arm(str)->bool`, `nudge_for_arm(str)->str|None`, `pareto_table(list[dict])->dict[str,dict]`, `embedders_for_arms(list[str])->list[str]` — used identically across tasks.
- **No cargo gate** (Python-only). DO run `python3 -m pytest benchmarks/tests/test_claude_bench.py`. The subprocess-heavy paths (`_index_repo`/`_assert_ready`/`run_claude`) are NOT unit-tested (they shell out); the pure helpers are. A real preflight+run is exercised in Workstream C.
- **Out of scope:** Workstream C (running the sweep, analysis, the gated default-flip PRs) — its own effort once B lands. The `sx-rerank` arm is intentionally NOT in `SX_CONFIG_ARMS` (precondition-gated; add it in C only if qwen3-reranker loads).
