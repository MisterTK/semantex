#!/usr/bin/env python3
"""Full-body trust-bundle A/B (2026-06-04).

Tests the hypothesis from the substitution probe: returning the COMPLETE code
body inline (SEMANTEX_MCP_FULL_CODE=1) + an honest completeness marker makes the
agent stop RE-READING files semantex already cited (baseline: 37/37 post-sx reads
were DUPLICATES of returned files). The lever is the body, not a confidence score.

Two arms, STEERED (SEMANTEX_MD nudge so the agent reliably calls semantex — full
bodies only matter AFTER a call; the nudge is CONSTANT across arms so the only
difference is full_code, making the re-read delta cleanly attributable):

  off  = sx-lateon     (200-char previews + file:line pointer — the current default)
  on   = sx-full-code  (full bodies inline + [COMPLETE:…]/[top K of N …] marker)

Primary metric: post-semantex DUPLICATE re-reads per run (want it DOWN).
Guards:  NEW reads (recall — must stay ~0), CCB (the bigger response must net
         BELOW the re-reads it prevents), num_turns. Quality is a separate judged
         run if this shows a positive signal — a marker that suppresses a NEEDED
         read would surface there.

Usage:
    python3 benchmarks/probe_fullbody_ab.py run      # COSTS API tokens (~$10)
    python3 benchmarks/probe_fullbody_ab.py parse     # FREE — rebuild table from raw
Env: PATH must include target/release; macOS ORT_DYLIB_PATH; SEMANTEX_BINARY set.
"""
from __future__ import annotations

import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import claude_bench as cb  # noqa: E402
import probe_substitution as P  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "results" / "probe-fullbody-ab"
ARMS = [("off", "sx-lateon"), ("on", "sx-full-code")]


def _run() -> None:
    # Force the SEMANTEX_MD steer for BOTH arms (decouple from nudge_for_arm, which
    # only steers sx-lateon-steered). The nudge is identical across arms, so it
    # cannot confound the full_code delta — it only ensures the agent calls semantex.
    cb.nudge_for_arm = lambda _arm: cb.SEMANTEX_MD  # type: ignore[assignment]
    for label, arm in ARMS:
        raw_dir = OUT / f"raw_{label}"
        print(f"\n===== ARM {label} ({arm}) =====", file=sys.stderr)
        P._run_arm(arm, raw_dir, steered=True)


def _rows(label: str, arm: str) -> list[dict]:
    raw_dir = OUT / f"raw_{label}"
    rows = []
    for q in cb.QUESTIONS:
        for rep in range(1, P.REPS + 1):
            path = raw_dir / f"gin_{q['id']}_r{rep}.jsonl"
            if not path.exists() or path.stat().st_size == 0:
                continue
            raw = path.read_text()
            d = P.parse_run(raw)
            stream = cb.parse_claude_stream(raw)
            d.update({
                "label": label, "q": q["id"], "type": q["type"], "rep": rep,
                "ccb": stream.get("ccb", 0),
                "num_turns": stream.get("num_turns", 0),
                "native": stream.get("native_tool_calls", 0),
            })
            rows.append(d)
    return rows


def _agg(rows: list[dict]) -> dict:
    n = len(rows)
    sx_runs = [r for r in rows if r["sx_call_count"] > 0]
    ccb = [r["ccb"] for r in rows if r["ccb"]]
    return {
        "n": n,
        "adopt": f"{len(sx_runs)}/{n}",
        "sx_calls_mean": round(st.mean([r["sx_call_count"] for r in rows]), 2) if rows else 0,
        "post_sx_total": sum(r["n_post_sx"] for r in rows),
        "dup_total": sum(r["n_dup"] for r in rows),
        "new_total": sum(r["n_new"] for r in rows),
        "dup_per_run": round(sum(r["n_dup"] for r in rows) / n, 2) if n else 0,
        "ccb_mean": int(st.mean(ccb)) if ccb else 0,
        "turns_mean": round(st.mean([r["num_turns"] for r in rows]), 2) if rows else 0,
        "native_mean": round(st.mean([r["native"] for r in rows]), 2) if rows else 0,
    }


def _parse() -> None:
    a = {label: _agg(_rows(label, arm)) for label, arm in ARMS}
    off, on = a["off"], a["on"]
    cols = ["n", "adopt", "sx_calls_mean", "post_sx_total", "dup_total", "new_total",
            "dup_per_run", "ccb_mean", "turns_mean", "native_mean"]
    print("\n## Full-body trust-bundle A/B (gin, steered) — full_code OFF vs ON\n")
    print(f"| metric | off (previews) | on (full body) |")
    print(f"|---|---|---|")
    for c in cols:
        print(f"| {c} | {off[c]} | {on[c]} |")

    def pct(o, n):
        return "n/a" if not o else f"{(n - o) / o * 100:+.0f}%"
    print("\n### Deltas (on − off; negative = improvement on dup/CCB)")
    print(f"- DUPLICATE re-reads/run: {off['dup_per_run']} → {on['dup_per_run']}  ({pct(off['dup_per_run'], on['dup_per_run'])})")
    print(f"- CCB mean:              {off['ccb_mean']} → {on['ccb_mean']}  ({pct(off['ccb_mean'], on['ccb_mean'])})")
    print(f"- NEW (recall) reads:    {off['new_total']} → {on['new_total']}  (must stay low)")
    print(f"- turns mean:            {off['turns_mean']} → {on['turns_mean']}")
    print(f"- native calls mean:     {off['native_mean']} → {on['native_mean']}")
    print("\nVERDICT: full bodies help IFF dup_per_run drops materially AND new_total stays ~flat AND ccb_mean does not balloon.")
    OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "parse"
    if cmd == "run":
        _run()
        _parse()
    elif cmd == "parse":
        _parse()
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
