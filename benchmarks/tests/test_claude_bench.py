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
    assert cb.nudge_for_arm("sx-lateon") is None
    assert cb.nudge_for_arm("sx-coderank") is None
    assert cb.nudge_for_arm("graphify") == cb.GRAPHIFY_MD


def test_judge_skips_already_scored_and_is_resumable(tmp_path, monkeypatch):
    # The judge must be resumable: an already-scored answer is NOT re-judged (no wasted
    # API $ / survives a mid-judge crash), and its score is preserved. Hardens against
    # the claude-CLI-vanishes-mid-run failure.
    import types
    out = tmp_path / "out"; out.mkdir()
    results = [
        {"arm": "builtin", "question_id": "Q1", "answer": "a1", "rep": 1, "repo": "/r",
         "quality": 5, "quality_reason": "pre"},
        {"arm": "sx-lateon", "question_id": "Q1", "answer": "a2", "rep": 1, "repo": "/r"},
    ]
    (out / "all_results.json").write_text(cb.json.dumps(results))
    spawned = []

    class FakeProc:
        stdout = cb.json.dumps({"result": '{"score": 3, "reason": "ok"}'})

    monkeypatch.setattr(cb, "_spawn_claude", lambda cmd, **kw: spawned.append(cmd) or FakeProc())
    monkeypatch.setattr(cb, "load_api_key", lambda: "k")
    monkeypatch.setattr(cb, "arm_config_dir", lambda a: tmp_path)
    monkeypatch.setattr(cb, "BENCH_HOME", tmp_path)
    monkeypatch.setattr(cb, "QUESTIONS",
                        [{"id": "Q1", "question": "q", "type": "t", "bucket": "semantic"}])
    cb.cmd_judge(types.SimpleNamespace(input=str(out), judge_model="m"))
    assert len(spawned) == 1                # only the unscored answer was judged
    saved = cb.json.loads((out / "all_results.json").read_text())
    assert saved[0]["quality"] == 5         # preexisting score preserved (resume)
    assert saved[1]["quality"] == 3         # newly scored


def test_run_resume_skips_existing_and_keeps_them(tmp_path, monkeypatch):
    # --resume: a crashed run is completed by re-running with --resume; cells whose raw
    # file exists are NOT re-called (no wasted API $) but ARE loaded into all_results so
    # the final file is whole.
    import types
    out = tmp_path / "out"; (out / "raw").mkdir(parents=True)
    (tmp_path / "gin").mkdir()
    (out / "raw" / "gin_Q1_sx-lateon_r1.json").write_text(
        cb.json.dumps({"arm": "sx-lateon", "ccb": 42, "_marker": "preexisting"}))
    calls = []
    monkeypatch.setattr(cb, "run_single",
                        lambda q, repo, arm, rep, key: calls.append((q["id"], arm, rep)) or
                        {"arm": arm, "ccb": 1, "_marker": "fresh"})
    monkeypatch.setattr(cb, "load_api_key", lambda: "k")
    monkeypatch.setattr(cb, "QUESTIONS",
                        [{"id": "Q1", "type": "architecture", "bucket": "semantic", "question": "q"}])
    args = types.SimpleNamespace(model="m", repos=[str(tmp_path / "gin")], output=str(out),
                                 reps=2, arms=["sx-lateon"], resume=True)
    cb.cmd_run(args)
    assert ("Q1", "sx-lateon", 1) not in calls  # existing cell NOT re-run
    assert ("Q1", "sx-lateon", 2) in calls       # missing cell WAS run
    allr = cb.json.loads((out / "all_results.json").read_text())
    assert any(r.get("_marker") == "preexisting" for r in allr)  # existing loaded into final
    assert any(r.get("_marker") == "fresh" for r in allr)


def test_deep_lean_arm_for_clean_sweep():
    # sx-deep-lean = depth=deep (completes the answer in fewer turns) + budget=6000
    # (the lean budget that trims the deep payload below builtin CCB). The clean-sweep
    # candidate: beat builtin on quality AND CCB AND latency AND turns. Bare (no steer).
    assert "sx-deep-lean" in cb.SX_CONFIG_ARMS
    env = cb.SX_CONFIG_ARMS["sx-deep-lean"]
    assert env["SEMANTEX_MCP_DEPTH"] == "deep"
    assert env["SEMANTEX_MCP_BUDGET"] == cb.SX_CONFIG_ARMS["sx-budget-low"]["SEMANTEX_MCP_BUDGET"]
    assert env["SEMANTEX_EMBEDDER"] == "lateon-colbert"
    assert cb.nudge_for_arm("sx-deep-lean") is None  # bare, not steered


def test_steered_diagnostic_arm_isolates_adoption_gap():
    # sx-lateon-steered is the labeled adoption-gap diagnostic: SAME embedder/env as
    # sx-lateon (so the only difference is the steer), but it DOES get the SEMANTEX_MD
    # nudge. The steered-minus-bare CCB/quality delta = the adoption gap. Never a headline.
    assert "sx-lateon-steered" in cb.SX_CONFIG_ARMS
    assert cb.SX_CONFIG_ARMS["sx-lateon-steered"] == cb.SX_CONFIG_ARMS["sx-lateon"]
    assert cb.nudge_for_arm("sx-lateon-steered") == cb.SEMANTEX_MD
    assert cb.nudge_for_arm("sx-lateon") is None  # the bare half of the delta
    assert cb.is_semantex_arm("sx-lateon-steered")  # MCP allowed, not blocked like builtin


def test_report_aggregates_all_arms_three_axes(tmp_path):
    import json
    rows = []
    for arm, ccb, q, secs in [("builtin", 100000, 4.0, 90), ("sx-lateon", 60000, 4.2, 45),
                              ("sx-coderank", 65000, 4.1, 80)]:
        for _rep in range(2):
            rows.append({"arm": arm, "ccb": ccb, "quality": q, "wall_secs": secs,
                         "num_turns": 5, "peak_context": ccb, "tool_calls": 3,
                         "cost_usd": 0.1, "caf": 1.5, "bucket": "semantic",
                         "question_id": "Q1", "question_type": "architecture"})
    (tmp_path / "all_results.json").write_text(json.dumps(rows))
    table = cb.pareto_table(rows)
    assert set(table) == {"builtin", "sx-lateon", "sx-coderank"}
    assert table["sx-lateon"]["ccb"] == 60000
    assert table["sx-lateon"]["quality"] == 4.2
    assert table["sx-lateon"]["wall_secs"] == 45
    assert table["sx-lateon"]["n"] == 2


def test_pareto_table_skips_error_rows():
    rows = [{"arm": "sx-lateon", "ccb": 60000, "quality": 4.0, "wall_secs": 40, "num_turns": 4},
            {"arm": "sx-lateon", "error": "empty"}]
    t = cb.pareto_table(rows)
    assert t["sx-lateon"]["n"] == 1  # error row excluded


def test_plain_semantex_arm_has_no_env_key():
    # bare-vs-env boundary: a config-arm emits env; plain `semantex` must NOT.
    assert "env" not in cb.mcp_config_for("semantex")["mcpServers"]["semantex"]
    assert "env" in cb.mcp_config_for("sx-coderank")["mcpServers"]["semantex"]


def test_embedders_for_arms():
    embs = cb.embedders_for_arms(["builtin", "sx-lateon", "sx-coderank", "sx-stacked"])
    assert embs == ["lateon-colbert", "coderank-137m"]  # dedup, order-stable
    assert cb.embedders_for_arms(["builtin"]) == []
    assert cb.embedders_for_arms(["sx-budget-low"]) == ["lateon-colbert"]


def test_assert_ready_false_without_meta(tmp_path):
    assert cb._assert_ready(str(tmp_path), "lateon-colbert") is False


def test_parse_stream_records_tool_names():
    import json
    # minimal claude stream-json: two assistant turns w/ tool_use blocks + a result.
    events = [
        {"type": "assistant", "message": {"usage": {"input_tokens": 100, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
            "content": [
                {"type": "tool_use", "name": "mcp__semantex__semantex_agent"},
                {"type": "tool_use", "name": "Grep"},
                {"type": "tool_use", "name": "Read"}]}},
        {"type": "assistant", "message": {"usage": {"input_tokens": 100, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
            "content": [{"type": "tool_use", "name": "mcp__semantex__semantex_agent"}]}},
        {"type": "result", "result": "done", "num_turns": 2},
    ]
    raw = "\n".join(json.dumps(e) for e in events)
    m = cb.parse_claude_stream(raw)
    assert m["tool_calls"] == 4
    assert m["tool_calls_by_name"]["mcp__semantex__semantex_agent"] == 2
    assert m["tool_calls_by_name"]["Grep"] == 1
    assert m["tool_calls_by_name"]["Read"] == 1
    assert m["sx_tool_calls"] == 2          # semantex MCP calls
    assert m["native_tool_calls"] == 2      # Grep + Read


def test_pareto_table_includes_tool_usage():
    rows = [{"arm": "sx-lateon", "ccb": 60000, "quality": 4.0, "wall_secs": 40,
             "num_turns": 4, "sx_tool_calls": 2, "native_tool_calls": 6},
            {"arm": "builtin", "ccb": 100000, "quality": 4.0, "wall_secs": 80,
             "num_turns": 5, "sx_tool_calls": 0, "native_tool_calls": 20}]
    t = cb.pareto_table(rows)
    assert t["sx-lateon"]["sx_tool_calls"] == 2
    assert t["sx-lateon"]["native_tool_calls"] == 6
    assert t["builtin"]["sx_tool_calls"] == 0
