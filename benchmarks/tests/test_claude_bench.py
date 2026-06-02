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
