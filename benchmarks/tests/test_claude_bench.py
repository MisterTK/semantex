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
