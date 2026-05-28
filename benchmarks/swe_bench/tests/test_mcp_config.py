from swe_bench_harness.conditions import Condition
from swe_bench_harness.tools.mcp_config import build_mcp_config


def _make_condition(**overrides) -> Condition:
    defaults = dict(
        id="c1_baseline",
        label="x",
        agent_model="claude-sonnet-4-6",
        semantex_enabled=False,
        semantex_features="",
        semantex_llm_features=False,
    )
    defaults.update(overrides)
    return Condition(**defaults)


def test_baseline_returns_none():
    c1 = _make_condition()
    assert build_mcp_config(c1) is None


def test_c2_no_llm_uses_default_binary(monkeypatch):
    monkeypatch.delenv("SEMANTEX_MCP_BINARY", raising=False)
    c2 = _make_condition(
        id="c2_semantex_no_llm",
        semantex_enabled=True,
        semantex_features="",
        semantex_llm_features=False,
    )
    cfg = build_mcp_config(c2)
    assert cfg is not None
    server = cfg["mcpServers"]["semantex"]
    assert server["command"] == "semantex"
    assert server["args"] == ["mcp"]
    # no LLM env vars when llm_features disabled
    env = server.get("env", {})
    assert "SEMANTEX_LLM_PROVIDER" not in env
    assert "SEMANTEX_LLM_MODEL" not in env


def test_c3_with_llm_uses_llm_binary_and_sets_env(monkeypatch):
    monkeypatch.delenv("SEMANTEX_LLM_BINARY", raising=False)
    c3 = _make_condition(
        id="c3_semantex_with_llm",
        semantex_enabled=True,
        semantex_features="llm",
        semantex_llm_features=True,
        semantex_llm_provider="anthropic",
        semantex_llm_model="claude-haiku-4-5-20251001",
        semantex_llm_fallback_provider="google",
        semantex_llm_fallback_model="gemini-2.5-flash",
    )
    cfg = build_mcp_config(c3)
    server = cfg["mcpServers"]["semantex"]
    assert server["command"] == "semantex-llm"
    assert server["args"] == ["mcp"]
    env = server["env"]
    assert env["SEMANTEX_LLM_PROVIDER"] == "anthropic"
    assert env["SEMANTEX_LLM_MODEL"] == "claude-haiku-4-5-20251001"
    assert env["SEMANTEX_LLM_FALLBACK_PROVIDER"] == "google"
    assert env["SEMANTEX_LLM_FALLBACK_MODEL"] == "gemini-2.5-flash"


def test_env_overrides_binary_path(monkeypatch):
    monkeypatch.setenv("SEMANTEX_BINARY", "/custom/path/to/semantex")
    c2 = _make_condition(
        id="c2_semantex_no_llm",
        semantex_enabled=True,
        semantex_features="",
        semantex_llm_features=False,
    )
    cfg = build_mcp_config(c2)
    assert cfg["mcpServers"]["semantex"]["command"] == "/custom/path/to/semantex"


def test_inherits_api_keys_when_set(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-anthropic")
    monkeypatch.setenv("GOOGLE_API_KEY", "g-test-google")
    monkeypatch.setenv("GEMINI_API_KEY", "g-test-gemini")
    c3 = _make_condition(
        id="c3_semantex_with_llm",
        semantex_enabled=True,
        semantex_features="llm",
        semantex_llm_features=True,
        semantex_llm_provider="anthropic",
        semantex_llm_model="claude-haiku-4-5",
    )
    cfg = build_mcp_config(c3)
    env = cfg["mcpServers"]["semantex"]["env"]
    assert env["ANTHROPIC_API_KEY"] == "sk-test-anthropic"
    assert env["GOOGLE_API_KEY"] == "g-test-google"
    assert env["GEMINI_API_KEY"] == "g-test-gemini"
