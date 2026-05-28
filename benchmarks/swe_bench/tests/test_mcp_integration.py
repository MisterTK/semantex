"""Validate build_mcp_config() output against the schema OpenHands actually uses.

OpenHands' Agent constructor takes mcp_config as `dict[str, Any]` (loose),
but openhands.sdk.mcp.utils.create_mcp_tools() validates it via
fastmcp.mcp_config.MCPConfig.model_validate(...). That is the real contract.

These tests feed our dict through MCPConfig and assert it parses into a
StdioMCPServer with the fields we set. We do NOT spawn semantex-mcp; that's
a Phase 7 operator-handoff prerequisite.
"""
from __future__ import annotations

import pytest
from fastmcp.mcp_config import MCPConfig, StdioMCPServer

from swe_bench_harness.conditions import Condition
from swe_bench_harness.tools.mcp_config import build_mcp_config


def _c3() -> Condition:
    return Condition(
        id="c3_semantex_with_llm",
        label="x",
        agent_model="claude-sonnet-4-6",
        semantex_enabled=True,
        semantex_features="llm",
        semantex_llm_features=True,
        semantex_llm_provider="anthropic",
        semantex_llm_model="claude-haiku-4-5-20251001",
        semantex_llm_fallback_provider="google",
        semantex_llm_fallback_model="gemini-2.5-flash",
    )


def test_c3_dict_validates_as_fastmcp_stdio_server(monkeypatch):
    """Our dict must parse cleanly through the same schema OpenHands uses."""
    monkeypatch.delenv("SEMANTEX_LLM_BINARY", raising=False)
    cfg_dict = build_mcp_config(_c3())
    assert cfg_dict is not None

    validated = MCPConfig.model_validate(cfg_dict)

    server = validated.mcpServers["semantex"]
    assert isinstance(server, StdioMCPServer), (
        f"expected stdio transport, got {type(server).__name__}"
    )
    assert server.command == "semantex-llm"
    assert server.args == ["mcp"]
    assert server.env["SEMANTEX_LLM_PROVIDER"] == "anthropic"
    assert server.env["SEMANTEX_LLM_MODEL"] == "claude-haiku-4-5-20251001"


def test_baseline_no_mcp_servers_to_validate():
    """C1 returns None — nothing to feed the schema, which is the correct
    no-MCP path."""
    c1 = Condition(
        id="c1_baseline",
        label="x",
        agent_model="claude-sonnet-4-6",
        semantex_enabled=False,
        semantex_features="",
        semantex_llm_features=False,
    )
    assert build_mcp_config(c1) is None


def test_schema_rejects_missing_command():
    """Negative control: confirm the schema we're validating against is strict
    enough to catch a missing 'command' — otherwise the positive test above
    would be meaningless."""
    bad = {"mcpServers": {"semantex": {"args": []}}}
    with pytest.raises(Exception):
        MCPConfig.model_validate(bad)
