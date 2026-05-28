"""Build OpenHands mcp_config dict from a Condition.

We register the existing crates/semantex-mcp server rather than building a
custom OpenHands Tool. The Condition selects which binary (default vs
--features llm build) and whether to pass LLM env vars.

Shape contract: OpenHands' Agent.mcp_config is typed `dict[str, Any]` but is
validated by `fastmcp.mcp_config.MCPConfig.model_validate(...)` inside
openhands.sdk.mcp.utils.create_mcp_tools — see tests/test_mcp_integration.py.
"""
from __future__ import annotations

import os

from swe_bench_harness.conditions import Condition


_INHERIT_KEYS = ("SEMANTEX_MAX_RSS_MB", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY")


def build_mcp_config(condition: Condition) -> dict | None:
    """Return an OpenHands mcp_config dict, or None if semantex is disabled."""
    if not condition.semantex_enabled:
        return None

    if condition.semantex_features == "llm":
        command = os.environ.get("SEMANTEX_MCP_LLM_BINARY", "semantex-mcp-llm")
    else:
        command = os.environ.get("SEMANTEX_MCP_BINARY", "semantex-mcp")

    env: dict[str, str] = {}
    for k in _INHERIT_KEYS:
        v = os.environ.get(k)
        if v is not None:
            env[k] = v

    if condition.semantex_llm_features:
        if condition.semantex_llm_provider:
            env["SEMANTEX_LLM_PROVIDER"] = condition.semantex_llm_provider
        if condition.semantex_llm_model:
            env["SEMANTEX_LLM_MODEL"] = condition.semantex_llm_model
        if condition.semantex_llm_fallback_provider:
            env["SEMANTEX_LLM_FALLBACK_PROVIDER"] = condition.semantex_llm_fallback_provider
        if condition.semantex_llm_fallback_model:
            env["SEMANTEX_LLM_FALLBACK_MODEL"] = condition.semantex_llm_fallback_model

    return {
        "mcpServers": {
            "semantex": {
                "command": command,
                "args": [],
                "env": env,
            }
        }
    }
