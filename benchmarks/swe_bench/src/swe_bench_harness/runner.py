"""Drive one (instance, condition, replicate) of the SWE-bench benchmark.

The runner instantiates an OpenHands Agent + Conversation in-process, drives
it against a single SWE-bench instance, and captures:
  - the final unified-diff patch (via ``git diff HEAD`` on the workspace)
  - per-turn token records (one entry per LLM completion call)
  - wall-clock duration
  - any error string

Only ``run_one`` is the public entrypoint; ``_invoke_openhands`` is patched
out by tests and isolated from ``run_one``'s error-swallowing wrapper so that
one bad instance does not crash the Task 4.4 orchestrator.

API contract observed against openhands-sdk 1.24.0:
  * ``Agent.tools`` is ``list[openhands.sdk.Tool]`` — Tool is a Pydantic spec
    ``{name: str, params: dict}``, NOT a tool-class instance. The agent
    materializes ``ToolDefinition`` from this spec via the tool registry.
  * ``Conversation(agent=..., workspace=str|Path, max_iteration_per_run=...)``
    returns a ``LocalConversation``. Drive it with
    ``conversation.send_message(text)`` then ``conversation.run()`` (no args).
  * Per-call token usage lives on ``llm.metrics.token_usages`` (list of
    ``TokenUsage`` — one per LLM completion). Fields are ``prompt_tokens``,
    ``completion_tokens``, ``cache_read_tokens``, ``cache_write_tokens``,
    ``response_id``, etc. We translate to Anthropic-style names
    (``input_tokens`` / ``output_tokens`` / ``cache_creation_input_tokens``
    / ``cache_read_input_tokens``) so downstream metrics math matches the
    Anthropic Usage object that CCB calculations reference.
  * Tool calls per turn are derived from ``conversation.state.events`` —
    ``ActionEvent`` instances carry ``tool_name`` and ``llm_response_id``,
    which we group against each TokenUsage's ``response_id``.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .conditions import Condition
from .dataset import Instance
from .tools.mcp_config import build_mcp_config


@dataclass
class RunResult:
    instance_id: str
    condition_id: str
    replicate: int
    patch: str = ""
    turns: list[dict] = field(default_factory=list)
    wall_clock_secs: float = 0.0
    error: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


@dataclass
class _OpenHandsResult:
    """Internal: what ``_invoke_openhands`` returns. Tests mock this."""

    patch: str
    turns: list[dict]


def _extract_workspace_patch(repo_path: Path, base_commit: str) -> str:
    """``git diff <base_commit>`` — captures both committed-since-base and
    uncommitted working-tree changes. Excludes `.semantex/` (our index dir)
    and `.gitignore` (agents sometimes mutate it to mask `.semantex/`).
    Returns "" if the path is not a git repo.

    Diffing against base_commit rather than HEAD is critical: agents
    sometimes commit their fix inside the workspace, which would leave
    `git diff HEAD` empty and silently lose the patch."""
    proc = subprocess.run(
        ["git", "diff", base_commit, "--", ".", ":(exclude).semantex", ":(exclude).gitignore"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout


def _tool_calls_by_response_id(events: list[Any]) -> dict[str, list[str]]:
    """Group tool_name from ActionEvents keyed by llm_response_id.

    Imported lazily so a default (no-openhands) test environment doesn't
    pay the import cost in the run_one wrapper path.
    """
    try:
        from openhands.sdk.event.llm_convertible.action import ActionEvent
    except Exception:  # noqa: BLE001 — if openhands isn't importable here we have bigger issues
        return {}

    out: dict[str, list[str]] = defaultdict(list)
    for ev in events:
        if isinstance(ev, ActionEvent):
            out[ev.llm_response_id].append(ev.tool_name)
    return dict(out)


def _invoke_openhands(
    *,
    repo_path: Path,
    instance: Instance,
    condition: Condition,
    max_turns: int,
) -> _OpenHandsResult:
    """Drive the real OpenHands agent against one SWE-bench instance.

    Imports are local to keep ``run_one``'s mocked test path zero-cost and
    to keep the module importable when openhands-sdk isn't installed.
    """
    from openhands.sdk import LLM, Agent, Conversation, Tool
    from pydantic import SecretStr

    # Importing these submodules triggers ToolDefinition registration via
    # @register_tool decorators (openhands-tools 1.24.x). Without these
    # imports the registry is empty and Agent fails resolving tool specs.
    import openhands.tools.terminal  # noqa: F401
    import openhands.tools.file_editor  # noqa: F401
    import openhands.tools.task_tracker  # noqa: F401

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY env var required for runner")

    llm = LLM(
        usage_id="agent",
        model=condition.agent_model,
        api_key=SecretStr(api_key),
    )

    # Built-in tools — same across all conditions for fair comparison.
    # Tool is a Pydantic spec (name + params); the agent resolves these
    # via openhands.sdk.tool.registry from the openhands-tools package.
    # NOTE: tool registry names (lowercase snake_case) — these are the names
    # the openhands-tools package uses when calling @register_tool. The class
    # names (TerminalTool/FileEditorTool/TaskTrackerTool) are NOT the
    # registry keys, even though Tool(name=...) silently accepts either.
    tools = [
        Tool(name="terminal"),
        Tool(name="file_editor"),
        Tool(name="task_tracker"),
    ]

    mcp_cfg = build_mcp_config(condition)

    agent_kwargs: dict[str, Any] = {"llm": llm, "tools": tools}
    if mcp_cfg is not None:
        agent_kwargs["mcp_config"] = mcp_cfg
    agent = Agent(**agent_kwargs)

    # System prompt is intentionally identical across all conditions — the
    # only treatment variable is whether semantex MCP is registered.
    prompt = (
        f"You are working on this GitHub issue from the {instance.repo} repo:\n\n"
        f"{instance.problem_statement}\n\n"
        f"Make the minimal code changes needed to fix the issue. "
        f"Do not modify test files. When done, ensure your changes are "
        f"saved to disk in the workspace."
    )

    conversation = Conversation(
        agent=agent,
        workspace=str(repo_path),
        max_iteration_per_run=max_turns,
        # No persistence — we capture what we need in-process before close.
        persistence_dir=None,
    )

    try:
        conversation.send_message(prompt)
        conversation.run()

        # Group tool calls by llm_response_id so we can attribute them
        # to the same TokenUsage record.
        events = list(getattr(conversation.state, "events", []) or [])
        tools_by_response = _tool_calls_by_response_id(events)

        turns: list[dict] = []
        metrics = getattr(llm, "metrics", None)
        token_usages = getattr(metrics, "token_usages", None) or []
        for usage in token_usages:
            response_id = getattr(usage, "response_id", "") or ""
            turns.append(
                {
                    "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                    "output_tokens": int(
                        getattr(usage, "completion_tokens", 0) or 0
                    ),
                    "cache_creation_input_tokens": int(
                        getattr(usage, "cache_write_tokens", 0) or 0
                    ),
                    "cache_read_input_tokens": int(
                        getattr(usage, "cache_read_tokens", 0) or 0
                    ),
                    "tool_calls": list(tools_by_response.get(response_id, [])),
                }
            )

        patch = _extract_workspace_patch(repo_path, instance.base_commit)
        return _OpenHandsResult(patch=patch, turns=turns)
    finally:
        # LocalConversation registers atexit for close, but explicit close
        # frees the workspace and any spawned MCP child processes promptly.
        try:
            conversation.close()
        except Exception:  # noqa: BLE001 — best effort
            pass


def run_one(
    *,
    instance: Instance,
    condition: Condition,
    replicate: int,
    repo_cache_root: Path,
    max_turns: int = 75,
) -> RunResult:
    """Run one (instance, condition, replicate). Never raises — failures
    are recorded on the ``RunResult.error`` field so the orchestrator can
    keep going across the remaining instances."""
    repo_path = repo_cache_root / instance.instance_id
    result = RunResult(
        instance_id=instance.instance_id,
        condition_id=condition.id,
        replicate=replicate,
    )
    t0 = time.monotonic()
    try:
        oh = _invoke_openhands(
            repo_path=repo_path,
            instance=instance,
            condition=condition,
            max_turns=max_turns,
        )
        result.patch = oh.patch
        result.turns = oh.turns
    except Exception as e:  # noqa: BLE001 — by design; surface as result.error
        result.error = str(e)
    result.wall_clock_secs = time.monotonic() - t0
    return result
