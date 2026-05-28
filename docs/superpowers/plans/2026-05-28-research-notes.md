# Research Notes for SWE-bench Harness Plan (Task 0.1)

Verified: 2026-05-27

All four sections were verified by **execution** (not docs-only) in a fresh
`python3.12 -m venv /tmp/research-venv312`. Python 3.14 was tried first but is
unsupported by the OpenHands SDK transitive deps (`litellm`, `pillow`).

## 1. SWE-bench Verified dataset

- Source: `princeton-nlp/SWE-bench_Verified`
- Rows: **500** (matches plan)
- Columns (exact, all 13):
  - `repo`
  - `instance_id`
  - `base_commit`
  - `patch`
  - `test_patch`
  - `problem_statement`
  - `hints_text`
  - `created_at`
  - `version`
  - `FAIL_TO_PASS`
  - `PASS_TO_PASS`
  - `environment_setup_commit`
  - `difficulty`
- All four required columns (`instance_id`, `repo`, `base_commit`, `problem_statement`) are present.
- Extra useful columns the plan should keep in mind: `version`, `environment_setup_commit`, `difficulty`, `FAIL_TO_PASS`, `PASS_TO_PASS`, `hints_text` (worth stripping/recording for "no-hints" condition fairness).
- Loader call: `load_dataset("princeton-nlp/SWE-bench_Verified", split="test")` — no auth token required for read; HF Hub serves it anonymously, with a rate-limit warning (set `HF_TOKEN` env for benchmark runs).
- Verification: **executed**

## 2. swebench harness CLI

- Install: `pip install swebench>=4.1.0` (verified version: `4.1.0`)
- Entry point: `python -m swebench.harness.run_evaluation`
- Required flags:
  - `-p / --predictions_path PATH` — JSON/JSONL file with predictions (set to literal string `'gold'` to use gold patches).
  - `-id / --run_id RUN_ID` — required; identifies the run.
- Important flags for our use:
  - `-d / --dataset_name DATASET_NAME` (default `SWE-bench/SWE-bench_Lite`). For Verified pass **`princeton-nlp/SWE-bench_Verified`** (matches HF dataset ID we load in §1).
  - `-s / --split SPLIT` (default `test`).
  - `-i / --instance_ids ID [ID ...]` — restrict to a subset of IDs (space-separated).
  - `--max_workers N` (default `4`; the help text says "should be <= 75% of CPU cores").
  - `-t / --timeout SECONDS` (default `1800`).
  - `--cache_level {none,base,env,instance}` (default `env`).
  - `--namespace NAMESPACE` (default `swebench`; pass `none` to use no namespace).
  - `--modal MODAL` — set to True to run on Modal cloud.
  - `--report_dir DIR` (default `.`) — **BEWARE: only used by `--rewrite_reports` mode; the normal final-report write ignores it (see report-path quirk below).**
- Prediction file schema (per `KEY_*` constants in `swebench.harness.constants`):
  - `instance_id` → instance ID (required)
  - `model_name_or_path` → our model/condition identifier (required; used in path)
  - `model_patch` → the diff (the `KEY_PREDICTION` constant)
- Report-path layout (verified by reading `swebench/harness/run_evaluation.py` and `swebench/harness/reporting.py`):
  - **Per-instance**: `logs/run_evaluation/{run_id}/{model_name_or_path.replace('/','__')}/{instance_id}/report.json`
    - The `logs/run_evaluation/` prefix is the constant `RUN_EVALUATION_LOG_DIR` (relative to CWD).
  - **Aggregate run report**: written to **CWD** as `{model_name_or_path.replace('/','__')}.{run_id}.json` — `make_run_report` in `reporting.py` does NOT join with `report_dir`, it just does `Path(f"{model}.{run_id}.json")`.
- Notes:
  - Harness requires Docker for normal local runs (`docker.from_env()` is called in `main()` before `run_instances`). The only Docker-free path is `--modal True`.
  - Multimodal split has an explicit guard that prints a warning and returns without running.
- Verification: **executed** (`--help` invocation and source-read of `run_evaluation.py` + `reporting.py`).

## 3. OpenHands

- **Chosen tag: `v1.24.0`** (released 2026-05-27 — today). This is a stable release, not RC/beta. SHA `fdc2bdf06f4d19af44d7ee4fbf3f742624d80d82`.
- Repo: **`https://github.com/OpenHands/software-agent-sdk`** — NOT `All-Hands-AI/OpenHands` (see major plan adjustment below).
- The original repo `All-Hands-AI/OpenHands` (latest stable `1.7.0`, SHA `43bd8d12e59d2e9254fa8b490ad3e1234902e33b`) is now ONLY the GUI/Cloud backend. The agent SDK has been factored out into a separate repo at `OpenHands/software-agent-sdk`. The README confirms: "The SDK is a composable Python library that contains all of our agentic tech... [view the source](https://github.com/OpenHands/software-agent-sdk/)".
- PyPI packages (both at `1.24.0`):
  - `openhands-sdk` — core (`from openhands.sdk import LLM, Agent, Conversation, Tool, Action, Observation`)
  - `openhands-tools` — built-in tools (`from openhands.tools.{terminal,file_editor,task_tracker} import ...`)
- Tool registration mechanism (TWO supported patterns):
  1. **Custom tool subclass + `register_tool`** (the path the plan assumes):
     - Subclass `Action` (pydantic `BaseModel`) with `Field`-typed params.
     - Subclass `Observation` with a `to_llm_content` property returning `Sequence[TextContent | ImageContent]`.
     - Subclass `ToolExecutor[ActionT, ObservationT]` with `__call__(action, conversation=None) -> ObservationT`.
     - Subclass `ToolDefinition[ActionT, ObservationT]` with a `@classmethod create(cls, conv_state, **params) -> Sequence[ToolDefinition]`.
     - Register: `register_tool(MyTool.name, MyTool)` then reference via `Tool(name=MyTool.name)` in `Agent(tools=[...])`.
  2. **MCP server via `mcp_config`** (alternative, lower-code):
     - `Agent(llm=llm, tools=tools, mcp_config={"mcpServers": {"semantex": {"command": "...", "args": [...]}}})`
     - We already ship an MCP server in `crates/semantex-mcp/`, so this is the cheaper path.
- Custom tool exact import paths (from `examples/01_standalone_sdk/02_custom_tools.py`):
  ```python
  from openhands.sdk import LLM, Agent, Conversation, Tool, Action, Observation, ToolDefinition, TextContent, ImageContent, Event, LLMConvertibleEvent, get_logger
  from openhands.sdk.tool import ToolExecutor, register_tool
  ```
- Model configuration:
  - `LLM(model=..., api_key=SecretStr(...), base_url=...)` constructor.
  - Examples read env vars: `LLM_MODEL`, `LLM_API_KEY`, `LLM_BASE_URL`.
  - Models are routed through LiteLLM under the hood — so for Sonnet 4.6 pass the LiteLLM model name (e.g. `claude-sonnet-4-5-20250929`, or whichever Anthropic model id we pick at run time) plus `ANTHROPIC_API_KEY` or pass it as `api_key=SecretStr(os.getenv("ANTHROPIC_API_KEY"))`.
  - NOTE: there is no SDK-level constant pinning Sonnet 4.6 — the LiteLLM model string is the source of truth. The plan should set `LLM_MODEL` in the condition config.
- In-process vs Docker:
  - **In-process is fully supported**: `Conversation(agent=agent, workspace=cwd)` where `cwd` is a local path string. Used in `examples/01_standalone_sdk/01_hello_world.py`. No Docker required.
  - Docker / remote / cloud workspaces are alternative options under `openhands-workspace` (`docker/`, `cloud/`, `apptainer/`, `remote_api/`) and `examples/02_remote_agent_server/`.
  - For our use (driving many checkouts on a single machine), **in-process is the right choice**.
- Verification: **executed** — installed `openhands-sdk==1.24.0` + `openhands-tools==1.24.0` in `/tmp/research-venv312` and confirmed:
  - `openhands.sdk` exports `LLM`, `Agent`, `Conversation`, `Tool`, `Action`, `Observation`, `ToolDefinition`.
  - `openhands.sdk.tool` exports `ToolExecutor`, `register_tool`.
  - `openhands.tools.{terminal,file_editor,task_tracker}` all import; `.name` attrs are `"terminal"`, `"file_editor"`, `"task_tracker"`.
  - SDK reports `OpenHands SDK v1.24.0` on import.

## 4. Anthropic SDK Usage fields

- Package: **`anthropic 0.104.1`** (latest on PyPI as of verification).
- `anthropic.types.Usage` is a pydantic model. Field names from `Usage.__annotations__` and `Usage.model_fields` (identical):
  - `cache_creation`            (object — nested breakdown, e.g. ephemeral vs persistent)
  - `cache_creation_input_tokens` (int — total cache-write tokens; matches plan)
  - `cache_read_input_tokens`    (int — cache-hit tokens; matches plan)
  - `inference_geo`              (object — region info, can ignore)
  - `input_tokens`               (int — matches plan)
  - `output_tokens`              (int — matches plan)
  - `server_tool_use`            (object — server-side tool usage, can ignore for our agent which uses client tools)
  - `service_tier`               (string — e.g. `standard`, `priority`)
- **All four field names the plan assumes (`input_tokens`, `output_tokens`, `cache_creation_input_tokens`, `cache_read_input_tokens`) exist verbatim.** No rename needed in Task 6.1.
- Note: `cache_creation` (singular, nested) is NEW and exposes finer cache-write breakdown — Task 6.1 can ignore it but should add a TODO if we ever want per-cache-tier cost analysis.
- Verification: **executed** (`Usage.__annotations__`, `Usage.model_fields`).

## Implications for the plan

The plan's Phase 0 / 3 / 4 / 5 / 6 tasks need the following adjustments. None block research; all are factual updates the implementer must apply when writing code.

### A. OpenHands source-of-truth changed (affects Task 0.3, 3.2, 3.3, 4.2)

The plan likely points at `https://github.com/All-Hands-AI/OpenHands.git` for the submodule. That repo no longer contains the agent SDK at 1.7.0. **Update Task 0.3 to use `https://github.com/OpenHands/software-agent-sdk.git` at tag `v1.24.0`** (SHA `fdc2bdf06f4d19af44d7ee4fbf3f742624d80d82`).

  Alternative (recommended) — skip the submodule entirely and just pin in `requirements.txt`:
  ```
  openhands-sdk==1.24.0
  openhands-tools==1.24.0
  ```
  PyPI distributions are available and verified to install cleanly on Python 3.12. Submodules add CI burden that PyPI pins do not.

### B. Python version pin (affects Task 0.2)

The OpenHands SDK transitively requires `litellm` + `pillow` wheels that **do not have Python 3.14 distributions** as of today. The SDK's own ruff config targets `py313`. Scaffold the benchmark project at **Python 3.12 or 3.13**; do not let CI float to 3.14.

### C. Tool registration API names (affects Task 3.2)

If the plan's example code references `BaseTool`, `Tool`, or `AgentAction` from older OpenHands versions (pre-1.0 used `openhands.agenthub.*` and `openhands.events.action.AgentAction`), it is stale. The 1.24.0 API is:

```python
from openhands.sdk import Action, Observation, Tool
from openhands.sdk.tool import ToolDefinition, ToolExecutor, register_tool
```

`Tool` is the *reference* used in `Agent(tools=[Tool(name=...)])`. `ToolDefinition` is the *base class* a custom tool inherits from. Don't confuse them.

### D. Tool option: MCP server vs custom ToolDefinition (affects Task 3.1, 3.2)

The plan assumes Task 3.1 builds a `SemantexClient` subprocess wrapper and Task 3.2 wraps it in a custom `ToolDefinition`. **Cheaper alternative**: register `crates/semantex-mcp` directly via `Agent(mcp_config={"mcpServers": {"semantex": {"command": "semantex-mcp", "args": [...]}}})` — no `ToolDefinition` subclass needed, no Action/Observation pydantic models, no executor wiring. This is `examples/01_standalone_sdk/13_get_llm_metrics.py` (which uses `mcp-server-fetch` the same way).

  Recommend the implementer **try MCP first** and only fall back to a custom ToolDefinition if MCP transport overhead or schema control becomes a problem. This could collapse 3.1 + 3.2 into ~50 LoC of YAML/dict config.

### E. Anthropic Usage extras (affects Task 6.1)

The plan's four target fields are all present and named correctly. Task 6.1 can also opportunistically capture `service_tier` (string) so reports can distinguish priority vs standard tier — useful for cost analysis if we ever burst onto priority.

### F. swebench aggregate-report path quirk (affects Task 5.1)

The harness's `--report_dir` flag is a half-implemented feature: it's only honored when `--rewrite_reports` is True. In normal eval mode, `make_run_report` writes the final aggregate to `Path(f"{model_name_or_path.replace('/','__')}.{run_id}.json")` — i.e., **the current working directory**, ignoring `--report_dir`. Per-instance reports DO go under `logs/run_evaluation/{run_id}/{model_name}/{instance_id}/report.json` (also relative to CWD via `RUN_EVALUATION_LOG_DIR`).

  **Task 5.1 must `os.chdir()` (or `subprocess(cwd=...)`) into the desired output directory before invoking the harness**, or parse `report.json` from the harness's CWD after the run completes. Don't trust `--report_dir`.

### G. Docker requirement for swebench harness (affects Task 5.1, 5.2)

`swebench.harness.run_evaluation.main()` calls `docker.from_env()` unconditionally (unless `--modal True` is passed). Task 5.1 must document the Docker prerequisite — and Task 5.2's smoke test cannot run on a machine without Docker. The plan should either:
  - Require Docker Desktop on macOS dev machine, OR
  - Add a `--modal` path (paid Modal account required).

### H. HF rate-limit warning (affects Task 1.1)

`load_dataset("princeton-nlp/SWE-bench_Verified", split="test")` works anonymously but emits "set HF_TOKEN" warning. For repeatable production runs, the harness scripts should accept `HF_TOKEN` via env (silent if set).

### Summary of items to flag in PR description

- [ ] OpenHands SDK source moved to `OpenHands/software-agent-sdk`, pinned at `v1.24.0` (PyPI: `openhands-sdk`/`openhands-tools` 1.24.0).
- [ ] Python pinned to 3.12 (avoid 3.14 transitive dep gaps).
- [ ] Tool API: `ToolDefinition` + `ToolExecutor` + `register_tool`; consider MCP-only path.
- [ ] swebench `--report_dir` quirk: aggregate report writes to CWD.
- [ ] Docker required for swebench harness local runs.
