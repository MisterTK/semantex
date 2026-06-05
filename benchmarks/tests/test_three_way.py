"""Unit tests for the three-way harness: the pure nudge `decide()` logic + the
runner's hermetic-config construction. All FREE (no API calls, no subprocess)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # import the modules
import three_way_nudge as nudge  # noqa: E402
import three_way as tw  # noqa: E402


# ── decide(): Read ─────────────────────────────────────────────────────────


def test_read_source_file_nudges():
    # A normal source/doc Read SHOULD nudge (the tool replaces grep+read).
    assert nudge.decide("Read", {"file_path": "packages/core/src/agent.ts"},
                        ".semantex", "semantex") == "NUDGE"
    assert nudge.decide("Read", {"file_path": "docs/architecture.md"},
                        ".semantex", "semantex") == "NUDGE"
    assert nudge.decide("Read", {"file_path": "/abs/path/main.py"},
                        ".serena", "") == "NUDGE"


def test_read_own_dir_is_silent():
    # Reading the tool's OWN output dir must never nudge "use the tool".
    assert nudge.decide("Read", {"file_path": ".semantex/meta.json"},
                        ".semantex", "semantex") is None
    assert nudge.decide("Read", {"file_path": "/repo/.semantex/index.bin"},
                        ".semantex", "semantex") is None
    assert nudge.decide("Read", {"file_path": "graphify-out/graph.json"},
                        "graphify-out", "graphify") is None
    assert nudge.decide("Read", {"file_path": "/repo/.serena/cache.db"},
                        ".serena", "") is None


def test_read_build_vcs_dirs_silent():
    for p in (".git/config", "node_modules/react/index.js", "target/debug/foo",
              "dist/bundle.js", "build/out.o", "/repo/node_modules/x/y.ts"):
        assert nudge.decide("Read", {"file_path": p}, ".semantex", "semantex") is None, p


def test_read_lock_files_silent():
    for p in ("Cargo.lock", "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
              "go.sum", "poetry.lock", "/repo/packages/a/package-lock.json"):
        assert nudge.decide("Read", {"file_path": p}, ".semantex", "semantex") is None, p


def test_read_binary_media_silent():
    for p in ("logo.png", "icon.svg", "photo.jpg", "doc.pdf", "archive.zip",
              "font.woff2", "lib.so", "app.wasm", "data.sqlite", "bundle.min.js",
              "x.map", "y.dylib"):
        assert nudge.decide("Read", {"file_path": p}, ".semantex", "semantex") is None, p


def test_read_missing_path_silent():
    assert nudge.decide("Read", {}, ".semantex", "semantex") is None
    assert nudge.decide("Read", {"file_path": ""}, ".semantex", "semantex") is None
    assert nudge.decide("Read", {"file_path": 123}, ".semantex", "semantex") is None


# ── decide(): Grep / Glob ──────────────────────────────────────────────────


def test_grep_glob_always_nudge():
    assert nudge.decide("Grep", {"pattern": "useAgent"}, ".semantex", "semantex") == "NUDGE"
    assert nudge.decide("Glob", {"pattern": "**/*.ts"}, ".semantex", "semantex") == "NUDGE"


# ── decide(): Bash ─────────────────────────────────────────────────────────


def test_bash_search_commands_nudge():
    for cmd in ("grep -rn foo .", "rg useAgent", "ag pattern", "ack thing",
                "find . -name '*.ts'", "fd foo", "fgrep x", "egrep y",
                "git grep needle"):
        assert nudge.decide("Bash", {"command": cmd}, ".semantex", "semantex") == "NUDGE", cmd


def test_bash_self_cmd_is_silent():
    # graphify arm: the agent running `graphify query ...` must NOT be nudged.
    assert nudge.decide("Bash", {"command": 'graphify query "how does X work"'},
                        "graphify-out", "graphify") is None
    assert nudge.decide("Bash", {"command": 'graphify explain Foo'},
                        "graphify-out", "graphify") is None
    # semantex self-cmd silent too (if the agent shells out to the CLI).
    assert nudge.decide("Bash", {"command": "semantex 'find the router'"},
                        ".semantex", "semantex") is None


def test_bash_non_search_commands_silent():
    for cmd in ("ls -la", "cat file.txt", "npm test", "node script.js",
                "echo hi", "git status", "git log"):
        assert nudge.decide("Bash", {"command": cmd}, ".semantex", "semantex") is None, cmd


def test_bash_empty_silent():
    assert nudge.decide("Bash", {"command": ""}, ".semantex", "semantex") is None
    assert nudge.decide("Bash", {}, ".semantex", "semantex") is None


def test_bash_empty_self_cmd_still_nudges_search():
    # serena arm has TW_SELF_CMD="" — a `grep` there should still nudge (empty
    # self_cmd never matches a real command).
    assert nudge.decide("Bash", {"command": "grep -r foo ."}, ".serena", "") == "NUDGE"


# ── decide(): malformed / other ────────────────────────────────────────────


def test_malformed_input_returns_none():
    assert nudge.decide("Read", None, ".semantex", "semantex") is None
    assert nudge.decide("Read", "not-a-dict", ".semantex", "semantex") is None
    assert nudge.decide("Edit", {"file_path": "x.ts"}, ".semantex", "semantex") is None
    assert nudge.decide("Write", {"file_path": "x.ts"}, ".semantex", "semantex") is None
    assert nudge.decide("", {}, ".semantex", "semantex") is None


# ── main(): stdin → JSON output (the wire contract) ────────────────────────


def test_main_emits_additional_context_on_nudge(monkeypatch, capsys):
    import io
    import json
    payload = {"tool_name": "Grep", "tool_input": {"pattern": "x"}}
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))
    monkeypatch.setenv("TW_NUDGE", "USE THE TOOL")
    monkeypatch.setenv("TW_OWN_DIR", ".semantex")
    monkeypatch.setenv("TW_SELF_CMD", "semantex")
    rc = nudge.main()
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["hookSpecificOutput"]["hookEventName"] == "PreToolUse"
    assert out["hookSpecificOutput"]["additionalContext"] == "USE THE TOOL"


def test_main_emits_empty_on_silent(monkeypatch, capsys):
    import io
    import json
    payload = {"tool_name": "Read", "tool_input": {"file_path": ".semantex/x.bin"}}
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))
    monkeypatch.setenv("TW_OWN_DIR", ".semantex")
    monkeypatch.setenv("TW_SELF_CMD", "semantex")
    rc = nudge.main()
    assert rc == 0
    assert json.loads(capsys.readouterr().out) == {}


def test_main_fails_open_on_garbage_stdin(monkeypatch, capsys):
    import io
    import json
    monkeypatch.setattr("sys.stdin", io.StringIO("{not json at all"))
    rc = nudge.main()
    assert rc == 0
    assert json.loads(capsys.readouterr().out) == {}  # fail open → {}, never blocks


def test_main_never_denies(monkeypatch, capsys):
    # The hook must NEVER emit a deny decision — only ever add context or {}.
    import io
    import json
    for payload in ({"tool_name": "Grep", "tool_input": {}},
                    {"tool_name": "Read", "tool_input": {"file_path": "a.ts"}},
                    {"tool_name": "Bash", "tool_input": {"command": "grep x"}}):
        monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))
        monkeypatch.setenv("TW_NUDGE", "n")
        nudge.main()
        out = json.loads(capsys.readouterr().out)
        assert "permissionDecision" not in str(out)
        assert "deny" not in str(out).lower()


# ── runner: hermetic config construction (FREE — no subprocess) ────────────


def test_arms_registry_complete():
    assert set(tw.ARM_NAMES) == {"builtin", "semantex", "graphify", "serena"}
    # semantex arm uses the lateon-colbert embedder (matches the indexed backend).
    assert tw.ARMS["semantex"]["mcp"]["semantex"]["env"]["SEMANTEX_EMBEDDER"] == "lateon-colbert"
    # graphify has NO mcp (Bash CLI).
    assert tw.ARMS["graphify"]["mcp"] is None
    # serena MCP launches via uvx with the verified args.
    serena = tw.ARMS["serena"]["mcp"]["serena"]
    assert serena["command"] == "/opt/homebrew/bin/uvx"
    assert "start-mcp-server" in serena["args"]
    assert "--project" in serena["args"] and tw.REPO in serena["args"]


def test_builtin_has_no_hook_and_no_mcp():
    assert tw.ARMS["builtin"]["nudge"] is None
    assert tw.settings_for("builtin") == {"hooks": {}}
    assert tw.mcp_config_for("builtin") == {"mcpServers": {}}


def test_settings_for_uses_common_nudge_script_with_arm_env():
    # Every tool arm shares the IDENTICAL nudge script; only the env differs.
    for arm, own, self_cmd in (("semantex", ".semantex", "semantex"),
                               ("graphify", "graphify-out", "graphify"),
                               ("serena", ".serena", "")):
        s = tw.settings_for(arm)
        hooks = s["hooks"]["PreToolUse"]
        matchers = {h["matcher"] for h in hooks}
        assert matchers == {"Read|Grep|Glob", "Bash"}, arm
        cmd = hooks[0]["hooks"][0]["command"]
        assert tw.NUDGE_SCRIPT in cmd, arm
        assert f"TW_OWN_DIR='{own}'" in cmd, arm
        assert f"TW_SELF_CMD='{self_cmd}'" in cmd, arm
        assert hooks[0]["hooks"][0]["timeout"] == 5
        # All three tool arms use the SAME script path (identical adoption mechanism).
        assert cmd.endswith(tw.NUDGE_SCRIPT)


def test_all_tool_arms_share_one_script():
    scripts = set()
    for arm in ("semantex", "graphify", "serena"):
        cmd = tw.settings_for(arm)["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
        scripts.add(cmd.split("python3 ", 1)[1])
    assert scripts == {tw.NUDGE_SCRIPT}  # ONE script, equal adoption


def test_tool_flags_grant_each_arm_its_mcp():
    sx = tw.tool_flags("semantex")
    assert "--allowedTools" in sx and "mcp__semantex__*" in sx
    assert "--disallowedTools" in sx and "Skill" in sx
    se = tw.tool_flags("serena")
    assert "mcp__serena__*" in se and "Skill" in se
    # graphify: native tools + Skill blocked, no MCP glob granted.
    gf = tw.tool_flags("graphify")
    assert "mcp__serena__*" not in gf and "mcp__semantex__*" not in gf
    assert "Bash" in gf
    # builtin: MCP globs explicitly DENIED (native floor).
    bi = tw.tool_flags("builtin")
    assert bi.index("--disallowedTools") >= 0
    assert "mcp__semantex__*" in bi and "mcp__serena__*" in bi
    assert "--allowedTools" in bi  # native tools still allowed


def test_mcp_config_shapes():
    assert tw.mcp_config_for("semantex")["mcpServers"]["semantex"]["args"] == ["mcp"]
    assert tw.mcp_config_for("graphify")["mcpServers"] == {}
    assert "serena" in tw.mcp_config_for("serena")["mcpServers"]


def test_question_by_id_resolves_qids():
    assert tw.question_by_id("Q1")["type"] == "architecture"
    assert tw.question_by_id("Q3")["type"] == "deep_technical"
    assert tw.question_by_id("Q5")["type"] == "feature_planning"
    import pytest
    with pytest.raises(KeyError):
        tw.question_by_id("Q99")


def test_own_tool_call_counters():
    assert tw._own_tool_calls("semantex", {"mcp__semantex__semantex_agent": 2, "Read": 3}) == 2
    assert tw._own_tool_calls("serena", {"mcp__serena__find_symbol": 1, "Grep": 4}) == 1
    assert tw._own_tool_calls("builtin", {"Read": 5}) == 0


def test_count_graphify_bash_from_stream():
    import json
    events = [
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "name": "Bash", "input": {"command": 'graphify query "x"'}},
            {"type": "tool_use", "name": "Bash", "input": {"command": "ls -la"}},
            {"type": "tool_use", "name": "Read", "input": {"file_path": "a.ts"}}]}},
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "name": "Bash", "input": {"command": "graphify explain Foo"}}]}},
        {"type": "result", "result": "done"},
    ]
    raw = "\n".join(json.dumps(e) for e in events)
    assert tw._count_graphify_bash(raw) == 2  # two graphify Bash calls, ls/Read excluded
