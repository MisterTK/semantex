from pathlib import Path

import pytest

from swe_bench_harness.conditions import Condition, load_conditions


CONFIG = Path(__file__).parent.parent / "config" / "conditions.yaml"


def test_loads_three_conditions():
    conds = load_conditions(CONFIG)
    assert set(conds.keys()) == {"c1_baseline", "c2_semantex_no_llm", "c3_semantex_with_llm"}


def test_baseline_has_no_semantex():
    conds = load_conditions(CONFIG)
    c1 = conds["c1_baseline"]
    assert not c1.semantex_enabled
    assert c1.semantex_features == ""
    assert not c1.semantex_llm_features


def test_c2_has_semantex_no_llm():
    conds = load_conditions(CONFIG)
    c2 = conds["c2_semantex_no_llm"]
    assert c2.semantex_enabled
    assert c2.semantex_features == ""
    assert not c2.semantex_llm_features


def test_c3_has_semantex_and_llm():
    conds = load_conditions(CONFIG)
    c3 = conds["c3_semantex_with_llm"]
    assert c3.semantex_enabled
    assert c3.semantex_features == "llm"
    assert c3.semantex_llm_features
    assert c3.semantex_llm_provider == "anthropic"
    assert c3.semantex_llm_model.startswith("claude-haiku")
    assert c3.semantex_llm_fallback_provider == "google"
    assert c3.semantex_llm_fallback_model.startswith("gemini")


def test_all_use_sonnet_4_6():
    conds = load_conditions(CONFIG)
    assert all(c.agent_model == "claude-sonnet-4-6" for c in conds.values())
