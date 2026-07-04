import shutil

import pytest

from relevance_harness.ripgrep_baseline import extract_keywords, rank_files_by_keyword_hits


def test_extract_keywords_pulls_identifiers_and_drops_stopwords():
    text = "Login fails when authenticate_user receives a unicode password."
    kws = extract_keywords(text)
    assert "authenticate_user" in kws
    assert "unicode" in kws
    assert "password" in kws
    low = [k.lower() for k in kws]
    assert "when" not in low
    assert "the" not in low


def test_extract_keywords_is_deterministic_and_ordered_first_seen():
    text = "PoolEmpty raised by connections.acquire when connections is empty"
    assert extract_keywords(text) == extract_keywords(text)
    kws = extract_keywords(text)
    assert kws.index("PoolEmpty") < kws.index("connections.acquire")


def test_extract_keywords_dedupes_case_insensitively():
    text = "auth_token missing; AUTH_TOKEN required; Auth_Token expected"
    kws = extract_keywords(text)
    assert len([k for k in kws if k.lower() == "auth_token"]) == 1


def test_extract_keywords_respects_max_keywords():
    text = " ".join(f"identifier_{i}" for i in range(30))
    assert len(extract_keywords(text, max_keywords=5)) == 5


def test_rank_files_by_keyword_hits_empty_keywords_returns_empty(tmp_path):
    assert rank_files_by_keyword_hits(tmp_path, []) == []


def test_rank_files_by_keyword_hits_missing_binary_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        rank_files_by_keyword_hits(tmp_path, ["foo"], rg_binary="definitely-not-a-real-binary")


@pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep not on PATH")
def test_rank_files_by_keyword_hits_orders_by_match_count(tmp_path):
    (tmp_path / "auth.py").write_text("def login(username, password):\n    return password\n")
    (tmp_path / "util.py").write_text("def helper():\n    pass\n")
    ranked = rank_files_by_keyword_hits(tmp_path, ["password", "login"])
    assert ranked[0] == "auth.py"
    assert "util.py" not in ranked


@pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep not on PATH")
def test_rank_files_by_keyword_hits_deterministic_tie_break(tmp_path):
    (tmp_path / "b.py").write_text("marker\n")
    (tmp_path / "a.py").write_text("marker\n")
    ranked = rank_files_by_keyword_hits(tmp_path, ["marker"])
    # equal hit counts -> path-ascending tie-break
    assert ranked == ["a.py", "b.py"]
