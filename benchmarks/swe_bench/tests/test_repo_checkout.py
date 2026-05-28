import subprocess

import pytest

from swe_bench_harness.repo_checkout import RepoCheckout, checkout


@pytest.fixture
def fake_repo(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=r, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=r, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=r, check=True)
    (r / "a.txt").write_text("v1")
    subprocess.run(["git", "add", "-A"], cwd=r, check=True)
    subprocess.run(["git", "commit", "-qm", "v1"], cwd=r, check=True)
    sha_v1 = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=r, text=True
    ).strip()
    (r / "a.txt").write_text("v2")
    subprocess.run(["git", "commit", "-qam", "v2"], cwd=r, check=True)
    return r, sha_v1


def test_checkout_returns_repo_at_sha(tmp_path, fake_repo):
    src, sha_v1 = fake_repo
    dest = tmp_path / "out"
    co = checkout(repo_url=str(src), sha=sha_v1, dest=dest)
    assert isinstance(co, RepoCheckout)
    assert co.path == dest
    assert co.sha == sha_v1
    assert (dest / "a.txt").read_text() == "v1"


def test_checkout_is_idempotent(tmp_path, fake_repo):
    src, sha_v1 = fake_repo
    dest = tmp_path / "out"
    checkout(repo_url=str(src), sha=sha_v1, dest=dest)
    co2 = checkout(repo_url=str(src), sha=sha_v1, dest=dest)
    assert co2.sha == sha_v1
    assert (dest / "a.txt").read_text() == "v1"


def test_checkout_invalid_sha_raises(tmp_path, fake_repo):
    src, _ = fake_repo
    dest = tmp_path / "out"
    with pytest.raises(subprocess.CalledProcessError):
        checkout(repo_url=str(src), sha="0" * 40, dest=dest)


def test_checkout_rejects_non_hex_sha(tmp_path, fake_repo):
    src, _ = fake_repo
    dest = tmp_path / "out"
    with pytest.raises(ValueError, match="sha must be"):
        checkout(repo_url=str(src), sha="not-a-sha", dest=dest)


def test_checkout_rejects_repo_url_starting_with_dash(tmp_path, fake_repo):
    _, sha = fake_repo
    dest = tmp_path / "out"
    with pytest.raises(ValueError, match="repo_url must not start"):
        checkout(repo_url="--upload-pack=evil", sha=sha, dest=dest)
