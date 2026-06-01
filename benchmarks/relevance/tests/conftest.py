import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Make the sibling swe_bench harness importable for SWE-loc reuse.
SWE_BENCH_SRC = ROOT.parent / "swe_bench" / "src"
if SWE_BENCH_SRC.is_dir():
    sys.path.insert(0, str(SWE_BENCH_SRC))


@pytest.fixture
def fixtures_dir() -> Path:
    return ROOT / "fixtures"
