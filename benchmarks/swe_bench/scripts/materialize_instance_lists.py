"""One-time: write Phase A (100 seeded) and Phase B (all 500) instance ID lists."""
from pathlib import Path

from swe_bench_harness.dataset import load_verified, select_subset


CONFIG = Path(__file__).parent.parent / "config"
PHASE_A_SEED = 20260528  # fixed; never change


def main() -> None:
    insts = load_verified()
    assert len(insts) == 500, f"expected 500 Verified instances, got {len(insts)}"

    phase_a = select_subset(insts, n=100, seed=PHASE_A_SEED)
    phase_b = sorted(insts, key=lambda i: i.instance_id)

    CONFIG.mkdir(exist_ok=True)
    (CONFIG / "instances_phase_a.txt").write_text(
        "\n".join(i.instance_id for i in phase_a) + "\n"
    )
    (CONFIG / "instances_phase_b.txt").write_text(
        "\n".join(i.instance_id for i in phase_b) + "\n"
    )
    print(f"phase A: {len(phase_a)}  phase B: {len(phase_b)}")


if __name__ == "__main__":
    main()
