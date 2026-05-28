from swe_bench_harness.metrics import ccb, cost_usd, num_turns, tool_distribution


TURNS = [
    {"input_tokens": 1000, "output_tokens": 50, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0, "tool_calls": ["grep"]},
    {"input_tokens": 200,  "output_tokens": 80, "cache_creation_input_tokens": 900, "cache_read_input_tokens": 0, "tool_calls": ["read", "read"]},
    {"input_tokens": 50,   "output_tokens": 60, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 1100, "tool_calls": ["edit"]},
]


def test_num_turns():
    assert num_turns(TURNS) == 3


def test_ccb_sums_attended_context_per_turn():
    # turn 1: 1000+0+0 = 1000
    # turn 2: 200+900+0 = 1100
    # turn 3: 50+0+1100 = 1150
    # total: 3250
    assert ccb(TURNS) == 3250


def test_cost_uses_pricing_table():
    pricing = {
        "claude-sonnet-4-6": {
            "input_per_mtok": 3.0,
            "output_per_mtok": 15.0,
            "cache_write_per_mtok": 3.75,
            "cache_read_per_mtok": 0.30,
        }
    }
    expected = sum([
        (1000 * 3.0 + 50 * 15.0) / 1e6,
        (200 * 3.0 + 80 * 15.0 + 900 * 3.75) / 1e6,
        (50 * 3.0 + 60 * 15.0 + 1100 * 0.30) / 1e6,
    ])
    assert abs(cost_usd(TURNS, model="claude-sonnet-4-6", pricing=pricing) - expected) < 1e-9


def test_tool_distribution_counts_calls():
    d = tool_distribution(TURNS)
    assert d == {"grep": 1, "read": 2, "edit": 1}
