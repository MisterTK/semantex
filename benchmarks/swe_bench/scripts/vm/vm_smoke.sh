#!/usr/bin/env bash
# 1-instance smoke validation on the VM. Picks the first phase A instance,
# pre-indexes it (clone+checkout+index), then runs the runner under C1 with
# max_turns=20 to validate the full pipeline before triggering pre-index +
# Phase A at scale. Cost: ~$1.50, time: ~5-10min.
set -euo pipefail

cd "$HOME/semantex/benchmarks/swe_bench"
. .venv/bin/activate
set -a
. "$HOME/semantex/.env"
set +a
export SEMANTEX_BINARY="$HOME/semantex/target/release/semantex"
export SEMANTEX_LLM_BINARY="$HOME/semantex/target-llm/release/semantex"
export SWE_BENCH_REPO_CACHE="$HOME/swe_repos"
mkdir -p "$SWE_BENCH_REPO_CACHE"

SMOKE_ID=$(head -1 config/instances_phase_a.txt)
echo "smoke instance: $SMOKE_ID"

echo "=== pre-index $SMOKE_ID ==="
python -c "
import os
from pathlib import Path
from swe_bench_harness.dataset import load_verified
from swe_bench_harness.repo_checkout import checkout
from swe_bench_harness.indexer import index_repo

cache = Path(os.environ['SWE_BENCH_REPO_CACHE'])
sid = '$SMOKE_ID'
inst = next(i for i in load_verified() if i.instance_id == sid)
dest = cache / sid
print(f'clone {inst.repo} @ {inst.base_commit[:12]}...')
checkout(repo_url=f'https://github.com/{inst.repo}.git', sha=inst.base_commit, dest=dest)
print('index...')
r = index_repo(repo_path=dest, semantex_binary=os.environ['SEMANTEX_BINARY'], timeout_secs=1800)
print(f'ok={r.ok} duration={r.duration_secs:.1f}s err={r.error[:200]}')
"

echo "=== run_one C1 smoke ==="
python -c "
import json, os
from pathlib import Path
from swe_bench_harness.conditions import load_conditions
from swe_bench_harness.dataset import load_verified
from swe_bench_harness.runner import run_one

sid = '$SMOKE_ID'
inst = next(i for i in load_verified() if i.instance_id == sid)
conds = load_conditions(Path('config/conditions.yaml'))
res = run_one(
    instance=inst, condition=conds['c1_baseline'], replicate=0,
    repo_cache_root=Path(os.environ['SWE_BENCH_REPO_CACHE']), max_turns=20,
)
print(f'patch length: {len(res.patch)}')
print(f'turns: {len(res.turns)}')
print(f'wall: {res.wall_clock_secs:.1f}s')
print(f'error: {res.error[:300]}')
if res.turns:
    t0, tN = res.turns[0], res.turns[-1]
    print(f'turn 0:  in={t0[\"input_tokens\"]} out={t0[\"output_tokens\"]} cw={t0[\"cache_creation_input_tokens\"]} cr={t0[\"cache_read_input_tokens\"]} tools={t0[\"tool_calls\"]}')
    print(f'turn -1: in={tN[\"input_tokens\"]} out={tN[\"output_tokens\"]} cw={tN[\"cache_creation_input_tokens\"]} cr={tN[\"cache_read_input_tokens\"]} tools={tN[\"tool_calls\"]}')
Path('$HOME/smoke_c1.json').write_text(res.to_json())
print('result: $HOME/smoke_c1.json')
"
