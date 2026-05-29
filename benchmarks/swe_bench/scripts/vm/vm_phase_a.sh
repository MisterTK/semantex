#!/usr/bin/env bash
# Launch Phase A on the VM (after pre-index has completed).
# Runs inside a detached tmux session named "phasea".
# Resumable: set RUN_ID env to resume an existing run; otherwise generates a new one.
set -euo pipefail
cd "$HOME/semantex/benchmarks/swe_bench"

# Pre-flight: confirm pre-index is sufficiently complete
READY=$(find "$HOME/swe_repos" -name updated_at -type f 2>/dev/null | wc -l | tr -d ' ')
TOTAL=$(wc -l < config/instances_phase_a.txt | tr -d ' ')
echo "pre-indexed: $READY / $TOTAL"
if [ "$READY" -lt 90 ]; then
  echo "ABORT: fewer than 90 instances pre-indexed."
  echo "Wait for pre-index to finish (or rerun with reduced --workers if it OOM'd)."
  exit 1
fi

RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)-phase_a}"
echo "RUN_ID=$RUN_ID"

# Already running?
if tmux has-session -t phasea 2>/dev/null; then
  echo "tmux session 'phasea' already exists. Attach with: tmux attach -t phasea"
  exit 0
fi

tmux new-session -d -s phasea "
  cd ~/semantex/benchmarks/swe_bench
  . .venv/bin/activate
  set -a; . ~/semantex/.env; set +a
  export SEMANTEX_BINARY=\$HOME/semantex/target/release/semantex
  export SEMANTEX_LLM_BINARY=\$HOME/semantex/target-llm/release/semantex
  export SWE_BENCH_REPO_CACHE=\$HOME/swe_repos
  python -m scripts.run --phase a --replicates 2 --workers 8 --max-turns 75 --run-id $RUN_ID 2>&1 | tee \$HOME/phase_a.log
"
echo "Phase A launched in tmux session 'phasea'. RUN_ID=$RUN_ID"
echo
echo "Monitor:   tmux attach -t phasea      (detach with Ctrl-b d)"
echo "Tail log:  tail -f ~/phase_a.log"
echo "Progress:  ls ~/semantex/benchmarks/swe_bench/results/$RUN_ID/runs/ | wc -l   # /600"
echo
echo "When complete:"
echo "  cd ~/semantex/benchmarks/swe_bench && . .venv/bin/activate"
echo "  python -m scripts.submit --run-id $RUN_ID"
