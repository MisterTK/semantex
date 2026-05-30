#!/usr/bin/env bash
# Self-healing pipeline driver. Idempotent: figures out where we are and
# resumes the right phase. Suitable for @reboot cron OR manual launch.
# Logs everything to ~/resume.log so we have a record across preemptions.
set -uo pipefail

LOG=$HOME/resume.log
exec > >(tee -a "$LOG") 2>&1
echo
echo "=== resume.sh $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

HARNESS=$HOME/semantex/benchmarks/swe_bench
CACHE=$HOME/swe_repos
cd "$HARNESS"

# Source env
. .venv/bin/activate
set -a; . "$HOME/semantex/.env"; set +a
# NOTE: Do NOT set ORT_DYLIB_PATH. semantex auto-provisions the correct ONNX
# Runtime (>= the version `ort` requires) into ~/.semantex/runtime on first use.
# The old /usr/local/lib/libonnxruntime.so.1.20.1 was 1.20 — below ort rc.11's
# 1.23 floor — and would defeat auto-provisioning (a user-set ORT_DYLIB_PATH
# wins). `download-models` (run in vm_bootstrap.sh) pre-populates it.
export SEMANTEX_BINARY="$HOME/semantex/target/release/semantex"
export SEMANTEX_LLM_BINARY="$HOME/semantex/target-llm/release/semantex"
export SWE_BENCH_REPO_CACHE="$CACHE"

INDEXED=$(find "$CACHE" -name updated_at -type f 2>/dev/null | wc -l | tr -d ' ')
TOTAL=$(wc -l < config/instances_phase_a.txt | tr -d ' ')

# Phase order:
#   1. Pre-index until we have >=90 of 100 instances indexed (allow up to 10 failures)
#   2. Phase A: run /home/*/swe_repos exists with indexes; produce 600 unit JSON files
#   3. Submit: aggregate + swebench eval + report

# --- Phase 1: pre-index ---
if [ "$INDEXED" -lt 90 ]; then
  echo "pre-index: $INDEXED / $TOTAL — launching"
  if tmux has-session -t preindex 2>/dev/null; then
    echo "preindex session already running"
  else
    tmux new-session -d -s preindex "
      cd $HARNESS
      . .venv/bin/activate
      set -a; . \$HOME/semantex/.env; set +a
      export SWE_BENCH_REPO_CACHE=$CACHE
      # Indexing is memory-bound, not CPU-bound: the next-plaid k-means distance
      # GEMM peaks at ~rayon_threads x block. With the vendored chunk_size_data
      # patch (block ~167MB) + RAYON_NUM_THREADS=2, each worker peaks ~4-5 GB, so
      # ~16 parallel workers fit in 128 GB at full index quality. SEMANTEX_NO_RLIMIT
      # disables the 24 GB virtual-address cap (irrelevant on a dedicated VM).
      export RAYON_NUM_THREADS=2
      export SEMANTEX_NO_RLIMIT=1
      export SWE_BENCH_INDEX_TIMEOUT=7200
      python -m scripts.pre_index --phase a --workers 16 --semantex-bin $SEMANTEX_BINARY 2>&1 | tee \$HOME/pre_index.log
    "
    echo "preindex tmux session started"
  fi
  exit 0
fi

# --- Phase 2: Phase A run ---
echo "pre-index complete: $INDEXED / $TOTAL"

# Reuse existing RUN_ID if we have one tracked; else generate
RUN_ID_FILE=$HOME/run_id.txt
if [ -f "$RUN_ID_FILE" ]; then
  RUN_ID=$(cat "$RUN_ID_FILE")
else
  RUN_ID="$(date +%Y%m%d-%H%M%S)-phase_a"
  echo "$RUN_ID" > "$RUN_ID_FILE"
fi
echo "RUN_ID=$RUN_ID"

RESULTS=$HARNESS/results/$RUN_ID/runs
EXISTING=$(ls "$RESULTS" 2>/dev/null | wc -l | tr -d ' ')
TARGET=600  # 100 inst × 3 conds × 2 reps
echo "phase-A: $EXISTING / $TARGET units complete"

if [ "$EXISTING" -lt "$TARGET" ]; then
  if tmux has-session -t phasea 2>/dev/null; then
    echo "phasea session already running"
  else
    tmux new-session -d -s phasea "
      cd $HARNESS
      . .venv/bin/activate
      set -a; . \$HOME/semantex/.env; set +a
      export SEMANTEX_BINARY=$SEMANTEX_BINARY
      export SEMANTEX_LLM_BINARY=$SEMANTEX_LLM_BINARY
      export SWE_BENCH_REPO_CACHE=$CACHE
      python -m scripts.run --phase a --replicates 2 --workers 8 --max-turns 75 --run-id $RUN_ID 2>&1 | tee \$HOME/phase_a.log
    "
    echo "phasea tmux session started"
  fi
  exit 0
fi

# --- Phase 3: submit (swebench eval + report) ---
echo "phase A complete: $EXISTING / $TARGET"

if [ -f "$HARNESS/results/$RUN_ID/report.md" ]; then
  echo "report already generated: $HARNESS/results/$RUN_ID/report.md"
  exit 0
fi

if tmux has-session -t submit 2>/dev/null; then
  echo "submit session already running"
  exit 0
fi

tmux new-session -d -s submit "
  cd $HARNESS
  . .venv/bin/activate
  python -m scripts.submit --run-id $RUN_ID 2>&1 | tee \$HOME/submit.log
"
echo "submit tmux session started"
