#!/usr/bin/env bash
# VM bootstrap: system deps → rust → semantex builds (default + LLM) → python venv + harness.
# Idempotent: safe to re-run if it fails partway.
set -euo pipefail
cd "$HOME/semantex"

log() { echo "=== $1 ==="; }

log "1. System packages"
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -qq -y \
  build-essential pkg-config libssl-dev libsqlite3-dev \
  git tmux jq htop python3.12 python3.12-venv python3.12-dev \
  ca-certificates curl docker.io

log "2. Rust toolchain"
if ! [ -x "$HOME/.cargo/bin/cargo" ]; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
    sh -s -- -y --default-toolchain stable --profile minimal
fi
. "$HOME/.cargo/env"
cargo --version

log "3. Building semantex (default, no LLM)"
cargo build --release -p semantex-cli 2>&1 | tail -3
ls -lh target/release/semantex

log "4. Building semantex (--features llm)"
cargo build --release -p semantex-cli --features llm --target-dir target-llm 2>&1 | tail -3
ls -lh target-llm/release/semantex

log "5. Python venv + harness install"
cd "$HOME/semantex/benchmarks/swe_bench"
python3.12 -m venv .venv
. .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -e ".[dev]"

log "6. Pytest validation"
pytest -q 2>&1 | tail -3

log "BOOTSTRAP COMPLETE"
echo "Source: $HOME/semantex"
echo "Default binary: $HOME/semantex/target/release/semantex"
echo "LLM binary:     $HOME/semantex/target-llm/release/semantex"
echo "Venv:           $HOME/semantex/benchmarks/swe_bench/.venv"
