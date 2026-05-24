# v0.3 Public Benchmark Sweep — Operational Plan

**Spec reference:** `docs/superpowers/specs/2026-05-24-semantex-v0.3-sota-design.md` §4.
**Scope of this doc:** what the human runs (and in what order) to produce the
benchmark numbers that ship with v0.3. Scaffolding (run_public.py, latency_bench.py,
COMPETITORS.md, the v0_3/ modules) is already in place.

---

## 1. Hardware & prerequisites

* macOS or Linux. Windows: untested for the sweep — file an issue if you need it.
* 32+ GB RAM recommended (running graphify + multiple HNSW indices simultaneously).
* 80+ GB free disk (datasets + competitor indices + transcripts).
* `python3 ≥ 3.10`, `cargo` (latest stable), `git`, `gh` (Copilot rows).
* Anthropic API key (env `ANTHROPIC_API_KEY`) for the B3 agent sweep — billed.
* Optional: Voyage/OpenAI API keys for the dense baseline rows.

---

## 2. Order of operations

### Stage 1 — verify the scaffold (≈10 min)

```bash
# 1. Build a fresh release binary
cargo build --release -p semantex-cli
export SEMANTEX_BIN=$(pwd)/target/release/semantex

# 2. Detect competitors
python3 benchmarks/v0_3/competitor_registry.py

# 3. Smoke test
python3 benchmarks/run_public.py --tool=semantex --benchmark=latency \
    --output=results/v0.3/smoke --target-repo=$(pwd)
cat results/v0.3/smoke/SUMMARY.json
```

If smoke fails: investigate before continuing. Do NOT run the multi-day sweep
on a broken scaffold.

### Stage 2 — prep B1 datasets (≈4–8 hours, mostly download time)

```bash
mkdir -p benchmarks/datasets/{codesearchnet-adv,cosqa,codexglue,swe-bench-lite-retrieval}
```

* **CodeSearchNet adversarial split**
  * URL: https://huggingface.co/datasets/code_search_net (use `adv` split)
  * Download: `huggingface-cli download code_search_net --repo-type dataset`
  * License: MIT — safe to commit; commit only the validation/test splits, not train.
* **CoSQA**
  * URL: https://github.com/Jun-jie-Huang/CoSQA
  * License: MIT.
* **CodeXGLUE — code-search task**
  * URL: https://github.com/microsoft/CodeXGLUE
  * License: MIT.
* **SWE-bench-Lite — retrieval split**
  * URL: https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite
  * License: MIT.

Write each dataset's MANIFEST.json (same shape as nl2code-v1/MANIFEST.json) so
`benchmarks/v0_3/b1_public_retrieval.py` can flip `present=true` for it.

### Stage 3 — curate B2 (≈8–16 hours of human time)

`benchmarks/datasets/nl2code-v1/MANIFEST.json` ships with 10 seed slots
(status="needs_curation"). Steps per row:

1. Read the seed's `question` and `notes`.
2. Check out the target repo at a recent tag — fill in `target_sha`.
3. Open the candidate `expected_files` and confirm the answer is actually there.
4. Add a second expected file if the answer spans modules.
5. Set `status="active"`, `curator=<your github handle>`.

Then draft 40 more rows (8 per language). Per spec §4.4 anti-gaming clause: the
10-row "blind set" must be drafted by an EXTERNAL contributor — do not look at
their queries until publication day.

### Stage 4 — full sweep

```bash
# Latency (B4) — minutes
python3 benchmarks/run_public.py --tool=semantex --benchmark=b4 \
    --target-repo=/path/to/indexed/repo \
    --output=results/v0.3/sweep-$(date +%Y%m%d)/

# Public retrieval (B1) — depends on dataset/tool combinations, ~30 min – 4 h
python3 benchmarks/run_public.py --tool=all --benchmark=b1 \
    --output=results/v0.3/sweep-$(date +%Y%m%d)/

# Owned NL→code (B2) — ~1 h once curated
python3 benchmarks/run_public.py --tool=all --benchmark=b2 \
    --output=results/v0.3/sweep-$(date +%Y%m%d)/

# Agent CCB (B3) — the big one. 5 repos × 5 questions × 5 tools × 3 reps.
# Budget: 4–12 hours wall, $50–$200 in API credits.
python3 benchmarks/run_b3.py --tool=semantex --reps=3 \
    --repos /path/rust-repo /path/python-repo /path/ts-repo /path/go-repo /path/java-repo \
    --output=results/v0.3/sweep-$(date +%Y%m%d)/b3-semantex/
# ... repeat per tool slug ...

# Head-to-head (B5) — aggregator, runs in seconds
python3 benchmarks/run_public.py --tool=all --benchmark=b5 \
    --output=results/v0.3/sweep-$(date +%Y%m%d)/
```

### Stage 5 — apply cap conditions

Read `results/v0.3/sweep-<date>/SUMMARY.json` against spec §4.7 (re-stated in
`docs/BENCHMARK-v0.3.md`'s checklist). If any condition trips, downgrade the
claim BEFORE writing the wins section. The honest losses section gets written
by a different actor than the wins section — that rule is structural,
non-negotiable.

### Stage 6 — leaderboard submissions

Once `docs/BENCHMARK-v0.3.md` is finalized:

* **CodeSearchNet leaderboard**: submission instructions at
  https://github.com/github/CodeSearchNet#leaderboard-submission — open a PR
  with our model's per-task scores in their JSON schema.
* **SWE-bench**: https://www.swebench.com/submit.html — submit predictions.jsonl.
* **CodeXGLUE**: submit via https://microsoft.github.io/CodeXGLUE/ leaderboard page.
* **HuggingFace MTEB code-retrieval track**: optional — exposes our model name
  to the broader retrieval community.

---

## 3. Reproducibility commitments

Per spec §4.5:

1. Commit raw transcripts (JSONL per agent run) to `benchmarks/results/v0.3/` —
   compressed if size > 100 MB.
2. Commit per-tool VERSION strings (`COMPETITOR_VERSIONS.txt` written by
   run_public.py at sweep start).
3. Commit the exact dataset SHAs used (already in each MANIFEST.json).
4. Run paired t-tests on B3 with n≥3 and report effect sizes + CIs in the
   results doc. Rows below p<0.05 are flagged "not significant" rather than
   reported as wins.

---

## 4. What this scaffold does NOT do

The W8 scaffolding deliberately stops short of:

* Running the actual B1/B2/B3 sweeps (multi-day; humans drive).
* Wiring competitor runners (each tool's invocation contract is fragile and
  changes between releases; we keep manifests honest by pinning at sweep time).
* Submitting to public leaderboards (humans verify the data first).
* Drafting the release announcement narrative.

These are the spec's §6.7 "what can't be parallelized" items.
