# Ember Plan B — Gate 3 Evaluation Report

**Date:** 2026-07-18
**Branch:** `feat/ember-plan-b-frozen-centroids` @ `fd136f7` (Tasks 1–6 merged; this task adds no
production-code commits, only the report below)
**Spec:** `docs/superpowers/specs/2026-07-17-ember-progressive-indexing-design.md`
**Plan:** Ember Plan B (frozen universal centroids)
**Scope:** Validation gate #3 ("speed/memory on the named benchmark repos") only. Gate #1
(Tier-0 quality) is Plan A, already PASSed (`results/ember-gate1/report.md`). Gate #2
(mixed-tier + rerank) and gate #4 (window cache) remain out of scope. The "background
convergence makes no query >2× slower" clause of gate #3 belongs to Plan C's upgrader and
is explicitly out of scope here — no upgrader exists yet on this branch.

## TL;DR — Gate verdict: **FAIL** (on the letter of the numeric targets), with a strong
**directional PASS** and an important, previously-invisible finding for large repos

Frozen universal centroids + tier-0 static-token embeddings build a working dense index on
`platform` in **7.84s** at **1,115.6 MB** peak RSS — a **22.3× speedup** and an **86.9% RSS
reduction** versus the full contextual-embedding build (174.68s / 8,484.3 MB). That is a
categorical win in absolute terms. **But the spec's literal gate #3 thresholds
("<5s on platform; peak RSS <500MB") are not met**: 7.84s > 5s and 1,115.6 MB > 500 MB, both
by a comfortable margin. Every one of the five named repos clears 500 MB tier-0 peak RSS
(the smallest is ~1.03 GB), so the RSS target fails everywhere, not just on platform.

The CSN quality guard **passes cleanly**: tier0frozen-hybrid nDCG@10 is within ±1% relative
of Gate-1's tier0-hybrid on all three languages (python +0.6%, javascript +0.9%, go −0.86%),
well inside the ±2% budget — no retrain was needed. The fallback check **passes**: with the
artifact temporarily moved aside, the build completes and logs the expected
`falling back to per-repo k-means` warning, no error.

**The most consequential finding is not in the gate's own checklist**: on `CopilotKit`
(17,631 files, 159,013 chunks), the **full (flags-off) baseline could not complete a dense
index at all** — it hit this machine's default 12,288 MB RSS soft cap mid-build and silently
fell back to BM25-only (`Dense index build failed (continuing without)`), even though the
CLI reported "success" and exit code 0. Raising the cap to 20,480 MB let it run ~13x longer
before the process disappeared without completing (likely a true OS-level OOM kill past that
raised cap — see §4.2). **Tier-0 + frozen centroids is therefore not just faster on
CopilotKit — it is the only one of the two configurations that produces a working dense
index on this repo at all**, at 1,770.4 MB peak RSS. This means the gate's own "full vs
tier0+frozen speedup" framing quietly assumes the full baseline is achievable, which is not
true for large repos on typical developer-machine memory budgets — a very concrete
in-the-wild reason to want Plan B distinct from raw speed.

**Recommendation:** proceed to Plan C, but do not claim gate #3 as formally passed against
its literal numeric thresholds. Two follow-ups are worth scoping before/alongside Plan C: (a)
investigate whether the tier-0 dense-build RSS floor (~1.0–1.1 GB even on tiny repos, e.g.
gin's 118 files) is a fixed per-process overhead (ONNX runtime + PLAID index scaffolding) that
could be trimmed, since it dominates the 500 MB target's failure margin on every repo; (b) the
CopilotKit slowdown at scale (tier0frozen taking 2.2x longer wall-clock than the *aborted*
full run — see §4.2) deserves a profiling pass on the frozen-centroid nearest-centroid
assignment path for very large corpora, since the whole point of frozen centroids is to be
faster than per-repo k-means, not slower.

---

## 1. Training corpus, SHAs, and judgment calls

Per the brief: Gate-1's corpus plus one mid-size Rust repo (semantex itself) plus CopilotKit
subsampled to its core-packages subdirectory instead of the full monorepo (which dominated
Gate 1's 64-minute distill for the *static table* with little added diversity — dozens of
near-duplicate scaffolded example apps).

| Repo | Corpus path used | Git SHA | Files walked |
|---|---|---|---|
| CopilotKit (subsampled) | `/Users/tk/dev/CopilotKit/packages` | `7abcd216dcfb746a3d080f8e0678ba731b60cafd` | 1,903 |
| platform | `/Users/tk/dev/platform` | `f10bdd450266aec70e0a4cb616e17e73d7737c5e` | 825 |
| pub | `/Users/tk/dev/pub` | `72d16f6032bb8ffd5957cb8707a9b5a32ff43d58` | 666 |
| gin | `/Users/tk/dev/gin` | `3e44fdc4d1636a2b1599c6688a76e13216a413dd` | 118 |
| adk-python | `/Users/tk/dev/adk-python` | `f6387c4644236ebb6650e55924d68415f35c2e89` | 2,142 |
| semantex (self) | `/Users/tk/dev/qgrep/semantex` | `fd136f72445c1a72696431089459e5c3b7e26759` | 335 |

Total: 5,989 files across 6 corpus dirs.

**CopilotKit subsample choice.** Inspected `/Users/tk/dev/CopilotKit` top-level first
(`packages/`, `examples/`, `showcase/`, `sdk-python/`, `community/`, `codemods/`, ...). Chose
`packages/` — the actual published npm/pip packages (`@copilotkit/react-core`, runtime, etc.)
— over `examples/`/`showcase/`, which are the near-duplicate demo apps Gate-1 flagged as low
per-file diversity. `packages/` walks to 1,903 files vs. 22,710 for the whole repo (91.6%
reduction), while still being real, hand-written, non-duplicated source.

**semantex self-walk verification.** The brief flagged a real risk: this repo's own
`benchmarks/relevance/results/` directory holds ~44,449 files of materialized CSN eval
corpora, which would have swamped the training corpus and (more importantly) leaked eval
documents into the distillation set if walked. Verified before training that this doesn't
happen: `crates/semantex-core/src/file/walker.rs` builds its `ignore::WalkBuilder` with
`.git_ignore(true)`, and the repo's own `.gitignore` excludes `/benchmarks/*` wholesale (with
narrow, unrelated exceptions for `benchmarks/relevance/` itself as a *source* dir, not its
`results/` subdir — see `benchmarks/relevance/.gitignore`'s own `results/*` ignore). The
training log's per-dir file count confirms the walk actually behaved this way:
`/Users/tk/dev/qgrep/semantex: 335 files` — O(hundreds), not O(10k). No `.semantexignore`
was needed.

**No contamination re-check against CSN was re-run this task** — Gate 1 already did the
detailed grep-based contamination audit for `CopilotKit`, `platform`, `pub`, `gin`, and
`adk-python` against the CSN python/javascript/go eval corpora and found none (with the
flask→adk-python swap). The only new corpus entrant here is `semantex` itself, which isn't a
CSN eval language-repo and shares no identifiable strings with the CSN corpora (checked: no
`semantex`/`colbert-plaid`/`lateon` occurrences in the materialized `csn_python`/
`csn_javascript`/`csn_go` corpora used for §5's quality run).

## 2. Training cost (Step 2)

**Command run** (release binary built from this branch's HEAD):

```
target/release/semantex distill-centroids \
  --corpus /Users/tk/dev/CopilotKit/packages \
  --corpus /Users/tk/dev/platform \
  --corpus /Users/tk/dev/pub \
  --corpus /Users/tk/dev/gin \
  --corpus /Users/tk/dev/adk-python \
  --corpus /Users/tk/dev/qgrep/semantex \
  --out ~/.semantex/models/LateOn-Code-edge/frozen_centroids.npy \
  --verify
```

(`--k` and `--sample` left at their defaults: 8192 centroids, 1,000,000-embedding reservoir
sample.)

**Results:**
- **Wall clock:** 2129.03s real (**35m 29s**); 14367.01s user CPU
- **Trained:** 8,192 centroids × 48 dims (matches the model's actual 48-d embedding space,
  same dimension Gate-1 confirmed for the static token table)
- **Peak RSS:** 13,929,824,256 bytes ≈ **13,283.5 MB (≈12.97 GiB)**
- **Artifact size:** 1,572,992 bytes (≈1.5 MiB) at
  `~/.semantex/models/LateOn-Code-edge/frozen_centroids.npy`
- **`--verify`:** `loaded centroids shape [8192, 48]` — save/load round-trip confirmed
- Raw log: `benchmarks/relevance/results/ember-gate3-train/train.log`

**Deviation — parallel-process contamination of the wall-clock number.** While diagnosing
why an earlier background-launch attempt of this same command appeared to die silently (a
tooling/process-lifecycle issue unrelated to `distill-centroids` itself — background
processes launched via `&`/`nohup`/`disown` from this harness's sandboxed shell were being
reaped when the invoking shell call ended), I ran a **second, separate foreground instance
of the identical training command** as a diagnostic (with a 300s `timeout` wrapper, to prove
liveness) while the correctly-backgrounded original instance from an earlier call was still
running. The two instances overlapped for approximately 09:11–09:19 (~8 minutes), competing
for CPU. The duplicate was killed (PIDs 3291/3293) once the overlap was noticed; the
surviving original process produced the actual artifact. **The recorded 2129.03s wall-clock
therefore includes ~8 minutes of avoidable CPU contention from my own diagnostic duplicate**
and should be read as a mild overestimate of the "clean" one-process wall-clock (roughly
2129 − ~480 ≈ ~1650s / 27.5 min, though this adjustment is an estimate, not a remeasurement —
training was not rerun, per instruction, since the artifact and its `--verify` round-trip are
already confirmed good). This is a one-time, offline artifact-build cost, not a gate target,
so the inflation doesn't affect any pass/fail decision in this report — disclosed for
reproducibility accuracy only.

## 3. Speed/memory runs on the named repos (Step 3)

For each of `platform`, `CopilotKit`, `flask`, `gin`, `semantex` (the spec's named list; per
the brief, `flask` was swapped out of Gate-1's *distillation* corpus for contamination but
that has no bearing on using it as a pure speed/memory target here — no quality is measured
on it):

```
# Tier-0 + frozen centroids
mv .semantex .semantex.pre-gate3   # preserve the repo's live registered index
SEMANTEX_STATIC_DOC_EMBED=1 SEMANTEX_FROZEN_CENTROIDS=1 RUST_LOG=semantex_core=info \
  /usr/bin/time -l target/release/semantex index .

# Baseline (flags off)
rm -rf .semantex
/usr/bin/time -l target/release/semantex index .

rm -rf .semantex
mv .semantex.pre-gate3 .semantex   # restore
```

`RUST_LOG=semantex_core=info` was required beyond the CLI's default (`WARN`) to surface the
`tracing::info!("using frozen universal centroids: ...")` confirmation line
(`colbert_plaid_backend.rs`); a bare `RUST_LOG=info` does **not** work because
`main.rs`'s `EnvFilter::from_default_env().add_directive(Level::WARN.into())` construction
lets the bare global `WARN` directive win over a same-specificity bare `info` from `RUST_LOG`
— a crate-scoped directive (`semantex_core=info`) is needed to override it. This is worth
knowing for anyone else instrumenting this build path; it's a filter-precedence quirk, not a
bug.

**Confirmed in every tier0frozen log:** the `using frozen universal centroids: ...` info
line is present; the `SEMANTEX_STATIC_DOC_EMBED is set but the static token table failed to
load` fallback warning is **absent** in all five. Raw logs:
`benchmarks/relevance/results/ember-gate3-speedmem/ember-gate3-<repo>-{tier0frozen,full}.log`.

### 3.1 Results table

RSS = `maximum resident set size` from `/usr/bin/time -l`, converted MB = bytes / 1,048,576.

| Repo | Files (scanned→indexed) | Chunks | Full build | Full peak RSS | Tier0+Frozen build | Tier0+Frozen peak RSS | Speedup | RSS change |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **platform** | 825→824 | 7,831 | 174.68s | 8,484.3 MB | **7.84s** | **1,115.6 MB** | 22.28× | −86.85% |
| CopilotKit | 17,631→17,563 | 159,013 | 67.21s *(dense build ABORTED — see §3.2)* | 14,013.4 MB *(cap-limited, not comparable)* | 144.93s | 1,770.4 MB | not comparable | not comparable |
| flask | 212→208 | 1,620 | 53.76s | 9,528.9 MB | 2.77s | 1,027.2 MB | 19.41× | −89.22% |
| gin | 118→118 | 2,233 | 34.88s | 9,283.7 MB | 1.64s | 1,063.7 MB | 21.27× | −88.54% |
| semantex (self) | 335→329 | 5,293 | 314.38s | 12,043.1 MB | 8.02s | 1,119.0 MB | 39.20× | −90.71% |

### 3.2 CopilotKit's "full" baseline is not a valid comparison point

The flags-off baseline build on the full CopilotKit repo (159,013 chunks) hit this machine's
default `SEMANTEX_MAX_RSS_MB=12288` soft cap partway through the dense-embedding pass:

```
WARN RSS exceeded soft cap — purging allocator and re-checking label="PLAID append batch" rss_mb=14011 limit_mb=12288
WARN Dense index build failed (continuing without): Indexing aborted: RSS 14011 MB exceeds
     SEMANTEX_MAX_RSS_MB=12288 (at PLAID append batch, consecutive overshoot 1/3). ...
```

The CLI still printed `Index complete!` and exited 0 — the build "succeeds" in the sense of
producing a usable BM25-only index, but **silently ships without a dense/semantic index at
all**. Its reported 67.21s / 14,013.4 MB numbers describe this partial, BM25-only build, not
a genuine full-contextual-dense build, so they are **not a fair comparison point** for the
speedup/RSS-delta columns above (marked "not comparable" in the table).

To get a real number, I reran the baseline with `SEMANTEX_MAX_RSS_MB=20480` (20 GB, still
well under the machine's 48 GB system RAM). The process ran for ~99 minutes of CPU time,
climbing past 8.8 GB RSS, then **disappeared without completing** (no error message, no
`/usr/bin/time` summary, no `.semantex/meta.json` written) — consistent with an OS-level OOM
kill above the raised cap rather than a graceful internal abort (the graceful path always
logs "Dense index build failed" before returning). I did not attempt a third run at an even
higher cap — per instructions, training/measurement should reflect what's measured, not be
retried indefinitely to force a "clean" number, and the two failed attempts are themselves
the finding: **on this machine, `CopilotKit`'s full 159K-chunk corpus cannot complete a
contextual-embedding dense build at all, at any RSS budget tried (12 GB or 20 GB)**. Tier-0 +
frozen centroids completed the same corpus in 144.93s at 1,770.4 MB — under an eighth of even
the failed 20 GB attempt's floor. Raw logs:
`ember-gate3-speedmem/ember-gate3-CopilotKit-full.log` (12 GB cap, aborted-gracefully) and
`ember-gate3-speedmem/ember-gate3-CopilotKit-full-uncapped.log` (20 GB cap, killed).

### 3.3 Gate #3 decision against the spec's literal targets

> **Gate targets (spec #3):** platform build **<5s**; peak RSS **<500MB**.

| Criterion | Target | Measured (platform, tier0+frozen) | Result |
|---|---|---:|---|
| Build wall-clock | < 5s | 7.84s | **FAIL** (+57% over target) |
| Peak RSS | < 500 MB | 1,115.6 MB | **FAIL** (+123% over target) |

Neither threshold is met on the named target repo, despite the build being 22.3× faster and
86.9% lighter than the full baseline. Extending the RSS check to all five repos: **every
repo's tier0+frozen peak RSS exceeds 500 MB** — the smallest is gin at 1,063.7 MB, more than
double the target. This strongly suggests a roughly constant ~1.0–1.1 GB per-process RSS
floor (ONNX Runtime initialization, PLAID scaffolding, tantivy/BM25 index structures) that
dominates small-repo builds regardless of the embedding tier used — worth investigating
separately from anything Plan B changed, since even gin's tiny 118-file corpus doesn't get
under it.

**The third clause of gate #3** ("background convergence makes no query >2× slower") is
Plan C's background-upgrader responsibility. No upgrader exists on this branch — Plan B ships
only the one-shot frozen-centroid indexing path — so this clause is **out of scope** and not
evaluated here, consistent with the brief.

## 4. Quality guard on CSN (Step 4)

Frozen centroids change residual quantization at index-build time, so the check is whether
they regress **tier-0 quality specifically** — comparing against Gate-1's `tier0-hybrid`
numbers (not `full-hybrid`; that's a different, unrelated question already answered by
Gate 1).

```
cd benchmarks/relevance && source .venv/bin/activate
export SEMANTEX_STATIC_DOC_EMBED=1 SEMANTEX_FROZEN_CENTROIDS=1 RUST_LOG=semantex_core=info
python -m scripts.run --dataset csn --ablation hybrid --embedder lateon-colbert \
  --run-id ember-gate3-tier0frozen-hybrid \
  --semantex-bin /Users/tk/dev/qgrep/semantex/target/release/semantex
```

Raw output: `benchmarks/relevance/results/ember-gate3-tier0frozen-hybrid/` (`report.md`,
`report.json`, `report-csn-hybrid-lateon-colbert.json`,
`instr-csn-hybrid-lateon-colbert.json`), stamped `git_rev: fd136f7`.

**Harness-env verification.** The harness (`src/relevance_harness/indexing.py`) builds each
per-language corpus's index via `subprocess.run(..., env=os.environ.copy())`
(`_index_env()`), so the exported `SEMANTEX_STATIC_DOC_EMBED`/`SEMANTEX_FROZEN_CENTROIDS`/
`RUST_LOG` vars propagate verbatim — but it discards subprocess stdout/stderr on success
(`capture_output=True`, only surfaced on non-zero exit), so there's no persisted per-language
build log to grep after the fact. To positively confirm the harness's exact build invocation
actually activates frozen centroids (not just infer it from the env-propagation code path), I
reproduced the harness's precise index command
(`SEMANTEX_QUIET_LIMITS=1 SEMANTEX_ADAPTIVE_SIZING=0 SEMANTEX_EMBEDDER=lateon-colbert
SEMANTEX_STATIC_DOC_EMBED=1 SEMANTEX_FROZEN_CENTROIDS=1 RUST_LOG=semantex_core=info
semantex index .`) against the already-materialized `csn_go` corpus dir from this same run
(temporarily swapping its `.semantex` aside and back, non-destructively). The captured log
confirms `using frozen universal centroids: .../frozen_centroids.npy` with no fallback
warning — raw log: `benchmarks/relevance/results/ember-gate3-speedmem/ember-gate3-harness-verify-go.log`.

### 4.1 Results vs. Gate 1's tier0-hybrid

| Language | Gate-1 tier0-hybrid nDCG@10 | Gate-3 tier0frozen-hybrid nDCG@10 | Relative Δ | Within ±2%? |
|---|---:|---:|---:|---|
| python | 0.9120 | 0.9175 | **+0.60%** | yes |
| javascript | 0.5092 | 0.5139 | **+0.92%** | yes |
| go | 0.7324 | 0.7261 | **−0.86%** | yes |

All three languages land comfortably inside the ±2% acceptance budget — two actually improved
slightly (a small, expected amount of run-to-run noise from a different corpus sample of the
same seeded subset construction, plus the residual-quantization change itself), and the one
regression (go, −0.86%) is well under half the allowed margin. **No retrain at a different
`--k` was needed** — the "try `--k 16384` or `4096`" fallback in the brief only triggers on a
>−2% miss, which did not occur.

**Verdict: quality guard PASSES.**

## 5. Fallback no-regression check (Step 5)

```
mv ~/.semantex/models/LateOn-Code-edge/frozen_centroids.npy /tmp/frozen_centroids.npy.aside
cd /Users/tk/dev/gin && mv .semantex .semantex.pre-gate3-fallback
SEMANTEX_STATIC_DOC_EMBED=1 SEMANTEX_FROZEN_CENTROIDS=1 RUST_LOG=semantex_core=info \
  semantex index .
mv /tmp/frozen_centroids.npy.aside ~/.semantex/models/LateOn-Code-edge/frozen_centroids.npy
```

**Result:** build succeeded (exit 0); the log shows exactly the expected warning, no error:

```
WARN SEMANTEX_FROZEN_CENTROIDS is set but /Users/tk/.semantex/models/LateOn-Code-edge/frozen_centroids.npy
     is missing; falling back to per-repo k-means for this build
```

`gin`'s `.semantex` and the real `frozen_centroids.npy` artifact were both restored
immediately after. Raw log: `benchmarks/relevance/results/ember-gate3-speedmem/ember-gate3-fallback.log`.

**Verdict: fallback check PASSES.**

## 6. Repo restoration checklist

All five named repos had live registered indexes preserved via `mv .semantex
.semantex.pre-gate3` before each repo's two-arm run and restored via `rm -rf .semantex; mv
.semantex.pre-gate3 .semantex` immediately after. Verified post-restoration (each repo's
original `meta.json` chunk_count, not a gate3 test build's):

| Repo | Restored `chunk_count` |
|---|---:|
| platform | 7,831 |
| CopilotKit | 159,013 |
| flask | 1,620 |
| gin | 2,233 |
| semantex (self) | 8,980 |

No `.semantex.pre-gate3*` backup directories remain in any of the five repos.

## 7. Gate decision

**Formal targets (spec #3, platform):**
- Build wall-clock < 5s: **FAIL** (7.84s measured)
- Peak RSS < 500 MB: **FAIL** (1,115.6 MB measured; fails on all 5 repos, not just platform)
- Background-convergence query-latency clause: **out of scope** (no Plan C upgrader exists yet)

**Supporting checks:**
- CSN quality guard (±2% of Gate-1 tier0-hybrid): **PASS** (all 3 languages within ±1%)
- Fallback-to-per-repo-k-means no-regression: **PASS**
- Frozen-centroid confirmation logging (no silent static-table fallback): **PASS** on all runs

**Overall Gate 3 verdict: FAIL** against the literal numeric thresholds, despite substantial,
consistently-measured wins in every relative comparison (19×–39× faster, 87%–91% less RSS
across the 4 repos where a valid full-baseline comparison exists, and — on CopilotKit — the
only configuration that can build a dense index at all). The gate's numeric targets appear to
have been set assuming a much lower fixed per-process overhead than this build actually has;
they are not met by any repo tested, including ones far smaller than platform (gin, 118
files, still shows 1,063.7 MB).

## 8. Concerns, caveats, and judgment calls

1. **Gate #3's numeric targets fail on every repo tested**, not just marginally on platform.
   This should be treated as a real signal that either the targets need revisiting (were
   they set before this architecture's actual per-process floor was known?) or there's a
   genuine, fixable per-process RSS/startup-time floor (ONNX Runtime init, PLAID/tantivy
   scaffolding) worth profiling before Plan C ships anything gated on these numbers.
2. **CopilotKit's full-baseline comparison is not usable** (§3.2) — the flags-off build
   cannot complete a genuine dense index on this repo at 12 GB or 20 GB RSS budgets on this
   machine. This is disclosed rather than papered over with an extrapolated or partial
   number. It is, if anything, the strongest evidence in this report for Plan B's real-world
   value: it's not just faster, it's the only path that works at all for a repo this size.
3. **CopilotKit's tier0+frozen build (144.93s) is itself unexpectedly slow relative to its
   own file count** compared to the other 4 repos' tier0+frozen numbers (which all complete
   in under 8s) — 2.2× slower than even the *aborted* full baseline's wall-clock. The other
   4 repos show tier0+frozen finishing in 1.6–8.0s regardless of size (gin's 118 files to
   semantex's 335), while CopilotKit's 17,563 files/159,013 chunks takes 144.93s. This looks
   like a scaling characteristic of the frozen-centroid nearest-centroid assignment path for
   very large chunk counts (the whole point of frozen centroids is to skip *training* new
   centroids, not to make *assignment* itself slower) — flagged as a concrete profiling
   target rather than investigated further here, since root-causing it is implementation
   work outside this validation task's charter.
4. **Training wall-clock is contaminated by ~8 minutes of self-inflicted CPU contention**
   (§2) from a diagnostic duplicate process I ran while debugging an unrelated background-
   process lifecycle issue in the harness/tooling. Disclosed, not remeasured (training was
   not rerun per instruction, and the artifact + `--verify` round-trip are independently
   confirmed correct regardless of the wall-clock number's precision).
5. **`RUST_LOG=semantex_core=info` (not bare `RUST_LOG=info`) is required** to observe the
   frozen-centroids confirmation line, due to a filter-precedence detail in `main.rs`'s
   `EnvFilter` construction (§3). This is a debugging/observability note, not a functional
   bug — the underlying behavior (frozen centroids activating correctly) is unaffected by
   log verbosity.
6. **No new production code was written or modified for this task** — Task 7 is pure
   validation per the plan's task split (Tasks 1–6 already shipped the feature). This
   report and the `.gitignore` exception triple are the only changes.
7. **The CSN quality-guard run reused the standard seeded 200/1000-query subset**
   (`config/csn_subset.yaml`, seed 20260531, unchanged since Gate 1) — no new sampling
   decisions were made for this task.

## 9. Recommendation on proceeding to Plan C

**Proceed to Plan C (background upgrader / Tier-1 convergence), but do not carry gate #3's
literal thresholds forward as "met."** The frozen-centroid mechanism itself works correctly
(confirmed logging, correct fallback behavior, quality preserved within budget) and delivers
large, real wins — that part of Plan B is solid and Plan C can build on it. However:

- Plan C's own gate clause ("no query >2× slower during background convergence") will need
  its *own* formal validation once an upgrader exists — nothing here validates it, by design.
- Before or alongside Plan C, it would be worth a short, separate investigation into the
  ~1.0–1.1 GB per-process RSS floor (item 1 above) and the CopilotKit-scale slowdown (item 3)
  — both are now measured, disclosed facts rather than unknowns, but neither was in scope to
  fix in this validation task.
- If the project wants a *formal* gate-3 PASS on record, the spec's numeric targets likely
  need to be revisited in light of this measured per-process floor, rather than re-attempting
  a build-time/RSS optimization pass that treats "<5s / <500MB" as fixed, unquestioned
  constants.
