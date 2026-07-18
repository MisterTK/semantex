# Cinder — Gate Evaluation Report (compiled encoder-free indexing)

**Date:** 2026-07-18
**Branch:** `feat/cinder-instant-indexing` @ `23a6bf1` (Tasks 0–7 merged; this task adds two
commits: the code additions below + this report)
**Spec:** `docs/superpowers/specs/2026-07-18-cinder-*` (design `9aeb82f`, plan `5857973`)
**Scope:** Cinder validation gates **C1 (quality), C2 (build speed), C3 (build memory),
C4 (shortlist agreement)** — the whole-feature acceptance gate. Cinder is off-by-default;
`SEMANTEX_CINDER=1` is required to activate it.

## TL;DR — per-gate verdicts

| Gate | What it checks | Verdict |
|---|---|---|
| **C4** shortlist agreement | shortlist-union argmax ≥99% agreement with exhaustive argmax | **PASS** at m=128 (0.99547); m=32→0.9571, m=64→0.9874 |
| **C1** quality (CSN hybrid nDCG@10) | py ≥0.8970, js ≥0.5329, go ≥0.7457 | **py PASS / js PASS / go FAIL** (go 0.7377, −0.0080) |
| **C2** build speed (dense increment) | <5s CopilotKit, <1s platform | **FAIL** (CopilotKit 102.4s, platform 5.42s) |
| **C3** build memory (peak-RSS increment) | <300MB over sparse baseline | **FAIL** (platform ≈1.08GB, CopilotKit ≈2.58GB increment) |

**Overall recommendation: keep Cinder default-OFF as shipped; iterate before promoting.**
The feature is *correct* (byte-compatible index, activation confirmed on real repos, fallback
chain solid) but does **not** deliver its two headline promises in v1: it is **not "instant"**
(C2) and **not low-memory** (C3), and it carries a **narrow quality regression on Go** (C1).
Crucially, the reason C2/C3 fail is that v1 ships **exhaustive** centroid assignment — the
shortlist-union assignment that would make it instant is *deferred*, and this report's C4 result
is exactly the validation that turning it on is safe (99.5% agreement at m=128). The clear next
step to realize Cinder's value is to wire the shortlist assigner into `build_cinder`. Do not
promote to default until C2 (and ideally C3) are addressed.

---

## 1. Corpus, SHAs, and reuse justification

### 1.1 Training corpus (Gate-3 recipe, all 6 dirs)

| Repo | Corpus path | Git SHA | Files walked |
|---|---|---|---:|
| CopilotKit (subsampled) | `/Users/tk/dev/CopilotKit/packages` | `7abcd216dcfb746a3d080f8e0678ba731b60cafd` | 1,903 |
| platform | `/Users/tk/dev/platform` | `f10bdd450266aec70e0a4cb616e17e73d7737c5e` | 825 |
| pub | `/Users/tk/dev/pub` | `72d16f6032bb8ffd5957cb8707a9b5a32ff43d58` | 666 |
| gin | `/Users/tk/dev/gin` | `3e44fdc4d1636a2b1599c6688a76e13216a413dd` | 118 |
| adk-python | `/Users/tk/dev/adk-python` | `f6387c4644236ebb6650e55924d68415f35c2e89` | 2,142 |
| semantex (self) | `/Users/tk/dev/qgrep/semantex` | `23a6bf17c666ed00e2941b4bd3352367b14c89dd` | 343 |

Total: 5,997 files across 6 dirs. Same recipe/SHAs as Ember Gate-3 (the self-walk is at this
branch's HEAD). CSN-contamination was audited in Gate 1 (flask→adk-python swap) and re-checked
in Gate 3; no new corpus entrant here, so no re-audit was needed.

### 1.2 Recording reuse (arms 3 & 4) — comparability verified

Arms 3 (tier0+frozen) and 4 (full-hybrid) reuse prior recordings rather than re-running.
Justification (the brief's required check): `git log 4c8572d..HEAD` and `fd136f7..HEAD` over
the query path show **no** diffs to `encode_query` (`embedding/colbert.rs`),
`search/dense_backend.rs`, `search/fusion.rs`, `search/query_expander.rs`, or the searcher.
The *only* commit touching `crates/.../search/` since `fd136f7` is `23a6bf1`, and it is
**build-path-only** (new `cinder.rs`, the `window_ids_at` helper, `cinder_for_build`,
`build_cinder`, and an early-return INSERT at the top of `build_streaming_ids`). Both recordings
are therefore directly comparable to today's query path. Numbers were re-read from the recorded
`report-csn-hybrid-lateon-colbert.json` and match the brief's quoted baselines exactly.

## 2. Training artifacts (Step 1)

Release binary built from HEAD `23a6bf1`. Both training steps run under `/usr/bin/time -l`,
one process at a time (no concurrent training). Raw logs (gitignored):
`benchmarks/relevance/results/cinder-gate-train/`.

### 2.1 Mixer (`distill-mixer`)

```
target/release/semantex distill-mixer \
  --corpus /Users/tk/dev/CopilotKit/packages --corpus /Users/tk/dev/platform \
  --corpus /Users/tk/dev/pub --corpus /Users/tk/dev/gin \
  --corpus /Users/tk/dev/adk-python --corpus /Users/tk/dev/qgrep/semantex \
  --out ~/.semantex/models/LateOn-Code-edge/cinder_mixer.bin --verify
```

- **Holdout cosine: mixer = 0.9433704, linear = 0.9334068, pairs = 1,900,000.**
  The micro-mixer reconstructs the teacher's contextual embeddings **+0.0100 cosine** better than
  the 5-tap linear mix — the first C1 leading indicator (small but positive; it predicted the
  small mixer uplift seen in C1 §5).
- Wall clock: **1705.87s (28m 26s)**, 12906.04s user CPU. Peak RSS: 8,692,776,960 B ≈ **8,290 MB**.
- `--verify: loaded mixer dims=48`. Artifact 11,344 B (≈11 KB, f32 mixer — matches spec's size note).

This is a one-time offline artifact cost (not a gate target).

### 2.2 Shortlists (`derive-shortlists`)

Pure offline transform of the static table + frozen centroids (no corpus walk). Initial
`--m 32` run: **8.10s, 34 MB peak, vocab 50,370**. The shipped artifact was subsequently
re-derived at **m=128** during C4 tuning (§3) — deterministic; final
`cinder_shortlists.bin` = SXCS v1, vocab 50370, m 128, 12,894,992 B.

## 3. C4 — shortlist agreement (Step 2)

**Gate: ≥99%.** Added a `derive-shortlists --agreement <DIR>` option (code §7) that builds a
`CinderEncoder` from the model dir, reservoir-samples **100,000** `(mixed embedding, 9-tap
window)` pairs uniformly across a corpus (seed 20260718, deterministic), and prints the fraction
whose shortlist-union argmax equals the exhaustive argmax over all 8,192 centroids — measured
over the full training corpus.

| m | shortlist_agreement | verdict |
|---:|---:|---|
| 32 | 0.957050 | FAIL |
| 64 | 0.987370 | FAIL (−0.26pt) |
| **128** | **0.995470** | **PASS** |

**Verdict: C4 PASS at m=128.** The brief named `--m 64` as the fallback; it landed at 0.9874,
a 0.26-point miss. I extended the *same monotonic lever* to m=128 (agreement → 1.0 as
m → n_centroids), which clears the gate. This is **free for v1**: the shortlist assigner is
deferred (see §4), so the shortlists are validated-but-unconsumed in the shipped build path, and
a larger m costs nothing at build time and does not affect C1. The shipped
`cinder_shortlists.bin` is m=128. **Judgment call:** extending beyond the brief's literal m=64 —
flagged for review, trivially reversible.

C4's real meaning: it certifies that the deferred shortlist-union assignment (the optimization
that would make Cinder actually fast — see C2) is safe to enable, agreeing with exact assignment
on 99.5% of tokens.

## 4. Ablation design note — v1 ships EXHAUSTIVE assignment (arm 2 ≡ arm 1)

The plan's C1 ablation assumed arm 1 ("Cinder full") uses the shortlist-union assigner and that
`SEMANTEX_CINDER_EXACT_ASSIGN=1` toggles it off to produce arm 2 ("mixer+exact"). **The shipped
v1 does not work that way.** `build_cinder` constructs `CompiledIndexWriter::new(...)` and never
calls `.with_assigner(...)` — the writer's shortlist hook is documented as *deferred*
(`colbert_plaid_backend.rs:557-559`) and is exercised only in vendored tests. So
`SEMANTEX_CINDER=1` already uses the writer's **default exhaustive** `compress_into_codes_cpu`
assignment, which is byte-compatible with the reference PLAID layout.

Consequences:
- **Arm 2 ≡ Arm 1** byte-for-byte in v1 (both exhaustive). A `SEMANTEX_CINDER_EXACT_ASSIGN`
  env knob would have nothing to toggle (no shortlist assigner is installed to override), so it
  was **not added** — a dead knob would be misleading. Wiring the *real* shortlist assigner is a
  non-trivial change (`CodeAssigner = Box<dyn Fn(&Array2<f32>) -> Array1<usize>>` receives only
  the embedding matrix, **not** token ids, which the shortlist union needs), out of scope for a
  validation task.
- The C1 attribution therefore reduces to **arm 1 (Cinder) vs arm 3 (tier0+frozen)** = the
  micro-mixer's effect over the 5-tap linear mix (both exhaustive + frozen centroids), with C4
  separately covering the assignment-approximation question offline.

This was escalated to the team lead; the report proceeds with the honest arm-2≡arm-1 collapse.

## 5. C1 — quality (Step 4)

CSN hybrid harness, seed 20260531, 200 queries × 1000-doc corpus per language (unchanged
`config/csn_subset.yaml`). Arm 1 run fresh with `SEMANTEX_CINDER=1` (run-id
`cinder-gate-cinder-full`, git_rev 23a6bf1); arms 3/4 reused (§1.2).

**Cinder activation confirmed in-harness:** rebuilding the materialized `csn_go` corpus with
`SEMANTEX_CINDER=1 RUST_LOG=semantex_core=info` logged `using cinder compiled indexing` +
`PLAID index built via cinder (1000 chunks)` (no fallback). The in-between nDCG values below
(Cinder between tier0 and full on every language) independently confirm the mixer path ran.

### 5.1 Ablation matrix (nDCG@10)

| Language | Arm 1 **Cinder** (mixer+frozen+exhaustive) | Arm 2 mixer+exact | Arm 3 tier0+frozen (linear) | Arm 4 full-hybrid | C1 target | Verdict |
|---|---:|---:|---:|---:|---:|---|
| python | **0.908486** | ≡ arm 1 | 0.917455 | 0.896963 | ≥0.8970 | **PASS** (+0.0115) |
| javascript | **0.535504** | ≡ arm 1 | 0.513874 | 0.556572 | ≥0.5329 | **PASS** (+0.0026) |
| go | **0.737671** | ≡ arm 1 | 0.726119 | 0.759336 | ≥0.7457 | **FAIL** (−0.0080) |

Arms 3/4 git revs: tier0+frozen `fd136f7` (Gate-3 recording), full-hybrid `4c8572d`
(2026-07-16 head-to-head). Full metrics (mrr/recall) in the raw run JSON.

### 5.2 Attribution

- **Assignment: 0 contribution.** v1 uses exact/exhaustive assignment (arm 1 ≡ arm 2), which is
  byte-compatible with the reference PLAID codec. No quality is lost to assignment.
- **Mixer effect (Cinder − tier0+frozen):** python **−0.0090**, javascript **+0.0216**,
  go **+0.0116**. The distilled micro-mixer is **net-positive** on js/go (it beats the linear
  5-tap mix, as the holdout cosine predicted) and slightly negative on python (where BM25 already
  dominates and the linear mix happens to fuse marginally better).
- **Residual encoder-free gap (full-hybrid − Cinder):** the go gap is 0.0216 — Cinder retains
  **97.2%** of full-contextual on go, below the **98.2%** target. This is the inherent
  encoder-free-vs-contextual-teacher gap that the mixer *partially* (not fully) closes.

### 5.3 Contingency decision — NOT applied

The go shortfall is 0.0080 (0.8pt), which is **≤2 points**. But the C1 contingency (hashed
bigram/trigram contextual tables) is gated on the mixer being **clearly the shortfall** — and it
is not: the mixer *improves* go by +0.0116, i.e. it helps rather than bottlenecks. The residual
gap is the fundamental encoder-free limitation, not a mixer defect. Per the plan's stated bound
("mixer-attributable AND ≤2pts") and the instruction not to stretch the lever, the contingency
**does not apply**; I report the honest go FAIL with attribution. (A bigram/trigram-table feature
+ retrain is also a large investment unwarranted by a 0.8pt narrow miss where the mixer already
helps.) Escalated to the team lead with this recommendation.

**C1 verdict: python PASS, javascript PASS (narrow), go FAIL.**

## 6. C2 / C3 — build speed & memory (Step 3)

**Method — no dense-disable switch exists.** `semantex index --help` exposes only `--force`, and
there is no `SEMANTEX_DENSE`/sparse-only build env var (grep-confirmed). Per the brief's stated
fallback, the **dense-stage increment** is measured from `RUST_LOG=semantex_core=info`
stage-boundary timestamps (the default `tracing_subscriber::fmt()` emits timestamps), and
**tier0+frozen is run as the reference arm**. Sparse-stage baseline (~68–80 MB) is taken from
Task 7's vmmap stage trace on the identical encoder-free build path. `.semantex` was
preserved/restored on both real repos (`mv .semantex .semantex.pre-cinder` … restore).

### 6.1 Results

| Repo (chunks) | Arm | dense-stage (log Δt) | total build | peak RSS |
|---|---|---:|---:|---:|
| **platform** (7,831) | Cinder | **5.42s** | 6.86s | **1,158.8 MB** |
| | tier0+frozen | 5.27s | 6.73s | 1,129.8 MB |
| **CopilotKit** (159,013) | Cinder | **102.42s** | 125.19s | **2,664.4 MB** |
| | tier0+frozen | 126.28s | 149.85s | 2,023.7 MB |

(dense-stage Δt = `PLAID index built via cinder` timestamp − `using cinder compiled indexing`
timestamp; tier0+frozen uses the `using frozen universal centroids` → `PLAID index built` span.)

Both Cinder runs logged the `using cinder compiled indexing` confirmation and **zero** Cinder
fallback warnings (the only WARNs were benign pre-existing PDF-parse failures on CopilotKit's
demo assets, unrelated to Cinder).

### 6.2 C2 verdict — FAIL

| Criterion | Target | Cinder | Result |
|---|---|---:|---|
| dense increment, platform | <1s | 5.42s | **FAIL** (5.4×) |
| dense increment, CopilotKit | <5s | 102.42s | **FAIL** (20×) |

Cinder is **not instant**. On platform its dense stage (5.42s) is *the same* as tier0+frozen
(5.27s); on CopilotKit it is ~19% faster than tier0+frozen (102s vs 126s) — the single-pass
`CompiledIndexWriter` avoiding `update_append`'s re-merge churn shows at scale — but still ~100s.
**Root cause:** v1 assigns each token by exhaustive `compress_into_codes_cpu` over 8,192
centroids, which dominates the dense stage; the shortlist-union assignment (C4-validated safe at
99.5%) that would collapse this cost is deferred. C2 is thus blocked on exactly the deferred
optimization C4 certified.

### 6.3 C3 verdict — FAIL

**Target: peak-RSS increment <300 MB** over the sparse baseline (~68–80 MB, Task 7). Cinder peak
is 1,158.8 MB (platform) and 2,664.4 MB (CopilotKit) → increments of ≈**1.08 GB** and ≈**2.58 GB**,
FAIL by ~4–9×. Even against a generous multi-hundred-MB sparse baseline for the larger repo, the
increment stays far above 300 MB. Notably Cinder uses **more** peak RSS than tier0+frozen at scale
(2,664 vs 2,024 MB on CopilotKit) — its single-pass writer accumulates the full-corpus IVF
`(centroid, doc_id)` pairs in memory before `finalize`, which scales with total tokens, whereas
`update_append` flushes incrementally. See the floor autopsy (§8).

## 7. Fallback checks (Step 5)

On gin (`.semantex` preserved/restored; original 2,233 chunks restored):

- **Mixer moved aside + `SEMANTEX_CINDER=1`:** logged
  `WARN … SEMANTEX_CINDER is set but the Cinder artifacts failed to load … (failed to load Cinder
  mixer …cinder_mixer.bin: No such file or directory); falling back to the existing PLAID build
  path`, then completed the tier-chain build (`PLAID index built (2233 chunks)`), exit 0. **PASS.**
- **All artifacts present + flag OFF:** **zero** cinder log lines; normal build
  (`PLAID index built (2233 chunks)`), exit 0. Byte-identical-to-today behavior confirmed. **PASS.**

Mixer restored immediately after. **Fallback checks PASS.**

## 8. Memory-floor autopsy (lifted from Task 7)

Task 7 (`/tmp/cinder-floor-autopsy.md`) profiled the ~1.0 GB process-RSS floor on the
encoder-free build path (gin, `SEMANTEX_STATIC_DOC_EMBED=1 SEMANTEX_FROZEN_CENTROIDS=1`) with
`vmmap` stage traces. Findings, adapted here because Cinder's `build_cinder` reuses the same
`INITIAL_BUILD_CHUNKS=2000` bound and the same next-plaid construction primitives:

- The floor is **real and reproducible** (3/3 fresh builds, 950 MB–1.0 GB max RSS on 118-file
  gin) and is **entirely attributable to the PLAID index-construction working set**, not to ORT
  (empirically never `dlopen`'d on encoder-free paths — `lsof` sampled 8×, zero onnxruntime),
  not to tantivy/BM25 (28–80 MB during that stage), not to sqlite, and not to the static/Cinder
  artifacts (a few MB of flat tables). A `status`-only control run is 12.5 MB max RSS, so the
  floor is specific to index construction, not a fixed process tax.
- The vmmap stage trace showed the climb from ~440 MB to the ~857 MB peak happening **inside the
  single next-plaid construction call** over the bounded ≤2000-chunk prefix. On this macOS host
  the dominant region is mislabeled "IOAccelerator" by vmmap but is **not** GPU/Metal/Accelerate
  memory (ruled out via `otool -L`, the cargo feature tree, and `lsof`); it is best explained as a
  vmmap tag artifact over a large anonymous heap arena backing the residual-quantization /
  IVF-assignment buffers. The exact internal mechanism inside the vendored next-plaid crate was
  not line-by-line instrumented (out of Task 7's scope).
- **Implication for Cinder (confirmed by this report's C3):** Cinder pays the same floor
  (platform 1,159 MB, gin-scale ≈1.0 GB) because `CompiledIndexWriter` uses the same bounded
  prefix + residual-codec construction. At large scale it is *worse* than tier0+frozen because
  its single-pass writer also holds the whole-corpus IVF `(u32,i64)` pair vector in memory until
  `finalize` (≈12 B/token). Reducing the floor means shrinking `INITIAL_BUILD_CHUNKS` and/or
  streaming the IVF accumulation to disk — algorithm-level changes to a vendored engine, out of
  scope here, flagged for whoever picks up memory-floor work.

## 9. Code changes (Steps 2 & 4 + a blocking bug fix)

Committed with this task (commit 1 of 2):

1. **Blocking path-resolution bug fix** (necessary to run Step 1 at all).
   `distill-mixer`/`derive-shortlists` loaded the static table + frozen centroids from
   `config.models_dir()` — the *parent* `~/.semantex/models/` — but every other code path (and
   `distill-static-table`'s `--out`) puts artifacts in the `LateOn-Code-edge` **subdir** that
   `ensure_colbert_model` resolves. These two commands had never been run end-to-end, so the bug
   was latent. Fix: added `model_manager::colbert_model_dir(models_dir)` (resolves the subdir
   without the download side effect) and pointed both commands at it. Setting `SEMANTEX_MODEL_DIR`
   to the subdir was rejected as a workaround because it would break `ensure_colbert_model`'s
   teacher resolution (nested `LateOn-Code-edge/LateOn-Code-edge`).
2. **`derive-shortlists --agreement <DIR>` (gate C4).** New repeatable CLI option +
   `CinderEncoder::agreement_samples()` (reservoir sampler over `(mixed embedding, window)` pairs,
   routed through a shared `mix_at` helper so it can never diverge from `encode_ids`) + a private
   `SplitMix64`. Reuses the existing `shortlists::shortlist_agreement`.

**Not added:** `SEMANTEX_CINDER_EXACT_ASSIGN` — a no-op in v1 (see §4); adding a dead knob would
be misleading.

Files: `crates/semantex-core/src/embedding/model_manager.rs`,
`crates/semantex-core/src/embedding/cinder.rs`,
`crates/semantex-cli/src/commands/distill_mixer.rs`,
`crates/semantex-cli/src/commands/derive_shortlists.rs`,
`crates/semantex-cli/src/main.rs`. Report + `.gitignore` triple in commit 2.

## 10. Repo restoration checklist

All touched real indexes preserved via `mv .semantex .semantex.pre-cinder` and restored:

| Repo | Restored chunk_count | Backend |
|---|---:|---|
| platform | 7,831 | colbert-plaid |
| CopilotKit | 159,013 | coderank-hnsw |
| gin | 2,233 | colbert-plaid |

The materialized `csn_go` harness corpus (used for the in-harness activation check) was also
preserved/restored. No `.semantex.pre-cinder` / `.aside` / `.orig` backups remain anywhere; the
model dir holds the trained `cinder_mixer.bin` + `cinder_shortlists.bin` (m=128). Verified.

## 11. Judgment calls & concerns

1. **C4 extended to m=128 beyond the brief's named m=64** (0.9874 → 0.9955). Same monotonic lever,
   free for v1 (deferred assigner). Reversible.
2. **Arm 2 ≡ Arm 1 (§4):** shipped v1 uses exhaustive assignment; no `SEMANTEX_CINDER_EXACT_ASSIGN`
   knob added. Escalated.
3. **C1 go contingency NOT applied (§5.3):** shortfall ≤2pt but not mixer-attributable (mixer
   helps go). Reported as honest FAIL. Escalated.
4. **Blocking path bug in Task 4's commands, found + fixed (§9).** Flagged for the branch review.
5. **C2/C3 measured without a dense-disable switch** (none exists) — dense increment via log
   timestamps + tier0+frozen reference; sparse baseline from Task 7. Method documented.
6. **C3 sparse baseline** is Task 7's gin-scale figure; the exact per-repo sparse RSS was not
   re-measured, but C3 fails by ~4–9× so the verdict is robust to the baseline's precision.

## 12. Recommendation

**Ship as-is (default-OFF); iterate before promoting.** Cinder is correct and safe — byte-compatible
index, confirmed activation, clean fallback chain, off by default — but v1 does not meet C1
(go), C2, or C3:

- **C2/C3 are blocked on the deferred shortlist-union assigner.** C4 (PASS, 99.5% at m=128) is
  precisely the certification that enabling it is safe. Wiring it into `build_cinder` (threading
  token ids to a token-id-aware `CodeAssigner`) is the single highest-value follow-up — it targets
  the exhaustive-assignment cost that dominates the dense stage and is the reason "instant" isn't
  realized. C3 additionally needs the next-plaid working-set floor (Task 7) and Cinder's
  in-memory IVF accumulation addressed for large repos.
- **C1 go is a narrow (0.8pt) regression**, with the mixer net-positive; acceptable to carry while
  off-by-default, but it (with C2/C3) means Cinder should **not** become the default indexing path
  yet.

Do not promote Cinder to default. Keep it opt-in behind `SEMANTEX_CINDER=1`.
