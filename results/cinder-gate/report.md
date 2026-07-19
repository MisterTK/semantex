# Cinder — Gate Evaluation Report (compiled encoder-free indexing)

**Date:** 2026-07-18 (updated post-Task-10)
**Branch:** `feat/cinder-instant-indexing` @ `ffd0f3c` (Task-10)
**History:** Tasks 0–7 merged. Task 8's first pass measured the four gates against **v1**, which
silently shipped **exhaustive** centroid assignment (preserved verbatim in **Appendix A**).
**Task 6b** (`ba378b2`) wired the real shortlist-union assigner into `build_cinder`; Task 8's second
pass re-validated the gates and added `SEMANTEX_CINDER_EXACT_ASSIGN` (`0e258b3`, preserved in the
**Post-Task-6b** section). **Task 9** found the shortlist assigner was a single-threaded
scalar loop (the entire ~8× C2 regression vs the rayon-parallel exhaustive path) and **parallelized
it** (Post-Task-9 section). **Task 10 (this task)** parallelized the two remaining dense-stage
hotspots the Task-9 floor breakdown named — the `CinderEncoder` static+mixer token encode (~35%)
and the vendored `CompiledIndexWriter` residual subtraction (part of the ~53% writer) — see the
**Post-Task-10** section, now the current/primary record for C2/C3. Post-Task-9, Post-Task-6b, and
Appendix A are retained as the honest before/after ledger.
**Spec:** `docs/superpowers/specs/2026-07-18-cinder-*` (design `9aeb82f`, plan `5857973`)
**Scope:** Cinder validation gates **C1 (quality), C2 (build speed), C3 (build memory),
C4 (shortlist agreement)**. Cinder is off-by-default; `SEMANTEX_CINDER=1` activates it.

## TL;DR — final (post-Task-10) per-gate verdicts

| Gate | What it checks | Verdict (post-Task-10 parallel encode + residual, on top of Task-9 parallel assign) |
|---|---|---|
| **C4** shortlist agreement | shortlist-union argmax ≥99% agreement w/ exhaustive argmax | **PASS** at m=128 (0.99547) — unchanged (assigner code untouched) |
| **C1** quality (CSN hybrid nDCG@10) | py ≥0.8970, js ≥0.5329, go ≥0.7457 | **py PASS / js PASS / go FAIL** (go 0.7382, −0.0075) — unchanged (byte-identical codes, proven) |
| **C2** build speed (dense increment) | <5s CopilotKit, <1s platform | **FAIL — improved ~26% more** (CopilotKit 106.8s→**79.4s**, platform 6.0s→**4.45s**); both hotspots now parallel but the serial writer IO + IVF accumulation + sublinear encode floor keeps it far over target |
| **C3** build memory (peak-RSS increment) | <300MB over sparse baseline | **FAIL** (CopilotKit 2232MB, platform 750MB) — **rose ~+267MB/+84MB from Task-9** as the expected cost of parallel buffers; verdict unchanged (floored ~1GB+ by next-plaid construction + O(corpus) IVF) |

**Task-10 headline:** Task 9's floor breakdown named the two remaining dense-stage costs — serial
`CinderEncoder::encode_documents_with_window_ids` static+mixer encode (~35%) and the vendored
next-plaid residual/quantize/IVF writer path (~53%). Task 10 parallelized both **parallelizable**
pieces: the encode (`rayon::par_iter` over documents, order-preserving → byte-identical) and the
writer's residual subtraction (mirroring the already-parallel `compress_and_residuals_cpu`,
bit-identical). `quantize_residuals` in the shared pack tail was **already** rayon-parallel, so it
was left untouched. Result: a **real, reproducible ~26% dense-stage reduction** on both gate repos
(CopilotKit 106.8s→79.4s, platform 6.0s→4.45s; user-CPU ≈10× wall confirms parallelism is live).
**But the ~65% the Task-9 estimate hoped for did not materialize:** within the "53% writer" only the
residual subtract was serial (quantize was already parallel; the `.npy` IO writes + the O(corpus)
IVF `(centroid,doc_id)` accumulation + `concat_buffer` copy are inherently serial and untouched), and
the encode's speedup is sublinear (tokenization + per-token Vec allocation under allocator
contention). By Amdahl's law that serial writer/finalize residue now dominates. **C2 stays FAIL by
15.9×/4.4×.** Peak RSS rose modestly (parallel buffers). Recommendation: keep Cinder default-OFF;
genuinely hitting <5s/<1s needs a lighter codec or GPU assist, not more CPU parallelism — the CPU
serial floor is now the binding constraint. See **Post-Task-10** below.

### ⚠️ Headline finding (Task 6b) — wiring the real shortlist assigner INVERTS the v1 hypothesis

> **RESOLVED by Task 9 (see Post-Task-9 below):** the inversion was caused by the assigner being a
> *single-threaded scalar loop* while the exhaustive path is *rayon-parallel + batched matmul*. Task 9
> parallelized the assigner; the ~8× slowdown is gone and the shortlist path is now on par with
> exhaustive. The analysis below is the Task-6b state, retained for the ledger.

The v1 report (Appendix A) hypothesized that C2 failed *only* because v1 shipped **exhaustive**
assignment, and that wiring the C4-certified shortlist-union assigner would make Cinder "instant."
**Task 6b wired it — and it made C2 ~7–8× SLOWER, not faster.** The shortlist path is a per-token
scalar closure (build the shortlist union → `sort_unstable` → dedup → scalar dot products over
~m·|window| candidates, m=128); the exhaustive path is a single batched-BLAS
`compress_into_codes_cpu` that vectorizes the 8,192-centroid argmax "for free." On CPU the batched
matmul wins decisively despite doing more nominal FLOPs. Measured dense-stage build time, same
repos, same preserve/restore + `/usr/bin/time -l` protocol:

| repo | v1 exhaustive (Appendix A §6) | EXACT re-measure (this run) | **real shortlist (Task 6b)** | shortlist slowdown |
|---|---:|---:|---:|---:|
| platform (7,831 chunks) | 5.42s | 5.68s | **44.62s** | 7.9× |
| CopilotKit (159,013) | 102.42s | 102.42s (reused ref) | **787.81s** | 7.7× |
| gin (2,233) | — | 1.25s | **8.72s** | 7.0× |

My EXACT (`SEMANTEX_CINDER_EXACT_ASSIGN=1`) platform re-measure (5.68s) reproduces v1's exhaustive
5.42s, cross-validating that the v1 "Cinder" arm was exhaustive and that reusing v1's CopilotKit
exhaustive number as the EXACT reference is legitimate. Quality is unaffected (C1 shortlist ≈ exact
within ≤0.002 nDCG, exactly as C4's 99.5% agreement predicts) and peak memory actually **drops**
(per-token processing avoids the batched score matrix) — but the speed regression means **C2 is now
failed harder, not fixed.**

**Overall recommendation (unchanged in direction): keep Cinder default-OFF; do NOT promote.** The
feature is correct (byte-compatible index, confirmed activation, clean fallback chain), C4-safe, and
quality-neutral, but meets none of C1(go)/C2/C3. The shortlist-assignment optimization the v1 report
named as "the fix" is, as currently implemented, a *pessimization* on CPU. See §V.5 for the practical
implication (exhaustive is both faster **and** marginally higher-quality here) and §V.8 for the
recommendation and the out-of-scope follow-up (a vectorized/batched shortlist).

---

# Post-Task-10 (v4) — parallelized token encode + writer residual subtract (PRIMARY / CURRENT for C2, C3)

Task 10 targeted the two dense-stage costs the Post-Task-9 floor breakdown (§IX.2) named as the
remaining floor after assignment stopped being the bottleneck: the **serial static+mixer token
encode (~35%)** and the **vendored next-plaid residual/quantize/IVF writer path (~53%)**. Both were
parallelized where parallelizable; the change is verified byte-identical (so C1/C4 cannot move) and
the C2/C3 numbers below are the current record.

## X.1 The two fixes

- **Part A — `CinderEncoder` token encode** (`crates/semantex-core/src/embedding/cinder.rs`,
  commit `2767ad3`). `encode_documents` and `encode_documents_with_window_ids` were serial
  `texts.iter().map(…)` loops. Each document's encoding is fully independent — it reads only the
  shared immutable table/mixer/alignment (`CinderEncoder` is `Sync`, compiler-enforced by
  `par_iter`) and owns its scratch buffer — so both were switched to `rayon::par_iter`. Collecting
  an **indexed** parallel iterator preserves order, so row `i` of the output is still document `i`
  and the result is byte-identical to the serial map. `encode_documents_with_window_ids` is
  `build_cinder`'s build-path hot loop feeding `add_document_with_ids`; order preservation keeps
  each `(embedding, window)` pair matched to its document.

- **Part B — `CompiledIndexWriter` residual subtraction** (`vendor/next-plaid/src/compiled.rs`,
  commit `ffd0f3c`). `finish_encoded_chunk` computed `residuals = embedding − centroid[code]` with a
  serial `axis_iter_mut` loop over token rows. Each row depends only on its own embedding and its own
  assigned centroid, so it now mirrors the **already-parallel** `compress_and_residuals_cpu`
  (`index.rs`) form exactly:
  `residuals.axis_iter_mut(Axis(0)).into_par_iter().zip(batch_codes.as_slice().unwrap().par_iter()).for_each(…)`.
  Bit-identical to the serial pass (no cross-row interaction, no reduction, no within-row
  reordering) — only wall-clock changes. **`codec.quantize_residuals` in the shared `pack_encoded_chunk`
  tail was checked and is *already* rayon-parallel over rows** (`codec.rs:356`), so it was left
  untouched — there was no second serial lever to pull there.

No new dependency (`rayon` is already used by both crates). No change to `shortlists.rs`,
`shortlist_argmax`, or any assigner semantics.

## X.2 Correctness — byte-identity proven, not assumed

- **Part A** emits byte-identical embeddings/window-ids: order-preserving `par_iter().collect()` over
  a pure per-document function. The model-gated `zero_mixer_equals_static_center_lookup_end_to_end`
  test (which runs the parallelized `encode_documents` against the **real** LateOn-Code-edge
  tokenizer and asserts row-for-row equality vs `StaticTokenEmbedder`) **passed** in
  `cargo test --workspace`. Identical encode output → identical assigner input → identical codes.
- **Part B** is guarded directly by next-plaid's own `finish_encoded_chunk_matches_encode_index_chunk`
  byte-identity test (added by `fe39706`), plus `output_is_byte_identical_to_create_index_files`,
  `default_no_assigner_path_stays_byte_identical`, `shortlist_assigner_agrees_with_exhaustive_default`,
  and the id-aware wiring test. All ran via the **workspace-member dance** (temporarily add
  `vendor/next-plaid` to `[workspace] members`, `cargo test -p next-plaid`, revert `Cargo.toml`/
  `Cargo.lock` — verified `git diff --stat` zero on both) and **passed unchanged: 142 lib tests, 0
  failures.**
- `cargo test --workspace` = **all green** (1245 semantex-core lib tests + all crates, 0 failures).
- Both index builds produced the **exact original chunk counts** (gin 2,233 / platform 7,831 /
  CopilotKit 159,013) with no fallback warnings, confirming the parallel paths build a correct index.

## X.3 C2 — build speed (dense-stage increment), same protocol as §V.3 / §IX.2

`.semantex` preserved/restored per repo, `/usr/bin/time -l`, `RUST_LOG=semantex_core=info`,
`SEMANTEX_CINDER=1`; dense Δt = `PLAID index built via cinder` − `using cinder compiled indexing`
timestamps. Release binary at HEAD `ffd0f3c`. Raw logs: `results/cinder-gate/task10-evidence/`.

| repo (chunks) | Task-9 parallel assign | **Task-10 + parallel encode + residual** | Δ | exhaustive ref | target | verdict |
|---|---:|---:|---:|---:|---:|---|
| gin (2,233) | 1.31s | **1.08s** | −18% | 1.25s | (not a gate) | on par |
| platform (7,831) | 6.00s | **4.45s** (4.436 / 4.485, two runs) | −26% | 5.41s | <1s | **FAIL** (4.4×) |
| CopilotKit (159,013) | 106.76s | **79.39s** | −26% | 102.42s | <5s | **FAIL** (15.9×) |

Parallelism is genuinely live: platform user-CPU 60.2s over 5.9s wall ≈ **10.2×**; CopilotKit
1085.7s over 102.1s wall ≈ **10.6×**. The two platform runs (4.436s, 4.485s) are within 1% — the
number is solid, not noise.

**The ~26% gain is real and reproducible, but well short of the Task-9 estimate (~2s platform /
~30–40s CopilotKit).** Why the shortfall, honestly:

- **The "53% writer" was only partly serial.** It is assign + residual-subtract + quantize + `.npy`
  IO + IVF accumulation. Assign was already parallel (Task 9); **quantize was already parallel**
  (codec.rs); only the residual subtract was the serial lever this task pulled. The remaining
  writer work — atomic `.npy` writes per chunk, the O(corpus) `ivf_pairs.push((centroid,doc_id))`
  loop, and the `concat_buffer` copy — is inherently serial and **untouched**.
- **The ~35% encode parallelized sublinearly.** Its dominant cost is `build_doc_token_ids` →
  `tokenizer.encode` plus a per-token `Vec<Vec<f32>>` window gather; under rayon these hit allocator
  contention, so wall-clock scales well below core count.
- By **Amdahl's law**, that serial writer/finalize residue + sublinear encode is now the binding
  floor. No further CPU parallelism of these two functions can close a 15.9×/4.4× gap.

## X.4 C3 — build memory (peak RSS), same protocol as §V.4

| repo | Task-9 | **Task-10** | Δ | increment vs sparse (~68–80 MB) | verdict |
|---|---:|---:|---:|---:|---|
| gin (2,233) | 651.8 MB | **655.2 MB** | +3 MB | — | (not a gate repo) |
| platform (7,831) | 670.2 MB | **750.5 MB** (750.5 / 757.6) | **+84 MB** | ≈675 MB | **FAIL** |
| CopilotKit (159,013) | 1965.5 MB | **2232.1 MB** | **+267 MB** | ≈2152 MB | **FAIL** |

**C3 stays FAIL, and peak RSS rose** — the expected, reproducible cost of parallel buffers (rayon
holds multiple in-flight per-document encode outputs / row slices concurrently; the two platform runs
agree within ~1%, so the increase is genuine, not noise). This does **not** change the verdict: C3
was already ~6× over the 300 MB gate, floored by the **~1 GB next-plaid index-construction working
set** (Appendix A §8) plus the single-pass writer's whole-corpus in-memory IVF `(u32,i64)`
accumulation (≈12 B/token) — neither of which Task 10 touches. But it is an honest regression worth
stating: **the parallelism buys ~26% speed for ~+13% peak RSS on CopilotKit.** If C3 is ever pursued,
that trade must be accounted for (or bounded, e.g. a rayon thread cap) alongside the architectural
floor that dominates it.

## X.5 C1 / C4 sanity (post-fix)

- **C1 (quality): UNCHANGED — provably.** Both changes emit byte-identical codes (Part A: order-
  preserving encode → identical assigner input; Part B: proven bit-identical by
  `finish_encoded_chunk_matches_encode_index_chunk`). Byte-identical codes → identical index →
  identical nDCG, so the §V.2 / §IX.3 verdict stands verbatim: **py 0.9066 PASS / js 0.5354 PASS /
  go 0.7382 FAIL**. A CSN re-run would deterministically reproduce those numbers; it is redundant,
  not skipped — matching Task 9's approach.
- **C4 (agreement): UNCHANGED.** `shortlist_argmax` / `shortlist_agreement` are byte-for-byte
  untouched by this task. **PASS at m=128 (0.99547).**

## X.6 Tests, restoration, harness note

- **Tests:** `cargo test --workspace` all green (0 failures); `cargo test -p next-plaid` via the
  workspace-member dance = 142 lib tests, 0 failures (byte-identity suite unchanged); `Cargo.toml`/
  `Cargo.lock` reverted to zero diff after the dance. clippy (pinned 1.91) introduced **zero** new
  warnings in the changed code (the 10 pre-existing warnings live in unrelated files).
- **Model-gated `cinder_build_produces_searchable_index`:** fails under bare `cargo test` on an ORT
  `dlopen` (`libonnxruntime.dylib` not on the test binary's load path) — the **same pre-existing
  harness gap** Task 9 documented (§IX.4), which occurs *before* any Cinder code runs and is
  unrelated to this change. The real CLI provisions ORT at runtime, and all three C2 builds above ran
  clean end-to-end through it, which is the stronger end-to-end validation.
- **Repo restoration — verified:** gin 2,233 / platform 7,831 / CopilotKit 159,013 chunks, all
  restored. No `.semantex.pre-cinder`/`.aside` backups remain. `.gitignore` side effects handled:
  CopilotKit's build re-appended `.semantex/` (working tree was clean pre-run) → **reverted to
  committed**; platform's `.gitignore` was only stat-dirty (empty content diff) → clean; gin's
  `.gitignore` retains its **pre-existing** (pre-session) `.semantex/` append unchanged (hash
  `5dbfed10…`, not this task's doing).

## X.7 Recommendation (updated)

**Keep Cinder default-OFF.** Task 10 removes the two named CPU-parallelizable hotspots and delivers a
real ~26% dense-stage win at byte-identical quality — the assignment, encode, and residual-subtract
paths are all now rayon-parallel, and `quantize_residuals` already was. But **C2's absolute <5s/<1s
targets remain unmet by 15.9×/4.4×**, and the binding constraint is now the *serial* remainder of the
writer (per-chunk `.npy` IO, O(corpus) IVF accumulation, `concat_buffer`) + finalize (IVF
sort/dedup) + sublinear encode scaling — none of which more CPU parallelism can meaningfully move.
**C3** additionally rose ~+267 MB (CopilotKit) as the cost of parallel buffers and remains FAIL,
floored regardless by the ~1 GB next-plaid construction working set + in-memory IVF. Consistent with
Task 9's caveat: genuinely hitting the spec on this hardware likely needs a **lighter codec** (fewer
residual bytes / a cheaper writer) or **GPU assist**, not further CPU parallelization. The
CPU-parallel dense pipeline is now essentially exhausted as a lever.

---

# Post-Task-9 (v3) — parallelized shortlist assigner (superseded for C2/C3 by Post-Task-10; retained as ledger)

Task 9 targeted the C2 regression the Post-Task-6b section documented (real shortlist assignment ran
~8× SLOWER than exhaustive). **Root cause confirmed by profiling:** the `build_cinder`
`IdAwareCodeAssigner` closure ran a *single-threaded scalar* `for` loop over every token row, calling
`shortlist_argmax` serially — while the exhaustive fallback (`Codec::compress_into_codes_cpu`) is
*rayon-parallel over row batches + a batched `batch.dot(&centroids.t())` matmul*. That asymmetry
(serial+scalar-random-gather vs parallel+matmul) is the entire slowdown, despite the shortlist
touching ~1072 candidates/token vs exhaustive's 8192.

**Profiling (serial, 8192 centroids, dim 48, m=128, 9-tap window):** FULL `shortlist_argmax`
38.1 µs/token, of which dot-product+gather **66%**, union-build (sort+dedup) 19%. Dominant cost is the
per-candidate dot+gather, and the whole loop was single-threaded.

## IX.1 The fix

`crates/semantex-core/src/search/colbert_plaid_backend.rs::build_cinder` — the assigner closure now
parallelizes across the chunk's token rows:
`(0..batch.nrows()).into_par_iter().map_init(Vec::<u16>::new, |scratch, r| shortlist_argmax(…))
.collect()`. Same `shortlist_argmax`, same union/argmax/tie-break; rayon's indexed `collect` restores
row order so `out[r]` is byte-identical to the serial value **regardless of thread count**. `map_init`
reuses one tiny scratch buffer per worker (no per-token alloc, no RSS growth). No new dependency
(`rayon` already used throughout). No change to `shortlists.rs` semantics.

## IX.2 C2 — build speed (dense-stage increment), same protocol as §V.3

| repo (chunks) | Post-Task-6b serial shortlist | **Task-9 parallel shortlist** | exhaustive ref | target | verdict |
|---|---:|---:|---:|---:|---|
| gin (2,233) | 8.72s | **1.31s** (6.7×) | 1.25s | (not a gate) | on par |
| platform (7,831) | 44.62s | **6.00s** (7.4×) | 5.41s (same-binary EXACT) | <1s | **FAIL** (6.0×) |
| CopilotKit (159,013) | 787.81s | **106.76s** (7.4×) | 102.42s (ref) | <5s | **FAIL** (21×) |

**The ~8× Task-6b regression is fully eliminated** — parallel shortlist is now within a few % of
exhaustive on all three repos. **But C2's absolute targets are still unmet**, because exhaustive
*itself* is ~102s/5.4s (20×/5× over target). After the fix the assignment is only ~4s of CopilotKit's
107s (106.8 − 102.4). Raw logs + the bench harness: `results/cinder-gate/task9-evidence/` +
`bench_cinder.sh`.

**Floor breakdown (temporary stage timers, since reverted):**

| repo | dense total | encode (serial mixer) | writer (assign+residual+quantize+IO) | finalize |
|---|---:|---:|---:|---:|
| gin | 1.31s | 0.40s (30%) | 0.64s (49%) | 0.16s (12%) |
| platform | 6.00s | 2.13s (35%) | 3.16s (53%) | 0.60s (10%) |

The dense stage is dominated by the **serial `CinderEncoder::encode_documents_with_window_ids`
static+mixer encode (~35%)** and the **vendored next-plaid residual/quantize/IVF writer path (~53%)**
— both independent of the assignment method. **No assignment optimization can pass C2**; a second
lever (bitset/vectorized union) was evaluated and rejected as verdict-irrelevant (it would shave
<1% of the total). Hitting <5s/<1s requires parallelizing that broader pipeline (encode in-crate +
the vendored residual/quantize path) — escalated as a separate, larger effort.

## IX.3 C1 / C3 / C4 sanity (post-fix)

- **C1 (quality): UNCHANGED — provably.** The parallel assigner emits **byte-identical codes** to the
  serial path (new test `parallel_map_init_matches_serial_argmax` + the model-gated end-to-end test +
  `out[r]` being a pure function of row `r`). Byte-identical codes → identical index → identical nDCG,
  so the §V.2 verdict stands verbatim: **py 0.9066 PASS / js 0.5354 PASS / go 0.7382 FAIL**. A CSN
  re-run would deterministically reproduce those numbers; it is redundant, not skipped.
- **C3 (memory): NOT regressed.** Peak RSS is within noise of the Post-Task-6b serial numbers
  (gin 651.8 MB, platform 670.2 MB, CopilotKit 1965.5 MB); per-thread scratch is ~2 KB. C3 stays
  **FAIL** vs the 300 MB gate (the ~1 GB next-plaid construction floor, §V.4/§8), still below
  exhaustive's peak. Task 9 neither helped nor hurt C3.
- **C4 (agreement): UNCHANGED.** `shortlist_argmax` / `shortlist_agreement` are byte-for-byte
  untouched. **PASS at m=128 (0.99547)**.

## IX.4 Tests & restoration

`cargo test --workspace` = **1501 passed, 0 failed** (all Post-Task-6b tests unchanged + the new
determinism test). Model-gated `cinder_build_produces_searchable_index` **PASS (3.30s)** with
`ORT_DYLIB_PATH` set (bare `cargo test` fails it on an ORT `dlopen` *before* indexing — a pre-existing
harness gap, not this change). All three benchmark repos restored to original chunk counts
(gin 2,233 / platform 7,831 / CopilotKit 159,013); no `.semantex.pre-cinder` backups remain.

## IX.5 Recommendation (updated)

**Keep Cinder default-OFF.** Task 9 removes the reason the shortlist assigner looked like a
pessimization — it is now on par with exhaustive at lower memory and byte-identical quality — so the
shortlist path is a legitimate default *for the assignment step*. But C2's absolute <5s/<1s targets
are **not** met and cannot be met by assignment work alone: the binding constraint is the serial
encode + vendored residual/quantize/IVF construction. Recommended next effort (escalated, not done
here): (1) parallelize `encode_documents_with_window_ids` (in-crate, safe, ~35% of floor);
(2) parallelize next-plaid's `finish_encoded_chunk` residual subtraction + confirm `quantize_residuals`
parallelism (~53%, vendored → workspace-member dance); (3) the ~1 GB construction floor still gates
C3. Even (1)+(2) is estimated to land platform ~2s / CopilotKit ~30–40s — a big win but still short of
target on CPU, so genuinely hitting the spec likely needs a lighter codec or GPU assist.

---

# Post-Task-6b (v2) — corrected measurements (superseded for C2 by Post-Task-9; retained as ledger)

All measurements below were taken from release HEAD `0e258b3` (Task 6b real shortlist assigner +
this task's diagnostic), one process at a time, using the same protocols as the v1 record. Each
benchmark repo's `.semantex` was preserved (`mv .semantex .semantex.pre-cinder`) and restored; all
touched repos verified back to their exact original chunk counts (§V.6).

## V.1 What changed since v1 (Appendix A)

- **Task 6b (`ba378b2`, opus-reviewed & approved):** `build_cinder` now installs a real
  `IdAwareCodeAssigner` closure that wraps `shortlists::shortlist_argmax` via
  `CompiledIndexWriter::with_id_aware_assigner` / `add_document_with_ids`, threading per-token window
  ids through the writer. So `SEMANTEX_CINDER=1` now genuinely uses the shortlist-union assignment,
  not the writer's default exhaustive scan. (Appendix A §4 documents the v1 state where these were
  identical — that section is now historical.)
- **This task (`0e258b3`):** added `SEMANTEX_CINDER_EXACT_ASSIGN=1`, which (with `SEMANTEX_CINDER=1`)
  forces the writer's default exhaustive assignment by *not* installing the id-aware assigner — the
  clean toggle that makes the C1 ablation's arm (1) "Cinder full" and arm (2) "mixer+exact" genuinely
  differ, and isolates the assignment-approximation effect from the mixer's contribution. Off by
  default → shortlist is the production path. Covered by an end-to-end test asserting that with the
  flag set, on-disk codes equal the exhaustive full-scan argmax.

Query-path invariance for arm-3/4 reuse re-verified: `git diff --name-only 23a6bf1..0e258b3` touches
only build-path files (`colbert_plaid_backend.rs::build_cinder`, `cinder.rs`, `model_manager.rs`,
the two CLI commands, `main.rs`) — **no** `encode_query`/`dense_backend`/`fusion`/`query_expander`/
searcher changes — so the reused tier0+frozen (`fd136f7`) and full-hybrid (`4c8572d`) recordings stay
comparable.

## V.2 C1 — quality (real shortlist vs forced-exact), git `0e258b3`

CSN hybrid harness, seed 20260531, 200 queries × 1000-doc corpus per language (`config/csn_subset.yaml`).
Arms 1 & 2 re-run fresh at this HEAD (run-ids `cinder-gate-8resume-cinder-shortlist` /
`cinder-gate-8resume-mixer-exact`); arms 3 & 4 reused (query path unchanged, §V.1).

| Language | **Arm 1 Cinder** (real shortlist) | Arm 2 mixer+exact (exhaustive) | Arm 3 tier0+frozen | Arm 4 full-hybrid | C1 target | Verdict |
|---|---:|---:|---:|---:|---:|---|
| python | **0.9066** | 0.9085 | 0.917455 | 0.896963 | ≥0.8970 | **PASS** (+0.0096) |
| javascript | **0.5354** | 0.5356 | 0.513874 | 0.556572 | ≥0.5329 | **PASS** (+0.0025) |
| go | **0.7382** | 0.7394 | 0.726119 | 0.759336 | ≥0.7457 | **FAIL** (−0.0075) |

**Assignment-approximation effect (arm 1 shortlist − arm 2 exact):** python −0.0019, javascript
−0.0002, go −0.0012 — all within ±0.002, i.e. quality-neutral, exactly as C4's 99.5% agreement
predicts. (Exhaustive is marginally *higher* on all three — see §V.5.) **The real shortlist
assignment does not move quality.**

**Mixer effect (arm 1 − arm 3 tier0+frozen):** python −0.0109, javascript +0.0215, go +0.0121 — the
distilled micro-mixer beats the linear 5-tap mix on js/go, slightly negative on python (BM25
dominates there). Same story as v1. **Residual encoder-free gap (arm 4 − arm 1):** go 0.0211, i.e.
Cinder retains 97.2% of full-contextual on go (below the 98.2% target) — the inherent
encoder-free-vs-contextual-teacher gap the mixer only partially closes.

**C1 verdict: python PASS, javascript PASS (narrow), go FAIL (−0.0075).** Essentially identical to v1
— confirming the assignment method (shortlist vs exhaustive) is not what drives C1.

**C1 go contingency — NOT applied.** Shortfall 0.0075 (≤2pt), but (i) not mixer-attributable (the
mixer *improves* go by +0.0121) and (ii) not assignment-attributable (arm 2 exact also fails go at
0.7394, −0.0063). The shortfall is the fundamental encoder-free gap, so per the plan's bound
("mixer-attributable AND ≤2pts") the hashed bigram/trigram contingency does not apply; the honest go
FAIL is reported.

## V.3 C2 — build speed (dense-stage increment)

Method identical to Appendix A §6 (no dense-disable switch exists; dense-stage Δt =
`PLAID index built via cinder` log timestamp − `using cinder compiled indexing` timestamp under
`RUST_LOG=semantex_core=info`). Both Cinder runs logged the `using cinder compiled indexing`
confirmation and **zero** Cinder fallback warnings (CopilotKit's only WARNs were 11 benign
pre-existing PDF-parse failures on demo assets).

| Repo (chunks) | **real shortlist (Task 6b)** | EXACT re-measure | v1 exhaustive (App. A) | target | verdict |
|---|---:|---:|---:|---:|---|
| platform (7,831) | **44.62s** | 5.68s | 5.42s | <1s | **FAIL** (44.6×) |
| CopilotKit (159,013) | **787.81s** | 102.42s (ref) | 102.42s | <5s | **FAIL** (157×) |

**C2 verdict: FAIL, and materially worse than v1.** Wiring the shortlist assigner took platform
5.42s → 44.6s and CopilotKit 102s → 788s. Root cause is the scalar-closure vs batched-BLAS gap
described in the headline finding — this is a real, reproducible ~7–8× slowdown across three repos of
very different sizes (gin 7.0×, CopilotKit 7.7×, platform 7.9×), not noise.

Raw `/usr/bin/time -l` timing+RSS logs for the platform and gin shortlist/exact A/B pairs (re-run
independently, corroborating the table above within run-to-run variance: platform 44.63s/5.54s, gin
8.69s/1.23s) are now persisted on disk at
`benchmarks/relevance/results/cinder-gate-8resume-evidence/`.

## V.4 C3 — build memory (peak RSS)

Peak RSS from `/usr/bin/time -l` (`maximum resident set size`), converted bytes/1,048,576. Gate:
peak-RSS increment <300 MB over the sparse baseline (~68–80 MB, Task 7).

| Repo | **real shortlist (Task 6b)** | EXACT re-measure | v1 exhaustive (App. A) | increment vs sparse | verdict |
|---|---:|---:|---:|---:|---|
| platform (7,831) | **671.9 MB** | 1089.0 MB | 1158.8 MB | ≈592 MB | **FAIL** |
| CopilotKit (159,013) | **1965.9 MB** | 2664.4 MB (ref) | 2664.4 MB | ≈1886 MB | **FAIL** |
| gin (2,233) | 650.0 MB | 1058.5 MB | — | — | (not a gate repo) |

**C3 verdict: FAIL, but the shortlist path uses LESS memory than exhaustive** (platform −417 MB,
CopilotKit −698 MB, gin −408 MB). The exhaustive `compress_into_codes_cpu` materializes a large
batched centroid-score matrix; the shortlist closure processes one token at a time and never does.
So Task 6b trades **speed for memory**: ~8× slower but ~0.4–0.7 GB lower peak. Both still exceed the
300 MB increment gate (the ~1 GB next-plaid construction floor — Appendix A §8 — dominates either
way), so C3 stays FAIL, but the direction of the tradeoff is worth stating plainly rather than as a
strict win/loss.

## V.5 C4 (unchanged) + the practical assignment-choice data point

C4 is an offline mechanism gate (shortlist-union argmax vs exhaustive argmax agreement over 100k
sampled `(mixed-embedding, window)` pairs); it did not change with Task 6b and remains **PASS at
m=128 (0.99547)** — see Appendix A §3 for the derivation. C4 certifies that the shortlist assignment
is *quality-safe* (it agrees with exhaustive on 99.5% of tokens), and V.2 confirms that safety in
end-to-end nDCG.

**Practical implication (a data point for the ship/iterate decision, not resolved here):** on this
CPU host the exhaustive assignment is **both faster (§V.3, ~8×) and marginally higher-quality**
(§V.2, +0.0002…+0.0019 nDCG across all three languages). There is therefore a straightforward
argument for keeping **exhaustive** as Cinder's practical default going forward rather than the
shortlist path — i.e. Cinder's value (encoder-free static+mixer indexing) is realized on the
exhaustive assignment, and the shortlist assigner, though correct and C4-certified, is currently all
cost (speed) and no benefit on CPU. This is a call for the final whole-branch review / the user, not
this report.

## V.6 Fallback checks (re-verified at `0e258b3`) + repo restoration

On gin (`.semantex` preserved/restored; mixer artifact moved aside and restored):

- **Mixer moved aside + `SEMANTEX_CINDER=1`:** logged `WARN … SEMANTEX_CINDER is set but the Cinder
  artifacts failed to load … (failed to load Cinder mixer …cinder_mixer.bin: No such file or
  directory); falling back to the existing PLAID build path`, then completed the tier-chain build
  (`PLAID index built (2233 chunks)`, **not** "via cinder"), exit 0. **PASS.**
- **All artifacts present + flag OFF:** **zero** cinder log lines; normal build
  (`PLAID index built (2233 chunks)`), exit 0. Byte-identical-to-today behavior. **PASS.**

**Repo restoration — all verified:** CopilotKit 159,013 chunks, platform 7,831, gin 2,233 (+ mixer
restored). No `.semantex.pre-cinder`/`.pre-fb`/`.aside` backups remain anywhere. During cleanup a
pre-existing `.semantex/` auto-append to `.gitignore` (a semantex-index-build side effect, predating
this session) was found in **gin** (unstaged) and **CopilotKit** (staged) and reverted to committed
state; platform's `.gitignore` was clean. Untracked cross-tool artifacts (`.semantexignore`,
`graphify-out/`, platform `docs/plans/*`) were left as-is — pre-existing, not this session's.

An interrupted-session leftover was also cleaned before measuring: CopilotKit's live `.semantex` was
an incomplete cinder build (lock file, no `meta.json`, 453 MB stray WAL) with the true 159,013-chunk
original preserved in `.semantex.pre-cinder`; the incomplete build was discarded and the original
restored.

## V.7 Code changes (this task)

- **`SEMANTEX_CINDER_EXACT_ASSIGN` diagnostic** (`0e258b3`) in
  `crates/semantex-core/src/search/colbert_plaid_backend.rs::build_cinder`: env-gated branch that
  installs the shortlist assigner (default) or leaves the writer on its exhaustive default (flag set).
  Plus a `(v)` block in the ignored end-to-end `cinder_build_produces_searchable_index` test asserting
  on-disk codes equal the exhaustive full-scan argmax when the flag is set (ran with `--ignored` →
  PASS). `cargo test -p semantex-core` = 1244 lib tests pass, 0 fail.
- **Task 6b wiring fix** (`ba378b2`, landed before this task): real shortlist-union assigner in
  `build_cinder` (the code this report re-validates). Not authored here, listed for the ledger.

## V.8 Recommendation (updated for the corrected numbers)

**Keep Cinder default-OFF (opt-in behind `SEMANTEX_CINDER=1`); do NOT promote to default.** It is
correct, C4-safe, and quality-neutral, but fails C1(go), C2, and C3, and — the key change from v1 —
the shortlist-union assigner that v1 named as the fix for C2 is, as implemented, a CPU pessimization
that makes C2 ~8× worse.

- **C2 is not solved by Task 6b; the natural follow-up is a *vectorized/batched* shortlist
  implementation** (build a per-flush candidate-union matrix and do one batched matmul + argmax over
  the union, so the theoretical O(~128) vs O(8192) advantage is realized instead of being eaten by
  per-token scalar work + `sort_unstable`). That is a real engineering task and is **explicitly out of
  scope for this validation** — not attempted here.
- **Assignment-default question (for the whole-branch review / user):** since exhaustive is faster and
  marginally higher-quality on CPU (§V.5), consider making exhaustive Cinder's practical default and
  treating the shortlist assigner as experimental until it is vectorized. The
  `SEMANTEX_CINDER_EXACT_ASSIGN` knob added here makes this a one-line env toggle to A/B.
- **C3** additionally needs the ~1 GB next-plaid construction floor (Appendix A §8) and (for the
  exhaustive path) the batched score-matrix addressed for large repos.
- **C1 go** is a narrow (0.75pt) regression with the mixer net-positive — acceptable to carry while
  off-by-default.

Do not promote Cinder to default.

---

# Appendix A — v1 (pre-Task-6b) record (SUPERSEDED; retained for history)

> **The sections below are the original Task-8 first-pass report, measured against v1 which silently
> shipped EXHAUSTIVE centroid assignment (before Task 6b wired the real shortlist assigner). They are
> retained verbatim as research signal per this branch's ledger. Where they conflict with the
> Post-Task-6b sections above, the above is authoritative. Specifically: §4's "arm 2 ≡ arm 1" and §9's
> "Not added: SEMANTEX_CINDER_EXACT_ASSIGN" describe the v1 state only — the diagnostic and the real
> shortlist arm both exist as of `0e258b3` (see §V.1/§V.7), and §12's recommendation is superseded by
> §V.8.**

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
