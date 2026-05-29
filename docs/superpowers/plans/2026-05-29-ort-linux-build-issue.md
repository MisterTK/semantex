# ort 2.0.0-rc.11 Linux build — root-cause notes + path forward

> **Status:** Mac builds work; Linux builds produce non-functional `semantex index`
> binaries. Blocks all semantex-enabled SWE-bench conditions (C2, C3) until fixed.

## Symptom

On Linux x86_64, the `semantex` binary builds cleanly and runs `--version`,
`--help`, `download-models`, and `index` on directories where no files reach
the embedding stage. As soon as it tries to embed even one Python file:

```
thread 'main' panicked at .../ort-2.0.0-rc.11/src/lib.rs:196:14:
`OrtGetApiBase` must be present in ONNX Runtime dylib:
DlSym { source: "/home/.../target/release/semantex: undefined symbol: OrtGetApiBase" }
```

Two telling details:
1. The `DlSym` source path is **the semantex binary**, not `libonnxruntime.so`.
2. A minimal C program calling `dlopen("/usr/local/lib/libonnxruntime.so") +
   dlsym("OrtGetApiBase")` against the **same library** succeeds and returns
   the symbol address.

So glibc dlsym works on the lib; Rust's `libloading::Library::get()` via
`ort 2.0.0-rc.11`'s `load-dynamic` path does not.

## Why download-binaries fails on Linux

The default config (used by Mac builds) is:
```toml
ort = { version = "2.0.0-rc.11", features = ["download-binaries"] }
```

`download-binaries` runs `ort-sys`'s build script to fetch pyke's prebuilt
ONNX Runtime binary at compile time. On Linux x86_64 this **silently produces
a broken build**: `cargo build --release` finishes successfully but
`target/release/build/ort-sys-*/out/` is empty (vs. populated on macOS).
The resulting `semantex` binary has no ONNX Runtime linked in, so at runtime
the `OrtGetApiBase` symbol is missing from its export table.

Root cause of the silent build failure was not isolated — possibly a network
or extraction issue specific to pyke's Linux x86_64 tarball that fails
without surfacing an error to cargo.

## Why load-dynamic also fails

Switching to `load-dynamic` should bypass this — the binary just dlopens
libonnxruntime at runtime. But it doesn't, because of cargo feature unification:

- `fastembed 5.9` (transitive via semantex-core) default feature includes
  `ort-download-binaries-native-tls`, which adds `ort/download-binaries` to
  the union.
- `next-plaid-onnx 1.3.1` (transitive via semantex-core) declares
  `[dependencies.ort]` without `default-features = false`, so `ort/default`
  features (including `download-binaries`) flow in.

Result: cargo's resolved ort feature set includes BOTH `load-dynamic` AND
`download-binaries`. ort's behavior with both features active is undefined
and ends up using the broken static-link path.

## What was tried and discarded

(All exhaustively tested 2026-05-29 overnight; see git history for full detail.)

1. **`crates/semantex-core/Cargo.toml` → `default-features = false, features = ["load-dynamic"]`**
   on both `fastembed` and `ort`. Verified via `cargo tree -e features` that
   download-binaries was no longer in our crate's directly-declared union.
   → Still failed: transitive deps re-introduced download-binaries.

2. **Workspace `[patch.crates-io]` pointing ort to a local fork** with
   `download-binaries` and `tls-native` removed from `[features].default`.
   Verified via `cargo tree -e features` that the resolved tree no longer
   contained `download-binaries`. Verified panic source path was the patched
   fork.
   → Still failed with same `dlsym` error. This is what proves the problem
   is not just feature unification — `load-dynamic`'s runtime code path
   itself is broken under this libloading/glibc/ort combination.

3. **`ORT_DYLIB_PATH` (absolute + versioned)**, **`LD_PRELOAD`**,
   **`LD_LIBRARY_PATH=/usr/local/lib`**, **`pip install onnxruntime==1.20.1`**
   so fastembed's fallback path finds the lib in `.venv/lib/python3.12/...`.
   → All failed with the same panic. Library opens (confirmed via strace
   showing `openat(...libonnxruntime.so...)`) but `libloading::Library::get`
   reports the symbol missing.

4. **`cargo clean --release` between every attempt** to bypass any incremental
   cache that might have preserved the broken static link.
   → No change.

## Recommended path forward

In order of expected effort:

### Tactic 1 — Try a different ort version (1-2 hours)

`2.0.0-rc.11` is a release candidate. The `dlsym`-on-binary behavior could
be a known bug fixed downstream. Try, in order:

- ort 2.0.0 stable if it has shipped
- ort 2.0.0-rc.12 / rc.13 / latest 2.x
- ort 1.x (last in the 1.x line) — semantex's API surface looks compatible
  with either major, but we'd need to verify

For each: bump `crates/semantex-core/Cargo.toml`, `cargo update -p ort`,
rebuild on Linux, run `semantex index ~/swe_repos/django__django-10973` and
confirm it produces chunks. **First version that works wins.**

### Tactic 2 — `ORT_LIB_LOCATION` build env (3-5 hours)

The most Rust-idiomatic path and what ort actually supports cleanly. Goal:
make ort-sys static-link against a known-good system-installed ONNX Runtime,
sidestepping both the broken download-binaries path and the broken
load-dynamic path.

Concrete steps:

1. Download official ONNX Runtime 1.20+ release from
   `https://github.com/microsoft/onnxruntime/releases` (NOT pyke's repackage).
2. Extract to `/opt/onnxruntime` on Linux build hosts.
3. Add a workspace-root `build.rs` (or document a manual env var) that sets
   `ORT_LIB_LOCATION=/opt/onnxruntime` before cargo build.
4. ort-sys's `build/main.rs` reads `vars::SYSTEM_LIB_LOCATION` and emits
   `cargo:rustc-link-lib=onnxruntime` — proper static linking, no
   load-dynamic, no fastembed-feature battle.
5. Result: self-contained binary, no runtime env vars needed.
6. Mac builds keep using `download-binaries` (works on macOS); document both
   paths.

A workspace-level `build.rs` that auto-downloads + sets the env var would
solve this for every contributor automatically, similar to how the Mac
download-binaries path does today.

### Tactic 3 — Docker build (1-2 hours, fallback)

If both above fail:

- Add `Dockerfile.linux-build` based on `ubuntu:24.04` that pre-installs
  ONNX Runtime + rustup at known versions
- Build semantex inside the container; extract `target/release/semantex` via
  `docker cp`
- Document in README that Linux production builds use this Dockerfile

Bypasses whichever obscure interaction breaks the host build, at the cost of
adding a Docker build dependency.

## Quick environment restoration

The GCP VM `swe-bench-phase-a` boot disk was preserved (`--keep-disks=all`
on delete). To resume work with the full cached state (rust toolchain,
~/swe_repos with 100 cloned SWE-bench instances, etc.):

```bash
gcloud compute instances create swe-bench-phase-a \
  --project=search-ahmed \
  --zone=us-central1-a \
  --machine-type=n2-standard-32 \
  --provisioning-model=STANDARD \
  --disk=name=swe-bench-phase-a,boot=yes,auto-delete=yes \
  --scopes=cloud-platform \
  --metadata=enable-oslogin=TRUE
```

Then `gcloud compute ssh swe-bench-phase-a --zone=us-central1-a` and start
testing version sweeps in `~/semantex`.

## What's already in place

- `crates/semantex-core/Cargo.toml` — reverted to the working Mac config
  (`features = ["download-binaries"]`)
- `benchmarks/swe_bench/scripts/vm/vm_bootstrap.sh` — full VM bootstrap
  (system deps + rustup + semantex build + Python venv); idempotent
- `benchmarks/swe_bench/scripts/vm/vm_resume.sh` — self-healing pipeline
  driver suitable for `@reboot` cron
- C1-only Phase A baseline data preserved locally at
  `benchmarks/swe_bench/results/20260529-054239-phase_a_c1/` (77.5% across
  102 units, $232 spend)
- `benchmarks/swe_bench/scripts/compare_conditions.py` — ready to emit
  C2-vs-C1 / C3-vs-C1 delta tables (paired by instance, McNemar + paired
  bootstrap) the moment C2/C3 unit JSONs exist

## What the win looks like

A Linux build that runs:

```bash
ORT_DYLIB_PATH=... semantex index ~/swe_repos/django__django-10973
```

and produces a populated `.semantex/` directory with non-zero chunk count,
without panicking. With that, the existing benchmark harness can run Phase A
with `--conditions c1_baseline,c2_semantex_no_llm,c3_semantex_with_llm`,
the comparison table from `scripts.compare_conditions` becomes meaningful,
and the project answers its core question: "does semantex help, where, and
by how much?"
