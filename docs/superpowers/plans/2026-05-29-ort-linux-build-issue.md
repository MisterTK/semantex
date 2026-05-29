# ort / ONNX Runtime cross-platform build — root cause + fix

> **Status (2026-05-29, revised):** RESOLVED on macOS; ready for Linux
> confirmation. The original root-cause analysis in this doc was **wrong**; the
> real cause and the shipped fix are below. Supersedes the first revision.

## TL;DR

It was never an `ort` bug, a Linux bug, or a network/CDN bug. `next-plaid-onnx`
forces `ort` into **pure `load-dynamic` mode on every platform**, so the binary
never embeds ONNX Runtime and must `dlopen` a `libonnxruntime` at runtime. macOS
"worked" only because Homebrew happened to have a compatible library installed;
Linux had none (or an incompatible one). Fix: semantex now **provisions the
official Microsoft ONNX Runtime (≥ the version `ort` requires) at runtime**,
caching it under `~/.semantex/runtime/<version>/`, exactly like it already
caches the ColBERT model — and points `ORT_DYLIB_PATH` at it. No build-time CDN
dependency, no system package required, identical on Linux/macOS/Windows.

## Root cause (definitive, from the exact locked crate sources)

Locked versions: `ort`/`ort-sys` `2.0.0-rc.11`, `fastembed` `5.9.0`,
`next-plaid-onnx` `1.3.1`.

1. **`next-plaid-onnx 1.3.1` forces a contradictory `ort` feature set.** Its
   manifest declares `ort` with `features = ["ndarray", "load-dynamic"]` **and
   omits `default-features = false`**, so it pulls `ort`'s *default* set (which
   includes `download-binaries`) **plus** `load-dynamic` (→ `ort-sys/disable-linking`).
   Cargo features are additive across the graph, so **nothing in semantex's own
   `Cargo.toml` can remove `download-binaries` or `disable-linking`** — both are
   always present.

2. **`disable-linking` wins and short-circuits the build.** `ort-sys`'s
   `build/main.rs` begins with:
   ```rust
   if env::var("DOCS_RS").is_ok() || cfg!(feature = "disable-linking") {
       // load-dynamic enabled → no need to link, and we don't download anything
       return; // ← no download, no static link. out/ is empty BY DESIGN.
   }
   ```
   So `download-binaries` is a complete **no-op**, and the empty
   `target/release/build/ort-sys-*/out/` the previous revision called a "silent
   build failure" is correct, intended behavior. The binary is in **pure
   `load-dynamic` mode on macOS and Linux alike** and always needs a runtime lib.

3. **`ort` rc.11 requires ONNX Runtime ≥ 1.23.** `ort-sys/src/version.rs` sets
   `ORT_API_VERSION = 23`, and `ort`'s `load_dylib_from_path` hard-errors if the
   loaded library's minor version is `< 23`. (pyke's own prebuilt for rc.11 is
   `ms@1.23.2`.)

4. **macOS worked by accident; Linux didn't.** `find_ort_dylib()` found
   Homebrew's `libonnxruntime.dylib` (1.26.0 on the dev Mac — ≥ 1.23, so it
   passes) and set `ORT_DYLIB_PATH`. The Linux VM had no resolvable, compatible
   `.so`, so `ort`'s lazy `setup_api()` panicked on `OrtGetApiBase`.

### Why the previous revision's experiments all failed

| Previous claim / attempt | Reality |
|---|---|
| "`download-binaries` silently produces a broken Linux build" | It never runs — `disable-linking` early-returns. Empty `out/` is expected. |
| "`load-dynamic`'s runtime path is itself broken" | `load-dynamic` was the *only* active mode the whole time, on both OSes. It works fine on Mac. |
| Tactic 1: "try other `ort` versions" | Pointless — not an `ort` bug. (rc.12 just moves the floor to ONNX Runtime 1.24.) |
| Tactic 2: "`ORT_LIB_LOCATION` static link" | **Impossible while `load-dynamic` is forced** — the `ORT_LIB_LOCATION` block in `build/main.rs` sits *after* the `disable-linking` early-return, so it is unreachable. |
| Item 3: `pip install onnxruntime==1.20.1` | **1.20 < 1.23** → `ort` rejects it as incompatible; also wrong filename for the OS loader. |

## What shipped

All in `crates/`, repo-agnostic, OSS-quality (env-overridable, no hardcoded
filesystem paths; the version + MS release URL are infra constants like the
existing HuggingFace model URL).

- **`crates/semantex-core/src/embedding/runtime_manager.rs`** (new):
  - `ONNXRUNTIME_VERSION = "1.23.2"` — **must be bumped in lockstep with `ort`**
    (it has to satisfy `ort_sys::ORT_API_VERSION`'s minor-version floor).
  - `asset_name()` — host `(os, arch)` → official MS release asset
    (`onnxruntime-{linux-x64,linux-aarch64,osx-arm64,osx-x86_64,win-x64,win-arm64}-1.23.2.{tgz,zip}`).
  - `find_onnxruntime(root)` — locate an already-cached lib (no download).
  - `ensure_onnxruntime(root)` — download + extract the shared library into
    `<root>/<version>/lib/` (atomic temp-dir swap), reusing
    `model_manager::download_file`. `.tgz` via `flate2`+`tar` (unix), `.zip` via
    `zip` (windows). Override the source with `SEMANTEX_ONNXRUNTIME_BASE_URL`
    (internal mirror / airgap).
- **`crates/semantex-core/Cargo.toml`** — `ort` now declared
  `default-features = false, features = ["ndarray", "std", "load-dynamic"]`
  (honest about the actual mode; the old `download-binaries` was a no-op). Added
  `flate2`+`tar` (non-windows) and `zip` (windows) for extraction.
- **`crates/semantex-cli/src/main.rs`** — startup `resolve_ort_dylib()`:
  for ONNX-loading commands, prefer the managed runtime (cache → provision),
  fall back to a system lib only when offline; sets `ORT_DYLIB_PATH` (absolute)
  before threads spawn. Respects a user-set `ORT_DYLIB_PATH`. Added
  `command_wants_onnxruntime()` arg-peek gate so lightweight subcommands
  (`status`/`stop`/`validate`/…) never trigger a download.
- **`crates/semantex-cli/src/commands/download.rs`** — `semantex download-models`
  now also provisions the ONNX Runtime, making it a complete offline-prep step.

Tests: 4 unit tests in `runtime_manager` (asset map, dylib predicate, cache
find, extraction filter) + 2 arg-peek tests. `cargo clippy` clean; default build
remains genai-free.

### Verified on macOS (arm64)

```
$ SEMANTEX_HOME=/tmp/fresh semantex download-models
Provisioned ONNX Runtime 1.23.2 at /tmp/fresh/runtime/1.23.2/lib/libonnxruntime.1.23.2.dylib
$ SEMANTEX_HOME=/tmp/fresh semantex index <repo>     # ORT_DYLIB_PATH unset in env
Index complete!  Chunks created: 3
```
The embedding stage `dlopen`'d the provisioned 1.23.2 library and produced
chunks — the exact step that panicked on Linux — with no system lib involved.

## Confirm on Linux

```bash
# fresh box, no onnxruntime installed, ORT_DYLIB_PATH unset
unset ORT_DYLIB_PATH
SEMANTEX_HOME=/tmp/sx semantex download-models          # provisions onnxruntime-linux-x64-1.23.2
SEMANTEX_HOME=/tmp/sx semantex index ~/swe_repos/django__django-10973
# expect: non-zero "Chunks created", no OrtGetApiBase panic
```
Then run Phase A with `--conditions c1_baseline,c2_semantex_no_llm,c3_semantex_with_llm`
and `scripts.compare_conditions` becomes meaningful.

## Immediate manual unblock (if needed before redeploying the binary)

The shipped fix auto-provisions, so a rebuilt binary needs no manual step. If you
must unblock an *existing* binary on the VM right now:
```bash
curl -L -o ort.tgz https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-linux-x64-1.23.2.tgz
tar xf ort.tgz -C /opt
export ORT_DYLIB_PATH=/opt/onnxruntime-linux-x64-1.23.2/lib/libonnxruntime.so   # absolute
semantex index ~/swe_repos/django__django-10973
```

## Maintenance note

When bumping the `ort` dependency, check the new `ort_sys::ORT_API_VERSION` and
update `ONNXRUNTIME_VERSION` in `runtime_manager.rs` to a matching ONNX Runtime
release (same minor or newer). A mismatch surfaces at runtime as
`ort … expected version 1.<N>.x or newer`.
