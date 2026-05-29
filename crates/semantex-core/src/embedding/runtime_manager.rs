//! ONNX Runtime shared-library provisioning for the `load-dynamic` linking mode.
//!
//! semantex's dependency graph forces `ort` into pure dynamic-loading mode:
//! `next-plaid-onnx` depends on `ort` with the `load-dynamic` feature, which
//! enables `ort-sys/disable-linking`. `ort-sys`'s build script early-returns on
//! `disable-linking`, so the binary never embeds ONNX Runtime — on **every**
//! platform. At runtime `ort` must `dlopen` a `libonnxruntime` shared library
//! resolved through `ORT_DYLIB_PATH` (or the OS loader search path).
//!
//! To make this work uniformly across operating systems without requiring a
//! system package (apt/brew/etc.), semantex downloads the official Microsoft
//! ONNX Runtime release and caches it under `~/.semantex/runtime/<version>/lib`,
//! mirroring how the ColBERT model is cached under `~/.semantex/models`.
//!
//! ## Version pinning
//!
//! `ort` records the ONNX Runtime C API version it was built against in
//! `ort_sys::ORT_API_VERSION` and **rejects** any runtime whose minor version is
//! lower (see `ort`'s `load_dylib_from_path`, which errors with "expected
//! version 1.N.x or newer"). `ort 2.0.0-rc.11` binds `ORT_API_VERSION = 23`, so
//! the runtime must be ONNX Runtime 1.23 or newer. [`ONNXRUNTIME_VERSION`] must
//! therefore be bumped in lockstep with the `ort` dependency in `Cargo.toml`.

use anyhow::{Context, Result, bail};
use std::fs;
use std::path::{Path, PathBuf};

/// ONNX Runtime version to provision. Must satisfy `ort`'s `ORT_API_VERSION`
/// minor-version floor (>= 1.23 for `ort 2.0.0-rc.11`). Bump alongside `ort`.
pub const ONNXRUNTIME_VERSION: &str = "1.23.2";

/// Base URL for official Microsoft ONNX Runtime release assets. Overridable via
/// `SEMANTEX_ONNXRUNTIME_BASE_URL` for airgapped/internal mirrors. The full URL
/// is `<base>/v<version>/<asset>`.
const ONNXRUNTIME_RELEASE_BASE: &str = "https://github.com/microsoft/onnxruntime/releases/download";

/// Environment override for the release base URL (internal mirror / airgap).
const BASE_URL_ENV: &str = "SEMANTEX_ONNXRUNTIME_BASE_URL";

/// Return the Microsoft release asset filename for the host platform, e.g.
/// `onnxruntime-linux-x64-1.23.2.tgz`. Errors for targets Microsoft does not
/// publish a prebuilt shared library for.
pub fn asset_name() -> Result<String> {
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;
    let stem = match (os, arch) {
        ("linux", "x86_64") => "onnxruntime-linux-x64",
        ("linux", "aarch64") => "onnxruntime-linux-aarch64",
        ("macos", "aarch64") => "onnxruntime-osx-arm64",
        ("macos", "x86_64") => "onnxruntime-osx-x86_64",
        ("windows", "x86_64") => "onnxruntime-win-x64",
        ("windows", "aarch64") => "onnxruntime-win-arm64",
        _ => bail!(
            "no prebuilt ONNX Runtime is published for {os}/{arch}; \
             install ONNX Runtime >= {ONNXRUNTIME_VERSION} and set ORT_DYLIB_PATH to its libonnxruntime"
        ),
    };
    let ext = if os == "windows" { "zip" } else { "tgz" };
    Ok(format!("{stem}-{ONNXRUNTIME_VERSION}.{ext}"))
}

/// Directory holding the extracted shared libraries for the pinned version:
/// `<runtime_root>/<version>/lib`.
fn lib_dir(runtime_root: &Path) -> PathBuf {
    runtime_root.join(ONNXRUNTIME_VERSION).join("lib")
}

/// True if `name` is the loadable ONNX Runtime shared library for this OS.
/// Microsoft ships a versioned real file plus an unversioned symlink; tar
/// extraction keeps only the regular file (e.g. `libonnxruntime.so.1.23.2`),
/// which `dlopen` loads fine by absolute path.
#[cfg(all(unix, not(target_os = "macos")))]
fn is_onnxruntime_dylib_name(name: &str) -> bool {
    name.starts_with("libonnxruntime.so")
}
#[cfg(target_os = "macos")]
fn is_onnxruntime_dylib_name(name: &str) -> bool {
    name.starts_with("libonnxruntime")
        && std::path::Path::new(name)
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("dylib"))
}
#[cfg(target_os = "windows")]
fn is_onnxruntime_dylib_name(name: &str) -> bool {
    name.eq_ignore_ascii_case("onnxruntime.dll")
}

/// Locate an already-provisioned ONNX Runtime shared library under
/// `<runtime_root>/<version>/lib`, without downloading. Returns the absolute
/// path to the loadable library, or `None` if not present.
pub fn find_onnxruntime(runtime_root: &Path) -> Option<PathBuf> {
    let dir = lib_dir(runtime_root);
    fs::read_dir(&dir).ok()?.flatten().map(|e| e.path()).find(|p| {
        p.is_file()
            && p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(is_onnxruntime_dylib_name)
    })
}

/// Ensure the pinned ONNX Runtime shared library is available under
/// `<runtime_root>/<version>/lib`, downloading and extracting the official
/// Microsoft release if needed. Returns the absolute path to the library
/// (suitable for `ORT_DYLIB_PATH`). Idempotent: a no-op when already present.
pub fn ensure_onnxruntime(runtime_root: &Path) -> Result<PathBuf> {
    if let Some(path) = find_onnxruntime(runtime_root) {
        return Ok(path);
    }

    let asset = asset_name()?;
    let base = std::env::var(BASE_URL_ENV).unwrap_or_else(|_| ONNXRUNTIME_RELEASE_BASE.to_string());
    let url = format!("{}/v{ONNXRUNTIME_VERSION}/{asset}", base.trim_end_matches('/'));

    let version_dir = runtime_root.join(ONNXRUNTIME_VERSION);
    fs::create_dir_all(&version_dir)
        .with_context(|| format!("Failed to create {}", version_dir.display()))?;

    let archive_path = version_dir.join(&asset);
    tracing::info!("Downloading ONNX Runtime {ONNXRUNTIME_VERSION} ({asset})...");
    crate::embedding::model_manager::download_file(&url, &archive_path)
        .with_context(|| format!("Failed to download ONNX Runtime from {url}"))?;

    // Extract into a partial dir, then atomically swap into place so a
    // half-extracted lib/ is never observed as complete.
    let partial = version_dir.join(".lib.partial");
    let _ = fs::remove_dir_all(&partial);
    extract_libs(&archive_path, &partial).context("Failed to extract ONNX Runtime archive")?;
    let final_dir = lib_dir(runtime_root);
    let _ = fs::remove_dir_all(&final_dir);
    fs::rename(&partial, &final_dir).with_context(|| {
        format!("Failed to move {} -> {}", partial.display(), final_dir.display())
    })?;
    let _ = fs::remove_file(&archive_path);

    find_onnxruntime(runtime_root)
        .context("ONNX Runtime shared library not found in archive after extraction")
}

/// Extract the ONNX Runtime shared libraries from the `lib/` directory of a
/// `.tgz` (Linux/macOS) archive into `dest`, skipping headers and symlinks.
#[cfg(not(target_os = "windows"))]
fn extract_libs(archive: &Path, dest: &Path) -> Result<()> {
    use flate2::read::GzDecoder;
    use tar::Archive;

    fs::create_dir_all(dest)?;
    let file = fs::File::open(archive)
        .with_context(|| format!("Failed to open {}", archive.display()))?;
    let mut tar = Archive::new(GzDecoder::new(file));
    let mut extracted = 0usize;
    for entry in tar.entries()? {
        let mut entry = entry?;
        if !entry.header().entry_type().is_file() {
            continue; // skip dirs and the unversioned symlinks
        }
        let path = entry.path()?.into_owned();
        let in_lib = path.components().any(|c| c.as_os_str() == "lib");
        let file_name = path.file_name().and_then(|n| n.to_str()).map(str::to_owned);
        // Keep exactly what `find_onnxruntime` looks for — the shared object,
        // not headers or the `.pc`/symlink siblings.
        if let Some(name) = file_name
            && in_lib
            && is_onnxruntime_dylib_name(&name)
        {
            entry
                .unpack(dest.join(&name))
                .with_context(|| format!("Failed to unpack {name}"))?;
            extracted += 1;
        }
    }
    if extracted == 0 {
        bail!("no libonnxruntime entry found under lib/ in {}", archive.display());
    }
    Ok(())
}

/// Extract the ONNX Runtime DLLs from the `lib/` directory of a `.zip`
/// (Windows) archive into `dest`.
#[cfg(target_os = "windows")]
fn extract_libs(archive: &Path, dest: &Path) -> Result<()> {
    use std::io;

    fs::create_dir_all(dest)?;
    let file = fs::File::open(archive)
        .with_context(|| format!("Failed to open {}", archive.display()))?;
    let mut zip = zip::ZipArchive::new(file)?;
    let mut extracted = 0usize;
    for i in 0..zip.len() {
        let mut entry = zip.by_index(i)?;
        if !entry.is_file() {
            continue;
        }
        let name = entry.name().replace('\\', "/");
        let in_lib = name.split('/').any(|c| c == "lib");
        let file_name = name.rsplit('/').next().unwrap_or("").to_owned();
        if in_lib && file_name.to_ascii_lowercase().starts_with("onnxruntime") && file_name.to_ascii_lowercase().ends_with(".dll") {
            let mut out = fs::File::create(dest.join(&file_name))
                .with_context(|| format!("Failed to create {file_name}"))?;
            io::copy(&mut entry, &mut out)?;
            extracted += 1;
        }
    }
    if extracted == 0 {
        bail!("no onnxruntime DLL found under lib/ in {}", archive.display());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn asset_name_matches_host_and_pinned_version() {
        // On any supported host this resolves; assert the shape rather than a
        // fixed string so the test holds across the CI platform matrix.
        let asset = asset_name().expect("supported host platform");
        assert!(asset.starts_with("onnxruntime-"), "unexpected asset: {asset}");
        assert!(asset.contains(ONNXRUNTIME_VERSION), "missing version: {asset}");
        let ext = if std::env::consts::OS == "windows" { ".zip" } else { ".tgz" };
        assert!(asset.ends_with(ext), "unexpected extension: {asset}");
    }

    #[test]
    fn dylib_name_predicate_is_os_specific() {
        #[cfg(all(unix, not(target_os = "macos")))]
        {
            assert!(is_onnxruntime_dylib_name("libonnxruntime.so"));
            assert!(is_onnxruntime_dylib_name("libonnxruntime.so.1.23.2"));
            assert!(!is_onnxruntime_dylib_name("libonnxruntime.dylib"));
            assert!(!is_onnxruntime_dylib_name("onnxruntime_test.h"));
        }
        #[cfg(target_os = "macos")]
        {
            assert!(is_onnxruntime_dylib_name("libonnxruntime.dylib"));
            assert!(is_onnxruntime_dylib_name("libonnxruntime.1.23.2.dylib"));
            assert!(!is_onnxruntime_dylib_name("libonnxruntime.so.1.23.2"));
        }
        #[cfg(target_os = "windows")]
        {
            assert!(is_onnxruntime_dylib_name("onnxruntime.dll"));
            assert!(is_onnxruntime_dylib_name("ONNXRUNTIME.DLL"));
            assert!(!is_onnxruntime_dylib_name("onnxruntime_providers_shared.dll"));
        }
    }

    #[test]
    fn find_returns_none_when_absent_and_path_when_present() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = tmp.path();
        assert!(find_onnxruntime(root).is_none());

        let dir = lib_dir(root);
        fs::create_dir_all(&dir).unwrap();
        let name = if cfg!(target_os = "macos") {
            "libonnxruntime.1.23.2.dylib"
        } else if cfg!(target_os = "windows") {
            "onnxruntime.dll"
        } else {
            "libonnxruntime.so.1.23.2"
        };
        let lib = dir.join(name);
        fs::write(&lib, b"stub").unwrap();
        assert_eq!(find_onnxruntime(root).as_deref(), Some(lib.as_path()));
    }

    // Validates the extraction filter: only the real shared library under
    // `lib/` is kept; headers and the unversioned symlink are dropped.
    #[cfg(not(target_os = "windows"))]
    #[test]
    fn extract_keeps_only_lib_shared_objects() {
        use flate2::write::GzEncoder;

        let tmp = tempfile::tempdir().expect("tempdir");
        let archive = tmp.path().join("ort.tgz");

        // Build a miniature MS-layout archive.
        {
            let tgz = fs::File::create(&archive).unwrap();
            let enc = GzEncoder::new(tgz, flate2::Compression::fast());
            let mut builder = tar::Builder::new(enc);

            let real_lib = if cfg!(target_os = "macos") {
                "onnxruntime-x/lib/libonnxruntime.1.23.2.dylib"
            } else {
                "onnxruntime-x/lib/libonnxruntime.so.1.23.2"
            };
            let mut add_file = |path: &str, bytes: &[u8]| {
                let mut header = tar::Header::new_gnu();
                header.set_size(bytes.len() as u64);
                header.set_entry_type(tar::EntryType::Regular);
                header.set_mode(0o644);
                header.set_cksum();
                builder.append_data(&mut header, path, bytes).unwrap();
            };
            add_file(real_lib, b"\x7fELF-stub");
            add_file("onnxruntime-x/include/onnxruntime_c_api.h", b"// header");
            add_file("onnxruntime-x/lib/libonnxruntime.pc", b"# pkg-config");
            builder.into_inner().unwrap().finish().unwrap();
        }

        let dest = tmp.path().join("out-lib");
        extract_libs(&archive, &dest).expect("extraction");

        let entries: Vec<String> = fs::read_dir(&dest)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(entries.len(), 1, "only the shared lib should be kept: {entries:?}");
        assert!(entries[0].starts_with("libonnxruntime"));
        // And the freshly-extracted lib is discoverable.
        let root = tmp.path().join("root");
        let realdir = lib_dir(&root);
        fs::create_dir_all(realdir.parent().unwrap()).unwrap();
        fs::rename(&dest, &realdir).unwrap();
        assert!(find_onnxruntime(&root).is_some());
    }
}
