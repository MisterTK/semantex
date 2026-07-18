pub mod connect;
pub mod disconnect;
pub mod distill_centroids;
pub mod distill_corpus;
pub mod distill_static_table;
pub mod download;
pub mod federated;
pub mod history;
pub mod hooks;
pub mod index;
pub mod install;
#[cfg(feature = "llm")]
pub mod llm_status;
pub mod mcp;
pub mod search;
pub mod serve;
pub mod skills_generate;
pub mod status;
pub mod stop;
pub mod validate;
pub mod watch;

/// Return the path of the currently-running executable as an `OsString`.
///
/// On success, this is the actual binary that was invoked (e.g.
/// `target/release/semantex`), so subprocesses we spawn will use the same
/// build rather than whatever `semantex` resolves to on `$PATH`.  On failure
/// (exotic platforms, deleted-exe races, etc.) we fall back to the bare name
/// `"semantex"` so behaviour is never worse than before.
pub(crate) fn self_exe() -> std::ffi::OsString {
    std::env::current_exe().map_or_else(
        |_| std::ffi::OsString::from("semantex"),
        std::path::PathBuf::into_os_string,
    )
}

/// Spawn `semantex index <dir>` in a background process (fire-and-forget).
/// Shared by hooks and search commands.
///
/// Every caller of this function is an *unattended* auto-index trigger (a
/// session-start hook, or `semantex search` auto-building on first use) —
/// never the explicit `semantex index <path>` command a human or a test
/// harness runs on purpose. So this refuses anything under a system temp
/// root at any depth: nobody asked for a throwaway scratch directory a
/// session happened to open in to become a permanently tracked project. An
/// explicit `semantex index /tmp/some-fixture` remains unaffected — that
/// goes through [`crate::commands::index::run`] directly, not this function.
pub(crate) fn spawn_background_index(project_path: &std::path::Path) {
    if semantex_core::index::registry::is_under_system_temp_root(project_path) {
        return;
    }
    if let Err(e) = std::process::Command::new(self_exe())
        .arg("index")
        .arg(project_path)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
    {
        eprintln!("semantex: failed to spawn background indexer: {e}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `self_exe()` must resolve to the currently-running test binary, not the
    /// bare name "semantex".  The test binary is always a real path on disk, so
    /// it must be absolute and non-empty, and it must match `current_exe()`.
    #[test]
    fn self_exe_matches_current_exe() {
        let result = self_exe();
        assert!(!result.is_empty(), "self_exe() returned an empty OsString");

        // The test binary always has a real path on disk, so current_exe()
        // succeeds here — assert directly. (The OsString::from("semantex")
        // fallback only fires on exotic platforms we don't run tests on.)
        let expected =
            std::env::current_exe().expect("current_exe() must succeed in the test binary");
        assert_eq!(
            result,
            expected.as_os_str(),
            "self_exe() should equal current_exe() when it succeeds"
        );
        // The path must be absolute so spawned subprocesses are unambiguous.
        assert!(
            expected.is_absolute(),
            "current_exe() returned a relative path: {}",
            expected.display()
        );
    }
}
