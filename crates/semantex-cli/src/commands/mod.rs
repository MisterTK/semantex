pub mod connect;
pub mod disconnect;
pub mod download;
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

/// Spawn `semantex index <dir>` in a background process (fire-and-forget).
/// Shared by hooks and search commands.
pub(crate) fn spawn_background_index(project_path: &std::path::Path) {
    if let Err(e) = std::process::Command::new("semantex")
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
