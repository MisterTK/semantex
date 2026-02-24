pub mod connect;
pub mod disconnect;
pub mod download;
pub mod hooks;
pub mod index;
pub mod install;
pub mod mcp;
pub mod search;
pub mod serve;
pub mod status;
pub mod stop;
pub mod watch;

/// Spawn `sage index <dir>` in a background process (fire-and-forget).
/// Shared by hooks and search commands.
pub(crate) fn spawn_background_index(project_path: &std::path::Path) {
    match std::env::current_exe() {
        Ok(exe) => {
            if let Err(e) = std::process::Command::new(&exe)
                .arg("index")
                .arg(project_path)
                .stdin(std::process::Stdio::null())
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .spawn()
            {
                eprintln!("sage: failed to spawn background indexer: {e}");
            }
        }
        Err(e) => eprintln!("sage: cannot determine executable path: {e}"),
    }
}
