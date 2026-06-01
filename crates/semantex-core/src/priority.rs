//! Background scheduling priority for index builds.
//!
//! Indexing is background work a developer shouldn't feel. The portable first
//! line of defense is using only a fraction of the cores (see
//! [`SingleVectorEmbedder::for_indexing`](crate::embedding::single_vector::SingleVectorEmbedder::for_indexing)).
//! On top of that, a *pure* indexing process (the `semantex index` / `watch`
//! CLI, or a future subprocess-isolated daemon build) can lower its whole-process
//! CPU priority so it yields to the developer's editor/compiler under load.
//!
//! This is intentionally process-scoped: it must NOT be called from a process
//! that also serves search queries (the MCP/serve daemon), because it would
//! slow query responses under contention. It is also best-effort — failures are
//! ignored; the worst case is "indexing runs at normal priority."
//!
//! Note on niceness semantics: a lowered priority only matters when something
//! else wants the CPU. On an otherwise-idle machine the build still runs at full
//! speed, so this costs nothing in the common single-developer case.

/// Lower the **current process's** CPU scheduling priority to a background
/// level. Best-effort; never panics or returns an error.
///
/// * Unix: `nice(+10)` via `setpriority(PRIO_PROCESS, self, 10)`.
/// * Windows: `BELOW_NORMAL_PRIORITY_CLASS`.
///
/// Threads spawned *after* this call (e.g. the ORT intra-op pool and the rayon
/// pool created during a build) inherit the lowered priority on Linux, so the
/// whole build runs in the background.
pub fn lower_process_to_background() {
    #[cfg(unix)]
    {
        // SAFETY: setpriority is a standard POSIX call; who=0 targets the
        // current process. The return value is ignored intentionally.
        unsafe {
            let _ = libc::setpriority(libc::PRIO_PROCESS, 0, 10);
        }
    }
    #[cfg(windows)]
    {
        use windows_sys::Win32::System::Threading::{
            BELOW_NORMAL_PRIORITY_CLASS, GetCurrentProcess, SetPriorityClass,
        };
        // SAFETY: GetCurrentProcess returns a pseudo-handle that needs no close;
        // SetPriorityClass on it is documented and safe. Result ignored.
        unsafe {
            let _ = SetPriorityClass(GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS);
        }
    }
}
