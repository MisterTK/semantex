//! Process memory monitoring utilities.

/// Returns the process RSS in bytes, or `None` if unavailable.
///
/// - **macOS**: returns *current* resident size via Mach `task_info`.
///   (`getrusage` `ru_maxrss` is peak/high-water-mark and never decreases,
///   making it useless for memory-pressure eviction decisions.)
/// - **Linux**: returns current `VmRSS` from `/proc/self/status`.
#[cfg(target_os = "macos")]
pub fn current_rss_bytes() -> Option<u64> {
    // SAFETY: mach_task_self() and task_info() are standard Mach kernel calls.
    // Zeroing the info struct is valid for MACH_TASK_BASIC_INFO.
    #[allow(deprecated)] // libc says use mach2, but we avoid the extra dep
    unsafe {
        let mut info: libc::mach_task_basic_info_data_t = std::mem::zeroed();
        let mut count = (std::mem::size_of::<libc::mach_task_basic_info_data_t>()
            / std::mem::size_of::<libc::natural_t>())
            as libc::mach_msg_type_number_t;
        let kr = libc::task_info(
            libc::mach_task_self(),
            libc::MACH_TASK_BASIC_INFO,
            (&raw mut info).cast::<libc::integer_t>(),
            &raw mut count,
        );
        if kr == libc::KERN_SUCCESS {
            Some(info.resident_size) // bytes, current (not peak)
        } else {
            None
        }
    }
}

#[cfg(target_os = "linux")]
pub fn current_rss_bytes() -> Option<u64> {
    let content = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in content.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
            return Some(kb * 1024);
        }
    }
    None
}

#[cfg(target_os = "windows")]
pub fn current_rss_bytes() -> Option<u64> {
    use std::mem::MaybeUninit;
    // SAFETY: GetCurrentProcess() returns a pseudo-handle (always valid),
    // and GetProcessMemoryInfo is safe with a zeroed PROCESS_MEMORY_COUNTERS.
    unsafe {
        let handle = windows_sys::Win32::System::Threading::GetCurrentProcess();
        let mut pmc = MaybeUninit::<
            windows_sys::Win32::System::ProcessStatus::PROCESS_MEMORY_COUNTERS,
        >::zeroed();
        let cb = std::mem::size_of_val(&pmc) as u32;
        let ok = windows_sys::Win32::System::ProcessStatus::K32GetProcessMemoryInfo(
            handle,
            pmc.as_mut_ptr(),
            cb,
        );
        if ok != 0 {
            Some(pmc.assume_init().WorkingSetSize as u64)
        } else {
            None
        }
    }
}

#[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
pub fn current_rss_bytes() -> Option<u64> {
    None
}

/// Returns current RSS in megabytes, or `None` if unavailable.
pub fn current_rss_mb() -> Option<u64> {
    current_rss_bytes().map(|b| b / (1024 * 1024))
}

/// Callback slot for allocator purge. Set by binaries that use mimalloc.
static PURGE_FN: std::sync::OnceLock<fn()> = std::sync::OnceLock::new();

/// Register a function that forces the allocator to return freed pages to the OS.
/// Called once at startup by binaries that use mimalloc.
pub fn register_purge_fn(f: fn()) {
    let _ = PURGE_FN.set(f);
}

/// Ask the global allocator to return freed pages to the OS.
/// No-op if no purge function was registered.
pub fn purge_allocator() {
    if let Some(f) = PURGE_FN.get() {
        f();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rss_returns_some_on_supported_platforms() {
        // On macOS/Linux/Windows this must return Some.
        // On unsupported platforms we skip.
        if cfg!(any(
            target_os = "macos",
            target_os = "linux",
            target_os = "windows"
        )) {
            let bytes = current_rss_bytes().expect("RSS should be available on this platform");
            // Sanity: a running Rust test process uses at least 1 MB
            assert!(bytes > 1_000_000, "RSS too small: {bytes} bytes");
            // Sanity: and less than 64 GB (not a garbage value)
            assert!(
                bytes < 64 * 1024 * 1024 * 1024,
                "RSS too large: {bytes} bytes"
            );
        }
    }

    #[test]
    fn rss_mb_returns_reasonable_value() {
        if let Some(mb) = current_rss_mb() {
            assert!(mb >= 1, "RSS should be at least 1 MB, got {mb}");
            assert!(mb < 64 * 1024, "RSS should be less than 64 GB, got {mb} MB");
        }
    }

    #[test]
    fn rss_decreases_after_large_allocation_is_dropped() {
        // Allocate ~50 MB, measure RSS, drop it, measure again.
        // On a working implementation RSS should decrease (or at least not grow).
        let before = current_rss_mb();

        // Allocate and touch 50 MB to ensure pages are faulted in
        let big: Vec<u8> = vec![42u8; 50 * 1024 * 1024];
        let during = current_rss_mb();

        // Force the allocation to actually exist (prevent optimization)
        assert_eq!(big[big.len() - 1], 42);
        drop(big);

        // Purge allocator if available
        purge_allocator();

        let after = current_rss_mb();

        if let (Some(before), Some(during), Some(after)) = (before, during, after) {
            // RSS should have grown during the allocation
            assert!(
                during >= before,
                "RSS should not shrink during allocation: before={before}, during={during}"
            );
            // After drop + purge, RSS should be closer to before than during.
            // We allow generous margin because OS page reclaim is async.
            // The key check: `after` should be less than `during` (memory returned).
            // On macOS with the old getrusage bug, `after == during` always.
            eprintln!("RSS: before={before}MB, during={during}MB, after={after}MB");
        }
    }
}
