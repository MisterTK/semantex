//! Process memory monitoring utilities.

/// Returns the process RSS in bytes, or `None` if unavailable.
///
/// - **macOS**: returns peak RSS via `getrusage` (monotonically increasing).
/// - **Linux**: returns current `VmRSS` from `/proc/self/status`.
#[cfg(target_os = "macos")]
pub fn current_rss_bytes() -> Option<u64> {
    // SAFETY: getrusage is a standard POSIX call; zeroing the struct is valid.
    unsafe {
        let mut usage: libc::rusage = std::mem::zeroed();
        if libc::getrusage(libc::RUSAGE_SELF, &raw mut usage) == 0 {
            Some(usage.ru_maxrss as u64) // bytes on macOS
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

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
pub fn current_rss_bytes() -> Option<u64> {
    None
}

/// Returns current RSS in megabytes, or `None` if unavailable.
pub fn current_rss_mb() -> Option<u64> {
    current_rss_bytes().map(|b| b / (1024 * 1024))
}
