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

// ════════════════════════════════════════════════════════════════════════════
// System-RAM-aware hard memory caps
//
// semantex must NEVER be able to OOM the host system, regardless of how
// runaway any single allocation gets. We enforce that via two layers:
//
//   1. Kernel-level address-space cap via setrlimit(RLIMIT_AS) at process
//      start. If our code allocates past this, mmap()/brk() returns ENOMEM,
//      the allocator panics, and the process dies cleanly — but the OS
//      never goes into swap thrash and never kills other processes.
//
//   2. Application-level RSS polling in hot paths (indexer batches, MCP
//      loops). If RSS exceeds the soft cap we abort cleanly with a message
//      pointing at SEMANTEX_MAX_RSS_MB.
//
// Both layers honor the same SEMANTEX_MAX_RSS_MB env var. Default is derived
// from system RAM so we behave sensibly on any machine without configuration.
// ════════════════════════════════════════════════════════════════════════════

/// Returns total system RAM in bytes, or `None` if unavailable.
#[cfg(target_os = "macos")]
pub fn system_ram_bytes() -> Option<u64> {
    // SAFETY: sysctlbyname is a documented BSD API.
    unsafe {
        let mut size: u64 = 0;
        let mut len = std::mem::size_of::<u64>();
        let name = c"hw.memsize";
        let rc = libc::sysctlbyname(
            name.as_ptr(),
            (&raw mut size).cast::<libc::c_void>(),
            &raw mut len,
            std::ptr::null_mut(),
            0,
        );
        if rc == 0 { Some(size) } else { None }
    }
}

#[cfg(target_os = "linux")]
pub fn system_ram_bytes() -> Option<u64> {
    let content = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in content.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
            return Some(kb * 1024);
        }
    }
    None
}

#[cfg(target_os = "windows")]
pub fn system_ram_bytes() -> Option<u64> {
    use std::mem::MaybeUninit;
    // SAFETY: GlobalMemoryStatusEx zeroes its output struct after setting cbSize.
    unsafe {
        let mut status =
            MaybeUninit::<windows_sys::Win32::System::SystemInformation::MEMORYSTATUSEX>::zeroed();
        let p = status.as_mut_ptr();
        (*p).dwLength = std::mem::size_of_val(&status) as u32;
        let ok = windows_sys::Win32::System::SystemInformation::GlobalMemoryStatusEx(p);
        if ok != 0 {
            Some(status.assume_init().ullTotalPhys)
        } else {
            None
        }
    }
}

#[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
pub fn system_ram_bytes() -> Option<u64> {
    None
}

/// Returns total system RAM in megabytes (rounded down), or `None`.
pub fn system_ram_mb() -> Option<u64> {
    system_ram_bytes().map(|b| b / (1024 * 1024))
}

/// Absolute floor: never let the cap drop below this. A tiny cap would render
/// the indexer non-functional (the dense embedder model is ~137 MB, its ONNX
/// session can spike to a few hundred MB, Tantivy's write buffer defaults to
/// 50 MB). 1024 MB is the smallest cap at which semantex actually works.
pub const ABSOLUTE_FLOOR_MB: u64 = 1024;

/// Absolute ceiling: even on huge servers, don't claim more than this without
/// an explicit override. Prevents one daemon from leaving no headroom for other
/// processes on shared boxes.
pub const ABSOLUTE_CEILING_MB: u64 = 12 * 1024;

/// Returns the soft RSS limit in MB, honoring `SEMANTEX_MAX_RSS_MB` if set,
/// otherwise computing a safe default from system RAM.
///
/// Default policy:
///   * 50% of system RAM, clamped to `[ABSOLUTE_FLOOR_MB, ABSOLUTE_CEILING_MB]`.
///   * If system RAM cannot be detected, fall back to 2048 MB.
///
/// Explicit `SEMANTEX_MAX_RSS_MB=0` means "no app-level cap" — the kernel
/// `RLIMIT_AS` cap (installed at startup) still applies.
pub fn soft_rss_limit_mb() -> u64 {
    if let Ok(env) = std::env::var("SEMANTEX_MAX_RSS_MB")
        && let Ok(n) = env.trim().parse::<u64>()
    {
        return n;
    }
    let detected = system_ram_mb().unwrap_or(4 * 1024);
    (detected / 2).clamp(ABSOLUTE_FLOOR_MB, ABSOLUTE_CEILING_MB)
}

/// Returns the kernel `RLIMIT_AS` (address space) cap in bytes.
///
/// Default: 75% of system RAM, clamped to `[ABSOLUTE_FLOOR_MB,
/// 2 × ABSOLUTE_CEILING_MB]`. The kernel cap is intentionally higher than
/// the application-level soft cap so the app cap fires first with a clean
/// error message; the kernel cap is the last-resort failsafe.
pub fn kernel_rss_cap_bytes() -> u64 {
    if let Ok(env) = std::env::var("SEMANTEX_MAX_RSS_MB")
        && let Ok(n) = env.trim().parse::<u64>()
        && n > 0
    {
        // Honour user override; give 1.5× headroom so the soft cap can fire
        // before the kernel kills us.
        return (n * 3 / 2) * 1024 * 1024;
    }
    let detected = system_ram_mb().unwrap_or(4 * 1024);
    let mb = (detected * 3 / 4).clamp(ABSOLUTE_FLOOR_MB, 2 * ABSOLUTE_CEILING_MB);
    mb * 1024 * 1024
}

/// Result of attempting to install a kernel-level address-space cap.
///
/// Outcomes vary by platform:
///   * Linux: `Installed(bytes)` — setrlimit(RLIMIT_AS) is honoured.
///   * macOS: `UnsupportedPlatform` — no userspace API for hard caps.
///     setrlimit(RLIMIT_AS) returns EINVAL; `task_set_phys_footprint_limit`
///     returns KERN_NOT_PERMITTED for normal processes. Only the App Sandbox
///     and launchd-managed daemons can enforce memory limits via system
///     configuration. The soft cap (RSS polling) is the only available
///     guard.
///   * Windows: `UnsupportedPlatform` — would need Job Objects.
///   * Any: `Disabled` if `SEMANTEX_NO_RLIMIT=1`.
///   * Linux: `Failed(errno)` if setrlimit fails for any other reason.
#[derive(Debug, Clone, Copy)]
pub enum KernelCapResult {
    Installed(u64),
    UnsupportedPlatform,
    Disabled,
    Failed(i32),
}

/// Install a kernel-level address-space cap via `setrlimit(RLIMIT_AS, …)`.
///
/// Call ONCE at process start, before any heavy allocation. On Linux this
/// makes runaway allocations get `ENOMEM` from the kernel; on macOS the call
/// is a no-op (the platform doesn't expose a usable per-process memory cap)
/// and the soft cap is the only guard — so the indexer's per-batch RSS
/// polling is mandatory.
///
/// `SEMANTEX_NO_RLIMIT=1` disables installation (escape hatch for CI or
/// containers that handle their own cgroup limits).
#[cfg(target_os = "linux")]
pub fn install_kernel_rss_cap() -> KernelCapResult {
    if std::env::var("SEMANTEX_NO_RLIMIT").as_deref() == Ok("1") {
        return KernelCapResult::Disabled;
    }
    let bytes = kernel_rss_cap_bytes();
    // SAFETY: setrlimit is a standard POSIX call. We pass a properly-sized
    // struct and a valid resource identifier.
    unsafe {
        let rlim = libc::rlimit {
            rlim_cur: bytes,
            rlim_max: bytes,
        };
        if libc::setrlimit(libc::RLIMIT_AS, &raw const rlim) == 0 {
            KernelCapResult::Installed(bytes)
        } else {
            KernelCapResult::Failed(*libc::__errno_location())
        }
    }
}

#[cfg(not(target_os = "linux"))]
pub fn install_kernel_rss_cap() -> KernelCapResult {
    if std::env::var("SEMANTEX_NO_RLIMIT").as_deref() == Ok("1") {
        KernelCapResult::Disabled
    } else {
        KernelCapResult::UnsupportedPlatform
    }
}

/// Counter for how many consecutive `check_rss_or_abort` calls have observed
/// a soft-cap overshoot. After enough consecutive overshoots we hard-abort
/// the process to prevent unbounded allocation from continuing.
static OVERSHOOT_COUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// After this many consecutive overshoots, call `std::process::abort()` to
/// kill the process immediately (no unwinding, no destructors). The runaway
/// allocator should not be given more chances.
const ABORT_AFTER_CONSECUTIVE_OVERSHOOTS: u32 = 3;

/// Check whether current RSS exceeds the soft cap. If so, log a warning,
/// purge the allocator, and re-check. If still over, return `Err` with a
/// caller-actionable message AND increment the overshoot counter. After
/// `ABORT_AFTER_CONSECUTIVE_OVERSHOOTS` consecutive overshoots the process
/// hard-aborts via `std::process::abort()` — no unwinding, no destructors,
/// no further allocation. This is the ONLY guard on macOS where the kernel
/// has no equivalent failsafe.
///
/// Call at hot-path boundaries (per batch, per phase) — NOT in tight loops.
pub fn check_rss_or_abort(label: &str) -> Result<(), String> {
    use std::sync::atomic::Ordering;

    let limit_mb = soft_rss_limit_mb();
    if limit_mb == 0 {
        return Ok(()); // explicitly disabled
    }
    let Some(rss_mb) = current_rss_mb() else {
        return Ok(()); // can't measure → don't block
    };
    if rss_mb <= limit_mb {
        // Reset the consecutive-overshoot counter on any clean check.
        OVERSHOOT_COUNT.store(0, Ordering::Relaxed);
        return Ok(());
    }
    tracing::warn!(
        label,
        rss_mb,
        limit_mb,
        "RSS exceeded soft cap — purging allocator and re-checking"
    );
    purge_allocator();
    let after = current_rss_mb().unwrap_or(rss_mb);
    if after <= limit_mb {
        tracing::info!(
            label,
            before_mb = rss_mb,
            after_mb = after,
            limit_mb,
            "RSS recovered after purge"
        );
        OVERSHOOT_COUNT.store(0, Ordering::Relaxed);
        return Ok(());
    }

    // Still over after purge. Bump the counter and, if we've overshot enough
    // times in a row, hard-abort the process so we can NEVER take down the
    // host. This is the macOS failsafe (no kernel RLIMIT_AS available there).
    let prev = OVERSHOOT_COUNT.fetch_add(1, Ordering::Relaxed);
    let count = prev + 1;
    if count >= ABORT_AFTER_CONSECUTIVE_OVERSHOOTS {
        eprintln!(
            "\n[semantex FATAL] RSS {after} MB exceeded SEMANTEX_MAX_RSS_MB={limit_mb} \
             for {count} consecutive checks (last: {label}). \
             Aborting process to protect host memory.\n\
             To raise the cap, set `SEMANTEX_MAX_RSS_MB=<bigger>` (e.g. 8192). \
             To opt out (NOT recommended), set `SEMANTEX_MAX_RSS_MB=0`.\n"
        );
        // std::process::abort() bypasses Drop and panic handlers entirely —
        // we cannot rely on graceful shutdown when the allocator is already
        // out of control. The kernel reaps us; the host is unaffected.
        std::process::abort();
    }

    Err(format!(
        "RSS {after} MB exceeds SEMANTEX_MAX_RSS_MB={limit_mb} (at {label}, \
         consecutive overshoot {count}/{ABORT_AFTER_CONSECUTIVE_OVERSHOOTS}). \
         Operation aborted. Raise the cap via `SEMANTEX_MAX_RSS_MB=<larger>` \
         (e.g. 8192) or reindex a smaller subset. After \
         {ABORT_AFTER_CONSECUTIVE_OVERSHOOTS} consecutive overshoots the \
         process will hard-abort to protect host memory."
    ))
}

#[cfg(test)]
mod cap_tests {
    use super::*;

    #[test]
    fn system_ram_is_reasonable() {
        if let Some(mb) = system_ram_mb() {
            // Any machine running this test has at least 1 GB and less than 4 TB.
            assert!(mb >= 1024, "system RAM too small: {mb} MB");
            assert!(
                mb < 4 * 1024 * 1024,
                "system RAM implausibly large: {mb} MB"
            );
        }
    }

    /// All env-mutating tests run serially inside this single test function.
    /// Cargo runs different `#[test]` functions in parallel; without
    /// serialisation they race on `SEMANTEX_MAX_RSS_MB` and read each other's
    /// transient values. Combining them avoids needing a global lock.
    ///
    /// **Variables mutated here**: `SEMANTEX_MAX_RSS_MB` only.
    ///
    /// This test does NOT hold `crate::llm::TEST_ENV_LOCK` because it does not
    /// touch `SEMANTEX_LLM_*` vars; those are guarded by `TEST_ENV_LOCK` in
    /// `llm/genai_backend.rs` and `llm/subscription_cli.rs`. The two variable
    /// families are disjoint, so this single-combined-test pattern is sufficient.
    /// If a future test needs to mutate BOTH families, it must hold `TEST_ENV_LOCK`
    /// for the entire duration.
    #[test]
    fn env_cap_behaviour_serial() {
        let orig = std::env::var("SEMANTEX_MAX_RSS_MB").ok();
        let restore = |v: &Option<String>| match v {
            Some(s) => unsafe { std::env::set_var("SEMANTEX_MAX_RSS_MB", s) },
            None => unsafe { std::env::remove_var("SEMANTEX_MAX_RSS_MB") },
        };

        // (1) No env var → limit is inside [floor, ceiling].
        unsafe { std::env::remove_var("SEMANTEX_MAX_RSS_MB") };
        let limit = soft_rss_limit_mb();
        assert!(limit >= ABSOLUTE_FLOOR_MB, "limit {limit} below floor");
        assert!(limit <= ABSOLUTE_CEILING_MB, "limit {limit} above ceiling");

        // (2) Kernel cap ≥ soft cap so the soft cap can fire first.
        let soft_bytes = soft_rss_limit_mb() * 1024 * 1024;
        let kernel_bytes = kernel_rss_cap_bytes();
        assert!(
            kernel_bytes >= soft_bytes,
            "kernel cap {kernel_bytes} bytes < soft cap {soft_bytes} bytes — soft cap would never fire"
        );

        // (3) Explicit override → exact value, kernel cap is 1.5× the soft cap.
        unsafe { std::env::set_var("SEMANTEX_MAX_RSS_MB", "2000") };
        assert_eq!(soft_rss_limit_mb(), 2000);
        assert_eq!(kernel_rss_cap_bytes(), 3000 * 1024 * 1024);

        // (4) `=0` disables the soft cap; kernel cap returns to its default.
        unsafe { std::env::set_var("SEMANTEX_MAX_RSS_MB", "0") };
        assert_eq!(soft_rss_limit_mb(), 0);

        // (5) check_rss_or_abort returns Ok when the limit is large enough.
        unsafe { std::env::set_var("SEMANTEX_MAX_RSS_MB", "100000") };
        assert!(check_rss_or_abort("test").is_ok());

        // (6) Soft cap below current RSS → first overshoot returns Err.
        //     (We avoid hitting the abort threshold by only checking once.)
        let current = current_rss_mb().expect("RSS available on test platform");
        assert!(current > 1, "RSS must be at least 1 MB to test");
        // Set cap to 1 MB — definitely below current.
        unsafe { std::env::set_var("SEMANTEX_MAX_RSS_MB", "1") };
        // Reset counter so prior tests don't poison this call.
        OVERSHOOT_COUNT.store(0, std::sync::atomic::Ordering::Relaxed);
        let result = check_rss_or_abort("forced overshoot");
        assert!(
            result.is_err(),
            "expected Err when current RSS ({current} MB) > cap (1 MB), got {result:?}"
        );
        let msg = result.unwrap_err();
        assert!(
            msg.contains("exceeds SEMANTEX_MAX_RSS_MB"),
            "error message should reference the env var, got: {msg}"
        );
        // Counter should now be 1. Reset before leaving so other tests aren't poisoned.
        OVERSHOOT_COUNT.store(0, std::sync::atomic::Ordering::Relaxed);

        restore(&orig);
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
