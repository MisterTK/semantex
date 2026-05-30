//! Cross-repo index-build concurrency gate.
//!
//! # Why
//!
//! semantex is meant to run on *every* repo a developer works in: an assistant
//! auto-indexes on first use, and many sessions across many repos can be active
//! at once. A full index build holds the chunk's token embeddings plus the
//! k-means working set in RAM — several GB per build. If ten repos start a full
//! build at the same moment (e.g. right after the machine reboots and ten
//! assistants reconnect), the box thrashes or OOMs.
//!
//! The historical mitigation was to cripple every build to a single ORT thread
//! so many could coexist. That made indexing pathologically slow. We instead
//! **bound how many heavy builds run at once** and then let each one use real
//! CPU ([`ColbertEmbedder::for_indexing`](crate::embedding::colbert::ColbertEmbedder::for_indexing)).
//! Few-fast-serialized beats many-slow-parallel on wall-clock, peak memory, and
//! the common single-repo case.
//!
//! # Mechanism
//!
//! A counting semaphore built from `N` advisory lock files under
//! `~/.semantex/build-slots/`. To acquire, try `File::try_lock` on each slot in
//! turn and hold the first that succeeds; if all are held, wait and retry. The
//! held [`BuildSlot`] releases the lock on drop. Advisory `flock`-style locks
//! are released by the OS when the holding process exits, so a crashed builder
//! never wedges a slot — no stale-lock cleanup needed.
//!
//! Only **full** builds take a slot. Incremental updates (a handful of changed
//! files, no k-means) are cheap and already serialized per-repo by
//! `.semantex.lock`, so they bypass the gate to keep restarts instant.

use std::fs::File;
use std::path::PathBuf;
use std::time::{Duration, Instant};

/// Env override for the maximum number of concurrent full builds.
const ENV_MAX_BUILDS: &str = "SEMANTEX_MAX_CONCURRENT_BUILDS";

/// Each full build is budgeted at roughly this much RAM when deriving the
/// default concurrency from system memory. Big repos (tens of thousands of
/// chunks) peak here between held token embeddings and the k-means working set.
const RAM_GB_PER_BUILD: u64 = 16;

/// Default concurrency when system RAM can't be detected.
const FALLBACK_CONCURRENCY: usize = 1;

/// Returns the maximum number of concurrent full index builds.
///
/// `SEMANTEX_MAX_CONCURRENT_BUILDS` overrides (clamped to ≥ 1). Otherwise
/// `max(1, RAM_GB / 16)`: 16 GB → 1, 32 GB → 2, 64 GB → 4. Conservative on
/// purpose — the cost of queueing a build is latency; the cost of over-
/// committing is an OOM that takes down the developer's whole machine.
pub fn max_concurrent_builds() -> usize {
    if let Some(n) = std::env::var(ENV_MAX_BUILDS)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        return n.max(1);
    }
    match crate::memory::system_ram_mb() {
        Some(mb) => ((mb / 1024) / RAM_GB_PER_BUILD).max(1) as usize,
        None => FALLBACK_CONCURRENCY,
    }
}

/// Directory holding the slot lock files. `None` if the home dir is unknown
/// (in which case the gate is skipped — better to build than to wedge).
fn slots_dir() -> Option<PathBuf> {
    dirs::home_dir().map(|h| h.join(".semantex").join("build-slots"))
}

/// A held build slot. Releasing the underlying file lock happens automatically
/// when this is dropped (or when the process exits).
#[must_use = "dropping the BuildSlot immediately releases the slot"]
pub struct BuildSlot {
    _file: File,
}

/// How long to wait between sweeps of the slots when all are currently held.
const POLL_INTERVAL: Duration = Duration::from_millis(250);

/// Safety valve: if we somehow wait this long for a slot, proceed ungated
/// rather than block a build forever. Far beyond any real build time.
const MAX_WAIT: Duration = Duration::from_hours(1);

/// Acquire one of the `max_concurrent_builds()` global build slots, blocking
/// until one is free.
///
/// Returns `None` (build proceeds ungated) only if the home directory or slots
/// directory can't be set up — we never want a path/permission quirk to *block*
/// indexing entirely. `log_waiting` is invoked once if the caller has to wait,
/// so the CLI can tell the user why it's pausing.
pub fn acquire(log_waiting: impl FnOnce()) -> Option<BuildSlot> {
    let dir = slots_dir()?;
    if std::fs::create_dir_all(&dir).is_err() {
        return None;
    }
    let limit = max_concurrent_builds();

    // First full sweep: grab any immediately-free slot without sleeping.
    if let Some(slot) = try_sweep(&dir, limit) {
        return Some(slot);
    }

    // All slots busy — wait. Announce once, then poll.
    log_waiting();
    let start = Instant::now();
    loop {
        std::thread::sleep(POLL_INTERVAL);
        if let Some(slot) = try_sweep(&dir, limit) {
            return Some(slot);
        }
        // Safety valve against a wedged slot holder that didn't release its OS
        // lock: proceed ungated rather than block a build forever.
        if start.elapsed() > MAX_WAIT {
            return None;
        }
    }
}

/// Try each slot once; return the first we can lock.
fn try_sweep(dir: &std::path::Path, limit: usize) -> Option<BuildSlot> {
    for i in 0..limit {
        let path = dir.join(format!("slot-{i}.lock"));
        let Ok(file) = File::create(&path) else {
            continue;
        };
        // Ok => we hold this slot. Err => held by another builder, or a
        // filesystem that doesn't support locking (NFS); skip it. If every slot
        // errors we fall through and the caller retries / eventually proceeds.
        if file.try_lock().is_ok() {
            return Some(BuildSlot { _file: file });
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_override_is_honored_and_floored() {
        // SAFETY: single-threaded test; restored implicitly at process exit.
        unsafe { std::env::set_var(ENV_MAX_BUILDS, "3") };
        assert_eq!(max_concurrent_builds(), 3);
        unsafe { std::env::set_var(ENV_MAX_BUILDS, "0") };
        assert_eq!(max_concurrent_builds(), 1, "must floor at 1");
        unsafe { std::env::remove_var(ENV_MAX_BUILDS) };
    }

    #[test]
    fn default_scales_with_ram_and_is_at_least_one() {
        unsafe { std::env::remove_var(ENV_MAX_BUILDS) };
        // Whatever the host RAM, the default must be a sane positive number.
        assert!(max_concurrent_builds() >= 1);
    }

    #[test]
    fn acquire_then_release_lets_next_caller_in() {
        // With the env forcing a single slot, the first guard must hold it and
        // the second sweep must find nothing — then dropping frees it.
        unsafe { std::env::set_var(ENV_MAX_BUILDS, "1") };
        let dir = match slots_dir() {
            Some(d) => d,
            None => return, // no home dir in this sandbox; nothing to assert
        };
        let _ = std::fs::create_dir_all(&dir);
        let first = try_sweep(&dir, 1);
        assert!(first.is_some(), "first acquire should get the only slot");
        assert!(
            try_sweep(&dir, 1).is_none(),
            "no slot should be free while the first is held"
        );
        drop(first);
        assert!(
            try_sweep(&dir, 1).is_some(),
            "slot should be reusable after release"
        );
        unsafe { std::env::remove_var(ENV_MAX_BUILDS) };
    }
}
