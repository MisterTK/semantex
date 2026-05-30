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
/// `SEMANTEX_MAX_CONCURRENT_BUILDS` overrides (a zero/invalid value falls back
/// to the default). Otherwise `max(1, RAM_GB / 16)`: 16 GB → 1, 32 GB → 2,
/// 64 GB → 4. Conservative on purpose — the cost of queueing a build is latency;
/// the cost of over-committing is an OOM that takes down the whole machine.
pub fn max_concurrent_builds() -> usize {
    let ram_default = match crate::memory::system_ram_mb() {
        Some(mb) => (((mb / 1024) / RAM_GB_PER_BUILD).max(1)) as usize,
        None => FALLBACK_CONCURRENCY,
    };
    crate::config::env_usize(ENV_MAX_BUILDS, ram_default)
}

/// Directory holding the slot lock files, under the shared semantex home (which
/// honors `SEMANTEX_HOME` and falls back to a temp dir).
fn slots_dir() -> PathBuf {
    crate::config::SemantexConfig::semantex_home().join("build-slots")
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
/// Returns `None` (build proceeds ungated) only if the slots directory or its
/// lock files can't be created — we never want a path/permission quirk to
/// *block* indexing entirely. `log_waiting` is invoked at most once, the first
/// time we have to wait, so the CLI can tell the user why it's pausing.
pub fn acquire(log_waiting: impl FnOnce()) -> Option<BuildSlot> {
    let dir = slots_dir();
    if std::fs::create_dir_all(&dir).is_err() {
        return None;
    }
    // Open every slot file ONCE, then re-probe these handles with `try_lock` on
    // each poll — re-opening per poll would be needless syscall churn over a
    // wait that can last minutes. A held lock is tied to the open file handle,
    // so we keep the one we lock and drop the rest. (Ok => we hold it; Err =>
    // held by another builder, or a lock-less FS like NFS — skip and retry.)
    let mut files: Vec<File> = (0..max_concurrent_builds())
        .filter_map(|i| File::create(dir.join(format!("slot-{i}.lock"))).ok())
        .collect();
    if files.is_empty() {
        return None;
    }

    let mut log_waiting = Some(log_waiting);
    let start = Instant::now();
    loop {
        if let Some(i) = files.iter().position(|f| f.try_lock().is_ok()) {
            return Some(BuildSlot {
                _file: files.swap_remove(i),
            });
        }
        // Announce the wait exactly once (the first time around).
        if let Some(f) = log_waiting.take() {
            f();
        }
        // Safety valve against a wedged slot holder that didn't release its OS
        // lock: proceed ungated rather than block a build forever.
        if start.elapsed() > MAX_WAIT {
            return None;
        }
        std::thread::sleep(POLL_INTERVAL);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // These tests mutate the process-global ENV_MAX_BUILDS, so they must not
    // run concurrently with each other.
    static ENV_GUARD: Mutex<()> = Mutex::new(());

    #[test]
    fn env_override_is_honored() {
        let _g = ENV_GUARD.lock().unwrap();
        // SAFETY: serialized by ENV_GUARD; restored before releasing the lock.
        unsafe { std::env::set_var(ENV_MAX_BUILDS, "3") };
        assert_eq!(max_concurrent_builds(), 3);
        // A nonsensical "0" falls back to the (always ≥ 1) RAM default.
        unsafe { std::env::set_var(ENV_MAX_BUILDS, "0") };
        assert!(max_concurrent_builds() >= 1, "must stay ≥ 1");
        unsafe { std::env::remove_var(ENV_MAX_BUILDS) };
    }

    #[test]
    fn default_scales_with_ram_and_is_at_least_one() {
        let _g = ENV_GUARD.lock().unwrap();
        unsafe { std::env::remove_var(ENV_MAX_BUILDS) };
        // Whatever the host RAM, the default must be a sane positive number.
        assert!(max_concurrent_builds() >= 1);
    }

    #[test]
    fn slot_is_held_until_dropped_then_reusable() {
        let _g = ENV_GUARD.lock().unwrap();
        unsafe { std::env::set_var(ENV_MAX_BUILDS, "1") };
        let slot_path = slots_dir().join("slot-0.lock");

        let first = acquire(|| {});
        assert!(first.is_some(), "first acquire should get the only slot");

        // A fresh handle to the same slot must observe it locked while held.
        let probe = File::create(&slot_path).unwrap();
        assert!(probe.try_lock().is_err(), "slot must be locked while held");
        drop(probe);

        drop(first);
        let probe2 = File::create(&slot_path).unwrap();
        assert!(
            probe2.try_lock().is_ok(),
            "slot should be free after the guard is dropped"
        );
        unsafe { std::env::remove_var(ENV_MAX_BUILDS) };
    }
}
