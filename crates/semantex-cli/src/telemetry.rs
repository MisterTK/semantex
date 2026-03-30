/// Opt-out usage telemetry — fire-and-forget, never blocks the main path.
///
/// What we collect (and nothing else):
///   - event name (command run: "index", "search", "watch", "serve", "mcp", …)
///   - semantex version (compile-time constant)
///   - OS ("macos" | "linux" | "windows")
///   - CPU architecture ("x86_64" | "aarch64")
///   - a stable anonymous machine ID stored in ~/.semantex/telemetry_id
///
/// We never collect query content, file paths, or any project data.
///
/// Opt out at any time:
///   export SEMANTEX_NO_TELEMETRY=1   # permanent in your shell profile
///   export DO_NOT_TRACK=1            # honoured per https://consoledonottrack.com
///
/// Telemetry is also automatically disabled when CI=1 is set.
use std::thread;
use std::time::Duration;

/// PostHog project write key — safe to ship in the binary (capture-only, no read access).
/// Replace with your own key from https://app.posthog.com/settings/project
const POSTHOG_KEY: &str = "phc_UEenKOEhH6eTI11OwQgo5qxOumaPRHiBSgnqXBy5o6V";

/// PostHog ingestion endpoint.
const ENDPOINT: &str = "https://app.posthog.com/capture/";

/// Returns true when the user has opted out.
pub fn is_opted_out() -> bool {
    // Honoured opt-out signals
    if std::env::var("SEMANTEX_NO_TELEMETRY").is_ok() {
        return true;
    }
    if std::env::var("DO_NOT_TRACK")
        .map(|v| v == "1")
        .unwrap_or(false)
    {
        return true;
    }
    // Suppress in CI environments
    if std::env::var("CI").is_ok() {
        return true;
    }
    if POSTHOG_KEY.is_empty() {
        return true;
    }
    false
}

/// Return (or create) a stable anonymous ID stored at `~/.semantex/telemetry_id`.
///
/// The ID is a random-ish UUID generated once on first run.  It identifies
/// a machine, not a person, and contains no personal information.
fn anonymous_id() -> String {
    let id_path = dirs::home_dir().map(|h| h.join(".semantex").join("telemetry_id"));

    // Try to read an existing ID
    if let Some(path) = &id_path
        && let Ok(contents) = std::fs::read_to_string(path)
    {
        let id = contents.trim().to_string();
        if id.len() >= 8 {
            return id;
        }
    }

    // Generate a new pseudo-random UUID and persist it
    let id = generate_id();
    if let Some(path) = &id_path {
        let _ = std::fs::create_dir_all(path.parent().expect("has parent"));
        let _ = std::fs::write(path, &id);
    }
    id
}

/// Generate a UUID-v4-shaped identifier from time + PID without external crates.
fn generate_id() -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let nanos = u64::from(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos(),
    );
    let pid = u64::from(std::process::id());

    // Mix host info to make IDs less correlated across simultaneous installs
    let mut hasher = DefaultHasher::new();
    nanos.hash(&mut hasher);
    pid.hash(&mut hasher);
    if let Ok(host) = std::env::var("HOSTNAME").or_else(|_| std::env::var("COMPUTERNAME")) {
        host.hash(&mut hasher);
    }
    let h = hasher.finish();
    let h2 = h.wrapping_mul(0x9e37_79b9_7f4a_7c15).wrapping_add(nanos);

    format!(
        "{:08x}-{:04x}-4{:03x}-{:04x}-{:012x}",
        h & 0xffff_ffff,
        (h >> 32) & 0xffff,
        (h >> 48) & 0x0fff,
        ((h2 >> 16) & 0x3fff) | 0x8000,
        h2 & 0x0000_ffff_ffff
    )
}

/// Fire-and-forget: record that `command` was run.
///
/// This spawns a background thread and returns immediately.  The thread is
/// killed when the process exits, so there's no added latency on the hot path.
pub fn track(command: &'static str) {
    if is_opted_out() {
        return;
    }

    let distinct_id = anonymous_id();
    let version = env!("CARGO_PKG_VERSION");
    let os = if cfg!(target_os = "macos") {
        "macos"
    } else if cfg!(target_os = "linux") {
        "linux"
    } else if cfg!(target_os = "windows") {
        "windows"
    } else {
        "unknown"
    };
    let arch = if cfg!(target_arch = "x86_64") {
        "x86_64"
    } else if cfg!(target_arch = "aarch64") {
        "aarch64"
    } else {
        "other"
    };

    // Build the PostHog capture payload
    let body = format!(
        r#"{{"api_key":"{POSTHOG_KEY}","event":"command_run","distinct_id":"{distinct_id}","properties":{{"command":"{command}","version":"{version}","os":"{os}","arch":"{arch}","$lib":"semantex"}}}}"#,
    );

    thread::spawn(move || {
        // Build an agent with a short timeout so a slow network never blocks
        let config = ureq::config::Config::builder()
            .timeout_global(Some(Duration::from_secs(3)))
            .build();
        let agent = ureq::Agent::new_with_config(config);
        let _ = agent
            .post(ENDPOINT)
            .header("content-type", "application/json")
            .send(body.as_bytes());
    });
}
