use anyhow::Result;

use crate::client;

/// Stop the persistent client.
pub fn run() -> Result<()> {
    if client::stop_client()? {
        eprintln!("Persistent client stopped.");
    } else {
        eprintln!("No persistent client running.");
    }

    Ok(())
}
