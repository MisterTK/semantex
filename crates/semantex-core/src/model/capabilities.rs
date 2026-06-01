//! `ModelCapabilities` + capability→backend negotiation.
use serde::{Deserialize, Serialize};

/// Engine-negotiable model capabilities. Missing manifest fields default to the
/// conservative single-vector profile so an unknown future capability is OFF for
/// existing models (they keep working). Task 5 adds the negotiation fn.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ModelCapabilities {
    /// `true` → per-token vectors (ColBERT/PLAID MaxSim); `false` → single-vector.
    #[serde(default)]
    pub multi_vector: bool,
}
