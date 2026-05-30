#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::struct_excessive_bools)]
#![allow(clippy::return_self_not_must_use)]

pub mod chunking;
pub mod config;
pub mod embedding;
pub mod file;
pub mod index;
#[cfg(feature = "llm")]
pub mod llm;
pub mod memory;
pub mod priority;
pub mod search;
pub mod server;
pub mod types;

pub use config::SemantexConfig;
pub use types::*;
