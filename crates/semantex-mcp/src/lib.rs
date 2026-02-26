#![allow(clippy::missing_errors_doc)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::similar_names)]
#![allow(clippy::cast_possible_truncation)]

pub mod protocol;
pub mod server;

pub use server::McpServer;
