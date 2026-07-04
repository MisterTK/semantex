#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::similar_names)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::too_many_lines)]

pub mod docs_context;
pub mod memory_tools;
pub mod protocol;
pub mod server;

#[cfg(feature = "http")]
pub mod http_transport;

pub use server::McpServer;
