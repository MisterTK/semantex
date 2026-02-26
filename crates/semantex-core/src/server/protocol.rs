use crate::search::SearchMetrics;
use serde::{Deserialize, Serialize};

/// Magic byte prefix to distinguish binary (bincode) from JSON protocol.
/// A JSON message always starts with '{' (0x7b), so 0x00 is unambiguous.
pub const BINARY_MAGIC: u8 = 0x00;

/// Request sent from client to daemon over TCP
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Request {
    Search(SearchRequest),
    Health,
    Shutdown,
}

/// Search request parameters
#[derive(Debug, Serialize, Deserialize)]
pub struct SearchRequest {
    pub query: String,
    #[serde(default = "default_max_results")]
    pub max_results: usize,
    #[serde(default = "default_true")]
    pub use_dense: bool,
    #[serde(default = "default_true")]
    pub use_sparse: bool,
    #[serde(default)]
    pub use_rerank: bool,
    #[serde(default)]
    pub include_types: Vec<String>,
    #[serde(default)]
    pub exclude_types: Vec<String>,
    #[serde(default)]
    pub code_only: bool,
    #[serde(default = "default_true")]
    pub include_content: bool,
    #[serde(default)]
    pub snippet: bool,
    /// Grep parity mode: exact+BM25 only, no dense/rerank, exhaustive
    #[serde(default)]
    pub grep_mode: bool,
    /// Optional regex pattern from `-e` flag for regex-semantic hybrid search
    #[serde(default)]
    pub regex_pattern: Option<String>,
}

fn default_max_results() -> usize {
    10
}

fn default_true() -> bool {
    true
}

/// Response sent from daemon to client
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Response {
    Search(SearchResponse),
    Health(HealthResponse),
    Shutdown(ShutdownResponse),
    Error(ErrorResponse),
}

/// Search results response
#[derive(Debug, Serialize, Deserialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResultItem>,
    pub duration_ms: u64,
    pub dense_count: usize,
    pub sparse_count: usize,
    pub fused_count: usize,
    #[serde(default)]
    pub metrics: Option<SearchMetrics>,
}

/// A single result item in the response
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SearchResultItem {
    pub file: String,
    pub start_line: u32,
    pub end_line: u32,
    pub score: f32,
    pub source: String,
    pub chunk_type: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default)]
    pub content: Option<String>,
}

/// Health check response
#[derive(Debug, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub uptime_s: u64,
    pub searches: u64,
}

/// Shutdown acknowledgment
#[derive(Debug, Serialize, Deserialize)]
pub struct ShutdownResponse {
    pub status: String,
}

/// Error response
#[derive(Debug, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub message: String,
}

// --- Binary (bincode) wire format ---
//
// Binary messages use a simple framing:
//   [BINARY_MAGIC: 1 byte] [length: 4 bytes LE] [bincode payload: length bytes]
//
// The first byte lets the listener auto-detect whether a client is sending
// JSON (starts with '{') or binary (starts with 0x00).

/// Binary request type tag for bincode (serde tags don't work well with bincode).
#[derive(Debug, Serialize, Deserialize)]
pub enum BinaryRequest {
    Search(SearchRequest),
    Health,
    Shutdown,
}

/// Binary response type for bincode.
#[derive(Debug, Serialize, Deserialize)]
pub enum BinaryResponse {
    Search(SearchResponse),
    Health(HealthResponse),
    Shutdown(ShutdownResponse),
    Error(ErrorResponse),
}

impl From<BinaryRequest> for Request {
    fn from(br: BinaryRequest) -> Self {
        match br {
            BinaryRequest::Search(s) => Request::Search(s),
            BinaryRequest::Health => Request::Health,
            BinaryRequest::Shutdown => Request::Shutdown,
        }
    }
}

impl From<Response> for BinaryResponse {
    fn from(r: Response) -> Self {
        match r {
            Response::Search(s) => BinaryResponse::Search(s),
            Response::Health(h) => BinaryResponse::Health(h),
            Response::Shutdown(s) => BinaryResponse::Shutdown(s),
            Response::Error(e) => BinaryResponse::Error(e),
        }
    }
}

/// Encode a binary request into a framed message: [0x00][len:4 LE][bincode]
pub fn encode_binary_request(req: &BinaryRequest) -> Vec<u8> {
    let payload =
        bincode::serde::encode_to_vec(req, bincode::config::standard()).expect("bincode serialize");
    let len = payload.len() as u32;
    let mut buf = Vec::with_capacity(1 + 4 + payload.len());
    buf.push(BINARY_MAGIC);
    buf.extend_from_slice(&len.to_le_bytes());
    buf.extend_from_slice(&payload);
    buf
}

/// Encode a binary response into a framed message: [0x00][len:4 LE][bincode]
pub fn encode_binary_response(resp: &BinaryResponse) -> Vec<u8> {
    let payload = bincode::serde::encode_to_vec(resp, bincode::config::standard())
        .expect("bincode serialize");
    let len = payload.len() as u32;
    let mut buf = Vec::with_capacity(1 + 4 + payload.len());
    buf.push(BINARY_MAGIC);
    buf.extend_from_slice(&len.to_le_bytes());
    buf.extend_from_slice(&payload);
    buf
}

/// Decode a binary response from raw bytes (after stripping the magic+length frame).
pub fn decode_binary_response(data: &[u8]) -> Result<BinaryResponse, bincode::error::DecodeError> {
    bincode::serde::decode_from_slice(data, bincode::config::standard()).map(|(v, _)| v)
}

/// Decode a binary request from raw bytes (after stripping the magic+length frame).
pub fn decode_binary_request(data: &[u8]) -> Result<BinaryRequest, bincode::error::DecodeError> {
    bincode::serde::decode_from_slice(data, bincode::config::standard()).map(|(v, _)| v)
}
