use crate::search::SearchMetrics;
use serde::{Deserialize, Serialize};

/// Magic byte prefix to distinguish binary (postcard) from JSON protocol.
/// A JSON message always starts with '{' (0x7b), so 0x00 is unambiguous.
pub const BINARY_MAGIC: u8 = 0x00;

/// Daemon binary-protocol version (v0.4.1 W-Index #2).
///
/// Encoded as the second byte of every binary frame, immediately after
/// `BINARY_MAGIC`. A client and daemon that disagree on this byte refuse to
/// decode the frame and surface a clean error instead of silently
/// mis-deserializing a postcard payload.
///
/// Bump this constant whenever the binary frame layout, the `BinaryRequest` /
/// `BinaryResponse` tag set, or any non-self-describing postcard schema in
/// those types changes. Postcard does not encode field/type tags, so silent
/// drift is a real risk and the version byte is the cheapest way to catch it.
pub const BINARY_PROTOCOL_VERSION: u8 = 1;

/// Error type returned by binary decode when the framing is malformed.
///
/// Wraps both postcard decode errors and our own version/magic mismatches so
/// callers (listener, client) can render a single uniform error path.
#[derive(Debug)]
pub enum BinaryFrameError {
    /// The version byte following `BINARY_MAGIC` did not match
    /// `BINARY_PROTOCOL_VERSION`. The mismatched value is included so the
    /// log message can distinguish "wrong version" from "garbage byte".
    UnsupportedVersion { expected: u8, got: u8 },
    /// Postcard failed to decode the payload (truncated, schema drift, etc.).
    Decode(postcard::Error),
}

impl std::fmt::Display for BinaryFrameError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedVersion { expected, got } => {
                write!(
                    f,
                    "unsupported binary protocol version: expected {expected}, got {got}"
                )
            }
            Self::Decode(e) => write!(f, "postcard decode error: {e}"),
        }
    }
}

impl std::error::Error for BinaryFrameError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::UnsupportedVersion { .. } => None,
            Self::Decode(e) => Some(e),
        }
    }
}

impl From<postcard::Error> for BinaryFrameError {
    fn from(e: postcard::Error) -> Self {
        Self::Decode(e)
    }
}

/// Request sent from client to daemon over TCP
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Request {
    Search(SearchRequest),
    Health,
    Shutdown,
    GraphWalk(GraphWalkRequest),
    MultiSearch(MultiSearchRequest),
    DeepSearch(DeepSearchRequest),
    Agent(AgentRequest),
}

/// Search request parameters
#[derive(Debug, Serialize, Deserialize, Clone)]
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
    /// When true, include content for the top-1 result even when include_content=false.
    /// Used by --refs mode to auto-peek the highest-confidence result.
    #[serde(default)]
    pub auto_peek_top: bool,
}

fn default_max_results() -> usize {
    10
}

fn default_true() -> bool {
    true
}

fn default_deep_max_results() -> usize {
    20
}

/// Deep search request: search, read, and summarize into a prose answer
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DeepSearchRequest {
    pub query: String,
    #[serde(default = "default_deep_max_results")]
    pub max_results: usize,
    #[serde(default = "default_true")]
    pub use_graph: bool,
}

/// Deep search source reference
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DeepSearchSource {
    pub file: String,
    pub start_line: u32,
    pub end_line: u32,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub kind: Option<String>,
}

/// Deep search metrics
#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct DeepResponseMetrics {
    pub search_ms: u64,
    pub triage_ms: u64,
    pub graph_ms: u64,
    pub read_ms: u64,
    pub summarize_ms: u64,
    pub total_ms: u64,
    pub chunks_searched: usize,
    pub chunks_read: usize,
    /// Confidence zone: "high", "medium", "low", or "no_results"
    #[serde(default)]
    pub confidence_zone: String,
}

/// Deep search response: prose answer + source refs + metrics
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DeepSearchResponse {
    pub answer: String,
    pub sources: Vec<DeepSearchSource>,
    pub metrics: DeepResponseMetrics,
    /// Normalized confidence (0.0–1.0) for the overall result quality.
    #[serde(default)]
    pub confidence: f32,
}

/// Response sent from daemon to client
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Response {
    Search(SearchResponse),
    Health(HealthResponse),
    Shutdown(ShutdownResponse),
    Error(ErrorResponse),
    GraphWalk(GraphWalkResponse),
    MultiSearch(MultiSearchResponse),
    DeepSearch(DeepSearchResponse),
    Agent(AgentResponse),
}

/// Search results response
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SearchResponse {
    pub results: Vec<SearchResultItem>,
    pub duration_ms: u64,
    pub dense_count: usize,
    pub sparse_count: usize,
    pub fused_count: usize,
    #[serde(default)]
    pub metrics: Option<SearchMetrics>,
    /// Confidence hint for the agent: "high", "medium", "low", or "none"
    /// Based on top result score and score gap to second result.
    #[serde(default)]
    pub confidence: Option<String>,
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
    #[serde(default)]
    pub kind: Option<String>,
    #[serde(default)]
    pub summary: Option<String>,
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

/// Multi-query batch request: run N searches in a single round trip
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MultiSearchRequest {
    pub queries: Vec<SearchRequest>,
}

/// Multi-query batch response: one SearchResponse per query
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MultiSearchResponse {
    pub responses: Vec<SearchResponse>,
}

/// Graph walk request: resolve symbol and return its structural neighbors
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GraphWalkRequest {
    pub symbol: String,
}

/// Graph walk response: callers, callees, type refs, hierarchy for a symbol
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GraphWalkResponse {
    pub target: Vec<SearchResultItem>,
    pub callers: Vec<SearchResultItem>,
    pub callees: Vec<SearchResultItem>,
    pub type_refs: Vec<SearchResultItem>,
    pub hierarchy: Vec<SearchResultItem>,
}

// --- Binary (postcard) wire format ---
//
// Binary messages use a versioned framing (v0.4.1 W-Index #2):
//   [BINARY_MAGIC: 1 byte] [BINARY_PROTOCOL_VERSION: 1 byte]
//   [length: 4 bytes LE] [postcard payload: length bytes]
//
// The first byte lets the listener auto-detect whether a client is sending
// JSON (starts with '{') or binary (starts with 0x00). The second byte is a
// protocol version so client/daemon mismatch surfaces a clean error instead
// of a silently-misdeserialized postcard payload.

/// Binary request type tag for postcard (serde tags don't work well with non-self-describing formats).
#[derive(Debug, Serialize, Deserialize)]
pub enum BinaryRequest {
    Search(SearchRequest),
    Health,
    Shutdown,
    GraphWalk(GraphWalkRequest),
    MultiSearch(MultiSearchRequest),
    DeepSearch(DeepSearchRequest),
    Agent(AgentRequest),
}

/// Binary response type for postcard.
#[derive(Debug, Serialize, Deserialize)]
pub enum BinaryResponse {
    Search(SearchResponse),
    Health(HealthResponse),
    Shutdown(ShutdownResponse),
    Error(ErrorResponse),
    GraphWalk(GraphWalkResponse),
    MultiSearch(MultiSearchResponse),
    DeepSearch(DeepSearchResponse),
    Agent(AgentResponse),
}

impl From<BinaryRequest> for Request {
    fn from(br: BinaryRequest) -> Self {
        match br {
            BinaryRequest::Search(s) => Request::Search(s),
            BinaryRequest::Health => Request::Health,
            BinaryRequest::Shutdown => Request::Shutdown,
            BinaryRequest::GraphWalk(g) => Request::GraphWalk(g),
            BinaryRequest::MultiSearch(m) => Request::MultiSearch(m),
            BinaryRequest::DeepSearch(d) => Request::DeepSearch(d),
            BinaryRequest::Agent(a) => Request::Agent(a),
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
            Response::GraphWalk(g) => BinaryResponse::GraphWalk(g),
            Response::MultiSearch(m) => BinaryResponse::MultiSearch(m),
            Response::DeepSearch(d) => BinaryResponse::DeepSearch(d),
            Response::Agent(a) => BinaryResponse::Agent(a),
        }
    }
}

/// Encode a binary request into a framed message:
/// `[BINARY_MAGIC][len:4 LE][BINARY_PROTOCOL_VERSION][postcard]`.
///
/// `len` covers the version byte and the postcard payload. Placing the
/// version inside the length-prefixed body lets the existing
/// `[MAGIC][len:4][body]` framing readers parse the frame unchanged; only
/// the decoder, which consumes byte 0 of `body` as the version, has new
/// behaviour.
pub fn encode_binary_request(req: &BinaryRequest) -> Vec<u8> {
    let payload = postcard::to_stdvec(req).expect("postcard serialize");
    let body_len = 1 + payload.len();
    let len = body_len as u32;
    let mut buf = Vec::with_capacity(BINARY_FRAME_HEADER_LEN + payload.len());
    buf.push(BINARY_MAGIC);
    buf.extend_from_slice(&len.to_le_bytes());
    buf.push(BINARY_PROTOCOL_VERSION);
    buf.extend_from_slice(&payload);
    buf
}

/// Encode a binary response into a framed message:
/// `[BINARY_MAGIC][len:4 LE][BINARY_PROTOCOL_VERSION][postcard]`.
pub fn encode_binary_response(resp: &BinaryResponse) -> Vec<u8> {
    let payload = postcard::to_stdvec(resp).expect("postcard serialize");
    let body_len = 1 + payload.len();
    let len = body_len as u32;
    let mut buf = Vec::with_capacity(BINARY_FRAME_HEADER_LEN + payload.len());
    buf.push(BINARY_MAGIC);
    buf.extend_from_slice(&len.to_le_bytes());
    buf.push(BINARY_PROTOCOL_VERSION);
    buf.extend_from_slice(&payload);
    buf
}

/// Length of the fixed-size frame header that precedes the length-prefixed
/// body: `[MAGIC: 1 byte][len: 4 bytes LE]`. The body itself begins with the
/// 1-byte protocol version followed by the postcard payload.
pub const BINARY_FRAME_HEADER_LEN: usize = 1 + 4;

/// Decode a binary response body. The body is `[VERSION: 1 byte][postcard]`,
/// i.e. exactly the bytes the caller read off the wire as the length-prefixed
/// payload. Returns `UnsupportedVersion` if byte 0 disagrees with
/// `BINARY_PROTOCOL_VERSION`; postcard errors surface as `Decode`.
///
/// An empty `data` slice returns `UnsupportedVersion { got: 0 }` to surface
/// the truncation; the empty frame is never a valid response.
pub fn decode_binary_response(data: &[u8]) -> Result<BinaryResponse, BinaryFrameError> {
    let Some((&version, payload)) = data.split_first() else {
        return Err(BinaryFrameError::UnsupportedVersion {
            expected: BINARY_PROTOCOL_VERSION,
            got: 0,
        });
    };
    if version != BINARY_PROTOCOL_VERSION {
        return Err(BinaryFrameError::UnsupportedVersion {
            expected: BINARY_PROTOCOL_VERSION,
            got: version,
        });
    }
    Ok(postcard::from_bytes::<BinaryResponse>(payload)?)
}

/// Decode a binary request body. The body is `[VERSION: 1 byte][postcard]`.
/// Returns `UnsupportedVersion` on mismatch and `Decode` on postcard failure.
pub fn decode_binary_request(data: &[u8]) -> Result<BinaryRequest, BinaryFrameError> {
    let Some((&version, payload)) = data.split_first() else {
        return Err(BinaryFrameError::UnsupportedVersion {
            expected: BINARY_PROTOCOL_VERSION,
            got: 0,
        });
    };
    if version != BINARY_PROTOCOL_VERSION {
        return Err(BinaryFrameError::UnsupportedVersion {
            expected: BINARY_PROTOCOL_VERSION,
            got: version,
        });
    }
    Ok(postcard::from_bytes::<BinaryRequest>(payload)?)
}

// --- Agent types ---

use crate::search::agent_classifier::AgentRoute;

/// Request for agent-orchestrated search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentRequest {
    pub query: String,
    /// Override automatic classification with a specific route.
    #[serde(default)]
    pub route: Option<AgentRoute>,
    /// Response budget in bytes. Default: 12000 (~3K tokens).
    #[serde(default)]
    pub budget: Option<usize>,
    /// Include full source code blocks (for analytical queries).
    #[serde(default)]
    pub full_code: bool,
}

/// Response from agent-orchestrated search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentResponse {
    pub route: AgentRoute,
    pub formatted: String,
    pub metrics: AgentMetrics,
}

/// Performance metrics for agent queries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMetrics {
    pub classify_us: u64,
    pub search_ms: u64,
    pub format_ms: u64,
    pub total_ms: u64,
    pub fallback_used: bool,
    pub result_count: usize,
}

#[cfg(test)]
mod agent_protocol_tests {
    use super::*;

    #[test]
    fn test_agent_request_json_roundtrip() {
        let req = AgentRequest {
            query: "how does auth work?".into(),
            route: None,
            budget: Some(8000),
            full_code: false,
        };
        let json = serde_json::to_string(&req).unwrap();
        let parsed: AgentRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.query, "how does auth work?");
        assert_eq!(parsed.budget, Some(8000));
    }

    #[test]
    fn test_agent_request_with_route() {
        use crate::search::agent_classifier::AgentRoute;
        let json = r#"{"query":"AuthService","route":"exact_symbol"}"#;
        let req: AgentRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.route, Some(AgentRoute::ExactSymbol));
    }

    #[test]
    fn test_agent_response_json_roundtrip() {
        use crate::search::agent_classifier::AgentRoute;
        let resp = AgentResponse {
            route: AgentRoute::Semantic,
            formatted: "[route: semantic]\n\nresults here".into(),
            metrics: AgentMetrics {
                classify_us: 5,
                search_ms: 17,
                format_ms: 0,
                total_ms: 18,
                fallback_used: false,
                result_count: 3,
            },
        };
        let json = serde_json::to_string(&resp).unwrap();
        let parsed: AgentResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.route, AgentRoute::Semantic);
        assert!(parsed.formatted.contains("[route: semantic]"));
    }

    #[test]
    fn test_agent_binary_roundtrip() {
        let req = BinaryRequest::Agent(AgentRequest {
            query: "test query".into(),
            route: None,
            budget: None,
            full_code: false,
        });
        let encoded = encode_binary_request(&req);
        // v0.4.1 W-Index #2: frame is [MAGIC][len:4][VERSION][postcard]; the
        // length-prefixed body (decode input) begins at BINARY_FRAME_HEADER_LEN
        // and starts with the version byte.
        assert_eq!(encoded[0], BINARY_MAGIC);
        assert_eq!(encoded[BINARY_FRAME_HEADER_LEN], BINARY_PROTOCOL_VERSION);
        let decoded = decode_binary_request(&encoded[BINARY_FRAME_HEADER_LEN..]).unwrap();
        match decoded {
            BinaryRequest::Agent(r) => assert_eq!(r.query, "test query"),
            _ => panic!("Wrong variant"),
        }
    }

    /// v0.4.1 W-Index #2: a frame whose version byte is something other than
    /// `BINARY_PROTOCOL_VERSION` must produce a clean `UnsupportedVersion`
    /// error rather than silently mis-decoding.
    #[test]
    fn unsupported_version_rejected_on_decode() {
        // Build a synthetic body with a bogus version prefix.
        let bogus_version = 99u8;
        let postcard_payload = postcard::to_stdvec(&BinaryRequest::Health).unwrap();
        let mut body = vec![bogus_version];
        body.extend_from_slice(&postcard_payload);

        let err = decode_binary_request(&body).expect_err("must reject");
        match err {
            BinaryFrameError::UnsupportedVersion { expected, got } => {
                assert_eq!(expected, BINARY_PROTOCOL_VERSION);
                assert_eq!(got, bogus_version);
            }
            BinaryFrameError::Decode(e) => panic!("expected version mismatch, got decode: {e}"),
        }
    }

    /// Symmetric check on the response side.
    #[test]
    fn unsupported_version_rejected_on_response_decode() {
        let postcard_payload = postcard::to_stdvec(&BinaryResponse::Health(HealthResponse {
            status: "ok".into(),
            uptime_s: 0,
            searches: 0,
        }))
        .unwrap();
        let mut body = vec![42u8];
        body.extend_from_slice(&postcard_payload);

        let err = decode_binary_response(&body).expect_err("must reject");
        assert!(matches!(
            err,
            BinaryFrameError::UnsupportedVersion {
                expected: BINARY_PROTOCOL_VERSION,
                got: 42
            }
        ));
    }

    /// An empty body (length-prefixed truncation) must surface as
    /// `UnsupportedVersion { got: 0 }` rather than panic.
    #[test]
    fn empty_frame_rejected_cleanly() {
        let err = decode_binary_request(&[]).expect_err("must reject");
        assert!(matches!(
            err,
            BinaryFrameError::UnsupportedVersion {
                expected: BINARY_PROTOCOL_VERSION,
                got: 0
            }
        ));
    }
}
