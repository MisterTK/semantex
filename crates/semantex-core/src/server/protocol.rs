use crate::search::SearchMetrics;
use serde::{Deserialize, Serialize};

/// Magic byte prefix to distinguish binary (postcard) from JSON protocol.
/// A JSON message always starts with '{' (0x7b), so 0x00 is unambiguous.
pub const BINARY_MAGIC: u8 = 0x00;

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
// Binary messages use a simple framing:
//   [BINARY_MAGIC: 1 byte] [length: 4 bytes LE] [postcard payload: length bytes]
//
// The first byte lets the listener auto-detect whether a client is sending
// JSON (starts with '{') or binary (starts with 0x00).

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

/// Encode a binary request into a framed message: [0x00][len:4 LE][postcard]
pub fn encode_binary_request(req: &BinaryRequest) -> Vec<u8> {
    let payload = postcard::to_stdvec(req).expect("postcard serialize");
    let len = payload.len() as u32;
    let mut buf = Vec::with_capacity(1 + 4 + payload.len());
    buf.push(BINARY_MAGIC);
    buf.extend_from_slice(&len.to_le_bytes());
    buf.extend_from_slice(&payload);
    buf
}

/// Encode a binary response into a framed message: [0x00][len:4 LE][postcard]
pub fn encode_binary_response(resp: &BinaryResponse) -> Vec<u8> {
    let payload = postcard::to_stdvec(resp).expect("postcard serialize");
    let len = payload.len() as u32;
    let mut buf = Vec::with_capacity(1 + 4 + payload.len());
    buf.push(BINARY_MAGIC);
    buf.extend_from_slice(&len.to_le_bytes());
    buf.extend_from_slice(&payload);
    buf
}

/// Decode a binary response from raw bytes (after stripping the magic+length frame).
pub fn decode_binary_response(data: &[u8]) -> Result<BinaryResponse, postcard::Error> {
    postcard::from_bytes::<BinaryResponse>(data)
}

/// Decode a binary request from raw bytes (after stripping the magic+length frame).
pub fn decode_binary_request(data: &[u8]) -> Result<BinaryRequest, postcard::Error> {
    postcard::from_bytes::<BinaryRequest>(data)
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
        let decoded = decode_binary_request(&encoded[5..]).unwrap(); // Skip magic + length
        match decoded {
            BinaryRequest::Agent(r) => assert_eq!(r.query, "test query"),
            _ => panic!("Wrong variant"),
        }
    }
}
