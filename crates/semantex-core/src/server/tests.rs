#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::float_cmp, clippy::module_inception)]
mod tests {
    use crate::server::handler;
    use crate::server::protocol::*;

    #[test]
    fn test_parse_search_request() {
        let json = r#"{"type":"search","query":"error handling","max_results":5}"#;
        let req: Request = serde_json::from_str(json).unwrap();
        match req {
            Request::Search(s) => {
                assert_eq!(s.query, "error handling");
                assert_eq!(s.max_results, 5);
                assert!(s.use_dense);
                assert!(s.use_sparse);
                assert!(!s.use_rerank);
                assert!(s.include_content);
                assert!(!s.snippet);
                assert!(!s.code_only);
            }
            _ => panic!("Expected Search request"),
        }
    }

    #[test]
    fn test_parse_search_request_with_filters() {
        let json = r#"{
            "type": "search",
            "query": "auth middleware",
            "max_results": 10,
            "use_dense": true,
            "use_sparse": false,
            "use_rerank": false,
            "include_types": ["ts", "dart"],
            "exclude_types": ["md"],
            "include_content": false,
            "snippet": false,
            "code_only": true
        }"#;
        let req: Request = serde_json::from_str(json).unwrap();
        match req {
            Request::Search(s) => {
                assert_eq!(s.query, "auth middleware");
                assert_eq!(s.max_results, 10);
                assert!(s.use_dense);
                assert!(!s.use_sparse);
                assert!(!s.use_rerank);
                assert_eq!(s.include_types, vec!["ts", "dart"]);
                assert_eq!(s.exclude_types, vec!["md"]);
                assert!(!s.include_content);
                assert!(s.code_only);
            }
            _ => panic!("Expected Search request"),
        }
    }

    #[test]
    fn test_parse_health_request() {
        let json = r#"{"type":"health"}"#;
        let req: Request = serde_json::from_str(json).unwrap();
        assert!(matches!(req, Request::Health));
    }

    #[test]
    fn test_parse_shutdown_request() {
        let json = r#"{"type":"shutdown"}"#;
        let req: Request = serde_json::from_str(json).unwrap();
        assert!(matches!(req, Request::Shutdown));
    }

    #[test]
    fn test_search_request_defaults() {
        let json = r#"{"type":"search","query":"test"}"#;
        let req: Request = serde_json::from_str(json).unwrap();
        match req {
            Request::Search(s) => {
                assert_eq!(s.max_results, 10);
                assert!(s.use_dense);
                assert!(s.use_sparse);
                assert!(!s.use_rerank);
                assert!(s.include_content);
                assert!(!s.snippet);
                assert!(!s.code_only);
                assert!(s.include_types.is_empty());
                assert!(s.exclude_types.is_empty());
            }
            _ => panic!("Expected Search request"),
        }
    }

    #[test]
    fn test_serialize_search_response() {
        let response = Response::Search(SearchResponse {
            results: vec![SearchResultItem {
                file: "src/main.rs".to_string(),
                start_line: 10,
                end_line: 25,
                score: 0.87,
                source: "Hybrid".to_string(),
                chunk_type: "AstNode".to_string(),
                name: Some("main".to_string()),
                language: Some("rust".to_string()),
                content: None,
                kind: None,
                summary: None,
            }],
            duration_ms: 31,
            dense_count: 20,
            sparse_count: 20,
            fused_count: 28,
            metrics: None,
            confidence: None,
            disambiguation: None,
        });

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"type\":\"search\""));
        assert!(json.contains("src/main.rs"));
        assert!(json.contains("\"duration_ms\":31"));
        // content is null when None (included for postcard compatibility)
        assert!(json.contains("\"content\":null"));
    }

    #[test]
    fn test_serialize_health_response() {
        let response = Response::Health(HealthResponse {
            status: "ok".to_string(),
            uptime_s: 3600,
            searches: 42,
        });

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"status\":\"ok\""));
        assert!(json.contains("\"uptime_s\":3600"));
        assert!(json.contains("\"searches\":42"));
    }

    #[test]
    fn test_serialize_error_response() {
        let response = Response::Error(ErrorResponse {
            message: "Something went wrong".to_string(),
        });

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"type\":\"error\""));
        assert!(json.contains("Something went wrong"));
    }

    #[test]
    fn test_serialize_shutdown_response() {
        let response = Response::Shutdown(ShutdownResponse {
            status: "shutting_down".to_string(),
        });

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"type\":\"shutdown\""));
        assert!(json.contains("shutting_down"));
    }

    #[test]
    fn test_roundtrip_search_result_item() {
        let item = SearchResultItem {
            file: "src/lib.rs".to_string(),
            start_line: 1,
            end_line: 50,
            score: 0.95,
            source: "Dense".to_string(),
            chunk_type: "AstNode".to_string(),
            name: Some("SemantexServer".to_string()),
            language: Some("rust".to_string()),
            content: Some("pub struct SemantexServer {}".to_string()),
            kind: None,
            summary: None,
        };

        let json = serde_json::to_string(&item).unwrap();
        let deserialized: SearchResultItem = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.file, "src/lib.rs");
        assert_eq!(deserialized.start_line, 1);
        assert_eq!(deserialized.end_line, 50);
        assert_eq!(deserialized.score, 0.95);
        assert_eq!(deserialized.name.as_deref(), Some("SemantexServer"));
        assert_eq!(
            deserialized.content.as_deref(),
            Some("pub struct SemantexServer {}")
        );
    }

    #[test]
    fn test_invalid_request_type() {
        let json = r#"{"type":"unknown_command"}"#;
        let result = serde_json::from_str::<Request>(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_tcp_protocol_roundtrip() {
        // Simulate what happens over the wire: request → JSON line → parse → handle → response → JSON line
        let request = serde_json::json!({
            "type": "search",
            "query": "error handling",
            "max_results": 5,
            "use_dense": true,
            "use_sparse": true,
            "use_rerank": false,
            "include_types": ["rs"],
            "include_content": false
        });

        let request_line = format!("{request}\n");
        let parsed: Request = serde_json::from_str(request_line.trim()).unwrap();
        assert!(matches!(parsed, Request::Search(_)));

        // Simulate response
        let response = Response::Search(SearchResponse {
            results: vec![],
            duration_ms: 15,
            dense_count: 0,
            sparse_count: 0,
            fused_count: 0,
            metrics: None,
            confidence: None,
            disambiguation: None,
        });
        let response_line = format!("{}\n", serde_json::to_string(&response).unwrap());

        // Verify response parses correctly
        let parsed_response: serde_json::Value =
            serde_json::from_str(response_line.trim()).unwrap();
        assert_eq!(parsed_response["type"], "search");
        assert_eq!(parsed_response["duration_ms"], 15);
    }

    // --- Binary protocol tests ---

    #[test]
    fn test_binary_request_search_roundtrip() {
        let req = BinaryRequest::Search(SearchRequest {
            query: "error handling in authentication".to_string(),
            max_results: 10,
            use_dense: true,
            use_sparse: true,
            use_rerank: false,
            include_types: vec!["rs".to_string(), "py".to_string()],
            exclude_types: vec!["md".to_string()],
            code_only: true,
            include_content: true,
            snippet: false,
            grep_mode: false,
            regex_pattern: None,
            auto_peek_top: false,
        });

        let encoded = encode_binary_request(&req);

        // Verify framing: magic byte + 4-byte length + body. v0.4.1 W-Index #2
        // moves the version byte INSIDE the length-prefixed body, so the header
        // layout stays `[MAGIC][len:4]` (5 bytes) and the body begins with
        // `[VERSION][postcard]`.
        assert_eq!(encoded[0], BINARY_MAGIC);
        let len = u32::from_le_bytes(encoded[1..5].try_into().unwrap()) as usize;
        assert_eq!(encoded.len(), BINARY_FRAME_HEADER_LEN + len);
        assert_eq!(encoded[BINARY_FRAME_HEADER_LEN], BINARY_PROTOCOL_VERSION);

        // Decode the payload
        let decoded = decode_binary_request(&encoded[BINARY_FRAME_HEADER_LEN..]).unwrap();
        match decoded {
            BinaryRequest::Search(s) => {
                assert_eq!(s.query, "error handling in authentication");
                assert_eq!(s.max_results, 10);
                assert!(s.use_dense);
                assert!(s.use_sparse);
                assert!(!s.use_rerank);
                assert_eq!(s.include_types, vec!["rs", "py"]);
                assert_eq!(s.exclude_types, vec!["md"]);
                assert!(s.code_only);
                assert!(s.include_content);
                assert!(!s.snippet);
            }
            _ => panic!("Expected BinaryRequest::Search"),
        }
    }

    #[test]
    fn test_binary_request_health_roundtrip() {
        let req = BinaryRequest::Health;
        let encoded = encode_binary_request(&req);
        assert_eq!(encoded[0], BINARY_MAGIC);

        let decoded = decode_binary_request(&encoded[BINARY_FRAME_HEADER_LEN..]).unwrap();
        assert!(matches!(decoded, BinaryRequest::Health));
    }

    #[test]
    fn test_binary_request_shutdown_roundtrip() {
        let req = BinaryRequest::Shutdown;
        let encoded = encode_binary_request(&req);

        let decoded = decode_binary_request(&encoded[BINARY_FRAME_HEADER_LEN..]).unwrap();
        assert!(matches!(decoded, BinaryRequest::Shutdown));
    }

    #[test]
    fn test_binary_response_search_roundtrip() {
        let resp = BinaryResponse::Search(SearchResponse {
            results: vec![
                SearchResultItem {
                    file: "src/main.rs".to_string(),
                    start_line: 10,
                    end_line: 25,
                    score: 0.87,
                    source: "Hybrid".to_string(),
                    chunk_type: "AstNode".to_string(),
                    name: Some("main".to_string()),
                    language: Some("rust".to_string()),
                    content: Some("fn main() {\n    println!(\"hello\");\n}".to_string()),
                    kind: None,
                    summary: None,
                },
                SearchResultItem {
                    file: "src/lib.rs".to_string(),
                    start_line: 1,
                    end_line: 5,
                    score: 0.72,
                    source: "Dense".to_string(),
                    chunk_type: "TextWindow".to_string(),
                    name: None,
                    language: None,
                    content: None,
                    kind: None,
                    summary: None,
                },
            ],
            duration_ms: 31,
            dense_count: 20,
            sparse_count: 15,
            fused_count: 28,
            metrics: None,
            confidence: None,
            disambiguation: None,
        });

        let encoded = encode_binary_response(&resp);
        assert_eq!(encoded[0], BINARY_MAGIC);

        let decoded = decode_binary_response(&encoded[BINARY_FRAME_HEADER_LEN..]).unwrap();
        match decoded {
            BinaryResponse::Search(sr) => {
                assert_eq!(sr.results.len(), 2);
                assert_eq!(sr.results[0].file, "src/main.rs");
                assert_eq!(sr.results[0].score, 0.87);
                assert_eq!(sr.results[0].name.as_deref(), Some("main"));
                assert!(sr.results[0].content.is_some());
                assert_eq!(sr.results[1].file, "src/lib.rs");
                assert!(sr.results[1].name.is_none());
                assert!(sr.results[1].content.is_none());
                assert_eq!(sr.duration_ms, 31);
                assert_eq!(sr.dense_count, 20);
                assert_eq!(sr.sparse_count, 15);
                assert_eq!(sr.fused_count, 28);
            }
            _ => panic!("Expected BinaryResponse::Search"),
        }
    }

    #[test]
    fn test_binary_response_health_roundtrip() {
        let resp = BinaryResponse::Health(HealthResponse {
            status: "ok".to_string(),
            uptime_s: 3600,
            searches: 42,
        });

        let encoded = encode_binary_response(&resp);
        let decoded = decode_binary_response(&encoded[BINARY_FRAME_HEADER_LEN..]).unwrap();
        match decoded {
            BinaryResponse::Health(h) => {
                assert_eq!(h.status, "ok");
                assert_eq!(h.uptime_s, 3600);
                assert_eq!(h.searches, 42);
            }
            _ => panic!("Expected BinaryResponse::Health"),
        }
    }

    #[test]
    fn test_binary_response_error_roundtrip() {
        let resp = BinaryResponse::Error(ErrorResponse {
            message: "Index not found".to_string(),
        });

        let encoded = encode_binary_response(&resp);
        let decoded = decode_binary_response(&encoded[BINARY_FRAME_HEADER_LEN..]).unwrap();
        match decoded {
            BinaryResponse::Error(e) => {
                assert_eq!(e.message, "Index not found");
            }
            _ => panic!("Expected BinaryResponse::Error"),
        }
    }

    #[test]
    fn test_binary_magic_distinguishes_from_json() {
        // JSON always starts with '{' (0x7b)
        let json_bytes = b"{\"type\":\"health\"}";
        assert_ne!(json_bytes[0], BINARY_MAGIC);
        assert_eq!(BINARY_MAGIC, 0x00);

        // Binary always starts with 0x00
        let binary = encode_binary_request(&BinaryRequest::Health);
        assert_eq!(binary[0], BINARY_MAGIC);
    }

    #[test]
    fn test_search_result_item_kind_summary() {
        let item = SearchResultItem {
            file: "src/main.rs".to_string(),
            start_line: 1,
            end_line: 10,
            score: 0.9,
            source: "Dense".to_string(),
            chunk_type: "AstNode".to_string(),
            name: Some("my_fn".to_string()),
            language: Some("rust".to_string()),
            content: None,
            kind: Some("fn".to_string()),
            summary: Some("Does something useful".to_string()),
        };

        let json = serde_json::to_string(&item).unwrap();
        assert!(json.contains("\"kind\":\"fn\""));
        assert!(json.contains("\"summary\":\"Does something useful\""));

        let decoded: SearchResultItem = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.kind.as_deref(), Some("fn"));
        assert_eq!(decoded.summary.as_deref(), Some("Does something useful"));
    }

    #[test]
    fn test_search_result_item_kind_none_backward_compat() {
        // Old daemon JSON without kind/summary should deserialize with None
        let old_json = r#"{
            "file": "src/lib.rs",
            "start_line": 1,
            "end_line": 5,
            "score": 0.8,
            "source": "Sparse",
            "chunk_type": "TextWindow",
            "name": null,
            "language": null,
            "content": null
        }"#;
        let item: SearchResultItem = serde_json::from_str(old_json).unwrap();
        assert!(item.kind.is_none());
        assert!(item.summary.is_none());
    }

    #[test]
    fn test_graph_walk_request_roundtrip() {
        let req = BinaryRequest::GraphWalk(GraphWalkRequest {
            symbol: "my_function".to_string(),
        });
        let encoded = encode_binary_request(&req);
        let decoded = decode_binary_request(&encoded[BINARY_FRAME_HEADER_LEN..]).unwrap();
        match decoded {
            BinaryRequest::GraphWalk(g) => assert_eq!(g.symbol, "my_function"),
            _ => panic!("Expected GraphWalk"),
        }
    }

    #[test]
    fn test_graph_walk_response_roundtrip() {
        let resp = BinaryResponse::GraphWalk(GraphWalkResponse {
            target: vec![SearchResultItem {
                file: "src/storage.rs".to_string(),
                start_line: 10,
                end_line: 30,
                score: 0.0,
                source: "GraphWalk".to_string(),
                chunk_type: "AstNode".to_string(),
                name: Some("get_edges".to_string()),
                language: Some("rust".to_string()),
                content: None,
                kind: Some("fn".to_string()),
                summary: Some("Get call edges".to_string()),
            }],
            callers: vec![],
            callees: vec![],
            type_refs: vec![],
            hierarchy: vec![],
        });
        let encoded = encode_binary_response(&resp);
        let decoded = decode_binary_response(&encoded[BINARY_FRAME_HEADER_LEN..]).unwrap();
        match decoded {
            BinaryResponse::GraphWalk(g) => {
                assert_eq!(g.target.len(), 1);
                assert_eq!(g.target[0].name.as_deref(), Some("get_edges"));
                assert_eq!(g.target[0].kind.as_deref(), Some("fn"));
            }
            _ => panic!("Expected GraphWalk response"),
        }
    }

    #[test]
    fn test_binary_request_to_request_conversion() {
        let bin_req = BinaryRequest::Search(SearchRequest {
            query: "test".to_string(),
            max_results: 5,
            use_dense: true,
            use_sparse: true,
            use_rerank: false,
            include_types: vec![],
            exclude_types: vec![],
            code_only: false,
            include_content: true,
            snippet: false,
            grep_mode: false,
            regex_pattern: None,
            auto_peek_top: false,
        });

        let req: Request = bin_req.into();
        assert!(matches!(req, Request::Search(_)));

        let bin_health = BinaryRequest::Health;
        let req: Request = bin_health.into();
        assert!(matches!(req, Request::Health));

        let bin_shutdown = BinaryRequest::Shutdown;
        let req: Request = bin_shutdown.into();
        assert!(matches!(req, Request::Shutdown));
    }

    #[test]
    fn test_response_to_binary_response_conversion() {
        let resp = Response::Search(SearchResponse {
            results: vec![],
            duration_ms: 10,
            dense_count: 0,
            sparse_count: 0,
            fused_count: 0,
            metrics: None,
            confidence: None,
            disambiguation: None,
        });
        let bin: BinaryResponse = resp.into();
        assert!(matches!(bin, BinaryResponse::Search(_)));

        let resp = Response::Health(HealthResponse {
            status: "ok".to_string(),
            uptime_s: 0,
            searches: 0,
        });
        let bin: BinaryResponse = resp.into();
        assert!(matches!(bin, BinaryResponse::Health(_)));

        let resp = Response::Error(ErrorResponse {
            message: "err".to_string(),
        });
        let bin: BinaryResponse = resp.into();
        assert!(matches!(bin, BinaryResponse::Error(_)));
    }

    #[test]
    fn test_binary_serialization_size() {
        // Binary should be significantly smaller than JSON for the same payload
        let item = SearchResultItem {
            file: "src/server/handler.rs".to_string(),
            start_line: 10,
            end_line: 25,
            score: 0.87,
            source: "Hybrid".to_string(),
            chunk_type: "AstNode".to_string(),
            name: Some("handle_search".to_string()),
            language: Some("rust".to_string()),
            content: Some(
                "fn handle_search(&self, req: SearchRequest) -> Response { ... }".to_string(),
            ),
            kind: None,
            summary: None,
        };

        let json_size = serde_json::to_string(&item).unwrap().len();
        let postcard_size = postcard::to_stdvec(&item).unwrap().len();

        // postcard should be smaller than JSON
        assert!(
            postcard_size < json_size,
            "postcard ({postcard_size} bytes) should be smaller than JSON ({json_size} bytes)"
        );
    }

    #[test]
    fn test_binary_protocol_with_large_content() {
        // Test with a realistic large content payload
        let large_content = "fn ".to_string() + &"a".repeat(10_000);
        let resp = BinaryResponse::Search(SearchResponse {
            results: vec![SearchResultItem {
                file: "src/big_file.rs".to_string(),
                start_line: 1,
                end_line: 500,
                score: 0.95,
                source: "Dense".to_string(),
                chunk_type: "AstNode".to_string(),
                name: Some("large_function".to_string()),
                language: Some("rust".to_string()),
                content: Some(large_content.clone()),
                kind: None,
                summary: None,
            }],
            duration_ms: 50,
            dense_count: 10,
            sparse_count: 5,
            fused_count: 12,
            metrics: None,
            confidence: None,
            disambiguation: None,
        });

        let encoded = encode_binary_response(&resp);
        let decoded = decode_binary_response(&encoded[BINARY_FRAME_HEADER_LEN..]).unwrap();
        match decoded {
            BinaryResponse::Search(sr) => {
                assert_eq!(
                    sr.results[0].content.as_deref(),
                    Some(large_content.as_str())
                );
            }
            _ => panic!("Expected BinaryResponse::Search"),
        }
    }

    #[test]
    fn test_auto_peek_top_field_roundtrip() {
        // auto_peek_top=true roundtrips through binary protocol
        let req = BinaryRequest::Search(SearchRequest {
            query: "test".to_string(),
            max_results: 5,
            use_dense: true,
            use_sparse: true,
            use_rerank: false,
            include_types: vec![],
            exclude_types: vec![],
            code_only: false,
            include_content: false,
            snippet: false,
            grep_mode: false,
            regex_pattern: None,
            auto_peek_top: true,
        });
        let encoded = encode_binary_request(&req);
        let decoded = decode_binary_request(&encoded[BINARY_FRAME_HEADER_LEN..]).unwrap();
        match decoded {
            BinaryRequest::Search(s) => assert!(s.auto_peek_top),
            _ => panic!("Expected Search"),
        }
    }

    #[test]
    fn test_confidence_field_roundtrip() {
        let resp = BinaryResponse::Search(SearchResponse {
            results: vec![],
            duration_ms: 10,
            dense_count: 0,
            sparse_count: 0,
            fused_count: 0,
            metrics: None,
            confidence: Some("high".to_string()),
            disambiguation: None,
        });
        let encoded = encode_binary_response(&resp);
        let decoded = decode_binary_response(&encoded[BINARY_FRAME_HEADER_LEN..]).unwrap();
        match decoded {
            BinaryResponse::Search(sr) => {
                assert_eq!(sr.confidence.as_deref(), Some("high"));
            }
            _ => panic!("Expected Search response"),
        }
    }

    #[test]
    fn test_binary_multi_search_roundtrip() {
        let make_req = |q: &str| SearchRequest {
            query: q.to_string(),
            max_results: 5,
            use_dense: true,
            use_sparse: true,
            use_rerank: false,
            include_types: vec![],
            exclude_types: vec![],
            code_only: false,
            include_content: false,
            snippet: false,
            grep_mode: false,
            regex_pattern: None,
            auto_peek_top: true,
        };

        let req = BinaryRequest::MultiSearch(MultiSearchRequest {
            queries: vec![
                make_req("query one"),
                make_req("query two"),
                make_req("query three"),
            ],
        });

        let encoded = encode_binary_request(&req);
        assert_eq!(encoded[0], BINARY_MAGIC);

        let decoded = decode_binary_request(&encoded[BINARY_FRAME_HEADER_LEN..]).unwrap();
        match decoded {
            BinaryRequest::MultiSearch(mr) => {
                assert_eq!(mr.queries.len(), 3);
                assert_eq!(mr.queries[0].query, "query one");
                assert_eq!(mr.queries[2].query, "query three");
                assert!(mr.queries[0].auto_peek_top);
            }
            _ => panic!("Expected MultiSearch"),
        }
    }

    #[test]
    fn test_binary_multi_search_response_roundtrip() {
        let make_resp = |conf: &str| SearchResponse {
            results: vec![],
            duration_ms: 5,
            dense_count: 0,
            sparse_count: 0,
            fused_count: 0,
            metrics: None,
            confidence: Some(conf.to_string()),
            disambiguation: None,
        };

        let resp = BinaryResponse::MultiSearch(MultiSearchResponse {
            responses: vec![make_resp("high"), make_resp("low")],
        });

        let encoded = encode_binary_response(&resp);
        let decoded = decode_binary_response(&encoded[BINARY_FRAME_HEADER_LEN..]).unwrap();
        match decoded {
            BinaryResponse::MultiSearch(mr) => {
                assert_eq!(mr.responses.len(), 2);
                assert_eq!(mr.responses[0].confidence.as_deref(), Some("high"));
                assert_eq!(mr.responses[1].confidence.as_deref(), Some("low"));
            }
            _ => panic!("Expected MultiSearch response"),
        }
    }

    #[test]
    fn test_search_request_auto_peek_top_default() {
        // Old JSON without auto_peek_top deserializes with default false
        let json = r#"{"type":"search","query":"test"}"#;
        let req: Request = serde_json::from_str(json).unwrap();
        match req {
            Request::Search(s) => assert!(!s.auto_peek_top),
            _ => panic!("Expected Search"),
        }
    }

    #[test]
    fn test_search_response_confidence_default() {
        // Old JSON without confidence deserializes with None
        let json = r#"{"type":"search","results":[],"duration_ms":0,"dense_count":0,"sparse_count":0,"fused_count":0}"#;
        let resp: Response = serde_json::from_str(json).unwrap();
        match resp {
            Response::Search(s) => assert!(s.confidence.is_none()),
            _ => panic!("Expected Search"),
        }
    }

    #[test]
    fn test_binary_tcp_protocol_simulation() {
        // Simulate the full binary protocol flow as it would happen over TCP:
        // Client sends: [0x00][len:4 LE][postcard request]
        // Server reads, processes, responds: [0x00][len:4 LE][postcard response]

        // 1. Client encodes request
        let client_req = BinaryRequest::Search(SearchRequest {
            query: "authentication flow".to_string(),
            max_results: 5,
            use_dense: true,
            use_sparse: true,
            use_rerank: false,
            include_types: vec![],
            exclude_types: vec![],
            code_only: false,
            include_content: true,
            snippet: false,
            grep_mode: false,
            regex_pattern: None,
            auto_peek_top: false,
        });
        let wire_request = encode_binary_request(&client_req);

        // 2. Server reads first byte = 0x00 → binary path
        assert_eq!(wire_request[0], BINARY_MAGIC);

        // 3. Server reads length (v0.4.1 W-Index #2: header stays
        // `[MAGIC][len:4]`; version byte moved into the length-prefixed body).
        let len = u32::from_le_bytes(wire_request[1..5].try_into().unwrap()) as usize;

        // 4. Server reads and decodes body. `decode_binary_request` verifies
        // the leading version byte against `BINARY_PROTOCOL_VERSION`.
        assert_eq!(
            wire_request[BINARY_FRAME_HEADER_LEN],
            BINARY_PROTOCOL_VERSION
        );
        let decoded_req = decode_binary_request(
            &wire_request[BINARY_FRAME_HEADER_LEN..BINARY_FRAME_HEADER_LEN + len],
        )
        .unwrap();
        match &decoded_req {
            BinaryRequest::Search(s) => assert_eq!(s.query, "authentication flow"),
            _ => panic!("Expected search"),
        }

        // 5. Server converts to Request and processes
        let _request: Request = decoded_req.into();

        // 6. Server encodes response
        let server_resp = BinaryResponse::Search(SearchResponse {
            results: vec![],
            duration_ms: 25,
            dense_count: 5,
            sparse_count: 5,
            fused_count: 8,
            metrics: None,
            confidence: None,
            disambiguation: None,
        });
        let wire_response = encode_binary_response(&server_resp);

        // 7. Client reads response — same header layout as the request.
        assert_eq!(wire_response[0], BINARY_MAGIC);
        let resp_len = u32::from_le_bytes(wire_response[1..5].try_into().unwrap()) as usize;
        assert_eq!(
            wire_response[BINARY_FRAME_HEADER_LEN],
            BINARY_PROTOCOL_VERSION
        );
        let decoded_resp = decode_binary_response(
            &wire_response[BINARY_FRAME_HEADER_LEN..BINARY_FRAME_HEADER_LEN + resp_len],
        )
        .unwrap();
        match decoded_resp {
            BinaryResponse::Search(sr) => {
                assert_eq!(sr.duration_ms, 25);
                assert!(sr.results.is_empty());
            }
            _ => panic!("Expected search response"),
        }
    }

    #[test]
    fn test_deep_search_request_roundtrip() {
        let req = BinaryRequest::DeepSearch(DeepSearchRequest {
            query: "how does search work".to_string(),
            max_results: 20,
            use_graph: true,
        });
        let encoded = encode_binary_request(&req);
        let decoded = decode_binary_request(&encoded[BINARY_FRAME_HEADER_LEN..]).unwrap();
        match decoded {
            BinaryRequest::DeepSearch(d) => {
                assert_eq!(d.query, "how does search work");
                assert_eq!(d.max_results, 20);
                assert!(d.use_graph);
            }
            _ => panic!("Expected DeepSearch request"),
        }
    }

    #[test]
    fn test_deep_search_response_roundtrip() {
        let resp = BinaryResponse::DeepSearch(DeepSearchResponse {
            answer: "The search is implemented in hybrid.rs.".to_string(),
            sources: vec![DeepSearchSource {
                file: "src/search/hybrid.rs".to_string(),
                start_line: 10,
                end_line: 50,
                name: Some("search".to_string()),
                kind: Some("fn".to_string()),
            }],
            metrics: DeepResponseMetrics {
                search_ms: 12,
                triage_ms: 1,
                graph_ms: 3,
                read_ms: 2,
                summarize_ms: 5,
                total_ms: 23,
                chunks_searched: 20,
                chunks_read: 6,
                confidence_zone: "high".to_string(),
            },
            confidence: 0.85,
        });
        let encoded = encode_binary_response(&resp);
        let decoded = decode_binary_response(&encoded[BINARY_FRAME_HEADER_LEN..]).unwrap();
        match decoded {
            BinaryResponse::DeepSearch(d) => {
                assert_eq!(d.sources.len(), 1);
                assert_eq!(d.sources[0].file, "src/search/hybrid.rs");
                assert_eq!(d.metrics.search_ms, 12);
                assert_eq!(d.metrics.confidence_zone, "high");
                assert!((d.confidence - 0.85).abs() < f32::EPSILON);
            }
            _ => panic!("Expected DeepSearch response"),
        }
    }

    // --- display_summary integration tests ---

    #[test]
    fn test_chunk_to_item_uses_display_summary() {
        use crate::chunking::structured_meta::{SemanticRole, StructuredChunkMeta};
        use crate::types::{AstNodeKind, Chunk, ChunkType};
        use std::path::PathBuf;

        let mut meta = StructuredChunkMeta {
            name: Some("handle_search".to_string()),
            signature: Some("fn handle_search(&self, req: SearchRequest) -> Response".to_string()),
            calls: vec![
                "self.searcher.search".to_string(),
                "push".to_string(), // trivial
            ],
            called_by: vec!["handle".to_string()],
            semantic_role: Some(SemanticRole::Handler),
            kind: Some("fn".to_string()),
            ..Default::default()
        };
        meta.generate_nl_summary(); // BM25 summary still generated for indexing

        let chunk = Chunk {
            id: 0,
            file_path: PathBuf::from("src/handler.rs"),
            start_line: 42,
            end_line: 78,
            content: "fn handle_search(...) { ... }".to_string(),
            chunk_type: ChunkType::AstNode {
                name: "handle_search".to_string(),
                kind: AstNodeKind::Function,
                language: "rust".to_string(),
                structured_meta: Some(Box::new(meta)),
            },
        };

        let item = handler::chunk_to_item(&chunk);
        let summary = item.summary.expect("summary should be present");

        // Should contain structured display format, not NL prose
        assert!(
            summary.contains("fn handle_search(&self, req: SearchRequest) -> Response"),
            "signature missing: {summary}"
        );
        // Should NOT contain expanded identifiers from NL summary
        assert!(
            !summary.contains("handle search"),
            "NL expansion should not appear: {summary}"
        );
        // Trivial calls should be filtered
        assert!(!summary.contains("push"), "trivial call leaked: {summary}");
        // Non-trivial calls should be present
        assert!(
            summary.contains("self.searcher.search"),
            "non-trivial call missing: {summary}"
        );
        // called_by should appear
        assert!(summary.contains("called_by: handle"), "{summary}");
        // role tag should appear
        assert!(summary.contains("[handler]"), "{summary}");
    }

    #[test]
    fn test_chunk_to_item_text_window_no_summary() {
        use crate::types::{Chunk, ChunkType};
        use std::path::PathBuf;

        let chunk = Chunk {
            id: 0,
            file_path: PathBuf::from("README.md"),
            start_line: 1,
            end_line: 20,
            content: "# Hello\nSome text".to_string(),
            chunk_type: ChunkType::TextWindow { window_index: 0 },
        };

        let item = handler::chunk_to_item(&chunk);
        assert!(item.summary.is_none(), "TextWindow should have no summary");
    }

    // --- daemon panic-isolation integration test ---

    /// End-to-end proof that a panic while handling ONE connection is caught and
    /// isolated by the accept loop, so the daemon keeps serving every other
    /// caller instead of dying and taking the in-RAM model down with it.
    ///
    /// Flow:
    ///   1. Spawn a real `Listener` (sparse-only, empty index) on a background
    ///      thread, exactly as the daemon does.
    ///   2. Send a Search whose query is `TEST_PANIC_SENTINEL` — the handler
    ///      panics. The CLIENT sees an error (connection dropped mid-response).
    ///   3. Send a NORMAL Health request and assert it SUCCEEDS. This is the
    ///      load-bearing assertion: pre-fix the accept-loop thread is dead and
    ///      this would time out / refuse; post-fix the daemon is alive.
    #[test]
    fn daemon_survives_handler_panic() {
        use crate::config::SemantexConfig;
        use crate::index::storage::ChunkStore;
        use crate::search::hybrid::HybridSearcher;
        use crate::server::listener::Listener;
        use std::sync::Arc;
        use std::sync::atomic::AtomicBool;
        use std::time::Duration;
        use tempfile::TempDir;

        // 1. Build a minimal sparse-only searcher over an empty index, mirroring
        //    handler.rs::build_empty_searcher (the established test harness).
        let dir = TempDir::new().expect("tempdir");
        let semantex_dir = dir.path().join(".semantex");
        std::fs::create_dir_all(&semantex_dir).unwrap();
        {
            let _store =
                ChunkStore::open(&semantex_dir.join("chunks.db")).expect("create empty store");
        }
        let config = SemantexConfig::default();
        let searcher =
            HybridSearcher::open_sparse_only(&semantex_dir, &config).expect("open searcher");

        let port_file = semantex_dir.join("semantex.port");
        let shutdown = Arc::new(AtomicBool::new(false));
        let listener = Listener::bind(
            &port_file,
            searcher,
            dir.path().to_path_buf(),
            Duration::from_secs(30),
            shutdown.clone(),
        )
        .expect("bind listener");
        let port = listener.port();

        // 2. Run the accept loop on a background thread (as the daemon does).
        let server_thread = std::thread::spawn(move || {
            let _ = listener.run();
        });

        // 3. Fire the panic-inducing request. The handler panics; the client's
        //    read fails because the connection drops mid-response. We only
        //    require that the panic-isolation logic does NOT propagate — i.e.
        //    the loop survives.
        let panic_req = BinaryRequest::Search(SearchRequest {
            query: super::super::handler::TEST_PANIC_SENTINEL.to_string(),
            max_results: 5,
            use_dense: false,
            use_sparse: true,
            use_rerank: false,
            include_types: vec![],
            exclude_types: vec![],
            code_only: false,
            include_content: false,
            snippet: false,
            grep_mode: false,
            regex_pattern: None,
            auto_peek_top: false,
        });
        let panic_result = send_binary_for_test(port, &panic_req, 3);
        assert!(
            panic_result.is_err(),
            "panic-sentinel request should fail at the client (connection dropped); got {panic_result:?}"
        );

        // 4. LOAD-BEARING: a normal request must still succeed, proving the
        //    accept loop survived the panic and the daemon keeps serving.
        let health_req = BinaryRequest::Health;
        let health_result = send_binary_for_test(port, &health_req, 5);
        assert!(
            health_result.is_ok(),
            "daemon must keep serving after an isolated handler panic; \
             health request failed: {health_result:?} — the accept loop likely died"
        );
        match health_result.unwrap() {
            BinaryResponse::Health(h) => assert_eq!(h.status, "ok"),
            other => panic!("expected Health response, got {other:?}"),
        }

        // Clean shutdown of the background daemon thread.
        shutdown.store(true, std::sync::atomic::Ordering::Relaxed);
        let _ = send_binary_for_test(port, &BinaryRequest::Shutdown, 2);
        let _ = server_thread.join();
    }

    /// F2 + concurrency (real socket, multi-threaded): several clients firing
    /// searches at the daemon simultaneously must each get a well-formed
    /// response (each connection runs on its own handler thread), and a
    /// graceful `Shutdown` request must receive its "shutting_down" response
    /// BEFORE the daemon tears down — the accept loop drains in-flight
    /// connection threads with a bounded grace period instead of exiting the
    /// moment the shutdown flag is observed (which raced the response write).
    #[test]
    fn daemon_serves_concurrent_clients_and_shutdown_response_lands() {
        use crate::config::SemantexConfig;
        use crate::index::storage::ChunkStore;
        use crate::search::hybrid::HybridSearcher;
        use crate::server::listener::Listener;
        use std::sync::Arc;
        use std::sync::atomic::AtomicBool;
        use std::time::Duration;
        use tempfile::TempDir;

        // Same minimal sparse-only harness as `daemon_survives_handler_panic`.
        let dir = TempDir::new().expect("tempdir");
        let semantex_dir = dir.path().join(".semantex");
        std::fs::create_dir_all(&semantex_dir).unwrap();
        {
            let _store =
                ChunkStore::open(&semantex_dir.join("chunks.db")).expect("create empty store");
        }
        let config = SemantexConfig::default();
        let searcher =
            HybridSearcher::open_sparse_only(&semantex_dir, &config).expect("open searcher");

        let port_file = semantex_dir.join("semantex.port");
        let shutdown = Arc::new(AtomicBool::new(false));
        let listener = Listener::bind(
            &port_file,
            searcher,
            dir.path().to_path_buf(),
            Duration::from_secs(30),
            shutdown.clone(),
        )
        .expect("bind listener");
        let port = listener.port();

        let server_thread = std::thread::spawn(move || {
            let _ = listener.run();
        });

        // 1. Concurrent clients: 4 threads each firing a search — every one
        //    must receive a Search response (empty index → empty results).
        let make_search = || {
            BinaryRequest::Search(SearchRequest {
                query: "concurrent daemon clients".to_string(),
                max_results: 5,
                use_dense: false,
                use_sparse: true,
                use_rerank: false,
                include_types: vec![],
                exclude_types: vec![],
                code_only: false,
                include_content: false,
                snippet: false,
                grep_mode: false,
                regex_pattern: None,
                auto_peek_top: false,
            })
        };
        let clients: Vec<_> = (0..4)
            .map(|_| {
                let req = make_search();
                std::thread::spawn(move || send_binary_for_test(port, &req, 10))
            })
            .collect();
        for client in clients {
            let resp = client
                .join()
                .unwrap()
                .expect("concurrent search must succeed");
            assert!(
                matches!(resp, BinaryResponse::Search(_)),
                "expected Search response, got {resp:?}"
            );
        }

        // 2. F2 load-bearing assertion: the Shutdown caller must receive the
        //    "shutting_down" response — pre-drain, run() could return (and in
        //    the real daemon, the process could exit) between the flag being
        //    set and the response bytes being written.
        let resp = send_binary_for_test(port, &BinaryRequest::Shutdown, 5)
            .expect("Shutdown response must be written before the daemon exits");
        assert!(
            matches!(resp, BinaryResponse::Shutdown(_)),
            "expected Shutdown response, got {resp:?}"
        );

        server_thread.join().unwrap();
    }

    /// Minimal binary-protocol client for the integration test: connect, send a
    /// framed request, read the framed response. Mirrors
    /// `server::mod::send_binary_request` but lives here to keep the test
    /// self-contained.
    fn send_binary_for_test(
        port: u16,
        req: &BinaryRequest,
        read_timeout_s: u64,
    ) -> anyhow::Result<BinaryResponse> {
        use std::io::{Read, Write};
        use std::net::TcpStream;
        use std::time::Duration;

        let frame = encode_binary_request(req);
        let addr = std::net::SocketAddr::from(([127, 0, 0, 1], port));
        let mut stream = TcpStream::connect_timeout(&addr, Duration::from_secs(5))?;
        stream.set_read_timeout(Some(Duration::from_secs(read_timeout_s)))?;
        stream.set_write_timeout(Some(Duration::from_secs(5)))?;

        stream.write_all(&frame)?;
        stream.flush()?;

        let mut magic = [0u8; 1];
        stream.read_exact(&mut magic)?;
        if magic[0] != BINARY_MAGIC {
            anyhow::bail!("expected binary response, got 0x{:02x}", magic[0]);
        }
        let mut len_buf = [0u8; 4];
        stream.read_exact(&mut len_buf)?;
        let len = u32::from_le_bytes(len_buf) as usize;
        let mut body = vec![0u8; len];
        stream.read_exact(&mut body)?;
        decode_binary_response(&body).map_err(|e| anyhow::anyhow!("decode failed: {e}"))
    }

    #[test]
    fn test_chunk_to_item_minimal_astnode_no_meta() {
        use crate::types::{AstNodeKind, Chunk, ChunkType};
        use std::path::PathBuf;

        let chunk = Chunk {
            id: 0,
            file_path: PathBuf::from("src/lib.rs"),
            start_line: 1,
            end_line: 5,
            content: "fn foo() {}".to_string(),
            chunk_type: ChunkType::AstNode {
                name: "foo".to_string(),
                kind: AstNodeKind::Function,
                language: "rust".to_string(),
                structured_meta: None,
            },
        };

        let item = handler::chunk_to_item(&chunk);
        // No structured_meta → no summary
        assert!(item.summary.is_none(), "no meta should yield no summary");
    }
}
