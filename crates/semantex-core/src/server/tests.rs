#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::float_cmp, clippy::module_inception)]
mod tests {
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
            }],
            duration_ms: 31,
            dense_count: 20,
            sparse_count: 20,
            fused_count: 28,
            metrics: None,
        });

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"type\":\"search\""));
        assert!(json.contains("src/main.rs"));
        assert!(json.contains("\"duration_ms\":31"));
        // content is null when None (included for bincode compatibility)
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
        });

        let encoded = encode_binary_request(&req);

        // Verify framing: magic byte + 4-byte length + payload
        assert_eq!(encoded[0], BINARY_MAGIC);
        let len = u32::from_le_bytes(encoded[1..5].try_into().unwrap()) as usize;
        assert_eq!(encoded.len(), 1 + 4 + len);

        // Decode the payload
        let decoded = decode_binary_request(&encoded[5..]).unwrap();
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

        let decoded = decode_binary_request(&encoded[5..]).unwrap();
        assert!(matches!(decoded, BinaryRequest::Health));
    }

    #[test]
    fn test_binary_request_shutdown_roundtrip() {
        let req = BinaryRequest::Shutdown;
        let encoded = encode_binary_request(&req);

        let decoded = decode_binary_request(&encoded[5..]).unwrap();
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
                },
            ],
            duration_ms: 31,
            dense_count: 20,
            sparse_count: 15,
            fused_count: 28,
            metrics: None,
        });

        let encoded = encode_binary_response(&resp);
        assert_eq!(encoded[0], BINARY_MAGIC);

        let decoded = decode_binary_response(&encoded[5..]).unwrap();
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
        let decoded = decode_binary_response(&encoded[5..]).unwrap();
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
        let decoded = decode_binary_response(&encoded[5..]).unwrap();
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
        };

        let json_size = serde_json::to_string(&item).unwrap().len();
        let bincode_size = bincode::serde::encode_to_vec(&item, bincode::config::standard())
            .unwrap()
            .len();

        // bincode should be smaller than JSON
        assert!(
            bincode_size < json_size,
            "bincode ({bincode_size} bytes) should be smaller than JSON ({json_size} bytes)"
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
            }],
            duration_ms: 50,
            dense_count: 10,
            sparse_count: 5,
            fused_count: 12,
            metrics: None,
        });

        let encoded = encode_binary_response(&resp);
        let decoded = decode_binary_response(&encoded[5..]).unwrap();
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
    fn test_binary_tcp_protocol_simulation() {
        // Simulate the full binary protocol flow as it would happen over TCP:
        // Client sends: [0x00][len:4 LE][bincode request]
        // Server reads, processes, responds: [0x00][len:4 LE][bincode response]

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
        });
        let wire_request = encode_binary_request(&client_req);

        // 2. Server reads first byte = 0x00 → binary path
        assert_eq!(wire_request[0], BINARY_MAGIC);

        // 3. Server reads length
        let len = u32::from_le_bytes(wire_request[1..5].try_into().unwrap()) as usize;

        // 4. Server reads and decodes payload
        let decoded_req = decode_binary_request(&wire_request[5..5 + len]).unwrap();
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
        });
        let wire_response = encode_binary_response(&server_resp);

        // 7. Client reads response
        assert_eq!(wire_response[0], BINARY_MAGIC);
        let resp_len = u32::from_le_bytes(wire_response[1..5].try_into().unwrap()) as usize;
        let decoded_resp = decode_binary_response(&wire_response[5..5 + resp_len]).unwrap();
        match decoded_resp {
            BinaryResponse::Search(sr) => {
                assert_eq!(sr.duration_ms, 25);
                assert!(sr.results.is_empty());
            }
            _ => panic!("Expected search response"),
        }
    }
}
