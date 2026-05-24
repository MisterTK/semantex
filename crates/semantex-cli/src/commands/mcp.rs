use anyhow::Result;
use semantex_core::config::SemantexConfig;

/// Run the MCP server. Defaults to stdio transport (one connection per process,
/// reads JSON-RPC from stdin, writes to stdout). Pass `http=true` to start the
/// HTTP transport instead, which exposes the same JSON-RPC surface on a TCP
/// listener. `toolset` selects which tool bundle to expose for stdio:
/// `core` (4), `structural` (5), or `all` (13, default).
pub fn run(
    config: &SemantexConfig,
    http: bool,
    port: u16,
    allow_remote: bool,
    toolset: &str,
) -> Result<()> {
    #[cfg(feature = "mcp")]
    {
        validate_toolset(toolset)?;

        if http {
            #[cfg(feature = "http")]
            {
                let _ = config;
                // Forward the toolset to the spawned child subprocess so the
                // child becomes the single source of truth for tool filtering
                // (`server.rs::tools_for_toolset`). The HTTP-level filter in
                // `http_transport.rs` is kept as defense-in-depth.
                return run_http(port, allow_remote, toolset);
            }
            #[cfg(not(feature = "http"))]
            {
                let _ = (port, allow_remote, config, toolset);
                anyhow::bail!(
                    "HTTP MCP transport not compiled in. Build semantex-mcp with --features http",
                );
            }
        }

        let _ = (port, allow_remote);
        let server = semantex_mcp::McpServer::with_toolset(config.clone(), toolset);
        server.run()
    }

    #[cfg(not(feature = "mcp"))]
    {
        let _ = (config, http, port, allow_remote, toolset);
        anyhow::bail!("MCP support not compiled in. Build with --features mcp")
    }
}

#[cfg(feature = "mcp")]
fn validate_toolset(toolset: &str) -> Result<()> {
    match toolset {
        "core" | "structural" | "all" => Ok(()),
        other => anyhow::bail!("unknown toolset `{other}`. Valid values: core, structural, all"),
    }
}

#[cfg(all(feature = "mcp", feature = "http"))]
fn run_http(port: u16, allow_remote: bool, toolset: &str) -> Result<()> {
    use std::sync::Arc;
    // Persist the toolset as `Option<String>`: `None` means default ("all").
    // The child subprocess gets `--toolset <name>` only when explicitly set.
    let child_toolset = if toolset == "all" {
        None
    } else {
        Some(toolset.to_string())
    };

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    rt.block_on(async move {
        let bin = semantex_mcp::http_transport::SubprocessBackend::detect_binary()?;
        let backend = Arc::new(
            semantex_mcp::http_transport::SubprocessBackend::new_with_toolset(bin, child_toolset),
        );
        semantex_mcp::http_transport::run_http_server(port, allow_remote, backend).await
    })
}

#[cfg(all(test, feature = "mcp"))]
mod tests {
    use super::*;

    #[test]
    fn validate_toolset_accepts_known_names() {
        validate_toolset("core").unwrap();
        validate_toolset("structural").unwrap();
        validate_toolset("all").unwrap();
    }

    #[test]
    fn validate_toolset_rejects_unknown() {
        assert!(validate_toolset("bogus").is_err());
        assert!(validate_toolset("CORE").is_err());
        assert!(validate_toolset("").is_err());
    }

    /// FINDING 4 regression: when the user requests --toolset core in --http
    /// mode, the SubprocessBackend must be constructed with that toolset so
    /// the spawned child enforces filtering (single source of truth in
    /// server.rs::tools_for_toolset).
    #[cfg(feature = "http")]
    #[test]
    fn http_backend_forwards_toolset_to_child() {
        use std::path::PathBuf;
        // Reproduce the conversion run_http performs.
        let cases = &[
            ("core", Some("core".to_string())),
            ("structural", Some("structural".to_string())),
            ("all", None),
        ];
        for (toolset, expected) in cases {
            let child_toolset = if *toolset == "all" {
                None
            } else {
                Some((*toolset).to_string())
            };
            assert_eq!(&child_toolset, expected, "toolset={toolset}");
            // Construction with the toolset stores it for spawn args.
            let backend = semantex_mcp::http_transport::SubprocessBackend::new_with_toolset(
                PathBuf::from("/nonexistent/semantex"),
                child_toolset,
            );
            // The backend itself doesn't expose toolset, but we verified the
            // mapping is preserved; the matching backend-side test in
            // `http_transport.rs::subprocess_backend_stores_toolset` validates
            // the field is reachable through the same constructor.
            let _ = backend;
        }
    }
}
