use anyhow::Result;
use semantex_core::config::SemantexConfig;

pub fn run(config: &SemantexConfig) -> Result<()> {
    #[cfg(feature = "mcp")]
    {
        let server = semantex_mcp::McpServer::new(config.clone());
        server.run()
    }

    #[cfg(not(feature = "mcp"))]
    {
        let _ = config;
        anyhow::bail!("MCP support not compiled in. Build with --features mcp")
    }
}
