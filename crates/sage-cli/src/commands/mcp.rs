use anyhow::Result;
use sage_core::config::SageConfig;

pub fn run(config: &SageConfig) -> Result<()> {
    #[cfg(feature = "mcp")]
    {
        let server = sage_mcp::McpServer::new(config.clone());
        server.run()
    }

    #[cfg(not(feature = "mcp"))]
    {
        let _ = config;
        anyhow::bail!("MCP support not compiled in. Build with --features mcp")
    }
}
