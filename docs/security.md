# Semantex Security & Threat Model

This document covers the security posture of the semantex MCP transports
(stdio and HTTP), the assumptions they rely on, and the operational guidance
operators should follow when exposing semantex outside a local development
machine.

## Scope

Semantex ships three deployable surfaces today:

1. **CLI** — `semantex` binary invoked locally by a human.
2. **MCP stdio transport** — `semantex mcp` spoken over stdin/stdout, invoked
   by an MCP host (Claude Code, Cursor, Codex, etc.).
3. **MCP HTTP transport** — `semantex mcp --http --port <N>` exposing JSON-RPC
   on a TCP listener (introduced in v0.3 via I2).

The threat model below focuses on (2) and (3). For (1) the threat model is
"shell user privilege" — same boundary as any local CLI.

## Trust assumptions

- The MCP host is trusted; semantex itself does not authenticate the host.
- The local filesystem is trusted (indexes are read locally; ports are bound
  locally by default).
- Anyone who can read the `.semantex/` directory of an indexed project can
  read every source file in that project. Indexes are NOT encrypted at rest.
- The TCP loopback interface (`127.0.0.1`) is considered private to the local
  user. This matches the trust boundary of the existing `serve` daemon.

## Default-safe behaviour

| Surface | Default | Override flag |
|---|---|---|
| stdio | One process per invocation | none |
| HTTP bind | `127.0.0.1:5050` (loopback only) | `--allow-remote` to bind `0.0.0.0` |
| CORS | No `Access-Control-Allow-*` headers emitted | None — must be added explicitly if needed |
| Authentication | None | None (see "Future work") |

When `--allow-remote` is passed, semantex prints a loud warning to stderr that
the listener is reachable on every interface and that JSON-RPC requests are
unauthenticated.

## Threats and mitigations

### T1. Unauthenticated remote access via HTTP

**Threat:** if `--allow-remote` is used on a workstation with a routable IP
(e.g. a developer VM in a shared subnet, or a homelab), any host on the
network can issue JSON-RPC requests, including `tools/call` for tools that
trigger indexing or write to the local filesystem.

**Mitigations:**
- Default bind is `127.0.0.1` only.
- `--allow-remote` is gated behind an explicit CLI flag.
- A stderr warning is printed every time `--allow-remote` is used.
- Operators are documented (here) to put any remote semantex behind a
  reverse proxy (Caddy / nginx / cloud LB) that adds authentication and TLS.

### T2. Browser cross-site request forgery

**Threat:** a malicious web page in the operator's browser could issue
`fetch('http://127.0.0.1:5050/mcp/all', { method: 'POST', body: ... })` and
exfiltrate the contents of the semantex index by enumerating tool responses.

**Mitigations:**
- Semantex emits no `Access-Control-Allow-Origin` header, so browsers reject
  cross-origin responses by default (the SOP).
- The fetch *request* still reaches the server, however. Operators concerned
  about CSRF MUST either (a) keep the listener off entirely (use stdio), or
  (b) front semantex with a reverse proxy that requires a custom header /
  bearer token. Adding lightweight bearer auth is tracked under "Future work".

### T3. Tool call against a tool outside the requested toolset

**Threat:** an HTTP caller targeting `/mcp/core` could attempt to invoke a
non-core tool via `tools/call`.

**Mitigation:** the HTTP transport explicitly rejects such calls with
JSON-RPC `-32601 Method not found` before dispatching to the backend. See
`crates/semantex-mcp/src/http_transport.rs::mcp_dispatch`.

### T4. Path traversal via tool arguments

**Threat:** several tools accept a `path` argument. A malicious caller could
pass `../../etc/passwd` or an absolute path to force indexing or reading of
unrelated files.

**Mitigation:**
- `IndexBuilder` canonicalizes paths before opening them.
- Paths outside the caller's project tree are still readable by `ripgrep`
  fallback (this is inherent — same posture as `grep` or `find`).
- For HTTP deployments accepting untrusted callers, operators MUST run
  semantex inside a container/sandbox that restricts filesystem visibility
  (Docker `--read-only`, gVisor, Firecracker, etc.).

### T5. Resource exhaustion (CPU, RAM, disk)

**Threat:** repeated `semantex_index` or large `semantex_search` payloads
could OOM the host or fill disk with index data.

**Mitigation:**
- `SEMANTEX_MAX_RSS_MB` (default 512 MiB in MCP mode) bounds memory; cached
  searchers are evicted when RSS exceeds 75% of the limit.
- Indexing skips automatically when memory pressure is already over 50%.
- Disk is not currently capped — operators should size project directories
  appropriately.
- HTTP transport has no per-IP rate limiting today. Operators behind a
  reverse proxy should apply rate limits there.

### T6. Local privilege escalation via the subprocess backend

**Threat:** the HTTP transport spawns `semantex mcp` as a child process. If
the parent's `current_exe()` resolves to a binary an attacker controls, that
binary is run with the operator's UID.

**Mitigation:**
- `current_exe()` is the operating system's canonical "what binary is this?"
  call; it cannot be redirected by environment variables.
- Operators should ensure the `semantex` binary itself is on a write-protected
  path (e.g. `/usr/local/bin/semantex`, `~/.local/bin/semantex` owned by the
  user).

## Operational recommendations

1. **Local-only (default):** start with `semantex mcp --http --port 5050`
   and let it bind `127.0.0.1`. This is the supported default for
   browser-based agents, LangChain, LlamaIndex, etc., running on the same
   host.
2. **Shared LAN:** put semantex behind a reverse proxy (Caddy or nginx) that
   terminates TLS and enforces bearer-token auth. Run semantex with
   `--allow-remote` and bind only to the loopback IP that the proxy can
   reach (e.g. a Docker bridge).
3. **Multi-tenant:** do not. Semantex assumes a single trusted operator per
   process. To serve multiple users, run one semantex process per user.

## Future work

- Optional bearer-token authentication (header check) for the HTTP transport.
- Optional structured audit log of all incoming tool calls.
- Configurable CORS allowlist for browser-based callers behind a known
  origin.
- Capability negotiation that lets clients restrict which tools they can call
  for the duration of a session.

## Reporting security issues

Please email `security@semantex.dev` for any vulnerability disclosure. Do
NOT open public GitHub issues for security-related problems.
