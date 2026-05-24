# Semantex Distribution & Registry Submission Guide

Semantex ships an official MCP `server.json` at the repo root that conforms to
the 2025-12-11 schema. This document captures the per-registry submission
requirements gathered during the v0.3 release. Actual submissions are
performed by the maintainer at release time; this file is a checklist, not an
automated submission pipeline.

## Source of truth

- **Manifest:** `server.json` at the repo root.
- **Schema:** `https://static.modelcontextprotocol.io/schemas/2025-12-11/server.schema.json`
- **Programmatic validation:** `cargo test -p semantex-mcp http_transport::tests::server_json_at_repo_root_is_well_formed`

Bump the `version` field in `server.json` in lock-step with the
`workspace.package.version` field in the root `Cargo.toml`. The CI release
job verifies the two match before publishing artifacts.

## Registries we target

| Registry | Status (v0.3) | Submission type | Notes |
|---|---|---|---|
| Official MCP Registry (`registry.modelcontextprotocol.io`) | Submit on release | API + `server.json` | Canonical source; downstream registries mirror it. |
| Smithery (`smithery.ai`) | Submit on release | GitHub OAuth + `smithery.yaml` | Smithery prefers Node-shape servers; for a Rust binary we publish a wrapper container. |
| Anthropic Official MCP Server List | Pull-request to `modelcontextprotocol/servers` | README entry + repo link | Curated list, not auto-imported. |
| Antigravity MCP Store | Submit on release | JSON/YAML manifest | Honours environment-variable substitution; never include raw secrets. |
| Cline Marketplace | GitHub issue on `cline/mcp-marketplace` | Issue template + logo | Manual vetting (~2 days). Add `llms-install.md` only if README is insufficient. |
| Cursor Marketplace | `cursor.com/marketplace/publish` | Plugin bundle | Curated rollout starting Feb 2026; expect manual review. |

## Per-registry checklists

### 1. Official MCP Registry

**What to submit:** the `server.json` itself, plus a publisher identity.

**Steps:**
1. Verify the version in `server.json` matches the new release tag.
2. Verify the OCI image referenced under `packages[0].identifier` exists and
   is signed.
3. POST the manifest to `registry.modelcontextprotocol.io/api/v1/servers`
   using a publisher API key.
4. Confirm the entry appears at `https://registry.modelcontextprotocol.io/servers/io.github.MisterTK/semantex`.

**Open questions:**
- Whether the registry now requires Sigstore signatures on the OCI image. As
  of writing the registry README only mentioned this as a recommendation;
  re-verify at submission time.

### 2. Smithery

**What to submit:** a `smithery.yaml` plus a GitHub repo URL. Smithery's
default expectation is a Node entrypoint, so for a Rust binary we publish
the Docker container `ghcr.io/mistertk/semantex` and point Smithery at the
container as the runtime.

**Steps:**
1. Add a `smithery.yaml` at the repo root with:
   ```yaml
   runtime: container
   image: ghcr.io/mistertk/semantex
   command: ["mcp"]
   ```
   (Note: this file is NOT shipped in v0.3 to avoid churn. Add it during the
   first Smithery submission cycle.)
2. Push the OCI image to GHCR with the matching version tag.
3. Connect the GitHub repo at `smithery.ai/new` and select the `server.json`
   manifest path.
4. Confirm the listing renders correctly (description, environment variables,
   transport options).

**Open questions:**
- Smithery's TypeScript-first documentation does not yet enumerate a stable
  contract for non-Node servers. Confirm with their support during first
  submission.

### 3. Anthropic Official MCP Server List

**What to submit:** a pull request to
`https://github.com/modelcontextprotocol/servers` adding an entry to the
appropriate section of the top-level `README.md`.

**Steps:**
1. Fork `modelcontextprotocol/servers`.
2. Add an entry under "Community Servers" (alphabetical) of the form:
   ```markdown
   - **[Semantex](https://github.com/MisterTK/semantex)** — Fully local
     semantic code search (ColBERT + BM25 + reranking) exposed as MCP tools.
   ```
3. Open the PR; respond to maintainer feedback.

### 4. Antigravity MCP Store

**What to submit:** a structured store manifest. Antigravity supports both
JSON and YAML. The fields overlap heavily with the official `server.json`,
so we expect to be able to submit `server.json` directly once their API
accepts schema-conformant input.

**Steps:**
1. From `antigravity.google/docs/mcp`, sign in with a publisher account.
2. Upload `server.json`. Confirm Antigravity infers transport, environment
   variables, and version correctly.
3. Verify that `${SEMANTEX_MAX_RSS_MB}` and other env vars listed in
   `server.json` are exposed as user-configurable settings.

**Notes:**
- Antigravity's docs are explicit about environment-variable substitution:
  consumers configure values, the editor injects them. Our manifest lists
  these as `isRequired: false` with defaults so the server runs out of the
  box.

### 5. Cline Marketplace

**What to submit:** an issue on `cline/mcp-marketplace` with a repo URL and
a 400x400 PNG logo.

**Steps:**
1. Open a new "Server Submission" issue with template fields:
   - GitHub repo URL
   - One-line description
   - Logo PNG (400×400)
   - Categories (e.g. `code-search`, `developer-tools`)
2. If your README does not have a one-paste install snippet for the Cline
   agent, add an `llms-install.md` at the repo root with copy-pasteable
   `claude_desktop_config.json` JSON and the equivalent Cline configuration.
3. Wait ~2 business days for review.

**Notes:**
- Cline reviews lean on README quality + maintainer reputation + security
  considerations. Our README already calls out the security posture; link to
  `docs/security.md` from the submission issue.

### 6. Cursor Marketplace

**What to submit:** a plugin bundle at `cursor.com/marketplace/publish`.

**Steps:**
1. Sign in with the maintainer's Cursor account.
2. Choose "Submit a plugin" → "MCP server".
3. Provide the repo URL, manifest URL (`server.json`), and a one-paragraph
   description targeted at Cursor users.
4. Cursor's review is manual today (the marketplace launched curated with
   first-party partners). Expect some back-and-forth.

**Open questions:**
- Whether Cursor accepts the `server.json` manifest directly or requires
  Cursor-specific metadata. Confirm at submission time.

## Maintainer release checklist

When cutting a new release:

1. [ ] Bump `version` in `Cargo.toml`, `server.json`, and any per-package
       `version` fields.
2. [ ] Run `cargo test --workspace --lib` to confirm `server.json` still
       validates (the `server_json_at_repo_root_is_well_formed` test).
3. [ ] Publish the OCI image (`ghcr.io/mistertk/semantex:<version>`).
4. [ ] Tag the release in git.
5. [ ] POST `server.json` to the official MCP Registry API.
6. [ ] Update the relevant entries on Smithery, Antigravity, Cline, Cursor.
7. [ ] Update the Anthropic `modelcontextprotocol/servers` README PR if the
       description has changed.

## Out of scope for v0.3

- Automated multi-registry submission. Each registry has a slightly
  different review model and we want the maintainer to retain ownership of
  the rollout. CI publishes the artifacts; the maintainer submits.
- npm distribution of semantex. We are deliberately not shipping a Node
  wrapper because it adds runtime overhead and obscures the security model
  (the Rust binary is the source of truth). Registries that require an npm
  package will receive the OCI image instead.
- Sigstore / cosign signatures on the OCI image. Planned for v0.3.1 once
  the official MCP Registry confirms its signing policy.
