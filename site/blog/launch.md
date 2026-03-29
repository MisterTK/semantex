---
title: "I built a semantic code search engine in Rust. Here's what I learned."
date: 2026-03-14
author: Thomas Kraus
author_bio: "Global Head of AI & SVP at Onix. Previously 5 years at Google Cloud Professional Services. 25+ years in enterprise tech. Austin, TX."
slug: launch
description: "Semantex combines ColBERT dense embeddings with BM25 sparse search to find code by meaning — locally, in 17ms. Open source, Apache-2.0."
tags: [rust, search, ai, code-search, colbert, bm25, mcp]
canonical_url: https://semantex.dev/blog/launch
---

# I built a semantic code search engine in Rust. Here's what I learned.

I asked grep to find the authentication flow in a 500-file TypeScript codebase. It returned 47 matches across 23 files. The actual authentication logic — the OAuth handshake, the token refresh, the session management — was in none of them. Grep found every file that mentioned the *word* "auth." It didn't find the files that *implemented* authentication.

This wasn't a one-off frustration. As someone who works with AI coding agents daily — watching Claude, Codex, Cursor chew through codebases — I kept seeing the same failure mode. An agent asks "how does authentication work here?" and falls into a loop: grep for "auth," read the first file, grep for "verify.*token," read the next file, grep for "session," read another. Each tool call adds to the context window, and in transformer-based models, that context never shrinks. The cost is quadratic — each call resends the entire accumulated history. By the time the agent finds what it needs (if it ever does), it has burned through 6,000+ tokens of search overhead to retrieve maybe 200 tokens of actual answer.

I wanted something that could answer "how does authentication work in this codebase" in one call. It needed to be local (I'm not sending proprietary code to a cloud API), fast (sub-100ms for interactive use), and actually good at retrieval (not just "better than grep," but measurably good against ground truth). Nothing I found met all three requirements. So I built it.

## What semantex does

Semantex is a hybrid semantic + keyword code search engine, written in Rust, that runs entirely on your machine. You give it a natural language query — "how does the database connection pool work" — and it returns the ranked code chunks that actually answer your question, not just the files that contain your search terms.

The key insight is straightforward: combine state-of-the-art dense retrieval (ColBERT per-token embeddings with PLAID indexing) with traditional BM25 keyword search (via Tantivy), and fuse the results with query-adaptive weights. Dense search understands concepts. Sparse search finds exact identifiers. Neither alone is sufficient. Together, they cover each other's blind spots.

Semantex ships as a standalone CLI tool and as an MCP server that plugs directly into AI coding assistants — Claude Code, Cursor, Windsurf, Cline, Continue, Codex, OpenCode.

## Why existing tools fall short

Let me be specific about the landscape, because I have a lot of respect for several of these tools.

**grep and ripgrep.** ripgrep is a genuine masterpiece of systems engineering — Andrew Gallant's work on it is one of the best examples of Rust done right. I use it every day and semantex depends on similar techniques internally. But grep-family tools find *text*, not *concepts*. If you search for "how does authentication work," you get every file containing the substring "auth." The human (or the AI agent) still has to read 23 files to figure out which ones actually implement the auth flow. Grep gives you a haystack. You still need to find the needle.

**Bloop.** Was the most promising entrant. Semantic code search, local-first, well-designed. [Archived January 2025](https://github.com/BloopAI/bloop). Dead project.

**Sourcegraph Cody.** Killed free and pro plans in June 2025. Enterprise-only now, which means it's irrelevant for individual developers and small teams.

**GitHub Copilot semantic search.** Sends your code to Microsoft's infrastructure. Requires a paid subscription. Only works with GitHub-hosted repositories. If you're working on proprietary code in a private GitLab, or on a codebase that can't leave your machine for compliance reasons, this isn't an option.

**SeaGOAT.** A Python-based semantic code search tool that uses MiniLM embeddings (all-MiniLM-L6-v2). Decent idea, and credit to the maintainers for building it. But single-vector embeddings fundamentally lack the token-level granularity that code search demands, and the absence of BM25 hybrid retrieval means it misses exact identifier matches. Also, Python — which means the inference + search pipeline is 10-50x slower than it could be.

**The gap.** There's no production-quality, local-first, open-source semantic code search with state-of-the-art retrieval. That's what semantex fills.

## Architecture deep dive

This is the section I'd want to read if I were evaluating this tool, so I'll go deep.

### Dense retrieval: ColBERT and PLAID

Most embedding-based search tools use single-vector models like MiniLM or Jina — they encode an entire passage into one 384- or 768-dimensional vector, then rank by cosine similarity. This works for paragraph-level semantic similarity, but it's lossy for code. When a query says "database connection pool" and the relevant code says `pool = ConnectionFactory.create(max_idle=10)`, a single-vector model has to somehow compress both the query and the code into fixed-size vectors where cosine similarity captures the match. It often doesn't.

ColBERT takes a different approach: *per-token embeddings with late interaction*. Instead of one vector per passage, ColBERT produces one vector per token. Matching is done via MaxSim — for each query token, find the maximum similarity to any document token, then sum across query tokens. This means the query token "pool" can directly match the document token "pool" while "database" matches "Connection" and "Factory." The alignment happens at the token level, which is exactly the granularity you need for code.

The downside of ColBERT is storage and search cost — per-token embeddings are much larger than single-vector embeddings. PLAID (Performance-optimized Late Interaction using Decontextualized token representations) solves this with product quantization and centroid-based pruning for fast retrieval. Semantex uses the ColBERT LateOn-Code-edge model — a ~17MB int8 ONNX model producing 48-dimensional per-token embeddings. It runs locally on CPU via ONNX Runtime, no GPU required. On macOS, it optionally accelerates with CoreML.

### Sparse retrieval: BM25 via Tantivy

Why both dense and sparse? Because dense search has a critical failure mode: exact identifiers.

If you search for `fhirBaseUrl`, you want BM25 to find that exact token. Dense embeddings might map it to something in the neighborhood of "FHIR base URL" semantically, but the embedding space doesn't guarantee that `fhirBaseUrl` in the query lands near `fhirBaseUrl` in the document — especially when the model was trained on natural language, not camelCase identifiers.

BM25, by contrast, is perfect for this: it's a literal token matching algorithm with tf-idf weighting. If the token exists in the document, BM25 finds it.

Semantex uses [Tantivy](https://github.com/quickwit-oss/tantivy) for its BM25 index — an excellent Rust search library (think "Lucene, but Rust"). The tokenization pipeline includes Snowball English stemming so that "authenticate" matches "authentication," plus a code-aware tokenizer that splits camelCase and snake_case identifiers into constituent tokens.

On top of raw code, the BM25 index is enriched with six layers of natural language annotations: function/class summaries, parameter descriptions, return value semantics, type annotations, semantic role classifications, and import context. This means a BM25 query for "validate user input" can match a function whose code never contains those words, but whose generated summary says "Validates incoming request parameters."

### Fusion: Triple CC with query-adaptive weights

The naive approach to combining dense and sparse results is Reciprocal Rank Fusion (RRF) — merge the rank lists with `1/(k + rank)` scoring. RRF is simple and robust, but it throws away score magnitudes. A dense result with similarity 0.95 and a dense result with similarity 0.31 get treated as "rank 1" and "rank 2" — the information that the first result is vastly more confident is lost.

Semantex uses Triple CC (Convex Combination) instead: normalize each source's scores to [0, 1] by dividing by the top score, then combine with per-source weights. Three sources contribute — dense (ColBERT), sparse (BM25), and exact substring match — each with its own weight.

The weights adapt per query type. A query classifier examines the input and categorizes it as Identifier (`getUserById`), Keyword (`auth`), Semantic (`how does the authentication flow work`), or Mixed. Identifier queries heavily favor exact match (weight 5.0) and BM25 (0.6) over dense (0.2). Semantic queries lean toward dense (0.4) and sparse (0.5) with a lower exact weight (0.8). On top of the static weights, a dynamic adaptation layer (DAT-lite) adjusts based on per-channel confidence — if one channel returns a very strong top result, its weight gets boosted at search time.

This adaptive fusion is what makes semantex competitive on both exact identifier lookups *and* natural language semantic queries, without needing the user to switch modes.

### AST-aware chunking

Naive text chunking — split every N tokens with overlap — works for prose. For code, it's destructive. A 512-token window might split a function in half, or combine the tail of one function with the head of another, creating chunks that are semantically incoherent.

Semantex uses tree-sitter to parse source code into ASTs across 23 languages (Rust, Python, JavaScript, TypeScript, Go, Java, C, C++, Ruby, PHP, C#, Dart, Scala, Kotlin, Swift, Elixir, Lua, Haskell, OCaml, Zig, R, HTML, Svelte), then chunks at function, class, method, and struct boundaries. Each chunk maps to a meaningful unit of code.

During indexing, each chunk gets enriched with metadata that feeds both the dense and sparse search channels:

- **Natural language summary** — a generated description of what the code does, so semantic queries match even when the code uses domain-specific naming
- **Type annotations** — parameter types, return types, trait bounds
- **Semantic role classification** — 11 roles (Service, Controller, Model, Repository, Utility, Config, Test, Migration, Middleware, Documentation, Sanitizer, ErrorHandler, etc.) assigned by pattern matching on function names and structural signals
- **Import resolution** — extracted import/use statements across 8 languages, resolved to relative paths when possible
- **Trait/impl relations** — which types implement which traits, with type hierarchy tracking
- **Docstrings** — extracted and structured separately from code content

This means the search index contains both the raw code *and* human-readable descriptions of what it does. When you search for "error handling middleware," both the code containing `catch` and `recover` and the NL summary containing "handles errors in the request pipeline" contribute to the match.

### Graph propagation

Code doesn't exist in isolation. A function that validates authentication tokens is connected to the function that issues them, the middleware that calls the validator, and the route handler that depends on the middleware. For architectural queries — "how does the data flow from API to database" — you need to follow these connections.

Semantex builds a cross-file graph at index time: call edges (function A calls function B), type references (struct A contains field of type B), and type hierarchy (struct A implements trait B). At search time, for semantic and architectural queries, scores propagate through the graph — if a function scores highly, its callers and callees get a fraction of that score, pulling structurally related code into the result set.

The propagation is query-adaptive. Identifier queries use minimal propagation (you asked for a specific symbol, not its neighborhood). Semantic queries use 1-hop propagation. Architectural queries use 2-hop propagation with higher decay weights, surfacing the broader structural context.

This is still the youngest part of the architecture. Graph resolution rates vary by codebase — in the benchmark codebase (725 files), we resolve about 9% of call edges and 22% of type references. There's significant room for improvement here, and I'll discuss the honest numbers below.

## Benchmarks

### Methodology

Benchmarks without methodology are marketing. Here's exactly how these numbers were produced.

**Codebase:** A real production TypeScript/Dart codebase — 725 files, 6,369 indexed chunks. Not a toy project.

**Queries:** 30 queries, manually written to cover three categories:
- **Exact (8):** Identifier lookups like `fhirBaseUrl` or `familyAuth` — things grep should excel at
- **Semantic (14):** Natural language questions like "how does the connection lifecycle work" or "where is PII handled"
- **Architectural (8):** Cross-cutting questions like "how does data flow from API to database" or "what's the auth interceptor pattern across Flutter and backend"

**Ground truth:** For each query, I manually labeled the set of relevant files. This is subjective and imperfect — there is no "correct" answer for "how does authentication work" — but it's the best available approach for measuring retrieval quality.

**Metric:** F1 score — the harmonic mean of precision (what fraction of returned results are relevant) and recall (what fraction of relevant files were returned). F1 penalizes both false positives and false negatives.

**Comparison:** grep (with manually optimized query terms — I gave grep every advantage) vs. semantex with default settings.

### Results

| Query Type | grep F1 | semantex F1 | Delta |
|---|---|---|---|
| **Overall (30q)** | 0.454 | **0.610** | **+34%** |
| Exact (8q) | 0.606 | **0.645** | +6% |
| Semantic (14q) | 0.463 | **0.568** | +23% |
| Architectural (8q) | 0.285 | **0.650** | **+128%** |

| Performance | grep | semantex | Delta |
|---|---|---|---|
| Warm search latency | 64ms | **17ms** | **3.8x faster** |

The headline number: **128% improvement on architectural queries.** These are exactly the queries where AI agents struggle most and spend the most tokens — "how does auth work end to end," "what's the data flow from ingestion to storage." Grep is nearly useless for these (F1=0.285). Semantex gets them right more than twice as often.

On exact identifier queries, semantex still beats grep, but only by 6%. This is expected — BM25 and grep are solving essentially the same problem for exact string matching, and grep is very good at it. The small edge comes from the fusion layer surfacing contextually related chunks alongside the exact match.

On latency: 17ms warm search. This is faster than grep on the same codebase because the search hits pre-built indices (PLAID + Tantivy + SQLite) rather than scanning raw files. Cold start (first search, building the index) takes 10-30 seconds depending on codebase size and whether the ONNX model is already cached.

### AI agent efficiency

Separate from retrieval quality, I benchmarked the impact on AI agent workflows: 10 agents (Claude Sonnet 4.6), 5 real-world questions, each question answered by one agent using semantex MCP tools and one agent using Grep/Glob/Read.

| Metric | With semantex | Without | Delta |
|---|---|---|---|
| Total tokens | 212K | 355K | **-40%** |
| Cumulative context burden | 2.2M | 6.8M | **-67%** |
| Tool calls | 39 | 86 | **-55%** |
| Wall-clock time | 513s | 609s | -16% |
| Answer quality | Comprehensive | Comprehensive | Tie |

The -40% token number is the billing metric — it directly reduces API costs. The -67% context burden number is arguably more important — it measures the total amount of text the model had to attend to across all turns. Fewer turns with smaller contexts means quadratically less attention waste, which translates to better reasoning quality as conversations grow longer.

Answer quality was a tie across all 5 questions. Both approaches reached comprehensive, accurate answers. Semantex just got there with half the tool calls and 40% fewer tokens.

## Honest failure analysis

Every benchmark has failures. Here are semantex's, because these are what tell you whether the tool fits your use case.

**Q4 — `fhirBaseUrl` (F1=0.22).** An exact identifier query that should be trivial. Semantex finds 1 of 3 files containing this identifier. The issue is camelCase splitting: the BM25 tokenizer splits `fhirBaseUrl` into `fhir`, `base`, `url` — three very common tokens that match broadly. The exact substring match channel should rescue this, but the identifier appears in contexts where surrounding code scores higher on other channels, pushing the right files down. This is a real bug in the fusion logic that I haven't fixed yet.

**Q7 — family auth routing (F1=0.20).** Even grep scores 0.00 on this one. The relevant file (`family.ts`) implements auth-related routing but doesn't contain the word "auth" or "authentication" anywhere — it uses route guards and middleware patterns that are semantically about auth but lexically invisible to both keyword and semantic search. This is a fundamental limitation: if the relevant code doesn't contain semantically related terms, no amount of embedding magic will find it.

**Q17 — auth interceptor, cross-framework (F1=0.40).** The query asks about the authentication interceptor pattern across Flutter and the backend. Semantex finds the backend interceptors but misses `dio_client.dart` — the Dart HTTP client that implements the equivalent pattern. Cross-framework queries are hard because the same concept is expressed with completely different vocabulary and structure across language ecosystems.

**Q20 — parallel API calls (F1=0.25).** The query asks where the codebase makes parallel API calls. The relevant code uses `Promise.allSettled` — a runtime pattern that isn't captured in chunk summaries or type annotations. Semantic search works on static code structure; it can't infer runtime behavior from source text alone.

These aren't edge cases I'm brushing under the rug. They're structural limitations of the approach. Semantic search fails when: (1) relevant code doesn't contain semantically related terms, (2) the answer requires cross-framework vocabulary translation, or (3) the relevant pattern is a runtime behavior rather than a static code structure. I'm working on all three, but they're genuinely hard problems.

## The MCP angle — why this matters for AI

The CLI is useful for humans. The MCP server is what makes semantex transformative for AI coding workflows.

When an AI agent uses grep to explore a codebase, it enters a loop: search, read, search again, read again. Each iteration adds to the context window. The agent is doing what amounts to a breadth-first search through the file system, using a tool (grep) that can only match text, not concepts. By the time it finds what it needs, it has accumulated thousands of tokens of search noise that it will re-read on every subsequent turn.

Semantex's `semantex_deep` tool collapses this loop into a single call. The agent asks "how does authentication work," and semantex internally does the search, reads the relevant code, expands through the call graph, and returns a pre-digested answer with source references. One tool call. No context accumulation. The agent gets the answer and moves on to reasoning about it.

The numbers bear this out: 39 tool calls vs. 86, 212K tokens vs. 355K, same answer quality. The agent spends its token budget on reasoning instead of searching.

## Technical choices

**Why Rust.** 17ms warm search latency. Embedding model inference, BM25 search, graph traversal, and result formatting all happen in a single process with no serialization overhead. Rust's ownership model makes it straightforward to hold a PLAID index, a Tantivy index, and a SQLite database in memory simultaneously without worrying about data races. The tree-sitter bindings are native. The ONNX Runtime bindings (via `ort`) are thin wrappers around the C API. Everything compiles to a single static binary.

**Why not Python.** I prototyped in Python first. The embedding inference alone (via sentence-transformers) took 200ms per query. Adding BM25 (via rank-bm25) added another 50ms. Graph traversal in Python with networkx was 30ms. Total: ~280ms before any I/O. The Rust version does all of this in 17ms. For a tool that needs to feel instant in an interactive workflow, 16x matters.

**Why ONNX Runtime.** Portable inference across macOS (ARM and Intel), Linux, and Windows without requiring users to install Python, PyTorch, or CUDA. The CoreML execution provider on macOS gives a further speedup for free. The model is 17MB — small enough to download on first run without the user noticing.

**Why Apache-2.0.** Maximum adoption. Compatible with enterprise use, compatible with embedding in other tools, no copyleft concerns. I want people to use this, not to read license terms.

## What's next

Semantex is at v0.1. It works, it's measurably better than grep for the queries that matter most, and the MCP integration makes it practical for daily use with AI agents. But there's a lot left to do.

**Better graph resolution.** 9% call graph resolution and 22% type reference resolution are not good enough. The graph propagation feature has the right architecture but needs better cross-file symbol resolution — especially for dynamic languages like JavaScript and Python where import resolution is ambiguous.

**Multi-repo search.** Right now, semantex indexes one project at a time. For microservice architectures where the auth logic lives in a shared library, you need cross-repo search.

**Language-specific semantic understanding.** The current approach is language-agnostic by design — the same pipeline handles Rust, Python, TypeScript, Go. But language-specific knowledge (Rust's trait system, Go's interface satisfaction, Python's duck typing) would improve both chunking and graph resolution.

**Community feedback.** I've benchmarked against one production codebase. I need data on where semantex fails in the wild — different languages, different architectures, different query patterns. If you try it and it misses something, I want to hear about it.

## Try it

```bash
curl -fsSL https://raw.githubusercontent.com/MisterTK/semantex/main/install.sh | sh
```

Or build from source (Rust 1.91+):

```bash
git clone https://github.com/MisterTK/semantex.git
cd semantex && cargo install --path crates/semantex-cli
```

Add it to your editor as an MCP server:

```json
{
  "mcpServers": {
    "semantex": {
      "command": "semantex",
      "args": ["mcp"]
    }
  }
}
```

First search auto-indexes your project. No configuration required.

**GitHub:** [github.com/MisterTK/semantex](https://github.com/MisterTK/semantex)
**License:** Apache-2.0
**Built with:** [Tantivy](https://github.com/quickwit-oss/tantivy), [next-plaid](https://github.com/lightonai/next-plaid), [tree-sitter](https://tree-sitter.github.io/), [ONNX Runtime](https://onnxruntime.ai/)

Your code stays on your machine. Always.
