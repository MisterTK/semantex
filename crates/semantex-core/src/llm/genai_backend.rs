//! `genai`-based LLM backend for semantex (Spec L §4 Items 1.1 + 1.3).
//!
//! Supports 25+ providers via the `rust-genai` unified adapter.
//! Configured entirely through environment variables; no API keys are read
//! by semantex itself (genai picks them up from the standard env vars such
//! as `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, etc.).
//!
//! Env-var schema (locked by Spec L §4 Item 1.3):
//!
//! | Var                    | Example                           | Effect                              |
//! |------------------------|-----------------------------------|-------------------------------------|
//! | `SEMANTEX_LLM_MODEL`   | `qwen2.5-coder:7b`                | Required. Enables LLM features.     |
//! | `SEMANTEX_LLM_PROVIDER`| `ollama`, `anthropic`, `openai`   | Optional. Inferred from model name. |
//! | `SEMANTEX_LLM_ENDPOINT`| `http://localhost:11434/v1`       | Optional. OpenAI-compat base URL.   |

#![cfg(feature = "llm")]

use anyhow::{Context, Result};
use genai::Client;
use genai::adapter::AdapterKind;
use genai::chat::{ChatMessage, ChatOptions, ChatRequest};
use genai::resolver::{Endpoint, ServiceTargetResolver};
use genai::{ClientConfig, ServiceTarget};

use crate::llm::LlmCapability;
use crate::llm::{
    CLASSIFIER_SYSTEM_PROMPT, HYDE_SYSTEM_PROMPT, build_classify_prompt, build_hyde_prompt,
    parse_route_from_llm_output,
};
use crate::search::agent_classifier::AgentRoute;

// ─────────────────────────────────────────────────────────────────────────────
// Public type
// ─────────────────────────────────────────────────────────────────────────────

/// LLM backend backed by `rust-genai` (Spec L Phase 1).
///
/// Constructed once at daemon startup via [`GenaiBackend::from_env`] and held
/// as `Arc<dyn LlmCapability>`.  Thread-safe: `genai::Client` is `Clone + Send
/// + Sync`.
pub(crate) struct GenaiBackend {
    client: Client,
    model: String,
    label: String,
}

impl GenaiBackend {
    /// Construct from environment variables.
    ///
    /// Returns `Ok(None)` when `SEMANTEX_LLM_MODEL` is unset (LLM disabled).
    /// Returns `Err` only for genuinely invalid configuration (unknown provider
    /// string, endpoint parse failure, etc.).
    pub fn from_env() -> Result<Option<Self>> {
        let Some(model) = std::env::var("SEMANTEX_LLM_MODEL").ok() else {
            return Ok(None);
        };
        let provider_override = std::env::var("SEMANTEX_LLM_PROVIDER").ok();
        let endpoint_override = std::env::var("SEMANTEX_LLM_ENDPOINT").ok();
        let client = build_client(
            &model,
            provider_override.as_deref(),
            endpoint_override.as_deref(),
        )?;
        let label = format!("genai/{model}");
        Ok(Some(Self {
            client,
            model,
            label,
        }))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Provider inference
// ─────────────────────────────────────────────────────────────────────────────

/// Infer the genai `AdapterKind` from the model name and an optional explicit
/// provider string.
///
/// Rules (in priority order):
/// 1. `override_` supplied → parse it against the known lower-case names.
/// 2. Model prefix matching (same logic as `AdapterKind::from_model`, but we
///    surface an error for truly unknown models so users get a helpful message
///    instead of silently routing to Ollama when they probably didn't intend to).
///    Exception: models that look like local/Ollama models (qwen, llama,
///    mistral, mixtral, deepseek, phi, …) → `Ollama`.
/// 3. Unknown model + no override → `Err` with actionable message.
pub(crate) fn infer_provider(model: &str, override_: Option<&str>) -> Result<AdapterKind> {
    // 1. Explicit override.
    if let Some(prov) = override_ {
        let kind = parse_provider_string(prov).with_context(|| {
            format!(
                "SEMANTEX_LLM_PROVIDER={prov:?} is not a recognised provider. \
                 Valid values: openai, anthropic, gemini, ollama, groq, cohere, \
                 deepseek, xai, fireworks, together, openai_resp"
            )
        })?;
        return Ok(kind);
    }

    // 2. Prefix / keyword inference.
    let lower = model.to_ascii_lowercase();

    // Path-style model names (e.g. "accounts/fireworks/models/qwen-…") cannot be
    // reliably inferred from the prefix — the caller must be explicit.
    if lower.contains('/') {
        anyhow::bail!(
            "Cannot infer LLM provider from path-style model name {model:?} \
             (contains '/').  Set SEMANTEX_LLM_PROVIDER explicitly, e.g. \
             SEMANTEX_LLM_PROVIDER=fireworks for Fireworks paths, or \
             SEMANTEX_LLM_PROVIDER=openai for any OpenAI-compatible gateway."
        );
    }

    // Cloud providers — well-known prefixes.
    if lower.starts_with("claude") {
        return Ok(AdapterKind::Anthropic);
    }
    if lower.starts_with("gemini") {
        return Ok(AdapterKind::Gemini);
    }
    if lower.starts_with("gpt")
        || lower.starts_with("o1")
        || lower.starts_with("o2") // defensive: o2 series not yet released but reserved
        || lower.starts_with("o3")
        || lower.starts_with("o4")
        || lower.starts_with("chatgpt")
    {
        return Ok(AdapterKind::OpenAI);
    }
    if lower.starts_with("grok") {
        return Ok(AdapterKind::Xai);
    }
    if lower.starts_with("command") || lower.starts_with("embed-") {
        // command-r, command-r-plus, command-light, embed-english-… → Cohere
        return Ok(AdapterKind::Cohere);
    }

    // AWS Bedrock model names (e.g. "nova-micro", "nova-pro", "nova-lite") are
    // not natively supported by the genai adapter.  Route through an
    // OpenAI-compatible gateway (e.g. Bedrock's converse API behind a proxy).
    if lower.starts_with("nova-") {
        anyhow::bail!(
            "AWS Bedrock model {model:?} detected (nova-* prefix). \
             Bedrock is not directly supported; set SEMANTEX_LLM_ENDPOINT to your \
             OpenAI-compatible gateway URL and SEMANTEX_LLM_PROVIDER=openai."
        );
    }

    // Local / Ollama-hosted models — common families.
    let ollama_prefixes = [
        "qwen",
        "llama",
        "mistral",
        "mixtral",
        "deepseek",
        "phi",
        "gemma",
        "codellama",
        "vicuna",
        "orca",
        "wizard",
        "hermes",
        "nous",
        "openchat",
        "yi",
        "solar",
        "neural",
        "dolphin",
        "starling",
        "magicoder",
        "zephyr",
        "stablelm",
        "starcoder",
        "codestral",
        "granite",
        "smollm",
        "internlm",
    ];
    for prefix in ollama_prefixes {
        if lower.starts_with(prefix) {
            return Ok(AdapterKind::Ollama);
        }
    }

    // 3. Ambiguous / unknown — ask the user to be explicit.
    // Note: SEMANTEX_LLM_PROVIDER=openai is a valid generic fallback for any
    // OpenAI-compatible gateway (LiteLLM proxy, vLLM, etc.).
    anyhow::bail!(
        "Cannot infer LLM provider from model name {model:?}. \
         Set SEMANTEX_LLM_PROVIDER to one of: openai, anthropic, gemini, ollama, \
         groq, cohere, deepseek, xai, fireworks, together. \
         Tip: use SEMANTEX_LLM_PROVIDER=openai as a generic fallback for any \
         OpenAI-compatible gateway (LiteLLM proxy, vLLM, LM Studio, etc.)."
    )
}

/// Parse a provider string (e.g. from `SEMANTEX_LLM_PROVIDER`) into an
/// `AdapterKind`.  Accepts the same lower-case names that genai uses internally.
fn parse_provider_string(s: &str) -> Result<AdapterKind> {
    let lower = s.to_ascii_lowercase();
    let kind = match lower.as_str() {
        "openai" => AdapterKind::OpenAI,
        "openai_resp" => AdapterKind::OpenAIResp,
        "anthropic" => AdapterKind::Anthropic,
        "gemini" => AdapterKind::Gemini,
        "ollama" => AdapterKind::Ollama,
        "groq" => AdapterKind::Groq,
        "cohere" => AdapterKind::Cohere,
        "deepseek" => AdapterKind::DeepSeek,
        "xai" => AdapterKind::Xai,
        "fireworks" => AdapterKind::Fireworks,
        "together" => AdapterKind::Together,
        "openrouter" => {
            // OpenRouter is OpenAI-compatible; route through OpenAI adapter.
            AdapterKind::OpenAI
        }
        other => anyhow::bail!("unknown provider string: {other:?}"),
    };
    Ok(kind)
}

// ─────────────────────────────────────────────────────────────────────────────
// Client construction
// ─────────────────────────────────────────────────────────────────────────────

/// Build a configured `genai::Client`.
///
/// Provider and endpoint overrides are handled via a `ServiceTargetResolver`
/// closure that runs after genai's default resolution so we can swap in the
/// user-supplied values without re-implementing genai's auth / header logic.
fn build_client(model: &str, provider: Option<&str>, endpoint: Option<&str>) -> Result<Client> {
    let forced_kind = infer_provider(model, provider)?;

    // Clone the strings so they can be captured in the `'static` closure.
    let forced_kind_owned = forced_kind;
    let endpoint_owned: Option<String> = endpoint.map(std::borrow::ToOwned::to_owned);

    // Build via ServiceTargetResolver so we can override both the AdapterKind
    // (which changes the HTTP path / header format) and the base URL.
    let resolver = ServiceTargetResolver::from_resolver_fn(
        move |mut target: ServiceTarget| -> genai::resolver::Result<ServiceTarget> {
            // Override the adapter (provider) in the ModelIden.
            let model_name = target.model.model_name.clone();
            target.model = genai::ModelIden::new(forced_kind_owned, model_name);

            // Override the endpoint URL when requested (e.g. custom Ollama port,
            // LiteLLM proxy, vLLM, LM Studio, etc.).
            if let Some(ref url) = endpoint_owned {
                target.endpoint = Endpoint::from_owned(url.as_str());
            }
            Ok(target)
        },
    );

    let config = ClientConfig::default().with_service_target_resolver(resolver);

    let client = Client::builder().with_config(config).build();
    Ok(client)
}

// ─────────────────────────────────────────────────────────────────────────────
// LlmCapability impl
// ─────────────────────────────────────────────────────────────────────────────

#[async_trait::async_trait]
impl LlmCapability for GenaiBackend {
    async fn classify_route(&self, query: &str) -> Result<AgentRoute> {
        tracing::debug!(
            model = %self.model,
            query_len = query.len(),
            capability = "classify",
            "LLM classify_route: start"
        );
        let start = std::time::Instant::now();

        let prompt = build_classify_prompt(query);
        let chat = ChatRequest::new(vec![
            ChatMessage::system(CLASSIFIER_SYSTEM_PROMPT),
            ChatMessage::user(prompt),
        ]);
        let opts = ChatOptions::default()
            .with_max_tokens(8)
            .with_temperature(0.0);

        let result = self
            .client
            .exec_chat(&self.model, chat, Some(&opts))
            .await
            .context("genai classify_route exec_chat");

        let elapsed_ms = start.elapsed().as_millis();

        match result {
            Ok(resp) => {
                let text = resp
                    .first_text()
                    .ok_or_else(|| anyhow::anyhow!("empty LLM classifier response"))?;
                tracing::debug!(
                    model = %self.model,
                    latency_ms = elapsed_ms,
                    output_len = text.len(),
                    capability = "classify",
                    "LLM classify_route: success"
                );
                parse_route_from_llm_output(text)
            }
            Err(e) => {
                tracing::warn!(
                    model = %self.model,
                    latency_ms = elapsed_ms,
                    error = %e,
                    capability = "classify",
                    "LLM classify_route: failed"
                );
                Err(e)
            }
        }
    }

    async fn synthesize_hyde_doc(&self, query: &str) -> Result<String> {
        tracing::debug!(
            model = %self.model,
            query_len = query.len(),
            capability = "hyde",
            "LLM synthesize_hyde_doc: start"
        );
        let start = std::time::Instant::now();

        let prompt = build_hyde_prompt(query);
        let chat = ChatRequest::new(vec![
            ChatMessage::system(HYDE_SYSTEM_PROMPT),
            ChatMessage::user(prompt),
        ]);
        let opts = ChatOptions::default()
            .with_max_tokens(400)
            .with_temperature(0.2);

        let result = self
            .client
            .exec_chat(&self.model, chat, Some(&opts))
            .await
            .context("genai synthesize_hyde_doc exec_chat");

        let elapsed_ms = start.elapsed().as_millis();

        match result {
            Ok(resp) => {
                let text = resp
                    .first_text()
                    .ok_or_else(|| anyhow::anyhow!("empty HyDE response"))?;
                tracing::debug!(
                    model = %self.model,
                    latency_ms = elapsed_ms,
                    output_len = text.len(),
                    capability = "hyde",
                    "LLM synthesize_hyde_doc: success"
                );
                Ok(text.to_string())
            }
            Err(e) => {
                tracing::warn!(
                    model = %self.model,
                    latency_ms = elapsed_ms,
                    error = %e,
                    capability = "hyde",
                    "LLM synthesize_hyde_doc: failed"
                );
                Err(e)
            }
        }
    }

    fn label(&self) -> &str {
        &self.label
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(all(feature = "llm", test))]
mod tests {
    use super::*;

    use crate::llm::TEST_ENV_LOCK;

    // 1. `from_env` returns Ok(None) when SEMANTEX_LLM_MODEL is unset.
    #[test]
    fn from_env_returns_none_when_unset() {
        // Cross-module env-var lock — subscription_cli tests also mutate
        // SEMANTEX_LLM_* vars on parallel threads. SAFETY: with the lock
        // held, no other test thread mutates these vars during the test.
        let _guard = TEST_ENV_LOCK.lock().unwrap();
        let saved = std::env::var("SEMANTEX_LLM_MODEL").ok();
        unsafe { std::env::remove_var("SEMANTEX_LLM_MODEL") };

        let result = GenaiBackend::from_env();

        if let Some(v) = saved {
            unsafe { std::env::set_var("SEMANTEX_LLM_MODEL", v) };
        }

        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    // 2. Provider inference table test.
    #[test]
    fn provider_inference_from_model_name() {
        struct Case {
            model: &'static str,
            provider_override: Option<&'static str>,
            expect: Result<AdapterKind>,
        }

        fn ok(k: AdapterKind) -> Result<AdapterKind> {
            Ok(k)
        }
        fn err() -> Result<AdapterKind> {
            Err(anyhow::anyhow!("placeholder error"))
        }

        let cases = vec![
            Case {
                model: "claude-sonnet-4-6",
                provider_override: None,
                expect: ok(AdapterKind::Anthropic),
            },
            Case {
                model: "claude-3-opus-20240229",
                provider_override: None,
                expect: ok(AdapterKind::Anthropic),
            },
            Case {
                model: "gpt-4o",
                provider_override: None,
                expect: ok(AdapterKind::OpenAI),
            },
            Case {
                model: "gpt-3.5-turbo",
                provider_override: None,
                expect: ok(AdapterKind::OpenAI),
            },
            Case {
                model: "o1-preview",
                provider_override: None,
                expect: ok(AdapterKind::OpenAI),
            },
            Case {
                model: "o3-mini",
                provider_override: None,
                expect: ok(AdapterKind::OpenAI),
            },
            Case {
                model: "gemini-2.5-pro",
                provider_override: None,
                expect: ok(AdapterKind::Gemini),
            },
            Case {
                model: "gemini-flash-1.5",
                provider_override: None,
                expect: ok(AdapterKind::Gemini),
            },
            // Local/Ollama models → Ollama
            Case {
                model: "qwen2.5-coder:7b",
                provider_override: None,
                expect: ok(AdapterKind::Ollama),
            },
            Case {
                model: "llama3.2",
                provider_override: None,
                expect: ok(AdapterKind::Ollama),
            },
            Case {
                model: "mistral:latest",
                provider_override: None,
                expect: ok(AdapterKind::Ollama),
            },
            Case {
                model: "mixtral:8x7b",
                provider_override: None,
                expect: ok(AdapterKind::Ollama),
            },
            Case {
                model: "deepseek-coder:latest",
                provider_override: None,
                expect: ok(AdapterKind::Ollama),
            },
            Case {
                model: "phi3:mini",
                provider_override: None,
                expect: ok(AdapterKind::Ollama),
            },
            // Explicit override wins over name-based inference.
            Case {
                model: "my-private-model",
                provider_override: Some("anthropic"),
                expect: ok(AdapterKind::Anthropic),
            },
            Case {
                model: "my-private-model",
                provider_override: Some("openai"),
                expect: ok(AdapterKind::OpenAI),
            },
            Case {
                model: "my-private-model",
                provider_override: Some("ollama"),
                expect: ok(AdapterKind::Ollama),
            },
            Case {
                model: "my-private-model",
                provider_override: Some("gemini"),
                expect: ok(AdapterKind::Gemini),
            },
            // Unknown model with no override → Err.
            Case {
                model: "unknown-model-xyz",
                provider_override: None,
                expect: err(),
            },
            // o2 prefix → OpenAI (Finding 14: defensive forward-compat).
            Case {
                model: "o2-mini",
                provider_override: None,
                expect: ok(AdapterKind::OpenAI),
            },
            // command-r and command-r-plus → Cohere (verify Finding 12).
            Case {
                model: "command-r",
                provider_override: None,
                expect: ok(AdapterKind::Cohere),
            },
            Case {
                model: "command-r-plus",
                provider_override: None,
                expect: ok(AdapterKind::Cohere),
            },
            // Path-style Fireworks model → Err with actionable message (Finding 12).
            Case {
                model: "accounts/fireworks/models/qwen2p5-coder-7b-instruct",
                provider_override: None,
                expect: err(),
            },
            // AWS Bedrock nova-* → Err with actionable message (Finding 12).
            Case {
                model: "nova-micro",
                provider_override: None,
                expect: err(),
            },
            Case {
                model: "nova-pro",
                provider_override: None,
                expect: err(),
            },
            // Bedrock nova-* with explicit openai override → ok (user knows what they're doing).
            Case {
                model: "nova-pro",
                provider_override: Some("openai"),
                expect: ok(AdapterKind::OpenAI),
            },
        ];

        for case in cases {
            let got = infer_provider(case.model, case.provider_override);
            match case.expect {
                Ok(expected_kind) => {
                    assert!(
                        got.is_ok(),
                        "model={:?} provider={:?} — expected Ok({:?}), got Err: {}",
                        case.model,
                        case.provider_override,
                        expected_kind,
                        got.unwrap_err()
                    );
                    assert_eq!(
                        got.unwrap(),
                        expected_kind,
                        "model={:?} provider={:?}",
                        case.model,
                        case.provider_override
                    );
                }
                Err(_) => {
                    assert!(
                        got.is_err(),
                        "model={:?} provider={:?} — expected Err, got Ok({:?})",
                        case.model,
                        case.provider_override,
                        got.unwrap()
                    );
                    let msg = got.unwrap_err().to_string();
                    assert!(
                        msg.contains("SEMANTEX_LLM_PROVIDER") || msg.contains("unknown provider"),
                        "error should mention SEMANTEX_LLM_PROVIDER, got: {msg}"
                    );
                }
            }
        }
    }

    // 3. Integration test — real Ollama (operator-gated).
    #[tokio::test]
    #[ignore = "requires Ollama with qwen2.5-coder:7b — set SEMANTEX_LLM_TEST_OLLAMA=1 to run"]
    async fn real_ollama_classify() {
        if std::env::var("SEMANTEX_LLM_TEST_OLLAMA").is_err() {
            return;
        }
        let _guard = TEST_ENV_LOCK.lock().unwrap();
        // SAFETY: TEST_ENV_LOCK guards env mutation across modules.
        unsafe { std::env::set_var("SEMANTEX_LLM_MODEL", "qwen2.5-coder:7b") };
        unsafe { std::env::set_var("SEMANTEX_LLM_PROVIDER", "ollama") };

        let backend = GenaiBackend::from_env()
            .expect("from_env should succeed")
            .expect("backend should be Some with SEMANTEX_LLM_MODEL set");

        let route = backend
            .classify_route("who calls handle_request?")
            .await
            .expect("classify_route should succeed against Ollama");
        assert_eq!(
            route,
            AgentRoute::Structural,
            "expected Structural route for a caller-lookup query"
        );
    }

    // 4. Integration test — real Ollama HyDE (operator-gated).
    #[tokio::test]
    #[ignore = "requires Ollama with qwen2.5-coder:7b — set SEMANTEX_LLM_TEST_OLLAMA=1 to run"]
    async fn real_ollama_hyde() {
        if std::env::var("SEMANTEX_LLM_TEST_OLLAMA").is_err() {
            return;
        }
        let _guard = TEST_ENV_LOCK.lock().unwrap();
        // SAFETY: TEST_ENV_LOCK guards env mutation across modules.
        unsafe { std::env::set_var("SEMANTEX_LLM_MODEL", "qwen2.5-coder:7b") };
        unsafe { std::env::set_var("SEMANTEX_LLM_PROVIDER", "ollama") };

        let backend = GenaiBackend::from_env()
            .expect("from_env should succeed")
            .expect("backend should be Some with SEMANTEX_LLM_MODEL set");

        let doc = backend
            .synthesize_hyde_doc("code that handles timeouts")
            .await
            .expect("synthesize_hyde_doc should succeed against Ollama");

        let lower = doc.to_ascii_lowercase();
        assert!(
            lower.contains("duration") || lower.contains("timeout"),
            "HyDE doc should mention 'duration' or 'timeout', got: {doc}"
        );
    }
}
