//! Server-sided custom provider management. Instead of compiling a crate
//! and provider module per OpenAI-compatible endpoint, this crate fetches a
//! manifest (`providers.json`) from a configurable URL and describes every
//! provider declaratively: its base URL, fallback API key, display name and
//! where its model list comes from.

use std::collections::BTreeMap;

use anyhow::{Context as _, Result};
use futures::{AsyncReadExt, future::BoxFuture};
use http_client::{
    AsyncBody, CustomHeaders, HttpClient, HttpRequestExt, Method, Request as HttpRequest,
    RequestBuilderExt,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Default endpoint serving the provider manifest.
pub const DEFAULT_PROVIDERS_URL: &str = "https://api.cognix.sryze.cc/providers.json";

/// Context window assumed when neither the model list nor the manifest entry
/// reports one.
pub const DEFAULT_CONTEXT_WINDOW: u64 = openai_protocol::DEFAULT_CONTEXT_LENGTH;

/// The root `providers.json` document.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct ProvidersManifest {
    pub providers: BTreeMap<String, ProviderConfig>,
}

/// Default refresh cadence, in minutes, for re-fetching the manifest.
pub const DEFAULT_REFRESH_INTERVAL_MINUTES: u64 = 10;

/// One provider entry in the manifest.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct ProviderConfig {
    /// Whether this provider should be registered at all.
    pub active: bool,
    /// API base URL, e.g. `https://api.tokenrouter.com/v1`.
    pub base_url: String,
    /// Fallback API key used when the user has none stored (env var or
    /// system keychain). Keys the user stored themselves always win.
    pub api_key: Option<String>,
    /// Display name in the UI; defaults to the title-cased provider key.
    pub name: Option<String>,
    /// Where the provider's model list comes from: URLs tried in order
    /// (first success wins), plus hardcoded entries used when every URL
    /// fails (or when no URL is given).
    pub models: Vec<ModelSource>,
}

/// A single `models` entry: either a models-list URL or a hardcoded model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ModelSource {
    Url(String),
    /// The `{"hardcoded": {...}}` form used by the manifest format.
    Hardcoded(HardcodedWrapper),
    /// A bare hardcoded model object.
    Model(HardcodedModel),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HardcodedWrapper {
    pub hardcoded: HardcodedModel,
}

/// A hardcoded model described by the manifest.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct HardcodedModel {
    /// The model's id on the wire, e.g. `qwen/qwen3.8-max-free`.
    pub id: String,
    /// Display name; defaults to the id.
    pub name: Option<String>,
    /// Accepted input modalities; contains `image` when the model accepts
    /// images.
    pub input: Vec<String>,
    /// Context window size.
    pub context_window: Option<u64>,
    /// Whether tool calls are supported.
    pub tools: bool,
    /// Whether reasoning/thinking content is supported.
    pub thinking: bool,
    /// Reasoning-effort values the model accepts.
    pub reasoning_efforts: Vec<String>,
    /// Which of `reasoning_efforts` to send when the user picks none.
    pub default_reasoning_effort: Option<String>,
}

impl Default for ProviderConfig {
    fn default() -> Self {
        Self {
            active: true,
            base_url: String::new(),
            api_key: None,
            name: None,
            models: Vec::new(),
        }
    }
}

impl Default for HardcodedModel {
    fn default() -> Self {
        Self {
            id: String::new(),
            name: None,
            input: vec!["text".to_string()],
            context_window: None,
            tools: true,
            thinking: true,
            reasoning_efforts: Vec::new(),
            default_reasoning_effort: None,
        }
    }
}

impl Default for ModelSource {
    fn default() -> Self {
        ModelSource::Url(String::new())
    }
}

/// A model offered by a dynamic provider, after merging manifest and
/// model-list information.
#[derive(Clone, Debug, PartialEq)]
pub struct DiscoveredModel {
    pub openai_model: openai_protocol::Model,
    pub reasoning_efforts: Vec<String>,
    pub default_reasoning_effort: Option<String>,
}

impl ProviderConfig {
    /// The models-list URLs of this provider, in the order they should be
    /// tried.
    pub fn model_urls(&self) -> Vec<&str> {
        self.models
            .iter()
            .filter_map(|source| match source {
                ModelSource::Url(url) => Some(url.as_str()),
                ModelSource::Hardcoded(_) | ModelSource::Model(_) => None,
            })
            .collect()
    }

    /// The hardcoded models declared for this provider.
    pub fn hardcoded_models(&self) -> Vec<HardcodedModel> {
        self.models
            .iter()
            .filter_map(|source| match source {
                ModelSource::Hardcoded(wrapper) => Some(wrapper.hardcoded.clone()),
                ModelSource::Model(model) => Some(model.clone()),
                ModelSource::Url(_) => None,
            })
            .collect()
    }
}

/// Full URL of a provider's chat-completions endpoint, derived from the
/// shape of its base URL:
///
/// - already ends with `/chat/completions` → used as-is,
/// - ends with a version segment (`/v1`, `/api/v1`, …) → append
///   `/chat/completions`,
/// - otherwise → append `/v1/chat/completions`.
pub fn chat_completions_url(base_url: &str) -> String {
    let trimmed = base_url.trim_end_matches('/');
    if trimmed.ends_with("/chat/completions") {
        return trimmed.to_string();
    }
    let version_segment = trimmed
        .rsplit('/')
        .next()
        .is_some_and(|segment| segment.starts_with('v') && segment[1..].chars().all(|c| c.is_ascii_digit()));
    if version_segment {
        format!("{trimmed}/chat/completions")
    } else {
        format!("{trimmed}/v1/chat/completions")
    }
}

/// The registry id for a manifest key: `cognix.{key}`.
///
/// Keys must not contain `/`: model references are written
/// `provider_id/model_id` and split on the first slash, so a slash in the
/// provider id would make them unaddressable. [`parse_manifest`] skips such
/// keys.
pub fn provider_id_for(key: &str) -> String {
    format!("cognix.{key}")
}

/// Whether a manifest key produces a usable provider id. Rejects empty keys
/// and keys containing `/` (see [`provider_id_for`]).
pub fn is_valid_provider_key(key: &str) -> bool {
    !key.is_empty() && !key.contains('/')
}

/// Fetches and parses the manifest from `url`.
pub async fn fetch_manifest(client: &dyn HttpClient, url: &str) -> Result<ProvidersManifest> {
    let body = get_string(client, url, None, &CustomHeaders::default()).await?;
    parse_manifest(&body)
}

/// Parses a manifest document, skipping entries that fail to deserialize
/// instead of rejecting the whole document.
pub fn parse_manifest(body: &str) -> Result<ProvidersManifest> {
    let value: Value =
        serde_json::from_str(body).context("failed to parse custom providers manifest")?;
    let Some(providers) = value.get("providers").and_then(Value::as_object) else {
        return Ok(ProvidersManifest::default());
    };

    let mut manifest = ProvidersManifest::default();
    for (key, entry) in providers {
        if !is_valid_provider_key(key) {
            log::warn!("skipping custom provider `{key}`: keys must be non-empty and contain no `/`");
            continue;
        }
        match serde_json::from_value::<ProviderConfig>(entry.clone()) {
            Ok(config) => {
                manifest.providers.insert(key.clone(), config);
            }
            Err(error) => {
                log::warn!("skipping custom provider `{key}`: {error}");
            }
        }
    }
    Ok(manifest)
}

/// Fetches a model list from `urls`, trying them in order; the first
/// successful response wins.
pub async fn fetch_models(
    client: &dyn HttpClient,
    urls: &[&str],
    api_key: Option<&str>,
    extra_headers: &CustomHeaders,
) -> Result<Vec<DiscoveredModel>> {
    let mut last_error = None;
    for url in urls {
        match fetch_models_from_url(client, url, api_key, extra_headers).await {
            Ok(models) => return Ok(models),
            Err(error) => {
                log::warn!("failed to fetch custom provider models from {url}: {error:#}");
                last_error = Some(error);
            }
        }
    }
    Err(last_error
        .unwrap_or_else(|| anyhow::anyhow!("custom provider declares no model list URL")))
}

async fn fetch_models_from_url(
    client: &dyn HttpClient,
    url: &str,
    api_key: Option<&str>,
    extra_headers: &CustomHeaders,
) -> Result<Vec<DiscoveredModel>> {
    let body = get_string(client, url, api_key, extra_headers).await?;
    parse_models_list(&body)
}

/// Parses an OpenAI-style `/models` listing into discovered models. Entries
/// whose `type` field is present and not `text` (image/speech models on
/// mixed catalogs) are skipped, mirroring the NextRouter behavior.
pub fn parse_models_list(body: &str) -> Result<Vec<DiscoveredModel>> {
    let value: Value = serde_json::from_str(body).context("failed to parse models list")?;
    let Some(data) = value.get("data").and_then(Value::as_array) else {
        return Ok(Vec::new());
    };

    let mut models = Vec::new();
    for entry in data {
        let Some(id) = entry.get("id").and_then(Value::as_str) else {
            continue;
        };
        let entry_type = entry.get("type").and_then(Value::as_str);
        if entry_type.is_some_and(|model_type| model_type != "text") {
            continue;
        }
        let reasoning_efforts: Vec<String> = entry
            .get("reasoning_efforts")
            .and_then(Value::as_array)
            .map(|efforts| {
                efforts
                    .iter()
                    .filter_map(|effort| effort.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();
        let default_reasoning_effort = entry
            .get("default_reasoning_effort")
            .and_then(Value::as_str)
            .map(str::to_string);
        let supports_images = entry
            .get("input")
            .and_then(Value::as_array)
            .is_some_and(|inputs| inputs.iter().any(|input| input.as_str() == Some("image")));

        models.push(DiscoveredModel {
            openai_model: openai_protocol::Model {
                name: id.to_string(),
                display_name: None,
                max_tokens: DEFAULT_CONTEXT_WINDOW,
                supports_tools: true,
                supports_images,
                supports_thinking: true,
                supports_reasoning_effort: !reasoning_efforts.is_empty(),
            },
            reasoning_efforts,
            default_reasoning_effort,
        });
    }
    Ok(models)
}

/// Converts a hardcoded manifest model into a discovered one.
pub fn discovered_from_hardcoded(model: &HardcodedModel) -> DiscoveredModel {
    let supports_images = model.input.iter().any(|input| input == "image");
    DiscoveredModel {
        openai_model: openai_protocol::Model {
            name: model.id.clone(),
            display_name: model.name.clone(),
            max_tokens: model.context_window.unwrap_or(DEFAULT_CONTEXT_WINDOW),
            supports_tools: model.tools,
            supports_images,
            supports_thinking: model.thinking,
            supports_reasoning_effort: !model.reasoning_efforts.is_empty(),
        },
        reasoning_efforts: model.reasoning_efforts.clone(),
        default_reasoning_effort: model.default_reasoning_effort.clone(),
    }
}

/// GETs `uri` and returns the body, failing on non-success statuses.
fn get_string<'a>(
    client: &'a dyn HttpClient,
    uri: &'a str,
    api_key: Option<&'a str>,
    extra_headers: &'a CustomHeaders,
) -> BoxFuture<'a, Result<String>> {
    Box::pin(async move {
        let request = HttpRequest::builder()
            .method(Method::GET)
            .uri(uri)
            .header("Accept", "application/json")
            .when_some(api_key, |builder, api_key| {
                builder.header("Authorization", format!("Bearer {api_key}"))
            })
            .extra_headers(extra_headers)
            .body(AsyncBody::default())?;

        let mut response = client.send(request).await?;
        let mut body = String::new();
        response.body_mut().read_to_string(&mut body).await?;
        anyhow::ensure!(
            response.status().is_success(),
            "Failed to connect to {uri}: {} {}",
            response.status(),
            body,
        );
        Ok(body)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The manifest shape from the feature request: URL string plus
    /// `{"hardcoded": {...}}` entries, camelCase keys.
    const MANIFEST: &str = r#"{
        "providers": {
            "glm": {
                "active": true,
                "baseUrl": "https://glm.cognix.sryze.cc/v1",
                "apiKey": "sk-example",
                "name": "Cognix-GLM",
                "models": ["https://glm.cognix.sryze.cc/v1/models"]
            },
            "tokenrouter": {
                "active": false,
                "baseUrl": "https://api.tokenrouter.com/v1",
                "apiKey": "sk-example",
                "models": [
                    "https://api.tokenrouter.com/v1/models",
                    {"hardcoded": {"id": "qwen/qwen3.8-max-free", "name": "Qwen Free (TokenRouter)", "input": ["text"], "contextWindow": 131072}}
                ]
            },
            "nim": {
                "active": true,
                "baseUrl": "https://integrate.api.nvidia.com/v1/chat/completions",
                "apiKey": "sk-example",
                "models": ["https://integrate.api.nvidia.com/v1/models"]
            }
        }
    }"#;

    #[test]
    fn parses_manifest_with_defaults() {
        let manifest = parse_manifest(MANIFEST).unwrap();
        assert_eq!(manifest.providers.len(), 3);

        let glm = &manifest.providers["glm"];
        assert!(glm.active);
        assert_eq!(glm.base_url, "https://glm.cognix.sryze.cc/v1");
        assert_eq!(glm.name.as_deref(), Some("Cognix-GLM"));
        assert_eq!(glm.model_urls(), vec!["https://glm.cognix.sryze.cc/v1/models"]);
        assert!(glm.hardcoded_models().is_empty());

        // active defaults to true; api_key is optional.
        let tokenrouter = &manifest.providers["tokenrouter"];
        assert!(!tokenrouter.active);

        let nim = &manifest.providers["nim"];
        assert_eq!(nim.api_key.as_deref(), Some("sk-example"));
    }

    #[test]
    fn parses_hardcoded_models_in_both_forms() {
        let manifest = parse_manifest(
            r#"{"providers": {"p": {"models": [
                {"hardcoded": {"id": "wrapped"}},
                {"id": "bare", "tools": false, "contextWindow": 4096}
            ]}}}"#,
        )
        .unwrap();
        let models = manifest.providers["p"].hardcoded_models();
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].id, "wrapped");
        assert!(models[0].tools);
        assert_eq!(models[0].input, vec!["text"]);

        assert_eq!(models[1].id, "bare");
        assert!(!models[1].tools);
        assert_eq!(models[1].context_window, Some(4096));
    }

    #[test]
    fn skips_bad_entries_but_keeps_good_ones() {
        let manifest = parse_manifest(
            r#"{"providers": {"good": {"baseUrl": "https://example.com"},
                                  "bad": {"baseUrl": 42}}}"#,
        )
        .unwrap();
        assert_eq!(manifest.providers.len(), 1);
        assert!(manifest.providers.contains_key("good"));
    }

    #[test]
    fn empty_manifest_is_valid() {
        let manifest = parse_manifest("{}").unwrap();
        assert!(manifest.providers.is_empty());
        let manifest = parse_manifest("not json").is_err();
        assert!(manifest);
    }

    #[test]
    fn derives_chat_completions_urls() {
        // Bare host: append /v1/chat/completions.
        assert_eq!(
            chat_completions_url("http://localhost:8080"),
            "http://localhost:8080/v1/chat/completions"
        );
        // Versioned base: append only /chat/completions.
        assert_eq!(
            chat_completions_url("https://api.tokenrouter.com/v1"),
            "https://api.tokenrouter.com/v1/chat/completions"
        );
        assert_eq!(
            chat_completions_url("https://api.kilo.ai/api/v1"),
            "https://api.kilo.ai/api/v1/chat/completions"
        );
        // Already a full endpoint: used as-is.
        assert_eq!(
            chat_completions_url("https://integrate.api.nvidia.com/v1/chat/completions"),
            "https://integrate.api.nvidia.com/v1/chat/completions"
        );
        // Trailing slashes are ignored when deciding.
        assert_eq!(
            chat_completions_url("https://example.com/v1/"),
            "https://example.com/v1/chat/completions"
        );
        // Non-numeric version segments are treated as a bare host.
        assert_eq!(
            chat_completions_url("https://example.com/api"),
            "https://example.com/api/v1/chat/completions"
        );
    }

    #[test]
    fn provider_ids_are_prefixed() {
        assert_eq!(provider_id_for("glm"), "cognix.glm");
    }

    #[test]
    fn rejects_provider_keys_that_break_model_references() {
        // A slash in the key would corrupt `provider_id/model_id` references.
        assert!(!is_valid_provider_key("my/org"));
        assert!(!is_valid_provider_key(""));
        assert!(is_valid_provider_key("glm"));
        assert!(is_valid_provider_key("next-router"));

        let manifest = parse_manifest(
            r#"{"providers": {
                "my/org": {"baseUrl": "https://example.com"},
                "": {"baseUrl": "https://example.com"},
                "good": {"baseUrl": "https://example.com"}
            }}"#,
        )
        .unwrap();
        assert_eq!(manifest.providers.len(), 1);
        assert!(manifest.providers.contains_key("good"));
    }

    #[test]
    fn parses_model_lists_with_efforts_and_filters_non_text() {
        let body = r#"{
            "data": [
                {"id": "glm-5.2", "reasoning_efforts": ["high", "max"], "default_reasoning_effort": "high"},
                {"id": "flux-2-dev", "type": "image"},
                {"id": "legacy"},
                {"id": "vision-model", "input": ["text", "image"]}
            ]
        }"#;
        let models = parse_models_list(body).unwrap();
        assert_eq!(models.len(), 3);

        assert_eq!(models[0].openai_model.name, "glm-5.2");
        assert_eq!(models[0].reasoning_efforts, vec!["high", "max"]);
        assert_eq!(
            models[0].default_reasoning_effort.as_deref(),
            Some("high")
        );
        assert!(models[0].openai_model.supports_reasoning_effort);

        assert_eq!(models[1].openai_model.name, "legacy");

        assert!(models[2].openai_model.supports_images);
    }

    #[test]
    fn converts_hardcoded_model_to_discovered() {
        let hardcoded = HardcodedModel {
            id: "qwen/qwen3.8-max-free".to_string(),
            name: Some("Qwen Free".to_string()),
            input: vec!["text".to_string(), "image".to_string()],
            context_window: Some(262_144),
            tools: false,
            thinking: true,
            reasoning_efforts: vec!["low".to_string()],
            default_reasoning_effort: Some("low".to_string()),
        };
        let discovered = discovered_from_hardcoded(&hardcoded);
        assert_eq!(discovered.openai_model.name, "qwen/qwen3.8-max-free");
        assert!(discovered.openai_model.supports_images);
        assert!(!discovered.openai_model.supports_tools);
        assert!(discovered.openai_model.supports_thinking);
        assert_eq!(discovered.openai_model.max_tokens, 262_144);
        assert_eq!(discovered.reasoning_efforts, vec!["low"]);
    }

    #[test]
    fn missing_data_key_yields_empty_model_list() {
        let models = parse_models_list(r#"{"object": "list"}"#).unwrap();
        assert!(models.is_empty());
    }

    #[test]
    fn fetches_manifest_and_models_over_http() {
        let client = http_client::FakeHttpClient::create(|request| {
            let uri = request.uri().to_string();
            async move {
                let (status, body) = match uri.as_str() {
                    "https://example.com/providers.json" => (200, r#"{"providers": {"glm": {"baseUrl": "https://glm.example.com/v1"}}}"#.to_string()),
                    "https://glm.example.com/v1/models" => (200, r#"{"data": [{"id": "glm-5.2"}]}"#.to_string()),
                    "https://unstable.example.com/models" => (500, "boom".to_string()),
                    _ => (404, String::new()),
                };
                Ok(http_client::http::Response::builder()
                    .status(status)
                    .body(http_client::AsyncBody::from(body))
                    .unwrap())
            }
        });

        let manifest = futures::executor::block_on(fetch_manifest(
            client.as_ref(),
            "https://example.com/providers.json",
        ))
        .unwrap();
        assert_eq!(manifest.providers.len(), 1);
        let glm = &manifest.providers["glm"];
        assert!(glm.model_urls().is_empty());

        // URL list: first success wins.
        let models = futures::executor::block_on(fetch_models(
            client.as_ref(),
            &[
                "https://unstable.example.com/models",
                "https://glm.example.com/v1/models",
            ],
            None,
            &CustomHeaders::default(),
        ))
        .unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].openai_model.name, "glm-5.2");

        // All URLs failing surfaces an error rather than an empty list.
        let failed = futures::executor::block_on(fetch_models(
            client.as_ref(),
            &["https://unstable.example.com/models"],
            None,
            &CustomHeaders::default(),
        ));
        assert!(failed.is_err());
    }
}
