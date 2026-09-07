pub use openai_protocol::{
    ChatCompletionRequest, ChatMessage, ChoiceDelta, ErrorEnvelope, FunctionChunk, FunctionContent,
    FunctionDefinition, ImageUrl, MessageContent, MessagePart, Model, ResponseMessageDelta,
    ResponseStreamEvent, ResponseStreamResult, StreamOptions, ToolCall, ToolCallChunk,
    ToolCallContent, ToolChoice, ToolDefinition, Usage, VERSIONED_CHAT_COMPLETIONS_PATH, get_json,
};

use anyhow::{Context as _, Result};
use http_client::{CustomHeaders, HttpClient};
use openai_protocol::ProviderSpec;
use serde::Deserialize;
use std::borrow::Cow;

pub const KILO_MODELS_URL: &str = "https://api.kilo.ai/api/openrouter/models";
pub const KILO_API_URL: &str = "https://api.kilo.ai/api/openrouter";

/// Only model ids carrying this suffix are usable without credentials;
/// every other model on the gateway is paid and must not be offered.
const FREE_MODEL_SUFFIX: &str = ":free";

/// Describes how Kilo speaks the OpenAI chat-completions protocol. The base
/// URL already ends with a version prefix, so the endpoint path is
/// `/chat/completions` rather than `/v1/chat/completions`.
pub const SPEC: ProviderSpec = ProviderSpec {
    chat_completions_path: Cow::Borrowed(VERSIONED_CHAT_COMPLETIONS_PATH),
    label: Cow::Borrowed("Kilo"),
};

/// The name shown in Zed's UI for a Kilo model: the OpenRouter-style
/// `vendor/model:tag` id reduced to its model segment.
pub fn display_name_for(model: &Model) -> &str {
    let name = model
        .name
        .strip_suffix(FREE_MODEL_SUFFIX)
        .unwrap_or(&model.name);
    name.rsplit_once('/')
        .map_or(name, |(_, model_name)| model_name)
}

#[derive(Debug, Deserialize)]
struct ModelListResponse {
    #[serde(default)]
    data: Vec<ModelEntry>,
}

#[derive(Debug, Deserialize)]
struct ModelEntry {
    id: String,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    context_length: Option<u64>,
    #[serde(default)]
    architecture: Option<ModelArchitecture>,
    #[serde(default)]
    supported_parameters: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct ModelArchitecture {
    #[serde(default)]
    input_modalities: Vec<String>,
}

impl ModelEntry {
    fn into_model(self) -> Model {
        let input_modalities = self
            .architecture
            .map(|architecture| architecture.input_modalities)
            .unwrap_or_default();
        let supports_reasoning_effort = self
            .supported_parameters
            .iter()
            .any(|parameter| parameter == "reasoning_effort");
        Model {
            name: self.id,
            display_name: self.name,
            max_tokens: self
                .context_length
                .unwrap_or(openai_protocol::DEFAULT_CONTEXT_LENGTH),
            supports_tools: self
                .supported_parameters
                .iter()
                .any(|parameter| parameter == "tools"),
            supports_images: input_modalities.iter().any(|modality| modality == "image"),
            supports_thinking: self.supported_parameters.iter().any(|parameter| {
                matches!(
                    parameter.as_str(),
                    "reasoning" | "include_reasoning" | "reasoning_effort"
                )
            }),
            supports_reasoning_effort,
        }
    }
}

fn parse_models(body: &str) -> Result<Vec<Model>> {
    let response: ModelListResponse =
        serde_json::from_str(body).context("failed to parse Kilo models list")?;
    Ok(response
        .data
        .into_iter()
        .filter(|entry| entry.id.contains(FREE_MODEL_SUFFIX))
        .map(ModelEntry::into_model)
        .collect())
}

pub async fn fetch_models(client: &dyn HttpClient) -> Result<Vec<Model>> {
    let body = get_json(
        client,
        KILO_MODELS_URL,
        None,
        &CustomHeaders::default(),
        &SPEC,
    )
    .await?;
    parse_models(&body)
}

pub async fn stream_chat_completion(
    client: &dyn HttpClient,
    api_url: &str,
    request: ChatCompletionRequest,
    extra_headers: &CustomHeaders,
) -> Result<futures::stream::BoxStream<'static, Result<ResponseStreamEvent>>> {
    // Free models require no Authorization header.
    openai_protocol::stream_chat_completion(client, api_url, None, request, extra_headers, &SPEC)
        .await
}
