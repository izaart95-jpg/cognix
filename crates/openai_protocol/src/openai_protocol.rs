use anyhow::{Result, anyhow};
use futures::{AsyncBufReadExt, AsyncReadExt, StreamExt, io::BufReader, stream::BoxStream};
use http_client::{
    AsyncBody, CustomHeaders, HttpClient, HttpRequestExt, Method, Request as HttpRequest,
    RequestBuilderExt, http,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::borrow::Cow;

/// Path of the chat-completions endpoint relative to a bare API base URL,
/// e.g. `http://localhost:8080`.
pub const CHAT_COMPLETIONS_PATH: &str = "/v1/chat/completions";
/// Path of the chat-completions endpoint relative to an API base URL that
/// already ends with a version prefix, e.g. `https://api.kilo.ai/api/v1`.
pub const VERSIONED_CHAT_COMPLETIONS_PATH: &str = "/chat/completions";
/// A model exposed to the rest of Zed, after merging API discovery with
/// user-configured overrides.
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct Model {
    pub name: String,
    pub display_name: Option<String>,
    pub max_tokens: u64,
    pub supports_tools: bool,
    pub supports_images: bool,
    pub supports_thinking: bool,
    /// Whether `reasoning_effort` may be sent on the wire for this model.
    #[serde(default)]
    pub supports_reasoning_effort: bool,
}

impl Model {
    pub fn new(
        name: &str,
        display_name: Option<&str>,
        max_tokens: Option<u64>,
        supports_tools: bool,
        supports_images: bool,
        supports_thinking: bool,
    ) -> Self {
        Self {
            name: name.to_owned(),
            display_name: display_name.map(ToString::to_string),
            max_tokens: max_tokens.unwrap_or(DEFAULT_CONTEXT_LENGTH),
            supports_tools,
            supports_images,
            supports_thinking,
            supports_reasoning_effort: false,
        }
    }

    pub fn display_name(&self) -> &str {
        self.display_name.as_deref().unwrap_or(&self.name)
    }
}

/// Fallback context window used when a model list entry carries none.
pub const DEFAULT_CONTEXT_LENGTH: u64 = 131_072;

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolChoice {
    Auto,
    Required,
    None,
}

#[derive(Clone, Deserialize, Serialize, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolDefinition {
    Function { function: FunctionDefinition },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FunctionDefinition {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Option<Value>,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum ChatMessage {
    Assistant {
        #[serde(default)]
        content: Option<MessageContent>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reasoning_content: Option<String>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        tool_calls: Vec<ToolCall>,
    },
    User {
        content: MessageContent,
    },
    System {
        content: MessageContent,
    },
    Tool {
        content: MessageContent,
        tool_call_id: String,
    },
}

#[derive(Serialize, Deserialize, Debug, Eq, PartialEq)]
#[serde(untagged)]
pub enum MessageContent {
    Plain(String),
    Multipart(Vec<MessagePart>),
}

impl MessageContent {
    pub fn push_part(&mut self, part: MessagePart) {
        match self {
            MessageContent::Plain(text) => {
                *self =
                    MessageContent::Multipart(vec![MessagePart::Text { text: text.clone() }, part]);
            }
            MessageContent::Multipart(parts) if parts.is_empty() => match part {
                MessagePart::Text { text } => *self = MessageContent::Plain(text),
                MessagePart::Image { .. } => *self = MessageContent::Multipart(vec![part]),
            },
            MessageContent::Multipart(parts) => parts.push(part),
        }
    }
}

impl From<Vec<MessagePart>> for MessageContent {
    fn from(mut parts: Vec<MessagePart>) -> Self {
        if let [MessagePart::Text { text }] = parts.as_mut_slice() {
            MessageContent::Plain(std::mem::take(text))
        } else {
            MessageContent::Multipart(parts)
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Eq, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MessagePart {
    Text {
        text: String,
    },
    #[serde(rename = "image_url")]
    Image {
        image_url: ImageUrl,
    },
}

#[derive(Serialize, Deserialize, Clone, Debug, Eq, PartialEq)]
pub struct ImageUrl {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Eq, PartialEq)]
pub struct ToolCall {
    pub id: String,
    #[serde(flatten)]
    pub content: ToolCallContent,
}

#[derive(Serialize, Deserialize, Debug, Eq, PartialEq)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ToolCallContent {
    Function { function: FunctionContent },
}

#[derive(Serialize, Deserialize, Debug, Eq, PartialEq)]
pub struct FunctionContent {
    pub name: String,
    pub arguments: String,
}

#[derive(Serialize, Debug)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ToolDefinition>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
}

/// Asks the server to include a final `usage` chunk in the stream.
#[derive(Serialize, Debug)]
pub struct StreamOptions {
    pub include_usage: bool,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Usage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ErrorEnvelope {
    pub message: String,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(untagged)]
pub enum ResponseStreamResult {
    Ok(ResponseStreamEvent),
    Err { error: ErrorEnvelope },
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ResponseStreamEvent {
    pub model: String,
    pub object: String,
    pub choices: Vec<ChoiceDelta>,
    pub usage: Option<Usage>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ChoiceDelta {
    pub index: u32,
    pub delta: ResponseMessageDelta,
    pub finish_reason: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Eq, PartialEq)]
pub struct ResponseMessageDelta {
    pub content: Option<String>,
    /// Reasoning models emit their chain of thought in a dedicated
    /// `reasoning_content` field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallChunk>>,
}

#[derive(Serialize, Deserialize, Debug, Eq, PartialEq)]
pub struct ToolCallChunk {
    pub index: usize,
    pub id: Option<String>,
    pub function: Option<FunctionChunk>,
}

#[derive(Serialize, Deserialize, Debug, Eq, PartialEq)]
pub struct FunctionChunk {
    pub name: Option<String>,
    pub arguments: Option<String>,
}

/// Describes how a provider speaks the OpenAI chat-completions protocol.
#[derive(Clone, Debug)]
pub struct ProviderSpec {
    /// Path of the chat-completions endpoint relative to the provider's API
    /// base URL: [`CHAT_COMPLETIONS_PATH`] or
    /// [`VERSIONED_CHAT_COMPLETIONS_PATH`].
    pub chat_completions_path: Cow<'static, str>,
    /// Human-readable label used in error messages (e.g. `GLM`).
    pub label: Cow<'static, str>,
}

/// Streams a chat completion from an OpenAI-compatible server
/// (`POST {api_url}{spec.chat_completions_path}`), parsing the SSE body into
/// [`ResponseStreamEvent`]s.
pub async fn stream_chat_completion(
    client: &dyn HttpClient,
    api_url: &str,
    api_key: Option<&str>,
    request: ChatCompletionRequest,
    extra_headers: &CustomHeaders,
    spec: &ProviderSpec,
) -> Result<BoxStream<'static, Result<ResponseStreamEvent>>> {
    let uri = format!("{}{}", api_url, spec.chat_completions_path);
    let request_builder = http::Request::builder()
        .method(Method::POST)
        .uri(uri)
        .header("Content-Type", "application/json")
        .when_some(api_key, |builder, api_key| {
            builder.header("Authorization", format!("Bearer {api_key}"))
        });

    let request = request_builder
        .extra_headers(extra_headers)
        .body(AsyncBody::from(serde_json::to_string(&request)?))?;
    let mut response = client.send(request).await?;
    if response.status().is_success() {
        Ok(sse_event_stream(response.into_body()))
    } else {
        let mut body = String::new();
        response.body_mut().read_to_string(&mut body).await?;
        anyhow::bail!(
            "Failed to connect to {} API: {} {}",
            spec.label,
            response.status(),
            body,
        );
    }
}

/// Parses the SSE body of a successful chat-completions response into
/// [`ResponseStreamEvent`]s. Exposed so providers with custom request headers
/// can still share the protocol's stream decoding.
pub fn sse_event_stream(
    body: http_client::AsyncBody,
) -> BoxStream<'static, Result<ResponseStreamEvent>> {
    parse_sse_lines(BufReader::new(body)).boxed()
}

/// Reads the SSE `data:` lines of a successful response body, skipping the
/// `[DONE]` sentinel, and parses each line as a `T`-shaped chunk.
fn parse_sse_lines<T: for<'de> Deserialize<'de>>(
    reader: BufReader<http_client::AsyncBody>,
) -> impl futures::Stream<Item = Result<T>> {
    reader.lines().filter_map(|line| async move {
        match line {
            Ok(line) => {
                let line = line.strip_prefix("data: ")?;
                if line == "[DONE]" {
                    None
                } else {
                    match serde_json::from_str::<StreamResult<T>>(line) {
                        Ok(StreamResult::Ok(response)) => Some(Ok(response)),
                        Ok(StreamResult::Err { error }) => Some(Err(anyhow!(error.message))),
                        Err(error) => Some(Err(anyhow!(error))),
                    }
                }
            }
            Err(error) => Some(Err(anyhow!(error))),
        }
    })
}

/// A single SSE chunk: either the expected payload or an error envelope.
#[derive(Deserialize)]
#[serde(untagged)]
enum StreamResult<T> {
    Ok(T),
    Err { error: ErrorEnvelope },
}

/// Performs `GET {uri}` and returns the response body as a string, failing
/// with a provider-labeled error unless the status is successful. Used for the
/// OpenAI-style `/models` listings.
pub async fn get_json(
    client: &dyn HttpClient,
    uri: &str,
    api_key: Option<&str>,
    extra_headers: &CustomHeaders,
    spec: &ProviderSpec,
) -> Result<String> {
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
        "Failed to connect to {} API: {} {}",
        spec.label,
        response.status(),
        body,
    );
    Ok(body)
}

/// Response shape shared by OpenAI-style `/models` listings: a `data` array
/// whose entries carry at least an `id`. Providers read their own extra fields
/// off the entry.
#[derive(Deserialize, Debug)]
pub struct ModelListResponse<E> {
    #[serde(default)]
    pub data: Vec<E>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_streaming_reasoning_and_tool_calls() {
        let event = serde_json::json!({
            "model": "glm-5.2",
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": null,
                        "reasoning_content": "thinking...",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "function": { "name": "weather", "arguments": "{\"city\":" }
                            }
                        ]
                    },
                    "finish_reason": null
                }
            ]
        });
        let event: ResponseStreamEvent = serde_json::from_value(event).unwrap();
        let delta = &event.choices[0].delta;
        assert_eq!(delta.reasoning_content.as_deref(), Some("thinking..."));
        assert_eq!(delta.tool_calls.as_ref().unwrap().len(), 1);
    }

    #[test]
    fn parses_streaming_error_envelope() {
        let payload = r#"{"error": {"message": "rate limited"}}"#;
        let result: ResponseStreamResult = serde_json::from_str(payload).unwrap();
        match result {
            ResponseStreamResult::Err { error } => assert_eq!(error.message, "rate limited"),
            ResponseStreamResult::Ok(_) => panic!("expected error envelope"),
        }
    }

    #[test]
    fn single_part_message_content_serializes_as_plain_string() {
        let content: MessageContent = vec![MessagePart::Text {
            text: "hello".to_string(),
        }]
        .into();
        assert_eq!(serde_json::to_string(&content).unwrap(), r#""hello""#);
    }

    #[test]
    fn request_omits_optional_fields() {
        let request = ChatCompletionRequest {
            model: "m".to_string(),
            messages: vec![],
            stream: true,
            max_tokens: None,
            stop: None,
            temperature: None,
            tools: vec![],
            tool_choice: None,
            stream_options: None,
            reasoning_effort: None,
        };
        let value = serde_json::to_value(&request).unwrap();
        assert!(value.get("reasoning_effort").is_none());
        assert!(value.get("stream_options").is_none());
        assert!(value.get("tools").is_none());
        assert_eq!(value["stream"], serde_json::json!(true));
    }
}
