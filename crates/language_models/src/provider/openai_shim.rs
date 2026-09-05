//! Shared plumbing for providers that speak the OpenAI chat-completions
//! wire protocol via the `openai_protocol` crate (llama.cpp, Kilo, Zen, and
//! the server-managed custom providers). It owns the two parts every such
//! provider would otherwise copy: translating a
//! [`LanguageModelRequest`] into an [`openai_protocol::ChatCompletionRequest`]
//! and mapping streamed [`openai_protocol::ResponseStreamEvent`]s back into
//! Zed completion events.

use std::collections::HashMap;
use std::pin::Pin;

use anyhow::Result;
use futures::{Stream, StreamExt};
use language_model::util::parse_tool_arguments;
use language_model::{
    LanguageModelCompletionError, LanguageModelCompletionEvent, LanguageModelRequest,
    LanguageModelToolChoice, LanguageModelToolResultContent, LanguageModelToolUse, MessageContent,
    Role, StopReason, TokenUsage,
};
use openai_protocol::{
    ChatCompletionRequest, ChatMessage, FunctionContent, FunctionDefinition, ImageUrl,
    MessageContent as WireMessageContent, MessagePart, ResponseStreamEvent, StreamOptions,
    ToolCall, ToolCallContent, ToolChoice as WireToolChoice, ToolDefinition,
};

/// The per-model capabilities a provider needs when building a request.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RequestCapabilities {
    pub supports_tools: bool,
    pub supports_thinking: bool,
}

/// Builds a wire request for `model_name`.
///
/// `reasoning_effort` is sent only when the model supports it; providers that
/// never send it pass `None`.
pub fn build_request(
    model_name: &str,
    supports_images: bool,
    capabilities: RequestCapabilities,
    request: LanguageModelRequest,
    provider_label: &str,
    reasoning_effort: Option<String>,
) -> Result<ChatCompletionRequest> {
    if request.contains_custom_tool_input() {
        anyhow::bail!("{provider_label} does not support custom tools");
    }

    let supports_tools = capabilities.supports_tools;
    let supports_thinking = capabilities.supports_thinking;
    let mut messages = Vec::new();

    for message in request.messages {
        let mut reasoning_content: Option<String> = None;
        for content in message.content {
            match content {
                MessageContent::Text(text) => add_message_content_part(
                    MessagePart::Text { text },
                    message.role,
                    &mut messages,
                    if supports_thinking && message.role == Role::Assistant {
                        reasoning_content.take()
                    } else {
                        None
                    },
                ),
                MessageContent::Thinking { text, .. } => {
                    if supports_thinking && message.role == Role::Assistant && !text.is_empty() {
                        reasoning_content.get_or_insert_default().push_str(&text);
                    }
                }
                MessageContent::RedactedThinking(_) => {}
                MessageContent::Compaction(_) => {}
                MessageContent::Image(image) => {
                    if supports_images {
                        add_message_content_part(
                            MessagePart::Image {
                                image_url: ImageUrl {
                                    url: image.to_base64_url(),
                                    detail: None,
                                },
                            },
                            message.role,
                            &mut messages,
                            if supports_thinking && message.role == Role::Assistant {
                                reasoning_content.take()
                            } else {
                                None
                            },
                        );
                    }
                }
                MessageContent::ToolUse(tool_use) => {
                    let input = tool_use.input.as_json().ok_or_else(|| {
                        anyhow::anyhow!("{provider_label} does not support custom tool calls")
                    })?;
                    let tool_call = ToolCall {
                        id: tool_use.id.to_string(),
                        content: ToolCallContent::Function {
                            function: FunctionContent {
                                name: tool_use.name.to_string(),
                                arguments: serde_json::to_string(input).unwrap_or_default(),
                            },
                        },
                    };

                    if let Some(ChatMessage::Assistant {
                        tool_calls,
                        reasoning_content: message_reasoning_content,
                        ..
                    }) = messages.last_mut()
                    {
                        append_reasoning_content(
                            message_reasoning_content,
                            reasoning_content.take(),
                        );
                        tool_calls.push(tool_call);
                    } else {
                        messages.push(ChatMessage::Assistant {
                            content: None,
                            reasoning_content: reasoning_content.take(),
                            tool_calls: vec![tool_call],
                        });
                    }
                }
                MessageContent::ToolResult(tool_result) => {
                    let content: Vec<MessagePart> = tool_result
                        .content
                        .iter()
                        .filter_map(|part| match part {
                            LanguageModelToolResultContent::Text(text) => Some(MessagePart::Text {
                                text: text.to_string(),
                            }),
                            LanguageModelToolResultContent::Image(image) => {
                                if supports_images {
                                    Some(MessagePart::Image {
                                        image_url: ImageUrl {
                                            url: image.to_base64_url(),
                                            detail: None,
                                        },
                                    })
                                } else {
                                    None
                                }
                            }
                        })
                        .collect();

                    messages.push(ChatMessage::Tool {
                        content: content.into(),
                        tool_call_id: tool_result.tool_use_id.to_string(),
                    });
                }
            }
        }
    }

    let tools: Vec<ToolDefinition> = if supports_tools {
        request
            .tools
            .into_iter()
            .map(|tool| {
                let input_schema = match tool.input {
                    language_model::LanguageModelRequestToolInput::Function {
                        input_schema,
                        ..
                    } => input_schema,
                    language_model::LanguageModelRequestToolInput::Custom { .. } => {
                        return Err(anyhow::anyhow!(
                            "{provider_label} does not support custom tools"
                        ));
                    }
                };
                Ok(ToolDefinition::Function {
                    function: FunctionDefinition {
                        name: tool.name,
                        description: Some(tool.description),
                        parameters: Some(input_schema),
                    },
                })
            })
            .collect::<Result<_>>()?
    } else {
        Vec::new()
    };
    let tool_choice = if tools.is_empty() {
        None
    } else {
        request.tool_choice.map(|choice| match choice {
            LanguageModelToolChoice::Auto => WireToolChoice::Auto,
            LanguageModelToolChoice::Any => WireToolChoice::Required,
            LanguageModelToolChoice::None => WireToolChoice::None,
        })
    };

    Ok(ChatCompletionRequest {
        model: model_name.to_string(),
        messages,
        stream: true,
        max_tokens: None,
        stop: if request.stop.is_empty() {
            None
        } else {
            Some(request.stop)
        },
        temperature: request.temperature,
        tools,
        tool_choice,
        stream_options: Some(StreamOptions {
            include_usage: true,
        }),
        reasoning_effort,
    })
}

/// Appends `new_part` to the last message when it has the same role (merging
/// reasoning into an assistant message), else starts a new message.
fn add_message_content_part(
    new_part: MessagePart,
    role: Role,
    messages: &mut Vec<ChatMessage>,
    reasoning_content: Option<String>,
) {
    match (role, messages.last_mut()) {
        (Role::User, Some(ChatMessage::User { content }))
        | (Role::System, Some(ChatMessage::System { content })) => {
            content.push_part(new_part);
        }
        (
            Role::Assistant,
            Some(ChatMessage::Assistant {
                content: Some(content),
                reasoning_content: message_reasoning_content,
                ..
            }),
        ) => {
            append_reasoning_content(message_reasoning_content, reasoning_content);
            content.push_part(new_part);
        }
        _ => {
            messages.push(match role {
                Role::User => ChatMessage::User {
                    content: WireMessageContent::from(vec![new_part]),
                },
                Role::Assistant => ChatMessage::Assistant {
                    content: Some(WireMessageContent::from(vec![new_part])),
                    reasoning_content,
                    tool_calls: Vec::new(),
                },
                Role::System => ChatMessage::System {
                    content: WireMessageContent::from(vec![new_part]),
                },
            });
        }
    }
}

fn append_reasoning_content(target: &mut Option<String>, content: Option<String>) {
    let Some(content) = content else {
        return;
    };
    if content.is_empty() {
        return;
    }
    target.get_or_insert_default().push_str(&content);
}

/// Accumulates streamed tool-call chunks by index and maps finish reasons to
/// Zed stop reasons.
pub struct ResponseStreamMapper {
    tool_calls_by_index: HashMap<usize, RawToolCall>,
}

impl ResponseStreamMapper {
    pub fn new() -> Self {
        Self {
            tool_calls_by_index: HashMap::default(),
        }
    }

    pub fn map_stream(
        mut self,
        events: Pin<Box<dyn Send + Stream<Item = Result<ResponseStreamEvent>>>>,
    ) -> impl Stream<Item = Result<LanguageModelCompletionEvent, LanguageModelCompletionError>>
    {
        events.flat_map(move |event| {
            futures::stream::iter(match event {
                Ok(event) => self.map_event(event),
                Err(error) => vec![Err(LanguageModelCompletionError::from(error))],
            })
        })
    }

    pub fn map_event(
        &mut self,
        event: ResponseStreamEvent,
    ) -> Vec<Result<LanguageModelCompletionEvent, LanguageModelCompletionError>> {
        let mut events = Vec::new();

        if let Some(usage) = event.usage {
            events.push(Ok(LanguageModelCompletionEvent::UsageUpdate(TokenUsage {
                input_tokens: usage.prompt_tokens,
                output_tokens: usage.completion_tokens,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
            })));
        }

        if let Some(choice) = event.choices.into_iter().next() {
            if let Some(reasoning_content) = choice.delta.reasoning_content {
                events.push(Ok(LanguageModelCompletionEvent::Thinking {
                    text: reasoning_content,
                    signature: None,
                }));
            }

            if let Some(content) = choice.delta.content {
                if !content.is_empty() {
                    events.push(Ok(LanguageModelCompletionEvent::Text(content)));
                }
            }

            if let Some(tool_calls) = choice.delta.tool_calls {
                for tool_call in tool_calls {
                    let entry = self.tool_calls_by_index.entry(tool_call.index).or_default();

                    if let Some(tool_id) = tool_call.id {
                        entry.id = tool_id;
                    }

                    if let Some(function) = tool_call.function {
                        if let Some(name) = function.name {
                            // Only the first chunk carries the function name;
                            // later chunks send an empty name with arguments.
                            if !name.is_empty() {
                                entry.name = name;
                            }
                        }

                        if let Some(arguments) = function.arguments {
                            entry.arguments.push_str(&arguments);
                        }
                    }
                }
            }

            if let Some(finish_reason) = choice.finish_reason.as_deref() {
                match finish_reason {
                    "stop" => {
                        events.push(Ok(LanguageModelCompletionEvent::Stop(StopReason::EndTurn)));
                    }
                    "tool_calls" => {
                        events.extend(self.tool_calls_by_index.drain().map(|(_, tool_call)| {
                            match parse_tool_arguments(&tool_call.arguments) {
                                Ok(input) => Ok(LanguageModelCompletionEvent::ToolUse(
                                    LanguageModelToolUse {
                                        id: tool_call.id.into(),
                                        name: tool_call.name.into(),
                                        is_input_complete: true,
                                        input: language_model::LanguageModelToolUseInput::Json(
                                            input,
                                        ),
                                        raw_input: tool_call.arguments,
                                        thought_signature: None,
                                    },
                                )),
                                Err(error) => {
                                    Ok(LanguageModelCompletionEvent::ToolUseJsonParseError {
                                        id: tool_call.id.into(),
                                        tool_name: tool_call.name.into(),
                                        raw_input: tool_call.arguments.into(),
                                        json_parse_error: error.to_string(),
                                    })
                                }
                            }
                        }));

                        events.push(Ok(LanguageModelCompletionEvent::Stop(StopReason::ToolUse)));
                    }
                    "length" => {
                        events.push(Ok(LanguageModelCompletionEvent::Stop(
                            StopReason::MaxTokens,
                        )));
                    }
                    unexpected => {
                        log::warn!("Unexpected finish_reason: {unexpected:?}");
                        events.push(Ok(LanguageModelCompletionEvent::Stop(StopReason::EndTurn)));
                    }
                }
            }
        }

        events
    }
}

#[derive(Default)]
struct RawToolCall {
    id: String,
    name: String,
    arguments: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use language_model::{
        LanguageModelRequest, LanguageModelRequestMessage, LanguageModelRequestTool,
        LanguageModelRequestToolInput, LanguageModelToolChoice,
    };
    use openai_protocol::{ChatMessage, MessageContent as WireMessageContent};

    fn assistant_thinking_request() -> LanguageModelRequest {
        LanguageModelRequest {
            messages: vec![LanguageModelRequestMessage {
                role: Role::Assistant,
                content: vec![
                    MessageContent::Thinking {
                        text: "reasoning".to_string(),
                        signature: None,
                    },
                    MessageContent::Text("answer".to_string()),
                ],
                cache: false,
                reasoning_details: None,
            }],
            ..Default::default()
        }
    }

    #[test]
    fn preserves_assistant_thinking_when_supported() {
        let request = build_request(
            "test-model",
            false,
            RequestCapabilities {
                supports_tools: false,
                supports_thinking: true,
            },
            assistant_thinking_request(),
            "test",
            None,
        )
        .unwrap();

        match &request.messages[0] {
            ChatMessage::Assistant {
                content: Some(WireMessageContent::Plain(content)),
                reasoning_content: Some(reasoning_content),
                tool_calls,
            } => {
                assert_eq!(content, "answer");
                assert_eq!(reasoning_content, "reasoning");
                assert!(tool_calls.is_empty());
            }
            message => panic!("unexpected message: {message:?}"),
        }
    }

    #[test]
    fn drops_assistant_thinking_when_unsupported() {
        let request = build_request(
            "test-model",
            false,
            RequestCapabilities {
                supports_tools: false,
                supports_thinking: false,
            },
            assistant_thinking_request(),
            "test",
            None,
        )
        .unwrap();

        match &request.messages[0] {
            ChatMessage::Assistant {
                content: Some(WireMessageContent::Plain(content)),
                reasoning_content,
                ..
            } => {
                assert_eq!(content, "answer");
                assert!(reasoning_content.is_none());
            }
            message => panic!("unexpected message: {message:?}"),
        }
    }

    #[test]
    fn forwards_reasoning_effort_when_given() {
        let request = build_request(
            "glm-5.2",
            false,
            RequestCapabilities {
                supports_tools: false,
                supports_thinking: true,
            },
            LanguageModelRequest::default(),
            "test",
            Some("high".to_string()),
        )
        .unwrap();

        assert_eq!(request.reasoning_effort.as_deref(), Some("high"));

        let without_effort = build_request(
            "glm-5.2",
            false,
            RequestCapabilities {
                supports_tools: false,
                supports_thinking: true,
            },
            LanguageModelRequest::default(),
            "test",
            None,
        )
        .unwrap();
        assert!(without_effort.reasoning_effort.is_none());
        // The field is skipped on the wire when unset.
        let value = serde_json::to_value(&without_effort).unwrap();
        assert!(value.get("reasoning_effort").is_none());
    }

    #[test]
    fn filters_tools_when_model_does_not_support_them() {
        let request = LanguageModelRequest {
            tools: vec![LanguageModelRequestTool {
                name: "weather".into(),
                description: "Get the weather".into(),
                input: LanguageModelRequestToolInput::Function {
                    input_schema: serde_json::json!({"type": "object"}),
                    use_input_streaming: false,
                },
            }],
            tool_choice: Some(LanguageModelToolChoice::Auto),
            ..Default::default()
        };

        let filtered = build_request(
            "m",
            false,
            RequestCapabilities {
                supports_tools: false,
                supports_thinking: false,
            },
            request.clone(),
            "test",
            None,
        )
        .unwrap();
        assert!(filtered.tools.is_empty());
        assert!(filtered.tool_choice.is_none());

        let kept = build_request(
            "m",
            false,
            RequestCapabilities {
                supports_tools: true,
                supports_thinking: false,
            },
            request,
            "test",
            None,
        )
        .unwrap();
        assert_eq!(kept.tools.len(), 1);
        assert!(kept.tool_choice.is_some());
    }

    #[test]
    fn maps_usage_text_and_stop_events() {
        let mut mapper = ResponseStreamMapper::new();
        let events = mapper.map_event(ResponseStreamEvent {
            model: "m".to_string(),
            object: "chat.completion.chunk".to_string(),
            choices: vec![openai_protocol::ChoiceDelta {
                index: 0,
                delta: openai_protocol::ResponseMessageDelta {
                    content: Some("hello".to_string()),
                    reasoning_content: None,
                    tool_calls: None,
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: Some(openai_protocol::Usage {
                prompt_tokens: 11,
                completion_tokens: 7,
                total_tokens: 18,
            }),
        });

        assert_eq!(events.len(), 3);

        let usage = events[0].as_ref().unwrap();
        assert!(matches!(
            usage,
            LanguageModelCompletionEvent::UsageUpdate(TokenUsage {
                input_tokens: 11,
                output_tokens: 7,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
            })
        ));

        let text = events[1].as_ref().unwrap();
        assert!(matches!(
            text,
            LanguageModelCompletionEvent::Text(text) if text == "hello"
        ));

        let stop = events[2].as_ref().unwrap();
        assert!(matches!(
            stop,
            LanguageModelCompletionEvent::Stop(StopReason::EndTurn)
        ));
    }
}
