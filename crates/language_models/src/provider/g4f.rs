use anyhow::Result;
use collections::{HashMap, HashSet};
use credentials_provider::CredentialsProvider;
use fs::Fs;
use futures::Stream;
use futures::{FutureExt, StreamExt, future::BoxFuture, stream::BoxStream};
use gpui::{App, AsyncApp, Context, Entity, Task, TaskExt};
use http_client::{CustomHeaders, HttpClient};
use language_model::util::parse_tool_arguments;
use language_model::{
    ApiKeyState, AuthenticateError, EnvVar, IconOrSvg, InlineDescription, LanguageModel,
    LanguageModelCompletionError, LanguageModelCompletionEvent, LanguageModelEffortLevel,
    LanguageModelId, LanguageModelName, LanguageModelProvider, LanguageModelProviderId,
    LanguageModelProviderName, LanguageModelProviderState, LanguageModelRequest,
    LanguageModelToolChoice, LanguageModelToolResultContent, LanguageModelToolUse, MessageContent,
    ProviderSettingsView, RateLimiter, Role, StopReason, SubPageProviderSettings, TokenUsage,
    env_var,
};
use g4f::{
    G4F_API_URL, ModelEntry, Props, get_models, get_props, stream_chat_completion,
    stream_model_events,
};
pub use settings::LlamaCppAvailableModel as AvailableModel;
use settings::{Settings, SettingsStore, update_settings_file};
use std::pin::Pin;
use std::sync::LazyLock;
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::time::Duration;
use ui::{
    ButtonLike, ButtonLink, ConfiguredApiCard, Divider, List, ListBulletItem, Tooltip, prelude::*,
};
use ui_input::InputField;
use util::ResultExt;

use crate::AllLanguageModelSettings;

const G4F_DOWNLOAD_URL: &str = "https://github.com/xtekky/gpt4free";
const G4F_MODELS_URL: &str = "https://g4f.space/v1/models";

const PROVIDER_ID: LanguageModelProviderId = LanguageModelProviderId::new("cognix.g4f");
const PROVIDER_NAME: LanguageModelProviderName = LanguageModelProviderName::new("Cognix-G4F");

const API_KEY_ENV_VAR_NAME: &str = "G4F_API_KEY";
static API_KEY_ENV_VAR: LazyLock<EnvVar> = env_var!(API_KEY_ENV_VAR_NAME);
// The default key set here is public
const MODEL_EVENT_RECONNECT_INTERVAL: Duration = Duration::from_secs(5);
const ASSUMED_UNLOADED_CONTEXT: u64 = 131_072;

// ====================================================================
// Reasoning-effort configuration
// --------------------------------------------------------------------
// To add/remove a model's reasoning-effort support, edit this list.
// Matching is case-insensitive substring match on the model name.
//
// Tuple layout per entry:
//   (model_name_match, &[(effort_value, display_label), ...], default_effort_value)
// ====================================================================
const REASONING_EFFORT_MODELS: &[(&str, &[(&str, &str)], &str)] = &[
    ("glm-5.2", &[("high", "High"), ("max", "Max")], "high"),
];

fn reasoning_effort_config(model_name: &str) -> Option<&'static (&'static str, &'static [(&'static str, &'static str)], &'static str)> {
    REASONING_EFFORT_MODELS
        .iter()
        .find(|(match_str, _, _)| {
            model_name
                .to_lowercase()
                .contains(&match_str.to_lowercase())
        })
}

fn model_supports_reasoning_effort(model_name: &str) -> bool {
    reasoning_effort_config(model_name).is_some()
}

fn supported_effort_levels_for(model_name: &str) -> Vec<LanguageModelEffortLevel> {
    let Some((_, levels, default)) = reasoning_effort_config(model_name) else {
        return Vec::new();
    };
    levels
        .iter()
        .map(|(value, label)| LanguageModelEffortLevel {
            name: (*label).into(),
            value: (*value).into(),
            is_default: *value == *default,
        })
        .collect()
}

fn resolve_reasoning_effort(
    request: &LanguageModelRequest,
    model_name: &str,
) -> Option<String> {
    let (_, levels, default) = reasoning_effort_config(model_name)?;

    if !request.thinking_allowed {
        return None;
    }

    let chosen = request
        .thinking_effort
        .as_deref()
        .filter(|effort| levels.iter().any(|(v, _)| *v == *effort))
        .unwrap_or(default);

    Some(chosen.to_string())
}

#[derive(Default, Debug, Clone, PartialEq)]
pub struct G4fSettings {
    pub api_url: String,
    pub auto_discover: bool,
    pub available_models: Vec<AvailableModel>,
    pub context_window: Option<u64>,
    pub custom_headers: CustomHeaders,
}

pub struct G4fLanguageModelProvider {
    http_client: Arc<dyn HttpClient>,
    state: Entity<State>,
    capability_cells: CapabilityCells,
    loading_progress: LoadingProgress,
}

pub struct State {
    api_key_state: ApiKeyState,
    credentials_provider: Arc<dyn CredentialsProvider>,
    http_client: Arc<dyn HttpClient>,
    fetched_models: Vec<g4f::Model>,
    fetch_model_task: Option<Task<Result<()>>>,
    model_event_task: Option<Task<()>>,
    capability_cells: CapabilityCells,
    loading_progress: LoadingProgress,
}

impl State {
    fn is_authenticated(&self) -> bool {
        !self.fetched_models.is_empty()
    }

    fn set_api_key(&mut self, api_key: Option<String>, cx: &mut Context<Self>) -> Task<Result<()>> {
        let credentials_provider = self.credentials_provider.clone();
        let api_url = G4fLanguageModelProvider::api_url(cx);
        let task = self.api_key_state.store(
            api_url,
            api_key,
            |this| &mut this.api_key_state,
            credentials_provider,
            cx,
        );

        self.fetched_models.clear();
        self.model_event_task = None;
        write_recover(&self.loading_progress).clear();
        cx.spawn(async move |this, cx| {
            let result = task.await;
            this.update(cx, |this, cx| this.restart_fetch_models_task(cx))
                .ok();
            result
        })
    }

    fn authenticate(&mut self, cx: &mut Context<Self>) -> Task<Result<(), AuthenticateError>> {
        let credentials_provider = self.credentials_provider.clone();
        let api_url = G4fLanguageModelProvider::api_url(cx);
        let load_key_task = self.api_key_state.load_if_needed(
            api_url,
            |this| &mut this.api_key_state,
            credentials_provider,
            cx,
        );

        if self.is_authenticated() {
            return Task::ready(Ok(()));
        }

        cx.spawn(async move |this, cx| {
            match load_key_task.await {
                Ok(()) | Err(AuthenticateError::CredentialsNotFound) => {}
                Err(error) => {
                    log::warn!("failed to load g4f API key: {error}");
                }
            }
            let fetch_models_task = this.update(cx, |this, cx| this.fetch_models(cx))?;
            match fetch_models_task.await {
                Ok(()) => Ok(()),
                Err(err) => {
                    let connection_refused = err.chain().any(|cause| {
                        cause
                            .downcast_ref::<std::io::Error>()
                            .is_some_and(|io_err| {
                                io_err.kind() == std::io::ErrorKind::ConnectionRefused
                            })
                    });
                    if connection_refused {
                        Err(AuthenticateError::ConnectionRefused)
                    } else {
                        Err(AuthenticateError::Other(err))
                    }
                }
            }
        })
    }

    fn fetch_models(&mut self, cx: &mut Context<Self>) -> Task<Result<()>> {
        let http_client = Arc::clone(&self.http_client);
        let settings = G4fLanguageModelProvider::settings(cx);
        let api_url = G4fLanguageModelProvider::api_url(cx);
        let api_key = self.api_key_state.key(&api_url).or_else(|| Some("g4f_u_moizdt_98e713da83fda691c8da8e39a8e5f0658a0daa58a7f3bcab_f49c7608".to_string().into()));
        let extra_headers = settings.custom_headers.clone();

        cx.spawn(async move |this, cx| {
            let entries = match get_models(
                http_client.as_ref(),
                &api_url,
                api_key.as_deref(),
                &extra_headers,
            )
            .await
            {
                Ok(entries) => entries,
                Err(err) => {
                    this.update(cx, |this, cx| {
                        this.fetched_models.clear();
                        cx.notify();
                    })
                   .ok();
                   return Err(err);
                }
            };

            let is_router = entries.iter().any(ModelEntry::is_router_entry);

            let loading_ids: HashSet<String> = entries
                .iter()
                .filter(|entry| entry.is_loading())
                .map(|entry| entry.id.clone())
                .collect();

            let models: Vec<g4f::Model> = if is_router {
                let tasks = entries.into_iter().map(|entry| {
                    let http_client = Arc::clone(&http_client);
                    let api_url = api_url.clone();
                    let api_key = api_key.clone();
                    let extra_headers = extra_headers.clone();
                    async move {
                        let props = if entry.is_loaded() {
                            get_props(
                                http_client.as_ref(),
                                &api_url,
                                api_key.as_deref(),
                                Some(&entry.id),
                                &extra_headers,
                            )
                            .await
                            .log_err()
                        } else {
                            None
                        };
                        model_from_entry(&entry, props.as_ref())
                    }
                });
                futures::stream::iter(tasks)
                    .buffer_unordered(5)
                    .collect()
                    .await
            } else {
                let props = get_props(
                    http_client.as_ref(),
                    &api_url,
                    api_key.as_deref(),
                    None,
                    &extra_headers,
                )
                .await
                .log_err();
                entries
                    .iter()
                    .map(|entry| model_from_entry(entry, props.as_ref()))
                    .collect()
            };

            this.update(cx, |this, cx| {
                this.fetched_models = models;
                let effective = compute_effective_models(
                    &this.fetched_models,
                    G4fLanguageModelProvider::settings(cx),
                );
                sync_capability_cells(&this.capability_cells, &effective);
                write_recover(&this.loading_progress).retain(|id, _| loading_ids.contains(id));
                if is_router {
                    if this.model_event_task.is_none() {
                        this.start_model_event_stream(cx);
                    }
                } else {
                    this.model_event_task = None;
                }
                cx.notify();
            })
        })
    }

    fn start_model_event_stream(&mut self, cx: &mut Context<Self>) {
        let http_client = Arc::clone(&self.http_client);
        let api_url = G4fLanguageModelProvider::api_url(cx);
        let api_key = self.api_key_state.key(&api_url).or_else(|| Some("g4f_u_moizdt_98e713da83fda691c8da8e39a8e5f0658a0daa58a7f3bcab_f49c7608".to_string().into()));
        let extra_headers = G4fLanguageModelProvider::settings(cx)
            .custom_headers
            .clone();

        self.model_event_task = Some(cx.spawn(async move |this, cx| {
            loop {
                match stream_model_events(
                    http_client.as_ref(),
                    &api_url,
                    api_key.as_deref(),
                    &extra_headers,
                )
                .await
                {
                    Ok(mut events) => {
                        while let Some(event) = events.next().await {
                            let Some(event) = event.log_err() else {
                                continue;
                            };
                            if let Some(exit_code) = event.load_failure() {
                                log::error!(
                                    "g4f model {} failed to load (exit code {exit_code})",
                                    event.model
                                );
                            }
                            if let Some(progress) = event.load_progress() {
                                let label = SharedString::from(progress.progress_label());
                                if this
                                    .update(cx, |this, cx| {
                                        write_recover(&this.loading_progress)
                                            .insert(event.model.clone(), label);
                                        cx.notify();
                                    })
                                    .is_err()
                                {
                                    return;
                                }
                                continue;
                            }
                            if !event.changes_model_state() {
                                continue;
                            }
                            if this
                                .update(cx, |this, cx| {
                                    write_recover(&this.loading_progress).remove(&event.model);
                                    this.restart_fetch_models_task(cx);
                                })
                                .is_err()
                            {
                                return;
                            }
                        }
                    }
                    Err(error) => {
                        log::warn!("g4f model event stream unavailable: {error:#}");
                    }
                }

                cx.background_executor()
                    .timer(MODEL_EVENT_RECONNECT_INTERVAL)
                    .await;
                if this.update(cx, |_, _| ()).is_err() {
                    return;
                }
            }
        }));
    }

    fn restart_fetch_models_task(&mut self, cx: &mut Context<Self>) {
        let task = self.fetch_models(cx);
        self.fetch_model_task.replace(task);
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct LiveCapabilities {
    max_tokens: u64,
    supports_tools: bool,
    supports_thinking: bool,
}

impl LiveCapabilities {
    fn of(model: &g4f::Model) -> Self {
        Self {
            max_tokens: model.max_tokens,
            supports_tools: model.supports_tools,
            supports_thinking: model.supports_thinking,
        }
    }
}

type CapabilityCells = Arc<RwLock<HashMap<String, LiveCapabilities>>>;
type LoadingProgress = Arc<RwLock<HashMap<String, SharedString>>>;

fn read_recover<T>(lock: &RwLock<T>) -> RwLockReadGuard<'_, T> {
    lock.read().unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn write_recover<T>(lock: &RwLock<T>) -> RwLockWriteGuard<'_, T> {
    lock.write()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn compute_effective_models(
    fetched_models: &[g4f::Model],
    settings: &G4fSettings,
) -> HashMap<String, g4f::Model> {
    let mut models: HashMap<String, g4f::Model> = HashMap::default();
    if settings.auto_discover {
        for model in fetched_models {
            let mut model = model.clone();
            if let Some(context_window) = settings.context_window {
                model.max_tokens = context_window;
            }
            models.insert(model.name.clone(), model);
        }
    }
    merge_settings_into_models(
        &mut models,
        &settings.available_models,
        settings.context_window,
    );
    models
}

fn sync_capability_cells(cells: &CapabilityCells, effective: &HashMap<String, g4f::Model>) {
    let mut cells = write_recover(cells);
    for model in effective.values() {
        cells.insert(model.name.clone(), LiveCapabilities::of(model));
    }
}

fn model_from_entry(entry: &ModelEntry, props: Option<&Props>) -> g4f::Model {
    let max_tokens = props
        .and_then(Props::context_length)
        .or_else(|| entry.meta.as_ref().and_then(|meta| meta.n_ctx))
        .or_else(|| entry.meta.as_ref().and_then(|meta| meta.n_ctx_train))
        .unwrap_or(ASSUMED_UNLOADED_CONTEXT);
    let supports_tools = match props {
        Some(props) => props.supports_tools(),
        None => !entry.is_loaded(),
    };
    let supports_images = props.is_some_and(Props::supports_images) || entry.supports_images_hint();
    let supports_thinking = props.is_some_and(Props::supports_thinking);

    g4f::Model::new(
        &entry.id,
        Some(&display_name_for(&entry.id)),
        Some(max_tokens),
        supports_tools,
        supports_images,
        supports_thinking,
    )
}

fn display_name_for(id: &str) -> String {
    let base = id.rsplit(['/', '\\']).next().unwrap_or(id);
    base.strip_suffix(".gguf").unwrap_or(base).to_string()
}

fn telemetry_id_for(id: &str) -> String {
    format!("{PROVIDER_ID}/{}", display_name_for(id))
}

impl G4fLanguageModelProvider {
    pub fn new(
        http_client: Arc<dyn HttpClient>,
        credentials_provider: Arc<dyn CredentialsProvider>,
        cx: &mut App,
    ) -> Self {
        let capability_cells: CapabilityCells = Arc::new(RwLock::new(HashMap::default()));
        let loading_progress: LoadingProgress = Arc::new(RwLock::new(HashMap::default()));
        let this = Self {
            http_client: http_client.clone(),
            capability_cells: capability_cells.clone(),
            loading_progress: loading_progress.clone(),
            state: cx.new(|cx| {
                cx.observe_global::<SettingsStore>({
                    let mut last_settings = G4fLanguageModelProvider::settings(cx).clone();
                    move |this: &mut State, cx| {
                        let current_settings = G4fLanguageModelProvider::settings(cx);
                        let settings_changed = current_settings != &last_settings;
                        if settings_changed {
                            let url_changed = last_settings.api_url != current_settings.api_url;
                            last_settings = current_settings.clone();
                            if url_changed {
                                let credentials_provider = this.credentials_provider.clone();
                                let api_url = Self::api_url(cx);
                                this.api_key_state.handle_url_change(
                                    api_url,
                                    |this| &mut this.api_key_state,
                                    credentials_provider,
                                    cx,
                                );
                                this.fetched_models.clear();
                                this.model_event_task = None;
                                write_recover(&this.loading_progress).clear();
                                this.authenticate(cx).detach();
                            }
                            cx.notify();
                        }
                    }
                })
                .detach();

                State {
                    http_client,
                    fetched_models: Default::default(),
                    fetch_model_task: None,
                    model_event_task: None,
                    capability_cells,
                    loading_progress,
                    api_key_state: ApiKeyState::new(Self::api_url(cx), (*API_KEY_ENV_VAR).clone()),
                    credentials_provider,
                }
            }),
        };
        this.state
            .update(cx, |state, cx| state.restart_fetch_models_task(cx));
        this
    }

    fn settings(cx: &App) -> &G4fSettings {
        &AllLanguageModelSettings::get_global(cx).g4f
    }

    fn api_url(cx: &App) -> SharedString {
        let api_url = &Self::settings(cx).api_url;
        if api_url.is_empty() {
            G4F_API_URL.into()
        } else {
            SharedString::new(api_url.as_str())
        }
    }

    fn has_custom_url(cx: &App) -> bool {
        let api_url = &Self::settings(cx).api_url;
        !api_url.is_empty() && api_url != G4F_API_URL
    }
}

impl LanguageModelProviderState for G4fLanguageModelProvider {
    type ObservableEntity = State;

    fn observable_entity(&self) -> Option<Entity<Self::ObservableEntity>> {
        Some(self.state.clone())
    }
}

impl LanguageModelProvider for G4fLanguageModelProvider {
    fn id(&self) -> LanguageModelProviderId {
        PROVIDER_ID
    }

    fn name(&self) -> LanguageModelProviderName {
        PROVIDER_NAME
    }

    fn icon(&self) -> IconOrSvg {
        IconOrSvg::Icon(IconName::Cognix)
    }

    fn default_model(&self, _: &App) -> Option<Arc<dyn LanguageModel>> {
        None
    }

    fn default_fast_model(&self, _: &App) -> Option<Arc<dyn LanguageModel>> {
        None
    }

    fn provided_models(&self, cx: &App) -> Vec<Arc<dyn LanguageModel>> {
        let settings = G4fLanguageModelProvider::settings(cx);
        let effective = compute_effective_models(&self.state.read(cx).fetched_models, settings);

        sync_capability_cells(&self.capability_cells, &effective);
        let mut models = effective
            .into_values()
            .map(|model| {
                Arc::new(G4fLanguageModel {
                    id: LanguageModelId::from(model.name.clone()),
                    name: model.name.clone(),
                    display_name: model.display_name().to_string(),
                    fallback_capabilities: LiveCapabilities::of(&model),
                    supports_images: model.supports_images,
                    capability_cells: self.capability_cells.clone(),
                    loading_progress: self.loading_progress.clone(),
                    http_client: self.http_client.clone(),
                    request_limiter: RateLimiter::new(4),
                    state: self.state.clone(),
                }) as Arc<dyn LanguageModel>
            })
            .collect::<Vec<_>>();
        models.sort_by_key(|model| model.name());
        models
    }

    fn is_authenticated(&self, cx: &App) -> bool {
        self.state.read(cx).is_authenticated()
    }

    fn authenticate(&self, cx: &mut App) -> Task<Result<(), AuthenticateError>> {
        self.state.update(cx, |state, cx| state.authenticate(cx))
    }

    fn settings_view(&self, _cx: &mut App) -> Option<ProviderSettingsView> {
        let state = self.state.clone();
        Some(ProviderSettingsView::SubPage(
            SubPageProviderSettings::new(move |window, cx| {
                cx.new(|cx| ConfigurationView::new(state.clone(), window, cx))
                    .into()
            })
            .description(InlineDescription::Text(
                "Experience the most powerful AI models without barriers.".into(),
            )),
        ))
    }
}

pub struct G4fLanguageModel {
    id: LanguageModelId,
    name: String,
    display_name: String,
    capability_cells: CapabilityCells,
    fallback_capabilities: LiveCapabilities,
    supports_images: bool,
    loading_progress: LoadingProgress,
    http_client: Arc<dyn HttpClient>,
    request_limiter: RateLimiter,
    state: Entity<State>,
}

impl G4fLanguageModel {
    fn capabilities(&self) -> LiveCapabilities {
        read_recover(&self.capability_cells)
            .get(&self.name)
            .copied()
            .unwrap_or(self.fallback_capabilities)
    }

    fn loading_label(&self) -> Option<SharedString> {
        read_recover(&self.loading_progress)
            .get(&self.name)
            .cloned()
    }

    fn to_g4f_request(
        &self,
        request: LanguageModelRequest,
    ) -> Result<g4f::ChatCompletionRequest> {
        build_g4f_request(
            &self.name,
            self.supports_images,
            self.capabilities(),
            request,
        )
    }

    fn stream_completion(
        &self,
        request: g4f::ChatCompletionRequest,
        cx: &AsyncApp,
    ) -> BoxFuture<
        'static,
        Result<futures::stream::BoxStream<'static, Result<g4f::ResponseStreamEvent>>>,
    > {
        let http_client = self.http_client.clone();
        let (api_key, api_url, extra_headers) = self.state.read_with(cx, |state, cx| {
            let api_url = G4fLanguageModelProvider::api_url(cx);
            let extra_headers = G4fLanguageModelProvider::settings(cx)
                .custom_headers
                .clone();
            (state.api_key_state.key(&api_url).or_else(|| Some("g4f_u_moizdt_98e713da83fda691c8da8e39a8e5f0658a0daa58a7f3bcab_f49c7608".to_string().into())), api_url, extra_headers)
        });

        let future = self.request_limiter.stream(async move {
            let stream = stream_chat_completion(
                http_client.as_ref(),
                &api_url,
                api_key.as_deref(),
                request,
                &extra_headers,
            )
            .await?;
            Ok(stream)
        });

        async move { Ok(future.await?.boxed()) }.boxed()
    }
}

fn build_g4f_request(
    model_name: &str,
    supports_images: bool,
    capabilities: LiveCapabilities,
    request: LanguageModelRequest,
) -> Result<g4f::ChatCompletionRequest> {
    if request.contains_custom_tool_input() {
        anyhow::bail!("g4f does not support custom tools");
    }

    let supports_tools = capabilities.supports_tools;
    let supports_thinking =
        capabilities.supports_thinking || model_supports_reasoning_effort(model_name);    let mut messages = Vec::new();
    let reasoning_effort = resolve_reasoning_effort(&request, model_name);

    for message in request.messages {
        let mut reasoning_content: Option<String> = None;
        for content in message.content {
            match content {
                MessageContent::Text(text) => add_message_content_part(
                    g4f::MessagePart::Text { text },
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
                            g4f::MessagePart::Image {
                                image_url: g4f::ImageUrl {
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
                        anyhow::anyhow!("g4f does not support custom tool calls")
                    })?;
                    let tool_call = g4f::ToolCall {
                        id: tool_use.id.to_string(),
                        content: g4f::ToolCallContent::Function {
                            function: g4f::FunctionContent {
                                name: tool_use.name.to_string(),
                                arguments: serde_json::to_string(input).unwrap_or_default(),
                            },
                        },
                    };

                    if let Some(g4f::ChatMessage::Assistant {
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
                        messages.push(g4f::ChatMessage::Assistant {
                            content: None,
                            reasoning_content: reasoning_content.take(),
                            tool_calls: vec![tool_call],
                        });
                    }
                }
                MessageContent::ToolResult(tool_result) => {
                    let content: Vec<g4f::MessagePart> = tool_result
                        .content
                        .iter()
                        .filter_map(|part| match part {
                            LanguageModelToolResultContent::Text(text) => {
                                Some(g4f::MessagePart::Text {
                                    text: text.to_string(),
                                })
                            }
                            LanguageModelToolResultContent::Image(image) => {
                                if supports_images {
                                    Some(g4f::MessagePart::Image {
                                        image_url: g4f::ImageUrl {
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

                    messages.push(g4f::ChatMessage::Tool {
                        content: content.into(),
                        tool_call_id: tool_result.tool_use_id.to_string(),
                    });
                }
            }
        }
    }

    let tools: Vec<g4f::ToolDefinition> = if supports_tools {
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
                        return Err(anyhow::anyhow!("g4f does not support custom tools"));
                    }
                };
                Ok(g4f::ToolDefinition::Function {
                    function: g4f::FunctionDefinition {
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
            LanguageModelToolChoice::Auto => g4f::ToolChoice::Auto,
            LanguageModelToolChoice::Any => g4f::ToolChoice::Required,
            LanguageModelToolChoice::None => g4f::ToolChoice::None,
        })
    };

    Ok(g4f::ChatCompletionRequest {
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
        stream_options: Some(g4f::StreamOptions {
            include_usage: true,
        }),
        reasoning_effort,
    })
}

impl LanguageModel for G4fLanguageModel {
    fn id(&self) -> LanguageModelId {
        self.id.clone()
    }

    fn name(&self) -> LanguageModelName {
        match self.loading_label() {
            Some(label) => LanguageModelName::from(format!("{} · {}", self.display_name, label)),
            None => LanguageModelName::from(self.display_name.clone()),
        }
    }

    fn provider_id(&self) -> LanguageModelProviderId {
        PROVIDER_ID
    }

    fn provider_name(&self) -> LanguageModelProviderName {
        PROVIDER_NAME
    }

    fn supports_tools(&self) -> bool {
        self.capabilities().supports_tools
    }

    fn supports_tool_choice(&self, choice: LanguageModelToolChoice) -> bool {
        self.supports_tools()
            && match choice {
                LanguageModelToolChoice::Auto => true,
                LanguageModelToolChoice::Any => true,
                LanguageModelToolChoice::None => true,
            }
    }

    fn supports_images(&self) -> bool {
        self.supports_images
    }

    fn supports_thinking(&self) -> bool {
        self.capabilities().supports_thinking
            || model_supports_reasoning_effort(&self.name)
    }
    
    fn supported_effort_levels(&self) -> Vec<LanguageModelEffortLevel> {
        supported_effort_levels_for(&self.name)
    }

    fn telemetry_id(&self) -> String {
        telemetry_id_for(&self.name)
    }

    fn max_token_count(&self) -> u64 {
        self.capabilities().max_tokens
    }

    fn stream_completion(
        &self,
        request: LanguageModelRequest,
        cx: &AsyncApp,
    ) -> BoxFuture<
        'static,
        Result<
            BoxStream<'static, Result<LanguageModelCompletionEvent, LanguageModelCompletionError>>,
            LanguageModelCompletionError,
        >,
    > {
        let request = match self.to_g4f_request(request) {
            Ok(request) => request,
            Err(error) => return async move { Err(error.into()) }.boxed(),
        };
        let completions = self.stream_completion(request, cx);
        async move {
            let mapper = G4fEventMapper::new();
            Ok(mapper.map_stream(completions.await?).boxed())
        }
        .boxed()
    }
}

struct G4fEventMapper {
    tool_calls_by_index: HashMap<usize, RawToolCall>,
}

impl G4fEventMapper {
    fn new() -> Self {
        Self {
            tool_calls_by_index: HashMap::default(),
        }
    }

    pub fn map_stream(
        mut self,
        events: Pin<Box<dyn Send + Stream<Item = Result<g4f::ResponseStreamEvent>>>>,
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
        event: g4f::ResponseStreamEvent,
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
                        log::warn!("Unexpected g4f finish_reason: {unexpected:?}");
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

fn add_message_content_part(
    new_part: g4f::MessagePart,
    role: Role,
    messages: &mut Vec<g4f::ChatMessage>,
    reasoning_content: Option<String>,
) {
    match (role, messages.last_mut()) {
        (Role::User, Some(g4f::ChatMessage::User { content }))
        | (Role::System, Some(g4f::ChatMessage::System { content })) => {
            content.push_part(new_part);
        }
        (
            Role::Assistant,
            Some(g4f::ChatMessage::Assistant {
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
                Role::User => g4f::ChatMessage::User {
                    content: g4f::MessageContent::from(vec![new_part]),
                },
                Role::Assistant => g4f::ChatMessage::Assistant {
                    content: Some(g4f::MessageContent::from(vec![new_part])),
                    reasoning_content,
                    tool_calls: Vec::new(),
                },
                Role::System => g4f::ChatMessage::System {
                    content: g4f::MessageContent::from(vec![new_part]),
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

fn merge_settings_into_models(
    models: &mut HashMap<String, g4f::Model>,
    available_models: &[AvailableModel],
    context_window: Option<u64>,
) {
    for setting_model in available_models {
        if let Some(model) = models.get_mut(&setting_model.name) {
            if context_window.is_none() {
                model.max_tokens = setting_model.max_tokens;
            }
            if setting_model.display_name.is_some() {
                model.display_name = setting_model.display_name.clone();
            }
            if let Some(supports_tools) = setting_model.supports_tools {
                model.supports_tools = supports_tools;
            }
            if let Some(supports_images) = setting_model.supports_images {
                model.supports_images = supports_images;
            }
            if let Some(supports_thinking) = setting_model.supports_thinking {
                model.supports_thinking = supports_thinking;
            }
        } else {
            models.insert(
                setting_model.name.clone(),
                g4f::Model {
                    name: setting_model.name.clone(),
                    display_name: setting_model.display_name.clone(),
                    max_tokens: context_window.unwrap_or(setting_model.max_tokens),
                    supports_tools: setting_model.supports_tools.unwrap_or(false),
                    supports_images: setting_model.supports_images.unwrap_or(false),
                    supports_thinking: setting_model.supports_thinking.unwrap_or(false),
                },
            );
        }
    }
}

struct ConfigurationView {
    api_key_editor: Entity<InputField>,
    api_url_editor: Entity<InputField>,
    context_window_editor: Entity<InputField>,
    state: Entity<State>,
}

impl ConfigurationView {
    pub fn new(state: Entity<State>, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let api_key_editor = cx.new(|cx| InputField::new(window, cx, "sk-...").label("API key"));

        let api_url_editor = cx.new(|cx| {
            let input = InputField::new(window, cx, G4F_API_URL).label("API URL");
            input.set_text(&G4fLanguageModelProvider::api_url(cx), window, cx);
            input
        });

        let context_window_editor = cx.new(|cx| {
            let input = InputField::new(window, cx, "8192").label("Context Window");
            if let Some(context_window) = G4fLanguageModelProvider::settings(cx).context_window
            {
                input.set_text(&context_window.to_string(), window, cx);
            }
            input
        });

        cx.observe(&state, |_, _, cx| {
            cx.notify();
        })
        .detach();

        Self {
            api_key_editor,
            api_url_editor,
            context_window_editor,
            state,
        }
    }

    fn retry_connection(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let has_api_url = G4fLanguageModelProvider::has_custom_url(cx);
        let has_api_key = self
            .state
            .read_with(cx, |state, _| state.api_key_state.has_key());
        if !has_api_url {
            self.save_api_url(cx);
        }
        if !has_api_key {
            self.save_api_key(&Default::default(), window, cx);
        }

        self.state.update(cx, |state, cx| {
            state.restart_fetch_models_task(cx);
        });
    }

    fn save_api_key(&mut self, _: &menu::Confirm, window: &mut Window, cx: &mut Context<Self>) {
        let api_key = self.api_key_editor.read(cx).text(cx).trim().to_string();
        if api_key.is_empty() {
            return;
        }

        self.api_key_editor
            .update(cx, |input, cx| input.set_text("", window, cx));

        let state = self.state.clone();
        cx.spawn_in(window, async move |_, cx| {
            state
                .update(cx, |state, cx| state.set_api_key(Some(api_key), cx))
                .await
        })
        .detach_and_log_err(cx);
    }

    fn reset_api_key(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.api_key_editor
            .update(cx, |input, cx| input.set_text("", window, cx));

        let state = self.state.clone();
        cx.spawn_in(window, async move |_, cx| {
            state
                .update(cx, |state, cx| state.set_api_key(None, cx))
                .await
        })
        .detach_and_log_err(cx);

        cx.notify();
    }

    fn save_api_url(&self, cx: &mut Context<Self>) {
        let api_url = self.api_url_editor.read(cx).text(cx).trim().to_string();
        let current_url = G4fLanguageModelProvider::api_url(cx);
        if !api_url.is_empty() && &api_url != &current_url {
            let fs = <dyn Fs>::global(cx);
            update_settings_file(fs, cx, move |settings, _| {
                settings
                    .language_models
                    .get_or_insert_default()
                    .g4f
                    .get_or_insert_default()
                    .api_url = Some(api_url);
            });
        }
    }

    fn reset_api_url(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.api_url_editor
            .update(cx, |input, cx| input.set_text("", window, cx));
        let fs = <dyn Fs>::global(cx);
        update_settings_file(fs, cx, |settings, _cx| {
            if let Some(settings) = settings
                .language_models
                .as_mut()
                .and_then(|models| models.g4f.as_mut())
            {
                settings.api_url = Some(G4F_API_URL.into());
            }
        });
        cx.notify();
    }

    fn save_context_window(&mut self, cx: &mut Context<Self>) {
        let context_window_str = self
            .context_window_editor
            .read(cx)
            .text(cx)
            .trim()
            .to_string();
        let current_context_window = G4fLanguageModelProvider::settings(cx).context_window;

        if let Ok(context_window) = context_window_str.parse::<u64>() {
            if Some(context_window) != current_context_window {
                let fs = <dyn Fs>::global(cx);
                update_settings_file(fs, cx, move |settings, _| {
                    settings
                        .language_models
                        .get_or_insert_default()
                        .g4f
                        .get_or_insert_default()
                        .context_window = Some(context_window);
                });
            }
        } else if context_window_str.is_empty() && current_context_window.is_some() {
            let fs = <dyn Fs>::global(cx);
            update_settings_file(fs, cx, move |settings, _| {
                settings
                    .language_models
                    .get_or_insert_default()
                    .g4f
                    .get_or_insert_default()
                    .context_window = None;
            });
        }
    }

    fn reset_context_window(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.context_window_editor
            .update(cx, |input, cx| input.set_text("", window, cx));
        let fs = <dyn Fs>::global(cx);
        update_settings_file(fs, cx, |settings, _cx| {
            if let Some(settings) = settings
                .language_models
                .as_mut()
                .and_then(|models| models.g4f.as_mut())
            {
                settings.context_window = None;
            }
        });
        cx.notify();
    }

    fn render_instructions(cx: &App) -> Div {
        v_flex()
            .gap_2()
            .child(
                Label::new(
                    "Run g4f locally, or connect to a \
                remote g4f server.",
                )
                .color(Color::Muted),
            )
            .child(Label::new("To use a local G4F server:").color(Color::Muted))
            .child(
                List::new()
                    .child(
                        ListBulletItem::new("")
                            .child(Label::new("Install G4F from PyPi or source.").color(Color::Muted))
                            .child(ButtonLink::new("G4F", G4F_DOWNLOAD_URL)),
                    )
                    .child(
                        ListBulletItem::new("")
                            .child(
                                Label::new("Start the server:").color(Color::Muted),
                            )
                            .child(Label::new("python -m g4f --port 8080 --debug").inline_code(cx)),
                    )
                    .child(
                        ListBulletItem::new(
                            "Click 'Connect' below to start using G4F in Cognix",
                        )
                        .label_color(Color::Muted),
                    ),
            )
            .child(
                Label::new(
                    "Alternatively, you can connect to the default remote G4F server by leaving the \
                URL and API key as it is by defualt:",
                )
                .color(Color::Muted),
            )
    }

    fn render_api_key_editor(&self, cx: &Context<Self>) -> impl IntoElement {
        let state = self.state.read(cx);
        let env_var_set = state.api_key_state.is_from_env_var();
        let has_key = state.api_key_state.has_key();
        let using_fallback = !has_key && state.is_authenticated();

        let configured_card_label = if env_var_set {
            format!("API key set in {API_KEY_ENV_VAR_NAME} environment variable.")
        } else if using_fallback {
            "Using default API key as fallback".to_string()
        } else {
            "API key configured".to_string()
        };

        let api_key_control = if !has_key && !using_fallback {
            self.api_key_editor.clone().into_any_element()
        } else {
            ConfiguredApiCard::new("llama-cpp-clone-reset-key", configured_card_label)
                .disabled(env_var_set || using_fallback)
                .on_click(cx.listener(|this, _, window, cx| this.reset_api_key(window, cx)))
                .when(env_var_set, |this| {
                    this.tooltip_label(format!(
                        "To reset your API key, unset the {API_KEY_ENV_VAR_NAME} environment variable."
                    ))
                })
                .into_any_element()
        };

        v_flex()
            .on_action(cx.listener(Self::save_api_key))
            .child(api_key_control)
            .gap_1p5()
            .mb_2()
            .child(
                Label::new(format!(
                    "You can also set the {API_KEY_ENV_VAR_NAME} environment variable and restart Cognix."
                ))
                .size(LabelSize::Small)
                .color(Color::Muted),
            )
    }

    fn render_context_window_editor(&self, cx: &Context<Self>) -> Div {
        let settings = G4fLanguageModelProvider::settings(cx);
        let custom_context_window_set = settings.context_window.is_some();

        if custom_context_window_set {
            h_flex()
                .p_1()
                .justify_between()
                .rounded_md()
                .border_1()
                .border_color(cx.theme().colors().border_variant)
                .bg(cx.theme().colors().background.opacity(0.5))
                .child(
                    h_flex()
                        .gap_1()
                        .child(Icon::new(IconName::Check).color(Color::Success))
                        .child(Label::new(format!(
                            "Context Window: {}",
                            settings.context_window.unwrap_or_default()
                        ))),
                )
                .child(
                    Button::new("reset-context-window", "Reset")
                        .style(ButtonStyle::Outlined)
                        .label_size(LabelSize::Small)
                        .start_icon(Icon::new(IconName::Undo).size(IconSize::Small))
                        .on_click(
                            cx.listener(|this, _, window, cx| {
                                this.reset_context_window(window, cx)
                            }),
                        ),
                )
        } else {
            v_flex()
                .on_action(
                    cx.listener(|this, _: &menu::Confirm, _window, cx| {
                        this.save_context_window(cx)
                    }),
                )
                .child(self.context_window_editor.clone())
                .gap_1p5()
                .child(
                    Label::new("Default: Discovered from the server")
                        .size(LabelSize::Small)
                        .color(Color::Muted),
                )
        }
    }

    fn render_api_url_editor(&self, cx: &Context<Self>) -> Div {
        let api_url = G4fLanguageModelProvider::api_url(cx);
        let custom_api_url_set = api_url != G4F_API_URL;

        if custom_api_url_set {
            h_flex()
                .p_1()
                .justify_between()
                .rounded_md()
                .border_1()
                .border_color(cx.theme().colors().border_variant)
                .bg(cx.theme().colors().background.opacity(0.5))
                .child(
                    h_flex()
                        .gap_1()
                        .child(Icon::new(IconName::Check).color(Color::Success))
                        .child(Label::new(api_url)),
                )
                .child(
                    Button::new("reset-api-url", "Reset API URL")
                        .style(ButtonStyle::Outlined)
                        .label_size(LabelSize::Small)
                        .start_icon(Icon::new(IconName::Undo).size(IconSize::Small))
                        .on_click(
                            cx.listener(|this, _, window, cx| this.reset_api_url(window, cx)),
                        ),
                )
        } else {
            v_flex()
                .on_action(cx.listener(|this, _: &menu::Confirm, _window, cx| {
                    this.save_api_url(cx);
                    cx.notify();
                }))
                .gap_1p5()
                .child(self.api_url_editor.clone())
        }
    }
}

impl Render for ConfigurationView {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let is_authenticated = self.state.read(cx).is_authenticated();

        v_flex()
            .gap_2()
            .child(Headline::new("G4F").size(HeadlineSize::Small))
            .child(Self::render_instructions(cx))
            .child(self.render_api_url_editor(cx))
            .child(self.render_context_window_editor(cx))
            .child(self.render_api_key_editor(cx))
            .child(Divider::horizontal())
            .child(
                h_flex()
                    .pt_2()
                    .w_full()
                    .justify_between()
                    .gap_2()
                    .child(
                        h_flex()
                            .w_full()
                            .gap_2()
                            .map(|this| {
                                if is_authenticated {
                                    this.child(
                                        Button::new("llama-cpp-clone-webui", "Check API health")
                                            .style(ButtonStyle::OutlinedGhost)
                                            .size(ButtonSize::Medium)
                                            .end_icon(
                                                Icon::new(IconName::ArrowUpRight)
                                                    .size(IconSize::XSmall)
                                                    .color(Color::Muted),
                                            )
                                            .on_click(move |_, _, cx| {
                                                let url =
                                                    G4fLanguageModelProvider::api_url(cx);
                                                cx.open_url(&url);
                                            })
                                            .into_any_element(),
                                    )
                                    .child(
                                        Button::new("llama-cpp-clone-site", "G4F")
                                            .style(ButtonStyle::OutlinedGhost)
                                            .size(ButtonSize::Medium)
                                            .end_icon(
                                                Icon::new(IconName::ArrowUpRight)
                                                    .size(IconSize::XSmall)
                                                    .color(Color::Muted),
                                            )
                                            .on_click(move |_, _, cx| {
                                                cx.open_url(G4F_DOWNLOAD_URL)
                                            })
                                            .into_any_element(),
                                    )
                                } else {
                                    this.child(
                                        Button::new("download_llama_cpp_clone_button", "Get G4F")
                                            .style(ButtonStyle::OutlinedGhost)
                                            .size(ButtonSize::Medium)
                                            .end_icon(
                                                Icon::new(IconName::ArrowUpRight)
                                                    .size(IconSize::XSmall)
                                                    .color(Color::Muted),
                                            )
                                            .on_click(move |_, _, cx| {
                                                cx.open_url(G4F_DOWNLOAD_URL)
                                            })
                                            .into_any_element(),
                                    )
                                }
                            })
                            .child(
                                Button::new("view-models", "Browse G4F Models")
                                    .style(ButtonStyle::OutlinedGhost)
                                    .size(ButtonSize::Medium)
                                    .end_icon(
                                        Icon::new(IconName::ArrowUpRight)
                                            .size(IconSize::XSmall)
                                            .color(Color::Muted),
                                    )
                                    .on_click(move |_, _, cx| cx.open_url(G4F_MODELS_URL)),
                            ),
                    )
                    .map(|this| {
                        if is_authenticated {
                            this.child(
                                ButtonLike::new("connected")
                                    .size(ButtonSize::Medium)
                                    .child(
                                        h_flex()
                                            .gap_1()
                                            .child(Icon::new(IconName::Check).color(Color::Success))
                                            .child(Label::new("Connected")),
                                    )
                                    .child(
                                        IconButton::new("refresh-models", IconName::RotateCcw)
                                            .icon_size(IconSize::Small)
                                            .tooltip(Tooltip::text("Refresh Models"))
                                            .on_click(cx.listener(|this, _, window, cx| {
                                                this.state.update(cx, |state, _| {
                                                    state.fetched_models.clear();
                                                });
                                                this.retry_connection(window, cx);
                                            })),
                                    ),
                            )
                        } else {
                            this.child(
                                Button::new("retry_llama_cpp_clone_models", "Connect")
                                    .style(ButtonStyle::Outlined)
                                    .size(ButtonSize::Medium)
                                    .start_icon(
                                        Icon::new(IconName::PlayOutlined).size(IconSize::XSmall),
                                    )
                                    .on_click(cx.listener(move |this, _, window, cx| {
                                        this.retry_connection(window, cx)
                                    })),
                            )
                        }
                    }),
            )
    }
}
