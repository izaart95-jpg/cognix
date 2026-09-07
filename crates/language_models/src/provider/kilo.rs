use anyhow::Result;
use collections::HashMap;
use futures::{FutureExt, StreamExt, future::BoxFuture, stream::BoxStream};
use gpui::{App, AppContext, AsyncApp, Entity, SharedString, Task};
use http_client::{CustomHeaders, HttpClient};
use kilo::KILO_API_URL;
use language_model::{
    AuthenticateError, IconOrSvg, LanguageModel, LanguageModelCompletionError,
    LanguageModelCompletionEvent, LanguageModelEffortLevel, LanguageModelId, LanguageModelName,
    LanguageModelProvider, LanguageModelProviderId, LanguageModelProviderName,
    LanguageModelProviderState, LanguageModelRequest, LanguageModelToolChoice,
    ProviderSettingsView, RateLimiter,
};
pub use settings::LlamaCppAvailableModel as AvailableModel;
use settings::{Settings, SettingsStore};
use std::sync::Arc;

use ui::IconName;

use crate::provider::openai_shim;

const PROVIDER_ID: LanguageModelProviderId = LanguageModelProviderId::new("cognix.kilo");
const PROVIDER_NAME: LanguageModelProviderName = LanguageModelProviderName::new("Cognix-Kilo");

// ====================================================================
// Reasoning-effort configuration
// --------------------------------------------------------------------
// Only models that advertise `reasoning_effort` in their supported
// parameters receive the parameter on the wire; for everyone else it is
// omitted entirely.
// ====================================================================

const REASONING_EFFORT_LEVELS: &[(&str, &str)] =
    &[("low", "Low"), ("medium", "Medium"), ("high", "High")];

const DEFAULT_REASONING_EFFORT: &str = "medium";

fn supported_effort_levels() -> Vec<LanguageModelEffortLevel> {
    REASONING_EFFORT_LEVELS
        .iter()
        .map(|(value, label)| LanguageModelEffortLevel {
            name: (*label).into(),
            value: (*value).into(),
            is_default: *value == DEFAULT_REASONING_EFFORT,
        })
        .collect()
}

fn resolve_reasoning_effort(request: &LanguageModelRequest) -> Option<String> {
    if !request.thinking_allowed {
        return None;
    }

    let chosen = request
        .thinking_effort
        .as_deref()
        .filter(|effort| REASONING_EFFORT_LEVELS.iter().any(|(v, _)| *v == *effort))
        .unwrap_or(DEFAULT_REASONING_EFFORT);

    Some(chosen.to_string())
}

// ====================================================================
// Hardcoded models
// --------------------------------------------------------------------
// Free models are discovered from the gateway registry at startup; this
// list only covers the window before the first successful fetch (or its
// failure, e.g. while offline).
// ====================================================================
fn hardcoded_models() -> Vec<kilo::Model> {
    vec![
        kilo::Model {
            name: "nvidia/nemotron-3-super-120b-a12b:free".to_string(),
            display_name: None,
            max_tokens: 262_144,
            supports_tools: true,
            supports_images: false,
            supports_thinking: true,
            supports_reasoning_effort: true,
        },
        kilo::Model {
            name: "stepfun/step-3.7-flash:free".to_string(),
            display_name: None,
            max_tokens: 262_144,
            supports_tools: true,
            supports_images: true,
            supports_thinking: true,
            supports_reasoning_effort: false,
        },
    ]
}

#[derive(Default, Debug, Clone, PartialEq)]
pub struct KiloSettings {
    pub api_url: String,
    pub available_models: Vec<AvailableModel>,
    pub custom_headers: CustomHeaders,
}

pub struct KiloLanguageModelProvider {
    http_client: Arc<dyn HttpClient>,
    state: Entity<State>,
}

pub struct State {
    /// Free models from the gateway registry; empty until the first
    /// successful fetch.
    fetched_models: Vec<kilo::Model>,
}

impl KiloLanguageModelProvider {
    pub fn new(http_client: Arc<dyn HttpClient>, cx: &mut App) -> Self {
        let state = cx.new(|cx| {
            cx.observe_global::<SettingsStore>(|_this: &mut State, cx| {
                cx.notify();
            })
            .detach();
            State {
                fetched_models: Vec::new(),
            }
        });

        // Fetch the free-model registry in the background and cache it in State.
        let fetch_client = http_client.clone();
        let fetch_state = state.clone();
        cx.spawn(
            async move |cx| match kilo::fetch_models(fetch_client.as_ref()).await {
                Ok(models) if !models.is_empty() => {
                    let _ = cx.update(|cx| {
                        fetch_state.update(cx, |state, cx| {
                            state.fetched_models = models;
                            cx.notify();
                        })
                    });
                }
                Ok(_) => {
                    log::warn!("Kilo model registry returned no free models; using fallback models")
                }
                Err(error) => {
                    log::warn!("failed to fetch Kilo models: {error:#}; using fallback models")
                }
            },
        )
        .detach();
        Self { http_client, state }
    }

    fn create_language_model(&self, model: kilo::Model) -> Arc<dyn LanguageModel> {
        Arc::new(KiloLanguageModel {
            id: LanguageModelId::from(model.name.clone()),
            name: model.name.clone(),
            display_name: kilo::display_name_for(&model).to_string(),
            supports_tools: model.supports_tools,
            supports_images: model.supports_images,
            supports_thinking: model.supports_thinking,
            supports_reasoning_effort: model.supports_reasoning_effort,
            max_tokens: model.max_tokens,
            http_client: self.http_client.clone(),
            request_limiter: RateLimiter::new(4),
            state: self.state.clone(),
        })
    }

    /// Registry models (response order preserved), falling back to the
    /// hardcoded list until the fetch succeeds.
    fn models(&self, cx: &App) -> Vec<kilo::Model> {
        let fetched = self.state.read(cx).fetched_models.clone();
        if fetched.is_empty() {
            hardcoded_models()
        } else {
            fetched
        }
    }

    fn settings(cx: &App) -> &KiloSettings {
        &crate::AllLanguageModelSettings::get_global(cx).kilo
    }

    fn api_url(cx: &App) -> SharedString {
        let api_url = &Self::settings(cx).api_url;
        if api_url.is_empty() {
            KILO_API_URL.into()
        } else {
            SharedString::new(api_url.as_str())
        }
    }
}

impl LanguageModelProviderState for KiloLanguageModelProvider {
    type ObservableEntity = State;

    fn observable_entity(&self) -> Option<Entity<Self::ObservableEntity>> {
        Some(self.state.clone())
    }
}

impl LanguageModelProvider for KiloLanguageModelProvider {
    fn id(&self) -> LanguageModelProviderId {
        PROVIDER_ID
    }

    fn name(&self) -> LanguageModelProviderName {
        PROVIDER_NAME
    }

    fn icon(&self) -> IconOrSvg {
        IconOrSvg::Icon(IconName::Cognix)
    }

    fn default_model(&self, cx: &App) -> Option<Arc<dyn LanguageModel>> {
        self.models(cx)
            .into_iter()
            .next()
            .map(|model| self.create_language_model(model))
    }

    fn default_fast_model(&self, cx: &App) -> Option<Arc<dyn LanguageModel>> {
        self.models(cx)
            .into_iter()
            .next()
            .map(|model| self.create_language_model(model))
    }

    fn provided_models(&self, cx: &App) -> Vec<Arc<dyn LanguageModel>> {
        let settings = Self::settings(cx);
        let mut models: HashMap<String, kilo::Model> = HashMap::default();
        for model in self.models(cx) {
            models.insert(model.name.clone(), model);
        }

        for setting_model in &settings.available_models {
            if let Some(model) = models.get_mut(&setting_model.name) {
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
                model.max_tokens = setting_model.max_tokens;
            } else {
                models.insert(
                    setting_model.name.clone(),
                    kilo::Model {
                        name: setting_model.name.clone(),
                        display_name: setting_model.display_name.clone(),
                        max_tokens: setting_model.max_tokens,
                        supports_tools: setting_model.supports_tools.unwrap_or(true),
                        supports_images: setting_model.supports_images.unwrap_or(false),
                        supports_thinking: setting_model.supports_thinking.unwrap_or(true),
                        supports_reasoning_effort: true,
                    },
                );
            }
        }

        let mut models = models.into_values().collect::<Vec<_>>();
        models.sort_by_key(|model| model.name.clone());
        models
            .into_iter()
            .map(|model| self.create_language_model(model))
            .collect()
    }

    fn is_authenticated(&self, _cx: &App) -> bool {
        true
    }

    fn authenticate(&self, _cx: &mut App) -> Task<Result<(), AuthenticateError>> {
        // Kilo's free models require no credentials.
        Task::ready(Ok(()))
    }

    fn settings_view(&self, _cx: &mut App) -> Option<ProviderSettingsView> {
        None
    }

    fn set_api_key(&self, _api_key: Option<String>, _cx: &mut App) -> Task<Result<()>> {
        // Kilo's free models require no credentials.
        Task::ready(Ok(()))
    }
}

pub struct KiloLanguageModel {
    id: LanguageModelId,
    name: String,
    display_name: String,
    supports_tools: bool,
    supports_images: bool,
    supports_thinking: bool,
    supports_reasoning_effort: bool,
    max_tokens: u64,
    http_client: Arc<dyn HttpClient>,
    request_limiter: RateLimiter,
    state: Entity<State>,
}

impl KiloLanguageModel {
    fn to_kilo_request(
        &self,
        request: LanguageModelRequest,
    ) -> Result<kilo::ChatCompletionRequest> {
        let reasoning_effort =
            resolve_reasoning_effort(&request).filter(|_| self.supports_reasoning_effort);
        openai_shim::build_request(
            &self.name,
            self.supports_images,
            openai_shim::RequestCapabilities {
                // Kilo's free models accept tool definitions whether or not the
                // registry advertises them, so tools are never filtered out.
                supports_tools: true,
                supports_thinking: true,
            },
            request,
            "Kilo",
            reasoning_effort,
        )
    }
    fn stream_completion(
        &self,
        request: kilo::ChatCompletionRequest,
        cx: &AsyncApp,
    ) -> BoxFuture<
        'static,
        Result<futures::stream::BoxStream<'static, Result<kilo::ResponseStreamEvent>>>,
    > {
        let http_client = self.http_client.clone();
        let (api_url, extra_headers) = self.state.read_with(cx, |_, cx| {
            (
                KiloLanguageModelProvider::api_url(cx),
                KiloLanguageModelProvider::settings(cx)
                    .custom_headers
                    .clone(),
            )
        });

        let future = self.request_limiter.stream(async move {
            let stream = kilo::stream_chat_completion(
                http_client.as_ref(),
                &api_url,
                request,
                &extra_headers,
            )
            .await?;
            Ok(stream)
        });

        async move { Ok(future.await?.boxed()) }.boxed()
    }
}

impl LanguageModel for KiloLanguageModel {
    fn id(&self) -> LanguageModelId {
        self.id.clone()
    }

    fn name(&self) -> LanguageModelName {
        LanguageModelName::from(self.display_name.clone())
    }

    fn provider_id(&self) -> LanguageModelProviderId {
        PROVIDER_ID
    }

    fn provider_name(&self) -> LanguageModelProviderName {
        PROVIDER_NAME
    }

    fn supports_tools(&self) -> bool {
        self.supports_tools
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
        self.supports_thinking
    }

    fn supported_effort_levels(&self) -> Vec<LanguageModelEffortLevel> {
        supported_effort_levels()
    }

    fn telemetry_id(&self) -> String {
        format!("{PROVIDER_ID}/{}", self.name)
    }

    fn max_token_count(&self) -> u64 {
        self.max_tokens
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
        let request = match self.to_kilo_request(request) {
            Ok(request) => request,
            Err(error) => return async move { Err(error.into()) }.boxed(),
        };
        let completions = self.stream_completion(request, cx);
        async move {
            let mapper = openai_shim::ResponseStreamMapper::new();
            Ok(mapper.map_stream(completions.await?).boxed())
        }
        .boxed()
    }
}
