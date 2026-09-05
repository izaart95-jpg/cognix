use anyhow::Result;
use collections::HashMap;
use credentials_provider::CredentialsProvider;
use futures::{FutureExt, StreamExt, future::BoxFuture};
use gpui::{App, AppContext, AsyncApp, Context, Entity, SharedString, Task};
use http_client::{CustomHeaders, HttpClient};
use language_model::{
    ApiKeyConfiguration, ApiKeyState, AuthenticateError, EnvVar, IconOrSvg, LanguageModel,
    LanguageModelCompletionError, LanguageModelCompletionEvent, LanguageModelId, LanguageModelName,
    LanguageModelProvider, LanguageModelProviderId, LanguageModelProviderName,
    LanguageModelProviderState, LanguageModelRequest, LanguageModelToolChoice,
    ProviderSettingsView, RateLimiter, env_var,
};
pub use settings::LlamaCppAvailableModel as AvailableModel;
use settings::{Settings, SettingsStore};
use std::sync::{Arc, LazyLock};

use ui::IconName;

use crate::provider::openai_shim;

const PROVIDER_ID: LanguageModelProviderId = LanguageModelProviderId::new("cognix.zen");
const PROVIDER_NAME: LanguageModelProviderName = LanguageModelProviderName::new("Zen");

const API_KEY_ENV_VAR_NAME: &str = "ZEN_API_KEY";
static API_KEY_ENV_VAR: LazyLock<EnvVar> = env_var!(API_KEY_ENV_VAR_NAME);

/// Free-tier models that work without an API key. The upstream only accepts
/// model ids ending in `-free` for the anonymous `public` key.
fn hardcoded_models() -> Vec<zen::Model> {
    vec![
        zen::Model::new("big-pickle", None, None, true, false, true),
        zen::Model::new("mimo-v2.5-free", None, None, true, false, true),
        zen::Model::new("nemotron-3.5-lightning-free", None, None, true, false, true),
    ]
}

#[derive(Default, Debug, Clone, PartialEq)]
pub struct ZenSettings {
    pub api_url: String,
    pub available_models: Vec<AvailableModel>,
    pub custom_headers: CustomHeaders,
}

pub struct ZenLanguageModelProvider {
    http_client: Arc<dyn HttpClient>,
    state: Entity<State>,
}

pub struct State {
    api_key_state: ApiKeyState,
    credentials_provider: Arc<dyn CredentialsProvider>,
    /// Models from the upstream registry; empty until the first successful fetch.
    fetched_models: Vec<zen::Model>,
}

impl State {
    fn set_api_key(&mut self, api_key: Option<String>, cx: &mut Context<Self>) -> Task<Result<()>> {
        let credentials_provider = self.credentials_provider.clone();
        let api_url = ZenLanguageModelProvider::api_url(cx);
        self.api_key_state.store(
            api_url,
            api_key,
            |this| &mut this.api_key_state,
            credentials_provider,
            cx,
        )
    }

    fn authenticate(&mut self, cx: &mut Context<Self>) -> Task<Result<(), AuthenticateError>> {
        let credentials_provider = self.credentials_provider.clone();
        let api_url = ZenLanguageModelProvider::api_url(cx);
        self.api_key_state.load_if_needed(
            api_url,
            |this| &mut this.api_key_state,
            credentials_provider,
            cx,
        )
    }
}

impl ZenLanguageModelProvider {
    pub fn new(
        http_client: Arc<dyn HttpClient>,
        credentials_provider: Arc<dyn CredentialsProvider>,
        cx: &mut App,
    ) -> Self {
        let state = cx.new(|cx| {
            cx.observe_global::<SettingsStore>(|this: &mut State, cx| {
                let credentials_provider = this.credentials_provider.clone();
                let api_url = Self::api_url(cx);
                this.api_key_state.handle_url_change(
                    api_url,
                    |this| &mut this.api_key_state,
                    credentials_provider,
                    cx,
                );
                cx.notify();
            })
            .detach();
            State {
                api_key_state: ApiKeyState::new(Self::api_url(cx), (*API_KEY_ENV_VAR).clone()),
                credentials_provider,
                fetched_models: Vec::new(),
            }
        });

        // Fetch the upstream model list in the background and cache it in
        // State; the anonymous "public" key lists at least the free models.
        let fetch_client = http_client.clone();
        let fetch_state = state.clone();
        cx.spawn(async move |cx| {
            let (api_url, extra_headers) = cx.update(|cx| {
                let api_url = ZenLanguageModelProvider::api_url(cx).to_string();
                let extra_headers = ZenLanguageModelProvider::settings(cx)
                    .custom_headers
                    .clone();
                (api_url, extra_headers)
            });

            match zen::fetch_models(
                fetch_client.as_ref(),
                &api_url,
                zen::PUBLIC_API_KEY,
                &extra_headers,
            )
            .await
            {
                Ok(models) if !models.is_empty() => {
                    let _ = cx.update(|cx| {
                        fetch_state.update(cx, |state, cx| {
                            state.fetched_models = models;
                            cx.notify();
                        })
                    });
                }
                Ok(_) => log::warn!("Zen model registry is empty; using fallback models"),
                Err(error) => {
                    log::warn!("failed to fetch Zen models: {error:#}; using fallback models")
                }
            }
        })
        .detach();

        Self { http_client, state }
    }

    fn create_language_model(&self, model: zen::Model) -> Arc<dyn LanguageModel> {
        Arc::new(ZenLanguageModel {
            id: LanguageModelId::from(model.name.clone()),
            name: model.name.clone(),
            display_name: model.display_name().to_string(),
            supports_tools: model.supports_tools,
            supports_images: model.supports_images,
            max_tokens: model.max_tokens,
            http_client: self.http_client.clone(),
            request_limiter: RateLimiter::new(4),
            state: self.state.clone(),
        })
    }

    /// Upstream models (response order preserved), falling back to the
    /// hardcoded free list until the fetch succeeds.
    fn models(&self, cx: &App) -> Vec<zen::Model> {
        let fetched = self.state.read(cx).fetched_models.clone();
        if fetched.is_empty() {
            hardcoded_models()
        } else {
            fetched
        }
    }

    fn settings(cx: &App) -> &ZenSettings {
        &crate::AllLanguageModelSettings::get_global(cx).zen
    }

    fn api_url(cx: &App) -> SharedString {
        let api_url = &Self::settings(cx).api_url;
        if api_url.is_empty() {
            zen::ZEN_API_URL.into()
        } else {
            SharedString::new(api_url.as_str())
        }
    }
}

impl LanguageModelProviderState for ZenLanguageModelProvider {
    type ObservableEntity = State;

    fn observable_entity(&self) -> Option<Entity<Self::ObservableEntity>> {
        Some(self.state.clone())
    }
}

impl LanguageModelProvider for ZenLanguageModelProvider {
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
        let mut models: HashMap<String, zen::Model> = HashMap::default();
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
                    zen::Model {
                        name: setting_model.name.clone(),
                        display_name: setting_model.display_name.clone(),
                        max_tokens: setting_model.max_tokens,
                        supports_tools: setting_model.supports_tools.unwrap_or(true),
                        supports_images: setting_model.supports_images.unwrap_or(false),
                        supports_thinking: setting_model.supports_thinking.unwrap_or(true),
                        supports_reasoning_effort: false,
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
        // Anonymous access with the "public" key always works for free-tier
        // models.
        true
    }

    fn authenticate(&self, cx: &mut App) -> Task<Result<(), AuthenticateError>> {
        self.state.update(cx, |state, cx| state.authenticate(cx))
    }

    fn settings_view(&self, cx: &mut App) -> Option<ProviderSettingsView> {
        let state = self.state.read(cx);
        Some(ProviderSettingsView::ApiKey(ApiKeyConfiguration::new(
            state.api_key_state.has_key(),
            state.api_key_state.is_from_env_var(),
            state.api_key_state.env_var_name().clone(),
            zen::ZEN_API_URL.into(),
        )))
    }

    fn set_api_key(&self, api_key: Option<String>, cx: &mut App) -> Task<Result<()>> {
        self.state
            .update(cx, |state, cx| state.set_api_key(api_key, cx))
    }
}

pub struct ZenLanguageModel {
    id: LanguageModelId,
    name: String,
    display_name: String,
    supports_tools: bool,
    supports_images: bool,
    max_tokens: u64,
    http_client: Arc<dyn HttpClient>,
    request_limiter: RateLimiter,
    state: Entity<State>,
}

impl ZenLanguageModel {
    fn to_zen_request(&self, request: LanguageModelRequest) -> Result<zen::ChatCompletionRequest> {
        openai_shim::build_request(
            &self.name,
            self.supports_images,
            openai_shim::RequestCapabilities {
                supports_tools: true,
                supports_thinking: true,
            },
            request,
            "Zen",
            None,
        )
    }

    fn stream_completion(
        &self,
        request: zen::ChatCompletionRequest,
        cx: &AsyncApp,
    ) -> BoxFuture<
        'static,
        Result<futures::stream::BoxStream<'static, Result<zen::ResponseStreamEvent>>>,
    > {
        let http_client = self.http_client.clone();
        let (api_key, api_url, extra_headers) = self.state.read_with(cx, |state, cx| {
            let api_url = ZenLanguageModelProvider::api_url(cx);
            let extra_headers = ZenLanguageModelProvider::settings(cx)
                .custom_headers
                .clone();
            // Fall back to the anonymous key so free-tier models work
            // without any configuration.
            (
                state
                    .api_key_state
                    .key(&api_url)
                    .or_else(|| Some(SharedString::from(zen::PUBLIC_API_KEY).into())),
                api_url,
                extra_headers,
            )
        });

        let future = self.request_limiter.stream(async move {
            let stream = zen::stream_chat_completion(
                http_client.as_ref(),
                &api_url,
                api_key.as_deref().unwrap_or(zen::PUBLIC_API_KEY),
                request,
                &extra_headers,
            )
            .await?;
            Ok(stream)
        });

        async move { Ok(future.await?.boxed()) }.boxed()
    }
}

impl LanguageModel for ZenLanguageModel {
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

    fn supports_tool_choice(&self, _choice: LanguageModelToolChoice) -> bool {
        self.supports_tools()
    }

    fn supports_images(&self) -> bool {
        self.supports_images
    }

    fn supports_thinking(&self) -> bool {
        true
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
            futures::stream::BoxStream<
                'static,
                Result<LanguageModelCompletionEvent, LanguageModelCompletionError>,
            >,
            LanguageModelCompletionError,
        >,
    > {
        let request = match self.to_zen_request(request) {
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
