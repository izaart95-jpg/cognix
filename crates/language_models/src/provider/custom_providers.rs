//! Server-sided custom providers. One generic provider implementation,
//! driven by a manifest (`providers.json`) fetched from a configurable URL,
//! replaces the per-provider crates that used to be compiled in for every
//! OpenAI-compatible endpoint.
//!
//! [`init`] owns the manifest lifecycle: it applies the last-good cached
//! manifest at startup, then fetches the manifest over the network and
//! re-applies it periodically, diffing it against what is currently
//! registered so server-side changes go live without an app update.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use collections::HashMap;
use credentials_provider::CredentialsProvider;
use custom_providers::{
    DiscoveredModel, ModelSource, ProviderConfig, ProvidersManifest, chat_completions_url,
    discovered_from_hardcoded, fetch_manifest, fetch_models, provider_id_for,
};
use fs::Fs;
use futures::{FutureExt, StreamExt, future::BoxFuture, stream::BoxStream};
use gpui::{App, AppContext, AsyncApp, Context, Entity, Global, SharedString, Task};
use http_client::{CustomHeaders, HttpClient};
use language_model::{
    ApiKeyConfiguration, ApiKeyState, AuthenticateError, EnvVar, IconOrSvg, LanguageModel,
    LanguageModelCompletionError, LanguageModelCompletionEvent, LanguageModelEffortLevel,
    LanguageModelId, LanguageModelName, LanguageModelProvider, LanguageModelProviderId,
    LanguageModelProviderName, LanguageModelProviderState, LanguageModelRegistry,
    LanguageModelRequest, LanguageModelToolChoice, ProviderSettingsView, RateLimiter,
};
use paths::data_dir;
use settings::{Settings, SettingsStore};
use ui::IconName;
use util::ResultExt;

use crate::AllLanguageModelSettings;
use crate::provider::openai_shim;

// Re-exported so `crate::provider::custom_providers` and the
// `custom_providers` crate resolve to the same items; the provider module
// shadows the crate name inside this crate.
pub use ::custom_providers::{DEFAULT_PROVIDERS_URL, DEFAULT_REFRESH_INTERVAL_MINUTES};

/// How long to wait before retrying a failed manifest fetch.
const FETCH_RETRY_INTERVAL: Duration = Duration::from_secs(30);

/// Resolved settings for the `custom_providers` section.
#[derive(Debug, Clone, PartialEq)]
pub struct CustomProvidersSettings {
    pub url: String,
    pub refresh_interval_minutes: u64,
}

impl CustomProvidersSettings {
    fn get(cx: &App) -> &Self {
        &AllLanguageModelSettings::get_global(cx).custom_providers
    }

    fn manifest_url(cx: &App) -> SharedString {
        let url = Self::get(cx).url.as_str();
        if url.is_empty() {
            custom_providers::DEFAULT_PROVIDERS_URL.into()
        } else {
            SharedString::new(url)
        }
    }

    fn refresh_interval(cx: &App) -> Duration {
        Duration::from_secs(Self::get(cx).refresh_interval_minutes * 60)
    }
}

/// Holds the registry state (and its fetch loop) alive for the app's
/// lifetime. The field is intentionally never read: the entity is captured
/// when the global is constructed, and dropping the global cancels the
/// loop, so its mere presence is the point.
#[allow(dead_code)]
struct GlobalCustomProvidersRegistry(Entity<CustomProvidersRegistryState>);

impl Global for GlobalCustomProvidersRegistry {}

/// Owns the manifest fetch loop and the set of currently registered dynamic
/// providers, keyed by manifest key.
pub struct CustomProvidersRegistryState {
    http_client: Arc<dyn HttpClient>,
    credentials_provider: Arc<dyn CredentialsProvider>,
    registry: Entity<LanguageModelRegistry>,
    registered: HashMap<String, Entity<CustomProviderState>>,
    fetch_task: Option<Task<()>>,
}

/// Live state of one dynamic provider: credentials and the model list.
pub struct CustomProviderState {
    base_url: SharedString,
    server_api_key: Option<String>,
    api_key_state: ApiKeyState,
    credentials_provider: Arc<dyn CredentialsProvider>,
    http_client: Arc<dyn HttpClient>,
    models_config: Vec<ModelSource>,
    fetched_models: Vec<DiscoveredModel>,
    fetch_task: Option<Task<()>>,
}

/// The provider registered with the [`LanguageModelRegistry`] for one
/// manifest entry.
pub struct CustomLanguageModelProvider {
    id: LanguageModelProviderId,
    name: LanguageModelProviderName,
    label: String,
    chat_completions_url: SharedString,
    http_client: Arc<dyn HttpClient>,
    state: Entity<CustomProviderState>,
}

pub struct CustomLanguageModel {
    id: LanguageModelId,
    provider_id: LanguageModelProviderId,
    provider_name: LanguageModelProviderName,
    name: String,
    display_name: String,
    capabilities: ModelCapabilities,
    /// Reasoning-effort values accepted by this model, with the default, if
    /// the manifest or model list declared them. Snapshotted at creation
    /// like `capabilities` so the trait methods need no context access.
    reasoning_efforts: Option<(Vec<String>, String)>,
    label: String,
    chat_completions_url: SharedString,
    http_client: Arc<dyn HttpClient>,
    request_limiter: RateLimiter,
    state: Entity<CustomProviderState>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct ModelCapabilities {
    supports_tools: bool,
    supports_images: bool,
    supports_thinking: bool,
    max_tokens: u64,
}

/// Starts the dynamic-provider lifecycle. Must run after the static
/// providers are registered so manifest keys colliding with built-in ids can
/// be detected.
pub fn init(
    registry: Entity<LanguageModelRegistry>,
    http_client: Arc<dyn HttpClient>,
    credentials_provider: Arc<dyn CredentialsProvider>,
    cx: &mut Context<LanguageModelRegistry>,
) {
    let state = cx.new(|cx| {
        cx.observe_global::<SettingsStore>({
            let mut last_url = CustomProvidersSettings::manifest_url(cx);
            let mut last_interval = CustomProvidersSettings::refresh_interval(cx);
            move |this: &mut CustomProvidersRegistryState, cx| {
                let url = CustomProvidersSettings::manifest_url(cx);
                let interval = CustomProvidersSettings::refresh_interval(cx);
                if url != last_url || interval != last_interval {
                    last_url = url;
                    last_interval = interval;
                    this.restart_fetch_loop(cx);
                }
            }
        })
        .detach();

        CustomProvidersRegistryState {
            http_client: http_client.clone(),
            credentials_provider: credentials_provider.clone(),
            registry: registry.clone(),
            registered: HashMap::default(),
            fetch_task: None,
        }
    });
    cx.set_global(GlobalCustomProvidersRegistry(state.clone()));

    state.update(cx, |state, cx| {
        state.load_cached_manifest(cx);
        state.restart_fetch_loop(cx);
    });
}

impl CustomProvidersRegistryState {
    /// Applies the last-good manifest cached on disk so providers exist
    /// before the network answers.
    fn load_cached_manifest(&mut self, cx: &mut Context<Self>) {
        let fs = <dyn Fs>::global(cx);
        let cache_path = manifest_cache_path();
        cx.spawn(async move |this, cx| match fs.load(&cache_path).await {
            Ok(contents) => {
                match custom_providers::parse_manifest(&contents) {
                    Ok(manifest) => {
                        this.update(cx, |this, cx| this.apply_manifest(manifest, cx))
                            .ok();
                    }
                    Err(error) => {
                        log::warn!("cached custom providers manifest is invalid: {error:#}");
                    }
                }
            }
            Err(error) => {
                // A missing cache on first launch is expected; only report
                // real read failures.
                if !cache_path.exists() {
                    return;
                }
                log::warn!("failed to read cached custom providers manifest: {error:#}");
            }
        })
        .detach();
    }

    fn restart_fetch_loop(&mut self, cx: &mut Context<Self>) {
        let http_client = self.http_client.clone();
        let url = CustomProvidersSettings::manifest_url(cx).to_string();
        let interval = CustomProvidersSettings::refresh_interval(cx);
        // Dropping the previous task cancels the in-flight fetch and its
        // timer chain.
        self.fetch_task = Some(cx.spawn(async move |this, cx| {
            loop {
                match fetch_manifest(http_client.as_ref(), &url).await {
                    Ok(manifest) => {
                        let raw = serde_json::to_string(&manifest).log_err();
                        if this
                            .update(cx, |this, cx| {
                                this.apply_manifest(manifest, cx);
                            })
                            .is_err()
                        {
                            return;
                        }
                        if let Some(raw) = raw {
                            let fs = cx.update(|cx| <dyn Fs>::global(cx));
                            let cache_path = manifest_cache_path();
                            cx.background_spawn(async move {
                                fs.atomic_write(cache_path, raw).await.log_err();
                            })
                            .detach();
                        }
                    }
                    Err(error) => {
                        log::warn!("failed to fetch custom providers manifest: {error:#}");
                        // Retry failures on a short interval so a transient
                        // outage recovers quickly; the long refresh cadence
                        // resumes after the next success.
                        cx.background_executor().timer(FETCH_RETRY_INTERVAL).await;
                        if this.update(cx, |_, _| ()).is_err() {
                            return;
                        }
                        continue;
                    }
                }

                // Zero interval means "fetch only at startup and on settings
                // changes"; the observer restarts the loop when that happens.
                if interval.is_zero() {
                    return;
                }
                cx.background_executor().timer(interval).await;
                if this.update(cx, |_, _| ()).is_err() {
                    return;
                }
            }
        }));
    }

    /// Diffs `manifest` against the registered set and applies the changes:
    /// new entries register, changed entries are replaced, removed or
    /// inactive entries unregister.
    fn apply_manifest(&mut self, manifest: ProvidersManifest, cx: &mut Context<Self>) {
        let registry = self.registry.clone();
        let mut seen: HashMap<&str, ()> = HashMap::default();

        for (key, config) in &manifest.providers {
            if !config.active || config.base_url.is_empty() {
                continue;
            }
            let provider_id = LanguageModelProviderId::from(provider_id_for(key));

            // A manifest entry must never shadow a built-in provider (e.g.
            // a `zen` key colliding with the static `cognix.zen`).
            if !self.registered.contains_key(key)
                && registry.read(cx).provider(&provider_id).is_some()
            {
                log::warn!(
                    "ignoring custom provider `{key}`: provider id `{provider_id}` is already registered"
                );
                continue;
            }

            seen.insert(key, ());

            let unchanged = self
                .registered
                .get(key)
                .is_some_and(|state| state.read(cx).config_matches(config));
            if unchanged {
                // Same provider: just refresh its model list so catalog
                // changes go live without re-registering.
                let state = self.registered.get(key).cloned();
                if let Some(state) = state {
                    state.update(cx, |state, cx| state.refresh_models(cx));
                }
            } else {
                if self.registered.contains_key(key) {
                    registry.update(cx, |registry, cx| {
                        registry.unregister_provider(provider_id.clone(), cx);
                    });
                }
                let provider = build_provider(
                    key,
                    config,
                    &self.http_client,
                    &self.credentials_provider,
                    cx,
                );
                self.registered.insert(key.clone(), provider.state.clone());
                registry.update(cx, |registry, cx| {
                    registry.register_provider(Arc::new(provider), cx);
                });
            }
        }

        // Unregister providers that disappeared or became inactive.
        let removed: Vec<String> = self
            .registered
            .keys()
            .filter(|key| !seen.contains_key(key.as_str()))
            .cloned()
            .collect();
        for key in removed {
            if let Some(state) = self.registered.remove(&key) {
                let provider_id = LanguageModelProviderId::from(provider_id_for(&key));
                registry.update(cx, |registry, cx| {
                    registry.unregister_provider(provider_id, cx);
                });
                state.update(cx, |state, cx| {
                    state.fetch_task = None;
                    cx.notify();
                });
            }
        }
    }
}

fn manifest_cache_path() -> PathBuf {
    data_dir().join("custom_providers.json")
}

fn build_provider(
    key: &str,
    config: &ProviderConfig,
    http_client: &Arc<dyn HttpClient>,
    credentials_provider: &Arc<dyn CredentialsProvider>,
    cx: &mut App,
) -> CustomLanguageModelProvider {
    let display_name = config.name.clone().unwrap_or_else(|| title_case(key));
    let base_url = SharedString::new(config.base_url.as_str());
    let endpoint = SharedString::new(chat_completions_url(&config.base_url));

    let state = cx.new(|_| CustomProviderState {
        base_url: base_url.clone(),
        server_api_key: config.api_key.clone(),
        api_key_state: ApiKeyState::new(
            base_url.clone(),
            EnvVar::new(env_var_name(key).into()),
        ),
        credentials_provider: credentials_provider.clone(),
        http_client: http_client.clone(),
        models_config: config.models.clone(),
        fetched_models: Vec::new(),
        fetch_task: None,
    });

    CustomLanguageModelProvider {
        id: LanguageModelProviderId::from(provider_id_for(key)),
        name: LanguageModelProviderName::from(display_name.clone()),
        label: display_name,
        chat_completions_url: endpoint,
        http_client: http_client.clone(),
        state,
    }
}

/// `NEXTROUTER_API_KEY`-style env var name derived from a manifest key.
fn env_var_name(key: &str) -> String {
    format!("{}_API_KEY", key.to_uppercase().replace(['.', '-'], "_"))
}

/// Title-cased display name fallback: `justwoker` → `Justwoker`,
/// `next-router` → `Next Router`.
fn title_case(key: &str) -> String {
    key.split(['-', '_', '.'])
        .map(|word| {
            let mut characters = word.chars();
            match characters.next() {
                Some(first) => first.to_uppercase().collect::<String>() + characters.as_str(),
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

impl CustomProviderState {
    fn config_matches(&self, config: &ProviderConfig) -> bool {
        self.base_url.as_ref() == config.base_url
            && self.server_api_key == config.api_key
            && self.models_config == config.models
    }

    /// The API key to use: a user-stored key (env var or system keychain)
    /// always wins over the manifest's fallback key.
    fn resolved_api_key(&self) -> Option<String> {
        self.api_key_state
            .key(&self.base_url)
            .map(|key| key.to_string())
            .or_else(|| self.server_api_key.clone())
    }

    fn hardcoded_models(&self) -> Vec<DiscoveredModel> {
        self.models_config
            .iter()
            .filter_map(|source| match source {
                ModelSource::Hardcoded(wrapper) => {
                    Some(discovered_from_hardcoded(&wrapper.hardcoded))
                }
                ModelSource::Model(model) => Some(discovered_from_hardcoded(model)),
                ModelSource::Url(_) => None,
            })
            .collect()
    }

    /// Fetched models when available, else the manifest's hardcoded list.
    fn effective_models(&self) -> Vec<DiscoveredModel> {
        if self.fetched_models.is_empty() {
            self.hardcoded_models()
        } else {
            self.fetched_models.clone()
        }
    }

    fn refresh_models(&mut self, cx: &mut Context<Self>) {
        let urls: Vec<String> = self
            .models_config
            .iter()
            .filter_map(|source| match source {
                ModelSource::Url(url) => Some(url.clone()),
                _ => None,
            })
            .collect();
        if urls.is_empty() {
            return;
        }

        let http_client = self.http_client.clone();
        let api_key = self.resolved_api_key();
        let provider_label = self.base_url.clone();
        self.fetch_task = Some(cx.spawn(async move |this, cx| {
            let url_refs: Vec<&str> = urls.iter().map(String::as_str).collect();
            match fetch_models(
                http_client.as_ref(),
                &url_refs,
                api_key.as_deref(),
                &CustomHeaders::default(),
            )
            .await
            {
                Ok(models) => {
                    this.update(cx, |this, cx| {
                        if !models.is_empty() {
                            this.fetched_models = models;
                        }
                        cx.notify();
                    })
                    .ok();
                }
                Err(error) => {
                    // Hardcoded entries (if any) keep serving as the list.
                    log::warn!("failed to fetch models from {provider_label}: {error:#}");
                }
            }
        }));
    }

    fn authenticate(&mut self, cx: &mut Context<Self>) -> Task<Result<(), AuthenticateError>> {
        let credentials_provider = self.credentials_provider.clone();
        let api_url = self.base_url.clone();
        self.api_key_state.load_if_needed(
            api_url,
            |this| &mut this.api_key_state,
            credentials_provider,
            cx,
        )
    }
}

impl LanguageModelProviderState for CustomLanguageModelProvider {
    type ObservableEntity = CustomProviderState;

    fn observable_entity(&self) -> Option<Entity<Self::ObservableEntity>> {
        Some(self.state.clone())
    }
}

impl LanguageModelProvider for CustomLanguageModelProvider {
    fn id(&self) -> LanguageModelProviderId {
        self.id.clone()
    }

    fn name(&self) -> LanguageModelProviderName {
        self.name.clone()
    }

    fn icon(&self) -> IconOrSvg {
        IconOrSvg::Icon(IconName::Cognix)
    }

    fn default_model(&self, cx: &App) -> Option<Arc<dyn LanguageModel>> {
        self.state
            .read(cx)
            .effective_models()
            .first()
            .map(|model| self.create_language_model(model))
    }

    fn default_fast_model(&self, cx: &App) -> Option<Arc<dyn LanguageModel>> {
        self.default_model(cx)
    }

    fn provided_models(&self, cx: &App) -> Vec<Arc<dyn LanguageModel>> {
        self.state
            .read(cx)
            .effective_models()
            .iter()
            .map(|model| self.create_language_model(model))
            .collect()
    }

    fn is_authenticated(&self, cx: &App) -> bool {
        !self.state.read(cx).effective_models().is_empty()
    }

    fn authenticate(&self, cx: &mut App) -> Task<Result<(), AuthenticateError>> {
        self.state.update(cx, |state, cx| {
            let task = state.authenticate(cx);
            state.refresh_models(cx);
            task
        })
    }

    fn settings_view(&self, cx: &mut App) -> Option<ProviderSettingsView> {
        let state = self.state.read(cx);
        Some(ProviderSettingsView::ApiKey(ApiKeyConfiguration::new(
            state.api_key_state.has_key(),
            state.api_key_state.is_from_env_var(),
            state.api_key_state.env_var_name().clone(),
            state.base_url.clone(),
        )))
    }

    fn set_api_key(&self, api_key: Option<String>, cx: &mut App) -> Task<Result<()>> {
        self.state.update(cx, |state, cx| {
            let credentials_provider = state.credentials_provider.clone();
            let api_url = state.base_url.clone();
            state.api_key_state.store(
                api_url,
                api_key,
                |this| &mut this.api_key_state,
                credentials_provider,
                cx,
            )
        })
    }
}

impl CustomLanguageModelProvider {
    fn create_language_model(&self, model: &DiscoveredModel) -> Arc<dyn LanguageModel> {
        let openai_model = &model.openai_model;
        let reasoning_efforts = if model.reasoning_efforts.is_empty() {
            None
        } else {
            let default = model
                .default_reasoning_effort
                .clone()
                .or_else(|| model.reasoning_efforts.first().cloned());
            default.map(|default| (model.reasoning_efforts.clone(), default))
        };
        Arc::new(CustomLanguageModel {
            id: LanguageModelId::from(openai_model.name.clone()),
            provider_id: self.id.clone(),
            provider_name: self.name.clone(),
            name: openai_model.name.clone(),
            display_name: openai_model.display_name().to_string(),
            capabilities: ModelCapabilities {
                supports_tools: openai_model.supports_tools,
                supports_images: openai_model.supports_images,
                supports_thinking: openai_model.supports_thinking,
                max_tokens: openai_model.max_tokens,
            },
            reasoning_efforts,
            label: self.label.clone(),
            chat_completions_url: self.chat_completions_url.clone(),
            http_client: self.http_client.clone(),
            request_limiter: RateLimiter::new(4),
            state: self.state.clone(),
        })
    }
}

impl LanguageModel for CustomLanguageModel {
    fn id(&self) -> LanguageModelId {
        self.id.clone()
    }

    fn name(&self) -> LanguageModelName {
        LanguageModelName::from(self.display_name.clone())
    }

    fn provider_id(&self) -> LanguageModelProviderId {
        self.provider_id.clone()
    }

    fn provider_name(&self) -> LanguageModelProviderName {
        self.provider_name.clone()
    }

    fn supports_tools(&self) -> bool {
        self.capabilities.supports_tools
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
        self.capabilities.supports_images
    }

    fn supports_thinking(&self) -> bool {
        self.capabilities.supports_thinking
    }

    fn supported_effort_levels(&self) -> Vec<LanguageModelEffortLevel> {
        let Some((efforts, default)) = &self.reasoning_efforts else {
            return Vec::new();
        };
        efforts
            .iter()
            .map(|value| LanguageModelEffortLevel {
                name: SharedString::from(title_case_level(value)),
                value: SharedString::from(value.as_str()),
                is_default: value == default,
            })
            .collect()
    }

    fn telemetry_id(&self) -> String {
        format!("{}/{}", self.provider_id, self.name)
    }

    fn max_token_count(&self) -> u64 {
        self.capabilities.max_tokens
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
        let supports_thinking = self.capabilities.supports_thinking
            || self.reasoning_efforts.is_some();
        let reasoning_effort = self.resolve_reasoning_effort(&request);
        let wire_request = match openai_shim::build_request(
            &self.name,
            self.capabilities.supports_images,
            openai_shim::RequestCapabilities {
                supports_tools: self.capabilities.supports_tools,
                supports_thinking,
            },
            request,
            &self.label,
            reasoning_effort,
        ) {
            Ok(request) => request,
            Err(error) => return async move { Err(error.into()) }.boxed(),
        };

        let api_key = self
            .state
            .read_with(cx, |state, _| state.resolved_api_key());
        let endpoint = self.chat_completions_url.clone();
        let http_client = self.http_client.clone();
        let label = self.label.clone();

        let future = self.request_limiter.stream(async move {
            let spec = openai_protocol::ProviderSpec {
                chat_completions_path: "".into(),
                label: label.into(),
            };
            // The base URL passed here is the full endpoint; the spec's path
            // is empty because `stream_chat_completion` concatenates the two.
            let stream = openai_protocol::stream_chat_completion(
                http_client.as_ref(),
                endpoint.as_ref(),
                api_key.as_deref(),
                wire_request,
                &CustomHeaders::default(),
                &spec,
            )
            .await?;
            Ok(stream)
        });

        async move {
            let mapper = openai_shim::ResponseStreamMapper::new();
            Ok(mapper.map_stream(future.await?.boxed()).boxed())
        }
        .boxed()
    }
}

impl CustomLanguageModel {
    /// Mirrors the reasoning-effort resolution of the previous glm/nim
    /// providers: the user's chosen effort when valid, else the configured
    /// default; nothing when thinking isn't allowed.
    fn resolve_reasoning_effort(&self, request: &LanguageModelRequest) -> Option<String> {
        let (efforts, default) = self.reasoning_efforts.clone()?;
        if !request.thinking_allowed {
            return None;
        }
        Some(
            request
                .thinking_effort
                .as_deref()
                .filter(|effort| efforts.iter().any(|level| level == *effort))
                .unwrap_or(&default)
                .to_string(),
        )
    }
}

fn title_case_level(value: &str) -> String {
    let mut characters = value.chars();
    match characters.next() {
        Some(first) => first.to_uppercase().collect::<String>() + characters.as_str(),
        None => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::TestAppContext;
    use language_model::LanguageModelRegistry;

    fn manifest_glm() -> ProvidersManifest {
        custom_providers::parse_manifest(
            r#"{"providers": {
                "glm": {
                    "baseUrl": "https://glm.example.com/v1",
                    "name": "Cognix-GLM",
                    "models": [
                        "https://glm.example.com/v1/models",
                        {"hardcoded": {"id": "glm-fallback", "contextWindow": 65536}}
                    ]
                }
            }}"#,
        )
        .unwrap()
    }

    fn registry_state(
        cx: &mut TestAppContext,
        http_client: Arc<dyn HttpClient>,
    ) -> Entity<CustomProvidersRegistryState> {
        cx.update(|cx| {
            let settings_store = ::settings::SettingsStore::test(cx);
            cx.set_global(settings_store);
            cx.set_global(db::AppDatabase::test_new());

            let registry = cx.new(|_| LanguageModelRegistry::default());
            cx.new(|_| CustomProvidersRegistryState {
                http_client,
                credentials_provider: fake_credentials_provider(),
                registry,
                registered: HashMap::default(),
                fetch_task: None,
            })
        })
    }

    fn fake_credentials_provider() -> Arc<dyn CredentialsProvider> {
        struct FakeCredentialsProvider;
        impl CredentialsProvider for FakeCredentialsProvider {
            fn read_credentials<'a>(
                &'a self,
                _url: &'a str,
                _cx: &'a gpui::AsyncApp,
            ) -> std::pin::Pin<
                Box<dyn Future<Output = Result<Option<(String, Vec<u8>)>>> + 'a>,
            > {
                Box::pin(async { Ok(None) })
            }

            fn write_credentials<'a>(
                &'a self,
                _url: &'a str,
                _username: &'a str,
                _password: &'a [u8],
                _cx: &'a gpui::AsyncApp,
            ) -> std::pin::Pin<Box<dyn Future<Output = Result<()>> + 'a>> {
                Box::pin(async { Ok(()) })
            }

            fn delete_credentials<'a>(
                &'a self,
                _url: &'a str,
                _cx: &'a gpui::AsyncApp,
            ) -> std::pin::Pin<Box<dyn Future<Output = Result<()>> + 'a>> {
                Box::pin(async { Ok(()) })
            }
        }
        Arc::new(FakeCredentialsProvider)
    }

    fn provider_ids(registry: &Entity<LanguageModelRegistry>, cx: &TestAppContext) -> Vec<String> {
        registry.read_with(cx, |registry, _| {
            registry
                .providers()
                .into_iter()
                .map(|provider| provider.id().0.to_string())
                .collect()
        })
    }

    #[gpui::test]
    fn test_manifest_applies_registers_and_unregisters(cx: &mut TestAppContext) {
        let http_client = http_client::FakeHttpClient::with_404_response();
        let state = registry_state(cx, http_client);
        let registry = state.read_with(cx, |state, _| state.registry.clone());

        // The manifest registers the `glm` provider.
        state.update(cx, |state, cx| state.apply_manifest(manifest_glm(), cx));
        assert_eq!(provider_ids(&registry, cx), vec!["cognix.glm".to_string()]);

        // An inactive entry alongside it changes nothing.
        let manifest_with_inactive = custom_providers::parse_manifest(
            r#"{"providers": {
                "glm": {"baseUrl": "https://glm.example.com/v1", "name": "Cognix-GLM"},
                "tokenrouter": {"active": false, "baseUrl": "https://api.tokenrouter.com/v1"}
            }}"#,
        )
        .unwrap();
        state.update(cx, |state, cx| {
            state.apply_manifest(manifest_with_inactive, cx)
        });
        assert_eq!(provider_ids(&registry, cx), vec!["cognix.glm".to_string()]);

        // An empty manifest unregisters everything.
        state.update(cx, |state, cx| {
            state.apply_manifest(ProvidersManifest::default(), cx)
        });
        assert!(provider_ids(&registry, cx).is_empty());
    }

    #[gpui::test]
    fn test_changed_entry_is_replaced(cx: &mut TestAppContext) {
        let http_client = http_client::FakeHttpClient::with_404_response();
        let state = registry_state(cx, http_client);
        let registry = state.read_with(cx, |state, _| state.registry.clone());

        state.update(cx, |state, cx| state.apply_manifest(manifest_glm(), cx));
        let first_state = state.read_with(cx, |state, _| state.registered["glm"].clone());

        // Same key, changed base URL: the provider must be replaced, not
        // reused.
        let changed = custom_providers::parse_manifest(
            r#"{"providers": {"glm": {"baseUrl": "https://glm-changed.example.com/v1", "models": []}}}"#,
        )
        .unwrap();
        state.update(cx, |state, cx| state.apply_manifest(changed, cx));
        let second_state = state.read_with(cx, |state, _| state.registered["glm"].clone());
        assert_ne!(first_state.entity_id(), second_state.entity_id());

        // The registry still holds exactly one provider for the key.
        assert_eq!(provider_ids(&registry, cx), vec!["cognix.glm".to_string()]);
    }

    #[gpui::test]
    fn test_unchanged_entry_keeps_its_provider_entity(cx: &mut TestAppContext) {
        let http_client = http_client::FakeHttpClient::with_404_response();
        let state = registry_state(cx, http_client);

        state.update(cx, |state, cx| state.apply_manifest(manifest_glm(), cx));
        let first_state = state.read_with(cx, |state, _| state.registered["glm"].clone());

        // Re-applying the same manifest must not replace the provider.
        state.update(cx, |state, cx| state.apply_manifest(manifest_glm(), cx));
        let second_state = state.read_with(cx, |state, _| state.registered["glm"].clone());
        assert_eq!(first_state.entity_id(), second_state.entity_id());
    }

    #[gpui::test]
    fn test_manifest_key_colliding_with_registered_provider_is_skipped(
        cx: &mut TestAppContext,
    ) {
        let http_client = http_client::FakeHttpClient::with_404_response();
        let state = registry_state(cx, http_client);
        let registry = state.read_with(cx, |state, _| state.registry.clone());

        // First manifest registers `zen` as a dynamic provider, which puts
        // `cognix.zen` in the registry.
        let manifest = custom_providers::parse_manifest(
            r#"{"providers": {"zen": {"baseUrl": "https://zen.example.com/v1"}}}"#,
        )
        .unwrap();
        state.update(cx, |state, cx| state.apply_manifest(manifest, cx));
        assert_eq!(provider_ids(&registry, cx), vec!["cognix.zen".to_string()]);

        // Simulate the id now belonging to a built-in provider: forget the
        // dynamic bookkeeping but keep the registry entry.
        state.update(cx, |state, _| {
            state.registered.remove("zen");
        });

        // A manifest with a `zen` key must be skipped rather than shadow it.
        let manifest = custom_providers::parse_manifest(
            r#"{"providers": {"zen": {"baseUrl": "https://zen.example.com/v1"}}}"#,
        )
        .unwrap();
        state.update(cx, |state, cx| state.apply_manifest(manifest, cx));
        assert_eq!(provider_ids(&registry, cx), vec!["cognix.zen".to_string()]);
        assert!(state.read_with(cx, |state, _| state.registered.is_empty()));
    }

    #[gpui::test]
    fn test_hardcoded_models_serve_as_fallback(cx: &mut TestAppContext) {
        let http_client = http_client::FakeHttpClient::with_404_response();
        let state = registry_state(cx, http_client);
        let registry = state.read_with(cx, |state, _| state.registry.clone());

        state.update(cx, |state, cx| state.apply_manifest(manifest_glm(), cx));

        // The models URL 404s (fake client), so the hardcoded entry is used.
        cx.run_until_parked();
        let models = registry.read_with(cx, |registry, cx| {
            registry
                .provider(&LanguageModelProviderId::from("cognix.glm".to_string()))
                .map(|provider| provider.provided_models(cx))
                .unwrap_or_default()
                .into_iter()
                .map(|model| model.id().0.to_string())
                .collect::<Vec<_>>()
        });
        assert_eq!(models, vec!["glm-fallback".to_string()]);
    }

    #[test]
    fn test_env_var_name_is_derived_from_key() {
        assert_eq!(env_var_name("nextrouter"), "NEXTROUTER_API_KEY");
        assert_eq!(env_var_name("justwoker"), "JUSTWOKER_API_KEY");
        assert_eq!(env_var_name("my.provider"), "MY_PROVIDER_API_KEY");
        assert_eq!(env_var_name("a-b"), "A_B_API_KEY");
    }

    #[test]
    fn test_title_case_falls_back_to_display_name() {
        assert_eq!(title_case("justwoker"), "Justwoker");
        assert_eq!(title_case("next-router"), "Next Router");
        assert_eq!(title_case("glm"), "Glm");
    }

    /// End-to-end: a manifest entry streams a completion through the
    /// OpenAI protocol, from `LanguageModelRequest` to Zed completion
    /// events, hitting the derived endpoint URL.
    #[gpui::test]
    async fn test_streams_completion_end_to_end(cx: &mut gpui::TestAppContext) {
        use language_model::{LanguageModelRequest, Role, MessageContent};

        let http_client = http_client::FakeHttpClient::create(|request| {
            let uri = request.uri().to_string();
            async move {
                let (status, body) = match uri.as_str() {
                    "https://glm.example.com/v1/chat/completions" => {
                        let sse = concat!(
                            "data: {\"model\":\"glm-5.2\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hel\"},\"finish_reason\":null}]}\n\n",
                            "data: {\"model\":\"glm-5.2\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"lo\"},\"finish_reason\":null}]}\n\n",
                            "data: {\"model\":\"glm-5.2\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":2,\"total_tokens\":5}}\n\n",
                            "data: [DONE]\n\n",
                        );
                        (200, sse.to_string())
                    }
                    _ => (404, String::new()),
                };
                Ok(http_client::http::Response::builder()
                    .status(status)
                    .body(http_client::AsyncBody::from(body))
                    .unwrap())
            }
        });

        let state = cx.update(|cx| {
            let settings_store = ::settings::SettingsStore::test(cx);
            cx.set_global(settings_store);
            cx.set_global(db::AppDatabase::test_new());
            let registry = cx.new(|_| LanguageModelRegistry::default());
            cx.new(|_| CustomProvidersRegistryState {
                http_client,
                credentials_provider: fake_credentials_provider(),
                registry,
                registered: HashMap::default(),
                fetch_task: None,
            })
        });

        let manifest = custom_providers::parse_manifest(
            r#"{"providers": {"glm": {"baseUrl": "https://glm.example.com/v1", "models": []}}}"#,
        )
        .unwrap();
        state.update(cx, |state, cx| state.apply_manifest(manifest, cx));

        let registry = state.read_with(cx, |state, _| state.registry.clone());

        // The first manifest declares no models, so the provider exposes none
        // (the fake server 404s any /models URL); re-apply with a hardcoded
        // model to make the provider usable.
        let manifest = custom_providers::parse_manifest(
            r#"{"providers": {"glm": {"baseUrl": "https://glm.example.com/v1",
                "models": [{"hardcoded": {"id": "glm-5.2", "reasoningEfforts": ["high"], "defaultReasoningEffort": "high"}}]}}}"#,
        )
        .unwrap();
        state.update(cx, |state, cx| state.apply_manifest(manifest, cx));

        let model = registry.read_with(cx, |registry, cx| {
            registry
                .provider(&LanguageModelProviderId::from("cognix.glm".to_string()))
                .and_then(|provider| provider.default_model(cx))
                .expect("hardcoded model should be exposed")
        });

        let events = cx
            .spawn(async move |cx| {
                let stream = model
                    .stream_completion(
                        LanguageModelRequest {
                            messages: vec![language_model::LanguageModelRequestMessage {
                                role: Role::User,
                                content: vec![MessageContent::Text("hi".to_string())],
                                cache: false,
                                reasoning_details: None,
                            }],
                            ..Default::default()
                        },
                        &cx,
                    )
                    .await?
                    .collect::<Vec<_>>()
                    .await;
                anyhow::Ok(stream)
            })
            .await
            .unwrap();

        let texts: Vec<String> = events
            .iter()
            .filter_map(|event| match event {
                Ok(LanguageModelCompletionEvent::Text(text)) => Some(text.clone()),
                Ok(LanguageModelCompletionEvent::Stop(
                    language_model::StopReason::EndTurn,
                )) => None,
                Ok(_) => None,
                Err(error) => panic!("stream error: {error:?}"),
            })
            .collect();
        assert_eq!(texts, vec!["Hel".to_string(), "lo".to_string()]);

        // The stream ends with a Stop event.
        assert!(events.iter().any(|event| matches!(
            event,
            Ok(LanguageModelCompletionEvent::Stop(language_model::StopReason::EndTurn))
        )));
    }
}
