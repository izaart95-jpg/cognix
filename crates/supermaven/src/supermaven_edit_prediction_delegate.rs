//! The `EditPredictionDelegate` implementation backed by Supermaven.

use std::ops::Range;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use edit_prediction_types::{
    DataCollectionState, EditPrediction, EditPredictionDelegate, EditPredictionDiscardReason,
    EditPredictionIconSet, EditPredictionRequestTrigger,
};
use gpui::{App, AppContext as _, AsyncApp, Context, Entity, Task};
use http_client::HttpClient;
use icons::IconName;
use language::{Anchor, Buffer, EditPreview, ToOffset};

use crate::agent::{DerivedCompletion, SmAgent};
use crate::binary_fetcher;

/// How long we keep polling the streamed states for more items after the last
/// refresh. The first websocket connect takes ~6 seconds, so a short window
/// would miss the very first completion.
const POLL_WINDOW: Duration = Duration::from_secs(5);

/// How often we re-derive the visible completion while the model is streaming.
const POLL_INTERVAL: Duration = Duration::from_millis(25);

/// The currently-visible suggestion from the perspective of a buffer.
#[derive(Clone)]
struct CurrentSuggestion {
    snapshot: language::BufferSnapshot,
    range: Range<Anchor>,
    text: Arc<str>,
    edit_preview: EditPreview,
}

pub struct SupermavenEditPredictionDelegate {
    http_client: Arc<dyn HttpClient>,
    agent: Option<SmAgent>,
    /// How many times a dead agent has been restarted, for diagnostics.
    agent_restart_count: usize,
    current_suggestion: Option<CurrentSuggestion>,
    pending_refresh: Option<Task<Result<()>>>,
}

impl SupermavenEditPredictionDelegate {
    pub fn new(http_client: Arc<dyn HttpClient>) -> Self {
        Self {
            http_client,
            agent: None,
            agent_restart_count: 0,
            current_suggestion: None,
            pending_refresh: None,
        }
    }

    /// Returns a task that resolves to a running agent, starting one if this
    /// is the first call or restarting one that died (the agent owns its own
    /// credential file, so a restart needs no re-activation).
    fn agent(&mut self, cx: &mut Context<Self>) -> Task<Result<SmAgent>> {
        if let Some(agent) = self.agent.clone() {
            if agent.is_running() {
                return Task::ready(Ok(agent));
            }
            log::info!("Supermaven agent exited; restarting it");
            self.agent = None;
            self.agent_restart_count += 1;
        }
        let http_client = self.http_client.clone();
        cx.background_spawn(async move {
            let binary_path = binary_fetcher::ensure_binary(http_client.as_ref()).await?;
            SmAgent::start(&binary_path).await
        })
    }

    fn refresh(
        &mut self,
        buffer: Entity<Buffer>,
        cursor_position: Anchor,
        debounce_duration: Duration,
        cx: &mut Context<Self>,
    ) {
        let agent_task = self.agent(cx);
        self.pending_refresh = Some(cx.spawn(async move |this, cx| {
            if !debounce_duration.is_zero() {
                cx.background_executor().timer(debounce_duration).await;
            }

            let agent = agent_task.await?;

            // Cache the running agent so future refreshes skip the download
            // and spawn.
            this.update(cx, |this, _| {
                this.agent.get_or_insert_with(|| agent.clone());
            })?;

            let (path, full_text, prefix, cursor_offset) = buffer
                .read_with(cx, |buffer, cx| {
                    let snapshot = buffer.snapshot();
                    let full_text = snapshot.text();
                    let cursor_offset = cursor_position.to_offset(&snapshot);
                    let prefix: String = snapshot.text_for_range(0..cursor_offset).collect();
                    let path = buffer
                        .file()
                        .and_then(|file| file.as_local())
                        .map(|local_file| local_file.abs_path(cx));
                    (path, full_text, prefix, cursor_offset)
                });

            let Some(path) = path else {
                anyhow::bail!("Supermaven only supports local files");
            };

            if full_text.len() > binary_fetcher::HARD_SIZE_LIMIT {
                anyhow::bail!(
                    "file is {} bytes, over the {} byte limit",
                    full_text.len(),
                    binary_fetcher::HARD_SIZE_LIMIT
                );
            }

            agent
                .submit_state(&path, &prefix, &full_text, cursor_offset)
                .await?;

            // Poll while the model streams, so ghost text grows as chunks
            // arrive. Stop when the window lapses or every state has ended.
            let poll_deadline = Instant::now() + POLL_WINDOW;
            loop {
                // Always render at least once: a deduped submission reuses
                // an already-complete state whose items still hold a
                // completion worth showing.
                let derived = agent.derive_completion(&prefix);
                let suggestion =
                    build_suggestion(&buffer, &cursor_position, derived, cx).await;

                this.update(cx, |this, cx| {
                    this.current_suggestion = suggestion;
                    cx.notify();
                })?;

                if !agent.is_streaming() || Instant::now() > poll_deadline {
                    break;
                }
                cx.background_executor().timer(POLL_INTERVAL).await;
            }

            anyhow::Ok(())
        }));
    }
}

/// Turns a derived completion into a suggestion, or `None` when the buffer
/// has moved on (e.g. a `dedent` whose deletion range is no longer
/// whitespace) or the buffer was released.
async fn build_suggestion(
    buffer: &Entity<Buffer>,
    cursor_position: &Anchor,
    derived: Option<DerivedCompletion>,
    cx: &mut AsyncApp,
) -> Option<CurrentSuggestion> {
    let derived = derived?;

    let edits = buffer.read_with(cx, |buffer, _cx| {
        let snapshot = buffer.snapshot();
        let cursor_offset = cursor_position.to_offset(&snapshot);
        let range_start = cursor_offset.saturating_sub(derived.prior_delete);

        // When the model asked us to delete backwards (`dedent`), the
        // characters in that range should be whitespace; otherwise the
        // buffer no longer matches what the model predicted.
        if derived.prior_delete > 0 {
            let deletion_range_text: String =
                snapshot.text_for_range(range_start..cursor_offset).collect();
            if !deletion_range_text.chars().all(char::is_whitespace) {
                return None;
            }
        }

        let edits: Arc<[(Range<Anchor>, Arc<str>)]> = vec![(
            snapshot.anchor_after(range_start)..snapshot.anchor_after(cursor_offset),
            Arc::from(derived.text.as_str()),
        )]
        .into();
        Some(edits)
    })?;

    let edit_preview = buffer
        .update(cx, |buffer, cx| buffer.preview_edits(edits.clone(), cx))
        .await;

    let snapshot = buffer.read_with(cx, |buffer, _| buffer.snapshot());

    Some(CurrentSuggestion {
        snapshot,
        range: edits[0].0.clone(),
        text: edits[0].1.clone(),
        edit_preview,
    })
}

impl EditPredictionDelegate for SupermavenEditPredictionDelegate {
    fn name() -> &'static str {
        "supermaven"
    }

    fn display_name() -> &'static str {
        "Supermaven"
    }

    fn show_predictions_in_menu() -> bool {
        true
    }

    fn show_tab_accept_marker() -> bool {
        true
    }

    fn icons(&self, _cx: &App) -> EditPredictionIconSet {
        EditPredictionIconSet::new(IconName::Cognix)
    }

    fn data_collection_state(&self, _cx: &App) -> DataCollectionState {
        DataCollectionState::Unsupported
    }

    fn is_refreshing(&self, _cx: &App) -> bool {
        self.pending_refresh.is_some()
    }

    fn is_enabled(
        &self,
        _buffer: &Entity<Buffer>,
        _cursor_position: language::Anchor,
        _cx: &App,
    ) -> bool {
        // Supermaven's free tier is anonymous and machine-keyed; there is no
        // sign-in state to gate on. The agent reports a server-side kill
        // switch through `set { key: "disabled" }`, which we honor here.
        !self
            .agent
            .as_ref()
            .is_some_and(|agent| agent.status().disabled)
    }

    fn refresh(
        &mut self,
        buffer: Entity<Buffer>,
        cursor_position: language::Anchor,
        debounce_duration: Duration,
        _trigger: EditPredictionRequestTrigger,
        cx: &mut Context<Self>,
    ) {
        self.refresh(buffer, cursor_position, debounce_duration, cx);
    }

    fn accept(&mut self, _cx: &mut Context<Self>) {
        self.current_suggestion.take();
    }

    fn discard(&mut self, _reason: EditPredictionDiscardReason, _cx: &mut Context<Self>) {
        self.current_suggestion.take();
    }

    fn suggest(
        &mut self,
        buffer: &Entity<Buffer>,
        _cursor_position: language::Anchor,
        cx: &mut Context<Self>,
    ) -> Option<EditPrediction> {
        let suggestion = self.current_suggestion.as_ref()?;
        let buffer = buffer.read(cx);
        if suggestion.snapshot.remote_id() != buffer.remote_id() {
            return None;
        }

        let edits = edit_prediction_types::interpolate_edits(
            &suggestion.snapshot,
            &buffer.snapshot(),
            &[(suggestion.range.clone(), suggestion.text.clone())],
        )
        .filter(|edits| !edits.is_empty())?;

        Some(EditPrediction::Local {
            id: None,
            edits,
            cursor_position: None,
            edit_preview: Some(suggestion.edit_preview.clone()),
        })
    }
}
