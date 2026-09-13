//! Hosts the `sm-agent` child process and implements the client side of the
//! state-update protocol.
//!
//! The flow mirrors the reference implementation:
//!
//! 1. spawn `sm-agent stdio`, keeping its stdin open for the session lifetime,
//! 2. send `greeting`, then activate the free tier,
//! 3. on every document change, submit a `state_update` carrying the entire
//!    file content and the cursor's byte offset,
//! 4. accumulate streamed `response` items per state,
//! 5. when asked for a completion, pick the best retained state by comparing
//!    each state's query-time prefix against the current buffer text and
//!    stripping the characters the user typed since.
//!
//! Every editor shares a single agent session (see [`SmAgent::shared`]), and
//! retained states are keyed by file path so that files edited side by side
//! never see each other's completions.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Weak};

use anyhow::{Context as _, Result};
use futures::channel::mpsc;
use futures::{AsyncBufReadExt as _, AsyncReadExt as _, AsyncWriteExt as _, StreamExt as _};
use http_client::HttpClient;
use parking_lot::Mutex;
use util::command::{Child, Stdio, new_command};

use crate::binary_fetcher;
use crate::protocol::{
    CompletionItem, InboundMessage, OutboundMessage, SM_MESSAGE_PREFIX, StateUpdate,
    StateUpdateEntry,
};

/// The number of most-recent states kept around for prefix matching. One
/// session serves every open file, so this must leave room for several
/// files' in-flight submissions; completions arrive within seconds of their
/// submission, so anything much older is already stale.
const MAX_STATE_ID_RETENTION: u32 = 128;

/// The application-wide agent session. Held weakly so the child process is
/// killed (via `kill_on_drop`) as soon as the last edit-prediction delegate
/// drops its handle.
static SHARED_SESSION: Mutex<Option<Weak<SmAgentInner>>> = Mutex::new(None);

/// Serializes session starts so that concurrent callers (several editors
/// opening at once) don't each spawn their own `sm-agent`.
static SESSION_START: smol::lock::Mutex<()> = smol::lock::Mutex::new(());

/// The status of the agent's connection to the Supermaven backend.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AgentStatus {
    pub is_connected: bool,
    pub status_text: Option<String>,
    pub tier: Option<String>,
    /// The agent told us it is disabled (e.g. server-side kill switch).
    pub disabled: bool,
}

/// A completion that has been derived from the agent's streamed items and is
/// ready to be shown to the user.
#[derive(Clone, Debug)]
pub struct DerivedCompletion {
    /// The text to insert at the cursor.
    pub text: String,
    /// How many characters before the cursor to delete before inserting
    /// (indentation fix-ups from `dedent` items).
    pub prior_delete: usize,
    /// True when the model has finished streaming this state.
    pub is_complete: bool,
}

struct State {
    /// The file this state was submitted for, so completions are never
    /// shared across files.
    path: String,
    /// The full document text before the cursor, captured at query time. Used
    /// to reconcile streamed predictions against what the user has typed
    /// since.
    prefix: Arc<str>,
    /// The document content submitted with this state, used to dedup
    /// identical submissions (the prefix alone misses edits made after the
    /// cursor).
    content: Arc<str>,
    /// All response items streamed for this state so far.
    items: Vec<CompletionItem>,
    /// True when `finish_edit` or `end` was seen for this state.
    has_ended: bool,
}

/// Handle to the sm-agent child process. Cloning shares the same underlying
/// session.
#[derive(Clone)]
pub struct SmAgent {
    inner: Arc<SmAgentInner>,
}

struct SmAgentInner {
    /// Serializes state submissions so the dedup check, the state-id
    /// allocation, and the map insert are atomic. Sessions are shared by
    /// every editor, so submissions can arrive concurrently.
    submissions: Mutex<()>,
    state_map: Mutex<HashMap<u32, State>>,
    /// The id of the most recent state submitted for each file path, used to
    /// dedup identical submissions per file.
    last_state_by_path: Mutex<HashMap<String, u32>>,
    /// Guarded by `submissions`; a plain counter so `submit_state` can read
    /// and bump it under one lock.
    next_state_id: Mutex<u32>,
    status: Mutex<AgentStatus>,
    dust_strings: Mutex<Vec<String>>,
    writer: mpsc::UnboundedSender<String>,
    child: Mutex<Option<Child>>,
}

impl SmAgent {
    /// Returns the application-wide agent session shared by every editor,
    /// starting `sm-agent` on the first call and restarting it if it died.
    pub async fn shared(http_client: &dyn HttpClient) -> Result<Self> {
        if let Some(agent) = live_shared_session() {
            return Ok(agent);
        }

        // Serialize starts so concurrent first callers don't spawn one agent
        // each (that would reintroduce per-editor processes).
        let _start_guard = SESSION_START.lock().await;
        if let Some(agent) = live_shared_session() {
            return Ok(agent);
        }

        let binary_path = binary_fetcher::ensure_binary(http_client).await?;
        let agent = Self::start(&binary_path).await?;
        *SHARED_SESSION.lock() = Some(Arc::downgrade(&agent.inner));
        Ok(agent)
    }

    /// Spawns the agent, sends the greeting and the free-tier activation, and
    /// starts the stdout reader loop.
    pub async fn start(binary_path: &Path) -> Result<Self> {
        // Spawn through `util::command` so the agent runs silently: on
        // Windows this applies CREATE_NO_WINDOW to the stdio-only agent,
        // without which a console window pops up in the foreground.
        let mut child = new_command(binary_path)
            .arg("stdio")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .with_context(|| format!("spawning sm-agent at {}", binary_path.display()))?;

        let stdin = child.stdin.take().context("sm-agent stdin was not piped")?;
        let stdout = child
            .stdout
            .take()
            .context("sm-agent stdout was not piped")?;
        let stderr = child
            .stderr
            .take()
            .context("sm-agent stderr was not piped")?;

        let (writer_tx, writer_rx) = mpsc::unbounded::<String>();
        let inner = Arc::new(SmAgentInner {
            submissions: Mutex::new(()),
            state_map: Mutex::new(HashMap::default()),
            last_state_by_path: Mutex::new(HashMap::default()),
            next_state_id: Mutex::new(1),
            status: Mutex::new(AgentStatus::default()),
            dust_strings: Mutex::new(Vec::new()),
            writer: writer_tx,
            child: Mutex::new(Some(child)),
        });

        // Write loop: one JSON object per line, flushed. It owns the child's
        // stdin and holds no reference to the session, so the session can be
        // dropped (and the child killed) while it is parked on the channel;
        // dropping the last sender then ends the loop and closes stdin.
        smol::spawn(async move {
            let mut stdin = stdin;
            let mut writer_rx = writer_rx;
            while let Some(line) = writer_rx.next().await {
                if stdin.write_all(line.as_bytes()).await.is_err()
                    || stdin.write_all(b"\n").await.is_err()
                    || stdin.flush().await.is_err()
                {
                    log::error!("sm-agent stdin write failed; the agent likely exited");
                    break;
                }
            }
        })
        .detach();

        // Read loop: split stdout into lines; lines with the protocol prefix
        // are decoded, everything else is agent log output. It holds only a
        // weak reference so it never keeps a dead session (or its child
        // process) alive.
        {
            let weak = Arc::downgrade(&inner);
            smol::spawn(async move {
                let mut reader = smol::io::BufReader::new(stdout);
                let mut line = String::new();
                loop {
                    line.clear();
                    match reader.read_line(&mut line).await {
                        Ok(0) => break, // EOF: the agent exited
                        Ok(_) => {}
                        Err(error) => {
                            log::debug!("sm-agent stdout read error: {error}");
                            break;
                        }
                    }

                    let line = line.trim_end_matches(['\n', '\r']);
                    if let Some(payload) = line.strip_prefix(SM_MESSAGE_PREFIX) {
                        match serde_json::from_str::<InboundMessage>(payload) {
                            Ok(message) => {
                                if let Some(inner) = weak.upgrade() {
                                    inner.handle_inbound_message(message);
                                }
                            }
                            Err(error) => {
                                log::debug!("sm-agent sent an undecodable message: {error}");
                            }
                        }
                    } else if !line.trim().is_empty() {
                        log::debug!("sm-agent: {line}");
                    }
                }
                if let Some(inner) = weak.upgrade() {
                    inner.status.lock().is_connected = false;
                }
            })
            .detach();
        }

        // Drain stderr so a chatty agent can never block on a full pipe.
        smol::spawn(async move {
            let mut stderr = stderr;
            let mut buffer = [0u8; 4096];
            while let Ok(read) = stderr.read(&mut buffer).await {
                if read == 0 {
                    break;
                }
            }
        })
        .detach();

        let this = Self { inner };
        this.send(OutboundMessage::Greeting {
            allow_gitignore: false,
        })?;
        // Always activate the free tier on spawn: with an existing
        // credential this is a harmless ack, without one it provisions an
        // anonymous key. Skipping it makes the backend reject us with
        // `Unauthorized (fatal)`.
        this.send(OutboundMessage::UseFreeVersion)?;

        Ok(this)
    }

    pub fn status(&self) -> AgentStatus {
        self.inner.status.lock().clone()
    }

    /// True while the child process has not exited.
    pub fn is_running(&self) -> bool {
        let mut child_guard = self.inner.child.lock();
        match child_guard.as_mut() {
            Some(child) => matches!(
                child.try_status(),
                Ok(None) // still running
            ),
            None => false,
        }
    }

    /// Sends a message to the agent. A failed send means the child died.
    /// The channel is unbounded, so this never blocks; the writer task is
    /// what actually performs the (async) stdin writes.
    fn send(&self, message: OutboundMessage) -> Result<()> {
        let line = serde_json::to_string(&message).context("serializing an sm-agent message")?;
        self.inner
            .writer
            .unbounded_send(line)
            .map_err(|_| anyhow::anyhow!("sm-agent is not running"))?;
        Ok(())
    }

    /// Notifies the agent that a file changed on disk.
    pub fn inform_file_changed(&self, path: &Path) -> Result<()> {
        let path = path
            .to_str()
            .with_context(|| format!("converting {} to a string", path.display()))?
            .to_string();
        self.send(OutboundMessage::InformFileChanged { path })
    }

    /// Submits a new document state, deduplicating against the last state we
    /// sent for this file. Returns the state id that is now authoritative,
    /// which may be an already-existing id when nothing changed.
    pub fn submit_state(
        &self,
        path: &Path,
        prefix_before_cursor: &str,
        full_text: &str,
        cursor_byte_offset: usize,
    ) -> Result<u32> {
        let path_string = path
            .to_str()
            .with_context(|| format!("converting {} to a string", path.display()))?
            .to_string();

        // Serialize the whole submit so concurrent submissions from other
        // editors sharing this session can't interleave between the dedup
        // check and the id allocation.
        let _submission_guard = self.inner.submissions.lock();

        let previous_state = self
            .inner
            .last_state_by_path
            .lock()
            .get(path_string.as_str())
            .copied()
            .and_then(|last_id| {
                self.inner
                    .state_map
                    .lock()
                    .get(&last_id)
                    .map(|state| (last_id, state.prefix.clone(), state.content.clone()))
            });

        if let Some((previous_id, previous_prefix, previous_content)) = previous_state
            && previous_prefix.as_ref() == prefix_before_cursor
            && previous_content.as_ref() == full_text
        {
            // Nothing changed since the last submission for this file; reuse
            // the old state id and avoid network traffic.
            return Ok(previous_id);
        }

        let state_id = *self.inner.next_state_id.lock();
        let mut updates = vec![StateUpdateEntry::CursorUpdate {
            path: path_string.clone(),
            offset: cursor_byte_offset,
        }];
        updates.push(StateUpdateEntry::FileUpdate {
            path: path_string.clone(),
            content: full_text.to_string(),
        });
        self.send(OutboundMessage::StateUpdate(StateUpdate {
            new_id: state_id.to_string(),
            updates,
        }))?;

        *self.inner.next_state_id.lock() = state_id + 1;
        self.inner.state_map.lock().insert(
            state_id,
            State {
                path: path_string.clone(),
                prefix: Arc::from(prefix_before_cursor),
                content: Arc::from(full_text),
                items: Vec::new(),
                has_ended: false,
            },
        );
        self.inner
            .last_state_by_path
            .lock()
            .insert(path_string, state_id);
        self.purge_old_states(state_id);
        Ok(state_id)
    }

    fn purge_old_states(&self, current_state_id: u32) {
        let oldest_retained = current_state_id.saturating_sub(MAX_STATE_ID_RETENTION);
        self.inner
            .state_map
            .lock()
            .retain(|id, _| *id >= oldest_retained && *id <= current_state_id);
        self.inner
            .last_state_by_path
            .lock()
            .retain(|_, id| *id >= oldest_retained && *id <= current_state_id);
    }

    /// Returns the completion to show for the given file and current prefix
    /// (all text before the cursor), choosing the best retained state and
    /// stripping the characters the user typed since the query.
    pub fn derive_completion(
        &self,
        path: &Path,
        current_prefix: &str,
    ) -> Option<DerivedCompletion> {
        let path_string = path.to_str()?;
        let state_map = self.inner.state_map.lock();
        let mut best: Option<(u32, String, usize, bool)> = None;

        for (id, state) in state_map.iter() {
            if state.path != path_string {
                continue;
            }
            let Some(remaining) = strip_prefix(state, current_prefix) else {
                continue;
            };
            let Some(derived) = shape_completion(&remaining) else {
                continue;
            };
            let is_better = match &best {
                // Longest completion wins; ties go to the newest state.
                Some((best_id, best_text, _, _)) => {
                    derived.text.len() > best_text.len()
                        || (derived.text.len() == best_text.len() && *id > *best_id)
                }
                None => true,
            };
            if is_better {
                best = Some((*id, derived.text, derived.prior_delete, derived.is_complete));
            }
        }

        best.map(|(_, text, prior_delete, is_complete)| DerivedCompletion {
            text,
            prior_delete,
            is_complete,
        })
    }

    /// True when any retained state for this file still has items streaming
    /// in.
    pub fn is_streaming(&self, path: &Path) -> bool {
        let Some(path_string) = path.to_str() else {
            return false;
        };
        self.inner
            .state_map
            .lock()
            .values()
            .any(|state| state.path == path_string && !state.has_ended)
    }
}

/// Returns the shared session if it exists and its child is still running.
fn live_shared_session() -> Option<SmAgent> {
    let shared = SHARED_SESSION.lock();
    shared
        .as_ref()
        .and_then(|weak| weak.upgrade())
        .map(|inner| SmAgent { inner })
        .filter(|agent| agent.is_running())
}

impl SmAgentInner {
    fn handle_inbound_message(&self, message: InboundMessage) {
        match message {
            InboundMessage::Response { state_id, items } => {
                let Ok(state_id) = state_id.parse::<u32>() else {
                    log::debug!("sm-agent response for unknown stateId {state_id:?}");
                    return;
                };
                let mut state_map = self.state_map.lock();
                if let Some(state) = state_map.get_mut(&state_id) {
                    if state.has_ended {
                        // The model finished this edit; a new alternative
                        // after a barrier is not for the current view.
                        return;
                    }
                    for item in items {
                        if matches!(item, CompletionItem::FinishEdit | CompletionItem::End) {
                            state.has_ended = true;
                        }
                        state.items.push(item);
                    }
                }
            }
            InboundMessage::Metadata { dust_strings } => {
                // Retained for future use: the backend's junk-token list,
                // which the reference implementation uses to suppress
                // garbage-only suggestions.
                *self.dust_strings.lock() = dust_strings;
            }
            InboundMessage::ConnectionStatus {
                is_connected,
                status_text,
            } => {
                let mut status = self.status.lock();
                status.is_connected = is_connected;
                status.status_text = status_text;
            }
            InboundMessage::UserStatus { tier, .. } => {
                self.status.lock().tier = Some(tier);
            }
            InboundMessage::ServiceTier { display } => {
                self.status.lock().tier = Some(display);
            }
            InboundMessage::Set { key, value } => {
                if key == "disabled" {
                    let disabled = value.as_deref() != Some("false");
                    self.status.lock().disabled = disabled;
                }
            }
            InboundMessage::ActivationRequest { activate_url } => {
                if activate_url.is_empty() {
                    log::debug!("sm-agent acknowledged activation");
                } else {
                    log::info!("Supermaven activation URL: {activate_url}");
                }
            }
            InboundMessage::ActivationSuccess => {}
            InboundMessage::Passthrough { passthrough } => {
                self.handle_inbound_message(*passthrough);
            }
        }
    }
}

/// The portion of a state that is still compatible with what the user has
/// typed since the query: the user's new keystrokes are stripped from the
/// front of the streamed items. Returns `None` when the state is stale (the
/// user typed something the model did not predict).
fn strip_prefix<'a>(state: &'a State, current_prefix: &str) -> Option<RemainingCompletion<'a>> {
    let state_prefix = state.prefix.as_ref();
    if !current_prefix.starts_with(state_prefix) {
        return None;
    }

    // The user's new keystrokes since the query.
    let user_input = &current_prefix[state_prefix.len()..];
    let mut items = &state.items[..];
    let mut leading_text: Option<&'a str> = None;

    if !user_input.is_empty() {
        // Walk the streamed items, consuming the typed characters from the
        // front. `dedent` deletes backwards relative to the prediction, so it
        // must not be consumed against typed input.
        let mut consumed = 0usize;
        let mut index = 0;
        while consumed < user_input.len() && index < items.len() {
            match &items[index] {
                CompletionItem::Text { text } => {
                    let remaining_input = &user_input[consumed..];
                    if remaining_input.len() <= text.len() {
                        if !text.starts_with(remaining_input) {
                            return None;
                        }
                        // The typed input ends inside (or exactly at the end
                        // of) this item; the item's suffix is what remains
                        // of the prediction.
                        leading_text = Some(&text[remaining_input.len()..]);
                        consumed = user_input.len();
                        index += 1;
                    } else {
                        if !remaining_input.starts_with(text.as_str()) {
                            return None;
                        }
                        consumed += text.len();
                        index += 1;
                    }
                }
                CompletionItem::Dedent { .. }
                | CompletionItem::Barrier
                | CompletionItem::FinishEdit
                | CompletionItem::End => {
                    return None;
                }
                CompletionItem::Delete { .. } => {
                    // Delete items refer to buffer lines, not to the
                    // prediction text; the typed input stays.
                    index += 1;
                }
            }
        }
        if consumed < user_input.len() {
            // Not enough predicted text to account for what the user typed.
            return None;
        }
        items = &items[index..];
    }

    Some(RemainingCompletion {
        leading_text,
        items,
    })
}

struct RemainingCompletion<'a> {
    /// The unconsumed suffix of the text item the typed input ended inside,
    /// if any.
    leading_text: Option<&'a str>,
    items: &'a [CompletionItem],
}

struct ShapedCompletion {
    text: String,
    prior_delete: usize,
    is_complete: bool,
}

/// Walks the streamed items and accumulates the text to insert, stopping at
/// the first edit-unit boundary with non-whitespace output.
fn shape_completion(remaining: &RemainingCompletion) -> Option<ShapedCompletion> {
    let mut text = String::new();
    let mut prior_delete = 0usize;
    let mut is_complete = false;

    if let Some(leading) = remaining.leading_text {
        text.push_str(leading);
    }

    for item in remaining.items {
        match item {
            CompletionItem::Text { text: chunk } => {
                text.push_str(chunk);
            }
            CompletionItem::Dedent { text: dedent_text } => {
                prior_delete += dedent_text.len();
            }
            CompletionItem::FinishEdit | CompletionItem::End => {
                is_complete = true;
                break;
            }
            CompletionItem::Barrier => {
                // A new alternative begins after the barrier; complete with
                // what we have.
                break;
            }
            CompletionItem::Delete { .. } => {
                if !text.trim().is_empty() {
                    break;
                }
            }
        }
    }

    // A leading newline means a block completion: only show through the last
    // complete line, keeping its trailing newline.
    let is_block = text.starts_with('\n');
    if is_block && let Some(last_newline) = text.rfind('\n') {
        text.truncate(last_newline + 1);
    }

    // Drop trailing whitespace. For block completions the final newline is
    // meaningful (it ends the inserted line) and is kept.
    let trimmed_end = text.trim_end_matches(|c: char| c != '\n' && c.is_whitespace());
    let trailing_whitespace_len = text.len() - trimmed_end.len();
    text.truncate(trimmed_end.len());

    if text.trim().is_empty() {
        return None;
    }

    // Ignore whitespace-only output that was all we got.
    if text.chars().all(char::is_whitespace) {
        return None;
    }

    // Whitespace that the deletion would have eaten from the buffer must not
    // double up with the insertion.
    prior_delete = prior_delete.saturating_sub(trailing_whitespace_len);

    Some(ShapedCompletion {
        text,
        prior_delete,
        is_complete,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_PATH: &str = "test.py";

    fn state_with_items(path: &str, prefix: &str, items: Vec<CompletionItem>) -> State {
        State {
            path: path.to_string(),
            prefix: Arc::from(prefix),
            content: Arc::from(""),
            items,
            has_ended: false,
        }
    }

    /// A session with pre-populated states and no child process, for testing
    /// state selection without spawning anything. The returned receiver
    /// keeps the writer channel open so `submit_state` can send.
    fn test_session(
        states: impl IntoIterator<Item = (u32, State)>,
    ) -> (SmAgent, mpsc::UnboundedReceiver<String>) {
        let (writer, writer_rx) = mpsc::unbounded();
        (
            SmAgent {
                inner: Arc::new(SmAgentInner {
                    submissions: Mutex::new(()),
                    state_map: Mutex::new(states.into_iter().collect()),
                    last_state_by_path: Mutex::new(HashMap::default()),
                    next_state_id: Mutex::new(1),
                    status: Mutex::new(AgentStatus::default()),
                    dust_strings: Mutex::new(Vec::new()),
                    writer,
                    child: Mutex::new(None),
                }),
            },
            writer_rx,
        )
    }

    #[test]
    fn test_derive_completion_scopes_to_file() {
        let (session, _writer_rx) = test_session([(
            7,
            state_with_items(
                "a.py",
                "",
                vec![
                    CompletionItem::Text {
                        text: "from a".to_string(),
                    },
                    CompletionItem::FinishEdit,
                ],
            ),
        )]);

        // The same prefix in another file must not reuse a.py's prediction.
        assert!(session.derive_completion(Path::new("b.py"), "").is_none());
        let derived = session
            .derive_completion(Path::new("a.py"), "")
            .expect("the state for a.py should match");
        assert_eq!(derived.text, "from a");
    }

    #[test]
    fn test_is_streaming_scopes_to_file() {
        let (session, _writer_rx) = test_session([(
            3,
            state_with_items(
                "a.py",
                "",
                vec![CompletionItem::Text {
                    text: "x".to_string(),
                }],
            ),
        )]);

        assert!(session.is_streaming(Path::new("a.py")));
        assert!(!session.is_streaming(Path::new("b.py")));
    }

    #[test]
    fn test_submit_state_dedups_per_file() {
        let (session, _writer_rx) = test_session([]);
        let a = Path::new("/proj/a.py");
        let b = Path::new("/proj/b.py");

        let first = session.submit_state(a, "x = ", "x = 1\n", 5).unwrap();
        // Same file, same content: deduped to the same state id.
        let again = session.submit_state(a, "x = ", "x = 1\n", 5).unwrap();
        assert_eq!(first, again);
        // Same content in another file: a new state.
        let other = session.submit_state(b, "x = ", "x = 1\n", 5).unwrap();
        assert_eq!(other, first + 1);
    }

    /// A stand-in for `sm-agent` that answers every `state_update` with one
    /// streamed text item followed by `finish_edit`, echoing the state id.
    #[cfg(unix)]
    const FAKE_AGENT_SCRIPT: &str = r#"#!/bin/sh
while IFS= read -r line; do
    case "$line" in
        *'"kind":"state_update"'*)
            id=$(printf '%s' "$line" | sed -n 's/.*"newId":"\([0-9][0-9]*\)".*/\1/p')
            printf 'SM-MESSAGE {"kind":"response","stateId":"%s","items":[{"kind":"text","text":"pletion"}]}\n' "$id"
            printf 'SM-MESSAGE {"kind":"response","stateId":"%s","items":[{"kind":"finish_edit"}]}\n' "$id"
            ;;
    esac
done
"#;

    /// Verifies the session talks stdio end-to-end through the same spawn
    /// path used for the real agent (`util::command::new_command`, which on
    /// Windows additionally applies CREATE_NO_WINDOW).
    #[test]
    #[cfg(unix)]
    fn test_agent_streams_completions_over_stdio() {
        let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
        let script_path = temp_dir.path().join("fake-sm-agent");
        std::fs::write(&script_path, FAKE_AGENT_SCRIPT).expect("failed to write script");
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&script_path, std::fs::Permissions::from_mode(0o755))
                .expect("failed to make script executable");
        }

        let agent = smol::block_on(async {
            let agent = SmAgent::start(&script_path)
                .await
                .expect("failed to start agent");
            assert!(agent.is_running());

            let path = Path::new("/proj/a.py");
            let state_id = agent
                .submit_state(path, "x = ", "x = 1\n", 5)
                .expect("failed to submit state");
            assert_eq!(state_id, 1);
            agent
        });

        // Poll for the streamed completion; the reader loop runs on the
        // global smol executor.
        let path = Path::new("/proj/a.py");
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        let derived = loop {
            if let Some(derived) = agent.derive_completion(path, "x = ") {
                break derived;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "no completion streamed back over stdio"
            );
            std::thread::sleep(std::time::Duration::from_millis(25));
        };
        assert_eq!(derived.text, "pletion");
        assert!(derived.is_complete);

        // The session dies with its last handle, killing the child.
        drop(agent);
    }

    #[test]
    fn test_strip_prefix_discards_stale_state() {
        let state = state_with_items(
            TEST_PATH,
            "greet(",
            vec![CompletionItem::Text {
                text: "ame):".to_string(),
            }],
        );
        // The user typed something the model didn't predict.
        assert!(strip_prefix(&state, "greet(f").is_none());
        // The user typed exactly what the model predicted: the typed part is
        // stripped and the rest of the prediction survives.
        let remaining = strip_prefix(&state, "greet(am").unwrap();
        let shaped = shape_completion(&remaining).unwrap();
        assert_eq!(shaped.text, "e):");
    }

    #[test]
    fn test_strip_prefix_requires_state_prefix_of_current() {
        let state = state_with_items(
            TEST_PATH,
            "greet(x",
            vec![CompletionItem::Text {
                text: ")".to_string(),
            }],
        );
        // The buffer diverged before the query prefix.
        assert!(strip_prefix(&state, "greet(y").is_none());
    }

    #[test]
    fn test_strip_prefix_consumes_across_items() {
        let state = state_with_items(
            TEST_PATH,
            "ab",
            vec![
                CompletionItem::Text {
                    text: "cd".to_string(),
                },
                CompletionItem::Text {
                    text: "ef".to_string(),
                },
            ],
        );
        let remaining = strip_prefix(&state, "abcd").unwrap();
        // The typed input consumed the first item entirely; only the second
        // item remains.
        assert_eq!(remaining.items.len(), 1);
        let shaped = shape_completion(&remaining).unwrap();
        assert_eq!(shaped.text, "ef");
    }

    #[test]
    fn test_strip_prefix_ends_inside_item() {
        let state = state_with_items(
            TEST_PATH,
            "greet(",
            vec![
                CompletionItem::Text {
                    text: "ame".to_string(),
                },
                CompletionItem::Text {
                    text: "):".to_string(),
                },
            ],
        );
        // The typed input ends inside the first item; the item's suffix is
        // kept as the leading text of the remaining completion.
        let remaining = strip_prefix(&state, "greet(a").unwrap();
        assert_eq!(remaining.leading_text, Some("me"));
        assert_eq!(remaining.items.len(), 1);
        let shaped = shape_completion(&remaining).unwrap();
        assert_eq!(shaped.text, "me):");
    }

    #[test]
    fn test_shape_completion_accumulates_text() {
        let state = state_with_items(
            TEST_PATH,
            "",
            vec![
                CompletionItem::Text {
                    text: "str".to_string(),
                },
                CompletionItem::Text {
                    text: "):".to_string(),
                },
                CompletionItem::FinishEdit,
            ],
        );
        let remaining = strip_prefix(&state, "").unwrap();
        let shaped = shape_completion(&remaining).unwrap();
        assert_eq!(shaped.text, "str):");
        assert!(shaped.is_complete);
    }

    #[test]
    fn test_shape_completion_stops_at_barrier() {
        let state = state_with_items(
            TEST_PATH,
            "",
            vec![
                CompletionItem::Text {
                    text: "one".to_string(),
                },
                CompletionItem::Barrier,
                CompletionItem::Text {
                    text: "two".to_string(),
                },
            ],
        );
        let remaining = strip_prefix(&state, "").unwrap();
        let shaped = shape_completion(&remaining).unwrap();
        assert_eq!(shaped.text, "one");
    }

    #[test]
    fn test_shape_completion_trailing_newline_block() {
        // A completion whose first chunk starts with a newline is a block
        // completion: only the part through the last newline is shown.
        let state = state_with_items(
            TEST_PATH,
            "",
            vec![
                CompletionItem::Text {
                    text: "\n    return x\n".to_string(),
                },
                CompletionItem::Text {
                    text: "    y = 2".to_string(),
                },
                CompletionItem::FinishEdit,
            ],
        );
        let remaining = strip_prefix(&state, "").unwrap();
        let shaped = shape_completion(&remaining).unwrap();
        assert_eq!(shaped.text, "\n    return x\n");
    }

    #[test]
    fn test_shape_completion_newline_only_renders_nothing() {
        let state = state_with_items(
            TEST_PATH,
            "",
            vec![
                CompletionItem::Text {
                    text: "\n".to_string(),
                },
                CompletionItem::FinishEdit,
            ],
        );
        let remaining = strip_prefix(&state, "").unwrap();
        assert!(shape_completion(&remaining).is_none());
    }

    #[test]
    fn test_shape_completion_dedent_prior_delete() {
        let state = state_with_items(
            TEST_PATH,
            "",
            vec![
                CompletionItem::Dedent {
                    text: "  ".to_string(),
                },
                CompletionItem::Text {
                    text: "x = 1".to_string(),
                },
                CompletionItem::FinishEdit,
            ],
        );
        let remaining = strip_prefix(&state, "").unwrap();
        let shaped = shape_completion(&remaining).unwrap();
        assert_eq!(shaped.text, "x = 1");
        assert_eq!(shaped.prior_delete, 2);
    }

    #[test]
    fn test_shape_completion_rejects_whitespace_only() {
        let state = state_with_items(
            TEST_PATH,
            "",
            vec![CompletionItem::Text {
                text: "\n".to_string(),
            }],
        );
        let remaining = strip_prefix(&state, "").unwrap();
        assert!(shape_completion(&remaining).is_none());
    }
}
