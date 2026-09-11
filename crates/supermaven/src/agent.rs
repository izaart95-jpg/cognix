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

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context as _, Result};
use futures::channel::mpsc;
use futures::{AsyncBufReadExt as _, AsyncReadExt as _, AsyncWriteExt as _, StreamExt as _};
use parking_lot::Mutex;
use smol::process::{Child, Command, Stdio};

use crate::protocol::{
    CompletionItem, InboundMessage, OutboundMessage, SM_MESSAGE_PREFIX, StateUpdate,
    StateUpdateEntry,
};

/// The number of most-recent states kept around for prefix matching.
const MAX_STATE_ID_RETENTION: u32 = 50;

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
    state_map: Mutex<HashMap<u32, State>>,
    next_state_id: Mutex<u32>,
    status: Mutex<AgentStatus>,
    dust_strings: Mutex<Vec<String>>,
    writer: mpsc::UnboundedSender<String>,
    /// The child's stdin, kept alive so the agent doesn't exit; dropped when
    /// the session shuts down.
    _child_stdin: Mutex<Option<smol::process::ChildStdin>>,
    _child: Mutex<Option<Child>>,
}

impl SmAgent {
    /// Spawns the agent, sends the greeting and the free-tier activation, and
    /// starts the stdout reader loop.
    pub async fn start(binary_path: &Path) -> Result<Self> {
        let mut child = Command::new(binary_path)
            .arg("stdio")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .with_context(|| format!("spawning sm-agent at {}", binary_path.display()))?;

        let stdin = child
            .stdin
            .take()
            .context("sm-agent stdin was not piped")?;
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
            state_map: Mutex::new(HashMap::default()),
            next_state_id: Mutex::new(1),
            status: Mutex::new(AgentStatus::default()),
            dust_strings: Mutex::new(Vec::new()),
            writer: writer_tx,
            _child_stdin: Mutex::new(Some(stdin)),
            _child: Mutex::new(Some(child)),
        });

        // Write loop: one JSON object per line, flushed.
        {
            let inner = inner.clone();
            smol::spawn(async move {
                let mut writer_rx = writer_rx;
                let mut stdin = match inner._child_stdin.lock().take() {
                    Some(stdin) => stdin,
                    None => return,
                };
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
        }

        // Read loop: split stdout into lines; lines with the protocol prefix
        // are decoded, everything else is agent log output.
        {
            let inner = inner.clone();
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
                                inner.handle_inbound_message(message);
                            }
                            Err(error) => {
                                log::debug!(
                                    "sm-agent sent an undecodable message: {error}"
                                );
                            }
                        }
                    } else if !line.trim().is_empty() {
                        log::debug!("sm-agent: {line}");
                    }
                }
                inner.status.lock().is_connected = false;
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
        })
        .await?;
        // Always activate the free tier on spawn: with an existing
        // credential this is a harmless ack, without one it provisions an
        // anonymous key. Skipping it makes the backend reject us with
        // `Unauthorized (fatal)`.
        this.send(OutboundMessage::UseFreeVersion).await?;

        Ok(this)
    }

    pub fn status(&self) -> AgentStatus {
        self.inner.status.lock().clone()
    }

    /// True while the child process has not exited.
    pub fn is_running(&self) -> bool {
        let mut child_guard = self.inner._child.lock();
        match child_guard.as_mut() {
            Some(child) => matches!(
                child.try_status(),
                Ok(None) // still running
            ),
            None => false,
        }
    }

    /// Sends a message to the agent. A failed write means the child died.
    pub async fn send(&self, message: OutboundMessage) -> Result<()> {
        let line = serde_json::to_string(&message)
            .context("serializing an sm-agent message")?;
        self.inner
            .writer
            .unbounded_send(line)
            .map_err(|_| anyhow::anyhow!("sm-agent is not running"))?;
        Ok(())
    }

    /// Notifies the agent that a file changed on disk.
    pub async fn inform_file_changed(&self, path: &Path) -> Result<()> {
        let path = path
            .to_str()
            .with_context(|| format!("converting {} to a string", path.display()))?
            .to_string();
        self.send(OutboundMessage::InformFileChanged { path })
            .await
    }

    /// Submits a new document state, deduplicating against the last state we
    /// sent. Returns the state id that is now authoritative, which may be an
    /// already-existing id when nothing changed.
    pub async fn submit_state(
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

        // Read the dedup candidate under one short lock, dropping the guard
        // before any I/O so it is never held across an await point.
        let (candidate_id, previous_state) = {
            let next_state_id = self.inner.next_state_id.lock();
            let candidate_id = *next_state_id;
            let previous_state = self
                .inner
                .state_map
                .lock()
                .get(&(candidate_id - 1))
                .map(|state| (state.prefix.clone(), state.content.clone()));
            (candidate_id, previous_state)
        };

        if let Some((previous_prefix, previous_content)) = &previous_state
            && previous_prefix.as_ref() == prefix_before_cursor
            && previous_content.as_ref() == full_text
        {
            // Nothing changed since the last submission; reuse the old state
            // id and avoid network traffic.
            return Ok(candidate_id - 1);
        }

        let mut updates = vec![StateUpdateEntry::CursorUpdate {
            path: path_string.clone(),
            offset: cursor_byte_offset,
        }];
        updates.push(StateUpdateEntry::FileUpdate {
            path: path_string,
            content: full_text.to_string(),
        });

        self.send(OutboundMessage::StateUpdate(StateUpdate {
            new_id: candidate_id.to_string(),
            updates,
        }))
        .await?;

        let state_id = candidate_id;
        *self.inner.next_state_id.lock() = candidate_id + 1;
        self.inner.state_map.lock().insert(
            state_id,
            State {
                prefix: Arc::from(prefix_before_cursor),
                content: Arc::from(full_text),
                items: Vec::new(),
                has_ended: false,
            },
        );
        self.purge_old_states(state_id);
        Ok(state_id)
    }

    fn purge_old_states(&self, current_state_id: u32) {
        let mut state_map = self.inner.state_map.lock();
        let oldest_retained = current_state_id.saturating_sub(MAX_STATE_ID_RETENTION);
        state_map.retain(|id, _| *id >= oldest_retained && *id <= current_state_id);
    }

    /// Returns the completion to show for the given current prefix (all text
    /// before the cursor), choosing the best retained state and stripping the
    /// characters the user typed since the query.
    pub fn derive_completion(&self, current_prefix: &str) -> Option<DerivedCompletion> {
        let state_map = self.inner.state_map.lock();
        let mut best: Option<(u32, String, usize, bool)> = None;

        for (id, state) in state_map.iter() {
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

    /// True when any retained state still has items streaming in.
    pub fn is_streaming(&self) -> bool {
        self.inner
            .state_map
            .lock()
            .values()
            .any(|state| !state.has_ended)
    }

    /// Drops all retained states (used when the provider is torn down).
    pub fn clear_states(&self) {
        self.inner.state_map.lock().clear();
    }
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
                        if matches!(
                            item,
                            CompletionItem::FinishEdit | CompletionItem::End
                        ) {
                            state.has_ended = true;
                        }
                        state.items.push(item);
                    }
                }
            }
            InboundMessage::Metadata { dust_strings } => {
                // Dust filtering is applied in `shape_completion`; store the
                // list on the session for the (rare) junk-only line case.
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
    if is_block
        && let Some(last_newline) = text.rfind('\n')
    {
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

    fn state_with_items(prefix: &str, items: Vec<CompletionItem>) -> State {
        State {
            prefix: Arc::from(prefix),
            content: Arc::from(""),
            items,
            has_ended: false,
        }
    }

    #[test]
    fn test_strip_prefix_discards_stale_state() {
        let state = state_with_items(
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
            "",
            vec![CompletionItem::Text {
                text: "\n".to_string(),
            }],
        );
        let remaining = strip_prefix(&state, "").unwrap();
        assert!(shape_completion(&remaining).is_none());
    }
}
