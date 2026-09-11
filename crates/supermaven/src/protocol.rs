//! Protocol types for the Supermaven `sm-agent` stdio JSON-lines interface.
//!
//! The wire format is one JSON object per `\n`-terminated line. Messages sent
//! to the agent go over its stdin; messages emitted by the agent arrive on its
//! stdout prefixed with the literal `SM-MESSAGE ` (note the trailing space).
//!
//! Field names on the wire are camelCase (`allowGitignore`, `stateId`,
//! `newId`, ...), while variant tags are snake_case (`use_free_version`,
//! `state_update`, ...). Both are reproduced exactly as observed on the wire.

use serde::{Deserialize, Serialize};

/// The prefix that the agent puts in front of every protocol message it
/// writes to stdout. Lines without this prefix are plain Rust log output.
pub const SM_MESSAGE_PREFIX: &str = "SM-MESSAGE ";

/// Messages we send to the agent, one JSON object per line on stdin.
#[derive(Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum OutboundMessage {
    /// Sent immediately after spawning the agent.
    Greeting {
        #[serde(rename = "allowGitignore")]
        allow_gitignore: bool,
    },
    /// Activates the anonymous, machine-keyed free tier. Safe to re-send on
    /// every spawn: with an existing credential it is a harmless ack.
    UseFreeVersion,
    /// Informs the agent that a file changed on disk.
    InformFileChanged { path: String },
    /// Requests completions for a new document state.
    StateUpdate(StateUpdate),
}

/// The payload of a `state_update` message.
#[derive(Debug, Serialize)]
pub struct StateUpdate {
    /// Our monotonically-increasing correlation id, echoed back in responses
    /// as `stateId`.
    #[serde(rename = "newId")]
    pub new_id: String,
    pub updates: Vec<StateUpdateEntry>,
}

/// One entry in the `updates` list of a [`StateUpdate`].
#[derive(Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum StateUpdateEntry {
    /// The cursor position as a byte offset from the start of the document.
    CursorUpdate { path: String, offset: usize },
    /// The entire file content.
    FileUpdate { path: String, content: String },
}

/// Messages the agent sends to us on stdout (after the `SM-MESSAGE ` prefix).
#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum InboundMessage {
    /// Streamed completion chunks for a previously-submitted state.
    Response {
        #[serde(rename = "stateId")]
        state_id: String,
        items: Vec<CompletionItem>,
    },
    /// Junk-token list used to suppress garbage-only suggestions.
    Metadata {
        #[serde(default, rename = "dustStrings")]
        dust_strings: Vec<String>,
    },
    /// Browser activation link for the Pro tier. An empty `activate_url` is an
    /// acknowledgement (e.g. of `use_free_version`).
    ActivationRequest {
        #[serde(default, rename = "activateUrl")]
        activate_url: String,
    },
    ActivationSuccess,
    ConnectionStatus {
        #[serde(rename = "isConnected")]
        is_connected: bool,
        #[serde(default, rename = "statusText")]
        status_text: Option<String>,
    },
    UserStatus { tier: String },
    ServiceTier { display: String },
    /// Settings pushed by the agent. We only care about the `disabled` key.
    Set {
        #[serde(default)]
        key: String,
        #[serde(default)]
        value: Option<String>,
    },
    /// A message nested inside another message.
    Passthrough {
        passthrough: Box<InboundMessage>,
    },
}

/// A single streamed item of a completion response.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CompletionItem {
    /// Literal text to insert.
    Text { text: String },
    /// Characters to delete backwards before inserting (indentation fix).
    Dedent { text: String },
    /// Delete the following buffer line if it matches `verify`.
    Delete {
        #[serde(default)]
        verify: String,
    },
    /// End of one edit attempt; a new alternative begins after it.
    Barrier,
    /// End of this edit unit.
    FinishEdit,
    /// End of the whole response.
    End,
}
