//! Supermaven edit predictions: hosts the `sm-agent` binary and turns its
//! streamed completions into editor suggestions.

mod agent;
mod binary_fetcher;
mod protocol;
mod supermaven_edit_prediction_delegate;

pub use agent::{AgentStatus, SmAgent};
pub use binary_fetcher::{HARD_SIZE_LIMIT, arch, binary_path, ensure_binary, platform};
pub use supermaven_edit_prediction_delegate::SupermavenEditPredictionDelegate;
