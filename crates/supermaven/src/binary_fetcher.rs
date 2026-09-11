//! Locating, downloading, verifying, and caching the Supermaven `sm-agent`
//! binary.
//!
//! The binary is never cached in a temp directory: it lives under
//! [`paths::data_dir`] in a versioned directory so that partial downloads can
//! never shadow a good binary, and a future protocol bump can invalidate the
//! cache by bumping [`CACHE_GENERATION`].

use std::io::Write;
use std::path::PathBuf;

use anyhow::{Context as _, Result, bail};
use futures::AsyncReadExt as _;
use http_client::{AsyncBody, HttpClient};
use serde::Deserialize;
use sha2::{Digest, Sha256};

/// Cache-generation constant. The reference plugin versions its cache
/// directory `v20`; keep that so a future protocol change can bump it.
const CACHE_GENERATION: &str = "v20";

/// The discovery endpoint that tells us where today's binary lives.
const DOWNLOAD_PATH_API: &str = "https://supermaven.com/api/download-path-v2";

/// The discovery API only accepts `editor=neovim` today; any other value is
/// rejected with `{"error": "Download path not found."}`.
const EDITOR_PARAM: &str = "neovim";

/// Refuse to send documents larger than this to the backend, mirroring the
/// reference implementation's hard size limit.
pub const HARD_SIZE_LIMIT: usize = 10 * 1024 * 1024;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct DownloadPath {
    download_url: String,
    #[allow(dead_code)] // part of the API contract; our upgrade signal
    version: u64,
    sha256_hash: String,
}

/// Returns the platform string the discovery API expects for this OS.
pub fn platform() -> Result<&'static str> {
    match std::env::consts::OS {
        "linux" => Ok("linux"),
        "macos" => Ok("macosx"),
        "windows" => Ok("windows"),
        other => bail!("unsupported Supermaven platform: {other}"),
    }
}

/// Returns the architecture string the discovery API expects for this CPU.
pub fn arch() -> Result<&'static str> {
    match std::env::consts::ARCH {
        "x86_64" => Ok("x86_64"),
        "aarch64" => Ok("aarch64"),
        other => bail!("unsupported Supermaven architecture: {other}"),
    }
}

/// The directory the agent binary is cached in:
/// `<data_dir>/supermaven/binary/v20/<platform>-<arch>`.
pub fn binary_cache_dir() -> Result<PathBuf> {
    Ok(paths::data_dir()
        .join("supermaven")
        .join("binary")
        .join(CACHE_GENERATION)
        .join(format!("{}-{}", platform()?, arch()?)))
}

/// The full path of the cached agent binary.
pub fn binary_path() -> Result<PathBuf> {
    let file_name = if cfg!(target_os = "windows") {
        "sm-agent.exe"
    } else {
        "sm-agent"
    };
    Ok(binary_cache_dir()?.join(file_name))
}

/// Ensures the agent binary is present and executable, downloading and
/// verifying it if necessary. Returns its path.
pub async fn ensure_binary(http_client: &dyn HttpClient) -> Result<PathBuf> {
    let destination = binary_path()?;
    if destination.is_file() {
        return Ok(destination);
    }

    let cache_dir = binary_cache_dir()?;
    std::fs::create_dir_all(&cache_dir)
        .with_context(|| format!("creating {}", cache_dir.display()))?;

    let download_path = fetch_download_path(http_client).await?;

    // Download to a uniquely-named file in the same directory so that a
    // partial download can never shadow a good binary, then atomically move
    // it into place.
    let temp_file = cache_dir.join(format!(
        "sm-agent.download.{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or(0)
    ));
    let result = download_verify_and_install(http_client, &download_path, &temp_file, &destination)
        .await;
    if result.is_err() {
        // Best-effort cleanup of a partial download.
        let _ = std::fs::remove_file(&temp_file);
    }
    result?;

    Ok(destination)
}

async fn fetch_download_path(http_client: &dyn HttpClient) -> Result<DownloadPath> {
    let url = format!(
        "{DOWNLOAD_PATH_API}?platform={}&arch={}&editor={EDITOR_PARAM}",
        platform()?,
        arch()?
    );
    let mut response = http_client
        .get(&url, AsyncBody::empty(), true)
        .await
        .context("requesting Supermaven download path")?;

    if !response.status().is_success() {
        bail!(
            "Supermaven download-path API returned status {}",
            response.status()
        );
    }

    let mut body = Vec::new();
    response
        .body_mut()
        .read_to_end(&mut body)
        .await
        .context("reading Supermaven download-path response")?;
    let download_path: DownloadPath =
        serde_json::from_slice(&body).context("parsing Supermaven download-path response")?;
    anyhow::ensure!(
        !download_path.download_url.is_empty(),
        "Supermaven download-path API returned an empty download URL"
    );
    Ok(download_path)
}

async fn download_verify_and_install(
    http_client: &dyn HttpClient,
    download_path: &DownloadPath,
    temp_file: &std::path::Path,
    destination: &std::path::Path,
) -> Result<()> {
    let mut response = http_client
        .get(&download_path.download_url, AsyncBody::empty(), true)
        .await
        .context("downloading sm-agent binary")?;
    anyhow::ensure!(
        response.status().is_success(),
        "downloading sm-agent failed with status {}",
        response.status()
    );

    let mut body = Vec::new();
    response
        .body_mut()
        .read_to_end(&mut body)
        .await
        .context("reading sm-agent binary")?;

    let mut hasher = Sha256::new();
    hasher.update(&body);
    let digest = hex_lower(&hasher.finalize());
    anyhow::ensure!(
        digest == download_path.sha256_hash,
        "sm-agent sha256 mismatch: expected {}, got {digest}",
        download_path.sha256_hash
    );

    // Write through std so the executable bit (set below) can't race with an
    // async writer that hasn't flushed yet.
    let mut file = std::fs::File::create(temp_file)
        .with_context(|| format!("creating {}", temp_file.display()))?;
    file.write_all(&body)
        .and_then(|_| file.flush())
        .with_context(|| format!("writing {}", temp_file.display()))?;
    drop(file);

    // Moving via rename-then-chmod keeps a partially-installed binary from
    // ever being observed as executable.
    std::fs::rename(temp_file, destination)
        .with_context(|| format!("installing {}", destination.display()))?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(destination, std::fs::Permissions::from_mode(0o755))
            .with_context(|| format!("making {} executable", destination.display()))?;
    }

    log::info!(
        "Installed sm-agent binary at {} ({} bytes, sha256 {digest})",
        destination.display(),
        body.len()
    );
    Ok(())
}

fn hex_lower(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        let _ = write!(out, "{byte:02x}");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_platform_and_arch_matrix() {
        // Every supported platform/arch combination must produce a usable
        // cache path.
        let cache_dir = binary_cache_dir();
        assert!(cache_dir.is_ok(), "{cache_dir:?}");
        let path = binary_path();
        assert!(path.is_ok(), "{path:?}");
    }

    #[test]
    fn test_hex_lower() {
        assert_eq!(hex_lower(&[0x00, 0xff, 0x10]), "00ff10");
    }

    #[test]
    fn test_decode_download_path_response() {
        let body = r#"{
            "downloadUrl": "https://example.com/sm-agent",
            "version": 8,
            "sha256Hash": "157a2df3"
        }"#;
        let parsed: DownloadPath = serde_json::from_str(body).unwrap();
        assert_eq!(parsed.download_url, "https://example.com/sm-agent");
        assert_eq!(parsed.sha256_hash, "157a2df3");
        assert_eq!(parsed.version, 8);
    }

    #[test]
    fn test_inbound_message_decoding() {
        use crate::protocol::{InboundMessage, SM_MESSAGE_PREFIX};

        let line = format!(
            "{SM_MESSAGE_PREFIX}{{\"kind\":\"response\",\"stateId\":\"5\",\"items\":[{{\"kind\":\"text\",\"text\":\"str\"}}]}}"
        );
        let response = line.strip_prefix(SM_MESSAGE_PREFIX).unwrap();
        let message: InboundMessage = serde_json::from_str(response).unwrap();
        match message {
            InboundMessage::Response { state_id, items } => {
                assert_eq!(state_id, "5");
                assert_eq!(items.len(), 1);
                assert!(matches!(
                    items[0],
                    crate::protocol::CompletionItem::Text { ref text } if text == "str"
                ));
            }
            other => panic!("unexpected message: {other:?}"),
        }
    }

    #[test]
    fn test_outbound_message_serialization() {
        use crate::protocol::{OutboundMessage, StateUpdate, StateUpdateEntry};

        let greeting =
            serde_json::to_string(&OutboundMessage::Greeting { allow_gitignore: false }).unwrap();
        assert_eq!(
            greeting,
            r#"{"kind":"greeting","allowGitignore":false}"#
        );

        let free = serde_json::to_string(&OutboundMessage::UseFreeVersion).unwrap();
        assert_eq!(free, r#"{"kind":"use_free_version"}"#);

        let state = serde_json::to_string(&OutboundMessage::StateUpdate(StateUpdate {
            new_id: "17".to_string(),
            updates: vec![
                StateUpdateEntry::CursorUpdate {
                    path: "/tmp/demo.py".to_string(),
                    offset: 51,
                },
                StateUpdateEntry::FileUpdate {
                    path: "/tmp/demo.py".to_string(),
                    content: "def greet(name):\n".to_string(),
                },
            ],
        }))
        .unwrap();
        assert_eq!(
            state,
            r#"{"kind":"state_update","newId":"17","updates":[{"kind":"cursor_update","path":"/tmp/demo.py","offset":51},{"kind":"file_update","path":"/tmp/demo.py","content":"def greet(name):\n"}]}"#
        );
    }
}
