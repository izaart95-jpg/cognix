"""
Watch for zed.exe, upload to gofile, and POST the gofile response
to an ngrok endpoint (with the ngrok-skip-browser-warning header).
"""

import os
import time
import json
import requests
from pathlib import Path

# --- Configuration ---------------------------------------------------------
USERPROFILE     = os.environ.get("USERPROFILE", "")
TARGET_FILE     = Path(USERPROFILE) / "Desktop" / "zed" / "target" / "release" / "zed.exe"
POST_URL        = "https://heteronomous-bridally-minerva.ngrok-free.dev"
RETRY_INTERVAL  = 15          # seconds between checks / retries
GOFILE_API      = "https://api.gofile.io"
NGROK_HEADERS   = {"ngrok-skip-browser-warning": "true"}
# ---------------------------------------------------------------------------

def get_gofile_server() -> str:
    """Ask gofile for the best available server."""
    r = requests.get(f"{GOFILE_API}/servers", timeout=30)
    r.raise_for_status()
    data = r.json()
    if data.get("status") != "ok":
        raise RuntimeError(f"gofile servers request failed: {data}")
    servers = data["data"]["servers"]
    if not servers:
        raise RuntimeError("no gofile servers returned")
    return servers[0]["name"]


def upload_to_gofile(filepath: Path) -> dict:
    """Upload a file to gofile and return the parsed response dict."""
    server = get_gofile_server()
    upload_url = f"https://{server}.gofile.io/contents/uploadfile"
    print(f"[gofile] using server {server} -> {upload_url}")

    file_size = filepath.stat().st_size
    print(f"[gofile] uploading {filepath} ({file_size:,} bytes)...")

    with filepath.open("rb") as fh:
        files = {"file": (filepath.name, fh, "application/octet-stream")}
        r = requests.post(upload_url, files=files, timeout=600)
    r.raise_for_status()
    data = r.json()
    if data.get("status") != "ok":
        raise RuntimeError(f"gofile upload failed: {data}")
    return data["data"]


def post_to_ngrok(payload: dict) -> requests.Response:
    """POST the gofile payload to the ngrok endpoint with the skip header."""
    print(f"[ngrok] POSTing payload to {POST_URL}")
    r = requests.post(
        POST_URL,
        data=json.dumps(payload),
        headers={
            **NGROK_HEADERS,
            "Content-Type": "application/json",
        },
        timeout=60,
    )
    return r


def main() -> None:
    print(f"[*] watching for: {TARGET_FILE}")
    while True:
        if TARGET_FILE.exists() and TARGET_FILE.is_file():
            print(f"[+] found: {TARGET_FILE}")
            try:
                gofile_resp = upload_to_gofile(TARGET_FILE)
                print(f"[+] gofile response: {json.dumps(gofile_resp, indent=2)}")

                ngrok_resp = post_to_ngrok(gofile_resp)
                print(f"[+] ngrok POST status: {ngrok_resp.status_code}")
                print(f"[+] ngrok POST body:   {ngrok_resp.text}")
                print("[*] done.")
                return
            except Exception as e:
                print(f"[!] error: {e!r}")
                print(f"[*] retrying in {RETRY_INTERVAL}s ...")
                time.sleep(RETRY_INTERVAL)
                continue
        else:
            print(f"[-] not found; retrying in {RETRY_INTERVAL}s ...")
            time.sleep(RETRY_INTERVAL)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[*] interrupted by user")
