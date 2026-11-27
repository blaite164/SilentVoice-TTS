"""
installer/mirror_omnivoice_runtime_gitlab.py — Uploads OmniVoice Windows Runtime
assets to GitLab Generic Packages and creates/updates the matching GitLab Release.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any, Dict, List
from urllib.parse import quote

import requests


GITLAB_PROJECT = "blaite164%2Fsilentvoice-tts"
RUNTIME_ASSETS: tuple[tuple[str, str], ...] = (
    ("omnivoice-runtime-win-vulkan-x64.zip", "package"),
    ("omnivoice-runtime-win-cuda-x64.zip", "package"),
    ("SHA256SUMS.txt", "other"),
)


def main(argv: List[str]) -> int:
    token = os.environ.get("GITLAB_TOKEN", "").strip()
    if not token:
        print("GITLAB_TOKEN not configured — skipping GitLab mirror.")
        return 0

    tag = os.environ.get("RELEASE_TAG", "omnivoice-runtime-v1").strip() or "omnivoice-runtime-v1"
    assets_dir = Path(argv[1] if len(argv) > 1 else "release_assets")
    ver = tag.removeprefix("omnivoice-runtime-").lstrip("vV") or "1"

    headers = {"PRIVATE-TOKEN": token}
    base_pkg = f"https://gitlab.com/api/v4/projects/{GITLAB_PROJECT}/packages/generic/omnivoice-runtime/{ver}"

    links: List[Dict[str, str]] = []
    for fname, link_type in RUNTIME_ASSETS:
        fpath = assets_dir / fname
        if not fpath.is_file():
            raise FileNotFoundError(f"Missing runtime asset: {fpath}")
        url = f"{base_pkg}/{fname}"
        print(f"Uploading {fname} to GitLab Generic Packages (v{ver})...")
        with fpath.open("rb") as fh:
            resp = requests.put(url, headers=headers, data=fh, timeout=(30, 600))
            resp.raise_for_status()
        links.append({"name": fname, "url": url, "link_type": link_type})

    sha_text = (assets_dir / "SHA256SUMS.txt").read_text(encoding="utf-8-sig").strip()
    desc = (
        f"OmniVoice Windows Runtime ({tag})\n\n"
        f"Patched Windows x64 `tts-server.exe` runtime for SilentVoice TTS (Vulkan + CUDA 12.4).\n\n"
        f"**SHA-256:**\n```text\n{sha_text}\n```"
    )

    rel_api = f"https://gitlab.com/api/v4/projects/{GITLAB_PROJECT}/releases"
    tag_q = quote(tag, safe="")
    check = requests.get(f"{rel_api}/{tag_q}", headers=headers, timeout=20)
    if check.status_code == 200:
        resp = requests.put(f"{rel_api}/{tag_q}", headers=headers, json={"description": desc}, timeout=30)
        resp.raise_for_status()
        existing_raw: Any = requests.get(f"{rel_api}/{tag_q}/assets/links", headers=headers, timeout=20).json()
        existing_names = {
            str(item.get("name") or "")
            for item in (existing_raw if isinstance(existing_raw, list) else [])
            if isinstance(item, dict)
        }
        for lnk in links:
            if lnk["name"] not in existing_names:
                requests.post(f"{rel_api}/{tag_q}/assets/links", headers=headers, json=lnk, timeout=20).raise_for_status()
    else:
        payload = {
            "name": f"OmniVoice Windows Runtime ({tag})",
            "tag_name": tag,
            "ref": "main",
            "description": desc,
            "assets": {"links": links},
        }
        resp = requests.post(rel_api, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()

    print(f"GitLab runtime mirror release published: {tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
