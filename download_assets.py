"""Download or verify the public A2 streaming weights and label map."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile
import urllib.request

from assets import (ASSETS, DEFAULT_LABELS, DEFAULT_WEIGHTS, LABELS_SHA256,
                    LABELS_URL, WEIGHTS_SHA256, WEIGHTS_URL, sha256_file)


def download(url: str, destination: Path, expected_sha256: str) -> None:
    if destination.is_file():
        if sha256_file(destination) != expected_sha256:
            raise ValueError(f"Checksum mismatch: {destination}. Move the file aside and retry.")
        print(f"Verified {destination.name}", flush=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "MoViNet-A2-demo/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response:
            with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".part", delete=False) as stream:
                temporary = Path(stream.name)
                size = 0
                while block := response.read(1024 * 1024):
                    stream.write(block)
                    size += len(block)
                    print(f"\rDownloading {destination.name}: {size / 1_000_000:.1f} MB", end="", flush=True)
        print()
        if sha256_file(temporary) != expected_sha256:
            raise ValueError(f"Downloaded artifact failed SHA-256 verification: {url}")
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-dir", type=Path, default=ASSETS)
    args = parser.parse_args()
    try:
        download(WEIGHTS_URL, args.assets_dir / DEFAULT_WEIGHTS.name, WEIGHTS_SHA256)
        download(LABELS_URL, args.assets_dir / DEFAULT_LABELS.name, LABELS_SHA256)
    except Exception as error:
        print(f"Asset setup failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
