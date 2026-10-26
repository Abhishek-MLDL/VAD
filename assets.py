"""Pinned upstream artifacts used by the demo."""
from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
UPSTREAM_COMMIT = "c2d1edf48fc6c5259707f9d833f22171b4f63493"
WEIGHTS_URL = (
    "https://raw.githubusercontent.com/Atze00/MoViNet-pytorch/"
    + UPSTREAM_COMMIT + "/weights/modelA2_stream_statedict_v3"
)
LABELS_URL = (
    "https://raw.githubusercontent.com/tensorflow/models/"
    "f8af2291cced43fc9f1d9b41ddbf772ae7b0d7d2/official/projects/movinet/"
    "files/kinetics_600_labels.txt"
)
WEIGHTS_SHA256 = "a3128a6bced7c0942c0e34bbb848035a5cfcaa226ae22c0d1f753e3739e9927a"
LABELS_SHA256 = "8dfca4202520ea8291ce0242ef33b006d0a0772b409663c090a2876bc1978dcf"
DEFAULT_WEIGHTS = ASSETS / "movinet_a2_stream.pt"
DEFAULT_LABELS = ASSETS / "kinetics_600_labels.txt"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_labels(path: Path = DEFAULT_LABELS) -> list[str]:
    """Enforce this checkpoint's exact 600-class ordering."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing labels: {path}. Run python download_assets.py")
    if sha256_file(path) != LABELS_SHA256:
        raise ValueError("Labels differ from the pinned Kinetics-600 label map.")
    labels = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    if len(labels) != 600 or len(set(labels)) != 600 or not all(labels):
        raise ValueError("Expected exactly 600 unique, nonempty labels.")
    return labels
