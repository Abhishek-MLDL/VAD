"""Video sampling, preprocessing, annotation, and optional audio remuxing.

The sampling timestamps use OpenCV's constant-frame-rate interpretation of a
video. Convert variable-frame-rate footage to a constant frame rate first when
precise presentation timestamps matter.
"""

from __future__ import annotations

import hashlib
import math
import numbers
from pathlib import Path
import re
import subprocess
from typing import Sequence

import cv2
import numpy as np


VIDEO_EXTENSIONS = frozenset(
    {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".mpg", ".mpeg", ".mts", ".m2ts"}
)


class FrameSampler:
    """Select each first decoded frame at or after an absolute sampling time.

    For example, 29.97 FPS sampled at 5 FPS selects frames 0, 6, 12, ... .
    Requested rates above the source rate are clamped to the source rate, so a
    source frame is never duplicated. Call ``take`` once for each decoded frame
    in increasing frame-index order. Skipped indexes do not shift the schedule.
    """

    def __init__(self, source_fps: float, requested_fps: float):
        self.source_fps = self._positive_finite(source_fps, "source_fps")
        self.requested_fps = self._positive_finite(requested_fps, "requested_fps")
        self._effective_fps = min(self.source_fps, self.requested_fps)
        self._next_tick = 0
        self._last_index = -1

    @staticmethod
    def _positive_finite(value: float, name: str) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a finite positive number.") from exc
        if not math.isfinite(result) or result <= 0:
            raise ValueError(f"{name} must be a finite positive number.")
        return result

    @property
    def effective_fps(self) -> float:
        return self._effective_fps

    def take(self, frame_index: int) -> bool:
        if isinstance(frame_index, bool) or not isinstance(frame_index, numbers.Integral):
            raise ValueError("frame_index must be a nonnegative integer.")
        frame_index = int(frame_index)
        if frame_index < 0:
            raise ValueError("frame_index must be a nonnegative integer.")
        if frame_index < self._last_index:
            raise ValueError("Frame indexes must be supplied in increasing order.")
        if frame_index == self._last_index:
            return False
        self._last_index = frame_index
        # Calculate from the absolute clock instead of repeatedly adding a
        # rounded frame interval, which drifts on non-integer source rates.
        next_frame = math.ceil(
            self._next_tick * self.source_fps / self._effective_fps - 1e-9
        )
        if frame_index < next_frame:
            return False
        self._next_tick = (
            math.floor(frame_index * self._effective_fps / self.source_fps + 1e-9) + 1
        )
        return True


def _validate_bgr(frame: np.ndarray) -> None:
    if (
        not isinstance(frame, np.ndarray)
        or frame.dtype != np.uint8
        or frame.ndim != 3
        or frame.shape[2] != 3
        or frame.shape[0] == 0
        or frame.shape[1] == 0
    ):
        raise ValueError("Expected a nonempty H x W x 3 uint8 BGR frame from OpenCV.")


def preprocess_bgr(frame: np.ndarray, size: int = 224) -> np.ndarray:
    """Return contiguous RGB, CHW, float32 pixels in [0, 1].

    This demo uses a direct bilinear square resize, matching its selected
    checkpoint's documented inference pipeline. No mean/std normalization or
    additional crop is applied.
    """
    _validate_bgr(frame)
    if isinstance(size, bool) or not isinstance(size, numbers.Integral) or size <= 0:
        raise ValueError("size must be a positive integer.")
    resized = cv2.resize(frame, (int(size), int(size)), interpolation=cv2.INTER_LINEAR)
    rgb_chw = resized[:, :, ::-1].transpose(2, 0, 1)
    return np.ascontiguousarray(rgb_chw, dtype=np.float32) / np.float32(255.0)


def discover_videos(path: Path) -> list[Path]:
    """Resolve an input video or recursively discover supported video files."""
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input does not exist: {path}")
    if path.is_file():
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"Unsupported video extension: {path.suffix or '(none)'}")
        return [path]
    if not path.is_dir():
        raise ValueError(f"Input is neither a file nor a directory: {path}")
    videos = sorted(
        (candidate.resolve() for candidate in path.rglob("*")
         if candidate.is_file() and candidate.suffix.lower() in VIDEO_EXTENSIONS),
        key=lambda candidate: (str(candidate).casefold(), str(candidate)),
    )
    if not videos:
        raise ValueError(f"No supported video files found in: {path}")
    return videos


def output_slug(path: Path) -> str:
    """Combine a portable filename stem and the resolved source-path hash."""
    resolved = Path(path).expanduser().resolve()
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", resolved.stem).strip("._-") or "video"
    digest = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:8]
    return f"{stem[:80]}_{digest}"


def _wrap_line(text: str, max_width: int, font: int, scale: float, thickness: int) -> list[str]:
    """Wrap even labels without spaces so every rendered line fits the frame."""
    result: list[str] = []
    current = ""
    for character in str(text):
        candidate = current + character
        width = cv2.getTextSize(candidate, font, scale, thickness)[0][0]
        if current and width > max_width:
            result.append(current.rstrip())
            current = character.lstrip()
        else:
            current = candidate
    result.append(current.rstrip())
    return result


def draw_overlay(frame: np.ndarray, lines: Sequence[str], active: bool = False) -> np.ndarray:
    """Draw readable status text on a copy of a BGR frame, including small ones."""
    _validate_bgr(frame)
    result = frame.copy()
    if not lines:
        return result
    height, width = result.shape[:2]
    padding = max(1, min(12, min(height, width) // 30))
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.25, min(0.65, width / 1000.0))
    thickness = 1 if scale < 0.6 else 2
    (_, text_height), baseline = cv2.getTextSize("Ag", font, scale, thickness)
    line_height = text_height + baseline + max(2, padding // 2)
    available_width = max(1, width - 2 * padding)
    rendered = [wrapped for line in lines
                for wrapped in _wrap_line(str(line), available_width, font, scale, thickness)]
    max_lines = max(1, (height - 2 * padding) // line_height)
    if len(rendered) > max_lines:
        rendered = rendered[:max_lines]
        if width >= 35:
            rendered[-1] = "..."
    box_height = min(height, 2 * padding + line_height * len(rendered))
    panel = result[:box_height].copy()
    panel[:] = (8, 8, 8)
    result[:box_height] = cv2.addWeighted(result[:box_height], 0.25, panel, 0.75, 0)
    status_color = (90, 235, 110) if active else (245, 245, 245)
    for index, line in enumerate(rendered):
        y = padding + text_height + index * line_height
        if y >= height:
            break
        cv2.putText(result, line, (padding, y), font, scale, status_color,
                    thickness, cv2.LINE_AA)
    return result


def mux_audio(annotated: Path, source: Path, output: Path, ffmpeg: str = "ffmpeg") -> None:
    """Copy annotated video and optionally re-encode source audio as AAC.

    The output ends at the annotated video's frame-count/FPS duration. A short
    audio stream therefore cannot truncate the video, and a long source audio
    stream cannot lengthen it. The source and annotation are never overwritten.
    FFmpeg must be installed separately and available through ``ffmpeg``.
    """
    annotated, source, output = (Path(item).expanduser().resolve()
                                 for item in (annotated, source, output))
    for video in (annotated, source):
        if not video.is_file():
            raise FileNotFoundError(f"Video does not exist: {video}")
    if output in (annotated, source):
        raise ValueError("Audio output must be different from both input paths.")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite audio output: {output}")
    capture = cv2.VideoCapture(str(annotated))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Cannot inspect annotated video: {annotated}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or not math.isfinite(count) or fps <= 0 or count <= 0:
            raise RuntimeError("Annotated video has no reliable frame count/FPS; cannot preserve duration.")
        duration = count / fps
    finally:
        capture.release()
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-n",
        "-i", str(annotated), "-i", str(source),
        "-map", "0:v:0", "-map", "1:a?", "-c:v", "copy", "-c:a", "aac",
        "-t", f"{duration:.9f}", str(output),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise RuntimeError(f"FFmpeg was not found: {ffmpeg!r}. Install it or set --ffmpeg.") from exc
    if result.returncode:
        details = result.stderr.strip() or result.stdout.strip() or "no error details"
        raise RuntimeError(f"FFmpeg audio mux failed (exit {result.returncode}): {details}")
