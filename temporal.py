"""Conservative temporal candidates from a single-label video classifier.

The returned intervals describe sustained classifier evidence, not learned action
boundaries. In particular, a rejected prediction does not establish "no action".
All thresholds use the original probability distribution over every class.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real

import numpy as np


_EPSILON = 1e-9


def _finite_number(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


@dataclass(frozen=True)
class TriggerConfig:
    """Uncalibrated demo thresholds; tune with representative labeled videos."""

    enter_threshold: float = 0.5
    exit_threshold: float = 0.3
    margin: float = 0.1
    min_observations: int = 3
    min_duration_seconds: float = 0.4
    gap_seconds: float = 0.6

    def __post_init__(self) -> None:
        for name in ("enter_threshold", "exit_threshold", "margin"):
            value = _finite_number(name, getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.exit_threshold > self.enter_threshold:
            raise ValueError("exit_threshold must not exceed enter_threshold")
        if (isinstance(self.min_observations, bool)
                or not isinstance(self.min_observations, Integral)
                or self.min_observations < 2):
            raise ValueError("min_observations must be an integer of at least 2")
        if _finite_number("min_duration_seconds", self.min_duration_seconds) < 0:
            raise ValueError("min_duration_seconds must be nonnegative")
        if _finite_number("gap_seconds", self.gap_seconds) <= 0:
            raise ValueError("gap_seconds must be greater than zero")


@dataclass
class _Evidence:
    class_id: int
    start: float
    last_support: float
    context_start: float
    window_id: int
    count: int
    probability_sum: float
    peak: float
    confirmed_at: float | None = None

    def support(self, timestamp: float, probability: float,
                context_start: float) -> None:
        self.last_support = timestamp
        self.context_start = min(self.context_start, context_start)
        self.count += 1
        self.probability_sum += probability
        self.peak = max(self.peak, probability)


class TemporalDetector:
    """Maintain at most one confirmed action candidate at a time.

    An event starts after consecutive qualifying global top-1 observations for
    one allowed class meet both the count and observed-duration requirements.
    Once confirmed, its original class only needs to meet ``exit_threshold``.
    An unsupported gap strictly greater than ``gap_seconds`` closes the event
    before the current observation is considered, including when that current
    observation would otherwise support the same class.

    A changed window ID closes confirmed evidence and discards tentative
    evidence *before* considering ``eligible``. An ineligible observation also
    clears evidence, so warmup or invalid model context cannot bridge events.
    Use a fresh detector for each video; timestamps must increase strictly.
    """

    def __init__(self, labels: list[str], allowed_ids: set[int] | None,
                 config: TriggerConfig) -> None:
        if (not labels or any(not isinstance(label, str) or not label.strip()
                              for label in labels)):
            raise ValueError("labels must contain nonempty strings")
        if len(set(labels)) != len(labels):
            raise ValueError("labels must be unique")
        if not isinstance(config, TriggerConfig):
            raise TypeError("config must be a TriggerConfig")
        self.labels = list(labels)
        self.config = config
        if allowed_ids is None:
            self.allowed_ids = None
        else:
            if any(isinstance(index, bool) or not isinstance(index, Integral)
                   or not 0 <= index < len(labels) for index in allowed_ids):
                raise ValueError("allowed_ids must be valid integer class IDs")
            self.allowed_ids = frozenset(int(index) for index in allowed_ids)
        self._evidence: _Evidence | None = None
        self._last_timestamp: float | None = None
        self._window_id: int | None = None

    @property
    def active_label(self) -> str | None:
        evidence = self._evidence
        if evidence is None or evidence.confirmed_at is None:
            return None
        return self.labels[evidence.class_id]

    def _close(self, reason: str) -> list[dict]:
        evidence, self._evidence = self._evidence, None
        if evidence is None or evidence.confirmed_at is None:
            return []
        return [{
            "label": self.labels[evidence.class_id],
            "class_id": evidence.class_id,
            "start_seconds": evidence.start,
            "end_seconds": evidence.last_support,
            "confirmed_at_seconds": evidence.confirmed_at,
            "peak_probability": evidence.peak,
            "mean_probability": evidence.probability_sum / evidence.count,
            "supported_observations": evidence.count,
            "context_start_seconds": evidence.context_start,
            "context_end_seconds": evidence.last_support,
            "window_id": evidence.window_id,
            "closed_by": reason,
        }]

    def _qualifying_top_class(self, probabilities: np.ndarray) -> int | None:
        top_id = int(np.argmax(probabilities))
        if self.allowed_ids is not None and top_id not in self.allowed_ids:
            return None
        score = float(probabilities[top_id])
        runner_up = (float(np.partition(probabilities, -2)[-2])
                     if len(probabilities) > 1 else 0.0)
        if (score + _EPSILON < self.config.enter_threshold
                or score - runner_up + _EPSILON < self.config.margin):
            return None
        return top_id

    def _validate_observation(self, timestamp: float,
                              probabilities: np.ndarray,
                              context_start: float, window_id: int,
                              eligible: bool) -> tuple[float, np.ndarray, float, int]:
        timestamp = _finite_number("timestamp", timestamp)
        context_start = _finite_number("context_start", context_start)
        if timestamp < 0 or context_start < 0 or context_start > timestamp:
            raise ValueError("require 0 <= context_start <= timestamp")
        if self._last_timestamp is not None and timestamp <= self._last_timestamp:
            raise ValueError("timestamps must increase strictly")
        if (isinstance(window_id, bool) or not isinstance(window_id, Integral)
                or window_id < 0):
            raise ValueError("window_id must be a nonnegative integer")
        window_id = int(window_id)
        if self._window_id is not None and window_id < self._window_id:
            raise ValueError("window_id must not decrease")
        if not isinstance(eligible, (bool, np.bool_)):
            raise ValueError("eligible must be a boolean")
        try:
            values = np.asarray(probabilities, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("probabilities must be a numeric vector") from error
        if values.shape != (len(self.labels),):
            raise ValueError(f"probabilities must have shape ({len(self.labels)},)")
        if (not np.all(np.isfinite(values))
                or np.any(values < 0.0) or np.any(values > 1.0)):
            raise ValueError("probabilities must be finite values in [0, 1]")
        if not np.isclose(values.sum(), 1.0, rtol=0.0, atol=1e-4):
            raise ValueError("probabilities must sum to 1; pass full softmax scores")
        return timestamp, values, context_start, window_id

    def update(self, timestamp: float, probabilities: np.ndarray,
               context_start: float, window_id: int,
               eligible: bool = True) -> list[dict]:
        """Consume one sampled prediction, returning newly completed events."""
        timestamp, values, context_start, window_id = self._validate_observation(
            timestamp, probabilities, context_start, window_id, eligible)
        completed: list[dict] = []
        if self._window_id is not None and window_id != self._window_id:
            completed.extend(self._close("window_reset"))
        self._window_id = window_id
        self._last_timestamp = timestamp
        if not eligible:
            completed.extend(self._close("ineligible"))
            return completed

        evidence = self._evidence
        if (evidence is not None
                and timestamp - evidence.last_support
                > self.config.gap_seconds + _EPSILON):
            completed.extend(self._close("gap"))
            evidence = None

        if evidence is not None and evidence.confirmed_at is not None:
            probability = float(values[evidence.class_id])
            if probability + _EPSILON >= self.config.exit_threshold:
                evidence.support(timestamp, probability, context_start)
            return completed

        top_id = self._qualifying_top_class(values)
        if top_id is None:
            self._evidence = None
            return completed
        if evidence is None or evidence.class_id != top_id:
            self._evidence = _Evidence(
                class_id=top_id, start=timestamp, last_support=timestamp,
                context_start=context_start, window_id=window_id, count=1,
                probability_sum=float(values[top_id]), peak=float(values[top_id]))
        else:
            evidence.support(timestamp, float(values[top_id]), context_start)

        evidence = self._evidence
        if (evidence.count >= self.config.min_observations
                and evidence.last_support - evidence.start + _EPSILON
                >= self.config.min_duration_seconds):
            evidence.confirmed_at = timestamp
        return completed

    def finish(self, reason: str = "end_of_video") -> list[dict]:
        """Flush confirmed evidence once and discard any tentative candidate."""
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason must be a nonempty string")
        return self._close(reason)
