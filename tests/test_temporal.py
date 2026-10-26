"""Temporal logic tests use synthetic probabilities, not model predictions."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from temporal import TemporalDetector, TriggerConfig


class TemporalDetectorTests(unittest.TestCase):
    def detector(self, allowed_ids=None, **options):
        return TemporalDetector(["hugging", "kissing", "walking"], allowed_ids,
                                TriggerConfig(**options))

    @staticmethod
    def observe(detector, timestamp, scores=(0.8, 0.1, 0.1), **kwargs):
        return detector.update(timestamp, np.array(scores),
                               kwargs.pop("context_start", 0.0),
                               kwargs.pop("window_id", 0), **kwargs)

    def confirm(self, detector):
        for timestamp in (0.0, 0.2, 0.4):
            self.assertEqual(self.observe(detector, timestamp), [])
        self.assertEqual(detector.active_label, "hugging")

    def test_count_and_observed_span_both_required(self):
        detector = self.detector()
        for timestamp in (0.0, 0.1, 0.2):
            self.observe(detector, timestamp)
        self.assertIsNone(detector.active_label)
        self.observe(detector, 0.4)
        event, = detector.finish()
        self.assertEqual(event["supported_observations"], 4)
        self.assertEqual(event["confirmed_at_seconds"], 0.4)
        self.assertEqual(event["start_seconds"], 0.0)
        self.assertEqual(event["end_seconds"], 0.4)

    def test_single_spike_and_unconfirmed_eof_emit_nothing(self):
        detector = self.detector()
        self.observe(detector, 0)
        self.observe(detector, 0.2, (0.34, 0.33, 0.33))
        self.assertEqual(detector.finish(), [])
        detector = self.detector()
        self.observe(detector, 0)
        self.observe(detector, 0.2)
        self.assertEqual(detector.finish(), [])

    def test_excluded_global_top1_is_not_replaced_by_allowed_runner_up(self):
        detector = self.detector(allowed_ids={0}, enter_threshold=0.3,
                                 margin=0.0)
        for timestamp in (0.0, 0.2, 0.4, 0.6):
            self.observe(detector, timestamp, (0.4, 0.5, 0.1))
        self.assertEqual(detector.finish(), [])

    def test_selected_subset_is_never_renormalized(self):
        detector = self.detector(allowed_ids={0})
        for timestamp in (0.0, 0.2, 0.4, 0.6):
            self.observe(detector, timestamp, (0.4, 0.3, 0.3))
        self.assertEqual(detector.finish(), [])

    def test_empty_allowlist_disables_events(self):
        detector = self.detector(allowed_ids=set())
        for timestamp in (0.0, 0.2, 0.4):
            self.observe(detector, timestamp)
        self.assertEqual(detector.finish(), [])

    def test_margin_uses_all_classes_and_ties_do_not_enter(self):
        detector = self.detector(enter_threshold=0.45, margin=0.1)
        for timestamp in (0.0, 0.2, 0.4):
            self.observe(detector, timestamp, (0.49, 0.48, 0.03))
        self.assertEqual(detector.finish(), [])

    def test_hysteresis_supports_original_class_below_entry(self):
        detector = self.detector()
        self.confirm(detector)
        self.observe(detector, 0.6, (0.35, 0.45, 0.2))
        event, = detector.finish()
        self.assertEqual(event["end_seconds"], 0.6)
        self.assertEqual(event["supported_observations"], 4)
        self.assertAlmostEqual(event["mean_probability"], (2.4 + 0.35) / 4)
        self.assertEqual(event["peak_probability"], 0.8)

    def test_gap_ends_at_last_support_not_expiration(self):
        detector = self.detector()
        self.confirm(detector)
        self.observe(detector, 0.6, (0.1, 0.45, 0.45))
        self.observe(detector, 0.8, (0.1, 0.45, 0.45))
        self.assertEqual(self.observe(detector, 1.0, (0.1, 0.45, 0.45)), [])
        event, = self.observe(detector, 1.1, (0.1, 0.45, 0.45))
        self.assertEqual(event["end_seconds"], 0.4)
        self.assertEqual(event["closed_by"], "gap")
        self.assertIsNone(detector.active_label)

    def test_short_gap_is_bridged_without_counting_unsupported_scores(self):
        detector = self.detector()
        self.confirm(detector)
        self.observe(detector, 0.6, (0.1, 0.45, 0.45))
        self.observe(detector, 0.8, (0.4, 0.4, 0.2))
        event, = detector.finish()
        self.assertEqual(event["end_seconds"], 0.8)
        self.assertEqual(event["supported_observations"], 4)
        self.assertAlmostEqual(event["mean_probability"], 0.7)

    def test_long_sample_gap_cannot_be_bridged_by_supporting_score(self):
        detector = self.detector()
        self.confirm(detector)
        event, = self.observe(detector, 1.1)
        self.assertEqual(event["end_seconds"], 0.4)
        self.assertEqual(event["closed_by"], "gap")
        self.assertIsNone(detector.active_label)
        self.assertEqual(detector.finish(), [])

    def test_pending_candidate_requires_consecutive_same_class(self):
        detector = self.detector()
        self.observe(detector, 0.0)
        self.observe(detector, 0.2, (0.1, 0.8, 0.1))
        self.observe(detector, 0.4)
        self.observe(detector, 0.6)
        self.assertIsNone(detector.active_label)
        self.observe(detector, 0.8)
        event, = detector.finish()
        self.assertEqual(event["start_seconds"], 0.4)

    def test_window_reset_precedes_warmup_and_never_reuses_evidence(self):
        detector = self.detector()
        self.confirm(detector)
        event, = self.observe(detector, 0.6, window_id=1,
                              context_start=0.6, eligible=False)
        self.assertEqual(event["closed_by"], "window_reset")
        self.assertEqual(event["window_id"], 0)
        self.assertEqual(event["end_seconds"], 0.4)
        self.assertIsNone(detector.active_label)
        for timestamp in (0.8, 1.0):
            self.observe(detector, timestamp, window_id=1, context_start=0.6)
        self.assertIsNone(detector.active_label)
        self.observe(detector, 1.2, window_id=1, context_start=0.6)
        event, = detector.finish()
        self.assertEqual(event["window_id"], 1)
        self.assertEqual(event["start_seconds"], 0.8)
        self.assertEqual(event["context_start_seconds"], 0.6)
        self.assertEqual(event["context_end_seconds"], 1.2)

    def test_warmup_predictions_do_not_count(self):
        detector = self.detector()
        for timestamp in (0.0, 0.2, 0.4):
            self.observe(detector, timestamp, eligible=False)
        self.observe(detector, 0.6)
        self.observe(detector, 0.8)
        self.assertEqual(detector.finish(), [])

    def test_ineligible_context_closes_active_candidate(self):
        detector = self.detector()
        self.confirm(detector)
        event, = self.observe(detector, 0.6, eligible=False)
        self.assertEqual(event["closed_by"], "ineligible")
        self.assertEqual(event["end_seconds"], 0.4)

    def test_finish_is_idempotent_and_reason_is_preserved(self):
        detector = self.detector()
        self.confirm(detector)
        event, = detector.finish("decode_stopped")
        self.assertEqual(event["closed_by"], "decode_stopped")
        self.assertEqual(detector.finish(), [])
        self.assertIsNone(detector.active_label)

    def test_new_detectors_do_not_share_video_state(self):
        first = self.detector()
        self.confirm(first)
        second = self.detector()
        self.assertIsNone(second.active_label)
        self.observe(second, 0.0)
        self.assertEqual(second.finish(), [])
        self.assertEqual(first.active_label, "hugging")

    def test_invalid_configuration(self):
        cases = [dict(enter_threshold=-0.1), dict(exit_threshold=0.8),
                 dict(margin=float("nan")), dict(min_observations=1),
                 dict(min_observations=2.5), dict(min_observations=True),
                 dict(min_duration_seconds=-1), dict(gap_seconds=0),
                 dict(gap_seconds=float("inf"))]
        for options in cases:
            with self.subTest(options=options), self.assertRaises(ValueError):
                TriggerConfig(**options)

    def test_invalid_labels_and_allowed_ids(self):
        cases = [([], None), (["a", "a"], None), ([""], None),
                 (["a"], {1}), (["a"], {-1}), (["a"], {True})]
        for labels, allowed in cases:
            with self.subTest(labels=labels, allowed=allowed):
                with self.assertRaises(ValueError):
                    TemporalDetector(labels, allowed, TriggerConfig())

    def test_invalid_probability_vectors(self):
        cases = [[0.8, 0.2], [[0.8, 0.1, 0.1]], [0.8, 0.1, np.nan],
                 [1.1, -0.1, 0.0], [0.8, 0.0, 0.0], [np.inf, 0.0, 0.0]]
        for values in cases:
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.detector().update(0.0, np.asarray(values), 0.0, 0)

    def test_invalid_time_context_and_window_values(self):
        cases = [(-1, 0, 0), (np.nan, 0, 0), (0, 1, 0),
                 (1, -1, 0), (1, np.inf, 0), (1, 0, -1), (1, 0, True)]
        for timestamp, context, window in cases:
            with self.subTest(values=(timestamp, context, window)):
                with self.assertRaises(ValueError):
                    self.observe(self.detector(), timestamp,
                                 context_start=context, window_id=window)
        detector = self.detector()
        self.observe(detector, 0.2, window_id=1)
        for timestamp in (0.1, 0.2):
            with self.assertRaises(ValueError):
                self.observe(detector, timestamp, window_id=1)
        with self.assertRaises(ValueError):
            self.observe(detector, 0.4, window_id=0)

    def test_rejected_update_does_not_change_detector_state(self):
        detector = self.detector()
        self.observe(detector, 0.0)
        with self.assertRaises(ValueError):
            self.observe(detector, 0.2, (0.0, 0.0, 0.0))
        self.observe(detector, 0.2)
        self.observe(detector, 0.4)
        event, = detector.finish()
        self.assertEqual(event["supported_observations"], 3)


if __name__ == "__main__":
    unittest.main()
