"""Pipeline tests use decoded synthetic videos and deterministic fake scores.

These verify video timing, state isolation, and output semantics, not pretrained
model accuracy. No weights, GPU, network connection, or FFmpeg are required.
"""

from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_video import build_parser, process_video, resolve_classes, validate_args
from videoio import output_slug


JUMPING_JACKS = 242


class FakePredictor:
    """600-class normalized scores with explicitly observable mutable state."""

    def __init__(self):
        self.reset_calls = 0
        self.total_calls = 0
        self.since_reset = 0
        self.observations: list[dict] = []
        self.invalid_shape = False

    def reset(self):
        self.reset_calls += 1
        self.since_reset = 0

    def predict(self, frame):
        self.total_calls += 1
        self.since_reset += 1
        self.observations.append({
            "since_reset": self.since_reset,
            "shape": frame.shape,
            "dtype": frame.dtype,
            "minimum": float(frame.min()),
            "maximum": float(frame.max()),
        })
        if self.invalid_shape:
            return np.zeros(599, np.float32)
        scores = np.zeros(600, np.float32)
        scores[JUMPING_JACKS] = 0.98
        scores[1] = 0.02
        return scores

    def metadata(self):
        return {"name": "synthetic test predictor", "classes": 600}


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        # A full-size synthetic label map keeps tests independent of downloads.
        self.labels = [f"class_{index:03d}" for index in range(600)]
        self.labels[JUMPING_JACKS] = "jumping jacks"
        self.model = FakePredictor()

    def fixture(self, *, name="source.mp4", fps=5.0, count=20):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (48, 32))
        if not writer.isOpened():
            self.skipTest("This OpenCV build cannot encode the MP4 fixtures.")
        try:
            for index in range(count):
                frame = np.zeros((32, 48, 3), np.uint8)
                frame[:] = (index * 7 % 240, 80, 220)
                writer.write(frame)
        finally:
            writer.release()
        return path

    def arguments(self, source, *extra):
        args = build_parser().parse_args([
            "--input", str(source), "--warmup-seconds", "0",
            "--state-window-seconds", "0", "--min-observations", "2",
            "--min-duration-seconds", "0.2", *extra,
        ])
        validate_args(args)
        return args

    def process(self, source, *, destination="results", extra=(), allowed_ids=None):
        with redirect_stdout(io.StringIO()):
            return process_video(
                source, self.root / destination, self.model, self.labels,
                allowed_ids, self.arguments(source, *extra),
            )

    def records(self, destination="results"):
        return [json.loads(line) for line in
                (self.root / destination / "predictions.jsonl").read_text().splitlines()]

    def events(self, destination="results"):
        return json.loads((self.root / destination / "events.json").read_text())

    def test_low_source_fps_is_not_duplicated_and_annotated_timing_is_preserved(self):
        source = self.fixture(fps=4, count=16)
        result = self.process(source, extra=("--sample-fps", "5", "--all-scores"))
        self.assertEqual(result["processed_frames"], 16)
        self.assertEqual(result["sampled_frames"], 16)
        self.assertEqual(result["sample_fps"], 4)
        self.assertAlmostEqual(result["duration_seconds"], 4)
        self.assertEqual(result["dimensions"], [48, 32])
        self.assertEqual(result["model"]["name"], "synthetic test predictor")
        records = self.records()
        self.assertEqual([row["frame_index"] for row in records], list(range(16)))
        self.assertEqual([row["timestamp_seconds"] for row in records],
                         [index / 4 for index in range(16)])
        self.assertEqual(len(records[0]["probabilities"]), 600)
        self.assertAlmostEqual(records[0]["probabilities"][JUMPING_JACKS], .98, places=6)
        self.assertEqual(records[0]["top_k"][0]["class_id"], JUMPING_JACKS)
        event, = self.events()
        self.assertEqual(event["label"], "jumping jacks")
        self.assertEqual(event["confirmed_at_seconds"], .25)
        self.assertEqual(event["end_seconds"], 3.75)
        self.assertEqual(event["supported_observations"], 16)
        capture = cv2.VideoCapture(result["annotated_video"])
        try:
            self.assertTrue(capture.isOpened())
            self.assertAlmostEqual(capture.get(cv2.CAP_PROP_FPS), 4, places=3)
            decoded = 0
            while capture.read()[0]:
                decoded += 1
            self.assertEqual(decoded, 16)
        finally:
            capture.release()
        for observation in self.model.observations:
            self.assertEqual(observation["shape"], (3, 224, 224))
            self.assertEqual(observation["dtype"], np.float32)
            self.assertGreaterEqual(observation["minimum"], 0)
            self.assertLessEqual(observation["maximum"], 1)

    def test_sampling_skips_frames_without_changing_output_frame_count(self):
        source = self.fixture(fps=10, count=20)
        result = self.process(source, extra=("--sample-fps", "5"))
        self.assertEqual(result["processed_frames"], 20)
        self.assertEqual(result["sampled_frames"], 10)
        self.assertEqual([row["frame_index"] for row in self.records()], list(range(0, 20, 2)))
        capture = cv2.VideoCapture(result["annotated_video"])
        try:
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 20)
            self.assertAlmostEqual(capture.get(cv2.CAP_PROP_FPS), 10, places=3)
        finally:
            capture.release()

    def test_every_video_and_state_window_begins_with_clean_model_state(self):
        first = self.fixture(name="first.mp4", count=6)
        second = self.fixture(name="second.mp4", count=6)
        self.process(first, destination="first_results", extra=("--no-video",))
        first_total = self.model.total_calls
        self.process(second, destination="second_results", extra=("--no-video",))
        self.assertEqual(self.model.observations[0]["since_reset"], 1)
        self.assertEqual(self.model.observations[first_total]["since_reset"], 1)
        self.assertEqual(self.model.since_reset, 0)  # final cleanup resets buffers
        self.assertEqual(self.model.reset_calls, 6)  # entry, first window, final cleanup per video
        self.assertEqual(self.events("first_results")[0]["start_seconds"], 0)
        self.assertEqual(self.events("second_results")[0]["start_seconds"], 0)

    def test_window_resets_suppress_warmup_and_cannot_bridge_candidates(self):
        source = self.fixture(fps=5, count=20)
        result = self.process(source, extra=(
            "--no-video", "--state-window-seconds", "2", "--warmup-seconds", ".75",
        ))
        self.assertEqual(result["state_windows"], 2)
        records = self.records()
        self.assertEqual([row["window_id"] for row in records], [0] * 10 + [1] * 10)
        self.assertEqual([row["eligible"] for row in records],
                         ([False] * 4 + [True] * 6) * 2)
        self.assertEqual(self.model.observations[10]["since_reset"], 1)
        events = self.events()
        self.assertEqual(len(events), 2)
        self.assertAlmostEqual(events[0]["start_seconds"], .8)
        self.assertAlmostEqual(events[0]["end_seconds"], 1.8)
        self.assertEqual(events[0]["closed_by"], "window_reset")
        self.assertAlmostEqual(events[1]["start_seconds"], 2.8)
        self.assertEqual(events[1]["context_start_seconds"], 2)
        self.assertEqual(events[1]["closed_by"], "end_of_video")

    def test_processing_limit_reports_intentional_stop_and_flushes_event(self):
        source = self.fixture(fps=5, count=20)
        result = self.process(source, extra=("--no-video", "--max-seconds", "1"))
        self.assertEqual(result["processed_frames"], 5)
        self.assertEqual(result["sampled_frames"], 5)
        self.assertTrue(result["processing_limit_reached"])
        self.assertAlmostEqual(result["duration_seconds"], 1)
        self.assertEqual(result["warnings"], [])
        self.assertEqual(self.events()[0]["closed_by"], "processing_limit")

    def test_explicit_fps_override_drives_sampling_and_timestamps(self):
        source = self.fixture(fps=4, count=16)
        result = self.process(source, extra=("--no-video", "--fps", "8"))
        self.assertTrue(result["fps_overridden"])
        self.assertAlmostEqual(result["reported_fps"], 4, places=3)
        self.assertEqual(result["fps"], 8)
        self.assertEqual(result["duration_seconds"], 2)
        self.assertEqual(result["sampled_frames"], 10)
        self.assertEqual([row["frame_index"] for row in self.records()],
                         [0, 2, 4, 5, 7, 8, 10, 12, 13, 15])

    def test_low_fps_default_gap_allows_evidence_but_explicit_short_gap_warns(self):
        source = self.fixture(fps=2, count=6)
        adaptive = self.process(source, destination="adaptive", extra=("--no-video",))
        self.assertEqual(adaptive["trigger_config"]["gap_seconds"], .75)
        self.assertEqual(adaptive["candidate_events"], 1)
        short = self.process(source, destination="short", extra=(
            "--no-video", "--gap-seconds", ".3",
        ))
        self.assertEqual(short["candidate_events"], 0)
        self.assertEqual(len(short["warnings"]), 1)
        self.assertIn("shorter than a sample interval", short["warnings"][0])

    def test_empty_allowlist_logs_predictions_without_candidates(self):
        source = self.fixture(count=6)
        result = self.process(source, extra=("--no-video",), allowed_ids=set())
        self.assertEqual(result["candidate_events"], 0)
        self.assertEqual(result["selected_classes"], [])
        self.assertEqual(self.events(), [])
        self.assertTrue(self.records())
        self.assertEqual(self.records()[0]["selected_class_probabilities"], {})

    def test_selected_class_scores_are_original_probabilities(self):
        source = self.fixture(count=6)
        result = self.process(source, extra=("--no-video",), allowed_ids={JUMPING_JACKS})
        self.assertEqual(result["selected_classes"], ["jumping jacks"])
        score = self.records()[0]["selected_class_probabilities"]["jumping jacks"]
        self.assertAlmostEqual(score, .98, places=6)  # never renormalize selected labels

    def test_unique_slug_outputs_and_existing_destination_refusal(self):
        first = self.fixture(name="one/same.mp4", count=3)
        second = self.fixture(name="two/same.mp4", count=3)
        first_slug, second_slug = output_slug(first), output_slug(second)
        self.assertNotEqual(first_slug, second_slug)
        first_result = self.process(first, destination=first_slug, extra=("--no-video",))
        second_result = self.process(second, destination=second_slug, extra=("--no-video",))
        self.assertNotEqual(first_result["output_directory"], second_result["output_directory"])
        before = (self.root / first_slug / "metadata.json").read_bytes()
        resets_before = self.model.reset_calls
        with self.assertRaises(FileExistsError):
            self.process(first, destination=first_slug, extra=("--no-video",))
        self.assertEqual((self.root / first_slug / "metadata.json").read_bytes(), before)
        self.assertEqual(self.model.reset_calls, resets_before)

    def test_broken_or_empty_input_fails_and_resets_state(self):
        broken = self.root / "broken.mp4"
        broken.write_bytes(b"This is not a video file.")
        with self.assertRaisesRegex(RuntimeError, "Cannot open video"):
            self.process(broken, extra=("--no-video",))
        self.assertEqual(self.model.since_reset, 0)
        empty_capture = mock.Mock()
        empty_capture.isOpened.return_value = True
        empty_capture.get.side_effect = lambda prop: 5.0 if prop == cv2.CAP_PROP_FPS else 0.0
        empty_capture.read.return_value = (False, None)
        with mock.patch("run_video.cv2.VideoCapture", return_value=empty_capture):
            with self.assertRaisesRegex(RuntimeError, "No frames decoded"):
                self.process(broken, destination="empty_results", extra=("--no-video",))
        empty_capture.release.assert_called_once()
        self.assertEqual(self.model.since_reset, 0)

    def test_prediction_failure_resets_model_and_next_video_can_succeed(self):
        source = self.fixture(count=6)
        self.model.invalid_shape = True
        with self.assertRaisesRegex(ValueError, "shape"):
            self.process(source, destination="failed", extra=("--no-video",))
        self.assertEqual(self.model.since_reset, 0)
        self.model.invalid_shape = False
        result = self.process(source, destination="recovered", extra=("--no-video",))
        self.assertEqual(result["status"], "success")
        self.assertEqual(self.model.observations[1]["since_reset"], 1)

    def test_class_selection_rejects_unknown_and_malformed_maps(self):
        self.assertEqual(resolve_classes(self.labels, "jumping jacks"), {JUMPING_JACKS})
        self.assertIsNone(resolve_classes(self.labels, "all"))
        with self.assertRaisesRegex(ValueError, "Unknown Kinetics-600"):
            resolve_classes(self.labels, "hugs")
        path = self.root / "classes.json"
        for content in ({"classes": ["jumping jacks"]}, [242], "jumping jacks"):
            path.write_text(json.dumps(content))
            with self.assertRaisesRegex(ValueError, "JSON array"):
                resolve_classes(self.labels, classes_file=path)
        path.write_text("[]")
        self.assertEqual(resolve_classes(self.labels, classes_file=path), set())

    def test_cli_rejects_invalid_numeric_settings_and_conflicting_outputs(self):
        invalid_options = [
            ("--sample-fps=nan",), ("--sample-fps=0",), ("--fps=0",),
            ("--max-seconds=inf",), ("--state-window-seconds=-1",),
            ("--warmup-seconds=nan",), ("--state-window-seconds=1", "--warmup-seconds=1"),
            ("--threads=-1",), ("--top-k=0",), ("--top-k=601",),
            ("--enter-threshold=1.1",), ("--exit-threshold=.8", "--enter-threshold=.5"),
            ("--margin=nan",), ("--min-observations=1",),
            ("--min-duration-seconds=-1",), ("--gap-seconds=0",),
            ("--no-video", "--keep-audio"),
        ]
        for options in invalid_options:
            with self.subTest(options=options):
                args = build_parser().parse_args(["--input", "test.mp4", *options])
                with self.assertRaises(ValueError):
                    validate_args(args)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            build_parser().parse_args([
                "--input", "test.mp4", "--classes", "jumping jacks",
                "--classes-file", "classes.json",
            ])


if __name__ == "__main__":
    unittest.main()
