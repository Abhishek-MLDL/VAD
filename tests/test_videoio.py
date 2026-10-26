"""Tests that do not require weights, PyTorch, video codecs, or FFmpeg."""

from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from videoio import FrameSampler, discover_videos, draw_overlay, mux_audio, output_slug, preprocess_bgr


class FrameSamplerTests(unittest.TestCase):
    def test_noninteger_rate_follows_absolute_target_times(self):
        sampler = FrameSampler(29.97, 5.0)
        selected = [index for index in range(30_000) if sampler.take(index)]
        # Every target is met by the earliest available source frame. This
        # catches cumulative drift caused by rounding the sampling interval.
        for tick, frame_index in enumerate(selected):
            target = tick / 5.0
            self.assertGreaterEqual(frame_index / 29.97 + 1e-10, target)
            if frame_index:
                self.assertLess((frame_index - 1) / 29.97, target)
        self.assertEqual(selected[0], 0)
        self.assertIn(1001, selected)

    def test_source_rate_below_requested_never_duplicates(self):
        sampler = FrameSampler(2.0, 5.0)
        self.assertEqual(sampler.effective_fps, 2.0)
        for index in range(25):
            self.assertTrue(sampler.take(index))
            self.assertFalse(sampler.take(index))

    def test_skipped_frames_do_not_restart_schedule(self):
        sampler = FrameSampler(30.0, 5.0)
        self.assertTrue(sampler.take(0))
        self.assertTrue(sampler.take(14))
        self.assertFalse(sampler.take(15))
        self.assertTrue(sampler.take(18))
        with self.assertRaises(ValueError):
            sampler.take(17)

    def test_invalid_parameters(self):
        for value in (0, -2, float("nan"), float("inf"), "invalid"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    FrameSampler(value, 5)
                with self.assertRaises(ValueError):
                    FrameSampler(30, value)
        for index in (-1, 0.5, True):
            with self.assertRaises(ValueError):
                FrameSampler(30, 5).take(index)


class PreprocessingTests(unittest.TestCase):
    def test_bgr_conversion_layout_and_normalization(self):
        frame = np.zeros((9, 17, 3), dtype=np.uint8)
        frame[:] = (10, 80, 255)
        result = preprocess_bgr(frame, size=7)
        self.assertEqual(result.shape, (3, 7, 7))
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(result.flags.c_contiguous)
        np.testing.assert_allclose(result[:, 0, 0], [1.0, 80 / 255, 10 / 255], rtol=1e-6)
        np.testing.assert_array_equal(frame[0, 0], [10, 80, 255])

    def test_square_bilinear_resize_retains_both_extremes(self):
        frame = np.zeros((1, 2, 3), dtype=np.uint8)
        frame[0, 1] = 255
        result = preprocess_bgr(frame, size=3)
        np.testing.assert_allclose(result[:, 1, 0], 0)
        np.testing.assert_allclose(result[:, 1, -1], 1)
        np.testing.assert_allclose(result[:, 1, 1], 128 / 255, atol=1 / 255)

    def test_invalid_frames_and_size(self):
        for frame in (np.zeros((3, 3), np.uint8), np.zeros((0, 3, 3), np.uint8),
                      np.zeros((3, 3, 3), np.float32)):
            with self.assertRaises(ValueError):
                preprocess_bgr(frame)
        for size in (0, -1, 2.5, True):
            with self.assertRaises(ValueError):
                preprocess_bgr(np.zeros((2, 2, 3), np.uint8), size)

    def test_overlay_does_not_mutate_input_and_handles_tiny_frames(self):
        for shape in ((720, 1280, 3), (48, 64, 3), (1, 1, 3)):
            frame = np.full(shape, 100, dtype=np.uint8)
            result = draw_overlay(frame, ["hugging: 0.72", "very long description " * 15], active=True)
            self.assertEqual(result.shape, frame.shape)
            self.assertEqual(result.dtype, np.uint8)
            self.assertIsNot(result, frame)
            self.assertTrue(np.all(frame == 100))


class FileAndMuxTests(unittest.TestCase):
    def test_discovery_recurses_sorts_and_accepts_uppercase(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "nested").mkdir()
            expected = [root / "A.MP4", root / "nested" / "b.mov"]
            for path in expected + [root / "ignore.txt"]:
                path.touch()
            self.assertEqual(discover_videos(root), [path.resolve() for path in expected])
            self.assertEqual(discover_videos(expected[0]), [expected[0].resolve()])
            with self.assertRaises(ValueError):
                discover_videos(root / "ignore.txt")
            with self.assertRaises(FileNotFoundError):
                discover_videos(root / "missing")

    def test_empty_input_and_stable_unique_output_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                discover_videos(root)
            first, second = root / "one" / "my video.mp4", root / "two" / "my video.mp4"
            self.assertEqual(output_slug(first), output_slug(first))
            self.assertNotEqual(output_slug(first), output_slug(second))
            self.assertRegex(output_slug(first), r"^my_video_[0-9a-f]{8}$")

    def test_mux_preserves_video_duration_without_shell_or_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotated, source, output = root / "annotated.mp4", root / "source.mp4", root / "audio.mp4"
            annotated.touch()
            source.touch()
            capture = mock.Mock()
            capture.isOpened.return_value = True
            capture.get.side_effect = [25.0, 75.0]
            process = mock.Mock(returncode=0, stderr="", stdout="")
            with mock.patch("videoio.cv2.VideoCapture", return_value=capture), \
                 mock.patch("videoio.subprocess.run", return_value=process) as run:
                mux_audio(annotated, source, output, ffmpeg="custom ffmpeg")
            command = run.call_args.args[0]
            self.assertEqual(command[0], "custom ffmpeg")
            self.assertIn("-n", command)
            self.assertNotIn("-shortest", command)
            self.assertEqual(command[command.index("-t") + 1], "3.000000000")
            self.assertIn("1:a?", command)
            self.assertNotIn("shell", run.call_args.kwargs)
            capture.release.assert_called_once()
            output.touch()
            with self.assertRaises(FileExistsError):
                mux_audio(annotated, source, output)


if __name__ == "__main__":
    unittest.main()
