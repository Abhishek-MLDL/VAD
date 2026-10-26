"""Optional real-checkpoint tests; run after installing PyTorch and downloading assets.

These validate streaming execution and reset behavior, not recognition accuracy.
No model or network download is initiated by the tests themselves.
"""
from pathlib import Path
import unittest

import numpy as np

from assets import DEFAULT_WEIGHTS

try:
    import torch
    import einops  # noqa: F401 -- required by the vendored model
    import yacs  # noqa: F401 -- required by its configuration
except ImportError as exc:
    DEPENDENCY_ERROR = str(exc)
else:
    DEPENDENCY_ERROR = ""


@unittest.skipIf(bool(DEPENDENCY_ERROR), f"Optional model dependencies unavailable: {DEPENDENCY_ERROR}")
@unittest.skipUnless(Path(DEFAULT_WEIGHTS).is_file(), "Download the A2-Stream checkpoint with download_assets.py first")
class RealCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from model import MoViNetA2Stream
        cls.adapter = MoViNetA2Stream(DEFAULT_WEIGHTS, device="cpu", threads=4)
        generator = np.random.default_rng(20261026)
        cls.frames = generator.random((3, 3, 224, 224), dtype=np.float32)

    def setUp(self):
        self.adapter.reset()

    def tearDown(self):
        self.adapter.reset()

    def test_reset_replays_an_identical_video_without_cross_video_state(self):
        first = [self.adapter.predict(frame).copy() for frame in self.frames]
        self.adapter.reset()
        second = [self.adapter.predict(frame).copy() for frame in self.frames]
        for before, after in zip(first, second):
            self.assertEqual(before.shape, (600,))
            self.assertTrue(np.isfinite(before).all())
            self.assertTrue(np.all(before >= 0))
            self.assertAlmostEqual(float(before.sum()), 1.0, places=5)
            np.testing.assert_allclose(before, after, rtol=1e-5, atol=1e-6)

    def test_frame_stream_matches_a_single_chunk_final_prediction(self):
        for frame in self.frames:
            streamed = self.adapter.predict(frame).copy()
        self.adapter.reset()
        # N,C,T,H,W. A fresh chunk and the same sequence frame by frame must
        # produce the same final cumulative prediction for a causal model.
        chunk = torch.from_numpy(np.ascontiguousarray(self.frames.transpose(1, 0, 2, 3)))[None]
        with torch.inference_mode():
            logits = self.adapter.model(chunk)
            chunked = logits.softmax(dim=1)[0].numpy()
        self.assertEqual(tuple(logits.shape), (1, 600))
        np.testing.assert_allclose(streamed, chunked, rtol=1e-4, atol=1e-5)

    def test_reset_clears_cumulative_pool_counters_and_stream_buffers(self):
        self.adapter.predict(self.frames[0])
        pools = [module for module in self.adapter.model.modules()
                 if hasattr(module, "n_cumulated_values")]
        causal_modules = [module for module in self.adapter.model.modules()
                          if hasattr(module, "reset_activation")]
        self.assertTrue(pools)
        self.assertTrue(any(module.n_cumulated_values > 0 for module in pools))
        self.assertTrue(any(module.activation is not None for module in causal_modules))
        self.adapter.reset()
        self.assertTrue(all(module.n_cumulated_values == 0 for module in pools))
        self.assertTrue(all(module.activation is None for module in causal_modules))


if __name__ == "__main__":
    unittest.main()
