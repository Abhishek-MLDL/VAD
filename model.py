"""Native PyTorch, FP32 MoViNet-A2-Stream inference.

One instance owns mutable streaming buffers. Use it serially; create separate
instances for concurrent videos. TensorFlow and torchvision are not required.
"""
from __future__ import annotations

import importlib.metadata
from pathlib import Path

import numpy as np

from assets import DEFAULT_WEIGHTS, UPSTREAM_COMMIT, WEIGHTS_SHA256, sha256_file


class MoViNetA2Stream:
    def __init__(self, weights: Path = DEFAULT_WEIGHTS, device: str = "auto", threads: int = 4):
        try:
            import torch
            from vendor.movinets import MoViNet
            from vendor.movinets.config import _C
        except ImportError as error:
            raise RuntimeError("Install PyTorch and requirements.txt first; see README.md.") from error
        weights = Path(weights).expanduser().resolve()
        if not weights.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {weights}. Run python download_assets.py")
        self.weights_sha256 = sha256_file(weights)
        if self.weights_sha256 != WEIGHTS_SHA256:
            raise ValueError("Checkpoint does not match the pinned pretrained A2-Stream weights.")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("Use cpu, cuda, cuda:N, or auto. Other devices are unverified.")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable. Install a matching CUDA PyTorch wheel or use --device cpu.")
        if threads > 0:
            torch.set_num_threads(threads)
        # pretrained=False avoids an implicit, unpinned torch.hub download.
        # These options must be explicit when loading stream weights manually.
        self.model = MoViNet(
            _C.MODEL.MoViNetA2.clone(), causal=True, pretrained=False,
            num_classes=600, conv_type="2plus1d", tf_like=True,
        )
        state = torch.load(str(weights), map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()
        self.model.requires_grad_(False)
        self.torch = torch
        self.reset()

    def reset(self) -> None:
        """Reset convolutions, cumulative pooling buffers, and frame counters."""
        self.model.clean_activation_buffers()

    def predict(self, rgb_chw: np.ndarray) -> np.ndarray:
        """Consume one RGB [3,224,224] float32 frame; return 600 probabilities."""
        if (not isinstance(rgb_chw, np.ndarray) or rgb_chw.shape != (3, 224, 224)
                or rgb_chw.dtype != np.float32 or not np.isfinite(rgb_chw).all()
                or rgb_chw.min() < 0 or rgb_chw.max() > 1):
            raise ValueError("Expected RGB float32 [3,224,224] pixels in [0,1].")
        torch = self.torch
        frame = torch.from_numpy(np.ascontiguousarray(rgb_chw))[None, :, None].to(self.device)
        with torch.inference_mode():
            logits = self.model(frame)  # B,C,T,H,W input; [1,600] logits output.
            if tuple(logits.shape) != (1, 600):
                raise RuntimeError(f"Unexpected model output shape: {tuple(logits.shape)}")
            probabilities = logits.softmax(dim=1)[0].cpu().numpy()
        if not np.isfinite(probabilities).all():
            raise RuntimeError("Model produced non-finite probabilities.")
        return probabilities

    def metadata(self) -> dict:
        torch = self.torch
        return {
            "name": "MoViNet-A2-Stream", "backend": "pytorch",
            "implementation": "Atze00/MoViNet-pytorch (unofficial port)",
            "upstream_commit": UPSTREAM_COMMIT, "weights_sha256": self.weights_sha256,
            "torch_version": torch.__version__, "numpy_version": np.__version__,
            "einops_version": importlib.metadata.version("einops"),
            "device": str(self.device), "precision": "float32",
            "cuda_runtime": torch.version.cuda,
            "device_name": (torch.cuda.get_device_name(self.device)
                            if self.device.type == "cuda" else "CPU"),
            "threads": torch.get_num_threads(), "classes": 600,
            "input": "RGB float32 [1,3,1,224,224], direct bilinear resize, range [0,1]",
        }
