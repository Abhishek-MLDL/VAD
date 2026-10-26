# Delivery validation

Validated on 2026-10-26 with Python 3.12.14, Windows x64, PyTorch 2.6.0+cpu,
NumPy 1.26.4, OpenCV 4.10.0, einops 0.8.1 and 4 PyTorch CPU threads.

## Automated checks

`python -m unittest discover -s tests -v` completed with **49 passed, 0 skipped,
0 failed**. This includes:

- 3 real-checkpoint tests: valid 600-class probabilities, identical predictions
  after resetting a repeated input sequence, frame/chunk final-prediction
  agreement, and clearing cumulative counters and activation buffers.
- 14 video pipeline tests, including real MP4 encoding/decoding with controlled
  model predictions: correct source/output frame counts and FPS, sampling,
  low-FPS handling, warmup and reset boundaries, EOF, processing limits, class
  filtering, error cleanup, and existing-output protection.
- 21 temporal tests: global top-1 selection, original probabilities, persistence,
  hysteresis, gap closure at last supported time, warmup, resets, and input
  validation.
- 11 sampling, preprocessing, file discovery, overlay, and audio command tests.

The included checkpoint and label map passed their pinned SHA-256 checks.

## Full CLI smoke run

The [official TensorFlow tutorial sample GIF](https://raw.githubusercontent.com/tensorflow/models/f8af2291cced43fc9f1d9b41ddbf772ae7b0d7d2/official/projects/movinet/files/jumpingjack.gif)
was repeated four times, encoded as a 30 FPS MP4, and given a synthetic sine-wave
audio track to exercise audio preservation. This creates a short repeatable
functional fixture, rather than an accuracy benchmark. The
[tutorial](https://www.tensorflow.org/hub/tutorials/movinet) attributes the original
footage to Coach Bobby Bluford under CC BY. The footage is not redistributed in
this package.

The actual pretrained PyTorch model was run through `run_video.py` with default
temporal settings, `--all-scores`, `--device cpu`, and `--keep-audio`.

| Check | Observed result |
|---|---|
| Input | 195 frames, 30 FPS, 6.5 seconds |
| Model sampling | 33 observations, target 5 FPS |
| Candidate | `jumping jacks`, class 242 |
| Evidence interval | 1.0–6.4 seconds; confirmed at 1.4 seconds |
| Annotated video | 195 frames, 30 FPS, 6.5 seconds |
| Audio-preserving output | AAC audio present; full FFmpeg decode succeeded |
| Logs | JSONL/CSV predictions, JSON/CSV events, metadata and batch summary produced |
| Warnings in processing metadata | None |

An annotated frame was visually inspected for readable labels. This verifies
execution and output integrity. It does not measure recognition accuracy,
false-positive rate, boundary accuracy, long-video behavior, GPU performance, or
quality under occlusion/blur. Hugging/kissing accuracy and suitable thresholds
must be measured on representative MomentPilot recordings. CUDA execution was
not exercised on this CPU-only test environment.
