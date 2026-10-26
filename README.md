# MoViNet-A2-Stream video demo in PyTorch

This is a complete standalone demo for local videos or folders of videos. It uses
the pretrained **MoViNet-A2-Stream**, with native PyTorch execution and 600
Kinetics classes. The checkpoint and model source are included. TensorFlow,
torchvision, a Git installation, and an account on a model hosting service are
not needed.

It writes timestamped top-k action predictions, sustained action candidates in
JSON/CSV, an annotated MP4, and processing metadata. An optional class list
selects MomentPilot highlight candidates such as hugging, kissing, clapping,
celebrating, and opening a present.

## 1. Install

Use a new environment with **Python 3.10–3.12**. The verified setup is Python
3.12, PyTorch 2.6.0 CPU, NumPy 1.26.4 and OpenCV 4.10.0 on Windows. CUDA execution
is supported by the code but was not tested on this machine.

Extract the ZIP, then open a terminal **inside `movinet_a2_stream_demo`**.

Windows PowerShell, CPU:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe download_assets.py
.\.venv\Scripts\python.exe run_video.py --input "C:\videos\input.mp4" --output results --device cpu
```

Linux, CPU:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python download_assets.py
python run_video.py --input /path/to/input.mp4 --output results --device cpu
```

For an NVIDIA GPU on Linux or Windows, install the CUDA wheel **instead of the
CPU wheel** in a new environment, then install `requirements.txt`:

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"
python run_video.py --input input.mp4 --output results --device cuda
```

Use a driver compatible with that CUDA build. See the [official PyTorch 2.6.0
installation commands](https://pytorch.org/get-started/previous-versions/#v260)
for other builds. The demo uses FP32 without autocast. It does not require a
locally installed CUDA compiler.

The ZIP includes the 29.7 MB checkpoint. `download_assets.py` verifies SHA-256
checksums and downloads missing assets from pinned upstream URLs. It never
silently replaces a mismatched file. Once dependencies and assets are installed,
inference works offline.

## 2. Run

All 600 classes, CPU or automatically selected CUDA:

```bash
python run_video.py --input input.mp4 --output results
```

Only create events for the example highlight classes:

```bash
python run_video.py --input input.mp4 --classes-file highlight_classes.json --output results
```

Only hugging and kissing:

```bash
python run_video.py --input input.mp4 --classes "hugging (not baby),hugging baby,kissing" --output results
```

Process a directory recursively:

```bash
python run_video.py --input videos --output results --device cuda
```

Quick 20-second run, JSON/CSV only:

```bash
python run_video.py --input input.mp4 --max-seconds 20 --no-video --all-scores
```

Retain original audio in a second annotated MP4; install FFmpeg separately:

```bash
python run_video.py --input input.mp4 --keep-audio
```

Use `--ffmpeg "C:\ffmpeg\bin\ffmpeg.exe"` if FFmpeg is not on PATH. Default
`annotated.mp4` is silent. The optional audio file preserves the processed video
duration. The model itself uses visual input only.

An empty JSON class list (`[]`) disables event creation while retaining action
prediction logs. Class strings must exactly match `assets/kinetics_600_labels.txt`.
Unknown names fail early. Every run creates a new directory; inputs and previous
results are never overwritten. Batch failures are recorded and processing
continues with subsequent videos; any failure produces exit code 1.

## 3. Outputs

```text
results/run_<UTC timestamp>_<unique id>/<video name>_<path hash>/
    predictions.jsonl
    predictions.csv
    events.json
    events.csv
    metadata.json
    annotated.mp4                 # unless --no-video
    annotated_with_audio.mp4      # if --keep-audio
```

`batch_summary.json` is written in the run directory. Failed videos receive an
`error.json`; partial outputs from them must not be treated as completed results.

| Output | Meaning |
|---|---|
| `predictions.jsonl` | One observation per sampled frame; source frame/time, state window, top-k probabilities, selected-class probabilities, warmup flag and active candidate |
| `predictions.csv` | Compact top-1 timeline |
| `events.json/csv` | Sustained classifier evidence: label, first/last supporting time, confirmation time, peak/mean score and closure reason |
| `metadata.json` | Model/version/checksum, device, preprocessing, actual FPS, thresholds, decoded counts, durations, warnings and timing |

`--all-scores` adds the complete 600-score vector to each JSONL observation in
label-map order. All times are seconds from the start of the source video.
`event_id` is unique within one video's results; use the video result directory
together with it as a batch-wide identifier.

## 4. Model and temporal behavior

The input is RGB float32 in `[0,1]`, resized directly to **224 × 224**, sampled at
**5 fps**, and passed as `[1,3,1,224,224]`. There is no ImageNet mean/std
normalization. Sampling uses source frames at or after each scheduled sampling
time and never duplicates a frame. The saved timestamp is the actual decoded
frame index divided by source FPS.

The model runs in evaluation mode under `torch.inference_mode()`. It returns
600 logits; the code applies softmax once across **all 600 classes**. The selected
highlight list does not change or renormalize these probabilities.

MoViNet retains convolution buffers and cumulative pooling. For videos with
changing actions, this demo resets **all** buffers/counters every 10 seconds and
between videos. It logs predictions during the first 1 second after each reset
but suppresses candidate creation during that warmup. State windows are recorded
in the output. These resets bound history but can split an event or miss a short
action around a reset boundary. Predictions still describe accumulated context
inside the current window, so they can lag an action change.

Default candidate logic:

1. The **global** top-1 class must be selected, its probability at least `0.50`,
   and its lead over the second class at least `0.10`.
2. The same class must satisfy those conditions for at least 3 consecutive
   observations spanning at least `0.4` seconds.
3. Once confirmed, its score may fall to `0.30`. A sufficiently long unsupported
   gap ends it at the **last supported observation**, rather than at timeout.
4. End of video and state resets close confirmed events; tentative spikes are
   discarded. Confirmation time is recorded separately from evidence start.

The default tolerated gap is the larger of 0.6 seconds and 1.5 maximum sampling
intervals, allowing low-FPS footage to function. Explicit `--gap-seconds` overrides
that choice. Tune these demo thresholds against your own labeled recordings.

For example:

```bash
python run_video.py --input input.mp4 --classes-file highlight_classes.json --enter-threshold 0.45 --exit-threshold 0.25 --margin 0.08 --min-observations 3 --min-duration-seconds 0.4 --gap-seconds 0.8
```

To retain whole-video streaming history instead:

```bash
python run_video.py --input input.mp4 --state-window-seconds 0
```

`python run_video.py --help` lists every option, including sampling, warmup,
thread count, FPS override, and output controls.

## 5. What this demo detects

This is pretrained **single-label video action recognition** with heuristic
temporal event extraction. It provides useful action/highlight candidates, not
person boxes or trained action onset/offset localization. Although several
classes can have scores, this model and event extractor are not a multi-person,
multi-label action detector.

Kinetics-600 includes `hugging (not baby)`, `hugging baby`, `kissing`, `clapping`
and many other useful labels. It has no universal **no action/background** class.
Low scores, excluded labels and absence of an emitted candidate mean
**uncertain/no selected candidate**, not proof that the video contains no action.
Softmax confidence is not calibrated actionness. Evaluate background footage as
well as positive examples before using events as automatic capture decisions.

For MomentPilot, use the emitted intervals and their context/scores as additional
evidence alongside existing scene, audio and image-quality rules. The standalone
demo does not alter the existing project or implement its abnormal-quality
module. Occlusion, motion blur, camera shake, blocking and low-quality footage
can reduce recognition reliability.

OpenCV timestamps assume constant frame rate. For phone footage with variable
frame rate or editing requirements needing exact presentation timestamps,
transcode first, for example:

```bash
ffmpeg -i phone.mov -vf fps=30 -c:v libx264 -crf 18 -c:a aac input_cfr.mp4
```

Use `--fps` only when the real CFR is known and metadata is wrong; it changes the
timestamp interpretation and is not a VFR repair. Frame decode failures and
inconsistent reported counts are surfaced in errors or metadata warnings.

## 6. Reuse from Python

```python
import cv2
from assets import load_labels
from model import MoViNetA2Stream
from videoio import preprocess_bgr

labels = load_labels()
recognizer = MoViNetA2Stream(device="auto")
capture = cv2.VideoCapture("input.mp4")
recognizer.reset()
try:
    # This minimal example feeds every frame. The CLI additionally samples at 5 fps.
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        probabilities = recognizer.predict(preprocess_bgr(frame))
        index = int(probabilities.argmax())
        print(labels[index], float(probabilities[index]))
finally:
    capture.release()
    recognizer.reset()
```

One model instance owns mutable state and must process videos serially. For
concurrent videos, use separate instances. Call `reset()` before changing videos;
never reuse one video's buffers for another.

## 7. Tests and provenance

```bash
python -m unittest discover -s tests -v
```

Tests cover sampling/timestamps, color/range preprocessing, conservative event
logic, state resets, checkpoint inference and video output. Real-model tests
require the bundled weights and runtime dependencies. See `VALIDATION.md` for the
checks performed on this delivery and their limitations.

The native PyTorch architecture and converted checkpoint come from the
[Atze00/MoViNet-pytorch project](https://github.com/Atze00/MoViNet-pytorch), pinned to
commit `c2d1edf48fc6c5259707f9d833f22171b4f63493`. This is an **unofficial PyTorch
port** of Google's model. Source, weights and labels are pinned and verified;
see `THIRD_PARTY_NOTICES.md` and `vendor/MOVINET_LICENSE.txt`.

Primary references: [official MoViNet repository](https://github.com/tensorflow/models/tree/master/official/projects/movinet),
[official streaming tutorial](https://www.tensorflow.org/hub/tutorials/movinet),
[PyTorch port's pretrained/parity tests](https://github.com/Atze00/MoViNet-pytorch/blob/c2d1edf48fc6c5259707f9d833f22171b4f63493/tests/test_pretrained_models.py).
