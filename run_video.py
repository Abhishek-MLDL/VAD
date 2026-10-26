"""Classify local videos with native PyTorch MoViNet-A2-Stream."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
import uuid

import cv2
import numpy as np

from assets import DEFAULT_LABELS, DEFAULT_WEIGHTS, load_labels
from model import MoViNetA2Stream
from temporal import TemporalDetector, TriggerConfig
from videoio import (FrameSampler, discover_videos, draw_overlay, mux_audio,
                     output_slug, preprocess_bgr)

EVENT_COLUMNS = [
    "event_id", "label", "class_id", "start_seconds", "end_seconds",
    "confirmed_at_seconds", "peak_probability", "mean_probability",
    "supported_observations", "context_start_seconds", "context_end_seconds",
    "window_id", "closed_by",
]


def write_json(path: Path, content: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def resolve_classes(labels: list[str], classes: str = "all", classes_file: Path | None = None) -> set[int] | None:
    if classes_file is not None:
        names = json.loads(classes_file.read_text(encoding="utf-8"))
        if not isinstance(names, list) or any(not isinstance(item, str) for item in names):
            raise ValueError("Classes file must be a JSON array of exact label strings.")
    elif classes.strip().lower() == "all":
        return None
    else:
        names = [name.strip() for name in classes.split(",") if name.strip()]
    unknown = sorted(set(names) - set(labels))
    if unknown:
        raise ValueError(f"Unknown Kinetics-600 labels: {unknown}. See assets/kinetics_600_labels.txt")
    return {labels.index(name) for name in names}


def process_video(source: Path, destination: Path, model: MoViNetA2Stream,
                  labels: list[str], allowed_ids: set[int] | None, args: argparse.Namespace) -> dict:
    """Process one video serially; also usable with a predictor implementing reset/predict/metadata."""
    destination.mkdir(parents=True, exist_ok=False)
    capture = cv2.VideoCapture(str(source))
    writer = None
    started = time.perf_counter()
    events: list[dict] = []
    warnings: list[str] = []
    processed = samples = resets = 0
    stopped_at_limit = False
    window_id = -1
    context_start = 0.0
    last_observation: dict | None = None
    silent_path = destination / "annotated.mp4"
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {source}")
        reported_fps = float(capture.get(cv2.CAP_PROP_FPS))
        fps = args.fps if args.fps is not None else reported_fps
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Video has missing/invalid FPS. Supply its real constant frame rate with --fps.")
        raw_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        expected_frames = int(raw_count) if math.isfinite(raw_count) and raw_count > 0 else None
        sampler = FrameSampler(fps, args.sample_fps)
        # At low source frame rates, allow enough time between adjacent observations.
        max_sample_interval = math.ceil(fps / sampler.effective_fps - 1e-9) / fps
        gap = args.gap_seconds if args.gap_seconds is not None else max(0.6, 1.5 * max_sample_interval)
        if gap + 1e-9 < max_sample_interval:
            warnings.append("Configured gap is shorter than a sample interval; events may never confirm.")
        trigger_config = TriggerConfig(
            enter_threshold=args.enter_threshold, exit_threshold=args.exit_threshold,
            margin=args.margin, min_observations=args.min_observations,
            min_duration_seconds=args.min_duration_seconds, gap_seconds=gap,
        )
        detector = TemporalDetector(labels, allowed_ids, trigger_config)
        dimensions = None
        model.reset()  # Always reset even after a failed previous video.
        predictions_path = destination / "predictions.jsonl"
        csv_path = destination / "predictions.csv"
        with predictions_path.open("w", encoding="utf-8") as predictions, csv_path.open("w", newline="", encoding="utf-8") as csv_stream:
            columns = ["frame_index", "timestamp_seconds", "window_id", "context_start_seconds",
                       "eligible", "top1_label", "top1_probability", "top1_top2_margin", "active_candidate"]
            csv_writer = csv.DictWriter(csv_stream, fieldnames=columns)
            csv_writer.writeheader()
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                timestamp = processed / fps
                if args.max_seconds is not None and timestamp >= args.max_seconds:
                    stopped_at_limit = True
                    break
                current_dimensions = (frame.shape[1], frame.shape[0])
                if dimensions is None:
                    dimensions = current_dimensions
                    if not args.no_video:
                        # MP4/MPEG4 encoding generally expects even dimensions.
                        output_size = (dimensions[0] + dimensions[0] % 2,
                                       dimensions[1] + dimensions[1] % 2)
                        writer = cv2.VideoWriter(str(silent_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, output_size)
                        if not writer.isOpened():
                            raise RuntimeError("Cannot create annotated MP4. Try --no-video or install an OpenCV build with MP4 support.")
                elif current_dimensions != dimensions:
                    raise RuntimeError("Video resolution changed midstream; split or transcode the input first.")
                if sampler.take(processed):
                    if window_id < 0 or (args.state_window_seconds > 0
                            and timestamp - context_start >= args.state_window_seconds - 1e-9):
                        events.extend(detector.finish("window_reset"))
                        model.reset()
                        context_start = timestamp
                        window_id += 1
                        resets += 1
                    probabilities = model.predict(preprocess_bgr(frame))
                    eligible = timestamp - context_start + 1e-9 >= args.warmup_seconds
                    events.extend(detector.update(timestamp, probabilities, context_start, window_id, eligible))
                    ranking = np.argsort(-probabilities, kind="stable")[:args.top_k]
                    first, second = np.argsort(-probabilities, kind="stable")[:2]
                    row = {
                        "frame_index": processed, "timestamp_seconds": timestamp,
                        "window_id": window_id, "context_start_seconds": context_start,
                        "eligible": bool(eligible), "top1_label": labels[int(first)],
                        "top1_probability": float(probabilities[first]),
                        "top1_top2_margin": float(probabilities[first] - probabilities[second]),
                        "active_candidate": detector.active_label,
                    }
                    csv_writer.writerow(row)
                    record = {
                        **row,
                        "top_k": [{"class_id": int(i), "label": labels[int(i)],
                                   "probability": float(probabilities[i])} for i in ranking],
                        "selected_class_probabilities": ({labels[i]: float(probabilities[i]) for i in sorted(allowed_ids)}
                                                         if allowed_ids is not None else None),
                    }
                    if args.all_scores:
                        record["probabilities"] = probabilities.tolist()
                    predictions.write(json.dumps(record, allow_nan=False) + "\n")
                    last_observation = record
                    samples += 1
                    if samples == 1 or samples % max(1, round(sampler.effective_fps * 10)) == 0:
                        print(f"  {source.name}: {timestamp:.1f}s | {row['top1_label']} {row['top1_probability']:.3f}", flush=True)
                if writer is not None:
                    latest = last_observation
                    lines = [f"MoViNet-A2-Stream | t={timestamp:.2f}s | window={window_id}"]
                    if latest is not None:
                        status = ("WARMUP" if not latest["eligible"] else
                                  f"CANDIDATE: {latest['active_candidate']}" if latest["active_candidate"] else
                                  "UNCERTAIN / no selected candidate")
                        lines.append(status)
                        lines.extend(f"{item['label']}: {item['probability']:.3f}" for item in latest["top_k"][:3])
                    annotated = draw_overlay(frame, lines, active=bool(latest and latest["active_candidate"]))
                    if annotated.shape[1::-1] != output_size:
                        annotated = cv2.copyMakeBorder(annotated, 0, dimensions[1] % 2,
                                                      0, dimensions[0] % 2, cv2.BORDER_REPLICATE)
                    writer.write(annotated)
                processed += 1
            events.extend(detector.finish("processing_limit" if stopped_at_limit else "end_of_video"))
        if processed == 0:
            raise RuntimeError("No frames decoded from the input video.")
        if expected_frames is not None and not stopped_at_limit and processed < expected_frames - max(2, expected_frames * .01):
            warnings.append(f"Decoded {processed} of {expected_frames} reported frames; the file may be truncated or metadata inaccurate.")
        if writer is not None:
            writer.release()
            writer = None
        for index, event in enumerate(events, 1):
            event["event_id"] = f"event_{index:05d}"
        write_json(destination / "events.json", events)
        with (destination / "events.csv").open("w", newline="", encoding="utf-8") as stream:
            event_writer = csv.DictWriter(stream, fieldnames=EVENT_COLUMNS)
            event_writer.writeheader()
            event_writer.writerows(events)
        audio_path = None
        if args.keep_audio:
            audio_path = destination / "annotated_with_audio.mp4"
            mux_audio(silent_path, source, audio_path, args.ffmpeg)
        result = {
            "status": "success", "source": str(source.resolve()),
            "output_directory": str(destination.resolve()),
            "processed_frames": processed, "sampled_frames": samples,
            "duration_seconds": processed / fps, "fps": fps,
            "reported_fps": reported_fps if math.isfinite(reported_fps) else None,
            "fps_overridden": args.fps is not None,
            "reported_frame_count": expected_frames,
            "dimensions": list(dimensions), "sample_fps": sampler.effective_fps,
            "state_windows": resets, "processing_limit_reached": stopped_at_limit,
            "candidate_events": len(events), "warnings": warnings,
            "wall_seconds": time.perf_counter() - started,
            "model": model.metadata(), "opencv_version": cv2.__version__,
            "trigger_config": asdict(trigger_config),
            "selected_classes": "all" if allowed_ids is None else [labels[i] for i in sorted(allowed_ids)],
            "state_window_seconds": args.state_window_seconds,
            "warmup_seconds": args.warmup_seconds,
            "timestamp_method": "decoded frame index / source FPS; assumes constant frame rate",
            "event_semantics": "heuristic intervals of sustained classifier evidence, not trained action boundaries",
            "annotated_video": str(silent_path.resolve()) if not args.no_video else None,
            "annotated_video_with_audio": str(audio_path.resolve()) if audio_path else None,
        }
        write_json(destination / "metadata.json", result)
        return result
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        model.reset()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="Video file or directory (recursive)")
    parser.add_argument("--output", type=Path, default=Path("results"), help="Parent directory; each run gets a new subdirectory")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--threads", type=int, default=4, help="PyTorch CPU threads; 0 leaves its setting unchanged")
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument("--fps", type=float, help="Override source FPS; use only for known CFR footage")
    parser.add_argument("--state-window-seconds", type=float, default=10.0, help="Reset all streaming buffers after this duration; 0 retains whole-video history")
    parser.add_argument("--warmup-seconds", type=float, default=1.0, help="Log predictions but suppress candidates after each reset")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--classes", default="all", help="all or comma-separated exact labels")
    group.add_argument("--classes-file", type=Path, help="JSON array of selected labels; [] logs predictions without events")
    parser.add_argument("--enter-threshold", type=float, default=.5)
    parser.add_argument("--exit-threshold", type=float, default=.3)
    parser.add_argument("--margin", type=float, default=.1)
    parser.add_argument("--min-observations", type=int, default=3)
    parser.add_argument("--min-duration-seconds", type=float, default=.4)
    parser.add_argument("--gap-seconds", type=float, help="Default: max(0.6, 1.5 x maximum sample interval)")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--all-scores", action="store_true", help="Include all 600 scores in each JSONL observation")
    parser.add_argument("--no-video", action="store_true", help="Only write predictions, events, and metadata")
    parser.add_argument("--keep-audio", action="store_true", help="Also mux source audio into a second video; requires FFmpeg")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="FFmpeg executable name or path")
    parser.add_argument("--max-seconds", type=float, help="Process only the first N seconds for a quick test")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    for name in ("sample_fps", "fps", "max_seconds"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError(f"{name.replace('_', '-')} must be finite and positive")
    for name in ("state_window_seconds", "warmup_seconds"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name.replace('_', '-')} must be finite and nonnegative")
    if args.state_window_seconds and args.warmup_seconds >= args.state_window_seconds:
        raise ValueError("Warmup must be shorter than the state window.")
    if args.threads < 0 or not 1 <= args.top_k <= 600:
        raise ValueError("threads must be >=0 and top-k must be in [1,600]")
    if args.keep_audio and args.no_video:
        raise ValueError("--keep-audio requires annotated video output; remove --no-video")
    TriggerConfig(args.enter_threshold, args.exit_threshold, args.margin,
                  args.min_observations, args.min_duration_seconds,
                  .6 if args.gap_seconds is None else args.gap_seconds)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_args(args)
        args.input = args.input.expanduser().resolve()
        labels = load_labels(args.labels)
        allowed_ids = resolve_classes(labels, args.classes, args.classes_file)
        sources = discover_videos(args.input)
        output_root = args.output.expanduser().resolve()
        if args.input.is_dir():
            sources = [source for source in sources if output_root != source.parent and output_root not in source.parents]
        if not sources:
            raise ValueError("No input videos remain after excluding the output directory. Choose a separate --output directory.")
        model = MoViNetA2Stream(args.weights, args.device, args.threads)
        run_name = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
        run_directory = output_root / run_name
        run_directory.mkdir(parents=True, exist_ok=False)
        print(f"Device: {model.device}. Results: {run_directory}", flush=True)
        summary: list[dict] = []
        for source in sources:
            destination = run_directory / output_slug(source)
            try:
                result = process_video(source, destination, model, labels, allowed_ids, args)
                print(f"  Completed: {result['candidate_events']} candidates -> {destination}", flush=True)
            except Exception as error:
                result = {"status": "failed", "source": str(source), "error": str(error),
                          "output_directory": str(destination)}
                if destination.is_dir():
                    write_json(destination / "error.json", result)
                print(f"  Failed {source.name}: {error}", file=sys.stderr, flush=True)
            summary.append(result)
            write_json(run_directory / "batch_summary.json", summary)
        return int(any(item["status"] != "success" for item in summary))
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
