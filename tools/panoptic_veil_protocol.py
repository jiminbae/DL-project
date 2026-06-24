#!/usr/bin/env python3
"""Materialize a small CMU Panoptic Studio based VEIL evaluation set.

The tool is intentionally lightweight: it can pull only the requested clip
window from Panoptic's public video URLs with FFmpeg, then crop a protected
target image from a candidate face box. It does not mirror full source videos.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_ENDPOINT = "http://domedb.perception.cs.cmu.edu"
DEFAULT_SEQUENCES = (
    "170221_haggling_b1",
    "170221_haggling_b2",
    "170221_haggling_b3",
    "170224_haggling_b1",
    "170224_haggling_b2",
    "170224_haggling_b3",
)
DEFAULT_CAMERAS = ("hd_00_08", "hd_00_27", "hd_00_00", "hd_00_25")
DEFAULT_TIMES = (60.0, 180.0, 240.0)


class ProtocolError(RuntimeError):
    """Raised for invalid Panoptic protocol input or failed materialization."""


def _write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    path = path.expanduser().resolve()
    if not path.exists():
        raise ProtocolError(f"Candidate file does not exist: {path}")
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ProtocolError(f"{path}:{line_number}: invalid JSON") from exc
    if not records:
        raise ProtocolError(f"Candidate file is empty: {path}")
    return records


def parse_csv_values(value: str | None, defaults: Sequence[str]) -> list[str]:
    if value is None:
        return list(defaults)
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated value")
    return items


def normalize_camera(camera: str) -> str:
    camera = str(camera).strip()
    if camera.startswith("hd_"):
        return camera
    if camera.isdigit():
        return f"hd_00_{int(camera):02d}"
    raise ProtocolError(f"Invalid Panoptic HD camera id: {camera}")


def build_panoptic_video_url(endpoint: str, sequence: str, camera: str) -> str:
    endpoint = endpoint.rstrip("/")
    camera = normalize_camera(camera)
    return f"{endpoint}/webdata/dataset/{sequence}/videos/hd_shared_crf20/{camera}.mp4"


def make_clip_id(record: dict) -> str:
    if record.get("clip_id"):
        return str(record["clip_id"])
    sequence = required_str(record, "sequence")
    camera = normalize_camera(required_str(record, "camera"))
    start = required_float(record, "start_seconds", aliases=("start_sec",))
    return f"{sequence}_{camera}_{int(round(start * 10000)):010d}"


def required_str(record: dict, key: str) -> str:
    value = record.get(key)
    if value is None or str(value).strip() == "":
        raise ProtocolError(f"missing_candidate_field: {key}")
    return str(value)


def required_float(record: dict, key: str, aliases: Sequence[str] = ()) -> float:
    raw_value = record.get(key)
    if raw_value is None:
        for alias in aliases:
            raw_value = record.get(alias)
            if raw_value is not None:
                break
    if raw_value is None:
        raise ProtocolError(f"missing_candidate_field: {key}")
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"invalid_candidate_field: {key}={raw_value!r}") from exc
    if not math.isfinite(value):
        raise ProtocolError(f"invalid_candidate_field: {key}={raw_value!r}")
    return value


def _ffprobe_for_ffmpeg(ffmpeg_path: str) -> str:
    sibling = Path(ffmpeg_path).with_name("ffprobe")
    if sibling.exists():
        return str(sibling)
    ffprobe_path = shutil.which("ffprobe")
    if ffprobe_path is None:
        raise ProtocolError("ffprobe executable not found")
    return ffprobe_path


def _probe_video_dimensions(video: str, ffprobe_path: str) -> tuple[int, int]:
    command = [
        ffprobe_path,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "json",
        video,
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise ProtocolError(f"ffprobe_dimensions_failed: {completed.stderr[-1000:].strip()}")
    try:
        payload = json.loads(completed.stdout)
        stream = payload["streams"][0]
        width = int(stream["width"])
        height = int(stream["height"])
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"ffprobe did not report video dimensions for {video}") from exc
    if width <= 0 or height <= 0:
        raise ProtocolError(f"invalid_video_dimensions: {width}x{height}")
    return width, height


def _validate_box_values(box: Sequence[float]) -> tuple[float, float, float, float]:
    if len(box) != 4:
        raise ProtocolError(f"invalid_reference_box: expected 4 values, got {len(box)}")
    try:
        x1, y1, x2, y2 = (float(value) for value in box)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"invalid_reference_box: non-numeric value in {list(box)}") from exc
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
        raise ProtocolError(f"invalid_reference_box: non-finite value in {list(box)}")
    if x2 <= x1 or y2 <= y1:
        raise ProtocolError(f"invalid_reference_box: expected x2>x1 and y2>y1, got {list(box)}")
    return x1, y1, x2, y2


def expanded_pixel_box(
    box: Sequence[float],
    width: int,
    height: int,
    padding: float,
    *,
    normalized: bool | None = None,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = _validate_box_values(box)
    if normalized is None:
        normalized = all(0.0 <= value <= 1.0 for value in (x1, y1, x2, y2))
    if normalized:
        if not all(0.0 <= value <= 1.0 for value in (x1, y1, x2, y2)):
            raise ProtocolError(f"invalid_reference_box: normalized values outside [0, 1]: {list(box)}")
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height

    box_w = x2 - x1
    box_h = y2 - y1
    x1 -= box_w * padding
    x2 += box_w * padding
    y1 -= box_h * padding
    y2 += box_h * padding
    px1 = max(0, min(width - 1, int(math.floor(x1))))
    py1 = max(0, min(height - 1, int(math.floor(y1))))
    px2 = max(1, min(width, int(math.ceil(x2))))
    py2 = max(1, min(height, int(math.ceil(y2))))
    if px2 <= px1 or py2 <= py1:
        raise ProtocolError(
            f"empty_crop: box={list(box)}, image_size={width}x{height}, crop=({px1},{py1},{px2},{py2})"
        )
    return px1, py1, px2, py2


def select_reference_box(record: dict) -> tuple[list[float], str]:
    if "reference_box" in record:
        return list(record["reference_box"]), "reference_box"

    faces = record.get("reference_faces")
    if not isinstance(faces, list) or not faces:
        raise ProtocolError("missing_reference_box: provide reference_box or reference_faces")

    if record.get("target_face_index") is not None:
        try:
            index = int(record["target_face_index"])
        except (TypeError, ValueError) as exc:
            raise ProtocolError(f"invalid_target_face_index: {record['target_face_index']!r}") from exc
        if index < 0 or index >= len(faces):
            raise ProtocolError(f"invalid_target_face_index: {index} outside 0..{len(faces) - 1}")
        face = faces[index]
        if "bbox" not in face:
            raise ProtocolError(f"missing_reference_box: reference_faces[{index}].bbox")
        return list(face["bbox"]), f"reference_faces[{index}].bbox"

    def face_score(face: dict) -> tuple[float, float]:
        bbox = face.get("bbox", [0, 0, 0, 0])
        try:
            x1, y1, x2, y2 = _validate_box_values(bbox)
        except ProtocolError:
            return (-1.0, -1.0)
        return (float(face.get("min_side", min(x2 - x1, y2 - y1))), (x2 - x1) * (y2 - y1))

    index, face = max(enumerate(faces), key=lambda item: face_score(item[1]))
    if "bbox" not in face:
        raise ProtocolError(f"missing_reference_box: reference_faces[{index}].bbox")
    return list(face["bbox"]), f"reference_faces[{index}].bbox"


def _run_ffmpeg(command: list[str], failure_prefix: str, timeout: float | None = None) -> None:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise ProtocolError(f"{failure_prefix}_timeout: exceeded {timeout:.1f}s") from exc
    if completed.returncode != 0:
        raise ProtocolError(f"{failure_prefix}: {completed.stderr[-2000:].strip()}")


def extract_clip(
    source_video: str,
    start: float,
    duration: float,
    output_path: Path,
    ffmpeg_path: str,
    overwrite: bool,
    timeout: float | None = None,
) -> None:
    if start < 0 or duration <= 0:
        raise ProtocolError(f"invalid_time_window: start={start}, duration={duration}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-ss",
        f"{start:.3f}",
        "-i",
        source_video,
        "-t",
        f"{duration:.3f}",
        "-map",
        "0:v:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    _run_ffmpeg(command, "ffmpeg_clip_extract_failed", timeout=timeout)
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise ProtocolError(f"clip_write_failed: {output_path}")


def extract_frame(
    source_video: str,
    timestamp: float,
    output_path: Path,
    ffmpeg_path: str,
    overwrite: bool,
    timeout: float | None = None,
) -> None:
    if timestamp < 0:
        raise ProtocolError(f"invalid_reference_timestamp: {timestamp}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-ss",
        f"{timestamp:.3f}",
        "-i",
        source_video,
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(output_path),
    ]
    _run_ffmpeg(command, "ffmpeg_reference_frame_failed", timeout=timeout)
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise ProtocolError(f"reference_frame_write_failed: {output_path}")


def extract_target_crop(
    source_video: str,
    timestamp: float,
    box: Sequence[float],
    output_path: Path,
    padding: float,
    ffmpeg_path: str,
    ffprobe_path: str,
    *,
    normalized: bool | None,
    overwrite: bool,
    timeout: float | None = None,
) -> tuple[int, int, int, int]:
    if timestamp < 0:
        raise ProtocolError(f"invalid_reference_timestamp: {timestamp}")
    width, height = _probe_video_dimensions(source_video, ffprobe_path)
    x1, y1, x2, y2 = expanded_pixel_box(box, width, height, padding, normalized=normalized)
    crop_w = x2 - x1
    crop_h = y2 - y1
    if min(crop_w, crop_h) < 8:
        raise ProtocolError(f"crop_too_small: crop_size={crop_w}x{crop_h}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-ss",
        f"{timestamp:.3f}",
        "-i",
        source_video,
        "-frames:v",
        "1",
        "-vf",
        f"crop={crop_w}:{crop_h}:{x1}:{y1}",
        "-q:v",
        "2",
        str(output_path),
    ]
    _run_ffmpeg(command, "ffmpeg_target_crop_failed", timeout=timeout)
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise ProtocolError(f"target_write_failed: {output_path}")
    return x1, y1, x2, y2


def source_video_for_record(record: dict, endpoint: str) -> str:
    if record.get("source_video_url"):
        return str(record["source_video_url"])
    sequence = required_str(record, "sequence")
    camera = required_str(record, "camera")
    return build_panoptic_video_url(endpoint, sequence, camera)


def materialize(
    candidates_path: Path,
    output_dir: Path,
    *,
    endpoint: str = DEFAULT_ENDPOINT,
    ffmpeg_binary: str = "ffmpeg",
    overwrite: bool = False,
    check_only: bool = False,
    default_split: str = "pilot",
    default_target_padding: float = 0.5,
) -> dict:
    records = read_jsonl(candidates_path)
    output_dir = output_dir.expanduser().resolve()

    ffmpeg_path = shutil.which(ffmpeg_binary)
    if ffmpeg_path is None and not check_only:
        raise ProtocolError(f"ffmpeg executable not found: {ffmpeg_binary}")
    ffprobe_path = _ffprobe_for_ffmpeg(ffmpeg_path) if ffmpeg_path and not check_only else None

    results: list[dict] = []
    counters = Counter()
    for record in records:
        result = dict(record)
        try:
            clip_id = make_clip_id(record)
            split = str(record.get("split") or default_split)
            source_video = source_video_for_record(record, endpoint)
            start = required_float(record, "start_seconds", aliases=("start_sec",))
            duration = required_float(record, "duration_seconds", aliases=("duration_sec",))
            reference_timestamp = required_float(
                record,
                "reference_timestamp_seconds",
                aliases=("reference_timestamp", "reference_sec"),
            )
            target_padding = float(record.get("target_padding", default_target_padding))
            if target_padding < 0:
                raise ProtocolError(f"invalid_target_padding: {target_padding}")
            box, box_source = select_reference_box(record)
            normalized = record.get("reference_box_format") == "normalized"
            if record.get("reference_box_format") == "pixel":
                normalized = False

            clip_path = output_dir / "clips" / split / f"{clip_id}.mp4"
            target_path = output_dir / "targets" / split / f"{clip_id}_target.jpg"
            frame_path = output_dir / "frames" / f"{clip_id}_ref.jpg"
            result.update(
                {
                    "clip_id": clip_id,
                    "split": split,
                    "source_video_url": source_video,
                    "clip_path": str(clip_path),
                    "target_image_path": str(target_path),
                    "reference_frame_path": str(frame_path),
                    "reference_box_source": box_source,
                    "target_padding": target_padding,
                }
            )

            if check_only:
                result["status"] = "ready"
                counters["ready"] += 1
                results.append(result)
                continue

            assert ffmpeg_path is not None
            assert ffprobe_path is not None
            if overwrite or not clip_path.exists():
                extract_clip(source_video, start, duration, clip_path, ffmpeg_path, overwrite, timeout=120.0)
            if overwrite or not frame_path.exists():
                extract_frame(source_video, reference_timestamp, frame_path, ffmpeg_path, overwrite, timeout=60.0)
            if overwrite or not target_path.exists():
                crop = extract_target_crop(
                    source_video,
                    reference_timestamp,
                    box,
                    target_path,
                    target_padding,
                    ffmpeg_path,
                    ffprobe_path,
                    normalized=normalized if "reference_box_format" in record else None,
                    overwrite=overwrite,
                    timeout=60.0,
                )
                result["target_crop_pixel_box"] = list(crop)

            result["status"] = "materialized"
            counters["materialized"] += 1
        except ProtocolError as exc:
            status = "materialize_failed"
            message = str(exc)
            if message.startswith("missing_candidate_field"):
                status = "invalid_candidate"
            elif "reference_box" in message or "target_face_index" in message:
                status = "target_crop_failed"
            elif message.startswith("ffmpeg_clip") or message.startswith("clip_write"):
                status = "clip_extract_failed"
            elif message.startswith("ffmpeg_reference_frame") or message.startswith("reference_frame_write"):
                status = "reference_frame_failed"
            elif message.startswith("ffmpeg_target_crop") or message.startswith("target_write") or message.startswith("crop_"):
                status = "target_crop_failed"
            result.update({"status": status, "failure_reason": message})
            counters[status] += 1
        results.append(result)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "materialized_manifest.jsonl", results)
    failures = [
        {
            "clip_id": result.get("clip_id"),
            "sequence": result.get("sequence"),
            "camera": result.get("camera"),
            "status": result.get("status"),
            "reason": result.get("failure_reason"),
        }
        for result in results
        if result.get("status") not in {"materialized", "ready"}
    ]
    summary = {
        "protocol": "Panoptic-VEIL Selective Anonymization Protocol",
        "candidates": str(candidates_path.expanduser().resolve()),
        "output_dir": str(output_dir),
        "endpoint": endpoint,
        "check_only": check_only,
        "counts": dict(sorted(counters.items())),
        "failures": failures,
    }
    with (output_dir / "materialize_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, sort_keys=True)
    return summary


def detect_faces_in_image(image_path: Path, model_path: Path, confidence: float) -> list[dict]:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ProtocolError("ultralytics is required for scan; install the VEIL model requirements") from exc

    model = YOLO(str(model_path))
    detections = model.predict(str(image_path), conf=confidence, verbose=False)
    faces: list[dict] = []
    for result in detections:
        if result.boxes is None:
            continue
        for box in result.boxes:
            x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
            conf = float(box.conf[0]) if box.conf is not None else 0.0
            min_side = min(x2 - x1, y2 - y1)
            faces.append(
                {
                    "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    "conf": round(conf, 4),
                    "min_side": round(min_side, 1),
                    "area": round((x2 - x1) * (y2 - y1), 1),
                    "swap_size_ok": min_side >= 110.0,
                }
            )
    faces.sort(key=lambda face: (-face["area"], -face["conf"]))
    return faces


def scan(args: argparse.Namespace) -> dict:
    ffmpeg_path = shutil.which(args.ffmpeg)
    if ffmpeg_path is None:
        raise ProtocolError(f"ffmpeg executable not found: {args.ffmpeg}")
    sequences = parse_csv_values(args.sequences, DEFAULT_SEQUENCES)
    cameras = [normalize_camera(item) for item in parse_csv_values(args.cameras, DEFAULT_CAMERAS)]
    times = [float(item) for item in parse_csv_values(args.times, [str(value) for value in DEFAULT_TIMES])]
    output_dir = args.output_dir.expanduser().resolve()
    frame_dir = output_dir / "frames"

    records: list[dict] = []
    counters = Counter()
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        for sequence in sequences:
            for camera in cameras:
                source_video = build_panoptic_video_url(args.endpoint, sequence, camera)
                for timestamp in times:
                    record = {
                        "sequence": sequence,
                        "camera": camera,
                        "timestamp_seconds": timestamp,
                        "source_video_url": source_video,
                    }
                    try:
                        frame_path = temp_root / f"{sequence}_{camera}_{int(timestamp)}.jpg"
                        extract_frame(source_video, timestamp, frame_path, ffmpeg_path, overwrite=True, timeout=args.ffmpeg_timeout)
                        faces = detect_faces_in_image(frame_path, args.model_path, args.confidence)
                        keep_path = frame_dir / frame_path.name
                        keep_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(frame_path, keep_path)
                        record.update(
                            {
                                "status": "scanned",
                                "frame_path": str(keep_path),
                                "face_count": len(faces),
                                "swap_size_count": sum(1 for face in faces if face["swap_size_ok"]),
                                "reference_faces": faces,
                            }
                        )
                        counters["scanned"] += 1
                    except ProtocolError as exc:
                        record.update({"status": "scan_failed", "failure_reason": str(exc)})
                        counters["scan_failed"] += 1
                    records.append(record)

    _write_jsonl(output_dir / "scan_results.jsonl", records)
    candidates = [
        record
        for record in records
        if record.get("status") == "scanned" and int(record.get("swap_size_count", 0)) >= args.min_swap_size_faces
    ]
    _write_jsonl(output_dir / "candidate_frames.jsonl", candidates)
    summary = {
        "output_dir": str(output_dir),
        "endpoint": args.endpoint,
        "counts": dict(sorted(counters.items())),
        "candidate_frames": len(candidates),
        "parameters": {
            "sequences": sequences,
            "cameras": cameras,
            "times": times,
            "confidence": args.confidence,
            "min_swap_size_faces": args.min_swap_size_faces,
        },
    }
    with (output_dir / "scan_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, sort_keys=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create and materialize a Panoptic-based VEIL protocol.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan_parser = subparsers.add_parser("scan", help="Scan Panoptic remote frames for promising multi-face candidates.")
    scan_parser.add_argument("--output-dir", type=Path, required=True)
    scan_parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    scan_parser.add_argument("--sequences")
    scan_parser.add_argument("--cameras")
    scan_parser.add_argument("--times")
    scan_parser.add_argument("--model-path", type=Path, default=Path("models/veil/weights/yolo26x-face.pt"))
    scan_parser.add_argument("--confidence", type=float, default=0.45)
    scan_parser.add_argument("--min-swap-size-faces", type=int, default=2)
    scan_parser.add_argument("--ffmpeg", default="ffmpeg")
    scan_parser.add_argument("--ffmpeg-timeout", type=float, default=45.0)

    materialize_parser = subparsers.add_parser("materialize", help="Extract Panoptic clips and protected target crops.")
    materialize_parser.add_argument("--candidates", type=Path, required=True)
    materialize_parser.add_argument("--output-dir", type=Path, required=True)
    materialize_parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    materialize_parser.add_argument("--ffmpeg", default="ffmpeg")
    materialize_parser.add_argument("--overwrite", action="store_true")
    materialize_parser.add_argument("--check-only", action="store_true")
    materialize_parser.add_argument("--default-split", default="pilot")
    materialize_parser.add_argument("--default-target-padding", type=float, default=0.5)
    return parser


def run_materialize(args: argparse.Namespace) -> int:
    if args.default_target_padding < 0:
        raise ProtocolError("default-target-padding must be non-negative")
    summary = materialize(
        args.candidates,
        args.output_dir,
        endpoint=args.endpoint,
        ffmpeg_binary=args.ffmpeg,
        overwrite=args.overwrite,
        check_only=args.check_only,
        default_split=args.default_split,
        default_target_padding=args.default_target_padding,
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "scan":
            print(json.dumps(scan(args), ensure_ascii=False))
            return 0
        if args.command == "materialize":
            return run_materialize(args)
        parser.error(f"Unknown command: {args.command}")
    except ProtocolError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
