#!/usr/bin/env python3
"""Build a reproducible selective-anonymization protocol from AVA ActiveSpeaker.

The tool deliberately does not download source videos. It consumes the public AVA
ActiveSpeaker annotations and optionally materializes clips from videos that the
user has already obtained and is authorized to process.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import io
import json
import math
import shutil
import statistics
import subprocess
import sys
import tarfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence, TextIO


CSV_COLUMNS = (
    "video_id",
    "frame_timestamp",
    "x1",
    "y1",
    "x2",
    "y2",
    "label",
    "entity_id",
)
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".mov", ".avi")


@dataclass(frozen=True)
class FaceAnnotation:
    video_id: str
    timestamp: float
    x1: float
    y1: float
    x2: float
    y2: float
    label: str
    entity_id: str
    source_file: str

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def min_side(self) -> float:
        return min(self.width, self.height)

    @property
    def box(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass(frozen=True)
class ClipCandidate:
    clip_id: str
    split: str
    video_id: str
    start_sec: float
    end_sec: float
    duration_sec: float
    protected_entity_id: str
    reference_timestamp: float
    reference_box: list[float]
    entity_ids: list[str]
    num_entities: int
    overlap_ratio: float
    protected_coverage: float
    protected_median_face_area: float
    protected_median_min_side: float
    observed_timestamps: int
    score: float


class ProtocolError(RuntimeError):
    """Raised for invalid input or a failed materialization step."""


def _is_header(row: Sequence[str]) -> bool:
    if not row:
        return False
    first = row[0].strip().lower()
    joined = ",".join(cell.strip().lower() for cell in row)
    return first in {"video_id", "video"} or "frame_timestamp" in joined


def _parse_row(row: Sequence[str], source_file: str, line_number: int) -> FaceAnnotation | None:
    if not row or all(not cell.strip() for cell in row):
        return None
    if len(row) < 8:
        raise ProtocolError(
            f"{source_file}:{line_number}: expected at least 8 CSV columns, got {len(row)}"
        )

    try:
        timestamp = float(row[1])
        x1, y1, x2, y2 = (float(row[i]) for i in range(2, 6))
    except ValueError as exc:
        raise ProtocolError(
            f"{source_file}:{line_number}: invalid numeric value in AVA row"
        ) from exc

    video_id = row[0].strip()
    label = row[6].strip()
    entity_id = row[7].strip()

    if not video_id or not entity_id:
        raise ProtocolError(f"{source_file}:{line_number}: empty video_id or entity_id")
    if x2 <= x1 or y2 <= y1:
        return None
    if not all(-1e-6 <= value <= 1.0 + 1e-6 for value in (x1, y1, x2, y2)):
        raise ProtocolError(
            f"{source_file}:{line_number}: normalized box is outside [0, 1]"
        )

    return FaceAnnotation(
        video_id=video_id,
        timestamp=timestamp,
        x1=max(0.0, min(1.0, x1)),
        y1=max(0.0, min(1.0, y1)),
        x2=max(0.0, min(1.0, x2)),
        y2=max(0.0, min(1.0, y2)),
        label=label,
        entity_id=entity_id,
        source_file=source_file,
    )


def _read_csv_stream(stream: TextIO, source_file: str) -> Iterator[FaceAnnotation]:
    reader = csv.reader(stream)
    for line_number, row in enumerate(reader, start=1):
        if line_number == 1 and _is_header(row):
            continue
        annotation = _parse_row(row, source_file, line_number)
        if annotation is not None:
            yield annotation


def iter_annotations(path: Path) -> Iterator[FaceAnnotation]:
    """Yield annotations from a CSV, directory of CSVs, or tar archive."""
    path = path.expanduser().resolve()
    if not path.exists():
        raise ProtocolError(f"Annotation path does not exist: {path}")

    if path.is_dir():
        csv_paths = sorted(p for p in path.rglob("*.csv") if p.is_file())
        if not csv_paths:
            raise ProtocolError(f"No CSV files found under: {path}")
        for csv_path in csv_paths:
            with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
                yield from _read_csv_stream(stream, str(csv_path))
        return

    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            yield from _read_csv_stream(stream, str(path))
        return

    if tarfile.is_tarfile(path):
        with tarfile.open(path, "r:*") as archive:
            members = sorted(
                (m for m in archive.getmembers() if m.isfile() and m.name.lower().endswith(".csv")),
                key=lambda m: m.name,
            )
            if not members:
                raise ProtocolError(f"No CSV files found in archive: {path}")
            for member in members:
                extracted = archive.extractfile(member)
                if extracted is None:
                    continue
                with io.TextIOWrapper(extracted, encoding="utf-8-sig", newline="") as stream:
                    yield from _read_csv_stream(stream, f"{path}!{member.name}")
        return

    raise ProtocolError(
        "Unsupported annotation input. Use a CSV file, a directory containing CSVs, "
        "or a .tar/.tar.bz2/.tar.gz archive."
    )


def load_annotations(path: Path, limit_videos: int | None = None) -> dict[str, list[FaceAnnotation]]:
    grouped: dict[str, list[FaceAnnotation]] = defaultdict(list)
    allowed: set[str] | None = set() if limit_videos else None

    for annotation in iter_annotations(path):
        if allowed is not None and annotation.video_id not in allowed:
            if len(allowed) >= int(limit_videos):
                continue
            allowed.add(annotation.video_id)
        grouped[annotation.video_id].append(annotation)

    for rows in grouped.values():
        rows.sort(key=lambda item: (item.timestamp, item.entity_id))

    if not grouped:
        raise ProtocolError("No valid AVA ActiveSpeaker rows were loaded")
    return dict(grouped)


def parse_split_ratios(value: str) -> list[tuple[str, int]]:
    """Parse train=80,val=10,test=10 and validate that values sum to 100."""
    result: list[tuple[str, int]] = []
    seen: set[str] = set()
    for item in value.split(","):
        if "=" not in item:
            raise argparse.ArgumentTypeError("Split ratios must look like train=80,val=10,test=10")
        name, raw_ratio = item.split("=", 1)
        name = name.strip()
        if not name or name in seen:
            raise argparse.ArgumentTypeError("Split names must be non-empty and unique")
        try:
            ratio = int(raw_ratio)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("Split ratios must be integers") from exc
        if ratio < 0:
            raise argparse.ArgumentTypeError("Split ratios must be non-negative")
        seen.add(name)
        result.append((name, ratio))
    if sum(ratio for _, ratio in result) != 100:
        raise argparse.ArgumentTypeError("Split ratios must sum to 100")
    return result


def assign_split(video_id: str, seed: int, split_ratios: Sequence[tuple[str, int]]) -> str:
    digest = hashlib.sha1(f"{seed}:{video_id}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    cumulative = 0
    for name, ratio in split_ratios:
        cumulative += ratio
        if bucket < cumulative:
            return name
    return split_ratios[-1][0]


def _timestamp_key(value: float) -> float:
    return round(value, 3)


def _reference_score(row: FaceAnnotation) -> float:
    center_x = (row.x1 + row.x2) * 0.5
    center_y = (row.y1 + row.y2) * 0.5
    center_distance = math.sqrt((center_x - 0.5) ** 2 + (center_y - 0.5) ** 2)
    center_factor = max(0.35, 1.0 - center_distance)
    edge_margin = min(row.x1, row.y1, 1.0 - row.x2, 1.0 - row.y2)
    edge_factor = max(0.35, min(1.0, edge_margin / 0.08))
    return row.area * center_factor * edge_factor


def _window_candidates(
    video_id: str,
    rows: Sequence[FaceAnnotation],
    *,
    window_seconds: float,
    step_seconds: float,
    min_faces: int,
    min_overlap_ratio: float,
    min_entity_coverage: float,
    min_normalized_face_size: float,
    seed: int,
    split_ratios: Sequence[tuple[str, int]],
) -> list[ClipCandidate]:
    timestamps = [row.timestamp for row in rows]
    min_ts = timestamps[0]
    max_ts = timestamps[-1]
    if max_ts - min_ts < window_seconds:
        return []

    start = math.ceil(min_ts / step_seconds) * step_seconds
    final_start = max_ts - window_seconds
    candidates: list[ClipCandidate] = []
    split = assign_split(video_id, seed, split_ratios)

    while start <= final_start + 1e-9:
        end = start + window_seconds
        left = bisect.bisect_left(timestamps, start)
        right = bisect.bisect_left(timestamps, end)
        window_rows = rows[left:right]
        if not window_rows:
            start += step_seconds
            continue

        by_time: dict[float, dict[str, FaceAnnotation]] = defaultdict(dict)
        by_entity: dict[str, list[FaceAnnotation]] = defaultdict(list)
        for row in window_rows:
            key = _timestamp_key(row.timestamp)
            previous = by_time[key].get(row.entity_id)
            if previous is None or row.area > previous.area:
                by_time[key][row.entity_id] = row
            by_entity[row.entity_id].append(row)

        observed_timestamps = len(by_time)
        if observed_timestamps == 0:
            start += step_seconds
            continue

        eligible: dict[str, dict[str, float]] = {}
        for entity_id, entity_rows in by_entity.items():
            entity_timestamps = {_timestamp_key(row.timestamp) for row in entity_rows}
            coverage = len(entity_timestamps) / observed_timestamps
            median_min_side = statistics.median(row.min_side for row in entity_rows)
            median_area = statistics.median(row.area for row in entity_rows)
            if coverage >= min_entity_coverage and median_min_side >= min_normalized_face_size:
                eligible[entity_id] = {
                    "coverage": coverage,
                    "median_min_side": median_min_side,
                    "median_area": median_area,
                }

        if len(eligible) < min_faces:
            start += step_seconds
            continue

        eligible_ids = set(eligible)
        overlap_frames = sum(
            1
            for visible in by_time.values()
            if len(eligible_ids.intersection(visible)) >= min_faces
        )
        overlap_ratio = overlap_frames / observed_timestamps
        if overlap_ratio < min_overlap_ratio:
            start += step_seconds
            continue

        protected_entity_id = max(
            eligible,
            key=lambda entity_id: (
                eligible[entity_id]["median_area"],
                eligible[entity_id]["coverage"],
                entity_id,
            ),
        )
        protected_rows = by_entity[protected_entity_id]
        reference = max(protected_rows, key=_reference_score)
        protected_stats = eligible[protected_entity_id]
        entity_ids = sorted(eligible)
        score = (
            overlap_ratio
            * protected_stats["coverage"]
            * math.sqrt(max(protected_stats["median_area"], 1e-12))
        )
        clip_hash = hashlib.sha1(
            f"{video_id}:{start:.3f}:{end:.3f}:{protected_entity_id}".encode("utf-8")
        ).hexdigest()[:10]
        clip_id = f"{video_id}_{int(round(start * 1000)):010d}_{clip_hash}"

        candidates.append(
            ClipCandidate(
                clip_id=clip_id,
                split=split,
                video_id=video_id,
                start_sec=round(start, 3),
                end_sec=round(end, 3),
                duration_sec=round(window_seconds, 3),
                protected_entity_id=protected_entity_id,
                reference_timestamp=round(reference.timestamp, 3),
                reference_box=[round(value, 6) for value in reference.box],
                entity_ids=entity_ids,
                num_entities=len(entity_ids),
                overlap_ratio=round(overlap_ratio, 6),
                protected_coverage=round(protected_stats["coverage"], 6),
                protected_median_face_area=round(protected_stats["median_area"], 8),
                protected_median_min_side=round(protected_stats["median_min_side"], 8),
                observed_timestamps=observed_timestamps,
                score=round(score, 8),
            )
        )
        start += step_seconds

    return candidates


def _temporal_overlap_ratio(a: ClipCandidate, b: ClipCandidate) -> float:
    intersection = max(0.0, min(a.end_sec, b.end_sec) - max(a.start_sec, b.start_sec))
    shorter = min(a.duration_sec, b.duration_sec)
    return 0.0 if shorter <= 0 else intersection / shorter


def select_candidates(
    grouped: dict[str, list[FaceAnnotation]],
    *,
    window_seconds: float,
    step_seconds: float,
    min_faces: int,
    min_overlap_ratio: float,
    min_entity_coverage: float,
    min_normalized_face_size: float,
    max_clips_per_video: int,
    max_selected_overlap: float,
    seed: int,
    split_ratios: Sequence[tuple[str, int]],
) -> list[ClipCandidate]:
    selected: list[ClipCandidate] = []

    for video_id in sorted(grouped):
        candidates = _window_candidates(
            video_id,
            grouped[video_id],
            window_seconds=window_seconds,
            step_seconds=step_seconds,
            min_faces=min_faces,
            min_overlap_ratio=min_overlap_ratio,
            min_entity_coverage=min_entity_coverage,
            min_normalized_face_size=min_normalized_face_size,
            seed=seed,
            split_ratios=split_ratios,
        )
        candidates.sort(key=lambda item: (-item.score, item.start_sec, item.clip_id))
        accepted: list[ClipCandidate] = []
        for candidate in candidates:
            if any(
                _temporal_overlap_ratio(candidate, existing) > max_selected_overlap
                for existing in accepted
            ):
                continue
            accepted.append(candidate)
            if len(accepted) >= max_clips_per_video:
                break
        selected.extend(sorted(accepted, key=lambda item: item.start_sec))

    selected.sort(key=lambda item: (item.split, item.video_id, item.start_sec))
    return selected


def _write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def write_protocol_outputs(
    output_dir: Path,
    candidates: Sequence[ClipCandidate],
    grouped: dict[str, list[FaceAnnotation]],
    parameters: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_records = [asdict(candidate) for candidate in candidates]
    _write_jsonl(output_dir / "manifest.jsonl", manifest_records)

    if manifest_records:
        with (output_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as stream:
            fieldnames = list(manifest_records[0].keys())
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for record in manifest_records:
                csv_record = dict(record)
                csv_record["reference_box"] = json.dumps(csv_record["reference_box"])
                csv_record["entity_ids"] = json.dumps(csv_record["entity_ids"], ensure_ascii=False)
                writer.writerow(csv_record)

    by_clip = {candidate.clip_id: candidate for candidate in candidates}
    selected_rows: list[dict] = []
    candidates_by_video: dict[str, list[ClipCandidate]] = defaultdict(list)
    for candidate in candidates:
        candidates_by_video[candidate.video_id].append(candidate)

    for video_id, video_candidates in candidates_by_video.items():
        for row in grouped[video_id]:
            for candidate in video_candidates:
                if candidate.start_sec <= row.timestamp < candidate.end_sec:
                    selected_rows.append(
                        {
                            "clip_id": candidate.clip_id,
                            "split": candidate.split,
                            "video_id": row.video_id,
                            "frame_timestamp": f"{row.timestamp:.3f}",
                            "x1": f"{row.x1:.6f}",
                            "y1": f"{row.y1:.6f}",
                            "x2": f"{row.x2:.6f}",
                            "y2": f"{row.y2:.6f}",
                            "label": row.label,
                            "entity_id": row.entity_id,
                            "is_protected": int(row.entity_id == by_clip[candidate.clip_id].protected_entity_id),
                        }
                    )

    selected_annotation_path = output_dir / "selected_annotations.csv"
    with selected_annotation_path.open("w", encoding="utf-8", newline="") as stream:
        fieldnames = [
            "clip_id",
            "split",
            "video_id",
            "frame_timestamp",
            "x1",
            "y1",
            "x2",
            "y2",
            "label",
            "entity_id",
            "is_protected",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected_rows)

    split_counts = Counter(candidate.split for candidate in candidates)
    unique_videos_by_split = {
        split: len({candidate.video_id for candidate in candidates if candidate.split == split})
        for split in split_counts
    }
    summary = {
        "protocol": "AVA-VEIL Selective Anonymization Protocol",
        "num_clips": len(candidates),
        "num_source_videos": len({candidate.video_id for candidate in candidates}),
        "clips_by_split": dict(sorted(split_counts.items())),
        "source_videos_by_split": dict(sorted(unique_videos_by_split.items())),
        "selected_annotation_rows": len(selected_rows),
        "parameters": parameters,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, sort_keys=True)


def read_manifest(path: Path) -> list[dict]:
    path = path.expanduser().resolve()
    if not path.exists():
        raise ProtocolError(f"Manifest does not exist: {path}")
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ProtocolError(f"{path}:{line_number}: invalid JSON") from exc
            records.append(record)
    if not records:
        raise ProtocolError(f"Manifest is empty: {path}")
    return records


def find_video(videos_dir: Path, video_id: str, extensions: Sequence[str]) -> Path | None:
    for extension in extensions:
        extension = extension if extension.startswith(".") else f".{extension}"
        candidate = videos_dir / f"{video_id}{extension}"
        if candidate.exists():
            return candidate
    matches = [p for p in videos_dir.rglob(f"{video_id}.*") if p.suffix.lower() in extensions]
    return sorted(matches)[0] if matches else None


def _expanded_pixel_box(box: Sequence[float], width: int, height: int, padding: float) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    px1, py1, px2, py2 = x1 * width, y1 * height, x2 * width, y2 * height
    box_w = px2 - px1
    box_h = py2 - py1
    px1 -= box_w * padding
    px2 += box_w * padding
    py1 -= box_h * padding
    py2 += box_h * padding
    return (
        max(0, min(width - 1, int(math.floor(px1)))),
        max(0, min(height - 1, int(math.floor(py1)))),
        max(1, min(width, int(math.ceil(px2)))),
        max(1, min(height, int(math.ceil(py2)))),
    )


def extract_reference_crop(
    video_path: Path,
    timestamp: float,
    box: Sequence[float],
    output_path: Path,
    padding: float,
) -> None:
    try:
        import cv2  # Imported lazily so selection does not require OpenCV.
    except ImportError as exc:
        raise ProtocolError("OpenCV is required for materialize: pip install opencv-python-headless") from exc

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ProtocolError(f"Cannot open source video: {video_path}")
    try:
        capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp) * 1000.0)
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise ProtocolError(f"Cannot read reference frame at {timestamp:.3f}s from {video_path}")

    height, width = frame.shape[:2]
    x1, y1, x2, y2 = _expanded_pixel_box(box, width, height, padding)
    if x2 <= x1 or y2 <= y1:
        raise ProtocolError(f"Invalid reference crop for {video_path} at {timestamp:.3f}s")
    crop = frame[y1:y2, x1:x2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), crop):
        raise ProtocolError(f"Failed to write target crop: {output_path}")


def materialize(
    manifest_path: Path,
    videos_dir: Path,
    output_dir: Path,
    *,
    local_time_offset: float,
    ffmpeg_binary: str,
    extensions: Sequence[str],
    target_padding: float,
    overwrite: bool,
    check_only: bool,
) -> dict:
    records = read_manifest(manifest_path)
    videos_dir = videos_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    normalized_extensions = tuple(
        extension.lower() if extension.startswith(".") else f".{extension.lower()}"
        for extension in extensions
    )

    ffmpeg_path = shutil.which(ffmpeg_binary)
    if not check_only and ffmpeg_path is None:
        raise ProtocolError(f"ffmpeg executable not found: {ffmpeg_binary}")

    results: list[dict] = []
    counters = Counter()
    for record in records:
        source_video = find_video(videos_dir, record["video_id"], normalized_extensions)
        result = dict(record)
        if source_video is None:
            result.update({"status": "missing_video", "source_video": None})
            counters["missing_video"] += 1
            results.append(result)
            continue

        local_start = float(record["start_sec"]) - local_time_offset
        local_reference = float(record["reference_timestamp"]) - local_time_offset
        if local_start < 0 or local_reference < 0:
            result.update(
                {
                    "status": "invalid_time_offset",
                    "source_video": str(source_video),
                    "local_start_sec": local_start,
                    "local_reference_sec": local_reference,
                }
            )
            counters["invalid_time_offset"] += 1
            results.append(result)
            continue

        split = record["split"]
        clip_path = output_dir / "clips" / split / f"{record['clip_id']}.mp4"
        target_path = output_dir / "targets" / split / f"{record['clip_id']}_target.jpg"
        result.update(
            {
                "source_video": str(source_video),
                "local_start_sec": round(local_start, 3),
                "local_reference_sec": round(local_reference, 3),
                "clip_path": str(clip_path),
                "target_image_path": str(target_path),
            }
        )

        if check_only:
            result["status"] = "ready"
            counters["ready"] += 1
            results.append(result)
            continue

        clip_path.parent.mkdir(parents=True, exist_ok=True)
        if overwrite or not clip_path.exists():
            command = [
                str(ffmpeg_path),
                "-hide_banner",
                "-loglevel",
                "error",
                "-y" if overwrite else "-n",
                "-ss",
                f"{local_start:.3f}",
                "-i",
                str(source_video),
                "-t",
                f"{float(record['duration_sec']):.3f}",
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
                str(clip_path),
            ]
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
            if completed.returncode != 0:
                result.update(
                    {
                        "status": "ffmpeg_failed",
                        "ffmpeg_stderr": completed.stderr[-2000:],
                    }
                )
                counters["ffmpeg_failed"] += 1
                results.append(result)
                continue

        if overwrite or not target_path.exists():
            try:
                extract_reference_crop(
                    source_video,
                    local_reference,
                    record["reference_box"],
                    target_path,
                    target_padding,
                )
            except ProtocolError as exc:
                result.update({"status": "target_crop_failed", "error": str(exc)})
                counters["target_crop_failed"] += 1
                results.append(result)
                continue

        result["status"] = "materialized"
        counters["materialized"] += 1
        results.append(result)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "materialized_manifest.jsonl", results)
    summary = {
        "manifest": str(manifest_path.expanduser().resolve()),
        "videos_dir": str(videos_dir),
        "output_dir": str(output_dir),
        "local_time_offset": local_time_offset,
        "check_only": check_only,
        "counts": dict(sorted(counters.items())),
    }
    with (output_dir / "materialize_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, sort_keys=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create and materialize an AVA-based VEIL selective-anonymization protocol."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    select_parser = subparsers.add_parser(
        "select", help="Select reproducible multi-person clips from AVA ActiveSpeaker annotations."
    )
    select_parser.add_argument("--annotations", type=Path, required=True)
    select_parser.add_argument("--output-dir", type=Path, required=True)
    select_parser.add_argument("--window-seconds", type=float, default=10.0)
    select_parser.add_argument("--step-seconds", type=float, default=5.0)
    select_parser.add_argument("--min-faces", type=int, default=2)
    select_parser.add_argument("--min-overlap-ratio", type=float, default=0.50)
    select_parser.add_argument("--min-entity-coverage", type=float, default=0.50)
    select_parser.add_argument("--min-normalized-face-size", type=float, default=0.06)
    select_parser.add_argument("--max-clips-per-video", type=int, default=3)
    select_parser.add_argument("--max-selected-overlap", type=float, default=0.25)
    select_parser.add_argument("--seed", type=int, default=2026)
    select_parser.add_argument(
        "--split-ratios",
        type=parse_split_ratios,
        default=parse_split_ratios("train=80,val=10,test=10"),
    )
    select_parser.add_argument("--limit-videos", type=int)

    materialize_parser = subparsers.add_parser(
        "materialize", help="Extract clips and protected target crops from local source videos."
    )
    materialize_parser.add_argument("--manifest", type=Path, required=True)
    materialize_parser.add_argument("--videos-dir", type=Path, required=True)
    materialize_parser.add_argument("--output-dir", type=Path, required=True)
    materialize_parser.add_argument(
        "--local-time-offset",
        type=float,
        default=0.0,
        help="Subtract this value from AVA timestamps for locally trimmed videos (often 900).",
    )
    materialize_parser.add_argument("--ffmpeg", default="ffmpeg")
    materialize_parser.add_argument(
        "--extensions", nargs="+", default=list(VIDEO_EXTENSIONS)
    )
    materialize_parser.add_argument("--target-padding", type=float, default=0.25)
    materialize_parser.add_argument("--overwrite", action="store_true")
    materialize_parser.add_argument("--check-only", action="store_true")
    return parser


def validate_select_args(args: argparse.Namespace) -> None:
    if args.window_seconds <= 0 or args.step_seconds <= 0:
        raise ProtocolError("window-seconds and step-seconds must be positive")
    if args.min_faces < 2:
        raise ProtocolError("min-faces must be at least 2 for selective anonymization")
    for name in ("min_overlap_ratio", "min_entity_coverage", "max_selected_overlap"):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            raise ProtocolError(f"{name.replace('_', '-')} must be in [0, 1]")
    if not 0.0 < args.min_normalized_face_size <= 1.0:
        raise ProtocolError("min-normalized-face-size must be in (0, 1]")
    if args.max_clips_per_video <= 0:
        raise ProtocolError("max-clips-per-video must be positive")


def run_select(args: argparse.Namespace) -> int:
    validate_select_args(args)
    grouped = load_annotations(args.annotations, args.limit_videos)
    candidates = select_candidates(
        grouped,
        window_seconds=args.window_seconds,
        step_seconds=args.step_seconds,
        min_faces=args.min_faces,
        min_overlap_ratio=args.min_overlap_ratio,
        min_entity_coverage=args.min_entity_coverage,
        min_normalized_face_size=args.min_normalized_face_size,
        max_clips_per_video=args.max_clips_per_video,
        max_selected_overlap=args.max_selected_overlap,
        seed=args.seed,
        split_ratios=args.split_ratios,
    )
    parameters = {
        "annotations": str(args.annotations.expanduser().resolve()),
        "window_seconds": args.window_seconds,
        "step_seconds": args.step_seconds,
        "min_faces": args.min_faces,
        "min_overlap_ratio": args.min_overlap_ratio,
        "min_entity_coverage": args.min_entity_coverage,
        "min_normalized_face_size": args.min_normalized_face_size,
        "max_clips_per_video": args.max_clips_per_video,
        "max_selected_overlap": args.max_selected_overlap,
        "seed": args.seed,
        "split_ratios": args.split_ratios,
        "limit_videos": args.limit_videos,
    }
    write_protocol_outputs(args.output_dir, candidates, grouped, parameters)
    print(
        json.dumps(
            {
                "status": "ok",
                "num_clips": len(candidates),
                "num_source_videos": len({candidate.video_id for candidate in candidates}),
                "output_dir": str(args.output_dir.expanduser().resolve()),
            },
            ensure_ascii=False,
        )
    )
    return 0


def run_materialize(args: argparse.Namespace) -> int:
    if args.target_padding < 0:
        raise ProtocolError("target-padding must be non-negative")
    summary = materialize(
        args.manifest,
        args.videos_dir,
        args.output_dir,
        local_time_offset=args.local_time_offset,
        ffmpeg_binary=args.ffmpeg,
        extensions=args.extensions,
        target_padding=args.target_padding,
        overwrite=args.overwrite,
        check_only=args.check_only,
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "select":
            return run_select(args)
        if args.command == "materialize":
            return run_materialize(args)
        parser.error(f"Unknown command: {args.command}")
    except ProtocolError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
