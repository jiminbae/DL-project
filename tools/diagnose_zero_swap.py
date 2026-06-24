#!/usr/bin/env python3
"""Diagnose accepted VEIL clips with no logged swaps."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence


class DiagnosticError(RuntimeError):
    """Raised when zero-swap diagnostics cannot be generated."""


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def bbox_min_side(row: dict) -> float | None:
    bbox = row.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(value) for value in bbox)
    except (TypeError, ValueError):
        return None
    return min(max(0.0, x2 - x1), max(0.0, y2 - y1))


def quantiles(values: list[float]) -> dict:
    if not values:
        return {}
    values = sorted(values)
    def pick(frac: float) -> float:
        index = min(len(values) - 1, max(0, round((len(values) - 1) * frac)))
        return round(values[index], 3)
    return {"min": pick(0.0), "p25": pick(0.25), "median": pick(0.5), "p75": pick(0.75), "max": pick(1.0)}


def classify_zero_swap(metrics: dict) -> str:
    non_target = int(metrics.get("non_target_face_rows", 0))
    swap = int(metrics.get("non_target_swap_rows", 0))
    blur = int(metrics.get("non_target_blur_rows", 0))
    preserve = int(metrics.get("non_target_preserve_rows", 0))
    unprocessed = int(metrics.get("non_target_unprocessed_rows", 0))
    unknown = int(metrics.get("non_target_unknown_rows", 0))
    if non_target == 0:
        return "no_valid_non_target"
    if swap == 0 and blur > 0 and preserve == 0 and unprocessed == 0 and unknown == 0:
        return "blur_only_success"
    if preserve > 0 or unprocessed > 0:
        return "privacy_exposure"
    if unknown > 0:
        return "metadata_logging_gap"
    return "other"


def extract_frame(video: Path, frame_idx: int, output_path: Path) -> bool:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selector = f"select=eq(n\\,{max(0, frame_idx - 1)})"
    completed = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video),
            "-vf",
            selector,
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0


def make_contact_sheet(original: Path, output: Path, clip_id: str, frame_indices: Sequence[int], output_path: Path) -> str | None:
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None

    temp_dir = output_path.parent / "_frames"
    panels = []
    for frame_idx in frame_indices:
        original_frame = temp_dir / f"{clip_id}_original_f{frame_idx:03d}.jpg"
        output_frame = temp_dir / f"{clip_id}_output_f{frame_idx:03d}.jpg"
        if not extract_frame(original, frame_idx, original_frame):
            continue
        if not extract_frame(output, frame_idx, output_frame):
            continue
        original_img = cv2.imread(str(original_frame))
        output_img = cv2.imread(str(output_frame))
        if original_img is None or output_img is None:
            continue

        def fit(image, width=480, height=270):
            h, w = image.shape[:2]
            scale = min(width / w, height / h)
            resized = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
            canvas = np.full((height, width, 3), 248, dtype=np.uint8)
            y = (height - resized.shape[0]) // 2
            x = (width - resized.shape[1]) // 2
            canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
            return canvas

        row = np.concatenate([fit(original_img), fit(output_img)], axis=1)
        cv2.putText(row, f"Frame {frame_idx}: Original | VEIL Output", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        panels.append(row)

    if not panels:
        return None
    sheet = np.concatenate(panels, axis=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    return str(output_path)


def diagnose(metadata_eval_dir: Path, materialized_manifest: Path, smoke_summary: Path, output_dir: Path) -> dict:
    metadata_eval_dir = metadata_eval_dir.expanduser().resolve()
    per_clip = read_csv(metadata_eval_dir / "per_clip_metrics.csv")
    actions = read_jsonl(metadata_eval_dir / "per_frame_actions.jsonl")
    manifest = {row["clip_id"]: row for row in read_jsonl(materialized_manifest.expanduser().resolve())}
    smoke = {row["clip_id"]: row for row in json.loads(smoke_summary.expanduser().resolve().read_text(encoding="utf-8"))}
    zero_swap = [
        row for row in per_clip
        if row.get("accepted") == "yes" and int(row.get("non_target_swap_rows", 0)) == 0
    ]

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics = []
    for metrics in zero_swap:
        clip_id = metrics["clip_id"]
        clip_actions = [row for row in actions if row["clip_id"] == clip_id]
        fallback_counts = Counter(reason for row in clip_actions for reason in row.get("fallback_reasons", []))
        quality_counts = Counter(row.get("quality") for row in clip_actions)
        state_counts = Counter(row["state"] for row in clip_actions)
        non_target_min_sides = [
            side for row in clip_actions
            if row.get("is_background") and (side := bbox_min_side(row)) is not None
        ]
        frame_indices = sorted({int(row["frame_idx"]) for row in clip_actions})
        sampled_frames = [frame_indices[0], frame_indices[len(frame_indices) // 2], frame_indices[-1]] if frame_indices else []
        timeline_path = output_dir / f"{clip_id}_timeline.csv"
        write_csv(timeline_path, clip_actions)
        contact_sheet = None
        if clip_id in manifest and clip_id in smoke:
            contact_sheet = make_contact_sheet(
                Path(manifest[clip_id]["clip_path"]),
                Path(smoke[clip_id]["output_video"]),
                clip_id,
                sampled_frames,
                output_dir / f"{clip_id}_contact_sheet.jpg",
            )

        diagnostic = {
            "clip_id": clip_id,
            "classification": classify_zero_swap(metrics),
            "total_frames": int(metrics.get("total_frames", 0)),
            "protected_face_rows": int(metrics.get("protected_face_rows", 0)),
            "non_target_face_rows": int(metrics.get("non_target_face_rows", 0)),
            "state_counts": dict(sorted(state_counts.items())),
            "quality_counts": dict(sorted((str(key), value) for key, value in quality_counts.items())),
            "fallback_reason_counts": dict(sorted(fallback_counts.items())),
            "non_target_face_min_side": quantiles(non_target_min_sides),
            "stable_face_ids": sorted({row["stable_face_id"] for row in clip_actions if row.get("stable_face_id") is not None}),
            "raw_track_ids": sorted({row["raw_track_id"] for row in clip_actions if row.get("raw_track_id") is not None}),
            "timeline_csv": str(timeline_path),
            "contact_sheet": contact_sheet,
            "notes": "Automatic classification is based on conservative metadata states and should be manually spot-checked.",
        }
        diagnostic_path = output_dir / f"{clip_id}_diagnostic.json"
        diagnostic_path.write_text(json.dumps(diagnostic, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        diagnostics.append(diagnostic)

    summary = {
        "zero_swap_accepted_clips": len(diagnostics),
        "diagnostics": diagnostics,
    }
    (output_dir / "zero_swap_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return {"output_dir": str(output_dir), "zero_swap_accepted_clips": len(diagnostics)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose accepted VEIL clips with no metadata-evaluator SWAP rows.")
    parser.add_argument("--metadata-eval-dir", type=Path, required=True)
    parser.add_argument("--materialized-manifest", type=Path, required=True)
    parser.add_argument("--smoke-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        print(json.dumps(diagnose(args.metadata_eval_dir, args.materialized_manifest, args.smoke_summary, args.output_dir), ensure_ascii=False))
    except DiagnosticError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
