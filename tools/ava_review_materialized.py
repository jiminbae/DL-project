#!/usr/bin/env python3
"""Create an HTML review page for materialized AVA-VEIL clips."""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence


REJECTION_REASONS = (
    "camera_cut",
    "wrong_target",
    "face_too_small",
    "target_crop_truncated",
    "single_person",
    "annotation_mismatch",
    "severe_occlusion",
    "other",
)


class ReviewError(RuntimeError):
    """Raised when review page generation cannot continue."""


def read_manifest(path: Path) -> list[dict]:
    path = path.expanduser().resolve()
    if not path.exists():
        raise ReviewError(f"Manifest does not exist: {path}")

    records: list[dict] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ReviewError(f"{path}:{line_number}: invalid JSON") from exc

    if not records:
        raise ReviewError(f"Manifest is empty: {path}")
    return records


def safe_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def relative_href(path: Path, base_dir: Path) -> str:
    return Path(os.path.relpath(path, base_dir)).as_posix()


def extract_frame(
    ffmpeg_path: str,
    clip_path: Path,
    timestamp: float,
    output_path: Path,
    *,
    overwrite: bool,
) -> str | None:
    if output_path.exists() and not overwrite:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{max(0.0, timestamp):.3f}",
        "-i",
        str(clip_path),
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(output_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        return completed.stderr[-1000:].strip() or "ffmpeg frame extraction failed"
    if not output_path.exists() or output_path.stat().st_size == 0:
        return "ffmpeg did not write review frame"
    return None


def frame_timestamps(duration_sec: float) -> list[tuple[str, float]]:
    duration = max(0.0, duration_sec)
    return [
        ("first", 0.0),
        ("middle", duration * 0.5),
        ("last", duration * 0.9 if duration > 0 else 0.0),
    ]


def prepare_review_records(
    records: Sequence[dict],
    review_dir: Path,
    *,
    ffmpeg_path: str,
    overwrite_frames: bool,
) -> tuple[list[dict], list[dict]]:
    review_records: list[dict] = []
    errors: list[dict] = []
    frames_dir = review_dir / "frames"

    for record in records:
        item = dict(record)
        clip_id = str(record.get("clip_id", "unknown_clip"))
        clip_path = Path(str(record.get("clip_path", ""))).expanduser()
        if not clip_path.is_absolute():
            clip_path = clip_path.resolve()
        duration = float(record.get("duration_sec", 0.0) or 0.0)

        frame_paths: dict[str, str] = {}
        frame_errors: list[str] = []
        if not clip_path.exists():
            frame_errors.append(f"missing clip: {clip_path}")
        else:
            for label, timestamp in frame_timestamps(duration):
                frame_path = frames_dir / f"{safe_stem(clip_id)}_{label}.jpg"
                error = extract_frame(
                    ffmpeg_path,
                    clip_path,
                    timestamp,
                    frame_path,
                    overwrite=overwrite_frames,
                )
                if error is None:
                    frame_paths[label] = relative_href(frame_path, review_dir)
                else:
                    frame_errors.append(f"{label}: {error}")

        if frame_errors:
            errors.append({"clip_id": clip_id, "errors": frame_errors})
        item["review_frame_paths"] = frame_paths
        item["review_errors"] = frame_errors
        review_records.append(item)

    return review_records, errors


def write_review_csv(records: Sequence[dict], csv_path: Path, *, overwrite: bool) -> bool:
    if csv_path.exists() and not overwrite:
        return False

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["clip_id", "accepted", "reason", "reviewer"])
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "clip_id": record.get("clip_id", ""),
                    "accepted": "",
                    "reason": "",
                    "reviewer": "",
                }
            )
    return True


def render_review_html(records: Sequence[dict], review_dir: Path, csv_path: Path, summary: dict) -> str:
    reason_options = "".join(f"<code>{html.escape(reason)}</code>" for reason in REJECTION_REASONS)
    cards: list[str] = []

    for record in records:
        clip_id = str(record.get("clip_id", ""))
        clip_path = Path(str(record.get("clip_path", ""))).expanduser()
        target_path = Path(str(record.get("target_image_path", ""))).expanduser()
        clip_href = relative_href(clip_path, review_dir) if clip_path.exists() else ""
        target_href = relative_href(target_path, review_dir) if target_path.exists() else ""

        frame_imgs = []
        for label in ("first", "middle", "last"):
            frame_href = record.get("review_frame_paths", {}).get(label)
            if frame_href:
                frame_imgs.append(
                    f'<figure><img src="{html.escape(frame_href)}" alt="{html.escape(label)} frame"><figcaption>{html.escape(label)}</figcaption></figure>'
                )
        frames_html = "\n".join(frame_imgs) or '<p class="warning">No review frames generated.</p>'

        errors = record.get("review_errors", [])
        errors_html = "" if not errors else '<p class="warning">' + html.escape("; ".join(errors)) + "</p>"
        target_html = (
            f'<img class="target" src="{html.escape(target_href)}" alt="target crop">'
            if target_href
            else '<div class="missing">Missing target image</div>'
        )
        video_html = (
            f'<video controls preload="metadata" src="{html.escape(clip_href)}"></video>'
            if clip_href
            else '<div class="missing">Missing clip video</div>'
        )

        metadata_rows = [
            ("clip_id", clip_id),
            ("split", record.get("split", "")),
            ("video_id", record.get("video_id", "")),
            ("protected_entity_id", record.get("protected_entity_id", "")),
            ("reference_timestamp", record.get("reference_timestamp", "")),
            ("local_reference_sec", record.get("local_reference_sec", "")),
            ("reference_box", record.get("reference_box", "")),
            ("num_entities", record.get("num_entities", "")),
            ("status", record.get("status", "")),
        ]
        metadata_html = "".join(
            f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(json.dumps(value, ensure_ascii=False) if isinstance(value, list) else str(value))}</td></tr>"
            for key, value in metadata_rows
        )

        cards.append(
            f"""<article class="clip-card" id="{html.escape(safe_stem(clip_id))}">
  <header><h2>{html.escape(clip_id)}</h2></header>
  <section class="media-grid">
    <div><h3>Target</h3>{target_html}</div>
    <div><h3>Clip</h3>{video_html}</div>
  </section>
  <section><h3>Frames</h3><div class="frames">{frames_html}</div></section>
  <section><h3>Manifest</h3><table>{metadata_html}</table>{errors_html}</section>
</article>"""
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AVA-VEIL Materialized Review</title>
  <style>
    :root {{ color-scheme: light; --ink: #172026; --muted: #66737f; --line: #d8dee4; --band: #f5f7f9; --accent: #0b6bcb; }}
    body {{ margin: 0; font-family: Arial, sans-serif; color: var(--ink); background: white; }}
    header.page {{ padding: 24px 32px; border-bottom: 1px solid var(--line); background: var(--band); }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    h2 {{ margin: 0; font-size: 18px; overflow-wrap: anywhere; }}
    h3 {{ margin: 0 0 8px; font-size: 14px; color: var(--muted); text-transform: uppercase; }}
    code {{ display: inline-block; margin: 2px 4px 2px 0; padding: 2px 6px; border: 1px solid var(--line); border-radius: 4px; background: white; }}
    .summary {{ display: flex; flex-wrap: wrap; gap: 12px; color: var(--muted); }}
    .clip-card {{ border-top: 2px solid var(--line); padding: 22px 0 30px; }}
    .media-grid {{ display: grid; grid-template-columns: minmax(180px, 280px) minmax(320px, 1fr); gap: 18px; margin: 16px 0; align-items: start; }}
    img.target {{ width: 100%; max-height: 360px; object-fit: contain; background: #111; }}
    video {{ width: 100%; max-height: 520px; background: #111; }}
    .frames {{ display: grid; grid-template-columns: repeat(3, minmax(160px, 1fr)); gap: 12px; margin-bottom: 16px; }}
    figure {{ margin: 0; }}
    figure img {{ width: 100%; background: #111; }}
    figcaption {{ margin-top: 4px; color: var(--muted); font-size: 13px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ border-top: 1px solid var(--line); padding: 7px 8px; text-align: left; vertical-align: top; }}
    th {{ width: 210px; color: var(--muted); font-weight: 600; }}
    .warning, .missing {{ color: #a64200; background: #fff4e6; border: 1px solid #ffd8a8; padding: 8px 10px; border-radius: 4px; }}
    a {{ color: var(--accent); }}
    @media (max-width: 760px) {{ .media-grid, .frames {{ grid-template-columns: 1fr; }} main {{ padding: 16px; }} header.page {{ padding: 18px; }} }}
  </style>
</head>
<body>
  <header class="page">
    <h1>AVA-VEIL Materialized Review</h1>
    <div class="summary">
      <span>Clips: {int(summary.get("num_records", 0))}</span>
      <span>Frame extraction errors: {int(summary.get("num_frame_error_records", 0))}</span>
      <span>Review CSV: <a href="{html.escape(relative_href(csv_path, review_dir))}">{html.escape(csv_path.name)}</a></span>
    </div>
    <p>Fill <code>accepted</code> with <code>1</code> or <code>0</code>. Allowed rejection reasons: {reason_options}</p>
  </header>
  <main>
    {''.join(cards)}
  </main>
</body>
</html>
"""


def generate_review_page(
    manifest_path: Path,
    review_dir: Path,
    *,
    ffmpeg_binary: str,
    overwrite_frames: bool = False,
    overwrite_csv: bool = False,
) -> dict:
    records = read_manifest(manifest_path)
    review_dir = review_dir.expanduser().resolve()
    review_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg_path = shutil.which(ffmpeg_binary)
    if ffmpeg_path is None:
        raise ReviewError(f"ffmpeg executable not found: {ffmpeg_binary}")

    review_records, frame_errors = prepare_review_records(
        records,
        review_dir,
        ffmpeg_path=ffmpeg_path,
        overwrite_frames=overwrite_frames,
    )
    csv_path = review_dir / "review.csv"
    csv_created = write_review_csv(review_records, csv_path, overwrite=overwrite_csv)

    summary = {
        "manifest": str(manifest_path.expanduser().resolve()),
        "review_dir": str(review_dir),
        "index_html": str(review_dir / "index.html"),
        "review_csv": str(csv_path),
        "review_csv_created": csv_created,
        "num_records": len(review_records),
        "num_frame_error_records": len(frame_errors),
        "frame_errors": frame_errors,
    }
    html_text = render_review_html(review_records, review_dir, csv_path, summary)
    (review_dir / "index.html").write_text(html_text, encoding="utf-8")
    (review_dir / "review_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate an HTML review page for materialized AVA-VEIL clips.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/ava_veil_materialized/materialized_manifest.jsonl"),
    )
    parser.add_argument(
        "--review-dir",
        type=Path,
        default=Path("data/ava_veil_materialized/review"),
    )
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--overwrite-frames", action="store_true")
    parser.add_argument("--overwrite-csv", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = generate_review_page(
            args.manifest,
            args.review_dir,
            ffmpeg_binary=args.ffmpeg,
            overwrite_frames=args.overwrite_frames,
            overwrite_csv=args.overwrite_csv,
        )
    except ReviewError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
