#!/usr/bin/env python3
"""Generate an HTML review page for corrected Panoptic benchmark outputs."""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence


class ReviewError(RuntimeError):
    """Raised when review generation fails."""


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def rel(path: Path, base: Path) -> str:
    return Path(os.path.relpath(path, base)).as_posix()


def run_ffmpeg(command: list[str]) -> str | None:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        return completed.stderr[-1200:].strip() or "ffmpeg failed"
    return None


def transcode_h264(ffmpeg: str, src: Path, dst: Path, *, overwrite: bool) -> str | None:
    if dst.exists() and dst.stat().st_size > 0 and not overwrite:
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    return run_ffmpeg(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(src),
            "-map",
            "0:v:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(dst),
        ]
    )


def extract_frame(ffmpeg: str, src: Path, dst: Path, timestamp: float, *, overwrite: bool) -> str | None:
    if dst.exists() and dst.stat().st_size > 0 and not overwrite:
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    return run_ffmpeg(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{max(0.0, timestamp):.3f}",
            "-i",
            str(src),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(dst),
        ]
    )


def run_output(benchmark_dir: Path, condition: str, clip_id: str) -> Path:
    return benchmark_dir / condition / "runs" / clip_id / f"{clip_id}_output.mp4"


def metric_by_clip(benchmark_dir: Path, condition: str) -> dict[str, dict]:
    path = benchmark_dir / condition / "metadata_eval" / "per_clip_metrics.csv"
    return {row["clip_id"]: row for row in read_csv(path)}


def write_review_csv(path: Path, clip_ids: Sequence[str], *, overwrite: bool) -> bool:
    if path.exists() and not overwrite:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "clip_id",
        "accepted",
        "target_preserved",
        "non_target_processed",
        "replacement_ok",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for clip_id in clip_ids:
            writer.writerow({"clip_id": clip_id})
    return True


def render_html(records: Sequence[dict], output_dir: Path, review_csv: Path, summary: dict) -> str:
    cards = []
    for record in records:
        clip_id = record["clip_id"]
        target_href = rel(Path(record["target_image_review"]), output_dir)
        metric = record["metrics"]
        metric_rows = "".join(
            f"<tr><th>{html.escape(k)}</th><td>{html.escape(str(v))}</td></tr>"
            for k, v in metric.items()
        )

        videos = []
        for key, label in (
            ("original_video", "Original"),
            ("full_video", "Full VEIL"),
            ("no_blur_video", "No Blur Fallback"),
            ("no_lock_video", "No Identity Lock"),
        ):
            path = record.get(key)
            if not path:
                continue
            videos.append(
                f'''<section class="video-panel"><h3>{html.escape(label)}</h3><video controls preload="metadata" src="{html.escape(rel(Path(path), output_dir))}"></video></section>'''
            )

        frames = []
        for label in ("first", "middle", "last"):
            row = []
            for key, title in (
                ("original", "Original"),
                ("full", "Full"),
                ("no_blur", "No Blur"),
                ("no_lock", "No Lock"),
            ):
                path = record.get("frames", {}).get(key, {}).get(label)
                if path:
                    row.append(
                        f'''<figure><img src="{html.escape(rel(Path(path), output_dir))}" alt="{html.escape(title)} {html.escape(label)}"><figcaption>{html.escape(title)} {html.escape(label)}</figcaption></figure>'''
                    )
            frames.append(f'''<div class="frame-row"><h4>{html.escape(label)}</h4><div class="frame-grid">{''.join(row)}</div></div>''')

        warnings = "".join(f"<li>{html.escape(error)}</li>" for error in record.get("errors", []))
        warning_html = f"<ul class=\"warnings\">{warnings}</ul>" if warnings else ""
        cards.append(
            f'''<article class="clip" id="{html.escape(clip_id)}">
  <header><h2>{html.escape(clip_id)}</h2></header>
  <section class="topline">
    <div><h3>Target Crop</h3><img class="target" src="{html.escape(target_href)}" alt="target crop"></div>
    <div><h3>Full Metrics</h3><table>{metric_rows}</table></div>
  </section>
  <section class="videos">{''.join(videos)}</section>
  <section class="frames"><h3>Preview Frames</h3>{''.join(frames)}</section>
  {warning_html}
</article>'''
        )

    return f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Corrected Panoptic VEIL Review</title>
  <style>
    :root {{ --ink:#161b22; --muted:#667085; --line:#d0d7de; --band:#f6f8fa; --accent:#0969da; }}
    body {{ margin:0; font-family:Arial, sans-serif; color:var(--ink); background:white; }}
    .page {{ padding:24px 32px; background:var(--band); border-bottom:1px solid var(--line); }}
    main {{ max-width:1280px; margin:0 auto; padding:22px; }}
    h1 {{ margin:0 0 8px; font-size:28px; }}
    h2 {{ margin:0; font-size:18px; overflow-wrap:anywhere; }}
    h3 {{ margin:0 0 8px; font-size:13px; color:var(--muted); text-transform:uppercase; }}
    h4 {{ margin:14px 0 8px; font-size:14px; color:var(--muted); }}
    code {{ padding:2px 5px; border:1px solid var(--line); border-radius:4px; background:white; }}
    .summary {{ display:flex; flex-wrap:wrap; gap:12px; color:var(--muted); }}
    .clip {{ padding:24px 0 34px; border-top:2px solid var(--line); }}
    .topline {{ display:grid; grid-template-columns:260px 1fr; gap:18px; margin-top:14px; align-items:start; }}
    .target {{ width:100%; max-height:340px; object-fit:contain; background:#111; }}
    .videos {{ display:grid; grid-template-columns:repeat(2, minmax(280px, 1fr)); gap:16px; margin-top:18px; }}
    video {{ width:100%; max-height:430px; background:#111; }}
    .frame-grid {{ display:grid; grid-template-columns:repeat(4, minmax(160px, 1fr)); gap:10px; }}
    figure {{ margin:0; }}
    figure img {{ width:100%; background:#111; }}
    figcaption {{ margin-top:4px; color:var(--muted); font-size:12px; }}
    table {{ width:100%; border-collapse:collapse; font-size:13px; }}
    th, td {{ border-top:1px solid var(--line); padding:6px 8px; text-align:left; }}
    th {{ width:220px; color:var(--muted); font-weight:600; }}
    .warnings {{ color:#9a3412; background:#fff7ed; border:1px solid #fed7aa; padding:10px 24px; }}
    a {{ color:var(--accent); }}
    @media (max-width: 860px) {{ .topline, .videos, .frame-grid {{ grid-template-columns:1fr; }} main {{ padding:14px; }} .page {{ padding:18px; }} }}
  </style>
</head>
<body>
  <header class="page">
    <h1>Corrected Panoptic VEIL Review</h1>
    <div class="summary">
      <span>Clips: {int(summary['num_clips'])}</span>
      <span>Videos transcoded: {int(summary['videos_transcoded'])}</span>
      <span>Errors: {int(summary['num_errors'])}</span>
      <span>CSV: <a href="{html.escape(rel(review_csv, output_dir))}">{html.escape(review_csv.name)}</a></span>
    </div>
    <p>Review <code>Full VEIL</code> for all clips. Ablation videos are included for the blur-only and identity-lock examples.</p>
  </header>
  <main>{''.join(cards)}</main>
</body>
</html>'''


def generate_review(
    benchmark_dir: Path,
    manifest_path: Path,
    output_dir: Path,
    *,
    ffmpeg: str,
    overwrite: bool,
    overwrite_csv: bool,
    include_ablation_examples: bool,
) -> dict:
    benchmark_dir = benchmark_dir.expanduser().resolve()
    manifest_path = manifest_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    ffmpeg_path = shutil.which(ffmpeg)
    if ffmpeg_path is None:
        raise ReviewError(f"ffmpeg not found: {ffmpeg}")

    summary = read_json(benchmark_dir / "benchmark_summary.json")
    clip_ids = summary["selected_clips"]
    manifest = {row["clip_id"]: row for row in read_jsonl(manifest_path)}
    full_metrics = metric_by_clip(benchmark_dir, "full")
    no_blur_metrics = metric_by_clip(benchmark_dir, "no_blur_fallback")
    no_lock_metrics = metric_by_clip(benchmark_dir, "no_identity_lock")

    output_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = output_dir / "videos"
    frames_dir = output_dir / "frames"
    targets_dir = output_dir / "targets"
    errors = []
    videos_transcoded = 0
    records = []

    blur_example = "170221_haggling_b2_hd_00_27_0001350000"
    lock_example = "170224_haggling_b1_hd_00_08_0001550000"

    for clip_id in clip_ids:
        row = manifest[clip_id]
        record = {
            "clip_id": clip_id,
            "metrics": {
                "target_coverage": full_metrics[clip_id]["target_coverage"],
                "non_target_face_rows": full_metrics[clip_id]["non_target_face_rows"],
                "swap_coverage": full_metrics[clip_id]["swap_coverage"],
                "blur_coverage": full_metrics[clip_id]["blur_coverage"],
                "full_exposure": full_metrics[clip_id]["non_target_exposure_rate"],
                "no_blur_exposure": no_blur_metrics[clip_id]["non_target_exposure_rate"],
                "no_lock_target_coverage": no_lock_metrics[clip_id]["target_coverage"],
            },
            "frames": {},
            "errors": [],
        }

        target_src = Path(row["target_image_path"])
        target_dst = targets_dir / f"{clip_id}_target.jpg"
        target_dst.parent.mkdir(parents=True, exist_ok=True)
        if not target_dst.exists() or overwrite:
            shutil.copy2(target_src, target_dst)
        record["target_image_review"] = str(target_dst)

        video_sources = {
            "original": Path(row["clip_path"]),
            "full": run_output(benchmark_dir, "full", clip_id),
        }
        if include_ablation_examples and clip_id == blur_example:
            video_sources["no_blur"] = run_output(benchmark_dir, "no_blur_fallback", clip_id)
        if include_ablation_examples and clip_id == lock_example:
            video_sources["no_lock"] = run_output(benchmark_dir, "no_identity_lock", clip_id)

        for key, src in video_sources.items():
            dst = videos_dir / f"{clip_id}_{key}.mp4"
            err = transcode_h264(ffmpeg_path, src, dst, overwrite=overwrite)
            if err:
                record["errors"].append(f"{key} transcode: {err}")
            else:
                videos_transcoded += int(overwrite or not dst.exists() or dst.stat().st_size > 0)
                record[f"{key}_video"] = str(dst)
                record.setdefault("frames", {})[key] = {}
                for label, timestamp in (("first", 0.0), ("middle", 5.0), ("last", 9.0)):
                    frame_dst = frames_dir / f"{clip_id}_{key}_{label}.jpg"
                    frame_err = extract_frame(ffmpeg_path, dst, frame_dst, timestamp, overwrite=overwrite)
                    if frame_err:
                        record["errors"].append(f"{key} {label} frame: {frame_err}")
                    else:
                        record["frames"][key][label] = str(frame_dst)

        errors.extend({"clip_id": clip_id, "error": error} for error in record["errors"])
        records.append(record)

    review_csv = output_dir / "review.csv"
    write_review_csv(review_csv, clip_ids, overwrite=overwrite_csv)
    page_summary = {
        "benchmark_dir": str(benchmark_dir),
        "manifest_path": str(manifest_path),
        "output_dir": str(output_dir),
        "index_html": str(output_dir / "index.html"),
        "review_csv": str(review_csv),
        "num_clips": len(records),
        "videos_transcoded": videos_transcoded,
        "num_errors": len(errors),
        "errors": errors,
        "ablation_examples": {
            "blur_fallback": blur_example,
            "identity_lock": lock_example,
        },
    }
    (output_dir / "index.html").write_text(render_html(records, output_dir, review_csv, page_summary), encoding="utf-8")
    write_json(output_dir / "review_summary.json", page_summary)
    return page_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate corrected Panoptic benchmark review HTML.")
    parser.add_argument("--benchmark-dir", type=Path, default=Path("data/panoptic_veil_materialized_10/benchmarks/panoptic10_ablation_fixed"))
    parser.add_argument("--manifest", type=Path, default=Path("data/panoptic_veil_materialized_10/materialized_manifest.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/panoptic_veil_materialized_10/benchmarks/panoptic10_ablation_fixed_review_html"))
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite-csv", action="store_true")
    parser.add_argument("--no-ablation-examples", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = generate_review(
            args.benchmark_dir,
            args.manifest,
            args.output_dir,
            ffmpeg=args.ffmpeg,
            overwrite=args.overwrite,
            overwrite_csv=args.overwrite_csv,
            include_ablation_examples=not args.no_ablation_examples,
        )
    except ReviewError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
