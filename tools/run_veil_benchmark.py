#!/usr/bin/env python3
"""Run VEIL benchmark conditions over materialized clips."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.evaluate_veil_metadata import evaluate as evaluate_metadata


CONDITIONS = {
    "full": {
        "description": "default VEIL with identity lock, face swap, and blur fallback",
        "enable_face_swap": True,
        "identity_lock_enabled": True,
        "blur_fallback_enabled": True,
    },
    "no_identity_lock": {
        "description": "temporal identity lock is disabled; only direct target matches are preserved",
        "enable_face_swap": True,
        "identity_lock_enabled": False,
        "blur_fallback_enabled": True,
    },
    "no_blur_fallback": {
        "description": "fallback blur is disabled after failed swaps or low-quality background faces",
        "enable_face_swap": True,
        "identity_lock_enabled": True,
        "blur_fallback_enabled": False,
    },
}


class BenchmarkError(RuntimeError):
    """Raised when a benchmark run cannot be completed."""


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.expanduser().resolve().open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_review(path: Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    with path.expanduser().resolve().open("r", encoding="utf-8", newline="") as stream:
        return {row["clip_id"]: row for row in csv.DictReader(stream)}


def accepted_clip_ids(review_rows: dict[str, dict]) -> set[str]:
    return {clip_id for clip_id, row in review_rows.items() if row.get("accepted") == "yes"}


def select_manifest_rows(
    manifest_path: Path,
    *,
    review_csv: Path | None = None,
    accepted_only: bool = False,
    clip_ids: Sequence[str] | None = None,
    clip_limit: int | None = None,
) -> list[dict]:
    rows = [row for row in read_jsonl(manifest_path) if row.get("status", "materialized") == "materialized"]
    wanted = set(clip_ids or [])
    if wanted:
        rows = [row for row in rows if row.get("clip_id") in wanted]

    review_rows = load_review(review_csv)
    if accepted_only:
        accepted = accepted_clip_ids(review_rows)
        rows = [row for row in rows if row.get("clip_id") in accepted]

    rows = sorted(rows, key=lambda row: row["clip_id"])
    if clip_limit is not None:
        rows = rows[:clip_limit]
    return rows


def validate_condition_names(names: Sequence[str]) -> list[str]:
    result = []
    for name in names:
        if name not in CONDITIONS:
            raise BenchmarkError(f"Unknown condition: {name}. Choices: {', '.join(sorted(CONDITIONS))}")
        result.append(name)
    return result


def main_hybrid_launcher(
    *,
    clip_path: Path,
    target_image_path: Path,
    output_video: Path,
    log_path: Path,
    face_metadata_path: Path,
    tracking_metadata_path: Path,
    condition: dict,
) -> str:
    payload = {
        "video_path": str(clip_path),
        "target_dir": str(target_image_path.parent),
        "target_pattern": target_image_path.name,
        "target_image_path": str(target_image_path),
        "output_path": str(output_video),
        "log_path": str(log_path),
        "metadata_path": str(face_metadata_path),
        "tracking_metadata_path": str(tracking_metadata_path),
        "enable_face_swap": bool(condition["enable_face_swap"]),
        "identity_lock_enabled": bool(condition["identity_lock_enabled"]),
        "blur_fallback_enabled": bool(condition["blur_fallback_enabled"]),
    }
    payload_literal = json.dumps(json.dumps(payload))
    return f'''
import json
from pathlib import Path
payload = json.loads({payload_literal})
import config
config.VIDEO_PATH = payload["video_path"]
config.TARGET_DIR = payload["target_dir"]
config.TARGET_PATTERN = payload["target_pattern"]
config.TARGET_IMAGE_PATH = payload["target_image_path"]
config.OUTPUT_PATH = payload["output_path"]
config.LOG_PATH = payload["log_path"]
config.METADATA_PATH = payload["metadata_path"]
config.TRACKING_METADATA_PATH = payload["tracking_metadata_path"]
config.ENABLE_FACE_SWAP = payload["enable_face_swap"]
Path(config.OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
Path(config.LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
Path(config.METADATA_PATH).parent.mkdir(parents=True, exist_ok=True)
Path(config.TRACKING_METADATA_PATH).parent.mkdir(parents=True, exist_ok=True)
import main_hybrid
if not payload["identity_lock_enabled"]:
    main_hybrid.is_known_target_face = lambda stable_face_id: False
    main_hybrid.is_recent_target_track = lambda raw_track_id, current_frame_idx: False
    main_hybrid.mark_target_seen = lambda stable_face_id, raw_track_id, current_frame_idx: None
    _prepare_track = main_hybrid.prepare_track
    def _direct_match_only_prepare_track(original_frame, track, target_embeddings, current_frame_idx):
        ctx = _prepare_track(original_frame, track, target_embeddings, current_frame_idx)
        ctx["is_target_final"] = bool(ctx.get("is_target"))
        ctx["is_background"] = not ctx["is_target_final"]
        return ctx
    main_hybrid.prepare_track = _direct_match_only_prepare_track
if not payload["blur_fallback_enabled"]:
    main_hybrid.apply_fallback_blur = lambda frame, bbox: frame
main_hybrid.main()
'''.strip()


def run_one_clip(
    row: dict,
    *,
    condition_name: str,
    condition: dict,
    condition_dir: Path,
    python_executable: str,
    veil_dir: Path,
    timeout_sec: int | None,
    skip_existing: bool,
) -> dict:
    clip_id = row["clip_id"]
    run_dir = condition_dir / "runs" / clip_id
    output_video = run_dir / f"{clip_id}_output.mp4"
    log_path = run_dir / f"tracking_target_{clip_id}_log.txt"
    face_metadata_path = run_dir / f"face_metadata_{clip_id}.json"
    tracking_metadata_path = run_dir / f"tracking_metadata_{clip_id}.json"
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    run_dir.mkdir(parents=True, exist_ok=True)

    if skip_existing and output_video.exists() and face_metadata_path.exists() and tracking_metadata_path.exists():
        return {
            "clip_id": clip_id,
            "condition": condition_name,
            "status": "skipped_existing",
            "returncode": 0,
            "elapsed_sec": 0.0,
            "clip_path": row["clip_path"],
            "target_image_path": row["target_image_path"],
            "output_video": str(output_video),
            "log_path": str(log_path),
            "face_metadata": str(face_metadata_path),
            "tracking_metadata": str(tracking_metadata_path),
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
        }

    for old_path in (output_video, log_path, face_metadata_path, tracking_metadata_path, stdout_path, stderr_path):
        if old_path.exists():
            old_path.unlink()

    code = main_hybrid_launcher(
        clip_path=Path(row["clip_path"]).expanduser().resolve(),
        target_image_path=Path(row["target_image_path"]).expanduser().resolve(),
        output_video=output_video.resolve(),
        log_path=log_path.resolve(),
        face_metadata_path=face_metadata_path.resolve(),
        tracking_metadata_path=tracking_metadata_path.resolve(),
        condition=condition,
    )
    command = [python_executable, "-c", code]
    started = time.perf_counter()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(
            command,
            cwd=str(veil_dir),
            stdout=stdout,
            stderr=stderr,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    elapsed = time.perf_counter() - started
    outputs_exist = output_video.exists() and face_metadata_path.exists() and tracking_metadata_path.exists()
    status = "completed" if completed.returncode == 0 and outputs_exist else "failed"
    return {
        "clip_id": clip_id,
        "condition": condition_name,
        "status": status,
        "returncode": completed.returncode,
        "elapsed_sec": round(elapsed, 3),
        "clip_path": row["clip_path"],
        "target_image_path": row["target_image_path"],
        "output_video": str(output_video),
        "log_path": str(log_path),
        "face_metadata": str(face_metadata_path),
        "tracking_metadata": str(tracking_metadata_path),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }


def aggregate_condition_metrics(output_dir: Path, condition_names: Sequence[str]) -> list[dict]:
    rows = []
    for condition_name in condition_names:
        aggregate_path = output_dir / condition_name / "metadata_eval" / "aggregate_metrics.json"
        if not aggregate_path.exists():
            continue
        payload = json.loads(aggregate_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "condition": condition_name,
                "clips": payload.get("clips", 0),
                "accepted_clips": payload.get("accepted_clips", 0),
                "protected_alteration_rate": payload.get("protected_alteration_rate", 0.0),
                "non_target_exposure_rate": payload.get("non_target_exposure_rate", 0.0),
                "anonymization_coverage": payload.get("anonymization_coverage", 0.0),
                "swap_coverage": payload.get("swap_coverage", 0.0),
                "blur_coverage": payload.get("blur_coverage", 0.0),
                "unknown_rate": payload.get("unknown_rate", 0.0),
                "mean_target_coverage": payload.get("mean_target_coverage", 0.0),
            }
        )
    return rows


def run_benchmark(args: argparse.Namespace) -> dict:
    manifest_path = args.manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    veil_dir = args.veil_dir.expanduser().resolve()
    condition_names = validate_condition_names(args.conditions)
    rows = select_manifest_rows(
        manifest_path,
        review_csv=args.review_csv,
        accepted_only=args.accepted_only,
        clip_ids=args.clip_id,
        clip_limit=args.clip_limit,
    )
    if not rows:
        raise BenchmarkError("No manifest rows selected")

    if args.dry_run:
        summary = {
            "dry_run": True,
            "conditions": condition_names,
            "selected_clips": [row["clip_id"] for row in rows],
            "output_dir": str(output_dir),
        }
        write_json(output_dir / "benchmark_plan.json", summary)
        return summary

    all_runs = []
    for condition_name in condition_names:
        condition = CONDITIONS[condition_name]
        condition_dir = output_dir / condition_name
        write_json(condition_dir / "condition_config.json", {"name": condition_name, **condition})
        condition_runs = []
        for row in rows:
            print(f"[{condition_name}] {row['clip_id']}", flush=True)
            try:
                run_result = run_one_clip(
                    row,
                    condition_name=condition_name,
                    condition=condition,
                    condition_dir=condition_dir,
                    python_executable=args.python,
                    veil_dir=veil_dir,
                    timeout_sec=args.timeout_sec,
                    skip_existing=args.skip_existing,
                )
            except subprocess.TimeoutExpired as exc:
                run_result = {
                    "clip_id": row["clip_id"],
                    "condition": condition_name,
                    "status": "timeout",
                    "returncode": None,
                    "elapsed_sec": args.timeout_sec,
                    "clip_path": row["clip_path"],
                    "target_image_path": row["target_image_path"],
                    "output_video": "",
                    "log_path": "",
                    "face_metadata": "",
                    "tracking_metadata": "",
                    "stdout": "",
                    "stderr": str(exc),
                }
            condition_runs.append(run_result)
            all_runs.append(run_result)
            write_json(condition_dir / "run_summary.json", condition_runs)

        completed = [run for run in condition_runs if run["status"] in ("completed", "skipped_existing")]
        if completed:
            evaluate_metadata(
                condition_dir / "runs",
                condition_dir / "metadata_eval",
                args.review_csv,
                blur_fallback_enabled=condition["blur_fallback_enabled"],
            )

    write_csv(output_dir / "run_summary.csv", all_runs)
    write_json(output_dir / "run_summary.json", all_runs)
    aggregate_rows = aggregate_condition_metrics(output_dir, condition_names)
    write_csv(output_dir / "condition_metrics.csv", aggregate_rows)
    summary = {
        "dry_run": False,
        "conditions": condition_names,
        "selected_clips": [row["clip_id"] for row in rows],
        "total_runs": len(all_runs),
        "completed_runs": sum(1 for row in all_runs if row["status"] in ("completed", "skipped_existing")),
        "failed_runs": sum(1 for row in all_runs if row["status"] not in ("completed", "skipped_existing")),
        "output_dir": str(output_dir),
        "condition_metrics_csv": str(output_dir / "condition_metrics.csv"),
    }
    write_json(output_dir / "benchmark_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run VEIL benchmark conditions over a materialized manifest.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--review-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--veil-dir", type=Path, default=Path("models/veil"))
    parser.add_argument("--conditions", nargs="+", default=["full", "no_identity_lock", "no_blur_fallback"])
    parser.add_argument("--accepted-only", action="store_true")
    parser.add_argument("--clip-id", action="append", default=[])
    parser.add_argument("--clip-limit", type=int)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--timeout-sec", type=int, default=1800)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        print(json.dumps(run_benchmark(args), ensure_ascii=False))
    except BenchmarkError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
