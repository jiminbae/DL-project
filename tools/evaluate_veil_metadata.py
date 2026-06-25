#!/usr/bin/env python3
"""Evaluate VEIL face metadata without assuming unavailable action fields."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence


STATES = ("PRESERVE", "SWAP", "BLUR", "UNPROCESSED", "FAILED", "UNKNOWN")
TRACK_LOG_RE = re.compile(
    r"Frame=(?P<frame>\d+).*?TrackID=(?P<track>\d+).*?SwapSuccess=(?P<swap>True|False)"
)


class EvaluationError(RuntimeError):
    """Raised when VEIL metadata cannot be evaluated."""


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvaluationError(f"Invalid JSON: {path}") from exc


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def find_run_dirs(runs_dir: Path) -> list[Path]:
    runs_dir = runs_dir.expanduser().resolve()
    if not runs_dir.exists():
        raise EvaluationError(f"Runs directory does not exist: {runs_dir}")
    run_dirs = sorted(path for path in runs_dir.iterdir() if path.is_dir())
    if not run_dirs:
        raise EvaluationError(f"No run directories found under: {runs_dir}")
    return run_dirs


def find_one(path: Path, pattern: str) -> Path | None:
    matches = sorted(path.glob(pattern))
    return matches[0] if matches else None


def load_review(path: Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    path = path.expanduser().resolve()
    if not path.exists():
        raise EvaluationError(f"Review CSV does not exist: {path}")
    with path.open("r", encoding="utf-8", newline="") as stream:
        return {row["clip_id"]: row for row in csv.DictReader(stream)}


def schema_report(face_paths: Sequence[Path], tracking_paths: Sequence[Path]) -> dict:
    file_reports: list[dict] = []
    all_fields: list[set[str]] = []
    examples: dict[str, object] = {}

    for path in face_paths:
        rows = read_json(path)
        if not isinstance(rows, list):
            raise EvaluationError(f"Expected list metadata: {path}")
        fields = set()
        for row in rows:
            fields.update(row)
            for key, value in row.items():
                examples.setdefault(key, value)
        all_fields.append(fields)
        file_reports.append({"path": str(path), "rows": len(rows), "fields": sorted(fields)})

    common_fields = sorted(set.intersection(*all_fields)) if all_fields else []
    union_fields = sorted(set.union(*all_fields)) if all_fields else []
    optional_fields = sorted(set(union_fields) - set(common_fields))

    tracking_reports: list[dict] = []
    for path in tracking_paths:
        rows = read_json(path)
        fields = sorted({key for row in rows for key in row}) if isinstance(rows, list) else []
        tracking_reports.append({"path": str(path), "rows": len(rows) if isinstance(rows, list) else None, "fields": fields})

    return {
        "face_metadata_files": len(face_paths),
        "tracking_metadata_files": len(tracking_paths),
        "common_fields": common_fields,
        "optional_fields": optional_fields,
        "file_reports": file_reports,
        "tracking_file_reports": tracking_reports,
        "example_values": {key: examples[key] for key in sorted(examples)},
        "action_field_present": any(field in union_fields for field in ("action", "processing_action", "final_action")),
        "swap_success_field_present": "swap_success" in union_fields,
    }


def parse_track_swap_log(log_path: Path | None) -> dict[tuple[int, int], bool]:
    if log_path is None or not log_path.exists():
        return {}
    result: dict[tuple[int, int], bool] = {}
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = TRACK_LOG_RE.search(line)
        if not match:
            continue
        result[(int(match.group("frame")), int(match.group("track")))] = match.group("swap") == "True"
    return result


def dedup_key(row: dict) -> tuple[int, str]:
    frame_idx = int(row.get("frame_idx", row.get("frame_number", -1)))
    stable_face_id = row.get("stable_face_id")
    if stable_face_id is not None:
        return frame_idx, f"face:{stable_face_id}"
    return frame_idx, f"track:{row.get('raw_track_id', row.get('track_id'))}"


def deduplicate_rows(rows: Sequence[dict]) -> list[dict]:
    by_key: dict[tuple[int, str], dict] = {}
    for row in rows:
        key = dedup_key(row)
        previous = by_key.get(key)
        if previous is None:
            by_key[key] = row
            continue
        # Prefer rows with explicit target/background labels and better quality.
        score = (
            int(bool(row.get("is_target"))),
            int(bool(row.get("is_background"))),
            int(row.get("quality") == "GOOD"),
        )
        previous_score = (
            int(bool(previous.get("is_target"))),
            int(bool(previous.get("is_background"))),
            int(previous.get("quality") == "GOOD"),
        )
        if score > previous_score:
            by_key[key] = row
    return [by_key[key] for key in sorted(by_key)]


def classify_row(row: dict, logged_swaps: dict[tuple[int, int], bool], *, blur_fallback_enabled: bool = True) -> tuple[str, str]:
    final_action = row.get("final_action")
    if isinstance(final_action, str):
        normalized = final_action.upper()
        if normalized in STATES:
            return normalized, "metadata_final_action"

    if row.get("is_target") is True:
        return "PRESERVE", "metadata_is_target"

    if row.get("is_background") is True:
        frame_idx = int(row.get("frame_idx", -1))
        raw_track_id = int(row.get("raw_track_id", row.get("track_id", -1)))
        logged = logged_swaps.get((frame_idx, raw_track_id))
        if logged is True:
            return "SWAP", "track_log_swap_success"
        if logged is False:
            if blur_fallback_enabled:
                return "BLUR", "track_log_swap_failed_fallback_blur"
            return "UNPROCESSED", "track_log_swap_failed_no_blur_fallback"

        fallback_reasons = row.get("fallback_reasons") or []
        if row.get("embedding_ok") is False:
            if blur_fallback_enabled:
                return "BLUR", "embedding_failed_fallback_blur"
            return "UNPROCESSED", "embedding_failed_no_blur_fallback"
        if row.get("quality") != "GOOD" or fallback_reasons:
            if blur_fallback_enabled:
                return "BLUR", "quality_or_fallback_blur"
            return "UNPROCESSED", "quality_or_fallback_no_blur_fallback"
        return "UNKNOWN", "background_good_but_swap_not_logged"

    return "UNKNOWN", "not_target_or_background"


def state_switches(actions: Sequence[dict], *, target_only: bool | None) -> int:
    by_identity: dict[str, list[dict]] = defaultdict(list)
    for action in actions:
        if target_only is True and not action["is_target"]:
            continue
        if target_only is False and action["is_target"]:
            continue
        identity = action["dedup_identity"]
        by_identity[identity].append(action)

    switches = 0
    for rows in by_identity.values():
        previous = None
        for row in sorted(rows, key=lambda item: item["frame_idx"]):
            state = row["state"]
            if previous is not None and state != previous:
                switches += 1
            previous = state
    return switches


def safe_div(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def evaluate_clip(clip_id: str, run_dir: Path, review: dict | None = None, *, blur_fallback_enabled: bool = True) -> tuple[dict, list[dict], list[dict]]:
    face_path = find_one(run_dir, "face_metadata*.json")
    if face_path is None:
        raise EvaluationError(f"No face_metadata*.json under: {run_dir}")
    rows = read_json(face_path)
    if not isinstance(rows, list):
        raise EvaluationError(f"Expected list metadata: {face_path}")

    log_path = find_one(run_dir, "tracking_target*_log.txt")
    logged_swaps = parse_track_swap_log(log_path)
    deduped = deduplicate_rows(rows)
    max_frame = max((int(row.get("frame_idx", 0)) for row in deduped), default=0)
    actions: list[dict] = []
    for row in deduped:
        state, reason = classify_row(row, logged_swaps, blur_fallback_enabled=blur_fallback_enabled)
        frame_idx, identity = dedup_key(row)
        action = {
            "clip_id": clip_id,
            "frame_idx": frame_idx,
            "raw_track_id": row.get("raw_track_id"),
            "stable_face_id": row.get("stable_face_id"),
            "dedup_identity": identity,
            "is_target": bool(row.get("is_target_final", row.get("is_target"))),
            "is_target_direct": bool(row.get("is_target_direct", row.get("is_target"))),
            "is_target_final": bool(row.get("is_target_final", row.get("is_target"))),
            "is_background": bool(row.get("is_background")),
            "state": state,
            "state_reason": reason,
            "final_action": row.get("final_action"),
            "swap_success": row.get("swap_success"),
            "blur_applied": row.get("blur_applied"),
            "quality": row.get("quality"),
            "fallback_reasons": row.get("fallback_reasons") or [],
            "embedding_ok": row.get("embedding_ok"),
            "target_similarity": row.get("target_similarity"),
            "bbox": row.get("bbox"),
        }
        actions.append(action)

    state_counts = Counter(action["state"] for action in actions)
    protected = [action for action in actions if action["is_target"]]
    non_target = [action for action in actions if action["is_background"]]
    protected_counts = Counter(action["state"] for action in protected)
    non_target_counts = Counter(action["state"] for action in non_target)
    unknown_rows = state_counts["UNKNOWN"]
    protected_denominator = len(protected)
    non_target_denominator = len(non_target)
    protected_frames = {action["frame_idx"] for action in protected}

    metrics = {
        "clip_id": clip_id,
        "accepted": (review or {}).get("accepted", ""),
        "total_frames": max_frame,
        "metadata_rows": len(rows),
        "deduped_face_rows": len(deduped),
        "duplicate_rows_removed": len(rows) - len(deduped),
        "frames_with_any_face": len({action["frame_idx"] for action in actions}),
        "protected_face_rows": len(protected),
        "non_target_face_rows": len(non_target),
        "protected_state_switches": state_switches(actions, target_only=True),
        "non_target_state_switches": state_switches(actions, target_only=False),
        "stable_id_count": len({action["stable_face_id"] for action in actions if action["stable_face_id"] is not None}),
        "raw_track_id_count": len({action["raw_track_id"] for action in actions if action["raw_track_id"] is not None}),
        "target_coverage": round(safe_div(len(protected_frames), max_frame), 6),
        "target_frame_count": len(protected_frames),
        "protected_alteration_rate": round(
            safe_div(protected_counts["SWAP"] + protected_counts["BLUR"] + protected_counts["FAILED"], protected_denominator),
            6,
        ),
        "non_target_exposure_rate": round(
            safe_div(non_target_counts["PRESERVE"] + non_target_counts["UNPROCESSED"], non_target_denominator),
            6,
        ),
        "anonymization_coverage": round(
            safe_div(non_target_counts["SWAP"] + non_target_counts["BLUR"], non_target_denominator),
            6,
        ),
        "swap_coverage": round(safe_div(non_target_counts["SWAP"], non_target_denominator), 6),
        "blur_coverage": round(safe_div(non_target_counts["BLUR"], non_target_denominator), 6),
        "non_target_unprocessed_rate": round(safe_div(non_target_counts["UNPROCESSED"], non_target_denominator), 6),
        "non_target_unknown_rate": round(safe_div(non_target_counts["UNKNOWN"], non_target_denominator), 6),
        "protected_unknown_rate": round(safe_div(protected_counts["UNKNOWN"], protected_denominator), 6),
        "unknown_rate": round(safe_div(unknown_rows, len(actions)), 6),
        "log_path": str(log_path) if log_path else "",
        "face_metadata_path": str(face_path),
    }
    for prefix, counts in (("protected", protected_counts), ("non_target", non_target_counts)):
        for state in STATES:
            metrics[f"{prefix}_{state.lower()}_rows"] = counts[state]
    for state in STATES:
        metrics[f"{state.lower()}_rows"] = state_counts[state]

    identity_rows: list[dict] = []
    by_identity: dict[str, list[dict]] = defaultdict(list)
    for action in actions:
        by_identity[action["dedup_identity"]].append(action)
    for identity, group in sorted(by_identity.items()):
        counts = Counter(action["state"] for action in group)
        identity_rows.append(
            {
                "clip_id": clip_id,
                "dedup_identity": identity,
                "stable_face_id": group[0].get("stable_face_id"),
                "raw_track_ids": json.dumps(sorted({action["raw_track_id"] for action in group}), ensure_ascii=False),
                "is_target_any": any(action["is_target"] for action in group),
                "rows": len(group),
                **{f"{state.lower()}_rows": counts[state] for state in STATES},
            }
        )

    return metrics, identity_rows, actions


def aggregate(per_clip: Sequence[dict]) -> dict:
    accepted = [row for row in per_clip if row.get("accepted") == "yes"]
    source = accepted or list(per_clip)
    totals = {f"{state.lower()}_rows": sum(int(row[f"{state.lower()}_rows"]) for row in source) for state in STATES}
    non_target_rows = sum(int(row["non_target_face_rows"]) for row in source)
    protected_rows = sum(int(row["protected_face_rows"]) for row in source)
    return {
        "clips": len(per_clip),
        "accepted_clips": len(accepted),
        "aggregate_source": "accepted" if accepted else "all",
        "protected_face_rows": protected_rows,
        "non_target_face_rows": non_target_rows,
        "states": totals,
        "protected_alteration_rate": round(
            safe_div(
                sum(int(row["protected_swap_rows"]) + int(row["protected_blur_rows"]) + int(row["protected_failed_rows"]) for row in source),
                protected_rows,
            ),
            6,
        ),
        "non_target_exposure_rate": round(
            safe_div(sum(int(row["non_target_preserve_rows"]) + int(row["non_target_unprocessed_rows"]) for row in source), non_target_rows),
            6,
        ),
        "anonymization_coverage": round(
            safe_div(sum(int(row["non_target_swap_rows"]) + int(row["non_target_blur_rows"]) for row in source), non_target_rows),
            6,
        ),
        "swap_coverage": round(safe_div(sum(int(row["non_target_swap_rows"]) for row in source), non_target_rows), 6),
        "blur_coverage": round(safe_div(sum(int(row["non_target_blur_rows"]) for row in source), non_target_rows), 6),
        "non_target_unprocessed_rate": round(safe_div(sum(int(row["non_target_unprocessed_rows"]) for row in source), non_target_rows), 6),
        "non_target_unknown_rate": round(safe_div(sum(int(row["non_target_unknown_rows"]) for row in source), non_target_rows), 6),
        "unknown_rate": round(
            safe_div(sum(int(row["unknown_rows"]) for row in source), sum(int(row["deduped_face_rows"]) for row in source)),
            6,
        ),
        "mean_target_coverage": round(statistics.mean(float(row["target_coverage"]) for row in source), 6) if source else 0.0,
    }


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(
    runs_dir: Path,
    output_dir: Path,
    review_csv: Path | None = None,
    *,
    blur_fallback_enabled: bool = True,
    clip_ids: Sequence[str] | None = None,
) -> dict:
    run_dirs = find_run_dirs(runs_dir)
    if clip_ids is not None:
        wanted = set(clip_ids)
        run_dirs = [run_dir for run_dir in run_dirs if run_dir.name in wanted]
        if not run_dirs:
            raise EvaluationError("No selected run directories found")
    review_rows = load_review(review_csv)
    face_paths = [path for run_dir in run_dirs if (path := find_one(run_dir, "face_metadata*.json")) is not None]
    tracking_paths = [path for run_dir in run_dirs if (path := find_one(run_dir, "tracking_metadata*.json")) is not None]

    schema = schema_report(face_paths, tracking_paths)
    per_clip: list[dict] = []
    per_identity: list[dict] = []
    per_frame_actions: list[dict] = []
    for run_dir in run_dirs:
        clip_id = run_dir.name
        clip_metrics, identity_rows, actions = evaluate_clip(clip_id, run_dir, review_rows.get(clip_id), blur_fallback_enabled=blur_fallback_enabled)
        per_clip.append(clip_metrics)
        per_identity.extend(identity_rows)
        per_frame_actions.extend(actions)

    output_dir = output_dir.expanduser().resolve()
    write_csv(output_dir / "per_clip_metrics.csv", per_clip)
    write_csv(output_dir / "per_identity_metrics.csv", per_identity)
    write_jsonl(output_dir / "per_frame_actions.jsonl", per_frame_actions)
    write_json(output_dir / "aggregate_metrics.json", aggregate(per_clip))
    write_json(output_dir / "schema_report.json", schema)
    config = {
        "runs_dir": str(runs_dir.expanduser().resolve()),
        "review_csv": str(review_csv.expanduser().resolve()) if review_csv else None,
        "classification_policy": {
            "target": "is_target=True -> PRESERVE",
            "swap": "background row with exact frame/track log SwapSuccess=True -> SWAP",
            "blur": "background row with exact log SwapSuccess=False, embedding failure, non-GOOD quality, or fallback reasons -> BLUR when fallback blur is enabled",
            "unprocessed": "the same rows -> UNPROCESSED when fallback blur is disabled",
            "unknown": "background GOOD rows without per-track swap log evidence -> UNKNOWN",
        },
        "deduplication_key": "(frame_idx, stable_face_id) else (frame_idx, raw_track_id)",
        "target_coverage": "unique protected frames / total frames",
        "blur_fallback_enabled": blur_fallback_enabled,
        "selected_clip_ids": sorted(clip_ids) if clip_ids is not None else None,
        "notes": [
            "Current VEIL face metadata does not contain a final action or swap_success field.",
            "SWAP is therefore under-counted to logged frame/track evidence; UNKNOWN is intentionally conservative.",
        ],
    }
    write_json(output_dir / "evaluation_config.json", config)
    return {
        "output_dir": str(output_dir),
        "clips": len(per_clip),
        "aggregate_metrics": str(output_dir / "aggregate_metrics.json"),
        "schema_report": str(output_dir / "schema_report.json"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate VEIL metadata into conservative action-state metrics.")
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--review-csv", type=Path)
    parser.add_argument("--clip-id", action="append", default=None, help="Evaluate only this run directory name; may be repeated.")
    parser.add_argument(
        "--blur-fallback-enabled",
        choices=("true", "false"),
        default="true",
        help="Set false for no-blur-fallback ablations so failed/low-quality background rows count as UNPROCESSED.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        blur_fallback_enabled = args.blur_fallback_enabled == "true"
        print(json.dumps(
            evaluate(
                args.runs_dir,
                args.output_dir,
                args.review_csv,
                blur_fallback_enabled=blur_fallback_enabled,
                clip_ids=args.clip_id,
            ),
            ensure_ascii=False,
        ))
    except EvaluationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
