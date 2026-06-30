#!/usr/bin/env python3
"""Copy lightweight benchmark results into a sanitized, git-trackable package."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence


class PackageError(RuntimeError):
    """Raised when benchmark result packaging fails."""


def sanitize_value(value):
    if isinstance(value, dict):
        return {key: sanitize_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_value(item) for item in value]
    if isinstance(value, str):
        if value.startswith("/"):
            return Path(value).name
        return value
    return value


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitize_value(payload), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def copy_text(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


def sanitize_csv(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        dst.write_text("", encoding="utf-8")
        return
    with dst.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: sanitize_value(value) for key, value in row.items()})


def package_results(benchmark_dir: Path, output_dir: Path, conditions: Sequence[str]) -> dict:
    benchmark_dir = benchmark_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not benchmark_dir.exists():
        raise PackageError(f"Benchmark directory does not exist: {benchmark_dir}")

    required = ["benchmark_summary.json", "condition_metrics.csv", "run_summary.csv", "run_summary.json"]
    for name in required:
        if not (benchmark_dir / name).exists():
            raise PackageError(f"Missing benchmark output: {benchmark_dir / name}")

    write_json(output_dir / "benchmark_summary.json", read_json(benchmark_dir / "benchmark_summary.json"))
    sanitize_csv(benchmark_dir / "condition_metrics.csv", output_dir / "condition_metrics.csv")
    sanitize_csv(benchmark_dir / "run_summary.csv", output_dir / "run_summary.csv")
    write_json(output_dir / "run_summary.json", read_json(benchmark_dir / "run_summary.json"))

    for condition in conditions:
        condition_dir = benchmark_dir / condition
        if not condition_dir.exists():
            continue
        write_json(output_dir / condition / "condition_config.json", read_json(condition_dir / "condition_config.json"))
        metadata_eval = condition_dir / "metadata_eval"
        for name in ("aggregate_metrics.json", "evaluation_config.json", "per_clip_metrics.csv"):
            src = metadata_eval / name
            if not src.exists():
                continue
            dst = output_dir / condition / name
            if src.suffix == ".csv":
                sanitize_csv(src, dst)
            else:
                write_json(dst, read_json(src))

    for name in (
        "analysis_summary.json",
        "analysis_summary.md",
        "manual_output_review.csv",
        "manual_output_review_summary.json",
        "manual_output_review_summary.md",
    ):
        src = benchmark_dir / name
        if not src.exists():
            continue
        dst = output_dir / name
        if src.suffix == ".csv":
            sanitize_csv(src, dst)
        elif src.suffix == ".json":
            write_json(dst, read_json(src))
        else:
            copy_text(src, dst)

    label = "Panoptic-20" if "20" in output_dir.name else "Panoptic-10"
    readme = f"""# {label} VEIL Ablation Results

This folder contains lightweight, sanitized result files for the VEIL accepted-subset ablation run.

Full videos and large per-frame JSONL files are intentionally excluded. Absolute local paths are reduced to file or directory names where possible.

Primary table: `condition_metrics.csv`.
"""
    if (output_dir / "manual_output_review_summary.md").exists():
        readme += "\nManual visual review is summarized in `manual_output_review_summary.md`.\n"
    if (output_dir / "analysis_summary.md").exists():
        readme += "\nBenchmark interpretation is summarized in `analysis_summary.md`.\n"
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    return {"output_dir": str(output_dir), "conditions": list(conditions)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Package corrected VEIL benchmark results for git/paper use.")
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--conditions", nargs="+", default=["full", "no_identity_lock", "no_blur_fallback", "selective_blur_only"])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        print(json.dumps(package_results(args.benchmark_dir, args.output_dir, args.conditions), ensure_ascii=False))
    except PackageError as exc:
        print(f"error: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
