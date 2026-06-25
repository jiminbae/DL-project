import argparse
import csv
import json
import tempfile
import unittest
from pathlib import Path

from tools.run_veil_benchmark import (
    CONDITIONS,
    aggregate_condition_metrics,
    main_hybrid_launcher,
    run_benchmark,
    select_manifest_rows,
    validate_condition_names,
)


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class RunVeilBenchmarkTest(unittest.TestCase):
    def test_select_manifest_rows_filters_accepted_and_clip_limit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = root / "manifest.jsonl"
            write_jsonl(
                manifest,
                [
                    {"clip_id": "b", "status": "materialized", "clip_path": "b.mp4", "target_image_path": "b.jpg"},
                    {"clip_id": "a", "status": "materialized", "clip_path": "a.mp4", "target_image_path": "a.jpg"},
                    {"clip_id": "c", "status": "failed", "clip_path": "c.mp4", "target_image_path": "c.jpg"},
                ],
            )
            review = root / "review.csv"
            review.write_text("clip_id,accepted\na,yes\nb,no\n", encoding="utf-8")
            rows = select_manifest_rows(manifest, review_csv=review, accepted_only=True, clip_limit=1)
            self.assertEqual([row["clip_id"] for row in rows], ["a"])

    def test_validate_condition_names_rejects_unknown(self):
        self.assertEqual(validate_condition_names(["full"]), ["full"])
        with self.assertRaisesRegex(Exception, "Unknown condition"):
            validate_condition_names(["missing"])

    def test_launcher_injects_ablation_monkeypatches(self):
        code = main_hybrid_launcher(
            clip_path=Path("clip.mp4"),
            protected_target_image_path=Path("protected/target.jpg"),
            replacement_image_path=Path("virtual/fake_face.jpg"),
            output_video=Path("out.mp4"),
            log_path=Path("log.txt"),
            face_metadata_path=Path("face.json"),
            tracking_metadata_path=Path("tracking.json"),
            condition=CONDITIONS["no_blur_fallback"],
        )
        self.assertIn("config.VIDEO_PATH", code)
        self.assertIn('config.TARGET_DIR = payload["target_dir"]', code)
        self.assertIn('config.TARGET_IMAGE_PATH = payload["replacement_image_path"]', code)
        self.assertIn('config.ENABLE_FALLBACK_BLUR = payload["blur_fallback_enabled"]', code)
        self.assertIn("protected/target.jpg", code)
        self.assertIn("virtual/fake_face.jpg", code)
        self.assertNotIn("apply_fallback_blur = lambda", code)

    def test_dry_run_writes_plan(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = root / "manifest.jsonl"
            write_jsonl(
                manifest,
                [{"clip_id": "clip_a", "status": "materialized", "clip_path": "a.mp4", "target_image_path": "a.jpg"}],
            )
            replacement = root / "fake_face.jpg"
            replacement.write_bytes(b"fake")
            args = argparse.Namespace(
                manifest=manifest,
                review_csv=None,
                output_dir=root / "bench",
                veil_dir=Path("models/veil"),
                conditions=["full"],
                replacement_image=replacement,
                accepted_only=False,
                clip_id=[],
                clip_limit=None,
                python="python",
                timeout_sec=1,
                skip_existing=False,
                dry_run=True,
            )
            summary = run_benchmark(args)
            self.assertTrue(summary["dry_run"])
            self.assertEqual(summary["replacement_image"], str(replacement.resolve()))
            self.assertTrue((root / "bench" / "benchmark_plan.json").exists())

    def test_aggregate_condition_metrics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            aggregate = root / "full" / "metadata_eval" / "aggregate_metrics.json"
            aggregate.parent.mkdir(parents=True)
            aggregate.write_text(
                json.dumps(
                    {
                        "clips": 1,
                        "accepted_clips": 1,
                        "protected_alteration_rate": 0.0,
                        "non_target_exposure_rate": 0.2,
                        "anonymization_coverage": 0.8,
                        "non_target_unprocessed_rate": 0.3,
                        "non_target_unknown_rate": 0.1,
                    }
                ),
                encoding="utf-8",
            )
            rows = aggregate_condition_metrics(root, ["full"])
            self.assertEqual(rows[0]["condition"], "full")
            self.assertEqual(rows[0]["non_target_exposure_rate"], 0.2)
            self.assertEqual(rows[0]["non_target_unprocessed_rate"], 0.3)
            self.assertEqual(rows[0]["non_target_unknown_rate"], 0.1)


if __name__ == "__main__":
    unittest.main()
