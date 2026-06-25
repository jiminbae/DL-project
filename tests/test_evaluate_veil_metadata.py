import json
import tempfile
import unittest
from pathlib import Path

from tools.evaluate_veil_metadata import (
    aggregate,
    classify_row,
    deduplicate_rows,
    evaluate,
    parse_track_swap_log,
)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class EvaluateVeilMetadataTest(unittest.TestCase):
    def test_classification_rules_are_conservative(self):
        logged = {(1, 2): True, (1, 3): False}
        self.assertEqual(
            classify_row({"is_target": True, "is_background": False}, logged)[0],
            "PRESERVE",
        )
        self.assertEqual(
            classify_row(
                {
                    "frame_idx": 1,
                    "raw_track_id": 2,
                    "is_background": True,
                    "quality": "GOOD",
                    "fallback_reasons": [],
                },
                logged,
            )[0],
            "SWAP",
        )
        self.assertEqual(
            classify_row(
                {
                    "frame_idx": 1,
                    "raw_track_id": 3,
                    "is_background": True,
                    "quality": "GOOD",
                    "fallback_reasons": [],
                },
                logged,
            )[0],
            "BLUR",
        )
        self.assertEqual(
            classify_row(
                {
                    "frame_idx": 2,
                    "raw_track_id": 4,
                    "is_background": True,
                    "quality": "GOOD",
                    "fallback_reasons": [],
                    "embedding_ok": True,
                },
                logged,
            )[0],
            "UNKNOWN",
        )
        self.assertEqual(
            classify_row(
                {
                    "frame_idx": 2,
                    "raw_track_id": 5,
                    "is_background": True,
                    "quality": "BAD",
                    "fallback_reasons": ["small_face_size"],
                    "embedding_ok": True,
                },
                logged,
            )[0],
            "BLUR",
        )

    def test_final_action_overrides_conservative_inference(self):
        self.assertEqual(
            classify_row(
                {
                    "frame_idx": 1,
                    "raw_track_id": 2,
                    "is_background": True,
                    "quality": "BAD",
                    "fallback_reasons": ["small_face_size"],
                    "final_action": "SWAP",
                },
                {},
            ),
            ("SWAP", "metadata_final_action"),
        )

    def test_no_blur_fallback_counts_failed_background_as_unprocessed(self):
        logged = {(1, 3): False}
        self.assertEqual(
            classify_row(
                {
                    "frame_idx": 1,
                    "raw_track_id": 3,
                    "is_background": True,
                    "quality": "GOOD",
                    "fallback_reasons": [],
                },
                logged,
                blur_fallback_enabled=False,
            )[0],
            "UNPROCESSED",
        )
        self.assertEqual(
            classify_row(
                {
                    "frame_idx": 2,
                    "raw_track_id": 5,
                    "is_background": True,
                    "quality": "BAD",
                    "fallback_reasons": ["small_face_size"],
                    "embedding_ok": True,
                },
                {},
                blur_fallback_enabled=False,
            )[0],
            "UNPROCESSED",
        )

    def test_deduplicates_by_frame_and_stable_face(self):
        rows = [
            {"frame_idx": 1, "raw_track_id": 7, "stable_face_id": 1, "quality": "BAD", "is_target": False},
            {"frame_idx": 1, "raw_track_id": 8, "stable_face_id": 1, "quality": "GOOD", "is_target": True},
            {"frame_idx": 1, "raw_track_id": 9, "stable_face_id": None, "quality": "GOOD", "is_target": False},
        ]
        deduped = deduplicate_rows(rows)
        self.assertEqual(len(deduped), 2)
        self.assertTrue(any(row.get("is_target") for row in deduped))

    def test_parse_track_swap_log(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "tracking_target1_log.txt"
            log_path.write_text(
                "Frame=10 TrackID=2 FaceID=1 Background=True SwapSuccess=True Quality=GOOD\n"
                "Frame=20 TrackID=3 FaceID=2 Background=True SwapSuccess=False Quality=BAD\n",
                encoding="utf-8",
            )
            self.assertEqual(parse_track_swap_log(log_path), {(10, 2): True, (20, 3): False})

    def test_evaluate_writes_metrics_and_schema(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run = root / "runs" / "clip_a"
            rows = [
                {
                    "frame_idx": 1,
                    "raw_track_id": 1,
                    "stable_face_id": 1,
                    "bbox": [0, 0, 10, 10],
                    "smoothed_bbox": [0, 0, 10, 10],
                    "is_target": True,
                    "is_background": False,
                    "target_similarity": 0.8,
                    "embedding_ok": True,
                    "quality": "GOOD",
                    "fallback_reasons": [],
                    "crop_path": None,
                },
                {
                    "frame_idx": 1,
                    "raw_track_id": 2,
                    "stable_face_id": 2,
                    "bbox": [20, 0, 40, 20],
                    "smoothed_bbox": [20, 0, 40, 20],
                    "is_target": False,
                    "is_background": True,
                    "target_similarity": 0.1,
                    "embedding_ok": True,
                    "quality": "GOOD",
                    "fallback_reasons": [],
                    "crop_path": None,
                },
                {
                    "frame_idx": 2,
                    "raw_track_id": 2,
                    "stable_face_id": 2,
                    "bbox": [21, 0, 41, 20],
                    "smoothed_bbox": [21, 0, 41, 20],
                    "is_target": False,
                    "is_background": True,
                    "target_similarity": 0.1,
                    "embedding_ok": True,
                    "quality": "BAD",
                    "fallback_reasons": ["small_face_size"],
                    "crop_path": None,
                },
                {
                    "frame_idx": 2,
                    "raw_track_id": 3,
                    "stable_face_id": 3,
                    "bbox": [50, 0, 70, 20],
                    "smoothed_bbox": [50, 0, 70, 20],
                    "is_target": True,
                    "is_background": False,
                    "target_similarity": 0.8,
                    "embedding_ok": True,
                    "quality": "GOOD",
                    "fallback_reasons": [],
                    "crop_path": None,
                    "final_action": "PRESERVE",
                },
            ]
            write_json(run / "face_metadata1.json", rows)
            write_json(run / "tracking_metadata1.json", [{"frame_idx": 1, "raw_track_id": 1, "bbox": [0, 0, 10, 10]}])
            (run / "tracking_target1_log.txt").write_text(
                "Frame=1 TrackID=2 FaceID=2 Background=True SwapSuccess=True Quality=GOOD\n",
                encoding="utf-8",
            )

            review = root / "review.csv"
            review.write_text("clip_id,accepted\nclip_a,yes\n", encoding="utf-8")
            summary = evaluate(root / "runs", root / "evaluation", review)
            self.assertEqual(summary["clips"], 1)

            aggregate_payload = json.loads((root / "evaluation" / "aggregate_metrics.json").read_text())
            self.assertEqual(aggregate_payload["accepted_clips"], 1)
            self.assertEqual(aggregate_payload["states"]["preserve_rows"], 2)
            self.assertEqual(aggregate_payload["states"]["swap_rows"], 1)
            self.assertEqual(aggregate_payload["states"]["blur_rows"], 1)
            self.assertEqual(aggregate_payload["mean_target_coverage"], 1.0)

            per_clip = (root / "evaluation" / "per_clip_metrics.csv").read_text(encoding="utf-8")
            self.assertIn("clip_a", per_clip)
            self.assertIn("target_frame_count", per_clip)
            self.assertIn("non_target_unknown_rate", per_clip)
            schema = json.loads((root / "evaluation" / "schema_report.json").read_text())
            self.assertTrue(schema["action_field_present"])
            self.assertFalse(schema["swap_success_field_present"])

    def test_aggregate_handles_zero_denominators(self):
        payload = aggregate(
            [
                {
                    "accepted": "yes",
                    "protected_face_rows": 0,
                    "non_target_face_rows": 0,
                    "deduped_face_rows": 0,
                    "target_coverage": 0,
                    "target_frame_count": 0,
                    **{f"{state}_rows": 0 for state in ("preserve", "swap", "blur", "unprocessed", "failed", "unknown")},
                    **{
                        f"protected_{state}_rows": 0
                        for state in ("preserve", "swap", "blur", "unprocessed", "failed", "unknown")
                    },
                    **{
                        f"non_target_{state}_rows": 0
                        for state in ("preserve", "swap", "blur", "unprocessed", "failed", "unknown")
                    },
                }
            ]
        )
        self.assertEqual(payload["anonymization_coverage"], 0.0)
        self.assertEqual(payload["protected_alteration_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
