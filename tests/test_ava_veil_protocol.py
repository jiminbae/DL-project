import csv
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from tools.ava_veil_protocol import (
    assign_split,
    iter_annotations,
    load_annotations,
    parse_split_ratios,
    select_candidates,
    write_protocol_outputs,
)


def write_synthetic_csv(path: Path, *, include_header: bool = False) -> None:
    rows = []
    for frame in range(20):
        timestamp = frame * 0.5
        rows.append(["videoA", timestamp, 0.10, 0.10, 0.30, 0.40, "NOT_SPEAKING", "person_big"])
        rows.append(["videoA", timestamp, 0.55, 0.15, 0.68, 0.36, "SPEAKING_AND_AUDIBLE", "person_small"])
    # Sparse third face should not pass the default coverage threshold.
    for frame in range(3):
        timestamp = frame * 0.5
        rows.append(["videoA", timestamp, 0.75, 0.20, 0.83, 0.32, "NOT_SPEAKING", "person_sparse"])

    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        if include_header:
            writer.writerow(["video_id", "frame_timestamp", "x1", "y1", "x2", "y2", "label", "entity_id"])
        writer.writerows(rows)


class AvaVeilProtocolTest(unittest.TestCase):
    def test_header_and_tar_input(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            csv_path = root / "sample.csv"
            write_synthetic_csv(csv_path, include_header=True)
            archive_path = root / "annotations.tar.bz2"
            with tarfile.open(archive_path, "w:bz2") as archive:
                archive.add(csv_path, arcname="train/sample.csv")

            rows = list(iter_annotations(archive_path))
            self.assertEqual(len(rows), 43)
            self.assertEqual(rows[0].video_id, "videoA")

    def test_selection_and_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            csv_path = root / "sample.csv"
            output_dir = root / "output"
            write_synthetic_csv(csv_path)
            grouped = load_annotations(csv_path)
            splits = parse_split_ratios("train=80,val=10,test=10")
            selected = select_candidates(
                grouped,
                window_seconds=5.0,
                step_seconds=2.5,
                min_faces=2,
                min_overlap_ratio=0.8,
                min_entity_coverage=0.8,
                min_normalized_face_size=0.05,
                max_clips_per_video=2,
                max_selected_overlap=0.25,
                seed=2026,
                split_ratios=splits,
            )
            self.assertTrue(selected)
            self.assertLessEqual(len(selected), 2)
            self.assertEqual(selected[0].protected_entity_id, "person_big")
            self.assertEqual(selected[0].num_entities, 2)

            write_protocol_outputs(output_dir, selected, grouped, {"test": True})
            manifest_lines = (output_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(manifest_lines), len(selected))
            first = json.loads(manifest_lines[0])
            self.assertEqual(first["protected_entity_id"], "person_big")
            self.assertTrue((output_dir / "selected_annotations.csv").exists())
            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["num_clips"], len(selected))

    def test_split_is_video_level_and_deterministic(self):
        splits = parse_split_ratios("train=70,val=15,test=15")
        first = assign_split("videoA", 1234, splits)
        second = assign_split("videoA", 1234, splits)
        self.assertEqual(first, second)
        self.assertIn(first, {"train", "val", "test"})


if __name__ == "__main__":
    unittest.main()
