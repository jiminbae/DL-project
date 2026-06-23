import csv
import json
import shutil
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path

from tools.ava_veil_protocol import (
    assign_split,
    iter_annotations,
    load_annotations,
    materialize,
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

    def test_materialize_extracts_target_crop_with_ffmpeg(self):
        if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
            self.skipTest("ffmpeg and ffprobe are required for materialize integration test")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            videos_dir = root / "videos"
            videos_dir.mkdir()
            video_path = videos_dir / "synthetic.mp4"
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=c=black:s=100x80:d=3:r=10",
                    "-vf",
                    "drawbox=x=20:y=16:w=30:h=32:color=red:t=fill",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    str(video_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            manifest_path = root / "manifest.jsonl"
            record = {
                "clip_id": "synthetic_0009010000_test",
                "split": "train",
                "video_id": "synthetic",
                "start_sec": 901.0,
                "end_sec": 902.0,
                "duration_sec": 1.0,
                "protected_entity_id": "person_red",
                "reference_timestamp": 901.0,
                "reference_box": [0.2, 0.2, 0.5, 0.6],
                "entity_ids": ["person_red", "person_other"],
                "num_entities": 2,
                "overlap_ratio": 1.0,
                "protected_coverage": 1.0,
                "protected_median_face_area": 0.12,
                "protected_median_min_side": 0.3,
                "observed_timestamps": 10,
                "score": 0.1,
            }
            manifest_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

            output_dir = root / "materialized"
            summary = materialize(
                manifest_path,
                videos_dir,
                output_dir,
                local_time_offset=900.0,
                ffmpeg_binary="ffmpeg",
                extensions=[".mp4"],
                target_padding=0.0,
                overwrite=True,
                check_only=False,
            )

            self.assertEqual(summary["counts"], {"materialized": 1})
            self.assertEqual(summary["failures"], [])
            self.assertTrue((output_dir / "clips" / "train" / "synthetic_0009010000_test.mp4").exists())
            target_path = output_dir / "targets" / "train" / "synthetic_0009010000_test_target.jpg"
            self.assertTrue(target_path.exists())

            completed = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "stream=width,height",
                    "-of",
                    "json",
                    str(target_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            dimensions = json.loads(completed.stdout)["streams"][0]
            self.assertEqual((int(dimensions["width"]), int(dimensions["height"])), (30, 32))

    def test_materialize_records_clear_target_crop_failures(self):
        if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
            self.skipTest("ffmpeg and ffprobe are required for materialize integration test")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            videos_dir = root / "videos"
            videos_dir.mkdir()
            video_path = videos_dir / "synthetic.mp4"
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=c=black:s=100x80:d=3:r=10",
                    "-vf",
                    "drawbox=x=20:y=16:w=30:h=32:color=red:t=fill",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    str(video_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            base_record = {
                "split": "train",
                "video_id": "synthetic",
                "start_sec": 901.0,
                "end_sec": 902.0,
                "duration_sec": 1.0,
                "protected_entity_id": "person_red",
                "reference_timestamp": 901.0,
                "reference_box": [0.2, 0.2, 0.5, 0.6],
                "entity_ids": ["person_red", "person_other"],
                "num_entities": 2,
                "overlap_ratio": 1.0,
                "protected_coverage": 1.0,
                "protected_median_face_area": 0.12,
                "protected_median_min_side": 0.3,
                "observed_timestamps": 10,
                "score": 0.1,
            }

            timestamp_record = dict(base_record, clip_id="synthetic_bad_timestamp", reference_timestamp=905.0)
            box_record = dict(base_record, clip_id="synthetic_bad_box", reference_box=[0.5, 0.2, 0.2, 0.6])
            manifest_path = root / "manifest.jsonl"
            manifest_path.write_text(
                "".join(json.dumps(record) + "\n" for record in (timestamp_record, box_record)),
                encoding="utf-8",
            )

            output_dir = root / "materialized"
            summary = materialize(
                manifest_path,
                videos_dir,
                output_dir,
                local_time_offset=900.0,
                ffmpeg_binary="ffmpeg",
                extensions=[".mp4"],
                target_padding=0.0,
                overwrite=True,
                check_only=False,
            )

            self.assertEqual(summary["counts"], {"target_crop_failed": 2})
            reasons = "\n".join(failure["reason"] for failure in summary["failures"])
            self.assertIn("invalid_reference_timestamp", reasons)
            self.assertIn("invalid_reference_box", reasons)

            manifest_rows = [
                json.loads(line)
                for line in (output_dir / "materialized_manifest.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([row["status"] for row in manifest_rows], ["target_crop_failed", "target_crop_failed"])
            self.assertIn("failure_reason", manifest_rows[0])
            self.assertIn("failure_reason", manifest_rows[1])

    def test_split_is_video_level_and_deterministic(self):
        splits = parse_split_ratios("train=70,val=15,test=15")
        first = assign_split("videoA", 1234, splits)
        second = assign_split("videoA", 1234, splits)
        self.assertEqual(first, second)
        self.assertIn(first, {"train", "val", "test"})


if __name__ == "__main__":
    unittest.main()
