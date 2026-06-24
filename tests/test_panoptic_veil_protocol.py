import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from tools.panoptic_veil_protocol import (
    build_panoptic_video_url,
    expanded_pixel_box,
    materialize,
    normalize_camera,
    select_reference_box,
)


class PanopticVeilProtocolTest(unittest.TestCase):
    def test_url_and_box_helpers(self):
        self.assertEqual(normalize_camera("25"), "hd_00_25")
        self.assertEqual(
            build_panoptic_video_url("http://example.test/", "170224_haggling_b1", "25"),
            "http://example.test/webdata/dataset/170224_haggling_b1/videos/hd_shared_crf20/hd_00_25.mp4",
        )
        self.assertEqual(expanded_pixel_box([0.2, 0.2, 0.5, 0.6], 100, 80, 0.0), (20, 16, 50, 48))
        self.assertEqual(expanded_pixel_box([20, 16, 50, 48], 100, 80, 0.0), (20, 16, 50, 48))

    def test_select_reference_box_uses_explicit_target_face_index(self):
        record = {
            "reference_faces": [
                {"bbox": [10, 10, 20, 20], "area": 100},
                {"bbox": [30, 30, 50, 60], "area": 600},
            ],
            "target_face_index": 1,
        }
        box, source = select_reference_box(record)
        self.assertEqual(box, [30, 30, 50, 60])
        self.assertEqual(source, "reference_faces[1].bbox")

    def test_materialize_extracts_clip_frame_and_target_crop(self):
        if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
            self.skipTest("ffmpeg and ffprobe are required for materialize integration test")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video_path = root / "synthetic.mp4"
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
                    "color=c=black:s=100x80:d=4:r=10",
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

            candidates_path = root / "candidates.jsonl"
            record = {
                "clip_id": "synthetic_panoptic_clip",
                "split": "pilot",
                "sequence": "synthetic_sequence",
                "camera": "hd_00_25",
                "source_video_url": str(video_path),
                "start_seconds": 1.0,
                "duration_seconds": 1.5,
                "reference_timestamp_seconds": 1.2,
                "reference_faces": [
                    {"bbox": [20, 16, 50, 48], "conf": 0.9, "min_side": 30.0, "area": 960.0}
                ],
                "target_face_index": 0,
                "target_padding": 0.0,
                "protected_target_policy": "synthetic_test_box",
            }
            candidates_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

            output_dir = root / "materialized"
            summary = materialize(candidates_path, output_dir, overwrite=True)

            self.assertEqual(summary["counts"], {"materialized": 1})
            self.assertEqual(summary["failures"], [])
            self.assertTrue((output_dir / "clips" / "pilot" / "synthetic_panoptic_clip.mp4").exists())
            self.assertTrue((output_dir / "frames" / "synthetic_panoptic_clip_ref.jpg").exists())
            target_path = output_dir / "targets" / "pilot" / "synthetic_panoptic_clip_target.jpg"
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

    def test_materialize_records_clear_failures(self):
        if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
            self.skipTest("ffmpeg and ffprobe are required for materialize integration test")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video_path = root / "synthetic.mp4"
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
                    "color=c=black:s=100x80:d=2:r=10",
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

            candidates_path = root / "candidates.jsonl"
            records = [
                {
                    "clip_id": "bad_box",
                    "sequence": "synthetic_sequence",
                    "camera": "hd_00_25",
                    "source_video_url": str(video_path),
                    "start_seconds": 0.0,
                    "duration_seconds": 1.0,
                    "reference_timestamp_seconds": 0.2,
                    "reference_box": [0.5, 0.2, 0.2, 0.6],
                },
                {
                    "clip_id": "missing_start",
                    "sequence": "synthetic_sequence",
                    "camera": "hd_00_25",
                    "source_video_url": str(video_path),
                    "duration_seconds": 1.0,
                    "reference_timestamp_seconds": 0.2,
                    "reference_box": [0.2, 0.2, 0.5, 0.6],
                },
            ]
            candidates_path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )

            output_dir = root / "materialized"
            summary = materialize(candidates_path, output_dir, overwrite=True)

            self.assertEqual(summary["counts"], {"invalid_candidate": 1, "target_crop_failed": 1})
            reasons = "\n".join(failure["reason"] for failure in summary["failures"])
            self.assertIn("invalid_reference_box", reasons)
            self.assertIn("missing_candidate_field: start_seconds", reasons)
            manifest_rows = [
                json.loads(line)
                for line in (output_dir / "materialized_manifest.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([row["status"] for row in manifest_rows], ["target_crop_failed", "invalid_candidate"])


if __name__ == "__main__":
    unittest.main()
