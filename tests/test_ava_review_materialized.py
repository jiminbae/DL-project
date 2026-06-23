import csv
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from tools.ava_review_materialized import generate_review_page


class AvaReviewMaterializedTest(unittest.TestCase):
    def test_generate_review_page_with_frames_and_csv(self):
        if shutil.which("ffmpeg") is None:
            self.skipTest("ffmpeg is required for review page frame extraction")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            clip_path = root / "clip.mp4"
            target_path = root / "target.jpg"
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
                    "testsrc=size=160x90:duration=1:rate=10",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    str(clip_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
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
                    "color=c=red:s=40x40:d=0.1",
                    "-frames:v",
                    "1",
                    str(target_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            manifest_path = root / "materialized_manifest.jsonl"
            record = {
                "clip_id": "clip_one",
                "split": "train",
                "video_id": "videoA",
                "protected_entity_id": "personA",
                "reference_timestamp": 901.0,
                "local_reference_sec": 1.0,
                "reference_box": [0.2, 0.2, 0.5, 0.6],
                "num_entities": 2,
                "duration_sec": 1.0,
                "status": "materialized",
                "clip_path": str(clip_path),
                "target_image_path": str(target_path),
            }
            manifest_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

            review_dir = root / "review"
            summary = generate_review_page(
                manifest_path,
                review_dir,
                ffmpeg_binary="ffmpeg",
                overwrite_frames=True,
            )

            self.assertEqual(summary["num_records"], 1)
            self.assertEqual(summary["num_frame_error_records"], 0)
            self.assertTrue((review_dir / "index.html").exists())
            self.assertTrue((review_dir / "review_summary.json").exists())
            self.assertEqual(len(list((review_dir / "frames").glob("clip_one_*.jpg"))), 3)

            with (review_dir / "review.csv").open("r", encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows, [{"clip_id": "clip_one", "accepted": "", "reason": "", "reviewer": ""}])

            html_text = (review_dir / "index.html").read_text(encoding="utf-8")
            self.assertIn("clip_one", html_text)
            self.assertIn("wrong_target", html_text)
            self.assertIn("<video", html_text)


if __name__ == "__main__":
    unittest.main()
