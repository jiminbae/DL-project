import csv
import json
import tempfile
import unittest
from pathlib import Path

from tools.package_benchmark_results import package_results


class PackageBenchmarkResultsTest(unittest.TestCase):
    def test_packages_and_sanitizes_lightweight_results(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bench = root / "bench"
            condition = bench / "full"
            (condition / "metadata_eval").mkdir(parents=True)
            (bench / "benchmark_summary.json").write_text(
                json.dumps({"output_dir": "/abs/local/bench", "completed_runs": 1}),
                encoding="utf-8",
            )
            (bench / "condition_metrics.csv").write_text(
                "condition,non_target_exposure_rate\nfull,0.0\n",
                encoding="utf-8",
            )
            (bench / "run_summary.csv").write_text(
                "clip_id,output_video\nclip_a,/abs/local/output.mp4\n",
                encoding="utf-8",
            )
            (bench / "run_summary.json").write_text(
                json.dumps([{"clip_id": "clip_a", "output_video": "/abs/local/output.mp4"}]),
                encoding="utf-8",
            )
            (condition / "condition_config.json").write_text(json.dumps({"name": "full"}), encoding="utf-8")
            (condition / "metadata_eval" / "aggregate_metrics.json").write_text(
                json.dumps({"non_target_exposure_rate": 0.0}),
                encoding="utf-8",
            )
            (condition / "metadata_eval" / "evaluation_config.json").write_text(
                json.dumps({"runs_dir": "/abs/local/runs"}),
                encoding="utf-8",
            )
            (condition / "metadata_eval" / "per_clip_metrics.csv").write_text(
                "clip_id,face_metadata_path\nclip_a,/abs/local/face_metadata.json\n",
                encoding="utf-8",
            )

            out = root / "paper_results"
            summary = package_results(bench, out, ["full"])
            self.assertEqual(summary["conditions"], ["full"])
            payload = json.loads((out / "benchmark_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["output_dir"], "bench")
            with (out / "run_summary.csv").open("r", encoding="utf-8", newline="") as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row["output_video"], "output.mp4")
            self.assertTrue((out / "README.md").exists())


if __name__ == "__main__":
    unittest.main()
