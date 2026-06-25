# Panoptic-10 VEIL Ablation Results

This folder contains lightweight, sanitized result files for the corrected Panoptic-10 accepted-subset ablation run.

Full videos and large per-frame JSONL files are intentionally excluded. Absolute local paths are reduced to file or directory names where possible.

Primary table: `condition_metrics.csv`.

## Manual review

Manual review was performed on the 9 accepted Panoptic clips using the generated review page for the `full` condition. The review CSV and summary are included as `manual_review.csv` and `manual_review_summary.json`.

Result: 8 of 9 clips were accepted. The remaining clip, `170224_haggling_b1_hd_00_08_0001550000`, was marked as a limitation case because the protected target is briefly blurred at the beginning, likely due to partial occlusion in the target reference crop. Non-target processing and replacement quality were still marked as acceptable for that clip.
