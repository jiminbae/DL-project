# Panoptic-20 VEIL Ablation Results

This folder contains lightweight, sanitized result files for the VEIL accepted-subset ablation run.

Full videos and large per-frame JSONL files are intentionally excluded. Absolute local paths are reduced to file or directory names where possible.

Primary table: `condition_metrics.csv`.

Manual visual review is summarized in `manual_output_review_summary.md`.

Benchmark interpretation is summarized in `analysis_summary.md`.

## Contents

- Final accepted set: 20 clips, listed in `accepted_manifest.jsonl` and summarized in `accepted_summary.json`.
- Benchmark: 60/60 runs completed across `full`, `no_identity_lock`, and `no_blur_fallback`.
- Source review summaries are included as `source_review_base20_summary.json` and `source_review_supplement10_summary.json`.
- Manual full-output review is included as `manual_output_review.csv` and `manual_output_review_summary.*`.

## Main Result

In the full condition, system-labeled non-target exposure was 0.000 and anonymization coverage was 1.000. The pipeline relied heavily on blur fallback: non-target swap coverage was 0.217 and blur coverage was 0.783. Disabling blur fallback increased non-target exposure to 0.783. Disabling identity lock reduced mean target coverage from 0.916 to 0.849.

## Caveat

Protected alteration metrics are based on system-labeled protected observations, not independent protected-person ground truth. Manual visual review found brief target alteration in pose or occlusion cases and blur-dominant non-target handling in several clips.
