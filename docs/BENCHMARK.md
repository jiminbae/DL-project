# VEIL Benchmark Runner

This document records the corrected Panoptic-VEIL ablation benchmark flow.

## Important Correction

The previous `panoptic10_ablation` run must not be used as the final paper result. The runner used the Panoptic protected target crop for both:

- protected identity registration, and
- InSwapper replacement identity.

Correct behavior requires two separate images:

- protected target crop: `manifest[].target_image_path`
- replacement face: `models/veil/virtual_face/fake_face.jpg`

The corrected runner now injects these separately:

```text
config.TARGET_DIR / TARGET_PATTERN = protected target crop
config.TARGET_IMAGE_PATH = replacement face image
```

## One-Shot Command

Run the corrected accepted-subset ablation with:

```bash
bash scripts/run_panoptic10_ablation_fixed.sh
```

Optional overrides:

```bash
PYTHON_BIN=/home/jmbae/veil/.venv/bin/python \
REPLACEMENT_IMAGE=models/veil/virtual_face/fake_face.jpg \
OUTPUT_DIR=data/panoptic_veil_materialized_10/benchmarks/panoptic10_ablation_fixed \
PAPER_RESULTS_DIR=paper_results/panoptic10_ablation_fixed \
bash scripts/run_panoptic10_ablation_fixed.sh
```

The script first writes a dry-run plan, then runs all four conditions over the 9 manually accepted Panoptic pilot clips, and finally packages lightweight sanitized result files under `paper_results/panoptic10_ablation_fixed/`.

## Conditions

- `full`: default VEIL with identity lock, face swap, and blur fallback.
- `no_identity_lock`: temporal identity lock is disabled; only direct target matches are preserved.
- `no_blur_fallback`: fallback blur is disabled after failed swaps or low-quality background faces.
- `selective_blur_only`: target-aware baseline that preserves protected targets and blurs all non-target faces without attempting face swapping.


## Smoke Test

Run a one-clip dry run for the selective blur-only baseline:

```bash
.venv/bin/python tools/run_veil_benchmark.py \
  --manifest data/panoptic_veil_accepted_20/accepted_manifest.jsonl \
  --review-csv data/panoptic_veil_accepted_20/source_review_accepted.csv \
  --output-dir /tmp/veil_selective_blur_smoke \
  --replacement-image models/veil/virtual_face/fake_face.jpg \
  --accepted-only \
  --clip-limit 1 \
  --conditions selective_blur_only \
  --python .venv/bin/python \
  --dry-run
```

## Outputs

Large local outputs:

- `data/panoptic_veil_materialized_10/benchmarks/panoptic10_ablation_fixed/benchmark_summary.json`
- `data/panoptic_veil_materialized_10/benchmarks/panoptic10_ablation_fixed/condition_metrics.csv`
- `data/panoptic_veil_materialized_10/benchmarks/panoptic10_ablation_fixed/<condition>/runs/<clip_id>/`
- `data/panoptic_veil_materialized_10/benchmarks/panoptic10_ablation_fixed/<condition>/metadata_eval/`

Git-trackable lightweight outputs:

- `paper_results/panoptic10_ablation_fixed/condition_metrics.csv`
- `paper_results/panoptic10_ablation_fixed/benchmark_summary.json`
- `paper_results/panoptic10_ablation_fixed/run_summary.csv`
- `paper_results/panoptic10_ablation_fixed/<condition>/aggregate_metrics.json`
- `paper_results/panoptic10_ablation_fixed/<condition>/per_clip_metrics.csv`

## Metric Notes

The corrected pipeline writes `final_action`, `swap_success`, `blur_applied`, `is_target_direct`, and `is_target_final` into face metadata. The evaluator uses `final_action` when present and falls back to conservative inference for older metadata.

`target_coverage` is computed as unique protected frames divided by total frames. Non-target partition metrics include `swap_coverage`, `blur_coverage`, `non_target_unprocessed_rate`, and `non_target_unknown_rate` with the same non-target denominator.

`protected_alteration_rate` is still system-metadata based, not independent ground-truth protected-person evaluation. Treat it as a pipeline-state metric until an evaluation-only protected identity association is added.
