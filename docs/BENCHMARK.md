# VEIL Benchmark Runner

This document records the reusable benchmark runner for Panoptic-VEIL ablation experiments.

## Command

```bash
python tools/run_veil_benchmark.py \
  --manifest data/panoptic_veil_materialized_10/materialized_manifest.jsonl \
  --review-csv data/panoptic_veil_materialized_10/review/panoptic_10_review.csv \
  --output-dir data/panoptic_veil_materialized_10/benchmarks/panoptic10_ablation \
  --accepted-only \
  --conditions full no_identity_lock no_blur_fallback \
  --python /home/jmbae/veil/.venv/bin/python \
  --timeout-sec 1800 \
  --skip-existing
```

The runner does not edit `models/veil/config.py`. Instead, each subprocess injects runtime config values before importing `main_hybrid.py`.

## Conditions

- `full`: default VEIL with identity lock, face swap, and blur fallback.
- `no_identity_lock`: temporal identity lock is disabled; only direct target matches are preserved.
- `no_blur_fallback`: fallback blur is disabled after failed swaps or low-quality background faces.

## Panoptic-10 Accepted-Subset Result

The benchmark was run on the 9 manually accepted Panoptic pilot clips.

| condition | clips | protected alteration rate | non-target exposure rate | anonymization coverage | swap coverage | blur coverage | unknown rate | mean target coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full | 9 | 0.000000 | 0.000000 | 0.551441 | 0.053420 | 0.498021 | 0.263534 | 0.941664 |
| no_identity_lock | 9 | 0.000000 | 0.000000 | 0.563838 | 0.052075 | 0.511763 | 0.273995 | 0.850686 |
| no_blur_fallback | 9 | 0.000000 | 0.498021 | 0.053420 | 0.053420 | 0.000000 | 0.263534 | 0.941664 |

## Interpretation

The `no_blur_fallback` ablation shows why fallback blur is part of VEIL's privacy boundary: the same accepted subset has a non-target exposure rate of 0.498021 when low-quality or failed-swap background faces are left unblurred.

The `no_identity_lock` ablation lowers mean target coverage from 0.941664 to 0.850686. This supports the claim that temporal/stable identity locking helps preserve the registered protected identity beyond isolated direct embedding matches.

As with the metadata evaluator, `SWAP` is conservative because current metadata does not store final per-row actions. Rows without exact swap-log evidence remain `UNKNOWN` rather than being counted as successful anonymization.

## Outputs

- `benchmark_summary.json`: selected clips and run counts.
- `run_summary.csv`: one row per condition/clip run.
- `condition_metrics.csv`: aggregate condition comparison table.
- `<condition>/runs/<clip_id>/`: output video, logs, stdout/stderr, and metadata for each clip.
- `<condition>/metadata_eval/`: conservative metadata evaluation for each condition.
