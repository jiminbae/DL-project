# VEIL Metadata Evaluation

This document describes the conservative metadata evaluator used for the
Panoptic-VEIL pilot results.

## Why the Rules Are Conservative

Current `face_metadata*.json` files do not contain a final `action` field and do
not store per-row `swap_success`. The final output video may contain preserved,
swapped, and blurred faces, but only part of that information is available in
metadata. The evaluator therefore uses explicit evidence only:

- `is_target=true` is classified as `PRESERVE`.
- A background row is classified as `SWAP` only when the tracking log contains
  the exact `(frame_idx, raw_track_id)` with `SwapSuccess=True`.
- A background row is classified as `BLUR` when the exact log entry says
  `SwapSuccess=False`, or when metadata indicates embedding failure, non-GOOD
  quality, or fallback reasons.
- A GOOD background row without per-track swap-log evidence is classified as
  `UNKNOWN`.

This intentionally under-counts `SWAP` and preserves uncertainty rather than
turning missing log evidence into a privacy claim.

## Outputs

Run:

```bash
python tools/evaluate_veil_metadata.py \
  --runs-dir data/panoptic_veil_materialized_10/veil_smoke_runs \
  --review-csv data/panoptic_veil_materialized_10/review/panoptic_10_review.csv \
  --output-dir data/panoptic_veil_materialized_10/evaluation/metadata_eval
```

For no-blur-fallback ablations, pass `--blur-fallback-enabled false` so failed
or low-quality background rows are counted as `UNPROCESSED` rather than `BLUR`.

The output directory contains:

- `schema_report.json`: observed metadata fields and examples.
- `evaluation_config.json`: classification and deduplication policy.
- `per_frame_actions.jsonl`: one classified action row per deduplicated face observation.
- `per_clip_metrics.csv`: clip-level counts and rates.
- `per_identity_metrics.csv`: identity-level state counts.
- `aggregate_metrics.json`: aggregate metrics over accepted clips when review CSV is provided.

## Zero-Swap Diagnostics

Accepted clips with no logged swaps are not automatically failures. Run:

```bash
python tools/diagnose_zero_swap.py \
  --metadata-eval-dir data/panoptic_veil_materialized_10/evaluation/metadata_eval \
  --materialized-manifest data/panoptic_veil_materialized_10/materialized_manifest.jsonl \
  --smoke-summary data/panoptic_veil_materialized_10/veil_smoke_runs/smoke_summary.json \
  --output-dir data/panoptic_veil_materialized_10/evaluation/zero_swap
```

The diagnostic separates `blur_only_success`, `no_valid_non_target`,
`privacy_exposure`, and `metadata_logging_gap`. For the current Panoptic-10
pilot, the only accepted zero-swap clip is classified as `blur_only_success`: all
non-target observations are blurred and no protected-target alteration is found.

## Metric Caveat

Because `SWAP` is only counted when per-track log evidence exists, swap coverage
is a lower bound. `UNKNOWN` rows should be inspected before making strong claims
about non-target exposure or anonymization coverage. For default VEIL runs,
fallback-eligible background rows are counted as `BLUR`; for no-blur-fallback
ablations, the same rows are counted as `UNPROCESSED`.
