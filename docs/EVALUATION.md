# VEIL Metadata Evaluation

This document describes the conservative metadata evaluator used for the
Panoptic-VEIL pilot results.

## Why the Rules Are Conservative

New benchmark metadata records `final_action`, `swap_success`, `blur_applied`,
`is_target_direct`, and `is_target_final`. When `final_action` is present, the
evaluator uses it directly. For older metadata without final actions, the
evaluator falls back to conservative evidence only:

- `final_action` in `PRESERVE`, `SWAP`, `BLUR`, `UNPROCESSED`, `FAILED`, or `UNKNOWN` is used directly.
- Without `final_action`, `is_target=true` is classified as `PRESERVE`.
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
- `per_clip_metrics.csv`: clip-level counts and rates, including unique-frame target coverage and non-target partition rates.
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

For new metadata, `SWAP`, `BLUR`, and `UNPROCESSED` come from `final_action`.
For older metadata, `SWAP` is only counted when per-track log evidence exists,
so swap coverage is a lower bound. `UNKNOWN` rows should be inspected before
making strong claims about non-target exposure or anonymization coverage. For
default VEIL runs, fallback-eligible background rows are counted as `BLUR`; for
no-blur-fallback ablations, the same rows are counted as `UNPROCESSED`.

## Independent Identity Verifier Evaluation

`tools/eval_identity_verifier.py` provides a separate evaluation path for
identity preservation and identity leakage. It does not reuse VEIL's internal
embedding cache or target-matching scores. The verifier is implemented behind a
small interface:

```python
Verifier.extract_embedding(image)
Verifier.similarity(a, b)
```

The script writes:

- `pair_scores.csv`: target and non-target pair-level verifier scores.
- `per_clip_identity_verifier.csv`: mean, median, and threshold accept rates for one clip.
- `aggregate_identity_verifier.csv`: aggregate summary using the same schema.
- `evaluation_config.json`: verifier backend, threshold, box source, and input provenance.
- `debug_crops/`: examples of high non-target similarity and low target similarity.

Example smoke command using the built-in `debug_color` backend:

```bash
.venv/bin/python tools/eval_identity_verifier.py \
  --original data/panoptic_veil_materialized_20/clips/pilot20/170221_haggling_b2_hd_00_27_0001350000.mp4 \
  --processed data/panoptic_veil_accepted_20/benchmarks/panoptic20_ablation/full/runs/170221_haggling_b2_hd_00_27_0001350000/170221_haggling_b2_hd_00_27_0001350000_output.mp4 \
  --action-metadata data/panoptic_veil_accepted_20/benchmarks/panoptic20_ablation/full/runs/170221_haggling_b2_hd_00_27_0001350000/face_metadata_170221_haggling_b2_hd_00_27_0001350000.json \
  --protected-reference data/panoptic_veil_materialized_20/targets/pilot20/170221_haggling_b2_hd_00_27_0001350000_target.jpg \
  --condition full \
  --clip-id 170221_haggling_b2_hd_00_27_0001350000 \
  --output-dir /tmp/veil_identity_verifier_smoke \
  --backend debug_color \
  --box-source metadata \
  --max-pairs 20 \
  --max-debug-crops 2
```

Important caveats:

- `debug_color` is only a deterministic smoke-test backend. It is not a
  paper-valid face-recognition verifier.
- The current environment has InsightFace available, but VEIL already uses
  InsightFace `buffalo_l` internally, so that checkpoint should not be used as
  the independent verifier.
- For paper results, plug in an independent backend such as AdaFace, MagFace,
  FaceNet, or a custom ONNX verifier, and prefer `--box-source gt` with external
  target/non-target box annotations when available.
- If `--box-source metadata` is used, verifier embeddings are separate from
  VEIL, but crop locations still come from VEIL action metadata.
