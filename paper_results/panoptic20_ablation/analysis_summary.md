# Panoptic-20 Ablation Analysis

- Completed runs: 60 / 60
- Failed runs: 0
- Full condition non-target exposure: 0.000
- Full condition anonymization coverage: 1.000
- Full condition swap / blur coverage: 0.217 / 0.783
- Mean target coverage, full vs no identity lock: 0.916 vs 0.849
- No-blur-fallback non-target exposure: 0.783

## Interpretation

The full VEIL condition processed all system-labeled non-target observations, with no measured non-target exposure. Most non-target processing came from blur fallback rather than successful face swap. Removing blur fallback exposed the same fraction of non-target observations that full VEIL would otherwise blur. Removing identity lock reduced mean target coverage, showing that temporal identity continuity helps retain protected targets.

## Caution

Protected alteration rate is based on system-labeled protected observations, not independent protected-person ground truth.

## Manual Visual Review

Manual full-output review found 8 clean qualitative candidates out of 20 clips. All clips remain usable for aggregate metrics, but visual limitations should be reported. Six clips show brief target alteration under target pose changes, side profile, or occlusion. Four clips perform correctly but blur a clearly visible non-target face rather than swapping it, likely due to face size or video quality. Two clips preserve the target but show repeated swap/blur alternation on a side-profile non-target face.

Use clean-pass clips for primary qualitative figures. Use blur-only or swap/blur flicker cases as limitation examples if needed.

Structured review files: `manual_output_review.csv`, `manual_output_review_summary.json`, and `manual_output_review_summary.md`.
