# Manual Output Review Summary

Reviewer: jimin

Reviewed full-condition outputs for the Panoptic-20 benchmark. All 20 clips remain usable for aggregate metrics, but only the clean-pass clips should be preferred for qualitative figures.

## Counts

- Reviewed clips: 20
- Clean qualitative candidates: 8
- Target transient alteration clips: 6
- Non-target blur quality caveat clips: 4
- Non-target swap/blur flicker caveat clips: 2

## Interpretation

The full pipeline anonymizes non-target observations in the metadata evaluation, but visual quality often depends on blur fallback. Target failures are brief and occur mainly when the protected target turns away, appears in side profile, or is occluded. Non-target visual artifacts include blur-only handling of visible faces and repeated swap/blur alternation for side-profile non-targets.

## Paper Use

Use clean-pass clips as primary qualitative examples. Use blur-only and swap/blur flicker clips as limitation examples if needed. Avoid claiming independent proof of zero protected-person alteration; describe the protected alteration metric as system-label based.
