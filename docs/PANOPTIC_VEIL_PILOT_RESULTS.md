# Panoptic-VEIL Pilot Results Draft

This note summarizes the current pilot evaluation of VEIL on CMU Panoptic Studio haggling clips. It is intended as paper/report draft material rather than final camera-ready text.

## Experimental Setup

VEIL is evaluated as a selective face de-identification pipeline: one registered protected identity is preserved, while non-protected faces are anonymized through InSwapper when the face quality gate permits swapping and through fallback handling otherwise. The pilot uses Panoptic haggling sequences because the earlier AVA and AMI pilots frequently produced faces that were too small or too compressed for reliable swap-quality evaluation.

The Panoptic pilot was constructed from public Panoptic HD video URLs without redistributing full source videos. Candidate frames were scanned with the VEIL YOLO face detector, and 10-second clips were materialized with FFmpeg. For each clip, one protected target crop was extracted from the reference frame and used as the registered identity.

## Data

- Dataset source: CMU Panoptic Studio haggling sequences.
- Pilot materialized clips: 10.
- Clip duration: 10 seconds each.
- Accepted after manual review: 9 / 10.
- Rejected after manual review: 1 / 10.
- Accepted manifest: data/panoptic_veil_materialized_10/review/accepted_manifest.jsonl.
- Accepted results table: data/panoptic_veil_materialized_10/evaluation/panoptic_accepted_9_evaluation_results.csv.

The rejected clip was 170221_haggling_b2_hd_00_27_0001950000. It was excluded because the target reference crop was mostly a side profile, which caused the protected target to be briefly blurred at the beginning of the output, although the rest of the clip was otherwise normal.

## Main Results

Across the 10 materialized clips, the VEIL pipeline completed successfully for all clips. Manual review accepted 9 clips. In the accepted subset, all 9 clips were judged to preserve the target identity and anonymize non-target identities.

Summary for the accepted 9 clips:

| Metric | Value |
| --- | ---: |
| Pipeline success on materialized clips | 10 / 10 |
| Manual accepted clips | 9 / 10 |
| Accepted clips with target preserved | 9 / 9 |
| Accepted clips with non-target anonymized | 9 / 9 |
| Accepted clips rated good visual quality | 9 / 9 |
| Accepted clips with logged swaps | 8 / 9 |
| Total logged swap successes on accepted clips | 174 |
| Total logged frames with swap success on accepted clips | 166 |
| Mean target coverage over 300 frames | 0.92 |

One accepted clip had no logged swap successes but was accepted by manual review because the non-target handling was still judged acceptable in the output. This distinction should be kept explicit: the manual acceptance criterion is based on the final de-identification result, while logged swap counts only measure frames where the InSwapper path was used successfully.

## Suggested Paper Wording

A concise results paragraph:

> We evaluated VEIL on a pilot set of 10 CMU Panoptic Studio haggling clips. All 10 clips were processed successfully by the pipeline. Manual review accepted 9 of 10 clips. In the accepted subset, the protected target identity was preserved in all clips, and non-target identities were judged anonymized in all clips. Eight of the nine accepted clips contained logged successful face swaps, with 174 total logged swap successes. The remaining accepted clip was handled acceptably through the pipeline's non-swap/fallback behavior.

A concise limitation paragraph:

> The rejected clip illustrates a limitation of target registration quality. When the protected reference crop is dominated by a side-profile view, the target embedding can be less stable, causing brief early-frame target blur before the target track stabilizes. This suggests that future protocol versions should require frontal or near-frontal target reference crops, or multiple target references per protected identity.

## Figure Candidates

Recommended qualitative examples:

1. 170224_haggling_b1_hd_00_25_0001450000
   - Strongest logged swap count among accepted clips: 33 total logged swap successes.
   - Useful as the main positive qualitative example.

2. 170224_haggling_b1_hd_00_25_0001750000
   - Previously validated pilot example.
   - Good for continuity with earlier smoke-test results.

3. 170221_haggling_b3_hd_00_27_0001350000
   - Good accepted clip from a different sequence.
   - Useful to show that the result is not limited to one Panoptic sequence/camera.

Failure/limitation example:

- 170221_haggling_b2_hd_00_27_0001950000
  - Side-profile target reference causes brief protected-target blur.

## Files To Use

- Review bundle with browser-compatible H.264 outputs: data/panoptic_veil_materialized_10/review_output_bundle_h264/index.html
- Accepted-only CSV: data/panoptic_veil_materialized_10/evaluation/panoptic_accepted_9_evaluation_results.csv
- All-clip CSV including rejected clip: data/panoptic_veil_materialized_10/evaluation/panoptic_10_evaluation_results.csv
- Aggregate summary JSON: data/panoptic_veil_materialized_10/evaluation/panoptic_accepted_9_summary.json
- Accepted manifest: data/panoptic_veil_materialized_10/review/accepted_manifest.jsonl

## Next Step

Use this pilot result to write the evaluation section and select qualitative frames. After the paper structure is stable, scale Panoptic from 9 accepted clips to a larger accepted set if more statistical weight is needed.
