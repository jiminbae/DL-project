# Panoptic-VEIL Selective Anonymization Protocol

This document describes the current primary evaluation path for VEIL after the
AVA and AMI pilots showed that many public meeting/movie clips are too small or
compressed for reliable face swapping.

The primary protocol uses CMU Panoptic Studio haggling sequences because they
provide multi-person HD video with faces large enough for VEIL's default
InSwapper gate. AVA/AMI can still be kept as secondary robustness material, but
Panoptic is the better pilot source for demonstrating the intended
preserve-target / anonymize-non-target behavior.

The repository does **not** redistribute Panoptic source videos. The protocol
stores candidate sequence/camera/timestamp metadata and materializes only local
evaluation clips from the public Panoptic video endpoint.

## What the Tool Produces

Running `materialize` creates:

- `clips/<split>/<clip_id>.mp4`: the video-only evaluation clip.
- `targets/<split>/<clip_id>_target.jpg`: the protected reference crop.
- `frames/<clip_id>_ref.jpg`: the reference frame used for target selection.
- `materialized_manifest.jsonl`: per-clip paths, status, and failure reasons.
- `materialize_summary.json`: counts and concrete failure summaries.

Running `scan` optionally samples remote Panoptic frames with the VEIL YOLO face
detector and writes:

- `scan_results.jsonl`: every sampled frame and its detections.
- `candidate_frames.jsonl`: frames with at least the requested number of
  swap-size faces.
- `scan_summary.json`: scan parameters and counts.

## Current Pilot Candidate

The current successful pilot clip is:

```text
sequence: 170224_haggling_b1
camera: hd_00_25
clip window: 175.0s to 185.0s
reference timestamp: 180.0s
target face: reference_faces[2], padding 0.5
```

This produces a 10-second, 300-frame clip with three visible faces. The final
VEIL smoke run preserved one target track and swapped the non-target track:

```text
output: data/panoptic_veil_materialized/veil_smoke_run_final/output_target22.mp4
metadata rows: 865
target rows: 300
background rows: 565
swap successes logged: 24
```

## Materialize the Pilot

Create or reuse a candidate JSONL row like this:

```json
{"clip_id":"170224_haggling_b1_hd_00_25_0001750000","split":"pilot","sequence":"170224_haggling_b1","camera":"hd_00_25","start_seconds":175.0,"duration_seconds":10.0,"reference_timestamp_seconds":180.0,"reference_faces":[{"bbox":[1586.5,244.0,1704.4,486.0],"conf":0.7746,"min_side":118.0,"area":28550.4,"swap_size_ok":true},{"bbox":[677.6,209.3,779.9,390.8],"conf":0.8611,"min_side":102.3,"area":18562.1,"swap_size_ok":false},{"bbox":[1304.8,327.0,1418.1,470.1],"conf":0.8867,"min_side":113.4,"area":16226.4,"swap_size_ok":true}],"target_face_index":2,"target_padding":0.5,"protected_target_policy":"manually_selected_stable_target_crop"}
```

Then run:

```bash
python tools/panoptic_veil_protocol.py materialize \
  --candidates data/panoptic_veil_pilot/candidates.jsonl \
  --output-dir data/panoptic_veil_materialized \
  --overwrite
```

The clip is pulled directly from:

```text
http://domedb.perception.cs.cmu.edu/webdata/dataset/170224_haggling_b1/videos/hd_shared_crf20/hd_00_25.mp4
```

No full Panoptic video download is required.

## Optional Candidate Scan

Use this when expanding beyond the current pilot:

```bash
python tools/panoptic_veil_protocol.py scan \
  --output-dir data/panoptic_veil_scan \
  --sequences 170224_haggling_b1,170224_haggling_b2,170224_haggling_b3 \
  --cameras 08,27,00,25 \
  --times 60,180,240 \
  --model-path models/veil/weights/yolo26x-face.pt
```

Promising frames should still be manually reviewed. In particular, pick a target
crop that is stable enough for InsightFace embedding, not merely the largest
face in the reference frame.

## Recommended Paper Framing

Use Panoptic as the primary controlled pilot benchmark:

- large enough faces for the default VEIL swap threshold;
- multiple simultaneous identities;
- reproducible candidate metadata;
- source video remains external and official.

Use AVA/AMI only as secondary robustness checks:

- AVA: realistic movie compression and failure-safe behavior;
- AMI: meeting-room low-resolution limits;
- neither should be the only source for the main swap-quality claim.

## Tests

The unit tests use synthetic video and candidate JSONL, so they do not contact
the Panoptic server:

```bash
python -m unittest discover -s tests -v
```
