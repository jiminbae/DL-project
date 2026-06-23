# AVA-VEIL Selective Anonymization Protocol

This document describes a reproducible way to build a VEIL evaluation set from
public AVA ActiveSpeaker face-track annotations without collecting a new dataset.

The protocol selects short multi-person windows, chooses one well-observed face
track as the protected identity, and records every other eligible track as a
non-target identity. It can then extract clips and protected reference crops from
source videos that you have already obtained and are authorized to process.

The repository does **not** download or redistribute source videos.

## What the tool produces

Running `select` creates:

- `manifest.jsonl`: one reproducible experiment record per selected clip.
- `manifest.csv`: spreadsheet-friendly version of the same manifest.
- `selected_annotations.csv`: all AVA face annotations inside selected windows,
  including an `is_protected` flag.
- `summary.json`: selection counts and the exact parameters used.

Running `materialize` creates:

- `clips/<split>/<clip_id>.mp4`: video-only evaluation clips.
- `targets/<split>/<clip_id>_target.jpg`: protected-identity reference crops.
- `materialized_manifest.jsonl`: manifest augmented with local file paths and
  status information.
- `materialize_summary.json`: missing-file and extraction statistics.

## 1. Obtain the annotations

Download the AVA ActiveSpeaker train and validation annotation archives from the
official AVA download page and keep the archives intact or extract them into a
directory. The script accepts either form.

Example local layout:

```text
data/
├── ava_activespeaker_train_v1.0.tar.bz2
├── ava_activespeaker_val_v1.0.tar.bz2
└── videos/
    ├── <video_id_1>.mp4
    ├── <video_id_2>.mp4
    └── ...
```

## 2. Run a small pilot selection

Start with a small number of videos so that threshold mistakes are cheap to fix.

```bash
python tools/ava_veil_protocol.py select \
  --annotations data/ava_activespeaker_train_v1.0.tar.bz2 \
  --output-dir data/ava_veil_pilot \
  --window-seconds 10 \
  --step-seconds 5 \
  --min-faces 2 \
  --min-overlap-ratio 0.50 \
  --min-entity-coverage 0.50 \
  --min-normalized-face-size 0.06 \
  --max-clips-per-video 3 \
  --limit-videos 5
```

Inspect:

```bash
cat data/ava_veil_pilot/summary.json
head data/ava_veil_pilot/manifest.csv
```

The protected identity is selected deterministically from eligible tracks using
face size and temporal coverage. Data splits are assigned at the **source-video
level**, so clips from one source video cannot leak across train/validation/test.

## 3. Prepare local source videos

Place each source video under one directory and name it using its AVA `video_id`:

```text
data/videos/<video_id>.mp4
```

The tool intentionally does not include a downloader. You are responsible for
obtaining videos in a way that follows the source platform's terms and your
institution's research policy.

AVA annotation timestamps refer to the source video's timeline. If your local
file contains the full source video, use the default `--local-time-offset 0`.
If your local file is a 15-minute excerpt that begins at the source video's
15:00 mark, use `--local-time-offset 900`.

## 4. Check local availability before extraction

```bash
python tools/ava_veil_protocol.py materialize \
  --manifest data/ava_veil_pilot/manifest.jsonl \
  --videos-dir data/videos \
  --output-dir data/ava_veil_materialized \
  --local-time-offset 0 \
  --check-only
```

Review `materialize_summary.json`. Fix missing filenames or the timestamp offset
before doing the expensive extraction.

## 5. Materialize clips and target images

Requirements:

- `ffmpeg` available on `PATH`.
- OpenCV installed; it is already listed in `models/requirements.txt`.

```bash
python tools/ava_veil_protocol.py materialize \
  --manifest data/ava_veil_pilot/manifest.jsonl \
  --videos-dir data/videos \
  --output-dir data/ava_veil_materialized \
  --local-time-offset 0
```

Use `--overwrite` only when you intentionally want to recreate existing clips and
reference crops.

## 6. Manual review checklist

Manually inspect at least the first 20 selected clips before running the full
benchmark. Reject or flag a clip when:

- the protected reference crop contains the wrong person;
- a camera cut occurs inside the selected window;
- the protected face is too small or severely occluded for most of the clip;
- the AVA `entity_id` visibly switches between people;
- fewer than two meaningful identities overlap in practice;
- subtitles, overlays, or editing effects dominate the face region.

Store review decisions in a separate CSV rather than editing the generated
manifest by hand. This preserves reproducibility.

Suggested columns:

```text
clip_id,accepted,reason,reviewer
```

## 7. Full protocol generation

After the pilot looks correct, remove `--limit-videos` and run train and
validation archives separately. Keep their outputs separate at first, then merge
accepted manifest rows with a small script or spreadsheet.

A practical first paper-scale target is 80-120 accepted clips, each 5-15 seconds
long, with one protected identity and at least one non-target identity.

## Selection parameters

- `--window-seconds`: clip duration.
- `--step-seconds`: sliding-window stride.
- `--min-faces`: minimum number of eligible identities.
- `--min-overlap-ratio`: fraction of annotated timestamps where at least
  `min-faces` eligible identities are simultaneously visible.
- `--min-entity-coverage`: minimum fraction of window timestamps where an
  identity must be annotated.
- `--min-normalized-face-size`: minimum median face width/height relative to the
  frame dimensions.
- `--max-clips-per-video`: prevents one source video from dominating the set.
- `--max-selected-overlap`: maximum temporal overlap allowed among selected clips
  from the same source video.
- `--seed`: deterministic video-level split assignment.
- `--split-ratios`: for example `train=80,val=10,test=10`.

## Tests

The tests use synthetic AVA-format annotations and do not require source videos.

```bash
python -m unittest discover -s tests -v
```

## Recommended research use

Treat this as an evaluation protocol, not as a newly collected dataset. In the
paper, publish the code, parameters, accepted `video_id`/timestamp manifest, and
review decisions. Do not redistribute source videos unless their individual
licenses explicitly allow it.
