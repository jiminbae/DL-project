#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
MANIFEST="${MANIFEST:-data/panoptic_veil_materialized_10/materialized_manifest.jsonl}"
REVIEW_CSV="${REVIEW_CSV:-data/panoptic_veil_materialized_10/review/panoptic_10_review.csv}"
REPLACEMENT_IMAGE="${REPLACEMENT_IMAGE:-models/veil/virtual_face/fake_face.jpg}"
OUTPUT_DIR="${OUTPUT_DIR:-data/panoptic_veil_materialized_10/benchmarks/panoptic10_ablation_fixed}"
PAPER_RESULTS_DIR="${PAPER_RESULTS_DIR:-paper_results/panoptic10_ablation_fixed}"
TIMEOUT_SEC="${TIMEOUT_SEC:-1800}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found or not executable: $PYTHON_BIN" >&2
  exit 2
fi

for path in "$MANIFEST" "$REVIEW_CSV" "$REPLACEMENT_IMAGE"; do
  if [[ ! -e "$path" ]]; then
    echo "Required path does not exist: $path" >&2
    exit 2
  fi
done

echo "[1/2] Dry-run benchmark plan"
"$PYTHON_BIN" tools/run_veil_benchmark.py \
  --manifest "$MANIFEST" \
  --review-csv "$REVIEW_CSV" \
  --output-dir "$OUTPUT_DIR" \
  --replacement-image "$REPLACEMENT_IMAGE" \
  --accepted-only \
  --conditions full no_identity_lock no_blur_fallback \
  --python "$PYTHON_BIN" \
  --timeout-sec "$TIMEOUT_SEC" \
  --dry-run

echo "[2/2] Running Panoptic-10 accepted ablation benchmark"
"$PYTHON_BIN" tools/run_veil_benchmark.py \
  --manifest "$MANIFEST" \
  --review-csv "$REVIEW_CSV" \
  --output-dir "$OUTPUT_DIR" \
  --replacement-image "$REPLACEMENT_IMAGE" \
  --accepted-only \
  --conditions full no_identity_lock no_blur_fallback \
  --python "$PYTHON_BIN" \
  --timeout-sec "$TIMEOUT_SEC" \
  --skip-existing

echo "[post] Packaging lightweight sanitized results"
"$PYTHON_BIN" tools/package_benchmark_results.py \
  --benchmark-dir "$OUTPUT_DIR" \
  --output-dir "$PAPER_RESULTS_DIR" \
  --conditions full no_identity_lock no_blur_fallback

echo "Done. Key outputs:"
echo "  $OUTPUT_DIR/benchmark_summary.json"
echo "  $OUTPUT_DIR/condition_metrics.csv"
echo "  $OUTPUT_DIR/<condition>/metadata_eval/aggregate_metrics.json"
echo "  $PAPER_RESULTS_DIR/condition_metrics.csv"
