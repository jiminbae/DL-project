import json
from pathlib import Path

def make_face_data(
    frame_idx,
    raw_track_id,
    stable_face_id,
    bbox,
    smoothed_bbox,
    is_target,
    is_background,
    target_sim,
    embedding_ok,
    quality,
    fallback_reasons,
    crop_path=None,
    is_target_direct=None,
    is_target_final=None,
    final_action=None,
    swap_success=None,
    blur_applied=None,
):
    if is_target_final is None:
        is_target_final = is_target

    return {
        "frame_idx": int(frame_idx),
        "raw_track_id": int(raw_track_id),
        "stable_face_id": int(stable_face_id) if stable_face_id is not None else None,
        "bbox": [int(v) for v in bbox],
        "smoothed_bbox": [int(v) for v in smoothed_bbox],
        "is_target": bool(is_target_final),
        "is_target_direct": bool(is_target_direct) if is_target_direct is not None else bool(is_target_final),
        "is_target_final": bool(is_target_final),
        "is_background": bool(is_background),
        "target_similarity": float(target_sim),
        "embedding_ok": bool(embedding_ok),
        "quality": str(quality),
        "fallback_reasons": list(fallback_reasons),
        "crop_path": crop_path,
        "final_action": final_action,
        "swap_success": bool(swap_success) if swap_success is not None else None,
        "blur_applied": bool(blur_applied) if blur_applied is not None else None,
    }


def save_metadata(metadata_path, all_face_metadata):
    metadata_path = Path(metadata_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(all_face_metadata, f, ensure_ascii=False, indent=2)