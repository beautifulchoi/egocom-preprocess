"""
Create clean original-image examples for the EgoCom preprocessing pipeline.

Outputs are raw/overlay/crop/tracking image files without title bars, borders,
resized panels, or plot-style composites.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import make_pipeline_figure_examples as base


DEFAULT_OUT_DIR = "/home/prj/data/egocom_holdout/1min/val/pipeline_figure_original_images"

_ORIGINAL_LOAD_MASK_DICT = base.load_mask_dict
_ORIGINAL_LIST_FRAME_PATHS = base.list_frame_paths
_MASK_CACHE: dict[Path, dict[int, dict[int, np.ndarray]]] = {}
_FRAME_PATH_CACHE: dict[Path, list[Path]] = {}


def cached_mask_dict(path: Path) -> dict[int, dict[int, np.ndarray]]:
    path = Path(path)
    if path not in _MASK_CACHE:
        _MASK_CACHE[path] = _ORIGINAL_LOAD_MASK_DICT(path)
    return _MASK_CACHE[path]


def cached_frame_paths(frame_dir: Path) -> list[Path]:
    frame_dir = Path(frame_dir)
    if frame_dir not in _FRAME_PATH_CACHE:
        _FRAME_PATH_CACHE[frame_dir] = _ORIGINAL_LIST_FRAME_PATHS(frame_dir)
    return _FRAME_PATH_CACHE[frame_dir]


def install_base_caches() -> None:
    base.load_mask_dict = cached_mask_dict
    base.list_frame_paths = cached_frame_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate clean original-image assets for the EgoCom preprocessing figure."
    )
    parser.add_argument("--split_root", default=base.DEFAULT_SPLIT_ROOT)
    parser.add_argument("--scene_key", default=base.DEFAULT_SCENE_KEY)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--tracking_frames", type=int, default=5)
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        json.dump(payload, file, indent=2)


def depth_refine_original(
    frame: np.ndarray,
    depth_map: np.ndarray,
    raw_persons: dict[int, np.ndarray],
    refined_persons: dict[int, np.ndarray],
    kept_segment_id: int,
    rejected_segment_id: int,
) -> np.ndarray:
    depth_color = base.depth_to_colormap(depth_map)
    if depth_color.shape[:2] != frame.shape[:2]:
        depth_color = cv2.resize(
            depth_color,
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    out = cv2.addWeighted(frame, 0.48, depth_color, 0.52, 0.0)
    kept_mask = refined_persons.get(kept_segment_id)
    rejected_mask = raw_persons.get(rejected_segment_id)
    if kept_mask is not None:
        out = base.overlay_mask(out, kept_mask, base.KEEP_COLOR_BGR, alpha=0.50, label=None)
    if rejected_mask is not None:
        out = base.overlay_mask(out, rejected_mask, base.REJECT_COLOR_BGR, alpha=0.62, label=None)
    return out


def segment_case_original(
    frame: np.ndarray,
    mask: np.ndarray | None,
    color: tuple[int, int, int],
) -> np.ndarray | None:
    if mask is None:
        return None
    out = (frame.astype(np.float32) * 0.30).astype(np.uint8)
    mask_bool = mask.astype(bool)
    out[mask_bool] = frame[mask_bool]
    return base.overlay_mask(out, mask_bool, color, alpha=0.55, label=None)


def masked_person_original(
    frame: np.ndarray,
    persons: dict[int, np.ndarray],
    person: base.SegmentMask,
) -> np.ndarray | None:
    mask = base.union_segments(persons, person.segment_ids)
    if mask is None:
        return None
    out = (frame.astype(np.float32) * 0.28).astype(np.uint8)
    out[mask] = frame[mask]
    return base.overlay_mask(
        out,
        mask,
        base.person_color(person.person_id),
        alpha=0.45,
        label=None,
    )


def find_frame_with_people(
    split_root: Path,
    video_name: str,
    people: tuple[base.SegmentMask, ...],
    preferred_frame_idx: int,
) -> tuple[int, Path, np.ndarray, dict[int, np.ndarray]] | None:
    mask_path = split_root / "person_mask" / video_name / "masks.pt"
    frame_paths = cached_frame_paths(split_root / "frame" / video_name)
    if not mask_path.exists() or not frame_paths:
        return None
    mask_dict = cached_mask_dict(mask_path)
    search_order = sorted(
        range(len(frame_paths)),
        key=lambda idx: (abs(idx - preferred_frame_idx), idx),
    )
    for frame_idx in search_order[:120]:
        persons = mask_dict.get(frame_idx, {})
        if not all(base.union_segments(persons, item.segment_ids) is not None for item in people[:2]):
            continue
        frame = cv2.imread(str(frame_paths[frame_idx]), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        return frame_idx, frame_paths[frame_idx], frame, persons
    return None


def mapping_views_for_example(
    split_root: Path,
    example: base.PipelineExample,
) -> list[dict[str, Any]]:
    remap_path = split_root / "person_face_mapping" / example.scene_key / "remap_all_chunks.json"
    remap = base.load_json(remap_path)
    chunk_payload = (remap.get("chunks") or {}).get(str(example.chunk), {})
    views: list[dict[str, Any]] = []

    preferred_videos = [example.source.video_name, example.target.video_name]
    for video_name in preferred_videos:
        payload = chunk_payload.get(video_name)
        if not isinstance(payload, dict):
            continue
        people = base.remap_people_from_payload(payload)[:2]
        if len(people) < 2:
            continue
        found = find_frame_with_people(
            split_root=split_root,
            video_name=video_name,
            people=people,
            preferred_frame_idx=(
                example.source.frame_idx if video_name == example.source.video_name else example.target.frame_idx
            ),
        )
        if found is None:
            continue
        frame_idx, frame_path, frame, persons = found
        views.append(
            {
                "video_name": video_name,
                "camera_person": int(payload.get("camera_person", -1)),
                "frame_idx": frame_idx,
                "frame_path": frame_path,
                "frame": frame,
                "persons": persons,
                "people": people,
            }
        )
        if len(views) == 2:
            return views

    for video_name, payload in chunk_payload.items():
        if any(view["video_name"] == video_name for view in views):
            continue
        if not isinstance(payload, dict):
            continue
        people = base.remap_people_from_payload(payload)[:2]
        if len(people) < 2:
            continue
        found = find_frame_with_people(
            split_root=split_root,
            video_name=video_name,
            people=people,
            preferred_frame_idx=example.target.frame_idx,
        )
        if found is None:
            continue
        frame_idx, frame_path, frame, persons = found
        views.append(
            {
                "video_name": video_name,
                "camera_person": int(payload.get("camera_person", -1)),
                "frame_idx": frame_idx,
                "frame_path": frame_path,
                "frame": frame,
                "persons": persons,
                "people": people,
            }
        )
        if len(views) == 2:
            return views
    return views


def crop_person_originals(
    frame: np.ndarray,
    persons: dict[int, np.ndarray],
    people: tuple[base.SegmentMask, ...],
) -> list[tuple[int, np.ndarray, tuple[int, int, int, int]]]:
    crops: list[tuple[int, np.ndarray, tuple[int, int, int, int]]] = []
    for item in people:
        mask = base.union_segments(persons, item.segment_ids)
        if mask is None:
            continue
        bbox = base.mask_bbox(mask, pad=18)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        crop = frame[y1:y2, x1:x2].copy()
        crop_mask = mask[y1:y2, x1:x2]
        crop[~crop_mask] = (crop[~crop_mask].astype(np.float32) * 0.18).astype(np.uint8)
        crop = base.overlay_mask(
            crop,
            crop_mask,
            base.person_color(item.person_id),
            alpha=0.40,
            label=None,
        )
        crops.append((item.person_id, crop, bbox))
    return crops


def tracking_originals(
    frame_paths: list[Path],
    refined_mask: dict[int, dict[int, np.ndarray]],
    center_idx: int,
    people: tuple[base.SegmentMask, ...],
    strip_count: int,
) -> list[tuple[int, Path, np.ndarray]]:
    strip_count = max(1, strip_count)
    half = strip_count // 2
    frame_indices = [center_idx + offset for offset in range(-half, half + 1)]
    frame_indices = [idx for idx in frame_indices if 0 <= idx < len(frame_paths)]
    while len(frame_indices) < strip_count and frame_indices:
        if frame_indices[0] > 0:
            frame_indices.insert(0, frame_indices[0] - 1)
        elif frame_indices[-1] + 1 < len(frame_paths):
            frame_indices.append(frame_indices[-1] + 1)
        else:
            break
    frame_indices = frame_indices[:strip_count]

    outputs: list[tuple[int, Path, np.ndarray]] = []
    for idx in frame_indices:
        frame = cv2.imread(str(frame_paths[idx]), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        out = frame.copy()
        persons = refined_mask.get(idx, {})
        for item in people:
            mask = base.union_segments(persons, item.segment_ids)
            if mask is None:
                continue
            out = base.overlay_mask(
                out,
                mask,
                base.person_color(item.person_id),
                alpha=0.50,
                label=None,
            )
        outputs.append((idx, frame_paths[idx], out))
    return outputs


def save_sample(
    split_root: Path,
    out_dir: Path,
    example: base.PipelineExample,
    tracking_frames: int,
) -> dict[str, Any]:
    sample_dir = out_dir / f"sample_{example.sample_index:02d}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    source_raw = cached_mask_dict(split_root / "person_mask" / example.source.video_name / "masks.pt")
    source_refined = cached_mask_dict(split_root / "refined_mask" / example.source.video_name / "mask.pt")
    target_raw = cached_mask_dict(split_root / "person_mask" / example.target.video_name / "masks.pt")
    target_refined_path = split_root / "refined_mask" / example.target.video_name / "mask.pt"
    target_refined = cached_mask_dict(target_refined_path) if target_refined_path.exists() else {}

    source_frame = cv2.imread(str(example.source.frame_path), cv2.IMREAD_COLOR)
    if source_frame is None:
        raise FileNotFoundError(f"Could not read frame: {example.source.frame_path}")
    depth_map = np.load(example.source.depth_path)

    target_frame_paths = cached_frame_paths(split_root / "frame" / example.target.video_name)
    target_frame_path = target_frame_paths[example.target.frame_idx]
    target_frame = cv2.imread(str(target_frame_path), cv2.IMREAD_COLOR)
    if target_frame is None:
        raise FileNotFoundError(f"Could not read frame: {target_frame_path}")

    raw_path = sample_dir / "01_raw_original.png"
    depth_refine_path = sample_dir / "02_depth_refine_kept_and_rejected.png"
    cv2.imwrite(str(raw_path), source_frame)
    source_raw_persons = source_raw.get(example.source.frame_idx, {})
    source_refined_persons = source_refined.get(example.source.frame_idx, {})
    cv2.imwrite(
        str(depth_refine_path),
        depth_refine_original(
            source_frame,
            depth_map,
            source_raw_persons,
            source_refined_persons,
            example.source.kept_segment_id,
            example.source.rejected_segment_id,
        ),
    )

    refine_case_dir = sample_dir / "02_depth_refine_cases"
    refine_case_dir.mkdir(exist_ok=True)
    refine_case_entries = []
    kept_case = segment_case_original(
        source_frame,
        source_refined_persons.get(example.source.kept_segment_id),
        base.KEEP_COLOR_BGR,
    )
    if kept_case is not None:
        kept_case_path = refine_case_dir / f"not_rejected_segment_{example.source.kept_segment_id}.png"
        cv2.imwrite(str(kept_case_path), kept_case)
        refine_case_entries.append(
            {
                "status": "not_rejected",
                "segment_id": example.source.kept_segment_id,
                "path": str(kept_case_path),
            }
        )
    rejected_case = segment_case_original(
        source_frame,
        source_raw_persons.get(example.source.rejected_segment_id),
        base.REJECT_COLOR_BGR,
    )
    if rejected_case is not None:
        rejected_case_path = refine_case_dir / f"rejected_segment_{example.source.rejected_segment_id}.png"
        cv2.imwrite(str(rejected_case_path), rejected_case)
        refine_case_entries.append(
            {
                "status": "rejected",
                "segment_id": example.source.rejected_segment_id,
                "path": str(rejected_case_path),
            }
        )

    mapping_dir = sample_dir / "03_common_person_mapping_masked"
    mapping_dir.mkdir(exist_ok=True)
    mapping_entries = []
    for view_index, view in enumerate(mapping_views_for_example(split_root, example), start=1):
        for person in view["people"][:2]:
            masked = masked_person_original(view["frame"], view["persons"], person)
            if masked is None:
                continue
            mapping_path = (
                mapping_dir
                / f"view_{view_index}_camera_person_{view['camera_person']}_captured_person_{person.person_id}.png"
            )
            cv2.imwrite(str(mapping_path), masked)
            mapping_entries.append(
                {
                    "view_index": view_index,
                    "video_name": view["video_name"],
                    "camera_person": view["camera_person"],
                    "captured_person_id": person.person_id,
                    "segment_ids": list(person.segment_ids),
                    "frame_idx": view["frame_idx"],
                    "frame_path": str(view["frame_path"]),
                    "path": str(mapping_path),
                }
            )

    crop_dir = sample_dir / "04_cropped_found_common_person"
    crop_dir.mkdir(exist_ok=True)
    crop_entries = []
    for person_id, crop, bbox in crop_person_originals(
        target_frame,
        target_raw.get(example.target.frame_idx, {}),
        example.target.people,
    ):
        crop_path = crop_dir / f"person_{person_id}.png"
        cv2.imwrite(str(crop_path), crop)
        crop_entries.append(
            {"person_id": person_id, "path": str(crop_path), "bbox": list(map(int, bbox)), "shape": list(crop.shape)}
        )

    tracking_dir = sample_dir / "05_tracked_person_masks_temporal"
    tracking_dir.mkdir(exist_ok=True)
    tracking_entries = []
    for frame_idx, frame_path, image in tracking_originals(
        target_frame_paths,
        target_refined,
        example.target.frame_idx,
        example.target.people,
        tracking_frames,
    ):
        out_path = tracking_dir / f"frame_{frame_idx:06d}.png"
        cv2.imwrite(str(out_path), image)
        tracking_entries.append({"frame_idx": frame_idx, "source_frame_path": str(frame_path), "path": str(out_path)})

    metadata = {
        "sample_index": example.sample_index,
        "scene_key": example.scene_key,
        "chunk": example.chunk,
        "source": {
            "video_name": example.source.video_name,
            "frame_idx": example.source.frame_idx,
            "frame_path": str(example.source.frame_path),
            "depth_path": str(example.source.depth_path),
            "kept_segment_id": example.source.kept_segment_id,
            "rejected_segment_id": example.source.rejected_segment_id,
        },
        "target": {
            "video_name": example.target.video_name,
            "camera_person": example.target.camera_person,
            "frame_idx": example.target.frame_idx,
            "frame_path": str(target_frame_path),
            "people": [
                {"person_id": item.person_id, "segment_ids": list(item.segment_ids)}
                for item in example.target.people
            ],
        },
        "outputs": {
            "raw_original": str(raw_path),
            "depth_refine_kept_and_rejected": str(depth_refine_path),
            "depth_refine_cases": refine_case_entries,
            "common_person_mapping_masked": mapping_entries,
            "cropped_found_common_person": crop_entries,
            "tracked_person_masks_temporal": tracking_entries,
        },
    }
    write_json(sample_dir / "metadata.json", metadata)
    return metadata


def main() -> None:
    args = parse_args()
    split_root = Path(args.split_root)
    out_dir = Path(args.out_dir)
    if args.num_samples < 1:
        raise ValueError("--num_samples must be >= 1")
    if not split_root.is_dir():
        raise FileNotFoundError(f"Missing split root: {split_root}")

    candidate_count = max(args.num_samples * 4, args.num_samples)
    candidates = base.collect_examples(split_root, args.scene_key, candidate_count)
    examples: list[base.PipelineExample] = []
    for candidate in candidates:
        views = mapping_views_for_example(split_root, candidate)
        if len(views) < 2:
            continue
        if not all(len(view["people"]) >= 2 for view in views[:2]):
            continue
        examples.append(
            base.PipelineExample(
                sample_index=len(examples) + 1,
                scene_key=candidate.scene_key,
                chunk=candidate.chunk,
                source=candidate.source,
                target=candidate.target,
            )
        )
        if len(examples) >= args.num_samples:
            break
    if len(examples) < args.num_samples:
        raise RuntimeError(
            f"Found only {len(examples)} examples with two valid mapping views; requested {args.num_samples}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = [save_sample(split_root, out_dir, example, args.tracking_frames) for example in examples]
    write_json(
        out_dir / "metadata.json",
        {
            "split_root": str(split_root),
            "scene_key": args.scene_key,
            "num_samples": len(metadata),
            "tracking_frames": args.tracking_frames,
            "samples": metadata,
        },
    )
    print(f"Saved {len(metadata)} clean original-image samples to {out_dir}")


if __name__ == "__main__":
    main()
