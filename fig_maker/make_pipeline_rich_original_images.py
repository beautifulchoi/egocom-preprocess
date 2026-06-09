"""
Generate rich original-frame assets for the EgoCom preprocessing pipeline figure.

This output avoids plot panels and title bars. Every visualization is paired with
its source original frame where useful. Rejected cases are saved as rejected-only
mask examples instead of a kept/rejected coexistence panel.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import make_pipeline_figure_examples as base


DEFAULT_OUT_DIR = "/home/prj/data/egocom_holdout/1min/val/pipeline_figure_rich_original_images"

_MASK_CACHE: dict[Path, dict[int, dict[int, np.ndarray]]] = {}
_FRAME_CACHE: dict[Path, list[Path]] = {}
_JSON_CACHE: dict[Path, Any] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate rich original-image pipeline examples.")
    parser.add_argument("--split_root", default=base.DEFAULT_SPLIT_ROOT)
    parser.add_argument("--scene_key", default=base.DEFAULT_SCENE_KEY)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--depth_examples_per_sample", type=int, default=5)
    parser.add_argument("--mapping_frames_per_view", type=int, default=3)
    parser.add_argument("--tracking_frames", type=int, default=5)
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        json.dump(payload, file, indent=2)


def load_mask(path: Path) -> dict[int, dict[int, np.ndarray]]:
    path = Path(path)
    if path not in _MASK_CACHE:
        _MASK_CACHE[path] = base.load_mask_dict(path)
    return _MASK_CACHE[path]


def frame_paths(path: Path) -> list[Path]:
    path = Path(path)
    if path not in _FRAME_CACHE:
        _FRAME_CACHE[path] = base.list_frame_paths(path)
    return _FRAME_CACHE[path]


def load_json(path: Path) -> Any:
    path = Path(path)
    if path not in _JSON_CACHE:
        _JSON_CACHE[path] = base.load_json(path)
    return _JSON_CACHE[path]


def save_image(path: Path, image: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)
    return str(path)


def depth_color_for_frame(depth_map: np.ndarray, frame: np.ndarray) -> np.ndarray:
    depth_color = base.depth_to_colormap(depth_map)
    if depth_color.shape[:2] != frame.shape[:2]:
        depth_color = cv2.resize(depth_color, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
    return depth_color


def mask_only(frame: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.55) -> np.ndarray:
    mask_bool = mask.astype(bool)
    out = (frame.astype(np.float32) * 0.28).astype(np.uint8)
    out[mask_bool] = frame[mask_bool]
    return base.overlay_mask(out, mask_bool, color, alpha=alpha, label=None)


def masked_person(frame: np.ndarray, persons: dict[int, np.ndarray], person: base.SegmentMask) -> np.ndarray | None:
    mask = base.union_segments(persons, person.segment_ids)
    if mask is None:
        return None
    return mask_only(frame, mask, base.person_color(person.person_id), alpha=0.45)


def clip_remap_payload(split_root: Path, scene_key: str, chunk: int) -> dict[str, Any]:
    remap_path = split_root / "person_face_mapping" / scene_key / "remap_all_chunks.json"
    remap = load_json(remap_path)
    return (remap.get("chunks") or {}).get(str(chunk), {})


def mapping_views(split_root: Path, example: base.PipelineExample) -> list[dict[str, Any]]:
    chunk_payload = clip_remap_payload(split_root, example.scene_key, example.chunk)
    preferred = [example.source.video_name, example.target.video_name]
    views: list[dict[str, Any]] = []
    video_order = preferred + [name for name in sorted(chunk_payload) if name not in preferred]
    for video_name in video_order:
        payload = chunk_payload.get(video_name)
        if not isinstance(payload, dict):
            continue
        people = base.remap_people_from_payload(payload)[:2]
        if len(people) < 2:
            continue
        mask_path = split_root / "person_mask" / video_name / "masks.pt"
        frames = frame_paths(split_root / "frame" / video_name)
        if not mask_path.exists() or not frames:
            continue
        masks = load_mask(mask_path)
        preferred_idx = example.source.frame_idx if video_name == example.source.video_name else example.target.frame_idx
        valid_indices = [
            idx for idx in range(len(frames))
            if all(base.union_segments(masks.get(idx, {}), person.segment_ids) is not None for person in people)
        ]
        if not valid_indices:
            continue
        valid_indices = sorted(valid_indices, key=lambda idx: (abs(idx - preferred_idx), idx))
        views.append(
            {
                "video_name": video_name,
                "camera_person": int(payload.get("camera_person", -1)),
                "people": people,
                "frame_paths": frames,
                "mask_dict": masks,
                "valid_indices": valid_indices,
            }
        )
        if len(views) >= 2:
            return views
    return views


def spread_indices(indices: list[int], count: int) -> list[int]:
    if count <= 0 or not indices:
        return []
    if len(indices) <= count:
        return indices
    positions = np.linspace(0, len(indices) - 1, count).round().astype(int).tolist()
    out: list[int] = []
    for pos in positions:
        idx = indices[pos]
        if idx not in out:
            out.append(idx)
    for idx in indices:
        if len(out) >= count:
            break
        if idx not in out:
            out.append(idx)
    return out[:count]


def depth_candidates(split_root: Path, video_name: str, preferred_idx: int, limit: int) -> list[dict[str, Any]]:
    raw_path = split_root / "person_mask" / video_name / "masks.pt"
    refined_path = split_root / "refined_mask" / video_name / "mask.pt"
    depth_dir = split_root / "da3" / "monocular" / video_name / "depth"
    frames = frame_paths(split_root / "frame" / video_name)
    if not raw_path.exists() or not refined_path.exists() or not frames:
        return []
    raw = load_mask(raw_path)
    refined = load_mask(refined_path)
    frame_order = sorted(range(len(frames)), key=lambda idx: (abs(idx - preferred_idx), idx))
    candidates: list[dict[str, Any]] = []
    used: set[tuple[int, int]] = set()
    for frame_idx in frame_order:
        depth_path = depth_dir / f"{frames[frame_idx].stem}.npy"
        if not depth_path.exists():
            continue
        raw_persons = raw.get(frame_idx, {})
        refined_persons = refined.get(frame_idx, {})
        rejected_ids = sorted(set(raw_persons) - set(refined_persons))
        if not rejected_ids:
            continue
        rejected_id = max(rejected_ids, key=lambda sid: int(raw_persons[sid].sum()))
        key = (frame_idx, int(rejected_id))
        if key in used:
            continue
        used.add(key)
        kept_ids = sorted(refined_persons)
        kept_id = max(kept_ids, key=lambda sid: int(refined_persons[sid].sum())) if kept_ids else None
        candidates.append(
            {
                "frame_idx": frame_idx,
                "frame_path": frames[frame_idx],
                "depth_path": depth_path,
                "kept_segment_id": None if kept_id is None else int(kept_id),
                "rejected_segment_id": int(rejected_id),
                "raw_persons": raw_persons,
                "refined_persons": refined_persons,
            }
        )
        if len(candidates) >= limit:
            return candidates
    return candidates


def save_depth_examples(split_root: Path, sample_dir: Path, example: base.PipelineExample, count: int) -> list[dict[str, Any]]:
    out_dir = sample_dir / "02_depth_refine_examples"
    entries: list[dict[str, Any]] = []
    candidates = depth_candidates(split_root, example.source.video_name, example.source.frame_idx, count)
    for case_idx, item in enumerate(candidates, start=1):
        case_dir = out_dir / f"example_{case_idx:02d}"
        frame = cv2.imread(str(item["frame_path"]), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        depth_map = np.load(item["depth_path"])
        depth_color = depth_color_for_frame(depth_map, frame)
        rejected_mask = item["raw_persons"].get(item["rejected_segment_id"])
        kept_mask = None
        if item["kept_segment_id"] is not None:
            kept_mask = item["refined_persons"].get(item["kept_segment_id"])

        paths = {
            "source_original": save_image(case_dir / "source_original.png", frame),
            "depth_colormap": save_image(case_dir / "depth_colormap.png", depth_color),
        }
        if kept_mask is not None:
            paths["not_rejected_segment"] = save_image(
                case_dir / f"not_rejected_segment_{item['kept_segment_id']}.png",
                mask_only(frame, kept_mask, base.KEEP_COLOR_BGR, alpha=0.50),
            )
        if rejected_mask is not None:
            paths["rejected_segment_only"] = save_image(
                case_dir / f"rejected_segment_{item['rejected_segment_id']}_only.png",
                mask_only(frame, rejected_mask, base.REJECT_COLOR_BGR, alpha=0.62),
            )
            paths["rejected_on_depth"] = save_image(
                case_dir / f"rejected_segment_{item['rejected_segment_id']}_on_depth.png",
                base.overlay_mask(depth_color, rejected_mask, base.REJECT_COLOR_BGR, alpha=0.62, label=None),
            )
        entries.append(
            {
                "case_index": case_idx,
                "video_name": example.source.video_name,
                "frame_idx": item["frame_idx"],
                "frame_path": str(item["frame_path"]),
                "depth_path": str(item["depth_path"]),
                "kept_segment_id": item["kept_segment_id"],
                "rejected_segment_id": item["rejected_segment_id"],
                "paths": paths,
                "note": "Rejected examples are rejected-only false-positive candidates, e.g. mirror/picture/reflection-like cases when present in the source data.",
            }
        )
    return entries


def save_mapping_examples(
    split_root: Path,
    sample_dir: Path,
    example: base.PipelineExample,
    frames_per_view: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    out_dir = sample_dir / "03_common_person_mapping_examples"
    crop_dir = sample_dir / "04_cropped_found_common_person_examples"
    mapping_entries: list[dict[str, Any]] = []
    crop_entries: list[dict[str, Any]] = []
    for view_idx, view in enumerate(mapping_views(split_root, example), start=1):
        selected = spread_indices(view["valid_indices"], frames_per_view)
        for local_idx, frame_idx in enumerate(selected, start=1):
            frame_path = view["frame_paths"][frame_idx]
            frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            persons = view["mask_dict"].get(frame_idx, {})
            frame_dir = out_dir / f"view_{view_idx}_camera_person_{view['camera_person']}" / f"frame_example_{local_idx:02d}"
            original_path = save_image(frame_dir / "source_original.png", frame)
            person_entries = []
            for person in view["people"][:2]:
                masked = masked_person(frame, persons, person)
                if masked is None:
                    continue
                masked_path = save_image(frame_dir / f"captured_person_{person.person_id}_masked.png", masked)
                person_entries.append(
                    {
                        "captured_person_id": person.person_id,
                        "segment_ids": list(person.segment_ids),
                        "masked_path": masked_path,
                    }
                )
                mask = base.union_segments(persons, person.segment_ids)
                bbox = base.mask_bbox(mask, pad=18) if mask is not None else None
                if bbox is not None:
                    x1, y1, x2, y2 = bbox
                    crop = frame[y1:y2, x1:x2].copy()
                    crop_mask = mask[y1:y2, x1:x2]
                    crop[~crop_mask] = (crop[~crop_mask].astype(np.float32) * 0.18).astype(np.uint8)
                    crop = base.overlay_mask(crop, crop_mask, base.person_color(person.person_id), alpha=0.40, label=None)
                    crop_path = save_image(
                        crop_dir / f"view_{view_idx}_frame_{local_idx:02d}_captured_person_{person.person_id}.png",
                        crop,
                    )
                    crop_entries.append(
                        {
                            "view_index": view_idx,
                            "camera_person": view["camera_person"],
                            "captured_person_id": person.person_id,
                            "frame_idx": frame_idx,
                            "frame_path": str(frame_path),
                            "bbox": list(map(int, bbox)),
                            "path": crop_path,
                        }
                    )
            mapping_entries.append(
                {
                    "view_index": view_idx,
                    "camera_person": view["camera_person"],
                    "video_name": view["video_name"],
                    "frame_example_index": local_idx,
                    "frame_idx": frame_idx,
                    "frame_path": str(frame_path),
                    "source_original_path": original_path,
                    "people": person_entries,
                }
            )
    return mapping_entries, crop_entries


def temporal_indices(center_idx: int, num_frames: int, count: int) -> list[int]:
    if count <= 0 or num_frames <= 0:
        return []
    half = count // 2
    indices = [idx for idx in range(center_idx - half, center_idx + half + 1) if 0 <= idx < num_frames]
    while len(indices) < count and indices:
        if indices[0] > 0:
            indices.insert(0, indices[0] - 1)
        elif indices[-1] + 1 < num_frames:
            indices.append(indices[-1] + 1)
        else:
            break
    return indices[:count]


def save_tracking_examples(split_root: Path, sample_dir: Path, example: base.PipelineExample, count: int) -> list[dict[str, Any]]:
    out_dir = sample_dir / "05_tracked_person_masks_temporal_both_views"
    entries: list[dict[str, Any]] = []
    for view_index, view in enumerate(mapping_views(split_root, example)[:2], start=1):
        refined_path = split_root / "refined_mask" / view["video_name"] / "mask.pt"
        if not refined_path.exists():
            continue
        refined = load_mask(refined_path)
        frames = view["frame_paths"]
        if not frames:
            continue
        center_idx = int(view["valid_indices"][0])
        view_dir = out_dir / f"view_{view_index}_camera_person_{view['camera_person']}"
        frame_entries = []
        for frame_idx in temporal_indices(center_idx, len(frames), count):
            frame = cv2.imread(str(frames[frame_idx]), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            overlay = frame.copy()
            for person in view["people"][:2]:
                mask = base.union_segments(refined.get(frame_idx, {}), person.segment_ids)
                if mask is not None:
                    overlay = base.overlay_mask(
                        overlay,
                        mask,
                        base.person_color(person.person_id),
                        alpha=0.50,
                        label=None,
                    )
            frame_dir = view_dir / f"frame_{frame_idx:06d}"
            frame_entries.append(
                {
                    "frame_idx": frame_idx,
                    "frame_path": str(frames[frame_idx]),
                    "source_original_path": save_image(frame_dir / "source_original.png", frame),
                    "tracked_overlay_path": save_image(frame_dir / "tracked_overlay.png", overlay),
                }
            )
        entries.append(
            {
                "view_index": view_index,
                "camera_person": view["camera_person"],
                "video_name": view["video_name"],
                "people": [
                    {"person_id": person.person_id, "segment_ids": list(person.segment_ids)}
                    for person in view["people"][:2]
                ],
                "center_frame_idx": center_idx,
                "frames": frame_entries,
            }
        )
    return entries


def valid_example(split_root: Path, example: base.PipelineExample, depth_count: int, mapping_frames: int) -> bool:
    if len(depth_candidates(split_root, example.source.video_name, example.source.frame_idx, depth_count)) < depth_count:
        return False
    views = mapping_views(split_root, example)
    if len(views) < 2:
        return False
    return all(len(view["valid_indices"]) >= mapping_frames for view in views[:2])


def save_sample(split_root: Path, out_dir: Path, example: base.PipelineExample, args: argparse.Namespace) -> dict[str, Any]:
    sample_dir = out_dir / f"sample_{example.sample_index:02d}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    raw_frame = cv2.imread(str(example.source.frame_path), cv2.IMREAD_COLOR)
    raw_path = None
    if raw_frame is not None:
        raw_path = save_image(sample_dir / "01_raw_before_tracking_original.png", raw_frame)
    depth_entries = save_depth_examples(split_root, sample_dir, example, args.depth_examples_per_sample)
    mapping_entries, crop_entries = save_mapping_examples(split_root, sample_dir, example, args.mapping_frames_per_view)
    tracking_entries = save_tracking_examples(split_root, sample_dir, example, args.tracking_frames)
    metadata = {
        "sample_index": example.sample_index,
        "scene_key": example.scene_key,
        "chunk": example.chunk,
        "source_video_name": example.source.video_name,
        "target_video_name": example.target.video_name,
        "raw_before_tracking_original_path": raw_path,
        "depth_refine_examples": depth_entries,
        "common_person_mapping_examples": mapping_entries,
        "cropped_found_common_person_examples": crop_entries,
        "tracked_person_masks_temporal": tracking_entries,
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

    candidates = base.collect_examples(split_root, args.scene_key, max(args.num_samples * 8, args.num_samples))
    examples: list[base.PipelineExample] = []
    for candidate in candidates:
        if not valid_example(split_root, candidate, args.depth_examples_per_sample, args.mapping_frames_per_view):
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
        raise RuntimeError(f"Found only {len(examples)} valid rich samples; requested {args.num_samples}")

    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = [save_sample(split_root, out_dir, example, args) for example in examples]
    write_json(
        out_dir / "metadata.json",
        {
            "split_root": str(split_root),
            "scene_key": args.scene_key,
            "num_samples": len(metadata),
            "depth_examples_per_sample": args.depth_examples_per_sample,
            "mapping_frames_per_view": args.mapping_frames_per_view,
            "tracking_frames": args.tracking_frames,
            "samples": metadata,
        },
    )
    print(f"Saved {len(metadata)} rich original-image samples to {out_dir}")


if __name__ == "__main__":
    main()
