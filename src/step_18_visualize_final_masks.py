"""
Visualize EgoCom whole-frame masks before filtering and after person remapping.

Inputs:
  /home/prj/data/egocom_holdout/1min/{split}/person_mask/{clip}/masks.pt
  /home/prj/data/egocom_holdout/1min/{split}/refined_mask/{clip}/mask.pt
  /home/prj/data/egocom_holdout/1min/{split}/person_face_mapping/{scene}/remap_all_chunks.json
  /home/prj/data/egocom_holdout/1min/{split}/frame/{clip}/*.jpg

Outputs:
  /home/prj/data/egocom_holdout/1min/{split}/final_vis/{scene}/{clip}/before/*.jpg
  /home/prj/data/egocom_holdout/1min/{split}/final_vis/{scene}/{clip}/after/*.jpg
  /home/prj/data/egocom_holdout/1min/{split}/final_vis/{scene}/{clip}/diff/*.jpg
  /home/prj/data/egocom_holdout/1min/{split}/final_vis/{scene}/{clip}/raw_frame/*.jpg
  /home/prj/data/egocom_holdout/1min/{split}/final_vis/{scene}/{clip}/masked_frame/*.jpg
  /home/prj/data/egocom_holdout/1min/{split}/final_vis/{scene}/{clip}/raw_mask_bw/*.jpg
  /home/prj/data/egocom_holdout/1min/{split}/final_vis/{scene}/{clip}/rejected_mask/*.jpg
  /home/prj/data/egocom_holdout/1min/{split}/final_vis/{scene}/{clip}/merged_track/*.jpg
  /home/prj/data/egocom_holdout/1min/{split}/final_vis/{scene}/{clip}/summary.json
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


DEFAULT_DATA_ROOT = "/home/prj/data/egocom_holdout/1min"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
CLIP_RE = re.compile(
    r"^(vid_\d+)__day_(?P<day>\d+)__con_(?P<con>\d+)__person_(?P<camera>\d+)"
    r"(?:_(?P<part>part\d+))?_chunk_(?P<chunk>\d+)$"
)
PERSON_COLORS_BGR = {
    1: (0, 0, 255),
    2: (0, 180, 0),
    3: (255, 0, 0),
}
SEGMENT_FALLBACK_COLORS_BGR = [
    (0, 180, 180),
    (180, 0, 180),
    (180, 180, 0),
    (70, 120, 255),
]


@dataclass(frozen=True)
class ClipJob:
    split: str
    scene_key: str
    clip_name: str
    camera_person: int
    chunk: int
    people_to_segments: dict[int, list[int]]
    frame_dir: Path
    raw_mask_path: Path
    refined_mask_path: Path
    output_dir: Path


def parse_comma_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_int_list(raw: str) -> list[int]:
    values: list[int] = []
    for item in parse_comma_list(raw):
        values.append(int(item))
    return values


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be >= 1")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return parsed


def selection_mode(value: str) -> str:
    choices = {"sampled", "changed", "changed_or_sampled"}
    if value not in choices:
        raise argparse.ArgumentTypeError(f"value must be one of: {', '.join(sorted(choices))}")
    return value


def change_scope(value: str) -> str:
    choices = {"mapped", "all"}
    if value not in choices:
        raise argparse.ArgumentTypeError(f"value must be one of: {', '.join(sorted(choices))}")
    return value


def parse_clip_name(clip_name: str) -> dict[str, Any] | None:
    match = CLIP_RE.match(clip_name)
    if match is None:
        return None

    day = match.group("day")
    con = match.group("con")
    part = match.group("part")
    scene_key = f"day_{day}__con_{con}"
    if part:
        scene_key = f"{scene_key}__{part}"

    return {
        "scene_key": scene_key,
        "camera_person": int(match.group("camera")),
        "chunk": int(match.group("chunk")),
    }


def get_safe_numpy_globals() -> list[Any]:
    safe: list[Any] = []
    try:
        import numpy._core.multiarray as multiarray

        safe.append(multiarray._reconstruct)
    except Exception:
        try:
            import numpy.core.multiarray as multiarray

            safe.append(multiarray._reconstruct)
        except Exception:
            pass

    for value in (
        np.ndarray,
        np.dtype,
        getattr(getattr(np, "dtypes", object), "BoolDType", None),
    ):
        if value is not None:
            safe.append(value)
    return safe


def load_torch(path: Path) -> Any:
    try:
        with torch.serialization.safe_globals(get_safe_numpy_globals()):
            return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def load_mask_dict(mask_path: Path) -> dict[int, dict[int, np.ndarray]]:
    raw = load_torch(mask_path)
    if not isinstance(raw, dict):
        raise TypeError(f"Expected mask dict in {mask_path}, got {type(raw)}")

    mask_dict: dict[int, dict[int, np.ndarray]] = {}
    for frame_idx_raw, persons_raw in raw.items():
        if not isinstance(persons_raw, dict):
            continue
        persons: dict[int, np.ndarray] = {}
        for segment_id_raw, mask_raw in persons_raw.items():
            mask = np.asarray(mask_raw).astype(bool)
            if mask.ndim == 3 and mask.shape[0] == 1:
                mask = mask[0]
            if mask.ndim != 2 or not mask.any():
                continue
            persons[int(segment_id_raw)] = mask
        if persons:
            mask_dict[int(frame_idx_raw)] = persons
    return mask_dict


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r") as file:
        return json.load(file)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        json.dump(payload, file, indent=2)


def list_frame_paths(frame_dir: Path) -> list[Path]:
    if not frame_dir.is_dir():
        return []
    return sorted(
        path
        for path in frame_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def discover_splits(data_root: Path, split_arg: str) -> list[str]:
    if split_arg != "all_existing":
        return parse_comma_list(split_arg)
    return sorted(
        split_dir.name
        for split_dir in data_root.iterdir()
        if split_dir.is_dir() and (split_dir / "person_face_mapping").is_dir()
    )


def people_to_segments_from_clip(clip_payload: dict[str, Any]) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    people = clip_payload.get("people", {})
    if not isinstance(people, dict):
        return out
    for person_id_raw, person_payload in people.items():
        if not isinstance(person_payload, dict):
            continue
        segment_ids = person_payload.get("merged_segment_ids", [])
        if not isinstance(segment_ids, list):
            segment_ids = []
        if not segment_ids and person_payload.get("primary_segment_id") is not None:
            segment_ids = [person_payload["primary_segment_id"]]
        parsed_segments = sorted({int(segment_id) for segment_id in segment_ids})
        if parsed_segments:
            out[int(person_id_raw)] = parsed_segments
    return out


def collect_jobs(args: argparse.Namespace) -> list[ClipJob]:
    data_root = Path(args.data_root)
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    selected_camera_persons = set(parse_int_list(args.camera_persons))
    jobs: list[ClipJob] = []
    for split in discover_splits(data_root, args.split):
        split_dir = data_root / split
        mapping_root = split_dir / "person_face_mapping"
        if not mapping_root.is_dir():
            print(f"[SKIP] {split}: missing {mapping_root}")
            continue

        for scene_dir in sorted(path for path in mapping_root.iterdir() if path.is_dir()):
            scene_key = scene_dir.name
            if args.scene_key and scene_key != args.scene_key:
                continue
            remap_path = scene_dir / "remap_all_chunks.json"
            if not remap_path.exists():
                print(f"[SKIP] {split}/{scene_key}: missing remap_all_chunks.json")
                continue
            remap = load_json(remap_path)
            chunks = remap.get("chunks", {})
            if not isinstance(chunks, dict):
                continue

            for chunk_payload in chunks.values():
                if not isinstance(chunk_payload, dict):
                    continue
                for clip_name, clip_payload in sorted(chunk_payload.items()):
                    if not isinstance(clip_payload, dict):
                        continue
                    parsed = parse_clip_name(clip_name)
                    if parsed is None:
                        continue
                    if int(parsed["camera_person"]) not in selected_camera_persons:
                        continue
                    if args.video and clip_name != args.video:
                        continue

                    people_to_segments = people_to_segments_from_clip(clip_payload)
                    if len(people_to_segments) < int(args.min_mapped_people):
                        continue

                    raw_mask_path = split_dir / "person_mask" / clip_name / "masks.pt"
                    refined_mask_path = split_dir / "refined_mask" / clip_name / "mask.pt"
                    frame_dir = split_dir / "frame" / clip_name
                    missing = [
                        str(path)
                        for path in (raw_mask_path, refined_mask_path, frame_dir)
                        if not path.exists()
                    ]
                    if missing:
                        print(f"[SKIP] {split}/{clip_name}: missing {', '.join(missing)}")
                        continue

                    jobs.append(
                        ClipJob(
                            split=split,
                            scene_key=scene_key,
                            clip_name=clip_name,
                            camera_person=int(parsed["camera_person"]),
                            chunk=int(parsed["chunk"]),
                            people_to_segments=people_to_segments,
                            frame_dir=frame_dir,
                            raw_mask_path=raw_mask_path,
                            refined_mask_path=refined_mask_path,
                            output_dir=split_dir / "final_vis" / scene_key / clip_name,
                        )
                    )
    return jobs


def resize_mask(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    if mask.shape[:2] == target_shape:
        return mask.astype(bool)
    height, width = target_shape
    return cv2.resize(
        mask.astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    y_indices, x_indices = np.where(mask.astype(bool))
    if len(y_indices) == 0 or len(x_indices) == 0:
        return None
    return (
        int(x_indices.min()),
        int(y_indices.min()),
        int(x_indices.max()) + 1,
        int(y_indices.max()) + 1,
    )


def color_for_person(person_id: int) -> tuple[int, int, int]:
    if person_id in PERSON_COLORS_BGR:
        return PERSON_COLORS_BGR[person_id]
    return SEGMENT_FALLBACK_COLORS_BGR[person_id % len(SEGMENT_FALLBACK_COLORS_BGR)]


def draw_label(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    x, y = origin
    y = max(20, y)
    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_overlay(
    frame_bgr: np.ndarray,
    frame_masks: dict[int, np.ndarray],
    people_to_segments: dict[int, list[int]],
    alpha: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    out = frame_bgr.copy()
    overlay = frame_bgr.copy()
    records: list[dict[str, Any]] = []

    for person_id, segment_ids in sorted(people_to_segments.items()):
        color = color_for_person(int(person_id))
        for segment_id in segment_ids:
            mask = frame_masks.get(int(segment_id))
            if mask is None:
                continue
            mask_bool = resize_mask(mask, frame_bgr.shape[:2])
            if not mask_bool.any():
                continue
            overlay[mask_bool] = color
            bbox = mask_bbox(mask_bool)
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            records.append(
                {
                    "person_id": int(person_id),
                    "segment_id": int(segment_id),
                    "bbox": [x1, y1, x2, y2],
                    "mask_pixels": int(mask_bool.sum()),
                }
            )

    cv2.addWeighted(overlay, float(alpha), out, 1.0 - float(alpha), 0.0, dst=out)
    for record in records:
        color = color_for_person(int(record["person_id"]))
        x1, y1, x2, y2 = [int(value) for value in record["bbox"]]
        cv2.rectangle(out, (x1, y1), (max(0, x2 - 1), max(0, y2 - 1)), color, 2)
        draw_label(out, f"person_{record['person_id']} mask {record['segment_id']}", (x1, y1 - 8), color)

    return out, records


def segments_for_change_scope(
    raw_frame_masks: dict[int, np.ndarray],
    refined_frame_masks: dict[int, np.ndarray],
    people_to_segments: dict[int, list[int]],
    scope: str,
) -> list[int]:
    if scope == "all":
        return sorted(set(raw_frame_masks) | set(refined_frame_masks))
    mapped_segments = set()
    for segment_ids in people_to_segments.values():
        mapped_segments.update(int(segment_id) for segment_id in segment_ids)
    return sorted(mapped_segments)


def masks_differ(raw_mask: np.ndarray | None, refined_mask: np.ndarray | None) -> bool:
    if raw_mask is None and refined_mask is None:
        return False
    if raw_mask is None or refined_mask is None:
        return True
    raw_bool = np.asarray(raw_mask).astype(bool)
    refined_bool = np.asarray(refined_mask).astype(bool)
    if raw_bool.shape != refined_bool.shape:
        refined_bool = resize_mask(refined_bool, raw_bool.shape[:2])
    return bool(np.any(raw_bool != refined_bool))


def frame_has_mask_change(
    raw_frame_masks: dict[int, np.ndarray],
    refined_frame_masks: dict[int, np.ndarray],
    people_to_segments: dict[int, list[int]],
    scope: str,
) -> bool:
    for segment_id in segments_for_change_scope(raw_frame_masks, refined_frame_masks, people_to_segments, scope):
        if masks_differ(raw_frame_masks.get(segment_id), refined_frame_masks.get(segment_id)):
            return True
    return False


def segment_person_lookup(people_to_segments: dict[int, list[int]]) -> dict[int, int]:
    lookup: dict[int, int] = {}
    for person_id, segment_ids in people_to_segments.items():
        for segment_id in segment_ids:
            lookup[int(segment_id)] = int(person_id)
    return lookup


def draw_diff_overlay(
    frame_bgr: np.ndarray,
    raw_frame_masks: dict[int, np.ndarray],
    refined_frame_masks: dict[int, np.ndarray],
    people_to_segments: dict[int, list[int]],
    scope: str,
    alpha: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    out = frame_bgr.copy()
    overlay = frame_bgr.copy()
    records: list[dict[str, Any]] = []
    person_by_segment = segment_person_lookup(people_to_segments)

    for segment_id in segments_for_change_scope(raw_frame_masks, refined_frame_masks, people_to_segments, scope):
        raw_mask = raw_frame_masks.get(segment_id)
        refined_mask = refined_frame_masks.get(segment_id)
        if raw_mask is None and refined_mask is None:
            continue

        base_shape = None
        if raw_mask is not None:
            base_shape = np.asarray(raw_mask).shape[:2]
        elif refined_mask is not None:
            base_shape = np.asarray(refined_mask).shape[:2]
        if base_shape is None:
            continue

        raw_bool = np.zeros(base_shape, dtype=bool) if raw_mask is None else np.asarray(raw_mask).astype(bool)
        refined_bool = (
            np.zeros(base_shape, dtype=bool)
            if refined_mask is None
            else resize_mask(np.asarray(refined_mask).astype(bool), base_shape)
        )
        if raw_bool.shape[:2] != frame_bgr.shape[:2]:
            raw_bool = resize_mask(raw_bool, frame_bgr.shape[:2])
            refined_bool = resize_mask(refined_bool, frame_bgr.shape[:2])

        removed = raw_bool & ~refined_bool
        added = refined_bool & ~raw_bool
        changed = removed | added
        if not changed.any():
            continue

        # Red means present before preprocessing and removed after filtering.
        overlay[removed] = (0, 0, 255)
        # Cyan is included for completeness if a processed mask ever contains extra pixels.
        overlay[added] = (255, 255, 0)
        bbox = mask_bbox(changed)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        person_id = person_by_segment.get(int(segment_id))
        label = f"removed mask {segment_id}" if person_id is None else f"removed person_{person_id} mask {segment_id}"
        records.append(
            {
                "person_id": person_id,
                "segment_id": int(segment_id),
                "bbox": [x1, y1, x2, y2],
                "removed_pixels": int(removed.sum()),
                "added_pixels": int(added.sum()),
                "label": label,
            }
        )

    cv2.addWeighted(overlay, float(alpha), out, 1.0 - float(alpha), 0.0, dst=out)
    for record in records:
        x1, y1, x2, y2 = [int(value) for value in record["bbox"]]
        color = (0, 0, 255) if int(record["removed_pixels"]) > 0 else (255, 255, 0)
        cv2.rectangle(out, (x1, y1), (max(0, x2 - 1), max(0, y2 - 1)), color, 2)
        draw_label(out, str(record["label"]), (x1, y1 - 8), color)

    if records:
        draw_label(out, "diff: red=removed mask before->after", (12, 24), (0, 0, 255))
    return out, records


def frame_has_enough_mapped_people(
    frame_masks: dict[int, np.ndarray],
    people_to_segments: dict[int, list[int]],
    min_people: int,
) -> bool:
    visible_people = 0
    for segment_ids in people_to_segments.values():
        if any(int(segment_id) in frame_masks for segment_id in segment_ids):
            visible_people += 1
    return visible_people >= min_people


def select_frame_indices(
    frame_paths: list[Path],
    raw_masks: dict[int, dict[int, np.ndarray]],
    refined_masks: dict[int, dict[int, np.ndarray]],
    people_to_segments: dict[int, list[int]],
    sample_every: int,
    max_frames: int,
    prefer_two_people: bool,
    mode: str,
    scope: str,
) -> list[int]:
    sample_every = max(1, int(sample_every))
    sampled = list(range(0, len(frame_paths), sample_every))

    if mode in {"changed", "changed_or_sampled"}:
        changed = [
            frame_idx
            for frame_idx in range(len(frame_paths))
            if frame_has_mask_change(
                raw_masks.get(frame_idx, {}),
                refined_masks.get(frame_idx, {}),
                people_to_segments,
                scope,
            )
        ]
        if changed or mode == "changed":
            if max_frames > 0:
                changed = changed[:max_frames]
            return changed

    min_people = min(2, len(people_to_segments)) if prefer_two_people else 1
    selected = [
        frame_idx
        for frame_idx in sampled
        if frame_has_enough_mapped_people(
            refined_masks.get(frame_idx, {}),
            people_to_segments,
            min_people,
        )
    ]
    if not selected and min_people > 1:
        selected = [
            frame_idx
            for frame_idx in sampled
            if frame_has_enough_mapped_people(
                refined_masks.get(frame_idx, {}),
                people_to_segments,
                1,
            )
        ]
    if not selected:
        selected = sampled
    if max_frames > 0:
        selected = selected[:max_frames]
    return selected


def union_mapped_masks(
    frame_masks: dict[int, np.ndarray],
    people_to_segments: dict[int, list[int]],
    target_shape: tuple[int, int],
) -> np.ndarray:
    union = np.zeros(target_shape, dtype=bool)
    for segment_ids in people_to_segments.values():
        for segment_id in segment_ids:
            mask = frame_masks.get(int(segment_id))
            if mask is None:
                continue
            union |= resize_mask(mask, target_shape)
    return union


def mapped_mask_for_person(
    frame_masks: dict[int, np.ndarray],
    people_to_segments: dict[int, list[int]],
    person_id: int,
    target_shape: tuple[int, int],
) -> np.ndarray:
    mask_union = np.zeros(target_shape, dtype=bool)
    for segment_id in people_to_segments.get(int(person_id), []):
        mask = frame_masks.get(int(segment_id))
        if mask is None:
            continue
        mask_union |= resize_mask(mask, target_shape)
    return mask_union


def visible_mapped_people(
    frame_masks: dict[int, np.ndarray],
    people_to_segments: dict[int, list[int]],
    target_shape: tuple[int, int],
) -> list[tuple[int, np.ndarray]]:
    visible: list[tuple[int, np.ndarray]] = []
    for person_id in sorted(people_to_segments):
        mask = mapped_mask_for_person(frame_masks, people_to_segments, int(person_id), target_shape)
        if mask.any():
            visible.append((int(person_id), mask))
    return visible


def build_masked_frame(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    outside_alpha: float,
    inside_alpha: float,
) -> np.ndarray:
    mask_bool = resize_mask(mask, frame_bgr.shape[:2])
    outside = (frame_bgr.astype(np.float32) * float(outside_alpha)).astype(np.uint8)
    person = frame_bgr.astype(np.float32)
    white = np.full_like(person, 255.0)
    inside = (person * float(inside_alpha) + white * (1.0 - float(inside_alpha))).astype(np.uint8)
    out = outside.copy()
    out[mask_bool] = inside[mask_bool]
    return out


def build_bw_mask(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    mask_bool = resize_mask(mask, target_shape)
    return (mask_bool.astype(np.uint8) * 255)


def draw_case_overlay(
    frame_bgr: np.ndarray,
    masks: list[tuple[int, np.ndarray, str, tuple[int, int, int]]],
    alpha: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    out = frame_bgr.copy()
    overlay = frame_bgr.copy()
    records: list[dict[str, Any]] = []
    for segment_id, mask, label, color in masks:
        mask_bool = resize_mask(mask, frame_bgr.shape[:2])
        if not mask_bool.any():
            continue
        overlay[mask_bool] = color
        bbox = mask_bbox(mask_bool)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        records.append(
            {
                "segment_id": int(segment_id),
                "label": label,
                "bbox": [x1, y1, x2, y2],
                "mask_pixels": int(mask_bool.sum()),
            }
        )
    cv2.addWeighted(overlay, float(alpha), out, 1.0 - float(alpha), 0.0, dst=out)
    for record in records:
        color = next(color for segment_id, _, label, color in masks if label == record["label"])
        x1, y1, x2, y2 = [int(value) for value in record["bbox"]]
        cv2.rectangle(out, (x1, y1), (max(0, x2 - 1), max(0, y2 - 1)), color, 2)
        draw_label(out, str(record["label"]), (x1, y1 - 8), color)
    return out, records


def save_rejected_mask_examples(
    job: ClipJob,
    raw_masks: dict[int, dict[int, np.ndarray]],
    refined_masks: dict[int, dict[int, np.ndarray]],
    frame_paths: list[Path],
    output_dir: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[dict[str, Any]] = []
    frame_indices = sorted(
        frame_idx
        for frame_idx in range(len(frame_paths))
        if frame_has_mask_change(raw_masks.get(frame_idx, {}), refined_masks.get(frame_idx, {}), job.people_to_segments, "all")
    )
    if int(args.rejected_frames_per_clip) > 0:
        frame_indices = frame_indices[: int(args.rejected_frames_per_clip)]

    for frame_idx in frame_indices:
        frame_bgr = cv2.imread(str(frame_paths[frame_idx]), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            continue
        raw_frame = raw_masks.get(frame_idx, {})
        refined_frame = refined_masks.get(frame_idx, {})
        masks: list[tuple[int, np.ndarray, str, tuple[int, int, int]]] = []
        for segment_id, raw_mask in sorted(raw_frame.items()):
            refined_mask = refined_frame.get(segment_id)
            if not masks_differ(raw_mask, refined_mask):
                continue
            label = f"rejected raw mask {segment_id}"
            masks.append((int(segment_id), raw_mask, label, (0, 0, 255)))
        if not masks:
            continue
        base_overlay, base_records = draw_overlay(
            frame_bgr=frame_bgr,
            frame_masks=raw_frame,
            people_to_segments=job.people_to_segments,
            alpha=args.alpha,
        )
        overlay, records = draw_case_overlay(base_overlay, masks, args.alpha)
        draw_label(overlay, "rejected mask: red=raw person mask removed", (12, 24), (0, 0, 255))
        out_path = output_dir / f"{frame_paths[frame_idx].stem}.jpg"
        if cv2.imwrite(str(out_path), overlay):
            saved.append(
                {
                    "frame_idx": int(frame_idx),
                    "frame_stem": frame_paths[frame_idx].stem,
                    "path": str(out_path),
                    "mapped_people_instances": base_records,
                    "rejected_instances": records,
                }
            )
    return saved


def save_merged_track_examples(
    job: ClipJob,
    refined_masks: dict[int, dict[int, np.ndarray]],
    frame_paths: list[Path],
    output_dir: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[dict[str, Any]] = []
    for person_id, segment_ids in sorted(job.people_to_segments.items()):
        if len(segment_ids) < 2:
            continue
        group_name = "_".join(str(segment_id) for segment_id in segment_ids)
        color = color_for_person(int(person_id))
        for segment_id in segment_ids:
            all_candidate_frames = [
                frame_idx
                for frame_idx in range(len(frame_paths))
                if int(segment_id) in refined_masks.get(frame_idx, {})
            ]
            candidate_frames = [
                frame_idx
                for frame_idx in all_candidate_frames
                if frame_has_enough_mapped_people(refined_masks.get(frame_idx, {}), job.people_to_segments, 2)
            ]
            if not candidate_frames:
                candidate_frames = all_candidate_frames
            if int(args.merged_frames_per_segment) > 0:
                candidate_frames = candidate_frames[: int(args.merged_frames_per_segment)]
            for frame_idx in candidate_frames:
                frame_bgr = cv2.imread(str(frame_paths[frame_idx]), cv2.IMREAD_COLOR)
                if frame_bgr is None:
                    continue
                overlay, records = draw_overlay(
                    frame_bgr=frame_bgr,
                    frame_masks=refined_masks.get(frame_idx, {}),
                    people_to_segments=job.people_to_segments,
                    alpha=args.alpha,
                )
                label = f"merged person_{person_id} mask {segment_id}"
                highlight_overlay, highlight_records = draw_case_overlay(
                    overlay,
                    [(int(segment_id), refined_masks[frame_idx][int(segment_id)], label, color)],
                    args.alpha,
                )
                draw_label(
                    highlight_overlay,
                    f"merged track: person_{person_id} = masks {','.join(map(str, segment_ids))}",
                    (12, 24),
                    color,
                )
                out_path = output_dir / f"person_{person_id}_masks_{group_name}" / f"mask_{segment_id}_{frame_paths[frame_idx].stem}.jpg"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                if cv2.imwrite(str(out_path), highlight_overlay):
                    saved.append(
                        {
                            "person_id": int(person_id),
                            "segment_id": int(segment_id),
                            "merged_segment_ids": [int(value) for value in segment_ids],
                            "frame_idx": int(frame_idx),
                            "frame_stem": frame_paths[frame_idx].stem,
                            "path": str(out_path),
                            "mapped_people_instances": records,
                            "highlight_instances": highlight_records,
                        }
                    )
    return saved


def clear_old_outputs(output_dir: Path) -> None:
    for child_name in ("before", "after", "diff", "raw_frame", "masked_frame", "raw_mask_bw", "rejected_mask", "merged_track"):
        child_dir = output_dir / child_name
        if not child_dir.is_dir():
            continue
        for path in child_dir.rglob("*.jpg"):
            path.unlink()


def process_job(job: ClipJob, args: argparse.Namespace) -> dict[str, Any]:
    summary_path = job.output_dir / "summary.json"
    if summary_path.exists() and not args.overwrite:
        print(f"[SKIP] {job.split}/{job.clip_name}: output exists")
        return {
            "status": "skipped",
            "reason": "existing_output",
            "split": job.split,
            "scene_key": job.scene_key,
            "clip_name": job.clip_name,
        }

    frame_paths = list_frame_paths(job.frame_dir)
    if not frame_paths:
        raise ValueError(f"No frames found: {job.frame_dir}")

    raw_masks = load_mask_dict(job.raw_mask_path)
    refined_masks = load_mask_dict(job.refined_mask_path)
    selected_frame_indices = select_frame_indices(
        frame_paths=frame_paths,
        raw_masks=raw_masks,
        refined_masks=refined_masks,
        people_to_segments=job.people_to_segments,
        sample_every=args.sample_every,
        max_frames=args.max_frames_per_clip,
        prefer_two_people=not args.allow_single_person_frames,
        mode=args.selection_mode,
        scope=args.change_scope,
    )

    before_dir = job.output_dir / "before"
    after_dir = job.output_dir / "after"
    diff_dir = job.output_dir / "diff"
    raw_frame_dir = job.output_dir / "raw_frame"
    masked_frame_dir = job.output_dir / "masked_frame"
    raw_mask_bw_dir = job.output_dir / "raw_mask_bw"
    rejected_dir = job.output_dir / "rejected_mask"
    merged_dir = job.output_dir / "merged_track"
    before_dir.mkdir(parents=True, exist_ok=True)
    after_dir.mkdir(parents=True, exist_ok=True)
    diff_dir.mkdir(parents=True, exist_ok=True)
    raw_frame_dir.mkdir(parents=True, exist_ok=True)
    masked_frame_dir.mkdir(parents=True, exist_ok=True)
    raw_mask_bw_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir.mkdir(parents=True, exist_ok=True)
    merged_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        clear_old_outputs(job.output_dir)

    saved_frames: list[dict[str, Any]] = []
    for frame_idx in selected_frame_indices:
        if frame_idx >= len(frame_paths):
            continue
        frame_path = frame_paths[frame_idx]
        frame_bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            continue

        before_overlay, before_records = draw_overlay(
            frame_bgr=frame_bgr,
            frame_masks=raw_masks.get(frame_idx, {}),
            people_to_segments=job.people_to_segments,
            alpha=args.alpha,
        )
        after_overlay, after_records = draw_overlay(
            frame_bgr=frame_bgr,
            frame_masks=refined_masks.get(frame_idx, {}),
            people_to_segments=job.people_to_segments,
            alpha=args.alpha,
        )
        diff_overlay, diff_records = draw_diff_overlay(
            frame_bgr=frame_bgr,
            raw_frame_masks=raw_masks.get(frame_idx, {}),
            refined_frame_masks=refined_masks.get(frame_idx, {}),
            people_to_segments=job.people_to_segments,
            scope=args.change_scope,
            alpha=args.alpha,
        )
        single_person_outputs: list[dict[str, Any]] = []
        for masked_person_id, person_mask in visible_mapped_people(
            raw_masks.get(frame_idx, {}),
            job.people_to_segments,
            frame_bgr.shape[:2],
        ):
            masked_frame = build_masked_frame(
                frame_bgr=frame_bgr,
                mask=person_mask,
                outside_alpha=args.masked_outside_alpha,
                inside_alpha=args.masked_inside_alpha,
            )
            raw_mask_bw = build_bw_mask(person_mask, frame_bgr.shape[:2])
            person_masked_dir = masked_frame_dir / f"person_{masked_person_id}"
            person_bw_dir = raw_mask_bw_dir / f"person_{masked_person_id}"
            person_masked_dir.mkdir(parents=True, exist_ok=True)
            person_bw_dir.mkdir(parents=True, exist_ok=True)
            masked_frame_path = person_masked_dir / f"{frame_path.stem}.jpg"
            raw_mask_bw_path = person_bw_dir / f"{frame_path.stem}.jpg"
            masked_frame_ok = cv2.imwrite(str(masked_frame_path), masked_frame)
            raw_mask_bw_ok = cv2.imwrite(str(raw_mask_bw_path), raw_mask_bw)
            if masked_frame_ok and raw_mask_bw_ok:
                single_person_outputs.append(
                    {
                        "person_id": int(masked_person_id),
                        "masked_frame_path": str(masked_frame_path),
                        "raw_mask_bw_path": str(raw_mask_bw_path),
                        "raw_mask_pixels": int(person_mask.sum()),
                    }
                )

        before_path = before_dir / f"{frame_path.stem}.jpg"
        after_path = after_dir / f"{frame_path.stem}.jpg"
        diff_path = diff_dir / f"{frame_path.stem}.jpg"
        raw_frame_path = raw_frame_dir / f"{frame_path.stem}.jpg"
        before_ok = cv2.imwrite(str(before_path), before_overlay)
        after_ok = cv2.imwrite(str(after_path), after_overlay)
        diff_ok = cv2.imwrite(str(diff_path), diff_overlay)
        raw_frame_ok = cv2.imwrite(str(raw_frame_path), frame_bgr)
        if before_ok and after_ok and diff_ok and raw_frame_ok:
            saved_frames.append(
                {
                    "frame_idx": int(frame_idx),
                    "frame_stem": frame_path.stem,
                    "before_path": str(before_path),
                    "after_path": str(after_path),
                    "diff_path": str(diff_path),
                    "raw_frame_path": str(raw_frame_path),
                    "single_person_mask_outputs": single_person_outputs,
                    "before_instances": before_records,
                    "after_instances": after_records,
                    "diff_instances": diff_records,
                }
            )

    rejected_examples = save_rejected_mask_examples(
        job=job,
        raw_masks=raw_masks,
        refined_masks=refined_masks,
        frame_paths=frame_paths,
        output_dir=rejected_dir,
        args=args,
    )
    merged_examples = save_merged_track_examples(
        job=job,
        refined_masks=refined_masks,
        frame_paths=frame_paths,
        output_dir=merged_dir,
        args=args,
    )

    summary = {
        "status": "ok",
        "split": job.split,
        "scene_key": job.scene_key,
        "clip_name": job.clip_name,
        "camera_person": int(job.camera_person),
        "chunk": int(job.chunk),
        "people_to_segments": {
            str(person_id): [int(segment_id) for segment_id in segment_ids]
            for person_id, segment_ids in sorted(job.people_to_segments.items())
        },
        "raw_mask_path": str(job.raw_mask_path),
        "refined_mask_path": str(job.refined_mask_path),
        "frame_dir": str(job.frame_dir),
        "num_frames": len(frame_paths),
        "sample_every": int(args.sample_every),
        "max_frames_per_clip": int(args.max_frames_per_clip),
        "selection_mode": str(args.selection_mode),
        "change_scope": str(args.change_scope),
        "masked_outside_alpha": float(args.masked_outside_alpha),
        "masked_inside_alpha": float(args.masked_inside_alpha),
        "num_selected_frames": len(selected_frame_indices),
        "num_saved_frames": len(saved_frames),
        "num_rejected_examples": len(rejected_examples),
        "num_merged_examples": len(merged_examples),
        "saved_frames": saved_frames,
        "rejected_examples": rejected_examples,
        "merged_examples": merged_examples,
    }
    write_json(summary_path, summary)
    print(
        f"[OK] {job.split}/{job.scene_key}/{job.clip_name}: "
        f"saved {len(saved_frames)}/{len(selected_frame_indices)} frames "
        f"rejected={len(rejected_examples)} merged={len(merged_examples)}"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Save whole-frame before/after visualizations for mapped EgoCom person masks."
    )
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Comma-separated splits, or all_existing to scan splits with person_face_mapping.",
    )
    parser.add_argument("--scene_key", type=str, default=None, help="Optional scene filter.")
    parser.add_argument("--video", type=str, default=None, help="Optional exact clip filter.")
    parser.add_argument(
        "--camera_persons",
        type=str,
        default="1,2,3",
        help="Comma-separated camera person ids to process.",
    )
    parser.add_argument(
        "--sample_every",
        type=positive_int,
        default=30,
        help="Save one sampled frame every N frames.",
    )
    parser.add_argument(
        "--max_frames_per_clip",
        type=nonnegative_int,
        default=0,
        help="Maximum frames to save per clip; 0 means no cap.",
    )
    parser.add_argument(
        "--min_mapped_people",
        type=positive_int,
        default=2,
        help="Skip clips with fewer mapped people than this.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.35,
        help="Mask overlay opacity.",
    )
    parser.add_argument(
        "--selection_mode",
        type=selection_mode,
        default="sampled",
        help="sampled saves regular samples; changed saves frames where masks differ; changed_or_sampled falls back to samples.",
    )
    parser.add_argument(
        "--change_scope",
        type=change_scope,
        default="all",
        help="Use all raw/refined masks or only final mapped masks when finding and drawing diffs.",
    )
    parser.add_argument(
        "--allow_single_person_frames",
        action="store_true",
        help="Do not prefer frames where two mapped people are visible.",
    )
    parser.add_argument(
        "--rejected_frames_per_clip",
        type=nonnegative_int,
        default=20,
        help="Maximum rejected-mask case images per clip; 0 means no cap.",
    )
    parser.add_argument(
        "--merged_frames_per_segment",
        type=nonnegative_int,
        default=3,
        help="Maximum merged-track case images per mask id; 0 means no cap.",
    )
    parser.add_argument(
        "--masked_outside_alpha",
        type=float,
        default=0.04,
        help="Brightness multiplier outside the raw mapped-person mask in masked_frame outputs.",
    )
    parser.add_argument(
        "--masked_inside_alpha",
        type=float,
        default=0.85,
        help="Original-frame blend inside the raw mapped-person mask; lower values make the masked region whiter.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 0.0 <= float(args.alpha) <= 1.0:
        raise ValueError("--alpha must be between 0 and 1")
    if not 0.0 <= float(args.masked_outside_alpha) <= 1.0:
        raise ValueError("--masked_outside_alpha must be between 0 and 1")
    if not 0.0 <= float(args.masked_inside_alpha) <= 1.0:
        raise ValueError("--masked_inside_alpha must be between 0 and 1")

    jobs = collect_jobs(args)
    if not jobs:
        print("No matching clips found.")
        return

    summaries = [process_job(job, args) for job in jobs]
    by_status: dict[str, int] = {}
    for summary in summaries:
        status = str(summary.get("status", "unknown"))
        by_status[status] = by_status.get(status, 0) + 1
    print(f"Done: {by_status}")


if __name__ == "__main__":
    main()
