"""
Lift EgoCom mapped person masks into per-frame camera-space depth summaries.

This step consumes refined masks, person-face mappings, saved face bboxes from
src/step_04_extract_person_embeding.py, DA3 monocular depth, and DA3 nested intrinsics.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from tqdm import tqdm


DEFAULT_DATA_ROOT = "/home/prj/data/egocom_holdout/1min"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
CLIP_RE = re.compile(
    r"^(vid_\d+)__day_(?P<day>\d+)__con_(?P<con>\d+)__person_(?P<camera>\d+)"
    r"(?:_(?P<part>part\d+))?_chunk_(?P<chunk>\d+)$"
)
STATUS_CODES = {
    "absent_seg": 0,
    "discontinuity_rejected": 1,
    "mask_fallback": 2,
    "face_intersection": 3,
    "insufficient_depth": 4,
    "missing_depth": 5,
    "empty_scaled_mask": 6,
    "missing_inputs": 7,
}


@dataclass(frozen=True)
class Assignment:
    split: str
    scene_key: str
    video_name: str
    camera_person: int
    person_id: int
    segment_ids: tuple[int, ...]


def parse_comma_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


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


def load_json(path: Path) -> Any:
    with path.open("r") as file:
        return json.load(file)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        json.dump(payload, file, indent=2)


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
            if mask.ndim == 2 and mask.any():
                persons[int(segment_id_raw)] = mask
        if persons:
            mask_dict[int(frame_idx_raw)] = persons
    return mask_dict


def load_embedding_dict(embedding_path: Path) -> dict[int, dict[str, Any]]:
    if not embedding_path.exists():
        return {}
    raw = load_torch(embedding_path)
    return raw if isinstance(raw, dict) else {}


def clip_scene_key(video_name: str) -> tuple[str, int] | None:
    match = CLIP_RE.match(video_name)
    if match is None:
        return None
    scene_key = f"day_{match.group('day')}__con_{match.group('con')}"
    part = match.group("part")
    if part:
        scene_key = f"{scene_key}__{part}"
    return scene_key, int(match.group("camera"))


def discover_mapping_paths(mapping_root: Path, scene_key: str | None) -> list[Path]:
    if mapping_root.is_file():
        return [mapping_root]
    paths = []
    for scene_dir in sorted(path for path in mapping_root.iterdir() if path.is_dir()):
        if scene_key and scene_dir.name != scene_key:
            continue
        remap_path = scene_dir / "remap_all_chunks.json"
        fallback_path = scene_dir / "mapping.json"
        if remap_path.exists():
            paths.append(remap_path)
        elif fallback_path.exists():
            paths.append(fallback_path)
    aggregate_path = mapping_root / "all_scene_mappings.json"
    if not paths and aggregate_path.exists():
        paths.append(aggregate_path)
    return paths


def add_assignment(
    grouped: dict[tuple[str, str, str, int], set[int]],
    split: str,
    scene_key: str,
    video_name: str,
    camera_person: int,
    person_id: int,
    segment_ids: list[int],
) -> None:
    valid_segment_ids = [int(value) for value in segment_ids if value is not None]
    if not valid_segment_ids:
        return
    key = (split, scene_key, video_name, int(person_id))
    grouped[key].update(valid_segment_ids)


def assignments_from_clip_payload(
    grouped: dict[tuple[str, str, str, int], set[int]],
    split: str,
    scene_key: str,
    video_name: str,
    payload: dict[str, Any],
) -> None:
    parsed = clip_scene_key(video_name)
    camera_person = int(payload.get("camera_person", parsed[1] if parsed else -1))

    people = payload.get("people")
    if isinstance(people, dict):
        for person_id_raw, person_payload in people.items():
            if not isinstance(person_payload, dict):
                continue
            segment_ids = person_payload.get("merged_segment_ids")
            if not segment_ids:
                segment_ids = [person_payload.get("primary_segment_id")]
            add_assignment(
                grouped,
                split,
                scene_key,
                video_name,
                camera_person,
                int(person_id_raw),
                [int(value) for value in segment_ids if value is not None],
            )
        return

    assignments = payload.get("assignments")
    if isinstance(assignments, dict):
        by_person: dict[int, list[int]] = defaultdict(list)
        for segment_id_raw, person_id_raw in assignments.items():
            if person_id_raw is None:
                continue
            by_person[int(person_id_raw)].append(int(segment_id_raw))
        for person_id, segment_ids in by_person.items():
            add_assignment(
                grouped,
                split,
                scene_key,
                video_name,
                camera_person,
                person_id,
                segment_ids,
            )


def collect_assignments(args: argparse.Namespace) -> list[Assignment]:
    mapping_root = Path(args.mapping_root)
    grouped: dict[tuple[str, str, str, int], set[int]] = defaultdict(set)
    for mapping_path in discover_mapping_paths(mapping_root, args.scene_key):
        data = load_json(mapping_path)
        scenes = data.items() if mapping_path.name == "all_scene_mappings.json" else [(data.get("scene_key"), data)]
        for scene_key_raw, scene_payload in scenes:
            if not isinstance(scene_payload, dict) or not scene_key_raw:
                continue
            scene_key = str(scene_key_raw)
            if args.scene_key and scene_key != args.scene_key:
                continue
            split = str(scene_payload.get("split") or args.split)

            chunks = scene_payload.get("chunks")
            if isinstance(chunks, dict):
                for chunk_payload in chunks.values():
                    if not isinstance(chunk_payload, dict):
                        continue
                    for video_name, clip_payload in chunk_payload.items():
                        if args.video and video_name != args.video:
                            continue
                        if isinstance(clip_payload, dict):
                            assignments_from_clip_payload(grouped, split, scene_key, video_name, clip_payload)

            clips = scene_payload.get("clips")
            if isinstance(clips, dict):
                for video_name, clip_payload in clips.items():
                    if args.video and video_name != args.video:
                        continue
                    if isinstance(clip_payload, dict):
                        assignments_from_clip_payload(grouped, split, scene_key, video_name, clip_payload)

    assignments = [
        Assignment(
            split=split,
            scene_key=scene_key,
            video_name=video_name,
            camera_person=int(clip_scene_key(video_name)[1] if clip_scene_key(video_name) else -1),
            person_id=int(person_id),
            segment_ids=tuple(sorted(segment_ids)),
        )
        for (split, scene_key, video_name, person_id), segment_ids in sorted(grouped.items())
    ]
    if args.limit is not None:
        assignments = assignments[: args.limit]
    return assignments


def list_frame_files(frame_dir: Path) -> list[Path]:
    if not frame_dir.is_dir():
        return []
    return sorted(
        path
        for path in frame_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def get_mask_image_shape(mask_dict: dict[int, dict[int, np.ndarray]]) -> tuple[int, int] | None:
    for persons in mask_dict.values():
        for mask in persons.values():
            if isinstance(mask, np.ndarray) and mask.ndim == 2:
                return int(mask.shape[0]), int(mask.shape[1])
    return None


def union_segment_mask(
    persons: dict[int, np.ndarray] | None,
    segment_ids: tuple[int, ...],
) -> np.ndarray | None:
    if not persons:
        return None
    masks = [np.asarray(persons[segment_id]).astype(bool) for segment_id in segment_ids if segment_id in persons]
    if not masks:
        return None
    out = np.zeros(masks[0].shape, dtype=bool)
    for mask in masks:
        if mask.shape == out.shape:
            out |= mask
    return out if out.any() else None


def resize_mask_to_depth(mask: np.ndarray, depth_shape: tuple[int, int]) -> np.ndarray:
    depth_h, depth_w = depth_shape
    resized = cv2.resize(mask.astype(np.uint8), (depth_w, depth_h), interpolation=cv2.INTER_NEAREST)
    return resized.astype(bool)


def scale_bbox_to_depth(
    bbox: np.ndarray,
    src_shape: tuple[int, int],
    dst_shape: tuple[int, int],
) -> np.ndarray:
    src_h, src_w = src_shape
    dst_h, dst_w = dst_shape
    x_scale = float(dst_w) / float(src_w)
    y_scale = float(dst_h) / float(src_h)
    x1, y1, x2, y2 = bbox.astype(np.float32).tolist()
    scaled = np.array([x1 * x_scale, y1 * y_scale, x2 * x_scale, y2 * y_scale], dtype=np.float32)
    scaled[0] = np.clip(np.floor(scaled[0]), 0, dst_w)
    scaled[1] = np.clip(np.floor(scaled[1]), 0, dst_h)
    scaled[2] = np.clip(np.ceil(scaled[2]), 0, dst_w)
    scaled[3] = np.clip(np.ceil(scaled[3]), 0, dst_h)
    return scaled.astype(np.float32)


def clip_bbox_to_image(bbox: np.ndarray, image_shape: tuple[int, int]) -> tuple[int, int, int, int] | None:
    height, width = image_shape
    x1, y1, x2, y2 = bbox.tolist()
    x1 = max(0, min(int(x1), width))
    y1 = max(0, min(int(y1), height))
    x2 = max(0, min(int(x2), width))
    y2 = max(0, min(int(y2), height))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def bbox_intersection_pixels(mask: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    clipped = clip_bbox_to_image(bbox, mask.shape)
    if clipped is None:
        return np.zeros((0, 2), dtype=np.int32)
    x1, y1, x2, y2 = clipped
    region = mask[y1:y2, x1:x2]
    if not region.any():
        return np.zeros((0, 2), dtype=np.int32)
    y_coords, x_coords = np.where(region)
    return np.stack([x_coords + x1, y_coords + y1], axis=1).astype(np.int32)


def mask_pixels(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.zeros((0, 2), dtype=np.int32)
    y_coords, x_coords = np.where(mask)
    return np.stack([x_coords, y_coords], axis=1).astype(np.int32)


def depth_valid_mask(values: np.ndarray) -> np.ndarray:
    return np.isfinite(values) & (values > 0)


def compute_depth_gradient_norm(depth_map: np.ndarray) -> np.ndarray:
    grad_y, grad_x = np.gradient(depth_map.astype(np.float32))
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    return grad_mag / np.maximum(depth_map.astype(np.float32), 1e-6)


def detect_low_discontinuity(
    scaled_mask: np.ndarray,
    depth_map: np.ndarray,
    kernel_size: int,
    compare_dilate_iters: int,
    min_boundary_pixels: int,
    gap_ratio_thresh: float,
    boundary_grad_p90_thresh: float,
) -> tuple[bool, dict[str, float | int | None]]:
    mask_u8 = scaled_mask.astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    compare_mask = mask_u8.copy()
    if compare_dilate_iters > 0:
        compare_mask = cv2.dilate(compare_mask, kernel, iterations=compare_dilate_iters)
    eroded = cv2.erode(compare_mask, kernel, iterations=1)
    dilated = cv2.dilate(compare_mask, kernel, iterations=1)
    inner_band = compare_mask.astype(bool) & ~eroded.astype(bool)
    outer_band = dilated.astype(bool) & ~compare_mask.astype(bool)
    boundary_band = inner_band | outer_band
    inner_depth = depth_map[inner_band]
    outer_depth = depth_map[outer_band]
    inner_depth = inner_depth[depth_valid_mask(inner_depth)]
    outer_depth = outer_depth[depth_valid_mask(outer_depth)]
    boundary_depth = depth_map[boundary_band]
    boundary_valid = depth_valid_mask(boundary_depth)
    metrics: dict[str, float | int | None] = {
        "inner_valid_count": int(inner_depth.size),
        "outer_valid_count": int(outer_depth.size),
        "gap_ratio": None,
        "boundary_grad_p90": None,
    }
    if (
        inner_depth.size < min_boundary_pixels
        or outer_depth.size < min_boundary_pixels
        or int(boundary_valid.sum()) < min_boundary_pixels
    ):
        return False, metrics
    median_inner = float(np.median(inner_depth))
    median_outer = float(np.median(outer_depth))
    gap_ratio = abs(median_inner - median_outer) / max(median_inner, 1e-6)
    gradient_norm = compute_depth_gradient_norm(depth_map)
    boundary_grad_p90 = float(np.percentile(gradient_norm[boundary_band], 90))
    metrics["gap_ratio"] = gap_ratio
    metrics["boundary_grad_p90"] = boundary_grad_p90
    return (
        gap_ratio < gap_ratio_thresh and boundary_grad_p90 < boundary_grad_p90_thresh,
        metrics,
    )


def get_frame_intrinsics(intrinsics: np.ndarray, frame_idx: int) -> np.ndarray:
    if intrinsics.ndim == 2:
        return intrinsics
    if intrinsics.ndim == 3:
        if frame_idx < 0 or frame_idx >= intrinsics.shape[0]:
            raise IndexError(f"Frame index {frame_idx} out of bounds for intrinsics shape {intrinsics.shape}")
        return intrinsics[frame_idx]
    raise ValueError(f"Unexpected intrinsics shape: {intrinsics.shape}")


def compute_region_means(
    pixel_coords: np.ndarray,
    depth_values: np.ndarray,
    intrinsics: np.ndarray,
) -> dict[str, float]:
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    x_pixels = pixel_coords[:, 0].astype(np.float32)
    y_pixels = pixel_coords[:, 1].astype(np.float32)
    z_values = depth_values.astype(np.float32)
    x_ray = (x_pixels - cx) / fx
    y_ray = (y_pixels - cy) / fy
    x_values = x_ray * z_values
    y_values = y_ray * z_values
    d_values = np.sqrt(x_values**2 + y_values**2 + z_values**2)
    return {
        "x_ray_mean": float(x_ray.mean()),
        "y_ray_mean": float(y_ray.mean()),
        "x_mean": float(x_values.mean()),
        "y_mean": float(y_values.mean()),
        "z_mean": float(z_values.mean()),
        "d_mean": float(d_values.mean()),
    }


def init_float_array(num_frames: int) -> np.ndarray:
    return np.full((num_frames,), np.nan, dtype=np.float32)


def init_bbox_array(num_frames: int) -> np.ndarray:
    return np.full((num_frames, 4), np.nan, dtype=np.float32)


def build_face_lookup(
    embedding_dict: dict[int, dict[str, Any]],
    segment_ids: tuple[int, ...],
) -> dict[int, list[np.ndarray]]:
    lookup: dict[int, list[np.ndarray]] = defaultdict(list)
    for segment_id in segment_ids:
        item = embedding_dict.get(int(segment_id))
        if not isinstance(item, dict):
            continue
        frame_indices = item.get("frame_indices", [])
        face_bboxes = item.get("face_bboxes", [])
        for frame_idx, bbox in zip(frame_indices, face_bboxes):
            arr = np.asarray(bbox, dtype=np.float32)
            if arr.shape == (4,) and np.all(np.isfinite(arr)):
                lookup[int(frame_idx)].append(arr)
    return dict(lookup)


def enclosing_bbox(bboxes: list[np.ndarray]) -> np.ndarray | None:
    valid = [np.asarray(bbox, dtype=np.float32) for bbox in bboxes if np.asarray(bbox).shape == (4,)]
    if not valid:
        return None
    stacked = np.stack(valid, axis=0)
    return np.array(
        [
            float(stacked[:, 0].min()),
            float(stacked[:, 1].min()),
            float(stacked[:, 2].max()),
            float(stacked[:, 3].max()),
        ],
        dtype=np.float32,
    )


def set_status(
    labels: np.ndarray,
    codes: np.ndarray,
    counts: Counter,
    index: int,
    label: str,
) -> None:
    labels[index] = label
    codes[index] = STATUS_CODES[label]
    counts[label] += 1


def build_video_output(
    assignment: Assignment,
    split_root: Path,
    min_face_pixels: int,
    min_valid_depth_pixels: int,
    min_boundary_pixels: int,
    discontinuity_kernel_size: int,
    discontinuity_compare_dilate_iters: int,
    discontinuity_gap_ratio_thresh: float,
    discontinuity_boundary_grad_p90_thresh: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    frame_dir = split_root / "frame" / assignment.video_name
    mask_path = split_root / "refined_mask" / assignment.video_name / "mask.pt"
    emb_path = split_root / "person_face_emb" / assignment.video_name / "embeding.pt"
    depth_dir = split_root / "da3" / "monocular" / assignment.video_name / "depth"
    intrinsics_path = split_root / "da3" / "nested" / assignment.video_name / "camera_params" / "intrinsics.npy"

    frame_paths = list_frame_files(frame_dir)
    if not frame_paths:
        raise FileNotFoundError(f"No frames found: {frame_dir}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Missing mask: {mask_path}")
    if not depth_dir.is_dir():
        raise FileNotFoundError(f"Missing depth dir: {depth_dir}")
    if not intrinsics_path.exists():
        raise FileNotFoundError(f"Missing intrinsics: {intrinsics_path}")

    mask_dict = load_mask_dict(mask_path)
    embedding_dict = load_embedding_dict(emb_path)
    face_lookup = build_face_lookup(embedding_dict, assignment.segment_ids)
    intrinsics = np.load(intrinsics_path)
    mask_shape = get_mask_image_shape(mask_dict)
    if mask_shape is None:
        raise ValueError(f"Failed to infer mask shape from {mask_path}")

    num_frames = len(frame_paths)
    status_code = np.full((num_frames,), -1, dtype=np.int32)
    status_label = np.full((num_frames,), "", dtype="<U32")
    region_pixel_count = np.zeros((num_frames,), dtype=np.int32)
    valid_depth_pixel_count = np.zeros((num_frames,), dtype=np.int32)
    face_bbox_orig = init_bbox_array(num_frames)
    face_bbox_depth = init_bbox_array(num_frames)
    discontinuity_gap_ratio = init_float_array(num_frames)
    discontinuity_boundary_grad_p90 = init_float_array(num_frames)
    x_ray_mean = init_float_array(num_frames)
    y_ray_mean = init_float_array(num_frames)
    x_mean = init_float_array(num_frames)
    y_mean = init_float_array(num_frames)
    z_mean = init_float_array(num_frames)
    d_mean = init_float_array(num_frames)
    frame_stems = np.array([path.stem for path in frame_paths], dtype="<U64")
    counts: Counter = Counter()
    depth_shape: tuple[int, int] | None = None

    for logical_frame_idx, frame_path in enumerate(frame_paths):
        mask = union_segment_mask(mask_dict.get(logical_frame_idx), assignment.segment_ids)
        if mask is None:
            set_status(status_label, status_code, counts, logical_frame_idx, "absent_seg")
            continue

        depth_path = depth_dir / f"{frame_path.stem}.npy"
        if not depth_path.exists():
            set_status(status_label, status_code, counts, logical_frame_idx, "missing_depth")
            continue
        depth_map = np.load(depth_path).astype(np.float32)
        if depth_map.ndim != 2:
            set_status(status_label, status_code, counts, logical_frame_idx, "missing_depth")
            continue
        depth_shape = (int(depth_map.shape[0]), int(depth_map.shape[1]))

        scaled_mask = resize_mask_to_depth(mask, depth_shape)
        if not scaled_mask.any():
            set_status(status_label, status_code, counts, logical_frame_idx, "empty_scaled_mask")
            continue

        low_discontinuity, discontinuity_metrics = detect_low_discontinuity(
            scaled_mask=scaled_mask,
            depth_map=depth_map,
            kernel_size=discontinuity_kernel_size,
            compare_dilate_iters=discontinuity_compare_dilate_iters,
            min_boundary_pixels=min_boundary_pixels,
            gap_ratio_thresh=discontinuity_gap_ratio_thresh,
            boundary_grad_p90_thresh=discontinuity_boundary_grad_p90_thresh,
        )
        if discontinuity_metrics["gap_ratio"] is not None:
            discontinuity_gap_ratio[logical_frame_idx] = float(discontinuity_metrics["gap_ratio"])
        if discontinuity_metrics["boundary_grad_p90"] is not None:
            discontinuity_boundary_grad_p90[logical_frame_idx] = float(discontinuity_metrics["boundary_grad_p90"])
        if low_discontinuity:
            set_status(status_label, status_code, counts, logical_frame_idx, "discontinuity_rejected")
            continue

        selected_pixels = None
        selected_label = "mask_fallback"
        frame_bboxes = face_lookup.get(logical_frame_idx, [])
        face_bbox = enclosing_bbox(frame_bboxes)
        if face_bbox is not None:
            face_bbox_orig[logical_frame_idx] = face_bbox
            scaled_bboxes = [scale_bbox_to_depth(bbox, mask_shape, depth_shape) for bbox in frame_bboxes]
            depth_bbox = enclosing_bbox(scaled_bboxes)
            if depth_bbox is not None:
                face_bbox_depth[logical_frame_idx] = depth_bbox
                face_region = np.zeros(depth_shape, dtype=bool)
                for scaled_bbox in scaled_bboxes:
                    face_pixels = bbox_intersection_pixels(scaled_mask, scaled_bbox)
                    if len(face_pixels) > 0:
                        face_region[face_pixels[:, 1], face_pixels[:, 0]] = True
                if int(face_region.sum()) >= min_face_pixels:
                    candidate_pixels = mask_pixels(face_region)
                    candidate_depth = depth_map[candidate_pixels[:, 1], candidate_pixels[:, 0]]
                    valid = depth_valid_mask(candidate_depth)
                    if int(valid.sum()) >= min_valid_depth_pixels:
                        selected_pixels = candidate_pixels[valid]
                        selected_label = "face_intersection"

        if selected_pixels is None:
            candidate_pixels = mask_pixels(scaled_mask)
            candidate_depth = depth_map[candidate_pixels[:, 1], candidate_pixels[:, 0]]
            valid = depth_valid_mask(candidate_depth)
            if int(valid.sum()) < min_valid_depth_pixels:
                set_status(status_label, status_code, counts, logical_frame_idx, "insufficient_depth")
                continue
            selected_pixels = candidate_pixels[valid]

        selected_depth = depth_map[selected_pixels[:, 1], selected_pixels[:, 0]].astype(np.float32)
        frame_intrinsics = get_frame_intrinsics(intrinsics, logical_frame_idx)
        means = compute_region_means(selected_pixels, selected_depth, frame_intrinsics)
        region_pixel_count[logical_frame_idx] = int(len(selected_pixels))
        valid_depth_pixel_count[logical_frame_idx] = int(len(selected_depth))
        x_ray_mean[logical_frame_idx] = means["x_ray_mean"]
        y_ray_mean[logical_frame_idx] = means["y_ray_mean"]
        x_mean[logical_frame_idx] = means["x_mean"]
        y_mean[logical_frame_idx] = means["y_mean"]
        z_mean[logical_frame_idx] = means["z_mean"]
        d_mean[logical_frame_idx] = means["d_mean"]
        set_status(status_label, status_code, counts, logical_frame_idx, selected_label)

    if depth_shape is None:
        sample_depth = np.load(depth_dir / f"{frame_paths[0].stem}.npy").astype(np.float32)
        depth_shape = (int(sample_depth.shape[0]), int(sample_depth.shape[1]))

    payload: dict[str, Any] = {
        "split": np.array(assignment.split),
        "scene_key": np.array(assignment.scene_key),
        "actual_person_id": np.array(assignment.person_id, dtype=np.int32),
        "video_name": np.array(assignment.video_name),
        "camera_person": np.array(assignment.camera_person, dtype=np.int32),
        "segment_ids": np.array(assignment.segment_ids, dtype=np.int32),
        "num_frames": np.array(num_frames, dtype=np.int32),
        "logical_frame_indices": np.arange(num_frames, dtype=np.int32),
        "frame_stems": frame_stems,
        "original_mask_height": np.array(mask_shape[0], dtype=np.int32),
        "original_mask_width": np.array(mask_shape[1], dtype=np.int32),
        "depth_height": np.array(depth_shape[0], dtype=np.int32),
        "depth_width": np.array(depth_shape[1], dtype=np.int32),
        "status_code": status_code,
        "status_label": status_label,
        "region_pixel_count": region_pixel_count,
        "valid_depth_pixel_count": valid_depth_pixel_count,
        "face_bbox_orig": face_bbox_orig,
        "face_bbox_depth": face_bbox_depth,
        "discontinuity_gap_ratio": discontinuity_gap_ratio,
        "discontinuity_boundary_grad_p90": discontinuity_boundary_grad_p90,
        "x_ray_mean": x_ray_mean,
        "y_ray_mean": y_ray_mean,
        "x_mean": x_mean,
        "y_mean": y_mean,
        "z_mean": z_mean,
        "d_mean": d_mean,
    }
    valid_mask = np.isfinite(z_mean)
    video_summary = {
        "split": assignment.split,
        "scene_key": assignment.scene_key,
        "actual_person_id": int(assignment.person_id),
        "video_name": assignment.video_name,
        "camera_person": int(assignment.camera_person),
        "segment_ids": [int(value) for value in assignment.segment_ids],
        "num_frames": int(num_frames),
        "num_valid_frames": int(valid_mask.sum()),
        "counts": {key: int(counts.get(key, 0)) for key in sorted(STATUS_CODES)},
        "z_mean_valid_mean": float(np.nanmean(z_mean)) if int(valid_mask.sum()) > 0 else None,
        "d_mean_valid_mean": float(np.nanmean(d_mean)) if int(valid_mask.sum()) > 0 else None,
    }
    return payload, video_summary


def process_assignment(
    assignment: Assignment,
    data_root: Path,
    output_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    person_output_dir = output_root / assignment.scene_key / f"person_{assignment.person_id}"
    output_path = person_output_dir / f"{assignment.video_name}.npz"
    summary_path = person_output_dir / "summary.json"
    if output_path.exists() and args.skip_existing:
        return {"status": "skipped_existing", "assignment": assignment}
    split_root = data_root / assignment.split
    payload, video_summary = build_video_output(
        assignment=assignment,
        split_root=split_root,
        min_face_pixels=args.min_face_pixels,
        min_valid_depth_pixels=args.min_valid_depth_pixels,
        min_boundary_pixels=args.min_boundary_pixels,
        discontinuity_kernel_size=args.discontinuity_kernel_size,
        discontinuity_compare_dilate_iters=args.discontinuity_compare_dilate_iters,
        discontinuity_gap_ratio_thresh=args.discontinuity_gap_ratio_thresh,
        discontinuity_boundary_grad_p90_thresh=args.discontinuity_boundary_grad_p90_thresh,
    )
    person_output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **payload)

    existing = {"videos": []}
    if summary_path.exists():
        try:
            existing = load_json(summary_path)
        except Exception:
            existing = {"videos": []}
    videos = [item for item in existing.get("videos", []) if item.get("video_name") != assignment.video_name]
    videos.append(video_summary)
    global_counts = Counter()
    for item in videos:
        global_counts.update({key: int(value) for key, value in item.get("counts", {}).items()})
    summary_payload = {
        "split": assignment.split,
        "scene_key": assignment.scene_key,
        "actual_person_id": int(assignment.person_id),
        "num_video_outputs": len(videos),
        "videos": sorted(videos, key=lambda item: item["video_name"]),
        "global_counts": {key: int(global_counts.get(key, 0)) for key in sorted(STATUS_CODES)},
        "config": {
            "min_face_pixels": int(args.min_face_pixels),
            "min_valid_depth_pixels": int(args.min_valid_depth_pixels),
            "min_boundary_pixels": int(args.min_boundary_pixels),
            "discontinuity_kernel_size": int(args.discontinuity_kernel_size),
            "discontinuity_compare_dilate_iters": int(args.discontinuity_compare_dilate_iters),
            "discontinuity_gap_ratio_thresh": float(args.discontinuity_gap_ratio_thresh),
            "discontinuity_boundary_grad_p90_thresh": float(args.discontinuity_boundary_grad_p90_thresh),
        },
    }
    write_json(summary_path, summary_payload)
    return {"status": "processed", "assignment": assignment, "summary": video_summary}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lift EgoCom mapped person masks into depth summaries.")
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--mapping_root", type=str, default=None)
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--scene_key", type=str, default=None)
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--limit", type=nonnegative_int, default=None)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--overwrite_summary", action="store_true")
    parser.add_argument("--min_face_pixels", type=positive_int, default=16)
    parser.add_argument("--min_valid_depth_pixels", type=positive_int, default=8)
    parser.add_argument("--min_boundary_pixels", type=positive_int, default=16)
    parser.add_argument("--discontinuity_kernel_size", type=positive_int, default=3)
    parser.add_argument("--discontinuity_compare_dilate_iters", type=nonnegative_int, default=1)
    parser.add_argument("--discontinuity_gap_ratio_thresh", type=float, default=0.04)
    parser.add_argument("--discontinuity_boundary_grad_p90_thresh", type=float, default=0.05)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data_root = Path(args.data_root)
    split_root = data_root / args.split
    args.mapping_root = args.mapping_root or str(split_root / "person_face_mapping")
    output_root = Path(args.output_root or split_root / "person_depth_lift")
    assignments = collect_assignments(args)
    if not assignments:
        print("No matching mapped person assignments found.")
        return
    if args.overwrite_summary and not args.skip_existing:
        summary_paths = {
            output_root / assignment.scene_key / f"person_{assignment.person_id}" / "summary.json"
            for assignment in assignments
        }
        for summary_path in summary_paths:
            if summary_path.exists():
                summary_path.unlink()

    print(f"Found {len(assignments)} mapped person-video assignment(s)")
    status_counts: Counter = Counter()
    errors = []
    for assignment in tqdm(assignments, desc="depth lift"):
        try:
            result = process_assignment(
                assignment=assignment,
                data_root=data_root,
                output_root=output_root,
                args=args,
            )
            status_counts[result["status"]] += 1
        except Exception as exc:
            status_counts["failed"] += 1
            errors.append(
                {
                    "scene_key": assignment.scene_key,
                    "video_name": assignment.video_name,
                    "person_id": int(assignment.person_id),
                    "error": str(exc),
                }
            )
            print(f"[ERROR] {assignment.scene_key}/{assignment.video_name}/person_{assignment.person_id}: {exc}")

    run_summary = {
        "num_assignments": len(assignments),
        "status_counts": {key: int(value) for key, value in sorted(status_counts.items())},
        "scene_key": args.scene_key,
        "video": args.video,
        "errors": errors,
        "output_root": str(output_root),
    }
    write_json(output_root / "run_summary.json", run_summary)
    print("Depth lift finished")
    print(f"Status counts: {dict(sorted(status_counts.items()))}")
    print(f"Output root: {output_root}")


if __name__ == "__main__":
    main()
