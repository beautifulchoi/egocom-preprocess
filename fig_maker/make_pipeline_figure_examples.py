"""
Create figure-ready examples for the EgoCom preprocessing pipeline.

The script uses existing holdout artifacts only. It does not run SAM, DA3,
face embedding extraction, or remapping.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


DEFAULT_SPLIT_ROOT = "/home/prj/data/egocom_holdout/1min/val"
DEFAULT_SCENE_KEY = "day_5__con_5"
DEFAULT_OUT_DIR = "/home/prj/data/egocom_holdout/1min/val/pipeline_figure_examples"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
CLIP_RE = re.compile(
    r"^(vid_\d+)__day_(?P<day>\d+)__con_(?P<con>\d+)__person_(?P<camera>\d+)"
    r"(?:_(?P<part>part\d+))?_chunk_(?P<chunk>\d+)$"
)

PERSON_COLORS_BGR = {
    1: (39, 127, 245),
    2: (68, 178, 88),
    3: (210, 95, 40),
    4: (150, 85, 190),
}
KEEP_COLOR_BGR = (52, 190, 80)
REJECT_COLOR_BGR = (36, 55, 230)
RAW_SEG_COLOR_BGR = (240, 175, 30)


@dataclass(frozen=True)
class SegmentMask:
    person_id: int
    segment_ids: tuple[int, ...]


@dataclass(frozen=True)
class SourceExample:
    video_name: str
    frame_idx: int
    frame_path: Path
    depth_path: Path
    kept_segment_id: int
    rejected_segment_id: int


@dataclass(frozen=True)
class TargetExample:
    video_name: str
    camera_person: int
    frame_idx: int
    people: tuple[SegmentMask, ...]


@dataclass(frozen=True)
class PipelineExample:
    sample_index: int
    scene_key: str
    chunk: int
    source: SourceExample
    target: TargetExample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate visual examples for the EgoCom preprocessing pipeline figure."
    )
    parser.add_argument("--split_root", default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--scene_key", default=DEFAULT_SCENE_KEY)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--tile_width", type=int, default=360)
    parser.add_argument("--tile_height", type=int, default=270)
    parser.add_argument("--tracking_frames", type=int, default=5)
    return parser.parse_args()


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


def load_mask_dict(path: Path) -> dict[int, dict[int, np.ndarray]]:
    raw = load_torch(path)
    if not isinstance(raw, dict):
        raise TypeError(f"Expected mask dict in {path}, got {type(raw)}")

    out: dict[int, dict[int, np.ndarray]] = {}
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
            out[int(frame_idx_raw)] = persons
    return out


def list_frame_paths(frame_dir: Path) -> list[Path]:
    if not frame_dir.is_dir():
        return []
    return sorted(
        path
        for path in frame_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def parse_clip(video_name: str) -> tuple[int, int] | None:
    match = CLIP_RE.match(video_name)
    if match is None:
        return None
    return int(match.group("camera")), int(match.group("chunk"))


def depth_to_colormap(depth_map: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth_map)
    if not np.any(valid):
        normalized = np.zeros(depth_map.shape, dtype=np.uint8)
    else:
        values = depth_map[valid].astype(np.float32)
        low, high = np.percentile(values, [2, 98])
        if high <= low:
            high = low + 1e-6
        clipped = np.clip(depth_map.astype(np.float32), low, high)
        normalized = ((clipped - low) / (high - low) * 255.0).astype(np.uint8)
        normalized[~valid] = 0
    return cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)


def person_color(person_id: int) -> tuple[int, int, int]:
    if person_id in PERSON_COLORS_BGR:
        return PERSON_COLORS_BGR[person_id]
    palette = list(PERSON_COLORS_BGR.values())
    return palette[person_id % len(palette)]


def resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape[:2] == shape:
        return mask.astype(bool)
    height, width = shape
    return cv2.resize(
        mask.astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)


def union_segments(
    persons: dict[int, np.ndarray] | None,
    segment_ids: tuple[int, ...],
) -> np.ndarray | None:
    if not persons:
        return None
    masks = [np.asarray(persons[sid]).astype(bool) for sid in segment_ids if sid in persons]
    if not masks:
        return None
    out = np.zeros(masks[0].shape, dtype=bool)
    for mask in masks:
        if mask.shape == out.shape:
            out |= mask
    return out if out.any() else None


def mask_bbox(mask: np.ndarray, pad: int = 8) -> tuple[int, int, int, int] | None:
    y_coords, x_coords = np.where(mask.astype(bool))
    if len(x_coords) == 0 or len(y_coords) == 0:
        return None
    height, width = mask.shape[:2]
    x1 = max(0, int(x_coords.min()) - pad)
    y1 = max(0, int(y_coords.min()) - pad)
    x2 = min(width, int(x_coords.max()) + 1 + pad)
    y2 = min(height, int(y_coords.max()) + 1 + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def draw_label(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    scale: float = 0.48,
) -> None:
    x, y = origin
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float = 0.5,
    label: str | None = None,
) -> np.ndarray:
    out = image.copy()
    mask_bool = resize_mask(mask, out.shape[:2])
    color_arr = np.array(color, dtype=np.uint8)
    out[mask_bool] = (
        out[mask_bool].astype(np.float32) * (1.0 - alpha)
        + color_arr.astype(np.float32) * alpha
    ).astype(np.uint8)
    contours, _ = cv2.findContours(mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, 2)
    if label:
        bbox = mask_bbox(mask_bool, pad=0)
        if bbox:
            x1, y1, _, _ = bbox
            draw_label(out, label, (x1, max(18, y1 - 6)), color)
    return out


def fit_image(image: np.ndarray, width: int, height: int, bg: tuple[int, int, int] = (245, 245, 245)) -> np.ndarray:
    canvas = np.full((height, width, 3), bg, dtype=np.uint8)
    src_h, src_w = image.shape[:2]
    scale = min(width / max(src_w, 1), height / max(src_h, 1))
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    x = (width - new_w) // 2
    y = (height - new_h) // 2
    canvas[y : y + new_h, x : x + new_w] = resized
    return canvas


def make_tile(image: np.ndarray, title: str, subtitle: str, width: int, height: int) -> np.ndarray:
    header_h = 46
    tile = np.full((height, width, 3), 248, dtype=np.uint8)
    body = fit_image(image, width, height - header_h, bg=(238, 238, 238))
    tile[header_h:, :] = body
    cv2.putText(tile, title, (12, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.putText(tile, subtitle[:54], (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (90, 90, 90), 1, cv2.LINE_AA)
    cv2.rectangle(tile, (0, 0), (width - 1, height - 1), (215, 215, 215), 1)
    return tile


def raw_tile(frame: np.ndarray) -> np.ndarray:
    return frame.copy()


def depth_refine_tile(
    frame: np.ndarray,
    depth_map: np.ndarray,
    raw_persons: dict[int, np.ndarray],
    refined_persons: dict[int, np.ndarray],
    kept_segment_id: int,
    rejected_segment_id: int,
) -> np.ndarray:
    depth_color = depth_to_colormap(depth_map)
    if depth_color.shape[:2] != frame.shape[:2]:
        depth_color = cv2.resize(depth_color, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
    out = cv2.addWeighted(frame, 0.48, depth_color, 0.52, 0.0)
    kept_mask = refined_persons.get(kept_segment_id)
    rejected_mask = raw_persons.get(rejected_segment_id)
    if kept_mask is not None:
        out = overlay_mask(out, kept_mask, KEEP_COLOR_BGR, alpha=0.50, label=f"kept seg {kept_segment_id}")
    if rejected_mask is not None:
        out = overlay_mask(out, rejected_mask, REJECT_COLOR_BGR, alpha=0.60, label=f"rejected seg {rejected_segment_id}")
    return out


def masked_people_tile(
    frame: np.ndarray,
    persons: dict[int, np.ndarray],
    people: tuple[SegmentMask, ...],
) -> np.ndarray:
    out = (frame.astype(np.float32) * 0.28).astype(np.uint8)
    for item in people:
        mask = union_segments(persons, item.segment_ids)
        if mask is None:
            continue
        visible = frame.copy()
        visible[~mask] = out[~mask]
        out[mask] = visible[mask]
        out = overlay_mask(
            out,
            mask,
            person_color(item.person_id),
            alpha=0.45,
            label=f"person {item.person_id}",
        )
    return out


def cropped_people_tile(
    frame: np.ndarray,
    persons: dict[int, np.ndarray],
    people: tuple[SegmentMask, ...],
    width: int = 640,
    height: int = 360,
) -> np.ndarray:
    crop_canvases: list[np.ndarray] = []
    for item in people:
        mask = union_segments(persons, item.segment_ids)
        if mask is None:
            crop_canvases.append(np.full((height, width // 2, 3), 238, dtype=np.uint8))
            continue
        bbox = mask_bbox(mask, pad=18)
        if bbox is None:
            crop_canvases.append(np.full((height, width // 2, 3), 238, dtype=np.uint8))
            continue
        x1, y1, x2, y2 = bbox
        crop = frame[y1:y2, x1:x2].copy()
        crop_mask = mask[y1:y2, x1:x2]
        crop[~crop_mask] = (crop[~crop_mask].astype(np.float32) * 0.18).astype(np.uint8)
        crop = overlay_mask(crop, crop_mask, person_color(item.person_id), alpha=0.40, label=f"person {item.person_id}")
        crop_canvases.append(fit_image(crop, width // 2, height, bg=(238, 238, 238)))
    while len(crop_canvases) < 2:
        crop_canvases.append(np.full((height, width // 2, 3), 238, dtype=np.uint8))
    out = np.concatenate(crop_canvases[:2], axis=1)
    cv2.line(out, (width // 2, 0), (width // 2, height), (225, 225, 225), 2)
    return out


def tracking_tile(
    frame_paths: list[Path],
    refined_mask: dict[int, dict[int, np.ndarray]],
    center_idx: int,
    people: tuple[SegmentMask, ...],
    strip_count: int,
) -> np.ndarray:
    strip_count = max(1, strip_count)
    half = strip_count // 2
    candidates = [center_idx + offset for offset in range(-half, half + 1)]
    candidates = [idx for idx in candidates if 0 <= idx < len(frame_paths)]
    while len(candidates) < strip_count and candidates:
        if candidates[0] > 0:
            candidates.insert(0, candidates[0] - 1)
        elif candidates[-1] + 1 < len(frame_paths):
            candidates.append(candidates[-1] + 1)
        else:
            break
    candidates = candidates[:strip_count]

    panels: list[np.ndarray] = []
    for idx in candidates:
        frame = cv2.imread(str(frame_paths[idx]), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        out = frame.copy()
        persons = refined_mask.get(idx, {})
        for item in people:
            mask = union_segments(persons, item.segment_ids)
            if mask is None:
                continue
            out = overlay_mask(out, mask, person_color(item.person_id), alpha=0.50, label=f"id {item.person_id}")
        draw_label(out, f"t{idx - center_idx:+d}", (10, 24), (40, 40, 40), scale=0.48)
        panels.append(fit_image(out, 180, 180, bg=(238, 238, 238)))
    if not panels:
        return np.full((180, 180 * strip_count, 3), 238, dtype=np.uint8)
    return np.concatenate(panels, axis=1)


def remap_people_from_payload(payload: dict[str, Any]) -> tuple[SegmentMask, ...]:
    people = payload.get("people")
    if not isinstance(people, dict):
        return tuple()
    out: list[SegmentMask] = []
    for person_id_raw, person_payload in people.items():
        if not isinstance(person_payload, dict):
            continue
        segment_ids = person_payload.get("merged_segment_ids")
        if not segment_ids:
            segment_ids = [person_payload.get("primary_segment_id")]
        clean = tuple(sorted(int(value) for value in segment_ids if value is not None))
        if clean:
            out.append(SegmentMask(person_id=int(person_id_raw), segment_ids=clean))
    return tuple(sorted(out, key=lambda item: item.person_id))


def find_source_example(split_root: Path, video_name: str) -> SourceExample | None:
    raw_path = split_root / "person_mask" / video_name / "masks.pt"
    refined_path = split_root / "refined_mask" / video_name / "mask.pt"
    frame_dir = split_root / "frame" / video_name
    depth_dir = split_root / "da3" / "monocular" / video_name / "depth"
    if not raw_path.exists() or not refined_path.exists() or not depth_dir.is_dir():
        return None

    raw_mask = load_mask_dict(raw_path)
    refined_mask = load_mask_dict(refined_path)
    frame_paths = list_frame_paths(frame_dir)
    if not frame_paths:
        return None

    for frame_idx in sorted(raw_mask):
        if frame_idx >= len(frame_paths):
            continue
        depth_path = depth_dir / f"{frame_paths[frame_idx].stem}.npy"
        if not depth_path.exists():
            continue
        raw_persons = raw_mask.get(frame_idx, {})
        refined_persons = refined_mask.get(frame_idx, {})
        rejected_ids = sorted(set(raw_persons) - set(refined_persons))
        kept_ids = sorted(refined_persons)
        if not rejected_ids or not kept_ids:
            continue
        kept_segment_id = max(kept_ids, key=lambda sid: int(refined_persons[sid].sum()))
        rejected_segment_id = max(rejected_ids, key=lambda sid: int(raw_persons[sid].sum()))
        return SourceExample(
            video_name=video_name,
            frame_idx=frame_idx,
            frame_path=frame_paths[frame_idx],
            depth_path=depth_path,
            kept_segment_id=int(kept_segment_id),
            rejected_segment_id=int(rejected_segment_id),
        )
    return None


def find_target_example(
    split_root: Path,
    chunk_payload: dict[str, Any],
    source_video_name: str,
    source_frame_idx: int,
) -> TargetExample | None:
    parsed_source = parse_clip(source_video_name)
    source_camera = parsed_source[0] if parsed_source else None
    candidates = []
    for video_name, payload in chunk_payload.items():
        if video_name == source_video_name:
            continue
        camera_person = int(payload.get("camera_person", -1))
        if source_camera is not None and camera_person == source_camera:
            continue
        people = remap_people_from_payload(payload)
        if len(people) < 2:
            continue
        candidates.append((video_name, camera_person, people[:2]))

    for video_name, camera_person, people in candidates:
        raw_path = split_root / "person_mask" / video_name / "masks.pt"
        frame_dir = split_root / "frame" / video_name
        if not raw_path.exists() or not frame_dir.is_dir():
            continue
        raw_mask = load_mask_dict(raw_path)
        frame_paths = list_frame_paths(frame_dir)
        if not frame_paths:
            continue
        search_order = sorted(
            range(len(frame_paths)),
            key=lambda idx: (abs(idx - source_frame_idx), idx),
        )
        for frame_idx in search_order[:90]:
            persons = raw_mask.get(frame_idx, {})
            if all(union_segments(persons, item.segment_ids) is not None for item in people):
                return TargetExample(
                    video_name=video_name,
                    camera_person=camera_person,
                    frame_idx=frame_idx,
                    people=people,
                )
    return None


def collect_examples(split_root: Path, scene_key: str, num_samples: int) -> list[PipelineExample]:
    remap_path = split_root / "person_face_mapping" / scene_key / "remap_all_chunks.json"
    if not remap_path.exists():
        raise FileNotFoundError(f"Missing remap file: {remap_path}")
    remap = load_json(remap_path)
    chunks = remap.get("chunks")
    if not isinstance(chunks, dict):
        raise ValueError(f"Unexpected remap format: {remap_path}")

    examples: list[PipelineExample] = []
    for chunk_key in sorted(chunks, key=lambda value: int(value)):
        chunk_payload = chunks[chunk_key]
        if not isinstance(chunk_payload, dict):
            continue
        videos = sorted(chunk_payload)
        for video_name in videos:
            source = find_source_example(split_root, video_name)
            if source is None:
                continue
            target = find_target_example(
                split_root=split_root,
                chunk_payload=chunk_payload,
                source_video_name=video_name,
                source_frame_idx=source.frame_idx,
            )
            if target is None:
                continue
            examples.append(
                PipelineExample(
                    sample_index=len(examples) + 1,
                    scene_key=scene_key,
                    chunk=int(chunk_key),
                    source=source,
                    target=target,
                )
            )
            if len(examples) >= num_samples:
                return examples
    return examples


def save_sample_tiles(
    split_root: Path,
    out_dir: Path,
    example: PipelineExample,
    tile_width: int,
    tile_height: int,
    tracking_frames: int,
) -> tuple[list[Path], dict[str, Any]]:
    sample_dir = out_dir / "tiles" / f"sample_{example.sample_index:02d}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    source_raw = load_mask_dict(split_root / "person_mask" / example.source.video_name / "masks.pt")
    source_refined = load_mask_dict(split_root / "refined_mask" / example.source.video_name / "mask.pt")
    target_raw = load_mask_dict(split_root / "person_mask" / example.target.video_name / "masks.pt")
    target_refined_path = split_root / "refined_mask" / example.target.video_name / "mask.pt"
    target_refined = load_mask_dict(target_refined_path) if target_refined_path.exists() else {}
    source_frame = cv2.imread(str(example.source.frame_path), cv2.IMREAD_COLOR)
    if source_frame is None:
        raise FileNotFoundError(f"Could not read frame: {example.source.frame_path}")
    depth_map = np.load(example.source.depth_path)

    target_frame_paths = list_frame_paths(split_root / "frame" / example.target.video_name)
    target_frame = cv2.imread(str(target_frame_paths[example.target.frame_idx]), cv2.IMREAD_COLOR)
    if target_frame is None:
        raise FileNotFoundError(f"Could not read frame: {target_frame_paths[example.target.frame_idx]}")

    stage_images = [
        (
            "01_raw",
            make_tile(
                raw_tile(source_frame),
                "1. Raw frame",
                f"{example.source.video_name} f{example.source.frame_idx}",
                tile_width,
                tile_height,
            ),
        ),
        (
            "02_depth_refine",
            make_tile(
                depth_refine_tile(
                    source_frame,
                    depth_map,
                    source_raw.get(example.source.frame_idx, {}),
                    source_refined.get(example.source.frame_idx, {}),
                    example.source.kept_segment_id,
                    example.source.rejected_segment_id,
                ),
                "2. Depth refine",
                f"kept {example.source.kept_segment_id}, rejected {example.source.rejected_segment_id}",
                tile_width,
                tile_height,
            ),
        ),
        (
            "03_common_person_mapping",
            make_tile(
                masked_people_tile(
                    target_frame,
                    target_raw.get(example.target.frame_idx, {}),
                    example.target.people,
                ),
                "3. Common mapping",
                f"{example.target.video_name} f{example.target.frame_idx}",
                tile_width,
                tile_height,
            ),
        ),
        (
            "04_cropped_common_person",
            make_tile(
                cropped_people_tile(
                    target_frame,
                    target_raw.get(example.target.frame_idx, {}),
                    example.target.people,
                ),
                "4. Cropped persons",
                "mask bbox crops",
                tile_width,
                tile_height,
            ),
        ),
        (
            "05_temporal_tracking",
            make_tile(
                tracking_tile(
                    target_frame_paths,
                    target_refined,
                    example.target.frame_idx,
                    example.target.people,
                    tracking_frames,
                ),
                "5. Tracked masks",
                "same real person id uses same color",
                tile_width,
                tile_height,
            ),
        ),
    ]

    tile_paths: list[Path] = []
    for stage_name, image in stage_images:
        path = sample_dir / f"{stage_name}.png"
        cv2.imwrite(str(path), image)
        tile_paths.append(path)

    row = np.concatenate([image for _, image in stage_images], axis=1)
    row_dir = out_dir / "rows"
    row_dir.mkdir(parents=True, exist_ok=True)
    row_path = row_dir / f"sample_{example.sample_index:02d}_pipeline.png"
    cv2.imwrite(str(row_path), row)

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
            "frame_path": str(target_frame_paths[example.target.frame_idx]),
            "people": [
                {"person_id": item.person_id, "segment_ids": list(item.segment_ids)}
                for item in example.target.people
            ],
        },
        "tile_paths": [str(path) for path in tile_paths],
        "row_path": str(row_path),
    }
    write_json(sample_dir / "metadata.json", metadata)
    return [*tile_paths, row_path], metadata


def build_summary_grid(row_paths: list[Path], out_path: Path) -> None:
    rows = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in row_paths]
    rows = [row for row in rows if row is not None]
    if not rows:
        return
    max_w = max(row.shape[1] for row in rows)
    normalized = []
    for row in rows:
        if row.shape[1] == max_w:
            normalized.append(row)
            continue
        canvas = np.full((row.shape[0], max_w, 3), 248, dtype=np.uint8)
        canvas[:, : row.shape[1]] = row
        normalized.append(canvas)
    grid = np.concatenate(normalized, axis=0)
    cv2.imwrite(str(out_path), grid)


def main() -> None:
    args = parse_args()
    split_root = Path(args.split_root)
    out_dir = Path(args.out_dir)
    if args.num_samples < 1:
        raise ValueError("--num_samples must be >= 1")
    if not split_root.is_dir():
        raise FileNotFoundError(f"Missing split root: {split_root}")

    examples = collect_examples(split_root, args.scene_key, args.num_samples)
    if len(examples) < args.num_samples:
        raise RuntimeError(f"Found only {len(examples)} examples; requested {args.num_samples}")

    out_dir.mkdir(parents=True, exist_ok=True)
    all_metadata: list[dict[str, Any]] = []
    row_paths: list[Path] = []
    for example in examples:
        _, metadata = save_sample_tiles(
            split_root=split_root,
            out_dir=out_dir,
            example=example,
            tile_width=args.tile_width,
            tile_height=args.tile_height,
            tracking_frames=args.tracking_frames,
        )
        all_metadata.append(metadata)
        row_paths.append(Path(metadata["row_path"]))

    grid_path = out_dir / "pipeline_examples_grid.png"
    build_summary_grid(row_paths, grid_path)
    write_json(
        out_dir / "metadata.json",
        {
            "split_root": str(split_root),
            "scene_key": args.scene_key,
            "num_samples": len(all_metadata),
            "tile_width": args.tile_width,
            "tile_height": args.tile_height,
            "tracking_frames": args.tracking_frames,
            "grid_path": str(grid_path),
            "samples": all_metadata,
        },
    )
    print(f"Saved {len(all_metadata)} pipeline examples to {out_dir}")
    print(f"Summary grid: {grid_path}")


if __name__ == "__main__":
    main()
