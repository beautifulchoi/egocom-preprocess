"""
Extract CLIP visual features from full-frame masked EgoCom person tracks.

Each mapped person track emits one CLIP feature per source frame. Frames where
the mapped person segment is absent are encoded from a full black RGB image so
the output sequence length stays aligned to the chunk frame sequence.

Inputs:
  /home/prj/data/egocom_holdout/1min/{split}/person_face_mapping/*/remap_all_chunks.json
  /home/prj/data/egocom_holdout/1min/{split}/refined_mask/{video}/mask.pt
  /home/prj/data/egocom_holdout/1min/{split}/frame/{video}/*.jpg

Outputs:
  /home/prj/data/egocom_holdout/1min/{split}/person_visual_clip_features/{scene}/person_{id}/{video}.pt
  /home/prj/data/egocom_holdout/1min/{split}/person_visual_clip_features/{scene}/summary.json
  /home/prj/data/egocom_holdout/1min/{split}/person_visual_clip_features/summary.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


DEFAULT_DATA_ROOT = "/home/prj/data/egocom_holdout/1min"
DEFAULT_MODEL_ID = "openai/clip-vit-large-patch14"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
CLIP_RE = re.compile(
    r"^(vid_\d+)__day_(?P<day>\d+)__con_(?P<con>\d+)__person_(?P<camera>\d+)"
    r"(?:_(?P<part>part\d+))?_chunk_(?P<chunk>\d+)$"
)


@dataclass(frozen=True)
class Assignment:
    split: str
    scene_key: str
    video_name: str
    camera_person: int
    person_id: int
    segment_ids: tuple[int, ...]
    mapping_path: Path


@dataclass
class MaskedFrame:
    frame_idx: int
    frame_stem: str
    frame_path: Path
    image: Image.Image
    mask_bbox: tuple[int, int, int, int] | None
    mask_pixel_count: int
    has_mask: bool
    status: str


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


def clip_scene_key(video_name: str) -> tuple[str, int] | None:
    match = CLIP_RE.match(video_name)
    if match is None:
        return None
    scene_key = f"day_{match.group('day')}__con_{match.group('con')}"
    part = match.group("part")
    if part:
        scene_key = f"{scene_key}__{part}"
    return scene_key, int(match.group("camera"))


def split_names(data_root: Path, split_arg: str) -> list[str]:
    if split_arg != "all_existing":
        return parse_comma_list(split_arg)
    return sorted(
        split_dir.name
        for split_dir in data_root.iterdir()
        if split_dir.is_dir()
        and (split_dir / "person_face_mapping").is_dir()
        and (split_dir / "refined_mask").is_dir()
        and (split_dir / "frame").is_dir()
    )


def default_mapping_root(data_root: Path, split: str) -> Path:
    return data_root / split / "person_face_mapping"


def default_output_root(data_root: Path, split: str) -> Path:
    return data_root / split / "person_visual_clip_features"


def discover_mapping_paths(mapping_root: Path, scene_key: str | None) -> list[Path]:
    if mapping_root.is_file():
        return [mapping_root]
    if not mapping_root.is_dir():
        return []

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
    sources: dict[tuple[str, str, str, int], Path],
    split: str,
    scene_key: str,
    video_name: str,
    person_id: int,
    segment_ids: list[int],
    mapping_path: Path,
) -> None:
    valid_segment_ids = [int(value) for value in segment_ids if value is not None]
    if not valid_segment_ids:
        return
    key = (split, scene_key, video_name, int(person_id))
    grouped[key].update(valid_segment_ids)
    sources.setdefault(key, mapping_path)


def assignments_from_clip_payload(
    grouped: dict[tuple[str, str, str, int], set[int]],
    sources: dict[tuple[str, str, str, int], Path],
    split: str,
    scene_key: str,
    video_name: str,
    payload: dict[str, Any],
    mapping_path: Path,
) -> None:
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
                sources,
                split,
                scene_key,
                video_name,
                int(person_id_raw),
                [int(value) for value in segment_ids if value is not None],
                mapping_path,
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
                sources,
                split,
                scene_key,
                video_name,
                person_id,
                segment_ids,
                mapping_path,
            )


def collect_assignments_for_split(
    split: str,
    mapping_root: Path,
    scene_key_filter: str | None,
    video_filter: str | None,
) -> list[Assignment]:
    grouped: dict[tuple[str, str, str, int], set[int]] = defaultdict(set)
    sources: dict[tuple[str, str, str, int], Path] = {}

    for mapping_path in discover_mapping_paths(mapping_root, scene_key_filter):
        data = load_json(mapping_path)
        if mapping_path.name == "all_scene_mappings.json":
            scenes = data.items()
        else:
            scenes = [(data.get("scene_key"), data)]

        for scene_key_raw, scene_payload in scenes:
            if not isinstance(scene_payload, dict) or not scene_key_raw:
                continue
            scene_key = str(scene_key_raw)
            if scene_key_filter and scene_key != scene_key_filter:
                continue
            scene_split = str(scene_payload.get("split") or split)
            if scene_split != split:
                continue

            chunks = scene_payload.get("chunks")
            if isinstance(chunks, dict):
                for chunk_payload in chunks.values():
                    if not isinstance(chunk_payload, dict):
                        continue
                    for video_name, clip_payload in chunk_payload.items():
                        if video_filter and video_name != video_filter:
                            continue
                        if isinstance(clip_payload, dict):
                            assignments_from_clip_payload(
                                grouped,
                                sources,
                                split,
                                scene_key,
                                video_name,
                                clip_payload,
                                mapping_path,
                            )

            clips = scene_payload.get("clips")
            if isinstance(clips, dict):
                for video_name, clip_payload in clips.items():
                    if video_filter and video_name != video_filter:
                        continue
                    if isinstance(clip_payload, dict):
                        assignments_from_clip_payload(
                            grouped,
                            sources,
                            split,
                            scene_key,
                            video_name,
                            clip_payload,
                            mapping_path,
                        )

    assignments = []
    for key, segment_ids in sorted(grouped.items()):
        split_name, scene_key, video_name, person_id = key
        parsed = clip_scene_key(video_name)
        assignments.append(
            Assignment(
                split=split_name,
                scene_key=scene_key,
                video_name=video_name,
                camera_person=int(parsed[1] if parsed else -1),
                person_id=int(person_id),
                segment_ids=tuple(sorted(segment_ids)),
                mapping_path=sources[key],
            )
        )
    return assignments


def list_frame_files(frame_dir: Path) -> list[Path]:
    if not frame_dir.is_dir():
        return []
    return sorted(
        path
        for path in frame_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def resize_mask(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_shape
    if mask.shape[:2] == target_shape:
        return mask.astype(bool)
    resized = cv2.resize(
        mask.astype(np.uint8),
        (target_w, target_h),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized.astype(bool)


def union_segment_mask(
    persons: dict[int, np.ndarray] | None,
    segment_ids: tuple[int, ...],
) -> np.ndarray | None:
    if not persons:
        return None
    masks = [
        np.asarray(persons[segment_id]).astype(bool)
        for segment_id in segment_ids
        if segment_id in persons
    ]
    if not masks:
        return None
    out = np.zeros(masks[0].shape, dtype=bool)
    for mask in masks:
        if mask.shape == out.shape:
            out |= mask
    return out if out.any() else None


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


def black_image(image_shape: tuple[int, int]) -> Image.Image:
    height, width = image_shape
    return Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8)).convert("RGB")


def infer_image_shape(
    frame_paths: list[Path],
    mask_dict: dict[int, dict[int, np.ndarray]],
) -> tuple[int, int]:
    for frame_path in frame_paths:
        frame_bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame_bgr is not None:
            return int(frame_bgr.shape[0]), int(frame_bgr.shape[1])
    for persons in mask_dict.values():
        for mask in persons.values():
            mask_arr = np.asarray(mask)
            if mask_arr.ndim == 2:
                return int(mask_arr.shape[0]), int(mask_arr.shape[1])
    raise RuntimeError("Could not infer frame shape from frames or masks")


def make_full_frame_masked_image(
    frame_path: Path,
    mask: np.ndarray,
) -> tuple[Image.Image, tuple[int, int, int, int] | None, int, str]:
    frame_bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if frame_bgr is None:
        return black_image(mask.shape[:2]), None, 0, "unreadable_frame"
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mask_bool = resize_mask(mask, frame_rgb.shape[:2])
    bbox = mask_bbox(mask_bool)
    if bbox is None:
        return black_image(frame_rgb.shape[:2]), None, 0, "empty_mask_after_resize"
    masked_rgb = np.zeros_like(frame_rgb)
    masked_rgb[mask_bool] = frame_rgb[mask_bool]
    return Image.fromarray(masked_rgb).convert("RGB"), bbox, int(mask_bool.sum()), "masked"


def select_visualization_indices(num_frames: int, num_samples: int) -> set[int]:
    if num_samples <= 0 or num_frames <= 0:
        return set()
    if num_frames <= num_samples:
        return set(range(num_frames))
    return {
        int(round(value))
        for value in np.linspace(0, num_frames - 1, num=num_samples)
    }


def save_visualizations(
    masked_frames: list[MaskedFrame],
    assignment: Assignment,
    output_root: Path,
    num_samples: int,
) -> list[str]:
    selected = select_visualization_indices(len(masked_frames), num_samples)
    if not selected:
        return []

    vis_dir = (
        output_root
        / assignment.scene_key
        / f"person_{assignment.person_id}"
        / "visualizations"
    )
    vis_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in vis_dir.glob(f"{assignment.video_name}__*.jpg"):
        stale_path.unlink()
    paths = []
    for row_index, masked in enumerate(masked_frames):
        if row_index not in selected:
            continue
        vis_path = vis_dir / f"{assignment.video_name}__{masked.frame_stem}.jpg"
        masked.image.save(vis_path, quality=92)
        paths.append(str(vis_path))
    return paths


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU")
        return "cpu"
    return device_arg


def load_clip(model_id: str, device: str):
    try:
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as exc:
        raise SystemExit("Missing dependency: install transformers and retry.") from exc

    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).eval().to(device)
    return model, processor


def get_image_feature_tensor(model, pixel_values: torch.Tensor) -> torch.Tensor:
    outputs = model.get_image_features(pixel_values=pixel_values)
    if isinstance(outputs, torch.Tensor):
        return outputs
    if hasattr(outputs, "pooler_output"):
        return outputs.pooler_output
    if isinstance(outputs, tuple) and outputs:
        return outputs[0]
    raise TypeError(f"Unsupported CLIP image feature output type: {type(outputs)!r}")


@torch.inference_mode()
def extract_clip_features(
    model,
    processor,
    images: list[Image.Image],
    device: str,
    batch_size: int,
    expected_dim: int,
) -> torch.Tensor:
    features = []
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size]
        inputs = processor(images=batch, return_tensors="pt", padding=True)
        pixel_values = inputs["pixel_values"].to(device)
        batch_features = get_image_feature_tensor(model, pixel_values)
        batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        features.append(batch_features.detach().cpu().float())

    image_features = torch.cat(features, dim=0)
    if image_features.ndim != 2 or int(image_features.shape[1]) != expected_dim:
        raise RuntimeError(
            f"Expected feature shape (*, {expected_dim}), got {tuple(image_features.shape)}"
        )
    return image_features


def collect_masked_frames(
    assignment: Assignment,
    split_root: Path,
) -> tuple[list[MaskedFrame], dict[str, Any]]:
    frame_dir = split_root / "frame" / assignment.video_name
    mask_path = split_root / "refined_mask" / assignment.video_name / "mask.pt"
    frame_paths = list_frame_files(frame_dir)
    if not frame_paths:
        raise FileNotFoundError(f"No frames found: {frame_dir}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Missing mask: {mask_path}")

    mask_dict = load_mask_dict(mask_path)
    image_shape = infer_image_shape(frame_paths, mask_dict)
    masked_frames = []
    frames_with_assignment_mask = 0
    unreadable_frames = 0
    empty_after_resize = 0
    absent_segment_frames = 0
    black_frames = 0

    for frame_idx, frame_path in enumerate(frame_paths):
        mask = union_segment_mask(mask_dict.get(frame_idx), assignment.segment_ids)
        if mask is None:
            absent_segment_frames += 1
            black_frames += 1
            masked_frames.append(
                MaskedFrame(
                    frame_idx=int(frame_idx),
                    frame_stem=frame_path.stem,
                    frame_path=frame_path,
                    image=black_image(image_shape),
                    mask_bbox=None,
                    mask_pixel_count=0,
                    has_mask=False,
                    status="absent_segment",
                )
            )
            continue
        frames_with_assignment_mask += 1
        image_payload = make_full_frame_masked_image(frame_path, mask)
        image, bbox, pixel_count, status = image_payload
        if status == "unreadable_frame":
            unreadable_frames += 1
            black_frames += 1
        elif status == "empty_mask_after_resize":
            empty_after_resize += 1
            black_frames += 1
        masked_frames.append(
            MaskedFrame(
                frame_idx=int(frame_idx),
                frame_stem=frame_path.stem,
                frame_path=frame_path,
                image=image,
                mask_bbox=bbox,
                mask_pixel_count=int(pixel_count),
                has_mask=status == "masked",
                status=status,
            )
        )

    diagnostics = {
        "num_frame_files": len(frame_paths),
        "num_mask_frames": len(mask_dict),
        "num_frames_with_assignment_mask": int(frames_with_assignment_mask),
        "num_absent_segment_frames": int(absent_segment_frames),
        "num_black_frames": int(black_frames),
        "num_masked_frames": int(sum(1 for item in masked_frames if item.has_mask)),
        "num_unreadable_frames": int(unreadable_frames),
        "num_empty_after_resize": int(empty_after_resize),
    }
    return masked_frames, diagnostics


def output_path_for(output_root: Path, assignment: Assignment) -> Path:
    return (
        output_root
        / assignment.scene_key
        / f"person_{assignment.person_id}"
        / f"{assignment.video_name}.pt"
    )


def process_assignment(
    assignment: Assignment,
    data_root: Path,
    output_root: Path,
    model,
    processor,
    device: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    out_path = output_path_for(output_root, assignment)
    if out_path.exists() and not args.overwrite:
        print(f"[SKIP] existing: {out_path}")
        return {
            "status": "skipped_existing",
            "split": assignment.split,
            "scene_key": assignment.scene_key,
            "video_name": assignment.video_name,
            "person_id": int(assignment.person_id),
            "output_path": str(out_path),
        }

    split_root = data_root / assignment.split
    masked_frames, diagnostics = collect_masked_frames(assignment, split_root)
    if not masked_frames:
        raise RuntimeError(
            f"No masked frames for {assignment.split}/{assignment.video_name} "
            f"person_{assignment.person_id} segments={assignment.segment_ids}"
        )

    visualization_paths = save_visualizations(
        masked_frames=masked_frames,
        assignment=assignment,
        output_root=output_root,
        num_samples=args.num_vis_samples,
    )
    features = extract_clip_features(
        model=model,
        processor=processor,
        images=[item.image for item in masked_frames],
        device=device,
        batch_size=args.batch_size,
        expected_dim=args.expected_dim,
    )

    payload = {
        "features": features,
        "frame_indices": [int(item.frame_idx) for item in masked_frames],
        "frame_stems": [item.frame_stem for item in masked_frames],
        "segment_ids": [int(value) for value in assignment.segment_ids],
        "has_masks": [bool(item.has_mask) for item in masked_frames],
        "frame_statuses": [item.status for item in masked_frames],
        "mask_bboxes": [
            None
            if item.mask_bbox is None
            else [int(value) for value in item.mask_bbox]
            for item in masked_frames
        ],
        "mask_pixel_counts": [int(item.mask_pixel_count) for item in masked_frames],
        "source_frame_paths": [str(item.frame_path) for item in masked_frames],
        "source_mask_path": str(split_root / "refined_mask" / assignment.video_name / "mask.pt"),
        "source_mapping_path": str(assignment.mapping_path),
        "split": assignment.split,
        "scene_key": assignment.scene_key,
        "video_name": assignment.video_name,
        "camera_person": int(assignment.camera_person),
        "person_id": int(assignment.person_id),
        "model_id": args.model_id,
        "image_mode": "full_frame_masked_black_background_else_black_frame",
        "feature_dim": int(features.shape[1]),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)

    return {
        "status": "processed",
        "split": assignment.split,
        "scene_key": assignment.scene_key,
        "video_name": assignment.video_name,
        "camera_person": int(assignment.camera_person),
        "person_id": int(assignment.person_id),
        "segment_ids": [int(value) for value in assignment.segment_ids],
        "num_features": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "output_path": str(out_path),
        "visualization_paths": visualization_paths,
        "diagnostics": diagnostics,
    }


def write_scene_summaries(output_roots: dict[str, Path], results: list[dict[str, Any]]) -> None:
    by_scene: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_scene[(str(result["split"]), str(result["scene_key"]))].append(result)

    for (split, scene_key), scene_results in sorted(by_scene.items()):
        output_root = output_roots[split]
        status_counts = Counter(str(item["status"]) for item in scene_results)
        summary = {
            "split": split,
            "scene_key": scene_key,
            "status_counts": dict(sorted(status_counts.items())),
            "num_tracks": len(scene_results),
            "num_processed_tracks": int(status_counts.get("processed", 0)),
            "num_features": int(sum(int(item.get("num_features", 0)) for item in scene_results)),
            "tracks": scene_results,
        }
        write_json(output_root / scene_key / "summary.json", summary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract CLIP features from full-frame masked EgoCom person tracks."
    )
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--mapping_root", type=str, default=None)
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--scene_key", type=str, default=None)
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch_size", type=positive_int, default=32)
    parser.add_argument("--expected_dim", type=positive_int, default=768)
    parser.add_argument("--num_vis_samples", type=nonnegative_int, default=8)
    parser.add_argument("--limit", type=nonnegative_int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    data_root = Path(args.data_root)
    if not data_root.is_dir():
        print(f"Data root does not exist: {data_root}", file=sys.stderr)
        return 2

    splits = split_names(data_root, args.split)
    if args.mapping_root and len(splits) != 1:
        print("--mapping_root is only supported with a single split", file=sys.stderr)
        return 2
    if args.output_root and len(splits) != 1:
        print("--output_root is only supported with a single split", file=sys.stderr)
        return 2

    assignments = []
    output_roots: dict[str, Path] = {}
    for split in splits:
        mapping_root = Path(args.mapping_root) if args.mapping_root else default_mapping_root(data_root, split)
        output_roots[split] = Path(args.output_root) if args.output_root else default_output_root(data_root, split)
        split_assignments = collect_assignments_for_split(
            split=split,
            mapping_root=mapping_root,
            scene_key_filter=args.scene_key,
            video_filter=args.video,
        )
        assignments.extend(split_assignments)

    if args.limit is not None:
        assignments = assignments[: args.limit]
    if not assignments:
        print("No matching mapped person tracks found.")
        return 2

    device = resolve_device(args.device)
    print(f"Using device: {device}")
    print(f"Loading CLIP model: {args.model_id}")
    model, processor = load_clip(args.model_id, device)
    print(f"Found {len(assignments)} mapped person tracks")

    results = []
    processed = 0
    skipped = 0
    failed = 0
    for assignment in tqdm(assignments, desc="tracks"):
        try:
            result = process_assignment(
                assignment=assignment,
                data_root=data_root,
                output_root=output_roots[assignment.split],
                model=model,
                processor=processor,
                device=device,
                args=args,
            )
            results.append(result)
            if result["status"] == "processed":
                processed += 1
            elif result["status"].startswith("skipped"):
                skipped += 1
        except Exception as exc:
            failed += 1
            result = {
                "status": "failed",
                "split": assignment.split,
                "scene_key": assignment.scene_key,
                "video_name": assignment.video_name,
                "camera_person": int(assignment.camera_person),
                "person_id": int(assignment.person_id),
                "segment_ids": [int(value) for value in assignment.segment_ids],
                "error": str(exc),
            }
            results.append(result)
            print(
                f"[ERROR] {assignment.split}/{assignment.video_name} "
                f"person_{assignment.person_id}: {exc}",
                file=sys.stderr,
            )

    write_scene_summaries(output_roots, results)
    for split, output_root in sorted(output_roots.items()):
        split_results = [item for item in results if item["split"] == split]
        status_counts = Counter(str(item["status"]) for item in split_results)
        summary = {
            "split": split,
            "model_id": args.model_id,
            "image_mode": "full_frame_masked_black_background_else_black_frame",
            "status_counts": dict(sorted(status_counts.items())),
            "num_tracks": len(split_results),
            "num_processed_tracks": int(status_counts.get("processed", 0)),
            "num_features": int(sum(int(item.get("num_features", 0)) for item in split_results)),
            "tracks": split_results,
        }
        write_json(output_root / "summary.json", summary)

    print(f"Done. processed={processed} skipped={skipped} failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
