#!/usr/bin/env python3
"""
Extract mask-pooled positional encoding features for EgoCom mapped person tracks.

Inputs:
  /home/prj/data/egocom_holdout/1min/{split}/person_face_mapping/*/remap_all_chunks.json
  /home/prj/data/egocom_holdout/1min/{split}/refined_mask/{video}/mask.pt
  /home/prj/data/egocom_holdout/1min/{split}/frame/{video}/*.jpg

Outputs:
  /home/prj/data/egocom_holdout/1min/{split}/person_pe_features/{scene}/person_{id}/{video}.pt
  /home/prj/data/egocom_holdout/1min/{split}/person_pe_features/{scene}/summary.json
  /home/prj/data/egocom_holdout/1min/{split}/person_pe_features/summary.json
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
from tqdm import tqdm


DEFAULT_DATA_ROOT = "/home/prj/data/egocom_holdout/1min"
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
class PooledFrame:
    frame_idx: int
    frame_stem: str
    frame_path: Path
    feature: torch.Tensor
    mask_pixel_count: int
    mask_grid_sum: float
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
    return data_root / split / "person_pe_features"


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


def build_sincos_pe(hidden_size: int, patches_per_side: int, pe_scale: float) -> torch.Tensor:
    try:
        from transformers.models.vit_mae.modeling_vit_mae import get_2d_sincos_pos_embed
    except ImportError as exc:
        raise SystemExit("Missing dependency: install transformers and retry.") from exc
    if hidden_size % 4 != 0:
        raise ValueError("--hidden_size must be divisible by 4 for 2D sinusoidal PE")
    pe = get_2d_sincos_pos_embed(hidden_size, patches_per_side) * pe_scale
    return torch.tensor(pe, dtype=torch.float32)


def rotary_axis_embedding(coords: torch.Tensor, axis_dim: int) -> torch.Tensor:
    inv_freq = 1.0 / (
        10000
        ** (torch.arange(0, axis_dim, 2, dtype=torch.float32) / float(axis_dim))
    )
    freqs = torch.einsum("n,d->nd", coords.float(), inv_freq)
    return torch.cat([freqs.sin(), freqs.cos()], dim=-1)


def build_rope_pe(hidden_size: int, patches_per_side: int, pe_scale: float) -> torch.Tensor:
    if hidden_size % 4 != 0:
        raise ValueError("--hidden_size must be divisible by 4 for 2D RoPE PE")
    axis_dim = hidden_size // 2
    coords = torch.arange(patches_per_side, dtype=torch.float32)
    y_coords, x_coords = torch.meshgrid(coords, coords, indexing="ij")
    y_embed = rotary_axis_embedding(y_coords.reshape(-1), axis_dim)
    x_embed = rotary_axis_embedding(x_coords.reshape(-1), axis_dim)
    return torch.cat([y_embed, x_embed], dim=-1).float() * float(pe_scale)


def build_pe_table(pe_type: str, hidden_size: int, patches_per_side: int, pe_scale: float) -> torch.Tensor:
    if pe_type == "sinusoidal":
        return build_sincos_pe(hidden_size, patches_per_side, pe_scale)
    if pe_type == "rope":
        return build_rope_pe(hidden_size, patches_per_side, pe_scale)
    raise ValueError(f"Unsupported PE type: {pe_type}")


def resize_mask_to_grid(mask: np.ndarray, patches_per_side: int) -> np.ndarray:
    mask_float = np.asarray(mask).astype(np.float32)
    if mask_float.shape[:2] == (patches_per_side, patches_per_side):
        return mask_float
    resized = cv2.resize(
        mask_float,
        (patches_per_side, patches_per_side),
        interpolation=cv2.INTER_AREA,
    )
    return np.clip(resized, 0.0, 1.0).astype(np.float32)


def mask_pooling(features: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    bsz = masks.shape[0]
    channels = features.shape[-1]
    features = features.view(-1, channels)
    masks = masks.view(bsz, -1)
    return torch.matmul(masks, features) / (masks.sum(dim=-1, keepdim=True) + 1e-6)


def collect_pooled_frames(
    assignment: Assignment,
    split_root: Path,
    pe_grid: torch.Tensor,
    patches_per_side: int,
) -> tuple[list[PooledFrame], dict[str, Any]]:
    frame_dir = split_root / "frame" / assignment.video_name
    mask_path = split_root / "refined_mask" / assignment.video_name / "mask.pt"
    frame_paths = list_frame_files(frame_dir)
    if not frame_paths:
        raise FileNotFoundError(f"No frames found: {frame_dir}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Missing mask: {mask_path}")

    mask_dict = load_mask_dict(mask_path)
    pooled_frames = []
    zero_feature = torch.zeros((pe_grid.shape[-1],), dtype=torch.float32)
    frames_with_assignment_mask = 0
    absent_segment_frames = 0
    empty_after_resize = 0

    for frame_idx, frame_path in enumerate(frame_paths):
        mask = union_segment_mask(mask_dict.get(frame_idx), assignment.segment_ids)
        if mask is None:
            absent_segment_frames += 1
            pooled_frames.append(
                PooledFrame(
                    frame_idx=int(frame_idx),
                    frame_stem=frame_path.stem,
                    frame_path=frame_path,
                    feature=zero_feature.clone(),
                    mask_pixel_count=0,
                    mask_grid_sum=0.0,
                    has_mask=False,
                    status="absent_segment",
                )
            )
            continue

        resized_mask = resize_mask_to_grid(mask, patches_per_side)
        mask_grid_sum = float(resized_mask.sum())
        if mask_grid_sum <= 0.0:
            empty_after_resize += 1
            pooled_frames.append(
                PooledFrame(
                    frame_idx=int(frame_idx),
                    frame_stem=frame_path.stem,
                    frame_path=frame_path,
                    feature=zero_feature.clone(),
                    mask_pixel_count=int(mask.sum()),
                    mask_grid_sum=0.0,
                    has_mask=False,
                    status="empty_mask_after_resize",
                )
            )
            continue

        frames_with_assignment_mask += 1
        mask_tensor = torch.from_numpy(resized_mask).view(1, patches_per_side, patches_per_side).float()
        feature = mask_pooling(pe_grid, mask_tensor)[0].float()
        pooled_frames.append(
            PooledFrame(
                frame_idx=int(frame_idx),
                frame_stem=frame_path.stem,
                frame_path=frame_path,
                feature=feature,
                mask_pixel_count=int(mask.sum()),
                mask_grid_sum=mask_grid_sum,
                has_mask=True,
                status="masked",
            )
        )

    diagnostics = {
        "num_frame_files": len(frame_paths),
        "num_mask_frames": len(mask_dict),
        "num_frames_with_assignment_mask": int(frames_with_assignment_mask),
        "num_absent_segment_frames": int(absent_segment_frames),
        "num_masked_frames": int(sum(1 for item in pooled_frames if item.has_mask)),
        "num_empty_after_resize": int(empty_after_resize),
    }
    return pooled_frames, diagnostics


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
    pe_grid: torch.Tensor,
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
    pooled_frames, diagnostics = collect_pooled_frames(
        assignment=assignment,
        split_root=split_root,
        pe_grid=pe_grid,
        patches_per_side=args.patches_per_side,
    )
    if not pooled_frames:
        raise RuntimeError(
            f"No PE frames for {assignment.split}/{assignment.video_name} "
            f"person_{assignment.person_id} segments={assignment.segment_ids}"
        )

    features = torch.stack([item.feature for item in pooled_frames], dim=0).float()
    payload = {
        "features": features,
        "frame_indices": [int(item.frame_idx) for item in pooled_frames],
        "frame_stems": [item.frame_stem for item in pooled_frames],
        "segment_ids": [int(value) for value in assignment.segment_ids],
        "has_masks": [bool(item.has_mask) for item in pooled_frames],
        "frame_statuses": [item.status for item in pooled_frames],
        "mask_pixel_counts": [int(item.mask_pixel_count) for item in pooled_frames],
        "mask_grid_sums": [float(item.mask_grid_sum) for item in pooled_frames],
        "source_frame_paths": [str(item.frame_path) for item in pooled_frames],
        "source_mask_path": str(split_root / "refined_mask" / assignment.video_name / "mask.pt"),
        "source_mapping_path": str(assignment.mapping_path),
        "split": assignment.split,
        "scene_key": assignment.scene_key,
        "video_name": assignment.video_name,
        "camera_person": int(assignment.camera_person),
        "person_id": int(assignment.person_id),
        "pe_type": args.pe_type,
        "patches_per_side": int(args.patches_per_side),
        "hidden_size": int(args.hidden_size),
        "pe_scale": float(args.pe_scale),
        "pooling": "mask_weighted_average_resized_to_patch_grid",
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
        description="Extract mask-pooled positional encoding features for EgoCom person tracks."
    )
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--mapping_root", type=str, default=None)
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--scene_key", type=str, default=None)
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--pe_type", choices=("sinusoidal", "rope"), default="sinusoidal")
    parser.add_argument("--patches_per_side", type=positive_int, default=16)
    parser.add_argument("--hidden_size", type=positive_int, default=768)
    parser.add_argument("--pe_scale", type=float, default=1.0)
    parser.add_argument("--limit", type=nonnegative_int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    data_root = Path(args.data_root)
    if not data_root.is_dir():
        print(f"Data root does not exist: {data_root}", file=sys.stderr)
        return 2
    if args.hidden_size % 4 != 0:
        print("--hidden_size must be divisible by 4", file=sys.stderr)
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

    pe_grid = build_pe_table(
        pe_type=args.pe_type,
        hidden_size=args.hidden_size,
        patches_per_side=args.patches_per_side,
        pe_scale=args.pe_scale,
    ).view(args.patches_per_side, args.patches_per_side, args.hidden_size)

    print(
        f"Using {args.pe_type} PE: "
        f"grid={args.patches_per_side}x{args.patches_per_side} dim={args.hidden_size}"
    )
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
                pe_grid=pe_grid,
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
            "pe_type": args.pe_type,
            "patches_per_side": int(args.patches_per_side),
            "hidden_size": int(args.hidden_size),
            "pe_scale": float(args.pe_scale),
            "pooling": "mask_weighted_average_resized_to_patch_grid",
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
