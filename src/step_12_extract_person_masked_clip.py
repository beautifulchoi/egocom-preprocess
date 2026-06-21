#!/usr/bin/env python3
"""
Extract mask-pooled CLIP patch-token features for EgoCom mapped person tracks.

Unlike src/step_11_extract_person_visual_clip.py, this extractor does not black out the
image before CLIP and does not use CLIP's CLS/global image feature. It runs CLIP
on the full RGB frame, reshapes patch tokens from a selected vision hidden
state to the patch grid, resizes the person mask to that grid, and mask-pools
the patch-token features.

Inputs:
  /home/prj/data/egocom_holdout/1min/{split}/person_face_mapping/*/remap_all_chunks.json
  /home/prj/data/egocom_holdout/1min/{split}/refined_mask/{video}/mask.pt
  /home/prj/data/egocom_holdout/1min/{split}/frame/{video}/*.jpg

Outputs:
  /home/prj/data/egocom_holdout/1min/{split}/person_masked_clip_features/{scene}/person_{id}/{video}.pt
  /home/prj/data/egocom_holdout/1min/{split}/person_masked_clip_features/{scene}/summary.json
  /home/prj/data/egocom_holdout/1min/{split}/person_masked_clip_features/summary.json
"""

from __future__ import annotations

import argparse
import math
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

from step_11_extract_person_visual_clip import (
    DEFAULT_DATA_ROOT,
    DEFAULT_MODEL_ID,
    Assignment,
    collect_assignments_for_split,
    default_mapping_root,
    list_frame_files,
    load_clip,
    load_mask_dict,
    nonnegative_int,
    positive_int,
    resolve_device,
    split_names,
    union_segment_mask,
    write_json,
)


@dataclass
class FrameMask:
    frame_idx: int
    frame_stem: str
    frame_path: Path
    image: Image.Image | None
    mask: np.ndarray | None
    mask_pixel_count: int
    mask_grid_sum: float
    has_mask: bool
    status: str


def default_output_root(data_root: Path, split: str) -> Path:
    return data_root / split / "person_masked_clip_features"


def load_rgb_image(frame_path: Path) -> Image.Image | None:
    frame_bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if frame_bgr is None:
        return None
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb).convert("RGB")


def resize_mask_to_grid(mask: np.ndarray, patches_per_side: int) -> np.ndarray:
    mask_float = np.asarray(mask).astype(np.float32)
    if mask_float.shape[:2] == (patches_per_side, patches_per_side):
        return np.clip(mask_float, 0.0, 1.0).astype(np.float32)
    resized = cv2.resize(
        mask_float,
        (patches_per_side, patches_per_side),
        interpolation=cv2.INTER_NEAREST,
    )
    return np.clip(resized, 0.0, 1.0).astype(np.float32)


def infer_patch_grid(num_patch_tokens: int) -> int:
    patches_per_side = int(math.isqrt(num_patch_tokens))
    if patches_per_side * patches_per_side != num_patch_tokens:
        raise RuntimeError(
            "Expected square CLIP patch-token grid after removing CLS, "
            f"got {num_patch_tokens} patch tokens"
        )
    return patches_per_side


def select_hidden_state(outputs: Any, layer: int) -> tuple[torch.Tensor, int]:
    if layer == -1:
        return outputs.last_hidden_state, -1

    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError("CLIP vision outputs did not include hidden_states")

    num_states = len(hidden_states)
    selected_index = layer if layer >= 0 else num_states + layer
    if selected_index < 0 or selected_index >= num_states:
        raise ValueError(
            f"--layer {layer} is out of range for {num_states} hidden states"
        )
    return hidden_states[selected_index], selected_index


def mask_pool_patch_tokens(
    patch_tokens: torch.Tensor,
    masks: torch.Tensor,
) -> torch.Tensor:
    bsz, num_patch_tokens, channels = patch_tokens.shape
    masks = masks.view(bsz, num_patch_tokens).to(
        device=patch_tokens.device,
        dtype=patch_tokens.dtype,
    )
    pooled = torch.bmm(masks.unsqueeze(1), patch_tokens).squeeze(1)
    return pooled / masks.sum(dim=-1, keepdim=True).clamp_min(1e-6)


def collect_frame_masks(
    assignment: Assignment,
    split_root: Path,
) -> tuple[list[FrameMask], dict[str, Any]]:
    frame_dir = split_root / "frame" / assignment.video_name
    mask_path = split_root / "refined_mask" / assignment.video_name / "mask.pt"
    frame_paths = list_frame_files(frame_dir)
    if not frame_paths:
        raise FileNotFoundError(f"No frames found: {frame_dir}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Missing mask: {mask_path}")

    mask_dict = load_mask_dict(mask_path)
    frame_masks = []
    frames_with_assignment_mask = 0
    unreadable_frames = 0
    absent_segment_frames = 0

    for frame_idx, frame_path in enumerate(frame_paths):
        image = load_rgb_image(frame_path)
        if image is None:
            unreadable_frames += 1
            frame_masks.append(
                FrameMask(
                    frame_idx=int(frame_idx),
                    frame_stem=frame_path.stem,
                    frame_path=frame_path,
                    image=None,
                    mask=None,
                    mask_pixel_count=0,
                    mask_grid_sum=0.0,
                    has_mask=False,
                    status="unreadable_frame",
                )
            )
            continue

        mask = union_segment_mask(mask_dict.get(frame_idx), assignment.segment_ids)
        if mask is None:
            absent_segment_frames += 1
            frame_masks.append(
                FrameMask(
                    frame_idx=int(frame_idx),
                    frame_stem=frame_path.stem,
                    frame_path=frame_path,
                    image=image,
                    mask=None,
                    mask_pixel_count=0,
                    mask_grid_sum=0.0,
                    has_mask=False,
                    status="absent_segment",
                )
            )
            continue

        frames_with_assignment_mask += 1
        frame_masks.append(
            FrameMask(
                frame_idx=int(frame_idx),
                frame_stem=frame_path.stem,
                frame_path=frame_path,
                image=image,
                mask=mask,
                mask_pixel_count=int(mask.sum()),
                mask_grid_sum=0.0,
                has_mask=True,
                status="masked",
            )
        )

    diagnostics = {
        "num_frame_files": len(frame_paths),
        "num_mask_frames": len(mask_dict),
        "num_frames_with_assignment_mask": int(frames_with_assignment_mask),
        "num_absent_segment_frames": int(absent_segment_frames),
        "num_masked_frames": int(sum(1 for item in frame_masks if item.has_mask)),
        "num_unreadable_frames": int(unreadable_frames),
    }
    return frame_masks, diagnostics


@torch.inference_mode()
def extract_masked_clip_features(
    model,
    processor,
    frame_masks: list[FrameMask],
    device: str,
    batch_size: int,
    expected_dim: int | None,
    layer: int,
    normalize: bool,
) -> tuple[torch.Tensor, int, int, int]:
    output_rows: list[torch.Tensor | None] = [None] * len(frame_masks)
    valid_indices = [
        index
        for index, item in enumerate(frame_masks)
        if item.image is not None and item.mask is not None and item.has_mask
    ]
    if valid_indices:
        clip_indices = valid_indices
    else:
        clip_indices = [
            index
            for index, item in enumerate(frame_masks)
            if item.image is not None
        ][:1]
    patches_per_side: int | None = None
    num_patch_tokens: int | None = None
    selected_hidden_state_index: int | None = None
    feature_dim: int | None = None

    for start in range(0, len(clip_indices), batch_size):
        batch_indices = clip_indices[start : start + batch_size]
        batch_items = [frame_masks[index] for index in batch_indices]
        inputs = processor(
            images=[item.image for item in batch_items],
            return_tensors="pt",
            padding=True,
        )
        pixel_values = inputs["pixel_values"].to(device)
        outputs = model.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_state, hidden_state_index = select_hidden_state(outputs, layer)
        selected_hidden_state_index = hidden_state_index

        patch_tokens = hidden_state[:, 1:, :]
        if patch_tokens.ndim != 3:
            raise RuntimeError(
                f"Expected patch tokens to be 3D, got {tuple(patch_tokens.shape)}"
            )

        batch_feature_dim = int(patch_tokens.shape[-1])
        if feature_dim is None:
            feature_dim = batch_feature_dim
        elif feature_dim != batch_feature_dim:
            raise RuntimeError(
                f"Feature dimension changed across batches: {feature_dim} vs {batch_feature_dim}"
            )

        batch_num_patch_tokens = int(patch_tokens.shape[1])
        batch_patches_per_side = infer_patch_grid(batch_num_patch_tokens)
        if patches_per_side is None:
            patches_per_side = batch_patches_per_side
            num_patch_tokens = batch_num_patch_tokens
        elif patches_per_side != batch_patches_per_side:
            raise RuntimeError(
                "Patch grid changed across batches: "
                f"{patches_per_side} vs {batch_patches_per_side}"
            )

        mask_arrays = []
        mask_row_offsets = []
        for row_offset, item in enumerate(batch_items):
            if item.mask is None or not item.has_mask:
                continue
            resized_mask = resize_mask_to_grid(item.mask, batch_patches_per_side)
            item.mask_grid_sum = float(resized_mask.sum())
            if item.mask_grid_sum <= 0.0:
                item.has_mask = False
                item.status = "empty_mask_after_resize"
                continue
            mask_arrays.append(resized_mask)
            mask_row_offsets.append(row_offset)

        if not mask_arrays:
            continue

        mask_tensor = torch.from_numpy(np.stack(mask_arrays, axis=0)).to(device)
        pooled = mask_pool_patch_tokens(patch_tokens[mask_row_offsets], mask_tensor)
        if normalize:
            pooled = pooled / pooled.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        pooled = pooled.detach().cpu().float()

        for pooled_offset, row_offset in enumerate(mask_row_offsets):
            item_index = batch_indices[row_offset]
            item = frame_masks[item_index]
            if item.has_mask:
                output_rows[item_index] = pooled[pooled_offset]

    if patches_per_side is None or num_patch_tokens is None or feature_dim is None:
        raise RuntimeError("No readable frames were available to run CLIP")
    if selected_hidden_state_index is None:
        selected_hidden_state_index = -1 if layer == -1 else layer

    if expected_dim is not None and feature_dim != expected_dim:
        raise RuntimeError(
            f"Expected feature shape (*, {expected_dim}), got hidden size {feature_dim}"
        )

    zero_feature = torch.zeros((feature_dim,), dtype=torch.float32)
    features = torch.stack(
        [row if row is not None else zero_feature.clone() for row in output_rows],
        dim=0,
    ).float()
    return features, patches_per_side, num_patch_tokens, selected_hidden_state_index


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
    frame_masks, diagnostics = collect_frame_masks(assignment, split_root)
    if not frame_masks:
        raise RuntimeError(
            f"No frames for {assignment.split}/{assignment.video_name} "
            f"person_{assignment.person_id} segments={assignment.segment_ids}"
        )

    features, patches_per_side, num_patch_tokens, selected_hidden_state_index = (
        extract_masked_clip_features(
            model=model,
            processor=processor,
            frame_masks=frame_masks,
            device=device,
            batch_size=args.batch_size,
            expected_dim=args.expected_dim,
            layer=args.layer,
            normalize=not args.no_normalize,
        )
    )

    diagnostics["num_empty_after_resize"] = int(
        sum(1 for item in frame_masks if item.status == "empty_mask_after_resize")
    )
    diagnostics["num_masked_frames"] = int(sum(1 for item in frame_masks if item.has_mask))

    payload = {
        "features": features,
        "frame_indices": [int(item.frame_idx) for item in frame_masks],
        "frame_stems": [item.frame_stem for item in frame_masks],
        "segment_ids": [int(value) for value in assignment.segment_ids],
        "has_masks": [bool(item.has_mask) for item in frame_masks],
        "frame_statuses": [item.status for item in frame_masks],
        "mask_pixel_counts": [int(item.mask_pixel_count) for item in frame_masks],
        "mask_grid_sums": [float(item.mask_grid_sum) for item in frame_masks],
        "source_frame_paths": [str(item.frame_path) for item in frame_masks],
        "source_mask_path": str(split_root / "refined_mask" / assignment.video_name / "mask.pt"),
        "source_mapping_path": str(assignment.mapping_path),
        "split": assignment.split,
        "scene_key": assignment.scene_key,
        "video_name": assignment.video_name,
        "camera_person": int(assignment.camera_person),
        "person_id": int(assignment.person_id),
        "model_id": args.model_id,
        "image_mode": "full_frame_rgb",
        "layer": int(args.layer),
        "selected_hidden_state_index": int(selected_hidden_state_index),
        "patches_per_side": int(patches_per_side),
        "num_patch_tokens": int(num_patch_tokens),
        "normalize_features": bool(not args.no_normalize),
        "pooling": "mask_weighted_average_resized_to_clip_patch_grid",
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
        "patches_per_side": int(patches_per_side),
        "num_patch_tokens": int(num_patch_tokens),
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
        description="Extract mask-pooled CLIP patch-token features for EgoCom person tracks."
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
    parser.add_argument("--expected_dim", type=positive_int, default=None)
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--limit", type=nonnegative_int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--no_normalize",
        action="store_true",
        help="Save raw mask-pooled hidden states instead of L2-normalized vectors.",
    )
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
            "image_mode": "full_frame_rgb",
            "layer": int(args.layer),
            "normalize_features": bool(not args.no_normalize),
            "pooling": "mask_weighted_average_resized_to_clip_patch_grid",
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
