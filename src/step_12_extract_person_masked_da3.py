#!/usr/bin/env python3
"""
Extract mask-pooled Depth Anything 3 feature-map vectors for EgoCom person tracks.

This extractor mirrors src/step_11_extract_person_masked_clip.py at the data/output level,
but uses DA3 spatial feature maps instead of CLIP patch tokens. It does not edit
the vendored DA3 source tree; it imports DA3 from _external/depth-anything-3/src
and uses the model's existing backbone feature-export path.

Inputs:
  /home/prj/data/egocom_holdout/1min/{split}/person_face_mapping/*/remap_all_chunks.json
  /home/prj/data/egocom_holdout/1min/{split}/refined_mask/{video}/mask.pt
  /home/prj/data/egocom_holdout/1min/{split}/frame/{video}/*.jpg

Outputs:
  /home/prj/data/egocom_holdout/1min/{split}/person_masked_da3_features/{scene}/person_{id}/{video}.pt
  /home/prj/data/egocom_holdout/1min/{split}/person_masked_da3_features/{scene}/summary.json
  /home/prj/data/egocom_holdout/1min/{split}/person_masked_da3_features/summary.json
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from step_11_extract_person_masked_clip import FrameMask, collect_frame_masks
from step_10_extract_person_visual_clip import (
    DEFAULT_DATA_ROOT,
    Assignment,
    collect_assignments_for_split,
    default_mapping_root,
    nonnegative_int,
    positive_int,
    resolve_device,
    split_names,
    write_json,
)


DA3_SRC = Path(__file__).resolve().parents[1] / "_external" / "depth-anything-3" / "src"
DEFAULT_MODEL_ID = "depth-anything/DA3METRIC-LARGE"
DEFAULT_LAYER = 23
DEFAULT_PROCESS_RES = 504
DEFAULT_PROCESS_RES_METHOD = "upper_bound_resize"
PATCH_SIZE = 14


def default_output_root(data_root: Path, split: str) -> Path:
    return data_root / split / "person_masked_da3_features"


def output_path_for(output_root: Path, assignment: Assignment) -> Path:
    return (
        output_root
        / assignment.scene_key
        / f"person_{assignment.person_id}"
        / f"{assignment.video_name}.pt"
    )


def resize_mask_nearest(mask: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    mask_uint8 = np.asarray(mask).astype(np.uint8)
    if mask_uint8.shape[:2] == (target_h, target_w):
        return mask_uint8.astype(bool)
    resized = cv2.resize(
        mask_uint8,
        (target_w, target_h),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized.astype(bool)


def boundary_resize_shape(
    width: int,
    height: int,
    target_size: int,
    method: str,
) -> tuple[int, int]:
    if method in ("upper_bound_resize", "upper_bound_crop"):
        bound = max(width, height)
    elif method in ("lower_bound_resize", "lower_bound_crop"):
        bound = min(width, height)
    else:
        raise ValueError(f"Unsupported process_res_method: {method}")

    if bound == target_size:
        return width, height
    scale = target_size / float(bound)
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def nearest_multiple(value: int, patch: int) -> int:
    down = (value // patch) * patch
    up = down + patch
    return max(1, up if abs(up - value) <= abs(value - down) else down)


def center_crop_mask(mask: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    height, width = mask.shape[:2]
    if target_h > height or target_w > width:
        return resize_mask_nearest(mask, target_h, target_w)
    top = max(0, (height - target_h) // 2)
    left = max(0, (width - target_w) // 2)
    return mask[top : top + target_h, left : left + target_w].astype(bool)


def transform_mask_to_da3_processed_grid(
    mask: np.ndarray,
    image: Image.Image,
    target_processed_hw: tuple[int, int],
    process_res: int,
    process_res_method: str,
) -> np.ndarray:
    """Apply DA3's resize/crop geometry to a mask, then align to batch output size."""
    orig_w, orig_h = image.size
    mask_processed = resize_mask_nearest(mask, orig_h, orig_w)

    resized_w, resized_h = boundary_resize_shape(
        orig_w,
        orig_h,
        process_res,
        process_res_method,
    )
    mask_processed = resize_mask_nearest(mask_processed, resized_h, resized_w)

    if process_res_method.endswith("resize"):
        final_w = nearest_multiple(resized_w, PATCH_SIZE)
        final_h = nearest_multiple(resized_h, PATCH_SIZE)
        mask_processed = resize_mask_nearest(mask_processed, final_h, final_w)
    elif process_res_method.endswith("crop"):
        final_w = (resized_w // PATCH_SIZE) * PATCH_SIZE
        final_h = (resized_h // PATCH_SIZE) * PATCH_SIZE
        mask_processed = center_crop_mask(mask_processed, final_h, final_w)
    else:
        raise ValueError(f"Unsupported process_res_method: {process_res_method}")

    target_h, target_w = target_processed_hw
    if mask_processed.shape[:2] != (target_h, target_w):
        mask_processed = center_crop_mask(mask_processed, target_h, target_w)
    return mask_processed.astype(bool)


def resize_mask_to_feature_grid(
    mask: np.ndarray,
    image: Image.Image,
    feature_h: int,
    feature_w: int,
    process_res: int,
    process_res_method: str,
) -> np.ndarray:
    processed_mask = transform_mask_to_da3_processed_grid(
        mask=mask,
        image=image,
        target_processed_hw=(feature_h * PATCH_SIZE, feature_w * PATCH_SIZE),
        process_res=process_res,
        process_res_method=process_res_method,
    )
    feature_mask = cv2.resize(
        processed_mask.astype(np.float32),
        (feature_w, feature_h),
        interpolation=cv2.INTER_NEAREST,
    )
    return np.clip(feature_mask, 0.0, 1.0).astype(np.float32)


def mask_pool_feature_maps(
    feature_maps: torch.Tensor,
    masks: torch.Tensor,
) -> torch.Tensor:
    if feature_maps.ndim != 4:
        raise RuntimeError(f"Expected DA3 feature maps to be 4D, got {tuple(feature_maps.shape)}")

    batch_size, feature_h, feature_w, channels = feature_maps.shape
    features_flat = feature_maps.view(batch_size, feature_h * feature_w, channels)
    masks_flat = masks.view(batch_size, feature_h * feature_w).to(
        device=feature_maps.device,
        dtype=feature_maps.dtype,
    )
    pooled = torch.bmm(masks_flat.unsqueeze(1), features_flat).squeeze(1)
    return pooled / masks_flat.sum(dim=-1, keepdim=True).clamp_min(1e-6)


class Da3FeatureExtractor:
    def __init__(self, model_id: str, device: str) -> None:
        if str(DA3_SRC) not in sys.path:
            sys.path.insert(0, str(DA3_SRC))
        try:
            from depth_anything_3.api import DepthAnything3
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Failed to import Depth Anything 3. Install the dependencies from "
                f"{DA3_SRC.parent / 'requirements.txt'}; missing module: {exc.name}"
            ) from exc

        self.device = torch.device(device)
        self.model = DepthAnything3.from_pretrained(model_id).to(device=self.device)
        self.model.eval()
        self.feature_net = self._select_feature_net(self.model.model)

    @staticmethod
    def _select_feature_net(model: torch.nn.Module) -> torch.nn.Module:
        if hasattr(model, "backbone") and hasattr(model, "_extract_auxiliary_features"):
            return model
        if hasattr(model, "da3"):
            nested_net = getattr(model, "da3")
            if hasattr(nested_net, "backbone") and hasattr(nested_net, "_extract_auxiliary_features"):
                return nested_net
        raise TypeError(f"Unsupported DA3 model structure for feature extraction: {type(model)!r}")

    @torch.inference_mode()
    def extract_feature_maps(
        self,
        images: list[Image.Image],
        layer: int,
        process_res: int,
        process_res_method: str,
    ) -> torch.Tensor:
        if not images:
            raise ValueError("No images provided for DA3 feature extraction")

        imgs_cpu, _, _ = self.model.input_processor(
            image=images,
            process_res=process_res,
            process_res_method=process_res_method,
            num_workers=1,
            print_progress=False,
            sequential=True,
        )
        imgs = imgs_cpu.to(self.device, non_blocking=True).float()
        model_input = imgs[:, None]
        height, width = int(model_input.shape[-2]), int(model_input.shape[-1])

        autocast_enabled = self.device.type == "cuda"
        autocast_dtype = (
            torch.bfloat16
            if autocast_enabled and torch.cuda.is_bf16_supported()
            else torch.float16
        )
        with torch.autocast(
            device_type=self.device.type,
            dtype=autocast_dtype,
            enabled=autocast_enabled,
        ):
            _, aux_feats = self.feature_net.backbone(
                model_input,
                cam_token=None,
                export_feat_layers=[layer],
                ref_view_strategy="first",
            )
            aux = self.feature_net._extract_auxiliary_features(
                aux_feats,
                [layer],
                height,
                width,
            )

        key = f"feat_layer_{layer}"
        if key not in aux:
            raise RuntimeError(f"DA3 did not return requested feature layer: {key}")
        feature_maps = aux[key]
        if feature_maps.ndim != 5 or int(feature_maps.shape[1]) != 1:
            raise RuntimeError(
                "Expected DA3 auxiliary feature shape (B, 1, Hf, Wf, C), "
                f"got {tuple(feature_maps.shape)}"
            )
        return feature_maps[:, 0].detach().cpu().float()


@torch.inference_mode()
def extract_masked_da3_features(
    extractor: Da3FeatureExtractor,
    frame_masks: list[FrameMask],
    batch_size: int,
    expected_dim: int | None,
    layer: int,
    process_res: int,
    process_res_method: str,
    normalize: bool,
) -> tuple[torch.Tensor, int, list[tuple[int, int] | None]]:
    output_rows: list[torch.Tensor | None] = [None] * len(frame_masks)
    feature_grid_shapes: list[tuple[int, int] | None] = [None] * len(frame_masks)
    valid_indices = [
        index
        for index, item in enumerate(frame_masks)
        if item.image is not None and item.mask is not None and item.has_mask
    ]
    if valid_indices:
        da3_indices = valid_indices
    else:
        da3_indices = [
            index
            for index, item in enumerate(frame_masks)
            if item.image is not None
        ][:1]

    feature_dim: int | None = None
    for start in range(0, len(da3_indices), batch_size):
        batch_indices = da3_indices[start : start + batch_size]
        batch_items = [frame_masks[index] for index in batch_indices]
        images = [item.image for item in batch_items]
        if any(image is None for image in images):
            raise RuntimeError("Internal error: DA3 batch contains unreadable frame")

        feature_maps = extractor.extract_feature_maps(
            images=[image for image in images if image is not None],
            layer=layer,
            process_res=process_res,
            process_res_method=process_res_method,
        )

        batch_feature_dim = int(feature_maps.shape[-1])
        if feature_dim is None:
            feature_dim = batch_feature_dim
        elif feature_dim != batch_feature_dim:
            raise RuntimeError(
                f"Feature dimension changed across batches: {feature_dim} vs {batch_feature_dim}"
            )

        feature_h, feature_w = int(feature_maps.shape[1]), int(feature_maps.shape[2])
        mask_arrays = []
        mask_row_offsets = []
        for row_offset, item in enumerate(batch_items):
            feature_grid_shapes[batch_indices[row_offset]] = (feature_h, feature_w)
            if item.mask is None or item.image is None or not item.has_mask:
                continue
            resized_mask = resize_mask_to_feature_grid(
                mask=item.mask,
                image=item.image,
                feature_h=feature_h,
                feature_w=feature_w,
                process_res=process_res,
                process_res_method=process_res_method,
            )
            item.mask_grid_sum = float(resized_mask.sum())
            if item.mask_grid_sum <= 0.0:
                item.has_mask = False
                item.status = "empty_mask_after_resize"
                continue
            mask_arrays.append(resized_mask)
            mask_row_offsets.append(row_offset)

        if not mask_arrays:
            continue

        mask_tensor = torch.from_numpy(np.stack(mask_arrays, axis=0))
        pooled = mask_pool_feature_maps(feature_maps[mask_row_offsets], mask_tensor)
        if normalize:
            pooled = pooled / pooled.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        pooled = pooled.detach().cpu().float()

        for pooled_offset, row_offset in enumerate(mask_row_offsets):
            item_index = batch_indices[row_offset]
            item = frame_masks[item_index]
            if item.has_mask:
                output_rows[item_index] = pooled[pooled_offset]

    if feature_dim is None:
        raise RuntimeError("No readable frames were available to run DA3")
    if expected_dim is not None and feature_dim != expected_dim:
        raise RuntimeError(
            f"Expected feature shape (*, {expected_dim}), got hidden size {feature_dim}"
        )

    zero_feature = torch.zeros((feature_dim,), dtype=torch.float32)
    features = torch.stack(
        [row if row is not None else zero_feature.clone() for row in output_rows],
        dim=0,
    ).float()
    return features, feature_dim, feature_grid_shapes


def process_assignment(
    assignment: Assignment,
    data_root: Path,
    output_root: Path,
    extractor: Da3FeatureExtractor,
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

    features, feature_dim, feature_grid_shapes = extract_masked_da3_features(
        extractor=extractor,
        frame_masks=frame_masks,
        batch_size=args.batch_size,
        expected_dim=args.expected_dim,
        layer=args.layer,
        process_res=args.process_res,
        process_res_method=args.process_res_method,
        normalize=not args.no_normalize,
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
        "feature_grid_shapes": [
            [int(shape[0]), int(shape[1])] if shape is not None else None
            for shape in feature_grid_shapes
        ],
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
        "process_res": int(args.process_res),
        "process_res_method": str(args.process_res_method),
        "normalize_features": bool(not args.no_normalize),
        "pooling": "mask_weighted_average_resized_to_da3_feature_grid",
        "feature_dim": int(feature_dim),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)

    unique_grids = sorted(
        {tuple(shape) for shape in feature_grid_shapes if shape is not None}
    )
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
        "feature_grid_shapes": [[int(h), int(w)] for h, w in unique_grids],
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
        description="Extract mask-pooled DA3 feature-map vectors for EgoCom person tracks."
    )
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--mapping_root", type=str, default=None)
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--scene_key", type=str, default=None)
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch_size", type=positive_int, default=8)
    parser.add_argument("--expected_dim", type=positive_int, default=None)
    parser.add_argument("--layer", type=nonnegative_int, default=DEFAULT_LAYER)
    parser.add_argument("--process_res", type=positive_int, default=DEFAULT_PROCESS_RES)
    parser.add_argument(
        "--process_res_method",
        type=str,
        default=DEFAULT_PROCESS_RES_METHOD,
        choices=[
            "upper_bound_resize",
            "upper_bound_crop",
            "lower_bound_resize",
            "lower_bound_crop",
        ],
    )
    parser.add_argument("--limit", type=nonnegative_int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--no_normalize",
        action="store_true",
        help="Save raw mask-pooled DA3 features instead of L2-normalized vectors.",
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
    print(f"Loading DA3 model: {args.model_id}")
    extractor = Da3FeatureExtractor(args.model_id, device)
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
                extractor=extractor,
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
            "process_res": int(args.process_res),
            "process_res_method": str(args.process_res_method),
            "normalize_features": bool(not args.no_normalize),
            "pooling": "mask_weighted_average_resized_to_da3_feature_grid",
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
