"""
Extract InternVL2 spatial text for mapped EgoCom person tracks.

For each mapped person track, this script sends a two-image prompt to InternVL2:
Image-1 is the full frame with only the mapped person mask visible, and Image-2
is the original frame. Frames with no usable mapped mask emit the literal text
"null" so the text sequence remains aligned to the source frame sequence.

Inputs:
  /home/prj/data/egocom_holdout/1min/{split}/person_face_mapping/*/remap_all_chunks.json
  /home/prj/data/egocom_holdout/1min/{split}/refined_mask/{video}/mask.pt
  /home/prj/data/egocom_holdout/1min/{split}/frame/{video}/*.jpg

Outputs:
  /home/prj/data/egocom_holdout/1min/{split}/person_spatial_internvl2_text/{scene}/person_{id}/{video}.txt
  /home/prj/data/egocom_holdout/1min/{split}/person_spatial_internvl2_text/{scene}/person_{id}/{video}.json
  /home/prj/data/egocom_holdout/1min/{split}/person_spatial_internvl2_text/{scene}/summary.json
  /home/prj/data/egocom_holdout/1min/{split}/person_spatial_internvl2_text/summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm

from extract_person_visual_clip import (
    Assignment,
    CLIP_RE,
    black_image,
    collect_assignments_for_split,
    default_mapping_root,
    list_frame_files,
    load_json,
    load_mask_dict,
    mask_bbox,
    nonnegative_int,
    positive_int,
    resize_mask,
    split_names,
    union_segment_mask,
    write_json,
)


DEFAULT_DATA_ROOT = "/home/prj/data/egocom_holdout/1min"
DEFAULT_MODEL_ID = "OpenGVLab/InternVL2-8B"
DEFAULT_PROMPT = (
    "Image-1: <image>\n"
    "Image-2: <image>\n"
    "Image-1 shows only the target person. Image-2 is the original frame. "
    "Locate the same target person in Image-2.\n"
    "Write one concise natural sentence describing the target person's spatial "
    "location in Image-2, including left/right and top/bottom position, "
    "approximate distance or size, facing direction, and occlusion if visible.\n"
    "Start directly with 'The target person'. "
    "Do not explain your reasoning. Do not use markdown, bullet lists, labels, "
    "or key-value formatting. Do not start with phrases like 'In Image-2', "
    "'To find', or 'The person in Image-1'."
)
DEFAULT_COMBINED_PROMPT = (
    "<image>\n"
    "The left half is Image-1, a masked reference frame showing only the target "
    "person. The right half is Image-2, the original full frame. Locate the "
    "same target person in the right half.\n"
    "Write one concise natural sentence describing the target person's spatial "
    "location within the original frame, including left/right and top/bottom "
    "position, approximate distance or size, facing direction, and occlusion if "
    "visible.\n"
    "Start directly with 'The target person'. "
    "Do not explain your reasoning. Do not use markdown, bullet lists, labels, "
    "or key-value formatting. Do not mention image halves, right half, left half, "
    "Image-1, or Image-2. "
    "Do not start with phrases like 'In Image-2', 'To find', or "
    "'The person in Image-1'."
)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class TextFrame:
    frame_idx: int
    frame_stem: str
    frame_path: Path
    masked_image: Image.Image | None
    original_image: Image.Image | None
    mask_bbox: tuple[int, int, int, int] | None
    mask_pixel_count: int
    has_mask: bool
    status: str


def clean_line(text: str) -> str:
    line = " ".join(str(text).strip().split())
    replacements = {
        "located in the right half of the original frame, ": "located ",
        "located in the right half of the frame, ": "located ",
        "located in the right half, ": "located ",
        "positioned in the right half of the original frame, ": "positioned ",
        "positioned in the right half of the frame, ": "positioned ",
        "positioned in the right half, ": "positioned ",
        "in the right half of the original frame, ": "",
        "in the right half of the frame, ": "",
        "in the right half, ": "",
    }
    for old, new in replacements.items():
        line = line.replace(old, new)
    return line or "null"


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU")
        return "cpu"
    return device_arg


def build_transform(input_size: int) -> T.Compose:
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: list[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            target_area = image_size * image_size * ratio[0] * ratio[1]
            if area > 0.5 * target_area:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = False,
) -> list[Image.Image]:
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = [
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    ]
    target_ratios = sorted(set(target_ratios), key=lambda item: item[0] * item[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))

    processed_images = []
    cols = target_width // image_size
    for block_idx in range(blocks):
        box = (
            (block_idx % cols) * image_size,
            (block_idx // cols) * image_size,
            ((block_idx % cols) + 1) * image_size,
            ((block_idx // cols) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))

    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def image_pair_to_pixel_values(
    masked_image: Image.Image,
    original_image: Image.Image,
    input_size: int,
    max_tiles: int,
) -> tuple[torch.Tensor, list[int]]:
    transform = build_transform(input_size=input_size)
    pixel_values_list = []
    num_patches_list = []
    for image in (masked_image, original_image):
        tiles = dynamic_preprocess(
            image,
            image_size=input_size,
            use_thumbnail=True,
            max_num=max_tiles,
        )
        pixel_values = torch.stack([transform(tile) for tile in tiles])
        num_patches_list.append(int(pixel_values.shape[0]))
        pixel_values_list.append(pixel_values)
    return torch.cat(pixel_values_list), num_patches_list


def image_to_pixel_values(
    image: Image.Image,
    input_size: int,
    max_tiles: int,
) -> tuple[torch.Tensor, int]:
    transform = build_transform(input_size=input_size)
    tiles = dynamic_preprocess(
        image,
        image_size=input_size,
        use_thumbnail=True,
        max_num=max_tiles,
    )
    pixel_values = torch.stack([transform(tile) for tile in tiles])
    return pixel_values, int(pixel_values.shape[0])


def combine_reference_query(
    masked_image: Image.Image,
    original_image: Image.Image,
) -> Image.Image:
    masked = masked_image.convert("RGB")
    original = original_image.convert("RGB")
    if original.size != masked.size:
        original = original.resize(masked.size, Image.BICUBIC)
    width, height = masked.size
    combined = Image.new("RGB", (width * 2, height))
    combined.paste(masked, (0, 0))
    combined.paste(original, (width, 0))
    return combined


def load_model(model_id: str, device: str, local_files_only: bool):
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: install transformers>=4.37.2 and retry."
        ) from exc

    dtype = torch.bfloat16 if torch.cuda.is_available() and device != "cpu" else torch.float32
    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        use_flash_attn=False,
        local_files_only=local_files_only,
    ).eval()
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        use_fast=False,
        local_files_only=local_files_only,
    )
    return model, tokenizer, dtype


def read_original_image(frame_path: Path, image_shape: tuple[int, int]) -> tuple[Image.Image, np.ndarray | None, str]:
    frame_bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if frame_bgr is None:
        return black_image(image_shape), None, "unreadable_frame"
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb).convert("RGB"), frame_rgb, "ok"


def infer_image_shape(frame_paths: list[Path], mask_dict: dict[int, dict[int, np.ndarray]]) -> tuple[int, int]:
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


def collect_text_frames(
    assignment: Assignment,
    split_root: Path,
    min_mask_pixels: int,
    max_frames: int | None,
) -> tuple[list[TextFrame], dict[str, Any]]:
    frame_dir = split_root / "frame" / assignment.video_name
    mask_path = split_root / "refined_mask" / assignment.video_name / "mask.pt"
    frame_paths = list_frame_files(frame_dir)
    if not frame_paths:
        raise FileNotFoundError(f"No frames found: {frame_dir}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Missing mask: {mask_path}")

    mask_dict = load_mask_dict(mask_path)
    image_shape = infer_image_shape(frame_paths, mask_dict)
    rows = []
    status_counts: Counter[str] = Counter()

    if max_frames is not None:
        frame_paths = frame_paths[:max_frames]

    for frame_idx, frame_path in enumerate(frame_paths):
        original_image, frame_rgb, original_status = read_original_image(frame_path, image_shape)
        if frame_rgb is None:
            status_counts[original_status] += 1
            rows.append(
                TextFrame(
                    frame_idx=int(frame_idx),
                    frame_stem=frame_path.stem,
                    frame_path=frame_path,
                    masked_image=None,
                    original_image=original_image,
                    mask_bbox=None,
                    mask_pixel_count=0,
                    has_mask=False,
                    status=original_status,
                )
            )
            continue

        mask = union_segment_mask(mask_dict.get(frame_idx), assignment.segment_ids)
        if mask is None:
            status_counts["absent_segment"] += 1
            rows.append(
                TextFrame(
                    frame_idx=int(frame_idx),
                    frame_stem=frame_path.stem,
                    frame_path=frame_path,
                    masked_image=None,
                    original_image=original_image,
                    mask_bbox=None,
                    mask_pixel_count=0,
                    has_mask=False,
                    status="absent_segment",
                )
            )
            continue

        mask_bool = resize_mask(mask, frame_rgb.shape[:2])
        bbox = mask_bbox(mask_bool)
        pixel_count = int(mask_bool.sum())
        if bbox is None:
            status = "empty_mask_after_resize"
        elif pixel_count < min_mask_pixels:
            status = "low_mask_pixels"
        else:
            status = "masked"

        status_counts[status] += 1
        masked_image = None
        has_mask = status == "masked"
        if has_mask:
            masked_rgb = np.zeros_like(frame_rgb)
            masked_rgb[mask_bool] = frame_rgb[mask_bool]
            masked_image = Image.fromarray(masked_rgb).convert("RGB")

        rows.append(
            TextFrame(
                frame_idx=int(frame_idx),
                frame_stem=frame_path.stem,
                frame_path=frame_path,
                masked_image=masked_image,
                original_image=original_image,
                mask_bbox=bbox,
                mask_pixel_count=pixel_count,
                has_mask=has_mask,
                status=status,
            )
        )

    diagnostics = {
        "num_frame_files": len(frame_paths),
        "num_mask_frames": len(mask_dict),
        "num_masked_frames": int(status_counts.get("masked", 0)),
        "num_null_frames": int(len(frame_paths) - status_counts.get("masked", 0)),
        "frame_status_counts": dict(sorted(status_counts.items())),
        "min_mask_pixels": int(min_mask_pixels),
    }
    return rows, diagnostics


def select_visualization_indices(rows: list[TextFrame], num_samples: int) -> set[int]:
    valid_indices = [idx for idx, row in enumerate(rows) if row.has_mask]
    if num_samples <= 0 or not valid_indices:
        return set()
    if len(valid_indices) <= num_samples:
        return set(valid_indices)
    selected_positions = np.linspace(0, len(valid_indices) - 1, num=num_samples)
    return {valid_indices[int(round(value))] for value in selected_positions}


def save_visualizations(
    rows: list[TextFrame],
    assignment: Assignment,
    output_root: Path,
    num_samples: int,
) -> list[str]:
    selected = select_visualization_indices(rows, num_samples)
    if not selected:
        return []

    vis_dir = output_root / assignment.scene_key / f"person_{assignment.person_id}" / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in vis_dir.glob(f"{assignment.video_name}__*.jpg"):
        stale_path.unlink()

    paths = []
    for row_index in selected:
        row = rows[row_index]
        if row.masked_image is None:
            continue
        vis_path = vis_dir / f"{assignment.video_name}__{row.frame_stem}.jpg"
        row.masked_image.save(vis_path, quality=92)
        paths.append(str(vis_path))
    return paths


@torch.inference_mode()
def describe_pair(
    model,
    tokenizer,
    row: TextFrame,
    prompt: str,
    input_size: int,
    max_tiles: int,
    max_new_tokens: int,
    device: str,
    dtype: torch.dtype,
) -> str:
    if row.masked_image is None or row.original_image is None:
        return "null"

    pixel_values, num_patches_list = image_pair_to_pixel_values(
        row.masked_image,
        row.original_image,
        input_size=input_size,
        max_tiles=max_tiles,
    )
    pixel_values = pixel_values.to(dtype=dtype, device=device)
    generation_config = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": False,
    }
    response = model.chat(
        tokenizer,
        pixel_values,
        prompt,
        generation_config,
        num_patches_list=num_patches_list,
    )
    return clean_line(response)


@torch.inference_mode()
def describe_combined_batch(
    model,
    tokenizer,
    rows: list[TextFrame],
    prompt: str,
    input_size: int,
    max_tiles: int,
    max_new_tokens: int,
    device: str,
    dtype: torch.dtype,
    batch_size: int,
) -> list[str]:
    texts = ["null"] * len(rows)
    valid_indices = [
        idx
        for idx, row in enumerate(rows)
        if row.has_mask and row.masked_image is not None and row.original_image is not None
    ]
    generation_config = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": False,
    }

    for start in range(0, len(valid_indices), batch_size):
        batch_indices = valid_indices[start : start + batch_size]
        pixel_values_list = []
        num_patches_list = []
        for row_index in batch_indices:
            row = rows[row_index]
            combined = combine_reference_query(row.masked_image, row.original_image)
            pixel_values, patch_count = image_to_pixel_values(
                combined,
                input_size=input_size,
                max_tiles=max_tiles,
            )
            pixel_values_list.append(pixel_values)
            num_patches_list.append(patch_count)

        pixel_values = torch.cat(pixel_values_list).to(dtype=dtype, device=device)
        responses = model.batch_chat(
            tokenizer,
            pixel_values,
            questions=[prompt] * len(batch_indices),
            generation_config=dict(generation_config),
            num_patches_list=num_patches_list,
        )
        for row_index, response in zip(batch_indices, responses):
            texts[row_index] = clean_line(response)
    return texts


def text_output_path(output_root: Path, assignment: Assignment) -> Path:
    return output_root / assignment.scene_key / f"person_{assignment.person_id}" / f"{assignment.video_name}.txt"


def chunk_number_from_video(video_name: str) -> int | None:
    match = CLIP_RE.match(video_name)
    if match is None:
        return None
    return int(match.group("chunk"))


def conflicted_scene_chunks(mapping_root: Path) -> dict[str, set[int]]:
    if not mapping_root.is_dir():
        return {}
    out: dict[str, set[int]] = {}
    for summary_path in sorted(mapping_root.glob("*/summary.json")):
        try:
            data = load_json(summary_path)
        except Exception:
            continue
        scene_key = summary_path.parent.name
        chunks: set[int] = set()
        for attempt in data.get("chunk_attempts", []):
            if int(attempt.get("conflict_segments", 0)) > 0:
                chunk = attempt.get("chunk")
                if chunk is not None:
                    chunks.add(int(chunk))
        if int(data.get("conflict_segments", 0)) > 0:
            chunk = data.get("selected_chunk", data.get("chunk"))
            if chunk is not None:
                chunks.add(int(chunk))
        if data.get("fallback_status") == "all_chunks_conflicted":
            for attempt in data.get("chunk_attempts", []):
                chunk = attempt.get("chunk")
                if chunk is not None:
                    chunks.add(int(chunk))
        if chunks:
            out[scene_key] = chunks
    return out


def filter_conflicted_assignments(
    assignments: list[Assignment],
    mapping_root: Path,
) -> tuple[list[Assignment], list[dict[str, Any]]]:
    conflicted = conflicted_scene_chunks(mapping_root)
    if not conflicted:
        return assignments, []
    kept = []
    excluded = []
    for assignment in assignments:
        chunk = chunk_number_from_video(assignment.video_name)
        scene_chunks = conflicted.get(assignment.scene_key, set())
        if chunk is not None and chunk in scene_chunks:
            excluded.append(
                {
                    "split": assignment.split,
                    "scene_key": assignment.scene_key,
                    "video_name": assignment.video_name,
                    "person_id": int(assignment.person_id),
                    "chunk": int(chunk),
                    "reason": "conflicted_mapping_chunk_attempt",
                }
            )
            continue
        kept.append(assignment)
    return kept, excluded


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def metadata_payload(
    assignment: Assignment,
    rows: list[TextFrame],
    texts: list[str],
    diagnostics: dict[str, Any],
    output_path: Path,
    visualization_paths: list[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    split_root = Path(args.data_root) / assignment.split
    return {
        "split": assignment.split,
        "scene_key": assignment.scene_key,
        "video_name": assignment.video_name,
        "camera_person": int(assignment.camera_person),
        "person_id": int(assignment.person_id),
        "segment_ids": [int(value) for value in assignment.segment_ids],
        "model_id": args.model_id,
        "prompt": args.prompt,
        "combined_prompt": args.combined_prompt,
        "pair_mode": args.pair_mode,
        "min_mask_pixels": int(args.min_mask_pixels),
        "num_lines": len(texts),
        "num_null_lines": int(sum(1 for text in texts if text == "null")),
        "text_path": str(output_path),
        "visualization_paths": visualization_paths,
        "source_mask_path": str(split_root / "refined_mask" / assignment.video_name / "mask.pt"),
        "source_mapping_path": str(assignment.mapping_path),
        "frame_indices": [int(row.frame_idx) for row in rows],
        "frame_stems": [row.frame_stem for row in rows],
        "source_frame_paths": [str(row.frame_path) for row in rows],
        "has_masks": [bool(row.has_mask) for row in rows],
        "frame_statuses": [row.status for row in rows],
        "mask_bboxes": [
            None if row.mask_bbox is None else [int(value) for value in row.mask_bbox]
            for row in rows
        ],
        "mask_pixel_counts": [int(row.mask_pixel_count) for row in rows],
        "diagnostics": diagnostics,
    }


def process_assignment(
    assignment: Assignment,
    data_root: Path,
    output_root: Path,
    model,
    tokenizer,
    device: str,
    dtype: torch.dtype,
    args: argparse.Namespace,
) -> dict[str, Any]:
    out_path = text_output_path(output_root, assignment)
    meta_path = out_path.with_suffix(".json")
    if out_path.exists() and meta_path.exists() and not args.overwrite:
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
    rows, diagnostics = collect_text_frames(
        assignment=assignment,
        split_root=split_root,
        min_mask_pixels=args.min_mask_pixels,
        max_frames=args.max_frames,
    )
    visualization_paths = save_visualizations(
        rows=rows,
        assignment=assignment,
        output_root=output_root,
        num_samples=args.num_vis_samples,
    )

    if args.dry_run:
        texts = [
            "null" if not row.has_mask else f"DRY_RUN {row.frame_stem}"
            for row in rows
        ]
    elif args.pair_mode == "combined":
        texts = describe_combined_batch(
            model=model,
            tokenizer=tokenizer,
            rows=rows,
            prompt=args.combined_prompt,
            input_size=args.input_size,
            max_tiles=args.max_tiles,
            max_new_tokens=args.max_new_tokens,
            device=device,
            dtype=dtype,
            batch_size=args.batch_size,
        )
    else:
        texts = []
        for row in rows:
            if not row.has_mask:
                texts.append("null")
                continue
            texts.append(
                describe_pair(
                    model=model,
                    tokenizer=tokenizer,
                    row=row,
                    prompt=args.prompt,
                    input_size=args.input_size,
                    max_tiles=args.max_tiles,
                    max_new_tokens=args.max_new_tokens,
                    device=device,
                    dtype=dtype,
                )
            )

    write_lines(out_path, texts)
    payload = metadata_payload(
        assignment=assignment,
        rows=rows,
        texts=texts,
        diagnostics=diagnostics,
        output_path=out_path,
        visualization_paths=visualization_paths,
        args=args,
    )
    write_json(meta_path, payload)

    return {
        "status": "processed",
        "split": assignment.split,
        "scene_key": assignment.scene_key,
        "video_name": assignment.video_name,
        "camera_person": int(assignment.camera_person),
        "person_id": int(assignment.person_id),
        "segment_ids": [int(value) for value in assignment.segment_ids],
        "num_lines": int(len(texts)),
        "num_null_lines": int(sum(1 for text in texts if text == "null")),
        "output_path": str(out_path),
        "metadata_path": str(meta_path),
        "visualization_paths": visualization_paths,
        "diagnostics": diagnostics,
    }


def write_summaries(output_roots: dict[str, Path], results: list[dict[str, Any]], args: argparse.Namespace) -> None:
    by_scene: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_scene[(str(result["split"]), str(result["scene_key"]))].append(result)

    for (split, scene_key), scene_results in sorted(by_scene.items()):
        status_counts = Counter(str(item["status"]) for item in scene_results)
        summary = {
            "split": split,
            "scene_key": scene_key,
            "model_id": args.model_id,
            "pair_mode": args.pair_mode,
            "min_mask_pixels": int(args.min_mask_pixels),
            "status_counts": dict(sorted(status_counts.items())),
            "num_tracks": len(scene_results),
            "num_processed_tracks": int(status_counts.get("processed", 0)),
            "num_lines": int(sum(int(item.get("num_lines", 0)) for item in scene_results)),
            "num_null_lines": int(sum(int(item.get("num_null_lines", 0)) for item in scene_results)),
            "tracks": scene_results,
        }
        write_json(output_roots[split] / scene_key / "summary.json", summary)

    for split, output_root in sorted(output_roots.items()):
        split_results = [item for item in results if item["split"] == split]
        status_counts = Counter(str(item["status"]) for item in split_results)
        summary = {
            "split": split,
            "model_id": args.model_id,
            "prompt": args.prompt,
            "combined_prompt": args.combined_prompt,
            "pair_mode": args.pair_mode,
            "min_mask_pixels": int(args.min_mask_pixels),
            "status_counts": dict(sorted(status_counts.items())),
            "num_tracks": len(split_results),
            "num_processed_tracks": int(status_counts.get("processed", 0)),
            "num_lines": int(sum(int(item.get("num_lines", 0)) for item in split_results)),
            "num_null_lines": int(sum(int(item.get("num_null_lines", 0)) for item in split_results)),
            "tracks": split_results,
        }
        write_json(output_root / "summary.json", summary)


def default_output_root(data_root: Path, split: str) -> Path:
    return data_root / split / "person_spatial_internvl2_text"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract InternVL2 spatial text from mapped EgoCom person tracks."
    )
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--mapping_root", type=str, default=None)
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--scene_key", type=str, default=None)
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--pair_mode", choices=("two_image", "combined"), default="two_image")
    parser.add_argument("--batch_size", type=positive_int, default=8)
    parser.add_argument("--input_size", type=positive_int, default=448)
    parser.add_argument("--max_tiles", type=positive_int, default=1)
    parser.add_argument("--max_new_tokens", type=positive_int, default=80)
    parser.add_argument("--min_mask_pixels", type=nonnegative_int, default=100)
    parser.add_argument("--max_frames", type=nonnegative_int, default=None)
    parser.add_argument("--num_vis_samples", type=nonnegative_int, default=8)
    parser.add_argument("--limit", type=nonnegative_int, default=None)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--combined_prompt", type=str, default=DEFAULT_COMBINED_PROMPT)
    parser.add_argument("--exclude_conflicted_videos", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
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
        if args.exclude_conflicted_videos:
            split_assignments, excluded = filter_conflicted_assignments(
                split_assignments,
                mapping_root,
            )
            if excluded:
                excluded_path = output_roots[split] / "excluded_conflicted_videos.json"
                write_json(
                    excluded_path,
                    {
                        "split": split,
                        "num_excluded_tracks": len(excluded),
                        "excluded_tracks": excluded,
                    },
                )
                print(
                    f"Excluded conflicted tracks for {split}: "
                    f"{len(excluded)} -> {excluded_path}"
                )
        assignments.extend(split_assignments)

    if args.limit is not None:
        assignments = assignments[: args.limit]
    if not assignments:
        print("No matching mapped person tracks found.")
        return 2

    device = resolve_device(args.device)
    model = tokenizer = dtype = None
    if not args.dry_run:
        print(f"Using device: {device}")
        print(f"Loading InternVL2 model: {args.model_id}")
        model, tokenizer, dtype = load_model(
            args.model_id,
            device=device,
            local_files_only=args.local_files_only,
        )
    else:
        dtype = torch.float32

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
                tokenizer=tokenizer,
                device=device,
                dtype=dtype,
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

    write_summaries(output_roots, results, args)
    print(f"Done. processed={processed} skipped={skipped} failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
