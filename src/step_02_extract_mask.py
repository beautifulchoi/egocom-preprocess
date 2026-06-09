"""
Extract SAM3 person masks for EgoCom holdout 1-minute chunks.

Default layout:
  /home/prj/data/egocom_holdout/1min/{split}/video/{video_name}.mp4
  /home/prj/data/egocom_holdout/1min/{split}/frame/{video_name}/frame00000.jpg
  /home/prj/data/egocom_holdout/1min/{split}/{video_name}/masks.pt
  /home/prj/data/egocom_holdout/1min/{split}/{video_name}/frame_overlay/frame00000.jpg

Examples:
  python src/step_02_extract_mask.py --splits train --limit 1
  python src/step_02_extract_mask.py --splits train,val --skip_existing --sam_gpus 0
  python src/step_02_extract_mask.py --splits train --num_shards 8 --shard_index 0
"""

import argparse
import gc
import inspect
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch


SAM3_DIR = Path("/home/prj/sam3")
if str(SAM3_DIR) not in sys.path:
    sys.path.append(str(SAM3_DIR))

DEFAULT_SAM3_CHECKPOINT = "/home/prj/sam3/checkpoints/sam3.pt"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi"}


@dataclass(frozen=True)
class VideoJob:
    split: str
    video_path: Optional[Path]
    frame_dir: Path
    output_dir: Path

    @property
    def video_name(self) -> str:
        if self.video_path is not None:
            return self.video_path.stem
        return self.frame_dir.name

    @property
    def rel_name(self) -> str:
        return f"{self.split}/{self.video_name}"


def parse_comma_list(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_gpu_list(raw: str) -> List[int]:
    raw = raw.strip()
    if not raw:
        return [0]
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def contiguous_shard(items: List[VideoJob], shard_index: int, num_shards: int) -> List[VideoJob]:
    if num_shards <= 0:
        raise ValueError(f"num_shards must be > 0, got {num_shards}")
    if not 0 <= shard_index < num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards}), got {shard_index}")

    total = len(items)
    base = total // num_shards
    remainder = total % num_shards
    start = shard_index * base + min(shard_index, remainder)
    stop = start + base + (1 if shard_index < remainder else 0)
    return items[start:stop]


def list_frame_paths(frame_dir: Path) -> List[Path]:
    if not frame_dir.is_dir():
        return []
    return sorted(
        path
        for path in frame_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def collect_video_jobs(args: argparse.Namespace) -> List[VideoJob]:
    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    video_name_filter = Path(args.video_name).stem if args.video_name else None
    jobs: List[VideoJob] = []

    for split in parse_comma_list(args.splits):
        split_dir = data_root / split
        video_dir = split_dir / "video"
        frame_root = split_dir / "frame"
        output_root = Path(args.output_root) if args.output_root else None
        if output_root is None:
            output_root = split_dir if not args.output_subdir else split_dir / args.output_subdir

        if args.frames_only:
            if not frame_root.is_dir():
                print(f"[SKIP] Missing frame directory: {frame_root}")
                continue
            frame_dirs = sorted(
                path
                for path in frame_root.iterdir()
                if path.is_dir() and list_frame_paths(path)
            )
            for frame_dir in frame_dirs:
                if video_name_filter and frame_dir.name != video_name_filter:
                    continue
                candidate_video = video_dir / f"{frame_dir.name}.mp4"
                jobs.append(
                    VideoJob(
                        split=split,
                        video_path=candidate_video if candidate_video.exists() else None,
                        frame_dir=frame_dir,
                        output_dir=output_root / frame_dir.name,
                    )
                )
            continue

        if not video_dir.is_dir():
            print(f"[SKIP] Missing video directory: {video_dir}")
            continue
        video_paths = sorted(
            path
            for path in video_dir.iterdir()
            if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
        )
        for video_path in video_paths:
            if video_name_filter and video_path.stem != video_name_filter:
                continue
            jobs.append(
                VideoJob(
                    split=split,
                    video_path=video_path,
                    frame_dir=frame_root / video_path.stem,
                    output_dir=output_root / video_path.stem,
                )
            )

    if video_name_filter and not jobs:
        raise FileNotFoundError(f"Video not found for --video_name: {args.video_name}")

    if args.limit is not None:
        jobs = jobs[: args.limit]

    if args.num_shards > 1:
        jobs = contiguous_shard(jobs, args.shard_index, args.num_shards)

    return jobs


def extract_all_video_frames(video_path: Path, output_dir: Path, jpg_quality: int) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_dir.mkdir(parents=True, exist_ok=True)
    write_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)]
    saved = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_path = output_dir / f"frame{saved:05d}.jpg"
        if not cv2.imwrite(str(frame_path), frame, write_params):
            cap.release()
            raise IOError(f"Failed to write frame: {frame_path}")
        saved += 1

    cap.release()

    if saved == 0:
        raise ValueError(f"No frames extracted from video: {video_path}")

    return {
        "video_path": str(video_path),
        "fps": fps,
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "saved_dir": str(output_dir),
        "num_extracted": saved,
        "frame_pattern": "frame%05d.jpg",
        "jpg_quality": jpg_quality,
    }


def ensure_frame_dir(job: VideoJob, jpg_quality: int) -> dict:
    frame_paths = list_frame_paths(job.frame_dir)
    if frame_paths:
        return {
            "created": False,
            "saved_dir": str(job.frame_dir),
            "num_extracted": len(frame_paths),
        }

    if job.video_path is None:
        raise FileNotFoundError(f"No frames found and no source video is available: {job.frame_dir}")

    print(f"  Extracting frames to {job.frame_dir}")
    metadata = extract_all_video_frames(
        video_path=job.video_path,
        output_dir=job.frame_dir,
        jpg_quality=jpg_quality,
    )
    metadata["created"] = True
    return metadata


def sam_outputs_to_mask_dict(outputs) -> dict:
    if outputs is None:
        return {}

    obj_ids = outputs.get("out_obj_ids", [])
    binary_masks = outputs.get("out_binary_masks", [])
    mask_dict = {}

    for idx, obj_id in enumerate(obj_ids):
        mask = binary_masks[idx]
        if isinstance(obj_id, torch.Tensor):
            obj_id = int(obj_id.detach().cpu().item())
        else:
            obj_id = int(obj_id)

        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu()
            if mask.ndim == 3 and mask.shape[0] == 1:
                mask = mask[0]
            mask_np = mask.numpy().astype(bool)
        else:
            mask_np = np.asarray(mask).astype(bool)
            if mask_np.ndim == 3 and mask_np.shape[0] == 1:
                mask_np = mask_np[0]

        if mask_np.any():
            mask_dict[obj_id] = mask_np

    return mask_dict


def merge_frame_masks(existing: dict, incoming: dict) -> dict:
    merged = dict(existing)
    for obj_id, mask in incoming.items():
        merged[obj_id] = mask
    return merged


def get_prompt_frame_indices(
    frame_count: int,
    prompt_frame_index: int,
    prompt_stride: int,
) -> List[int]:
    if frame_count <= 0:
        return []

    if prompt_stride <= 0:
        return [min(max(0, int(prompt_frame_index)), frame_count - 1)]

    frame_indices = list(range(0, frame_count, int(prompt_stride)))
    if (frame_count - 1) not in frame_indices:
        frame_indices.append(frame_count - 1)
    return frame_indices


def get_sam_masks(
    video_predictor,
    frame_dir: str,
    prompt: str,
    frame_count: int,
    prompt_frame_index: int,
    prompt_stride: int,
    propagate: bool,
) -> tuple[dict, dict]:
    for candidate in (
        "/home/prj/sam3",
        "/home/prj/old_files/play_sam3_audio/sam3",
    ):
        if candidate not in sys.path:
            sys.path.append(candidate)

    response = video_predictor.handle_request(
        request={
            "type": "start_session",
            "resource_path": frame_dir,
        }
    )
    session_id = response["session_id"]

    outputs_per_frame = {}
    prompt_indices = get_prompt_frame_indices(
        frame_count=frame_count,
        prompt_frame_index=prompt_frame_index,
        prompt_stride=prompt_stride,
    )
    prompt_frames_with_masks = 0
    propagated_frames_with_masks = 0

    try:
        for frame_index in prompt_indices:
            response = video_predictor.handle_request(
                request={
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": frame_index,
                    "text": prompt,
                }
            )
            # Official SAM3 video API returns the prompt-frame masks here.
            frame_masks = sam_outputs_to_mask_dict(response.get("outputs"))
            if frame_masks:
                outputs_per_frame[frame_index] = merge_frame_masks(
                    outputs_per_frame.get(frame_index, {}),
                    frame_masks,
                )
                prompt_frames_with_masks += 1

        if propagate:
            for response in video_predictor.handle_stream_request(
                request={
                    "type": "propagate_in_video",
                    "session_id": session_id,
                }
            ):
                frame_index = response["frame_index"]
                frame_masks = sam_outputs_to_mask_dict(response["outputs"])
                if frame_masks:
                    outputs_per_frame[frame_index] = merge_frame_masks(
                        outputs_per_frame.get(frame_index, {}),
                        frame_masks,
                    )
                    propagated_frames_with_masks += 1

        extraction_meta = {
            "prompt_frame_index": int(prompt_indices[0]) if prompt_indices else None,
            "prompt_stride": int(prompt_stride),
            "prompted_frame_count": len(prompt_indices),
            "prompt_frames_with_masks": prompt_frames_with_masks,
            "propagate": bool(propagate),
            "propagated_frames_with_masks": propagated_frames_with_masks,
        }
        return outputs_per_frame, extraction_meta
    finally:
        try:
            video_predictor.handle_request(
                request={
                    "type": "close_session",
                    "session_id": session_id,
                }
            )
        except Exception:
            pass
        clear_memory()


def select_mask_dict_persons(mask_dict, selection_mode: str, num_real_persons: int):
    person_frequency = Counter()
    for persons in mask_dict.values():
        person_frequency.update(persons.keys())

    sorted_persons = sorted(person_frequency.items(), key=lambda item: (-item[1], item[0]))
    if selection_mode == "all":
        kept_person_ids = sorted(person_frequency.keys())
    elif selection_mode == "topk":
        kept_person_ids = sorted(
            person_idx for person_idx, _ in sorted_persons[:num_real_persons]
        )
    else:
        raise ValueError(f"Unsupported selection_mode: {selection_mode}")

    kept_person_id_set = set(kept_person_ids)
    print("  Person detection frequency:")
    for person_idx, freq in sorted_persons:
        status = "KEEP" if person_idx in kept_person_id_set else "FILTER"
        print(f"    Person {person_idx}: {freq} frames - {status}")

    selected_mask_dict = {}
    for frame_idx, persons in mask_dict.items():
        selected_persons = {
            person_idx: mask
            for person_idx, mask in persons.items()
            if person_idx in kept_person_id_set
        }
        if selected_persons:
            selected_mask_dict[frame_idx] = selected_persons

    return selected_mask_dict, dict(person_frequency), kept_person_ids


def save_mask_visualizations(
    frame_dir: Path,
    mask_dict,
    output_dir: Path,
    alpha: float,
    sample_every: int,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = list_frame_paths(frame_dir)
    sample_every = max(1, int(sample_every))
    saved = 0

    palette = [
        (0, 0, 255),
        (0, 255, 0),
        (255, 0, 0),
        (0, 255, 255),
        (255, 0, 255),
        (255, 255, 0),
    ]

    for frame_idx, frame_path in enumerate(frame_paths):
        if frame_idx % sample_every != 0:
            continue

        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            continue

        overlay = frame.copy()
        for person_idx in sorted(mask_dict.get(frame_idx, {}).keys()):
            mask = mask_dict[frame_idx][person_idx]
            if not isinstance(mask, np.ndarray):
                continue

            mask_bool = mask.astype(bool)
            if not mask_bool.any():
                continue

            color = np.array(palette[int(person_idx) % len(palette)], dtype=np.uint8)
            overlay[mask_bool] = (
                overlay[mask_bool].astype(np.float32) * (1.0 - alpha)
                + color.astype(np.float32) * alpha
            ).astype(np.uint8)

            y_indices, x_indices = np.where(mask_bool)
            if len(y_indices) == 0 or len(x_indices) == 0:
                continue

            x_min, x_max = int(x_indices.min()), int(x_indices.max())
            y_min, y_max = int(y_indices.min()), int(y_indices.max())
            cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), color.tolist(), 2)
            cv2.putText(
                overlay,
                f"id {person_idx}",
                (x_min, max(20, y_min - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color.tolist(),
                2,
                cv2.LINE_AA,
            )

        overlay_path = output_dir / frame_path.name
        if cv2.imwrite(str(overlay_path), overlay):
            saved += 1

    return saved


def is_cuda_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    return "CUDA out of memory" in str(exc)


def clear_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass


def cleanup_predictor(predictor) -> None:
    if predictor is not None:
        try:
            predictor.shutdown()
        except Exception:
            pass
    clear_memory()


def load_predictor_for_profile(
    args: argparse.Namespace,
    sam_gpus: List[int],
    low_memory: bool,
):
    image_size = args.lowmem_image_size if low_memory else args.sam_image_size
    max_num_objects = (
        args.lowmem_max_num_objects if low_memory else args.sam_max_num_objects
    )
    offload_output = True if low_memory else args.sam_offload_output_to_cpu_for_eval
    print(
        f"Loading predictor profile={'lowmem' if low_memory else 'default'} "
        f"(image_size={image_size}, max_num_objects={max_num_objects}, "
        f"offload_output_to_cpu={offload_output})"
    )
    from sam3.model_builder import build_sam3_video_predictor

    candidate_kwargs = {
        "checkpoint_path": args.sam_checkpoint_path,
        "bpe_path": args.sam_bpe_path,
        "gpus_to_use": sam_gpus,
        "image_size": image_size,
        "max_num_objects": max_num_objects,
        "offload_output_to_cpu_for_eval": offload_output,
        "score_threshold_detection": args.score_threshold_detection,
        "det_nms_thresh": args.det_nms_thresh,
        "assoc_iou_thresh": args.assoc_iou_thresh,
        "trk_assoc_iou_thresh": args.trk_assoc_iou_thresh,
        "new_det_thresh": args.new_det_thresh,
        "compile": False,
    }
    signature = inspect.signature(build_sam3_video_predictor)
    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_var_kwargs:
        allowed_kwargs = candidate_kwargs
    else:
        allowed_kwargs = {
            key: value
            for key, value in candidate_kwargs.items()
            if key in signature.parameters
        }

    try:
        return build_sam3_video_predictor(**allowed_kwargs)
    except TypeError as exc:
        unsupported_kwargs = [
            "image_size",
            "max_num_objects",
            "offload_output_to_cpu_for_eval",
            "score_threshold_detection",
            "det_nms_thresh",
            "assoc_iou_thresh",
            "trk_assoc_iou_thresh",
            "new_det_thresh",
        ]
        if not any(key in str(exc) for key in unsupported_kwargs):
            raise
        fallback_kwargs = {
            key: value
            for key, value in candidate_kwargs.items()
            if key in {"checkpoint_path", "bpe_path", "gpus_to_use", "compile"}
        }
        print("  SAM3 API does not accept tuning kwargs; retrying with core args only")
        return build_sam3_video_predictor(**fallback_kwargs)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        json.dump(payload, file, indent=2)


def is_valid_existing_output(mask_path: Path, meta_path: Path, args: argparse.Namespace) -> bool:
    if not mask_path.exists() or not meta_path.exists():
        return False

    try:
        with meta_path.open() as file:
            meta = json.load(file)
    except (OSError, json.JSONDecodeError):
        return False

    sam_meta = meta.get("sam_extraction") or {}
    if bool(sam_meta.get("propagate")) != bool(args.propagate):
        return False
    if int(sam_meta.get("prompt_stride", -1)) != int(args.prompt_stride):
        return False
    if meta.get("prompt") != args.prompt:
        return False

    frame_count = meta.get("frame_count")
    frames_with_masks = meta.get("frames_with_masks")
    if not isinstance(frame_count, int) or frame_count <= 0:
        return False
    if not isinstance(frames_with_masks, int) or frames_with_masks <= 0:
        return False

    return True


def process_job(
    job: VideoJob,
    predictor,
    args: argparse.Namespace,
) -> dict:
    out_mask = job.output_dir / "masks.pt"
    out_meta = job.output_dir / "mask_meta.json"
    overlay_dir = job.output_dir / "frame_overlay"

    if args.skip_existing and out_mask.exists():
        if is_valid_existing_output(out_mask, out_meta, args):
            print("  [SKIP] valid masks.pt already exists")
            return {
                "status": "skipped",
                "reason": "existing_masks",
                "output_mask_path": str(out_mask),
            }
        print("  [REPROCESS] existing masks.pt does not match current tracker settings")

    frame_meta = ensure_frame_dir(job, jpg_quality=args.jpg_quality)
    frame_paths = list_frame_paths(job.frame_dir)
    if not frame_paths:
        raise ValueError(f"No frames found after preparation: {job.frame_dir}")

    sam_output, sam_extraction_meta = get_sam_masks(
        predictor,
        frame_dir=str(job.frame_dir),
        prompt=args.prompt,
        frame_count=len(frame_paths),
        prompt_frame_index=args.prompt_frame_index,
        prompt_stride=args.prompt_stride,
        propagate=args.propagate,
    )

    selected_masks, person_frequency, kept_person_ids = select_mask_dict_persons(
        sam_output,
        selection_mode=args.segment_selection_mode,
        num_real_persons=args.num_real_persons,
    )

    job.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(selected_masks, out_mask)

    overlays_saved = save_mask_visualizations(
        frame_dir=job.frame_dir,
        mask_dict=selected_masks,
        output_dir=overlay_dir,
        alpha=args.overlay_alpha,
        sample_every=args.overlay_sample_every,
    )

    meta = {
        "split": job.split,
        "video_path": str(job.video_path) if job.video_path is not None else None,
        "frame_dir": str(job.frame_dir),
        "frame_count": len(frame_paths),
        "frame_metadata": frame_meta,
        "prompt": args.prompt,
        "sam_extraction": sam_extraction_meta,
        "num_frames_with_output": len(selected_masks),
        "person_frequency": {str(key): int(value) for key, value in person_frequency.items()},
        "kept_person_ids": list(kept_person_ids),
        "num_segments": len(kept_person_ids),
        "segment_selection_mode": args.segment_selection_mode,
        "num_real_persons": (
            args.num_real_persons if args.segment_selection_mode == "topk" else None
        ),
        "mask_path": str(out_mask),
        "overlay_dir": str(overlay_dir),
        "overlay_sample_every": args.overlay_sample_every,
        "overlays_saved": overlays_saved,
    }
    write_json(out_meta, meta)

    del sam_output
    del selected_masks
    clear_memory()

    return {
        "status": "processed",
        "num_frames": len(frame_paths),
        "num_frames_with_output": meta["num_frames_with_output"],
        "num_segments": meta["num_segments"],
        "output_mask_path": str(out_mask),
        "overlay_dir": str(overlay_dir),
        "overlays_saved": overlays_saved,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract SAM3 person masks for EgoCom holdout 1-minute chunks"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/home/prj/data/egocom_holdout/1min",
        help="Root containing train/val/test split directories",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,val,test",
        help="Comma-separated split names to process",
    )
    parser.add_argument(
        "--video_name",
        type=str,
        default=None,
        help="Optional single video stem or filename to process",
    )
    parser.add_argument(
        "--frames_only",
        action="store_true",
        help="Collect jobs from existing split/frame folders instead of split/video files",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Optional explicit output root for one split, e.g. /path/to/train/person_mask",
    )
    parser.add_argument(
        "--output_subdir",
        type=str,
        default=None,
        help="Optional split-level output subdirectory, e.g. person_mask",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="person",
        help="SAM3 text prompt",
    )
    parser.add_argument(
        "--prompt_frame_index",
        type=int,
        default=0,
        help="Frame index used for the initial SAM3 text prompt",
    )
    parser.add_argument(
        "--prompt_stride",
        type=int,
        default=0,
        help="Optional extra text-prompt stride. 0 means prompt once and use video tracking.",
    )
    parser.add_argument(
        "--propagate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run SAM3 video propagation after the text prompt",
    )
    parser.add_argument(
        "--sam_gpus",
        type=str,
        default="0",
        help="Comma-separated SAM GPU ids",
    )
    parser.add_argument(
        "--sam_bpe_path",
        type=str,
        default=None,
        help="Optional SAM3 BPE vocab path",
    )
    parser.add_argument(
        "--sam_checkpoint_path",
        type=str,
        default=DEFAULT_SAM3_CHECKPOINT,
        help="Local SAM3 checkpoint path",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip videos that already have masks.pt",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only first N videos after filtering",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Number of contiguous shards",
    )
    parser.add_argument(
        "--shard_index",
        type=int,
        default=0,
        help="0-based shard index",
    )
    parser.add_argument(
        "--segment_selection_mode",
        type=str,
        choices=["all", "topk"],
        default="all",
        help="Keep all SAM tracks or only the most frequent top-k tracks",
    )
    parser.add_argument(
        "--num_real_persons",
        type=int,
        default=2,
        help="Number of tracks to keep when --segment_selection_mode topk is used",
    )
    parser.add_argument(
        "--overlay_sample_every",
        type=int,
        default=8,
        help="Save one overlay every N frames",
    )
    parser.add_argument(
        "--overlay_alpha",
        type=float,
        default=0.35,
        help="Mask overlay opacity",
    )
    parser.add_argument(
        "--jpg_quality",
        type=int,
        default=95,
        help="JPEG quality for extracted frames",
    )
    parser.add_argument(
        "--sam_image_size",
        type=int,
        default=1008,
        help="SAM image size for default profile",
    )
    parser.add_argument(
        "--sam_max_num_objects",
        type=int,
        default=-1,
        help="Maximum number of tracked objects for default profile",
    )
    parser.add_argument(
        "--sam_offload_output_to_cpu_for_eval",
        action="store_true",
        help="Offload tracker outputs/memory to CPU for default profile",
    )
    parser.add_argument(
        "--score_threshold_detection",
        type=float,
        default=0.5,
        help="Minimum detection confidence kept after sigmoid",
    )
    parser.add_argument(
        "--det_nms_thresh",
        type=float,
        default=0.1,
        help="Detection NMS IoU threshold",
    )
    parser.add_argument(
        "--assoc_iou_thresh",
        type=float,
        default=0.1,
        help="Detection-to-track association IoU threshold",
    )
    parser.add_argument(
        "--trk_assoc_iou_thresh",
        type=float,
        default=0.5,
        help="Track-to-detection matched IoU threshold",
    )
    parser.add_argument(
        "--new_det_thresh",
        type=float,
        default=0.7,
        help="Minimum confidence required before creating a new track",
    )
    parser.add_argument(
        "--retry_lowmem_on_oom",
        action="store_true",
        help="Retry once with low-memory SAM settings if CUDA OOM occurs",
    )
    parser.add_argument(
        "--lowmem_image_size",
        type=int,
        default=768,
        help="SAM image size for low-memory retry profile",
    )
    parser.add_argument(
        "--lowmem_max_num_objects",
        type=int,
        default=8,
        help="Maximum objects for low-memory retry profile",
    )
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Continue processing later videos after a per-video failure",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    jobs = collect_video_jobs(args)
    if not jobs:
        raise ValueError("No videos found for given filters")

    sam_gpus = parse_gpu_list(args.sam_gpus)
    print(f"Found {len(jobs)} video(s) to process")
    print(f"Loading SAM3 predictor on GPUs: {sam_gpus}")

    predictor_low_memory = False
    predictor = load_predictor_for_profile(args, sam_gpus, low_memory=False)

    summary = {
        "timestamp": time.time(),
        "data_root": args.data_root,
        "splits": parse_comma_list(args.splits),
        "prompt": args.prompt,
        "sam_gpus": sam_gpus,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "total_videos": len(jobs),
        "processed": 0,
        "skipped": 0,
        "failed": 0,
        "videos": {},
    }

    try:
        for idx, job in enumerate(jobs, 1):
            print(f"\n[{idx}/{len(jobs)}] {job.rel_name}")

            try:
                result = process_job(job, predictor, args)
            except Exception as exc:
                if is_cuda_oom(exc) and args.retry_lowmem_on_oom and not predictor_low_memory:
                    print(f"  [OOM] {exc}")
                    print("  Rebuilding SAM predictor with low-memory settings and retrying")
                    cleanup_predictor(predictor)
                    predictor = load_predictor_for_profile(args, sam_gpus, low_memory=True)
                    predictor_low_memory = True
                    result = process_job(job, predictor, args)
                elif args.continue_on_error:
                    print(f"  [FAIL] {exc}")
                    result = {
                        "status": "failed",
                        "error": str(exc),
                    }
                else:
                    raise

            status = result.get("status")
            if status == "processed":
                summary["processed"] += 1
            elif status == "skipped":
                summary["skipped"] += 1
            elif status == "failed":
                summary["failed"] += 1
            summary["videos"][job.rel_name] = result
    finally:
        cleanup_predictor(predictor)

    data_root = Path(args.data_root)
    if args.num_shards > 1:
        summary_path = (
            data_root
            / "_mask_shards"
            / f"shard_{args.shard_index:02d}_summary.json"
        )
    else:
        summary_path = data_root / "mask_extraction_summary.json"
    write_json(summary_path, summary)

    print("\n" + "=" * 60)
    print("Mask extraction completed")
    print(
        f"Processed: {summary['processed']}, "
        f"Skipped: {summary['skipped']}, Failed: {summary['failed']}"
    )
    print(f"Summary: {summary_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
