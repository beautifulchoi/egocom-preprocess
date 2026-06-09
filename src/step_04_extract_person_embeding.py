"""
Extract masked person face embeddings for EgoCom refined masks.

Inputs:
  /home/prj/data/egocom_holdout/1min/{split}/refined_mask/{video_name}/mask.pt
  /home/prj/data/egocom_holdout/1min/{split}/frame/{video_name}/*.jpg

Outputs:
  /home/prj/data/egocom_holdout/1min/{split}/person_embeding/{video_name}/embeding.pt
  /home/prj/data/egocom_holdout/1min/{split}/person_embeding/{video_name}/summary.json

Segments with no valid face detection are omitted from embeding.pt and recorded as invalid
in summary.json.

Examples:
  python src/step_04_extract_person_embeding.py --split train --video vid_001__day_1__con_1__person_1_part1_chunk_0001
  python src/step_04_extract_person_embeding.py --split train --sample_every 5 --overwrite
  python src/step_04_extract_person_embeding.py --mask_path /path/to/mask.pt --frame_dir /path/to/frames
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import CLIPImageProcessor, CLIPModel


DEFAULT_DATA_ROOT = "/home/prj/data/egocom_holdout/1min"
DEFAULT_STYLEID_MODEL_ID = "kwanY/styleid"
DEFAULT_INSIGHTFACE_MODEL = "antelopev2"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class VideoJob:
    split: str
    video_name: str
    mask_path: Path
    frame_dir: Path
    output_dir: Path


@dataclass(frozen=True)
class FaceCrop:
    segment_id: int
    frame_idx: int
    frame_stem: str
    image: Image.Image
    det_score: float
    face_bbox: list[int]
    person_bbox: list[int]


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


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return parsed


def list_frame_paths(frame_dir: Path) -> list[Path]:
    if not frame_dir.is_dir():
        return []
    return sorted(
        path
        for path in frame_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


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


def load_mask_dict(mask_path: Path) -> dict[int, dict[int, np.ndarray]]:
    with torch.serialization.safe_globals(get_safe_numpy_globals()):
        raw = torch.load(mask_path, map_location="cpu", weights_only=True)

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


def expand_bbox(
    bbox: tuple[int, int, int, int] | list[int],
    image_shape: tuple[int, int],
    padding: float,
) -> tuple[int, int, int, int]:
    height, width = image_shape
    x1, y1, x2, y2 = [int(v) for v in bbox]
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    pad = int(round(max(box_w, box_h) * padding))
    return (
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(width, x2 + pad),
        min(height, y2 + pad),
    )


def crop_array(
    array: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    return array[y1:y2, x1:x2]


def clip_face_bbox(
    bbox: np.ndarray | list[float],
    image_shape: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    height, width = image_shape
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox[:4]]
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def choose_face(
    faces: list[Any],
    mask_crop: np.ndarray,
    min_face_score: float,
) -> Any | None:
    best_face = None
    best_key: tuple[float, float, float] | None = None
    for face in faces:
        score = float(getattr(face, "det_score", 0.0))
        if score < min_face_score:
            continue
        bbox = clip_face_bbox(face.bbox, mask_crop.shape[:2])
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        face_area = float(max(1, (x2 - x1) * (y2 - y1)))
        overlap_pixels = float(mask_crop[y1:y2, x1:x2].sum())
        overlap_ratio = overlap_pixels / face_area
        key = (overlap_ratio, score, face_area)
        if best_key is None or key > best_key:
            best_key = key
            best_face = face
    return best_face


def segment_ids(mask_dict: dict[int, dict[int, np.ndarray]]) -> list[int]:
    ids = set()
    for persons in mask_dict.values():
        ids.update(int(segment_id) for segment_id in persons)
    return sorted(ids)


def infer_split_from_mask_path(mask_path: Path) -> str:
    parts = mask_path.parts
    if "refined_mask" in parts:
        refined_index = parts.index("refined_mask")
        if refined_index > 0:
            return parts[refined_index - 1]
    return "custom"


def infer_video_name(mask_path: Path) -> str:
    if mask_path.name == "mask.pt":
        return mask_path.parent.name
    return mask_path.stem


def discover_splits(data_root: Path, split_arg: str) -> list[str]:
    if split_arg != "all_existing":
        return parse_comma_list(split_arg)
    splits = []
    for split_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        if (split_dir / "refined_mask").is_dir() and (split_dir / "frame").is_dir():
            splits.append(split_dir.name)
    return splits


def collect_jobs(args: argparse.Namespace) -> list[VideoJob]:
    data_root = Path(args.data_root)
    if args.mask_path:
        mask_path = Path(args.mask_path)
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask path not found: {mask_path}")
        split = args.split if args.split != "all_existing" else infer_split_from_mask_path(mask_path)
        video_name = args.video or infer_video_name(mask_path)
        if args.frame_dir:
            frame_dir = Path(args.frame_dir)
        elif split != "custom":
            frame_dir = data_root / split / "frame" / video_name
        else:
            raise ValueError("--frame_dir is required when --mask_path is outside the default layout")

        if args.output_dir:
            output_dir = Path(args.output_dir)
        elif args.output_root:
            output_dir = Path(args.output_root) / video_name
        elif split != "custom":
            output_dir = data_root / split / "person_embeding" / video_name
        else:
            output_dir = mask_path.parent / "person_embeding"

        return [
            VideoJob(
                split=split,
                video_name=video_name,
                mask_path=mask_path,
                frame_dir=frame_dir,
                output_dir=output_dir,
            )
        ]

    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    jobs: list[VideoJob] = []
    video_filter = Path(args.video).stem if args.video else None
    for split in discover_splits(data_root, args.split):
        split_dir = data_root / split
        mask_root = split_dir / "refined_mask"
        frame_root = split_dir / "frame"
        if not mask_root.is_dir() or not frame_root.is_dir():
            print(f"[SKIP] {split}: missing refined_mask or frame directory")
            continue

        for mask_dir in sorted(path for path in mask_root.iterdir() if path.is_dir()):
            video_name = mask_dir.name
            if video_filter and video_name != video_filter:
                continue
            mask_path = mask_dir / "mask.pt"
            frame_dir = frame_root / video_name
            if not mask_path.exists():
                print(f"[SKIP] {split}/{video_name}: missing mask.pt")
                continue
            if not frame_dir.is_dir():
                print(f"[SKIP] {split}/{video_name}: missing frame directory")
                continue
            output_root = Path(args.output_root) if args.output_root else split_dir / "person_embeding"
            jobs.append(
                VideoJob(
                    split=split,
                    video_name=video_name,
                    mask_path=mask_path,
                    frame_dir=frame_dir,
                    output_dir=Path(args.output_dir) if args.output_dir else output_root / video_name,
                )
            )

    if args.start_index or args.stride != 1:
        jobs = [
            job
            for index, job in enumerate(jobs)
            if index >= args.start_index and (index - args.start_index) % args.stride == 0
        ]

    if args.limit is not None:
        jobs = jobs[: args.limit]
    return jobs


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU")
        return torch.device("cpu")
    return torch.device(device_arg)


def cuda_index(device: torch.device) -> int:
    if device.type != "cuda":
        return -1
    if device.index is None:
        return 0
    return int(device.index)


def repair_nested_insightface_pack(model_name: str) -> bool:
    model_dir = Path.home() / ".insightface" / "models" / model_name
    nested_dir = model_dir / model_name
    if not nested_dir.is_dir():
        return False

    copied = False
    for src_path in nested_dir.glob("*.onnx"):
        dst_path = model_dir / src_path.name
        if not dst_path.exists():
            shutil.copy2(src_path, dst_path)
            copied = True
    return copied


def load_face_detector(args: argparse.Namespace, device: torch.device):
    try:
        import onnxruntime as ort
        from insightface.app import FaceAnalysis
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Failed to import insightface. Install insightface and onnxruntime first."
        ) from exc

    providers = ["CPUExecutionProvider"]
    provider_options = [{}]
    if device.type == "cuda":
        available_providers = ort.get_available_providers()
        if "CUDAExecutionProvider" not in available_providers:
            raise RuntimeError(
                "CUDA was selected, but ONNXRuntime does not expose CUDAExecutionProvider. "
                "Install a compatible onnxruntime-gpu package or run with --device cpu. "
                f"Available providers: {available_providers}"
            )
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        provider_options = [{"device_id": cuda_index(device)}, {}]

    try:
        app = FaceAnalysis(
            name=args.insightface_model,
            allowed_modules=["detection"],
            providers=providers,
            provider_options=provider_options,
        )
    except AssertionError:
        if not repair_nested_insightface_pack(args.insightface_model):
            raise
        app = FaceAnalysis(
            name=args.insightface_model,
            allowed_modules=["detection"],
            providers=providers,
            provider_options=provider_options,
        )
    app.prepare(ctx_id=cuda_index(device), det_size=(args.det_size, args.det_size))
    return app


def load_styleid_model(model_id: str, device: torch.device) -> tuple[CLIPModel, CLIPImageProcessor, int]:
    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPImageProcessor.from_pretrained(model_id)
    model.eval()
    embedding_dim = int(getattr(model.config, "projection_dim", 0) or model.visual_projection.out_features)
    return model, processor, embedding_dim


def extract_face_crop(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    segment_id: int,
    frame_idx: int,
    frame_stem: str,
    face_detector: Any,
    args: argparse.Namespace,
) -> FaceCrop | None:
    frame_h, frame_w = frame_bgr.shape[:2]
    mask_bool = resize_mask(mask, (frame_h, frame_w))
    if int(mask_bool.sum()) < args.min_mask_pixels:
        return None

    bbox = mask_bbox(mask_bool)
    if bbox is None:
        return None

    person_bbox = expand_bbox(bbox, (frame_h, frame_w), args.person_padding)
    person_crop_bgr = crop_array(frame_bgr, person_bbox)
    mask_crop = crop_array(mask_bool, person_bbox)
    if person_crop_bgr.size == 0 or not mask_crop.any():
        return None

    masked_crop_bgr = person_crop_bgr.copy()
    masked_crop_bgr[~mask_crop] = 0

    faces = face_detector.get(masked_crop_bgr)
    face = choose_face(faces, mask_crop, args.min_face_score)
    if face is None:
        return None

    local_face_bbox = clip_face_bbox(face.bbox, person_crop_bgr.shape[:2])
    if local_face_bbox is None:
        return None

    px1, py1, _, _ = person_bbox
    lx1, ly1, lx2, ly2 = local_face_bbox
    global_face_bbox = (px1 + lx1, py1 + ly1, px1 + lx2, py1 + ly2)
    padded_face_bbox = expand_bbox(global_face_bbox, (frame_h, frame_w), args.face_padding)
    face_crop_bgr = crop_array(frame_bgr, padded_face_bbox)
    if face_crop_bgr.size == 0:
        return None

    face_crop_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
    return FaceCrop(
        segment_id=int(segment_id),
        frame_idx=int(frame_idx),
        frame_stem=frame_stem,
        image=Image.fromarray(face_crop_rgb),
        det_score=float(getattr(face, "det_score", 0.0)),
        face_bbox=[int(v) for v in padded_face_bbox],
        person_bbox=[int(v) for v in person_bbox],
    )


def embed_face_crops(
    crops: list[FaceCrop],
    model: CLIPModel,
    processor: CLIPImageProcessor,
    device: torch.device,
    batch_size: int,
) -> list[torch.Tensor | None]:
    embeddings: list[torch.Tensor | None] = []
    for start in range(0, len(crops), batch_size):
        batch = crops[start : start + batch_size]
        inputs = processor(images=[crop.image for crop in batch], return_tensors="pt").to(device)
        with torch.inference_mode():
            emb = model.get_image_features(**inputs)
            emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        emb = emb.detach().cpu()
        for row in emb:
            if torch.isfinite(row).all() and float(row.norm().item()) > 0:
                embeddings.append(row)
            else:
                embeddings.append(None)
    return embeddings


def initial_segment_stats(segment_id: int) -> dict[str, Any]:
    return {
        "segment_id": int(segment_id),
        "num_mask_frames": 0,
        "num_processed_mask_frames": 0,
        "num_detected_faces": 0,
        "num_valid_embeddings": 0,
        "num_no_face_frames": 0,
        "num_embedding_failures": 0,
        "is_valid_embedding": False,
        "embedding_shape": None,
        "valid_frame_indices": [],
        "valid_frame_stems": [],
        "no_face_frame_indices": [],
        "detection_scores": [],
        "face_bboxes": [],
        "person_bboxes": [],
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        json.dump(payload, file, indent=2)


def detector_providers(face_detector: Any) -> list[str]:
    det_model = getattr(face_detector, "det_model", None)
    session = getattr(det_model, "session", None)
    if session is None or not hasattr(session, "get_providers"):
        return []
    return [str(provider) for provider in session.get_providers()]


def detector_provider_options(face_detector: Any) -> dict[str, dict[str, str]]:
    det_model = getattr(face_detector, "det_model", None)
    session = getattr(det_model, "session", None)
    if session is None or not hasattr(session, "get_provider_options"):
        return {}
    return {
        str(provider): {str(key): str(value) for key, value in options.items()}
        for provider, options in session.get_provider_options().items()
    }


def process_job(
    job: VideoJob,
    args: argparse.Namespace,
    face_detector: Any,
    style_model: CLIPModel,
    processor: CLIPImageProcessor,
    embedding_dim: int,
    device: torch.device,
) -> dict[str, Any]:
    output_embedding_path = job.output_dir / "embeding.pt"
    summary_path = job.output_dir / "summary.json"
    if output_embedding_path.exists() and summary_path.exists() and not args.overwrite:
        print(f"[SKIP] {job.split}/{job.video_name}: person embeding exists")
        return {"status": "skipped", "reason": "existing_output", "video_name": job.video_name}

    frame_paths = list_frame_paths(job.frame_dir)
    if not frame_paths:
        raise ValueError(f"No frames found: {job.frame_dir}")

    mask_dict = load_mask_dict(job.mask_path)
    ids = segment_ids(mask_dict)
    stats = {segment_id: initial_segment_stats(segment_id) for segment_id in ids}
    embeddings_by_segment: dict[int, list[torch.Tensor]] = {segment_id: [] for segment_id in ids}
    face_crops: list[FaceCrop] = []
    sample_every = max(1, int(args.sample_every))
    max_frames = args.max_frames if args.max_frames is not None else len(frame_paths)
    processed_frame_count = 0
    frames_with_masks = 0
    missing_frame_indices = 0

    for frame_idx, frame_path in enumerate(frame_paths):
        if frame_idx >= max_frames:
            break
        persons = mask_dict.get(frame_idx)
        if not persons:
            continue
        frames_with_masks += 1
        for segment_id in persons:
            stats[int(segment_id)]["num_mask_frames"] += 1
        if frame_idx % sample_every != 0:
            continue

        frame_bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            missing_frame_indices += 1
            continue
        processed_frame_count += 1

        for segment_id, mask in persons.items():
            segment_id = int(segment_id)
            stats[segment_id]["num_processed_mask_frames"] += 1
            crop = extract_face_crop(
                frame_bgr=frame_bgr,
                mask=mask,
                segment_id=segment_id,
                frame_idx=frame_idx,
                frame_stem=frame_path.stem,
                face_detector=face_detector,
                args=args,
            )
            if crop is None:
                stats[segment_id]["num_no_face_frames"] += 1
                stats[segment_id]["no_face_frame_indices"].append(int(frame_idx))
                continue
            stats[segment_id]["num_detected_faces"] += 1
            face_crops.append(crop)

    embedded = embed_face_crops(
        crops=face_crops,
        model=style_model,
        processor=processor,
        device=device,
        batch_size=args.batch_size,
    )

    for crop, embedding in zip(face_crops, embedded):
        segment_stats = stats[crop.segment_id]
        if embedding is None:
            segment_stats["num_embedding_failures"] += 1
            continue
        embeddings_by_segment[crop.segment_id].append(embedding)
        segment_stats["num_valid_embeddings"] += 1
        segment_stats["valid_frame_indices"].append(int(crop.frame_idx))
        segment_stats["valid_frame_stems"].append(crop.frame_stem)
        segment_stats["detection_scores"].append(float(crop.det_score))
        segment_stats["face_bboxes"].append([int(value) for value in crop.face_bbox])
        segment_stats["person_bboxes"].append([int(value) for value in crop.person_bbox])

    output_embeddings: dict[int, dict[str, Any]] = {}
    for segment_id in ids:
        valid_embeddings = embeddings_by_segment[segment_id]
        if valid_embeddings:
            tensor = torch.stack(valid_embeddings, dim=0).to(dtype=torch.float32)
            stats[segment_id]["is_valid_embedding"] = True
            stats[segment_id]["embedding_shape"] = [int(dim) for dim in tensor.shape]
            output_embeddings[int(segment_id)] = {
                "embeddings": tensor,
                "frame_indices": [int(value) for value in stats[segment_id]["valid_frame_indices"]],
                "frame_stems": [str(value) for value in stats[segment_id]["valid_frame_stems"]],
                "detection_scores": [
                    float(value) for value in stats[segment_id]["detection_scores"]
                ],
                "face_bboxes": [
                    [int(coord) for coord in bbox]
                    for bbox in stats[segment_id]["face_bboxes"]
                ],
                "person_bboxes": [
                    [int(coord) for coord in bbox]
                    for bbox in stats[segment_id]["person_bboxes"]
                ],
            }
        else:
            stats[segment_id]["is_valid_embedding"] = False
            stats[segment_id]["embedding_shape"] = None

    total_valid_embeddings = int(sum(item["num_valid_embeddings"] for item in stats.values()))
    detected_faces_per_segment = {
        str(segment_id): int(stats[segment_id]["num_detected_faces"])
        for segment_id in ids
    }
    total_detected_faces = int(sum(detected_faces_per_segment.values()))
    valid_segment_ids = sorted(
        int(segment_id)
        for segment_id, item in stats.items()
        if bool(item["is_valid_embedding"])
    )
    invalid_segment_ids = sorted(set(ids) - set(valid_segment_ids))
    embedding_index = {
        str(segment_id): [
            {
                "row": row_index,
                "frame_idx": int(frame_idx),
                "frame_stem": str(frame_stem),
                "detection_score": float(score),
                "face_bbox": [int(value) for value in face_bbox],
                "person_bbox": [int(value) for value in person_bbox],
            }
            for row_index, (frame_idx, frame_stem, score, face_bbox, person_bbox) in enumerate(
                zip(
                    stats[segment_id]["valid_frame_indices"],
                    stats[segment_id]["valid_frame_stems"],
                    stats[segment_id]["detection_scores"],
                    stats[segment_id]["face_bboxes"],
                    stats[segment_id]["person_bboxes"],
                )
            )
        ]
        for segment_id in valid_segment_ids
    }

    summary = {
        "split": job.split,
        "video_name": job.video_name,
        "detected_faces_per_segment": detected_faces_per_segment,
        "total_detected_faces": total_detected_faces,
        "embedding_index": embedding_index,
        "source_mask_path": str(job.mask_path),
        "frame_dir": str(job.frame_dir),
        "output_embedding_path": str(output_embedding_path),
        "summary_path": str(summary_path),
        "styleid_model_id": args.styleid_model_id,
        "insightface_model": args.insightface_model,
        "insightface_detector_providers": detector_providers(face_detector),
        "insightface_detector_provider_options": detector_provider_options(face_detector),
        "styleid_device": str(next(style_model.parameters()).device),
        "embedding_dim": int(embedding_dim),
        "num_frames": len(frame_paths),
        "num_frames_with_masks": int(frames_with_masks),
        "num_processed_frames": int(processed_frame_count),
        "num_segments": len(ids),
        "segment_ids": ids,
        "valid_segment_ids": valid_segment_ids,
        "invalid_segment_ids": invalid_segment_ids,
        "num_valid_embeddings": total_valid_embeddings,
        "num_omitted_invalid_segments": len(invalid_segment_ids),
        "parameters": {
            "sample_every": int(args.sample_every),
            "max_frames": args.max_frames,
            "det_size": int(args.det_size),
            "min_face_score": float(args.min_face_score),
            "min_mask_pixels": int(args.min_mask_pixels),
            "person_padding": float(args.person_padding),
            "face_padding": float(args.face_padding),
            "batch_size": int(args.batch_size),
            "device": str(device),
        },
        "diagnostics": {
            "missing_or_unreadable_frames": int(missing_frame_indices),
        },
        "per_segment": {str(key): value for key, value in stats.items()},
    }

    job.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(output_embeddings, output_embedding_path)
    write_json(summary_path, summary)
    print(
        f"[OK] {job.split}/{job.video_name}: "
        f"valid_embeddings={total_valid_embeddings} "
        f"omitted_invalid_segments={len(invalid_segment_ids)} "
        f"valid_segments={valid_segment_ids}"
    )
    return {"status": "processed", **summary}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract StyleID face embeddings from masked EgoCom person segments."
    )
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--split",
        type=str,
        default="all_existing",
        help="Comma-separated splits, or all_existing to scan splits with refined masks.",
    )
    parser.add_argument("--video", type=str, default=None, help="Optional single video name.")
    parser.add_argument("--mask_path", type=str, default=None, help="Optional explicit mask.pt path.")
    parser.add_argument("--frame_dir", type=str, default=None, help="Frame directory for --mask_path.")
    parser.add_argument("--output_root", type=str, default=None, help="Optional output root for jobs.")
    parser.add_argument("--output_dir", type=str, default=None, help="Optional exact output directory.")
    parser.add_argument("--styleid_model_id", type=str, default=DEFAULT_STYLEID_MODEL_ID)
    parser.add_argument("--insightface_model", type=str, default=DEFAULT_INSIGHTFACE_MODEL)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--det_size", type=positive_int, default=640)
    parser.add_argument("--min_face_score", type=float, default=0.3)
    parser.add_argument("--min_mask_pixels", type=positive_int, default=25)
    parser.add_argument("--person_padding", type=nonnegative_float, default=0.08)
    parser.add_argument("--face_padding", type=nonnegative_float, default=0.25)
    parser.add_argument("--sample_every", type=positive_int, default=1)
    parser.add_argument("--batch_size", type=positive_int, default=16)
    parser.add_argument("--limit", type=nonnegative_int, default=None, help="Limit number of videos.")
    parser.add_argument("--start_index", type=nonnegative_int, default=0, help="Start at this 0-based job index.")
    parser.add_argument("--stride", type=positive_int, default=1, help="Process every Nth job after --start_index.")
    parser.add_argument("--max_frames", type=nonnegative_int, default=None, help="Limit frames per video.")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.min_face_score < 0:
        raise ValueError(f"--min_face_score must be >= 0, got {args.min_face_score}")
    if args.output_dir and not (args.mask_path or args.video):
        raise ValueError("--output_dir is only supported with --mask_path or --video")

    jobs = collect_jobs(args)
    if not jobs:
        print("No matching videos found.")
        return

    device = resolve_device(args.device)
    print(f"Using device: {device}")
    print(f"Loading InsightFace detector: {args.insightface_model}")
    face_detector = load_face_detector(args, device)
    print(f"Loading StyleID model: {args.styleid_model_id}")
    style_model, processor, embedding_dim = load_styleid_model(args.styleid_model_id, device)

    print(f"Found {len(jobs)} videos")
    processed = 0
    skipped = 0
    failed = 0
    for index, job in enumerate(jobs, start=1):
        print(f"[{index}/{len(jobs)}] {job.split}/{job.video_name}")
        try:
            result = process_job(
                job=job,
                args=args,
                face_detector=face_detector,
                style_model=style_model,
                processor=processor,
                embedding_dim=embedding_dim,
                device=device,
            )
            if result["status"] == "processed":
                processed += 1
            else:
                skipped += 1
        except Exception as exc:
            failed += 1
            print(f"[ERROR] {job.split}/{job.video_name}: {exc}")

    print(f"Done. processed={processed} skipped={skipped} failed={failed}")


if __name__ == "__main__":
    main()
