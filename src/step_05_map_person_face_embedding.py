"""
Map EgoCom refined-mask segment ids to person ids using face embeddings.

Inputs:
  /home/prj/data/egocom_holdout/1min/{split}/refined_mask/{clip}/mask.pt
  /home/prj/data/egocom_holdout/1min/{split}/refined_mask/{clip}/summary.json
  /home/prj/data/egocom_holdout/1min/{split}/person_face_emb/{clip}/embeding.pt
  /home/prj/data/egocom_holdout/1min/{split}/frame/{clip}/*.jpg

Outputs:
  /home/prj/data/egocom_holdout/1min/{split}/person_face_mapping/
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


DEFAULT_DATA_ROOT = "/home/prj/data/egocom_holdout/1min"
ALL_PERSON_IDS = {1, 2, 3}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
CLIP_RE = re.compile(
    r"^(vid_\d+)__day_(?P<day>\d+)__con_(?P<con>\d+)__person_(?P<camera>\d+)"
    r"(?:_(?P<part>part\d+))?_chunk_(?P<chunk>\d+)$"
)
PALETTE_BGR = [
    (0, 0, 255),
    (0, 180, 0),
    (255, 0, 0),
    (0, 180, 180),
    (180, 0, 180),
    (180, 180, 0),
]


@dataclass(frozen=True)
class ClipInfo:
    split: str
    name: str
    scene_key: str
    camera_person: int
    chunk: int
    mask_dir: Path
    emb_dir: Path
    frame_dir: Path


@dataclass
class SegmentEmbedding:
    segment_id: int
    embeddings: torch.Tensor
    frame_indices: list[int]
    frame_stems: list[str]
    detection_scores: list[float]
    aggregate: np.ndarray


@dataclass
class ClipData:
    info: ClipInfo
    top_segment_ids: list[int]
    selected_segment_ids: list[int]
    missing_embedding_segment_ids: list[int]
    low_embedding_count_segment_ids: list[int]
    embeddings: dict[int, SegmentEmbedding]
    mask_dict: dict[int, dict[int, np.ndarray]]


def parse_comma_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return parsed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be >= 1")
    return parsed


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


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r") as file:
        return json.load(file)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        json.dump(payload, file, indent=2)


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


def discover_splits(data_root: Path, split_arg: str) -> list[str]:
    if split_arg != "all_existing":
        return parse_comma_list(split_arg)
    return sorted(
        split_dir.name
        for split_dir in data_root.iterdir()
        if split_dir.is_dir()
        and (split_dir / "refined_mask").is_dir()
        and (split_dir / "person_face_emb").is_dir()
    )


def discover_scene_chunks(args: argparse.Namespace) -> dict[str, dict[int, list[ClipInfo]]]:
    data_root = Path(args.data_root)
    scenes: dict[str, dict[int, list[ClipInfo]]] = defaultdict(lambda: defaultdict(list))
    for split in discover_splits(data_root, args.split):
        split_dir = data_root / split
        mask_root = split_dir / "refined_mask"
        emb_root = split_dir / "person_face_emb"
        frame_root = split_dir / "frame"
        if not mask_root.is_dir():
            print(f"[SKIP] {split}: missing {mask_root}")
            continue

        for mask_dir in sorted(path for path in mask_root.iterdir() if path.is_dir()):
            parsed = parse_clip_name(mask_dir.name)
            if parsed is None:
                continue
            if args.scene_key and parsed["scene_key"] != args.scene_key:
                continue
            if args.video and mask_dir.name != args.video:
                continue
            info = ClipInfo(
                split=split,
                name=mask_dir.name,
                scene_key=parsed["scene_key"],
                camera_person=int(parsed["camera_person"]),
                chunk=int(parsed["chunk"]),
                mask_dir=mask_dir,
                emb_dir=emb_root / mask_dir.name,
                frame_dir=frame_root / mask_dir.name,
            )
            scenes[f"{split}/{parsed['scene_key']}"][int(parsed["chunk"])].append(info)
    return {
        scene_key: dict(sorted(chunks.items()))
        for scene_key, chunks in sorted(scenes.items())
    }


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


def count_mask_frames(mask_dict: dict[int, dict[int, np.ndarray]]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for persons in mask_dict.values():
        for segment_id in persons:
            segment_id = int(segment_id)
            counts[segment_id] = counts.get(segment_id, 0) + 1
    return counts


def top_counted_segments(mask_dir: Path, mask_dict: dict[int, dict[int, np.ndarray]], top_k: int) -> list[int]:
    summary_path = mask_dir / "summary.json"
    counts: dict[int, int] = {}
    if summary_path.exists():
        summary = load_json(summary_path)
        raw_counts = summary.get("remaining_person_frequency") or summary.get("person_frequency") or {}
        counts = {int(key): int(value) for key, value in raw_counts.items() if int(value) > 0}
    if not counts:
        counts = count_mask_frames(mask_dict)
    return [
        int(segment_id)
        for segment_id, _ in sorted(counts.items(), key=lambda item: (-int(item[1]), int(item[0])))[:top_k]
    ]


def normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12 or not np.isfinite(norm):
        return np.zeros_like(vector, dtype=np.float32)
    return (vector / norm).astype(np.float32)


def load_segment_embeddings(
    emb_path: Path,
    segment_ids: list[int],
    min_embeddings: int,
) -> tuple[dict[int, SegmentEmbedding], list[int]]:
    if not emb_path.exists():
        return {}, []
    raw = load_torch(emb_path)
    if not isinstance(raw, dict):
        return {}, []

    out: dict[int, SegmentEmbedding] = {}
    low_embedding_count_segment_ids: list[int] = []
    for segment_id in segment_ids:
        item = raw.get(int(segment_id))
        if not isinstance(item, dict) or "embeddings" not in item:
            continue
        tensor = item["embeddings"]
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2 or tensor.numel() == 0:
            continue
        tensor = tensor.detach().cpu().to(dtype=torch.float32)
        if int(tensor.shape[0]) < int(min_embeddings):
            low_embedding_count_segment_ids.append(int(segment_id))
            continue
        if not bool(torch.isfinite(tensor).all()):
            continue
        aggregate = normalize(tensor.mean(dim=0).numpy())
        if not np.any(aggregate):
            continue
        out[int(segment_id)] = SegmentEmbedding(
            segment_id=int(segment_id),
            embeddings=tensor,
            frame_indices=[int(value) for value in item.get("frame_indices", [])],
            frame_stems=[str(value) for value in item.get("frame_stems", [])],
            detection_scores=[float(value) for value in item.get("detection_scores", [])],
            aggregate=aggregate,
        )
    return out, low_embedding_count_segment_ids


def load_clip_data(info: ClipInfo, top_k: int, min_embeddings: int) -> ClipData | None:
    mask_path = info.mask_dir / "mask.pt"
    emb_path = info.emb_dir / "embeding.pt"
    if not mask_path.exists():
        print(f"[SKIP] {info.split}/{info.name}: missing mask.pt")
        return None
    mask_dict = load_mask_dict(mask_path)
    top_segment_ids = top_counted_segments(info.mask_dir, mask_dict, top_k)
    embeddings, low_embedding_count_segment_ids = load_segment_embeddings(
        emb_path,
        top_segment_ids,
        min_embeddings=min_embeddings,
    )
    selected_segment_ids = [segment_id for segment_id in top_segment_ids if segment_id in embeddings]
    missing_embedding_segment_ids = [
        segment_id
        for segment_id in top_segment_ids
        if segment_id not in embeddings and segment_id not in low_embedding_count_segment_ids
    ]
    return ClipData(
        info=info,
        top_segment_ids=top_segment_ids,
        selected_segment_ids=selected_segment_ids,
        missing_embedding_segment_ids=missing_embedding_segment_ids,
        low_embedding_count_segment_ids=low_embedding_count_segment_ids,
        embeddings=embeddings,
        mask_dict=mask_dict,
    )


def cosine_matrix(
    clip_a: ClipData,
    clip_b: ClipData,
) -> tuple[np.ndarray, list[int], list[int]]:
    ids_a = [segment_id for segment_id in clip_a.selected_segment_ids if segment_id in clip_a.embeddings]
    ids_b = [segment_id for segment_id in clip_b.selected_segment_ids if segment_id in clip_b.embeddings]
    matrix = np.zeros((len(ids_a), len(ids_b)), dtype=np.float32)
    for row, seg_a in enumerate(ids_a):
        for col, seg_b in enumerate(ids_b):
            matrix[row, col] = float(np.dot(clip_a.embeddings[seg_a].aggregate, clip_b.embeddings[seg_b].aggregate))
    return matrix, ids_a, ids_b


def best_match(matrix: np.ndarray, ids_a: list[int], ids_b: list[int]) -> dict[str, Any] | None:
    if matrix.size == 0:
        return None
    flat = matrix.reshape(-1)
    best_flat = int(np.argmax(flat))
    best_score = float(flat[best_flat])
    row, col = np.unravel_index(best_flat, matrix.shape)
    if flat.size > 1:
        second_best = float(np.partition(flat, -2)[-2])
        margin = best_score - second_best
    else:
        second_best = None
        margin = best_score
    return {
        "segment_id_a": int(ids_a[row]),
        "segment_id_b": int(ids_b[col]),
        "matrix_index": [int(row), int(col)],
        "top_similarity": best_score,
        "second_best_similarity": second_best,
        "top_margin": float(margin),
    }


def initial_video_result(clip: ClipData) -> dict[str, Any]:
    assignments = {str(segment_id): None for segment_id in clip.top_segment_ids}
    return {
        "clip_name": clip.info.name,
        "camera_person": int(clip.info.camera_person),
        "top_segment_ids": [int(value) for value in clip.top_segment_ids],
        "selected_embedding_segment_ids": [int(value) for value in clip.selected_segment_ids],
        "missing_embedding_segment_ids": [int(value) for value in clip.missing_embedding_segment_ids],
        "low_embedding_count_segment_ids": [int(value) for value in clip.low_embedding_count_segment_ids],
        "assignments": assignments,
        "unknown_segments": [int(value) for value in clip.top_segment_ids],
        "pair_evidence": [],
        "conflicts": [],
        "representatives": {},
    }


def add_assignment(
    result: dict[str, Any],
    segment_id: int,
    person_id: int,
    source: dict[str, Any],
) -> bool:
    key = str(int(segment_id))
    if key not in result["assignments"]:
        result["assignments"][key] = None
    current = result["assignments"].get(key)
    if current is None:
        result["assignments"][key] = int(person_id)
        return True
    if int(current) == int(person_id):
        return True

    result["assignments"][key] = None
    result["conflicts"].append(
        {
            "segment_id": int(segment_id),
            "existing_person_id": int(current),
            "new_person_id": int(person_id),
            "source": source,
        }
    )
    return False


def add_pair_evidence(
    result_a: dict[str, Any],
    result_b: dict[str, Any],
    clip_a: ClipData,
    clip_b: ClipData,
    expected_person: int,
    ids_a: list[int],
    ids_b: list[int],
    matrix: np.ndarray,
    match: dict[str, Any] | None,
    accepted: bool,
    reason: str,
    threshold: float,
) -> None:
    base = {
        "expected_common_person": int(expected_person),
        "accepted": bool(accepted),
        "reason": reason,
        "one_by_one_threshold": float(threshold),
        "matrix_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
    }
    match_payload = None
    if match is not None:
        match_payload = {
            "top_similarity": float(match["top_similarity"]),
            "second_best_similarity": match["second_best_similarity"],
            "top_margin": float(match["top_margin"]),
            "matrix_index": [int(value) for value in match["matrix_index"]],
        }

    result_a["pair_evidence"].append(
        {
            **base,
            "other_clip": clip_b.info.name,
            "local_segment_ids": [int(value) for value in ids_a],
            "other_segment_ids": [int(value) for value in ids_b],
            "similarity_matrix": matrix.astype(float).tolist(),
            "local_segment_id": None if match is None else int(match["segment_id_a"]),
            "other_segment_id": None if match is None else int(match["segment_id_b"]),
            "match": match_payload,
        }
    )
    result_b["pair_evidence"].append(
        {
            **base,
            "other_clip": clip_a.info.name,
            "local_segment_ids": [int(value) for value in ids_b],
            "other_segment_ids": [int(value) for value in ids_a],
            "similarity_matrix": matrix.T.astype(float).tolist(),
            "local_segment_id": None if match is None else int(match["segment_id_b"]),
            "other_segment_id": None if match is None else int(match["segment_id_a"]),
            "match": (
                None
                if match_payload is None
                else {
                    **match_payload,
                    "matrix_index": [int(match["matrix_index"][1]), int(match["matrix_index"][0])],
                }
            ),
        }
    )


def process_pair(
    clip_a: ClipData,
    clip_b: ClipData,
    result_a: dict[str, Any],
    result_b: dict[str, Any],
    threshold: float,
) -> None:
    expected = sorted(ALL_PERSON_IDS - {clip_a.info.camera_person, clip_b.info.camera_person})
    if len(expected) != 1:
        matrix, ids_a, ids_b = cosine_matrix(clip_a, clip_b)
        add_pair_evidence(
            result_a,
            result_b,
            clip_a,
            clip_b,
            expected_person=-1,
            ids_a=ids_a,
            ids_b=ids_b,
            matrix=matrix,
            match=None,
            accepted=False,
            reason="invalid_camera_pair",
            threshold=threshold,
        )
        return

    expected_person = int(expected[0])
    matrix, ids_a, ids_b = cosine_matrix(clip_a, clip_b)
    match = best_match(matrix, ids_a, ids_b)
    accepted = False
    reason = "empty_similarity_matrix"
    if match is not None:
        if matrix.shape == (1, 1) and float(match["top_similarity"]) < threshold:
            reason = "one_by_one_below_threshold"
        else:
            accepted = True
            reason = "accepted"

    add_pair_evidence(
        result_a,
        result_b,
        clip_a,
        clip_b,
        expected_person=expected_person,
        ids_a=ids_a,
        ids_b=ids_b,
        matrix=matrix,
        match=match,
        accepted=accepted,
        reason=reason,
        threshold=threshold,
    )
    if not accepted or match is None:
        return

    add_assignment(
        result_a,
        int(match["segment_id_a"]),
        expected_person,
        {"type": "pair_match", "other_clip": clip_b.info.name},
    )
    add_assignment(
        result_b,
        int(match["segment_id_b"]),
        expected_person,
        {"type": "pair_match", "other_clip": clip_a.info.name},
    )


def infer_remaining_assignments(clips: dict[str, ClipData], results: dict[str, dict[str, Any]]) -> None:
    for clip_name, clip in clips.items():
        result = results[clip_name]
        if result["conflicts"]:
            continue
        candidate_ids = [int(value) for value in clip.selected_segment_ids]
        if len(candidate_ids) != 2:
            continue

        assigned = {
            int(segment_id): int(person_id)
            for segment_id, person_id in result["assignments"].items()
            if person_id is not None and int(segment_id) in candidate_ids
        }
        if len(assigned) != 1:
            continue

        visible_persons = ALL_PERSON_IDS - {clip.info.camera_person}
        remaining_persons = visible_persons - set(assigned.values())
        remaining_segments = [segment_id for segment_id in candidate_ids if segment_id not in assigned]
        if len(remaining_persons) != 1 or len(remaining_segments) != 1:
            continue

        add_assignment(
            result,
            remaining_segments[0],
            next(iter(remaining_persons)),
            {"type": "inferred_remaining_visible_person"},
        )


def finalize_unknowns(results: dict[str, dict[str, Any]]) -> None:
    for result in results.values():
        result["unknown_segments"] = sorted(
            int(segment_id)
            for segment_id, person_id in result["assignments"].items()
            if person_id is None
        )


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
    bbox: tuple[int, int, int, int],
    image_shape: tuple[int, int],
    padding: float,
) -> tuple[int, int, int, int]:
    height, width = image_shape
    x1, y1, x2, y2 = [int(value) for value in bbox]
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    pad = int(round(max(box_w, box_h) * padding))
    return (
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(width, x2 + pad),
        min(height, y2 + pad),
    )


def list_frame_paths(frame_dir: Path) -> dict[str, Path]:
    if not frame_dir.is_dir():
        return {}
    return {
        path.stem: path
        for path in sorted(frame_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }


def blend_mask(frame: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    out = frame.copy()
    overlay = frame.copy()
    overlay[mask.astype(bool)] = color
    cv2.addWeighted(overlay, 0.35, out, 0.65, 0.0, dst=out)
    return out


def representative_for_segment(
    clip: ClipData,
    segment_id: int,
    person_id: int,
    output_dir: Path,
    save_vis: bool,
    crop_padding: float,
) -> tuple[dict[str, Any] | None, torch.Tensor | None]:
    segment = clip.embeddings.get(int(segment_id))
    if segment is None:
        return None, None

    rows = segment.embeddings.numpy().astype(np.float32)
    row_norms = np.linalg.norm(rows, axis=1, keepdims=True)
    normalized_rows = rows / np.clip(row_norms, 1e-12, None)
    scores = (normalized_rows @ segment.aggregate.reshape(-1, 1)).reshape(-1)
    order = np.argsort(-scores)
    frames_by_stem = list_frame_paths(clip.info.frame_dir)

    for row_index_raw in order:
        row_index = int(row_index_raw)
        if row_index >= len(segment.frame_indices) or row_index >= len(segment.frame_stems):
            continue
        frame_idx = int(segment.frame_indices[row_index])
        frame_stem = str(segment.frame_stems[row_index])
        persons = clip.mask_dict.get(frame_idx)
        if not persons or int(segment_id) not in persons:
            continue
        frame_path = frames_by_stem.get(frame_stem)
        if frame_path is None:
            continue
        frame_bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            continue
        mask = persons[int(segment_id)].astype(bool)
        if mask.shape[:2] != frame_bgr.shape[:2]:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (frame_bgr.shape[1], frame_bgr.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        bbox = mask_bbox(mask)
        if bbox is None:
            continue
        padded_bbox = expand_bbox(bbox, frame_bgr.shape[:2], crop_padding)
        x1, y1, x2, y2 = padded_bbox
        crop = frame_bgr[y1:y2, x1:x2].copy()
        if crop.size == 0:
            continue
        crop_path = None
        if save_vis:
            crop_mask = mask[y1:y2, x1:x2]
            color = PALETTE_BGR[int(segment_id) % len(PALETTE_BGR)]
            crop = blend_mask(crop, crop_mask, color)
            cv2.rectangle(crop, (0, 0), (max(0, crop.shape[1] - 1), max(0, crop.shape[0] - 1)), color, 2)
            crop_path = output_dir / "representative_crops" / clip.info.name / f"person_{int(person_id)}.jpg"
            crop_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(crop_path), crop)

        metadata = {
            "segment_id": int(segment_id),
            "row": int(row_index),
            "frame_idx": int(frame_idx),
            "frame_stem": frame_stem,
            "frame_path": str(frame_path),
            "similarity_to_aggregate": float(scores[row_index]),
            "detection_score": (
                None
                if row_index >= len(segment.detection_scores)
                else float(segment.detection_scores[row_index])
            ),
            "mask_bbox": [int(value) for value in bbox],
            "crop_bbox": [int(value) for value in padded_bbox],
            "crop_path": None if crop_path is None else str(crop_path),
        }
        return metadata, segment.embeddings[row_index].clone().to(dtype=torch.float32)

    return None, None


def attach_representatives(
    clips: dict[str, ClipData],
    results: dict[str, dict[str, Any]],
    scene_output_dir: Path,
    save_vis: bool,
    crop_padding: float,
) -> dict[str, dict[int, torch.Tensor]]:
    representative_tensors: dict[str, dict[int, torch.Tensor]] = {}
    for clip_name, result in results.items():
        clip = clips[clip_name]
        representative_tensors[clip_name] = {}
        for segment_id_raw, person_id in sorted(
            result["assignments"].items(), key=lambda item: int(item[0])
        ):
            if person_id is None:
                continue
            segment_id = int(segment_id_raw)
            metadata, tensor = representative_for_segment(
                clip=clip,
                segment_id=segment_id,
                person_id=int(person_id),
                output_dir=scene_output_dir,
                save_vis=save_vis,
                crop_padding=crop_padding,
            )
            if metadata is None or tensor is None:
                result["representatives"][str(segment_id)] = {
                    "person_id": int(person_id),
                    "status": "missing_representative_frame",
                }
                continue
            result["representatives"][str(segment_id)] = {
                "person_id": int(person_id),
                "status": "ok",
                **metadata,
            }
            representative_tensors[clip_name][segment_id] = tensor
    return representative_tensors


def build_scene_mapping(
    split: str,
    scene_key: str,
    infos: list[ClipInfo],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, ClipData]]:
    loaded: dict[str, ClipData] = {}
    for info in sorted(infos, key=lambda item: (item.camera_person, item.name)):
        data = load_clip_data(info, args.top_k, args.min_embeddings)
        if data is not None:
            loaded[info.name] = data

    results = {clip_name: initial_video_result(clip) for clip_name, clip in loaded.items()}
    camera_to_clips: dict[int, list[str]] = defaultdict(list)
    for clip_name, clip in loaded.items():
        camera_to_clips[clip.info.camera_person].append(clip_name)

    duplicate_cameras = {
        str(camera): names for camera, names in sorted(camera_to_clips.items()) if len(names) > 1
    }
    pair_count = 0
    camera_clip_names = {
        camera: sorted(names)[0]
        for camera, names in sorted(camera_to_clips.items())
        if names
    }
    cameras = sorted(camera_clip_names)
    for idx, camera_a in enumerate(cameras):
        for camera_b in cameras[idx + 1 :]:
            clip_a = loaded[camera_clip_names[camera_a]]
            clip_b = loaded[camera_clip_names[camera_b]]
            process_pair(
                clip_a,
                clip_b,
                results[clip_a.info.name],
                results[clip_b.info.name],
                threshold=float(args.one_by_one_threshold),
            )
            pair_count += 1

    infer_remaining_assignments(loaded, results)
    finalize_unknowns(results)

    total_segments = sum(len(result["assignments"]) for result in results.values())
    resolved_segments = sum(
        1
        for result in results.values()
        for person_id in result["assignments"].values()
        if person_id is not None
    )
    unresolved_segments = sum(len(result["unknown_segments"]) for result in results.values())
    conflict_segments = sum(len(result["conflicts"]) for result in results.values())
    matrix_shapes = Counter(
        tuple(evidence["matrix_shape"])
        for result in results.values()
        for evidence in result["pair_evidence"]
        if result["clip_name"] < evidence["other_clip"]
    )
    summary = {
        "split": split,
        "scene_key": scene_key,
        "num_input_clips": len(infos),
        "num_loaded_clips": len(loaded),
        "camera_persons": sorted(int(value) for value in camera_clip_names),
        "duplicate_cameras": duplicate_cameras,
        "num_pairs": int(pair_count),
        "total_segments": int(total_segments),
        "resolved_segments": int(resolved_segments),
        "unresolved_segments": int(unresolved_segments),
        "conflict_segments": int(conflict_segments),
        "matrix_shape_counts": {f"{key[0]}x{key[1]}": int(value) for key, value in matrix_shapes.items()},
        "one_by_one_threshold": float(args.one_by_one_threshold),
        "min_embeddings": int(args.min_embeddings),
    }
    chunk_values = sorted({int(info.chunk) for info in infos})
    if len(chunk_values) == 1:
        summary["chunk"] = int(chunk_values[0])
    scene_mapping = {
        "split": split,
        "scene_key": scene_key,
        "summary": summary,
        "clips": results,
    }
    return scene_mapping, summary, loaded


def process_scene(
    split: str,
    scene_key: str,
    infos: list[ClipInfo],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[int, torch.Tensor]]]:
    scene_mapping, summary, loaded = build_scene_mapping(split, scene_key, infos, args)
    output_root = Path(args.output_root) if args.output_root else Path(args.data_root) / split / "person_face_mapping"
    scene_output_dir = output_root / scene_key
    if args.overwrite and args.save_vis:
        shutil.rmtree(scene_output_dir / "representative_crops", ignore_errors=True)
    representative_tensors = attach_representatives(
        loaded,
        scene_mapping["clips"],
        scene_output_dir=scene_output_dir,
        save_vis=bool(args.save_vis),
        crop_padding=float(args.crop_padding),
    )
    return scene_mapping, summary, representative_tensors


def fallback_attempt_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunk": int(summary.get("chunk", -1)),
        "num_loaded_clips": int(summary.get("num_loaded_clips", 0)),
        "total_segments": int(summary.get("total_segments", 0)),
        "resolved_segments": int(summary.get("resolved_segments", 0)),
        "unresolved_segments": int(summary.get("unresolved_segments", 0)),
        "conflict_segments": int(summary.get("conflict_segments", 0)),
        "matrix_shape_counts": summary.get("matrix_shape_counts", {}),
    }


def unique_pair_evidence(scene_mapping: dict[str, Any]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for clip_name, result in sorted(scene_mapping["clips"].items()):
        for evidence in result.get("pair_evidence", []):
            other_clip = evidence.get("other_clip")
            if other_clip is None or str(clip_name) > str(other_clip):
                continue
            match = evidence.get("match") or {}
            pairs.append(
                {
                    "clip_a": clip_name,
                    "clip_b": other_clip,
                    "expected_common_person": evidence.get("expected_common_person"),
                    "accepted": bool(evidence.get("accepted")),
                    "reason": evidence.get("reason"),
                    "matrix_shape": evidence.get("matrix_shape"),
                    "local_segment_ids": evidence.get("local_segment_ids", []),
                    "other_segment_ids": evidence.get("other_segment_ids", []),
                    "local_segment_id": evidence.get("local_segment_id"),
                    "other_segment_id": evidence.get("other_segment_id"),
                    "top_similarity": match.get("top_similarity"),
                    "second_best_similarity": match.get("second_best_similarity"),
                    "top_margin": match.get("top_margin"),
                    "similarity_matrix": evidence.get("similarity_matrix", []),
                }
            )
    return pairs


def fallback_attempt_reasons(scene_mapping: dict[str, Any], summary: dict[str, Any]) -> list[str]:
    reasons: set[str] = set()
    if int(summary.get("num_loaded_clips", 0)) < 3:
        reasons.add("only_two_or_fewer_views")
    if int(summary.get("conflict_segments", 0)) > 0:
        reasons.add("conflicting_common_person_assignment")
    if int(summary.get("unresolved_segments", 0)) > 0:
        reasons.add("unresolved_assignment")

    for result in scene_mapping["clips"].values():
        if result.get("missing_embedding_segment_ids"):
            reasons.add("missing_embedding")
        if result.get("low_embedding_count_segment_ids"):
            reasons.add("low_embedding_count")
        for evidence in result.get("pair_evidence", []):
            if not evidence.get("accepted"):
                reasons.add(str(evidence.get("reason") or "rejected_pair"))
    if not reasons:
        reasons.add("clean")
    return sorted(reasons)


def fallback_attempt_detail(scene_mapping: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    clips: dict[str, Any] = {}
    for clip_name, result in sorted(scene_mapping["clips"].items()):
        clips[clip_name] = {
            "camera_person": int(result["camera_person"]),
            "top_segment_ids": [int(value) for value in result.get("top_segment_ids", [])],
            "selected_embedding_segment_ids": [
                int(value) for value in result.get("selected_embedding_segment_ids", [])
            ],
            "missing_embedding_segment_ids": [
                int(value) for value in result.get("missing_embedding_segment_ids", [])
            ],
            "low_embedding_count_segment_ids": [
                int(value) for value in result.get("low_embedding_count_segment_ids", [])
            ],
            "assignments": result.get("assignments", {}),
            "unknown_segments": [int(value) for value in result.get("unknown_segments", [])],
            "conflicts": result.get("conflicts", []),
        }
    return {
        **fallback_attempt_summary(summary),
        "reasons": fallback_attempt_reasons(scene_mapping, summary),
        "clips": clips,
        "pair_evidence": unique_pair_evidence(scene_mapping),
    }


def fallback_rank(summary: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        -int(summary.get("conflict_segments", 0)),
        int(summary.get("resolved_segments", 0)),
        -int(summary.get("unresolved_segments", 0)),
        int(summary.get("num_loaded_clips", 0)),
    )


def process_scene_with_chunk_fallback(
    split: str,
    scene_key: str,
    chunks: dict[int, list[ClipInfo]],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[int, torch.Tensor]]]:
    attempts: list[dict[str, Any]] = []
    attempt_details: list[dict[str, Any]] = []
    best: tuple[dict[str, Any], dict[str, Any], dict[str, ClipData]] | None = None
    selected: tuple[dict[str, Any], dict[str, Any], dict[str, ClipData]] | None = None

    for chunk, infos in sorted(chunks.items()):
        scene_mapping, summary, loaded = build_scene_mapping(split, scene_key, infos, args)
        attempts.append(fallback_attempt_summary(summary))
        attempt_details.append(fallback_attempt_detail(scene_mapping, summary))
        if best is None or fallback_rank(summary) > fallback_rank(best[1]):
            best = (scene_mapping, summary, loaded)
        if int(summary.get("conflict_segments", 0)) == 0:
            selected = (scene_mapping, summary, loaded)
            break

    if selected is None:
        if best is None:
            raise RuntimeError(f"No chunk attempts were built for {split}/{scene_key}")
        selected = best
        fallback_status = "all_chunks_conflicted"
    else:
        fallback_status = "resolved_without_conflict"

    scene_mapping, summary, loaded = selected
    selected_chunk = int(summary.get("chunk", -1))
    summary["selected_chunk"] = selected_chunk
    summary["num_chunk_attempts"] = len(attempts)
    summary["fallback_status"] = fallback_status
    summary["chunk_attempts"] = attempts
    scene_mapping["summary"] = summary
    scene_mapping["chunk_attempts"] = attempts
    scene_mapping["chunk_attempt_details"] = attempt_details

    output_root = Path(args.output_root) if args.output_root else Path(args.data_root) / split / "person_face_mapping"
    scene_output_dir = output_root / scene_key
    if args.overwrite and args.save_vis:
        shutil.rmtree(scene_output_dir / "representative_crops", ignore_errors=True)
    representative_tensors = attach_representatives(
        loaded,
        scene_mapping["clips"],
        scene_output_dir=scene_output_dir,
        save_vis=bool(args.save_vis),
        crop_padding=float(args.crop_padding),
    )
    return scene_mapping, summary, representative_tensors


def compact_scene_mapping(scene_mapping: dict[str, Any]) -> dict[str, Any]:
    compact_clips: dict[str, Any] = {}
    for clip_name, result in sorted(scene_mapping["clips"].items()):
        representatives: dict[str, Any] = {}
        for segment_id, rep in sorted(result.get("representatives", {}).items(), key=lambda item: int(item[0])):
            representatives[str(segment_id)] = {
                "person_id": rep.get("person_id"),
                "status": rep.get("status"),
                "frame_idx": rep.get("frame_idx"),
                "frame_stem": rep.get("frame_stem"),
                "embedding_path": "representative_embeddings.pt",
                "crop_path": rep.get("crop_path"),
            }

        compact_clips[clip_name] = {
            "camera_person": int(result["camera_person"]),
            "assignments": {
                str(segment_id): person_id
                for segment_id, person_id in sorted(
                    result["assignments"].items(), key=lambda item: int(item[0])
                )
            },
            "unknown_segments": [int(value) for value in result["unknown_segments"]],
            "representatives": representatives,
        }

    return {
        "split": scene_mapping["split"],
        "scene_key": scene_mapping["scene_key"],
        "summary": scene_mapping["summary"],
        "clips": compact_clips,
    }


def save_scene_outputs(
    split: str,
    scene_key: str,
    scene_mapping: dict[str, Any],
    scene_summary: dict[str, Any],
    representative_tensors: dict[str, dict[int, torch.Tensor]],
    args: argparse.Namespace,
) -> None:
    output_root = Path(args.output_root) if args.output_root else Path(args.data_root) / split / "person_face_mapping"
    scene_dir = output_root / scene_key
    mapping_path = scene_dir / "mapping.json"
    details_path = scene_dir / "details.json"
    summary_path = scene_dir / "summary.json"
    representative_path = scene_dir / "representative_embeddings.pt"
    if mapping_path.exists() and summary_path.exists() and not args.overwrite:
        print(f"[SKIP] {split}/{scene_key}: output exists")
        return
    compact_mapping = compact_scene_mapping(scene_mapping)
    write_json(mapping_path, compact_mapping)
    write_json(details_path, scene_mapping)
    write_json(summary_path, scene_summary)
    scene_dir.mkdir(parents=True, exist_ok=True)
    torch.save(representative_tensors, representative_path)
    print(
        f"[OK] {split}/{scene_key}: "
        f"chunk={scene_summary.get('selected_chunk', scene_summary.get('chunk', '-'))} "
        f"clips={scene_summary['num_loaded_clips']} "
        f"resolved={scene_summary['resolved_segments']}/{scene_summary['total_segments']} "
        f"conflicts={scene_summary['conflict_segments']} "
        f"attempts={scene_summary.get('num_chunk_attempts', 1)}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Map EgoCom refined-mask segment ids to person ids with face embedding similarity."
    )
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Comma-separated splits, or all_existing to scan splits with refined masks.",
    )
    parser.add_argument("--scene_key", type=str, default=None, help="Optional scene filter.")
    parser.add_argument("--video", type=str, default=None, help="Optional exact clip filter.")
    parser.add_argument("--output_root", type=str, default=None, help="Optional exact output root.")
    parser.add_argument("--top_k", type=positive_int, default=2)
    parser.add_argument(
        "--min_embeddings",
        type=positive_int,
        default=10,
        help="Reject selected segments with fewer than this many face embeddings.",
    )
    parser.add_argument("--one_by_one_threshold", type=positive_float, default=0.70)
    parser.add_argument("--crop_padding", type=positive_float, default=0.08)
    parser.add_argument("--save_vis", action="store_true", help="Save representative crop visualizations.")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data_root = Path(args.data_root)
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    scenes = discover_scene_chunks(args)
    if not scenes:
        print("No matching clips found.")
        return

    all_mappings: dict[str, Any] = {}
    all_summaries: dict[str, Any] = {}
    for full_scene_key, chunks in scenes.items():
        split, scene_key = full_scene_key.split("/", 1)
        scene_mapping, scene_summary, representative_tensors = process_scene_with_chunk_fallback(
            split=split,
            scene_key=scene_key,
            chunks=chunks,
            args=args,
        )
        save_scene_outputs(
            split=split,
            scene_key=scene_key,
            scene_mapping=scene_mapping,
            scene_summary=scene_summary,
            representative_tensors=representative_tensors,
            args=args,
        )
        all_mappings[full_scene_key] = scene_mapping
        all_summaries[full_scene_key] = scene_summary

    by_split: dict[str, list[str]] = defaultdict(list)
    for full_scene_key in all_mappings:
        split, _ = full_scene_key.split("/", 1)
        by_split[split].append(full_scene_key)

    for split, split_scene_keys in sorted(by_split.items()):
        output_root = Path(args.output_root) if args.output_root else data_root / split / "person_face_mapping"
        write_json(
            output_root / "all_scene_mappings.json",
            {
                key.split("/", 1)[1]: compact_scene_mapping(all_mappings[key])
                for key in sorted(split_scene_keys)
            },
        )
        write_json(
            output_root / "all_scene_summaries.json",
            {key.split("/", 1)[1]: all_summaries[key] for key in sorted(split_scene_keys)},
        )

    total_resolved = sum(int(summary["resolved_segments"]) for summary in all_summaries.values())
    total_segments = sum(int(summary["total_segments"]) for summary in all_summaries.values())
    print(
        "Done. "
        f"scenes={len(all_summaries)} "
        f"resolved={total_resolved}/{total_segments} "
        f"output_splits={','.join(sorted(by_split))}"
    )


if __name__ == "__main__":
    main()
