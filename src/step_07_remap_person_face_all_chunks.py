"""
Remap all EgoCom chunks using selected-chunk representative person embeddings.

This pass does not modify original masks. It writes per-chunk indications that
multiple segment ids can represent the same real person when a track disconnects
and reappears.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from map_person_face_embedding import (
    ALL_PERSON_IDS,
    CLIP_RE,
    DEFAULT_DATA_ROOT,
    IMAGE_SUFFIXES,
    PALETTE_BGR,
    ClipInfo,
    SegmentEmbedding,
    discover_splits,
    expand_bbox,
    list_frame_paths,
    load_json,
    load_mask_dict,
    load_torch,
    mask_bbox,
    normalize,
    parse_clip_name,
    positive_float,
    positive_int,
    write_json,
)


@dataclass
class RemapSegment:
    segment_id: int
    embeddings: torch.Tensor
    frame_indices: list[int]
    frame_stems: list[str]
    detection_scores: list[float]
    aggregate: np.ndarray
    frame_set: set[int]
    frame_range: list[int | None]


def parse_comma_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def discover_scene_chunks(args: argparse.Namespace) -> dict[str, dict[int, list[ClipInfo]]]:
    data_root = Path(args.data_root)
    scenes: dict[str, dict[int, list[ClipInfo]]] = defaultdict(lambda: defaultdict(list))
    for split in discover_splits(data_root, args.split):
        split_dir = data_root / split
        mask_root = split_dir / "refined_mask"
        emb_root = split_dir / "person_face_emb"
        frame_root = split_dir / "frame"
        if not mask_root.is_dir() or not emb_root.is_dir():
            continue
        for mask_dir in sorted(path for path in mask_root.iterdir() if path.is_dir()):
            parsed = parse_clip_name(mask_dir.name)
            if parsed is None:
                continue
            if args.scene_key and parsed["scene_key"] != args.scene_key:
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
    return {key: dict(sorted(value.items())) for key, value in sorted(scenes.items())}


def load_all_segment_embeddings(emb_path: Path, min_embeddings: int) -> tuple[dict[int, RemapSegment], list[int]]:
    if not emb_path.exists():
        return {}, []
    raw = load_torch(emb_path)
    if not isinstance(raw, dict):
        return {}, []

    out: dict[int, RemapSegment] = {}
    low_count: list[int] = []
    for segment_id_raw, item in sorted(raw.items(), key=lambda kv: int(kv[0])):
        segment_id = int(segment_id_raw)
        if not isinstance(item, dict) or "embeddings" not in item:
            continue
        tensor = item["embeddings"]
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2 or tensor.numel() == 0:
            continue
        tensor = tensor.detach().cpu().to(dtype=torch.float32)
        if int(tensor.shape[0]) < int(min_embeddings):
            low_count.append(segment_id)
            continue
        if not bool(torch.isfinite(tensor).all()):
            continue
        aggregate = normalize(tensor.mean(dim=0).numpy())
        if not np.any(aggregate):
            continue
        frame_indices = [int(value) for value in item.get("frame_indices", [])]
        frame_set = set(frame_indices)
        out[segment_id] = RemapSegment(
            segment_id=segment_id,
            embeddings=tensor,
            frame_indices=frame_indices,
            frame_stems=[str(value) for value in item.get("frame_stems", [])],
            detection_scores=[float(value) for value in item.get("detection_scores", [])],
            aggregate=aggregate,
            frame_set=frame_set,
            frame_range=[min(frame_set), max(frame_set)] if frame_set else [None, None],
        )
    return out, low_count


def load_scene_prototypes(scene_dir: Path) -> dict[int, dict[int, np.ndarray]]:
    mapping_path = scene_dir / "mapping.json"
    emb_path = scene_dir / "representative_embeddings.pt"
    if not mapping_path.exists() or not emb_path.exists():
        return {}
    mapping = load_json(mapping_path)
    representative = torch.load(emb_path, map_location="cpu", weights_only=False)
    prototypes: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
    for clip_name, clip in mapping.get("clips", {}).items():
        parsed = parse_clip_name(clip_name)
        if parsed is None:
            continue
        camera_person = int(parsed["camera_person"])
        clip_tensors = representative.get(clip_name, {})
        for segment_id_raw, person_id in clip.get("assignments", {}).items():
            if person_id is None:
                continue
            segment_id = int(segment_id_raw)
            tensor = clip_tensors.get(segment_id)
            if tensor is None:
                continue
            prototypes[camera_person][int(person_id)] = normalize(tensor.detach().cpu().numpy())
    return {camera: dict(persons) for camera, persons in prototypes.items()}


def score_segments(
    prototypes: dict[int, np.ndarray],
    segments: dict[int, RemapSegment],
) -> dict[int, list[dict[str, Any]]]:
    scored: dict[int, list[dict[str, Any]]] = {}
    for person_id, prototype in sorted(prototypes.items()):
        rows = []
        for segment_id, segment in sorted(segments.items()):
            rows.append(
                {
                    "segment_id": int(segment_id),
                    "similarity": float(np.dot(prototype, segment.aggregate)),
                    "num_embeddings": int(segment.embeddings.shape[0]),
                    "frame_range": segment.frame_range,
                }
            )
        scored[int(person_id)] = sorted(rows, key=lambda item: (-item["similarity"], item["segment_id"]))
    return scored


def greedy_primary_assignments(
    visible_persons: list[int],
    scores: dict[int, list[dict[str, Any]]],
) -> dict[int, dict[str, Any]]:
    candidates: list[tuple[float, int, int, dict[str, Any]]] = []
    for person_id in visible_persons:
        for row in scores.get(person_id, []):
            candidates.append((float(row["similarity"]), int(person_id), int(row["segment_id"]), row))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

    assigned_persons: set[int] = set()
    assigned_segments: set[int] = set()
    assignments: dict[int, dict[str, Any]] = {}
    for score, person_id, segment_id, row in candidates:
        if person_id in assigned_persons or segment_id in assigned_segments:
            continue
        assignments[person_id] = {
            "person_id": int(person_id),
            "primary_segment_id": int(segment_id),
            "primary_similarity": float(score),
            "primary_num_embeddings": int(row["num_embeddings"]),
            "primary_frame_range": row["frame_range"],
            "merged_segment_ids": [int(segment_id)],
            "merge_candidates": [],
            "rejected_merge_candidates": [],
        }
        assigned_persons.add(person_id)
        assigned_segments.add(segment_id)
        if len(assigned_persons) == len(visible_persons):
            break
    return assignments


def frame_overlap_count(a: set[int], b: set[int]) -> int:
    return len(a & b)


def add_merge_candidates(
    assignments: dict[int, dict[str, Any]],
    scores: dict[int, list[dict[str, Any]]],
    segments: dict[int, RemapSegment],
    merge_similarity: float,
    merge_margin: float,
    max_overlap_frames: int,
) -> None:
    primary_segments = {
        int(item["primary_segment_id"]) for item in assignments.values()
    }
    globally_merged = set(primary_segments)
    for person_id, assignment in sorted(assignments.items()):
        primary_id = int(assignment["primary_segment_id"])
        primary_score = float(assignment["primary_similarity"])
        merged_frames = set(segments[primary_id].frame_set)
        for row in scores.get(person_id, []):
            segment_id = int(row["segment_id"])
            if segment_id == primary_id or segment_id in globally_merged:
                continue
            similarity = float(row["similarity"])
            overlap = frame_overlap_count(merged_frames, segments[segment_id].frame_set)
            within_score = similarity >= merge_similarity and (primary_score - similarity) <= merge_margin
            low_overlap = overlap <= max_overlap_frames
            payload = {
                "segment_id": segment_id,
                "similarity": similarity,
                "score_delta_from_primary": float(primary_score - similarity),
                "frame_overlap_with_group": int(overlap),
                "frame_range": segments[segment_id].frame_range,
                "num_embeddings": int(segments[segment_id].embeddings.shape[0]),
            }
            if within_score and low_overlap:
                payload["reason"] = "similar_embedding_non_overlapping_frames"
                assignment["merged_segment_ids"].append(segment_id)
                assignment["merge_candidates"].append(payload)
                merged_frames.update(segments[segment_id].frame_set)
                globally_merged.add(segment_id)
            else:
                reasons = []
                if not within_score:
                    reasons.append("below_similarity_or_margin_threshold")
                if not low_overlap:
                    reasons.append("too_much_frame_overlap")
                payload["reasons"] = reasons
                assignment["rejected_merge_candidates"].append(payload)


def representative_for_remap_segment(
    info: ClipInfo,
    segment: RemapSegment,
    mask_dict: dict[int, dict[int, np.ndarray]],
    person_id: int,
    output_dir: Path,
    save_vis: bool,
    crop_padding: float,
) -> tuple[dict[str, Any] | None, torch.Tensor | None]:
    rows = segment.embeddings.numpy().astype(np.float32)
    row_norms = np.linalg.norm(rows, axis=1, keepdims=True)
    normalized_rows = rows / np.clip(row_norms, 1e-12, None)
    scores = (normalized_rows @ segment.aggregate.reshape(-1, 1)).reshape(-1)
    order = np.argsort(-scores)
    frames_by_stem = list_frame_paths(info.frame_dir)

    for row_index_raw in order:
        row_index = int(row_index_raw)
        if row_index >= len(segment.frame_indices) or row_index >= len(segment.frame_stems):
            continue
        frame_idx = int(segment.frame_indices[row_index])
        frame_stem = str(segment.frame_stems[row_index])
        persons = mask_dict.get(frame_idx)
        if not persons or int(segment.segment_id) not in persons:
            continue
        frame_path = frames_by_stem.get(frame_stem)
        if frame_path is None:
            continue
        frame_bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            continue
        mask = persons[int(segment.segment_id)].astype(bool)
        if mask.shape[:2] != frame_bgr.shape[:2]:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (frame_bgr.shape[1], frame_bgr.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        bbox = mask_bbox(mask)
        if bbox is None:
            continue
        crop_bbox = expand_bbox(bbox, frame_bgr.shape[:2], crop_padding)
        x1, y1, x2, y2 = crop_bbox
        crop = frame_bgr[y1:y2, x1:x2].copy()
        if crop.size == 0:
            continue
        crop_path = None
        if save_vis:
            crop_mask = mask[y1:y2, x1:x2]
            color = PALETTE_BGR[int(person_id) % len(PALETTE_BGR)]
            overlay = crop.copy()
            overlay[crop_mask] = color
            cv2.addWeighted(overlay, 0.35, crop, 0.65, 0.0, dst=crop)
            cv2.rectangle(crop, (0, 0), (max(0, crop.shape[1] - 1), max(0, crop.shape[0] - 1)), color, 2)
            crop_path = (
                output_dir
                / "remap_visualizations"
                / info.name
                / f"person_{int(person_id)}_segment_{int(segment.segment_id)}.jpg"
            )
            crop_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(crop_path), crop)
        metadata = {
            "segment_id": int(segment.segment_id),
            "person_id": int(person_id),
            "row": int(row_index),
            "frame_idx": int(frame_idx),
            "frame_stem": frame_stem,
            "frame_path": str(frame_path),
            "similarity_to_aggregate": float(scores[row_index]),
            "mask_bbox": [int(value) for value in bbox],
            "crop_bbox": [int(value) for value in crop_bbox],
            "crop_path": None if crop_path is None else str(crop_path),
        }
        return metadata, segment.embeddings[row_index].clone().to(dtype=torch.float32)
    return None, None


def remap_clip(
    info: ClipInfo,
    prototypes: dict[int, dict[int, np.ndarray]],
    scene_dir: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], dict[int, torch.Tensor]]:
    camera_prototypes = prototypes.get(int(info.camera_person), {})
    visible_persons = sorted(ALL_PERSON_IDS - {int(info.camera_person)})
    visible_persons = [person_id for person_id in visible_persons if person_id in camera_prototypes]

    segments, low_count = load_all_segment_embeddings(info.emb_dir / "embeding.pt", args.min_embeddings)
    mask_dict = load_mask_dict(info.mask_dir / "mask.pt") if (info.mask_dir / "mask.pt").exists() else {}
    scores = score_segments(
        {person_id: camera_prototypes[person_id] for person_id in visible_persons},
        segments,
    )
    assignments = greedy_primary_assignments(visible_persons, scores)
    add_merge_candidates(
        assignments,
        scores,
        segments,
        merge_similarity=float(args.merge_similarity),
        merge_margin=float(args.merge_margin),
        max_overlap_frames=int(args.max_overlap_frames),
    )

    representative_tensors: dict[int, torch.Tensor] = {}
    representatives: dict[str, Any] = {}
    for assignment in assignments.values():
        person_id = int(assignment["person_id"])
        for segment_id in assignment["merged_segment_ids"]:
            metadata, tensor = representative_for_remap_segment(
                info=info,
                segment=segments[int(segment_id)],
                mask_dict=mask_dict,
                person_id=person_id,
                output_dir=scene_dir,
                save_vis=bool(args.save_vis),
                crop_padding=float(args.crop_padding),
            )
            if metadata is not None and tensor is not None:
                representatives[str(segment_id)] = metadata
                representative_tensors[int(segment_id)] = tensor

    compact_people = {}
    for person_id, assignment in sorted(assignments.items()):
        compact_people[str(person_id)] = {
            "primary_segment_id": int(assignment["primary_segment_id"]),
            "merged_segment_ids": [int(value) for value in assignment["merged_segment_ids"]],
            "primary_similarity": float(assignment["primary_similarity"]),
            "representatives": {
                str(segment_id): representatives.get(str(segment_id))
                for segment_id in assignment["merged_segment_ids"]
                if str(segment_id) in representatives
            },
        }

    detail = {
        "clip_name": info.name,
        "camera_person": int(info.camera_person),
        "chunk": int(info.chunk),
        "visible_persons": visible_persons,
        "num_valid_segments": len(segments),
        "low_embedding_count_segment_ids": low_count,
        "scores": scores,
        "people": assignments,
    }
    compact = {
        "clip_name": info.name,
        "camera_person": int(info.camera_person),
        "chunk": int(info.chunk),
        "people": compact_people,
        "low_embedding_count_segment_ids": low_count,
    }
    return compact, detail, representative_tensors


def process_scene(full_scene_key: str, chunks: dict[int, list[ClipInfo]], args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[int, torch.Tensor]] | None]:
    split, scene_key = full_scene_key.split("/", 1)
    scene_dir = Path(args.data_root) / split / "person_face_mapping" / scene_key
    prototypes = load_scene_prototypes(scene_dir)
    if not prototypes:
        return (
            {"split": split, "scene_key": scene_key, "status": "missing_prototypes", "chunks": {}},
            {"split": split, "scene_key": scene_key, "status": "missing_prototypes"},
            None,
        )
    if args.overwrite and args.save_vis:
        shutil.rmtree(scene_dir / "remap_visualizations", ignore_errors=True)

    compact_chunks: dict[str, Any] = {}
    detail_chunks: dict[str, Any] = {}
    all_tensors: dict[str, dict[int, torch.Tensor]] = {}
    merged_groups = []
    total_clips = 0
    total_primary = 0
    total_merged_extra = 0

    for chunk, infos in sorted(chunks.items()):
        compact_chunks[str(chunk)] = {}
        detail_chunks[str(chunk)] = {}
        for info in sorted(infos, key=lambda item: (item.camera_person, item.name)):
            compact, detail, tensors = remap_clip(info, prototypes, scene_dir, args)
            compact_chunks[str(chunk)][info.name] = compact
            detail_chunks[str(chunk)][info.name] = detail
            all_tensors[info.name] = tensors
            total_clips += 1
            for person_id, person in compact["people"].items():
                total_primary += 1
                extra = max(0, len(person["merged_segment_ids"]) - 1)
                total_merged_extra += extra
                if extra > 0:
                    merged_groups.append(
                        {
                            "chunk": int(chunk),
                            "clip_name": info.name,
                            "camera_person": int(info.camera_person),
                            "person_id": int(person_id),
                            "primary_segment_id": int(person["primary_segment_id"]),
                            "merged_segment_ids": person["merged_segment_ids"],
                            "primary_similarity": person["primary_similarity"],
                        }
                    )

    summary = {
        "split": split,
        "scene_key": scene_key,
        "status": "ok",
        "num_chunks": len(chunks),
        "num_clips": total_clips,
        "num_primary_assignments": total_primary,
        "num_merged_groups": len(merged_groups),
        "num_extra_merged_segments": total_merged_extra,
        "merge_similarity": float(args.merge_similarity),
        "merge_margin": float(args.merge_margin),
        "max_overlap_frames": int(args.max_overlap_frames),
        "min_embeddings": int(args.min_embeddings),
    }
    compact_scene = {
        "split": split,
        "scene_key": scene_key,
        "summary": summary,
        "merged_groups": merged_groups,
        "chunks": compact_chunks,
    }
    detail_scene = {
        **compact_scene,
        "prototype_cameras": {
            str(camera): sorted(int(person) for person in persons)
            for camera, persons in prototypes.items()
        },
        "chunks": detail_chunks,
    }
    return compact_scene, detail_scene, all_tensors


def discover_selected_scenes(args: argparse.Namespace) -> dict[str, dict[int, list[ClipInfo]]]:
    chunks = discover_scene_chunks(args)
    out = {}
    for full_scene_key in chunks:
        split, scene_key = full_scene_key.split("/", 1)
        scene_dir = Path(args.data_root) / split / "person_face_mapping" / scene_key
        if (scene_dir / "mapping.json").exists() and (scene_dir / "representative_embeddings.pt").exists():
            out[full_scene_key] = chunks[full_scene_key]
    return out


def build_html_report(all_summaries: dict[str, Any]) -> str:
    rows = []
    for key, summary in sorted(all_summaries.items()):
        rows.append(
            "<tr>"
            f"<td>{key}</td>"
            f"<td>{summary.get('num_chunks', 0)}</td>"
            f"<td>{summary.get('num_clips', 0)}</td>"
            f"<td>{summary.get('num_primary_assignments', 0)}</td>"
            f"<td>{summary.get('num_merged_groups', 0)}</td>"
            f"<td>{summary.get('num_extra_merged_segments', 0)}</td>"
            f"<td>{summary.get('status')}</td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>EgoCom All-Chunk Remap Report</title>
  <style>
    body {{ font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 28px; color: #17212f; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border: 1px solid #cfd6df; padding: 8px 9px; text-align: left; }}
    th {{ background: #f5f7fa; }}
  </style>
</head>
<body>
  <h1>EgoCom All-Chunk Remap Report</h1>
  <p>Tracks are grouped as the same person by indication only; original masks are not modified.</p>
  <table>
    <thead><tr><th>Scene</th><th>Chunks</th><th>Clips</th><th>Primary Assignments</th><th>Merged Groups</th><th>Extra Merged Segments</th><th>Status</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</body>
</html>
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Remap all chunks with selected representative face prototypes.")
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split", type=str, default="all_existing")
    parser.add_argument("--scene_key", type=str, default=None)
    parser.add_argument("--min_embeddings", type=positive_int, default=10)
    parser.add_argument("--merge_similarity", type=positive_float, default=0.70)
    parser.add_argument("--merge_margin", type=positive_float, default=0.10)
    parser.add_argument("--max_overlap_frames", type=positive_int, default=5)
    parser.add_argument("--crop_padding", type=positive_float, default=0.08)
    parser.add_argument("--save_vis", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    scenes = discover_selected_scenes(args)
    if not scenes:
        print("No scenes with selected mapping prototypes found.")
        return

    all_summaries: dict[str, Any] = {}
    for full_scene_key, chunks in scenes.items():
        split, scene_key = full_scene_key.split("/", 1)
        scene_dir = Path(args.data_root) / split / "person_face_mapping" / scene_key
        compact, detail, tensors = process_scene(full_scene_key, chunks, args)
        write_json(scene_dir / "remap_all_chunks.json", compact)
        write_json(scene_dir / "remap_all_chunks_details.json", detail)
        if tensors is not None:
            torch.save(tensors, scene_dir / "remap_representative_embeddings.pt")
        all_summaries[full_scene_key] = compact["summary"]
        print(
            f"[OK] {full_scene_key}: chunks={compact['summary'].get('num_chunks', 0)} "
            f"clips={compact['summary'].get('num_clips', 0)} "
            f"merged_groups={compact['summary'].get('num_merged_groups', 0)}"
        )

    by_split: dict[str, dict[str, Any]] = defaultdict(dict)
    for full_scene_key, summary in all_summaries.items():
        split, scene_key = full_scene_key.split("/", 1)
        by_split[split][scene_key] = summary
    for split, summaries in sorted(by_split.items()):
        split_dir = Path(args.data_root) / split / "person_face_mapping"
        write_json(split_dir / "remap_all_chunks_summary.json", summaries)
    root = Path(args.data_root)
    write_json(root / "person_face_remap_all_chunks_summary.json", all_summaries)
    (root / "person_face_remap_all_chunks_report.html").write_text(build_html_report(all_summaries))

    total_merged = sum(int(item.get("num_merged_groups", 0)) for item in all_summaries.values())
    total_extra = sum(int(item.get("num_extra_merged_segments", 0)) for item in all_summaries.values())
    print(
        "Done. "
        f"scenes={len(all_summaries)} "
        f"merged_groups={total_merged} "
        f"extra_merged_segments={total_extra}"
    )


if __name__ == "__main__":
    main()
