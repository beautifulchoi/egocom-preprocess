#!/usr/bin/env python3
"""
Save final frame-aligned binary masks for EgoCom remapped person tracks.

Inputs:
  /home/prj/data/egocom_holdout/1min/{split}/person_face_mapping/*/remap_all_chunks.json
  /home/prj/data/egocom_holdout/1min/{split}/refined_mask/{video}/mask.pt
  /home/prj/data/egocom_holdout/1min/{split}/frame/{video}/*.jpg

Outputs:
  /home/prj/data/egocom_holdout/1min/{split}/final_mask/{scene}/chunk_XXXX/{video}/person_{id}/*.jpg
  /home/prj/data/egocom_holdout/1min/{split}/final_mask/{scene}/chunk_XXXX/{video}/summary.json
  /home/prj/data/egocom_holdout/1min/{split}/final_mask/summary.json
"""

from __future__ import annotations

import argparse
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

from step_11_extract_person_visual_clip import (
    CLIP_RE,
    DEFAULT_DATA_ROOT,
    Assignment,
    collect_assignments_for_split,
    default_mapping_root,
    infer_image_shape,
    list_frame_files,
    load_mask_dict,
    nonnegative_int,
    positive_int,
    resize_mask,
    split_names,
    union_segment_mask,
    write_json,
)


@dataclass(frozen=True)
class MaskExportJob:
    split: str
    scene_key: str
    video_name: str
    camera_person: int
    chunk_index: int
    people_to_segments: dict[int, tuple[int, ...]]
    mapping_path: Path


def bounded_jpg_quality(value: str) -> int:
    parsed = int(value)
    if parsed < 1 or parsed > 100:
        raise argparse.ArgumentTypeError("value must be in [1, 100]")
    return parsed


def default_output_root(data_root: Path, split: str) -> Path:
    return data_root / split / "final_mask"


def output_dir_for(output_root: Path, job: MaskExportJob) -> Path:
    return (
        output_root
        / job.scene_key
        / f"chunk_{job.chunk_index:04d}"
        / job.video_name
    )


def parse_chunk_index(video_name: str) -> tuple[int, int]:
    match = CLIP_RE.match(video_name)
    if match is None:
        return -1, -1
    return int(match.group("camera")), int(match.group("chunk"))


def collect_jobs_for_split(
    split: str,
    data_root: Path,
    scene_key_filter: str | None,
    video_filter: str | None,
) -> list[MaskExportJob]:
    assignments = collect_assignments_for_split(
        split=split,
        mapping_root=default_mapping_root(data_root, split),
        scene_key_filter=scene_key_filter,
        video_filter=video_filter,
    )
    grouped: dict[tuple[str, str, str], dict[int, set[int]]] = defaultdict(
        lambda: defaultdict(set)
    )
    mapping_sources: dict[tuple[str, str, str], Path] = {}
    camera_persons: dict[tuple[str, str, str], int] = {}

    for assignment in assignments:
        key = (assignment.split, assignment.scene_key, assignment.video_name)
        grouped[key][assignment.person_id].update(assignment.segment_ids)
        mapping_sources.setdefault(key, assignment.mapping_path)
        camera_persons.setdefault(key, assignment.camera_person)

    jobs = []
    for key, people in sorted(grouped.items()):
        split_name, scene_key, video_name = key
        parsed_camera, chunk_index = parse_chunk_index(video_name)
        camera_person = camera_persons.get(key, parsed_camera)
        jobs.append(
            MaskExportJob(
                split=split_name,
                scene_key=scene_key,
                video_name=video_name,
                camera_person=int(camera_person),
                chunk_index=int(chunk_index),
                people_to_segments={
                    int(person_id): tuple(sorted(int(value) for value in segment_ids))
                    for person_id, segment_ids in sorted(people.items())
                    if segment_ids
                },
                mapping_path=mapping_sources[key],
            )
        )
    return jobs


def read_frame_shapes(
    frame_paths: list[Path],
    fallback_shape: tuple[int, int],
) -> tuple[list[tuple[int, int]], int]:
    shapes = []
    unreadable = 0
    for frame_path in frame_paths:
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            shapes.append(fallback_shape)
            unreadable += 1
        else:
            shapes.append((int(frame.shape[0]), int(frame.shape[1])))
    return shapes, unreadable


def mask_to_jpg_array(
    mask: np.ndarray | None,
    target_shape: tuple[int, int],
) -> tuple[np.ndarray, int, str]:
    if mask is None:
        return np.zeros(target_shape, dtype=np.uint8), 0, "absent_segment"

    mask_bool = resize_mask(mask, target_shape)
    pixel_count = int(mask_bool.sum())
    if pixel_count == 0:
        return np.zeros(target_shape, dtype=np.uint8), 0, "empty_mask_after_resize"
    return (mask_bool.astype(np.uint8) * 255), pixel_count, "masked"


def save_gray_jpg(path: Path, image: np.ndarray, jpg_quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(
        str(path),
        image,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)],
    )
    if not ok:
        raise RuntimeError(f"Failed to write mask JPG: {path}")


def export_job(
    job: MaskExportJob,
    split_root: Path,
    output_root: Path,
    overwrite: bool,
    strict: bool,
    jpg_quality: int,
) -> dict[str, Any]:
    frame_dir = split_root / "frame" / job.video_name
    mask_path = split_root / "refined_mask" / job.video_name / "mask.pt"
    view_output_dir = output_dir_for(output_root, job)
    summary_path = view_output_dir / "summary.json"

    if view_output_dir.exists() and not overwrite:
        return {
            "split": job.split,
            "scene_key": job.scene_key,
            "video_name": job.video_name,
            "chunk_index": job.chunk_index,
            "status": "skipped_existing",
            "output_dir": str(view_output_dir),
        }
    if view_output_dir.exists() and overwrite:
        shutil.rmtree(view_output_dir)

    frame_paths = list_frame_files(frame_dir)
    if not frame_paths:
        message = f"No frames found: {frame_dir}"
        if strict:
            raise FileNotFoundError(message)
        return {
            "split": job.split,
            "scene_key": job.scene_key,
            "video_name": job.video_name,
            "chunk_index": job.chunk_index,
            "status": "missing_frames",
            "message": message,
        }
    if not mask_path.exists():
        message = f"Missing mask: {mask_path}"
        if strict:
            raise FileNotFoundError(message)
        return {
            "split": job.split,
            "scene_key": job.scene_key,
            "video_name": job.video_name,
            "chunk_index": job.chunk_index,
            "status": "missing_mask",
            "message": message,
        }

    mask_dict = load_mask_dict(mask_path)
    fallback_shape = infer_image_shape(frame_paths, mask_dict)
    frame_shapes, unreadable_frames = read_frame_shapes(frame_paths, fallback_shape)

    people_summaries: dict[str, Any] = {}
    totals = Counter()
    for person_id, segment_ids in sorted(job.people_to_segments.items()):
        person_dir = view_output_dir / f"person_{person_id}"
        person_counts = Counter()
        for frame_idx, frame_path in enumerate(frame_paths):
            mask = union_segment_mask(mask_dict.get(frame_idx), segment_ids)
            image, pixel_count, status = mask_to_jpg_array(
                mask=mask,
                target_shape=frame_shapes[frame_idx],
            )
            save_gray_jpg(person_dir / frame_path.name, image, jpg_quality)
            person_counts[status] += 1
            if pixel_count > 0:
                person_counts["nonempty"] += 1
            person_counts["saved"] += 1
            person_counts["mask_pixels"] += int(pixel_count)

        person_summary = {
            "person_id": int(person_id),
            "segment_ids": [int(value) for value in segment_ids],
            "output_dir": str(person_dir),
            "num_saved_masks": int(person_counts["saved"]),
            "num_nonempty_masks": int(person_counts["nonempty"]),
            "num_black_masks": int(
                person_counts["absent_segment"]
                + person_counts["empty_mask_after_resize"]
            ),
            "num_absent_segment_frames": int(person_counts["absent_segment"]),
            "num_empty_after_resize": int(person_counts["empty_mask_after_resize"]),
            "total_mask_pixels": int(person_counts["mask_pixels"]),
        }
        people_summaries[f"person_{person_id}"] = person_summary
        totals.update(person_counts)

    summary = {
        "split": job.split,
        "scene_key": job.scene_key,
        "video_name": job.video_name,
        "camera_person": int(job.camera_person),
        "chunk_index": int(job.chunk_index),
        "status": "processed",
        "mapping_path": str(job.mapping_path),
        "frame_dir": str(frame_dir),
        "mask_path": str(mask_path),
        "output_dir": str(view_output_dir),
        "jpg_quality": int(jpg_quality),
        "num_frame_files": int(len(frame_paths)),
        "num_mask_frames": int(len(mask_dict)),
        "num_people": int(len(job.people_to_segments)),
        "num_unreadable_frames": int(unreadable_frames),
        "num_saved_masks": int(totals["saved"]),
        "num_nonempty_masks": int(totals["nonempty"]),
        "num_black_masks": int(
            totals["absent_segment"] + totals["empty_mask_after_resize"]
        ),
        "people": people_summaries,
    }
    write_json(summary_path, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save final per-person binary mask JPGs from remapped refined masks."
    )
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--split",
        default="all_existing",
        help="Split name, comma-separated split names, or all_existing.",
    )
    parser.add_argument("--scene_key", default=None)
    parser.add_argument("--video", default=None, help="Optional exact view clip name.")
    parser.add_argument(
        "--limit",
        type=nonnegative_int,
        default=0,
        help="Maximum number of view clips to process after filtering; 0 means all.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--jpg_quality", type=bounded_jpg_quality, default=100)
    parser.add_argument(
        "--min_mapped_people",
        type=positive_int,
        default=1,
        help="Skip view clips with fewer mapped target people.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    split_summaries = []
    overall_counts = Counter()

    for split in split_names(data_root, args.split):
        split_root = data_root / split
        output_root = default_output_root(data_root, split)
        jobs = [
            job
            for job in collect_jobs_for_split(
                split=split,
                data_root=data_root,
                scene_key_filter=args.scene_key,
                video_filter=args.video,
            )
            if len(job.people_to_segments) >= args.min_mapped_people
        ]
        if args.limit:
            jobs = jobs[: args.limit]

        split_counts = Counter()
        view_summaries = []
        for job in tqdm(jobs, desc=f"{split}: final person masks"):
            summary = export_job(
                job=job,
                split_root=split_root,
                output_root=output_root,
                overwrite=args.overwrite,
                strict=args.strict,
                jpg_quality=args.jpg_quality,
            )
            view_summaries.append(summary)
            split_counts[summary.get("status", "unknown")] += 1
            if summary.get("status") == "processed":
                split_counts["num_saved_masks"] += int(summary["num_saved_masks"])
                split_counts["num_nonempty_masks"] += int(summary["num_nonempty_masks"])
                split_counts["num_black_masks"] += int(summary["num_black_masks"])

        split_summary = {
            "split": split,
            "data_root": str(data_root),
            "output_root": str(output_root),
            "num_jobs": int(len(jobs)),
            "counts": {key: int(value) for key, value in sorted(split_counts.items())},
            "views": view_summaries,
        }
        write_json(output_root / "summary.json", split_summary)
        split_summaries.append(split_summary)
        overall_counts.update(split_counts)

    print(
        {
            "splits": [summary["split"] for summary in split_summaries],
            "counts": {key: int(value) for key, value in sorted(overall_counts.items())},
        }
    )


if __name__ == "__main__":
    main()
