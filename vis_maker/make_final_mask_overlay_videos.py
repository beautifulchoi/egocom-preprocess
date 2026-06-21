#!/usr/bin/env python3
"""
Create debug videos by overlaying final per-person masks on original frames.

Inputs:
  /home/prj/data/egocom_holdout/1min/{split}/final_mask/{scene}/chunk_XXXX/{view_clip}/person_*/*.jpg
  /home/prj/data/egocom_holdout/1min/{split}/frame/{view_clip}/*.jpg

Outputs:
  /home/prj/data/egocom_holdout/1min/{split}/final_mask_overlay_video/{scene}/chunk_XXXX/{view_clip}.mp4
  /home/prj/data/egocom_holdout/1min/{split}/final_mask_overlay_video/summary.json
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm


DEFAULT_DATA_ROOT = "/home/prj/data/egocom_holdout/1min"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
CHUNK_RE = re.compile(r"^chunk_(?P<chunk>\d+)$")
PERSON_RE = re.compile(r"^person_(?P<person>\d+)$")

PERSON_COLORS_BGR = {
    1: (64, 96, 255),
    2: (64, 220, 64),
    3: (255, 128, 64),
    4: (220, 80, 220),
    5: (64, 220, 220),
    6: (180, 180, 80),
}
FALLBACK_COLORS_BGR = [
    (255, 64, 64),
    (64, 255, 255),
    (255, 64, 255),
    (160, 220, 80),
    (80, 160, 220),
]


@dataclass(frozen=True)
class OverlayJob:
    split: str
    scene_key: str
    chunk_name: str
    chunk_index: int
    view_clip: str
    mask_view_dir: Path
    frame_dir: Path
    output_path: Path
    person_dirs: tuple[Path, ...]


def parse_comma_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def unit_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0 or parsed > 1.0:
        raise argparse.ArgumentTypeError("value must be in [0, 1]")
    return parsed


def load_json(path: Path) -> Any:
    with path.open("r") as file:
        return json.load(file)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        json.dump(payload, file, indent=2)


def h264_temp_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}__opencv_tmp.mp4")


def transcode_to_h264(input_path: Path, output_path: Path, crf: int) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is not available for H.264 transcoding")
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-crf",
        str(crf),
        str(output_path),
    ]
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg H.264 transcode failed for {input_path}: {result.stderr.strip()}"
        )


def split_names(data_root: Path, split_arg: str) -> list[str]:
    if split_arg != "all_existing":
        return parse_comma_list(split_arg)
    return sorted(
        split_dir.name
        for split_dir in data_root.iterdir()
        if split_dir.is_dir()
        and (split_dir / "final_mask").is_dir()
        and (split_dir / "frame").is_dir()
    )


def chunk_index_from_name(chunk_name: str) -> int:
    match = CHUNK_RE.match(chunk_name)
    if match is None:
        return -1
    return int(match.group("chunk"))


def person_id_from_dir(person_dir: Path) -> int:
    match = PERSON_RE.match(person_dir.name)
    if match is None:
        return 0
    return int(match.group("person"))


def person_color(person_id: int) -> tuple[int, int, int]:
    if person_id in PERSON_COLORS_BGR:
        return PERSON_COLORS_BGR[person_id]
    return FALLBACK_COLORS_BGR[person_id % len(FALLBACK_COLORS_BGR)]


def list_frame_files(frame_dir: Path) -> list[Path]:
    if not frame_dir.is_dir():
        return []
    return sorted(
        path
        for path in frame_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def collect_jobs_for_split(
    data_root: Path,
    split: str,
    scene_key_filter: str | None,
    chunk_filter: int | None,
    video_filter: str | None,
) -> list[OverlayJob]:
    split_root = data_root / split
    final_mask_root = split_root / "final_mask"
    frame_root = split_root / "frame"
    output_root = split_root / "final_mask_overlay_video"
    if not final_mask_root.is_dir():
        return []

    jobs = []
    for scene_dir in sorted(path for path in final_mask_root.iterdir() if path.is_dir()):
        if scene_key_filter and scene_dir.name != scene_key_filter:
            continue
        for chunk_dir in sorted(path for path in scene_dir.iterdir() if path.is_dir()):
            chunk_index = chunk_index_from_name(chunk_dir.name)
            if chunk_filter is not None and chunk_index != chunk_filter:
                continue
            for view_dir in sorted(path for path in chunk_dir.iterdir() if path.is_dir()):
                if video_filter and view_dir.name != video_filter:
                    continue
                person_dirs = tuple(
                    sorted(
                        (
                            path
                            for path in view_dir.iterdir()
                            if path.is_dir() and PERSON_RE.match(path.name)
                        ),
                        key=person_id_from_dir,
                    )
                )
                if not person_dirs:
                    continue
                jobs.append(
                    OverlayJob(
                        split=split,
                        scene_key=scene_dir.name,
                        chunk_name=chunk_dir.name,
                        chunk_index=chunk_index,
                        view_clip=view_dir.name,
                        mask_view_dir=view_dir,
                        frame_dir=frame_root / view_dir.name,
                        output_path=(
                            output_root
                            / scene_dir.name
                            / chunk_dir.name
                            / f"{view_dir.name}.mp4"
                        ),
                        person_dirs=person_dirs,
                    )
                )
    return jobs


def load_mask(mask_path: Path, target_shape: tuple[int, int]) -> np.ndarray | None:
    if not mask_path.exists():
        return None
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    if mask.shape[:2] != target_shape:
        mask = cv2.resize(
            mask,
            (target_shape[1], target_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    return mask > 127


def draw_legend(
    frame: np.ndarray,
    people: list[tuple[int, tuple[int, int, int]]],
) -> None:
    if not people:
        return
    line_h = 22
    pad = 8
    box_w = 142
    box_h = pad * 2 + line_h * len(people)
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (8 + box_w, 8 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, dst=frame)
    for row, (person_id, color) in enumerate(people):
        y = 8 + pad + row * line_h
        cv2.rectangle(frame, (18, y + 3), (34, y + 17), color, -1)
        cv2.putText(
            frame,
            f"person_{person_id}",
            (42, y + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def overlay_frame(
    frame_bgr: np.ndarray,
    masks_by_person: list[tuple[int, np.ndarray]],
    background_alpha: float,
    foreground_alpha: float,
) -> np.ndarray:
    frame_float = frame_bgr.astype(np.float32)
    foreground_union = np.zeros(frame_bgr.shape[:2], dtype=bool)
    color_accum = np.zeros_like(frame_float)
    color_counts = np.zeros(frame_bgr.shape[:2], dtype=np.float32)
    people_for_legend = []

    for person_id, mask in masks_by_person:
        if mask is None or not mask.any():
            people_for_legend.append((person_id, person_color(person_id)))
            continue
        color = np.array(person_color(person_id), dtype=np.float32)
        foreground_union |= mask
        color_accum[mask] += color
        color_counts[mask] += 1.0
        people_for_legend.append((person_id, person_color(person_id)))

    output = (frame_float * float(background_alpha)).astype(np.float32)
    if foreground_union.any():
        counts = np.clip(color_counts[..., None], 1.0, None)
        color_layer = color_accum / counts
        output[foreground_union] = np.clip(
            frame_float[foreground_union]
            + color_layer[foreground_union] * foreground_alpha,
            0,
            255,
        )

    output_uint8 = np.clip(output, 0, 255).astype(np.uint8)
    for person_id, mask in masks_by_person:
        if mask is None or not mask.any():
            continue
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(output_uint8, contours, -1, person_color(person_id), 2)
    draw_legend(output_uint8, people_for_legend)
    return output_uint8


def make_video(
    job: OverlayJob,
    fps: float,
    background_alpha: float,
    foreground_alpha: float,
    overwrite: bool,
    codec: str,
    h264: bool,
    h264_crf: int,
) -> dict[str, Any]:
    if job.output_path.exists() and not overwrite:
        return {
            "status": "skipped_existing",
            "split": job.split,
            "scene_key": job.scene_key,
            "chunk_name": job.chunk_name,
            "view_clip": job.view_clip,
            "output_path": str(job.output_path),
        }

    frame_paths = list_frame_files(job.frame_dir)
    if not frame_paths:
        return {
            "status": "missing_frames",
            "split": job.split,
            "scene_key": job.scene_key,
            "chunk_name": job.chunk_name,
            "view_clip": job.view_clip,
            "frame_dir": str(job.frame_dir),
        }

    first_frame = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
    if first_frame is None:
        return {
            "status": "unreadable_first_frame",
            "split": job.split,
            "scene_key": job.scene_key,
            "chunk_name": job.chunk_name,
            "view_clip": job.view_clip,
            "frame_path": str(frame_paths[0]),
        }

    height, width = first_frame.shape[:2]
    job.output_path.parent.mkdir(parents=True, exist_ok=True)
    use_h264 = bool(h264 and shutil.which("ffmpeg"))
    writer_path = h264_temp_path(job.output_path) if use_h264 else job.output_path
    if use_h264 and writer_path.exists():
        writer_path.unlink()

    writer = cv2.VideoWriter(
        str(writer_path),
        cv2.VideoWriter_fourcc(*codec),
        float(fps),
        (int(width), int(height)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {writer_path}")

    counts = Counter()
    person_ids = [person_id_from_dir(person_dir) for person_dir in job.person_dirs]
    try:
        for frame_path in frame_paths:
            frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame is None:
                counts["unreadable_frames"] += 1
                continue
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
                counts["resized_frames"] += 1

            masks_by_person = []
            for person_dir, person_id in zip(job.person_dirs, person_ids):
                mask = load_mask(person_dir / frame_path.name, (height, width))
                if mask is None:
                    counts["missing_or_unreadable_masks"] += 1
                    mask = np.zeros((height, width), dtype=bool)
                elif not mask.any():
                    counts["empty_masks"] += 1
                else:
                    counts["nonempty_masks"] += 1
                masks_by_person.append((person_id, mask))

            writer.write(
                overlay_frame(
                    frame_bgr=frame,
                    masks_by_person=masks_by_person,
                    background_alpha=background_alpha,
                    foreground_alpha=foreground_alpha,
                )
            )
            counts["written_frames"] += 1
    finally:
        writer.release()

    if use_h264:
        transcode_to_h264(writer_path, job.output_path, h264_crf)
        writer_path.unlink(missing_ok=True)
        output_codec = "libx264/yuv420p"
        h264_status = "transcoded"
    elif h264:
        output_codec = codec
        h264_status = "ffmpeg_missing_fallback"
    else:
        output_codec = codec
        h264_status = "disabled"

    return {
        "status": "processed",
        "split": job.split,
        "scene_key": job.scene_key,
        "chunk_name": job.chunk_name,
        "chunk_index": int(job.chunk_index),
        "view_clip": job.view_clip,
        "frame_dir": str(job.frame_dir),
        "mask_view_dir": str(job.mask_view_dir),
        "output_path": str(job.output_path),
        "codec": output_codec,
        "h264_status": h264_status,
        "fps": float(fps),
        "width": int(width),
        "height": int(height),
        "persons": [f"person_{person_id}" for person_id in person_ids],
        "num_frame_files": int(len(frame_paths)),
        "counts": {key: int(value) for key, value in sorted(counts.items())},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create per-view debug MP4s with final person masks overlaid on source frames."
    )
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--split",
        default="all_existing",
        help="Split name, comma-separated split names, or all_existing.",
    )
    parser.add_argument("--scene_key", default=None)
    parser.add_argument("--chunk", type=nonnegative_int, default=None)
    parser.add_argument("--video", default=None, help="Optional exact view clip name.")
    parser.add_argument(
        "--limit",
        type=nonnegative_int,
        default=0,
        help="Maximum number of view videos to render after filtering; 0 means all.",
    )
    parser.add_argument("--fps", type=positive_float, default=5.0)
    parser.add_argument("--background_alpha", type=unit_float, default=0.5)
    parser.add_argument("--foreground_alpha", type=unit_float, default=0.5)
    parser.add_argument("--codec", default="mp4v", help="OpenCV temporary FourCC codec, default mp4v.")
    parser.add_argument(
        "--no_h264",
        action="store_true",
        help="Keep the raw OpenCV MP4 instead of transcoding to H.264/yuv420p with ffmpeg.",
    )
    parser.add_argument("--h264_crf", type=nonnegative_int, default=18)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    overall_counts = Counter()
    split_summaries = []

    for split in split_names(data_root, args.split):
        jobs = collect_jobs_for_split(
            data_root=data_root,
            split=split,
            scene_key_filter=args.scene_key,
            chunk_filter=args.chunk,
            video_filter=args.video,
        )
        if args.limit:
            jobs = jobs[: args.limit]

        split_counts = Counter()
        videos = []
        for job in tqdm(jobs, desc=f"{split}: final-mask overlays"):
            summary = make_video(
                job=job,
                fps=args.fps,
                background_alpha=args.background_alpha,
                foreground_alpha=args.foreground_alpha,
                overwrite=args.overwrite,
                codec=args.codec,
                h264=not args.no_h264,
                h264_crf=args.h264_crf,
            )
            videos.append(summary)
            split_counts[summary.get("status", "unknown")] += 1

        split_summary = {
            "split": split,
            "data_root": str(data_root),
            "output_root": str(data_root / split / "final_mask_overlay_video"),
            "num_jobs": int(len(jobs)),
            "counts": {key: int(value) for key, value in sorted(split_counts.items())},
            "videos": videos,
        }
        write_json(data_root / split / "final_mask_overlay_video" / "summary.json", split_summary)
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
