#!/usr/bin/env python3
"""
Build windowed EgoCom multimodal pair manifests from 1-minute chunk outputs.

Inputs:
  {data_root}/{chunk_root}/{split}/person_depth_lift/{scene}/person_{id}/*.npz
  {data_root}/{chunk_root}/{split}/person_visual_clip_features/{scene}/person_{id}/*.pt
  {data_root}/{chunk_root}/{split}/person_spatial_t5_features/{scene}/person_{id}/*.pt
  {data_root}/original/{split}/audio/*.wav
  {data_root}/original/{split}/video/*.MP4

Outputs:
  {data_root}/{output_tag}/{split}/audio
  {data_root}/{output_tag}/{split}/video
  {data_root}/{output_tag}/{split}/depth_xy_ray
  {data_root}/{output_tag}/{split}/clip_features
  {data_root}/{output_tag}/{split}/t5_text_features
  {data_root}/{output_tag}/{split}/manifest/manifest_mm.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch


DEFAULT_DATA_ROOT = Path("/home/prj/data/egocom_holdout")
CHUNK_RE = re.compile(r"^(?P<base>.+)_chunk_(?P<chunk>\d+)$")
CAMERA_RE = re.compile(r"__person_(?P<camera>\d+)(?:_|$)")
VIDEO_SUFFIXES = (".MP4", ".mp4", ".MOV", ".mov", ".M4V", ".m4v")


@dataclass(frozen=True)
class ChunkPath:
    chunk_index: int
    path: Path


@dataclass(frozen=True)
class ChunkInterval:
    chunk_index: int
    start_frame: int
    end_frame: int


@dataclass
class SplitIndex:
    depth: dict[tuple[str, int, str], list[ChunkPath]]
    clip: dict[tuple[str, int, str], list[ChunkPath]]
    text: dict[tuple[str, int, str], list[ChunkPath]]
    camera_to_base: dict[str, dict[int, str]]
    scenes: list[str]


@dataclass
class GeometryStream:
    num_frames: int
    chunk_intervals: list[ChunkInterval]
    frame_indices: np.ndarray
    frame_stems: np.ndarray
    status_label: np.ndarray
    valid_mask: np.ndarray
    x_ray: np.ndarray
    y_ray: np.ndarray
    d: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    region_pixel_count: np.ndarray
    source_chunk_index: np.ndarray


@dataclass
class ClipStream:
    num_frames: int
    chunk_intervals: list[ChunkInterval]
    features: torch.Tensor
    valid_mask: torch.Tensor
    frame_indices: np.ndarray
    frame_stems: np.ndarray
    frame_statuses: np.ndarray
    mask_pixel_counts: np.ndarray
    source_chunk_index: np.ndarray
    feature_dim: int
    model_id: str


@dataclass
class TextStream:
    num_frames: int
    chunk_intervals: list[ChunkInterval]
    features: torch.Tensor
    valid_mask: torch.Tensor
    frame_indices: np.ndarray
    frame_stems: np.ndarray
    frame_statuses: np.ndarray
    mask_pixel_counts: np.ndarray
    source_chunk_index: np.ndarray
    texts: np.ndarray
    encoded_texts: np.ndarray
    raw_null_mask: np.ndarray
    feature_dim: int
    text_model_id: str
    feature_model_id: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build windowed EgoCom manifests from chunked depth, CLIP, T5 text, and original audio.",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--chunk-root", default="1min")
    parser.add_argument("--original-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--output-tag", default=None)
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument(
        "--window-sec",
        type=float,
        default=None,
        help="Override split-specific window seconds. Defaults: train/val=6.0, test=4.0.",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=None,
        help="Override split-specific overlap. Defaults: train/val=0.5, test=0.0.",
    )
    parser.add_argument("--overlap-mode", choices=("ratio", "seconds"), default="ratio")
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--chunk-sec", type=float, default=60.0)
    parser.add_argument(
        "--max-audio-duration-sec",
        type=float,
        default=0.0,
        help="When positive, keep only pairs whose source and target original audio are shorter than this many seconds.",
    )
    parser.add_argument(
        "--audio-stem-allowlist",
        nargs="+",
        default=None,
        help=(
            "Optional original-audio allowlist. Values may be stems or .wav paths; "
            "when set, both source and target audio must be in this list."
        ),
    )
    parser.add_argument("--ignore-video-chunk-root", type=Path, default=None)
    parser.add_argument("--ignore-video-chunk-list", type=Path, default=None)
    parser.add_argument(
        "--write-video-windows",
        action="store_true",
        help="Also split original source/target videos into manifest-aligned MP4 windows.",
    )
    parser.add_argument(
        "--video-window-mode",
        choices=("accurate", "copy"),
        default="accurate",
        help="accurate re-encodes video for timestamp-accurate clips; copy is faster but keyframe-aligned.",
    )
    parser.add_argument(
        "--video-window-audio",
        choices=("window", "none"),
        default="window",
        help="When writing video windows, mux the matching generated WAV window into the MP4 unless set to none.",
    )
    parser.add_argument(
        "--test-min-geometry-valid-ratio",
        type=float,
        default=0.25,
        help="Minimum target geometry valid ratio for test windows. Set negative to disable.",
    )
    parser.add_argument("--missing-policy", choices=("skip", "strict"), default="skip")
    parser.add_argument("--scene-key", default=None)
    parser.add_argument("--limit-scenes", type=int, default=0)
    parser.add_argument("--limit-windows-per-pair", type=int, default=0)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_split_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def format_number(value: float) -> str:
    if math.isclose(value, round(value)):
        return str(int(round(value)))
    return f"{value:g}"


def make_output_tag(window_sec: float, overlap: float) -> str:
    return f"{format_number(window_sec)}s_overlap{format_number(overlap)}_v2"


def compute_stride_sec(window_sec: float, overlap: float, overlap_mode: str) -> float:
    if window_sec <= 0:
        raise ValueError("--window-sec must be positive")
    if overlap_mode == "ratio":
        if not (0.0 <= overlap < 1.0):
            raise ValueError("--overlap must be in [0, 1) when --overlap-mode ratio")
        return window_sec * (1.0 - overlap)
    if not (0.0 <= overlap < window_sec):
        raise ValueError("--overlap must be in [0, window_sec) when --overlap-mode seconds")
    return window_sec - overlap


def split_window_config(args: argparse.Namespace, split: str) -> tuple[float, float, float]:
    default_window_sec = 4.0 if split == "test" else 6.0
    default_overlap = 0.0 if split == "test" else 0.5
    window_sec = args.window_sec if args.window_sec is not None else default_window_sec
    overlap = args.overlap if args.overlap is not None else default_overlap
    stride_sec = compute_stride_sec(window_sec, overlap, args.overlap_mode)
    return window_sec, overlap, stride_sec


def parse_chunk_stem(stem: str) -> tuple[str, int] | None:
    match = CHUNK_RE.match(stem)
    if match is None:
        return None
    return match.group("base"), int(match.group("chunk"))


def chunk_name(base_video: str, chunk_index: int) -> str:
    return f"{base_video}_chunk_{chunk_index:04d}"


def camera_person_from_base(base_video: str) -> int | None:
    match = CAMERA_RE.search(base_video)
    if match is None:
        return None
    return int(match.group("camera"))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Any:
    with path.open("r") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def load_ignore_video_chunks(args: argparse.Namespace, split: str) -> tuple[set[str], str | None]:
    if args.ignore_video_chunk_list is not None:
        list_path = args.ignore_video_chunk_list
    else:
        root = args.ignore_video_chunk_root or (args.data_root / "ignore_video_chunks")
        list_path = root / f"{split}.txt"

    if not list_path.is_file():
        return set(), str(list_path)

    ignored: set[str] = set()
    with list_path.open("r") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            ignored.add(Path(line).name)
    return ignored, str(list_path)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


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


def conflicted_scene_chunks(mapping_root: Path) -> dict[str, set[int]]:
    if not mapping_root.is_dir():
        return {}
    out: dict[str, set[int]] = {}
    for summary_path in sorted(mapping_root.glob("*/summary.json")):
        try:
            data = load_json(summary_path)
        except Exception:
            continue
        scene = summary_path.parent.name
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
            out[scene] = chunks
    return out


def is_conflicted_chunk(conflicted: dict[str, set[int]], scene: str, chunk_index: int) -> bool:
    return chunk_index in conflicted.get(scene, set())


def discover_split_index(
    split_root: Path,
    split: str,
    conflicted: dict[str, set[int]] | None = None,
) -> SplitIndex | None:
    depth_root = split_root / "person_depth_lift"
    clip_root = split_root / "person_visual_clip_features"
    text_root = split_root / "person_spatial_t5_features"
    if not depth_root.is_dir() or not clip_root.is_dir() or not text_root.is_dir():
        return None
    conflicted = conflicted or {}

    depth: dict[tuple[str, int, str], list[ChunkPath]] = defaultdict(list)
    clip: dict[tuple[str, int, str], list[ChunkPath]] = defaultdict(list)
    text: dict[tuple[str, int, str], list[ChunkPath]] = defaultdict(list)
    camera_to_base: dict[str, dict[int, str]] = defaultdict(dict)

    def add_camera(scene: str, base_video: str) -> None:
        camera_person = camera_person_from_base(base_video)
        if camera_person is not None:
            prev = camera_to_base[scene].get(camera_person)
            if prev is not None and prev != base_video:
                raise ValueError(
                    f"Conflicting camera video for {split} {scene} person_{camera_person}: "
                    f"{prev} vs {base_video}"
                )
            camera_to_base[scene][camera_person] = base_video

    for person_dir in sorted(depth_root.glob("*/person_*")):
        if not person_dir.is_dir():
            continue
        scene = person_dir.parent.name
        target_person = int(person_dir.name.split("_", 1)[1])
        for npz_path in sorted(person_dir.glob("*.npz")):
            parsed = parse_chunk_stem(npz_path.stem)
            if parsed is None:
                continue
            base_video, chunk_index = parsed
            if is_conflicted_chunk(conflicted, scene, chunk_index):
                continue
            depth[(scene, target_person, base_video)].append(ChunkPath(chunk_index, npz_path))
            add_camera(scene, base_video)

    for person_dir in sorted(clip_root.glob("*/person_*")):
        if not person_dir.is_dir():
            continue
        scene = person_dir.parent.name
        target_person = int(person_dir.name.split("_", 1)[1])
        for pt_path in sorted(person_dir.glob("*.pt")):
            parsed = parse_chunk_stem(pt_path.stem)
            if parsed is None:
                continue
            base_video, chunk_index = parsed
            if is_conflicted_chunk(conflicted, scene, chunk_index):
                continue
            clip[(scene, target_person, base_video)].append(ChunkPath(chunk_index, pt_path))
            add_camera(scene, base_video)

    for person_dir in sorted(text_root.glob("*/person_*")):
        if not person_dir.is_dir():
            continue
        scene = person_dir.parent.name
        target_person = int(person_dir.name.split("_", 1)[1])
        for pt_path in sorted(person_dir.glob("*.pt")):
            parsed = parse_chunk_stem(pt_path.stem)
            if parsed is None:
                continue
            base_video, chunk_index = parsed
            if is_conflicted_chunk(conflicted, scene, chunk_index):
                continue
            text[(scene, target_person, base_video)].append(ChunkPath(chunk_index, pt_path))
            add_camera(scene, base_video)

    scenes = sorted(
        set(camera_to_base)
        | {key[0] for key in depth}
        | {key[0] for key in clip}
        | {key[0] for key in text}
    )
    for paths in depth.values():
        paths.sort(key=lambda item: item.chunk_index)
    for paths in clip.values():
        paths.sort(key=lambda item: item.chunk_index)
    for paths in text.values():
        paths.sort(key=lambda item: item.chunk_index)
    return SplitIndex(
        depth=dict(depth),
        clip=dict(clip),
        text=dict(text),
        camera_to_base=dict(camera_to_base),
        scenes=scenes,
    )


def chunk_start_frame(chunk_index: int, chunk_sec: float, fps: float) -> int:
    return int(round((chunk_index - 1) * chunk_sec * fps))


def chunk_frame_count(chunk_sec: float, fps: float) -> int:
    return int(round(chunk_sec * fps))


def build_chunk_intervals_from_lengths(
    chunk_lengths: list[tuple[int, int]],
    chunk_sec: float,
    fps: float,
) -> list[ChunkInterval]:
    intervals: list[ChunkInterval] = []
    previous_index: int | None = None
    previous_end = 0
    for chunk_index, length in sorted(chunk_lengths):
        if length < 0:
            raise ValueError(f"Negative frame length for chunk {chunk_index}: {length}")
        if previous_index is not None and chunk_index == previous_index + 1:
            start = previous_end
        else:
            start = chunk_start_frame(chunk_index, chunk_sec, fps)
        end = start + length
        intervals.append(ChunkInterval(chunk_index=chunk_index, start_frame=start, end_frame=end))
        previous_index = chunk_index
        previous_end = end
    return intervals


def list_frame_files(frame_dir: Path) -> list[Path]:
    if not frame_dir.is_dir():
        return []
    return sorted(
        path
        for path in frame_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


def video_chunk_intervals_from_frames(
    split_root: Path,
    base_video: str,
    chunk_sec: float,
    fps: float,
) -> list[ChunkInterval]:
    frame_root = split_root / "frame"
    chunk_lengths: list[tuple[int, int]] = []
    for frame_dir in sorted(frame_root.glob(f"{base_video}_chunk_*")):
        parsed = parse_chunk_stem(frame_dir.name)
        if parsed is None:
            continue
        parsed_base, chunk_index = parsed
        if parsed_base != base_video:
            continue
        chunk_lengths.append((chunk_index, len(list_frame_files(frame_dir))))
    return build_chunk_intervals_from_lengths(chunk_lengths, chunk_sec, fps)


def loaded_chunk_intervals(
    chunk_lengths: list[tuple[ChunkPath, int]],
    chunk_sec: float,
    fps: float,
    base_chunk_intervals: list[ChunkInterval] | None = None,
) -> list[ChunkInterval]:
    if base_chunk_intervals:
        start_by_chunk = {interval.chunk_index: interval.start_frame for interval in base_chunk_intervals}
        intervals: list[ChunkInterval] = []
        for chunk, length in sorted(chunk_lengths, key=lambda item: item[0].chunk_index):
            start = start_by_chunk.get(
                chunk.chunk_index,
                chunk_start_frame(chunk.chunk_index, chunk_sec, fps),
            )
            intervals.append(
                ChunkInterval(
                    chunk_index=chunk.chunk_index,
                    start_frame=start,
                    end_frame=start + length,
                )
            )
        return intervals
    return build_chunk_intervals_from_lengths(
        [(chunk.chunk_index, length) for chunk, length in chunk_lengths],
        chunk_sec,
        fps,
    )


def sec_to_frame_index(sec: float, fps: float) -> int:
    return int(math.floor(sec * fps + 0.5))


def frame_to_ms(frame_idx: int, fps: float) -> int:
    return int(round((frame_idx / fps) * 1000.0))


def overlapping_chunk_names(
    base_video: str,
    start_idx: int,
    end_idx: int,
    chunk_intervals: list[ChunkInterval],
) -> list[str]:
    return [chunk_name(base_video, idx) for idx in overlapping_chunk_indices(start_idx, end_idx, chunk_intervals)]


def overlapping_chunk_indices(
    start_idx: int,
    end_idx: int,
    chunk_intervals: list[ChunkInterval],
) -> list[int]:
    if end_idx <= start_idx:
        return []
    return [
        interval.chunk_index
        for interval in chunk_intervals
        if start_idx < interval.end_frame and end_idx > interval.start_frame
    ]


def safe_array(data: dict[str, np.ndarray], key: str, length: int, fill: float = np.nan) -> np.ndarray:
    if key in data:
        arr = np.asarray(data[key])
        return arr[:length]
    return np.full((length,), fill, dtype=np.float32)


def merge_geometry(
    chunks: list[ChunkPath],
    fps: float,
    chunk_sec: float,
    base_chunk_intervals: list[ChunkInterval] | None = None,
) -> GeometryStream:
    loaded: list[tuple[ChunkPath, dict[str, np.ndarray], int]] = []
    chunk_lengths: list[tuple[ChunkPath, int]] = []
    for chunk in chunks:
        data = load_npz(chunk.path)
        length = int(np.asarray(data.get("num_frames", len(data["d_mean"]))).item())
        loaded.append((chunk, data, length))
        chunk_lengths.append((chunk, length))

    chunk_intervals = loaded_chunk_intervals(chunk_lengths, chunk_sec, fps, base_chunk_intervals)
    interval_by_chunk = {interval.chunk_index: interval for interval in chunk_intervals}
    max_end = max((interval.end_frame for interval in chunk_intervals), default=0)

    frame_indices = np.arange(max_end, dtype=np.int32)
    frame_stems = np.full((max_end,), "", dtype=object)
    status_label = np.full((max_end,), "missing_chunk", dtype=object)
    x_ray = np.full((max_end,), np.nan, dtype=np.float32)
    y_ray = np.full((max_end,), np.nan, dtype=np.float32)
    d = np.full((max_end,), np.nan, dtype=np.float32)
    x = np.full((max_end,), np.nan, dtype=np.float32)
    y = np.full((max_end,), np.nan, dtype=np.float32)
    z = np.full((max_end,), np.nan, dtype=np.float32)
    region_pixel_count = np.zeros((max_end,), dtype=np.int32)
    source_chunk_index = np.zeros((max_end,), dtype=np.int32)

    for chunk, data, length in loaded:
        start = interval_by_chunk[chunk.chunk_index].start_frame
        end = start + length
        frame_stems[start:end] = np.asarray(data.get("frame_stems", [""] * length), dtype=object)[:length]
        status_label[start:end] = np.asarray(data.get("status_label", ["unknown"] * length), dtype=object)[:length]
        x_ray[start:end] = safe_array(data, "x_ray_mean", length)
        y_ray[start:end] = safe_array(data, "y_ray_mean", length)
        d[start:end] = safe_array(data, "d_mean", length)
        x[start:end] = safe_array(data, "x_mean", length)
        y[start:end] = safe_array(data, "y_mean", length)
        z[start:end] = safe_array(data, "z_mean", length)
        region_pixel_count[start:end] = safe_array(data, "region_pixel_count", length, fill=0).astype(np.int32)
        source_chunk_index[start:end] = chunk.chunk_index

    valid_mask = np.isfinite(x_ray) & np.isfinite(y_ray) & np.isfinite(d)
    return GeometryStream(
        num_frames=max_end,
        chunk_intervals=chunk_intervals,
        frame_indices=frame_indices,
        frame_stems=frame_stems,
        status_label=status_label,
        valid_mask=valid_mask,
        x_ray=x_ray,
        y_ray=y_ray,
        d=d,
        x=x,
        y=y,
        z=z,
        region_pixel_count=region_pixel_count,
        source_chunk_index=source_chunk_index,
    )


def merge_clip(
    chunks: list[ChunkPath],
    fps: float,
    chunk_sec: float,
    base_chunk_intervals: list[ChunkInterval] | None = None,
) -> ClipStream:
    loaded: list[tuple[ChunkPath, dict[str, Any], int]] = []
    chunk_lengths: list[tuple[ChunkPath, int]] = []
    feature_dim = 0
    model_id = ""
    for chunk in chunks:
        data = load_torch(chunk.path)
        features = data["features"]
        length = int(features.shape[0])
        feature_dim = int(data.get("feature_dim", features.shape[1]))
        model_id = str(data.get("model_id", model_id))
        local_indices = data.get("frame_indices", list(range(length)))
        if local_indices:
            length_extent = max(int(idx) for idx in local_indices) + 1
        else:
            length_extent = length
        loaded.append((chunk, data, length_extent))
        chunk_lengths.append((chunk, length_extent))

    chunk_intervals = loaded_chunk_intervals(chunk_lengths, chunk_sec, fps, base_chunk_intervals)
    interval_by_chunk = {interval.chunk_index: interval for interval in chunk_intervals}
    max_end = max((interval.end_frame for interval in chunk_intervals), default=0)

    features_all = torch.zeros((max_end, feature_dim), dtype=torch.float32)
    valid_mask = torch.zeros((max_end,), dtype=torch.bool)
    frame_indices = np.arange(max_end, dtype=np.int32)
    frame_stems = np.full((max_end,), "", dtype=object)
    frame_statuses = np.full((max_end,), "missing_chunk", dtype=object)
    mask_pixel_counts = np.zeros((max_end,), dtype=np.int32)
    source_chunk_index = np.zeros((max_end,), dtype=np.int32)

    for chunk, data, _length_extent in loaded:
        start = interval_by_chunk[chunk.chunk_index].start_frame
        features = data["features"].detach().cpu().float()
        local_indices = [int(idx) for idx in data.get("frame_indices", range(features.shape[0]))]
        stems = list(data.get("frame_stems", [""] * features.shape[0]))
        statuses = list(data.get("frame_statuses", ["unknown"] * features.shape[0]))
        pixel_counts = list(data.get("mask_pixel_counts", [0] * features.shape[0]))
        for row_idx, local_idx in enumerate(local_indices):
            global_idx = start + local_idx
            if global_idx >= max_end:
                continue
            features_all[global_idx] = features[row_idx]
            valid_mask[global_idx] = True
            frame_stems[global_idx] = stems[row_idx] if row_idx < len(stems) else ""
            frame_statuses[global_idx] = statuses[row_idx] if row_idx < len(statuses) else "unknown"
            mask_pixel_counts[global_idx] = int(pixel_counts[row_idx]) if row_idx < len(pixel_counts) else 0
            source_chunk_index[global_idx] = chunk.chunk_index

    return ClipStream(
        num_frames=max_end,
        chunk_intervals=chunk_intervals,
        features=features_all,
        valid_mask=valid_mask,
        frame_indices=frame_indices,
        frame_stems=frame_stems,
        frame_statuses=frame_statuses,
        mask_pixel_counts=mask_pixel_counts,
        source_chunk_index=source_chunk_index,
        feature_dim=feature_dim,
        model_id=model_id,
    )


def merge_text(
    chunks: list[ChunkPath],
    fps: float,
    chunk_sec: float,
    base_chunk_intervals: list[ChunkInterval] | None = None,
) -> TextStream:
    loaded: list[tuple[ChunkPath, dict[str, Any], int]] = []
    chunk_lengths: list[tuple[ChunkPath, int]] = []
    feature_dim = 0
    text_model_id = ""
    feature_model_id = ""
    for chunk in chunks:
        data = load_torch(chunk.path)
        features = data["features"]
        length = int(features.shape[0])
        feature_dim = int(data.get("feature_dim", features.shape[1]))
        text_model_id = str(data.get("text_model_id", text_model_id))
        feature_model_id = str(data.get("feature_model_id", feature_model_id))
        local_indices = data.get("frame_indices", list(range(length)))
        if local_indices:
            length_extent = max(int(idx) for idx in local_indices) + 1
        else:
            length_extent = length
        loaded.append((chunk, data, length_extent))
        chunk_lengths.append((chunk, length_extent))

    chunk_intervals = loaded_chunk_intervals(chunk_lengths, chunk_sec, fps, base_chunk_intervals)
    interval_by_chunk = {interval.chunk_index: interval for interval in chunk_intervals}
    max_end = max((interval.end_frame for interval in chunk_intervals), default=0)

    features_all = torch.zeros((max_end, feature_dim), dtype=torch.float32)
    valid_mask = torch.zeros((max_end,), dtype=torch.bool)
    frame_indices = np.arange(max_end, dtype=np.int32)
    frame_stems = np.full((max_end,), "", dtype=object)
    frame_statuses = np.full((max_end,), "missing_chunk", dtype=object)
    mask_pixel_counts = np.zeros((max_end,), dtype=np.int32)
    source_chunk_index = np.zeros((max_end,), dtype=np.int32)
    texts = np.full((max_end,), "", dtype=object)
    encoded_texts = np.full((max_end,), "", dtype=object)
    raw_null_mask = np.zeros((max_end,), dtype=np.bool_)

    for chunk, data, _length_extent in loaded:
        start = interval_by_chunk[chunk.chunk_index].start_frame
        features = data["features"].detach().cpu().float()
        local_indices = [int(idx) for idx in data.get("frame_indices", range(features.shape[0]))]
        stems = list(data.get("frame_stems", [""] * features.shape[0]))
        statuses = list(data.get("frame_statuses", ["unknown"] * features.shape[0]))
        pixel_counts = list(data.get("mask_pixel_counts", [0] * features.shape[0]))
        raw_texts = list(data.get("texts", [""] * features.shape[0]))
        t5_texts = list(data.get("encoded_texts", raw_texts))
        for row_idx, local_idx in enumerate(local_indices):
            global_idx = start + local_idx
            if global_idx >= max_end:
                continue
            raw_text = str(raw_texts[row_idx]) if row_idx < len(raw_texts) else ""
            encoded_text = str(t5_texts[row_idx]) if row_idx < len(t5_texts) else raw_text
            features_all[global_idx] = features[row_idx]
            valid_mask[global_idx] = True
            frame_stems[global_idx] = stems[row_idx] if row_idx < len(stems) else ""
            frame_statuses[global_idx] = statuses[row_idx] if row_idx < len(statuses) else "unknown"
            mask_pixel_counts[global_idx] = int(pixel_counts[row_idx]) if row_idx < len(pixel_counts) else 0
            source_chunk_index[global_idx] = chunk.chunk_index
            texts[global_idx] = raw_text
            encoded_texts[global_idx] = encoded_text
            raw_null_mask[global_idx] = raw_text.strip().lower() == "null"

    return TextStream(
        num_frames=max_end,
        chunk_intervals=chunk_intervals,
        features=features_all,
        valid_mask=valid_mask,
        frame_indices=frame_indices,
        frame_stems=frame_stems,
        frame_statuses=frame_statuses,
        mask_pixel_counts=mask_pixel_counts,
        source_chunk_index=source_chunk_index,
        texts=texts,
        encoded_texts=encoded_texts,
        raw_null_mask=raw_null_mask,
        feature_dim=feature_dim,
        text_model_id=text_model_id,
        feature_model_id=feature_model_id,
    )


def audio_duration_sec(path: Path) -> float:
    info = sf.info(path)
    return float(info.frames) / float(info.samplerate)


def audio_clip_path(audio_root: Path, base_video: str, clip_index: int, start_ms: int, end_ms: int) -> Path:
    name = f"{base_video}__clip_{clip_index:04d}__s{start_ms:08d}_e{end_ms:08d}.wav"
    return audio_root / base_video / name


def video_clip_path(video_root: Path, base_video: str, clip_index: int, start_ms: int, end_ms: int) -> Path:
    name = f"{base_video}__clip_{clip_index:04d}__s{start_ms:08d}_e{end_ms:08d}.mp4"
    return video_root / base_video / name


def find_original_video(video_root: Path, base_video: str) -> Path | None:
    for suffix in VIDEO_SUFFIXES:
        candidate = video_root / f"{base_video}{suffix}"
        if candidate.is_file():
            return candidate
    matches = sorted(path for path in video_root.glob(f"{base_video}.*") if path.suffix in VIDEO_SUFFIXES)
    return matches[0] if matches else None


def write_video_window(
    source_video: Path,
    output_path: Path,
    start_sec: float,
    end_sec: float,
    overwrite: bool,
    mode: str,
    audio_window: Path | None = None,
) -> None:
    if output_path.exists() and not overwrite:
        return
    ensure_dir(output_path.parent)
    duration_sec = end_sec - start_sec
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-ss",
        f"{start_sec:.6f}",
        "-i",
        str(source_video),
    ]
    if audio_window is not None:
        command.extend(["-i", str(audio_window)])
    command.extend(
        [
        "-t",
        f"{duration_sec:.6f}",
        "-map",
        "0:v:0",
        ]
    )
    if audio_window is not None:
        command.extend(["-map", "1:a:0", "-c:a", "aac", "-b:a", "192k", "-shortest"])
    else:
        command.append("-an")
    if mode == "copy":
        command.extend(["-c:v", "copy", "-avoid_negative_ts", "make_zero"])
    else:
        command.extend(
            [
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
            ]
        )
    command.append(str(output_path))
    subprocess.run(command, check=True)


def write_audio_window(
    source_audio: Path,
    output_path: Path,
    start_sec: float,
    end_sec: float,
    overwrite: bool,
    pad_end: bool = False,
) -> None:
    if output_path.exists() and not overwrite:
        return
    info = sf.info(source_audio)
    start_frame = int(round(start_sec * info.samplerate))
    stop_frame = int(round(end_sec * info.samplerate))
    if stop_frame > info.frames and not pad_end:
        raise ValueError(f"Audio window exceeds source length: {source_audio}")
    read_stop = min(stop_frame, info.frames)
    data, sr = sf.read(source_audio, start=start_frame, stop=read_stop, dtype="float32", always_2d=False)
    if stop_frame > info.frames:
        pad_frames = stop_frame - info.frames
        if data.ndim == 1:
            data = np.pad(data, (0, pad_frames), mode="constant")
        else:
            data = np.pad(data, ((0, pad_frames), (0, 0)), mode="constant")
    ensure_dir(output_path.parent)
    sf.write(output_path, data, sr)


def window_starts(
    max_duration_sec: float,
    window_sec: float,
    stride_sec: float,
    include_partial_tail: bool = False,
) -> list[float]:
    starts: list[float] = []
    start = 0.0
    epsilon = 1e-6
    while start + window_sec <= max_duration_sec + epsilon:
        starts.append(start)
        start += stride_sec
    if include_partial_tail and start < max_duration_sec - epsilon:
        starts.append(start)
    return starts


def save_geometry_window(
    path: Path,
    stream: GeometryStream,
    start_idx: int,
    end_idx: int,
    metadata: dict[str, Any],
    overwrite: bool,
) -> tuple[int, float]:
    window_len = end_idx - start_idx
    copy_end = min(end_idx, stream.num_frames)
    copy_len = max(0, copy_end - start_idx)
    frame_indices = np.arange(start_idx, end_idx, dtype=np.int32)
    frame_stems = np.full((window_len,), "", dtype=object)
    status_label = np.full((window_len,), "padded_after_end", dtype=object)
    valid = np.zeros((window_len,), dtype=np.bool_)
    x_ray = np.full((window_len,), np.nan, dtype=np.float32)
    y_ray = np.full((window_len,), np.nan, dtype=np.float32)
    d = np.full((window_len,), np.nan, dtype=np.float32)
    x = np.full((window_len,), np.nan, dtype=np.float32)
    y = np.full((window_len,), np.nan, dtype=np.float32)
    z = np.full((window_len,), np.nan, dtype=np.float32)
    region_pixel_count = np.zeros((window_len,), dtype=np.int32)
    source_chunk_index = np.zeros((window_len,), dtype=np.int32)
    if copy_len > 0:
        dst = slice(0, copy_len)
        src = slice(start_idx, copy_end)
        frame_stems[dst] = stream.frame_stems[src]
        status_label[dst] = stream.status_label[src]
        valid[dst] = stream.valid_mask[src]
        x_ray[dst] = stream.x_ray[src].astype(np.float32)
        y_ray[dst] = stream.y_ray[src].astype(np.float32)
        d[dst] = stream.d[src].astype(np.float32)
        x[dst] = stream.x[src].astype(np.float32)
        y[dst] = stream.y[src].astype(np.float32)
        z[dst] = stream.z[src].astype(np.float32)
        region_pixel_count[dst] = stream.region_pixel_count[src].astype(np.int32)
        source_chunk_index[dst] = stream.source_chunk_index[src].astype(np.int32)
    num_valid = int(valid.sum())
    if path.exists() and not overwrite:
        return num_valid, num_valid / float(window_len)
    ensure_dir(path.parent)
    np.savez_compressed(
        path,
        **{key: np.array(value) for key, value in metadata.items()},
        frame_indices=frame_indices,
        frame_stems=frame_stems,
        status_label=status_label,
        valid_mask=valid.astype(np.bool_),
        x_ray=x_ray,
        y_ray=y_ray,
        d=d,
        x=x,
        y=y,
        z=z,
        region_pixel_count=region_pixel_count,
        source_chunk_index=source_chunk_index,
        num_valid_frames=np.array(num_valid, dtype=np.int32),
        x_ray_valid_mean=np.array(np.nanmean(x_ray[valid]) if num_valid else np.nan, dtype=np.float32),
        y_ray_valid_mean=np.array(np.nanmean(y_ray[valid]) if num_valid else np.nan, dtype=np.float32),
        d_valid_mean=np.array(np.nanmean(d[valid]) if num_valid else np.nan, dtype=np.float32),
    )
    return num_valid, num_valid / float(window_len)


def save_clip_window(
    path: Path,
    stream: ClipStream,
    start_idx: int,
    end_idx: int,
    metadata: dict[str, Any],
    overwrite: bool,
) -> tuple[int, float]:
    window_len = end_idx - start_idx
    copy_end = min(end_idx, stream.num_frames)
    copy_len = max(0, copy_end - start_idx)
    features = torch.zeros((window_len, stream.feature_dim), dtype=torch.float32)
    valid = torch.zeros((window_len,), dtype=torch.bool)
    frame_indices = np.arange(start_idx, end_idx, dtype=np.int32)
    frame_stems = np.full((window_len,), "", dtype=object)
    frame_statuses = np.full((window_len,), "padded_after_end", dtype=object)
    mask_pixel_counts = np.zeros((window_len,), dtype=np.int32)
    source_chunk_index = np.zeros((window_len,), dtype=np.int32)
    if copy_len > 0:
        dst = slice(0, copy_len)
        src = slice(start_idx, copy_end)
        features[dst] = stream.features[src].clone()
        valid[dst] = stream.valid_mask[src].clone()
        frame_stems[dst] = stream.frame_stems[src]
        frame_statuses[dst] = stream.frame_statuses[src]
        mask_pixel_counts[dst] = stream.mask_pixel_counts[src].astype(np.int32)
        source_chunk_index[dst] = stream.source_chunk_index[src].astype(np.int32)
    num_valid = int(valid.sum().item())
    if path.exists() and not overwrite:
        return num_valid, num_valid / float(window_len)
    ensure_dir(path.parent)
    payload = {
        **metadata,
        "features": features,
        "feature_valid_mask": valid.clone(),
        "frame_indices": frame_indices.tolist(),
        "frame_stems": frame_stems.tolist(),
        "frame_statuses": frame_statuses.tolist(),
        "mask_pixel_counts": mask_pixel_counts.astype(int).tolist(),
        "source_chunk_index": source_chunk_index.astype(int).tolist(),
        "num_valid_frames": num_valid,
        "valid_ratio": num_valid / float(window_len),
        "feature_dim": stream.feature_dim,
        "model_id": stream.model_id,
    }
    torch.save(payload, path)
    return num_valid, num_valid / float(window_len)


def save_text_window(
    path: Path,
    stream: TextStream,
    start_idx: int,
    end_idx: int,
    metadata: dict[str, Any],
    overwrite: bool,
) -> tuple[int, float, int]:
    window_len = end_idx - start_idx
    copy_end = min(end_idx, stream.num_frames)
    copy_len = max(0, copy_end - start_idx)
    features = torch.zeros((window_len, stream.feature_dim), dtype=torch.float32)
    valid = torch.zeros((window_len,), dtype=torch.bool)
    frame_indices = np.arange(start_idx, end_idx, dtype=np.int32)
    frame_stems = np.full((window_len,), "", dtype=object)
    frame_statuses = np.full((window_len,), "padded_after_end", dtype=object)
    mask_pixel_counts = np.zeros((window_len,), dtype=np.int32)
    source_chunk_index = np.zeros((window_len,), dtype=np.int32)
    texts = np.full((window_len,), "", dtype=object)
    encoded_texts = np.full((window_len,), "", dtype=object)
    raw_null = np.zeros((window_len,), dtype=np.bool_)
    if copy_len > 0:
        dst = slice(0, copy_len)
        src = slice(start_idx, copy_end)
        features[dst] = stream.features[src].clone()
        valid[dst] = stream.valid_mask[src].clone()
        frame_stems[dst] = stream.frame_stems[src]
        frame_statuses[dst] = stream.frame_statuses[src]
        mask_pixel_counts[dst] = stream.mask_pixel_counts[src].astype(np.int32)
        source_chunk_index[dst] = stream.source_chunk_index[src].astype(np.int32)
        texts[dst] = stream.texts[src]
        encoded_texts[dst] = stream.encoded_texts[src]
        raw_null[dst] = stream.raw_null_mask[src]
    num_valid = int(valid.sum().item())
    num_null = int((raw_null & valid.numpy()).sum())
    if path.exists() and not overwrite:
        return num_valid, num_valid / float(window_len), num_null
    ensure_dir(path.parent)
    payload = {
        **metadata,
        "features": features,
        "feature_valid_mask": valid.clone(),
        "frame_indices": frame_indices.tolist(),
        "frame_stems": frame_stems.tolist(),
        "frame_statuses": frame_statuses.tolist(),
        "mask_pixel_counts": mask_pixel_counts.astype(int).tolist(),
        "source_chunk_index": source_chunk_index.astype(int).tolist(),
        "texts": texts.tolist(),
        "encoded_texts": encoded_texts.tolist(),
        "raw_null_mask": raw_null.astype(bool).tolist(),
        "num_valid_frames": num_valid,
        "valid_ratio": num_valid / float(window_len),
        "num_null_frames": num_null,
        "feature_dim": stream.feature_dim,
        "text_model_id": stream.text_model_id,
        "feature_model_id": stream.feature_model_id,
    }
    torch.save(payload, path)
    return num_valid, num_valid / float(window_len), num_null


def handle_missing(policy: str, message: str) -> bool:
    if policy == "strict":
        raise FileNotFoundError(message)
    return False


def build_split(
    split: str,
    args: argparse.Namespace,
    output_root: Path,
    original_root: Path,
    window_sec: float,
    overlap: float,
    stride_sec: float,
) -> dict[str, Any]:
    split_root = args.data_root / args.chunk_root / split
    split_output = output_root / split
    audio_output_root = split_output / "audio"
    video_output_root = split_output / "video"
    geometry_output_root = split_output / "depth_xy_ray"
    clip_output_root = split_output / "clip_features"
    text_output_root = split_output / "t5_text_features"
    manifest_path = split_output / "manifest" / "manifest_mm.jsonl"
    summary_path = split_output / "manifest" / "build_summary_mm.json"
    original_audio_root = original_root / split / "audio"
    original_video_root = original_root / split / "video"

    records: list[dict[str, Any]] = []
    drop_counts: Counter[str] = Counter()
    scene_summaries: list[dict[str, Any]] = []
    audio_written: set[Path] = set()
    video_written: set[Path] = set()
    video_chunk_interval_cache: dict[str, list[ChunkInterval]] = {}
    original_video_cache: dict[str, Path | None] = {}
    audio_duration_cache: dict[Path, float] = {}
    quantized_window_count = 0
    ignored_video_chunks, ignore_video_chunk_list_path = load_ignore_video_chunks(args, split)
    conflicted = conflicted_scene_chunks(split_root / "person_face_mapping")
    conflicted_chunk_count = sum(len(chunks) for chunks in conflicted.values())

    index = discover_split_index(split_root, split, conflicted)
    if index is None:
        drop_counts["missing_depth_clip_or_text_root"] += 1
        handle_missing(args.missing_policy, f"Missing depth, CLIP, or T5 text root for split {split}: {split_root}")
        summary = {
            "split": split,
            "skipped": True,
            "output_root": str(output_root),
            "ignore_video_chunk_list": ignore_video_chunk_list_path,
            "ignored_video_chunk_count": len(ignored_video_chunks),
            "conflicted_scene_chunk_count": conflicted_chunk_count,
            "drop_counts": dict(drop_counts),
            "emitted_record_count": 0,
            "manifest_path": str(manifest_path),
        }
        write_json(summary_path, summary)
        write_jsonl(manifest_path, [])
        return summary

    scenes = index.scenes
    if args.scene_key:
        scenes = [scene for scene in scenes if scene == args.scene_key]
    if args.limit_scenes:
        scenes = scenes[: args.limit_scenes]

    def get_video_chunk_intervals(scene_name: str, base_video: str) -> list[ChunkInterval]:
        if base_video not in video_chunk_interval_cache:
            intervals = video_chunk_intervals_from_frames(split_root, base_video, args.chunk_sec, args.fps)
            if not intervals:
                matching_indices = {
                    chunk.chunk_index
                    for mapping in (index.depth, index.clip, index.text)
                    for (item_scene, _person_id, item_base), chunks in mapping.items()
                    if item_scene == scene_name and item_base == base_video
                    for chunk in chunks
                } | conflicted.get(scene_name, set())
                intervals = build_chunk_intervals_from_lengths(
                    [
                        (chunk_index, chunk_frame_count(args.chunk_sec, args.fps))
                        for chunk_index in sorted(matching_indices)
                    ],
                    args.chunk_sec,
                    args.fps,
                )
            video_chunk_interval_cache[base_video] = intervals
        return video_chunk_interval_cache[base_video]

    def get_audio_duration(path: Path) -> float:
        if path not in audio_duration_cache:
            audio_duration_cache[path] = audio_duration_sec(path)
        return audio_duration_cache[path]

    audio_stem_allowlist = (
        {Path(value).stem for value in args.audio_stem_allowlist}
        if args.audio_stem_allowlist
        else None
    )

    def is_audio_stem_allowed(path: Path) -> bool:
        return audio_stem_allowlist is None or path.stem in audio_stem_allowlist

    def is_audio_duration_allowed(path: Path) -> bool:
        return args.max_audio_duration_sec <= 0.0 or get_audio_duration(path) < args.max_audio_duration_sec

    def get_original_video(base_video: str) -> Path | None:
        if base_video not in original_video_cache:
            original_video_cache[base_video] = find_original_video(original_video_root, base_video)
        return original_video_cache[base_video]

    window_frames = int(round(window_sec * args.fps))
    stride_frames = sec_to_frame_index(stride_sec, args.fps)
    window_ms = int(round(window_sec * 1000.0))
    stride_ms = int(round(stride_sec * 1000.0))

    for scene in scenes:
        camera_to_base = index.camera_to_base.get(scene, {})
        scene_summary: dict[str, Any] = {
            "scene_name": scene,
            "camera_person_ids": sorted(camera_to_base),
            "pair_stream_count": 0,
            "emitted_record_count": 0,
            "quantized_window_count": 0,
            "drop_counts": Counter(),
        }

        for src_person_id in sorted(camera_to_base):
            src_base = camera_to_base[src_person_id]
            src_audio_source = original_audio_root / f"{src_base}.wav"
            if not src_audio_source.is_file():
                scene_summary["drop_counts"]["missing_src_audio"] += 1
                drop_counts["missing_src_audio"] += 1
                if not handle_missing(args.missing_policy, f"Missing source audio: {src_audio_source}"):
                    continue
            if not is_audio_stem_allowed(src_audio_source):
                scene_summary["drop_counts"]["src_audio_stem_filtered"] += 1
                drop_counts["src_audio_stem_filtered"] += 1
                continue
            if not is_audio_duration_allowed(src_audio_source):
                scene_summary["drop_counts"]["src_audio_duration_filtered"] += 1
                drop_counts["src_audio_duration_filtered"] += 1
                continue

            for tgt_person_id in sorted(camera_to_base):
                if src_person_id == tgt_person_id:
                    continue
                tgt_base = camera_to_base[tgt_person_id]
                tgt_audio_source = original_audio_root / f"{tgt_base}.wav"
                pair_key = (scene, tgt_person_id, src_base)
                scene_summary["pair_stream_count"] += 1

                if not tgt_audio_source.is_file():
                    scene_summary["drop_counts"]["missing_tgt_audio"] += 1
                    drop_counts["missing_tgt_audio"] += 1
                    if not handle_missing(args.missing_policy, f"Missing target audio: {tgt_audio_source}"):
                        continue
                if not is_audio_stem_allowed(tgt_audio_source):
                    scene_summary["drop_counts"]["tgt_audio_stem_filtered"] += 1
                    drop_counts["tgt_audio_stem_filtered"] += 1
                    continue
                if not is_audio_duration_allowed(tgt_audio_source):
                    scene_summary["drop_counts"]["tgt_audio_duration_filtered"] += 1
                    drop_counts["tgt_audio_duration_filtered"] += 1
                    continue
                if pair_key not in index.depth:
                    scene_summary["drop_counts"]["missing_target_geometry"] += 1
                    drop_counts["missing_target_geometry"] += 1
                    if not handle_missing(args.missing_policy, f"Missing target geometry chunks: {pair_key}"):
                        continue
                if pair_key not in index.clip:
                    scene_summary["drop_counts"]["missing_target_clip"] += 1
                    drop_counts["missing_target_clip"] += 1
                    if not handle_missing(args.missing_policy, f"Missing target CLIP chunks: {pair_key}"):
                        continue
                if pair_key not in index.text:
                    scene_summary["drop_counts"]["missing_target_t5_text"] += 1
                    drop_counts["missing_target_t5_text"] += 1
                    if not handle_missing(args.missing_policy, f"Missing target T5 text chunks: {pair_key}"):
                        continue

                base_chunk_intervals = get_video_chunk_intervals(scene, src_base)
                geometry = merge_geometry(index.depth[pair_key], args.fps, args.chunk_sec, base_chunk_intervals)
                clip = merge_clip(index.clip[pair_key], args.fps, args.chunk_sec, base_chunk_intervals)
                text = merge_text(index.text[pair_key], args.fps, args.chunk_sec, base_chunk_intervals)
                max_duration_sec = min(
                    get_audio_duration(src_audio_source),
                    get_audio_duration(tgt_audio_source),
                    geometry.num_frames / args.fps,
                    clip.num_frames / args.fps,
                    text.num_frames / args.fps,
                )
                starts = window_starts(
                    max_duration_sec,
                    window_sec,
                    stride_sec,
                    include_partial_tail=False,
                )
                if args.limit_windows_per_pair:
                    starts = starts[: args.limit_windows_per_pair]
                if not starts:
                    scene_summary["drop_counts"]["no_complete_window"] += 1
                    drop_counts["no_complete_window"] += 1
                    continue

                for window_idx, start_sec in enumerate(starts, start=1):
                    if args.max_records and len(records) >= args.max_records:
                        break
                    end_sec = start_sec + window_sec
                    start_ms = int(round(start_sec * 1000.0))
                    end_ms = start_ms + window_ms
                    start_idx = sec_to_frame_index(start_sec, args.fps)
                    end_idx = start_idx + window_frames
                    aligned_start_ms = frame_to_ms(start_idx, args.fps)
                    aligned_end_ms = frame_to_ms(end_idx, args.fps)
                    if aligned_start_ms != start_ms or aligned_end_ms != end_ms:
                        scene_summary["quantized_window_count"] += 1
                        quantized_window_count += 1
                    if (
                        split != "test"
                        and (
                            end_idx > geometry.num_frames
                            or end_idx > clip.num_frames
                            or end_idx > text.num_frames
                        )
                    ):
                        scene_summary["drop_counts"]["window_exceeds_modality"] += 1
                        drop_counts["window_exceeds_modality"] += 1
                        continue
                    overlapped_source_chunks = overlapping_chunk_names(
                        src_base,
                        start_idx,
                        end_idx,
                        base_chunk_intervals,
                    )
                    if any(name in ignored_video_chunks for name in overlapped_source_chunks):
                        scene_summary["drop_counts"]["ignored_source_video_chunk"] += 1
                        drop_counts["ignored_source_video_chunk"] += 1
                        continue
                    overlapped_source_chunk_indices = overlapping_chunk_indices(
                        start_idx,
                        end_idx,
                        base_chunk_intervals,
                    )
                    if any(chunk in conflicted.get(scene, set()) for chunk in overlapped_source_chunk_indices):
                        scene_summary["drop_counts"]["conflicted_source_video_chunk"] += 1
                        drop_counts["conflicted_source_video_chunk"] += 1
                        continue

                    clip_valid = int(clip.valid_mask[start_idx:end_idx].sum().item())
                    if clip_valid == 0:
                        scene_summary["drop_counts"]["no_valid_clip_in_window"] += 1
                        drop_counts["no_valid_clip_in_window"] += 1
                        continue
                    clip_statuses = clip.frame_statuses[start_idx:end_idx]
                    if not any(status == "masked" for status in clip_statuses):
                        scene_summary["drop_counts"]["fully_absent_target_visual"] += 1
                        drop_counts["fully_absent_target_visual"] += 1
                        continue

                    text_valid = int(text.valid_mask[start_idx:end_idx].sum().item())
                    if text_valid == 0:
                        scene_summary["drop_counts"]["no_valid_t5_text_in_window"] += 1
                        drop_counts["no_valid_t5_text_in_window"] += 1
                        continue

                    geometry_valid = int(geometry.valid_mask[start_idx:end_idx].sum())
                    if geometry_valid == 0:
                        scene_summary["drop_counts"]["no_valid_geometry_in_window"] += 1
                        drop_counts["no_valid_geometry_in_window"] += 1
                        continue
                    geometry_valid_ratio = geometry_valid / float(window_frames)
                    if (
                        split == "test"
                        and args.test_min_geometry_valid_ratio >= 0.0
                        and geometry_valid_ratio < args.test_min_geometry_valid_ratio
                    ):
                        scene_summary["drop_counts"]["low_geometry_valid_ratio"] += 1
                        drop_counts["low_geometry_valid_ratio"] += 1
                        continue

                    src_audio_path = audio_clip_path(audio_output_root, src_base, window_idx, start_ms, end_ms)
                    tgt_audio_path = audio_clip_path(audio_output_root, tgt_base, window_idx, start_ms, end_ms)
                    src_video_path: Path | None = None
                    tgt_video_path: Path | None = None
                    for audio_source, audio_path in (
                        (src_audio_source, src_audio_path),
                        (tgt_audio_source, tgt_audio_path),
                    ):
                        if audio_path not in audio_written:
                            write_audio_window(
                                audio_source,
                                audio_path,
                                start_sec,
                                end_sec,
                                False,
                                pad_end=(split == "test"),
                            )
                            audio_written.add(audio_path)
                    if args.write_video_windows:
                        src_video_source = get_original_video(src_base)
                        tgt_video_source = get_original_video(tgt_base)
                        if src_video_source is None:
                            scene_summary["drop_counts"]["missing_src_video"] += 1
                            drop_counts["missing_src_video"] += 1
                            if not handle_missing(args.missing_policy, f"Missing source video: {src_base}"):
                                continue
                        if tgt_video_source is None:
                            scene_summary["drop_counts"]["missing_tgt_video"] += 1
                            drop_counts["missing_tgt_video"] += 1
                            if not handle_missing(args.missing_policy, f"Missing target video: {tgt_base}"):
                                continue
                        src_video_path = video_clip_path(video_output_root, src_base, window_idx, start_ms, end_ms)
                        tgt_video_path = video_clip_path(video_output_root, tgt_base, window_idx, start_ms, end_ms)
                        for video_source, video_path in (
                            (src_video_source, src_video_path),
                            (tgt_video_source, tgt_video_path),
                        ):
                            if video_source is None:
                                continue
                            if video_path not in video_written:
                                audio_window = None
                                if args.video_window_audio == "window":
                                    audio_window = src_audio_path if video_path == src_video_path else tgt_audio_path
                                write_video_window(
                                    video_source,
                                    video_path,
                                    start_sec,
                                    end_sec,
                                    args.overwrite,
                                    args.video_window_mode,
                                    audio_window,
                                )
                                video_written.add(video_path)

                    sidecar_name = (
                        f"{src_base}__tgt_person_{tgt_person_id}"
                        f"__clip_{window_idx:04d}__s{start_ms:08d}_e{end_ms:08d}"
                    )
                    geometry_path = (
                        geometry_output_root / scene / src_base / f"target_person_{tgt_person_id}" / f"{sidecar_name}.npz"
                    )
                    clip_path = (
                        clip_output_root / scene / src_base / f"target_person_{tgt_person_id}" / f"{sidecar_name}.pt"
                    )
                    text_path = (
                        text_output_root / scene / src_base / f"target_person_{tgt_person_id}" / f"{sidecar_name}.pt"
                    )
                    metadata = {
                        "split": split,
                        "scene_name": scene,
                        "src_video_name": src_base,
                        "tgt_video_name": tgt_base,
                        "src_person_id": int(src_person_id),
                        "tgt_person_id": int(tgt_person_id),
                        "clip_index": int(window_idx),
                        "clip_start_ms": int(start_ms),
                        "clip_end_ms": int(end_ms),
                        "aligned_clip_start_ms": int(aligned_start_ms),
                        "aligned_clip_end_ms": int(aligned_end_ms),
                        "fps": float(args.fps),
                        "window_sec": float(window_sec),
                        "stride_sec": float(stride_sec),
                        "window_num_frames": int(window_frames),
                    }
                    geometry_count, geometry_ratio = save_geometry_window(
                        geometry_path,
                        geometry,
                        start_idx,
                        end_idx,
                        metadata,
                        False,
                    )
                    clip_count, clip_ratio = save_clip_window(
                        clip_path,
                        clip,
                        start_idx,
                        end_idx,
                        metadata,
                        False,
                    )
                    text_count, text_ratio, text_null_count = save_text_window(
                        text_path,
                        text,
                        start_idx,
                        end_idx,
                        metadata,
                        args.overwrite,
                    )

                    record = {
                        "split": split,
                        "scene_name": scene,
                        "src_person": f"person_{src_person_id}",
                        "tgt_person": f"person_{tgt_person_id}",
                        "src_person_id": int(src_person_id),
                        "tgt_person_id": int(tgt_person_id),
                        "src_video_name": src_base,
                        "tgt_video_name": tgt_base,
                        "clip_start_ms": start_ms,
                        "clip_end_ms": end_ms,
                        "src_audio_path": str(src_audio_path),
                        "tgt_audio_path": str(tgt_audio_path),
                        "src_clip_filename": src_audio_path.name,
                        "tgt_clip_filename": tgt_audio_path.name,
                        "src_video_path": str(src_video_path) if src_video_path is not None else None,
                        "tgt_video_path": str(tgt_video_path) if tgt_video_path is not None else None,
                        "src_video_filename": src_video_path.name if src_video_path is not None else None,
                        "tgt_video_filename": tgt_video_path.name if tgt_video_path is not None else None,
                        "tgt_geometry_path": str(geometry_path),
                        "tgt_geometry_filename": geometry_path.name,
                        "tgt_geometry_num_valid_frames": geometry_count,
                        "tgt_geometry_valid_ratio": geometry_ratio,
                        "tgt_clip_feature_path": str(clip_path),
                        "tgt_clip_feature_filename": clip_path.name,
                        "tgt_clip_num_valid_frames": clip_count,
                        "tgt_clip_valid_ratio": clip_ratio,
                        "tgt_t5_text_feature_path": str(text_path),
                        "tgt_t5_text_feature_filename": text_path.name,
                        "tgt_t5_text_num_valid_frames": text_count,
                        "tgt_t5_text_valid_ratio": text_ratio,
                        "tgt_t5_text_num_null_frames": text_null_count,
                        "aligned_start_frame": start_idx,
                        "aligned_end_frame": end_idx,
                        "tgt_geometry_aligned_start_ms": aligned_start_ms,
                        "tgt_geometry_aligned_end_ms": aligned_end_ms,
                        "tgt_clip_aligned_start_ms": aligned_start_ms,
                        "tgt_clip_aligned_end_ms": aligned_end_ms,
                        "tgt_t5_text_aligned_start_ms": aligned_start_ms,
                        "tgt_t5_text_aligned_end_ms": aligned_end_ms,
                        "window_frames": window_frames,
                    }
                    records.append(record)
                    scene_summary["emitted_record_count"] += 1

                if args.max_records and len(records) >= args.max_records:
                    break
            if args.max_records and len(records) >= args.max_records:
                break

        scene_summary["drop_counts"] = dict(scene_summary["drop_counts"])
        scene_summaries.append(scene_summary)
        if args.max_records and len(records) >= args.max_records:
            break

    summary = {
        "split": split,
        "skipped": False,
        "data_root": str(args.data_root),
        "chunk_root": args.chunk_root,
        "original_root": str(original_root),
        "output_root": str(output_root),
        "ignore_video_chunk_list": ignore_video_chunk_list_path,
        "ignored_video_chunk_count": len(ignored_video_chunks),
        "conflicted_scene_chunk_count": conflicted_chunk_count,
        "partial_tail_padding": False,
        "max_audio_duration_sec": float(args.max_audio_duration_sec),
        "audio_stem_allowlist": sorted(audio_stem_allowlist) if audio_stem_allowlist is not None else None,
        "test_min_geometry_valid_ratio": args.test_min_geometry_valid_ratio,
        "window_sec": window_sec,
        "overlap": overlap,
        "overlap_mode": args.overlap_mode,
        "stride_sec": stride_sec,
        "fps": args.fps,
        "chunk_sec": args.chunk_sec,
        "window_ms": window_ms,
        "stride_ms": stride_ms,
        "window_frames": window_frames,
        "stride_frames": stride_frames,
        "scene_count": len(scene_summaries),
        "emitted_record_count": len(records),
        "audio_window_count": len(audio_written),
        "video_window_count": len(video_written),
        "write_video_windows": bool(args.write_video_windows),
        "video_window_mode": args.video_window_mode,
        "video_window_audio": args.video_window_audio,
        "quantized_window_count": quantized_window_count,
        "drop_counts": dict(drop_counts),
        "scene_summaries": scene_summaries,
        "manifest_path": str(manifest_path),
    }
    write_jsonl(manifest_path, records)
    write_json(summary_path, summary)
    return summary


def main() -> None:
    args = parse_args()
    original_root = args.original_root or (args.data_root / "original")
    splits = parse_split_list(args.splits)

    summaries = []
    for split in splits:
        window_sec, overlap, stride_sec = split_window_config(args, split)
        if args.output_root is not None:
            output_root = args.output_root
        elif args.output_tag is not None:
            output_root = args.data_root / args.output_tag
        else:
            output_root = args.data_root / make_output_tag(window_sec, overlap)
        summaries.append(
            build_split(
                split=split,
                args=args,
                output_root=output_root,
                original_root=original_root,
                window_sec=window_sec,
                overlap=overlap,
                stride_sec=stride_sec,
            )
        )

    print(
        json.dumps(
            {
                "splits": [
                    {
                        "split": item["split"],
                        "skipped": item.get("skipped", False),
                        "output_root": item.get("output_root"),
                        "manifest_path": item.get("manifest_path"),
                        "emitted_record_count": item.get("emitted_record_count", 0),
                        "drop_counts": item.get("drop_counts", {}),
                    }
                    for item in summaries
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
