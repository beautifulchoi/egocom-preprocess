#!/usr/bin/env python3
"""
Build windowed EgoCom multimodal pair manifests from 1-minute chunk outputs.

Inputs:
  {data_root}/{chunk_root}/{split}/person_depth_lift/{scene}/person_{id}/*.npz
  {data_root}/{chunk_root}/{split}/person_visual_clip_features/{scene}/person_{id}/*.pt
  {data_root}/original/{split}/audio/*.wav

Outputs:
  {data_root}/{output_tag}/{split}/audio
  {data_root}/{output_tag}/{split}/depth_xy_ray
  {data_root}/{output_tag}/{split}/clip_features
  {data_root}/{output_tag}/{split}/jsonl/manifest.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import re
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


@dataclass(frozen=True)
class ChunkPath:
    chunk_index: int
    path: Path


@dataclass
class SplitIndex:
    depth: dict[tuple[str, int, str], list[ChunkPath]]
    clip: dict[tuple[str, int, str], list[ChunkPath]]
    camera_to_base: dict[str, dict[int, str]]
    scenes: list[str]


@dataclass
class GeometryStream:
    num_frames: int
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
    features: torch.Tensor
    valid_mask: torch.Tensor
    frame_indices: np.ndarray
    frame_stems: np.ndarray
    frame_statuses: np.ndarray
    mask_pixel_counts: np.ndarray
    source_chunk_index: np.ndarray
    feature_dim: int
    model_id: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build windowed EgoCom manifests from chunked depth, CLIP, and original audio.",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--chunk-root", default="1min")
    parser.add_argument("--original-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--output-tag", default=None)
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--window-sec", type=float, default=5.0)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--overlap-mode", choices=("ratio", "seconds"), default="ratio")
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--chunk-sec", type=float, default=60.0)
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
    return f"{format_number(window_sec)}s_overlap{format_number(overlap)}"


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


def parse_chunk_stem(stem: str) -> tuple[str, int] | None:
    match = CHUNK_RE.match(stem)
    if match is None:
        return None
    return match.group("base"), int(match.group("chunk"))


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


def discover_split_index(split_root: Path, split: str) -> SplitIndex | None:
    depth_root = split_root / "person_depth_lift"
    clip_root = split_root / "person_visual_clip_features"
    if not depth_root.is_dir() or not clip_root.is_dir():
        return None

    depth: dict[tuple[str, int, str], list[ChunkPath]] = defaultdict(list)
    clip: dict[tuple[str, int, str], list[ChunkPath]] = defaultdict(list)
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
            clip[(scene, target_person, base_video)].append(ChunkPath(chunk_index, pt_path))
            add_camera(scene, base_video)

    scenes = sorted(set(camera_to_base) | {key[0] for key in depth} | {key[0] for key in clip})
    for paths in depth.values():
        paths.sort(key=lambda item: item.chunk_index)
    for paths in clip.values():
        paths.sort(key=lambda item: item.chunk_index)
    return SplitIndex(depth=dict(depth), clip=dict(clip), camera_to_base=dict(camera_to_base), scenes=scenes)


def chunk_start_frame(chunk_index: int, chunk_sec: float, fps: float) -> int:
    return int(round((chunk_index - 1) * chunk_sec * fps))


def sec_to_frame_index(sec: float, fps: float) -> int:
    return int(math.floor(sec * fps + 0.5))


def frame_to_ms(frame_idx: int, fps: float) -> int:
    return int(round((frame_idx / fps) * 1000.0))


def safe_array(data: dict[str, np.ndarray], key: str, length: int, fill: float = np.nan) -> np.ndarray:
    if key in data:
        arr = np.asarray(data[key])
        return arr[:length]
    return np.full((length,), fill, dtype=np.float32)


def merge_geometry(chunks: list[ChunkPath], fps: float, chunk_sec: float) -> GeometryStream:
    loaded: list[tuple[ChunkPath, dict[str, np.ndarray], int, int]] = []
    max_end = 0
    for chunk in chunks:
        data = load_npz(chunk.path)
        length = int(np.asarray(data.get("num_frames", len(data["d_mean"]))).item())
        start = chunk_start_frame(chunk.chunk_index, chunk_sec, fps)
        loaded.append((chunk, data, start, length))
        max_end = max(max_end, start + length)

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

    for chunk, data, start, length in loaded:
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


def merge_clip(chunks: list[ChunkPath], fps: float, chunk_sec: float) -> ClipStream:
    loaded: list[tuple[ChunkPath, dict[str, Any], int, int]] = []
    max_end = 0
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
        start = chunk_start_frame(chunk.chunk_index, chunk_sec, fps)
        loaded.append((chunk, data, start, length_extent))
        max_end = max(max_end, start + length_extent)

    features_all = torch.zeros((max_end, feature_dim), dtype=torch.float32)
    valid_mask = torch.zeros((max_end,), dtype=torch.bool)
    frame_indices = np.arange(max_end, dtype=np.int32)
    frame_stems = np.full((max_end,), "", dtype=object)
    frame_statuses = np.full((max_end,), "missing_chunk", dtype=object)
    mask_pixel_counts = np.zeros((max_end,), dtype=np.int32)
    source_chunk_index = np.zeros((max_end,), dtype=np.int32)

    for chunk, data, start, _length_extent in loaded:
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


def audio_duration_sec(path: Path) -> float:
    info = sf.info(path)
    return float(info.frames) / float(info.samplerate)


def audio_clip_path(audio_root: Path, base_video: str, clip_index: int, start_ms: int, end_ms: int) -> Path:
    name = f"{base_video}__clip_{clip_index:04d}__s{start_ms:08d}_e{end_ms:08d}.wav"
    return audio_root / base_video / name


def write_audio_window(
    source_audio: Path,
    output_path: Path,
    start_sec: float,
    end_sec: float,
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        return
    info = sf.info(source_audio)
    start_frame = int(round(start_sec * info.samplerate))
    stop_frame = int(round(end_sec * info.samplerate))
    if stop_frame > info.frames:
        raise ValueError(f"Audio window exceeds source length: {source_audio}")
    data, sr = sf.read(source_audio, start=start_frame, stop=stop_frame, dtype="float32", always_2d=False)
    ensure_dir(output_path.parent)
    sf.write(output_path, data, sr)


def window_starts(max_duration_sec: float, window_sec: float, stride_sec: float) -> list[float]:
    starts: list[float] = []
    start = 0.0
    epsilon = 1e-6
    while start + window_sec <= max_duration_sec + epsilon:
        starts.append(start)
        start += stride_sec
    return starts


def save_geometry_window(
    path: Path,
    stream: GeometryStream,
    start_idx: int,
    end_idx: int,
    metadata: dict[str, Any],
    overwrite: bool,
) -> tuple[int, float]:
    valid = stream.valid_mask[start_idx:end_idx]
    num_valid = int(valid.sum())
    if path.exists() and not overwrite:
        return num_valid, num_valid / float(end_idx - start_idx)
    ensure_dir(path.parent)
    x_ray = stream.x_ray[start_idx:end_idx].astype(np.float32)
    y_ray = stream.y_ray[start_idx:end_idx].astype(np.float32)
    d = stream.d[start_idx:end_idx].astype(np.float32)
    np.savez_compressed(
        path,
        **{key: np.array(value) for key, value in metadata.items()},
        frame_indices=stream.frame_indices[start_idx:end_idx].astype(np.int32),
        frame_stems=stream.frame_stems[start_idx:end_idx],
        status_label=stream.status_label[start_idx:end_idx],
        valid_mask=valid.astype(np.bool_),
        x_ray=x_ray,
        y_ray=y_ray,
        d=d,
        x=stream.x[start_idx:end_idx].astype(np.float32),
        y=stream.y[start_idx:end_idx].astype(np.float32),
        z=stream.z[start_idx:end_idx].astype(np.float32),
        region_pixel_count=stream.region_pixel_count[start_idx:end_idx].astype(np.int32),
        source_chunk_index=stream.source_chunk_index[start_idx:end_idx].astype(np.int32),
        num_valid_frames=np.array(num_valid, dtype=np.int32),
        x_ray_valid_mean=np.array(np.nanmean(x_ray[valid]) if num_valid else np.nan, dtype=np.float32),
        y_ray_valid_mean=np.array(np.nanmean(y_ray[valid]) if num_valid else np.nan, dtype=np.float32),
        d_valid_mean=np.array(np.nanmean(d[valid]) if num_valid else np.nan, dtype=np.float32),
    )
    return num_valid, num_valid / float(end_idx - start_idx)


def save_clip_window(
    path: Path,
    stream: ClipStream,
    start_idx: int,
    end_idx: int,
    metadata: dict[str, Any],
    overwrite: bool,
) -> tuple[int, float]:
    valid = stream.valid_mask[start_idx:end_idx]
    num_valid = int(valid.sum().item())
    if path.exists() and not overwrite:
        return num_valid, num_valid / float(end_idx - start_idx)
    ensure_dir(path.parent)
    payload = {
        **metadata,
        "features": stream.features[start_idx:end_idx].clone(),
        "feature_valid_mask": valid.clone(),
        "frame_indices": stream.frame_indices[start_idx:end_idx].astype(np.int32).tolist(),
        "frame_stems": stream.frame_stems[start_idx:end_idx].tolist(),
        "frame_statuses": stream.frame_statuses[start_idx:end_idx].tolist(),
        "mask_pixel_counts": stream.mask_pixel_counts[start_idx:end_idx].astype(int).tolist(),
        "source_chunk_index": stream.source_chunk_index[start_idx:end_idx].astype(int).tolist(),
        "num_valid_frames": num_valid,
        "valid_ratio": num_valid / float(end_idx - start_idx),
        "feature_dim": stream.feature_dim,
        "model_id": stream.model_id,
    }
    torch.save(payload, path)
    return num_valid, num_valid / float(end_idx - start_idx)


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
    stride_sec: float,
) -> dict[str, Any]:
    split_root = args.data_root / args.chunk_root / split
    split_output = output_root / split
    audio_output_root = split_output / "audio"
    geometry_output_root = split_output / "depth_xy_ray"
    clip_output_root = split_output / "clip_features"
    manifest_path = split_output / "jsonl" / "manifest.jsonl"
    summary_path = split_output / "jsonl" / "build_summary.json"
    original_audio_root = original_root / split / "audio"

    records: list[dict[str, Any]] = []
    drop_counts: Counter[str] = Counter()
    scene_summaries: list[dict[str, Any]] = []
    audio_written: set[Path] = set()
    quantized_window_count = 0

    index = discover_split_index(split_root, split)
    if index is None:
        drop_counts["missing_depth_or_clip_root"] += 1
        handle_missing(args.missing_policy, f"Missing depth or CLIP root for split {split}: {split_root}")
        summary = {
            "split": split,
            "skipped": True,
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

                geometry = merge_geometry(index.depth[pair_key], args.fps, args.chunk_sec)
                clip = merge_clip(index.clip[pair_key], args.fps, args.chunk_sec)
                max_duration_sec = min(
                    audio_duration_sec(src_audio_source),
                    audio_duration_sec(tgt_audio_source),
                    geometry.num_frames / args.fps,
                    clip.num_frames / args.fps,
                )
                starts = window_starts(max_duration_sec, window_sec, stride_sec)
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
                    if end_idx > geometry.num_frames or end_idx > clip.num_frames:
                        scene_summary["drop_counts"]["window_exceeds_modality"] += 1
                        drop_counts["window_exceeds_modality"] += 1
                        continue

                    geometry_valid = int(geometry.valid_mask[start_idx:end_idx].sum())
                    if geometry_valid == 0:
                        scene_summary["drop_counts"]["no_valid_geometry_in_window"] += 1
                        drop_counts["no_valid_geometry_in_window"] += 1
                        continue
                    clip_valid = int(clip.valid_mask[start_idx:end_idx].sum().item())
                    if clip_valid == 0:
                        scene_summary["drop_counts"]["no_valid_clip_in_window"] += 1
                        drop_counts["no_valid_clip_in_window"] += 1
                        continue

                    src_audio_path = audio_clip_path(audio_output_root, src_base, window_idx, start_ms, end_ms)
                    tgt_audio_path = audio_clip_path(audio_output_root, tgt_base, window_idx, start_ms, end_ms)
                    for audio_source, audio_path in (
                        (src_audio_source, src_audio_path),
                        (tgt_audio_source, tgt_audio_path),
                    ):
                        if audio_path not in audio_written:
                            write_audio_window(audio_source, audio_path, start_sec, end_sec, args.overwrite)
                            audio_written.add(audio_path)

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
                        args.overwrite,
                    )
                    clip_count, clip_ratio = save_clip_window(
                        clip_path,
                        clip,
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
                        "tgt_geometry_path": str(geometry_path),
                        "tgt_geometry_filename": geometry_path.name,
                        "tgt_geometry_num_valid_frames": geometry_count,
                        "tgt_geometry_valid_ratio": geometry_ratio,
                        "tgt_clip_feature_path": str(clip_path),
                        "tgt_clip_feature_filename": clip_path.name,
                        "tgt_clip_num_valid_frames": clip_count,
                        "tgt_clip_valid_ratio": clip_ratio,
                        "aligned_start_frame": start_idx,
                        "aligned_end_frame": end_idx,
                        "tgt_geometry_aligned_start_ms": aligned_start_ms,
                        "tgt_geometry_aligned_end_ms": aligned_end_ms,
                        "tgt_clip_aligned_start_ms": aligned_start_ms,
                        "tgt_clip_aligned_end_ms": aligned_end_ms,
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
        "window_sec": window_sec,
        "overlap": args.overlap,
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
    output_tag = args.output_tag or make_output_tag(args.window_sec, args.overlap)
    output_root = args.output_root or (args.data_root / output_tag)
    stride_sec = compute_stride_sec(args.window_sec, args.overlap, args.overlap_mode)
    splits = parse_split_list(args.splits)

    summaries = []
    for split in splits:
        summaries.append(
            build_split(
                split=split,
                args=args,
                output_root=output_root,
                original_root=original_root,
                window_sec=args.window_sec,
                stride_sec=stride_sec,
            )
        )

    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "splits": [
                    {
                        "split": item["split"],
                        "skipped": item.get("skipped", False),
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
