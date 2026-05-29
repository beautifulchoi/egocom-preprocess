#!/usr/bin/env python3
"""
Gather EgoCom 1-minute chunk outputs into compact whole-sequence artifacts.

Outputs are written under:
  /home/prj/data/egocom_holdout/gathering/{split}/{original_video}/{modality}

Chunk grouping removes only the trailing `_chunk_####` suffix. This preserves
scene, camera/person, and optional part identifiers in the original video name.
Ignored chunks are skipped from the compact output and recorded in timeline.json.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import soundfile as sf


DEFAULT_DATA_ROOT = Path("/home/prj/data/egocom_holdout")
CHUNK_RE = re.compile(r"^(?P<base>.+)_chunk_(?P<chunk>\d+)$")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_SUFFIXES = (".MP4", ".mp4", ".MOV", ".mov", ".M4V", ".m4v")
FRAME_LIKE_KEYS = {
    "frame_indices",
    "logical_frame_indices",
    "frame_stems",
    "frame_statuses",
    "status_label",
    "status_code",
    "has_masks",
    "feature_valid_mask",
    "valid_mask",
    "mask_pixel_counts",
    "mask_grid_sums",
    "feature_grid_shapes",
    "source_frame_paths",
    "source_chunk_index",
    "raw_null_mask",
    "texts",
    "encoded_texts",
    "mask_bboxes",
    "region_pixel_count",
    "valid_depth_pixel_count",
    "face_bbox_orig",
    "face_bbox_depth",
    "discontinuity_gap_ratio",
    "discontinuity_boundary_grad_p90",
    "x_ray_mean",
    "y_ray_mean",
    "x_mean",
    "y_mean",
    "z_mean",
    "d_mean",
}


@dataclass(frozen=True)
class ChunkRecord:
    base_video: str
    chunk_index: int
    name: str
    split_root: Path
    frame_dir: Path | None
    video_path: Path | None
    frame_count: int
    ignored: bool


@dataclass(frozen=True)
class TimelineRange:
    chunk_name: str
    chunk_index: int
    source_frame_start: int
    source_frame_end: int
    compact_frame_start: int
    compact_frame_end: int
    original_start_sec: float
    original_end_sec: float
    compact_start_sec: float
    compact_end_sec: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gather EgoCom chunked modalities into compact full-sequence outputs.",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--chunk-root", default="1min")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--chunk-sec", type=float, default=60.0)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--ignore-video-chunk-root", type=Path, default=None)
    parser.add_argument("--link-mode", choices=("hardlink", "symlink", "copy"), default="hardlink")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit-videos", type=int, default=None)
    parser.add_argument("--video", type=str, default=None, help="Optional original video/base name filter.")
    parser.add_argument(
        "--skip-video-concat",
        action="store_true",
        help="Skip ffmpeg video concatenation and only write timelines for the video modality.",
    )
    parser.add_argument(
        "--strict-torch-load",
        action="store_true",
        help="Do not fall back to pickle-based torch.load when weights_only loading fails.",
    )
    return parser.parse_args()


def split_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_chunk_stem(stem: str) -> tuple[str, int] | None:
    match = CHUNK_RE.match(stem)
    if match is None:
        return None
    return match.group("base"), int(match.group("chunk"))


def chunk_name(base_video: str, chunk_index: int) -> str:
    return f"{base_video}_chunk_{chunk_index:04d}"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    ensure_parent(path)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)


def list_frame_files(frame_dir: Path | None) -> list[Path]:
    if frame_dir is None or not frame_dir.is_dir():
        return []
    return sorted(
        path
        for path in frame_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def safe_json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return safe_json_value(value.item())
        return value.tolist()
    if torch.is_tensor(value):
        if value.ndim == 0:
            return safe_json_value(value.item())
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): safe_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json_value(item) for item in value]
    return value


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
        getattr(getattr(np, "dtypes", object), "Float32DType", None),
        getattr(getattr(np, "dtypes", object), "Int64DType", None),
        getattr(getattr(np, "dtypes", object), "Int32DType", None),
    ):
        if value is not None:
            safe.append(value)
    return safe


def load_torch(path: Path, strict: bool = False) -> Any:
    try:
        with torch.serialization.safe_globals(get_safe_numpy_globals()):
            return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        if strict:
            raise
        return torch.load(path, map_location="cpu", weights_only=False)


def load_json(path: Path) -> Any:
    with path.open("r") as handle:
        return json.load(handle)


def read_ignore_chunks(data_root: Path, ignore_root: Path | None, split: str) -> tuple[set[str], Path]:
    root = ignore_root or (data_root / "ignore_video_chunks")
    path = root / f"{split}.txt"
    if not path.is_file():
        return set(), path
    ignored: set[str] = set()
    with path.open("r") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            ignored.add(Path(line).stem)
    return ignored, path


def discover_video_path(video_root: Path, chunk_stem: str) -> Path | None:
    for suffix in VIDEO_SUFFIXES:
        candidate = video_root / f"{chunk_stem}{suffix}"
        if candidate.is_file():
            return candidate
    matches = sorted(path for path in video_root.glob(f"{chunk_stem}.*") if path.suffix in VIDEO_SUFFIXES)
    return matches[0] if matches else None


def discover_split_chunks(
    split_root: Path,
    ignored_chunks: set[str],
) -> dict[str, list[ChunkRecord]]:
    discovered: dict[str, dict[int, ChunkRecord]] = defaultdict(dict)
    frame_root = split_root / "frame"
    video_root = split_root / "video"
    stems: set[str] = set()

    if frame_root.is_dir():
        stems.update(path.name for path in frame_root.iterdir() if path.is_dir())
    if video_root.is_dir():
        stems.update(path.stem for path in video_root.iterdir() if path.is_file())

    for stem in sorted(stems):
        parsed = parse_chunk_stem(stem)
        if parsed is None:
            continue
        base_video, chunk_index = parsed
        frame_dir = frame_root / stem if (frame_root / stem).is_dir() else None
        video_path = discover_video_path(video_root, stem) if video_root.is_dir() else None
        discovered[base_video][chunk_index] = ChunkRecord(
            base_video=base_video,
            chunk_index=chunk_index,
            name=stem,
            split_root=split_root,
            frame_dir=frame_dir,
            video_path=video_path,
            frame_count=len(list_frame_files(frame_dir)),
            ignored=stem in ignored_chunks,
        )

    return {
        base_video: [items[index] for index in sorted(items)]
        for base_video, items in sorted(discovered.items())
    }


def expected_chunk_indices(chunks: list[ChunkRecord]) -> list[int]:
    if not chunks:
        return []
    return list(range(min(chunk.chunk_index for chunk in chunks), max(chunk.chunk_index for chunk in chunks) + 1))


def chunk_duration_sec(chunk: ChunkRecord, chunk_sec: float, fps: float) -> float:
    if chunk.frame_count > 0 and fps > 0:
        return chunk.frame_count / fps
    return chunk_sec


def original_ranges_by_chunk(
    all_chunks: list[ChunkRecord],
    chunk_sec: float,
    fps: float,
) -> dict[int, tuple[int, int, float, float]]:
    ranges: dict[int, tuple[int, int, float, float]] = {}
    frame_cursor = 0
    sec_cursor = 0.0
    for chunk in sorted(all_chunks, key=lambda item: item.chunk_index):
        duration_sec = chunk_duration_sec(chunk, chunk_sec, fps)
        frame_len = chunk.frame_count if chunk.frame_count > 0 else int(round(duration_sec * fps))
        ranges[chunk.chunk_index] = (frame_cursor, frame_cursor + frame_len, sec_cursor, sec_cursor + duration_sec)
        frame_cursor += frame_len
        sec_cursor += duration_sec
    return ranges


def build_ranges(
    all_chunks: list[ChunkRecord],
    included_chunks: list[ChunkRecord],
    chunk_sec: float,
    fps: float,
) -> list[TimelineRange]:
    ranges: list[TimelineRange] = []
    original_ranges = original_ranges_by_chunk(all_chunks, chunk_sec, fps)
    compact_frame = 0
    for chunk in included_chunks:
        source_start, source_end, original_start, original_end = original_ranges.get(
            chunk.chunk_index,
            (0, chunk.frame_count, 0.0, chunk_duration_sec(chunk, chunk_sec, fps)),
        )
        compact_start = compact_frame
        compact_end = compact_start + chunk.frame_count
        ranges.append(
            TimelineRange(
                chunk_name=chunk.name,
                chunk_index=chunk.chunk_index,
                source_frame_start=source_start,
                source_frame_end=source_end,
                compact_frame_start=compact_start,
                compact_frame_end=compact_end,
                original_start_sec=original_start,
                original_end_sec=original_end,
                compact_start_sec=compact_start / fps,
                compact_end_sec=compact_end / fps,
            )
        )
        compact_frame = compact_end
    return ranges


def modality_timeline(
    split: str,
    base_video: str,
    modality: str,
    source_root: Path,
    output_dir: Path,
    all_chunks: list[ChunkRecord],
    included_chunks: list[ChunkRecord],
    chunk_sec: float,
    fps: float,
    warnings: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    expected = expected_chunk_indices(all_chunks)
    seen = {chunk.chunk_index for chunk in all_chunks}
    ignored = [chunk.name for chunk in all_chunks if chunk.ignored]
    missing = [chunk_name(base_video, idx) for idx in expected if idx not in seen]
    ranges = build_ranges(all_chunks, included_chunks, chunk_sec, fps)
    expected_frames = int(round(chunk_sec * fps))
    timeline_warnings = list(warnings or [])
    for chunk in all_chunks:
        if chunk.frame_count and chunk.frame_count != expected_frames:
            timeline_warnings.append(
                f"{chunk.name}: frame_count={chunk.frame_count}, expected_regular_chunk_frames={expected_frames}"
            )
    payload: dict[str, Any] = {
        "split": split,
        "original_video_name": base_video,
        "modality": modality,
        "compact_mode": "compact_kept_chunks",
        "source_root": str(source_root),
        "output_dir": str(output_dir),
        "fps": float(fps),
        "chunk_sec": float(chunk_sec),
        "expected_chunk_indices": expected,
        "included_chunks": [chunk.name for chunk in included_chunks],
        "ignored_chunks": ignored,
        "missing_chunks": missing,
        "output_frame_count": int(sum(chunk.frame_count for chunk in included_chunks)),
        "expected_original_duration_sec": float(sum(chunk_duration_sec(chunk, chunk_sec, fps) for chunk in all_chunks)),
        "compact_duration_sec": float(sum(chunk.frame_count for chunk in included_chunks) / fps if fps else 0.0),
        "ranges": [range_item.__dict__ for range_item in ranges],
        "warnings": sorted(set(timeline_warnings)),
    }
    if extra:
        payload.update(safe_json_value(extra))
    return payload


def reset_output_dir(path: Path, overwrite: bool, dry_run: bool) -> None:
    if dry_run:
        return
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists, pass --overwrite: {path}")
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    path.mkdir(parents=True, exist_ok=True)


def place_file(src: Path, dst: Path, link_mode: str, overwrite: bool, dry_run: bool) -> None:
    if dry_run:
        return
    ensure_parent(dst)
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            raise FileExistsError(f"Output already exists, pass --overwrite: {dst}")
        dst.unlink()
    if link_mode == "copy":
        shutil.copy2(src, dst)
    elif link_mode == "symlink":
        os.symlink(src, dst)
    else:
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def save_torch(path: Path, payload: Any, dry_run: bool) -> None:
    if dry_run:
        return
    ensure_parent(path)
    torch.save(payload, path)


def save_npz(path: Path, dry_run: bool, **payload: Any) -> None:
    if dry_run:
        return
    ensure_parent(path)
    np.savez(path, **payload)


def write_text(path: Path, content: str, dry_run: bool) -> None:
    if dry_run:
        return
    ensure_parent(path)
    path.write_text(content)


def gather_frames(
    split: str,
    base_video: str,
    split_root: Path,
    output_base: Path,
    all_chunks: list[ChunkRecord],
    included_chunks: list[ChunkRecord],
    args: argparse.Namespace,
) -> dict[str, Any]:
    modality = "frame"
    output_dir = output_base / modality
    reset_output_dir(output_dir, args.overwrite, args.dry_run)
    frame_no = 1
    warnings: list[str] = []
    for chunk in included_chunks:
        files = list_frame_files(chunk.frame_dir)
        if not files:
            warnings.append(f"{chunk.name}: no frame files")
            continue
        for frame_path in files:
            dst = output_dir / f"frame_{frame_no:06d}{frame_path.suffix.lower()}"
            place_file(frame_path, dst, args.link_mode, args.overwrite, args.dry_run)
            frame_no += 1
    timeline = modality_timeline(
        split,
        base_video,
        modality,
        split_root / modality,
        output_dir,
        all_chunks,
        included_chunks,
        args.chunk_sec,
        args.fps,
        warnings,
    )
    write_json(output_dir / "timeline.json", timeline) if not args.dry_run else None
    return {"modality": modality, "output": str(output_dir), "timeline": timeline}


def concat_videos(video_paths: list[Path], output_path: Path, overwrite: bool, dry_run: bool) -> None:
    if dry_run or not video_paths:
        return
    ensure_parent(output_path)
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists, pass --overwrite: {output_path}")
        output_path.unlink()
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
        list_path = Path(handle.name)
        for video_path in video_paths:
            escaped = str(video_path).replace("'", "'\\''")
            handle.write(f"file '{escaped}'\n")
    try:
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y" if overwrite else "-n",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            "-c",
            "copy",
            str(output_path),
        ]
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError:
            command[-3:] = ["-c:v", "libx264", str(output_path)]
            subprocess.run(command, check=True)
    finally:
        list_path.unlink(missing_ok=True)


def gather_video(
    split: str,
    base_video: str,
    split_root: Path,
    output_base: Path,
    all_chunks: list[ChunkRecord],
    included_chunks: list[ChunkRecord],
    args: argparse.Namespace,
) -> dict[str, Any]:
    modality = "video"
    output_dir = output_base / modality
    reset_output_dir(output_dir, args.overwrite, args.dry_run)
    warnings: list[str] = []
    video_paths: list[Path] = []
    for chunk in included_chunks:
        if chunk.video_path is None:
            warnings.append(f"{chunk.name}: missing video file")
        else:
            video_paths.append(chunk.video_path)
    if not args.skip_video_concat and video_paths:
        concat_videos(video_paths, output_dir / f"{base_video}.mp4", args.overwrite, args.dry_run)
    timeline = modality_timeline(
        split,
        base_video,
        modality,
        split_root / modality,
        output_dir,
        all_chunks,
        included_chunks,
        args.chunk_sec,
        args.fps,
        warnings,
        {"video_concat_skipped": bool(args.skip_video_concat), "source_video_paths": [str(path) for path in video_paths]},
    )
    write_json(output_dir / "timeline.json", timeline) if not args.dry_run else None
    return {"modality": modality, "output": str(output_dir), "timeline": timeline}


def gather_audio(
    split: str,
    base_video: str,
    output_base: Path,
    all_chunks: list[ChunkRecord],
    included_chunks: list[ChunkRecord],
    args: argparse.Namespace,
) -> dict[str, Any]:
    modality = "audio"
    output_dir = output_base / modality
    reset_output_dir(output_dir, args.overwrite, args.dry_run)
    source_root = args.data_root / "original" / split / "audio"
    source = source_root / f"{base_video}.wav"
    output_path = output_dir / f"{base_video}.wav"
    warnings: list[str] = []
    written = False

    if not source.is_file():
        warnings.append(f"missing original audio: {source}")
    elif not args.dry_run:
        info = sf.info(source)
        chunks_audio: list[np.ndarray] = []
        sample_rate = int(info.samplerate)
        original_ranges = original_ranges_by_chunk(all_chunks, args.chunk_sec, args.fps)
        for chunk in included_chunks:
            _source_start_frame, _source_end_frame, start_sec, end_sec = original_ranges.get(
                chunk.chunk_index,
                (0, chunk.frame_count, 0.0, chunk_duration_sec(chunk, args.chunk_sec, args.fps)),
            )
            start_sample = int(round(start_sec * sample_rate))
            stop_sample = int(round(end_sec * sample_rate))
            if start_sample >= info.frames:
                warnings.append(f"{chunk.name}: audio start beyond source duration")
                continue
            if stop_sample > info.frames:
                warnings.append(f"{chunk.name}: audio stop clipped to source duration")
                stop_sample = info.frames
            data, sr = sf.read(source, start=start_sample, stop=stop_sample, dtype="float32", always_2d=True)
            if int(sr) != sample_rate:
                raise RuntimeError(f"Unexpected sample-rate change while reading {source}: {sr} vs {sample_rate}")
            chunks_audio.append(data)
        if chunks_audio:
            ensure_parent(output_path)
            audio = np.concatenate(chunks_audio, axis=0)
            sf.write(output_path, audio, sample_rate)
            written = True
        else:
            warnings.append("no included audio chunks were readable")

    timeline = modality_timeline(
        split,
        base_video,
        modality,
        source_root,
        output_dir,
        all_chunks,
        included_chunks,
        args.chunk_sec,
        args.fps,
        warnings,
        {
            "source_audio_path": str(source) if source.is_file() else None,
            "output_audio_path": str(output_path),
            "audio_compaction": "concatenate_kept_chunk_intervals",
            "audio_written": written,
        },
    )
    write_json(output_dir / "timeline.json", timeline) if not args.dry_run else None
    return {"modality": modality, "output": str(output_dir), "timeline": timeline}


def gather_da3(
    split: str,
    base_video: str,
    split_root: Path,
    output_base: Path,
    all_chunks: list[ChunkRecord],
    included_chunks: list[ChunkRecord],
    args: argparse.Namespace,
) -> dict[str, Any]:
    modality = "da3"
    source_root = split_root / modality
    output_dir = output_base / modality
    reset_output_dir(output_dir, args.overwrite, args.dry_run)
    warnings: list[str] = []

    frame_no = 1
    for subdir in ("depth", "vis"):
        frame_no = 1
        for chunk in included_chunks:
            chunk_dir = source_root / "monocular" / chunk.name / subdir
            files = sorted(path for path in chunk_dir.iterdir() if path.is_file()) if chunk_dir.is_dir() else []
            if not files:
                warnings.append(f"{chunk.name}: missing da3/monocular/{subdir}")
                continue
            for src in files:
                dst = output_dir / "monocular" / subdir / f"frame_{frame_no:06d}{src.suffix.lower()}"
                place_file(src, dst, args.link_mode, args.overwrite, args.dry_run)
                frame_no += 1

    camera_arrays: dict[str, list[np.ndarray]] = defaultdict(list)
    for chunk in included_chunks:
        camera_dir = source_root / "nested" / chunk.name / "camera_params"
        if not camera_dir.is_dir():
            warnings.append(f"{chunk.name}: missing da3/nested/camera_params")
            continue
        for npy in sorted(camera_dir.glob("*.npy")):
            camera_arrays[npy.name].append(np.load(npy))
    for name, arrays in sorted(camera_arrays.items()):
        try:
            merged = np.concatenate(arrays, axis=0)
        except ValueError:
            warnings.append(f"{name}: could not concatenate camera arrays; copied first array only")
            merged = arrays[0]
        if not args.dry_run:
            out_path = output_dir / "nested" / "camera_params" / name
            ensure_parent(out_path)
            np.save(out_path, merged)

    timeline = modality_timeline(
        split,
        base_video,
        modality,
        source_root,
        output_dir,
        all_chunks,
        included_chunks,
        args.chunk_sec,
        args.fps,
        warnings,
    )
    write_json(output_dir / "timeline.json", timeline) if not args.dry_run else None
    return {"modality": modality, "output": str(output_dir), "timeline": timeline}


def offset_mask_dict(raw: Any, compact_offset: int) -> dict[int, Any]:
    if not isinstance(raw, dict):
        raise TypeError(f"Expected mask dict, got {type(raw)}")
    out: dict[int, Any] = {}
    for frame_idx_raw, persons in raw.items():
        out[int(frame_idx_raw) + compact_offset] = persons
    return out


def gather_mask_root(
    split: str,
    base_video: str,
    split_root: Path,
    output_base: Path,
    modality: str,
    source_filename: str,
    args: argparse.Namespace,
    all_chunks: list[ChunkRecord],
    included_chunks: list[ChunkRecord],
) -> dict[str, Any]:
    source_root = split_root / modality
    output_dir = output_base / modality
    reset_output_dir(output_dir, args.overwrite, args.dry_run)
    warnings: list[str] = []
    merged: dict[int, Any] = {}
    compact_offset = 0
    source_summaries: list[dict[str, Any]] = []
    for chunk in included_chunks:
        chunk_dir = source_root / chunk.name
        source_path = chunk_dir / source_filename
        if not source_path.is_file():
            warnings.append(f"{chunk.name}: missing {source_filename}")
            compact_offset += chunk.frame_count
            continue
        if not args.dry_run:
            raw = load_torch(source_path, strict=args.strict_torch_load)
            merged.update(offset_mask_dict(raw, compact_offset))
        for sidecar in ("mask_meta.json", "summary.json"):
            sidecar_path = chunk_dir / sidecar
            if sidecar_path.is_file():
                try:
                    source_summaries.append({"chunk": chunk.name, "file": sidecar, "payload": load_json(sidecar_path)})
                except Exception as exc:
                    warnings.append(f"{chunk.name}: failed to read {sidecar}: {exc}")
        compact_offset += chunk.frame_count
    save_torch(output_dir / source_filename, merged, args.dry_run)
    timeline = modality_timeline(
        split,
        base_video,
        modality,
        source_root,
        output_dir,
        all_chunks,
        included_chunks,
        args.chunk_sec,
        args.fps,
        warnings,
        {"source_summaries": source_summaries},
    )
    write_json(output_dir / "timeline.json", timeline) if not args.dry_run else None
    return {"modality": modality, "output": str(output_dir), "timeline": timeline}


def gather_chunk_files(
    split: str,
    base_video: str,
    split_root: Path,
    output_base: Path,
    modality: str,
    args: argparse.Namespace,
    all_chunks: list[ChunkRecord],
    included_chunks: list[ChunkRecord],
) -> dict[str, Any]:
    source_root = split_root / modality
    output_dir = output_base / modality
    reset_output_dir(output_dir, args.overwrite, args.dry_run)
    warnings: list[str] = []
    copied: list[str] = []
    for chunk in included_chunks:
        chunk_dir = source_root / chunk.name
        if not chunk_dir.is_dir():
            warnings.append(f"{chunk.name}: missing {modality} directory")
            continue
        for src in sorted(path for path in chunk_dir.rglob("*") if path.is_file()):
            rel = src.relative_to(chunk_dir)
            dst = output_dir / "chunks" / chunk.name / rel
            place_file(src, dst, args.link_mode, args.overwrite, args.dry_run)
            copied.append(str(dst.relative_to(output_dir)))
    timeline = modality_timeline(
        split,
        base_video,
        modality,
        source_root,
        output_dir,
        all_chunks,
        included_chunks,
        args.chunk_sec,
        args.fps,
        warnings,
        {"copied_files": copied},
    )
    write_json(output_dir / "timeline.json", timeline) if not args.dry_run else None
    return {"modality": modality, "output": str(output_dir), "timeline": timeline}


def gather_person_face_emb(
    split: str,
    base_video: str,
    split_root: Path,
    output_base: Path,
    args: argparse.Namespace,
    all_chunks: list[ChunkRecord],
    included_chunks: list[ChunkRecord],
) -> dict[str, Any]:
    modality = "person_face_emb"
    source_root = split_root / modality
    output_dir = output_base / modality
    reset_output_dir(output_dir, args.overwrite, args.dry_run)
    warnings: list[str] = []
    merged: dict[str, Any] = {
        "split": split,
        "video_name": base_video,
        "layout": "chunk_namespaced_local_segments",
        "chunks": {},
    }
    chunk_offsets = {
        item.chunk_index: rng.compact_frame_start
        for item, rng in zip(included_chunks, build_ranges(all_chunks, included_chunks, args.chunk_sec, args.fps))
    }

    for chunk in included_chunks:
        chunk_dir = source_root / chunk.name
        emb_path = chunk_dir / "embeding.pt"
        summary_path = chunk_dir / "summary.json"
        if not emb_path.is_file():
            warnings.append(f"{chunk.name}: missing embeding.pt")
            continue
        if args.dry_run:
            merged["chunks"][chunk.name] = {"source_embedding_path": str(emb_path)}
            continue
        raw = load_torch(emb_path, strict=args.strict_torch_load)
        if not isinstance(raw, dict):
            warnings.append(f"{emb_path}: expected dict payload, got {type(raw)}")
            continue
        compact_offset = int(chunk_offsets.get(chunk.chunk_index, 0))
        chunk_payload: dict[str, Any] = {
            "source_embedding_path": str(emb_path),
            "compact_frame_offset": compact_offset,
            "segments": {},
        }
        for segment_id, payload in raw.items():
            if not isinstance(payload, dict):
                chunk_payload["segments"][str(segment_id)] = payload
                continue
            item = dict(payload)
            frame_indices = item.get("frame_indices")
            if isinstance(frame_indices, list):
                item["frame_indices"] = [int(value) + compact_offset for value in frame_indices]
                item["source_frame_indices"] = [int(value) for value in frame_indices]
            chunk_payload["segments"][str(segment_id)] = item
        if summary_path.is_file():
            try:
                chunk_payload["summary"] = load_json(summary_path)
            except Exception as exc:
                warnings.append(f"{summary_path}: failed to load summary: {exc}")
        merged["chunks"][chunk.name] = chunk_payload

    save_torch(output_dir / "embeding.pt", merged, args.dry_run)
    timeline = modality_timeline(
        split,
        base_video,
        modality,
        source_root,
        output_dir,
        all_chunks,
        included_chunks,
        args.chunk_sec,
        args.fps,
        warnings,
        {"output_embedding_path": str(output_dir / "embeding.pt")},
    )
    write_json(output_dir / "timeline.json", timeline) if not args.dry_run else None
    return {"modality": modality, "output": str(output_dir), "timeline": timeline}


def is_sequence_value(value: Any, expected_len: int) -> bool:
    if torch.is_tensor(value):
        return value.ndim >= 1 and int(value.shape[0]) == expected_len
    if isinstance(value, np.ndarray):
        return value.ndim >= 1 and int(value.shape[0]) == expected_len
    if isinstance(value, list):
        return len(value) == expected_len
    return False


def concat_values(values: list[Any]) -> Any:
    values = [value for value in values if value is not None]
    if not values:
        return None
    first = values[0]
    if torch.is_tensor(first):
        return torch.cat([value.detach().cpu() for value in values], dim=0)
    if isinstance(first, np.ndarray):
        return np.concatenate([np.asarray(value) for value in values], axis=0)
    out: list[Any] = []
    for value in values:
        out.extend(list(value))
    return out


def compact_frame_indices(lengths: list[int]) -> list[int]:
    total = sum(lengths)
    return list(range(total))


def merge_feature_payloads(
    chunks: list[tuple[ChunkRecord, Path, dict[str, Any]]],
    base_video: str,
    modality: str,
    strict: bool,
) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    lengths: list[int] = []
    for chunk, path, payload in chunks:
        if "features" in payload and torch.is_tensor(payload["features"]):
            lengths.append(int(payload["features"].shape[0]))
        elif "features" in payload and isinstance(payload["features"], np.ndarray):
            lengths.append(int(payload["features"].shape[0]))
        else:
            frame_indices = payload.get("frame_indices")
            if isinstance(frame_indices, list):
                lengths.append(len(frame_indices))
            else:
                warnings.append(f"{chunk.name}: cannot infer sequence length for {path.name}")
                lengths.append(chunk.frame_count)

    merged: dict[str, Any] = {}
    keys = sorted({key for _chunk, _path, payload in chunks for key in payload})
    for key in keys:
        if key in {"video_name", "source_frame_paths", "source_mask_path", "source_mapping_path", "text_path"}:
            continue
        values: list[Any] = []
        can_concat = True
        for (_chunk, _path, payload), length in zip(chunks, lengths):
            value = payload.get(key)
            if key == "frame_indices":
                values.append(list(range(length)))
            elif key == "frame_stems":
                values.append(list(value) if isinstance(value, list) else [""] * length)
            elif key == "source_chunk_index":
                values.append([_chunk.chunk_index] * length)
            elif is_sequence_value(value, length):
                values.append(value)
            elif key in FRAME_LIKE_KEYS:
                can_concat = False
                break
            else:
                if key not in merged:
                    merged[key] = safe_json_value(value)
        if values and can_concat:
            merged[key] = concat_values(values)

    merged["video_name"] = base_video
    merged["source_video_names"] = [chunk.name for chunk, _path, _payload in chunks]
    merged["source_paths"] = [str(path) for _chunk, path, _payload in chunks]
    source_chunk_index: list[int] = []
    for (chunk, _path, _payload), length in zip(chunks, lengths):
        source_chunk_index.extend([chunk.chunk_index] * length)
    merged["source_chunk_index"] = source_chunk_index
    merged["frame_indices"] = compact_frame_indices(lengths)
    merged["num_frames"] = int(sum(lengths))
    merged["gather_modality"] = modality
    return merged, warnings


def discover_person_chunk_files(source_root: Path, base_video: str, suffix: str) -> dict[tuple[str, str], list[tuple[int, Path]]]:
    grouped: dict[tuple[str, str], list[tuple[int, Path]]] = defaultdict(list)
    if not source_root.is_dir():
        return {}
    for person_dir in sorted(source_root.glob("*/person_*")):
        if not person_dir.is_dir():
            continue
        scene_key = person_dir.parent.name
        person_name = person_dir.name
        for path in sorted(person_dir.glob(f"{base_video}_chunk_*{suffix}")):
            parsed = parse_chunk_stem(path.stem)
            if parsed is None:
                continue
            parsed_base, chunk_index = parsed
            if parsed_base != base_video:
                continue
            grouped[(scene_key, person_name)].append((chunk_index, path))
    for paths in grouped.values():
        paths.sort(key=lambda item: item[0])
    return grouped


def included_chunk_by_index(included_chunks: list[ChunkRecord]) -> dict[int, ChunkRecord]:
    return {chunk.chunk_index: chunk for chunk in included_chunks}


def gather_person_pt_modality(
    split: str,
    base_video: str,
    split_root: Path,
    output_base: Path,
    modality: str,
    args: argparse.Namespace,
    all_chunks: list[ChunkRecord],
    included_chunks: list[ChunkRecord],
) -> dict[str, Any]:
    source_root = split_root / modality
    output_dir = output_base / modality
    reset_output_dir(output_dir, args.overwrite, args.dry_run)
    warnings: list[str] = []
    grouped = discover_person_chunk_files(source_root, base_video, ".pt")
    included_by_index = included_chunk_by_index(included_chunks)
    outputs: list[str] = []

    for (scene_key, person_name), indexed_paths in sorted(grouped.items()):
        chunks_with_payloads: list[tuple[ChunkRecord, Path, dict[str, Any]]] = []
        for chunk_index, path in indexed_paths:
            chunk = included_by_index.get(chunk_index)
            if chunk is None:
                continue
            if args.dry_run:
                chunks_with_payloads.append((chunk, path, {}))
                continue
            payload = load_torch(path, strict=args.strict_torch_load)
            if not isinstance(payload, dict):
                warnings.append(f"{path}: expected dict payload, got {type(payload)}")
                continue
            chunks_with_payloads.append((chunk, path, payload))
        if not chunks_with_payloads:
            continue
        out_path = output_dir / scene_key / person_name / f"{base_video}.pt"
        if not args.dry_run:
            merged, merge_warnings = merge_feature_payloads(
                chunks_with_payloads,
                base_video,
                modality,
                strict=args.strict_torch_load,
            )
            warnings.extend(merge_warnings)
            merged["scene_key"] = scene_key
            merged["person_id"] = int(person_name.split("_", 1)[1])
            save_torch(out_path, merged, args.dry_run)
        outputs.append(str(out_path.relative_to(output_dir)))

    timeline = modality_timeline(
        split,
        base_video,
        modality,
        source_root,
        output_dir,
        all_chunks,
        included_chunks,
        args.chunk_sec,
        args.fps,
        warnings,
        {"outputs": outputs},
    )
    write_json(output_dir / "timeline.json", timeline) if not args.dry_run else None
    return {"modality": modality, "output": str(output_dir), "timeline": timeline}


def gather_person_npz_modality(
    split: str,
    base_video: str,
    split_root: Path,
    output_base: Path,
    modality: str,
    args: argparse.Namespace,
    all_chunks: list[ChunkRecord],
    included_chunks: list[ChunkRecord],
) -> dict[str, Any]:
    source_root = split_root / modality
    output_dir = output_base / modality
    reset_output_dir(output_dir, args.overwrite, args.dry_run)
    warnings: list[str] = []
    grouped = discover_person_chunk_files(source_root, base_video, ".npz")
    included_by_index = included_chunk_by_index(included_chunks)
    outputs: list[str] = []

    for (scene_key, person_name), indexed_paths in sorted(grouped.items()):
        loaded: list[tuple[ChunkRecord, Path, dict[str, Any], int]] = []
        for chunk_index, path in indexed_paths:
            chunk = included_by_index.get(chunk_index)
            if chunk is None:
                continue
            if args.dry_run:
                loaded.append((chunk, path, {}, chunk.frame_count))
                continue
            with np.load(path, allow_pickle=False) as data:
                payload = {key: data[key] for key in data.files}
            length = int(np.asarray(payload.get("num_frames", chunk.frame_count)).item())
            loaded.append((chunk, path, payload, length))
        if not loaded:
            continue

        out_path = output_dir / scene_key / person_name / f"{base_video}.npz"
        if args.dry_run:
            outputs.append(str(out_path.relative_to(output_dir)))
            continue

        merged: dict[str, Any] = {}
        keys = sorted({key for _chunk, _path, payload, _length in loaded for key in payload})
        lengths = [length for _chunk, _path, _payload, length in loaded]
        for key in keys:
            values: list[Any] = []
            scalar_value: Any = None
            for chunk, _path, payload, length in loaded:
                value = payload.get(key)
                if key in {"video_name"}:
                    scalar_value = np.array(base_video)
                elif key in {"logical_frame_indices", "frame_indices"}:
                    values.append(np.arange(length, dtype=np.int32))
                elif isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == length:
                    values.append(value)
                elif key in FRAME_LIKE_KEYS:
                    values = []
                    break
                elif scalar_value is None:
                    scalar_value = value
            if values:
                merged[key] = np.concatenate(values, axis=0)
            elif scalar_value is not None:
                merged[key] = scalar_value
        merged["video_name"] = np.array(base_video)
        merged["num_frames"] = np.array(sum(lengths), dtype=np.int32)
        merged["source_chunk_index"] = np.asarray(
            [chunk.chunk_index for chunk, _path, _payload, length in loaded for _ in range(length)],
            dtype=np.int32,
        )
        merged["logical_frame_indices"] = np.arange(sum(lengths), dtype=np.int32)
        merged["source_paths"] = np.asarray([str(path) for _chunk, path, _payload, _length in loaded], dtype="<U512")
        save_npz(out_path, args.dry_run, **merged)
        outputs.append(str(out_path.relative_to(output_dir)))

    timeline = modality_timeline(
        split,
        base_video,
        modality,
        source_root,
        output_dir,
        all_chunks,
        included_chunks,
        args.chunk_sec,
        args.fps,
        warnings,
        {"outputs": outputs},
    )
    write_json(output_dir / "timeline.json", timeline) if not args.dry_run else None
    return {"modality": modality, "output": str(output_dir), "timeline": timeline}


def read_text_lines(path: Path) -> list[str]:
    return path.read_text().splitlines()


def gather_person_text_modality(
    split: str,
    base_video: str,
    split_root: Path,
    output_base: Path,
    modality: str,
    args: argparse.Namespace,
    all_chunks: list[ChunkRecord],
    included_chunks: list[ChunkRecord],
) -> dict[str, Any]:
    source_root = split_root / modality
    output_dir = output_base / modality
    reset_output_dir(output_dir, args.overwrite, args.dry_run)
    warnings: list[str] = []
    grouped = discover_person_chunk_files(source_root, base_video, ".json")
    included_by_index = included_chunk_by_index(included_chunks)
    outputs: list[str] = []

    for (scene_key, person_name), indexed_paths in sorted(grouped.items()):
        lines: list[str] = []
        metadata: dict[str, Any] = {}
        source_jsons: list[str] = []
        source_txts: list[str] = []
        frame_indices: list[int] = []
        frame_offset = 0
        for chunk_index, json_path in indexed_paths:
            chunk = included_by_index.get(chunk_index)
            if chunk is None:
                continue
            try:
                payload = load_json(json_path)
            except Exception as exc:
                warnings.append(f"{json_path}: failed to load json: {exc}")
                continue
            if not metadata:
                metadata = {
                    key: value
                    for key, value in payload.items()
                    if key not in FRAME_LIKE_KEYS and key not in {"video_name", "text_path", "visualization_paths"}
                }
            txt_path = json_path.with_suffix(".txt")
            chunk_lines = read_text_lines(txt_path) if txt_path.is_file() else []
            if not chunk_lines:
                warnings.append(f"{json_path}: missing or empty txt sidecar")
            lines.extend(chunk_lines)
            frame_indices.extend(range(frame_offset, frame_offset + len(chunk_lines)))
            frame_offset += len(chunk_lines)
            source_jsons.append(str(json_path))
            if txt_path.is_file():
                source_txts.append(str(txt_path))
        if not lines and not source_jsons:
            continue
        out_dir = output_dir / scene_key / person_name
        write_text(out_dir / f"{base_video}.txt", "\n".join(lines) + ("\n" if lines else ""), args.dry_run)
        metadata.update(
            {
                "split": split,
                "scene_key": scene_key,
                "video_name": base_video,
                "person_id": int(person_name.split("_", 1)[1]),
                "num_lines": len(lines),
                "frame_indices": frame_indices,
                "source_json_paths": source_jsons,
                "source_text_paths": source_txts,
                "text_path": str(out_dir / f"{base_video}.txt"),
            }
        )
        write_json(out_dir / f"{base_video}.json", safe_json_value(metadata)) if not args.dry_run else None
        outputs.append(str((out_dir / f"{base_video}.json").relative_to(output_dir)))

    timeline = modality_timeline(
        split,
        base_video,
        modality,
        source_root,
        output_dir,
        all_chunks,
        included_chunks,
        args.chunk_sec,
        args.fps,
        warnings,
        {"outputs": outputs},
    )
    write_json(output_dir / "timeline.json", timeline) if not args.dry_run else None
    return {"modality": modality, "output": str(output_dir), "timeline": timeline}


def gather_one_video(
    split: str,
    base_video: str,
    split_root: Path,
    chunks: list[ChunkRecord],
    args: argparse.Namespace,
) -> dict[str, Any]:
    output_base = (args.output_root or (args.data_root / "gathering")) / split / base_video
    included_chunks = [chunk for chunk in chunks if not chunk.ignored]
    results: list[dict[str, Any]] = []

    results.append(gather_frames(split, base_video, split_root, output_base, chunks, included_chunks, args))
    results.append(gather_video(split, base_video, split_root, output_base, chunks, included_chunks, args))
    results.append(gather_audio(split, base_video, output_base, chunks, included_chunks, args))
    results.append(gather_da3(split, base_video, split_root, output_base, chunks, included_chunks, args))
    results.append(
        gather_mask_root(
            split,
            base_video,
            split_root,
            output_base,
            "person_mask",
            "masks.pt",
            args,
            chunks,
            included_chunks,
        )
    )
    results.append(
        gather_mask_root(
            split,
            base_video,
            split_root,
            output_base,
            "refined_mask",
            "mask.pt",
            args,
            chunks,
            included_chunks,
        )
    )
    results.append(gather_person_face_emb(split, base_video, split_root, output_base, args, chunks, included_chunks))

    for modality in (
        "person_visual_clip_features",
        "person_masked_clip_features",
        "person_masked_da3_features",
        "person_pe_features",
        "person_spatial_t5_features",
    ):
        results.append(gather_person_pt_modality(split, base_video, split_root, output_base, modality, args, chunks, included_chunks))

    results.append(gather_person_npz_modality(split, base_video, split_root, output_base, "person_depth_lift", args, chunks, included_chunks))
    results.append(
        gather_person_text_modality(
            split,
            base_video,
            split_root,
            output_base,
            "person_spatial_internvl2_text",
            args,
            chunks,
            included_chunks,
        )
    )

    summary = {
        "split": split,
        "original_video_name": base_video,
        "output_dir": str(output_base),
        "dry_run": bool(args.dry_run),
        "num_source_chunks": len(chunks),
        "num_included_chunks": len(included_chunks),
        "ignored_chunks": [chunk.name for chunk in chunks if chunk.ignored],
        "modalities": results,
    }
    if not args.dry_run:
        write_json(output_base / "summary.json", safe_json_value(summary))
    return summary


def iter_limited(items: Iterable[tuple[str, list[ChunkRecord]]], limit: int | None) -> Iterable[tuple[str, list[ChunkRecord]]]:
    for index, item in enumerate(items):
        if limit is not None and index >= limit:
            break
        yield item


def main() -> None:
    args = parse_args()
    args.output_root = args.output_root or (args.data_root / "gathering")
    all_summaries: list[dict[str, Any]] = []

    for split in split_list(args.splits):
        split_root = args.data_root / args.chunk_root / split
        if not split_root.is_dir():
            print(f"[WARN] missing split root: {split_root}")
            continue
        ignored, ignore_path = read_ignore_chunks(args.data_root, args.ignore_video_chunk_root, split)
        grouped = discover_split_chunks(split_root, ignored)
        if args.video:
            grouped = {key: value for key, value in grouped.items() if key == args.video}
        print(
            f"[INFO] {split}: videos={len(grouped)} ignored_chunks={len(ignored)} "
            f"ignore_list={ignore_path}"
        )
        for base_video, chunks in iter_limited(grouped.items(), args.limit_videos):
            print(
                f"[INFO] gather {split}/{base_video}: "
                f"chunks={len(chunks)} included={sum(1 for chunk in chunks if not chunk.ignored)}"
            )
            summary = gather_one_video(split, base_video, split_root, chunks, args)
            all_summaries.append(summary)

    root_summary = {
        "output_root": str(args.output_root),
        "chunk_root": args.chunk_root,
        "splits": split_list(args.splits),
        "dry_run": bool(args.dry_run),
        "num_videos": len(all_summaries),
        "videos": [
            {
                "split": item["split"],
                "original_video_name": item["original_video_name"],
                "num_source_chunks": item["num_source_chunks"],
                "num_included_chunks": item["num_included_chunks"],
                "ignored_chunks": item["ignored_chunks"],
                "output_dir": item["output_dir"],
            }
            for item in all_summaries
        ],
    }
    if not args.dry_run:
        write_json(args.output_root / "summary.json", root_summary)
    print(f"[DONE] videos={len(all_summaries)} output_root={args.output_root} dry_run={args.dry_run}")


if __name__ == "__main__":
    main()
