"""
Filter EgoCom SAM person masks using DA3 monocular depth discontinuity.

Inputs:
  /home/prj/data/egocom_holdout/1min/{split}/person_mask/{video_name}/masks.pt
  /home/prj/data/egocom_holdout/1min/{split}/da3/monocular/{video_name}/depth/*.npy
  /home/prj/data/egocom_holdout/1min/{split}/frame/{video_name}/*.jpg

Outputs:
  /home/prj/data/egocom_holdout/1min/{split}/refined_mask/{video_name}/mask.pt
  /home/prj/data/egocom_holdout/1min/{split}/refined_mask/{video_name}/summary.json
  /home/prj/data/egocom_holdout/1min/{split}/refined_mask/{video_name}/vis/*.jpg
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


DEFAULT_DATA_ROOT = "/home/prj/data/egocom_holdout/1min"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
PALETTE_BGR = [
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 0),
]


@dataclass(frozen=True)
class VideoJob:
    split: str
    video_name: str
    mask_path: Path
    frame_dir: Path
    depth_dir: Path
    output_dir: Path


def parse_comma_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


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


def depth_to_colormap(depth_map: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth_map)
    if not np.any(valid):
        normalized = np.zeros(depth_map.shape, dtype=np.uint8)
    else:
        depth_valid = depth_map[valid].astype(np.float32)
        low, high = np.percentile(depth_valid, [2, 98])
        if high <= low:
            high = low + 1e-6
        clipped = np.clip(depth_map.astype(np.float32), low, high)
        normalized = ((clipped - low) / (high - low) * 255.0).astype(np.uint8)
        normalized[~valid] = 0
    return cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)


def depth_discontinuity_metrics(
    depth_map: np.ndarray,
    mask: np.ndarray,
    kernel: np.ndarray,
    min_mask_pixels: int,
    min_ring_pixels: int,
    local_edge_thresh: float,
    min_edge_fraction: float,
    boundary_quantile: float,
) -> dict[str, float] | None:
    depth_mask = resize_mask(mask, depth_map.shape[:2])
    if int(depth_mask.sum()) < min_mask_pixels:
        return None

    dilated = cv2.dilate(depth_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    ring = dilated & ~depth_mask
    if int(ring.sum()) < min_ring_pixels:
        return None

    finite = np.isfinite(depth_map)
    inside_values = depth_map[depth_mask & finite]
    ring_values = depth_map[ring & finite]
    if len(inside_values) < min_mask_pixels or len(ring_values) < min_ring_pixels:
        return None

    inside_median = float(np.median(inside_values))
    ring_median = float(np.median(ring_values))
    median_relative_diff = abs(inside_median - ring_median) / max(abs(ring_median), 1e-6)

    # Median-vs-ring is brittle when a person touches a similar-depth object,
    # e.g. sitting on a couch. Use a high quantile of outer-ring contrast too:
    # if any meaningful boundary portion sees different depth, keep the mask.
    outer_relative_to_inside = np.abs(ring_values - inside_median) / max(abs(inside_median), 1e-6)
    edge_fraction = float(np.mean(outer_relative_to_inside >= local_edge_thresh))
    outer_tail_score = float(np.percentile(outer_relative_to_inside, boundary_quantile))
    robust_boundary_score = outer_tail_score if edge_fraction >= min_edge_fraction else 0.0
    discontinuity_score = max(median_relative_diff, robust_boundary_score)

    return {
        "depth_discontinuity_score": float(discontinuity_score),
        "median_relative_depth_diff": float(median_relative_diff),
        "outer_tail_relative_depth_diff": float(outer_tail_score),
        "outer_edge_fraction": float(edge_fraction),
        "inside_median_depth": float(inside_median),
        "outer_median_depth": float(ring_median),
    }


def segment_ids(mask_dict: dict[int, dict[int, np.ndarray]]) -> list[int]:
    ids = set()
    for persons in mask_dict.values():
        ids.update(int(segment_id) for segment_id in persons)
    return sorted(ids)


def segment_frame_counts(mask_dict: dict[int, dict[int, np.ndarray]]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for persons in mask_dict.values():
        for segment_id in persons:
            counts[int(segment_id)] = counts.get(int(segment_id), 0) + 1
    return counts


def evaluate_frame_segments(
    mask_dict: dict[int, dict[int, np.ndarray]],
    frame_paths: list[Path],
    depth_dir: Path,
    args: argparse.Namespace,
) -> tuple[set[tuple[int, int]], list[dict[str, Any]], dict[int, dict[str, Any]], dict[str, Any]]:
    radius = max(0, int(args.dilate_radius))
    kernel_size = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    sample_every = max(1, int(args.sample_every))
    segment_stats: dict[int, dict[str, Any]] = {
        segment_id: {
            "segment_id": int(segment_id),
            "num_considered_frames": 0,
            "num_valid_frames": 0,
            "num_rejected_frames": 0,
            "scores": [],
            "median_scores": [],
            "outer_tail_scores": [],
            "outer_edge_fractions": [],
        }
        for segment_id in segment_ids(mask_dict)
    }
    rejected_instances: set[tuple[int, int]] = set()
    rejection_records: list[dict[str, Any]] = []
    missing_depth_frames = 0

    for frame_idx, frame_path in enumerate(frame_paths):
        if frame_idx % sample_every != 0:
            continue
        persons = mask_dict.get(frame_idx)
        if not persons:
            continue

        depth_path = depth_dir / f"{frame_path.stem}.npy"
        if not depth_path.exists():
            missing_depth_frames += 1
            continue
        depth_map = np.load(depth_path)

        for segment_id, mask in persons.items():
            stats = segment_stats.setdefault(
                int(segment_id),
                {
                    "segment_id": int(segment_id),
                    "num_considered_frames": 0,
                    "num_valid_frames": 0,
                    "num_rejected_frames": 0,
                    "scores": [],
                    "median_scores": [],
                    "outer_tail_scores": [],
                    "outer_edge_fractions": [],
                },
            )
            stats["num_considered_frames"] += 1
            metrics = depth_discontinuity_metrics(
                depth_map=depth_map,
                mask=mask,
                kernel=kernel,
                min_mask_pixels=args.min_mask_pixels,
                min_ring_pixels=args.min_ring_pixels,
                local_edge_thresh=args.local_edge_thresh,
                min_edge_fraction=args.min_edge_fraction,
                boundary_quantile=args.boundary_quantile,
            )
            if metrics is not None:
                score = float(metrics["depth_discontinuity_score"])
                stats["num_valid_frames"] += 1
                stats["scores"].append(score)
                stats["median_scores"].append(float(metrics["median_relative_depth_diff"]))
                stats["outer_tail_scores"].append(float(metrics["outer_tail_relative_depth_diff"]))
                stats["outer_edge_fractions"].append(float(metrics["outer_edge_fraction"]))
                if score < args.depth_diff_thresh:
                    rejected_instances.add((frame_idx, int(segment_id)))
                    stats["num_rejected_frames"] += 1
                    rejection_records.append(
                        {
                            "frame_idx": int(frame_idx),
                            "frame_stem": frame_path.stem,
                            "segment_id": int(segment_id),
                            **metrics,
                            "reason": "depth_diff_below_threshold",
                        }
                    )

    finalized_segment_stats: dict[int, dict[str, Any]] = {}
    rejected_scores_by_segment: dict[int, dict[str, float]] = {}
    for record in rejection_records:
        rejected_scores_by_segment.setdefault(int(record["segment_id"]), {})[
            str(record["frame_idx"])
        ] = float(record["depth_discontinuity_score"])

    for segment_id, raw_stats in segment_stats.items():
        scores = raw_stats.pop("scores")
        median_scores = raw_stats.pop("median_scores")
        outer_tail_scores = raw_stats.pop("outer_tail_scores")
        outer_edge_fractions = raw_stats.pop("outer_edge_fractions")
        median_score = float(np.median(scores)) if scores else None
        finalized_segment_stats[int(segment_id)] = {
            **raw_stats,
            "median_depth_discontinuity_score": median_score,
            "min_depth_discontinuity_score": float(np.min(scores)) if scores else None,
            "max_depth_discontinuity_score": float(np.max(scores)) if scores else None,
            "median_relative_depth_diff": float(np.median(median_scores)) if median_scores else None,
            "median_outer_tail_relative_depth_diff": (
                float(np.median(outer_tail_scores)) if outer_tail_scores else None
            ),
            "median_outer_edge_fraction": (
                float(np.median(outer_edge_fractions)) if outer_edge_fractions else None
            ),
            "rejected_frame_scores": rejected_scores_by_segment.get(int(segment_id), {}),
        }

    diagnostics = {"missing_depth_frames": missing_depth_frames}
    return rejected_instances, rejection_records, finalized_segment_stats, diagnostics


def refine_mask_dict(
    mask_dict: dict[int, dict[int, np.ndarray]],
    rejected_instances: set[tuple[int, int]],
) -> dict[int, dict[int, np.ndarray]]:
    refined: dict[int, dict[int, np.ndarray]] = {}
    for frame_idx, persons in mask_dict.items():
        kept = {
            int(segment_id): np.asarray(mask).astype(bool)
            for segment_id, mask in persons.items()
            if (int(frame_idx), int(segment_id)) not in rejected_instances
        }
        if kept:
            refined[int(frame_idx)] = kept
    return refined


def draw_rejected_overlay(
    frame: np.ndarray,
    depth_map: np.ndarray,
    mask: np.ndarray,
    segment_id: int,
    score: float | None,
) -> np.ndarray:
    depth_color = depth_to_colormap(depth_map)
    if depth_color.shape[:2] != frame.shape[:2]:
        depth_color = cv2.resize(
            depth_color,
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    canvas = cv2.addWeighted(frame, 0.55, depth_color, 0.45, 0.0)

    mask_bool = mask.astype(bool)
    color = np.array(PALETTE_BGR[segment_id % len(PALETTE_BGR)], dtype=np.uint8)
    canvas[mask_bool] = (
        canvas[mask_bool].astype(np.float32) * 0.45 + color.astype(np.float32) * 0.55
    ).astype(np.uint8)

    y_indices, x_indices = np.where(mask_bool)
    if len(y_indices) and len(x_indices):
        x_min, x_max = int(x_indices.min()), int(x_indices.max())
        y_min, y_max = int(y_indices.min()), int(y_indices.max())
        cv2.rectangle(canvas, (x_min, y_min), (x_max, y_max), color.tolist(), 2)
        score_text = "na" if score is None else f"{score:.4f}"
        label = f"rejected seg {segment_id} depth_diff {score_text}"
        cv2.putText(
            canvas,
            label,
            (x_min, max(24, y_min - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            label,
            (x_min, max(24, y_min - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color.tolist(),
            2,
            cv2.LINE_AA,
        )
    return canvas


def save_rejected_visualizations(
    rejected_instances: set[tuple[int, int]],
    stats: dict[int, dict[str, Any]],
    mask_dict: dict[int, dict[int, np.ndarray]],
    frame_paths: list[Path],
    depth_dir: Path,
    vis_dir: Path,
    vis_sample_every: int,
) -> int:
    vis_dir.mkdir(parents=True, exist_ok=True)
    for old_path in vis_dir.glob("*_rejected.jpg"):
        old_path.unlink()
    if not rejected_instances:
        return 0
    sample_every = max(1, int(vis_sample_every))
    saved = 0

    for frame_idx, frame_path in enumerate(frame_paths):
        if frame_idx % sample_every != 0:
            continue
        persons = mask_dict.get(frame_idx, {})
        present_rejected = sorted(
            segment_id
            for segment_id in persons
            if (int(frame_idx), int(segment_id)) in rejected_instances
        )
        if not present_rejected:
            continue

        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        depth_path = depth_dir / f"{frame_path.stem}.npy"
        if not depth_path.exists():
            continue
        depth_map = np.load(depth_path)

        for segment_id in present_rejected:
            score = None
            frame_rejections = stats.get(segment_id, {}).get("rejected_frame_scores", {})
            if isinstance(frame_rejections, dict):
                score = frame_rejections.get(str(frame_idx))
            overlay = draw_rejected_overlay(
                frame=frame,
                depth_map=depth_map,
                mask=persons[segment_id],
                segment_id=segment_id,
                score=score,
            )
            out_path = vis_dir / f"{frame_path.stem}_seg{segment_id}_rejected.jpg"
            if cv2.imwrite(str(out_path), overlay):
                saved += 1
    return saved


def discover_splits(data_root: Path, split_arg: str) -> list[str]:
    if split_arg != "all_existing":
        return parse_comma_list(split_arg)
    splits = []
    for split_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        if (split_dir / "person_mask").is_dir() and (split_dir / "da3" / "monocular").is_dir():
            splits.append(split_dir.name)
    return splits


def collect_jobs(args: argparse.Namespace) -> list[VideoJob]:
    data_root = Path(args.data_root)
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    jobs: list[VideoJob] = []
    video_filter = Path(args.video).stem if args.video else None
    for split in discover_splits(data_root, args.split):
        split_dir = data_root / split
        mask_root = split_dir / "person_mask"
        depth_root = split_dir / "da3" / "monocular"
        frame_root = split_dir / "frame"
        if not mask_root.is_dir() or not depth_root.is_dir():
            print(f"[SKIP] {split}: missing person_mask or da3/monocular")
            continue

        for mask_dir in sorted(path for path in mask_root.iterdir() if path.is_dir()):
            video_name = mask_dir.name
            if video_filter and video_name != video_filter:
                continue
            mask_path = mask_dir / "masks.pt"
            frame_dir = frame_root / video_name
            depth_dir = depth_root / video_name / "depth"
            if not mask_path.exists():
                print(f"[SKIP] {split}/{video_name}: missing {mask_path.name}")
                continue
            if not frame_dir.is_dir():
                print(f"[SKIP] {split}/{video_name}: missing frame directory")
                continue
            if not depth_dir.is_dir():
                print(f"[SKIP] {split}/{video_name}: missing depth directory")
                continue
            jobs.append(
                VideoJob(
                    split=split,
                    video_name=video_name,
                    mask_path=mask_path,
                    frame_dir=frame_dir,
                    depth_dir=depth_dir,
                    output_dir=split_dir / "refined_mask" / video_name,
                )
            )
    return jobs


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        json.dump(payload, file, indent=2)


def process_job(job: VideoJob, args: argparse.Namespace) -> dict[str, Any]:
    output_mask_path = job.output_dir / "mask.pt"
    summary_path = job.output_dir / "summary.json"
    if output_mask_path.exists() and not args.overwrite and not args.dry_run:
        print(f"[SKIP] {job.split}/{job.video_name}: refined mask exists")
        return {"status": "skipped", "reason": "existing_output", "video_name": job.video_name}

    frame_paths = list_frame_paths(job.frame_dir)
    if not frame_paths:
        raise ValueError(f"No frames found: {job.frame_dir}")

    mask_dict = load_mask_dict(job.mask_path)
    rejected_instances, rejection_records, segment_stats, diagnostics = evaluate_frame_segments(
        mask_dict, frame_paths, job.depth_dir, args
    )
    segments_with_rejections = sorted(
        {segment_id for _, segment_id in rejected_instances}
    )
    refined = refine_mask_dict(mask_dict, rejected_instances)
    original_counts = segment_frame_counts(mask_dict)
    remaining_counts = segment_frame_counts(refined)
    remaining_segments = sorted(remaining_counts)
    fully_removed_segments = [
        segment_id
        for segment_id in segment_ids(mask_dict)
        if segment_id not in remaining_counts
    ]
    per_segment_counts = {
        str(segment_id): {
            "original_frames": int(original_counts.get(segment_id, 0)),
            "rejected_frames": int(segment_stats.get(segment_id, {}).get("num_rejected_frames", 0)),
            "remaining_frames": int(remaining_counts.get(segment_id, 0)),
        }
        for segment_id in segment_ids(mask_dict)
    }
    original_segment_counts = {
        str(segment_id): int(original_counts.get(segment_id, 0))
        for segment_id in segment_ids(mask_dict)
    }
    rejected_segment_counts = {
        str(segment_id): int(segment_stats.get(segment_id, {}).get("num_rejected_frames", 0))
        for segment_id in segment_ids(mask_dict)
    }
    remaining_segment_counts = {
        str(segment_id): int(remaining_counts.get(segment_id, 0))
        for segment_id in segment_ids(mask_dict)
    }
    num_original_mask_instances = int(sum(original_counts.values()))
    num_remaining_mask_instances = int(sum(remaining_counts.values()))

    summary = {
        "split": job.split,
        "video_name": job.video_name,
        "person_frequency": original_segment_counts,
        "remaining_person_frequency": remaining_segment_counts,
        "rejected_person_frequency": rejected_segment_counts,
        "num_original_mask_instances": num_original_mask_instances,
        "num_rejected_mask_instances": len(rejection_records),
        "num_remaining_mask_instances": num_remaining_mask_instances,
        "original_segments": segment_ids(mask_dict),
        "remaining_segments": remaining_segments,
        "fully_removed_segments": fully_removed_segments,
        "source_mask_path": str(job.mask_path),
        "output_mask_path": str(output_mask_path),
        "frame_dir": str(job.frame_dir),
        "depth_dir": str(job.depth_dir),
        "num_frames": len(frame_paths),
        "num_original_frames_with_masks": len(mask_dict),
        "num_refined_frames_with_masks": len(refined),
        "segments_with_rejections": segments_with_rejections,
        "num_rejected_instances": len(rejection_records),
        "rejected_segments": rejection_records,
        "original_segment_counts": original_segment_counts,
        "rejected_segment_counts": rejected_segment_counts,
        "remaining_segment_counts": remaining_segment_counts,
        "per_segment_counts": per_segment_counts,
        "parameters": {
            "depth_diff_thresh": args.depth_diff_thresh,
            "local_edge_thresh": args.local_edge_thresh,
            "min_edge_fraction": args.min_edge_fraction,
            "boundary_quantile": args.boundary_quantile,
            "dilate_radius": args.dilate_radius,
            "min_mask_pixels": args.min_mask_pixels,
            "min_ring_pixels": args.min_ring_pixels,
            "sample_every": args.sample_every,
            "vis_sample_every": args.vis_sample_every,
        },
        "segment_stats": {str(key): value for key, value in segment_stats.items()},
    }
    if diagnostics:
        summary["diagnostics"] = diagnostics

    if args.dry_run:
        print(
            f"[DRY] {job.split}/{job.video_name}: "
            f"reject_instances={len(rejection_records)} "
            f"remaining_segments={remaining_segments}"
        )
        return {"status": "dry_run", **summary}

    job.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(refined, output_mask_path)
    vis_saved = save_rejected_visualizations(
        rejected_instances=rejected_instances,
        stats=segment_stats,
        mask_dict=mask_dict,
        frame_paths=frame_paths,
        depth_dir=job.depth_dir,
        vis_dir=job.output_dir / "vis",
        vis_sample_every=args.vis_sample_every,
    )
    summary["rejected_visualizations_saved"] = vis_saved
    write_json(summary_path, summary)

    print(
        f"[OK] {job.split}/{job.video_name}: "
        f"rejected_instances={len(rejection_records)} "
        f"remaining_segments={remaining_segments} vis={vis_saved}"
    )
    return {"status": "processed", **summary}


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Filter person mask segments whose depth does not differ from the surrounding ring."
    )
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--split",
        type=str,
        default="all_existing",
        help="Comma-separated splits, or all_existing to scan splits with masks and DA3 depth.",
    )
    parser.add_argument("--video", type=str, default=None, help="Optional single video name.")
    parser.add_argument("--depth_diff_thresh", type=float, default=0.06)
    parser.add_argument(
        "--local_edge_thresh",
        type=float,
        default=0.08,
        help="Relative depth difference that counts as local boundary edge evidence.",
    )
    parser.add_argument(
        "--min_edge_fraction",
        type=float,
        default=0.08,
        help="Minimum outer-ring pixel fraction that must show local edge evidence.",
    )
    parser.add_argument(
        "--boundary_quantile",
        type=float,
        default=90.0,
        help="Percentile of outer-ring relative depth contrast used for robust boundary evidence.",
    )
    parser.add_argument("--dilate_radius", type=nonnegative_int, default=5)
    parser.add_argument("--min_mask_pixels", type=positive_int, default=25)
    parser.add_argument("--min_ring_pixels", type=positive_int, default=25)
    parser.add_argument("--sample_every", type=positive_int, default=1)
    parser.add_argument(
        "--vis_sample_every",
        type=positive_int,
        default=1,
        help="Save every Nth rejected-frame visualization. Default 1 saves all rejected masks.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.depth_diff_thresh < 0:
        raise ValueError(f"--depth_diff_thresh must be >= 0, got {args.depth_diff_thresh}")
    if args.local_edge_thresh < 0:
        raise ValueError(f"--local_edge_thresh must be >= 0, got {args.local_edge_thresh}")
    if not 0.0 <= args.min_edge_fraction <= 1.0:
        raise ValueError(f"--min_edge_fraction must be in [0, 1], got {args.min_edge_fraction}")
    if not 0.0 <= args.boundary_quantile <= 100.0:
        raise ValueError(f"--boundary_quantile must be in [0, 100], got {args.boundary_quantile}")

    jobs = collect_jobs(args)
    if not jobs:
        print("No matching videos found.")
        return

    print(f"Found {len(jobs)} videos")
    processed = 0
    skipped = 0
    failed = 0
    for index, job in enumerate(jobs, start=1):
        print(f"[{index}/{len(jobs)}] {job.split}/{job.video_name}")
        try:
            result = process_job(job, args)
            if result["status"] in {"processed", "dry_run"}:
                processed += 1
            else:
                skipped += 1
        except Exception as exc:
            failed += 1
            print(f"[ERROR] {job.split}/{job.video_name}: {exc}")

    print(f"Done. processed={processed} skipped={skipped} failed={failed}")


if __name__ == "__main__":
    main()
