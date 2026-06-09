"""
Extract Depth Anything 3 outputs for EgoCom frame folders.

Outputs:
  output_dir/
    monocular/<video_name>/depth/<frame_stem>.npy
    monocular/<video_name>/vis/<frame_stem>.jpg  # sampled overlays
    nested/<video_name>/camera_params/intrinsics.npy
"""

from __future__ import annotations

import argparse
import gc
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch


DA3_SRC = Path(__file__).resolve().parents[1] / "_external" / "depth-anything-3" / "src"
if str(DA3_SRC) not in sys.path:
    sys.path.insert(0, str(DA3_SRC))


DEFAULT_INPUT_DIR = "/home/prj/egocom_preprocess/extract_output/frames"
DEFAULT_OUTPUT_DIR = "/home/prj/egocom_preprocess/extract_output/da3"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
MONOCULAR_MODEL_ID = "depth-anything/DA3METRIC-LARGE"
NESTED_MODEL_ID = "depth-anything/DA3NESTED-GIANT-LARGE"
DEFAULT_VIS_STRIDE = 8
DEFAULT_VIS_ALPHA = 0.9


def resolve_device(device_arg: str) -> torch.device:
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU")
        return torch.device("cpu")
    return torch.device(device_arg)


def list_frame_files(folder_path: Path) -> list[Path]:
    return sorted(
        path
        for path in folder_path.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def load_frames(image_paths: list[Path]) -> list[np.ndarray]:
    frames = []
    for image_path in image_paths:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise ValueError(f"Failed to read image: {image_path}")
        frames.append(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    return frames


def shape_of(value: object) -> tuple[int, ...] | str | None:
    if value is None:
        return None
    if hasattr(value, "shape"):
        return tuple(int(dim) for dim in value.shape)
    return str(type(value))


def write_summary(path: Path, summary: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")


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


def save_depth_overlay(
    image_rgb: np.ndarray,
    depth_map: np.ndarray,
    output_path: Path,
    alpha: float,
) -> None:
    depth_h, depth_w = depth_map.shape[:2]
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    if image_bgr.shape[:2] != (depth_h, depth_w):
        image_bgr = cv2.resize(image_bgr, (depth_w, depth_h), interpolation=cv2.INTER_AREA)

    depth_color = depth_to_colormap(depth_map)
    overlay = cv2.addWeighted(image_bgr, 1.0 - alpha, depth_color, alpha, 0.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), overlay)


def clear_memory(*objects: object) -> None:
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_da3_model(model_id: str, device: torch.device) -> torch.nn.Module:
    try:
        from depth_anything_3.api import DepthAnything3
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Failed to import Depth Anything 3. Install the dependencies from "
            f"{DA3_SRC.parent / 'requirements.txt'}; missing module: {exc.name}"
        ) from exc

    model = DepthAnything3.from_pretrained(model_id).to(device=device)
    model.eval()
    return model


def run_inference(
    model: torch.nn.Module,
    frames: list[np.ndarray],
    process_res: int,
):
    return model.inference(
        frames,
        ref_view_strategy="middle",
        use_ray_pose=True,
        process_res=process_res,
    )


def process_folder_nested(
    folder_path: Path,
    model: torch.nn.Module,
    output_base_dir: Path,
    process_res: int,
    skip_existing: bool,
) -> None:
    print(f"\n{'=' * 60}")
    print(f"Processing folder (NESTED intrinsics): {folder_path.name}")
    print(f"{'=' * 60}")

    image_paths = list_frame_files(folder_path)
    if not image_paths:
        print(f"Warning: No images found in {folder_path}, skipping...")
        return

    print(f"Loading {len(image_paths)} frames from {folder_path}...")
    frames = load_frames(image_paths)

    print("Running inference with NESTED model...")
    prediction = run_inference(model, frames, process_res)

    folder_output_dir = output_base_dir / "nested" / folder_path.name
    params_dir = folder_output_dir / "camera_params"
    if skip_existing and (params_dir / "intrinsics.npy").exists():
        print(f"Skipping existing nested intrinsics: {folder_path.name}")
        return

    params_dir.mkdir(parents=True, exist_ok=True)

    intrinsics = prediction.intrinsics
    if intrinsics is None:
        raise ValueError("Nested model did not return intrinsics")

    np.save(params_dir / "intrinsics.npy", intrinsics)

    write_summary(
        folder_output_dir / "summary.txt",
        {
            "folder": folder_path.name,
            "model_type": "nested",
            "model_id": NESTED_MODEL_ID,
            "num_frames": len(frames),
            "reference_index": len(frames) // 2,
            "frame_stems": [path.stem for path in image_paths],
            "intrinsics_shape": shape_of(intrinsics),
            "depth_shape": shape_of(prediction.depth),
        },
    )

    print(f"Saved intrinsics shape: {shape_of(intrinsics)}")
    print(f"Results saved to {folder_output_dir}")
    clear_memory(frames, prediction, intrinsics)


def process_folder_monocular(
    folder_path: Path,
    model: torch.nn.Module,
    output_base_dir: Path,
    process_res: int,
    vis_stride: int,
    vis_alpha: float,
    skip_existing: bool,
) -> None:
    print(f"\n{'=' * 60}")
    print(f"Processing folder (MONOCULAR metric depth): {folder_path.name}")
    print(f"{'=' * 60}")

    image_paths = list_frame_files(folder_path)
    if not image_paths:
        print(f"Warning: No images found in {folder_path}, skipping...")
        return

    print(f"Loading {len(image_paths)} frames from {folder_path}...")
    frames = load_frames(image_paths)

    print("Running inference with METRIC model...")
    prediction = run_inference(model, frames, process_res)

    folder_output_dir = output_base_dir / "monocular" / folder_path.name
    depth_dir = folder_output_dir / "depth"
    vis_dir = folder_output_dir / "vis"
    expected_vis_count = (
        (len(image_paths) + vis_stride - 1) // vis_stride if vis_stride > 0 else 0
    )
    if skip_existing:
        depth_count = len(list(depth_dir.glob("*.npy"))) if depth_dir.exists() else 0
        vis_count = len(list(vis_dir.glob("*.jpg"))) if vis_dir.exists() else 0
        if depth_count == len(image_paths) and vis_count >= expected_vis_count:
            print(f"Skipping existing monocular depth and vis: {folder_path.name}")
            return

    depth_dir.mkdir(parents=True, exist_ok=True)

    depth_maps = prediction.depth
    if len(depth_maps) != len(image_paths):
        raise ValueError(
            f"Depth map count {len(depth_maps)} does not match frame count {len(image_paths)}"
        )

    num_vis_saved = 0
    for frame_index, (image_path, image_rgb, depth_map) in enumerate(
        zip(image_paths, frames, depth_maps)
    ):
        np.save(depth_dir / f"{image_path.stem}.npy", depth_map)
        if vis_stride > 0 and frame_index % vis_stride == 0:
            save_depth_overlay(
                image_rgb,
                depth_map,
                vis_dir / f"{image_path.stem}.jpg",
                alpha=vis_alpha,
            )
            num_vis_saved += 1

    write_summary(
        folder_output_dir / "summary.txt",
        {
            "folder": folder_path.name,
            "model_type": "monocular",
            "model_id": MONOCULAR_MODEL_ID,
            "num_frames": len(frames),
            "reference_index": len(frames) // 2,
            "frame_stems": [path.stem for path in image_paths],
            "depth_maps_shape": shape_of(depth_maps),
            "is_metric": getattr(prediction, "is_metric", None),
            "vis_stride": vis_stride,
            "vis_alpha": vis_alpha,
            "num_vis_saved": num_vis_saved,
        },
    )

    print(f"Saved {len(depth_maps)} depth maps to {depth_dir}")
    print(f"Saved {num_vis_saved} depth overlays to {vis_dir}")
    print(f"Results saved to {folder_output_dir}")
    clear_memory(frames, prediction, depth_maps)


def get_frame_folders(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    subdirs = sorted(path for path in input_dir.iterdir() if path.is_dir())
    if not subdirs:
        raise ValueError(f"No subdirectories found in {input_dir}")
    return subdirs


def process_all_folders(
    folders: list[Path],
    output_dir: Path,
    device: torch.device,
    process_res: int,
    run_mode: str,
    vis_stride: int,
    vis_alpha: float,
    skip_existing: bool,
) -> None:
    if run_mode in {"both", "nested"}:
        print(f"Loading Depth-Anything-3 nested model: {NESTED_MODEL_ID}")
        nested_model = load_da3_model(NESTED_MODEL_ID, device)
        print(f"Nested model loaded on {device}")

        for index, folder in enumerate(folders, start=1):
            print(f"\n[NESTED {index}/{len(folders)}] {folder.name}")
            try:
                process_folder_nested(
                    folder, nested_model, output_dir, process_res, skip_existing
                )
            except Exception as exc:
                print(f"Error processing nested outputs for {folder.name}: {exc}")
                traceback.print_exc()
                continue

        del nested_model
        clear_memory()

    if run_mode in {"both", "monocular"}:
        print(f"Loading Depth-Anything-3 metric model: {MONOCULAR_MODEL_ID}")
        metric_model = load_da3_model(MONOCULAR_MODEL_ID, device)
        print(f"Metric model loaded on {device}")

        for index, folder in enumerate(folders, start=1):
            print(f"\n[MONOCULAR {index}/{len(folders)}] {folder.name}")
            try:
                process_folder_monocular(
                    folder,
                    metric_model,
                    output_dir,
                    process_res,
                    vis_stride,
                    vis_alpha,
                    skip_existing,
                )
            except Exception as exc:
                print(f"Error processing monocular outputs for {folder.name}: {exc}")
                traceback.print_exc()
                continue

        del metric_model
        clear_memory()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract DA3 nested intrinsics and DA3 metric monocular depth for frame folders."
        )
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing one subdirectory per video/frame sequence.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save DA3 outputs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for model inference, e.g. 'cuda', 'cuda:0', or 'cpu'.",
    )
    parser.add_argument(
        "--process_res",
        type=int,
        default=504,
        help="DA3 processing resolution passed to model.inference.",
    )
    parser.add_argument(
        "--run",
        type=str,
        default="both",
        choices=["both", "nested", "monocular"],
        help="Which DA3 outputs to extract.",
    )
    parser.add_argument(
        "--vis_stride",
        type=int,
        default=DEFAULT_VIS_STRIDE,
        help="Save one depth overlay every N frames during monocular extraction. Use 0 to disable.",
    )
    parser.add_argument(
        "--vis_alpha",
        type=float,
        default=DEFAULT_VIS_ALPHA,
        help="Depth colormap opacity for overlay visualizations. Higher values make the original frame more transparent.",
    )
    parser.add_argument(
        "--no_skip_existing",
        action="store_true",
        help="Recompute outputs even when existing depth/visualization or intrinsics files are complete.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    device = resolve_device(args.device)

    folders = get_frame_folders(input_dir)
    print(f"Found {len(folders)} folders to process")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Run mode: {args.run}")
    if not 0.0 <= args.vis_alpha <= 1.0:
        raise ValueError(f"--vis_alpha must be between 0 and 1, got {args.vis_alpha}")

    process_all_folders(
        folders=folders,
        output_dir=output_dir,
        device=device,
        process_res=args.process_res,
        run_mode=args.run,
        vis_stride=args.vis_stride,
        vis_alpha=args.vis_alpha,
        skip_existing=not args.no_skip_existing,
    )

    print("\n" + "=" * 60)
    print("All requested DA3 extraction passes completed")
    print(f"Results saved to {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
