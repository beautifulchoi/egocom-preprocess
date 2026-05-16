"""
Extract T5-XXL features from EgoCom InternVL2 person-track text.

Each input .txt file is expected to contain one text line per source frame. The
literal text "null" is encoded like any other line so the feature sequence stays
aligned with the frame sequence.

Inputs:
  /home/prj/data/egocom_holdout/1min/{split}/person_spatial_internvl2_text/**/*.txt

Outputs:
  /home/prj/data/egocom_holdout/1min/{split}/person_spatial_t5_features/{scene}/person_{id}/{video}.pt
  /home/prj/data/egocom_holdout/1min/{split}/person_spatial_t5_features/{scene}/summary.json
  /home/prj/data/egocom_holdout/1min/{split}/person_spatial_t5_features/summary.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm


DEFAULT_DATA_ROOT = "/home/prj/data/egocom_holdout/1min"
DEFAULT_MODEL_ID = "google/t5-v1_1-xxl"
DEFAULT_MAX_LENGTH = 256
TRACK_RE = re.compile(r"(?P<scene>[^/]+)/person_(?P<person_id>\d+)/(?P<video>[^/]+)\.txt$")


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


def load_json(path: Path) -> Any:
    with path.open("r") as file:
        return json.load(file)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        json.dump(payload, file, indent=2)


def split_names(data_root: Path, split_arg: str) -> list[str]:
    if split_arg != "all_existing":
        return parse_comma_list(split_arg)
    return sorted(
        split_dir.name
        for split_dir in data_root.iterdir()
        if split_dir.is_dir() and (split_dir / "person_spatial_internvl2_text").is_dir()
    )


def default_input_root(data_root: Path, split: str) -> Path:
    return data_root / split / "person_spatial_internvl2_text"


def default_output_root(data_root: Path, split: str) -> Path:
    return data_root / split / "person_spatial_t5_features"


def discover_text_files(input_root: Path, scene_key: str | None, video: str | None) -> list[Path]:
    if input_root.is_file():
        if input_root.suffix.lower() != ".txt":
            raise ValueError(f"Input file is not a .txt file: {input_root}")
        return [input_root]
    if not input_root.is_dir():
        return []

    paths = []
    for path in sorted(input_root.rglob("*.txt")):
        if not path.is_file():
            continue
        try:
            rel = path.relative_to(input_root)
        except ValueError:
            continue
        if len(rel.parts) < 3:
            continue
        if scene_key and rel.parts[0] != scene_key:
            continue
        if video and path.stem != video:
            continue
        paths.append(path)
    return paths


def parse_text_path(path: Path, input_root: Path) -> dict[str, Any]:
    try:
        rel = path.relative_to(input_root).as_posix()
    except ValueError:
        rel = path.as_posix()
    match = TRACK_RE.search(rel)
    if match is None:
        return {
            "scene_key": "unknown_scene",
            "person_id": -1,
            "video_name": path.stem,
        }
    return {
        "scene_key": match.group("scene"),
        "person_id": int(match.group("person_id")),
        "video_name": match.group("video"),
    }


def output_path_for(txt_path: Path, input_root: Path, output_root: Path) -> Path:
    if input_root.is_file():
        parsed = parse_text_path(txt_path, txt_path.parent.parent.parent)
        return (
            output_root
            / str(parsed["scene_key"])
            / f"person_{parsed['person_id']}"
            / f"{txt_path.stem}.pt"
        )
    return output_root / txt_path.relative_to(input_root).with_suffix(".pt")


def read_text_lines(path: Path) -> list[str]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [line if line else "null" for line in lines]


def lines_for_t5_encoding(lines: list[str]) -> list[str]:
    return ["" if line.strip().lower() == "null" else line for line in lines]


def pick_dtype(device: str | None) -> torch.dtype:
    if device == "cpu":
        return torch.float32
    if torch.cuda.is_available():
        return torch.bfloat16
    return torch.float32


def load_t5_encoder(
    model_id: str,
    device: str | None,
    dtype: torch.dtype,
    local_files_only: bool,
):
    try:
        from transformers import AutoTokenizer, T5EncoderModel
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: install transformers and sentencepiece, then retry."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        local_files_only=local_files_only,
    )
    model_kwargs = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        "local_files_only": local_files_only,
    }
    if device is None and torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"

    try:
        model = T5EncoderModel.from_pretrained(model_id, **model_kwargs).eval()
    except OSError as exc:
        if "TensorFlow weights" not in str(exc):
            raise
        model_kwargs["from_tf"] = True
        model = T5EncoderModel.from_pretrained(model_id, **model_kwargs).eval()
    if device is not None:
        model = model.to(device)
    return tokenizer, model


def model_device(model) -> torch.device:
    if hasattr(model, "device"):
        return torch.device(model.device)
    return next(model.parameters()).device


@torch.inference_mode()
def encode_lines(
    lines: list[str],
    tokenizer,
    model,
    max_length: int,
    batch_size: int,
) -> torch.Tensor:
    device = model_device(model)
    pooled_batches = []
    for start in range(0, len(lines), batch_size):
        batch = lines[start : start + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        outputs = model(**inputs)
        hidden = outputs.last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        pooled_batches.append(pooled.detach().cpu().float())
    return torch.cat(pooled_batches, dim=0)


def load_text_metadata(txt_path: Path) -> dict[str, Any]:
    meta_path = txt_path.with_suffix(".json")
    if not meta_path.exists():
        return {}
    try:
        data = load_json(meta_path)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def build_payload(
    features: torch.Tensor,
    texts: list[str],
    encoded_texts: list[str],
    txt_path: Path,
    input_root: Path,
    output_path: Path,
    metadata: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    parsed = parse_text_path(txt_path, input_root if input_root.is_dir() else input_root.parent.parent.parent)
    return {
        "features": features,
        "texts": texts,
        "encoded_texts": encoded_texts,
        "source_text_path": str(txt_path),
        "source_metadata_path": str(txt_path.with_suffix(".json")) if txt_path.with_suffix(".json").exists() else None,
        "output_path": str(output_path),
        "split": metadata.get("split", args.split),
        "scene_key": metadata.get("scene_key", parsed["scene_key"]),
        "video_name": metadata.get("video_name", parsed["video_name"]),
        "camera_person": metadata.get("camera_person"),
        "person_id": int(metadata.get("person_id", parsed["person_id"])),
        "segment_ids": metadata.get("segment_ids", []),
        "frame_indices": metadata.get("frame_indices", list(range(len(texts)))),
        "frame_stems": metadata.get("frame_stems", []),
        "source_frame_paths": metadata.get("source_frame_paths", []),
        "has_masks": metadata.get("has_masks", []),
        "frame_statuses": metadata.get("frame_statuses", []),
        "mask_bboxes": metadata.get("mask_bboxes", []),
        "mask_pixel_counts": metadata.get("mask_pixel_counts", []),
        "text_model_id": metadata.get("model_id"),
        "feature_model_id": args.model_id,
        "feature_dim": int(features.shape[1]),
        "num_lines": int(len(texts)),
        "num_null_lines": int(sum(1 for text in texts if text == "null")),
    }


def process_text_file(
    txt_path: Path,
    input_root: Path,
    output_root: Path,
    tokenizer,
    model,
    args: argparse.Namespace,
) -> dict[str, Any]:
    out_path = output_path_for(txt_path, input_root, output_root)
    if out_path.exists() and not args.overwrite:
        print(f"[SKIP] existing: {out_path}")
        parsed = parse_text_path(txt_path, input_root if input_root.is_dir() else input_root.parent.parent.parent)
        return {
            "status": "skipped_existing",
            "split": args.split,
            "scene_key": parsed["scene_key"],
            "video_name": parsed["video_name"],
            "person_id": int(parsed["person_id"]),
            "output_path": str(out_path),
        }

    texts = read_text_lines(txt_path)
    if not texts:
        raise ValueError(f"No text lines found in {txt_path}")

    encoded_texts = lines_for_t5_encoding(texts)
    features = encode_lines(
        lines=encoded_texts,
        tokenizer=tokenizer,
        model=model,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    expected_shape = (len(texts), int(args.expected_dim))
    if tuple(features.shape) != expected_shape:
        raise RuntimeError(
            f"got feature shape {tuple(features.shape)}, expected {expected_shape}"
        )
    if torch.isnan(features).any() or torch.isinf(features).any():
        raise RuntimeError("feature tensor contains NaN or Inf")

    metadata = load_text_metadata(txt_path)
    payload = build_payload(
        features=features,
        texts=texts,
        encoded_texts=encoded_texts,
        txt_path=txt_path,
        input_root=input_root,
        output_path=out_path,
        metadata=metadata,
        args=args,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)

    return {
        "status": "processed",
        "split": payload["split"],
        "scene_key": payload["scene_key"],
        "video_name": payload["video_name"],
        "camera_person": payload["camera_person"],
        "person_id": int(payload["person_id"]),
        "num_lines": int(payload["num_lines"]),
        "num_null_lines": int(payload["num_null_lines"]),
        "feature_dim": int(payload["feature_dim"]),
        "output_path": str(out_path),
        "source_text_path": str(txt_path),
    }


def write_summaries(output_roots: dict[str, Path], results: list[dict[str, Any]], args: argparse.Namespace) -> None:
    by_scene: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_scene[(str(result["split"]), str(result["scene_key"]))].append(result)

    for (split, scene_key), scene_results in sorted(by_scene.items()):
        status_counts = Counter(str(item["status"]) for item in scene_results)
        summary = {
            "split": split,
            "scene_key": scene_key,
            "model_id": args.model_id,
            "status_counts": dict(sorted(status_counts.items())),
            "num_tracks": len(scene_results),
            "num_processed_tracks": int(status_counts.get("processed", 0)),
            "num_lines": int(sum(int(item.get("num_lines", 0)) for item in scene_results)),
            "num_null_lines": int(sum(int(item.get("num_null_lines", 0)) for item in scene_results)),
            "tracks": scene_results,
        }
        write_json(output_roots[split] / scene_key / "summary.json", summary)

    for split, output_root in sorted(output_roots.items()):
        split_results = [item for item in results if item["split"] == split]
        status_counts = Counter(str(item["status"]) for item in split_results)
        summary = {
            "split": split,
            "model_id": args.model_id,
            "status_counts": dict(sorted(status_counts.items())),
            "num_tracks": len(split_results),
            "num_processed_tracks": int(status_counts.get("processed", 0)),
            "num_lines": int(sum(int(item.get("num_lines", 0)) for item in split_results)),
            "num_null_lines": int(sum(int(item.get("num_null_lines", 0)) for item in split_results)),
            "tracks": split_results,
        }
        write_json(output_root / "summary.json", summary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract T5-XXL features from InternVL2 EgoCom person text."
    )
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--input_root", type=str, default=None)
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--scene_key", type=str, default=None)
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max_length", type=positive_int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--batch_size", type=positive_int, default=16)
    parser.add_argument("--expected_dim", type=positive_int, default=4096)
    parser.add_argument("--limit", type=nonnegative_int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    data_root = Path(args.data_root)
    if not data_root.is_dir():
        print(f"Data root does not exist: {data_root}", file=sys.stderr)
        return 2

    splits = split_names(data_root, args.split)
    if args.input_root and len(splits) != 1:
        print("--input_root is only supported with a single split", file=sys.stderr)
        return 2
    if args.output_root and len(splits) != 1:
        print("--output_root is only supported with a single split", file=sys.stderr)
        return 2

    jobs: list[tuple[str, Path, Path, Path]] = []
    output_roots: dict[str, Path] = {}
    for split in splits:
        input_root = Path(args.input_root) if args.input_root else default_input_root(data_root, split)
        output_root = Path(args.output_root) if args.output_root else default_output_root(data_root, split)
        output_roots[split] = output_root
        try:
            text_files = discover_text_files(input_root, args.scene_key, args.video)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        for txt_path in text_files:
            jobs.append((split, input_root, output_root, txt_path))

    if args.limit is not None:
        jobs = jobs[: args.limit]
    if not jobs:
        print("No matching InternVL2 text files found.")
        return 2

    dtype = pick_dtype(args.device)
    print(f"Loading T5 encoder: {args.model_id}")
    tokenizer, model = load_t5_encoder(
        args.model_id,
        device=args.device,
        dtype=dtype,
        local_files_only=args.local_files_only,
    )
    print(f"Found {len(jobs)} text files")

    results = []
    processed = 0
    skipped = 0
    failed = 0
    for split, input_root, output_root, txt_path in tqdm(jobs, desc="texts"):
        args.split = split
        try:
            result = process_text_file(
                txt_path=txt_path,
                input_root=input_root,
                output_root=output_root,
                tokenizer=tokenizer,
                model=model,
                args=args,
            )
            results.append(result)
            if result["status"] == "processed":
                processed += 1
            elif result["status"].startswith("skipped"):
                skipped += 1
        except Exception as exc:
            failed += 1
            parsed = parse_text_path(txt_path, input_root if input_root.is_dir() else input_root.parent.parent.parent)
            result = {
                "status": "failed",
                "split": split,
                "scene_key": parsed["scene_key"],
                "video_name": parsed["video_name"],
                "person_id": int(parsed["person_id"]),
                "source_text_path": str(txt_path),
                "error": str(exc),
            }
            results.append(result)
            print(f"[ERROR] {txt_path}: {exc}", file=sys.stderr)

    write_summaries(output_roots, results, args)
    print(f"Done. processed={processed} skipped={skipped} failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
