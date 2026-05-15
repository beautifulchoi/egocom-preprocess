"""
Create compact HTML reports listing folders that contain merged segment tracks.
"""

from __future__ import annotations

import argparse
import html
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_DATA_ROOT = "/home/prj/data/egocom_holdout/1min"


def load_json(path: Path) -> Any:
    with path.open("r") as file:
        return json.load(file)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        json.dump(payload, file, indent=2)


def esc(value: Any) -> str:
    return html.escape(str(value))


def discover_splits(data_root: Path, split_arg: str) -> list[str]:
    if split_arg != "all_existing":
        return [item.strip() for item in split_arg.split(",") if item.strip()]
    return sorted(
        split_dir.name
        for split_dir in data_root.iterdir()
        if split_dir.is_dir() and (split_dir / "person_face_mapping").is_dir()
    )


def split_rows(data_root: Path, split: str) -> list[dict[str, Any]]:
    mapping_root = data_root / split / "person_face_mapping"
    rows_by_folder: dict[str, dict[str, Any]] = {}
    for remap_path in sorted(mapping_root.glob("*/remap_all_chunks.json")):
        scene_key = remap_path.parent.name
        remap = load_json(remap_path)
        for group in remap.get("merged_groups", []):
            clip_name = str(group["clip_name"])
            folder_path = remap_path.parent / "remap_visualizations" / clip_name
            key = str(folder_path)
            row = rows_by_folder.setdefault(
                key,
                {
                    "folder_path": key,
                    "scene_key": scene_key,
                    "clip_name": clip_name,
                    "merged_group_count": 0,
                    "extra_merged_segment_count": 0,
                },
            )
            row["merged_group_count"] += 1
            row["extra_merged_segment_count"] += max(
                0,
                len(group.get("merged_segment_ids", [])) - 1,
            )
    return sorted(rows_by_folder.values(), key=lambda item: item["folder_path"])


def count_visualization_folders(data_root: Path, split: str) -> int:
    mapping_root = data_root / split / "person_face_mapping"
    return sum(
        1
        for path in mapping_root.glob("*/remap_visualizations/*")
        if path.is_dir()
    )


def render_html(split: str, rows: list[dict[str, Any]], all_folder_count: int) -> str:
    total_groups = sum(int(row["merged_group_count"]) for row in rows)
    total_extra = sum(int(row["extra_merged_segment_count"]) for row in rows)
    ratio = f"{len(rows)} / {all_folder_count}"
    table_rows = []
    for row in rows:
        table_rows.append(
            "<tr>"
            f"<td><code>{esc(row['folder_path'])}</code></td>"
            f"<td>{esc(row['merged_group_count'])}</td>"
            f"<td>{esc(row['extra_merged_segment_count'])}</td>"
            "</tr>"
        )
    if not table_rows:
        table_rows.append("<tr><td colspan=\"3\">No merged segments found.</td></tr>")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{esc(split)} Merged Segment Report</title>
  <style>
    body {{ font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 28px; color: #17212f; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; letter-spacing: 0; }}
    p {{ color: #4f5b68; }}
    code {{ background: #eef1f5; padding: 1px 5px; border-radius: 5px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 16px; font-size: 13px; }}
    th, td {{ border: 1px solid #cfd6df; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #f5f7fa; }}
  </style>
</head>
<body>
  <h1>{esc(split)} Merged Segment Report</h1>
  <p>Folders with merged segment indications. Original masks are not modified.</p>
  <p><strong>Merged folders / total remap visualization folders: {ratio}</strong></p>
  <p>Merged groups: {total_groups}; extra merged segments: {total_extra}</p>
  <table>
    <thead>
      <tr><th>Folder Path</th><th>Merged Group Count</th><th>Extra Merged Segment Count</th></tr>
    </thead>
    <tbody>{''.join(table_rows)}</tbody>
  </table>
</body>
</html>
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Report folders containing merged segment remap outputs.")
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split", type=str, default="all_existing")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data_root = Path(args.data_root)
    summaries: dict[str, Any] = {}
    for split in discover_splits(data_root, args.split):
        rows = split_rows(data_root, split)
        all_folder_count = count_visualization_folders(data_root, split)
        out_dir = data_root / split / "person_face_mapping"
        write_json(out_dir / "merged_segments_report.json", rows)
        write_text(out_dir / "merged_segments_report.html", render_html(split, rows, all_folder_count))
        summaries[split] = {
            "folder_count": all_folder_count,
            "all_visualization_folder_count": all_folder_count,
            "merged_folder_count": len(rows),
            "merge_counted_over_total_video": f"{len(rows)} / {all_folder_count}",
            "merged_video_folder_count": len(rows),
            "total_video_folder_count": all_folder_count,
            "merged_group_count": sum(int(row["merged_group_count"]) for row in rows),
            "extra_merged_segment_count": sum(
                int(row["extra_merged_segment_count"]) for row in rows
            ),
            "report_html": str(out_dir / "merged_segments_report.html"),
            "report_json": str(out_dir / "merged_segments_report.json"),
        }
        print(
            f"[OK] {split}: merge_counted/total_video="
            f"{summaries[split]['merge_counted_over_total_video']} "
            f"merged_groups={summaries[split]['merged_group_count']} "
            f"extra={summaries[split]['extra_merged_segment_count']}"
        )
    write_json(data_root / "merged_segments_report_summary.json", summaries)


if __name__ == "__main__":
    main()
