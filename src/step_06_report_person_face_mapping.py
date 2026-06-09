"""
Build a cross-split report for EgoCom face-embedding person mapping outputs.

The report focuses on scenes/clips that do not have representative visualizations
because selected segments were not finally mapped.
"""

from __future__ import annotations

import argparse
import html
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_DATA_ROOT = "/home/prj/data/egocom_holdout/1min"


def load_json(path: Path) -> Any:
    with path.open("r") as file:
        return json.load(file)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        json.dump(payload, file, indent=2)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def scene_dirs(data_root: Path) -> list[Path]:
    return sorted(data_root.glob("*/person_face_mapping/*"))


def evidence_summary(pair_evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for evidence in pair_evidence:
        match = evidence.get("match") or {}
        out.append(
            {
                "other_clip": evidence.get("other_clip"),
                "expected_common_person": evidence.get("expected_common_person"),
                "accepted": bool(evidence.get("accepted")),
                "reason": evidence.get("reason"),
                "matrix_shape": evidence.get("matrix_shape"),
                "local_segment_id": evidence.get("local_segment_id"),
                "other_segment_id": evidence.get("other_segment_id"),
                "top_similarity": match.get("top_similarity"),
                "second_best_similarity": match.get("second_best_similarity"),
                "top_margin": match.get("top_margin"),
                "local_segment_ids": evidence.get("local_segment_ids", []),
                "other_segment_ids": evidence.get("other_segment_ids", []),
                "similarity_matrix": evidence.get("similarity_matrix", []),
            }
        )
    return out


def compact_attempt_details(attempts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for attempt in attempts:
        compact.append(
            {
                "chunk": attempt.get("chunk"),
                "reasons": attempt.get("reasons", []),
                "num_loaded_clips": attempt.get("num_loaded_clips"),
                "total_segments": attempt.get("total_segments"),
                "resolved_segments": attempt.get("resolved_segments"),
                "unresolved_segments": attempt.get("unresolved_segments"),
                "conflict_segments": attempt.get("conflict_segments"),
                "matrix_shape_counts": attempt.get("matrix_shape_counts", {}),
                "clips": attempt.get("clips", {}),
                "pair_evidence": attempt.get("pair_evidence", []),
            }
        )
    return compact


def clip_reasons(clip: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if clip.get("conflicts"):
        reasons.append("conflict")
    if clip.get("missing_embedding_segment_ids"):
        reasons.append("missing_embedding")
    if clip.get("low_embedding_count_segment_ids"):
        reasons.append("low_embedding_count")
    if clip.get("unknown_segments"):
        reasons.append("unresolved_assignment")
    for evidence in clip.get("pair_evidence", []):
        if not evidence.get("accepted"):
            reason = evidence.get("reason") or "rejected_pair"
            reasons.append(str(reason))
    return sorted(set(reasons))


def build_report(data_root: Path) -> dict[str, Any]:
    split_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "scenes": 0,
            "problem_scenes": 0,
            "clips": 0,
            "clips_without_representatives": 0,
            "total_segments": 0,
            "resolved_segments": 0,
            "unresolved_segments": 0,
            "conflict_segments": 0,
            "missing_embedding_segments": 0,
            "low_embedding_count_segments": 0,
            "fallback_scenes": 0,
            "all_chunks_conflicted_scenes": 0,
            "reason_counts": Counter(),
        }
    )
    all_scenes: list[dict[str, Any]] = []
    problem_scenes: list[dict[str, Any]] = []
    diagnostic_scenes: list[dict[str, Any]] = []

    for scene_dir in scene_dirs(data_root):
        if scene_dir.name in {"representative_crops"}:
            continue
        summary_path = scene_dir / "summary.json"
        details_path = scene_dir / "details.json"
        if not summary_path.exists() or not details_path.exists():
            continue

        split = scene_dir.parents[1].name
        scene_key = scene_dir.name
        summary = load_json(summary_path)
        details = load_json(details_path)
        scene_clip_count = 0
        scene_clips_without_representatives = 0
        scene_missing_embedding_segments = 0
        scene_low_embedding_count_segments = 0

        stats = split_stats[split]
        stats["scenes"] += 1
        stats["total_segments"] += int(summary.get("total_segments", 0))
        stats["resolved_segments"] += int(summary.get("resolved_segments", 0))
        stats["unresolved_segments"] += int(summary.get("unresolved_segments", 0))
        stats["conflict_segments"] += int(summary.get("conflict_segments", 0))
        if int(summary.get("num_chunk_attempts", 1)) > 1:
            stats["fallback_scenes"] += 1
        if summary.get("fallback_status") == "all_chunks_conflicted":
            stats["all_chunks_conflicted_scenes"] += 1

        scene_problem_clips: list[dict[str, Any]] = []
        for clip_name, clip in sorted(details.get("clips", {}).items()):
            stats["clips"] += 1
            scene_clip_count += 1
            missing = [int(value) for value in clip.get("missing_embedding_segment_ids", [])]
            low = [int(value) for value in clip.get("low_embedding_count_segment_ids", [])]
            unknown = [int(value) for value in clip.get("unknown_segments", [])]
            conflicts = clip.get("conflicts", [])
            representatives = clip.get("representatives", {})
            assignments = clip.get("assignments", {})
            resolved = {
                str(segment_id): person_id
                for segment_id, person_id in assignments.items()
                if person_id is not None
            }

            stats["missing_embedding_segments"] += len(missing)
            stats["low_embedding_count_segments"] += len(low)
            scene_missing_embedding_segments += len(missing)
            scene_low_embedding_count_segments += len(low)
            if not representatives:
                stats["clips_without_representatives"] += 1
                scene_clips_without_representatives += 1

            reasons = clip_reasons(clip)
            for reason in reasons:
                stats["reason_counts"][reason] += 1

            if reasons:
                scene_problem_clips.append(
                    {
                        "clip_name": clip_name,
                        "camera_person": clip.get("camera_person"),
                        "top_segment_ids": clip.get("top_segment_ids", []),
                        "selected_embedding_segment_ids": clip.get(
                            "selected_embedding_segment_ids", []
                        ),
                        "assignments": assignments,
                        "resolved_assignments": resolved,
                        "unknown_segments": unknown,
                        "missing_embedding_segment_ids": missing,
                        "low_embedding_count_segment_ids": low,
                        "conflicts": conflicts,
                        "representative_count": len(representatives),
                        "reasons": reasons,
                        "pair_evidence": evidence_summary(clip.get("pair_evidence", [])),
                    }
                )

        is_problem = (
            int(summary.get("unresolved_segments", 0)) > 0
            or int(summary.get("conflict_segments", 0)) > 0
            or bool(scene_problem_clips)
        )
        scene_entry = {
            "split": split,
            "scene_key": scene_key,
            "summary": summary,
            "scene_dir": str(scene_dir),
            "selected_chunk": summary.get("selected_chunk", summary.get("chunk")),
            "fallback_status": summary.get("fallback_status"),
            "chunk_attempts": summary.get("chunk_attempts", []),
            "chunk_attempt_details": compact_attempt_details(
                details.get("chunk_attempt_details", [])
            ),
            "scene_counts": {
                "clips": int(scene_clip_count),
                "clips_without_representatives": int(scene_clips_without_representatives),
                "missing_embedding_segments": int(scene_missing_embedding_segments),
                "low_embedding_count_segments": int(scene_low_embedding_count_segments),
            },
            "problem_clips": scene_problem_clips,
        }
        all_scenes.append(scene_entry)

        if is_problem:
            stats["problem_scenes"] += 1
            problem_scenes.append(scene_entry)
        if (
            is_problem
            or int(summary.get("num_chunk_attempts", 1)) > 1
            or int(summary.get("num_loaded_clips", 0)) < 3
        ):
            diagnostic_scenes.append(scene_entry)

    normalized_split_stats: dict[str, Any] = {}
    for split, stats in sorted(split_stats.items()):
        normalized = dict(stats)
        normalized["reason_counts"] = dict(sorted(stats["reason_counts"].items()))
        normalized_split_stats[split] = normalized

    total = {
        "splits": len(normalized_split_stats),
        "scenes": sum(item["scenes"] for item in normalized_split_stats.values()),
        "problem_scenes": sum(item["problem_scenes"] for item in normalized_split_stats.values()),
        "clips": sum(item["clips"] for item in normalized_split_stats.values()),
        "clips_without_representatives": sum(
            item["clips_without_representatives"] for item in normalized_split_stats.values()
        ),
        "total_segments": sum(item["total_segments"] for item in normalized_split_stats.values()),
        "resolved_segments": sum(
            item["resolved_segments"] for item in normalized_split_stats.values()
        ),
        "unresolved_segments": sum(
            item["unresolved_segments"] for item in normalized_split_stats.values()
        ),
        "conflict_segments": sum(
            item["conflict_segments"] for item in normalized_split_stats.values()
        ),
        "missing_embedding_segments": sum(
            item["missing_embedding_segments"] for item in normalized_split_stats.values()
        ),
        "low_embedding_count_segments": sum(
            item["low_embedding_count_segments"] for item in normalized_split_stats.values()
        ),
        "fallback_scenes": sum(
            item["fallback_scenes"] for item in normalized_split_stats.values()
        ),
        "all_chunks_conflicted_scenes": sum(
            item["all_chunks_conflicted_scenes"] for item in normalized_split_stats.values()
        ),
    }

    return {
        "data_root": str(data_root),
        "total": total,
        "splits": normalized_split_stats,
        "scenes": all_scenes,
        "problem_scenes": problem_scenes,
        "diagnostic_scenes": diagnostic_scenes,
    }


def subset_report(report: dict[str, Any], scenes: list[dict[str, Any]]) -> dict[str, Any]:
    split_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "scenes": 0,
            "problem_scenes": 0,
            "clips": 0,
            "clips_without_representatives": 0,
            "total_segments": 0,
            "resolved_segments": 0,
            "unresolved_segments": 0,
            "conflict_segments": 0,
            "missing_embedding_segments": 0,
            "low_embedding_count_segments": 0,
            "fallback_scenes": 0,
            "all_chunks_conflicted_scenes": 0,
            "reason_counts": Counter(),
        }
    )
    problem_scenes: list[dict[str, Any]] = []
    diagnostic_scenes: list[dict[str, Any]] = []
    for scene in scenes:
        split = scene["split"]
        summary = scene["summary"]
        scene_counts = scene.get("scene_counts", {})
        stats = split_stats[split]
        stats["scenes"] += 1
        stats["total_segments"] += int(summary.get("total_segments", 0))
        stats["resolved_segments"] += int(summary.get("resolved_segments", 0))
        stats["unresolved_segments"] += int(summary.get("unresolved_segments", 0))
        stats["conflict_segments"] += int(summary.get("conflict_segments", 0))
        stats["clips"] += int(scene_counts.get("clips", 0))
        stats["clips_without_representatives"] += int(
            scene_counts.get("clips_without_representatives", 0)
        )
        stats["missing_embedding_segments"] += int(
            scene_counts.get("missing_embedding_segments", 0)
        )
        stats["low_embedding_count_segments"] += int(
            scene_counts.get("low_embedding_count_segments", 0)
        )
        if int(summary.get("num_chunk_attempts", 1)) > 1:
            stats["fallback_scenes"] += 1
        if summary.get("fallback_status") == "all_chunks_conflicted":
            stats["all_chunks_conflicted_scenes"] += 1
        if scene.get("problem_clips"):
            stats["problem_scenes"] += 1
            problem_scenes.append(scene)
        if (
            scene.get("problem_clips")
            or int(summary.get("num_chunk_attempts", 1)) > 1
            or int(summary.get("num_loaded_clips", 0)) < 3
        ):
            diagnostic_scenes.append(scene)
        for clip in scene.get("problem_clips", []):
            for reason in clip.get("reasons", []):
                stats["reason_counts"][reason] += 1

    normalized_split_stats: dict[str, Any] = {}
    for split, stats in sorted(split_stats.items()):
        normalized = dict(stats)
        normalized["reason_counts"] = dict(sorted(stats["reason_counts"].items()))
        normalized_split_stats[split] = normalized

    total = {
        "splits": len(normalized_split_stats),
        "scenes": sum(item["scenes"] for item in normalized_split_stats.values()),
        "problem_scenes": sum(item["problem_scenes"] for item in normalized_split_stats.values()),
        "clips": sum(item["clips"] for item in normalized_split_stats.values()),
        "clips_without_representatives": sum(
            item["clips_without_representatives"] for item in normalized_split_stats.values()
        ),
        "total_segments": sum(item["total_segments"] for item in normalized_split_stats.values()),
        "resolved_segments": sum(
            item["resolved_segments"] for item in normalized_split_stats.values()
        ),
        "unresolved_segments": sum(
            item["unresolved_segments"] for item in normalized_split_stats.values()
        ),
        "conflict_segments": sum(
            item["conflict_segments"] for item in normalized_split_stats.values()
        ),
        "missing_embedding_segments": sum(
            item["missing_embedding_segments"] for item in normalized_split_stats.values()
        ),
        "low_embedding_count_segments": sum(
            item["low_embedding_count_segments"] for item in normalized_split_stats.values()
        ),
        "fallback_scenes": sum(
            item["fallback_scenes"] for item in normalized_split_stats.values()
        ),
        "all_chunks_conflicted_scenes": sum(
            item["all_chunks_conflicted_scenes"] for item in normalized_split_stats.values()
        ),
    }
    return {
        "data_root": report["data_root"],
        "total": total,
        "splits": normalized_split_stats,
        "scenes": scenes,
        "problem_scenes": problem_scenes,
        "diagnostic_scenes": diagnostic_scenes,
    }


def esc(value: Any) -> str:
    return html.escape(str(value))


def fmt_list(values: Any) -> str:
    if not values:
        return "-"
    if isinstance(values, dict):
        return esc(json.dumps(values, sort_keys=True))
    return esc(", ".join(str(value) for value in values))


def fmt_score(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return esc(value)


def fmt_matrix(matrix: Any) -> str:
    if not matrix:
        return "-"
    rows = []
    for row in matrix:
        rows.append("[" + ", ".join(fmt_score(value) for value in row) + "]")
    return "<br>".join(rows)


def render_attempt_rows(attempts: list[dict[str, Any]]) -> str:
    if not attempts:
        return "<tr><td colspan=\"8\">No chunk attempt diagnostics recorded.</td></tr>"
    rows: list[str] = []
    for attempt in attempts:
        pair_items: list[str] = []
        for evidence in attempt.get("pair_evidence", []):
            pair_items.append(
                "<li>"
                f"<code>{esc(evidence.get('clip_a'))}</code> vs "
                f"<code>{esc(evidence.get('clip_b'))}</code>: "
                f"expected person {esc(evidence.get('expected_common_person'))}; "
                f"accepted={esc(evidence.get('accepted'))}; "
                f"reason={esc(evidence.get('reason'))}; "
                f"shape={fmt_list(evidence.get('matrix_shape'))}; "
                f"local ids={fmt_list(evidence.get('local_segment_ids'))}; "
                f"other ids={fmt_list(evidence.get('other_segment_ids'))}; "
                f"chosen local={esc(evidence.get('local_segment_id'))}, "
                f"other={esc(evidence.get('other_segment_id'))}; "
                f"top={fmt_score(evidence.get('top_similarity'))}; "
                f"second={fmt_score(evidence.get('second_best_similarity'))}; "
                f"margin={fmt_score(evidence.get('top_margin'))}; "
                f"matrix=<span class=\"matrix\">{fmt_matrix(evidence.get('similarity_matrix'))}</span>"
                "</li>"
            )
        conflict_items: list[str] = []
        for clip_name, clip in sorted((attempt.get("clips") or {}).items()):
            for conflict in clip.get("conflicts", []):
                conflict_items.append(
                    "<li>"
                    f"<code>{esc(clip_name)}</code> segment {esc(conflict.get('segment_id'))}: "
                    f"existing person {esc(conflict.get('existing_person_id'))}, "
                    f"new person {esc(conflict.get('new_person_id'))}; "
                    f"source={esc(conflict.get('source'))}"
                    "</li>"
                )
        if not conflict_items:
            conflict_items.append("<li>No segment assignment conflicts.</li>")
        if not pair_items:
            pair_items.append("<li>No pair evidence.</li>")

        rows.append(
            "<tr>"
            f"<td>{esc(attempt.get('chunk'))}</td>"
            f"<td>{fmt_list(attempt.get('reasons'))}</td>"
            f"<td>{esc(attempt.get('resolved_segments'))}/{esc(attempt.get('total_segments'))}</td>"
            f"<td>{esc(attempt.get('unresolved_segments'))}</td>"
            f"<td>{esc(attempt.get('conflict_segments'))}</td>"
            f"<td>{esc(attempt.get('matrix_shape_counts'))}</td>"
            f"<td><ul>{''.join(conflict_items)}</ul></td>"
            f"<td><ul>{''.join(pair_items)}</ul></td>"
            "</tr>"
        )
    return "".join(rows)


def render_html(report: dict[str, Any]) -> str:
    split_rows = []
    for split, stats in report["splits"].items():
        split_rows.append(
            "<tr>"
            f"<td>{esc(split)}</td>"
            f"<td>{stats['scenes']}</td>"
            f"<td>{stats['problem_scenes']}</td>"
            f"<td>{stats['resolved_segments']}/{stats['total_segments']}</td>"
            f"<td>{stats['unresolved_segments']}</td>"
            f"<td>{stats['conflict_segments']}</td>"
            f"<td>{stats['missing_embedding_segments']}</td>"
            f"<td>{stats['low_embedding_count_segments']}</td>"
            f"<td>{stats['fallback_scenes']}</td>"
            f"<td>{stats['all_chunks_conflicted_scenes']}</td>"
            f"<td>{stats['clips_without_representatives']}</td>"
            "</tr>"
        )

    scene_sections = []
    scenes_to_render = report.get("diagnostic_scenes", report["problem_scenes"])
    if not scenes_to_render and len(report.get("scenes", [])) == 1:
        scenes_to_render = report["scenes"]
    for scene in scenes_to_render:
        clip_rows = []
        for clip in scene["problem_clips"]:
            evidence_items = []
            for evidence in clip["pair_evidence"]:
                score = evidence.get("top_similarity")
                score_text = "-" if score is None else f"{float(score):.3f}"
                evidence_items.append(
                    "<li>"
                    f"with <code>{esc(evidence.get('other_clip'))}</code>: "
                    f"expected person {esc(evidence.get('expected_common_person'))}, "
                    f"accepted={esc(evidence.get('accepted'))}, "
                    f"reason={esc(evidence.get('reason'))}, "
                    f"shape={fmt_list(evidence.get('matrix_shape'))}, "
                    f"local={esc(evidence.get('local_segment_id'))}, "
                    f"other={esc(evidence.get('other_segment_id'))}, "
                    f"score={score_text}"
                    "</li>"
                )
            clip_rows.append(
                "<tr>"
                f"<td><code>{esc(clip['clip_name'])}</code><br>camera person {esc(clip['camera_person'])}</td>"
                f"<td>{fmt_list(clip['reasons'])}</td>"
                f"<td>{fmt_list(clip['assignments'])}</td>"
                f"<td>{fmt_list(clip['unknown_segments'])}</td>"
                f"<td>{fmt_list(clip['missing_embedding_segment_ids'])}</td>"
                f"<td>{fmt_list(clip['low_embedding_count_segment_ids'])}</td>"
                f"<td>{len(clip['conflicts'])}</td>"
                f"<td><ul>{''.join(evidence_items)}</ul></td>"
                "</tr>"
            )
        if not clip_rows:
            clip_rows.append(
                "<tr><td colspan=\"8\">No unresolved, missing, low-count, or conflicted clips in this scene.</td></tr>"
            )

        summary = scene["summary"]
        scene_sections.append(
            "<section>"
            f"<h3>{esc(scene['split'])}/{esc(scene['scene_key'])}</h3>"
            "<p>"
            f"Resolved {esc(summary.get('resolved_segments'))}/{esc(summary.get('total_segments'))}; "
            f"unresolved {esc(summary.get('unresolved_segments'))}; "
            f"conflicts {esc(summary.get('conflict_segments'))}; "
            f"matrix shapes {esc(summary.get('matrix_shape_counts'))}."
            f" Selected chunk {esc(scene.get('selected_chunk'))}; "
            f"fallback status {esc(scene.get('fallback_status'))}."
            "</p>"
            "<h4>Chunk Attempts and Conflict Reasons</h4>"
            "<table>"
            "<thead><tr><th>Chunk</th><th>Reasons</th><th>Resolved</th><th>Unresolved</th>"
            "<th>Conflicts</th><th>Matrix Shapes</th><th>Conflict Details</th><th>Similarity Scores</th></tr></thead>"
            f"<tbody>{render_attempt_rows(scene.get('chunk_attempt_details', []))}</tbody>"
            "</table>"
            "<h4>Remaining Problem Clips</h4>"
            "<table>"
            "<thead><tr><th>Clip</th><th>Reasons</th><th>Assignments</th><th>Unknown</th>"
            "<th>Missing Emb</th><th>Low Count</th><th>Conflicts</th><th>Pair Evidence</th></tr></thead>"
            f"<tbody>{''.join(clip_rows)}</tbody>"
            "</table>"
            "</section>"
        )

    total = report["total"]
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>EgoCom Person Face Mapping Report</title>
  <style>
    body {{ font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; color: #17212f; background: #fff; }}
    main {{ max-width: 1320px; margin: 0 auto; padding: 32px 28px 56px; }}
    h1 {{ margin: 0 0 8px; font-size: 30px; letter-spacing: 0; }}
    h2 {{ margin: 34px 0 12px; font-size: 22px; letter-spacing: 0; }}
    h3 {{ margin: 28px 0 8px; font-size: 18px; letter-spacing: 0; }}
    p {{ color: #4f5b68; }}
    code {{ background: #eef1f5; padding: 1px 5px; border-radius: 5px; }}
    table {{ width: 100%; border-collapse: collapse; margin: 12px 0 22px; font-size: 13px; }}
    th, td {{ border: 1px solid #cfd6df; padding: 8px 9px; vertical-align: top; text-align: left; }}
    th {{ background: #f5f7fa; }}
    ul {{ margin: 0; padding-left: 18px; }}
    h4 {{ margin: 18px 0 8px; font-size: 15px; letter-spacing: 0; }}
    .matrix {{ display: inline-block; font-family: "SFMono-Regular", Consolas, monospace; background: #eef1f5; padding: 2px 5px; border-radius: 5px; margin-left: 3px; }}
    .cards {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 18px 0 22px; }}
    .card {{ border: 1px solid #cfd6df; border-radius: 8px; padding: 12px 14px; background: #f9fbfd; }}
    .label {{ color: #5b6472; font-size: 12px; }}
    .value {{ font-size: 21px; font-weight: 700; margin-top: 3px; }}
    @media (max-width: 900px) {{ .cards {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} main {{ padding: 24px 16px; }} }}
  </style>
</head>
<body>
<main>
  <h1>EgoCom Person Face Mapping Report</h1>
  <p>Generated from mapping outputs under <code>{esc(report['data_root'])}</code>. This report lists scenes and clips where representative visualizations may be absent because mappings were unresolved or conflicted.</p>
  <div class="cards">
    <div class="card"><div class="label">Scenes</div><div class="value">{total['scenes']}</div></div>
    <div class="card"><div class="label">Problem Scenes</div><div class="value">{total['problem_scenes']}</div></div>
    <div class="card"><div class="label">Resolved Segments</div><div class="value">{total['resolved_segments']}/{total['total_segments']}</div></div>
    <div class="card"><div class="label">Clips Without Reps</div><div class="value">{total['clips_without_representatives']}</div></div>
  </div>

  <h2>Split Summary</h2>
  <table>
    <thead>
      <tr><th>Split</th><th>Scenes</th><th>Problem Scenes</th><th>Resolved</th><th>Unresolved</th><th>Conflicts</th><th>Missing Emb</th><th>Low Count</th><th>Fallback Scenes</th><th>All Chunks Conflicted</th><th>Clips Without Reps</th></tr>
    </thead>
    <tbody>{''.join(split_rows)}</tbody>
  </table>

  <h2>Problem Scenes</h2>
  {''.join(scene_sections)}
</main>
</body>
</html>
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Report EgoCom person face mapping failures.")
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--output_html", type=str, default=None)
    return parser


def write_dataset_reports(data_root: Path, report: dict[str, Any]) -> list[Path]:
    written: list[Path] = []

    root_json = data_root / "person_face_mapping_report.json"
    root_html = data_root / "person_face_mapping_report.html"
    write_json(root_json, report)
    write_text(root_html, render_html(report))
    written.extend([root_json, root_html])

    scenes_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for scene in report.get("scenes", []):
        scenes_by_split[scene["split"]].append(scene)

    for split, scenes in sorted(scenes_by_split.items()):
        split_report = subset_report(report, scenes)
        split_dir = data_root / split / "person_face_mapping"
        split_json = split_dir / "report.json"
        split_html = split_dir / "report.html"
        write_json(split_json, split_report)
        write_text(split_html, render_html(split_report))
        written.extend([split_json, split_html])

        for scene in sorted(scenes, key=lambda item: item["scene_key"]):
            scene_report = subset_report(report, [scene])
            scene_dir = Path(scene["scene_dir"])
            scene_json = scene_dir / "report.json"
            scene_html = scene_dir / "report.html"
            write_json(scene_json, scene_report)
            write_text(scene_html, render_html(scene_report))
            written.extend([scene_json, scene_html])

    return written


def main() -> None:
    args = build_parser().parse_args()
    data_root = Path(args.data_root)
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    report = build_report(data_root)
    written = write_dataset_reports(data_root, report)
    if args.output_json:
        write_json(Path(args.output_json), report)
        written.append(Path(args.output_json))
    if args.output_html:
        write_text(Path(args.output_html), render_html(report))
        written.append(Path(args.output_html))

    total = report["total"]
    print(
        "Report written. "
        f"scenes={total['scenes']} "
        f"problem_scenes={total['problem_scenes']} "
        f"resolved={total['resolved_segments']}/{total['total_segments']} "
        f"files={len(written)}"
    )


if __name__ == "__main__":
    main()
