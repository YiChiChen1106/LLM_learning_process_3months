from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def load_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def summarize_annotations(path: Path) -> dict[str, object]:
    rows = load_jsonl(path)
    reviewed = [row for row in rows if row.get("is_correct") is not None]
    correct = [row for row in reviewed if row.get("is_correct") is True]
    incorrect = [row for row in reviewed if row.get("is_correct") is False]

    error_counts = Counter()
    for row in incorrect:
        error_counts.update(row.get("error_types", []))

    accuracy = len(correct) / len(reviewed) if reviewed else None
    return {
        "path": str(path),
        "total_rows": len(rows),
        "reviewed_rows": len(reviewed),
        "correct_rows": len(correct),
        "incorrect_rows": len(incorrect),
        "accuracy": accuracy,
        "error_type_counts": dict(sorted(error_counts.items())),
        "unreviewed_prompt_ids": [row["prompt_id"] for row in rows if row.get("is_correct") is None],
    }


def format_accuracy(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.2%}"


def write_markdown_summary(path: Path, summary: dict[str, object]) -> None:
    lines = [
        "# Evaluation Annotation Summary",
        "",
        f"- Source: `{summary['path']}`",
        f"- Total rows: {summary['total_rows']}",
        f"- Reviewed rows: {summary['reviewed_rows']}",
        f"- Correct rows: {summary['correct_rows']}",
        f"- Incorrect rows: {summary['incorrect_rows']}",
        f"- Accuracy: {format_accuracy(summary['accuracy'])}",
        "",
        "## Error Types",
        "",
        "| Error type | Count |",
        "| --- | ---: |",
    ]

    error_counts = summary["error_type_counts"]
    if error_counts:
        for error_type, count in error_counts.items():
            lines.append(f"| {error_type} | {count} |")
    else:
        lines.append("| none | 0 |")

    lines.extend(["", "## Unreviewed Prompt IDs", ""])
    unreviewed = summary["unreviewed_prompt_ids"]
    if unreviewed:
        lines.extend(f"- `{prompt_id}`" for prompt_id in unreviewed)
    else:
        lines.append("None.")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output-md", default="")
    args = parser.parse_args()

    summary = summarize_annotations(Path(args.annotations))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output_md:
        write_markdown_summary(Path(args.output_md), summary)
        print(f"已写入 {args.output_md}")


if __name__ == "__main__":
    main()
