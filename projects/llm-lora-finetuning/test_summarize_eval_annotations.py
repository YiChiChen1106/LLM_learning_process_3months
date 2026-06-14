from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def load_summarize_eval_annotations_module():
    module_path = Path(__file__).with_name("summarize_eval_annotations.py")
    spec = importlib.util.spec_from_file_location("llm_lora_summarize_eval_annotations", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_summarize_annotations_counts_accuracy_and_error_types(tmp_path):
    summarize_eval_annotations = load_summarize_eval_annotations_module()
    path = tmp_path / "annotations.jsonl"
    write_jsonl(
        path,
        [
            {
                "prompt_id": "lora",
                "run": "adapter-v1",
                "is_correct": True,
                "error_types": [],
                "note": "",
            },
            {
                "prompt_id": "qlora",
                "run": "adapter-v1",
                "is_correct": False,
                "error_types": ["factual_error", "concept_confusion"],
                "note": "Says QLoRA quantizes adapter.",
            },
            {
                "prompt_id": "adapter",
                "run": "adapter-v1",
                "is_correct": None,
                "error_types": [],
                "note": "not reviewed yet",
            },
        ],
    )

    summary = summarize_eval_annotations.summarize_annotations(path)

    assert summary == {
        "path": str(path),
        "total_rows": 3,
        "reviewed_rows": 2,
        "correct_rows": 1,
        "incorrect_rows": 1,
        "accuracy": 0.5,
        "error_type_counts": {"concept_confusion": 1, "factual_error": 1},
        "unreviewed_prompt_ids": ["adapter"],
    }


def test_write_markdown_summary(tmp_path):
    summarize_eval_annotations = load_summarize_eval_annotations_module()
    path = tmp_path / "summary.md"
    summary = {
        "path": "annotations.jsonl",
        "total_rows": 2,
        "reviewed_rows": 2,
        "correct_rows": 1,
        "incorrect_rows": 1,
        "accuracy": 0.5,
        "error_type_counts": {"factual_error": 1},
        "unreviewed_prompt_ids": [],
    }

    summarize_eval_annotations.write_markdown_summary(path, summary)

    text = path.read_text(encoding="utf-8")
    assert "# Evaluation Annotation Summary" in text
    assert "- Accuracy: 50.00%" in text
    assert "| factual_error | 1 |" in text
