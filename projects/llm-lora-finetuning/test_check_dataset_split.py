from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def load_check_dataset_split_module():
    module_path = Path(__file__).with_name("check_dataset_split.py")
    spec = importlib.util.spec_from_file_location("llm_lora_check_dataset_split", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_check_split_accepts_clean_rephrased_eval(tmp_path):
    check_dataset_split = load_check_dataset_split_module()
    train = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    write_jsonl(
        train,
        [
            {"instruction": "LoRA 是什么？", "response": "LoRA 是参数高效微调方法。"},
            {"instruction": "QLoRA 为什么省显存？", "response": "QLoRA 量化 frozen base weights。"},
        ],
    )
    write_jsonl(
        eval_path,
        [
            {"id": "lora_rephrase", "prompt": "有人说 LoRA 属于 pretraining，这对吗？"},
            {"id": "qlora_rephrase", "prompt": "QLoRA 省显存是不是因为 adapter 更少？"},
        ],
    )

    report = check_dataset_split.check_split(train, eval_path)

    assert report["train_count"] == 2
    assert report["eval_count"] == 2
    assert report["errors"] == []


def test_check_split_reports_exact_train_eval_overlap(tmp_path):
    check_dataset_split = load_check_dataset_split_module()
    train = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    write_jsonl(train, [{"instruction": "LoRA 是预训练方法吗？", "response": "不是。"}])
    write_jsonl(eval_path, [{"id": "leak", "prompt": "LoRA 是预训练方法吗？"}])

    report = check_dataset_split.check_split(train, eval_path)

    assert report["errors"] == [
        {
            "type": "exact_train_eval_overlap",
            "train_index": 0,
            "eval_index": 0,
            "text": "LoRA 是预训练方法吗？",
        }
    ]


def test_check_split_warns_about_high_similarity(tmp_path):
    check_dataset_split = load_check_dataset_split_module()
    train = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    write_jsonl(train, [{"instruction": "adapter checkpoint 为什么不能单独 generate？", "response": "不是完整模型。"}])
    write_jsonl(eval_path, [{"id": "near_leak", "prompt": "adapter checkpoint 为什么不能直接 generate？"}])

    report = check_dataset_split.check_split(train, eval_path, similarity_threshold=0.7)

    assert report["errors"] == []
    assert report["warnings"] == [
        {
            "type": "similar_train_eval_prompt",
            "train_index": 0,
            "eval_index": 0,
            "similarity": 0.82,
            "train_text": "adapter checkpoint 为什么不能单独 generate？",
            "eval_text": "adapter checkpoint 为什么不能直接 generate？",
        }
    ]
