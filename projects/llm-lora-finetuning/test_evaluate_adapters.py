from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def load_evaluate_adapters_module():
    module_path = Path(__file__).with_name("evaluate_adapters.py")
    spec = importlib.util.spec_from_file_location("llm_lora_evaluate_adapters", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_load_prompts_reads_jsonl_and_skips_blank_lines(tmp_path):
    evaluate_adapters = load_evaluate_adapters_module()
    path = tmp_path / "prompts.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"id": "lora-basic", "prompt": "什么是 LoRA？"}, ensure_ascii=False),
                "",
                json.dumps({"id": "adapter-generate", "prompt": "adapter 能单独 generate 吗？"}, ensure_ascii=False),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    prompts = evaluate_adapters.load_prompts(path)

    assert prompts == [
        {"id": "lora-basic", "prompt": "什么是 LoRA？"},
        {"id": "adapter-generate", "prompt": "adapter 能单独 generate 吗？"},
    ]


def test_write_jsonl_outputs_one_record_per_result(tmp_path):
    evaluate_adapters = load_evaluate_adapters_module()
    path = tmp_path / "nested" / "results.jsonl"
    results = [
        {"run": "base", "prompt_id": "lora-basic", "prompt": "什么是 LoRA？", "response": "base answer"},
        {"run": "v4", "prompt_id": "lora-basic", "prompt": "什么是 LoRA？", "response": "adapter answer"},
    ]

    evaluate_adapters.write_jsonl(path, results)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == results


def test_write_markdown_groups_results_by_prompt(tmp_path):
    evaluate_adapters = load_evaluate_adapters_module()
    path = tmp_path / "report.md"
    results = [
        {"run": "base", "prompt_id": "lora-basic", "prompt": "什么是 LoRA？", "response": "base answer"},
        {"run": "v4", "prompt_id": "lora-basic", "prompt": "什么是 LoRA？", "response": "adapter answer"},
        {"run": "v4", "prompt_id": "qlora-adapter", "prompt": "QLoRA 会量化 adapter 吗？", "response": "no"},
    ]

    evaluate_adapters.write_markdown(path, results)

    report = path.read_text(encoding="utf-8")
    assert "# Adapter Evaluation" in report
    assert "## lora-basic" in report
    assert "**Prompt:** 什么是 LoRA？" in report
    assert "### base" in report
    assert "base answer" in report
    assert "### v4" in report
    assert "adapter answer" in report
    assert "## qlora-adapter" in report
