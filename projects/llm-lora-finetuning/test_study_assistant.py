from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def load_study_assistant_module():
    module_path = Path(__file__).with_name("study_assistant.py")
    spec = importlib.util.spec_from_file_location("llm_lora_study_assistant", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_load_questions_reads_jsonl_and_skips_blank_lines(tmp_path):
    study_assistant = load_study_assistant_module()
    path = tmp_path / "questions.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "concept_lora_alpha",
                        "prompt": "lora_alpha 是不是参数数量？",
                        "expected": "alpha 不是参数数量，而是 scaling。",
                    },
                    ensure_ascii=False,
                ),
                "",
                json.dumps(
                    {
                        "id": "dataflow_attention_mask",
                        "prompt": "attention_mask 控制什么？",
                        "expected": "控制哪些 token 参与 attention。",
                    },
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    questions = study_assistant.load_questions(path)

    assert questions == [
        {
            "id": "concept_lora_alpha",
            "prompt": "lora_alpha 是不是参数数量？",
            "expected": "alpha 不是参数数量，而是 scaling。",
        },
        {
            "id": "dataflow_attention_mask",
            "prompt": "attention_mask 控制什么？",
            "expected": "控制哪些 token 参与 attention。",
        },
    ]


def test_find_question_by_id_returns_matching_question(tmp_path):
    study_assistant = load_study_assistant_module()
    path = tmp_path / "questions.jsonl"
    write_jsonl(
        path,
        [
            {"id": "first", "prompt": "第一题", "expected": "A"},
            {"id": "second", "prompt": "第二题", "expected": "B"},
        ],
    )

    question = study_assistant.find_question_by_id(study_assistant.load_questions(path), "second")

    assert question == {"id": "second", "prompt": "第二题", "expected": "B"}


def test_build_attempt_record_keeps_answer_grade_errors_and_feedback():
    study_assistant = load_study_assistant_module()
    question = {
        "id": "concept_qlora_memory",
        "prompt": "QLoRA 为什么更省显存？",
        "expected": "应答出 frozen base weights 4bit 量化加载。",
    }

    record = study_assistant.build_attempt_record(
        question=question,
        user_answer="因为 adapter 参数更少",
        grade="wrong",
        error_types=["concept_confusion"],
        feedback="QLoRA 省的是 frozen base weights 的存储显存。",
        review_tip="复习 adapter 参数量 vs base weight 存储显存。",
    )

    assert record == {
        "prompt_id": "concept_qlora_memory",
        "prompt": "QLoRA 为什么更省显存？",
        "expected": "应答出 frozen base weights 4bit 量化加载。",
        "user_answer": "因为 adapter 参数更少",
        "grade": "wrong",
        "error_types": ["concept_confusion"],
        "feedback": "QLoRA 省的是 frozen base weights 的存储显存。",
        "review_tip": "复习 adapter 参数量 vs base weight 存储显存。",
    }


def test_append_attempts_writes_jsonl_records(tmp_path):
    study_assistant = load_study_assistant_module()
    path = tmp_path / "nested" / "attempts.jsonl"
    records = [
        {
            "prompt_id": "concept_lora_alpha",
            "prompt": "lora_alpha 是不是参数数量？",
            "expected": "不是参数数量。",
            "user_answer": "不是",
            "grade": "partial",
            "error_types": ["incomplete_answer"],
            "feedback": "还要说 alpha 影响 scaling。",
            "review_tip": "复习 lora_alpha 和 rank 的区别。",
        }
    ]

    study_assistant.append_attempts(path, records)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == records


def test_cli_writes_utf8_stdout_for_chinese_text(tmp_path):
    script_path = Path(__file__).with_name("study_assistant.py")
    questions_path = tmp_path / "questions.jsonl"
    output_path = tmp_path / "attempts.jsonl"
    write_jsonl(
        questions_path,
        [
            {
                "id": "concept_qlora_memory",
                "prompt": "QLoRA 为什么更省显存？",
                "expected": "量化 frozen base weights。",
            }
        ],
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--questions",
            str(questions_path),
            "--prompt-id",
            "concept_qlora_memory",
            "--answer",
            "因为 base weight 被量化",
            "--output",
            str(output_path),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    stdout = result.stdout.decode("utf-8")
    assert "题目 ID: concept_qlora_memory" in stdout
    assert "QLoRA 为什么更省显存？" in stdout


def test_load_grading_rules_reads_jsonl_and_skips_blank_lines(tmp_path):
    study_assistant = load_study_assistant_module()
    path = tmp_path / "rules.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "prompt_id": "concept_qlora_memory",
                        "required": [{"label": "base weights", "terms": ["base weights"]}],
                        "forbidden": [],
                        "feedback_correct": "答到了核心点。",
                        "review_tip": "复习 QLoRA。",
                    },
                    ensure_ascii=False,
                ),
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rules = study_assistant.load_grading_rules(path)

    assert rules == [
        {
            "prompt_id": "concept_qlora_memory",
            "required": [{"label": "base weights", "terms": ["base weights"]}],
            "forbidden": [],
            "feedback_correct": "答到了核心点。",
            "review_tip": "复习 QLoRA。",
        }
    ]


def test_grade_answer_marks_correct_when_required_groups_are_hit():
    study_assistant = load_study_assistant_module()
    rule = {
        "prompt_id": "concept_qlora_memory",
        "required": [
            {"label": "frozen base weights", "terms": ["frozen base weights", "base weights"]},
            {"label": "4bit quantization", "terms": ["4bit", "量化"]},
        ],
        "forbidden": [],
        "feedback_correct": "答到了 QLoRA 省显存的核心原因。",
        "review_tip": "复习 QLoRA 省显存来源。",
    }

    result = study_assistant.grade_answer("因为 frozen base weights 被 4bit 量化加载", rule)

    assert result == {
        "grade": "correct",
        "error_types": [],
        "feedback": "答到了 QLoRA 省显存的核心原因。",
        "review_tip": "复习 QLoRA 省显存来源。",
    }


def test_grade_answer_marks_wrong_when_forbidden_term_is_hit():
    study_assistant = load_study_assistant_module()
    rule = {
        "prompt_id": "concept_qlora_memory",
        "required": [
            {"label": "frozen base weights", "terms": ["frozen base weights"]},
            {"label": "4bit quantization", "terms": ["4bit", "量化"]},
        ],
        "forbidden": [
            {
                "terms": ["adapter 参数更少", "adapter更少"],
                "error_type": "concept_confusion",
                "feedback": "QLoRA 不是靠 adapter 参数更少省显存。",
            }
        ],
        "feedback_correct": "答到了核心原因。",
        "review_tip": "复习 adapter 参数量 vs base weight 存储显存。",
    }

    result = study_assistant.grade_answer("因为 adapter 参数更少", rule)

    assert result == {
        "grade": "wrong",
        "error_types": ["concept_confusion"],
        "feedback": "QLoRA 不是靠 adapter 参数更少省显存。",
        "review_tip": "复习 adapter 参数量 vs base weight 存储显存。",
    }


def test_grade_answer_marks_partial_when_some_required_groups_are_missing():
    study_assistant = load_study_assistant_module()
    rule = {
        "prompt_id": "concept_qlora_memory",
        "required": [
            {"label": "frozen base weights", "terms": ["frozen base weights", "base weights"]},
            {"label": "4bit quantization", "terms": ["4bit", "量化"]},
        ],
        "forbidden": [],
        "feedback_correct": "答到了核心原因。",
        "review_tip": "复习 QLoRA 省显存来源。",
    }

    result = study_assistant.grade_answer("因为 frozen base weights 不训练", rule)

    assert result == {
        "grade": "partial",
        "error_types": ["incomplete_answer"],
        "feedback": "还缺少关键点: 4bit quantization。",
        "review_tip": "复习 QLoRA 省显存来源。",
    }


def test_cli_auto_grade_writes_grading_result(tmp_path):
    script_path = Path(__file__).with_name("study_assistant.py")
    questions_path = tmp_path / "questions.jsonl"
    rules_path = tmp_path / "rules.jsonl"
    output_path = tmp_path / "attempts.jsonl"
    write_jsonl(
        questions_path,
        [
            {
                "id": "concept_qlora_memory",
                "prompt": "QLoRA 为什么更省显存？",
                "expected": "量化 frozen base weights。",
            }
        ],
    )
    write_jsonl(
        rules_path,
        [
            {
                "prompt_id": "concept_qlora_memory",
                "required": [
                    {"label": "frozen base weights", "terms": ["frozen base weights", "base weights"]},
                    {"label": "4bit quantization", "terms": ["4bit", "量化"]},
                ],
                "forbidden": [],
                "feedback_correct": "答到了核心原因。",
                "review_tip": "复习 QLoRA 省显存来源。",
            }
        ],
    )

    subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--questions",
            str(questions_path),
            "--rules",
            str(rules_path),
            "--auto-grade",
            "--prompt-id",
            "concept_qlora_memory",
            "--answer",
            "因为 frozen base weights 会被 4bit 量化",
            "--output",
            str(output_path),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    record = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
    assert record["grade"] == "correct"
    assert record["error_types"] == []
    assert record["feedback"] == "答到了核心原因。"


def test_summarize_attempts_counts_grades_errors_categories_and_review_tips():
    study_assistant = load_study_assistant_module()
    records = [
        {
            "prompt_id": "concept_qlora_memory",
            "grade": "correct",
            "error_types": [],
            "review_tip": "",
        },
        {
            "prompt_id": "concept_lora_alpha",
            "grade": "partial",
            "error_types": ["incomplete_answer"],
            "review_tip": "复习 lora_alpha 和 rank 的区别。",
        },
        {
            "prompt_id": "dataflow_attention_mask",
            "grade": "wrong",
            "error_types": ["concept_confusion"],
            "review_tip": "复习 attention_mask vs labels=-100。",
        },
        {
            "prompt_id": "dataflow_loss_ignore_prompt",
            "grade": "wrong",
            "error_types": ["concept_confusion", "factual_error"],
            "review_tip": "复习 attention_mask vs labels=-100。",
        },
        {
            "prompt_id": "boundary_0_5b_best_use",
            "grade": "ungraded",
            "error_types": [],
            "review_tip": "",
        },
    ]

    summary = study_assistant.summarize_attempts(records)

    assert summary == {
        "total_attempts": 5,
        "graded_attempts": 4,
        "needs_review_attempts": 3,
        "accuracy": 0.25,
        "grade_counts": {"correct": 1, "partial": 1, "ungraded": 1, "wrong": 2},
        "error_type_counts": {
            "concept_confusion": 2,
            "factual_error": 1,
            "incomplete_answer": 1,
        },
        "category_counts": {"boundary": 1, "concept": 2, "dataflow": 2},
        "needs_review_by_category": {"concept": 1, "dataflow": 2},
        "weak_prompt_counts": {
            "concept_lora_alpha": 1,
            "dataflow_attention_mask": 1,
            "dataflow_loss_ignore_prompt": 1,
        },
        "review_tips": [
            "复习 attention_mask vs labels=-100。",
            "复习 lora_alpha 和 rank 的区别。",
        ],
    }


def test_write_review_markdown_includes_core_summary(tmp_path):
    study_assistant = load_study_assistant_module()
    path = tmp_path / "review.md"
    summary = {
        "total_attempts": 3,
        "graded_attempts": 3,
        "needs_review_attempts": 2,
        "accuracy": 1 / 3,
        "grade_counts": {"correct": 1, "partial": 1, "wrong": 1},
        "error_type_counts": {"concept_confusion": 1, "incomplete_answer": 1},
        "category_counts": {"concept": 2, "dataflow": 1},
        "needs_review_by_category": {"concept": 2},
        "weak_prompt_counts": {"concept_qlora_memory": 2},
        "review_tips": ["复习 QLoRA 省显存来源。"],
    }

    study_assistant.write_review_markdown(path, summary)

    text = path.read_text(encoding="utf-8")
    assert "# Study Attempt Review" in text
    assert "- Accuracy: 33.33%" in text
    assert "| concept_confusion | 1 |" in text
    assert "- 复习 QLoRA 省显存来源。" in text


def test_cli_review_writes_markdown_summary(tmp_path):
    script_path = Path(__file__).with_name("study_assistant.py")
    attempts_path = tmp_path / "attempts.jsonl"
    review_path = tmp_path / "review.md"
    write_jsonl(
        attempts_path,
        [
            {
                "prompt_id": "concept_qlora_memory",
                "grade": "wrong",
                "error_types": ["concept_confusion"],
                "review_tip": "复习 QLoRA 省显存来源。",
            }
        ],
    )

    subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--review",
            "--attempts",
            str(attempts_path),
            "--review-md",
            str(review_path),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    text = review_path.read_text(encoding="utf-8")
    assert "- Needs review attempts: 1" in text
    assert "| concept_confusion | 1 |" in text


def test_recommend_next_question_prefers_unanswered_question_from_weak_category():
    study_assistant = load_study_assistant_module()
    questions = [
        {"id": "concept_lora_alpha", "prompt": "alpha?", "expected": "scaling"},
        {"id": "dataflow_attention_mask", "prompt": "mask?", "expected": "attention"},
        {"id": "dataflow_loss_ignore_prompt", "prompt": "loss?", "expected": "-100"},
        {"id": "dataflow_collator_padding", "prompt": "padding?", "expected": "batch max"},
    ]
    attempts = [
        {"prompt_id": "concept_lora_alpha", "grade": "correct"},
        {"prompt_id": "dataflow_attention_mask", "grade": "wrong"},
        {"prompt_id": "dataflow_loss_ignore_prompt", "grade": "partial"},
    ]

    recommendation = study_assistant.recommend_next_question(questions, attempts)

    assert recommendation == {
        "prompt_id": "dataflow_collator_padding",
        "prompt": "padding?",
        "expected": "batch max",
        "category": "dataflow",
        "reason": "优先练习薄弱类别: dataflow。",
    }


def test_recommend_next_question_falls_back_to_weak_prompt_when_category_is_exhausted():
    study_assistant = load_study_assistant_module()
    questions = [
        {"id": "dataflow_attention_mask", "prompt": "mask?", "expected": "attention"},
        {"id": "dataflow_loss_ignore_prompt", "prompt": "loss?", "expected": "-100"},
    ]
    attempts = [
        {"prompt_id": "dataflow_attention_mask", "grade": "wrong"},
        {"prompt_id": "dataflow_loss_ignore_prompt", "grade": "wrong"},
        {"prompt_id": "dataflow_loss_ignore_prompt", "grade": "partial"},
    ]

    recommendation = study_assistant.recommend_next_question(questions, attempts)

    assert recommendation == {
        "prompt_id": "dataflow_loss_ignore_prompt",
        "prompt": "loss?",
        "expected": "-100",
        "category": "dataflow",
        "reason": "复习错题最多的题目: dataflow_loss_ignore_prompt。",
    }


def test_recommend_next_question_uses_first_unanswered_when_no_review_needed():
    study_assistant = load_study_assistant_module()
    questions = [
        {"id": "concept_lora_alpha", "prompt": "alpha?", "expected": "scaling"},
        {"id": "concept_lora_rank", "prompt": "rank?", "expected": "params"},
    ]
    attempts = [{"prompt_id": "concept_lora_alpha", "grade": "correct"}]

    recommendation = study_assistant.recommend_next_question(questions, attempts)

    assert recommendation == {
        "prompt_id": "concept_lora_rank",
        "prompt": "rank?",
        "expected": "params",
        "category": "concept",
        "reason": "没有明显薄弱错题，推荐未作答题目。",
    }


def test_cli_recommend_next_prints_recommendation(tmp_path):
    script_path = Path(__file__).with_name("study_assistant.py")
    questions_path = tmp_path / "questions.jsonl"
    attempts_path = tmp_path / "attempts.jsonl"
    write_jsonl(
        questions_path,
        [
            {"id": "concept_lora_alpha", "prompt": "alpha?", "expected": "scaling"},
            {"id": "dataflow_attention_mask", "prompt": "mask?", "expected": "attention"},
            {"id": "dataflow_loss_ignore_prompt", "prompt": "loss?", "expected": "-100"},
        ],
    )
    write_jsonl(
        attempts_path,
        [
            {"prompt_id": "dataflow_attention_mask", "grade": "wrong"},
        ],
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--recommend-next",
            "--questions",
            str(questions_path),
            "--attempts",
            str(attempts_path),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    stdout = result.stdout.decode("utf-8")
    assert '"prompt_id": "dataflow_loss_ignore_prompt"' in stdout
    assert "优先练习薄弱类别: dataflow。" in stdout
