from __future__ import annotations

import importlib.util
import json
import random
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


def test_recommend_next_question_prefers_random_unanswered_question_from_weak_subtopic():
    study_assistant = load_study_assistant_module()
    questions = [
        {"id": "concept_lora_alpha", "prompt": "alpha?", "expected": "scaling"},
        {"id": "dataflow_attention_mask", "prompt": "mask?", "expected": "attention"},
        {"id": "dataflow_loss_ignore_prompt", "prompt": "loss?", "expected": "-100"},
        {"id": "dataflow_collator_padding", "prompt": "padding?", "expected": "batch max"},
        {"id": "dataflow_trainer_batch", "prompt": "batch?", "expected": "tensors"},
    ]
    attempts = [
        {"prompt_id": "concept_lora_alpha", "grade": "correct"},
        {"prompt_id": "dataflow_attention_mask", "grade": "wrong"},
    ]

    recommendation = study_assistant.recommend_next_question(questions, attempts, rng=random.Random(5))

    assert recommendation == {
        "prompt_id": "dataflow_loss_ignore_prompt",
        "prompt": "loss?",
        "expected": "-100",
        "category": "dataflow",
        "subtopic": "labels 与 loss",
        "reason": "优先练习薄弱知识点: labels 与 loss。",
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
        "subtopic": "labels 与 loss",
        "reason": "复习错题最多的题目: dataflow_loss_ignore_prompt。",
    }


def test_recommend_next_question_uses_random_unanswered_when_no_review_needed():
    study_assistant = load_study_assistant_module()
    questions = [
        {"id": "concept_lora_alpha", "prompt": "alpha?", "expected": "scaling"},
        {"id": "concept_lora_rank", "prompt": "rank?", "expected": "params"},
        {"id": "concept_qlora_memory", "prompt": "memory?", "expected": "4bit"},
    ]
    attempts = [{"prompt_id": "concept_lora_alpha", "grade": "correct"}]

    recommendation = study_assistant.recommend_next_question(questions, attempts, rng=random.Random(0))

    assert recommendation == {
        "prompt_id": "concept_qlora_memory",
        "prompt": "memory?",
        "expected": "4bit",
        "category": "concept",
        "subtopic": "QLoRA 量化",
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
    assert '"subtopic": "labels 与 loss"' in stdout
    assert "优先练习薄弱知识点: labels 与 loss。" in stdout


def test_cli_quiz_asks_until_quit_and_saves_attempt(tmp_path):
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
            },
            {
                "id": "concept_lora_alpha",
                "prompt": "lora_alpha 是什么？",
                "expected": "scaling。",
            },
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
            },
            {
                "prompt_id": "concept_lora_alpha",
                "required": [{"label": "scaling", "terms": ["scaling", "缩放"]}],
                "forbidden": [],
                "feedback_correct": "答到了 alpha 的作用。",
                "review_tip": "复习 lora_alpha。",
            },
        ],
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--quiz",
            "--seed",
            "1",
            "--questions",
            str(questions_path),
            "--rules",
            str(rules_path),
            "--output",
            str(output_path),
            "--attempts",
            str(output_path),
        ],
        input="因为 frozen base weights 被 4bit 量化\nq\n",
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )

    stdout = result.stdout
    assert "输入 q 退出" in stdout
    assert "自动判分: correct" in stdout
    assert "退出 quiz" in stdout
    assert "标准要点:" not in stdout

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["prompt_id"] == "concept_qlora_memory"
    assert rows[0]["grade"] == "correct"


def test_cli_quiz_seed_controls_first_random_recommendation(tmp_path):
    script_path = Path(__file__).with_name("study_assistant.py")
    questions_path = tmp_path / "questions.jsonl"
    rules_path = tmp_path / "rules.jsonl"
    output_path = tmp_path / "attempts.jsonl"
    write_jsonl(
        questions_path,
        [
            {"id": "concept_lora_alpha", "prompt": "alpha?", "expected": "scaling"},
            {"id": "concept_lora_rank", "prompt": "rank?", "expected": "params"},
            {"id": "concept_qlora_memory", "prompt": "memory?", "expected": "4bit"},
        ],
    )
    write_jsonl(
        rules_path,
        [
            {
                "prompt_id": "concept_lora_alpha",
                "required": [{"label": "scaling", "terms": ["scaling"]}],
                "forbidden": [],
            },
            {
                "prompt_id": "concept_lora_rank",
                "required": [{"label": "params", "terms": ["参数"]}],
                "forbidden": [],
            },
            {
                "prompt_id": "concept_qlora_memory",
                "required": [{"label": "4bit", "terms": ["4bit"]}],
                "forbidden": [],
            },
        ],
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--quiz",
            "--seed",
            "5",
            "--questions",
            str(questions_path),
            "--rules",
            str(rules_path),
            "--output",
            str(output_path),
            "--attempts",
            str(output_path),
        ],
        input="4bit\nq\n",
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )

    stdout = result.stdout
    assert "题目 ID: concept_qlora_memory" in stdout

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["prompt_id"] == "concept_qlora_memory"


def default_rule_for(study_assistant, prompt_id: str) -> dict[str, object]:
    rules_path = Path(__file__).with_name("data") / "study_grading_rules.jsonl"
    return study_assistant.find_grading_rule(study_assistant.load_grading_rules(rules_path), prompt_id)


def test_default_grading_rules_cover_every_curated_eval_question():
    study_assistant = load_study_assistant_module()
    questions_path = Path(__file__).with_name("data") / "curated_eval.jsonl"
    rules_path = Path(__file__).with_name("data") / "study_grading_rules.jsonl"
    questions = study_assistant.load_questions(questions_path)
    rules = study_assistant.load_grading_rules(rules_path)
    question_ids = {str(question["id"]) for question in questions}
    rules_by_id = {str(rule["prompt_id"]): rule for rule in rules}

    missing = sorted(question_ids - set(rules_by_id))
    extra = sorted(set(rules_by_id) - question_ids)
    empty_required = sorted(
        prompt_id
        for prompt_id, rule in rules_by_id.items()
        if not rule.get("required")
    )

    assert missing == []
    assert extra == []
    assert empty_required == []


def test_default_rules_cover_key_correct_partial_and_wrong_answers():
    study_assistant = load_study_assistant_module()
    cases = [
        (
            "concept_qlora_memory",
            "冻结的基础模型参数用低比特量化加载，所以显存更省",
            "correct",
            [],
        ),
        (
            "concept_qlora_memory",
            "冻结的 base 权重不训练",
            "partial",
            ["incomplete_answer"],
        ),
        (
            "concept_qlora_memory",
            "因为 LoRA adapter 参数更少",
            "wrong",
            ["concept_confusion"],
        ),
        (
            "concept_qlora_trainable_params",
            "LoRA 和 QLoRA 都训练 lora_A/lora_B，base 权重量化不改变可训练参数量",
            "correct",
            [],
        ),
        (
            "concept_qlora_trainable_params",
            "它们训练的都是 adapter",
            "partial",
            ["incomplete_answer"],
        ),
        (
            "concept_qlora_trainable_params",
            "QLoRA 的 trainable params 更少",
            "wrong",
            ["concept_confusion"],
        ),
        (
            "concept_lora_alpha",
            "alpha 不是参数量，它控制 LoRA delta 的缩放强度",
            "correct",
            [],
        ),
        (
            "concept_lora_alpha",
            "alpha 是 scaling 系数",
            "partial",
            ["incomplete_answer"],
        ),
        (
            "concept_lora_alpha",
            "alpha 越大参数数量越多",
            "wrong",
            ["concept_confusion"],
        ),
        (
            "dataflow_attention_mask",
            "attention_mask 标记真实 token 和 pad token，让注意力不要关注 padding",
            "correct",
            [],
        ),
        (
            "dataflow_attention_mask",
            "attention_mask 用来标记 padding",
            "partial",
            ["incomplete_answer"],
        ),
        (
            "dataflow_attention_mask",
            "attention_mask 让 padding 不参与 loss",
            "wrong",
            ["concept_confusion"],
        ),
        (
            "dataflow_loss_ignore_prompt",
            "labels 里设成 -100 的位置会被 loss 忽略",
            "correct",
            [],
        ),
        (
            "dataflow_loss_ignore_prompt",
            "-100 是 mask 标记",
            "partial",
            ["incomplete_answer"],
        ),
        (
            "dataflow_loss_ignore_prompt",
            "attention_mask 控制 prompt 不算 loss",
            "wrong",
            ["concept_confusion"],
        ),
        (
            "concept_adapter_only",
            "只有 adapter_model.safetensors 也可以直接聊天",
            "wrong",
            ["concept_confusion"],
        ),
        (
            "concept_target_modules_choice",
            "target_modules 不重要，随便写也不影响训练",
            "wrong",
            ["concept_confusion"],
        ),
        (
            "concept_merge_adapter",
            "merge 后还需要单独加载 lora_A 和 lora_B",
            "wrong",
            ["concept_confusion"],
        ),
        (
            "concept_target_modules_attention",
            "PEFT 会把 LoRA 加到 tokenizer 和 embedding",
            "wrong",
            ["concept_confusion"],
        ),
    ]

    for prompt_id, answer, expected_grade, expected_errors in cases:
        result = study_assistant.grade_answer(answer, default_rule_for(study_assistant, prompt_id))
        assert result["grade"] == expected_grade, prompt_id
        assert result["error_types"] == expected_errors, prompt_id


def test_default_rules_cover_real_quiz_answers_and_input_method_typos():
    study_assistant = load_study_assistant_module()
    cases = [
        (
            "concept_lora_not_pretraining",
            "不对，LoRA不属于pretraining，属于参数高效微调",
            "correct",
            [],
        ),
        (
            "concept_lora_not_pretraining",
            "LoRA 不是预训练，他是参数高校微调的方法",
            "correct",
            [],
        ),
        (
            "concept_lora_not_full_update",
            "不是，base weight不更新，只更新adapter",
            "correct",
            [],
        ),
        (
            "concept_lora_not_full_update",
            "不是，他不更新base weights,只更新Lora weights，也就是adapter里的参数",
            "correct",
            [],
        ),
        (
            "concept_lora_low_rank_delta",
            "Lora能用较小的A/B低秩矩阵近似权重增量",
            "partial",
            ["incomplete_answer"],
        ),
        (
            "concept_lora_adapter_params",
            "Lora_A Lora_B",
            "correct",
            [],
        ),
        (
            "concept_lora_alpha",
            "不是参数数量，主要参与scaling,影响Lora delta的输出",
            "correct",
            [],
        ),
        (
            "concept_lora_alpha",
            "不是，主要影响scaling,影响lora delta的输出强度",
            "correct",
            [],
        ),
        (
            "concept_qlora_memory",
            "QLora会使frozen wight量化成4bit，不是因为adapter 参数更少",
            "correct",
            [],
        ),
        (
            "concept_qlora_trainable_params",
            "Lora 和 QLora的区别只在于frozen base weights 是否被 4bit 量化加载，他们的可训练参数数量都只跟adapter有关",
            "correct",
            [],
        ),
        (
            "concept_adapter_only",
            "不能",
            "correct",
            [],
        ),
        (
            "concept_target_modules_choice",
            "因为它指定了哪些module要参与LORA训练",
            "correct",
            [],
        ),
        (
            "concept_merge_adapter",
            "不需要",
            "correct",
            [],
        ),
        (
            "concept_target_modules_attention",
            "q_proj、k_proj、v_proj、o_proj",
            "correct",
            [],
        ),
    ]

    for prompt_id, answer, expected_grade, expected_errors in cases:
        result = study_assistant.grade_answer(answer, default_rule_for(study_assistant, prompt_id))
        assert result["grade"] == expected_grade, prompt_id
        assert result["error_types"] == expected_errors, prompt_id


def test_default_rules_cover_representative_answers_for_all_new_categories():
    study_assistant = load_study_assistant_module()
    cases = [
        ("concept_qlora_adapter_not_quantized", "通常不会，adapter 保持可训练精度，量化的是 frozen base weights", "correct"),
        ("concept_quantized_base_forward", "base weight 以 qweight 量化形式存储，forward 时先反量化，再加上 LoRA delta", "correct"),
        ("concept_adapter_contents", "不是压缩后的 Qwen，主要是 lora_A/lora_B 这些 adapter 权重", "correct"),
        ("dataflow_jsonl_dataset", "变成 Dataset 后方便 map/tokenize 和批处理，也能对接 Trainer 的数据接口", "correct"),
        ("dataflow_instruction_response_format", "因为模型训练连续 token 序列，chat template 把用户问题和 assistant response 拼成一段", "correct"),
        ("dataflow_tokenizer_ids", "模型不能直接处理字符串，要变成 input_ids 这样的 token id 才能查 embedding", "correct"),
        ("dataflow_labels_copy", "labels 提供预测下一个 token 的监督目标，模型内部会 shift", "correct"),
        ("dataflow_mask_vs_loss", "不是一件事，attention_mask 管 padding 是否参与 attention，labels=-100 管 loss 忽略", "correct"),
        ("dataflow_collator_padding", "collator 把 batch 补齐，labels 的 padding 应该补 -100", "correct"),
        ("dataflow_dynamic_padding", "动态 padding 补到当前 batch 最长，固定 padding 补到 max_length，可能浪费更多 padding", "correct"),
        ("dataflow_truncation", "超过 max_length 的部分会被截断，可能丢失 response 或监督信号", "correct"),
        ("dataflow_trainable_label_filter", "如果 labels 全是 -100，这条样本不贡献 loss，训练没有意义", "correct"),
        ("dataflow_gradient_accumulation", "不是每一步更新，而是累积 8 个小 batch 后更新一次，有效 batch 变大", "correct"),
        ("dataflow_bf16_cuda", "有 CUDA GPU 时用 bf16，没有 GPU 或不支持时避免启用", "correct"),
        ("dataflow_trainer_save_adapter", "PEFT LoRA 里主要保存 adapter 权重，不保存完整 base model", "correct"),
        ("dataflow_generate_eval", "base 是对照组，对比 base 和 base+adapter 才能判断 adapter 带来的变化", "correct"),
        ("failure_adapter_without_base", "adapter 不是完整模型，缺少 base 权重和配置，不能单独作为生成模型", "correct"),
        ("failure_wrong_base_model", "base 不一致会结构或权重语义不匹配，可能加载失败或输出异常", "correct"),
        ("failure_target_module_typo", "qproject 会匹配不到目标模块，可能报错或没有正确插入 LoRA", "correct"),
        ("failure_target_module_too_few", "不一定失败，但作用范围变小，效果和表达能力可能受限", "correct"),
        ("failure_pad_label_not_ignored", "padding 位置会参与 loss，模型被要求预测 padding，loss 被无意义位置污染", "correct"),
        ("failure_prompt_label_not_ignored", "prompt 也参与 loss 时，模型会学习复现用户问题，不再只聚焦 assistant response", "correct"),
        ("failure_eval_leakage", "eval prompt 原句出现在 train 里会泄漏，模型可能只是记住答案，不代表泛化", "correct"),
        ("failure_loss_only", "loss 下降不等于一定更好，还要固定 eval 和人工判断事实正确性", "correct"),
        ("failure_hallucinated_mechanism", "这属于 hallucinated_mechanism，因为它编造了不存在的流程", "correct"),
        ("failure_incomplete_answer", "不算完全正确，是 incomplete_answer，因为漏掉必须加载 base model 这个关键条件", "correct"),
        ("boundary_0_5b_best_use", "适合学习、调试训练流程、窄任务和低成本验证，不适合复杂通用推理", "correct"),
        ("boundary_0_5b_not_teacher", "因为 0.5B 参数规模小，复杂推理和事实稳定性有限，不够可靠", "correct"),
        ("boundary_toy_dataset_value", "toy 数据适合教学和调试训练闭环，帮助理解数据、训练、评测和标注流程", "correct"),
        ("boundary_from_toy_to_real", "应该先做固定评测集，再按错题补训练数据", "correct"),
        ("boundary_public_data", "不应该随便抓，要检查质量、许可、隐私和任务相关性", "correct"),
        ("boundary_synthetic_data", "有风险，强模型生成的数据可能有幻觉，需要人工筛选和固定评测验证", "correct"),
        ("boundary_bigger_model_when", "等数据稳定、评测稳定、训练流程稳定后，再为更强能力换 3B 或 7B", "correct"),
        ("boundary_personal_assistant", "可以做窄范围复习和固定格式总结，但不能完全替代强模型或人工判断", "correct"),
        ("boundary_eval_first", "因为没有固定评测就不知道是否变好，50 题 eval 让后续补数据更有方向", "correct"),
        ("boundary_iteration_loop", "用固定 eval 找错，按错误类型补数据，再训练下一轮", "correct"),
    ]

    for prompt_id, answer, expected_grade in cases:
        result = study_assistant.grade_answer(answer, default_rule_for(study_assistant, prompt_id))
        assert result["grade"] == expected_grade, prompt_id
        assert result["error_types"] == [], prompt_id


def test_default_rules_recognize_base_frozen_when_low_rank_delta_answer_is_partial():
    study_assistant = load_study_assistant_module()

    result = study_assistant.grade_answer(
        "Lora训练中base weight是frozen的，只更改Lora A Lora B这些低秩矩阵",
        default_rule_for(study_assistant, "concept_lora_low_rank_delta"),
    )

    assert result["grade"] == "partial"
    assert result["error_types"] == ["incomplete_answer"]
    assert "weight delta approximation" in result["feedback"]
    assert "base weights frozen" not in result["feedback"]
