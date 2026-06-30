from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path


GRADE_CHOICES = ("correct", "partial", "wrong", "ungraded")


def load_questions(path: Path) -> list[dict[str, object]]:
    questions = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            questions.append(json.loads(line))
    return questions


def load_grading_rules(path: Path) -> list[dict[str, object]]:
    rules = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rules.append(json.loads(line))
    return rules


def find_question_by_id(questions: list[dict[str, object]], prompt_id: str) -> dict[str, object]:
    for question in questions:
        if question.get("id") == prompt_id:
            return question
    raise ValueError(f"Question not found: {prompt_id}")


def find_grading_rule(rules: list[dict[str, object]], prompt_id: str) -> dict[str, object]:
    for rule in rules:
        if rule.get("prompt_id") == prompt_id:
            return rule
    raise ValueError(f"Grading rule not found: {prompt_id}")


def choose_question(
    questions: list[dict[str, object]],
    prompt_id: str | None = None,
    seed: int | None = None,
) -> dict[str, object]:
    if not questions:
        raise ValueError("No questions loaded.")
    if prompt_id:
        return find_question_by_id(questions, prompt_id)
    rng = random.Random(seed)
    return rng.choice(questions)


def normalize_text(text: str) -> str:
    return "".join(text.lower().split())


def contains_any(answer: str, terms: list[str]) -> bool:
    normalized_answer = normalize_text(answer)
    return any(normalize_text(term) in normalized_answer for term in terms)


def is_negated_match(answer: str, term: str) -> bool:
    normalized_answer = normalize_text(answer)
    normalized_term = normalize_text(term)
    negation_prefixes = [
        "不是因为",
        "并不是因为",
        "不是由于",
        "并不是由于",
        "不是靠",
        "不靠",
        "不算",
        "不能算",
        "不能",
        "不是",
        "并不是",
        "不",
    ]
    return any(f"{prefix}{normalized_term}" in normalized_answer for prefix in negation_prefixes)


def contains_non_negated_any(answer: str, terms: list[str]) -> bool:
    normalized_answer = normalize_text(answer)
    for term in terms:
        normalized_term = normalize_text(term)
        if normalized_term in normalized_answer and not is_negated_match(answer, term):
            return True
    return False


def unique_values(values: list[str]) -> list[str]:
    seen = set()
    unique = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def grade_answer(user_answer: str, rule: dict[str, object]) -> dict[str, object]:
    review_tip = str(rule.get("review_tip", ""))
    forbidden_feedbacks = []
    error_types = []
    for forbidden in rule.get("forbidden", []):
        if contains_non_negated_any(user_answer, forbidden.get("terms", [])):
            error_types.append(str(forbidden.get("error_type", "factual_error")))
            feedback = str(forbidden.get("feedback", "命中了错误说法。"))
            forbidden_feedbacks.append(feedback)

    if forbidden_feedbacks:
        return {
            "grade": "wrong",
            "error_types": unique_values(error_types),
            "feedback": " ".join(forbidden_feedbacks),
            "review_tip": review_tip,
        }

    required = rule.get("required", [])
    missing_labels = [
        str(group.get("label", "required point"))
        for group in required
        if not contains_any(user_answer, group.get("terms", []))
    ]
    matched_count = len(required) - len(missing_labels)

    if not missing_labels:
        return {
            "grade": "correct",
            "error_types": [],
            "feedback": str(rule.get("feedback_correct", "答到了核心要点。")),
            "review_tip": review_tip,
        }

    grade = "partial" if matched_count > 0 else "wrong"
    return {
        "grade": grade,
        "error_types": ["incomplete_answer"],
        "feedback": f"还缺少关键点: {', '.join(missing_labels)}。",
        "review_tip": review_tip,
    }


def build_attempt_record(
    question: dict[str, object],
    user_answer: str,
    grade: str = "ungraded",
    error_types: list[str] | None = None,
    feedback: str = "",
    review_tip: str = "",
) -> dict[str, object]:
    if grade not in GRADE_CHOICES:
        raise ValueError(f"grade must be one of: {', '.join(GRADE_CHOICES)}")
    return {
        "prompt_id": question["id"],
        "prompt": question["prompt"],
        "expected": question["expected"],
        "user_answer": user_answer,
        "grade": grade,
        "error_types": error_types or [],
        "feedback": feedback,
        "review_tip": review_tip,
    }


def append_attempts(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_attempts(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    attempts = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            attempts.append(json.loads(line))
    return attempts


def prompt_category(prompt_id: object) -> str:
    text = str(prompt_id or "unknown")
    return text.split("_", 1)[0] if "_" in text else text


def prompt_subtopic(prompt_id: object) -> str:
    text = str(prompt_id or "unknown")
    if "qlora" in text or "quantized_base" in text:
        return "QLoRA 量化"
    if "lora_" in text:
        return "LoRA 基础"
    if "adapter" in text or "merge" in text:
        return "adapter 与 merge"
    if "target_module" in text:
        return "target_modules"
    if any(key in text for key in ["jsonl", "dataset", "instruction_response", "tokenizer"]):
        return "Dataset / tokenizer"
    if any(key in text for key in ["attention_mask", "labels", "loss", "label_filter", "mask_vs_loss"]):
        return "labels 与 loss"
    if any(key in text for key in ["collator", "padding", "truncation"]):
        return "padding / collator"
    if any(key in text for key in ["gradient_accumulation", "bf16", "save_adapter"]):
        return "训练配置"
    if any(key in text for key in ["generate_eval", "eval_", "iteration_loop"]):
        return "评测闭环"
    if text.startswith("failure_"):
        return "失败模式"
    if any(key in text for key in ["0_5b", "bigger_model", "personal_assistant"]):
        return "模型能力边界"
    if any(key in text for key in ["toy_dataset", "from_toy", "public_data", "synthetic_data"]):
        return "数据来源与质量"
    return prompt_category(text)


def sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items()))


def top_counter_item(counter: Counter[str]) -> str:
    return sorted(counter.items(), key=lambda item: (-item[1], item[0]))[0][0]


def is_needs_review(record: dict[str, object]) -> bool:
    return str(record.get("grade")) in {"partial", "wrong"}


def summarize_attempts(records: list[dict[str, object]]) -> dict[str, object]:
    grade_counts = Counter(str(record.get("grade") or "ungraded") for record in records)
    category_counts = Counter(prompt_category(record.get("prompt_id")) for record in records)

    graded = [record for record in records if str(record.get("grade") or "ungraded") != "ungraded"]
    needs_review = [
        record
        for record in graded
        if str(record.get("grade")) in {"partial", "wrong"}
    ]

    error_type_counts = Counter()
    for record in needs_review:
        error_type_counts.update(str(error_type) for error_type in record.get("error_types", []))

    needs_review_by_category = Counter(prompt_category(record.get("prompt_id")) for record in needs_review)
    weak_prompt_counts = Counter(str(record.get("prompt_id")) for record in needs_review)
    review_tips = sorted(
        {
            str(record.get("review_tip"))
            for record in needs_review
            if str(record.get("review_tip") or "").strip()
        }
    )

    correct_count = grade_counts.get("correct", 0)
    accuracy = correct_count / len(graded) if graded else None
    return {
        "total_attempts": len(records),
        "graded_attempts": len(graded),
        "needs_review_attempts": len(needs_review),
        "accuracy": accuracy,
        "grade_counts": sorted_counter(grade_counts),
        "error_type_counts": sorted_counter(error_type_counts),
        "category_counts": sorted_counter(category_counts),
        "needs_review_by_category": sorted_counter(needs_review_by_category),
        "weak_prompt_counts": sorted_counter(weak_prompt_counts),
        "review_tips": review_tips,
    }


def build_recommendation(question: dict[str, object], reason: str) -> dict[str, object]:
    prompt_id = str(question["id"])
    return {
        "prompt_id": prompt_id,
        "prompt": question["prompt"],
        "expected": question["expected"],
        "category": prompt_category(prompt_id),
        "subtopic": prompt_subtopic(prompt_id),
        "reason": reason,
    }


def recommend_next_question(
    questions: list[dict[str, object]],
    records: list[dict[str, object]],
    rng: random.Random | None = None,
) -> dict[str, object]:
    if not questions:
        raise ValueError("No questions loaded.")

    rng = rng or random.Random()
    questions_by_id = {str(question["id"]): question for question in questions}
    attempted_ids = {str(record.get("prompt_id")) for record in records if record.get("prompt_id")}
    needs_review = [record for record in records if is_needs_review(record)]

    if needs_review:
        weak_subtopic_counts = Counter(prompt_subtopic(record.get("prompt_id")) for record in needs_review)
        weak_subtopic = top_counter_item(weak_subtopic_counts)
        weak_subtopic_candidates = [
            question
            for question in questions
            if prompt_subtopic(str(question["id"])) == weak_subtopic
            and str(question["id"]) not in attempted_ids
        ]
        if weak_subtopic_candidates:
            return build_recommendation(
                rng.choice(weak_subtopic_candidates),
                f"优先练习薄弱知识点: {weak_subtopic}。",
            )

        weak_category_counts = Counter(prompt_category(record.get("prompt_id")) for record in needs_review)
        weak_category = top_counter_item(weak_category_counts)
        weak_category_candidates = [
            question
            for question in questions
            if prompt_category(str(question["id"])) == weak_category
            and str(question["id"]) not in attempted_ids
        ]
        if weak_category_candidates:
            return build_recommendation(
                rng.choice(weak_category_candidates),
                f"优先练习薄弱类别: {weak_category}。",
            )

        weak_prompt_counts = Counter(
            str(record.get("prompt_id"))
            for record in needs_review
            if str(record.get("prompt_id")) in questions_by_id
        )
        if weak_prompt_counts:
            prompt_id = top_counter_item(weak_prompt_counts)
            return build_recommendation(
                questions_by_id[prompt_id],
                f"复习错题最多的题目: {prompt_id}。",
            )

    unanswered_candidates = [
        question for question in questions if str(question["id"]) not in attempted_ids
    ]
    if unanswered_candidates:
        return build_recommendation(
            rng.choice(unanswered_candidates),
            "没有明显薄弱错题，推荐未作答题目。",
        )

    return build_recommendation(questions[0], "所有题都已有记录，推荐从第一题开始复习。")


def grade_with_rules(
    user_answer: str,
    rules: list[dict[str, object]],
    prompt_id: str,
) -> dict[str, object]:
    try:
        return grade_answer(user_answer, find_grading_rule(rules, prompt_id))
    except ValueError:
        return {
            "grade": "ungraded",
            "error_types": [],
            "feedback": "这道题还没有自动判分规则，需要人工检查。",
            "review_tip": "",
        }


def run_quiz(
    questions: list[dict[str, object]],
    rules: list[dict[str, object]],
    attempts: list[dict[str, object]],
    output_path: Path,
    rng: random.Random | None = None,
) -> None:
    rng = rng or random.Random()
    print("进入 quiz 模式，输入 q 退出。")
    while True:
        recommendation = recommend_next_question(questions, attempts, rng=rng)
        question = find_question_by_id(questions, recommendation["prompt_id"])
        print("")
        print(f"推荐原因: {recommendation['reason']}")
        print(f"题目 ID: {question['id']}")
        print(f"题目: {question['prompt']}")
        user_answer = input("你的回答: ").strip()
        if user_answer.lower() in {"q", "quit", "exit"}:
            print("退出 quiz。")
            break

        grading = grade_with_rules(user_answer, rules, str(question["id"]))
        print(f"自动判分: {grading['grade']}")
        if grading["feedback"]:
            print(f"反馈: {grading['feedback']}")
        if grading["review_tip"]:
            print(f"复习建议: {grading['review_tip']}")

        record = build_attempt_record(
            question=question,
            user_answer=user_answer,
            grade=str(grading["grade"]),
            error_types=grading["error_types"],
            feedback=str(grading["feedback"]),
            review_tip=str(grading["review_tip"]),
        )
        append_attempts(output_path, [record])
        attempts.append(record)
        print(f"已保存作答记录: {output_path}")


def format_accuracy(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.2%}"


def append_table(lines: list[str], rows: dict[str, int]) -> None:
    lines.extend(["| Item | Count |", "| --- | ---: |"])
    if rows:
        lines.extend(f"| {item} | {count} |" for item, count in rows.items())
    else:
        lines.append("| none | 0 |")


def write_review_markdown(path: Path, summary: dict[str, object]) -> None:
    lines = [
        "# Study Attempt Review",
        "",
        f"- Total attempts: {summary['total_attempts']}",
        f"- Graded attempts: {summary['graded_attempts']}",
        f"- Needs review attempts: {summary['needs_review_attempts']}",
        f"- Accuracy: {format_accuracy(summary['accuracy'])}",
        "",
        "## Grade Counts",
        "",
    ]
    append_table(lines, summary["grade_counts"])

    lines.extend(["", "## Error Types", ""])
    append_table(lines, summary["error_type_counts"])

    lines.extend(["", "## Category Counts", ""])
    append_table(lines, summary["category_counts"])

    lines.extend(["", "## Needs Review By Category", ""])
    append_table(lines, summary["needs_review_by_category"])

    lines.extend(["", "## Weak Prompt IDs", ""])
    append_table(lines, summary["weak_prompt_counts"])

    lines.extend(["", "## Review Tips", ""])
    review_tips = summary["review_tips"]
    if review_tips:
        lines.extend(f"- {tip}" for tip in review_tips)
    else:
        lines.append("None.")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def default_questions_path() -> Path:
    return Path(__file__).with_name("data") / "curated_eval.jsonl"


def default_grading_rules_path() -> Path:
    return Path(__file__).with_name("data") / "study_grading_rules.jsonl"


def default_output_path() -> Path:
    return Path(__file__).parents[2] / "outputs" / "study_attempts.jsonl"


def configure_stdio() -> None:
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    configure_stdio()
    parser = argparse.ArgumentParser(description="Ask one study question and save an attempt record.")
    parser.add_argument("--quiz", action="store_true")
    parser.add_argument("--review", action="store_true")
    parser.add_argument("--recommend-next", action="store_true")
    parser.add_argument("--questions", type=Path, default=default_questions_path())
    parser.add_argument("--rules", type=Path, default=default_grading_rules_path())
    parser.add_argument("--output", type=Path, default=default_output_path())
    parser.add_argument("--attempts", type=Path, default=default_output_path())
    parser.add_argument("--review-md", type=Path, default=None)
    parser.add_argument("--prompt-id", default="")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--answer", default="")
    parser.add_argument("--auto-grade", action="store_true")
    parser.add_argument("--grade", choices=GRADE_CHOICES, default="ungraded")
    parser.add_argument("--error-type", action="append", default=[])
    parser.add_argument("--feedback", default="")
    parser.add_argument("--review-tip", default="")
    args = parser.parse_args()

    if args.review:
        summary = summarize_attempts(load_attempts(args.attempts))
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if args.review_md:
            write_review_markdown(args.review_md, summary)
            print(f"已写入复盘报告: {args.review_md}")
        return

    if args.recommend_next:
        recommendation = recommend_next_question(
            load_questions(args.questions),
            load_attempts(args.attempts),
            rng=random.Random(args.seed),
        )
        print(json.dumps(recommendation, ensure_ascii=False, indent=2))
        return

    if args.quiz:
        run_quiz(
            questions=load_questions(args.questions),
            rules=load_grading_rules(args.rules),
            attempts=load_attempts(args.attempts),
            output_path=args.output,
            rng=random.Random(args.seed),
        )
        return

    questions = load_questions(args.questions)
    question = choose_question(questions, prompt_id=args.prompt_id or None, seed=args.seed)

    print(f"题目 ID: {question['id']}")
    print(f"题目: {question['prompt']}")
    print(f"标准要点: {question['expected']}")
    user_answer = args.answer or input("你的回答: ").strip()

    if args.auto_grade:
        rules = load_grading_rules(args.rules)
        grading = grade_with_rules(user_answer, rules, str(question["id"]))
        args.grade = grading["grade"]
        args.error_type = grading["error_types"]
        args.feedback = grading["feedback"]
        args.review_tip = grading["review_tip"]
        print(f"自动判分: {args.grade}")
        if args.feedback:
            print(f"反馈: {args.feedback}")

    record = build_attempt_record(
        question=question,
        user_answer=user_answer,
        grade=args.grade,
        error_types=args.error_type,
        feedback=args.feedback,
        review_tip=args.review_tip,
    )
    append_attempts(args.output, [record])
    print(f"已保存作答记录: {args.output}")


if __name__ == "__main__":
    main()
