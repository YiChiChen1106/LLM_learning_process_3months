from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict[str, str]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["_line"] = line_number
            rows.append(row)
    return rows


def normalize_text(text: str) -> str:
    return "".join(text.lower().split())


def char_bigrams(text: str) -> set[str]:
    text = normalize_text(text)
    if len(text) < 2:
        return {text} if text else set()
    return {text[i : i + 2] for i in range(len(text) - 1)}


def similarity(left: str, right: str) -> float:
    left_bigrams = char_bigrams(left)
    right_bigrams = char_bigrams(right)
    if not left_bigrams and not right_bigrams:
        return 1.0
    if not left_bigrams or not right_bigrams:
        return 0.0
    return len(left_bigrams & right_bigrams) / len(left_bigrams | right_bigrams)


def round_similarity(value: float) -> float:
    return round(value, 2)


def check_split(train_path: Path, eval_path: Path, similarity_threshold: float = 0.85) -> dict[str, object]:
    train_rows = load_jsonl(train_path)
    eval_rows = load_jsonl(eval_path)

    train_texts = [row["instruction"] for row in train_rows]
    eval_texts = [row["prompt"] for row in eval_rows]

    errors = []
    warnings = []

    normalized_train = {}
    for index, text in enumerate(train_texts):
        normalized_train.setdefault(normalize_text(text), []).append((index, text))

    for eval_index, eval_text in enumerate(eval_texts):
        for train_index, train_text in normalized_train.get(normalize_text(eval_text), []):
            errors.append(
                {
                    "type": "exact_train_eval_overlap",
                    "train_index": train_index,
                    "eval_index": eval_index,
                    "text": train_text,
                }
            )

    for train_index, train_text in enumerate(train_texts):
        for eval_index, eval_text in enumerate(eval_texts):
            if normalize_text(train_text) == normalize_text(eval_text):
                continue
            score = round_similarity(similarity(train_text, eval_text))
            if score >= similarity_threshold:
                warnings.append(
                    {
                        "type": "similar_train_eval_prompt",
                        "train_index": train_index,
                        "eval_index": eval_index,
                        "similarity": score,
                        "train_text": train_text,
                        "eval_text": eval_text,
                    }
                )

    return {
        "train_path": str(train_path),
        "eval_path": str(eval_path),
        "train_count": len(train_rows),
        "eval_count": len(eval_rows),
        "errors": errors,
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--similarity-threshold", type=float, default=0.85)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    report = check_split(Path(args.train), Path(args.eval), similarity_threshold=args.similarity_threshold)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
        print(f"已写入 {out_path}")
    else:
        print(text)

    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
