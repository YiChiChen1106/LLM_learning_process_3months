from __future__ import annotations

import argparse
import gc
import json
from collections import OrderedDict
from pathlib import Path
from typing import Iterable


def load_prompts(path: Path) -> list[dict[str, str]]:
    prompts = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompts.append({"id": row["id"], "prompt": row["prompt"]})
    return prompts


def write_jsonl(path: Path, results: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")


def write_markdown(path: Path, results: Iterable[dict[str, str]]) -> None:
    grouped: OrderedDict[str, dict[str, object]] = OrderedDict()
    for result in results:
        prompt_id = result["prompt_id"]
        if prompt_id not in grouped:
            grouped[prompt_id] = {"prompt": result["prompt"], "items": []}
        grouped[prompt_id]["items"].append(result)

    lines = ["# Adapter Evaluation", ""]
    for prompt_id, group in grouped.items():
        lines.append(f"## {prompt_id}")
        lines.append("")
        lines.append(f"**Prompt:** {group['prompt']}")
        lines.append("")
        for item in group["items"]:
            lines.append(f"### {item['run']}")
            lines.append("")
            lines.append(item["response"].strip())
            lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def make_quantization_config(load_in_4bit: bool):
    import torch
    from transformers import BitsAndBytesConfig

    if not load_in_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def load_base_model(model_name: str, load_in_4bit: bool = False):
    import torch
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        quantization_config=make_quantization_config(load_in_4bit),
        trust_remote_code=True,
    )


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    import torch

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    device = next(model.parameters()).device
    inputs = tokenizer([text], return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = outputs[:, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens[0], skip_special_tokens=True)


def evaluate_run(run_name: str, model, tokenizer, prompts: list[dict[str, str]], max_new_tokens: int):
    model.eval()
    results = []
    for row in prompts:
        results.append(
            {
                "run": run_name,
                "prompt_id": row["id"],
                "prompt": row["prompt"],
                "response": generate_response(model, tokenizer, row["prompt"], max_new_tokens),
            }
        )
    return results


def parse_adapter_arg(value: str) -> tuple[str, str]:
    if "=" not in value:
        path = Path(value)
        return path.name, value
    name, path = value.split("=", 1)
    return name, path


def main() -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", default="eval_prompts.jsonl")
    parser.add_argument("--adapter", action="append", default=[], help="Adapter as name=path or path.")
    parser.add_argument("--output-jsonl", default="runs/eval_outputs.jsonl")
    parser.add_argument("--output-md", default="runs/eval_outputs.md")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--include-base", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load base model in 4bit for all runs.")
    args = parser.parse_args()

    prompts = load_prompts(Path(args.prompts))
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_results = []
    if args.include_base:
        model = load_base_model(args.model, load_in_4bit=args.load_in_4bit)
        all_results.extend(evaluate_run("base", model, tokenizer, prompts, args.max_new_tokens))
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for adapter in args.adapter:
        run_name, adapter_path = parse_adapter_arg(adapter)
        model = load_base_model(args.model, load_in_4bit=args.load_in_4bit)
        model = PeftModel.from_pretrained(model, adapter_path)
        all_results.extend(evaluate_run(run_name, model, tokenizer, prompts, args.max_new_tokens))
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_jsonl(Path(args.output_jsonl), all_results)
    write_markdown(Path(args.output_md), all_results)
    print(f"已写入 {args.output_jsonl}")
    print(f"已写入 {args.output_md}")


if __name__ == "__main__":
    main()
