from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments


def load_jsonl(path: str) -> Dataset:
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return Dataset.from_list(rows)


def format_prompt(example: dict[str, str]) -> str:
    return (
        "<|im_start|>user\n"
        f"{example['instruction']}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def format_example(example: dict[str, str]) -> str:
    return format_prompt(example) + f"{example['response']}<|im_end|>"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with Path(args.config).open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if cfg.get("load_in_4bit", False):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        quantization_config=quantization_config,
        trust_remote_code=True,
    )
    if cfg.get("load_in_4bit", False):
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=cfg["lora_rank"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_jsonl(cfg["train_file"])

    def tokenize(example: dict[str, str]) -> dict[str, list[int]]:
        prompt = format_prompt(example)
        text = format_example(example)
        prompt_ids = tokenizer(
            prompt,
            truncation=True,
            max_length=cfg["max_length"],
            padding=False,
        )["input_ids"]
        encoded = tokenizer(text, truncation=True, max_length=cfg["max_length"], padding="max_length")
        prompt_len = min(len(prompt_ids), len(encoded["input_ids"]))
        encoded["labels"] = [
            token_id if mask == 1 and i >= prompt_len else -100
            for i, (token_id, mask) in enumerate(zip(encoded["input_ids"], encoded["attention_mask"]))
        ]
        return encoded

    tokenized = dataset.map(tokenize, remove_columns=dataset.column_names)
    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        num_train_epochs=cfg["num_train_epochs"],
        logging_steps=cfg["logging_steps"],
        save_steps=cfg["save_steps"],
        bf16=torch.cuda.is_available(),
        report_to=["tensorboard"],
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized)
    trainer.train()
    trainer.save_model(cfg["output_dir"])


if __name__ == "__main__":
    main()
