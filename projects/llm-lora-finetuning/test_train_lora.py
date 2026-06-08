from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path


def load_train_lora_module():
    module_path = Path(__file__).with_name("train_lora.py")
    spec = importlib.util.spec_from_file_location("llm_lora_train_lora", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeDataset:
    def __init__(self, rows):
        self.rows = list(rows)
        self.column_names = list(self.rows[0].keys()) if self.rows else []

    @classmethod
    def from_list(cls, rows):
        return cls(rows)

    def map(self, fn, remove_columns=None):
        mapped_rows = []
        for row in self.rows:
            encoded = fn(row)
            if remove_columns:
                for column in remove_columns:
                    encoded.pop(column, None)
            mapped_rows.append(encoded)
        return FakeDataset(mapped_rows)

    def __getitem__(self, index):
        return self.rows[index]


def test_tokenize_masks_prompt_only_and_padding_labels(tmp_path, monkeypatch):
    captured = {}

    fake_torch = types.SimpleNamespace(
        bfloat16="bfloat16",
        float32="float32",
        cuda=types.SimpleNamespace(is_available=lambda: False),
    )
    fake_datasets = types.SimpleNamespace(Dataset=FakeDataset)
    fake_peft = types.SimpleNamespace(
        LoraConfig=lambda **kwargs: kwargs,
        get_peft_model=lambda model, config: model,
        prepare_model_for_kbit_training=lambda model: model,
    )

    class FakeTokenizer:
        pad_token = None
        eos_token = "<eos>"

        def __call__(self, text, truncation, max_length, padding):
            return {
                "input_ids": [101, 200, 0, 0],
                "attention_mask": [1, 1, 0, 0],
            }

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return FakeTokenizer()

    class FakeModel:
        def print_trainable_parameters(self):
            pass

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return FakeModel()

    class FakeTrainer:
        def __init__(self, model, args, train_dataset, data_collator=None):
            captured["train_dataset"] = train_dataset

        def train(self):
            pass

        def save_model(self, output_dir):
            pass

    class FakeTrainingArguments:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_transformers = types.SimpleNamespace(
        AutoModelForCausalLM=FakeAutoModel,
        AutoTokenizer=FakeAutoTokenizer,
        BitsAndBytesConfig=lambda **kwargs: kwargs,
        Trainer=FakeTrainer,
        TrainingArguments=FakeTrainingArguments,
    )
    fake_yaml = types.SimpleNamespace(
        safe_load=lambda stream: json.loads(stream.read()),
    )

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
    monkeypatch.setitem(sys.modules, "peft", fake_peft)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "yaml", fake_yaml)

    train_lora = load_train_lora_module()

    monkeypatch.setattr(train_lora, "AutoTokenizer", FakeAutoTokenizer)
    monkeypatch.setattr(train_lora, "AutoModelForCausalLM", FakeAutoModel)
    monkeypatch.setattr(train_lora, "Trainer", FakeTrainer)
    monkeypatch.setattr(train_lora, "TrainingArguments", FakeTrainingArguments)
    monkeypatch.setattr(train_lora, "LoraConfig", lambda **kwargs: kwargs)
    monkeypatch.setattr(train_lora, "get_peft_model", lambda model, config: model)

    train_file = tmp_path / "train.jsonl"
    train_file.write_text(json.dumps({"instruction": "hi", "response": "hello"}) + "\n", encoding="utf-8")
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        json.dumps(
            {
                "model_name": "fake-model",
                "load_in_4bit": False,
                "lora_rank": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "target_modules": ["q_proj"],
                "train_file": str(train_file),
                "max_length": 4,
                "output_dir": str(tmp_path / "out"),
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "learning_rate": 1e-4,
                "num_train_epochs": 1,
                "logging_steps": 1,
                "save_steps": 1,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(sys, "argv", ["train_lora.py", "--config", str(config_file)])

    train_lora.main()

    row = captured["train_dataset"][0]
    assert row["input_ids"] == [101, 200, 0, 0]
    assert row["attention_mask"] == [1, 1, 0, 0]
    assert row["labels"] == [-100, -100, -100, -100]


def test_tokenize_masks_prompt_and_padding_labels(tmp_path, monkeypatch):
    captured = {}

    fake_torch = types.SimpleNamespace(
        bfloat16="bfloat16",
        float32="float32",
        cuda=types.SimpleNamespace(is_available=lambda: False),
    )
    fake_datasets = types.SimpleNamespace(Dataset=FakeDataset)
    fake_peft = types.SimpleNamespace(
        LoraConfig=lambda **kwargs: kwargs,
        get_peft_model=lambda model, config: model,
        prepare_model_for_kbit_training=lambda model: model,
    )

    class FakeTokenizer:
        pad_token = None
        eos_token = "<eos>"
        padding_side = "right"

        def __call__(self, text, truncation, max_length, padding):
            if "hello" not in text:
                return {"input_ids": [10, 11, 12], "attention_mask": [1, 1, 1]}
            if padding == "max_length":
                return {
                    "input_ids": [10, 11, 12, 20, 21, 0],
                    "attention_mask": [1, 1, 1, 1, 1, 0],
                }
            return {
                "input_ids": [10, 11, 12, 20, 21],
                "attention_mask": [1, 1, 1, 1, 1],
            }

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return FakeTokenizer()

    class FakeModel:
        def print_trainable_parameters(self):
            pass

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return FakeModel()

    class FakeTrainer:
        def __init__(self, model, args, train_dataset, data_collator=None):
            captured["train_dataset"] = train_dataset

        def train(self):
            pass

        def save_model(self, output_dir):
            pass

    class FakeTrainingArguments:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_transformers = types.SimpleNamespace(
        AutoModelForCausalLM=FakeAutoModel,
        AutoTokenizer=FakeAutoTokenizer,
        BitsAndBytesConfig=lambda **kwargs: kwargs,
        Trainer=FakeTrainer,
        TrainingArguments=FakeTrainingArguments,
    )
    fake_yaml = types.SimpleNamespace(
        safe_load=lambda stream: json.loads(stream.read()),
    )

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
    monkeypatch.setitem(sys.modules, "peft", fake_peft)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "yaml", fake_yaml)

    train_lora = load_train_lora_module()

    monkeypatch.setattr(train_lora, "AutoTokenizer", FakeAutoTokenizer)
    monkeypatch.setattr(train_lora, "AutoModelForCausalLM", FakeAutoModel)
    monkeypatch.setattr(train_lora, "Trainer", FakeTrainer)
    monkeypatch.setattr(train_lora, "TrainingArguments", FakeTrainingArguments)
    monkeypatch.setattr(train_lora, "LoraConfig", lambda **kwargs: kwargs)
    monkeypatch.setattr(train_lora, "get_peft_model", lambda model, config: model)

    train_file = tmp_path / "train.jsonl"
    train_file.write_text(json.dumps({"instruction": "hi", "response": "hello"}) + "\n", encoding="utf-8")
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        json.dumps(
            {
                "model_name": "fake-model",
                "load_in_4bit": False,
                "lora_rank": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "target_modules": ["q_proj"],
                "train_file": str(train_file),
                "max_length": 6,
                "output_dir": str(tmp_path / "out"),
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "learning_rate": 1e-4,
                "num_train_epochs": 1,
                "logging_steps": 1,
                "save_steps": 1,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(sys, "argv", ["train_lora.py", "--config", str(config_file)])

    train_lora.main()

    row = captured["train_dataset"][0]
    assert row["input_ids"] == [10, 11, 12, 20, 21]
    assert row["attention_mask"] == [1, 1, 1, 1, 1]
    assert row["labels"] == [-100, -100, -100, 20, 21]


def test_trainer_uses_dynamic_padding_collator(tmp_path, monkeypatch):
    captured = {}

    class FakeTensor:
        def __init__(self, data):
            self.data = data
            self.shape = (len(data), len(data[0]) if data else 0)

        def tolist(self):
            return self.data

    fake_torch = types.SimpleNamespace(
        bfloat16="bfloat16",
        float32="float32",
        long="long",
        cuda=types.SimpleNamespace(is_available=lambda: False),
        tensor=lambda value, dtype=None: FakeTensor(value),
    )
    fake_datasets = types.SimpleNamespace(Dataset=FakeDataset)
    fake_peft = types.SimpleNamespace(
        LoraConfig=lambda **kwargs: kwargs,
        get_peft_model=lambda model, config: model,
        prepare_model_for_kbit_training=lambda model: model,
    )

    class FakeTokenizer:
        pad_token = None
        eos_token = "<eos>"
        padding_side = "right"
        pad_token_id = 0

        def __call__(self, text, truncation, max_length, padding):
            if "hello" not in text:
                return {"input_ids": [10, 11, 12], "attention_mask": [1, 1, 1]}
            if padding == "max_length":
                return {
                    "input_ids": [10, 11, 12, 20, 21, 0],
                    "attention_mask": [1, 1, 1, 1, 1, 0],
                }
            return {
                "input_ids": [10, 11, 12, 20, 21],
                "attention_mask": [1, 1, 1, 1, 1],
            }

        def pad(self, features, padding, pad_to_multiple_of=None, return_tensors=None):
            max_len = max(len(feature["input_ids"]) for feature in features)
            if pad_to_multiple_of:
                max_len = ((max_len + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of

            batch = {"input_ids": [], "attention_mask": []}
            for feature in features:
                pad_len = max_len - len(feature["input_ids"])
                batch["input_ids"].append(feature["input_ids"] + [self.pad_token_id] * pad_len)
                batch["attention_mask"].append(feature["attention_mask"] + [0] * pad_len)
            return {key: FakeTensor(value) for key, value in batch.items()}

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return FakeTokenizer()

    class FakeModel:
        def print_trainable_parameters(self):
            pass

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return FakeModel()

    class FakeTrainer:
        def __init__(self, model, args, train_dataset, data_collator=None):
            captured["train_dataset"] = train_dataset
            captured["data_collator"] = data_collator

        def train(self):
            pass

        def save_model(self, output_dir):
            pass

    class FakeTrainingArguments:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_transformers = types.SimpleNamespace(
        AutoModelForCausalLM=FakeAutoModel,
        AutoTokenizer=FakeAutoTokenizer,
        BitsAndBytesConfig=lambda **kwargs: kwargs,
        Trainer=FakeTrainer,
        TrainingArguments=FakeTrainingArguments,
    )
    fake_yaml = types.SimpleNamespace(
        safe_load=lambda stream: json.loads(stream.read()),
    )

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
    monkeypatch.setitem(sys.modules, "peft", fake_peft)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "yaml", fake_yaml)

    train_lora = load_train_lora_module()

    monkeypatch.setattr(train_lora, "AutoTokenizer", FakeAutoTokenizer)
    monkeypatch.setattr(train_lora, "AutoModelForCausalLM", FakeAutoModel)
    monkeypatch.setattr(train_lora, "Trainer", FakeTrainer)
    monkeypatch.setattr(train_lora, "TrainingArguments", FakeTrainingArguments)
    monkeypatch.setattr(train_lora, "LoraConfig", lambda **kwargs: kwargs)
    monkeypatch.setattr(train_lora, "get_peft_model", lambda model, config: model)

    train_file = tmp_path / "train.jsonl"
    train_file.write_text(json.dumps({"instruction": "hi", "response": "hello"}) + "\n", encoding="utf-8")
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        json.dumps(
            {
                "model_name": "fake-model",
                "load_in_4bit": False,
                "lora_rank": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "target_modules": ["q_proj"],
                "train_file": str(train_file),
                "max_length": 6,
                "output_dir": str(tmp_path / "out"),
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "learning_rate": 1e-4,
                "num_train_epochs": 1,
                "logging_steps": 1,
                "save_steps": 1,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(sys, "argv", ["train_lora.py", "--config", str(config_file)])

    train_lora.main()

    row = captured["train_dataset"][0]
    assert row["input_ids"] == [10, 11, 12, 20, 21]
    assert row["attention_mask"] == [1, 1, 1, 1, 1]
    assert row["labels"] == [-100, -100, -100, 20, 21]

    collator = captured["data_collator"]
    assert collator is not None

    batch = collator(
        [
            row,
            {
                "input_ids": [30, 31],
                "attention_mask": [1, 1],
                "labels": [-100, 40],
            },
        ]
    )
    assert batch["input_ids"].tolist() == [[10, 11, 12, 20, 21], [30, 31, 0, 0, 0]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 1, 1], [1, 1, 0, 0, 0]]
    assert batch["labels"].tolist() == [[-100, -100, -100, 20, 21], [-100, 40, -100, -100, -100]]
