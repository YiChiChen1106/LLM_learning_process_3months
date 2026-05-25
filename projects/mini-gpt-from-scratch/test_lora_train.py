from pathlib import Path
import tempfile
import unittest

import torch

from lora_checkpoint import count_lora_parameters
from lora_minigpt import LoRALinear
from model import MiniGPT
from train_lora import (
    build_lora_model_from_checkpoint,
    make_lora_optimizer,
    train_lora_adapter,
)


def load_torch_file(path: Path) -> dict[str, torch.Tensor]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


class TrainLoRAAdapterTest(unittest.TestCase):
    def make_checkpoint(self, path: Path) -> dict[str, object]:
        torch.manual_seed(0)
        model = MiniGPT(
            vocab_size=50,
            block_size=16,
            dim=128,
            num_heads=4,
            num_layers=4,
            dropout=0.0,
        )
        checkpoint = {
            "model": model.state_dict(),
            "stoi": {str(i): i for i in range(50)},
            "itos": {i: str(i) for i in range(50)},
            "config": {
                "vocab_size": 50,
                "block_size": 16,
                "dim": 128,
                "num_heads": 4,
                "num_layers": 4,
                "dropout": 0.0,
            },
        }
        torch.save(checkpoint, path)
        return checkpoint

    def test_build_lora_model_loads_base_weights_before_replacing_qkv(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "base.pt"
            checkpoint = self.make_checkpoint(checkpoint_path)

            model, _ = build_lora_model_from_checkpoint(
                checkpoint_path,
                r=8,
                alpha=16,
                device="cpu",
            )

            qkv = model.blocks[0].attn.qkv
            self.assertIsInstance(qkv, LoRALinear)
            self.assertTrue(
                torch.equal(
                    qkv.linear.weight,
                    checkpoint["model"]["blocks.0.attn.qkv.weight"],
                )
            )
            self.assertTrue(
                torch.equal(
                    qkv.linear.bias,
                    checkpoint["model"]["blocks.0.attn.qkv.bias"],
                )
            )

    def test_make_lora_optimizer_uses_only_trainable_lora_parameters(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "base.pt"
            self.make_checkpoint(checkpoint_path)
            model, _ = build_lora_model_from_checkpoint(checkpoint_path)

            optimizer = make_lora_optimizer(model, learning_rate=3e-4)

            optimizer_param_count = sum(
                parameter.numel()
                for group in optimizer.param_groups
                for parameter in group["params"]
            )
            trainable_param_count = sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            )
            trainable_names = [
                name
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            ]

            self.assertEqual(optimizer_param_count, trainable_param_count)
            self.assertEqual(trainable_param_count, 16_384)
            self.assertTrue(all("lora_" in name for name in trainable_names))

    def test_train_lora_adapter_saves_only_lora_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "base.pt"
            adapter_path = Path(tmp) / "adapter.pt"
            self.make_checkpoint(checkpoint_path)
            train_data = torch.arange(80, dtype=torch.long) % 50
            val_data = torch.arange(80, dtype=torch.long).flip(0) % 50

            result = train_lora_adapter(
                base_checkpoint=checkpoint_path,
                adapter_output=adapter_path,
                train_data=train_data,
                val_data=val_data,
                max_steps=1,
                eval_interval=1,
                patience=1,
                learning_rate=3e-4,
                batch_size=2,
                eval_iters=1,
                r=8,
                alpha=16,
                seed=0,
                device="cpu",
            )

            adapter_state = load_torch_file(adapter_path)

            self.assertEqual(result["trainable_params"], 16_384)
            self.assertEqual(count_lora_parameters(adapter_state), 16_384)
            self.assertTrue(adapter_state)
            self.assertTrue(all("lora_" in name for name in adapter_state))


if __name__ == "__main__":
    unittest.main()
