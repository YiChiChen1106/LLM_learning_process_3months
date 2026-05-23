import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from lora_checkpoint import save_lora_adapter, save_merged_checkpoint
from lora_minigpt import LoRALinear, merge_lora_linear, merge_qkv_lora, replace_qkv_with_lora
from model import MiniGPT
from sample import load_sampling_model


class MiniGPTLoRAMergeTest(unittest.TestCase):
    def make_base_model(self) -> MiniGPT:
        torch.manual_seed(0)
        return MiniGPT(
            vocab_size=50,
            block_size=16,
            dim=128,
            num_heads=4,
            num_layers=4,
            dropout=0.0,
        )

    def make_base_checkpoint(self, model: MiniGPT) -> dict:
        return {
            "model": model.state_dict(),
            "stoi": {"a": 0, "b": 1},
            "itos": {0: "a", 1: "b"},
            "config": {
                "vocab_size": 50,
                "block_size": 16,
                "dim": 128,
                "num_heads": 4,
                "num_layers": 4,
                "dropout": 0.0,
            },
        }

    def save_base_checkpoint(self, model: MiniGPT, path: Path) -> dict:
        checkpoint = self.make_base_checkpoint(model)
        torch.save(checkpoint, path)
        return checkpoint

    def train_lora_one_step(self, model: MiniGPT) -> None:
        torch.manual_seed(1)
        input_ids = torch.randint(0, 50, (2, 8))
        targets = torch.randint(0, 50, (2, 8))
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=3e-4,
        )

        logits = model(input_ids)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    def test_merge_lora_linear_returns_plain_linear_with_equivalent_output(self):
        torch.manual_seed(0)
        base = nn.Linear(128, 384, bias=True)
        lora_layer = LoRALinear(base, r=8, alpha=16)
        nn.init.normal_(lora_layer.lora_B.weight, mean=0.0, std=0.02)
        x = torch.randn(4, 128)

        merged = merge_lora_linear(lora_layer)

        self.assertIsInstance(merged, nn.Linear)
        self.assertNotIsInstance(merged, LoRALinear)
        self.assertEqual(tuple(merged.weight.shape), (384, 128))
        with torch.no_grad():
            self.assertTrue(torch.allclose(lora_layer(x), merged(x), atol=1e-6))

    def test_merge_qkv_lora_replaces_qkv_layers_and_preserves_logits(self):
        base_model = self.make_base_model()
        lora_model = self.make_base_model()
        lora_model.load_state_dict(base_model.state_dict())
        lora_model = replace_qkv_with_lora(lora_model, r=8, alpha=16)
        self.train_lora_one_step(lora_model)
        input_ids = torch.randint(0, 50, (2, 8))

        lora_model.eval()
        with torch.no_grad():
            before_merge = lora_model(input_ids)

        merged_model = merge_qkv_lora(lora_model)

        for block in merged_model.blocks:
            self.assertIsInstance(block.attn.qkv, nn.Linear)
            self.assertNotIsInstance(block.attn.qkv, LoRALinear)
        self.assertTrue(
            all("lora_" not in name for name in merged_model.state_dict())
        )
        with torch.no_grad():
            after_merge = merged_model(input_ids)
        self.assertTrue(torch.allclose(before_merge, after_merge, atol=1e-6))

    def test_save_merged_checkpoint_writes_plain_minigpt_checkpoint(self):
        base_model = self.make_base_model()
        lora_model = self.make_base_model()
        lora_model.load_state_dict(base_model.state_dict())
        lora_model = replace_qkv_with_lora(lora_model, r=8, alpha=16)
        self.train_lora_one_step(lora_model)
        input_ids = torch.randint(0, 50, (2, 8))

        lora_model.eval()
        with torch.no_grad():
            expected_logits = lora_model(input_ids)
        merged_model = merge_qkv_lora(lora_model)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "merged.pt"
            source_checkpoint = self.make_base_checkpoint(base_model)
            saved_checkpoint = save_merged_checkpoint(
                merged_model,
                source_checkpoint,
                checkpoint_path,
            )
            loaded_checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )

        self.assertEqual(saved_checkpoint["config"], loaded_checkpoint["config"])
        self.assertEqual(saved_checkpoint["stoi"], loaded_checkpoint["stoi"])
        self.assertEqual(saved_checkpoint["itos"], loaded_checkpoint["itos"])
        self.assertTrue(
            all("lora_" not in name for name in loaded_checkpoint["model"])
        )

        loaded_plain_model = MiniGPT(**loaded_checkpoint["config"])
        loaded_plain_model.load_state_dict(loaded_checkpoint["model"])
        loaded_plain_model.eval()
        with torch.no_grad():
            loaded_logits = loaded_plain_model(input_ids)
        self.assertTrue(torch.allclose(expected_logits, loaded_logits, atol=1e-6))

    def test_load_sampling_model_can_merge_adapter_and_save_plain_checkpoint(self):
        base_model = self.make_base_model()
        reference_model = self.make_base_model()
        reference_model.load_state_dict(base_model.state_dict())
        reference_model = replace_qkv_with_lora(reference_model, r=8, alpha=16)
        self.train_lora_one_step(reference_model)
        input_ids = torch.randint(0, 50, (2, 8))

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "base.pt"
            adapter_path = Path(tmpdir) / "adapter.pt"
            merged_path = Path(tmpdir) / "merged.pt"
            self.save_base_checkpoint(base_model, checkpoint_path)
            save_lora_adapter(reference_model, adapter_path)

            loaded_model, _, _ = load_sampling_model(
                checkpoint_path,
                device="cpu",
                lora_adapter=adapter_path,
                lora_rank=8,
                lora_alpha=16,
                merge_lora=True,
                save_merged_checkpoint_path=merged_path,
            )
            merged_checkpoint = torch.load(
                merged_path,
                map_location="cpu",
                weights_only=True,
            )

        self.assertIsInstance(loaded_model.blocks[0].attn.qkv, nn.Linear)
        self.assertNotIsInstance(loaded_model.blocks[0].attn.qkv, LoRALinear)
        self.assertTrue(all("lora_" not in name for name in merged_checkpoint["model"]))
        reference_model.eval()
        loaded_model.eval()
        with torch.no_grad():
            self.assertTrue(
                torch.allclose(reference_model(input_ids), loaded_model(input_ids), atol=1e-6)
            )


if __name__ == "__main__":
    unittest.main()
