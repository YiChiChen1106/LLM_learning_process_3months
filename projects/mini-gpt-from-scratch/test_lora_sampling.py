import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from lora_checkpoint import save_lora_adapter
from lora_minigpt import LoRALinear, replace_qkv_with_lora
from model import MiniGPT
from sample import load_sampling_model


class MiniGPTLoRASamplingTest(unittest.TestCase):
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

    def save_base_checkpoint(self, model: MiniGPT, path: Path) -> None:
        torch.save(
            {
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
            },
            path,
        )

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

    def test_load_sampling_model_without_adapter_keeps_plain_minigpt(self):
        base_model = self.make_base_model()
        input_ids = torch.randint(0, 50, (2, 8))

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "base.pt"
            self.save_base_checkpoint(base_model, checkpoint_path)
            loaded_model, stoi, itos = load_sampling_model(
                checkpoint_path,
                device="cpu",
            )

        self.assertIsInstance(loaded_model.blocks[0].attn.qkv, torch.nn.Linear)
        self.assertEqual(stoi["a"], 0)
        self.assertEqual(itos[1], "b")
        with torch.no_grad():
            self.assertTrue(torch.equal(base_model(input_ids), loaded_model(input_ids)))

    def test_load_sampling_model_with_adapter_matches_reference_lora_model(self):
        base_model = self.make_base_model()
        reference_model = self.make_base_model()
        reference_model.load_state_dict(base_model.state_dict())
        reference_model = replace_qkv_with_lora(reference_model, r=8, alpha=16)
        self.train_lora_one_step(reference_model)
        input_ids = torch.randint(0, 50, (2, 8))

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "base.pt"
            adapter_path = Path(tmpdir) / "adapter.pt"
            self.save_base_checkpoint(base_model, checkpoint_path)
            save_lora_adapter(reference_model, adapter_path)

            loaded_model, _, _ = load_sampling_model(
                checkpoint_path,
                device="cpu",
                lora_adapter=adapter_path,
                lora_rank=8,
                lora_alpha=16,
            )

        self.assertIsInstance(loaded_model.blocks[0].attn.qkv, LoRALinear)
        with torch.no_grad():
            self.assertTrue(torch.equal(reference_model(input_ids), loaded_model(input_ids)))

    def test_loaded_adapter_weights_can_be_recovered_from_sampling_model(self):
        source_model = replace_qkv_with_lora(self.make_base_model(), r=8, alpha=16)
        self.train_lora_one_step(source_model)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "base.pt"
            adapter_path = Path(tmpdir) / "adapter.pt"
            self.save_base_checkpoint(self.make_base_model(), checkpoint_path)
            saved_state = save_lora_adapter(source_model, adapter_path)

            loaded_model, _, _ = load_sampling_model(
                checkpoint_path,
                device="cpu",
                lora_adapter=adapter_path,
                lora_rank=8,
                lora_alpha=16,
            )

        loaded_state = loaded_model.state_dict()
        for name, tensor in saved_state.items():
            self.assertTrue(torch.equal(loaded_state[name].cpu(), tensor), name)


if __name__ == "__main__":
    unittest.main()
