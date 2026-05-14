import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from lora_checkpoint import (
    count_lora_parameters,
    get_lora_state_dict,
    load_lora_adapter,
    save_lora_adapter,
)
from lora_minigpt import replace_qkv_with_lora
from model import MiniGPT


class MiniGPTLoRACheckpointTest(unittest.TestCase):
    def make_lora_model(self) -> MiniGPT:
        torch.manual_seed(0)
        model = MiniGPT(
            vocab_size=50,
            block_size=16,
            dim=128,
            num_heads=4,
            num_layers=4,
            dropout=0.0,
        )
        return replace_qkv_with_lora(model, r=8, alpha=16)

    def train_one_step(self, model: MiniGPT) -> None:
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

    def test_get_lora_state_dict_keeps_only_lora_adapter_weights(self):
        model = self.make_lora_model()
        lora_state = get_lora_state_dict(model)

        self.assertEqual(len(lora_state), 8)
        self.assertTrue(all("lora_" in name for name in lora_state))
        self.assertNotIn("blocks.0.attn.qkv.linear.weight", lora_state)
        self.assertEqual(count_lora_parameters(lora_state), 16_384)

    def test_save_lora_adapter_writes_only_adapter_checkpoint(self):
        model = self.make_lora_model()
        self.train_one_step(model)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "adapter.pt"
            saved_state = save_lora_adapter(model, path)
            loaded_state = torch.load(path, map_location="cpu", weights_only=True)

        self.assertEqual(set(saved_state), set(loaded_state))
        self.assertEqual(count_lora_parameters(loaded_state), 16_384)
        self.assertTrue(all("lora_" in name for name in loaded_state))

    def test_load_lora_adapter_restores_saved_adapter_weights(self):
        source_model = self.make_lora_model()
        self.train_one_step(source_model)

        target_model = self.make_lora_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "adapter.pt"
            saved_state = save_lora_adapter(source_model, path)
            load_lora_adapter(target_model, path)

        target_state = get_lora_state_dict(target_model)
        for name, tensor in saved_state.items():
            self.assertTrue(torch.equal(tensor, target_state[name]), name)

    def test_same_base_model_and_same_adapter_produce_same_logits(self):
        source_model = self.make_lora_model()
        self.train_one_step(source_model)

        target_model = self.make_lora_model()
        input_ids = torch.randint(0, 50, (2, 8))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "adapter.pt"
            save_lora_adapter(source_model, path)
            load_lora_adapter(target_model, path)

        source_model.eval()
        target_model.eval()
        with torch.no_grad():
            source_logits = source_model(input_ids)
            target_logits = target_model(input_ids)

        self.assertTrue(torch.equal(source_logits, target_logits))


if __name__ == "__main__":
    unittest.main()
