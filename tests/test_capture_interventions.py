import unittest
import torch
from torch import nn
from nanoinspect.capture import Capture
from nanoinspect.interventions import Intervention, contrastive_direction


class Block(nn.Module):
    layer_type = "full_attention"
    def __init__(self):
        super().__init__()
        self.input_layernorm = nn.Identity()
        self.self_attn = nn.Identity()
        self.post_attention_layernorm = nn.Identity()
        self.mlp = nn.Identity()

    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        return x + self.mlp(self.post_attention_layernorm(x))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([Block()])

    def forward(self, x):
        return self.layers[0](x)


class CaptureInterventionTests(unittest.TestCase):
    def setUp(self):
        self.model = Model()
        self.x = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)

    def test_capture_is_read_only_and_copies(self):
        expected = self.model(self.x)
        with Capture(self.model, [(0, "post_mixer_residual")], positions=[1], metadata={"phase": "prefill"}) as capture:
            actual = self.model(self.x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(capture.records[0]["tensor"], (self.x * 2)[:, 1:2])
        self.x.zero_()
        self.assertGreater(capture.records[0]["tensor"].sum().item(), 0)
        self.assertEqual(capture.records[0]["estimated_bytes"], 16)

    def test_patch_only_selected_position_and_zero_steering(self):
        baseline = self.model(self.x)
        with Intervention(self.model, 0, "residual_output", [1], patch=torch.ones(1, 1, 4)):
            patched = self.model(self.x)
        torch.testing.assert_close(patched[:, [0, 2]], baseline[:, [0, 2]], rtol=0, atol=0)
        torch.testing.assert_close(patched[:, 1:2], torch.ones(1, 1, 4))
        with Intervention(self.model, 0, "residual_output", [1], direction=torch.ones(4), alpha=0):
            steered = self.model(self.x)
        torch.testing.assert_close(steered, baseline, rtol=0, atol=0)

    def test_cleanup_after_exceptions_and_budget(self):
        baseline = self.model(self.x)
        with self.assertRaises(MemoryError):
            with Capture(self.model, [(0, "residual_output")], positions=[0], max_bytes=1):
                self.model(self.x)
        with self.assertRaisesRegex(RuntimeError, "stop"):
            with Intervention(self.model, 0, "residual_output", [0], direction=torch.ones(4), alpha=1):
                raise RuntimeError("stop")
        torch.testing.assert_close(self.model(self.x), baseline, rtol=0, atol=0)
        self.assertFalse(self.model.layers[0]._forward_hooks)

    def test_contrastive_direction(self):
        direction, magnitude = contrastive_direction(torch.tensor([[2., 0.], [4., 0.]]), torch.zeros(2, 2))
        torch.testing.assert_close(direction, torch.tensor([1., 0.]))
        self.assertEqual(magnitude, 3.)


if __name__ == "__main__":
    unittest.main()
