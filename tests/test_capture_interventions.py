import unittest
import torch
from torch import nn
from nanoinspect.capture import Capture, REQUIRED_METADATA, estimate_bytes
from nanoinspect.interventions import Intervention, contrastive_direction, random_direction, hook_count


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
        self.layers = nn.ModuleList([Block(), Block()])

    def forward(self, x):
        return self.layers[1](self.layers[0](x))


class CaptureInterventionTests(unittest.TestCase):
    def setUp(self):
        self.model = Model()
        self.x = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)

    def test_capture_is_read_only_and_copies(self):
        expected = self.model(self.x)
        metadata = {"run": "unit", "phase": "prefill", "token_ids": [7, 8, 9], "model_revision": "rev"}
        with Capture(self.model, [(0, "post_mixer_residual")], positions=[1], metadata=metadata) as capture:
            actual = self.model(self.x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(capture.records[0]["tensor"], (self.x * 2)[:, 1:2])
        self.x.zero_()
        self.assertGreater(capture.records[0]["tensor"].sum().item(), 0)
        record = capture.records[0]
        self.assertEqual(record["bytes"], 16)
        for key in REQUIRED_METADATA:
            self.assertIn(key, record)
        self.assertEqual((record["run"], record["phase"], record["model_revision"]), ("unit", "prefill", "rev"))
        self.assertEqual(record["token_ids"], [[8]])
        self.assertEqual(record["positions"], [1])
        self.assertEqual(record["shape"], [1, 1, 4])
        self.assertEqual(record["dtype"], "torch.float32")

    def test_decode_offset_all_positions_and_metadata_defaults(self):
        with Capture(self.model, [(1, "residual_input")], positions="all",
                     metadata={"phase": "decode", "token_ids": [1, 2, 3, 4, 5], "position_offset": 2}) as capture:
            self.model(self.x)
        record = capture.records[0]
        self.assertEqual(record["positions"], [2, 3, 4])
        self.assertEqual(record["token_ids"], [[3, 4, 5]])
        self.assertIsNone(record["run"])
        self.assertIn("model_revision", record)

    def test_budget_is_estimated_before_running(self):
        selections = [(0, "residual_output"), (1, "residual_output")]
        expected = {"batch_size": 1, "hidden_size": 4, "element_size": 4}
        self.assertEqual(estimate_bytes(selections, [0, 1], **expected), 64)
        self.assertEqual(estimate_bytes(selections, "all", sequence_length=3, **expected), 96)
        with self.assertRaises(MemoryError):
            Capture(self.model, selections, positions=[0, 1], max_bytes=63, expected=expected)
        self.assertEqual(hook_count(self.model), 0)

    def test_patch_only_selected_position_and_zero_steering(self):
        baseline = self.model(self.x)
        with Intervention(self.model, 0, "residual_output", [1], patch=torch.ones(1, 1, 4)) as patch:
            patched = self.model(self.x)
        self.assertEqual(patch.calls, 1)
        torch.testing.assert_close(patched[:, [0, 2]], baseline[:, [0, 2]], rtol=0, atol=0)
        torch.testing.assert_close(patched[:, 1:2], torch.ones(1, 1, 4) * 4)  # layer 1 doubles twice
        with Intervention(self.model, 0, "residual_output", [1], direction=torch.ones(4), alpha=0):
            steered = self.model(self.x)
        torch.testing.assert_close(steered, baseline, rtol=0, atol=0)
        with Intervention(self.model, 1, "mixer_output", "all", direction=torch.ones(4), alpha=0.5):
            shifted = self.model(self.x)
        torch.testing.assert_close(shifted, baseline + 1.0, rtol=0, atol=0)  # +0.5 then doubled by the MLP residual

    def test_cleanup_after_exceptions_and_budget(self):
        baseline = self.model(self.x)
        with self.assertRaises(MemoryError):
            with Capture(self.model, [(0, "residual_output")], positions=[0], max_bytes=1):
                self.model(self.x)
        with self.assertRaisesRegex(RuntimeError, "stop"):
            with Intervention(self.model, 0, "residual_output", [0], direction=torch.ones(4), alpha=1):
                raise RuntimeError("stop")
        torch.testing.assert_close(self.model(self.x), baseline, rtol=0, atol=0)
        self.assertEqual(hook_count(self.model), 0)

    def test_patch_shape_and_direction_validation(self):
        with self.assertRaises(ValueError):
            with Intervention(self.model, 0, "residual_output", [0], patch=torch.ones(1, 2, 4)):
                self.model(self.x)
        with self.assertRaises(ValueError):
            Intervention(self.model, 0, "residual_output", [0], patch=torch.ones(1, 1, 4), direction=torch.ones(4))
        self.assertEqual(hook_count(self.model), 0)

    def test_post_mixer_residual_patch_reaches_the_residual_path(self):
        # Block: y = x + attn(x); out = y + mlp(norm(y)), all identities -> out = 2y.
        baseline = self.model(self.x)
        with Intervention(self.model, 0, "post_mixer_residual", [1], patch=torch.ones(1, 1, 4)) as patch:
            patched = self.model(self.x)
        with Intervention(self.model, 0, "residual_output", [1], patch=torch.full((1, 1, 4), 2.0)):
            equivalent = self.model(self.x)
        self.assertEqual(patch.calls, 1)
        torch.testing.assert_close(patched, equivalent, rtol=0, atol=0)  # residual and MLP both see the patch
        torch.testing.assert_close(patched[:, [0, 2]], baseline[:, [0, 2]], rtol=0, atol=0)
        with Intervention(self.model, 0, "post_mixer_residual", "all", direction=torch.ones(4), alpha=0.0):
            torch.testing.assert_close(self.model(self.x), baseline, rtol=0, atol=0)
        with Intervention(self.model, 0, "post_mixer_residual", [0], direction=torch.ones(4), alpha=1.0):
            shifted = self.model(self.x)
        torch.testing.assert_close(shifted[:, 0], baseline[:, 0] + 8.0, rtol=0, atol=0)  # +1, x2 by block 0 MLP, x4 by block 1
        self.assertEqual(hook_count(self.model), 0)

    def test_per_row_positions(self):
        x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        baseline = self.model(x)
        with Capture(self.model, [(0, "residual_input")], positions=[[0], [2]],
                     metadata={"token_ids": [[1, 2, 3], [4, 5, 6]]}) as capture:
            self.model(x)
        record = capture.records[0]
        torch.testing.assert_close(record["tensor"], torch.stack([x[0, [0]], x[1, [2]]]))
        self.assertEqual(record["positions"], [[0], [2]])
        self.assertEqual(record["token_ids"], [[1], [6]])
        with Intervention(self.model, 0, "residual_input", [[0], [2]], patch=torch.zeros(2, 1, 4)):
            patched = self.model(x)
        changed = (patched != baseline).any(-1)
        self.assertEqual(changed.tolist(), [[True, False, False], [False, False, True]])
        with self.assertRaises(ValueError):
            with Intervention(self.model, 0, "residual_input", [[0], [1, 2]], patch=torch.zeros(2, 1, 4)):
                self.model(x)

    def test_contrastive_and_random_directions(self):
        direction, magnitude = contrastive_direction(torch.tensor([[2., 0.], [4., 0.]]), torch.zeros(2, 2))
        torch.testing.assert_close(direction, torch.tensor([1., 0.]))
        self.assertEqual(magnitude, 3.)
        a, b = random_direction(16, 1, norm=2.5), random_direction(16, 1, norm=2.5)
        torch.testing.assert_close(a, b)
        self.assertAlmostEqual(a.norm().item(), 2.5, places=5)
        self.assertFalse(torch.equal(a, random_direction(16, 2, norm=2.5)))


if __name__ == "__main__":
    unittest.main()
