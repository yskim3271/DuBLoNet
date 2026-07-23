import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class StaticModelWiringTest(unittest.TestCase):
    def test_fixed_mask_gate_declares_no_parameter(self):
        path = ROOT / "src" / "models" / "backbone.py"
        module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        gate = next(
            node
            for node in module.body
            if isinstance(node, ast.ClassDef) and node.name == "FixedMaskGate"
        )
        parameter_calls = [
            node
            for node in ast.walk(gate)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "Parameter"
        ]
        self.assertEqual(parameter_calls, [])

    def test_backbones_wire_even_frequency_pad_and_output_crop(self):
        for relative_path in (
            "src/models/backbone.py",
            "src/models/latency_expert.py",
        ):
            with self.subTest(path=relative_path):
                source = (ROOT / relative_path).read_text(encoding="utf-8")
                self.assertIn("original_frequency_bins", source)
                self.assertIn("pad_even_frequency", source)
                self.assertIn(":original_frequency_bins", source.replace(" ", ""))

    def test_solver_persists_balanced_schedule_cursor(self):
        source = (ROOT / "src" / "solver.py").read_text(encoding="utf-8")
        self.assertIn("BalancedOperatingPointSchedule", source)
        self.assertIn("package['operating_schedule']", source)
        self.assertIn("load_state_dict(schedule_state)", source)


class StaticConfigWiringTest(unittest.TestCase):
    def test_bn_config_declares_locked_grid(self):
        source = (
            ROOT / "conf" / "experiments" / "lacosenet_v2_sfi_bn.yaml"
        ).read_text(encoding="utf-8")
        required = (
            "target_sample_rates: [8000, 16000, 22050, 24000, 32000, 44100, 48000]",
            "window_ms: 40.0",
            "hop_ms: 20.0",
            "latency_ids: [LA0, LA1, LA2, LA3]",
            "latency_sampling: balanced_grid",
            "fixed_mask_gate: true",
            "pad_even_frequency: true",
            "generator_only: true",
        )
        for item in required:
            with self.subTest(item=item):
                self.assertIn(item, source)

    def test_channel_ln_config_changes_only_normalization_candidate(self):
        source = (
            ROOT / "conf" / "experiments" / "lacosenet_v2_sfi_channel_ln.yaml"
        ).read_text(encoding="utf-8")
        self.assertIn("- lacosenet_v2_sfi_bn", source)
        self.assertIn("norm_type: channel_layer", source)
        self.assertIn("expert_scope: dsddb_only", source)


if __name__ == "__main__":
    unittest.main()
