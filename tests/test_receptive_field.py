import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from receptive_field import compute_receptive_field, rf_to_segment

PUBLIC_RECIPE = {
    "dense_depth": 4,
    "num_tsblock": 4,
    "time_block_num": 2,
    "time_dw_kernel_size": 3,
    "sca_kernel_size": 11,
    "time_block_kernel": [3, 5, 7, 11],
    "causal_ts_block": True,
    "hop_len": 100,
    "win_len": 400,
}


class CausalSCAReceptiveFieldTest(unittest.TestCase):
    def test_public_recipe_includes_causal_sca(self):
        result = compute_receptive_field(PUBLIC_RECIPE)

        self.assertEqual(result.components[0].rf_frames, 31)
        self.assertEqual(result.components[1].rf_frames, 177)
        self.assertEqual(result.components[2].rf_frames, 31)
        self.assertEqual(result.total_rf_frames, 237)
        self.assertEqual(result.total_rf_samples, 24000)
        self.assertEqual(result.total_rf_ms, 1500.0)
        self.assertFalse(result.global_time_dependency)

    def test_sca_adds_one_kernel_increment_per_time_cab(self):
        without_sca_span = dict(PUBLIC_RECIPE, sca_kernel_size=1)

        baseline = compute_receptive_field(without_sca_span)
        corrected = compute_receptive_field(PUBLIC_RECIPE)
        time_cab_count = (
            PUBLIC_RECIPE["num_tsblock"] * PUBLIC_RECIPE["time_block_num"]
        )

        self.assertEqual(baseline.total_rf_frames, 157)
        self.assertEqual(
            corrected.total_rf_frames - baseline.total_rf_frames,
            time_cab_count * (PUBLIC_RECIPE["sca_kernel_size"] - 1),
        )

    def test_rf_sized_segment_uses_corrected_finite_span(self):
        self.assertEqual(rf_to_segment(PUBLIC_RECIPE), 24000)


class NonCausalSCAReceptiveFieldTest(unittest.TestCase):
    def test_noncausal_time_sca_is_reported_as_global(self):
        result = compute_receptive_field(
            dict(PUBLIC_RECIPE, causal_ts_block=False)
        )

        self.assertTrue(result.global_time_dependency)
        self.assertIn(
            "Actual temporal dependency: full input sequence",
            result.summary(),
        )
        self.assertIn(
            "Frequency-stage pooling is global over frequency, not time",
            result.summary(),
        )

    def test_noncausal_global_dependency_has_no_finite_rf_segment(self):
        with self.assertRaisesRegex(ValueError, "full input sequence"):
            rf_to_segment(dict(PUBLIC_RECIPE, causal_ts_block=False))


class ReleasedRecipeCompatibilityTest(unittest.TestCase):
    def test_released_training_crop_remains_one_second(self):
        config = (ROOT / "conf" / "config.yaml").read_text(encoding="utf-8")

        self.assertIn("segment: 16000", config)
        self.assertNotIn("segment: auto", config)


if __name__ == "__main__":
    unittest.main()
