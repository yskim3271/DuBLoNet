import unittest

from src.v2_contract import (
    BalancedOperatingPointSchedule,
    LATENCY_FUTURE_FRAMES,
    SFIProfile,
    TARGET_SAMPLE_RATES,
    compute_lookahead_frames,
    decoder_padding_ratios_for_future_frames,
    exact_samples,
    validate_v2_grid,
)


class SFIProfileTest(unittest.TestCase):
    EXPECTED = {
        8000: (320, 160, 161, 161),
        16000: (640, 320, 321, 321),
        22050: (882, 441, 442, 443),
        24000: (960, 480, 481, 481),
        32000: (1280, 640, 641, 641),
        44100: (1764, 882, 883, 883),
        48000: (1920, 960, 961, 961),
    }

    def test_locked_stft_grid(self):
        for sample_rate, expected in self.EXPECTED.items():
            with self.subTest(sample_rate=sample_rate):
                profile = SFIProfile(sample_rate)
                actual = (
                    profile.fft_len,
                    profile.hop_len,
                    profile.frequency_bins,
                    profile.internal_frequency_bins,
                )
                self.assertEqual(actual, expected)
                self.assertEqual(profile.bin_hz, 25.0)

    def test_one_second_has_rate_independent_sequence_length(self):
        for sample_rate in TARGET_SAMPLE_RATES:
            profile = SFIProfile(sample_rate)
            self.assertEqual(profile.centered_frame_count(sample_rate), 51)

    def test_only_22050_needs_frequency_pad(self):
        padded_rates = {
            rate for rate in TARGET_SAMPLE_RATES if SFIProfile(rate).needs_even_frequency_pad
        }
        self.assertEqual(padded_rates, {22050})

    def test_fractional_sample_grid_is_rejected(self):
        with self.assertRaises(ValueError):
            exact_samples(22050, 25.0)


class LatencyContractTest(unittest.TestCase):
    def test_padding_implementation_matches_future_frame_contract(self):
        ratios = decoder_padding_ratios_for_future_frames(LATENCY_FUTURE_FRAMES)
        self.assertEqual(set(ratios), set(LATENCY_FUTURE_FRAMES))
        for latency_id, expected in LATENCY_FUTURE_FRAMES.items():
            with self.subTest(latency_id=latency_id):
                self.assertEqual(
                    compute_lookahead_frames(ratios[latency_id], depth=4),
                    expected,
                )

    def test_algorithmic_latency_grid(self):
        schedule = BalancedOperatingPointSchedule(seed=7)
        points = [schedule.next() for _ in range(schedule.cycle_size)]
        actual = {
            point.latency_id: point.algorithmic_latency_ms for point in points
        }
        self.assertEqual(
            actual,
            {"LA0": 20.0, "LA1": 40.0, "LA2": 60.0, "LA3": 80.0},
        )


class BalancedScheduleTest(unittest.TestCase):
    def test_each_cycle_contains_all_28_cells_once(self):
        schedule = BalancedOperatingPointSchedule(seed=2039)
        points = [schedule.next() for _ in range(schedule.cycle_size)]
        cells = {(point.sample_rate, point.latency_id) for point in points}
        self.assertEqual(len(points), 28)
        self.assertEqual(len(cells), 28)

    def test_schedule_resume_is_exact(self):
        original = BalancedOperatingPointSchedule(seed=2039)
        for _ in range(13):
            original.next()
        state = original.state_dict()
        expected = [original.next() for _ in range(40)]

        resumed = BalancedOperatingPointSchedule(seed=2039)
        resumed.load_state_dict(state)
        actual = [resumed.next() for _ in range(40)]
        self.assertEqual(actual, expected)

    def test_grid_constants_validate_without_tensor_runtime(self):
        validate_v2_grid()


if __name__ == "__main__":
    unittest.main()
