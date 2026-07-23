"""Pure-Python contracts for the LaCoSENet v2 operating grid.

This module intentionally has no torch, numpy, or Hydra dependency.  It is
therefore safe to validate on development machines that do not have the model
runtime installed.  Tensor-level correctness is covered separately by the
runtime validation manifest in ``tests/RUNTIME_VALIDATION.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


TARGET_SAMPLE_RATES: Tuple[int, ...] = (
    8_000,
    16_000,
    22_050,
    24_000,
    32_000,
    44_100,
    48_000,
)

LATENCY_FUTURE_FRAMES: Dict[str, int] = {
    "LA0": 0,
    "LA1": 1,
    "LA2": 2,
    "LA3": 3,
}

# Representative ratios for the current depth-4 DS_DDB implementation.  The
# public contract is the integer future-frame count above; these ratios are an
# implementation detail and are always validated before use.
DECODER_PADDING_RATIOS: Dict[str, Tuple[float, float]] = {
    "LA0": (1.00, 0.00),
    "LA1": (0.96, 0.04),
    "LA2": (0.92, 0.08),
    "LA3": (0.90, 0.10),
}


def _config_get(config, key, default=None):
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)
    getter = getattr(config, "get", None)
    if getter is not None:
        return getter(key, default)
    return getattr(config, key, default)


def exact_samples(sample_rate: int, duration_ms: float) -> int:
    """Convert a physical duration to samples and reject fractional grids."""
    value = int(sample_rate) * float(duration_ms) / 1000.0
    rounded = round(value)
    if abs(value - rounded) > 1e-9:
        raise ValueError(
            f"{duration_ms} ms is not an integer sample count at {sample_rate} Hz: {value}"
        )
    return int(rounded)


@dataclass(frozen=True)
class SFIProfile:
    sample_rate: int
    window_ms: float = 40.0
    hop_ms: float = 20.0

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.window_ms <= 0 or self.hop_ms <= 0:
            raise ValueError("window_ms and hop_ms must be positive")
        if self.hop_ms > self.window_ms:
            raise ValueError("hop_ms cannot exceed window_ms")
        # Fail during construction rather than much later in torch.stft.
        _ = self.win_len
        _ = self.hop_len

    @property
    def win_len(self) -> int:
        return exact_samples(self.sample_rate, self.window_ms)

    @property
    def hop_len(self) -> int:
        return exact_samples(self.sample_rate, self.hop_ms)

    @property
    def fft_len(self) -> int:
        return self.win_len

    @property
    def frequency_bins(self) -> int:
        return self.fft_len // 2 + 1

    @property
    def internal_frequency_bins(self) -> int:
        bins = self.frequency_bins
        return bins if bins % 2 == 1 else bins + 1

    @property
    def needs_even_frequency_pad(self) -> bool:
        return self.frequency_bins % 2 == 0

    @property
    def bin_hz(self) -> float:
        return self.sample_rate / self.fft_len

    def centered_frame_count(self, num_samples: int) -> int:
        """Frame count for torch.stft(center=True) with the v2 frontend."""
        if num_samples < 0:
            raise ValueError("num_samples cannot be negative")
        return num_samples // self.hop_len + 1

    def as_stft_args(self, compress_factor: float = 1.0) -> Dict[str, float | int]:
        return {
            "n_fft": self.fft_len,
            "hop_size": self.hop_len,
            "win_size": self.win_len,
            "compress_factor": float(compress_factor),
        }


def sfi_profile_from_config(config, sample_rate: int) -> SFIProfile:
    return SFIProfile(
        sample_rate=int(sample_rate),
        window_ms=float(_config_get(config, "window_ms", 40.0)),
        hop_ms=float(_config_get(config, "hop_ms", 20.0)),
    )


def compute_lookahead_frames(
    padding_ratio: Sequence[float],
    depth: int = 4,
) -> int:
    """Mirror AsymmetricConv2d's right-padding rounding without torch."""
    if len(padding_ratio) != 2:
        raise ValueError(f"padding_ratio must have two values: {padding_ratio}")
    left_ratio, right_ratio = (float(v) for v in padding_ratio)
    if abs(left_ratio + right_ratio - 1.0) > 1e-6:
        raise ValueError(f"padding_ratio must sum to one: {padding_ratio}")

    total_right = 0
    for index in range(depth):
        dilation = 2**index
        total_padding = dilation * 2
        left = round(total_padding * left_ratio)
        right = round(total_padding * right_ratio)
        if left + right != total_padding:
            right = total_padding - left
        total_right += right
    return total_right


def decoder_padding_ratios_for_future_frames(
    future_frames: Mapping[str, int],
    depth: int = 4,
) -> Dict[str, List[float]]:
    """Resolve the public LA contract to validated DS_DDB padding ratios."""
    ratios: Dict[str, List[float]] = {}
    for latency_id, expected in future_frames.items():
        if latency_id not in DECODER_PADDING_RATIOS:
            raise KeyError(
                f"No decoder padding implementation for {latency_id}; "
                f"expected one of {list(DECODER_PADDING_RATIOS)}"
            )
        ratio = DECODER_PADDING_RATIOS[latency_id]
        actual = compute_lookahead_frames(ratio, depth=depth)
        if actual != int(expected):
            raise ValueError(
                f"{latency_id} requests {expected} future frames but ratio "
                f"{ratio} produces {actual} at depth={depth}"
            )
        ratios[latency_id] = [float(ratio[0]), float(ratio[1])]
    return ratios


def validate_v2_grid(
    sample_rates: Iterable[int] = TARGET_SAMPLE_RATES,
    future_frames: Mapping[str, int] = LATENCY_FUTURE_FRAMES,
    window_ms: float = 40.0,
    hop_ms: float = 20.0,
    depth: int = 4,
) -> None:
    rates = tuple(int(rate) for rate in sample_rates)
    if len(rates) != len(set(rates)):
        raise ValueError(f"sample rates contain duplicates: {rates}")
    if set(future_frames) != set(LATENCY_FUTURE_FRAMES):
        raise ValueError(
            f"v2 latency ids must be {list(LATENCY_FUTURE_FRAMES)}: {future_frames}"
        )

    for rate in rates:
        profile = SFIProfile(rate, window_ms=window_ms, hop_ms=hop_ms)
        if abs(profile.bin_hz - 25.0) > 1e-9:
            raise ValueError(f"v2 frequency grid must be 25 Hz/bin: {profile}")
    decoder_padding_ratios_for_future_frames(future_frames, depth=depth)


@dataclass(frozen=True)
class OperatingPoint:
    sample_rate: int
    latency_id: str
    future_frames: int

    @property
    def algorithmic_latency_ms(self) -> float:
        # 40 ms centered window contributes 20 ms; each future frame is one
        # 20 ms hop under the locked v2 frontend.
        return 20.0 + self.future_frames * 20.0


class BalancedOperatingPointSchedule:
    """Seeded, resumable, exactly balanced 7-rate x 4-latency schedule."""

    def __init__(
        self,
        sample_rates: Sequence[int] = TARGET_SAMPLE_RATES,
        future_frames: Mapping[str, int] = LATENCY_FUTURE_FRAMES,
        seed: int = 2039,
    ) -> None:
        validate_v2_grid(sample_rates=sample_rates, future_frames=future_frames)
        self.sample_rates = tuple(int(rate) for rate in sample_rates)
        self.future_frames = {
            str(latency_id): int(frames)
            for latency_id, frames in future_frames.items()
        }
        self.seed = int(seed)
        self.cycle_index = 0
        self.position = 0
        self._cycle = self._build_cycle(self.cycle_index)

    @property
    def cycle_size(self) -> int:
        return len(self.sample_rates) * len(self.future_frames)

    def _build_cycle(self, cycle_index: int) -> List[OperatingPoint]:
        points = [
            OperatingPoint(rate, latency_id, frames)
            for rate in self.sample_rates
            for latency_id, frames in self.future_frames.items()
        ]
        random.Random(self.seed + int(cycle_index)).shuffle(points)
        return points

    def next(self) -> OperatingPoint:
        if self.position >= len(self._cycle):
            self.cycle_index += 1
            self.position = 0
            self._cycle = self._build_cycle(self.cycle_index)
        point = self._cycle[self.position]
        self.position += 1
        return point

    def state_dict(self) -> Dict[str, int | List[int] | Dict[str, int]]:
        return {
            "sample_rates": list(self.sample_rates),
            "future_frames": dict(self.future_frames),
            "seed": self.seed,
            "cycle_index": self.cycle_index,
            "position": self.position,
        }

    def load_state_dict(self, state: Mapping) -> None:
        expected_rates = list(self.sample_rates)
        expected_frames = dict(self.future_frames)
        if list(state["sample_rates"]) != expected_rates:
            raise ValueError("schedule sample rates do not match the current config")
        if dict(state["future_frames"]) != expected_frames:
            raise ValueError("schedule latency grid does not match the current config")
        if int(state["seed"]) != self.seed:
            raise ValueError("schedule seed does not match the current config")

        cycle_index = int(state["cycle_index"])
        position = int(state["position"])
        if cycle_index < 0 or not 0 <= position <= self.cycle_size:
            raise ValueError(f"invalid schedule cursor: cycle={cycle_index}, position={position}")
        self.cycle_index = cycle_index
        self.position = position
        self._cycle = self._build_cycle(cycle_index)


# Validate constants at import time.  This remains a pure-Python check.
validate_v2_grid()
