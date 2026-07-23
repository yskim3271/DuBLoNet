from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence

import torch
import torch.nn as nn

from src.models.backbone import (
    AsymmetricConv2d,
    DS_DDB,
    DenseEncoder,
    FixedMaskGate,
    LearnableSigmoid_2d,
    MaskDecoder,
    PhaseDecoder,
    TS_BLOCK,
    get_padding_2d,
    _make_norm2d,
    _validate_padding_ratio,
    complex_to_mag_pha,
    mag_pha_to_complex,
)
from src.v2_contract import (
    compute_lookahead_frames,
    decoder_padding_ratios_for_future_frames,
)


DEFAULT_LATENCY_IDS = ["L0", "L1", "L3", "L5"]
DEFAULT_PADDING_RATIOS = {
    "L0": [1.00, 0.00],
    "L1": [0.96, 0.04],
    "L3": [0.90, 0.10],
    "L5": [0.84, 0.16],
}
VALID_EXPERT_SCOPES = {"dsddb_bn", "dsddb_only", "asymconv_only", "full"}
VALID_EXPERT_TOPOLOGIES = {"encoder_decoder", "decoder_only"}


def _normalize_latency_ids(latency_ids: Optional[Sequence[str]]) -> List[str]:
    ids = list(latency_ids or DEFAULT_LATENCY_IDS)
    if not ids:
        raise ValueError("latency_ids must contain at least one id")
    if len(set(ids)) != len(ids):
        raise ValueError(f"latency_ids contains duplicates: {ids}")
    return ids


def _normalize_ratio_map(
    ratio_map: Optional[Mapping[str, Sequence[float]]],
    latency_ids: Sequence[str],
    name: str,
) -> Dict[str, List[float]]:
    source = ratio_map or DEFAULT_PADDING_RATIOS
    ratios: Dict[str, List[float]] = {}
    for latency_id in latency_ids:
        if latency_id not in source:
            raise KeyError(f"{name} is missing latency id '{latency_id}'")
        ratio = [float(v) for v in source[latency_id]]
        _validate_padding_ratio(ratio, f"{name}.{latency_id}")
        ratios[latency_id] = ratio
    return ratios


def _shared_ratio_from_map(
    ratios: Mapping[str, Sequence[float]],
    name: str,
) -> List[float]:
    """Return the common ratio and fail if a shared block has per-id ratios."""
    iterator = iter(ratios.items())
    first_latency_id, first_ratio = next(iterator)
    first_ratio = [float(v) for v in first_ratio]
    for latency_id, ratio in iterator:
        ratio = [float(v) for v in ratio]
        if any(abs(a - b) > 1e-6 for a, b in zip(first_ratio, ratio)):
            raise ValueError(
                f"{name} must be identical for all latency ids when "
                f"latency_expert.topology='decoder_only'. "
                f"Got {first_latency_id}={first_ratio} and {latency_id}={ratio}."
            )
    return first_ratio


class LatencyExpertMixin:
    latency_ids: List[str]
    active_latency_id: str

    def _validate_latency_id(self, latency_id: Optional[str]) -> str:
        latency_id = latency_id or self.active_latency_id
        if latency_id not in self.latency_ids:
            raise KeyError(
                f"Unknown latency_id '{latency_id}'. "
                f"Expected one of {self.latency_ids}."
            )
        return latency_id

    def set_latency_id(self, latency_id: str) -> None:
        self.active_latency_id = self._validate_latency_id(latency_id)
        for module in self.modules():
            if module is self:
                continue
            if hasattr(module, "latency_ids") and hasattr(module, "active_latency_id"):
                if self.active_latency_id not in module.latency_ids:
                    raise KeyError(
                        f"Child module {module.__class__.__name__} does not support "
                        f"latency_id '{self.active_latency_id}'."
                    )
                module.active_latency_id = self.active_latency_id


class LatencyExpertBatchNorm2d(nn.Module, LatencyExpertMixin):
    """BatchNorm2d with one set of parameters/statistics per latency id."""

    def __init__(self, num_features: int, latency_ids: Sequence[str], default_latency_id: str):
        super().__init__()
        self.latency_ids = list(latency_ids)
        self.active_latency_id = default_latency_id
        self.norms = nn.ModuleDict({
            latency_id: nn.BatchNorm2d(num_features)
            for latency_id in self.latency_ids
        })

    def forward(self, x: torch.Tensor, latency_id: Optional[str] = None) -> torch.Tensor:
        latency_id = self._validate_latency_id(latency_id)
        return self.norms[latency_id](x)


class LatencyExpertModuleDict(nn.Module, LatencyExpertMixin):
    """Select exactly one full module branch for the active latency id."""

    def __init__(
        self,
        experts: Mapping[str, nn.Module],
        latency_ids: Sequence[str],
        default_latency_id: str,
    ):
        super().__init__()
        self.latency_ids = list(latency_ids)
        self.active_latency_id = default_latency_id
        self.experts = nn.ModuleDict({
            latency_id: experts[latency_id]
            for latency_id in self.latency_ids
        })

    def forward(self, *args, latency_id: Optional[str] = None, **kwargs):
        latency_id = self._validate_latency_id(latency_id)
        return self.experts[latency_id](*args, **kwargs)


def _make_latency_expert_norm2d(
    num_features: int,
    norm_type: str,
    latency_ids: Sequence[str],
    default_latency_id: str,
    use_expert_norm: bool,
) -> nn.Module:
    if use_expert_norm and norm_type == "batch":
        return LatencyExpertBatchNorm2d(num_features, latency_ids, default_latency_id)
    return _make_norm2d(num_features, norm_type)


class LatencyExpertAsymmetricConv2d(nn.Module, LatencyExpertMixin):
    """AsymmetricConv2d branch selector for asymconv-only ablations."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding,
        padding_ratios: Mapping[str, Sequence[float]],
        latency_ids: Sequence[str],
        default_latency_id: str,
        stride=1,
        dilation=(1, 1),
        groups=1,
        bias=True,
    ):
        super().__init__()
        self.latency_ids = list(latency_ids)
        self.active_latency_id = default_latency_id
        self.padding_ratios = {
            latency_id: [float(v) for v in padding_ratios[latency_id]]
            for latency_id in self.latency_ids
        }
        self.experts = nn.ModuleDict({
            latency_id: AsymmetricConv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                padding_ratio=self.padding_ratios[latency_id],
                stride=stride,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
            for latency_id in self.latency_ids
        })

    def forward(self, x: torch.Tensor, latency_id: Optional[str] = None) -> torch.Tensor:
        latency_id = self._validate_latency_id(latency_id)
        return self.experts[latency_id](x)


class LatencyExpertDSDDB(nn.Module, LatencyExpertMixin):
    """Latency-specific DS_DDB branch selector.

    Only the selected branch executes during forward, preserving the active
    compute profile of a single-latency LaCo-SENet.
    """

    def __init__(
        self,
        dense_channel: int,
        padding_ratios: Mapping[str, Sequence[float]],
        latency_ids: Sequence[str],
        default_latency_id: str,
        kernel_size=(3, 3),
        depth: int = 4,
        causal: bool = False,
        norm_type: str = "batch",
    ):
        super().__init__()
        self.latency_ids = list(latency_ids)
        self.active_latency_id = default_latency_id
        self.padding_ratios = {
            latency_id: [float(v) for v in padding_ratios[latency_id]]
            for latency_id in self.latency_ids
        }
        self.experts = nn.ModuleDict({
            latency_id: DS_DDB(
                dense_channel=dense_channel,
                kernel_size=kernel_size,
                depth=depth,
                causal=causal,
                padding_ratio=self.padding_ratios[latency_id],
                norm_type=norm_type,
            )
            for latency_id in self.latency_ids
        })

    def forward(self, x: torch.Tensor, latency_id: Optional[str] = None) -> torch.Tensor:
        latency_id = self._validate_latency_id(latency_id)
        return self.experts[latency_id](x)


class AsymConvExpertDSDDB(nn.Module, LatencyExpertMixin):
    """DS_DDB variant where only the depthwise AsymmetricConv2d is expertized."""

    def __init__(
        self,
        dense_channel,
        padding_ratios: Mapping[str, Sequence[float]],
        latency_ids: Sequence[str],
        default_latency_id: str,
        kernel_size=(3, 3),
        depth: int = 4,
        causal: bool = False,
        norm_type: str = "batch",
    ):
        super().__init__()
        self.latency_ids = list(latency_ids)
        self.active_latency_id = default_latency_id
        self.dense_channel = dense_channel
        self.depth = depth
        self.dense_block = nn.ModuleList([])
        self.padding_ratios = {
            latency_id: [float(v) for v in padding_ratios[latency_id]]
            for latency_id in self.latency_ids
        }

        for i in range(depth):
            dil = 2 ** i
            padding = get_padding_2d(kernel_size, (dil, 1))
            channels = dense_channel * (i + 1)
            dense_conv = nn.Sequential(
                LatencyExpertAsymmetricConv2d(
                    channels,
                    channels,
                    kernel_size,
                    padding=padding,
                    padding_ratios=self.padding_ratios,
                    latency_ids=self.latency_ids,
                    default_latency_id=default_latency_id,
                    dilation=(dil, 1),
                    groups=channels,
                    bias=True,
                ),
                nn.Conv2d(
                    in_channels=channels,
                    out_channels=dense_channel,
                    kernel_size=1,
                    padding=0,
                    stride=1,
                    groups=1,
                    bias=True,
                ),
                _make_norm2d(dense_channel, norm_type),
                nn.PReLU(dense_channel),
            )
            self.dense_block.append(dense_conv)

    def forward(self, x: torch.Tensor, latency_id: Optional[str] = None) -> torch.Tensor:
        latency_id = self._validate_latency_id(latency_id)
        self.set_latency_id(latency_id)
        skip = x
        for i in range(self.depth):
            x = self.dense_block[i](skip)
            skip = torch.cat([x, skip], dim=1)
        return x


def _make_expert_dense_block(
    dense_channel: int,
    padding_ratios: Mapping[str, Sequence[float]],
    latency_ids: Sequence[str],
    default_latency_id: str,
    expert_scope: str,
    depth: int = 4,
    causal: bool = False,
    norm_type: str = "batch",
) -> nn.Module:
    if expert_scope in {"dsddb_bn", "dsddb_only"}:
        return LatencyExpertDSDDB(
            dense_channel=dense_channel,
            padding_ratios=padding_ratios,
            latency_ids=latency_ids,
            default_latency_id=default_latency_id,
            depth=depth,
            causal=causal,
            norm_type=norm_type,
        )
    if expert_scope == "asymconv_only":
        return AsymConvExpertDSDDB(
            dense_channel=dense_channel,
            padding_ratios=padding_ratios,
            latency_ids=latency_ids,
            default_latency_id=default_latency_id,
            depth=depth,
            causal=causal,
            norm_type=norm_type,
        )
    raise ValueError(
        f"Unknown expert_scope '{expert_scope}'. Expected one of {sorted(VALID_EXPERT_SCOPES)}."
    )


class LatencyExpertDenseEncoder(nn.Module, LatencyExpertMixin):
    def __init__(
        self,
        dense_channel: int,
        in_channel: int,
        padding_ratios: Mapping[str, Sequence[float]],
        latency_ids: Sequence[str],
        default_latency_id: str,
        depth: int = 4,
        causal: bool = False,
        norm_type: str = "batch",
        expert_scope: str = "dsddb_bn",
    ):
        super().__init__()
        self.latency_ids = list(latency_ids)
        self.active_latency_id = default_latency_id
        use_expert_norm = expert_scope == "dsddb_bn"
        self.dense_channel = dense_channel
        self.dense_conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, dense_channel, (1, 1)),
            _make_latency_expert_norm2d(
                dense_channel, norm_type, latency_ids, default_latency_id, use_expert_norm
            ),
            nn.PReLU(dense_channel),
        )
        self.dense_block = _make_expert_dense_block(
            dense_channel=dense_channel,
            padding_ratios=padding_ratios,
            latency_ids=latency_ids,
            default_latency_id=default_latency_id,
            expert_scope=expert_scope,
            depth=depth,
            causal=causal,
            norm_type=norm_type,
        )
        self.dense_conv_2 = nn.Sequential(
            nn.Conv2d(dense_channel, dense_channel, (1, 3), (1, 2)),
            _make_latency_expert_norm2d(
                dense_channel, norm_type, latency_ids, default_latency_id, use_expert_norm
            ),
            nn.PReLU(dense_channel),
        )

    def forward(self, x: torch.Tensor, latency_id: Optional[str] = None) -> torch.Tensor:
        latency_id = self._validate_latency_id(latency_id)
        self.set_latency_id(latency_id)
        x = self.dense_conv_1(x)
        x = self.dense_block(x)
        x = self.dense_conv_2(x)
        return x


class LatencyExpertMaskDecoder(nn.Module, LatencyExpertMixin):
    def __init__(
        self,
        dense_channel: int,
        n_fft: int,
        sigmoid_beta: float,
        padding_ratios: Mapping[str, Sequence[float]],
        latency_ids: Sequence[str],
        default_latency_id: str,
        out_channel: int = 1,
        depth: int = 4,
        causal: bool = False,
        norm_type: str = "batch",
        expert_scope: str = "dsddb_bn",
        mask_gate: str = "learnable",
    ):
        super().__init__()
        self.latency_ids = list(latency_ids)
        self.active_latency_id = default_latency_id
        use_expert_norm = expert_scope == "dsddb_bn"
        self.n_fft = n_fft
        self.dense_block = _make_expert_dense_block(
            dense_channel=dense_channel,
            padding_ratios=padding_ratios,
            latency_ids=latency_ids,
            default_latency_id=default_latency_id,
            expert_scope=expert_scope,
            depth=depth,
            causal=causal,
            norm_type=norm_type,
        )
        self.mask_conv = nn.Sequential(
            nn.ConvTranspose2d(dense_channel, dense_channel, (1, 3), (1, 2)),
            nn.Conv2d(dense_channel, out_channel, (1, 1)),
            _make_latency_expert_norm2d(
                out_channel, norm_type, latency_ids, default_latency_id, use_expert_norm
            ),
            nn.PReLU(out_channel),
            nn.Conv2d(out_channel, out_channel, (1, 1)),
        )
        if mask_gate == "fixed":
            self.lsigmoid = FixedMaskGate(beta=sigmoid_beta)
        elif mask_gate == "learnable":
            self.lsigmoid = LearnableSigmoid_2d(n_fft // 2 + 1, beta=sigmoid_beta)
        else:
            raise ValueError(
                f"Unknown mask_gate '{mask_gate}' (expected 'learnable' or 'fixed')"
            )

    def forward(self, x: torch.Tensor, latency_id: Optional[str] = None) -> torch.Tensor:
        latency_id = self._validate_latency_id(latency_id)
        self.set_latency_id(latency_id)
        x = self.dense_block(x)
        x = self.mask_conv(x)
        x = x.squeeze(1).transpose(1, 2)
        x = self.lsigmoid(x).transpose(1, 2).unsqueeze(1)
        return x


class LatencyExpertPhaseDecoder(nn.Module, LatencyExpertMixin):
    def __init__(
        self,
        dense_channel: int,
        padding_ratios: Mapping[str, Sequence[float]],
        latency_ids: Sequence[str],
        default_latency_id: str,
        out_channel: int = 1,
        depth: int = 4,
        causal: bool = False,
        norm_type: str = "batch",
        expert_scope: str = "dsddb_bn",
    ):
        super().__init__()
        self.latency_ids = list(latency_ids)
        self.active_latency_id = default_latency_id
        use_expert_norm = expert_scope == "dsddb_bn"
        self.dense_block = _make_expert_dense_block(
            dense_channel=dense_channel,
            padding_ratios=padding_ratios,
            latency_ids=latency_ids,
            default_latency_id=default_latency_id,
            expert_scope=expert_scope,
            depth=depth,
            causal=causal,
            norm_type=norm_type,
        )
        self.phase_conv = nn.Sequential(
            nn.ConvTranspose2d(dense_channel, dense_channel, (1, 3), (1, 2)),
            _make_latency_expert_norm2d(
                dense_channel, norm_type, latency_ids, default_latency_id, use_expert_norm
            ),
            nn.PReLU(dense_channel),
        )
        self.phase_conv_r = nn.Conv2d(dense_channel, out_channel, (1, 1))
        self.phase_conv_i = nn.Conv2d(dense_channel, out_channel, (1, 1))

    def forward(self, x: torch.Tensor, latency_id: Optional[str] = None) -> torch.Tensor:
        latency_id = self._validate_latency_id(latency_id)
        self.set_latency_id(latency_id)
        x = self.dense_block(x)
        x = self.phase_conv(x)
        x_r = self.phase_conv_r(x)
        x_i = self.phase_conv_i(x)
        return torch.atan2(x_i + 1e-8, x_r + 1e-8)


class LatencyExpertBackbone(nn.Module, LatencyExpertMixin):
    """LaCo-SENet backbone with configurable latency expert scope."""

    def __init__(
        self,
        win_len,
        hop_len,
        fft_len,
        dense_channel,
        sigmoid_beta,
        compress_factor,
        dense_depth=4,
        num_tsblock=4,
        time_dw_kernel_size=3,
        time_block_kernel=[3, 5, 7, 9],
        freq_block_kernel=[3, 11, 23, 31],
        time_block_num=2,
        freq_block_num=2,
        causal_ts_block=False,
        encoder_padding_ratio=(0.5, 0.5),
        decoder_padding_ratio=(0.5, 0.5),
        sca_kernel_size=11,
        norm_type="batch",
        latency_expert=None,
        sfi=None,
    ):
        super().__init__()
        latency_expert = latency_expert or {}
        self.latency_ids = _normalize_latency_ids(latency_expert.get("latency_ids"))
        self.default_latency_id = latency_expert.get("default_latency_id", self.latency_ids[0])
        if self.default_latency_id not in self.latency_ids:
            raise ValueError(
                f"default_latency_id '{self.default_latency_id}' is not in {self.latency_ids}"
            )
        self.active_latency_id = self.default_latency_id
        self.expert_scope = latency_expert.get("expert_scope", "dsddb_bn")
        if self.expert_scope not in VALID_EXPERT_SCOPES:
            raise ValueError(
                f"Unknown expert_scope '{self.expert_scope}'. "
                f"Expected one of {sorted(VALID_EXPERT_SCOPES)}."
            )
        self.expert_topology = latency_expert.get(
            "topology",
            latency_expert.get("expert_topology", "encoder_decoder"),
        )
        if self.expert_topology not in VALID_EXPERT_TOPOLOGIES:
            raise ValueError(
                f"Unknown latency_expert.topology '{self.expert_topology}'. "
                f"Expected one of {sorted(VALID_EXPERT_TOPOLOGIES)}."
            )

        configured_future_frames = latency_expert.get("future_frames")
        self.future_frames = (
            {
                str(latency_id): int(frames)
                for latency_id, frames in configured_future_frames.items()
            }
            if configured_future_frames is not None
            else None
        )
        if self.future_frames is not None and set(self.future_frames) != set(self.latency_ids):
            raise ValueError(
                "latency_expert.future_frames keys must match latency_ids: "
                f"ids={self.latency_ids}, future_frames={self.future_frames}"
            )

        encoder_ratio_map = latency_expert.get("encoder_padding_ratios")
        decoder_ratio_map = latency_expert.get("decoder_padding_ratios")

        # If a caller omits the expert maps, keep compatibility with the usual
        # single-ratio config by assigning that ratio to every expert.
        if encoder_ratio_map is None:
            if self.expert_topology == "decoder_only" and self.future_frames is not None:
                encoder_ratio_map = {
                    latency_id: [1.0, 0.0] for latency_id in self.latency_ids
                }
            else:
                encoder_ratio_map = {
                    latency_id: encoder_padding_ratio for latency_id in self.latency_ids
                }
        if decoder_ratio_map is None:
            if self.future_frames is not None:
                decoder_ratio_map = decoder_padding_ratios_for_future_frames(
                    self.future_frames, depth=dense_depth
                )
            else:
                decoder_ratio_map = {
                    latency_id: decoder_padding_ratio for latency_id in self.latency_ids
                }

        self.encoder_padding_ratios = _normalize_ratio_map(
            encoder_ratio_map, self.latency_ids, "encoder_padding_ratios"
        )
        self.decoder_padding_ratios = _normalize_ratio_map(
            decoder_ratio_map, self.latency_ids, "decoder_padding_ratios"
        )
        if self.future_frames is not None:
            for latency_id, expected in self.future_frames.items():
                actual = compute_lookahead_frames(
                    self.decoder_padding_ratios[latency_id], depth=dense_depth
                )
                if actual != expected:
                    raise ValueError(
                        f"{latency_id} requests {expected} future frames but its "
                        f"decoder padding produces {actual}"
                    )
        if self.expert_topology == "decoder_only":
            self.shared_encoder_padding_ratio = _shared_ratio_from_map(
                self.encoder_padding_ratios, "encoder_padding_ratios"
            )
            self.encoder_padding_ratio = self.shared_encoder_padding_ratio
        else:
            self.shared_encoder_padding_ratio = None
            self.encoder_padding_ratio = self.encoder_padding_ratios[self.default_latency_id]
        self.decoder_padding_ratio = self.decoder_padding_ratios[self.default_latency_id]

        self.win_len = win_len
        self.hop_len = hop_len
        self.fft_len = fft_len
        self.dense_channel = dense_channel
        self.dense_depth = dense_depth
        self.sigmoid_beta = sigmoid_beta
        self.compress_factor = compress_factor
        self.num_tsblock = num_tsblock
        self.causal_ts_block = causal_ts_block
        self.norm_type = norm_type
        self.sfi = sfi or {}
        self.sfi_enabled = bool(self.sfi.get("enabled", False))
        self.pad_even_frequency = bool(
            self.sfi.get("pad_even_frequency", self.sfi_enabled)
        )
        self.mask_gate_type = (
            "fixed"
            if self.sfi.get("fixed_mask_gate", self.sfi_enabled)
            else "learnable"
        )

        if self.expert_topology == "decoder_only":
            self.dense_encoder = DenseEncoder(
                dense_channel,
                in_channel=2,
                depth=dense_depth,
                causal=causal_ts_block,
                padding_ratio=self.encoder_padding_ratio,
                norm_type=norm_type,
            )
        elif self.expert_scope == "full":
            self.dense_encoder = LatencyExpertModuleDict(
                {
                    latency_id: DenseEncoder(
                        dense_channel,
                        in_channel=2,
                        depth=dense_depth,
                        causal=causal_ts_block,
                        padding_ratio=self.encoder_padding_ratios[latency_id],
                        norm_type=norm_type,
                    )
                    for latency_id in self.latency_ids
                },
                latency_ids=self.latency_ids,
                default_latency_id=self.default_latency_id,
            )
        else:
            self.dense_encoder = LatencyExpertDenseEncoder(
                dense_channel,
                in_channel=2,
                depth=dense_depth,
                causal=causal_ts_block,
                padding_ratios=self.encoder_padding_ratios,
                latency_ids=self.latency_ids,
                default_latency_id=self.default_latency_id,
                norm_type=norm_type,
                expert_scope=self.expert_scope,
            )
        self.sequence_block = nn.Sequential(
            *[
                TS_BLOCK(
                    dense_channel,
                    time_block_num,
                    freq_block_num,
                    time_dw_kernel_size,
                    time_block_kernel,
                    freq_block_kernel,
                    causal=causal_ts_block,
                    sca_kernel_size=sca_kernel_size,
                )
                for _ in range(num_tsblock)
            ]
        )
        if self.expert_scope == "full":
            self.mask_decoder = LatencyExpertModuleDict(
                {
                    latency_id: MaskDecoder(
                        dense_channel,
                        fft_len,
                        sigmoid_beta,
                        out_channel=1,
                        depth=dense_depth,
                        causal=causal_ts_block,
                        padding_ratio=self.decoder_padding_ratios[latency_id],
                        norm_type=norm_type,
                        mask_gate=self.mask_gate_type,
                    )
                    for latency_id in self.latency_ids
                },
                latency_ids=self.latency_ids,
                default_latency_id=self.default_latency_id,
            )
            self.phase_decoder = LatencyExpertModuleDict(
                {
                    latency_id: PhaseDecoder(
                        dense_channel,
                        out_channel=1,
                        depth=dense_depth,
                        causal=causal_ts_block,
                        padding_ratio=self.decoder_padding_ratios[latency_id],
                        norm_type=norm_type,
                    )
                    for latency_id in self.latency_ids
                },
                latency_ids=self.latency_ids,
                default_latency_id=self.default_latency_id,
            )
        else:
            self.mask_decoder = LatencyExpertMaskDecoder(
                dense_channel,
                fft_len,
                sigmoid_beta,
                out_channel=1,
                depth=dense_depth,
                causal=causal_ts_block,
                padding_ratios=self.decoder_padding_ratios,
                latency_ids=self.latency_ids,
                default_latency_id=self.default_latency_id,
                norm_type=norm_type,
                expert_scope=self.expert_scope,
                mask_gate=self.mask_gate_type,
            )
            self.phase_decoder = LatencyExpertPhaseDecoder(
                dense_channel,
                out_channel=1,
                depth=dense_depth,
                causal=causal_ts_block,
                padding_ratios=self.decoder_padding_ratios,
                latency_ids=self.latency_ids,
                default_latency_id=self.default_latency_id,
                norm_type=norm_type,
                expert_scope=self.expert_scope,
            )

    @property
    def is_latency_expert(self) -> bool:
        return True

    def get_latency_ids(self) -> List[str]:
        return list(self.latency_ids)

    def get_padding_ratio(self, latency_id: Optional[str] = None):
        latency_id = self._validate_latency_id(latency_id)
        return {
            "encoder_padding_ratio": self.encoder_padding_ratios[latency_id],
            "decoder_padding_ratio": self.decoder_padding_ratios[latency_id],
        }

    def get_future_frames(self, latency_id: Optional[str] = None) -> int:
        latency_id = self._validate_latency_id(latency_id)
        if self.future_frames is not None:
            return self.future_frames[latency_id]
        return compute_lookahead_frames(
            self.decoder_padding_ratios[latency_id], depth=self.dense_depth
        )

    def set_latency_id(self, latency_id: str) -> None:
        super().set_latency_id(latency_id)
        self.encoder_padding_ratio = self.encoder_padding_ratios[self.active_latency_id]
        self.decoder_padding_ratio = self.decoder_padding_ratios[self.active_latency_id]

    def forward(self, noisy_com: torch.Tensor, latency_id: Optional[str] = None):
        latency_id = self._validate_latency_id(latency_id)
        self.set_latency_id(latency_id)

        original_frequency_bins = noisy_com.shape[1]
        if self.pad_even_frequency and original_frequency_bins % 2 == 0:
            pad = noisy_com.new_zeros(
                noisy_com.shape[0], 1, noisy_com.shape[2], noisy_com.shape[3]
            )
            noisy_com = torch.cat((noisy_com, pad), dim=1)

        mag, pha = complex_to_mag_pha(noisy_com, stack_dim=-1)
        x = torch.stack((mag, pha), dim=1).permute(0, 1, 3, 2)
        x = self.dense_encoder(x)
        x = self.sequence_block(x)

        mask = self.mask_decoder(x).squeeze(1).transpose(1, 2)
        mask = mask[:, :original_frequency_bins, :]
        mag = mag[:, :original_frequency_bins, :]
        est_mag = mag * mask

        est_pha = self.phase_decoder(x).squeeze(1).transpose(1, 2)
        est_pha = est_pha[:, :original_frequency_bins, :]
        est_com = mag_pha_to_complex(est_mag, est_pha, stack_dim=-1)
        return est_mag, est_pha, est_com


def latency_expert_ids(model: nn.Module) -> List[str]:
    if hasattr(model, "get_latency_ids"):
        return model.get_latency_ids()
    return []


def set_latency_id(model: nn.Module, latency_id: str) -> None:
    if not hasattr(model, "set_latency_id"):
        raise TypeError(f"Model {model.__class__.__name__} does not support latency_id")
    model.set_latency_id(latency_id)
