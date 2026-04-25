from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.stft import mag_pha_to_complex, complex_to_mag_pha


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

def get_padding_2d(kernel_size, dilation=(1, 1)):
    return (int((kernel_size[0]*dilation[0] - dilation[0])/2), int((kernel_size[1]*dilation[1] - dilation[1])/2))

def _validate_padding_ratio(padding_ratio, name="padding_ratio"):
    left, right = padding_ratio
    assert abs(left + right - 1.0) < 1e-6, \
        f"{name} must sum to 1.0, got {left + right}"
    assert 0.0 <= left <= 1.0 and 0.0 <= right <= 1.0, \
        f"{name} values must be in [0, 1], got ({left}, {right})"

def _make_norm2d(num_features, norm_type="batch"):
    if norm_type == "batch":
        return nn.BatchNorm2d(num_features)
    elif norm_type == "instance":
        return nn.InstanceNorm2d(num_features, affine=True)
    else:
        raise ValueError(f"Unknown norm_type: {norm_type} (expected 'batch' or 'instance')")

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1, dilation=1, groups=1, bias=True):
        super(CausalConv1d, self).__init__()
        self.padding = padding * 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=0,
                              stride=stride,
                              dilation=dilation,
                              groups=groups,
                              bias=bias)

    def forward(self, x):
        x = F.pad(x, [self.padding, 0])
        return self.conv(x)

class CausalConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1, dilation=(1, 1), groups=1, bias=True):
        super(CausalConv2d, self).__init__()
        time_padding = padding[0] * 2
        freq_padding = padding[1]
        self.padding = (freq_padding, freq_padding, time_padding, 0)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              padding=0,
                              stride=stride,
                              dilation=dilation,
                              groups=groups,
                              bias=bias)

    def forward(self, x):
        x = F.pad(x, self.padding)
        return self.conv(x)

class AsymmetricConv2d(nn.Module):
    """
    2D convolution with asymmetric time-axis padding.

    This module enables flexible control over the receptive field by allowing
    different amounts of padding on the left (past) and right (future) of the
    time axis, while maintaining symmetric padding on the frequency axis.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (tuple): Convolution kernel size (time, freq)
        padding (tuple): Total padding per axis (time_padding, freq_padding)
        padding_ratio (tuple): How to split time padding (left_ratio, right_ratio)
            - (1.0, 0.0): Fully causal (all padding on left/past)
            - (0.5, 0.5): Symmetric (equal padding on both sides)
            - (0.8, 0.2): Asymmetric (80% left/past, 20% right/future)
        stride (int): Convolution stride
        dilation (tuple): Dilation rate for time and frequency axes
        groups (int): Number of groups for grouped convolution
        bias (bool): Whether to use bias

    Note:
        - Frequency axis always uses symmetric padding (no causal meaning)
        - Padding format for F.pad: (left, right, top, bottom)
        - Time axis corresponds to the height dimension (top/bottom in F.pad)

    Example:
        >>> conv = AsymmetricConv2d(64, 64, (3, 3), padding=(4, 1),
        ...                          padding_ratio=(0.75, 0.25))
        >>> # time_padding=4: left=3, right=1 (75%:25% split)
        >>> # freq_padding=1: symmetric padding
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding,
                 padding_ratio=(1.0, 0.0), stride=1, dilation=(1, 1),
                 groups=1, bias=True):
        super(AsymmetricConv2d, self).__init__()
        # padding[0] is the symmetric padding value (one-side)
        # For full RF preservation, total padding = padding[0] * 2
        # This matches CausalConv2d interface where time_padding = padding[0] * 2
        time_padding_total = padding[0] * 2
        freq_padding = padding[1]
        left_ratio, right_ratio = padding_ratio
        _validate_padding_ratio(padding_ratio)

        # Distribute time padding according to ratio
        time_padding_left = round(time_padding_total * left_ratio)
        time_padding_right = round(time_padding_total * right_ratio)

        # Verify rounding preserves total (important for correctness)
        actual_total = time_padding_left + time_padding_right
        if actual_total != time_padding_total:
            # Adjust right padding to ensure exact total
            time_padding_right = time_padding_total - time_padding_left

        assert time_padding_left + time_padding_right == time_padding_total, \
            f"Padding split failed: {time_padding_left} + {time_padding_right} != {time_padding_total}"

        # F.pad format: (left, right, top, bottom)
        # Time axis: top=left (past), bottom=right (future)
        # Freq axis: left, right (symmetric)
        self.padding = (freq_padding, freq_padding,
                       time_padding_left, time_padding_right)

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                             padding=0,
                             stride=stride,
                             dilation=dilation,
                             groups=groups,
                             bias=bias)

        # Store for inspection/debugging
        self.padding_ratio = padding_ratio
        self.time_padding_left = time_padding_left
        self.time_padding_right = time_padding_right

    def forward(self, x):
        x = F.pad(x, self.padding)
        return self.conv(x)

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"padding_ratio={self.padding_ratio}, "
                f"time_padding=({self.time_padding_left}, {self.time_padding_right}))")

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        B, C, T = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1) * y + bias.view(1, C, 1)
        return y
    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        B, C, T = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=2).sum(dim=0), grad_output.sum(dim=2).sum(
            dim=0), None

class LayerNorm1d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm1d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)
    
class Group_Prime_Kernel_FFN(nn.Module):
    def __init__(self, in_channel: int = 64, kernel_list: List[int] = [3, 11, 23, 31], causal: bool = False):
        super().__init__()
        self.in_channel = in_channel
        self.expand_ratio = len(kernel_list)
        self.mid_channel = self.in_channel * self.expand_ratio
        self.kernel_list = kernel_list
        self.causal = causal

        if causal:
            conv_fn = CausalConv1d
        else:
            conv_fn = nn.Conv1d

        self.proj_first = nn.Sequential(
            nn.Conv1d(self.in_channel, self.mid_channel, kernel_size=1))
        self.proj_last = nn.Sequential(
            nn.Conv1d(self.mid_channel, self.in_channel, kernel_size=1))
        self.norm = LayerNorm1d(self.in_channel)
        self.scale = nn.Parameter(torch.zeros((1, self.in_channel, 1)), requires_grad=True)

        for kernel_size in self.kernel_list:
            setattr(self, f"attn_{kernel_size}", nn.Sequential(
                conv_fn(self.in_channel, self.in_channel, kernel_size=kernel_size, groups=self.in_channel, padding=get_padding(kernel_size)),
                nn.Conv1d(self.in_channel, self.in_channel, kernel_size=1)
            ))
            setattr(self, f"conv_{kernel_size}", conv_fn(self.in_channel, self.in_channel, kernel_size=kernel_size, padding=get_padding(kernel_size), groups=self.in_channel))

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        x = self.proj_first(x)

        x_chunks = list(torch.chunk(x, self.expand_ratio, dim=1))
        for i in range(self.expand_ratio):
            x_chunks[i] = getattr(self, f"attn_{self.kernel_list[i]}")(x_chunks[i]) * getattr(self, f"conv_{self.kernel_list[i]}")(x_chunks[i])

        x = torch.cat(x_chunks, dim=1)
        x = self.proj_last(x) * self.scale + shortcut
        return x

class Channel_Attention_Block(nn.Module):
    def __init__(self, in_channels: int = 64, dw_kernel_size: int = 3,
                 causal: bool = False, sca_kernel_size: int = 11):
        super().__init__()

        dw_channel = in_channels * 2

        if causal:
            conv_fn = CausalConv1d
        else:
            conv_fn = nn.Conv1d

        self.norm = LayerNorm1d(in_channels)
        self.pwconv1 = nn.Conv1d(in_channels=in_channels, out_channels=dw_channel, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)
        self.dwconv = conv_fn(in_channels=dw_channel, out_channels=dw_channel, kernel_size=dw_kernel_size,
                              padding=get_padding(dw_kernel_size), stride=1, groups=dw_channel, bias=True)
        self.sg = SimpleGate()
        if causal:
            # SCA (Simplified Channel Attention): causal depthwise conv for
            # streaming-friendly local pooling (replaces global average pooling)
            self.sca = nn.Sequential(
                CausalConv1d(
                    in_channels=dw_channel // 2,
                    out_channels=dw_channel // 2,
                    kernel_size=sca_kernel_size,
                    padding=get_padding(sca_kernel_size),
                    groups=dw_channel // 2,   # depthwise
                    bias=False,
                ),
                nn.Conv1d(
                    in_channels=dw_channel // 2,
                    out_channels=dw_channel // 2,
                    kernel_size=1, padding=0, stride=1,
                    groups=1, bias=True,
                ),
            )
        else:
            # Non-causal: original global average pooling
            self.sca = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Conv1d(
                    in_channels=dw_channel // 2,
                    out_channels=dw_channel // 2,
                    kernel_size=1, padding=0, stride=1,
                    groups=1, bias=True,
                ),
            )
        self.pwconv2 = nn.Conv1d(in_channels=dw_channel // 2, out_channels=in_channels, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)
        self.beta = nn.Parameter(torch.zeros((1, in_channels, 1)), requires_grad=True)

    def forward(self, x):
        skip = x
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.dwconv(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.pwconv2(x)
        x = skip + x * self.beta
        return x

class TS_BLOCK(nn.Module):
    def __init__(
        self,
        dense_channel: int = 64,
        time_block_num: int = 2,
        freq_block_num: int = 2,
        time_dw_kernel_size: int = 3,
        time_block_kernel: List[int] = [3, 5, 7, 9],
        freq_block_kernel: List[int] = [3, 11, 23, 31],
        causal: bool = False,
        sca_kernel_size: int = 11
    ):
        super().__init__()
        self.dense_channel = dense_channel

        time_stage = nn.ModuleList([])
        freq_stage = nn.ModuleList([])

        # Time stage: use causal parameter as provided
        for _ in range(time_block_num):
            time_stage.append(
                nn.Sequential(
                    Channel_Attention_Block(in_channels=dense_channel, dw_kernel_size=time_dw_kernel_size,
                                            causal=causal, sca_kernel_size=sca_kernel_size),
                    Group_Prime_Kernel_FFN(in_channel=dense_channel, kernel_list=time_block_kernel, causal=causal),
                )
            )

        # Frequency stage: always non-causal (frequency axis has no causal meaning)
        for _ in range(freq_block_num):
            freq_stage.append(
                nn.Sequential(
                    Channel_Attention_Block(in_channels=dense_channel, dw_kernel_size=3, causal=False),
                    Group_Prime_Kernel_FFN(in_channel=dense_channel, kernel_list=freq_block_kernel, causal=False),
                )
            )

        self.time_stage = nn.Sequential(*time_stage)
        self.freq_stage = nn.Sequential(*freq_stage)

        self.beta_t = nn.Parameter(torch.zeros((1, dense_channel, 1)), requires_grad=True)
        self.beta_f = nn.Parameter(torch.zeros((1, dense_channel, 1)), requires_grad=True)

    def forward(self, x):
        B, C, T, F = x.size()
        x = x.permute(0, 3, 1, 2).reshape(B * F, C, T)

        x = self.time_stage(x) + x * self.beta_t
        x = x.view(B, F, C, T).permute(0, 3, 2, 1).reshape(B * T, C, F)

        x = self.freq_stage(x) + x * self.beta_f
        x = x.view(B, T, C, F).permute(0, 2, 1, 3)
        return x

class LearnableSigmoid_2d(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features, 1))
        self.slope.requires_grad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)

class DS_DDB(nn.Module):
    """
    Dense Dilated Depthwise Block with asymmetric padding support.

    Args:
        dense_channel: Number of dense channels
        kernel_size: Convolution kernel size (time, freq)
        depth: Number of dilated layers (default: 4)
        causal: Deprecated parameter, kept for compatibility but ignored
        padding_ratio: (left_ratio, right_ratio) for asymmetric padding on time axis (required)
            - (1.0, 0.0): Fully causal
            - (0.5, 0.5): Symmetric
            - Custom ratios: Asymmetric (e.g., (0.8333, 0.1667) for R=5)
            - Must sum to 1.0
    """
    def __init__(self, dense_channel, kernel_size=(3, 3), depth=4, causal=False, padding_ratio=(0.5, 0.5), norm_type="batch"):
        super().__init__()
        self.dense_channel = dense_channel
        self.depth = depth
        self.dense_block = nn.ModuleList([])

        _validate_padding_ratio(padding_ratio)
        self.padding_ratio = padding_ratio

        # Always use AsymmetricConv2d with specified padding_ratio
        conv_fn = lambda in_ch, out_ch, ks, dil, pad, groups, bias: AsymmetricConv2d(
            in_ch, out_ch, ks, padding=pad, padding_ratio=self.padding_ratio,
            dilation=dil, groups=groups, bias=bias
        )

        for i in range(depth):
            dil = 2 ** i
            padding = get_padding_2d(kernel_size, (dil, 1))
            dense_conv = nn.Sequential(
                conv_fn(dense_channel*(i+1), dense_channel*(i+1), kernel_size,
                       dil=(dil, 1), pad=padding, groups=dense_channel*(i+1), bias=True),
                nn.Conv2d(in_channels=dense_channel*(i+1), out_channels=dense_channel,
                         kernel_size=1, padding=0, stride=1, groups=1, bias=True),
                _make_norm2d(dense_channel, norm_type),
                nn.PReLU(dense_channel)
            )
            self.dense_block.append(dense_conv)

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            x = self.dense_block[i](skip)
            skip = torch.cat([x, skip], dim=1)
        return x

class DenseEncoder(nn.Module):
    """
    Dense Encoder with asymmetric padding.

    Args:
        dense_channel: Number of dense channels
        in_channel: Number of input channels
        depth: Depth of DS_DDB (default: 4)
        causal: Deprecated parameter, kept for compatibility but ignored
        padding_ratio: (left_ratio, right_ratio) for asymmetric padding (required)
            - Must sum to 1.0
    """
    def __init__(self, dense_channel, in_channel, depth=4, causal=False, padding_ratio=(0.5, 0.5), norm_type="batch"):
        super().__init__()
        self.dense_channel = dense_channel
        self.dense_conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, dense_channel, (1, 1)),
            _make_norm2d(dense_channel, norm_type),
            nn.PReLU(dense_channel))
        self.dense_block = DS_DDB(dense_channel, depth=depth, causal=causal, padding_ratio=padding_ratio, norm_type=norm_type)
        self.dense_conv_2 = nn.Sequential(
            nn.Conv2d(dense_channel, dense_channel, (1, 3), (1, 2)),
            _make_norm2d(dense_channel, norm_type),
            nn.PReLU(dense_channel))

    def forward(self, x):
        x = self.dense_conv_1(x)
        x = self.dense_block(x)
        x = self.dense_conv_2(x)
        return x

class MaskDecoder(nn.Module):
    """
    Mask Decoder with asymmetric padding.

    Args:
        dense_channel: Number of dense channels
        n_fft: FFT size
        sigmoid_beta: Beta parameter for learnable sigmoid
        out_channel: Number of output channels (default: 1)
        depth: Depth of DS_DDB (default: 4)
        causal: Deprecated parameter, kept for compatibility but ignored
        padding_ratio: (left_ratio, right_ratio) for asymmetric padding (required)
            - Must sum to 1.0
    """
    def __init__(self,
                 dense_channel,
                 n_fft,
                 sigmoid_beta,
                 out_channel=1,
                 depth=4,
                 causal=False,
                 padding_ratio=(0.5, 0.5),
                 norm_type="batch"):
        super().__init__()
        self.n_fft = n_fft
        self.dense_block = DS_DDB(dense_channel, depth=depth, causal=causal, padding_ratio=padding_ratio, norm_type=norm_type)
        self.mask_conv = nn.Sequential(
            nn.ConvTranspose2d(dense_channel, dense_channel, (1, 3), (1, 2)),
            nn.Conv2d(dense_channel, out_channel, (1, 1)),
            _make_norm2d(out_channel, norm_type),
            nn.PReLU(out_channel),
            nn.Conv2d(out_channel, out_channel, (1, 1))
        )
        self.lsigmoid = LearnableSigmoid_2d(n_fft//2+1, beta=sigmoid_beta)

    def forward(self, x):
        x = self.dense_block(x)
        x = self.mask_conv(x)
        x = x.squeeze(1).transpose(1, 2)
        x = self.lsigmoid(x).transpose(1,2).unsqueeze(1)
        return x

class PhaseDecoder(nn.Module):
    """
    Phase Decoder with asymmetric padding.

    Args:
        dense_channel: Number of dense channels
        out_channel: Number of output channels (default: 1)
        depth: Depth of DS_DDB (default: 4)
        causal: Deprecated parameter, kept for compatibility but ignored
        padding_ratio: (left_ratio, right_ratio) for asymmetric padding (required)
            - Must sum to 1.0
    """
    def __init__(self,
                 dense_channel,
                 out_channel=1,
                 depth=4,
                 causal=False,
                 padding_ratio=(0.5, 0.5),
                 norm_type="batch"):
        super().__init__()
        self.dense_block = DS_DDB(dense_channel, depth=depth, causal=causal, padding_ratio=padding_ratio, norm_type=norm_type)
        self.phase_conv = nn.Sequential(
            nn.ConvTranspose2d(dense_channel, dense_channel, (1, 3), (1, 2)),
            _make_norm2d(dense_channel, norm_type),
            nn.PReLU(dense_channel)
        )
        self.phase_conv_r = nn.Conv2d(dense_channel, out_channel, (1, 1))
        self.phase_conv_i = nn.Conv2d(dense_channel, out_channel, (1, 1))

    def forward(self, x):
        x = self.dense_block(x)
        x = self.phase_conv(x)
        x_r = self.phase_conv_r(x)
        x_i = self.phase_conv_i(x)
        # Add epsilon to prevent gradient explosion in atan2 backward
        x = torch.atan2(x_i + 1e-8, x_r + 1e-8)
        return x

class Backbone(nn.Module):
    """
    Backbone with asymmetric padding support for latency control.

    Args:
        encoder_padding_ratio: (left_ratio, right_ratio) for encoder asymmetric padding (required)
            - (1.0, 0.0): Fully causal (12.5ms latency, center=True)
            - (0.8333, 0.1667): R=5, 75.0ms latency
            - (0.5, 0.5): Symmetric (200.0ms latency)
            - Must sum to 1.0
        decoder_padding_ratio: (left_ratio, right_ratio) for decoder asymmetric padding (required)
            - Recommended: (1.0, 0.0) for Option B (decoder causal)
            - Must sum to 1.0
        causal_ts_block: Controls TS_BLOCK causality (True for causal time processing)

    Example (Option B configuration):
        model = Backbone(
            ...,
            encoder_padding_ratio=(0.8333, 0.1667),  # R=5, 75.0ms latency
            decoder_padding_ratio=(1.0, 0.0),         # Decoder causal
            causal_ts_block=True,                     # TS_BLOCK causal
        )
    """
    def __init__(self,
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
                 norm_type="batch"
                 ):
        super().__init__()
        self.win_len = win_len
        self.hop_len = hop_len
        self.fft_len = fft_len
        self.dense_channel = dense_channel
        self.dense_depth = dense_depth
        self.sigmoid_beta = sigmoid_beta
        self.compress_factor = compress_factor
        self.num_tsblock = num_tsblock
        self.causal_ts_block = causal_ts_block

        _validate_padding_ratio(encoder_padding_ratio, "encoder_padding_ratio")
        _validate_padding_ratio(decoder_padding_ratio, "decoder_padding_ratio")

        self.encoder_padding_ratio = encoder_padding_ratio
        self.decoder_padding_ratio = decoder_padding_ratio

        self.dense_encoder = DenseEncoder(dense_channel, in_channel=2, depth=dense_depth,
                                         causal=causal_ts_block, padding_ratio=encoder_padding_ratio,
                                         norm_type=norm_type)
        self.sequence_block = nn.Sequential(
            *[TS_BLOCK(dense_channel, time_block_num, freq_block_num, time_dw_kernel_size,
                      time_block_kernel, freq_block_kernel, causal=causal_ts_block,
                      sca_kernel_size=sca_kernel_size) for _ in range(num_tsblock)]
        )
        self.mask_decoder = MaskDecoder(dense_channel, fft_len, sigmoid_beta, out_channel=1,
                                       depth=dense_depth, causal=causal_ts_block, padding_ratio=decoder_padding_ratio,
                                       norm_type=norm_type)
        self.phase_decoder = PhaseDecoder(dense_channel, out_channel=1, depth=dense_depth,
                                         causal=causal_ts_block, padding_ratio=decoder_padding_ratio,
                                         norm_type=norm_type)

    def forward(self, noisy_com):
        # Input shape: [B, F, T, 2]
        mag, pha = complex_to_mag_pha(noisy_com, stack_dim=-1)  # [B, F, T] each

        x = torch.stack((mag, pha), dim=1).permute(0, 1, 3, 2)  # [B, 2, T, F]
        x = self.dense_encoder(x)
        x = self.sequence_block(x)

        # mask_decoder output: [B, 1, T, F] -> squeeze/transpose -> [B, F, T]
        mask = self.mask_decoder(x).squeeze(1).transpose(1, 2)
        est_mag = mag * mask  # [B, F, T]

        est_pha = self.phase_decoder(x).squeeze(1).transpose(1, 2)  # [B, F, T]
        est_com = mag_pha_to_complex(est_mag, est_pha, stack_dim=-1)

        return est_mag, est_pha, est_com