"""
LaCoSENet with Lookahead Buffering.

This module provides a streaming wrapper that addresses the asymmetric
convolution problem by buffering features and providing real lookahead context.
It supports models where encoder, decoder, or both have asymmetric padding.

Processing pipeline:
1. Input buffering + STFT future buffering:
   - Accumulate samples until a full processing window is available, including
     win_size/2 extra samples for real future context.
   - Use center=False STFT with manually prepended past context (win_size/2)
     and appended future context (win_size/2) to eliminate reflect padding artifacts.
2. Encoder + TS_BLOCK:
   - Run the encoder path immediately once the processing window is ready.
   - StatefulConv provides past context via internal state buffers.
   - "Input lookahead" provides future context for asymmetric encoder padding.
3. Feature buffer (decoder lookahead):
   - Accumulate encoder features until enough frames are available for decoder
     processing with lookahead.
4. Decoder:
   - Process an extended time window (current chunk + lookahead + STFT frames)
     and produce only the current chunk output for iSTFT overlap-add.
5. StateFramesContext:
   - Prevent lookahead and STFT-induced extra frames from corrupting streaming
     states by limiting state updates to the current chunk frames.

This approach provides:
- Real past context from StatefulConv buffering
- Real future context from delayed processing (encoder/decoder lookahead)
- Minimal additional latency beyond the required lookahead frames

Example (decoder-only asymmetric):
    >>> streaming = LaCoSENet.from_checkpoint(
    ...     chkpt_dir="path/to/checkpoint",
    ...     chunk_size=64,
    ...     encoder_lookahead=0,   # Encoder is causal
    ...     decoder_lookahead=7,   # Decoder needs 7 frame lookahead
    ... )

Example (both encoder and decoder asymmetric):
    >>> streaming = LaCoSENet.from_checkpoint(
    ...     chkpt_dir="path/to/checkpoint",
    ...     chunk_size=64,
    ...     encoder_lookahead=7,   # Encoder needs 7 frame lookahead
    ...     decoder_lookahead=7,   # Decoder needs 7 frame lookahead
    ... )
    >>>
    >>> for chunk in audio_stream:
    ...     output = streaming.process_samples(chunk)
    ...     if output is not None:
    ...         play(output)
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor

from src.stft import manual_istft_ola

if TYPE_CHECKING:
    from src.models.streaming.layers.reshape_free_stateful import (
        StatefulReshapeFreeTSBlock,
    )

logger = logging.getLogger(__name__)


class LaCoSENet(nn.Module):
    """
    Streaming wrapper for Backbone with lookahead buffering.

    This wrapper enables true streaming inference for models with asymmetric
    convolutions in encoder, decoder, or both. It provides real future context
    through lookahead buffering instead of zero-padding.

    Supported model configurations:
    - **Fully Causal**: encoder_lookahead=0, decoder_lookahead=0
    - **Asymmetric Decoder Only**: encoder_lookahead=0, decoder_lookahead>0
      Example: decoder_padding_ratio=(0.77, 0.23)
    - **Asymmetric Encoder+Decoder**: encoder_lookahead>0, decoder_lookahead>0
      Example: encoder/decoder padding_ratio=(0.77, 0.23)

    Processing pipeline:

    1. **Input buffering + STFT future buffering**:
       - Accumulates input samples until a full processing window is available,
         including win_size/2 extra samples for real future context.
       - Uses center=False STFT with manually prepended past context (win_size/2)
         and appended future context (win_size/2) to eliminate reflect padding.

    2. **Encoder Path**:
       - DenseEncoder + TS_BLOCK with StatefulConv
       - StatefulConv provides past context (left padding replacement)
       - Input lookahead provides future context (right padding replacement for
         asymmetric encoder padding)

    3. **Feature Buffer** (for decoder lookahead):
       - Stores encoded features until decoder_lookahead frames available
       - Bridges encoder output to decoder input with proper context

    4. **Decoder Path** (delayed processing):
       - Processes extended input: current + lookahead frames
       - Uses StateFramesContext to prevent state corruption from lookahead
       - Outputs only current chunk portion (lookahead frames discarded)

    5. **Cross-chunk OLA reconstruction**:
       - Manual iSTFT with cross-chunk overlap-add buffer
       - Decoder outputs chunk_size frames; OLA buffer carries over the tail
         (win_size - hop_size samples) between chunks for seamless reconstruction

    The total latency is:
        total_lookahead_frames = encoder_lookahead + decoder_lookahead
        stft_future_delay_samples = win_size / 2   (STFT future buffering for real context)
        latency_samples = total_lookahead_frames * hop_size + stft_future_delay_samples
        latency_seconds = latency_samples / sample_rate

    Attributes:
        encoder_lookahead: Frames needed for encoder (0 if encoder is causal)
        decoder_lookahead: Frames needed for decoder (0 if decoder is causal)
        total_lookahead: encoder_lookahead + decoder_lookahead
        latency_ms: Total latency in milliseconds
    """

    def __init__(
        self,
        model: nn.Module,
        chunk_size: int = 64,
        encoder_lookahead: int = 0,
        decoder_lookahead: int = 7,
        hop_size: int = 100,
        n_fft: int = 400,
        win_size: int = 400,
        compress_factor: float = 0.3,
        sample_rate: int = 16000,
        rf_sequence_block: Optional[nn.ModuleList] = None,
        freq_size: int = 100,
        disable_state_guard: bool = False,
    ):
        """
        Initialize LaCoSENet.

        Note: Use `from_checkpoint()` for easier initialization.

        Args:
            model: Backbone model (should already have StatefulConv applied)
            chunk_size: Number of STFT frames per chunk
            encoder_lookahead: Frames to delay encoder processing (for asymmetric encoder)
            decoder_lookahead: Frames to delay decoder processing (for asymmetric decoder)
            hop_size: STFT hop size in samples
            n_fft: FFT size
            win_size: Window size
            compress_factor: Magnitude compression factor
            sample_rate: Audio sample rate in Hz
            rf_sequence_block: Reshape-free TS_BLOCK ModuleList (if enabled)
            freq_size: Frequency bins (for state initialization)
            disable_state_guard: If True, disable StateFramesContext so all
                frames (including lookahead) update streaming state buffers.
                Used for ablation study of selective state update (C3).
        """
        super().__init__()

        # Ablation: disable selective state update
        self.disable_state_guard = disable_state_guard

        # Store model reference
        self.model = model
        self.model.eval()

        # STFT parameters
        self.hop_size = hop_size
        self.n_fft = n_fft
        self.win_size = win_size
        self.compress_factor = compress_factor
        self.sample_rate = sample_rate
        self.output_frequency_bins = self.n_fft // 2 + 1
        self.internal_frequency_bins = (
            self.output_frequency_bins + 1
            if getattr(self.model, "pad_even_frequency", False)
            and self.output_frequency_bins % 2 == 0
            else self.output_frequency_bins
        )
        # Streaming emulates center=True via center=False + manual context buffers
        # (win_size/2 past + win_size/2 future), adding stft_center_delay.
        self.stft_future_samples = self.win_size // 2
        self.stft_center_delay_samples = self.stft_future_samples

        # Streaming parameters
        self.chunk_size = chunk_size
        self.encoder_lookahead = encoder_lookahead
        self.decoder_lookahead = decoder_lookahead
        self.input_lookahead_frames = int(encoder_lookahead)
        self.total_lookahead = self.input_lookahead_frames + decoder_lookahead

        self.total_frames_needed = chunk_size + self.input_lookahead_frames
        # chunk includes stft_future_samples (W/2) for STFT right context.
        # Full STFT input = [past_context(W/2) | chunk_samples(N + W/2)]
        self.samples_per_chunk = (self.total_frames_needed - 1) * hop_size + self.stft_future_samples

        # Output step is ALWAYS chunk_size frames (i.e., chunk_size * hop_size samples).
        # This must match how we slide the input buffer; otherwise, time alignment drifts.
        self.output_frames_per_chunk = chunk_size
        self.output_samples_per_chunk = self.output_frames_per_chunk * hop_size

        # OLA buffer parameters: carry-over tail for cross-chunk overlap-add
        self.ola_tail_size = win_size - hop_size  # 300 for win=400, hop=100

        # Latency calculation
        self.latency_samples = self.total_lookahead * hop_size + self.stft_center_delay_samples
        self.latency_ms = self.latency_samples / sample_rate * 1000

        # Reshape-free TS_BLOCK support
        self.rf_sequence_block = rf_sequence_block
        self.use_reshape_free = rf_sequence_block is not None
        self.freq_size = freq_size
        self._rf_states: Optional[List[List[Dict[str, Tensor]]]] = None

        # Initialize buffers
        self._reset_buffers()

        # Config storage
        self._streaming_config: Dict[str, Any] = {}

    def _reset_buffers(self) -> None:
        """Reset all internal buffers."""
        self.input_buffer = torch.tensor([], dtype=torch.float32)
        self.feature_buffer: List[Dict[str, Any]] = []  # Encoded features for decoder lookahead
        self._buffered_frames = 0
        # STFT past context buffer for center emulation
        self._stft_context = torch.zeros(self.win_size // 2, dtype=torch.float32)

        # OLA buffer for cross-chunk overlap-add
        self._ola_buffer = torch.zeros(self.ola_tail_size, dtype=torch.float32)
        self._ola_norm = torch.zeros(self.ola_tail_size, dtype=torch.float32)

        # Initialize reshape-free states if enabled
        if self.use_reshape_free and self.rf_sequence_block is not None:
            self._rf_states = self._init_rf_states()

    def _init_rf_states(self) -> List[List[Dict[str, Tensor]]]:
        """
        Initialize reshape-free TS_BLOCK states.

        Returns:
            List of states for each TS_BLOCK, where each state is a list of
            block states (one per time_stage block).
        """
        if self.rf_sequence_block is None:
            return []

        device = self.device
        dtype = next(self.model.parameters()).dtype

        all_states = []
        for rf_block in self.rf_sequence_block:
            block_states = rf_block.init_state(
                batch_size=1,
                freq_size=self.freq_size,
                device=device,
                dtype=dtype,
            )
            all_states.append(block_states)

        return all_states

    def _reset_rf_states(self) -> None:
        """Reset reshape-free TS_BLOCK states."""
        if self.use_reshape_free:
            self._rf_states = self._init_rf_states()

    def reset_state(self) -> None:
        """
        Reset all streaming state for a new audio stream.

        This resets:
        1. Audio input buffers
        2. Feature buffer for decoder lookahead
        3. Stateful convolution state buffers (for encoder/decoder)
        4. Reshape-free TS_BLOCK states (if enabled)
        """
        self._reset_buffers()

        # Reset reshape-free states if enabled
        if self.use_reshape_free:
            self._reset_rf_states()

        # Reset stateful convolution state (for encoder/decoder)
        from src.models.streaming.converters.conv_converter import (
            reset_streaming_state,
        )
        reset_streaming_state(self.model)

    @property
    def device(self) -> torch.device:
        """Get the device of the underlying model."""
        return next(self.model.parameters()).device

    @property
    def streaming_config(self) -> Dict[str, Any]:
        """Get streaming configuration information."""
        return {
            **self._streaming_config,
            "chunk_size_frames": self.chunk_size,
            "encoder_lookahead": self.encoder_lookahead,
            "decoder_lookahead": self.decoder_lookahead,
            "input_lookahead_frames": self.input_lookahead_frames,
            "total_lookahead": self.total_lookahead,
            "stft_center_delay_samples": self.stft_center_delay_samples,
            "output_frames_per_chunk": self.output_frames_per_chunk,
            "samples_per_chunk": self.samples_per_chunk,
            "output_samples_per_chunk": self.output_samples_per_chunk,
            "latency_samples": self.latency_samples,
            "latency_ms": self.latency_ms,
            "hop_size": self.hop_size,
            "sample_rate": self.sample_rate,
            "use_reshape_free": self.use_reshape_free,
            "freq_size": self.freq_size,
        }

    @classmethod
    def from_checkpoint(
        cls,
        chkpt_dir: str,
        chkpt_file: str = "best.th",
        chunk_size: int = 64,
        encoder_lookahead: Optional[int] = None,
        decoder_lookahead: Optional[int] = None,
        use_reshape_free: bool = False,
        fold_bn: bool = False,
        device: Optional[str] = None,
        verbose: bool = True,
        disable_state_guard: bool = False,
        latency_id: Optional[str] = None,
        sample_rate: Optional[int] = None,
    ) -> "LaCoSENet":
        """
        Create LaCoSENet from a checkpoint directory.

        This automatically:
        1. Loads the model configuration and weights
        2. Converts convolutions to stateful versions
        3. Optionally converts TS_BLOCKs to reshape-free versions
        4. Reads encoder/decoder padding ratios from the model config for
           visibility/debugging (lookahead values are provided by the caller).

        Args:
            chkpt_dir: Path to checkpoint directory
            chkpt_file: Checkpoint file name (default: "best.th")
            chunk_size: Number of STFT frames per chunk
            encoder_lookahead: Frames for encoder lookahead. Set to >0 if encoder
                has asymmetric padding (e.g., padding_ratio=(0.77, 0.23)).
                Set to 0 if encoder is fully causal (padding_ratio=(1.0, 0.0)).
            decoder_lookahead: Frames for decoder lookahead. Set to >0 if decoder
                has asymmetric padding. Set to 0 if decoder is fully causal.
            use_reshape_free: Convert TS_BLOCKs to reshape-free versions.
                Benefits:
                - Eliminates 16 reshape operations per inference
                - Unifies batch dimension to B=1 for all states
                - Reduces state count by ~48% (100 -> 52)
                - Recommended for NPU deployment
            fold_bn: Apply BN folding for CPU inference.
            device: Device to load model on
            verbose: Print loading information
            disable_state_guard: Disable selective state update (ablation).
            latency_id: Latency branch to use for latency-expert checkpoints.
                A streaming wrapper is fixed to one latency id for its lifetime.
            sample_rate: Fixed sampling rate for this streaming instance.  SFI
                checkpoints derive their 40/20 ms STFT sizes from this value.

        Returns:
            LaCoSENet instance

        Example configurations:
            - Decoder-only asymmetric: encoder_lookahead=0, decoder_lookahead=7
            - Both asymmetric: encoder_lookahead=7, decoder_lookahead=7
            - With reshape-free: use_reshape_free=True (recommended for mobile/NPU)
        """
        from src.models.streaming.utils import prepare_streaming_model

        if verbose:
            print(f"Loading LaCoSENet from: {chkpt_dir}")

        config_path = os.path.join(chkpt_dir, ".hydra", "config.yaml")
        if os.path.exists(config_path) and (use_reshape_free or fold_bn):
            from omegaconf import OmegaConf

            conf = OmegaConf.load(config_path)
            if conf.model.get("type", "backbone") == "latency_expert_backbone":
                unsupported = []
                if use_reshape_free:
                    unsupported.append("reshape-free TS_BLOCK conversion")
                if fold_bn:
                    unsupported.append("BN folding")
                raise NotImplementedError(
                    "LatencyExpertBackbone streaming does not support "
                    + " or ".join(unsupported)
                    + " in v1."
                )

        if latency_id is not None and use_reshape_free:
            raise NotImplementedError(
                "LatencyExpertBackbone streaming does not support reshape-free "
                "TS_BLOCK conversion in v1."
            )
        if latency_id is not None and fold_bn:
            raise NotImplementedError(
                "LatencyExpertBackbone streaming does not support BN folding in v1."
            )

        # Use unified model preparation pipeline
        model, metadata = prepare_streaming_model(
            chkpt_dir=chkpt_dir,
            chkpt_file=chkpt_file,
            use_stateful_conv=True,  # Always use stateful conv for buffered streaming
            use_reshape_free=use_reshape_free,
            fold_bn=fold_bn,
            device=device,
            verbose=verbose,
        )

        # Extract model args from metadata
        model_args = metadata["model_args"]
        model_params = model_args
        stateful_conv_count = metadata.get("stateful_conv_count", 0)
        rf_sequence_block = metadata.get("rf_sequence_block", None)
        rf_block_count = metadata.get("rf_block_count", 0)

        is_latency_expert = getattr(model, "is_latency_expert", False)
        if is_latency_expert:
            if use_reshape_free:
                raise NotImplementedError(
                    "LatencyExpertBackbone streaming does not support reshape-free "
                    "TS_BLOCK conversion in v1."
                )
            if fold_bn:
                raise NotImplementedError(
                    "LatencyExpertBackbone streaming does not support BN folding in v1."
                )
            latency_ids = model.get_latency_ids()
            if latency_id is None:
                latency_id = getattr(model, "default_latency_id", latency_ids[0])
            if latency_id not in latency_ids:
                raise ValueError(
                    f"Unknown latency_id '{latency_id}'. Expected one of {latency_ids}."
                )
            model.set_latency_id(latency_id)

        # Get padding ratios from model config
        if is_latency_expert:
            ratio_info = model.get_padding_ratio(latency_id)
            enc_padding = ratio_info["encoder_padding_ratio"]
            dec_padding = ratio_info["decoder_padding_ratio"]
        else:
            enc_padding = getattr(model_params, 'encoder_padding_ratio', [1.0, 0.0])
            dec_padding = getattr(model_params, 'decoder_padding_ratio', [1.0, 0.0])

        from src.utils import compute_lookahead_frames

        if encoder_lookahead is None:
            encoder_lookahead = compute_lookahead_frames(
                enc_padding, getattr(model, "dense_depth", 4)
            )
        if decoder_lookahead is None:
            if is_latency_expert and hasattr(model, "get_future_frames"):
                decoder_lookahead = model.get_future_frames(latency_id)
            else:
                decoder_lookahead = compute_lookahead_frames(
                    dec_padding, getattr(model, "dense_depth", 4)
                )

        if verbose:
            if is_latency_expert:
                print(f"  Latency id: {latency_id}")
            print(f"  Encoder padding ratio: {enc_padding}")
            print(f"  Decoder padding ratio: {dec_padding}")

        def get_model_param(*names: str, default: Any) -> Any:
            for name in names:
                if hasattr(model_params, "get"):
                    value = model_params.get(name, None)
                else:
                    value = getattr(model_params, name, None)
                if value is not None:
                    return value
            return default

        # Get STFT parameters.  A v2 SFI streaming instance is fixed to one
        # sample rate for its entire lifetime so all state shapes stay stable.
        sfi_config = get_model_param("sfi", default={}) or {}
        active_sample_rate = int(
            sample_rate
            if sample_rate is not None
            else sfi_config.get("reference_sample_rate", 16000)
        )
        if sfi_config.get("enabled", False):
            from src.v2_contract import sfi_profile_from_config

            profile = sfi_profile_from_config(sfi_config, active_sample_rate)
            hop_size = profile.hop_len
            n_fft = profile.fft_len
            win_size = profile.win_len
        else:
            hop_size = get_model_param("hop_len", "hop_size", default=100)
            n_fft = get_model_param("fft_len", "n_fft", default=400)
            win_size = get_model_param("win_len", "win_size", default=400)
        compress_factor = get_model_param("compress_factor", default=0.3)

        # Calculate freq_size from actual encoder output (not STFT bins)
        # DenseEncoder's dense_conv_2 has stride=(1,2), halving freq dimension
        stft_freq = n_fft // 2 + 1
        internal_stft_freq = (
            stft_freq + 1
            if getattr(model, "pad_even_frequency", False) and stft_freq % 2 == 0
            else stft_freq
        )
        with torch.no_grad():
            dummy = torch.randn(
                1,
                2,
                4,
                internal_stft_freq,
                device=next(model.parameters()).device,
            )
            freq_size = model.dense_encoder(dummy).shape[3]

        # Create instance
        instance = cls(
            model=model,
            chunk_size=chunk_size,
            encoder_lookahead=encoder_lookahead,
            decoder_lookahead=decoder_lookahead,
            hop_size=hop_size,
            n_fft=n_fft,
            win_size=win_size,
            compress_factor=compress_factor,
            sample_rate=active_sample_rate,
            rf_sequence_block=rf_sequence_block,
            freq_size=freq_size,
            disable_state_guard=disable_state_guard,
        )

        # Store config
        instance._streaming_config = {
            "chkpt_dir": chkpt_dir,
            "use_reshape_free": use_reshape_free,
            "encoder_padding_ratio": enc_padding,
            "decoder_padding_ratio": dec_padding,
            "stateful_conv_count": stateful_conv_count,
            "rf_block_count": rf_block_count,
            "model_class": model.__class__.__name__,
            "latency_id": latency_id,
            "sample_rate": active_sample_rate,
        }

        # The dummy forward above runs through stateful encoder layers after model
        # conversion. Return a clean wrapper state for the first real stream.
        instance.reset_state()

        if verbose:
            print(f"  Chunk size: {chunk_size} frames")
            print(f"  Encoder lookahead: {encoder_lookahead} frames")
            print(f"  Decoder lookahead: {decoder_lookahead} frames")
            print(f"  Total latency: {instance.latency_ms:.1f}ms")
            if use_reshape_free:
                print(f"  Reshape-Free: {rf_block_count} TS_BLOCKs converted")

        return instance

    def _stft(self, audio: Tensor) -> Tensor:
        """Compute STFT (center=False) and return compressed complex spectrogram.

        Streaming uses center=False internally; center=True emulation is
        handled by process_samples which prepends past context.
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        hann_window = torch.hann_window(self.win_size).to(audio.device)
        stft_spec = torch.stft(
            audio, self.n_fft, hop_length=self.hop_size, win_length=self.win_size,
            window=hann_window, center=False, pad_mode='reflect',
            normalized=False, return_complex=True,
        )
        stft_spec = torch.view_as_real(stft_spec)
        mag = torch.sqrt(stft_spec.pow(2).sum(-1) + 1e-9)
        pha = torch.atan2(stft_spec[:, :, :, 1] + 1e-8, stft_spec[:, :, :, 0] + 1e-8)
        mag = torch.pow(mag, self.compress_factor)
        com = torch.stack((mag * torch.cos(pha), mag * torch.sin(pha)), dim=-1)

        return com

    def _process_encoder(
        self,
        spectrogram: Tensor,
    ) -> Tuple[Tensor, Tensor, int]:
        """
        Process spectrogram through Encoder + TS_BLOCK.

        This is the common encoder path for both immediate and buffered modes.
        When reshape-free is enabled, uses StatefulReshapeFreeTSBlock instead
        of the original TS_BLOCK with unified batch dimension.

        Args:
            spectrogram: Complex spectrogram [B, F, T, 2]

        Returns:
            Tuple of (mag, ts_out, valid_frames):
            - mag: Magnitude spectrogram [B, F, T]
            - ts_out: TS_BLOCK output [B, C, T, F]
            - valid_frames: Number of frames used for state update
        """
        from src.models.backbone import complex_to_mag_pha
        from src.models.streaming.utils import StateFramesContext

        if spectrogram.shape[1] != self.output_frequency_bins:
            raise ValueError(
                f"Streaming STFT produced {spectrogram.shape[1]} bins; "
                f"expected {self.output_frequency_bins} for n_fft={self.n_fft}"
            )
        if self.internal_frequency_bins != self.output_frequency_bins:
            pad = spectrogram.new_zeros(
                spectrogram.shape[0], 1, spectrogram.shape[2], spectrogram.shape[3]
            )
            spectrogram = torch.cat((spectrogram, pad), dim=1)

        _, _, T, _ = spectrogram.shape

        mag, pha = complex_to_mag_pha(spectrogram, stack_dim=-1)
        x = torch.stack((mag, pha), dim=1).permute(0, 1, 3, 2)  # [B, 2, T, F]

        # Restrict state updates to chunk_size frames.
        # T = total_frames_needed = chunk_size + input_lookahead_frames.
        # Only the first chunk_size frames should update streaming state;
        # the remaining input_lookahead frames provide future context only.
        valid_frames = min(T, self.chunk_size)

        with StateFramesContext(None if self.disable_state_guard else valid_frames):
            # Encoder (always use original model's encoder)
            encoded = self.model.dense_encoder(x)

            # TS_BLOCK: use reshape-free version if enabled
            if self.use_reshape_free and self.rf_sequence_block is not None:
                ts_out = self._process_rf_sequence_block(encoded)
            else:
                ts_out = self.model.sequence_block(encoded)  # [B, C, T, F]

        return mag, ts_out, valid_frames

    def _process_rf_sequence_block(self, x: Tensor) -> Tensor:
        """
        Process through reshape-free TS_BLOCKs with explicit state management.

        This uses StatefulReshapeFreeTSBlock which:
        - Eliminates reshape operations (no permute + contiguous)
        - Uses unified batch dimension (B=1 for all states)
        - freq_stage is stateless (no streaming state needed)

        Args:
            x: Encoder output [B, C, T, F]

        Returns:
            TS_BLOCK output [B, C, T, F]
        """
        if self.rf_sequence_block is None or self._rf_states is None:
            raise RuntimeError("Reshape-free sequence block not initialized")

        from src.models.streaming.utils import get_state_frames_context

        state_frames = get_state_frames_context()

        # Process through each TS_BLOCK with state
        for i, rf_block in enumerate(self.rf_sequence_block):
            x, new_states = rf_block(x, self._rf_states[i], state_frames=state_frames)
            self._rf_states[i] = new_states

        return x

    def _can_process_immediately(self, ts_out: Tensor) -> bool:
        """
        Determine if decoder can process immediately without buffering.

        Conditions for immediate processing:
        1. decoder_lookahead == 0
        2. encoder output frames >= chunk_size

        Args:
            ts_out: Encoder + TS_BLOCK output [B, C, T, F]

        Returns:
            True if immediate processing is possible, False otherwise.
        """
        if self.decoder_lookahead > 0:
            return False

        return ts_out.shape[2] >= self.chunk_size

    def _process_decoder_immediate(
        self,
        ts_out: Tensor,
        mag: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Process decoder immediately without feature buffer (decoder_lookahead=0).

        Only chunk_size frames are processed. Cross-chunk OLA handles
        overlap-add reconstruction without extra frames.

        Args:
            ts_out: Encoder + TS_BLOCK output [B, C, T, F]
            mag: Magnitude spectrogram [B, F, T]

        Returns:
            Tuple of (est_mag, est_pha) with T=chunk_size for OLA
        """
        from src.models.streaming.utils import StateFramesContext

        features = ts_out[:, :, :self.chunk_size, :]
        chunk_mag = mag[:, :, :self.chunk_size]

        with StateFramesContext(None if self.disable_state_guard else self.chunk_size):
            mask = self.model.mask_decoder(features).squeeze(1).transpose(1, 2)
            est_pha = self.model.phase_decoder(features).squeeze(1).transpose(1, 2)

        mask = mask[:, :self.output_frequency_bins, :]
        est_pha = est_pha[:, :self.output_frequency_bins, :]
        chunk_mag = chunk_mag[:, :self.output_frequency_bins, :]

        infer_type = getattr(self.model, 'infer_type', 'masking')
        if infer_type == 'masking':
            est_mag = chunk_mag * mask
        else:
            est_mag = mask

        return est_mag, est_pha

    def _process_decoder_buffered(
        self,
        ts_out: Tensor,
        mag: Tensor,
        valid_frames: int,
    ) -> Optional[Tuple[Tensor, Tensor]]:
        """
        Process decoder with feature buffering (decoder_lookahead > 0).

        This path accumulates encoded features in the feature buffer until
        enough frames are available for decoder processing with lookahead.

        Args:
            ts_out: Encoder + TS_BLOCK output [B, C, T, F]
            mag: Magnitude spectrogram [B, F, T]
            valid_frames: Number of frames used for encoder state update

        Returns:
            Tuple of (est_mag, est_pha) if output available, None otherwise
        """
        from src.models.streaming.utils import StateFramesContext

        # Step 1: Add to feature buffer
        # IMPORTANT: Only store valid_frames, not all T frames from STFT
        self.feature_buffer.append({
            'features': ts_out[:, :, :valid_frames, :],
            'mag': mag[:, :, :valid_frames],
            'frames': valid_frames,
        })
        self._buffered_frames += valid_frames

        # Step 2: Check if we have enough for decoder processing
        total_needed = self.chunk_size + self.decoder_lookahead
        if self._buffered_frames < total_needed:
            logger.debug(f"Buffering: {self._buffered_frames}/{total_needed}")
            return None

        # Step 3: Gather features for decoder processing
        all_features = torch.cat([buf['features'] for buf in self.feature_buffer], dim=2)
        all_mag = torch.cat([buf['mag'] for buf in self.feature_buffer], dim=2)

        extended_features = all_features[:, :, :total_needed, :]
        extended_mag = all_mag[:, :, :total_needed]

        # Step 4: Process decoder with extended input
        with StateFramesContext(None if self.disable_state_guard else self.chunk_size):
            mask = self.model.mask_decoder(extended_features).squeeze(1).transpose(1, 2)
            est_pha = self.model.phase_decoder(extended_features).squeeze(1).transpose(1, 2)

        mask = mask[:, :self.output_frequency_bins, :]
        est_pha = est_pha[:, :self.output_frequency_bins, :]
        extended_mag = extended_mag[:, :self.output_frequency_bins, :]

        # Apply mask
        infer_type = getattr(self.model, 'infer_type', 'masking')
        if infer_type == 'masking':
            est_mag = extended_mag * mask
        else:
            est_mag = mask

        # Step 5: Trim to chunk_size frames for OLA reconstruction
        est_mag = est_mag[:, :, :self.chunk_size]
        est_pha = est_pha[:, :, :self.chunk_size]

        # Step 6: Update feature buffer
        frames_to_remove = self.chunk_size
        removed = 0

        while removed < frames_to_remove and self.feature_buffer:
            buf = self.feature_buffer[0]
            if buf['frames'] <= (frames_to_remove - removed):
                removed += buf['frames']
                self.feature_buffer.pop(0)
            else:
                keep_frames = buf['frames'] - (frames_to_remove - removed)
                buf['features'] = buf['features'][:, :, -keep_frames:, :]
                buf['mag'] = buf['mag'][:, :, -keep_frames:]
                buf['frames'] = keep_frames
                removed = frames_to_remove

        self._buffered_frames -= removed

        return est_mag, est_pha

    def process_spectrogram_buffered(
        self,
        spectrogram: Tensor,
    ) -> Optional[Tuple[Tensor, Tensor]]:
        """
        Process spectrogram using adaptive encoder-decoder approach.

        Processing flow:
        1. Encoder + TS_BLOCK: Process immediately (common path)
        2. Decoder path: Conditional based on decoder_lookahead
           - If decoder_lookahead=0 AND enough frames: Immediate processing
           - Otherwise: Buffer features and wait for lookahead

        Args:
            spectrogram: Complex spectrogram [B, F, T, 2] where T = chunk_size

        Returns:
            Tuple of (est_mag, est_pha) if output available, None otherwise
            Output time dimension is chunk_size (for cross-chunk OLA)
        """
        # Step 1: Encoder + TS_BLOCK (common path)
        mag, ts_out, valid_frames = self._process_encoder(spectrogram)

        # Step 2: Decoder path (conditional)
        if self._can_process_immediately(ts_out):
            # Immediate mode: Use encoder output directly for decoder
            return self._process_decoder_immediate(ts_out, mag)
        else:
            # Buffered mode: Accumulate features and wait for lookahead
            return self._process_decoder_buffered(ts_out, mag, valid_frames)

    def _manual_istft_ola(self, est_mag: Tensor, est_pha: Tensor) -> Tensor:
        """
        Perform manual iSTFT with cross-chunk OLA buffer.

        Uses the OLA buffer state (_ola_buffer, _ola_norm) to carry over
        overlap-add tail between chunks. Returns exactly output_samples_per_chunk
        mature samples per call.

        Args:
            est_mag: [B, F, T] estimated magnitude (compressed)
            est_pha: [B, F, T] estimated phase

        Returns:
            [output_samples_per_chunk] mature audio samples
        """
        # Ensure OLA buffers are on the correct device
        if self._ola_buffer.device != est_mag.device:
            self._ola_buffer = self._ola_buffer.to(est_mag.device)
            self._ola_norm = self._ola_norm.to(est_mag.device)

        output, new_ola_buffer, new_ola_norm = manual_istft_ola(
            est_mag, est_pha,
            n_fft=self.n_fft,
            hop_size=self.hop_size,
            win_size=self.win_size,
            compress_factor=self.compress_factor,
            ola_buffer=self._ola_buffer,
            ola_norm=self._ola_norm,
        )

        # Update carry-over state
        self._ola_buffer = new_ola_buffer
        self._ola_norm = new_ola_norm

        return output[:self.output_samples_per_chunk]

    @torch.inference_mode()
    def process_samples(self, samples: Tensor) -> Optional[Tensor]:
        """
        Process incoming audio samples and return enhanced output if available.

        Args:
            samples: Input audio samples [T] or [B, T] (B must be 1)

        Returns:
            Enhanced audio samples [T] if output available, None otherwise
        """
        if samples.dim() == 2:
            if samples.shape[0] != 1:
                raise ValueError("Batch size must be 1 for streaming")
            samples = samples.squeeze(0)

        samples = samples.to(self.device)

        # Ensure STFT context is on the same device (needed on first call after reset)
        if self._stft_context.device != samples.device:
            self._stft_context = self._stft_context.to(samples.device)

        # Add to buffer
        self.input_buffer = torch.cat([self.input_buffer.to(self.device), samples])

        # Check if we have enough samples for one chunk
        if len(self.input_buffer) < self.samples_per_chunk:
            return None

        # Extract chunk for processing
        chunk_samples = self.input_buffer[:self.samples_per_chunk]

        # Prepend real past context for STFT center emulation.
        context_size = self.win_size // 2
        chunk_with_context = torch.cat([self._stft_context.to(chunk_samples.device), chunk_samples])
        spectrogram = self._stft(chunk_with_context)

        # Save context for next chunk
        advance = self.output_samples_per_chunk
        if advance >= context_size:
            self._stft_context = self.input_buffer[advance - context_size:advance].clone()
        else:
            need_from_prev = context_size - advance
            prev_part = self._stft_context[len(self._stft_context) - need_from_prev:]
            curr_part = self.input_buffer[:advance]
            self._stft_context = torch.cat([prev_part, curr_part]).clone()

        # Process through buffered model
        result = self.process_spectrogram_buffered(spectrogram)

        if result is None:
            # Still buffering for decoder lookahead
            # Advance input buffer to prepare for next chunk (by output size)
            self.input_buffer = self.input_buffer[self.output_samples_per_chunk:]
            return None

        est_mag, est_pha = result

        # Reconstruct audio via cross-chunk OLA
        valid_output = self._manual_istft_ola(est_mag, est_pha)

        # Update buffer
        self.input_buffer = self.input_buffer[self.output_samples_per_chunk:]

        return valid_output

    def process_audio(self, audio: Tensor) -> Tensor:
        """
        Process a complete audio signal in streaming fashion.

        Feeds audio through the per-chunk streaming pipeline (stateful conv +
        lookahead buffering + per-chunk iSTFT) and concatenates outputs.
        Silence padding is appended to flush all buffered data.

        Args:
            audio: Complete input audio [T] or [1, T]

        Returns:
            Enhanced audio [T'] where T' <= T
        """
        if audio.dim() == 2:
            audio = audio.squeeze(0)

        audio_length = len(audio)
        self.reset_state()
        outputs: List[Tensor] = []

        # Append silence to flush the pipeline
        flush_size = self.samples_per_chunk * (self.total_lookahead + 2)
        padded = torch.cat([
            audio,
            torch.zeros(flush_size, device=audio.device),
        ])

        # Feed in output-sized increments
        for i in range(0, len(padded), self.output_samples_per_chunk):
            chunk = padded[i:i + self.output_samples_per_chunk]
            if len(chunk) == 0:
                break
            result = self.process_samples(chunk)
            if result is not None and len(result) > 0:
                outputs.append(result)

        if not outputs:
            return torch.tensor([], device=audio.device)

        result = torch.cat(outputs)
        if len(result) > audio_length:
            result = result[:audio_length]
        return result

    def process_audio_fast(self, audio: Tensor) -> Tensor:
        """
        Process a complete audio signal using the 3-phase fast pipeline.

        Eliminates per-chunk STFT/iSTFT overhead by batching them.  The model
        forward pass (Phase 2) remains sequential because StatefulConv layers
        are state-dependent.

        Phase 1 — Batched STFT: replicate per-chunk context+samples slicing,
                  then one batched mag_pha_stft call.
        Phase 2 — Sequential process_spectrogram_buffered() calls (tight loop).
        Phase 3 — Batch iSTFT via manual_istft_ola() (single call).

        Produces bit-identical output to process_audio().

        Args:
            audio: Complete input audio [T] or [1, T]

        Returns:
            Enhanced audio [T'] where T' <= T
        """
        if audio.dim() == 2:
            audio = audio.squeeze(0)

        audio_length = len(audio)
        device = self.device
        audio = audio.to(device)

        # Reset all streaming state
        self.reset_state()

        # --- Phase 1: Batched STFT ---
        # Flush padding identical to process_audio()
        flush_size = self.samples_per_chunk * (self.total_lookahead + 2)
        padded = torch.cat([audio, torch.zeros(flush_size, device=device)])

        osp = self.output_samples_per_chunk
        # Prepend context_size zeros to replicate streaming context
        context_size = self.win_size // 2
        pre_padded = torch.cat([
            torch.zeros(context_size, device=device),
            padded,
        ])
        stft_input_len = context_size + self.samples_per_chunk

        # Number of spectrogram calls (same count as process_samples processing calls)
        n_calls = (len(pre_padded) - stft_input_len) // osp + 1

        # Extract overlapping windows using unfold (efficient, no copy)
        batch_input = pre_padded.unfold(0, stft_input_len, osp)  # [N, stft_input_len]
        batch_input = batch_input[:n_calls]

        # Single batched STFT (center=False: past/future context already in windows)
        hann_window = torch.hann_window(self.win_size).to(device)
        stft_spec = torch.stft(
            batch_input, self.n_fft, hop_length=self.hop_size, win_length=self.win_size,
            window=hann_window, center=False, pad_mode='reflect',
            normalized=False, return_complex=True,
        )
        stft_spec = torch.view_as_real(stft_spec)
        mag = torch.sqrt(stft_spec.pow(2).sum(-1) + 1e-9)
        pha = torch.atan2(stft_spec[:, :, :, 1] + 1e-8, stft_spec[:, :, :, 0] + 1e-8)
        mag = torch.pow(mag, self.compress_factor)
        batch_com = torch.stack((mag * torch.cos(pha), mag * torch.sin(pha)), dim=-1)
        # batch_com: [N, F, total_frames_needed, 2] — exact frame count, no skip needed

        # --- Phase 2: Sequential model forward (tight loop) ---
        all_mag: List[Tensor] = []
        all_pha: List[Tensor] = []

        for j in range(n_calls):
            chunk_com = batch_com[j:j + 1]  # [1, F, T, 2]
            result = self.process_spectrogram_buffered(chunk_com)

            if result is not None:
                est_mag, est_pha = result
                all_mag.append(est_mag)
                all_pha.append(est_pha)

        if not all_mag:
            return torch.tensor([], device=device)

        # --- Phase 3: Batch iSTFT (single call) ---
        cat_mag = torch.cat(all_mag, dim=2)  # [1, F, T_out]
        cat_pha = torch.cat(all_pha, dim=2)  # [1, F, T_out]

        output, _, _ = manual_istft_ola(
            cat_mag, cat_pha,
            n_fft=self.n_fft,
            hop_size=self.hop_size,
            win_size=self.win_size,
            compress_factor=self.compress_factor,
            ola_buffer=None,
            ola_norm=None,
        )

        if len(output) > audio_length:
            output = output[:audio_length]

        return output

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for nn.Module compatibility."""
        return self.process_audio(x)

    def __repr__(self) -> str:
        config = self._streaming_config
        rf_info = ""
        if self.use_reshape_free:
            rf_count = config.get('rf_block_count', 0)
            rf_info = f"  use_reshape_free=True ({rf_count} blocks),\n"
        return (
            f"{self.__class__.__name__}(\n"
            f"{rf_info}"
            f"  chunk_size={self.chunk_size},\n"
            f"  encoder_lookahead={self.encoder_lookahead},\n"
            f"  decoder_lookahead={self.decoder_lookahead},\n"
            f"  total_lookahead={self.total_lookahead},\n"
            f"  latency_ms={self.latency_ms:.2f},\n"
            f")"
        )
