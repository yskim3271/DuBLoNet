"""
ONNX LaCoSENet Wrapper.

This module provides an ONNX Runtime-based streaming wrapper that has the same
interface as LaCoSENet but uses ONNX for neural network inference.

Key features:
1. Same interface as LaCoSENet (from_checkpoint, process_samples, process_audio)
2. STFT/iSTFT processed on host in FP32
3. Neural network core runs via ONNX Runtime
4. Full support for encoder/decoder lookahead buffering

Usage:
    # Create from checkpoint (exports ONNX automatically)
    streaming = ONNXLaCoSENet.from_checkpoint(
        chkpt_dir="path/to/checkpoint",
        chunk_size=64,
        encoder_lookahead=0,
        decoder_lookahead=7,
    )

    # Process audio
    enhanced = streaming.process_audio(noisy_audio)

    # Or process in streaming fashion
    for chunk in audio_stream:
        output = streaming.process_samples(chunk)
        if output is not None:
            play(output)
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from src.stft import mag_pha_istft, mag_pha_stft

logger = logging.getLogger(__name__)


# =============================================================================
# QNN Execution Provider Configuration
# =============================================================================

@dataclass
class QNNConfig:
    """
    Configuration for QNN Execution Provider.

    This configures the Qualcomm AI Engine Direct (QNN) backend for ONNX Runtime,
    enabling hardware-accelerated inference on Qualcomm Snapdragon SoCs.

    Reference: https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html

    Attributes:
        backend_type: QNN backend - "htp" (NPU), "gpu", or "cpu"
        htp_performance_mode: HTP performance mode - "burst", "balanced", "low_power"
        soc_model: Target SoC model (e.g., "SM8550" for Snapdragon 8 Gen 2)
        enable_htp_fp16_precision: Enable FP16 precision on HTP (default False for INT8)
        vtcm_mb: VTCM memory allocation in MB
        context_cache_enabled: Enable QNN context binary caching
        context_cache_path: Path to save/load QNN context binary
        context_embed_mode: Embed context binary in ONNX file (1) or separate (0)
        disable_cpu_ep_fallback: Disable CPU EP fallback for strict QNN execution
    """
    backend_type: str = "htp"
    htp_performance_mode: str = "burst"
    soc_model: Optional[str] = None  # e.g., "SM8550"
    enable_htp_fp16_precision: bool = False
    vtcm_mb: int = 8
    context_cache_enabled: bool = False
    context_cache_path: Optional[str] = None
    context_embed_mode: int = 1  # 1 = embedded, 0 = separate
    disable_cpu_ep_fallback: bool = False

    def to_provider_options(self) -> Dict[str, str]:
        """Convert to ONNX Runtime provider options dict."""
        options = {
            "backend_type": self.backend_type,
            "htp_performance_mode": self.htp_performance_mode,
            "enable_htp_fp16_precision": "1" if self.enable_htp_fp16_precision else "0",
            "vtcm_mb": str(self.vtcm_mb),
        }
        if self.soc_model:
            options["soc_model"] = self.soc_model
        return options

    def to_session_options_entries(self) -> Dict[str, str]:
        """Get session-level config entries for context caching."""
        entries = {}
        if self.context_cache_enabled and self.context_cache_path:
            entries["ep.context_enable"] = "1"
            entries["ep.context_file_path"] = self.context_cache_path
            entries["ep.context_embed_mode"] = str(self.context_embed_mode)
        if self.disable_cpu_ep_fallback:
            entries["session.disable_cpu_ep_fallback"] = "1"
        return entries


def create_ort_session(
    onnx_path: str,
    providers: Optional[List[str]] = None,
    provider_options: Optional[List[Dict[str, str]]] = None,
    qnn_config: Optional[QNNConfig] = None,
    num_threads: Optional[int] = None,
    verbose: bool = False,
) -> Any:
    """
    Create ONNX Runtime InferenceSession with flexible backend configuration.

    Supports three modes:
    1. Simple: providers list only (e.g., ["CPUExecutionProvider"])
    2. Advanced: providers + provider_options lists
    3. QNN shortcut: QNNConfig object for Qualcomm hardware acceleration

    Args:
        onnx_path: Path to ONNX model file
        providers: List of execution providers (e.g., ["QNNExecutionProvider", "CPUExecutionProvider"])
        provider_options: List of option dicts matching providers list
        qnn_config: QNNConfig for simplified QNN EP setup (overrides providers/provider_options)
        num_threads: CPU intra-op thread count for ONNX Runtime
        verbose: Print configuration info

    Returns:
        onnxruntime.InferenceSession

    Example:
        # CPU only
        session = create_ort_session(path, providers=["CPUExecutionProvider"])

        # QNN HTP with context caching
        qnn_cfg = QNNConfig(
            backend_type="htp",
            htp_performance_mode="burst",
            soc_model="SM8550",
            context_cache_enabled=True,
            context_cache_path="model_qnn_ctx.onnx",
        )
        session = create_ort_session(path, qnn_config=qnn_cfg)

        # Manual provider options
        session = create_ort_session(
            path,
            providers=["QNNExecutionProvider", "CPUExecutionProvider"],
            provider_options=[{"backend_type": "htp"}, {}],
        )
    """
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError("onnxruntime is required. Install with: pip install onnxruntime")

    # Default to CPU
    if providers is None and qnn_config is None:
        providers = ["CPUExecutionProvider"]

    sess_options = ort.SessionOptions()
    if num_threads is not None:
        sess_options.intra_op_num_threads = int(num_threads)
        sess_options.inter_op_num_threads = 1

    # QNN Config takes precedence
    if qnn_config is not None:
        providers = ["QNNExecutionProvider", "CPUExecutionProvider"]
        provider_options = [qnn_config.to_provider_options(), {}]

        # Apply session-level config entries (context caching, etc.)
        for key, value in qnn_config.to_session_options_entries().items():
            sess_options.add_session_config_entry(key, value)

        if verbose:
            logger.info(f"QNN Config: backend={qnn_config.backend_type}, "
                       f"perf_mode={qnn_config.htp_performance_mode}, "
                       f"soc={qnn_config.soc_model}")
            if qnn_config.context_cache_enabled:
                logger.info(f"QNN Context Cache: {qnn_config.context_cache_path}")

    if verbose:
        logger.info(f"Creating ORT session: {onnx_path}")
        logger.info(f"  Providers: {providers}")
        if provider_options:
            logger.info(f"  Provider options: {provider_options}")

    # Create session
    if provider_options is not None:
        session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_options,
            providers=providers,
            provider_options=provider_options,
        )
    else:
        session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_options,
            providers=providers,
        )

    return session


@dataclass
class STFTConfig:
    """STFT configuration matching Backbone defaults."""
    n_fft: int = 400
    hop_size: int = 100
    win_size: int = 400
    compress_factor: float = 0.3
    sample_rate: int = 16000

    @property
    def freq_size(self) -> int:
        return self.n_fft // 2 + 1


class ONNXLaCoSENet:
    """
    ONNX Runtime-based streaming wrapper for Backbone.

    This class provides the same interface as LaCoSENet but uses
    ONNX Runtime for neural network inference. The STFT/iSTFT operations
    are performed on the host in FP32.

    The streaming pipeline is:
        audio -> input_buffer -> STFT (Host) -> mag/pha
             -> ONNX NN Core (with state I/O)
             -> est_mask/est_pha -> iSTFT (Host) -> output

    Lookahead buffering (encoder/decoder) works exactly as in LaCoSENet:
    - encoder_lookahead: Delays encoder processing for asymmetric encoder padding
    - decoder_lookahead: Buffers encoder features for asymmetric decoder padding

    Attributes:
        chunk_size: Number of STFT frames per processing chunk
        encoder_lookahead: Frames needed for encoder asymmetry (0 if fully causal)
        decoder_lookahead: Frames needed for decoder asymmetry (0 if fully causal)
        latency_ms: Total latency in milliseconds
    """

    def __init__(
        self,
        onnx_session: Any,  # onnxruntime.InferenceSession
        stft_config: STFTConfig,
        chunk_size: int = 64,
        encoder_lookahead: int = 0,
        decoder_lookahead: int = 7,
        stft_lookahead_frames: int = 1,
        infer_type: str = "masking",
        phase_output_mode: str = "atan2",
        state_names: Optional[List[str]] = None,
        state_init_fn: Optional[Any] = None,  # Callable to initialize states
        expected_time_frames: Optional[int] = None,
    ):
        """
        Initialize ONNXLaCoSENet.

        Note: Use `from_checkpoint()` for easier initialization.

        Args:
            onnx_session: ONNX Runtime session
            stft_config: STFT configuration
            chunk_size: Number of STFT frames per chunk
            encoder_lookahead: Frames to delay encoder processing
            decoder_lookahead: Frames to delay decoder processing
            stft_lookahead_frames: Extra frames for STFT overlap-add
            infer_type: "masking" or "mapping"
            phase_output_mode: "atan2" or "complex"
            state_names: Names of state tensors (from ONNX model)
            state_init_fn: Callable to initialize states with correct shapes
        """
        self.session = onnx_session
        self.stft_config = stft_config
        self.infer_type = infer_type
        self.phase_output_mode = phase_output_mode
        # For static-shape ONNX (INT8 PTQ), mag/pha time dimension must match exactly.
        # If set, we will pad/trim mag/pha to this length before session.run().
        self.expected_time_frames = expected_time_frames

        # Streaming parameters
        self.chunk_size = chunk_size
        self.encoder_lookahead = encoder_lookahead
        self.decoder_lookahead = decoder_lookahead
        self.stft_lookahead_frames = max(1, int(stft_lookahead_frames))
        self.input_lookahead_frames = int(encoder_lookahead)
        self.total_lookahead = self.input_lookahead_frames + decoder_lookahead

        # STFT parameters
        self.hop_size = stft_config.hop_size
        self.n_fft = stft_config.n_fft
        self.win_size = stft_config.win_size
        self.compress_factor = stft_config.compress_factor
        self.sample_rate = stft_config.sample_rate
        self.freq_size = stft_config.freq_size

        # Calculate samples per chunk
        self.total_frames_needed = chunk_size + self.input_lookahead_frames
        self.samples_per_chunk = (self.total_frames_needed - 1) * self.hop_size + self.win_size
        self.output_frames_per_chunk = chunk_size
        self.output_samples_per_chunk = self.output_frames_per_chunk * self.hop_size

        # Latency calculation (matches LaCoSENet)
        self.stft_center_delay_samples = self.win_size // 2
        self.latency_samples = self.total_lookahead * self.hop_size + self.stft_center_delay_samples
        self.latency_ms = self.latency_samples / self.sample_rate * 1000

        # ONNX state management
        self._state_names = state_names or []
        self._states: List[np.ndarray] = []
        self._state_init_fn = state_init_fn  # Function to create initial states

        # Initialize buffers
        self._reset_buffers()

        # Config storage
        self._streaming_config: Dict[str, Any] = {}

    def _reset_buffers(self) -> None:
        """Reset all internal buffers."""
        self.input_buffer = torch.tensor([], dtype=torch.float32)
        self.feature_buffer: List[Dict[str, Any]] = []
        self._first_chunk = True
        self._buffered_frames = 0
        self._stft_context: Optional[torch.Tensor] = None
        self._stft_context_frames = self.win_size // (2 * self.hop_size)

    def _init_onnx_states(self) -> None:
        """Initialize ONNX state tensors.

        Uses the state_init_fn if provided, which ensures correct batch dimensions
        for time_stage (B*F_enc) and freq_stage (B*T) layers.
        """
        if self._state_init_fn is not None:
            # Use the init function from StatefulExportableNNCore
            states = self._state_init_fn()
            self._states = [s.cpu().numpy() for s in states]
        else:
            # Fallback: Get state shapes from session inputs
            self._states = []
            for inp in self.session.get_inputs():
                if "state" in inp.name.lower():
                    # Get shape and create zero tensor
                    shape = list(inp.shape)
                    # Replace dynamic dimensions with actual values
                    for i, dim in enumerate(shape):
                        if isinstance(dim, str):
                            if "batch" in dim.lower():
                                shape[i] = 1
                            else:
                                # For other dynamic dims, use reasonable defaults
                                shape[i] = 1
                    self._states.append(np.zeros(shape, dtype=np.float32))

    def reset_state(self) -> None:
        """
        Reset all streaming state for a new audio stream.

        This resets:
        1. Audio input buffers
        2. Feature buffer for decoder lookahead
        3. ONNX state tensors
        """
        self._reset_buffers()
        self._init_onnx_states()

    @property
    def streaming_config(self) -> Dict[str, Any]:
        """Get streaming configuration information."""
        return {
            **self._streaming_config,
            "chunk_size_frames": self.chunk_size,
            "encoder_lookahead": self.encoder_lookahead,
            "decoder_lookahead": self.decoder_lookahead,
            "stft_lookahead_frames": self.stft_lookahead_frames,
            "input_lookahead_frames": self.input_lookahead_frames,
            "total_lookahead": self.total_lookahead,
            "output_frames_per_chunk": self.output_frames_per_chunk,
            "samples_per_chunk": self.samples_per_chunk,
            "output_samples_per_chunk": self.output_samples_per_chunk,
            "latency_samples": self.latency_samples,
            "latency_ms": self.latency_ms,
            "hop_size": self.hop_size,
            "sample_rate": self.sample_rate,
        }

    @classmethod
    def from_checkpoint(
        cls,
        chkpt_dir: str,
        chkpt_file: str = "best.th",
        chunk_size: int = 64,
        encoder_lookahead: int = 0,
        decoder_lookahead: int = 7,
        providers: Optional[List[str]] = None,
        provider_options: Optional[List[Dict[str, str]]] = None,
        qnn_config: Optional[QNNConfig] = None,
        onnx_path: Optional[str] = None,
        phase_output_mode: str = "atan2",
        export_use_dynamic_axes: bool = True,
        export_opset_version: int = 17,
        force_export: bool = False,
        use_reshape_free: bool = False,
        num_threads: Optional[int] = None,
        verbose: bool = True,
    ) -> "ONNXLaCoSENet":
        """
        Create ONNXLaCoSENet from a checkpoint directory.

        This automatically:
        1. Loads the PyTorch model
        2. Converts to StatefulExportableNNCore
        3. Exports to ONNX (if onnx_path not provided, uses temp file)
        4. Creates ONNX Runtime session

        Args:
            chkpt_dir: Path to checkpoint directory
            chkpt_file: Checkpoint file name (default: "best.th")
            chunk_size: Number of STFT frames per chunk
            encoder_lookahead: Frames for encoder lookahead
            decoder_lookahead: Frames for decoder lookahead
            providers: ONNX execution providers (default: CPU)
            provider_options: Provider-specific options (list matching providers)
            qnn_config: QNNConfig for Qualcomm hardware acceleration.
                If provided, overrides providers/provider_options.
            onnx_path: Path to save/load ONNX model. If None, uses temp file.
            phase_output_mode: "atan2" or "complex"
            export_use_dynamic_axes: If True, export with dynamic axes (batch/time).
                Set to False for INT8 static PTQ (static shapes required).
            export_opset_version: ONNX opset version for export.
            force_export: If True, export ONNX even if onnx_path already exists.
            use_reshape_free: If True, use StatefulReshapeFreeExportableNNCore
                which eliminates B*F/B*T reshape overhead in TS_BLOCK.
            num_threads: CPU intra-op thread count for ONNX Runtime.
            verbose: Print loading information

        Returns:
            ONNXLaCoSENet instance

        Example:
            # CPU (default)
            streaming = ONNXLaCoSENet.from_checkpoint(chkpt_dir="...")

            # QNN HTP (Qualcomm NPU)
            from src.models.onnx_export import QNNConfig
            qnn_cfg = QNNConfig(backend_type="htp", soc_model="SM8550")
            streaming = ONNXLaCoSENet.from_checkpoint(
                chkpt_dir="...",
                qnn_config=qnn_cfg,
            )
        """
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime is required. Install with: pip install onnxruntime")

        from src.models.streaming import LaCoSENet
        from src.models.onnx_export import (
            StatefulExportableNNCore,
            export_stateful_nncore_to_onnx,
        )
        from src.models.onnx_export.stateful_core_rf import (
            StatefulReshapeFreeExportableNNCore,
            export_stateful_rf_nncore_to_onnx,
        )

        # Default providers (qnn_config will override if provided)
        if providers is None and qnn_config is None:
            providers = ["CPUExecutionProvider"]

        chkpt_dir = Path(chkpt_dir)

        if verbose:
            print(f"Loading ONNXLaCoSENet from: {chkpt_dir}")

        # Step 1: Load PyTorch model via LaCoSENet
        pytorch_streaming = LaCoSENet.from_checkpoint(
            chkpt_dir=str(chkpt_dir),
            chkpt_file=chkpt_file,
            chunk_size=chunk_size,
            encoder_lookahead=encoder_lookahead,
            decoder_lookahead=decoder_lookahead,
            device="cpu",  # Export from CPU
            verbose=verbose,
        )

        model = pytorch_streaming.model
        stft_config = STFTConfig(
            n_fft=pytorch_streaming.n_fft,
            hop_size=pytorch_streaming.hop_size,
            win_size=pytorch_streaming.win_size,
            compress_factor=pytorch_streaming.compress_factor,
            sample_rate=pytorch_streaming.sample_rate,
        )

        # Step 2: Convert to exportable core
        if use_reshape_free:
            if verbose:
                print("  Converting to StatefulReshapeFreeExportableNNCore...")
            core = StatefulReshapeFreeExportableNNCore.from_backbone(
                model,
                convert_to_functional=True,
                phase_output_mode=phase_output_mode,
            )
        else:
            if verbose:
                print("  Converting to StatefulExportableNNCore...")
            core = StatefulExportableNNCore.from_backbone(
                model,
                convert_to_functional=True,
                phase_output_mode=phase_output_mode,
            )

        # Step 3: Export to ONNX
        # Export time_frames must cover the model inference window used by this wrapper.
        # We need:
        # - encoder_lookahead: future context required by asymmetric encoder layers
        # - decoder_lookahead: additional future context required by asymmetric decoder
        # This matches the PyTorch buffered streaming design where the decoder consumes
        # an extended window (current + lookahead) while state updates advance only
        # by chunk_size frames per step.
        input_lookahead_frames = int(encoder_lookahead)
        export_time_frames = chunk_size + input_lookahead_frames + decoder_lookahead

        if onnx_path is None:
            # Use temp file — must force export since the file is created empty
            temp_file = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
            onnx_path = temp_file.name
            temp_file.close()
            force_export = True

        if verbose:
            print(f"  Exporting to ONNX: {onnx_path}")
            print(f"  Export time_frames: {export_time_frames}")

        # Limit conv state updates to chunk_size frames even when exporting an extended window.
        # This mirrors StateFramesContext(chunk_size) behavior in PyTorch streaming.
        core.set_state_frames_for_update(chunk_size)

        # Export only when needed to avoid re-exporting on every run.
        # This is important for Android integration where the ONNX file must live
        # at a stable path and should be treated as an artifact.
        should_export = force_export or (not Path(onnx_path).exists())
        if should_export:
            export_fn = export_stateful_rf_nncore_to_onnx if use_reshape_free else export_stateful_nncore_to_onnx
            export_fn(
                core,
                onnx_path,
                batch_size=1,
                time_frames=export_time_frames,
                freq_size=stft_config.freq_size,
                opset_version=export_opset_version,
                verbose=verbose,
                use_dynamic_axes=export_use_dynamic_axes,
            )
        else:
            if verbose:
                print(f"  Reusing existing ONNX: {onnx_path}")

        # Step 4: Create ONNX Runtime session
        if verbose:
            if qnn_config is not None:
                print(f"  Creating ONNX session with QNN EP (backend={qnn_config.backend_type})")
            else:
                print(f"  Creating ONNX session with providers: {providers}")

        session = create_ort_session(
            onnx_path,
            providers=providers,
            provider_options=provider_options,
            qnn_config=qnn_config,
            num_threads=num_threads,
            verbose=verbose,
        )

        # Get state names from model
        state_names = core.get_state_names()

        # Extract infer_type
        infer_type = getattr(model, 'infer_type', 'masking')

        # Create state initialization function
        # This ensures correct batch dimensions for time_stage/freq_stage layers
        # Use export_time_frames to match ONNX export dimensions
        def state_init_fn():
            return core.init_states(
                batch_size=1,
                freq_size=stft_config.freq_size,
                time_frames=export_time_frames,  # Match ONNX export
                device=torch.device("cpu"),
                dtype=torch.float32,
            )

        # Create instance
        instance = cls(
            onnx_session=session,
            stft_config=stft_config,
            chunk_size=chunk_size,
            encoder_lookahead=encoder_lookahead,
            decoder_lookahead=decoder_lookahead,
            infer_type=infer_type,
            phase_output_mode=phase_output_mode,
            state_names=state_names,
            state_init_fn=state_init_fn,
        )

        # Store config
        # Determine actual providers used
        actual_providers = providers
        if qnn_config is not None:
            actual_providers = ["QNNExecutionProvider", "CPUExecutionProvider"]

        instance._streaming_config = {
            "chkpt_dir": str(chkpt_dir),
            "onnx_path": onnx_path,
            "providers": actual_providers,
            "provider_options": provider_options,
            "qnn_config": qnn_config.__dict__ if qnn_config else None,
            "num_states": len(state_names),
            **pytorch_streaming._streaming_config,
        }

        # Initialize states
        instance._init_onnx_states()

        if verbose:
            print(f"  Chunk size: {chunk_size} frames")
            print(f"  Encoder lookahead: {encoder_lookahead} frames")
            print(f"  Decoder lookahead: {decoder_lookahead} frames")
            print(f"  Total latency: {instance.latency_ms:.1f}ms")
            print(f"  Number of ONNX states: {len(state_names)}")

        return instance

    @classmethod
    def from_onnx_path(
        cls,
        onnx_path: str,
        chunk_size: int = 64,
        encoder_lookahead: int = 0,
        decoder_lookahead: int = 0,
        stft_config: Optional[STFTConfig] = None,
        providers: Optional[List[str]] = None,
        provider_options: Optional[List[Dict[str, str]]] = None,
        qnn_config: Optional[QNNConfig] = None,
        phase_output_mode: Optional[str] = None,
        infer_type: str = "masking",
        num_threads: Optional[int] = None,
        verbose: bool = True,
    ) -> "ONNXLaCoSENet":
        """
        Create ONNXLaCoSENet from an existing ONNX file.

        This is the recommended entry point for Android integration where the ONNX
        model is shipped as a fixed artifact (e.g., assets -> app files directory).

        Args:
            onnx_path: Path to the ONNX model (FP32 or INT8)
            chunk_size: Chunk size in STFT frames (must match export-time config)
            encoder_lookahead: Encoder lookahead in frames (must match export)
            decoder_lookahead: Decoder lookahead in frames (must match export)
            stft_config: STFT configuration (defaults to Backbone defaults)
            providers: ORT execution providers (default: CPU)
            provider_options: Provider-specific options (list matching providers)
            qnn_config: QNNConfig for Qualcomm hardware acceleration.
                If provided, overrides providers/provider_options.
            phase_output_mode: Override for phase output mode. If None, inferred from model outputs.
            infer_type: "masking" or "mapping"
            num_threads: CPU intra-op thread count for ONNX Runtime.
            verbose: Print loading information

        Example:
            # CPU (default)
            streaming = ONNXLaCoSENet.from_onnx_path("model_int8.onnx")

            # QNN HTP with context caching
            qnn_cfg = QNNConfig(
                backend_type="htp",
                soc_model="SM8550",
                context_cache_enabled=True,
                context_cache_path="model_qnn_ctx.onnx",
            )
            streaming = ONNXLaCoSENet.from_onnx_path(
                "model_int8.onnx",
                qnn_config=qnn_cfg,
            )
        """
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime is required. Install with: pip install onnxruntime")

        # Default providers (qnn_config will override if provided)
        if providers is None and qnn_config is None:
            providers = ["CPUExecutionProvider"]
        if stft_config is None:
            stft_config = STFTConfig()

        onnx_path = str(Path(onnx_path))
        if verbose:
            print(f"Loading ONNXLaCoSENet from ONNX: {onnx_path}")
            if qnn_config is not None:
                print(f"  QNN EP: backend={qnn_config.backend_type}, soc={qnn_config.soc_model}")
            else:
                print(f"  Providers: {providers}")

        session = create_ort_session(
            onnx_path,
            providers=providers,
            provider_options=provider_options,
            qnn_config=qnn_config,
            num_threads=num_threads,
            verbose=verbose,
        )

        # Infer expected time frames from the ONNX input shape (static export).
        expected_time_frames: Optional[int] = None
        try:
            mag_inp = next(inp for inp in session.get_inputs() if inp.name == "mag")
            if isinstance(mag_inp.shape, (list, tuple)) and len(mag_inp.shape) == 3:
                tdim = mag_inp.shape[2]
                if isinstance(tdim, int):
                    expected_time_frames = int(tdim)
        except StopIteration:
            expected_time_frames = None

        # Infer phase output mode if not specified.
        if phase_output_mode is None:
            out_names = [o.name for o in session.get_outputs()]
            if ("phase_real" in out_names) and ("phase_imag" in out_names):
                phase_output_mode = "complex"
            else:
                phase_output_mode = "atan2"

        # Infer state names from inputs (stable for static export).
        state_names = [i.name for i in session.get_inputs() if "state" in i.name.lower()]

        instance = cls(
            onnx_session=session,
            stft_config=stft_config,
            chunk_size=chunk_size,
            encoder_lookahead=encoder_lookahead,
            decoder_lookahead=decoder_lookahead,
            infer_type=infer_type,
            phase_output_mode=phase_output_mode,
            state_names=state_names,
            state_init_fn=None,  # Use static input shapes from ONNX
            expected_time_frames=expected_time_frames,
        )

        # Determine actual providers used
        actual_providers = providers
        if qnn_config is not None:
            actual_providers = ["QNNExecutionProvider", "CPUExecutionProvider"]

        instance._streaming_config = {
            "onnx_path": onnx_path,
            "providers": actual_providers,
            "provider_options": provider_options,
            "qnn_config": qnn_config.__dict__ if qnn_config else None,
            "num_states": len(state_names),
            "expected_time_frames": expected_time_frames,
        }

        instance._init_onnx_states()

        if verbose:
            print(f"  Phase output mode: {phase_output_mode}")
            if expected_time_frames is not None:
                print(f"  Expected time_frames (static): {expected_time_frames}")
            print(f"  Number of ONNX states: {len(state_names)}")
            print(f"  Total latency: {instance.latency_ms:.1f}ms")

        return instance

    def _stft(self, audio: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute STFT and return magnitude and phase."""
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        mag, pha, _ = mag_pha_stft(
            audio,
            n_fft=self.n_fft,
            hop_size=self.hop_size,
            win_size=self.win_size,
            compress_factor=self.compress_factor,
        )
        return mag, pha

    def _istft(self, mag: Tensor, pha: Tensor) -> Tensor:
        """Compute inverse STFT."""
        return mag_pha_istft(
            mag, pha,
            n_fft=self.n_fft,
            hop_size=self.hop_size,
            win_size=self.win_size,
            compress_factor=self.compress_factor,
        )

    def _run_onnx(
        self,
        mag: np.ndarray,
        pha: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """
        Run ONNX inference.

        Args:
            mag: Magnitude spectrogram [B, F, T]
            pha: Phase spectrogram [B, F, T]

        Returns:
            Tuple of (est_mask, est_pha, next_states)
        """
        # For static-shape exports (INT8 PTQ), pad/trim time dimension to match.
        if self.expected_time_frames is not None:
            t_expected = int(self.expected_time_frames)
            t_in = int(mag.shape[2])
            if t_in < t_expected:
                pad = t_expected - t_in
                mag = np.pad(mag, ((0, 0), (0, 0), (0, pad)), mode="constant")
                pha = np.pad(pha, ((0, 0), (0, 0), (0, pad)), mode="constant")
            elif t_in > t_expected:
                mag = mag[:, :, :t_expected]
                pha = pha[:, :, :t_expected]

        # Build input dict
        inputs = {
            "mag": mag.astype(np.float32),
            "pha": pha.astype(np.float32),
        }

        # Add states
        for i, state_name in enumerate(self._state_names):
            inputs[state_name] = self._states[i]

        # Run inference
        outputs = self.session.run(None, inputs)

        # Parse outputs based on phase_output_mode
        if self.phase_output_mode == "complex":
            est_mask = outputs[0]
            phase_real = outputs[1]
            phase_imag = outputs[2]
            # Compute atan2 on host
            est_pha = np.arctan2(phase_imag + 1e-8, phase_real + 1e-8)
            next_states = outputs[3:]
        else:
            est_mask = outputs[0]
            est_pha = outputs[1]
            next_states = outputs[2:]

        # Update states
        self._states = list(next_states)

        return est_mask, est_pha, next_states

    def _can_process_immediately(self) -> bool:
        """Check if decoder can process immediately."""
        return self.decoder_lookahead == 0

    def _process_immediate(
        self,
        mag: Tensor,
        pha: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Process immediately without feature buffering.

        Used when decoder_lookahead == 0.

        ONNX model is exported with the same model lookahead as the PyTorch wrapper.
        STFT lookahead is kept separate from model lookahead so it does not inflate
        reported algorithmic latency.

        Args:
            mag: Magnitude spectrogram [B, F, T] where T >= chunk_size
            pha: Phase spectrogram [B, F, T]

        Returns:
            Tuple of (est_mag, est_pha) for iSTFT
        """
        B, F, T = mag.shape
        # Use model input lookahead only; STFT lookahead is not model latency.
        frames_for_model = self.chunk_size + self.input_lookahead_frames
        frames_for_istft = self.chunk_size + self.stft_lookahead_frames

        # Process all frames through ONNX
        # (ONNX model was exported with chunk_size + input_lookahead_frames)
        frames_to_use = min(T, frames_for_model)
        mag_extended = mag[:, :, :frames_to_use]
        pha_extended = pha[:, :, :frames_to_use]

        mag_np = mag_extended.numpy()
        pha_np = pha_extended.numpy()
        est_mask, est_pha_out, _ = self._run_onnx(mag_np, pha_np)

        # Apply mask
        if self.infer_type == 'masking':
            est_mag_full = mag_np * est_mask
        else:
            est_mag_full = est_mask

        # Trim to frames_for_istft (may be less than frames_for_model if encoder_lookahead > stft_lookahead)
        est_mag = est_mag_full[:, :, :frames_for_istft]
        est_pha = est_pha_out[:, :, :frames_for_istft]

        return torch.from_numpy(est_mag), torch.from_numpy(est_pha)

    def _process_buffered(
        self,
        mag: Tensor,
        pha: Tensor,
    ) -> Optional[Tuple[Tensor, Tensor]]:
        """
        Process with feature buffering for decoder lookahead.

        Used when decoder_lookahead > 0.

        We buffer frames until we have enough for decoder lookahead.
        ONNX model processes exactly chunk_size frames (state dimensions are fixed).

        Args:
            mag: Magnitude spectrogram [B, F, T]
            pha: Phase spectrogram [B, F, T]

        Returns:
            Tuple of (est_mag, est_pha) if output available, None otherwise
        """
        B, F, T = mag.shape
        valid_frames = min(T, self.chunk_size)

        # Step 1: Add to feature buffer
        self.feature_buffer.append({
            'mag': mag[:, :, :valid_frames].numpy(),
            'pha': pha[:, :, :valid_frames].numpy(),
            'frames': valid_frames,
        })
        self._buffered_frames += valid_frames

        # Step 2: Check if we have enough for processing.
        # We need an extended window for the model so that asymmetric encoder/decoder
        # can see future context:
        #   total_needed = chunk_size + input_lookahead_frames + decoder_lookahead
        total_needed = self.chunk_size + self.input_lookahead_frames + self.decoder_lookahead
        if self._buffered_frames < total_needed:
            logger.debug(f"Buffering: {self._buffered_frames}/{total_needed}")
            return None

        # Step 3: Gather features
        all_mag = np.concatenate([buf['mag'] for buf in self.feature_buffer], axis=2)
        all_pha = np.concatenate([buf['pha'] for buf in self.feature_buffer], axis=2)

        # Step 4: Process frames through ONNX
        # ONNX model expects exactly export_time_frames, which we set to:
        #   chunk_size + input_lookahead_frames + decoder_lookahead
        frames_for_model = total_needed
        frames_for_istft = self.chunk_size + self.stft_lookahead_frames

        mag_chunk = all_mag[:, :, :frames_for_model]
        pha_chunk = all_pha[:, :, :frames_for_model]

        # Run ONNX model
        est_mask, est_pha_out, _ = self._run_onnx(mag_chunk, pha_chunk)

        # Apply mask
        if self.infer_type == 'masking':
            est_mag_full = mag_chunk * est_mask
        else:
            est_mag_full = est_mask

        # Step 5: Trim to frames_for_istft for iSTFT
        est_mag = est_mag_full[:, :, :frames_for_istft]
        est_pha = est_pha_out[:, :, :frames_for_istft]

        # Step 6: Update feature buffer - remove chunk_size frames
        frames_to_remove = self.chunk_size
        removed = 0

        while removed < frames_to_remove and self.feature_buffer:
            buf = self.feature_buffer[0]
            if buf['frames'] <= (frames_to_remove - removed):
                removed += buf['frames']
                self.feature_buffer.pop(0)
            else:
                keep_frames = buf['frames'] - (frames_to_remove - removed)
                buf['mag'] = buf['mag'][:, :, -keep_frames:]
                buf['pha'] = buf['pha'][:, :, -keep_frames:]
                buf['frames'] = keep_frames
                removed = frames_to_remove

        self._buffered_frames -= removed

        return torch.from_numpy(est_mag), torch.from_numpy(est_pha)

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

        # Add to buffer (keep on CPU for STFT)
        self.input_buffer = torch.cat([self.input_buffer, samples.cpu()])

        # Check if we have enough samples
        if len(self.input_buffer) < self.samples_per_chunk:
            return None

        # Extract chunk for processing
        chunk_samples = self.input_buffer[:self.samples_per_chunk]

        # Prepend STFT context
        context_size = self.win_size // 2
        if self._stft_context is not None:
            chunk_with_context = torch.cat([self._stft_context, chunk_samples])
            mag, pha = self._stft(chunk_with_context)
            mag = mag[:, :, self._stft_context_frames:]
            pha = pha[:, :, self._stft_context_frames:]
        else:
            mag, pha = self._stft(chunk_samples)

        # Save context for next chunk
        context_start = self.output_samples_per_chunk - context_size
        context_end = self.output_samples_per_chunk
        self._stft_context = self.input_buffer[context_start:context_end].clone()

        # Process through model
        if self._can_process_immediately():
            result = self._process_immediate(mag, pha)
        else:
            result = self._process_buffered(mag, pha)

        if result is None:
            # Still buffering
            self.input_buffer = self.input_buffer[self.output_samples_per_chunk:]
            return None

        est_mag, est_pha = result

        # Compute iSTFT
        output_audio = self._istft(est_mag, est_pha).squeeze(0)

        # Extract valid output region
        if self._first_chunk:
            self._first_chunk = False
            start_idx = self.win_size // 2
        else:
            start_idx = 0

        end_idx = start_idx + self.output_samples_per_chunk
        if end_idx > len(output_audio):
            end_idx = len(output_audio)
        valid_output = output_audio[start_idx:end_idx]

        # Update buffer
        self.input_buffer = self.input_buffer[self.output_samples_per_chunk:]

        return valid_output

    def process_audio(self, audio: Tensor) -> Tensor:
        """
        Process a complete audio signal in streaming fashion.

        Args:
            audio: Complete input audio [T] or [1, T]

        Returns:
            Enhanced audio [T]
        """
        if audio.dim() == 2:
            audio = audio.squeeze(0)

        self.reset_state()
        outputs = []

        # Process in increments
        increment = self.hop_size * 4

        for i in range(0, len(audio), increment):
            chunk = audio[i:i + increment]
            output = self.process_samples(chunk)
            if output is not None:
                outputs.append(output)

        # Flush remaining samples
        padding = torch.zeros(self.samples_per_chunk * 2)
        for _ in range(3):
            final_output = self.process_samples(padding)
            if final_output is not None:
                outputs.append(final_output)

        if outputs:
            return torch.cat(outputs)
        else:
            return torch.tensor([])

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  chunk_size={self.chunk_size},\n"
            f"  encoder_lookahead={self.encoder_lookahead},\n"
            f"  decoder_lookahead={self.decoder_lookahead},\n"
            f"  total_lookahead={self.total_lookahead},\n"
            f"  latency_ms={self.latency_ms:.2f},\n"
            f"  num_states={len(self._state_names)},\n"
            f")"
        )


__all__ = [
    "ONNXLaCoSENet",
    "STFTConfig",
    "QNNConfig",
    "create_ort_session",
]
