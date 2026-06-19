"""
Core utilities for streaming inference.

This module provides foundational utilities used across streaming components:

- StateFramesContext: Thread-local context for state frame control
- Model preparation pipeline utilities
- Stateful layer utilities

Merged from:
    - core/state_context.py
    - core/stateful_utils.py
    - core/model_builder.py

Example:
    >>> from src.models.streaming.utils import StateFramesContext, prepare_streaming_model
    >>>
    >>> # Prepare model
    >>> model, meta = prepare_streaming_model("path/to/checkpoint")
    >>>
    >>> # Use state frame context for hybrid streaming
    >>> with StateFramesContext(64):
    ...     output = model(extended_input)
"""

from __future__ import annotations

import os
import threading
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


# =============================================================================
# StateFramesContext (from core/state_context.py)
# =============================================================================
# CANONICAL THREAD-LOCAL STORAGE
# DO NOT DUPLICATE THIS ELSEWHERE IN THE CODEBASE
# All imports should come from this module

_state_frames_context = threading.local()


def get_state_frames_context() -> Optional[int]:
    """
    Get current state_frames context value.

    Returns:
        Number of frames to use for state update, or None if not set.
    """
    return getattr(_state_frames_context, 'value', None)


def set_state_frames_context(value: Optional[int]) -> None:
    """
    Set state_frames context value.

    Args:
        value: Number of frames to use for state update, or None to clear.
    """
    _state_frames_context.value = value


class StateFramesContext:
    """
    Context manager for setting state_frames during model forward pass.

    This allows processing extended input (current + lookahead) through
    the model while ensuring only the current frames update the state buffer.

    Example:
        >>> # Process 71 frames (64 current + 7 lookahead)
        >>> # but only update state for the first 64 frames
        >>> with StateFramesContext(64):
        ...     output = model(extended_input)

    Attributes:
        state_frames: Number of frames to use for state update
        prev_value: Previous context value (for nesting support)

    Note:
        Supports nesting - each context manager saves and restores
        the previous value on exit.
    """

    def __init__(self, state_frames: Optional[int]):
        """
        Initialize context manager.

        Args:
            state_frames: Number of frames to use for state update.
                If None, no limit is applied (all frames update state).
        """
        self.state_frames = state_frames
        self.prev_value: Optional[int] = None

    def __enter__(self) -> "StateFramesContext":
        """Enter context and set state_frames value."""
        self.prev_value = get_state_frames_context()
        set_state_frames_context(self.state_frames)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Exit context and restore previous value."""
        set_state_frames_context(self.prev_value)
        return False  # Don't suppress exceptions

    def __repr__(self) -> str:
        return f"StateFramesContext(state_frames={self.state_frames})"


# =============================================================================
# Stateful utilities (from core/stateful_utils.py)
# =============================================================================


def check_streaming_allowed(training: bool, layer_name: str = "StatefulLayer") -> None:
    """
    Check if streaming mode can be enabled.

    Streaming mode requires eval mode because:
    1. State buffers should not be part of gradient computation
    2. Dropout/BatchNorm behavior differs in train mode

    Args:
        training: Current training mode status (module.training)
        layer_name: Name for error message

    Raises:
        RuntimeError: If attempting to enable streaming during training
    """
    if training:
        raise RuntimeError(
            f"{layer_name}: Streaming mode is not supported during training. "
            "Call model.eval() first before enabling streaming."
        )


def check_batch_size_change(
    state: Optional[torch.Tensor],
    current_batch: int,
    layer_name: str = "StatefulLayer",
) -> bool:
    """
    Check for batch size change and warn if detected.

    Batch size changes during streaming indicate incorrect usage:
    - Each utterance should be processed independently
    - Mixing utterances in a batch corrupts state buffers

    Args:
        state: Current state buffer (may be None for first chunk)
        current_batch: Batch size of current input
        layer_name: Name for warning message

    Returns:
        True if state should be reset, False otherwise
    """
    if state is None:
        return False

    state_batch = state.shape[0]
    if state_batch != current_batch:
        warnings.warn(
            f"[{layer_name}] Batch size changed ({state_batch} -> {current_batch}). "
            f"Automatically resetting state. "
            f"This may indicate incorrect streaming usage.",
            RuntimeWarning,
        )
        return True
    return False


def sync_state_device_dtype(
    state: Optional[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """
    Sync state tensor to match input device/dtype.

    This handles cases where:
    - Model is moved to different device between chunks
    - Input dtype changes (e.g., mixed precision)

    Args:
        state: State buffer to sync (may be None)
        device: Target device
        dtype: Target dtype

    Returns:
        Synced state tensor, or None if input was None
    """
    if state is None:
        return None
    if state.device != device or state.dtype != dtype:
        return state.to(device=device, dtype=dtype)
    return state


# =============================================================================
# Model builder (from core/model_builder.py)
# =============================================================================


@dataclass
class ModelPrepConfig:
    """
    Model preparation configuration (streaming loop settings excluded).

    This covers the common settings for preparing a model for streaming:
    - Stateful convolution conversion
    - Reshape-free TS_BLOCK conversion

    Streaming-specific settings (chunk_size, lookahead, etc.) are handled
    by each wrapper class separately.

    Attributes:
        use_stateful_conv: Convert convolutions to stateful versions
        use_reshape_free: Convert TS_BLOCKs to reshape-free versions
            - Eliminates reshape operations (16 per inference)
            - Unifies batch dimension (all states have batch=1)
            - Reduces state count by ~48% (100 -> 52)
            - Recommended for NPU deployment
        device: Target device (auto-detect if None)
        verbose: Print loading information
    """

    use_stateful_conv: bool = True
    use_reshape_free: bool = False
    fold_bn: bool = False

    device: Optional[str] = None
    verbose: bool = True


def load_model_from_checkpoint(
    chkpt_dir: str,
    chkpt_file: str = "best.th",
    device: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[nn.Module, Any]:
    """
    Load model and config from checkpoint directory.

    Args:
        chkpt_dir: Path to checkpoint directory containing:
            - .hydra/config.yaml: Model configuration
            - best.th (or chkpt_file): Model weights
        chkpt_file: Checkpoint file name
        device: Target device (auto-detect if None)
        verbose: Print loading information

    Returns:
        Tuple of (model, model_args) where model_args is the OmegaConf node
        for compatibility with existing call-sites that use attribute access.

    Raises:
        FileNotFoundError: If config file not found
    """
    from omegaconf import OmegaConf

    from src.utils import load_checkpoint, load_model

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    config_path = os.path.join(chkpt_dir, ".hydra", "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    conf = OmegaConf.load(config_path)
    model_args = conf.model

    model = load_model(
        model_args,
        device,
    )
    model = load_checkpoint(model, chkpt_dir, chkpt_file, device)

    if verbose:
        print(f"  Model: Backbone")

    return model, model_args


def apply_reshape_free_tsblock(
    model: nn.Module,
    verbose: bool = True,
) -> Tuple[nn.Module, nn.ModuleList, int]:
    """
    Convert TS_BLOCKs to stateful reshape-free versions.

    This transformation:
    - Eliminates reshape operations (permute + contiguous)
    - Unifies batch dimension to 1 for all states
    - Reduces state count by ~48%
    - Uses Conv2d with axis-specific kernels instead of Conv1d

    Args:
        model: Model with TS_BLOCKs to convert
        verbose: Print information

    Returns:
        Tuple of (model, rf_sequence_block, num_ts_blocks):
        - model: Original model (sequence_block is NOT modified in-place)
        - rf_sequence_block: ModuleList of StatefulReshapeFreeTSBlocks
        - num_ts_blocks: Number of converted TS_BLOCKs
    """
    from src.models.streaming.converters.reshape_free_converter import (
        convert_sequence_block_to_stateful_reshape_free,
    )

    if not hasattr(model, "sequence_block"):
        if verbose:
            print("  No sequence_block found, skipping reshape-free conversion")
        return model, nn.ModuleList(), 0

    # Convert to stateful reshape-free (returns ModuleList)
    rf_sequence_block = convert_sequence_block_to_stateful_reshape_free(
        model.sequence_block,
    )

    num_blocks = len(rf_sequence_block)

    if verbose:
        print(f"  Applied Reshape-Free TS_BLOCK ({num_blocks} blocks)")
        print(f"    - Batch dimension unified to B=1")
        print(f"    - freq_stage now stateless")

    return model, rf_sequence_block, num_blocks


def apply_stateful_conv(
    model: nn.Module,
    device: str = "cpu",
    verbose: bool = True,
) -> Tuple[nn.Module, int]:
    """
    Convert conv layers to stateful versions and enable streaming mode.

    Args:
        model: Model to transform
        device: Target device
        verbose: Print information

    Returns:
        Tuple of (transformed_model, stateful_layer_count)
    """
    from src.models.streaming.converters.conv_converter import (
        convert_to_stateful,
        get_stateful_layer_count,
        set_streaming_mode,
    )

    model = convert_to_stateful(model, verbose=False, inplace=True)
    model.to(device)
    model.eval()
    set_streaming_mode(model, True)

    layer_counts = get_stateful_layer_count(model)

    if verbose:
        print(f"  Applied Stateful Convolutions ({layer_counts['total']} layers)")
        if layer_counts["StatefulCausalConv1d"] > 0:
            print(f"    - StatefulCausalConv1d: {layer_counts['StatefulCausalConv1d']}")
        if layer_counts["StatefulAsymmetricConv2d"] > 0:
            print(
                f"    - StatefulAsymmetricConv2d: {layer_counts['StatefulAsymmetricConv2d']}"
            )
        if layer_counts["StatefulCausalConv2d"] > 0:
            print(f"    - StatefulCausalConv2d: {layer_counts['StatefulCausalConv2d']}")

    return model, layer_counts["total"]


def prepare_streaming_model(
    chkpt_dir: str,
    chkpt_file: str = "best.th",
    config: Optional[ModelPrepConfig] = None,
    **kwargs,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Complete model preparation pipeline for streaming inference.

    This is the main entry point that combines all preparation steps:
    1. Load model from checkpoint
    2. BN Folding (if fold_bn)
    3. Optionally convert TS_BLOCKs to reshape-free versions
    4. Convert to Stateful Convolutions
    5. Channels-last + torch.compile (if fold_bn)

    Args:
        chkpt_dir: Checkpoint directory path
        chkpt_file: Checkpoint filename
        config: ModelPrepConfig instance (or pass kwargs)
        **kwargs: Override config values

    Returns:
        Tuple of (prepared_model, metadata_dict) where metadata contains:
            - chkpt_dir: Checkpoint directory
            - use_stateful_conv: Whether stateful conv was applied
            - stateful_conv_count: Number of converted layers
            - model_args: Original model configuration (OmegaConf node)

    Example:
        >>> model, meta = prepare_streaming_model(
        ...     "path/to/checkpoint",
        ... )
        >>> print(meta["model_args"].dense_channel)
    """
    # Build config from kwargs if not provided
    if config is None:
        # Filter kwargs to only include valid ModelPrepConfig fields
        valid_fields = {
            "use_stateful_conv",
            "use_reshape_free",
            "fold_bn",
            "device",
            "verbose",
        }
        config_kwargs = {k: v for k, v in kwargs.items() if k in valid_fields}
        config = ModelPrepConfig(**config_kwargs)
    else:
        # Allow kwargs to override config
        for k, v in kwargs.items():
            if hasattr(config, k):
                setattr(config, k, v)

    device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if config.verbose:
        print(f"Preparing streaming model from: {chkpt_dir}")
        print(f"  Device: {device}")
        if config.use_reshape_free:
            print(f"  Reshape-Free: enabled")

    # Step 1: Load model
    model, model_args = load_model_from_checkpoint(
        chkpt_dir, chkpt_file, device, config.verbose
    )

    # Step 2: BN Folding (before structural transforms preserve Sequential structure)
    bn_fused = 0
    if config.fold_bn:
        from src.models.streaming.cpu_optimizations import fold_batchnorm

        model, bn_fused = fold_batchnorm(model)
        if config.verbose:
            print(f"  BN Folding: {bn_fused} pairs fused")

    # Step 3: Apply Reshape-Free TS_BLOCK (if enabled)
    rf_sequence_block = None
    rf_block_count = 0
    if config.use_reshape_free:
        model, rf_sequence_block, rf_block_count = apply_reshape_free_tsblock(
            model,
            verbose=config.verbose,
        )

    # Step 4: Apply Stateful Conv (for encoder/decoder)
    stateful_count = 0
    if config.use_stateful_conv:
        model, stateful_count = apply_stateful_conv(model, device, config.verbose)

    cpu_stats = {}
    if config.fold_bn:
        cpu_stats = {
            "bn_fused": bn_fused,
        }

    # Build metadata
    metadata: Dict[str, Any] = {
        "chkpt_dir": chkpt_dir,
        "use_stateful_conv": config.use_stateful_conv,
        "stateful_conv_count": stateful_count,
        "use_reshape_free": config.use_reshape_free,
        "rf_sequence_block": rf_sequence_block,
        "rf_block_count": rf_block_count,
        "fold_bn": config.fold_bn,
        "cpu_stats": cpu_stats,
        "model_args": model_args,
    }

    return model, metadata


__all__ = [
    # State context
    "StateFramesContext",
    "get_state_frames_context",
    "set_state_frames_context",
    # Stateful utilities
    "check_streaming_allowed",
    "check_batch_size_change",
    "sync_state_device_dtype",
    # Model preparation
    "ModelPrepConfig",
    "prepare_streaming_model",
    "load_model_from_checkpoint",
    "apply_stateful_conv",
    "apply_reshape_free_tsblock",
]
