"""
BAFNet-plus source package

This package contains the complete BAFNet-plus implementation:

Core Library:
- data: Dataset and data augmentation
- solver: Training loop (Solver class)
- stft: STFT/iSTFT utilities
- utils: Utility functions

Executable Scripts:
- train: Training entry point
- enhance: Inference/enhancement
- evaluate: Evaluation with metrics
- compute_metrics: Metric computation
"""

__version__ = "0.1.0"

__all__ = [
    'Solver',
]


def __getattr__(name):
    """Load tensor-runtime components only when they are explicitly requested.

    Keeping package initialization lightweight allows pure configuration and
    operating-grid contracts to be validated without installing PyTorch.
    """
    if name == "Solver":
        from .solver import Solver

        return Solver
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
