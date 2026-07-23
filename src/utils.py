import os
import time
import torch
import numpy as np
import logging
import re
from typing import Dict, List, Optional, Tuple, Union, Any
from joblib import Parallel, delayed
from contextlib import contextmanager
import atexit
from pesq import pesq

def anti_wrapping_function(x):
    return torch.abs(x - torch.round(x / (2 * np.pi)) * 2 * np.pi)

def phase_losses(phase_r, phase_g):
    ip_loss = torch.mean(anti_wrapping_function(phase_r - phase_g))
    gd_loss = torch.mean(anti_wrapping_function(torch.diff(phase_r, dim=1) - torch.diff(phase_g, dim=1)))
    iaf_loss = torch.mean(anti_wrapping_function(torch.diff(phase_r, dim=2) - torch.diff(phase_g, dim=2)))

    return ip_loss + gd_loss + iaf_loss

# Reusable joblib Parallel pool (loky backend)
_JOBLIB_PARALLEL = None
_JOBLIB_WORKERS = None

def _get_joblib_parallel(workers: int):
    """Return a reusable joblib Parallel instance; recreate if worker size changed."""
    global _JOBLIB_PARALLEL, _JOBLIB_WORKERS
    if _JOBLIB_PARALLEL is None or _JOBLIB_WORKERS != workers:
        # Terminate existing pool if present (best-effort; uses private API)
        if _JOBLIB_PARALLEL is not None:
            try:
                _JOBLIB_PARALLEL._terminate_pool()
            except Exception:
                pass
        _JOBLIB_PARALLEL = Parallel(n_jobs=workers, backend="loky", prefer="processes")
        _JOBLIB_WORKERS = workers
    return _JOBLIB_PARALLEL

def batch_pesq(clean, noisy, workers=8, normalize=True):
    # Reuse a single loky process pool to avoid frequent creation/cleanup cycles
    parallel = _get_joblib_parallel(workers)
    pesq_score = parallel(delayed(pesq_loss)(c, n) for c, n in zip(clean, noisy))
    pesq_score = np.array(pesq_score)
    if -1 in pesq_score:
        return None
    if normalize:
        pesq_score = (pesq_score - 1) / 3.5
    return torch.FloatTensor(pesq_score)

def pesq_loss(clean, noisy, sr=16000):
    try:
        pesq_score = pesq(sr, clean, noisy, "wb")
    except:
        # error can happen due to silent period
        pesq_score = -1
    return pesq_score


def copy_state(state):
    return {k: v.cpu().clone() for k, v in state.items()}


def serialize_model(model):
    args, kwargs = model._init_args_kwargs
    state = copy_state(model.state_dict())
    return {"class": model.__class__, "args": args, "kwargs": kwargs, "state": state}


@contextmanager
def swap_state(model, state):
    """
    Context manager that swaps the state of a model, e.g:

        # model is in old state
        with swap_state(model, new_state):
            # model in new state
        # model back to old state
    """
    old_state = copy_state(model.state_dict())
    model.load_state_dict(state)
    try:
        yield
    finally:
        model.load_state_dict(old_state)


def pull_metric(history, name):
    out = []
    for metrics in history:
        if name in metrics:
            out.append(metrics[name])
    return out

def expand_path(path):
    return os.path.abspath(os.path.expanduser(path))

def basename(path):
    filename, ext = os.path.splitext(os.path.basename(path))
    return filename, ext



class LogProgress:
    """
    Sort of like tqdm but using log lines and not as real time.
    Args:
        - logger: logger obtained from `logging.getLogger`,
        - iterable: iterable object to wrap
        - updates (int): number of lines that will be printed, e.g.
            if `updates=5`, log every 1/5th of the total length.
        - total (int): length of the iterable, in case it does not support
            `len`.
        - name (str): prefix to use in the log.
        - level: logging level (like `logging.INFO`).
    """
    def __init__(self,
                 logger,
                 iterable,
                 updates=5,
                 total=None,
                 name="LogProgress",
                 level=logging.INFO):
        self.iterable = iterable
        self.total = total or len(iterable)
        self.updates = updates
        self.name = name
        self.logger = logger
        self.level = level

    def update(self, **infos):
        self._infos = infos
    
    def append(self, **infos):
        self._infos.update(**infos)
        
    def _append(self, info):
        self._infos.update(info)

    def __iter__(self):
        self._iterator = iter(self.iterable)
        self._index = -1
        self._infos = {}
        self._begin = time.time()
        return self

    def __next__(self):
        self._index += 1
        try:
            value = next(self._iterator)
        except StopIteration:
            raise
        else:
            return value
        finally:
            log_every = max(1, self.total // self.updates)
            # logging is delayed by 1 it, in order to have the metrics from update
            if self._index >= 1 and self._index % log_every == 0:
                self._log()

    def _log(self):
        self._speed = (1 + self._index) / (time.time() - self._begin)
        infos = " | ".join(f"{k} {v}" for k, v in self._infos.items())
        if self._speed < 1e-4:
            speed = "oo sec/it"
        elif self._speed < 0.1:
            speed = f"{1/self._speed:.1f} sec/it"
        else:
            speed = f"{self._speed:.1f} it/sec"
        out = f"{self.name} | {self._index}/{self.total} | {speed}"
        if infos:
            out += " | " + infos
        self.logger.log(self.level, out)


def colorize(text, color):
    """
    Display text with some ANSI color in the terminal.
    """
    code = f"\033[{color}m"
    restore = "\033[0m"
    return "".join([code, text, restore])


def bold(text):
    """
    Display text in bold in the terminal.
    """
    return colorize(text, "1")


# ============================================================================
# Model and Checkpoint Utilities
# ============================================================================

def load_model(model_params: Dict[str, Any], device: str = 'cuda'):
    """
    Create Backbone model instance.

    Args:
        model_params: Model parameters dictionary
        device: Device to load model on ('cuda' or 'cpu')

    Returns:
        Backbone instance loaded on specified device

    Example:
        >>> model = load_model(params, "cuda")
    """
    from omegaconf import OmegaConf
    import inspect

    if not isinstance(model_params, dict):
        model_params = OmegaConf.to_container(model_params, resolve=True)

    model_type = model_params.get("type", "backbone")

    if model_type == "backbone":
        from src.models.backbone import Backbone
        model_cls = Backbone
    elif model_type == "latency_expert_backbone":
        from src.models.latency_expert import LatencyExpertBackbone
        model_cls = LatencyExpertBackbone
    else:
        raise ValueError(
            f"Unknown model.type '{model_type}'. "
            "Expected 'backbone' or 'latency_expert_backbone'."
        )

    valid_keys = inspect.signature(model_cls.__init__).parameters
    filtered = {
        k: v for k, v in model_params.items()
        if k in valid_keys and k != "type"
    }
    model = model_cls(**filtered)
    return model.to(device)


def load_checkpoint(model: torch.nn.Module, chkpt_dir: str, chkpt_file: str, device: str = 'cuda'):
    """
    Load model checkpoint.

    Args:
        model: Model instance to load checkpoint into
        chkpt_dir: Directory containing checkpoint file
        chkpt_file: Checkpoint filename (e.g., "best.th", "checkpoint.th")
        device: Device for loading checkpoint

    Returns:
        Model with loaded checkpoint

    Example:
        >>> model = load_checkpoint(model, "outputs/exp", "best.th", "cuda")
    """
    chkpt_path = os.path.join(chkpt_dir, chkpt_file)
    chkpt = torch.load(chkpt_path, map_location=device, weights_only=False)
    model.load_state_dict(chkpt['model'])
    return model


class ConfigDict:
    """
    Convert nested dictionary to object with attribute access.

    Useful for converting checkpoint args to object-like structure.

    Example:
        >>> config = ConfigDict({'model': {'name': 'test', 'params': {'lr': 0.01}}})
        >>> config.model.name  # 'test'
        >>> config.model.params.lr  # 0.01
        >>> config.to_dict()  # {'model': {'name': 'test', 'params': {'lr': 0.01}}}
    """

    def __init__(self, d: Dict[str, Any]):
        for key, value in d.items():
            setattr(self, key, value if not isinstance(value, dict) else ConfigDict(value))

    def to_dict(self) -> Dict[str, Any]:
        """Convert ConfigDict back to regular dict for ** unpacking"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, ConfigDict):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result


def load_model_config_from_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """
    Load model configuration from checkpoint file.

    Args:
        checkpoint_path: Path to checkpoint file (e.g., "outputs/exp/best.th")

    Returns:
        Dictionary of model parameters (e.g., {'dense_channel': 64, ...})

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        KeyError: If checkpoint doesn't contain 'args' or 'args.model'

    Example:
        >>> config = load_model_config_from_checkpoint("outputs/backbone_exp/best.th")
        >>> print(config['dense_channel'])  # 64
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # Load checkpoint (CPU is sufficient for config extraction)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if 'args' not in checkpoint:
        raise KeyError(f"Checkpoint does not contain 'args' field: {checkpoint_path}")

    args = checkpoint['args']

    if not hasattr(args, 'model'):
        raise KeyError(f"Checkpoint args does not contain 'model' field: {checkpoint_path}")

    model_config = args.model

    return dict(model_config)


def parse_file_list(directory: str, list_file: str) -> List[str]:
    """
    Parse file list and return full paths.

    Args:
        directory: Base directory path
        list_file: Path to file containing list of filenames (one per line)

    Returns:
        List of full file paths

    Example:
        >>> noise_files = parse_file_list("data/noise", "dataset/noise_train.txt")
    """
    with open(list_file, 'r') as f:
        return [os.path.join(directory, line.strip()) for line in f]


def get_stft_args_from_config(model_args, sample_rate: Optional[int] = None) -> Dict[str, Any]:
    """
    Extract STFT arguments from model configuration.

    Args:
        model_args: Model configuration object (args.model)

    Returns:
        Dictionary of STFT arguments

    Example:
        >>> stft_args = get_stft_args_from_config(config.model)
        >>> # Returns: {"n_fft": 400, "hop_size": 100, "win_size": 400, "compress_factor": 0.3}
    """
    sfi_config = model_args.get("sfi", {})
    if sfi_config and sfi_config.get("enabled", False):
        from src.v2_contract import sfi_profile_from_config

        active_sample_rate = int(
            sample_rate
            if sample_rate is not None
            else sfi_config.get("reference_sample_rate", 16000)
        )
        profile = sfi_profile_from_config(sfi_config, active_sample_rate)
        return profile.as_stft_args(
            compress_factor=model_args.get("compress_factor", 1.0)
        )

    fft_len = model_args.fft_len
    return {
        "n_fft": fft_len,
        "hop_size": model_args.get("hop_len", fft_len // 4),
        "win_size": model_args.get("win_len", fft_len),
        "compress_factor": model_args.get("compress_factor", 1.0),
    }


def compute_lookahead_frames(padding_ratio, depth=4) -> int:
    """Compute total right (future) padding frames for a DS_DDB block.

    DS_DDB uses depth layers with kernel_size=3 and dilations=[1,2,4,8].
    Each layer: time_padding_total = 2 * dilation.
    Matches AsymmetricConv2d rounding logic (Python banker's rounding).

    Args:
        padding_ratio: (left_ratio, right_ratio) tuple that sums to 1.0
        depth: Number of dilated layers (default: 4)

    Returns:
        Total right padding frames (lookahead requirement)
    """
    left_ratio, right_ratio = padding_ratio
    total_right = 0
    for i in range(depth):
        dilation = 2 ** i
        time_padding_total = dilation * 2
        left = round(time_padding_total * left_ratio)
        right = round(time_padding_total * right_ratio)
        if left + right != time_padding_total:
            right = time_padding_total - left
        total_right += right
    return total_right


def compute_lookahead_from_config(model_config) -> Tuple[int, int]:
    """Compute encoder/decoder lookahead frames from model config.

    Args:
        model_config: OmegaConf model config node (or object with attrs)

    Returns:
        (encoder_lookahead, decoder_lookahead) in frames
    """
    latency_config = model_config.get("latency_expert", {})
    future_frames = latency_config.get("future_frames") if latency_config else None
    if future_frames is not None:
        latency_ids = list(latency_config.get("latency_ids", future_frames.keys()))
        latency_id = latency_config.get("default_latency_id", latency_ids[0])
        topology = latency_config.get("topology", "decoder_only")
        if topology != "decoder_only":
            raise ValueError(
                "Explicit v2 future_frames currently supports decoder_only topology"
            )
        return 0, int(future_frames[latency_id])

    depth = getattr(model_config, 'dense_depth', 4)
    enc_ratio = list(getattr(model_config, 'encoder_padding_ratio', [0.5, 0.5]))
    dec_ratio = list(getattr(model_config, 'decoder_padding_ratio', [0.5, 0.5]))
    enc_la = compute_lookahead_frames(enc_ratio, depth)
    dec_la = compute_lookahead_frames(dec_ratio, depth)
    return enc_la, dec_la
