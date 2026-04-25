"""Paired-wav dataset for cross-dataset evaluation (e.g., DNS Challenge).

Mirrors the output signature of ``VoiceBankDataset`` so that the existing
``enhance`` / ``evaluate`` pipelines work unchanged. Intended for any evaluation
set laid out as::

    <root>/noisy/<id>.wav
    <root>/clean/<id>.wav

DNS Challenge 2020 non-blind test set (synthetic/no_reverb) is supported via
its own id-based naming (``noisy_fileid_*.wav`` / ``clean_fileid_*.wav``): pass
``noisy_glob='noisy_fileid_*.wav'`` and ``id_regex`` accordingly, or use the
``layout='dns2020'`` preset.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import soundfile as sf
import torch
import torch.utils.data


def _load_wav(path: str, target_sr: int = 16000) -> np.ndarray:
    data, sr = sf.read(path, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != target_sr:
        raise ValueError(f"Expected sr={target_sr}, got {sr} for {path}")
    return data.astype("float32")


class PairedWavDataset(torch.utils.data.Dataset):
    """Paired (noisy, clean) waveform dataset from two parallel directories.

    Args:
        noisy_dir: directory containing noisy wavs.
        clean_dir: directory containing clean wavs. Matching is done by shared
            id (filename stem by default, or via ``noisy_regex``/``clean_fmt``).
        sampling_rate: expected sampling rate (default 16000).
        power_normalize: apply the same power normalization used by
            ``VoiceBankDataset`` — required to match training-time distribution.
        layout: optional preset. ``'simple'`` matches by filename stem;
            ``'dns2020'`` pairs ``noisy_fileid_<N>.wav`` with
            ``clean_fileid_<N>.wav`` (DNS Challenge 2020 synthetic/no_reverb).
        with_id / with_text: returned tuple shape matches ``VoiceBankDataset``.
    """

    def __init__(
        self,
        noisy_dir: str,
        clean_dir: str,
        sampling_rate: int = 16000,
        power_normalize: bool = True,
        layout: str = "simple",
        with_id: bool = False,
        with_text: bool = False,
        with_norm_factor: bool = False,
    ):
        self.sampling_rate = sampling_rate
        self.power_normalize = power_normalize
        self.with_id = with_id
        self.with_text = with_text
        self.with_norm_factor = with_norm_factor
        self.noisy_dir = Path(noisy_dir)
        self.clean_dir = Path(clean_dir)

        self.pairs: List[Tuple[Path, Path, str]] = self._index(layout)
        if not self.pairs:
            raise RuntimeError(
                f"No paired files found. noisy_dir={noisy_dir}, clean_dir={clean_dir}, layout={layout}"
            )

    def _index(self, layout: str) -> List[Tuple[Path, Path, str]]:
        if layout == "simple":
            return self._index_by_stem()
        if layout == "dns2020":
            return self._index_dns2020()
        raise ValueError(f"Unknown layout '{layout}'. Use 'simple' or 'dns2020'.")

    def _index_by_stem(self) -> List[Tuple[Path, Path, str]]:
        clean_map = {p.stem: p for p in self.clean_dir.glob("*.wav")}
        out = []
        for noisy_path in sorted(self.noisy_dir.glob("*.wav")):
            stem = noisy_path.stem
            if stem not in clean_map:
                continue
            out.append((noisy_path, clean_map[stem], stem))
        return out

    def _index_dns2020(self) -> List[Tuple[Path, Path, str]]:
        pat = re.compile(r"fileid_(\d+)")
        clean_map = {}
        for p in self.clean_dir.glob("clean_fileid_*.wav"):
            m = pat.search(p.name)
            if m:
                clean_map[m.group(1)] = p
        out = []
        # Noisy names vary: "noisy_fileid_<N>.wav" or "clnsp<N>_..._fileid_<N>.wav".
        # Match by fileid parsed from the filename.
        for noisy_path in sorted(self.noisy_dir.glob("*fileid_*.wav")):
            m = pat.search(noisy_path.name)
            if not m:
                continue
            fid = m.group(1)
            if fid not in clean_map:
                continue
            out.append((noisy_path, clean_map[fid], f"fileid_{fid}"))
        return out

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        noisy_path, clean_path, sample_id = self.pairs[index]
        noisy = _load_wav(str(noisy_path), self.sampling_rate)
        clean = _load_wav(str(clean_path), self.sampling_rate)

        # Align length defensively (DNS pairs are same length, but be safe).
        min_len = min(len(noisy), len(clean))
        noisy = noisy[:min_len]
        clean = clean[:min_len]

        if self.power_normalize:
            norm_factor = float(np.sqrt(noisy.shape[-1] / max(np.sum(noisy ** 2.0), 1e-12)))
            noisy = noisy * norm_factor
            clean = clean * norm_factor
        else:
            norm_factor = 1.0

        noisy_t = torch.from_numpy(noisy.copy())
        clean_t = torch.from_numpy(clean.copy())

        if self.with_text:
            base = (noisy_t, clean_t, sample_id, "")
        elif self.with_id:
            base = (noisy_t, clean_t, sample_id)
        else:
            base = (noisy_t, clean_t)
        if self.with_norm_factor:
            base = (*base, norm_factor)
        return base


class UnpairedWavDataset(torch.utils.data.Dataset):
    """Noisy-only dataset for no-reference metrics (e.g., DNSMOS on blind sets).

    Returns ``(noisy, id)`` tuples. Pairs a dummy ``clean`` tensor (zeros) so the
    signature remains compatible with pipelines expecting ``(noisy, clean, id, _)``;
    downstream consumers should skip intrusive metrics when ``clean`` is all zero.
    """

    def __init__(
        self,
        noisy_dir: str,
        sampling_rate: int = 16000,
        power_normalize: bool = True,
        with_id: bool = True,
        with_text: bool = True,
        with_norm_factor: bool = False,
    ):
        self.noisy_dir = Path(noisy_dir)
        self.sampling_rate = sampling_rate
        self.power_normalize = power_normalize
        self.with_id = with_id
        self.with_text = with_text
        self.with_norm_factor = with_norm_factor
        self.files = sorted(self.noisy_dir.glob("*.wav"))
        if not self.files:
            raise RuntimeError(f"No wavs in {noisy_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        path = self.files[index]
        noisy = _load_wav(str(path), self.sampling_rate)
        if self.power_normalize:
            norm_factor = float(np.sqrt(noisy.shape[-1] / max(np.sum(noisy ** 2.0), 1e-12)))
            noisy = noisy * norm_factor
        else:
            norm_factor = 1.0
        noisy_t = torch.from_numpy(noisy.copy())
        clean_t = torch.zeros_like(noisy_t)
        sample_id = path.stem
        if self.with_text:
            base = (noisy_t, clean_t, sample_id, "")
        elif self.with_id:
            base = (noisy_t, clean_t, sample_id)
        else:
            base = (noisy_t, clean_t)
        if self.with_norm_factor:
            base = (*base, norm_factor)
        return base
