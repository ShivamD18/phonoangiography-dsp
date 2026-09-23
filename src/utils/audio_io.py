"""Standardized audio loading and inspection helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def load_audio(path: str | Path, mono: bool = True) -> tuple[np.ndarray, int]:
    """Load an audio file as float samples and return ``(samples, sample_rate)``."""
    samples, sample_rate = sf.read(Path(path), always_2d=False, dtype="float32")
    if mono and samples.ndim == 2:
        samples = samples.mean(axis=1)
    return np.asarray(samples), int(sample_rate)


def inspect_audio(path: str | Path) -> dict[str, int | float | str]:
    """Return basic metadata without loading the complete waveform."""
    info = sf.info(Path(path))
    return {
        "path": str(path),
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "frames": info.frames,
        "duration_seconds": info.duration,
        "format": info.format,
        "subtype": info.subtype,
    }
