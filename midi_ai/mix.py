"""Mixing and mastering of rendered stems.

Combines per-track stems onto a virtual stage (pan / level / width / reverb
send), applies a convolution reverb, then a mastering chain (EQ, bus
compression, saturation, brickwall limiter, loudness normalisation).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import dsp
from .instruments import preset_of


@dataclass
class MixOptions:
    reverb: float = 0.30        # master reverb amount 0..1
    brightness: float = 0.0     # master high-shelf (dB)
    warmth: float = 0.0         # master low-shelf (dB)
    width: float = 0.0          # master stereo width 0..1
    loudness_db: float = -15.0  # target RMS in dBFS
    saturation: float = 0.0     # tape saturation drive (dB), 0 disables


def _pad_to(stem: np.ndarray, length: int) -> np.ndarray:
    if stem.shape[0] < length:
        return np.pad(stem, ((0, length - stem.shape[0]), (0, 0)))
    return stem[:length]


def mix_and_master(
    stems: list[np.ndarray],
    tracks,
    samplerate: int,
    ir: np.ndarray,
    options: MixOptions | None = None,
) -> np.ndarray:
    options = options or MixOptions()
    sr = samplerate

    if not stems:
        return np.zeros((sr, 2), dtype=np.float32)

    max_len = max(s.shape[0] for s in stems)
    dry = np.zeros((max_len, 2), dtype=np.float32)
    reverb_bus = np.zeros((max_len, 2), dtype=np.float32)

    for i, (stem, track) in enumerate(zip(stems, tracks)):
        preset = preset_of(track.program, track.is_drums)
        stem = _pad_to(stem, max_len).astype(np.float32)

        # Pan.
        pan_amount = 0.0 if track.is_drums else preset.pan
        stem = dsp.pan(stem, pan_amount)
        # Level trim.
        stem = stem * dsp.db_to_amp(preset.level_db)
        # Per-instrument stereo width.
        if preset.width > 0:
            stem = dsp.stereo_widen(stem, preset.width)

        dry += stem
        send = preset.reverb * options.reverb
        if send > 0:
            reverb_bus += stem * send

    # Convolution reverb (trim to dry length).
    wet = np.zeros_like(dry)
    if np.max(np.abs(reverb_bus)) > 1e-9:
        wet = dsp.convolve_stereo(reverb_bus, ir)[:max_len]
        wet = wet.astype(np.float32)
        # Normalise wet so it doesn't overwhelm.
        wet_rms = dsp.rms(wet)
        if wet_rms > 1e-9:
            wet = wet * (dsp.rms(dry) / wet_rms) * 0.9

    mix = dry + wet

    # ---- mastering chain -------------------------------------------------
    # Gentle high-pass to remove sub-sonic rumble.
    mix = dsp.apply_eq(mix, sr, [(28.0, "highpass", 0.707)])
    # Low shelf warmth.
    if options.warmth:
        mix = dsp.apply_eq(mix, sr, [(120.0, "lowshelf", 0.707, options.warmth)])
    # High shelf brightness / air.
    if options.brightness:
        mix = dsp.apply_eq(mix, sr, [(9000.0, "highshelf", 0.707, options.brightness)])

    # Bus compression.
    mix = dsp.compressor(mix, sr, threshold_db=-20.0, ratio=2.0,
                         attack_ms=25.0, release_ms=200.0, makeup_db=0.0)

    # Saturation.
    if options.saturation > 0:
        mix = dsp.soft_clip(mix, options.saturation)

    # Master stereo width.
    if options.width > 0:
        mix = dsp.stereo_widen(mix, options.width)

    # Loudness + limiting.
    mix = dsp.limiter(mix, sr, ceiling_db=-0.3, lookahead_ms=4.0)
    target_rms = dsp.db_to_amp(options.loudness_db)
    mix = dsp.normalize_rms(mix, target_rms=target_rms, peak_ceiling=0.97)

    return mix.astype(np.float32)
