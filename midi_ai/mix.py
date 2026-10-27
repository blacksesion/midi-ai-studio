"""Mixing and mastering of rendered stems.

Combines per-track stems onto a virtual stage (pan / level / width / reverb
send), applies a convolution reverb, then a mastering chain (EQ, bus
compression, saturation, brickwall limiter, loudness normalisation).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import dsp
from .instruments import preset_of


@dataclass
class MixOptions:
    reverb: float = 0.30        # master reverb amount 0..1
    reverb_size: float = 0.5    # 0..1 hall size
    brightness: float = 0.0     # master high-shelf (dB)
    warmth: float = 0.0         # master low-shelf (dB)
    width: float = 0.0          # master stereo width 0..1
    loudness_db: float = -15.0  # target LUFS
    saturation: float = 0.0     # tape saturation drive (dB), 0 disables
    multiband: bool = False     # multiband compression
    deesser: bool = False       # de-essing


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

        pan_amount = 0.0 if track.is_drums else preset.pan
        stem = dsp.pan(stem, pan_amount)
        stem = stem * dsp.db_to_amp(preset.level_db)
        if preset.width > 0:
            stem = dsp.stereo_widen(stem, preset.width)
        if getattr(preset, "bright_db", 0.0):
            stem = dsp.apply_eq(stem, sr, [(4000.0, "highshelf", 0.707, preset.bright_db)])

        dry += stem
        send = preset.reverb * options.reverb
        if send > 0:
            reverb_bus += stem * send

    ir_len = ir.shape[0] if ir.ndim >= 1 else 1
    total_len = max_len + max(0, ir_len - 1)
    dry = _pad_to(dry, total_len).astype(np.float32)
    reverb_bus = _pad_to(reverb_bus, total_len).astype(np.float32)
    wet = np.zeros((total_len, 2), dtype=np.float32)
    if np.max(np.abs(reverb_bus)) > 1e-9:
        wet = dsp.convolve_stereo(reverb_bus, ir)
        wet = _pad_to(wet, total_len).astype(np.float32)
        wet_rms = dsp.rms(wet)
        dry_rms = dsp.rms(dry)
        if wet_rms > 1e-9:
            wet = wet * (dry_rms / wet_rms) * 0.9

    mix = dry + wet

    mix = dsp.apply_eq(mix, sr, [(28.0, "highpass", 0.707)])
    if options.warmth:
        mix = dsp.apply_eq(mix, sr, [(120.0, "lowshelf", 0.707, options.warmth)])
    if options.brightness:
        mix = dsp.apply_eq(mix, sr, [(9000.0, "highshelf", 0.707, options.brightness)])

    if getattr(options, "multiband", False):
        mix = dsp.multiband_compressor(mix, sr, threshold_db=-20.0, ratio=2.0)
    else:
        mix = dsp.compressor(mix, sr, threshold_db=-20.0, ratio=2.0,
                              attack_ms=25.0, release_ms=200.0, makeup_db=0.0)
    if getattr(options, "deesser", False):
        mix = dsp.deesser(mix, sr)

    if options.saturation > 0:
        mix = dsp.soft_clip(mix, options.saturation)

    if options.width > 0:
        mix = dsp.stereo_widen(mix, options.width)

    mix = dsp.master_loudness(mix, sr, target_lufs=options.loudness_db, ceiling_db=-1.0)

    return mix.astype(np.float32)