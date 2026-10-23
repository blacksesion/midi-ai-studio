"""End-to-end rendering pipeline.

Ties together parsing, the expressive-performance model, SoundFont rendering
and mixing/mastering. Both the CLI and the web UI drive this module.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np
import soundfile as sf

from . import dsp
from .assets import ensure_model, ensure_soundfont
from .features import note_features, piece_velocity_stats
from .midi_parse import Song, load_midi, summarize
from .mix import MixOptions, mix_and_master
from .render import DEFAULT_SAMPLERATE, render_song_stems


@dataclass
class RenderOptions:
    ai_strength: float = 0.6        # 0 = untouched, 1 = fully expressive
    humanize: bool = True
    reverb: float = 0.30            # master reverb 0..1
    brightness: float = 0.0         # dB
    warmth: float = 0.0             # dB
    width: float = 0.0              # 0..1
    loudness_db: float = -15.0      # target RMS
    saturation: float = 0.5         # dB tape saturation
    sample_rate: int = DEFAULT_SAMPLERATE
    seed: int | None = 42
    use_model: bool = True

    def mix_options(self) -> MixOptions:
        return MixOptions(
            reverb=self.reverb,
            brightness=self.brightness,
            warmth=self.warmth,
            width=self.width,
            loudness_db=self.loudness_db,
            saturation=self.saturation,
        )


@dataclass
class RenderResult:
    audio: np.ndarray
    sample_rate: int
    duration: float
    tracks: list[dict]
    bpm: float
    elapsed: float
    model_used: bool
    notes_processed: int


def _apply_model(song: Song, model, strength: float, seed: int) -> int:
    """Humanise non-drum tracks in place; returns number of notes processed."""
    rng = np.random.default_rng(seed)
    processed = 0
    for track in song.tracks:
        if track.is_drums or not track.notes:
            continue
        notes = track.notes
        feats = note_features(notes, song.bpm, song.beats_per_bar,
                              total_beats=song.total_seconds * song.bpm / 60.0)
        pred = model.predict(feats, sample=True, rng=rng)
        v_z, onset_dev, dur_log = pred[:, 0], pred[:, 1], pred[:, 2]

        vel = np.array([n.velocity for n in notes], dtype=np.float64)
        mu, sigma = piece_velocity_stats(vel)

        for i, n in enumerate(notes):
            # Expressive velocity (denormalised from z-score).
            v_pred = np.exp(mu + v_z[i] * sigma) - 1.0
            v_pred = int(min(127, max(1, round(v_pred))))
            n.velocity = int(round(n.velocity * (1 - strength) + v_pred * strength))

            # Micro-timing (predicted in beats -> seconds).
            delta = onset_dev[i] * 60.0 / song.bpm * strength
            n.start = max(0.0, n.start + delta)

            # Articulation.
            ratio = np.exp(dur_log[i] * strength)
            dur = max(0.02, n.duration * ratio)
            n.end = n.start + dur
        processed += len(notes)
    return processed


def render_song(
    song: Song,
    options: RenderOptions | None = None,
    model=None,
    sf_path: str | None = None,
    ir: np.ndarray | None = None,
) -> RenderResult:
    opts = options or RenderOptions()
    t0 = time.perf_counter()

    model_used = False
    notes_processed = 0
    if opts.humanize and opts.ai_strength > 0 and model is not None:
        notes_processed = _apply_model(song, model, opts.ai_strength, opts.seed or 42)
        model_used = True

    sf_path = sf_path or ensure_soundfont()
    stems = render_song_stems(song, sf_path, samplerate=opts.sample_rate)

    if ir is None:
        ir = dsp.make_ir(opts.sample_rate, rt60=1.5 + 2.0 * opts.reverb,
                         pre_delay_ms=18.0, seed=7)

    audio = mix_and_master(stems, song.tracks, opts.sample_rate, ir, opts.mix_options())

    return RenderResult(
        audio=audio,
        sample_rate=opts.sample_rate,
        duration=float(len(audio) / opts.sample_rate),
        tracks=summarize(song),
        bpm=round(song.bpm, 2),
        elapsed=time.perf_counter() - t0,
        model_used=model_used,
        notes_processed=notes_processed,
    )


def _load_model():
    mp = ensure_model()
    if not mp:
        return None
    from .model import PerformanceModel
    return PerformanceModel.load(mp)


_MODEL_CACHE = {"loaded": False, "model": None}


def get_model():
    if not _MODEL_CACHE["loaded"]:
        _MODEL_CACHE["loaded"] = True
        _MODEL_CACHE["model"] = _load_model()
    return _MODEL_CACHE["model"]


def render_file(
    midi_path: str,
    out_path: str | None = None,
    options: RenderOptions | None = None,
) -> RenderResult:
    """Load, render and (optionally) write a MIDI file to WAV."""
    opts = options or RenderOptions()
    song = load_midi(midi_path)

    model = None
    if opts.use_model and opts.humanize and opts.ai_strength > 0:
        model = get_model()

    result = render_song(song, opts, model=model)

    if out_path:
        sf.write(out_path, result.audio, result.sample_rate, subtype="PCM_16")
    return result
