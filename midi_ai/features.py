"""Feature extraction for the expressive-performance model.

Features describe each note's *musical context* so a model can predict how a
human performer would shape it (dynamics, micro-timing, articulation). They are
computed identically for training data and for arbitrary input MIDI.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FEATURE_NAMES = [
    "pitch", "pitch_class", "beat_in_bar", "downbeat",
    "log_dur", "d_pitch_prev", "d_onset_prev", "d_pitch_next", "d_onset_next",
    "polyphony", "bar_pos", "piece_pos", "prev_dur",
]
N_FEATURES = len(FEATURE_NAMES)

GRID = 0.25  # 16th-note grid in quarter-note beats


@dataclass
class MechNote:
    start: float
    end: float
    pitch: int


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def note_features(
    notes: list,
    bpm: float,
    beats_per_bar: int = 4,
    total_beats: float | None = None,
) -> np.ndarray:
    """Compute an (N, N_FEATURES) float32 feature matrix for a list of notes.

    ``notes`` must be sorted by start time and expose ``.start`` (seconds),
    ``.end`` (seconds) and ``.pitch``. Features are computed from the notes as
    given (the "mechanical" reference); the model predicts expressive offsets.
    """
    n = len(notes)
    if n == 0:
        return np.zeros((0, N_FEATURES), dtype=np.float32)

    starts = np.array([x.start for x in notes], dtype=np.float64)
    ends = np.array([x.end for x in notes], dtype=np.float64)
    pitches = np.array([x.pitch for x in notes], dtype=np.float64)

    onset_beats = starts * bpm / 60.0
    dur_beats = np.maximum((ends - starts) * bpm / 60.0, 1e-4)

    if total_beats is None or total_beats <= 0:
        total_beats = float(onset_beats[-1] + 1.0)

    bpb = max(1, beats_per_bar)
    bar_index = np.floor(onset_beats / bpb)
    beat_in_bar = onset_beats - bar_index * bpb

    # Polyphony: number of notes sounding at each onset (including itself).
    # = count(start <= start_i) - count(end <= start_i)
    started_before = np.searchsorted(starts, starts, side="right")
    ended_before_or_at = np.searchsorted(ends, starts, side="left")
    polyphony = (started_before - ended_before_or_at).astype(np.float64)

    # Previous / next neighbour indices.
    prev_idx = np.arange(n) - 1
    next_idx = np.arange(n) + 1
    prev_idx = np.clip(prev_idx, 0, n - 1)
    next_idx = np.clip(next_idx, 0, n - 1)

    d_pitch_prev = _clip_vec((pitches - pitches[prev_idx]) / 24.0, -1.0, 1.0)
    d_pitch_next = _clip_vec((pitches[next_idx] - pitches) / 24.0, -1.0, 1.0)
    d_onset_prev = _clip_vec((onset_beats - onset_beats[prev_idx]) / 4.0, 0.0, 1.0)
    d_onset_next = _clip_vec((onset_beats[next_idx] - onset_beats) / 4.0, 0.0, 1.0)

    downbeat = np.zeros(n, dtype=np.float64)
    downbeat[beat_in_bar < 1e-6] = 1.0
    on_beat = np.abs(beat_in_bar - np.round(beat_in_bar)) < 1e-6
    downbeat[on_beat & (beat_in_bar >= 1e-6)] = 0.5

    feats = np.column_stack([
        pitches / 127.0,
        (pitches % 12.0) / 12.0,
        beat_in_bar / bpb,
        downbeat,
        np.log1p(dur_beats),
        d_pitch_prev,
        d_onset_prev,
        d_pitch_next,
        d_onset_next,
        _clip_vec(polyphony / 8.0, 0.0, 1.0),
        np.mod(bar_index, 8.0) / 8.0,
        _clip_vec(onset_beats / total_beats, 0.0, 1.0),
        np.log1p(np.maximum(dur_beats[prev_idx], 1e-4)),
    ]).astype(np.float32)

    return feats


def _clip_vec(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(x, lo, hi)


def quantize_to_grid(notes, bpm: float, grid: float = GRID) -> list:
    """Build a mechanical (quantised) reference from a list of notes.

    Returns a list of :class:`MechNote` objects suitable for :func:`note_features`.
    """
    out = []
    for n in notes:
        onset_beat = n.start * bpm / 60.0
        mech_beat = round(onset_beat / grid) * grid
        mech_start = mech_beat * 60.0 / bpm
        dur_beats = max(grid, round(max(1e-4, (n.end - n.start) * bpm / 60.0) / grid) * grid)
        mech_dur = dur_beats * 60.0 / bpm
        out.append(MechNote(mech_start, mech_start + mech_dur, n.pitch))
    return out


def piece_velocity_stats(velocities: np.ndarray) -> tuple[float, float]:
    """Per-piece log-velocity statistics used to normalise the v_z target."""
    v = np.log1p(np.asarray(velocities, dtype=np.float64))
    mu = float(v.mean())
    sigma = float(v.std()) + 1e-6
    return mu, sigma
