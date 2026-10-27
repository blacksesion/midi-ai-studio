"""Family-aware expressive shaping for non-piano instruments.

The neural model is trained on piano (MAESTRO). For other families we apply
musically-informed shaping: legato for strings/winds, accents, groove for
drums, register-based dynamics, and gentle micro-timing humanisation.
"""
from __future__ import annotations

import numpy as np

from .instruments import family_of


def _rng(seed):
    return np.random.default_rng(seed)


def shape_drums(track, song, strength, rng):
    notes = track.notes
    if not notes:
        return
    for n in notes:
        # velocity humanisation
        v = n.velocity + rng.integers(-6, 7) * strength
        n.velocity = int(min(127, max(1, v)))
        # micro-timing swing: push 16ths slightly
        beat = n.start * song.bpm / 60.0
        frac = beat % 0.25
        if 0.20 < frac < 0.30:
            n.start = max(0.0, n.start + 0.012 * strength)
            n.end = n.start + n.duration


def shape_melodic(track, song, strength, rng, family):
    notes = track.notes
    if not notes:
        return
    pitches = np.array([n.pitch for n in notes], dtype=float)
    # register-based dynamics: higher notes slightly louder
    reg = (pitches - 60.0) / 24.0
    legato_families = {"strings", "ensemble", "brass", "reed", "pipe", "synth_lead", "synth_pad"}
    is_legato = family in legato_families

    for i, n in enumerate(notes):
        # dynamics curve
        dyn = 1.0 + 0.08 * reg[i] * strength
        n.velocity = int(min(127, max(1, round(n.velocity * dyn))))
        # micro-timing: slight anticipation on downbeats
        beat = n.start * song.bpm / 60.0
        if abs(beat - round(beat)) < 1e-6:
            n.start = max(0.0, n.start - 0.008 * strength)
        # articulation
        if is_legato:
            n.end = max(n.end, n.start + n.duration * (1 + 0.15 * strength))
        else:
            # slight shortening for percussive
            n.end = n.start + max(0.02, n.duration * (1 - 0.05 * strength))


def shape_track(track, song, strength, seed):
    rng = _rng(seed)
    if track.is_drums:
        shape_drums(track, song, strength, rng)
        return
    fam = family_of(track.program)
    if fam == "piano":
        return  # handled by neural model
    shape_melodic(track, song, strength, rng, fam)