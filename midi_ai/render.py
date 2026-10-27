"""SoundFont-based rendering via TinySoundFont.

The SoundFont is loaded *once* (decompression is the dominant cost) and each
track is rendered to its own stereo stem with sample-accurate note/controller
scheduling. Stems are mixed/mastered elsewhere.
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import tinysoundfont

from .midi_parse import Track

DEFAULT_SAMPLERATE = 44100
CHANNELS = 2

_synth_cache: dict[tuple, tinysoundfont.Synth] = {}
_synth_lock = threading.RLock()
_thread_local = threading.local()


def make_synth(sf_path: str, samplerate: int = DEFAULT_SAMPLERATE,
               synth_gain_db: float = 6.0) -> tuple[tinysoundfont.Synth, int]:
    synth = tinysoundfont.Synth(gain=synth_gain_db, samplerate=samplerate)
    sfid = synth.sfload(sf_path, max_voices=512)
    return synth, sfid


def get_synth(sf_path: str, samplerate: int = DEFAULT_SAMPLERATE,
              synth_gain_db: float = 6.0) -> tuple[tinysoundfont.Synth, int]:
    """Return a cached Synth + sfid (SoundFont decompression is expensive)."""
    key = (os.path.abspath(sf_path), samplerate, synth_gain_db)
    with _synth_lock:
        if key not in _synth_cache:
            _synth_cache[key] = make_synth(sf_path, samplerate, synth_gain_db)
        synth, sfid = _synth_cache[key]
        return synth, sfid


def _read(synth: tinysoundfont.Synth, samples: int) -> np.ndarray:
    if samples <= 0:
        return np.zeros((0, CHANNELS), dtype=np.float32)
    buf = synth.generate_simple(samples)
    arr = np.frombuffer(buf, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((samples, CHANNELS), dtype=np.float32)
    arr = arr.reshape(-1, CHANNELS).copy()
    if arr.shape[0] < samples:
        arr = np.pad(arr, ((0, samples - arr.shape[0]), (0, 0)))
    return arr


def render_track_stem(
    synth: tinysoundfont.Synth,
    sfid: int,
    track: Track,
    samplerate: int = DEFAULT_SAMPLERATE,
    tail_seconds: float = 2.5,
) -> np.ndarray:
    """Render one track to a stereo float32 stem (samples, 2), reusing ``synth``."""
    synth.sounds_off()  # silence any leftover voices from a previous track
    synth.program_select(0, sfid, 0, track.program, is_drums=track.is_drums)
    synth.pitchbend_range(0, 2.0)

    sr = samplerate
    events: list[tuple[int, str, tuple]] = []
    for n in track.notes:
        on = int(round(n.start * sr))
        off = int(round(n.end * sr))
        if off <= on:
            off = on + 1
        vel = int(min(127, max(1, n.velocity)))
        events.append((on, "on", (n.pitch, vel)))
        events.append((off, "off", (n.pitch,)))
    for sec, cc, val in track.controls:
        idx = int(round(sec * sr))
        events.append((idx, "cc", (cc, val)))

    # Sort by time; for ties: note_off (0) < cc (1) < note_on (2).
    prio = {"off": 0, "cc": 1, "on": 2}
    events.sort(key=lambda e: (e[0], prio.get(e[1], 1)))

    chunks: list[np.ndarray] = []
    cur = 0
    for idx, kind, args in events:
        if idx > cur:
            chunks.append(_read(synth, idx - cur))
            cur = idx
        if kind == "on":
            synth.noteon(0, args[0], args[1])
        elif kind == "off":
            synth.noteoff(0, args[0])
        else:  # cc
            synth.control_change(0, args[0], args[1])

    chunks.append(_read(synth, int(round(tail_seconds * sr))))

    return np.concatenate(chunks, axis=0).astype(np.float32)


def _thread_synth(sf_path: str, samplerate: int, synth_gain_db: float):
    key = (os.path.abspath(sf_path), samplerate, synth_gain_db)
    if getattr(_thread_local, "key", None) != key:
        _thread_local.synth, _thread_local.sfid = make_synth(sf_path, samplerate, synth_gain_db)
        _thread_local.key = key
    return _thread_local.synth, _thread_local.sfid


def render_song_stems(
    song,
    sf_path: str,
    samplerate: int = DEFAULT_SAMPLERATE,
    synth_gain_db: float = 6.0,
    tail_seconds: float = 2.5,
    parallel: bool = True,
) -> list[np.ndarray]:
    """Render every track of a Song to a stem (SoundFont loaded per thread)."""
    tracks = list(song.tracks)

    def _render(t):
        synth, sfid = _thread_synth(sf_path, samplerate, synth_gain_db)
        return render_track_stem(synth, sfid, t, samplerate, tail_seconds)

    if not parallel or len(tracks) <= 1:
        synth, sfid = get_synth(sf_path, samplerate, synth_gain_db)
        return [render_track_stem(synth, sfid, t, samplerate, tail_seconds) for t in tracks]

    with ThreadPoolExecutor(max_workers=min(4, len(tracks))) as ex:
        return list(ex.map(_render, tracks))
