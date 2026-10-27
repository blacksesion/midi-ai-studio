"""End-to-end rendering pipeline.

Ties together parsing, the expressive-performance model, SoundFont rendering
and mixing/mastering. Both the CLI and the web UI drive this module.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import soundfile as sf

from . import dsp
from .assets import ensure_model, ensure_soundfont
from .features import note_features, piece_velocity_stats
from .midi_parse import Song, load_midi, summarize
from .mix import MixOptions, mix_and_master
from .render import DEFAULT_SAMPLERATE, render_song_stems, render_song_stems_iter


@dataclass
class RenderOptions:
    ai_strength: float = 0.6        # 0 = untouched, 1 = fully expressive
    timing_strength: float | None = None
    dynamics_strength: float | None = None
    articulation_strength: float | None = None
    humanize: bool = True
    reverb: float = 0.30            # master reverb 0..1
    reverb_size: float = 0.5        # 0..1
    brightness: float = 0.0         # dB
    warmth: float = 0.0             # dB
    width: float = 0.0              # 0..1
    loudness_db: float = -15.0      # target RMS
    saturation: float = 0.5         # dB tape saturation
    multiband: bool = False
    deesser: bool = False
    sample_rate: int = DEFAULT_SAMPLERATE
    seed: int | None = 42
    use_model: bool = True
    cache: bool = True
    max_duration: float | None = 900.0  # seconds; None disables the guard
    progress: object = None  # optional callable (frac: float) -> None

    def mix_options(self) -> MixOptions:
        return MixOptions(
            reverb=self.reverb,
            reverb_size=self.reverb_size,
            brightness=self.brightness,
            warmth=self.warmth,
            width=self.width,
            loudness_db=self.loudness_db,
            saturation=self.saturation,
            multiband=self.multiband,
            deesser=self.deesser,
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


def _apply_model(song: Song, model, strength: float, seed: int,
                 timing_strength: float | None = None,
                 dynamics_strength: float | None = None,
                 articulation_strength: float | None = None) -> int:
    """Humanise non-drum tracks in place; returns number of notes processed."""
    from .features import quantize_to_grid

    rng = np.random.default_rng(seed)
    processed = 0
    from .instruments import family_of

    for track in song.tracks:
        if track.is_drums or not track.notes:
            continue
        if family_of(track.program) != "piano":
            continue
        notes = track.notes
        # Mechanical (quantised) reference used both for features and targets.
        mech = quantize_to_grid(notes, song.bpm)
        feats = note_features(mech, song.bpm, song.beats_per_bar,
                              total_beats=song.total_seconds * song.bpm / 60.0)
        pred = model.predict(feats, sample=True, rng=rng)
        v_z, onset_dev, dur_log = pred[:, 0], pred[:, 1], pred[:, 2]

        ts = strength if timing_strength is None else timing_strength
        ds = strength if dynamics_strength is None else dynamics_strength
        ars = strength if articulation_strength is None else articulation_strength

        vel = np.array([n.velocity for n in notes], dtype=np.float64)
        mu, sigma = piece_velocity_stats(vel)

        for i, n in enumerate(notes):
            # Expressive velocity (denormalised from z-score).
            v_pred = np.exp(mu + v_z[i] * sigma) - 1.0
            v_pred = int(min(127, max(1, round(v_pred))))
            n.velocity = int(round(n.velocity * (1 - ds) + v_pred * ds))

            # Micro-timing: predicted offset (in beats) applied to the mechanical
            # onset, then blended with the original onset.
            mech_start = mech[i].start
            mech_dur = max(1e-4, mech[i].end - mech[i].start)
            target_start = mech_start + onset_dev[i] * 60.0 / song.bpm
            new_start = n.start * (1 - ts) + target_start * ts
            new_start = max(0.0, new_start)

            # Articulation: ratio applied to mechanical duration, blended.
            ratio = np.exp(dur_log[i] * ars)
            target_dur = mech_dur * ratio
            new_dur = n.duration * (1 - ars) + target_dur * ars
            new_dur = max(0.02, new_dur)
            n.start = new_start
            n.end = new_start + new_dur
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
    prog = getattr(opts, "progress", None)

    def _p(frac):
        if prog is not None:
            try:
                prog(float(frac))
            except Exception:
                pass

    _p(0.05)

    model_used = False
    notes_processed = 0
    if opts.humanize and opts.ai_strength > 0:
        from .expression import shape_track
        for track in song.tracks:
            shape_track(track, song, opts.ai_strength, opts.seed or 42)
        if model is not None:
            notes_processed = _apply_model(song, model, opts.ai_strength, opts.seed or 42,
                                           opts.timing_strength, opts.dynamics_strength,
                                           opts.articulation_strength)
            model_used = True
    _p(0.35)

    sf_path = sf_path or ensure_soundfont()

    # On small machines (and for long files) render/mix stems one at a time so
    # peak memory does not scale with track count.
    env_parallel = os.environ.get("MIDI_AI_PARALLEL_STEMS")
    if env_parallel is None:
        parallel = (os.cpu_count() or 1) >= 4 and len(song.tracks) > 1
    else:
        parallel = env_parallel not in ("0", "false", "False") and len(song.tracks) > 1
    tail = 2.5
    if parallel:
        stems = render_song_stems(song, sf_path, samplerate=opts.sample_rate,
                                  parallel=True)
        max_len = None
    else:
        end = max((n.end for t in song.tracks for n in t.notes), default=0.0)
        max_len = int((end + tail) * opts.sample_rate) + 2
        stems = render_song_stems_iter(song, sf_path, samplerate=opts.sample_rate,
                                       tail_seconds=tail)
    _p(0.7)

    if ir is None:
        ir = dsp.make_ir(opts.sample_rate, rt60=1.5 + 2.0 * opts.reverb_size,
                         pre_delay_ms=18.0, seed=7)

    audio = mix_and_master(stems, song.tracks, opts.sample_rate, ir,
                           opts.mix_options(), max_len=max_len)
    _p(1.0)

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
    from .model import load_model
    return load_model(mp)


_MODEL_CACHE = {"loaded": False, "model": None}
_MODEL_LOCK = threading.Lock()


def get_model():
    with _MODEL_LOCK:
        if not _MODEL_CACHE["loaded"]:
            _MODEL_CACHE["loaded"] = True
            _MODEL_CACHE["model"] = _load_model()
        return _MODEL_CACHE["model"]


CACHE_VERSION = 2  # bump when the render/mix chain changes audibly


def _cache_key(midi_path: str, opts: RenderOptions) -> str:
    import hashlib, json
    h = hashlib.sha256()
    with open(midi_path, "rb") as f:
        h.update(f.read())
    data = {k: v for k, v in opts.__dict__.items() if k != "progress"}
    data["_cache_version"] = CACHE_VERSION
    h.update(json.dumps(data, sort_keys=True, default=str).encode())
    return h.hexdigest()


def _prune_cache(cdir: str, max_files: int = 20, max_mb: int = 300):
    """Keep the render cache bounded so the container disk cannot fill up."""
    try:
        files = [
            (os.path.join(cdir, n), os.path.getmtime(os.path.join(cdir, n)),
             os.path.getsize(os.path.join(cdir, n)))
            for n in os.listdir(cdir)
            if n.endswith(".wav")
        ]
    except OSError:
        return
    files.sort(key=lambda x: x[1])
    total = 0
    keep = []
    for path, mtime, size in reversed(files):
        if len(keep) < max_files and total + size <= max_mb * 1024 * 1024:
            keep.append(path)
            total += size
        else:
            try:
                os.remove(path)
            except OSError:
                pass


def _check_duration(song: Song, opts: RenderOptions):
    if opts.max_duration and song.total_seconds > opts.max_duration:
        raise ValueError(
            f"song is {song.total_seconds / 60.0:.1f} min long; "
            f"the limit is {opts.max_duration / 60.0:.1f} min"
        )


def render_file(
    midi_path: str,
    out_path: str | None = None,
    options: RenderOptions | None = None,
) -> RenderResult:
    """Load, render and (optionally) write a MIDI file to WAV."""
    opts = options or RenderOptions()

    # Cache lookup.
    cache_path = None
    if opts.cache:
        from .assets import asset_dir
        cdir = os.path.join(asset_dir(), "cache")
        os.makedirs(cdir, exist_ok=True)
        cache_path = os.path.join(cdir, _cache_key(midi_path, opts) + ".wav")

    if cache_path and os.path.exists(cache_path):
        import soundfile as sf
        audio, sr = sf.read(cache_path)
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 1:
            audio = np.stack([audio, audio], axis=1)
        song = load_midi(midi_path)
        _check_duration(song, opts)
        result = RenderResult(
            audio=audio, sample_rate=sr,
            duration=float(len(audio) / sr), tracks=summarize(song),
            bpm=round(song.bpm, 2), elapsed=0.0, model_used=opts.use_model and opts.ai_strength > 0,
            notes_processed=0,
        )
        if out_path:
            save_audio(out_path, result.audio, result.sample_rate)
        return result

    song = load_midi(midi_path)
    _check_duration(song, opts)

    model = None
    if opts.use_model and opts.humanize and opts.ai_strength > 0:
        model = get_model()

    result = render_song(song, opts, model=model)

    if cache_path:
        try:
            save_audio(cache_path, result.audio, result.sample_rate)
            _prune_cache(os.path.dirname(cache_path))
        except Exception:
            pass
    if out_path:
        save_audio(out_path, result.audio, result.sample_rate)
    return result


def save_audio(path: str, audio: np.ndarray, sample_rate: int):
    """Write audio to disk, choosing codec by extension (wav/flac/ogg/mp3)."""
    import subprocess
    audio = np.asarray(audio, dtype=np.float32)
    ext = os.path.splitext(path)[1].lower()
    if ext in (".wav", ".flac", ".ogg"):
        subtype = "PCM_24" if ext == ".wav" else None
        sf.write(path, audio, sample_rate, subtype=subtype) if subtype else sf.write(path, audio, sample_rate)
        return
    if ext == ".mp3":
        # Try ffmpeg.
        tmp = path + ".wav"
        sf.write(tmp, audio, sample_rate, subtype="PCM_16")
        try:
            subprocess.run(["ffmpeg", "-y", "-i", tmp, "-b:a", "192k", path],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass
        return
    sf.write(path, audio, sample_rate, subtype="PCM_16")
