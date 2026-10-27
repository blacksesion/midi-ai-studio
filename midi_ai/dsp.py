"""Digital signal processing helpers for mixing and mastering.

All functions operate on float32 numpy arrays (mono or stereo). Stereo arrays
have shape (samples, 2).
"""

from __future__ import annotations

import numpy as np
from scipy import signal

EPS = 1e-9


# --------------------------------------------------------------------------- #
# Utility
# --------------------------------------------------------------------------- #
def db_to_amp(db: float) -> float:
    return 10.0 ** (db / 20.0)


def amp_to_db(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return 20.0 * np.log10(np.maximum(x, EPS))


def ensure_stereo(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        return np.stack([x, x], axis=1)
    if x.shape[1] == 1:
        return np.repeat(x, 2, axis=1)
    return x


# --------------------------------------------------------------------------- #
# Filters
# --------------------------------------------------------------------------- #
def biquad(freq: float, sr: float, kind: str, q: float = 0.707, gain_db: float = 0.0):
    """Return (b, a) for a biquad filter of `kind`.

    kinds: lowpass, highpass, lowshelf, highshelf, peaking, bandpass, notch.
    """
    w0 = 2 * np.pi * freq / sr
    alpha = np.sin(w0) / (2 * q)
    A = db_to_amp(gain_db)
    cosw, sinw = np.cos(w0), np.sin(w0)

    if kind == "lowpass":
        b0 = (1 - cosw) / 2
        b1, b2 = 1 - cosw, (1 - cosw) / 2
        a0, a1, a2 = 1 + alpha, -2 * cosw, 1 - alpha
    elif kind == "highpass":
        b0 = (1 + cosw) / 2
        b1, b2 = -(1 + cosw), (1 + cosw) / 2
        a0, a1, a2 = 1 + alpha, -2 * cosw, 1 - alpha
    elif kind == "lowshelf":
        b0 = A * ((A + 1) - (A - 1) * cosw + 2 * np.sqrt(A) * alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * cosw)
        b2 = A * ((A + 1) - (A - 1) * cosw - 2 * np.sqrt(A) * alpha)
        a0 = (A + 1) + (A - 1) * cosw + 2 * np.sqrt(A) * alpha
        a1 = -2 * ((A - 1) + (A + 1) * cosw)
        a2 = (A + 1) + (A - 1) * cosw - 2 * np.sqrt(A) * alpha
    elif kind == "highshelf":
        b0 = A * ((A + 1) + (A - 1) * cosw + 2 * np.sqrt(A) * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * cosw)
        b2 = A * ((A + 1) + (A - 1) * cosw - 2 * np.sqrt(A) * alpha)
        a0 = (A + 1) - (A - 1) * cosw + 2 * np.sqrt(A) * alpha
        a1 = 2 * ((A - 1) - (A + 1) * cosw)
        a2 = (A + 1) - (A - 1) * cosw - 2 * np.sqrt(A) * alpha
    elif kind == "peaking":
        b0 = 1 + alpha * A
        b1, b2 = -2 * cosw, 1 - alpha * A
        a0 = 1 + alpha / A
        a1, a2 = -2 * cosw, 1 - alpha / A
    else:
        raise ValueError(f"unknown biquad kind: {kind}")

    return (np.array([b0, b1, b2]) / a0, np.array([1.0, a1 / a0, a2 / a0]))


def sos_filters(specs: list[tuple], sr: float):
    """Build a single SOS matrix from a list of (freq, kind, q, gain_db)."""
    sos = []
    for spec in specs:
        if len(spec) == 2:
            freq, kind = spec
            q, gain = 0.707, 0.0
        elif len(spec) == 3:
            freq, kind, q = spec
            gain = 0.0
        else:
            freq, kind, q, gain = spec
        b, a = biquad(freq, sr, kind, q, gain)
        sos.append(np.concatenate([b, a]))
    return np.array(sos)


def apply_eq(x: np.ndarray, sr: float, specs: list[tuple]) -> np.ndarray:
    if not specs:
        return x
    sos = sos_filters(specs, sr)
    return signal.sosfilt(sos, x, axis=0).astype(np.float32)


# --------------------------------------------------------------------------- #
# Dynamics
# --------------------------------------------------------------------------- #
def _control_env(x: np.ndarray, sr: float, control_hz: float = 1200.0):
    """Reduce a per-sample envelope to a per-control-block max envelope."""
    hop = max(1, int(round(sr / control_hz)))
    pad = (-len(x)) % hop
    if pad:
        x2 = np.pad(x, (0, pad))
    else:
        x2 = x
    env = x2.reshape(-1, hop).max(axis=1)
    return env, hop


def _upsample_gain(gain: np.ndarray, hop: int, length: int) -> np.ndarray:
    ctrl_t = np.arange(len(gain)) * hop + hop / 2.0
    t = np.arange(length, dtype=np.float64)
    return np.interp(t, ctrl_t, gain)


def compressor(x: np.ndarray, sr: float, threshold_db: float = -18.0,
               ratio: float = 3.0, attack_ms: float = 20.0,
               release_ms: float = 180.0, makeup_db: float = 0.0) -> np.ndarray:
    """Soft-knee feed-forward stereo-linked compressor (control-rate gain)."""
    if x.ndim == 2:
        env = np.max(np.abs(x), axis=1)
    else:
        env = np.abs(x)

    env_c, hop = _control_env(env, sr)
    level = np.maximum(env_c, EPS)

    # Gain targets (vectorised over control samples).
    threshold = db_to_amp(threshold_db)
    knee = db_to_amp(threshold_db + 6.0)
    target = np.ones_like(level)
    above = level >= knee
    target[above] = (threshold * (level[above] / threshold) ** (1.0 / ratio)) / level[above]
    mid = (level >= threshold) & (level < knee)
    if mid.any():
        t = (amp_to_db(level[mid]) - threshold_db) / 6.0
        s = 1.0 / ratio + (1.0 - 1.0 / ratio) * t * t * (3.0 - 2.0 * t)
        target[mid] = (threshold * (level[mid] / threshold) ** s) / level[mid]

    attack = 1.0 - np.exp(-1.0 / (attack_ms / 1000.0 * sr / hop))
    release = 1.0 - np.exp(-1.0 / (release_ms / 1000.0 * sr / hop))

    gain = np.empty_like(target)
    g = 1.0
    for i in range(len(target)):
        c = attack if target[i] < g else release
        g += c * (target[i] - g)
        gain[i] = g

    gain_full = _upsample_gain(gain, hop, len(env))
    out = x * gain_full[..., None] if x.ndim == 2 else x * gain_full
    if makeup_db:
        out *= db_to_amp(makeup_db)
    return out.astype(np.float32)


def soft_clip(x: np.ndarray, drive_db: float = 0.0) -> np.ndarray:
    """Gentle tape-style saturation."""
    drive = db_to_amp(drive_db)
    y = np.tanh(x * drive) / np.tanh(drive) if drive > 1.0 else np.tanh(x)
    return (y * 0.95).astype(np.float32)


def limiter(x: np.ndarray, sr: float, ceiling_db: float = -0.5, lookahead_ms: float = 3.0):
    """Look-ahead brickwall peak limiter (vectorised)."""
    from scipy.ndimage import maximum_filter1d

    peak = np.max(np.abs(x), axis=1) if x.ndim == 2 else np.abs(x)
    look = max(2, int(lookahead_ms / 1000.0 * sr) + 1)
    future = maximum_filter1d(peak, size=look, mode="nearest")

    ceil = db_to_amp(ceiling_db)
    target = np.ones_like(future)
    mask = future > ceil
    target[mask] = ceil / future[mask]

    # Smooth the gain at control rate to avoid zipper noise.
    target_c, hop = _control_env(target, sr, control_hz=2000.0)
    alpha = 1.0 - np.exp(-1.0 / (lookahead_ms / 1000.0 * sr / hop))
    gain = np.empty_like(target_c)
    g = 1.0
    for i in range(len(target_c)):
        g = target_c[i] if target_c[i] < g else g + alpha * (target_c[i] - g)
        gain[i] = g

    gain_full = _upsample_gain(gain, hop, len(peak))
    out = x * gain_full[..., None] if x.ndim == 2 else x * gain_full
    return out.astype(np.float32)


# --------------------------------------------------------------------------- #
# Reverb (convolution with a generated impulse response)
# --------------------------------------------------------------------------- #
def make_ir(sr: float, rt60: float = 2.4, pre_delay_ms: float = 20.0,
            damping: float = 0.5, seed: int = 7, stereo: bool = True,
            length: float | None = None) -> np.ndarray:
    """Generate a realistic hall impulse response (mono or stereo).

    The IR is exponential-decay filtered noise with a few early reflections,
    high-frequency damping and (for stereo) decorrelated left/right channels.
    """
    rng = np.random.default_rng(seed)
    length = length or (rt60 * 1.5)
    n = int(length * sr)

    def one_channel():
        # Early reflections (discrete taps).
        tap_times = np.array([0.004, 0.009, 0.013, 0.020, 0.028, 0.037, 0.047, 0.060])
        tap_gains = db_to_amp(np.array([-8, -11, -13, -16, -18, -20, -22, -24]))
        ir = np.zeros(n, dtype=np.float32)
        for t, g in zip(tap_times, tap_gains):
            idx = int(t * sr)
            if idx < n:
                ir[idx] += g

        # Late reverberation: exponentially-decaying filtered noise.
        t = np.arange(n) / sr
        decay = 10.0 ** (-3.0 * t / rt60)
        noise = rng.standard_normal(n).astype(np.float32)
        # One-pole lowpass for damping (frequency-dependent decay).
        cutoff = 6000 * (1 - damping) + 800
        a = np.exp(-2 * np.pi * cutoff / sr)
        lp = signal.lfilter([1 - a], [1, -a], noise).astype(np.float32)
        ir += lp * decay * 0.35
        return ir

    # Pre-delay.
    pre = np.zeros(int(pre_delay_ms / 1000.0 * sr), dtype=np.float32)
    left = one_channel()
    if stereo:
        right = one_channel()
        ir = np.stack([left, right], axis=1)
    else:
        ir = left

    if pre.size:
        ir = np.concatenate([np.zeros((pre.size, ir.shape[1]), dtype=np.float32)
                             if ir.ndim == 2 else pre, ir], axis=0)
    # Normalise.
    ir /= (np.max(np.abs(ir)) + EPS)
    return ir.astype(np.float32)


def convolve_stereo(x: np.ndarray, ir: np.ndarray) -> np.ndarray:
    """Convolve a stereo signal with a (stereo) IR, returning stereo."""
    x = ensure_stereo(x)
    if ir.ndim == 1:
        return signal.fftconvolve(x, ir[:, None], mode="full").astype(np.float32)
    out_l = signal.fftconvolve(x[:, 0], ir[:, 0], mode="full")
    out_r = signal.fftconvolve(x[:, 1], ir[:, 1], mode="full")
    return np.stack([out_l, out_r], axis=1).astype(np.float32)


# --------------------------------------------------------------------------- #
# Stereo / panning
# --------------------------------------------------------------------------- #
def pan(stereo: np.ndarray, pan: float) -> np.ndarray:
    """Constant-power pan. pan in [-1, 1]."""
    stereo = ensure_stereo(stereo)
    pan = float(np.clip(pan, -1.0, 1.0))
    theta = (pan + 1.0) * np.pi / 4.0
    gl = np.cos(theta)
    gr = np.sin(theta)
    out = stereo.copy()
    out[:, 0] *= gl
    out[:, 1] *= gr
    return out.astype(np.float32)


def stereo_widen(x: np.ndarray, width: float) -> np.ndarray:
    """Mid/side stereo widening. width in [0, 1] (0 = none)."""
    x = ensure_stereo(x)
    if width <= 0:
        return x
    mid = (x[:, 0] + x[:, 1]) * 0.5
    side = (x[:, 0] - x[:, 1]) * 0.5
    side *= 1.0 + width
    out = np.stack([mid + side, mid - side], axis=1)
    return out.astype(np.float32)


# --------------------------------------------------------------------------- #
# Loudness
# --------------------------------------------------------------------------- #
def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)) + EPS))


def deesser(x: np.ndarray, sr: float, freq: float = 7000.0, threshold_db: float = -24.0,
            ratio: float = 3.0) -> np.ndarray:
    """Simple de-esser: compress a high band around `freq`."""
    # Split: low (below freq) + high (above freq)
    low = apply_eq(x, sr, [(freq, "lowpass", 0.707)])
    high = x - low
    high_c = compressor(high, sr, threshold_db=threshold_db, ratio=ratio,
                        attack_ms=5.0, release_ms=40.0)
    return (low + high_c).astype(np.float32)


def multiband_compressor(x: np.ndarray, sr: float, threshold_db: float = -20.0,
                        ratio: float = 2.0) -> np.ndarray:
    """3-band (low/mid/high) compressor summed back."""
    lo = apply_eq(x, sr, [(200.0, "lowpass", 0.707)])
    hi = apply_eq(x, sr, [(4000.0, "highpass", 0.707)])
    mid = x - lo - hi
    lo_c = compressor(lo, sr, threshold_db=threshold_db, ratio=ratio, attack_ms=30.0, release_ms=250.0)
    mid_c = compressor(mid, sr, threshold_db=threshold_db, ratio=ratio, attack_ms=20.0, release_ms=180.0)
    hi_c = compressor(hi, sr, threshold_db=threshold_db + 2.0, ratio=ratio + 0.5, attack_ms=10.0, release_ms=120.0)
    return (lo_c + mid_c + hi_c).astype(np.float32)


def normalize_rms(x: np.ndarray, target_rms: float = 0.18, peak_ceiling: float = 0.98) -> np.ndarray:
    g = target_rms / max(rms(x), EPS)
    out = x * g
    peak = np.max(np.abs(out))
    if peak > peak_ceiling:
        out *= peak_ceiling / peak
    return out.astype(np.float32)


def _k_weight(x: np.ndarray, sr: float) -> np.ndarray:
    y = apply_eq(x, sr, [(38.0, "highpass", 0.5)])
    y = apply_eq(y, sr, [(4000.0, "highshelf", 0.707, 4.0)])
    return y


def lufs(x: np.ndarray, sr: float) -> float:
    y = _k_weight(x, sr)
    if y.ndim == 2:
        ms = float(np.mean(np.square(y)))
    else:
        ms = float(np.mean(np.square(y)))
    return -0.691 + 10.0 * np.log10(ms + EPS)


def normalize_lufs(x: np.ndarray, sr: float, target_lufs: float = -14.0,
                   peak_ceiling: float = 0.97) -> np.ndarray:
    cur = lufs(x, sr)
    gain = 10.0 ** ((target_lufs - cur) / 20.0)
    out = x * gain
    peak = np.max(np.abs(out))
    if peak > peak_ceiling:
        out *= peak_ceiling / peak
    return out.astype(np.float32)


def master_loudness(x: np.ndarray, sr: float, target_lufs: float = -14.0,
                    ceiling_db: float = -1.0, iterations: int = 4) -> np.ndarray:
    """Iteratively gain + limit until the target LUFS is reached.

    Limiting reduces loudness slightly, so a single gain pass undershoots.
    Repeating gain -> limiter converges to the requested integrated loudness
    while keeping true peaks below the ceiling.
    """
    out = np.asarray(x, dtype=np.float32)
    for _ in range(iterations):
        cur = lufs(out, sr)
        if abs(cur - target_lufs) < 0.1:
            break
        gain = 10.0 ** ((target_lufs - cur) / 20.0)
        out = (out * gain).astype(np.float32)
        out = true_peak_limiter(out, sr, ceiling_db=ceiling_db)
    # Final safety clamp.
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    ceil = db_to_amp(ceiling_db)
    if peak > ceil:
        out = (out * (ceil / peak)).astype(np.float32)
    return out


def _inter_sample_peak(x: np.ndarray, block: int = 1 << 18) -> np.ndarray:
    """Approximate the true peak with 4x cubic (Catmull-Rom) interpolation.

    The naive approach of upsampling the whole signal at 4x allocates several
    copies of a 4x-length buffer, which is catastrophic for multi-minute
    songs. This walks the signal in blocks and keeps a single float32 peak
    envelope, bounding extra memory to a few megabytes.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None]
    n = x.shape[0]
    out = np.empty(n, dtype=np.float32)
    if n == 0:
        return out
    for start in range(0, n, block):
        end = min(n, start + block)
        s = max(0, start - 1)
        e = min(n, end + 2)
        seg = x[s:e]
        if seg.shape[0] < 4:
            seg = np.pad(seg, ((0, 4 - seg.shape[0]), (0, 0)))
        x0 = np.empty_like(seg)
        x0[1:] = seg[:-1]
        x0[0] = seg[0]
        x1 = seg
        x2 = np.empty_like(seg)
        x2[:-1] = seg[1:]
        x2[-1] = seg[-1]
        x3 = np.empty_like(seg)
        x3[1:-1] = seg[2:]
        x3[0] = seg[0]
        x3[-1] = seg[-1]
        peak = np.abs(x1).max(axis=1)
        for t in (0.25, 0.5, 0.75):
            y = 0.5 * (2.0 * x1 + (-x0 + x2) * t
                       + (2.0 * x0 - 5.0 * x1 + 4.0 * x2 - x3) * (t * t)
                       + (-x0 + 3.0 * x1 - 3.0 * x2 + x3) * (t * t * t))
            np.maximum(peak, np.abs(y).max(axis=1), out=peak)
        off = start - s
        out[start:end] = peak[off:off + (end - start)]
    return out


def true_peak_limiter(x: np.ndarray, sr: float, ceiling_db: float = -1.0,
                      oversample: int = 4) -> np.ndarray:
    """Memory-bounded look-ahead limiter with inter-sample peak detection.

    ``oversample`` is kept for API compatibility; peak detection always uses
    the 4x cubic approximation.
    """
    from scipy.ndimage import maximum_filter1d, minimum_filter1d
    from scipy.signal import lfilter

    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x

    ceil = db_to_amp(ceiling_db)
    peak = _inter_sample_peak(x)
    look = max(2, int(0.004 * sr) + 1)
    future = maximum_filter1d(peak, size=look, mode="nearest")
    del peak

    # Target gain (<= 1) at every sample.
    gain = np.minimum(1.0, ceil / np.maximum(future, 1e-9)).astype(np.float32)
    del future

    # Instant attack (hold the minimum over the look-ahead window) and a
    # smooth one-pole release. Taking the element-wise min guarantees the
    # gain never exceeds the requirement while avoiding a hard release.
    hold = minimum_filter1d(gain, size=look, mode="nearest")
    alpha = np.float32(0.2)
    smooth = lfilter([alpha], [np.float32(1.0), np.float32(-(1.0 - alpha))],
                     gain, zi=[(1.0 - alpha) * np.float32(1.0)])[0].astype(np.float32)
    out_gain = np.minimum(smooth, hold)
    del gain, hold, smooth

    if x.ndim == 2:
        out = x * out_gain[:, None]
    else:
        out = x * out_gain
    return out.astype(np.float32, copy=False)
