"""Asset management (soundfont, model, impulse responses).

Assets live under the project's ``assets/`` directory (or a user data dir if
not writable). The SoundFont is downloaded on first use; the model checkpoint
is trained locally via ``train.py``; impulse responses are generated in code.
"""

from __future__ import annotations

import os
import sys
import urllib.request

SOUNDFONT_URL = (
    "https://ftp.osuosl.org/pub/musescore/soundfont/"
    "MuseScore_General/MuseScore_General.sf3"
)
SOUNDFONT_FILENAME = "MuseScore_General.sf3"

MAESTRO_URL = (
    "https://storage.googleapis.com/magentadata/datasets/"
    "maestro/v3.0.0/maestro-v3.0.0-midi.zip"
)
MAESTRO_FILENAME = "maestro-v3.0.0-midi.zip"

_MODEL_URL = None  # set if you host a pre-trained checkpoint


def project_root() -> str:
    """Directory that contains the ``assets`` folder (repo root)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def asset_dir() -> str:
    root = project_root()
    candidate = os.path.join(root, "assets")
    if os.path.isdir(candidate):
        return candidate
    # Fallback to a user data directory.
    home = os.path.expanduser("~")
    data = os.path.join(home, ".midi_ai", "assets")
    os.makedirs(data, exist_ok=True)
    return data


def soundfont_path() -> str:
    return os.path.join(asset_dir(), "soundfonts", SOUNDFONT_FILENAME)


def model_path() -> str:
    env = os.environ.get("MIDI_AI_MODEL")
    if env:
        return env if os.path.isabs(env) else os.path.join(asset_dir(), "models", env)
    return os.path.join(asset_dir(), "models", "performance.pt")


def download(url: str, dest: str, description: str = "") -> str:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    tmp = dest + ".part"
    label = description or os.path.basename(dest)
    print(f"[assets] downloading {label} ...", flush=True)

    def report(count, block, total):
        if total > 0 and count % 40 == 0:
            pct = min(100, int(count * block * 100 / total))
            sys.stderr.write(f"\r[assets] {label} {pct}%          ")
            sys.stderr.flush()

    urllib.request.urlretrieve(url, tmp, reporthook=report)
    sys.stderr.write("\r" + " " * 60 + "\r")
    os.replace(tmp, dest)
    print(f"[assets] saved {label}", flush=True)
    return dest


def ensure_soundfont() -> str:
    path = soundfont_path()
    if not (os.path.exists(path) and os.path.getsize(path) > 0):
        download(SOUNDFONT_URL, path, "MuseScore General SoundFont")
    return path


def ensure_dataset() -> str:
    """Download the MAESTRO (MIDI-only) training dataset if missing."""
    path = os.path.join(asset_dir(), "training", MAESTRO_FILENAME)
    if not (os.path.exists(path) and os.path.getsize(path) > 0):
        download(MAESTRO_URL, path, "MAESTRO MIDI dataset")
    return path


def ensure_model() -> str | None:
    path = model_path()
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    if _MODEL_URL:
        return download(_MODEL_URL, path, "expressive model")
    return None
