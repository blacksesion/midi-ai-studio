"""Compare rendered output of two checkpoints (e.g. MLP vs seq CNN).

Usage:
    python scripts/evaluate_models.py examples/sample_band.mid \
        --model assets/models/performance_mlp_backup.pt \
        --model assets/models/performance.pt
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from midi_ai import dsp
from midi_ai.midi_parse import load_midi, summarize
from midi_ai.model import load_model
from midi_ai.pipeline import RenderOptions, render_song


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("midi")
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--outdir", default="tmp/eval")
    ap.add_argument("--ai-strength", type=float, default=0.6)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    song = load_midi(args.midi)
    print(f"[eval] {args.midi}: {len(song.tracks)} tracks, {song.total_seconds:.1f}s")
    for t in summarize(song):
        print(f"  - {t['instrument']:<22} {t['notes']:>5} notes")

    for path in args.model:
        name = os.path.splitext(os.path.basename(path))[0]
        model = load_model(path)
        s = load_midi(args.midi)
        res = render_song(s, RenderOptions(ai_strength=args.ai_strength), model=model)
        out = os.path.join(args.outdir, f"{name}.wav")
        sf.write(out, res.audio, res.sample_rate, subtype="PCM_24")
        peak = float(np.max(np.abs(res.audio)))
        print(f"[eval] {name}: dur={res.duration:.2f}s lufs={dsp.lufs(res.audio, res.sample_rate):.2f} "
              f"peak={dsp.amp_to_db(peak):.2f} dBFS -> {out}")


if __name__ == "__main__":
    main()