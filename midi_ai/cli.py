"""Command-line interface: `python -m midi_ai input.mid -o out.wav`."""

from __future__ import annotations

import argparse
import os
import sys

from . import __version__
from .pipeline import RenderOptions, render_file


def _clamp01(x: str) -> float:
    v = float(x)
    if not 0.0 <= v <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return v


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="midi-ai",
        description="Render MIDI files to near-realistic audio using AI "
                    "expressive performance modelling + high-quality synthesis.",
    )
    p.add_argument("input", help="input .mid file")
    p.add_argument("-o", "--out", help="output .wav path (default: <input>.wav)")
    p.add_argument("--ai-strength", type=_clamp01, default=0.6,
                   help="AI expressiveness strength 0..1 (default 0.6)")
    p.add_argument("--no-ai", action="store_true", help="disable the AI model")
    p.add_argument("--reverb", type=_clamp01, default=0.30, help="reverb amount 0..1")
    p.add_argument("--brightness", type=float, default=0.0, help="high-shelf EQ (dB)")
    p.add_argument("--warmth", type=float, default=0.0, help="low-shelf EQ (dB)")
    p.add_argument("--width", type=_clamp01, default=0.0, help="stereo width 0..1")
    p.add_argument("--loudness", type=float, default=-15.0, help="target RMS (dBFS)")
    p.add_argument("--saturation", type=float, default=0.5, help="tape saturation (dB)")
    p.add_argument("--rate", type=int, default=44100, help="sample rate")
    p.add_argument("--seed", type=int, default=42, help="random seed")
    p.add_argument("--version", action="version", version=f"midi-ai {__version__}")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if not os.path.exists(args.input):
        print(f"error: file not found: {args.input}", file=sys.stderr)
        return 1

    out = args.out or os.path.splitext(args.input)[0] + ".wav"
    opts = RenderOptions(
        ai_strength=0.0 if args.no_ai else args.ai_strength,
        humanize=not args.no_ai,
        reverb=args.reverb,
        brightness=args.brightness,
        warmth=args.warmth,
        width=args.width,
        loudness_db=args.loudness,
        saturation=args.saturation,
        sample_rate=args.rate,
        seed=args.seed,
        use_model=not args.no_ai,
    )

    print(f"Rendering {args.input} ...")
    result = render_file(args.input, out, opts)

    print(f"\n  bpm          : {result.bpm}")
    print(f"  tracks       : {len(result.tracks)}")
    for t in result.tracks:
        print(f"    - {t['instrument']:<22} {t['notes']:>6} notes  ({t['name']})")
    print(f"  duration     : {result.duration:.2f} s")
    print(f"  AI model     : {'on' if result.model_used else 'off'}"
          + (f" ({result.notes_processed} notes shaped)" if result.model_used else ""))
    print(f"  render time  : {result.elapsed:.2f} s")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
