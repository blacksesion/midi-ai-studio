"""MIDI-AI Studio — renders MIDI files with AI-assisted realism.

A pipeline that:
  1. Parses a Standard MIDI File (tempo map, tracks, notes).
  2. Analyzes each track (instrument, role, register, density).
  3. Applies a learned "expressive performance" model (trained on MAESTRO)
     to humanize velocity, micro-timing and articulation.
  4. Renders audio through a high-quality SoundFont synth.
  5. Mixes and masters the result (pan, EQ, convolution reverb, compression,
     saturation, loudness normalisation).
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
