import os
import tempfile

import numpy as np
import soundfile as sf

from midi_ai import dsp, midi_parse
from midi_ai.pipeline import render_file, RenderOptions


def test_parse_sample():
    song = midi_parse.load_midi("examples/sample_band.mid")
    assert song.tracks
    assert song.bpm == 120.0
    assert song.total_seconds > 0


def test_dsp_filters():
    x = np.zeros((1000, 2), dtype=np.float32)
    y = dsp.apply_eq(x, 44100, [(1000.0, "lowpass")])
    assert y.shape == x.shape


def test_render_smoke():
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "out.wav")
        res = render_file("examples/sample_band.mid", out, RenderOptions(ai_strength=0.3, use_model=False))
        assert os.path.exists(out)
        audio, sr = sf.read(out)
        assert audio.shape[0] > 0
        assert sr == 44100