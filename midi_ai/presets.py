"""Genre presets: bundle AI expression + mix settings for common styles."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Preset:
    name: str
    ai_strength: float = 0.6
    timing: float | None = None
    dynamics: float | None = None
    articulation: float | None = None
    reverb: float = 0.30
    brightness: float = 0.0
    warmth: float = 0.0
    width: float = 0.0
    saturation: float = 0.5
    loudness_db: float = -14.0


PRESETS: dict[str, Preset] = {
    "classical": Preset("classical", ai_strength=0.7, timing=0.5, dynamics=0.7, articulation=0.6,
                        reverb=0.35, warmth=1.0, width=0.15, saturation=0.2, loudness_db=-16.0),
    "jazz": Preset("jazz", ai_strength=0.65, timing=0.7, dynamics=0.6, articulation=0.5,
                   reverb=0.25, warmth=2.0, width=0.1, saturation=0.4, loudness_db=-14.0),
    "pop": Preset("pop", ai_strength=0.45, timing=0.4, dynamics=0.5, articulation=0.4,
                  reverb=0.20, brightness=2.0, width=0.25, saturation=0.6, loudness_db=-12.0),
    "cinematic": Preset("cinematic", ai_strength=0.75, timing=0.55, dynamics=0.8, articulation=0.7,
                        reverb=0.45, warmth=1.5, width=0.4, saturation=0.3, loudness_db=-16.0),
    "neutral": Preset("neutral", ai_strength=0.0, reverb=0.20, saturation=0.0),
}


def get(name: str) -> Preset:
    return PRESETS.get(name, PRESETS["neutral"])