"""General MIDI instrument table and mixing intelligence.

Provides names, instrument families, and per-family mixing presets used to
place each track on a believable "stage" (pan, level, reverb send, EQ).
"""

from __future__ import annotations

from dataclasses import dataclass

# Program names for General MIDI Level 1, index = program number (0..127).
GM_PROGRAMS = [
    # 0-7 piano
    "Acoustic Grand Piano", "Bright Acoustic Piano", "Electric Grand Piano",
    "Honky-tonk Piano", "Electric Piano 1", "Electric Piano 2", "Harpsichord",
    "Clavinet",
    # 8-15 chromatic percussion
    "Celesta", "Glockenspiel", "Music Box", "Vibraphone", "Marimba",
    "Xylophone", "Tubular Bells", "Dulcimer",
    # 16-23 organ
    "Drawbar Organ", "Percussive Organ", "Rock Organ", "Church Organ",
    "Reed Organ", "Accordion", "Harmonica", "Tango Accordion",
    # 24-31 guitar
    "Nylon Guitar", "Steel Guitar", "Jazz Guitar", "Clean Guitar",
    "Muted Guitar", "Overdriven Guitar", "Distortion Guitar", "Guitar Harmonics",
    # 32-39 bass
    "Acoustic Bass", "Electric Bass (finger)", "Electric Bass (pick)",
    "Fretless Bass", "Slap Bass 1", "Slap Bass 2", "Synth Bass 1", "Synth Bass 2",
    # 40-47 strings
    "Violin", "Viola", "Cello", "Contrabass", "Tremolo Strings",
    "Pizzicato Strings", "Orchestral Harp", "Timpani",
    # 48-55 ensemble
    "String Ensemble 1", "String Ensemble 2", "Synth Strings 1", "Synth Strings 2",
    "Choir Aahs", "Voice Oohs", "Synth Voice", "Orchestra Hit",
    # 56-63 brass
    "Trumpet", "Trombone", "Tuba", "Muted Trumpet", "French Horn", "Brass Section",
    "Synth Brass 1", "Synth Brass 2",
    # 64-71 reed
    "Soprano Sax", "Alto Sax", "Tenor Sax", "Baritone Sax", "Oboe",
    "English Horn", "Bassoon", "Clarinet",
    # 72-79 pipe
    "Piccolo", "Flute", "Recorder", "Pan Flute", "Blown Bottle", "Shakuhachi",
    "Whistle", "Ocarina",
    # 80-87 synth lead
    "Lead 1 (square)", "Lead 2 (sawtooth)", "Lead 3 (calliope)", "Lead 4 (chiff)",
    "Lead 5 (charang)", "Lead 6 (voice)", "Lead 7 (fifths)", "Lead 8 (bass+lead)",
    # 88-95 synth pad
    "Pad 1 (new age)", "Pad 2 (warm)", "Pad 3 (polysynth)", "Pad 4 (choir)",
    "Pad 5 (bowed)", "Pad 6 (metallic)", "Pad 7 (halo)", "Pad 8 (sweep)",
    # 96-103 synth effects
    "FX 1 (rain)", "FX 2 (soundtrack)", "FX 3 (crystal)", "FX 4 (atmosphere)",
    "FX 5 (brightness)", "FX 6 (goblins)", "FX 7 (echoes)", "FX 8 (sci-fi)",
    # 104-111 ethnic
    "Sitar", "Banjo", "Shamisen", "Koto", "Kalimba", "Bag Pipe", "Fiddle", "Shanai",
    # 112-119 percussive
    "Tinkle Bell", "Agogo", "Steel Drums", "Woodblock", "Taiko Drum",
    "Melodic Tom", "Synth Drum", "Reverse Cymbal",
    # 120-127 sound effects
    "Guitar Fret Noise", "Breath Noise", "Seashore", "Bird Tweet", "Telephone Ring",
    "Helicopter", "Applause", "Gunshot",
]

# Instrument family (used for mixing + role inference). Drum channel (9) is
# handled separately.
FAMILIES = {
    "piano":        list(range(0, 8)),
    "chrom_perc":   list(range(8, 16)),
    "organ":        list(range(16, 24)),
    "guitar":       list(range(24, 32)),
    "bass":         list(range(32, 40)),
    "strings":      list(range(40, 48)),
    "ensemble":     list(range(48, 56)),
    "brass":        list(range(56, 64)),
    "reed":         list(range(64, 72)),
    "pipe":         list(range(72, 80)),
    "synth_lead":   list(range(80, 88)),
    "synth_pad":    list(range(88, 96)),
    "synth_fx":     list(range(96, 104)),
    "ethnic":       list(range(104, 112)),
    "percussive":   list(range(112, 120)),
    "sound_fx":     list(range(120, 128)),
}

PROGRAM_FAMILY = {p: f for f, ps in FAMILIES.items() for p in ps}


@dataclass(frozen=True)
class MixPreset:
    pan: float = 0.0          # -1 (full left) .. +1 (full right)
    level_db: float = 0.0     # relative trim
    reverb: float = 0.20      # reverb send 0..1
    width: float = 0.0        # stereo width boost 0..1
    bright_db: float = 0.0    # high-shelf trim applied at mix time
    legato: bool = False      # hint: play overlapping (articulation)


# Per-family mix presets. Values chosen to create a natural stage.
FAMILY_PRESETS: dict[str, MixPreset] = {
    "piano":      MixPreset(pan=0.00, level_db=0.0, reverb=0.22, width=0.10),
    "chrom_perc": MixPreset(pan=0.12, level_db=-1.0, reverb=0.28, width=0.0),
    "organ":      MixPreset(pan=0.00, level_db=-1.0, reverb=0.10, width=0.0),
    "guitar":     MixPreset(pan=-0.20, level_db=0.0, reverb=0.16, width=0.0),
    "bass":       MixPreset(pan=0.05, level_db=-1.5, reverb=0.06, width=0.0),
    "strings":    MixPreset(pan=0.15, level_db=0.0, reverb=0.38, width=0.35, legato=True),
    "ensemble":   MixPreset(pan=-0.12, level_db=0.0, reverb=0.32, width=0.30, legato=True),
    "brass":      MixPreset(pan=-0.15, level_db=0.0, reverb=0.30, width=0.0, legato=True),
    "reed":       MixPreset(pan=0.10, level_db=0.0, reverb=0.24, width=0.0, legato=True),
    "pipe":       MixPreset(pan=0.15, level_db=-1.0, reverb=0.30, width=0.0, legato=True),
    "synth_lead": MixPreset(pan=0.18, level_db=-1.0, reverb=0.22, width=0.0, legato=True),
    "synth_pad":  MixPreset(pan=-0.10, level_db=-1.5, reverb=0.35, width=0.40, legato=True),
    "synth_fx":   MixPreset(pan=0.00, level_db=-2.0, reverb=0.30, width=0.30),
    "ethnic":     MixPreset(pan=0.05, level_db=0.0, reverb=0.20, width=0.0),
    "percussive": MixPreset(pan=0.00, level_db=0.0, reverb=0.12, width=0.0),
    "sound_fx":   MixPreset(pan=0.00, level_db=-3.0, reverb=0.05, width=0.0),
    "drums":      MixPreset(pan=0.00, level_db=0.0, reverb=0.10, width=0.0),
}

# Drum kit note names (GM), for analysis and display.
DRUM_NAMES = {
    35: "Acoustic Bass Drum", 36: "Bass Drum 1", 38: "Acoustic Snare",
    40: "Electric Snare", 42: "Closed Hi-Hat", 44: "Pedal Hi-Hat",
    46: "Open Hi-Hat", 49: "Crash Cymbal 1", 51: "Ride Cymbal 1",
    52: "Chinese Cymbal", 55: "Splash Cymbal", 57: "Crash Cymbal 2",
    59: "Ride Cymbal 2", 41: "Low Floor Tom", 43: "High Floor Tom",
    45: "Low Tom", 47: "Low-Mid Tom", 48: "Hi-Mid Tom", 50: "High Tom",
    37: "Side Stick", 39: "Hand Clap", 54: "Tambourine", 56: "Cowbell",
}


def family_of(program: int) -> str:
    return PROGRAM_FAMILY.get(program, "ensemble")


def preset_of(program: int, is_drums: bool = False) -> MixPreset:
    if is_drums:
        return FAMILY_PRESETS["drums"]
    return FAMILY_PRESETS[family_of(program)]


def program_name(program: int) -> str:
    return GM_PROGRAMS[program] if 0 <= program < 128 else f"Program {program}"
