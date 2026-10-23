"""Standard MIDI File parsing.

Converts a .mid file into a high-level :class:`Song` object with a tempo map,
time signatures, tracks and note events expressed in *seconds* (so downstream
code can apply sub-sample timing / humanisation without worrying about ticks).

Each distinct (channel) that carries notes becomes a :class:`Track`, so
multi-channel Type-1 tracks are handled correctly. Sustain-pedal and a few
other controllers are captured so the renderer can reproduce realistic
articulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import mido

from .instruments import program_name

DRUM_CHANNEL = 9

# Controllers we forward to the synth for realism.
FORWARD_CC = {1, 7, 10, 11, 64, 91, 93}


@dataclass
class Note:
    pitch: int
    velocity: int
    start: float  # seconds
    end: float    # seconds
    channel: int
    program: int

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class Track:
    name: str
    channel: int
    program: int
    is_drums: bool
    notes: list[Note] = field(default_factory=list)
    controls: list[tuple[float, int, int]] = field(default_factory=list)  # (sec, cc, value)

    @property
    def program_label(self) -> str:
        if self.is_drums:
            return "Drums"
        return program_name(self.program)

    @property
    def note_count(self) -> int:
        return len(self.notes)


@dataclass
class Song:
    path: str
    ticks_per_beat: int = 480
    tempo_map: list[tuple[int, int]] = field(default_factory=list)  # (tick, us/beat)
    time_signatures: list[tuple[int, int, int]] = field(default_factory=list)  # (tick, num, den)
    tracks: list[Track] = field(default_factory=list)

    # ---- derived --------------------------------------------------------
    bpm: float = 120.0
    beats_per_bar: int = 4
    total_seconds: float = 0.0

    @property
    def notes(self) -> Iterable[Note]:
        for t in self.tracks:
            yield from t.notes

    def tick_to_seconds(self, tick: int) -> float:
        """Convert an absolute tick to seconds using the tempo map."""
        tm = self.tempo_map or [(0, 500000)]
        secs = 0.0
        prev_tick = 0
        us = tm[0][1]
        for t, u in tm:
            if t >= tick:
                break
            secs += (t - prev_tick) * us / 1_000_000 / self.ticks_per_beat
            prev_tick = t
            us = u
        secs += (tick - prev_tick) * us / 1_000_000 / self.ticks_per_beat
        return secs

    def beat_of(self, seconds: float) -> float:
        return seconds * self.bpm / 60.0


def _accumulate_ticks(messages) -> Iterable[tuple[int, mido.Message]]:
    tick = 0
    for msg in messages:
        tick += msg.time
        yield tick, msg


def load_midi(path: str) -> Song:
    """Parse a MIDI file into a Song."""
    mid = mido.MidiFile(path)
    tpb = mid.ticks_per_beat or 480

    tempo_events: list[tuple[int, int]] = []
    ts_events: list[tuple[int, int, int]] = []

    # channel -> {notes: [...], controls: [...], program: int}
    # notes entries: (tick_on, tick_off, pitch, velocity)
    parts: dict[int, dict] = {}
    track_names: dict[int, str] = {}
    channel_names: dict[int, str] = {}

    def part(channel: int) -> dict:
        if channel not in parts:
            parts[channel] = {"notes": [], "controls": [], "program": 0}
        return parts[channel]

    for ti, tr in enumerate(mid.tracks):
        track_names.setdefault(ti, f"Track {ti + 1}")
        current_name = track_names[ti]
        pending_on: dict[tuple[int, int], tuple[int, int]] = {}

        for tick, msg in _accumulate_ticks(tr):
            if msg.type == "set_tempo":
                tempo_events.append((tick, msg.tempo))
            elif msg.type == "time_signature":
                ts_events.append((tick, msg.numerator, msg.denominator))
            elif msg.type == "track_name":
                current_name = msg.name or current_name
                track_names[ti] = current_name
            elif msg.type == "program_change":
                part(msg.channel)["program"] = msg.program
                channel_names.setdefault(msg.channel, current_name)
            elif msg.type == "control_change":
                if msg.control in FORWARD_CC:
                    part(msg.channel)["controls"].append((tick, msg.control, msg.value))
                    channel_names.setdefault(msg.channel, current_name)
            elif msg.type == "note_on":
                chan, pitch, vel = msg.channel, msg.note, msg.velocity
                channel_names.setdefault(chan, current_name)
                key = (chan, pitch)
                if vel > 0:
                    pending_on.setdefault(key, (tick, vel))
                else:
                    t_on = pending_on.pop(key, None)
                    if t_on is not None:
                        part(chan)["notes"].append((t_on[0], tick, pitch, t_on[1]))
            elif msg.type == "note_off":
                chan, pitch = msg.channel, msg.note
                channel_names.setdefault(chan, current_name)
                key = (chan, pitch)
                t_on = pending_on.pop(key, None)
                if t_on is not None:
                    part(chan)["notes"].append((t_on[0], tick, pitch, t_on[1]))

        for (chan, pitch), (t_on, vel) in pending_on.items():
            channel_names.setdefault(chan, current_name)
            part(chan)["notes"].append((t_on, t_on, pitch, vel))

    tempo_events.sort()
    ts_events.sort()
    if not tempo_events:
        tempo_events = [(0, 500000)]
    if not ts_events:
        ts_events = [(0, 4, 4)]

    song = Song(
        path=path,
        ticks_per_beat=tpb,
        tempo_map=tempo_events,
        time_signatures=ts_events,
    )
    song.bpm = 60_000_000 / tempo_events[0][1]
    song.beats_per_bar = ts_events[0][1] or 4

    for channel, data in sorted(parts.items()):
        notes = [
            Note(
                pitch=pitch,
                velocity=max(1, min(127, vel)),
                start=song.tick_to_seconds(t_on),
                end=song.tick_to_seconds(t_off),
                channel=channel,
                program=data["program"],
            )
            for t_on, t_off, pitch, vel in data["notes"]
        ]
        # Fix zero/negative durations.
        for n in notes:
            if n.end <= n.start:
                n.end = n.start + 0.05
        notes.sort(key=lambda n: (n.start, n.pitch))

        controls = [
            (song.tick_to_seconds(t), cc, val) for t, cc, val in sorted(data["controls"])
        ]

        if not notes:
            continue
        song.tracks.append(Track(
            name=channel_names.get(channel, f"Channel {channel + 1}"),
            channel=channel,
            program=data["program"],
            is_drums=(channel == DRUM_CHANNEL),
            notes=notes,
            controls=controls,
        ))

    ends = [n.end for t in song.tracks for n in t.notes]
    song.total_seconds = max(ends) if ends else 0.0
    song.tracks.sort(key=lambda t: -t.note_count)  # densest first (for display)

    return song


def summarize(song: Song) -> list[dict]:
    out = []
    for i, t in enumerate(song.tracks):
        out.append({
            "index": i,
            "name": t.name,
            "channel": t.channel,
            "program": t.program,
            "instrument": t.program_label,
            "is_drums": t.is_drums,
            "notes": t.note_count,
            "min_pitch": min((n.pitch for n in t.notes), default=0),
            "max_pitch": max((n.pitch for n in t.notes), default=0),
        })
    return out
