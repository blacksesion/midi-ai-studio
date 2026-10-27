"""Standard MIDI File parsing.

Converts a .mid file into a high-level :class:`Song` object with a tempo map,
time signatures, tracks and note events expressed in *seconds* (so downstream
code can apply sub-sample timing / humanisation without worrying about ticks).

Each distinct (channel, program) that carries notes becomes a :class:`Track`,
so multi-channel Type-1 tracks and mid-track program changes are handled
correctly. Sustain-pedal and a few other controllers are captured so the
renderer can reproduce realistic articulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import mido

from .instruments import program_name

DRUM_CHANNEL = 9

# Controllers we forward to the synth for realism.
FORWARD_CC = {1, 7, 10, 11, 64, 91, 93}

GRID_TICKS_DEFAULT = 120  # fallback 16th-ish when ticks_per_beat unknown


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

    def seconds_to_tick(self, sec: float) -> int:
        """Inverse-ish: tick for a given second (uses tempo map)."""
        # Linear search over tempo segments.
        tm = self.tempo_map or [(0, 500000)]
        secs = 0.0
        prev_tick = 0
        us = tm[0][1]
        for t, u in tm:
            seg_secs = (t - prev_tick) * us / 1_000_000 / self.ticks_per_beat
            if secs + seg_secs >= sec:
                return prev_tick + int((sec - secs) * 1_000_000 * self.ticks_per_beat / us)
            secs += seg_secs
            prev_tick = t
            us = u
        return prev_tick + int((sec - secs) * 1_000_000 * self.ticks_per_beat / us)

    def bpm_at(self, seconds: float) -> float:
        """Instantaneous BPM at a given time (from tempo map)."""
        tm = self.tempo_map or [(0, 500000)]
        secs = 0.0
        prev_tick = 0
        us = tm[0][1]
        for t, u in tm:
            seg_secs = (t - prev_tick) * us / 1_000_000 / self.ticks_per_beat
            if secs + seg_secs >= seconds:
                return 60_000_000 / us
            secs += seg_secs
            prev_tick = t
            us = u
        return 60_000_000 / us

    def beat_of(self, seconds: float) -> float:
        # Cumulative beats using tempo map.
        tm = self.tempo_map or [(0, 500000)]
        beats = 0.0
        secs = 0.0
        prev_tick = 0
        us = tm[0][1]
        for t, u in tm:
            seg_secs = (t - prev_tick) * us / 1_000_000 / self.ticks_per_beat
            if secs + seg_secs >= seconds:
                return beats + (seconds - secs) * (60.0 / (us / 1000.0))  # approx
            beats += (t - prev_tick) / self.ticks_per_beat
            secs += seg_secs
            prev_tick = t
            us = u
        return beats + (seconds - secs) * (60.0 / (us / 1000.0))

    def beats_between(self, s0: float, s1: float) -> float:
        return self.beat_of(s1) - self.beat_of(s0)


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

    # channel -> {notes: [(t_on, t_off, pitch, vel, program)], controls: [(tick,cc,val)], name, program}
    parts: dict[int, dict] = {}
    channel_names: dict[int, str] = {}

    def part(channel: int) -> dict:
        if channel not in parts:
            parts[channel] = {"notes": [], "controls": [], "program": 0, "name": f"Channel {channel + 1}"}
        return parts[channel]

    for ti, tr in enumerate(mid.tracks):
        current_name = f"Track {ti + 1}"
        current_programs: dict[int, int] = {}  # channel -> current program
        pending_on: dict[tuple[int, int], list[tuple[int, int, int]]] = {}  # (chan,pitch) -> list of (tick, vel, program)

        for tick, msg in _accumulate_ticks(tr):
            if msg.type == "set_tempo":
                tempo_events.append((tick, msg.tempo))
            elif msg.type == "time_signature":
                ts_events.append((tick, msg.numerator, msg.denominator))
            elif msg.type == "track_name":
                current_name = msg.name or current_name
            elif msg.type == "program_change":
                current_programs[msg.channel] = msg.program
                channel_names.setdefault(msg.channel, current_name)
                part(msg.channel)["program"] = msg.program
                part(msg.channel)["name"] = current_name
            elif msg.type == "control_change":
                if msg.control in FORWARD_CC:
                    part(msg.channel)["controls"].append((tick, msg.control, msg.value))
                    part(msg.channel)["name"] = current_name
                    channel_names.setdefault(msg.channel, current_name)
            elif msg.type == "note_on":
                chan, pitch, vel = msg.channel, msg.note, msg.velocity
                channel_names.setdefault(chan, current_name)
                part(chan)["name"] = current_name
                prog = current_programs.get(chan, part(chan)["program"])
                key = (chan, pitch)
                if vel > 0:
                    pending_on.setdefault(key, []).append((tick, vel, prog))
                else:
                    stack = pending_on.get(key)
                    if stack:
                        t_on, v_on, p_on = stack.pop()
                        part(chan)["notes"].append((t_on, tick, pitch, v_on, p_on))
            elif msg.type == "note_off":
                chan, pitch = msg.channel, msg.note
                channel_names.setdefault(chan, current_name)
                part(chan)["name"] = current_name
                key = (chan, pitch)
                stack = pending_on.get(key)
                if stack:
                    t_on, v_on, p_on = stack.pop()
                    part(chan)["notes"].append((t_on, tick, pitch, v_on, p_on))

        # Flush pending notes (held notes) with a short default duration.
        for (chan, pitch), stack in pending_on.items():
            for t_on, vel, prog in stack:
                channel_names.setdefault(chan, current_name)
                part(chan)["name"] = current_name
                part(chan)["notes"].append((t_on, t_on, pitch, vel, prog))

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

    # Group notes by (channel, program) into separate tracks.
    for channel, data in sorted(parts.items()):
        # bucket notes by program
        by_prog: dict[int, list] = {}
        for n in data["notes"]:
            by_prog.setdefault(n[4], []).append(n)

        for program, notes_raw in by_prog.items():
            notes = []
            for t_on, t_off, pitch, vel, prog in notes_raw:
                start = song.tick_to_seconds(t_on)
                if t_off <= t_on:
                    # default: one 16th note
                    t_off = t_on + max(1, tpb // 4)
                end = song.tick_to_seconds(t_off)
                if end <= start:
                    end = start + 0.05
                notes.append(Note(
                    pitch=pitch,
                    velocity=max(1, min(127, vel)),
                    start=start,
                    end=end,
                    channel=channel,
                    program=program,
                ))
            if not notes:
                continue
            notes.sort(key=lambda n: (n.start, n.pitch))

            # Controls are channel-level; attach to every program track so each
            # stem reproduces pedal/CC correctly.
            controls = [
                (song.tick_to_seconds(t), cc, val) for t, cc, val in sorted(data["controls"])
            ]

            is_drums = (channel == DRUM_CHANNEL)
            song.tracks.append(Track(
                name=data.get("name", channel_names.get(channel, f"Channel {channel + 1}")),
                channel=channel,
                program=program,
                is_drums=is_drums,
                notes=notes,
                controls=controls,
            ))

    # If controls ended up only on primary program track, that's fine.

    ends = [n.end for t in song.tracks for n in t.notes]
    song.total_seconds = max(ends) if ends else 0.0
    song.tracks.sort(key=lambda t: -t.note_count)

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
