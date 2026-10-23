"""Train the expressive-performance model on MAESTRO (piano) performances.

The model learns how a human pianist shapes a mechanical note sequence:
dynamics (velocity), micro-timing and articulation. The resulting checkpoint
is used at render time to humanise arbitrary MIDI.

Usage:
    python train.py [--max-samples 4000000] [--epochs 25] [--batch 16384]
"""

from __future__ import annotations

import argparse
import glob
import os
import zipfile

import numpy as np

from midi_ai.assets import asset_dir, ensure_dataset, model_path
from midi_ai.features import GRID, N_FEATURES, note_features, piece_velocity_stats
from midi_ai.midi_parse import load_midi
from midi_ai.model import ExpressiveMLP, PerformanceModel

EXTRACT_DIR = "maestro-v3.0.0"


class MechNote:
    __slots__ = ("start", "end", "pitch")

    def __init__(self, start, end, pitch):
        self.start = start
        self.end = end
        self.pitch = pitch


def extract_maestro() -> str:
    base = asset_dir()
    zip_path = ensure_dataset()
    out_dir = os.path.join(base, "training", EXTRACT_DIR)
    if not os.path.isdir(out_dir) or not glob.glob(os.path.join(out_dir, "**", "*.mid*"), recursive=True):
        print("[train] extracting MAESTRO MIDI ...", flush=True)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(os.path.dirname(out_dir))
    return out_dir


def build_dataset(data_dir: str, max_samples: int, seed: int = 0):
    files = sorted(glob.glob(os.path.join(data_dir, "**", "*.mid"), recursive=True)
                   + glob.glob(os.path.join(data_dir, "**", "*.midi"), recursive=True))
    rng = np.random.default_rng(seed)
    rng.shuffle(files)
    print(f"[train] found {len(files)} MIDI files", flush=True)

    Xs, ys = [], []
    total = 0
    for path in files:
        try:
            song = load_midi(path)
        except Exception:
            continue
        notes = [n for t in song.tracks for n in t.notes]
        if len(notes) < 30:
            continue
        bpm = max(30.0, min(400.0, song.bpm))
        bpb = max(1, song.beats_per_bar)
        total_beats = max(1.0, song.total_seconds * bpm / 60.0)

        # Build the mechanical (quantised) reference.
        mech = []
        velocities = np.array([n.velocity for n in notes], dtype=np.float64)
        mu, sigma = piece_velocity_stats(velocities)
        v_z = (np.log1p(velocities) - mu) / sigma

        onset_dev = np.empty(len(notes), dtype=np.float64)  # in quarter-note beats
        dur_log = np.empty(len(notes), dtype=np.float64)

        for i, n in enumerate(notes):
            onset_beat = n.start * bpm / 60.0
            mech_beat = round(onset_beat / GRID) * GRID
            mech_start = mech_beat * 60.0 / bpm

            dur_beats = max(1e-4, (n.end - n.start) * bpm / 60.0)
            mech_dur_beats = max(GRID, round(dur_beats / GRID) * GRID)
            mech_dur = mech_dur_beats * 60.0 / bpm

            mech.append(MechNote(mech_start, mech_start + mech_dur, n.pitch))
            onset_dev[i] = (n.start - mech_start) * bpm / 60.0
            dur_log[i] = np.log(max(n.duration, 1e-3) / max(mech_dur, 1e-3))

        feats = note_features(mech, bpm, bpb, total_beats)
        targets = np.column_stack([v_z, onset_dev, dur_log]).astype(np.float64)

        # Clip extreme targets.
        targets[:, 0] = np.clip(targets[:, 0], -2.5, 2.5)
        targets[:, 1] = np.clip(targets[:, 1], -0.75, 0.75)
        targets[:, 2] = np.clip(targets[:, 2], -1.2, 1.2)

        Xs.append(feats)
        ys.append(targets)
        total += len(feats)
        if total >= max_samples:
            break

    X = np.concatenate(Xs, axis=0).astype(np.float32)
    y = np.concatenate(ys, axis=0).astype(np.float32)
    print(f"[train] dataset ready: {X.shape[0]} notes x {X.shape[1]} features", flush=True)
    return X, y


def train(X: np.ndarray, y: np.ndarray, epochs: int, batch: int, seed: int = 0):
    import torch
    import torch.nn as nn

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device = {dev}", flush=True)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_val = int(len(idx) * 0.1)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    feat_mean = X[tr_idx].mean(axis=0)
    feat_std = X[tr_idx].std(axis=0) + 1e-6
    target_mean = y[tr_idx].mean(axis=0)
    target_std = y[tr_idx].std(axis=0) + 1e-6

    Xs = (X - feat_mean) / feat_std
    ys = (y - target_mean) / target_std

    Xtr = torch.from_numpy(Xs[tr_idx]).float().to(dev)
    ytr = torch.from_numpy(ys[tr_idx]).float().to(dev)
    Xva = torch.from_numpy(Xs[val_idx]).float().to(dev)
    yva = torch.from_numpy(ys[val_idx]).float().to(dev)

    model = ExpressiveMLP().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()

    n_train = Xtr.shape[0]
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train, device=dev)
        total_loss = 0.0
        nb = 0
        for i in range(0, n_train, batch):
            b = perm[i:i + batch]
            xb, yb = Xtr[b], ytr[b]
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            nb += 1
        sched.step()

        model.eval()
        with torch.no_grad():
            vloss = loss_fn(model(Xva), yva).item()
        print(f"[train] epoch {epoch + 1:02d}/{epochs}  "
              f"loss={total_loss / nb:.5f}  val={vloss:.5f}", flush=True)

    # Residual standard deviations (unexplained variance), in original units.
    model.eval()
    with torch.no_grad():
        pred_val = model(Xva).cpu().numpy() * target_std + target_mean
    yva_orig = y[val_idx]
    residuals = np.clip((yva_orig - pred_val).std(axis=0), 1e-3, None).astype(np.float32)
    print(f"[train] residuals (v_z, onset_ms, dur_log) = {residuals}", flush=True)

    return PerformanceModel(
        model,
        feat_mean.astype(np.float32),
        feat_std.astype(np.float32),
        residuals,
        meta={"dataset": "MAESTRO v3.0.0 (piano)", "train_notes": int(n_train)},
        target_mean=target_mean.astype(np.float32),
        target_std=target_std.astype(np.float32),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-samples", type=int, default=4_000_000)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data_dir = extract_maestro()
    X, y = build_dataset(data_dir, args.max_samples, args.seed)
    perf = train(X, y, args.epochs, args.batch, args.seed)

    out = args.out or model_path()
    perf.save(out)
    print(f"[train] saved checkpoint -> {out}", flush=True)


if __name__ == "__main__":
    main()
