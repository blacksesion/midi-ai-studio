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
from midi_ai.model import (ExpressiveMLP, ExpressiveSeqCNN, PerformanceModel,
                           SeqPerformanceModel)

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


def _build_from_files(files, max_samples):
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

        mech = []
        velocities = np.array([n.velocity for n in notes], dtype=np.float64)
        mu, sigma = piece_velocity_stats(velocities)
        v_z = (np.log1p(velocities) - mu) / sigma

        onset_dev = np.empty(len(notes), dtype=np.float64)
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
        targets[:, 0] = np.clip(targets[:, 0], -2.5, 2.5)
        targets[:, 1] = np.clip(targets[:, 1], -0.75, 0.75)
        targets[:, 2] = np.clip(targets[:, 2], -1.2, 1.2)

        Xs.append(feats)
        ys.append(targets)
        total += len(feats)
        if total >= max_samples:
            break

    if not Xs:
        return None, None
    X = np.concatenate(Xs, axis=0).astype(np.float32)
    y = np.concatenate(ys, axis=0).astype(np.float32)
    return X, y


def _build_sequences(files, max_samples, window=256, stride=128):
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

        mech = []
        velocities = np.array([n.velocity for n in notes], dtype=np.float64)
        mu, sigma = piece_velocity_stats(velocities)
        v_z = (np.log1p(velocities) - mu) / sigma

        onset_dev = np.empty(len(notes), dtype=np.float64)
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
        targets[:, 0] = np.clip(targets[:, 0], -2.5, 2.5)
        targets[:, 1] = np.clip(targets[:, 1], -0.75, 0.75)
        targets[:, 2] = np.clip(targets[:, 2], -1.2, 1.2)

        # sliding windows
        N = len(feats)
        i = 0
        while i + window <= N:
            Xs.append(feats[i:i + window])
            ys.append(targets[i:i + window])
            total += window
            i += stride
            if total >= max_samples:
                break
        if total >= max_samples:
            break

    if not Xs:
        return None, None
    X = np.stack(Xs).astype(np.float32)
    y = np.stack(ys).astype(np.float32)
    return X, y


def build_dataset(data_dir: str, max_samples: int, seed: int = 0, val_ratio: float = 0.1,
                  seq: bool = False, window: int = 256, stride: int = 128):
    files = sorted(glob.glob(os.path.join(data_dir, "**", "*.mid"), recursive=True)
                   + glob.glob(os.path.join(data_dir, "**", "*.midi"), recursive=True))
    rng = np.random.default_rng(seed)
    rng.shuffle(files)
    print(f"[train] found {len(files)} MIDI files", flush=True)

    n_val = max(1, int(len(files) * val_ratio))
    val_files, tr_files = files[:n_val], files[n_val:]

    per_tr = int(max_samples * (1 - val_ratio))
    per_va = int(max_samples * val_ratio)
    if seq:
        Xtr, ytr = _build_sequences(tr_files, per_tr, window, stride)
        Xva, yva = _build_sequences(val_files, per_va, window, stride)
    else:
        Xtr, ytr = _build_from_files(tr_files, per_tr)
        Xva, yva = _build_from_files(val_files, per_va)
    print(f"[train] train={len(Xtr)} val={len(Xva)}", flush=True)
    return Xtr, ytr, Xva, yva


def train(Xtr, ytr, Xva, yva, epochs: int, batch: int, seed: int = 0, seq: bool = False):
    import torch
    import torch.nn as nn

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device = {dev}", flush=True)
    use_amp = dev == "cuda"

    rng = np.random.default_rng(seed)
    if seq:
        feat_mean = Xtr.reshape(-1, Xtr.shape[-1]).mean(axis=0)
        feat_std = Xtr.reshape(-1, Xtr.shape[-1]).std(axis=0) + 1e-6
        target_mean = ytr.reshape(-1, ytr.shape[-1]).mean(axis=0)
        target_std = ytr.reshape(-1, ytr.shape[-1]).std(axis=0) + 1e-6
    else:
        feat_mean = Xtr.mean(axis=0)
        feat_std = Xtr.std(axis=0) + 1e-6
        target_mean = ytr.mean(axis=0)
        target_std = ytr.std(axis=0) + 1e-6

    Xtrn = (Xtr - feat_mean) / feat_std
    ytrn = (ytr - target_mean) / target_std
    Xvan = (Xva - feat_mean) / feat_std
    yvan = (yva - target_mean) / target_std

    Xtrt = torch.from_numpy(Xtrn).float().to(dev)
    ytrt = torch.from_numpy(ytrn).float().to(dev)
    Xvat = torch.from_numpy(Xvan).float().to(dev)
    yvat = torch.from_numpy(yvan).float().to(dev)

    if seq:
        model = ExpressiveSeqCNN().to(dev)
    else:
        model = ExpressiveMLP().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    best_val = float("inf")
    best_state = None
    patience = 5
    no_improve = 0
    n_train = Xtrt.shape[0]

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train, device=dev)
        total_loss = 0.0
        nb = 0
        for i in range(0, n_train, batch):
            b = perm[i:i + batch]
            xb, yb = Xtrt[b], ytrt[b]
            opt.zero_grad()
            if use_amp:
                with torch.amp.autocast("cuda"):
                    pred = model(xb)
                    loss = loss_fn(pred, yb)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            total_loss += loss.item()
            nb += 1
        sched.step()

        model.eval()
        with torch.no_grad():
            vloss = loss_fn(model(Xvat), yvat).item()
        print(f"[train] epoch {epoch + 1:02d}/{epochs}  "
              f"loss={total_loss / nb:.5f}  val={vloss:.5f}", flush=True)

        if vloss < best_val - 1e-6:
            best_val = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print("[train] early stopping", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        pred_val = model(Xvat).cpu().numpy() * target_std + target_mean
    yva_orig = yva
    diff = yva_orig - pred_val
    if seq:
        residuals = np.clip(diff.reshape(-1, 3).std(axis=0), 1e-3, None).astype(np.float32)
        mae = np.abs(diff.reshape(-1, 3)).mean(axis=0)
    else:
        residuals = np.clip(diff.std(axis=0), 1e-3, None).astype(np.float32)
        mae = np.abs(diff).mean(axis=0)
    print(f"[train] residuals (v_z, onset, dur) = {residuals}", flush=True)
    print(f"[train] MAE (v_z, onset, dur) = {mae}", flush=True)

    if seq:
        return SeqPerformanceModel(
            model,
            feat_mean.astype(np.float32),
            feat_std.astype(np.float32),
            residuals,
            meta={"dataset": "MAESTRO v3.0.0 (piano)", "train_notes": int(len(Xtr)) * (Xtr.shape[1] if Xtr.ndim == 3 else 1),
                  "arch": "seqcnn"},
            target_mean=target_mean.astype(np.float32),
            target_std=target_std.astype(np.float32),
        )
    return PerformanceModel(
        model,
        feat_mean.astype(np.float32),
        feat_std.astype(np.float32),
        residuals,
        meta={"dataset": "MAESTRO v3.0.0 (piano)", "train_notes": int(len(Xtr))},
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
    ap.add_argument("--seq", action="store_true", help="train sequential CNN model")
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--stride", type=int, default=128)
    args = ap.parse_args()

    data_dir = extract_maestro()
    Xtr, ytr, Xva, yva = build_dataset(data_dir, args.max_samples, args.seed,
                                         seq=args.seq, window=args.window, stride=args.stride)
    perf = train(Xtr, ytr, Xva, yva, args.epochs, args.batch, args.seed, seq=args.seq)

    out = args.out or model_path()
    perf.save(out)
    print(f"[train] saved checkpoint -> {out}", flush=True)


if __name__ == "__main__":
    main()