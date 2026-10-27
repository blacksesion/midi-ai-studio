"""The expressive-performance model.

A small feed-forward network that predicts, per note, how a human performer
would shape it relative to a mechanical reference:

  * ``v_z``         — normalised expressive velocity (z-score within the piece)
  * ``onset_beats`` — micro-timing offset in quarter-note beats
  * ``dur_log``     — log articulation ratio (expressive / mechanical duration)

The checkpoint also stores feature statistics and empirical residual standard
deviations so inference can add a controlled amount of natural variance.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn as nn

from .features import N_FEATURES

TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# On small servers torch would otherwise use every core and oversubscribe the
# CPU alongside the SoundFont renderer. One or two threads is plenty for a
# tiny CNN and keeps the box responsive.
_THREADS = max(1, int(os.environ.get(
    "MIDI_AI_TORCH_THREADS", str(min(2, os.cpu_count() or 1)))))
try:
    torch.set_num_threads(_THREADS)
    torch.set_num_interop_threads(1)
except (RuntimeError, ValueError):
    pass


class ExpressiveMLP(nn.Module):
    def __init__(self, in_dim: int = N_FEATURES, hidden: int = 256, out_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ExpressiveSeqCNN(nn.Module):
    """1D-CNN over the note sequence for phrase-aware expression."""

    def __init__(self, in_dim: int = N_FEATURES, hidden: int = 128, out_dim: int = 3,
                 kernel: int = 5):
        super().__init__()
        self.conv1 = nn.Conv1d(in_dim, hidden, kernel, padding=kernel // 2)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel, padding=kernel // 2)
        self.conv3 = nn.Conv1d(hidden, hidden // 2, kernel, padding=kernel // 2)
        self.act = nn.ReLU()
        self.out = nn.Linear(hidden // 2, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, F) -> (B, N, out_dim)
        h = x.transpose(1, 2)  # (B, F, N)
        h = self.act(self.conv1(h))
        h = self.act(self.conv2(h))
        h = self.act(self.conv3(h))
        h = h.transpose(1, 2)  # (B, N, H)
        return self.out(h)


# Residual standard deviations applied at inference (degrees of naturalness).
# These are recomputed during training and stored in the checkpoint.
DEFAULT_RESIDUALS = np.array([0.30, 0.06, 0.10], dtype=np.float32)  # v_z, onset_beats, dur_log


class PerformanceModel:
    """Wraps the torch model + normalisation statistics for easy inference."""

    def __init__(self, model: ExpressiveMLP, feat_mean: np.ndarray,
                 feat_std: np.ndarray, residuals: np.ndarray, meta: dict | None = None,
                 target_mean: np.ndarray | None = None,
                 target_std: np.ndarray | None = None):
        self.model = model
        self.feat_mean = feat_mean.astype(np.float32)
        self.feat_std = feat_std.astype(np.float32)
        self.residuals = residuals.astype(np.float32)
        self.target_mean = None if target_mean is None else target_mean.astype(np.float32)
        self.target_std = None if target_std is None else target_std.astype(np.float32)
        self.meta = meta or {}

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.feat_mean) / (self.feat_std + 1e-6)

    @torch.no_grad()
    def predict(self, feats: np.ndarray, sample: bool = True,
                rng: np.random.Generator | None = None) -> np.ndarray:
        """Return (N, 3) full-expressive predictions [v_z, onset_beats, dur_log].

        ``sample`` adds residual variance for a more human, less averaged
        result. Values are *full* deviations; the caller blends them with the
        original performance by an ``ai_strength`` factor.
        """
        if len(feats) == 0:
            return np.zeros((0, 3), dtype=np.float32)
        rng = rng or np.random.default_rng()
        dev = device_of(self.model)
        xn = torch.from_numpy(self._normalize(feats)).float().to(dev)
        y = self.model(xn).cpu().numpy().astype(np.float32)

        # De-standardise targets back to original units.
        if self.target_std is not None:
            mean = self.target_mean if self.target_mean is not None else np.zeros(3, dtype=np.float32)
            y = y * self.target_std + mean

        if sample:
            noise = rng.standard_normal(y.shape).astype(np.float32) * self.residuals
            y = y + noise

        # Clamp to musically sane ranges.
        y[:, 0] = np.clip(y[:, 0], -2.0, 2.0)
        y[:, 1] = np.clip(y[:, 1], -0.5, 0.5)
        y[:, 2] = np.clip(y[:, 2], -0.8, 0.8)
        return y

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
        torch.save({
            "state_dict": self.model.state_dict(),
            "feat_mean": self.feat_mean,
            "feat_std": self.feat_std,
            "residuals": self.residuals,
            "target_mean": self.target_mean,
            "target_std": self.target_std,
            "meta": self.meta,
            "in_dim": N_FEATURES,
            "hidden": self.model.net[0].out_features,
        }, path)

    @classmethod
    def load(cls, path: str, device: str | None = None) -> "PerformanceModel":
        device = device or TORCH_DEVICE
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = ExpressiveMLP(in_dim=ckpt["in_dim"], hidden=ckpt["hidden"])
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        model.eval()
        return cls(
            model,
            np.asarray(ckpt["feat_mean"], dtype=np.float32),
            np.asarray(ckpt["feat_std"], dtype=np.float32),
            np.asarray(ckpt.get("residuals", DEFAULT_RESIDUALS), dtype=np.float32),
            ckpt.get("meta", {}),
            np.asarray(ckpt["target_mean"], dtype=np.float32) if ckpt.get("target_mean") is not None else None,
            np.asarray(ckpt["target_std"], dtype=np.float32) if ckpt.get("target_std") is not None else None,
        )


class SeqPerformanceModel:
    """Wraps a sequential model with the same inference interface."""

    def __init__(self, model: ExpressiveSeqCNN, feat_mean, feat_std, residuals,
                 meta=None, target_mean=None, target_std=None):
        self.model = model
        self.feat_mean = feat_mean.astype(np.float32)
        self.feat_std = feat_std.astype(np.float32)
        self.residuals = residuals.astype(np.float32)
        self.target_mean = None if target_mean is None else target_mean.astype(np.float32)
        self.target_std = None if target_std is None else target_std.astype(np.float32)
        self.meta = meta or {}

    def _normalize(self, x):
        return (x - self.feat_mean) / (self.feat_std + 1e-6)

    @torch.no_grad()
    def predict(self, feats, sample=True, rng=None):
        if len(feats) == 0:
            return np.zeros((0, 3), dtype=np.float32)
        rng = rng or np.random.default_rng()
        dev = device_of(self.model)
        xn = self._normalize(feats)
        t = torch.from_numpy(xn).float().unsqueeze(0).to(dev)  # (1, N, F)
        y = self.model(t).squeeze(0).cpu().numpy().astype(np.float32)
        if self.target_std is not None:
            mean = self.target_mean if self.target_mean is not None else np.zeros(3, dtype=np.float32)
            y = y * self.target_std + mean
        if sample:
            noise = rng.standard_normal(y.shape).astype(np.float32) * self.residuals
            y = y + noise
        y[:, 0] = np.clip(y[:, 0], -2.0, 2.0)
        y[:, 1] = np.clip(y[:, 1], -0.5, 0.5)
        y[:, 2] = np.clip(y[:, 2], -0.8, 0.8)
        return y

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
        torch.save({
            "state_dict": self.model.state_dict(),
            "feat_mean": self.feat_mean,
            "feat_std": self.feat_std,
            "residuals": self.residuals,
            "target_mean": self.target_mean,
            "target_std": self.target_std,
            "meta": self.meta,
            "in_dim": N_FEATURES,
            "hidden": self.model.conv1.out_channels,
            "arch": "seqcnn",
            "kernel": self.model.conv1.kernel_size[0],
        }, path)

    @classmethod
    def load(cls, path, device=None):
        device = device or TORCH_DEVICE
        ckpt = torch.load(path, map_location=device, weights_only=False)
        arch = ckpt.get("arch", "mlp")
        if arch != "seqcnn":
            raise ValueError("not a seq checkpoint")
        model = ExpressiveSeqCNN(in_dim=ckpt["in_dim"], hidden=ckpt["hidden"],
                                 kernel=ckpt.get("kernel", 5))
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        model.eval()
        return cls(
            model,
            np.asarray(ckpt["feat_mean"], dtype=np.float32),
            np.asarray(ckpt["feat_std"], dtype=np.float32),
            np.asarray(ckpt.get("residuals", DEFAULT_RESIDUALS), dtype=np.float32),
            ckpt.get("meta", {}),
            np.asarray(ckpt["target_mean"], dtype=np.float32) if ckpt.get("target_mean") is not None else None,
            np.asarray(ckpt["target_std"], dtype=np.float32) if ckpt.get("target_std") is not None else None,
        )


def load_model(path: str, device: str | None = None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    arch = ckpt.get("arch", "mlp")
    if arch == "seqcnn":
        return SeqPerformanceModel.load(path, device)
    return PerformanceModel.load(path, device)


def device_of(model: nn.Module) -> torch.device:
    return next(model.parameters()).device
