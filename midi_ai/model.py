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


def device_of(model: nn.Module) -> torch.device:
    return next(model.parameters()).device
