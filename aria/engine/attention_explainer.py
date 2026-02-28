"""Attention-based anomaly explainer using PyTorch autoencoder.

A small autoencoder with self-attention that explains anomalies via:
- Feature contributions: which features have highest reconstruction error
- Temporal attention: which time steps the attention layer weighted most
- Contrastive explanation: "Looks like X but with unusually high Y"

Tier 4 only — requires torch. Falls back gracefully when unavailable.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Lazy torch import
_torch: Any = None
_nn: Any = None


def _ensure_torch():
    """Lazily import torch, returning True if available."""
    global _torch, _nn
    if _torch is not None:
        return True
    try:
        import torch
        from torch import nn

        _torch = torch
        _nn = nn
        return True
    except ImportError:
        return False


def _build_autoencoder(n_features: int, sequence_length: int, hidden_dim: int):
    """Build the attention autoencoder model.

    Architecture:
    - Encoder: Linear(n_features, hidden_dim) -> ReLU -> Linear(hidden_dim, hidden_dim//2)
    - Attention: Scaled dot-product self-attention over time steps
    - Decoder: Linear(hidden_dim//2, hidden_dim) -> ReLU -> Linear(hidden_dim, n_features)
    """
    if not _ensure_torch():
        return None

    class AttentionAutoencoder(_nn.Module):
        def __init__(self):
            super().__init__()
            half_dim = max(hidden_dim // 2, 4)

            # Encoder
            self.enc1 = _nn.Linear(n_features, hidden_dim)
            self.enc2 = _nn.Linear(hidden_dim, half_dim)

            # Self-attention over time steps
            self.query = _nn.Linear(half_dim, half_dim)
            self.key = _nn.Linear(half_dim, half_dim)
            self.value = _nn.Linear(half_dim, half_dim)
            self.attn_scale = half_dim**0.5

            # Decoder
            self.dec1 = _nn.Linear(half_dim, hidden_dim)
            self.dec2 = _nn.Linear(hidden_dim, n_features)

            self.relu = _nn.ReLU()

        def forward(self, x):
            # x: (batch, seq_len, n_features)
            # Encode
            h = self.relu(self.enc1(x))
            h = self.enc2(h)  # (batch, seq_len, half_dim)

            # Self-attention
            Q = self.query(h)
            K = self.key(h)
            V = self.value(h)
            attn_weights = _torch.softmax(_torch.bmm(Q, K.transpose(1, 2)) / self.attn_scale, dim=-1)
            h = _torch.bmm(attn_weights, V)

            # Decode
            out = self.relu(self.dec1(h))
            out = self.dec2(out)  # (batch, seq_len, n_features)

            return out, attn_weights

    return AttentionAutoencoder()


class AttentionExplainer:
    """Attention-based anomaly explainer (Tier 4).

    Args:
        n_features: Number of features per time step.
        sequence_length: Number of time steps in each window.
        hidden_dim: Hidden dimension of the autoencoder (default 32).
    """

    def __init__(
        self,
        n_features: int,
        sequence_length: int,
        hidden_dim: int = 32,
    ):
        self.n_features = n_features
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.torch_available = _ensure_torch()

        self._model = None
        self._normal_baseline: np.ndarray | None = None
        self._training_mean: np.ndarray | None = None
        self._training_std: np.ndarray | None = None
        self._trained = False

        if self.torch_available:
            self._model = _build_autoencoder(n_features, sequence_length, hidden_dim)

    @property
    def is_trained(self) -> bool:
        return self._trained

    def train(
        self,
        windows: np.ndarray,
        epochs: int = 20,
        learning_rate: float = 1e-3,
    ) -> dict[str, Any]:
        """Train the autoencoder on normal windows.

        Args:
            windows: Array of shape (n_samples, sequence_length, n_features).
            epochs: Training epochs.
            learning_rate: Adam learning rate.

        Returns:
            Training result dict with trained, final_loss, epochs.
        """
        if not self.torch_available or self._model is None:
            return {"trained": False, "reason": "torch not available"}

        # Store normal baseline statistics
        self._training_mean = windows.mean(axis=0)
        self._training_std = windows.std(axis=0) + 1e-8
        self._normal_baseline = self._training_mean

        # Normalize
        normed = (windows - self._training_mean) / self._training_std

        X = _torch.FloatTensor(normed)
        optimizer = _torch.optim.Adam(self._model.parameters(), lr=learning_rate)
        loss_fn = _nn.MSELoss()

        self._model.train()
        final_loss = 0.0

        for _epoch in range(epochs):
            optimizer.zero_grad()
            reconstructed, _attn = self._model(X)
            loss = loss_fn(reconstructed, X)
            loss.backward()
            optimizer.step()
            final_loss = loss.item()

        self._trained = True
        logger.info(
            f"Attention explainer trained: {epochs} epochs, final_loss={final_loss:.6f}, {windows.shape[0]} samples"
        )
        return {"trained": True, "final_loss": final_loss, "epochs": epochs}

    def explain(
        self,
        window: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Explain an anomalous window.

        Args:
            window: Array of shape (sequence_length, n_features).
            feature_names: Optional names for features.

        Returns:
            Dict with feature_contributions, temporal_attention,
            contrastive_explanation, anomaly_score.
        """
        empty: dict[str, Any] = {
            "feature_contributions": [],
            "temporal_attention": [],
            "contrastive_explanation": None,
            "anomaly_score": None,
        }

        if not self._trained or self._model is None:
            return empty

        names = feature_names or [f"feature_{i}" for i in range(self.n_features)]

        # Normalize using training stats
        normed = (window - self._training_mean) / self._training_std
        X = _torch.FloatTensor(normed).unsqueeze(0)  # (1, seq, feat)

        self._model.eval()
        with _torch.no_grad():
            reconstructed, attn_weights = self._model(X)

        # Reconstruction error per feature (averaged over time)
        recon_error = (X - reconstructed).squeeze(0).numpy()  # (seq, feat)
        per_feature_error = np.mean(np.abs(recon_error), axis=0)  # (feat,)

        # Normalize to contributions
        total_error = per_feature_error.sum()
        contributions = per_feature_error / total_error if total_error > 0 else np.zeros(self.n_features)

        # Sort by contribution
        sorted_idx = np.argsort(contributions)[::-1]
        feature_contributions = [
            {"feature": names[i], "contribution": round(float(contributions[i]), 4)}
            for i in sorted_idx
            if contributions[i] > 0.01
        ]

        # Temporal attention (averaged across attention heads / queries)
        attn = attn_weights.squeeze(0).numpy()  # (seq, seq)
        temporal_attention = np.mean(attn, axis=0).tolist()  # avg attention received
        temporal_attention = [round(float(v), 4) for v in temporal_attention]

        # Overall anomaly score (mean reconstruction error)
        anomaly_score = round(float(np.mean(np.abs(recon_error))), 4)

        # Contrastive explanation
        contrastive = self._build_contrastive(window, per_feature_error, names)

        return {
            "feature_contributions": feature_contributions,
            "temporal_attention": temporal_attention,
            "contrastive_explanation": contrastive,
            "anomaly_score": anomaly_score,
        }

    def _build_contrastive(
        self,
        window: np.ndarray,
        per_feature_error: np.ndarray,
        feature_names: list[str],
    ) -> str | None:
        """Build a contrastive explanation string.

        "Looks like [normal pattern] but with unusually high [feature]"
        """
        if self._normal_baseline is None:
            return None

        # Find the feature with highest deviation from normal
        window_mean = window.mean(axis=0)
        deviation = np.abs(window_mean - self._normal_baseline.mean(axis=0))
        top_idx = np.argmax(deviation)
        top_feature = feature_names[top_idx]
        direction = "high" if window_mean[top_idx] > self._normal_baseline.mean(axis=0)[top_idx] else "low"

        return f"Looks like normal pattern but with unusually {direction} {top_feature}"

    def get_stats(self) -> dict[str, Any]:
        """Return explainer statistics."""
        stats = {
            "is_trained": self._trained,
            "torch_available": self.torch_available,
            "n_features": self.n_features,
            "sequence_length": self.sequence_length,
            "hidden_dim": self.hidden_dim,
        }
        if self._model is not None:
            param_count = sum(p.numel() for p in self._model.parameters())
            stats["parameter_count"] = param_count
        return stats
