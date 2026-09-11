# src/fusion.py
import torch
import torch.nn as nn

from src.labels import NUM_CLASSES

FEATURE_DIM = 2048
N_TOKENS = 16


class FusionClassifier(nn.Module):
    """One class covers all three experiments: camera-only, radar-only, fused."""

    def __init__(self, use_s2=True, use_s1=True,
                 d_model=256, n_heads=4, dropout=0.2):
        super().__init__()
        assert use_s2 or use_s1, "need at least one modality"
        self.use_s2, self.use_s1 = use_s2, use_s1
        self.d_model = d_model

        if use_s2:
            self.proj_s2 = nn.Linear(FEATURE_DIM, d_model)
            self.type_s2 = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        if use_s1:
            self.proj_s1 = nn.Linear(FEATURE_DIM, d_model)
            self.type_s1 = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        self.pos = nn.Parameter(torch.randn(1, N_TOKENS, d_model) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.mixer = nn.TransformerEncoder(layer, num_layers=1)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, NUM_CLASSES),
        )

    def encode(self, s2=None, s1=None):
        """Fused token sequence: (B, 16 or 32, d_model)."""
        parts = []
        if self.use_s2:
            assert s2 is not None, "model built with use_s2=True but got s2=None"
            parts.append(self.proj_s2(s2) + self.pos + self.type_s2)
        if self.use_s1:
            assert s1 is not None, "model built with use_s1=True but got s1=None"
            parts.append(self.proj_s1(s1) + self.pos + self.type_s1)
        return self.mixer(torch.cat(parts, dim=1))

    def forward(self, s2=None, s1=None):
        return self.head(self.encode(s2, s1).mean(dim=1))