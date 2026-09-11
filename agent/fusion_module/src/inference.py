# src/inference.py
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.config import META_CSV, S1_DIR, S2_DIR
from src.dataset import S1_MEAN, S1_STD, S2_MEAN, S2_STD, _read_band
from src.device import get_device
from src.encoders import FrozenEncoder, S1_BANDS, S2_BANDS
from src.fusion import FusionClassifier
from src.labels import CLASSES

DEFAULT_CKPT = Path("checkpoints/fused_seed0.pt")


class FusionPredictor:
    """Optical + SAR land-cover predictor. Build once, call many times."""

    def __init__(self, ckpt_path=DEFAULT_CKPT, device=None):
        self.device = device or get_device()
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        self.config_name = ckpt["config_name"]
        self.model_kwargs = ckpt["model_kwargs"]
        self.test_metrics = ckpt["test_metrics"]
        self.classes = list(CLASSES)

        self.model = FusionClassifier(**self.model_kwargs).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

        self.encoders = {}
        if self.model_kwargs["use_s2"]:
            self.encoders["s2"] = FrozenEncoder("s2").to(self.device).eval()
        if self.model_kwargs["use_s1"]:
            self.encoders["s1"] = FrozenEncoder("s1").to(self.device).eval()

        meta = pd.read_csv(META_CSV)
        self.meta = meta.set_index("patch_id")
        self._meta_reset = meta

    # ---------- pairing ----------

    def resolve_pair(self, patch_id):
        """Sentinel-2 patch id -> its matching Sentinel-1 patch id."""
        if patch_id not in self.meta.index:
            raise KeyError(f"unknown patch_id: {patch_id!r}")
        return str(self.meta.loc[patch_id, "patch_id_s1"])

    def find_nearest(self, lat, lon, k=5):
        """Closest patches to a coordinate. Rough planar distance, fine at patch scale."""
        df = self._meta_reset
        d = np.hypot(df["lat"].values - lat,
                     (df["lon"].values - lon) * np.cos(np.radians(lat)))
        idx = np.argsort(d)[:k]
        return [
            {
                "patch_id": str(df["patch_id"].iloc[i]),
                "lat": float(df["lat"].iloc[i]),
                "lon": float(df["lon"].iloc[i]),
                "approx_km": round(float(d[i]) * 111.0, 2),
            }
            for i in idx
        ]

    # ---------- input loading ----------

    def _load_patch(self, patch_id, s1_patch_id):
        out = {}
        if self.model_kwargs["use_s2"]:
            arr = np.stack([_read_band(S2_DIR / patch_id / f"{patch_id}_{b}.tif")
                            for b in S2_BANDS])
            out["s2"] = torch.from_numpy((arr - S2_MEAN) / S2_STD).float()
        if self.model_kwargs["use_s1"]:
            arr = np.stack([_read_band(S1_DIR / s1_patch_id / f"{s1_patch_id}_{b}.tif")
                            for b in S1_BANDS])
            out["s1"] = torch.from_numpy((arr - S1_MEAN) / S1_STD).float()
        return out

    def _batch(self, patch_id):
        s1_id = self.resolve_pair(patch_id)
        loaded = self._load_patch(patch_id, s1_id)
        feats = {}
        for m, img in loaded.items():
            f = self.encoders[m](img.unsqueeze(0).to(self.device))  # (1, 2048, 4, 4)
            feats[m] = f.flatten(2).transpose(1, 2)                 # (1, 16, 2048)
        return feats, s1_id

    # ---------- public API ----------

    @torch.no_grad()
    def predict(self, patch_id, threshold=0.5, top_k=None):
        batch, s1_id = self._batch(patch_id)
        probs = torch.sigmoid(self.model(**batch))[0].float().cpu().numpy()

        ranked = sorted(zip(self.classes, probs.tolist()), key=lambda x: -x[1])
        labels = ([c for c, _ in ranked[:top_k]] if top_k is not None
                  else [c for c, p in ranked if p >= threshold])

        row = self.meta.loc[patch_id]
        return {
            "patch_id": patch_id,
            "s1_patch_id": s1_id,
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            "labels": labels,
            "probabilities": {c: round(p, 4) for c, p in ranked},
            "modalities_used": [m for m in ("s2", "s1") if self.model_kwargs[f"use_{m}"]],
            "model_test_mAP": round(self.test_metrics["mAP"], 4),
        }

    @torch.no_grad()
    def fused_tokens(self, patch_id):
        """Fused representation, (n_tokens, d_model). Input for Stage 2."""
        batch, _ = self._batch(patch_id)
        return self.model.encode(**batch)[0].cpu()