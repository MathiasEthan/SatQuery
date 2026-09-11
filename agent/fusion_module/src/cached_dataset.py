# src/cached_dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset

from src.config import DATA_ROOT

CACHE = DATA_ROOT.parent / "cache"


class CachedFeatures(Dataset):
    def __init__(self, split):
        d = np.load(CACHE / f"{split}.npz", allow_pickle=True)
        self.s2 = torch.from_numpy(d["s2"])          # (N, 16, 2048) float16
        self.s1 = torch.from_numpy(d["s1"])
        self.label = torch.from_numpy(d["label"]).float()
        self.patch_id = d["patch_id"]
        assert len(self.s2) == len(self.s1) == len(self.label)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, i):
        return {
            "s2": self.s2[i].float(),
            "s1": self.s1[i].float(),
            "label": self.label[i],
        }


class ShuffledFeatures(CachedFeatures):
    """Control: radar deliberately paired with the WRONG patch.

    Same architecture, same parameter count, same training — only the
    correspondence between the two modalities is destroyed. If the fused
    score drops to camera-only level here, the gain came from real radar
    information rather than from extra model capacity.
    """

    def __init__(self, split, seed=0):
        super().__init__(split)
        g = torch.Generator().manual_seed(seed)
        self.perm = torch.randperm(len(self.label), generator=g)

    def __getitem__(self, i):
        return {
            "s2": self.s2[i].float(),
            "s1": self.s1[self.perm[i]].float(),
            "label": self.label[i],
        }