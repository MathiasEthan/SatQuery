# src/dataset.py
import json

import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import Dataset

from configilm.extra.BENv2_utils import band_combi_to_mean_std
from src.config import META_CSV, SPLITS_DIR, S1_DIR, S2_DIR
from src.encoders import S1_BANDS, S2_BANDS
from src.labels import old_labels_to_multihot, NUM_CLASSES

INTERPOLATION = "120_nearest"
TARGET_SIZE = 120


def _stats(bands):
    mean, std = band_combi_to_mean_std(bands, interpolation=INTERPOLATION)
    return (np.asarray(mean, dtype=np.float32).reshape(-1, 1, 1),
            np.asarray(std, dtype=np.float32).reshape(-1, 1, 1))


S2_MEAN, S2_STD = _stats(S2_BANDS)
S1_MEAN, S1_STD = _stats(S1_BANDS)


def _read_labels(patch_id):
    path = S2_DIR / patch_id / f"{patch_id}_labels_metadata.json"
    with open(path) as f:
        return old_labels_to_multihot(json.load(f)["labels"])


def _read_band(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
    if arr.shape[0] != TARGET_SIZE:
        factor = TARGET_SIZE // arr.shape[0]
        assert factor * arr.shape[0] == TARGET_SIZE, f"bad size {arr.shape} in {path}"
        arr = np.repeat(np.repeat(arr, factor, axis=0), factor, axis=1)
    assert arr.shape == (TARGET_SIZE, TARGET_SIZE), f"{path}: {arr.shape}"
    return arr


class BenGeFusionDataset(Dataset):
    def __init__(self, split="train", verbose=True):
        meta = pd.read_csv(META_CSV)
        ids = pd.read_csv(SPLITS_DIR / f"ben-ge-8k_{split}.csv",
                          header=None, names=["patch_id"])
        merged = ids.merge(meta[["patch_id", "patch_id_s1"]],
                           on="patch_id", how="left")
        missing = int(merged["patch_id_s1"].isna().sum())
        assert missing == 0, f"{missing} patches in '{split}' have no S1 pair"

        keep, labels = [], []
        for pid in merged["patch_id"]:
            vec = _read_labels(pid)
            if vec is not None:
                keep.append(pid)
                labels.append(vec)

        dropped = len(merged) - len(keep)
        if verbose:
            print(f"[{split}] kept {len(keep)}, dropped {dropped} "
                  f"with no 19-class label")
        assert len(keep) > 0, f"'{split}' has no usable patches"

        self.pairs = merged[merged["patch_id"].isin(keep)].reset_index(drop=True)
        self.labels = {pid: vec for pid, vec in zip(keep, labels)}
        self.split = split
        self.num_classes = NUM_CLASSES

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        row = self.pairs.iloc[i]
        s2_id, s1_id = row["patch_id"], row["patch_id_s1"]

        s2 = np.stack([_read_band(S2_DIR / s2_id / f"{s2_id}_{b}.tif") for b in S2_BANDS])
        s1 = np.stack([_read_band(S1_DIR / s1_id / f"{s1_id}_{b}.tif") for b in S1_BANDS])

        s2 = (s2 - S2_MEAN) / S2_STD
        s1 = (s1 - S1_MEAN) / S1_STD

        return {
            "s2": torch.from_numpy(s2).float(),
            "s1": torch.from_numpy(s1).float(),
            "label": torch.from_numpy(self.labels[s2_id]),
            "patch_id": s2_id,
            "patch_id_s1": s1_id,
        }