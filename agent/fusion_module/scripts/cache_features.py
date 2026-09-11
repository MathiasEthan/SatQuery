# scripts/cache_features.py
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import BenGeFusionDataset
from src.encoders import FrozenEncoder
from src.device import get_device
from src.config import DATA_ROOT


def main():
    OUT = DATA_ROOT.parent / "cache"
    OUT.mkdir(exist_ok=True)
    device = get_device()

    encoders = {m: FrozenEncoder(m).to(device) for m in ["s2", "s1"]}

    for split in ["train", "val", "test"]:
        ds = BenGeFusionDataset(split)
        dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)

        feats = {"s2": [], "s1": []}
        labels, ids = [], []

        for batch in dl:
            for m in ["s2", "s1"]:
                f = encoders[m](batch[m].to(device))     # (B, 2048, 4, 4)
                f = f.flatten(2).transpose(1, 2)         # (B, 16, 2048)
                feats[m].append(f.cpu().numpy().astype(np.float16))
            labels.append(batch["label"].numpy())
            ids.extend(batch["patch_id"])
            print(f"\r{split}: {len(ids)}/{len(ds)}", end="")

        np.savez(
            OUT / f"{split}.npz",
            s2=np.concatenate(feats["s2"]),
            s1=np.concatenate(feats["s1"]),
            label=np.concatenate(labels),
            patch_id=np.array(ids),
        )
        print(f"  -> saved {OUT / f'{split}.npz'}")


if __name__ == "__main__":
    main()