# scripts/test_labels.py
import torch
from src.dataset import BenGeFusionDataset
from src.labels import CLASSES

ds = BenGeFusionDataset("train")
s = ds[0]
print(s["patch_id"], tuple(s["label"].shape))
print("positive:", [CLASSES[i] for i in torch.nonzero(s["label"]).flatten().tolist()])

counts = torch.stack([torch.from_numpy(v) for v in ds.labels.values()]).sum(0)
for name, c in sorted(zip(CLASSES, counts.tolist()), key=lambda x: -x[1]):
    print(f"  {int(c):5d}  {name}")