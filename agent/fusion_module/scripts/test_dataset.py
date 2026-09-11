# scripts/test_dataset.py
from src.dataset import BenGeFusionDataset

ds = BenGeFusionDataset("train")
print("patches:", len(ds))

s = ds[0]
print("id:", s["patch_id"], "->", s["patch_id_s1"])
for k in ["s2", "s1"]:
    t = s[k]
    print(f"{k}: {tuple(t.shape)}  mean={t.mean():+.3f}  std={t.std():.3f}  "
          f"min={t.min():+.2f}  max={t.max():+.2f}")