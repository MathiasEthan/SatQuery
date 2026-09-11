# scripts/peek_meta.py
import pandas as pd

ROOT = "/Users/burhanuddinamin/Documents/projects/SatQuery AI/dataset/ben-ge-8k"

meta = pd.read_csv(f"{ROOT}/ben-ge-8k_meta.csv")
print(meta.shape)
print(meta.columns.tolist())
print(meta.head(3).to_string())

train = pd.read_csv(SPLITS_DIR / "ben-ge-8k_train.csv", header=None, names=["patch_id"])
print("\ntrain split:", train.shape)
print(train.head(3).to_string())