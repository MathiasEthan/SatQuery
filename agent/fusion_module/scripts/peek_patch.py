# scripts/peek_patch.py
import glob, os
import rasterio

ROOT = "/Users/burhanuddinamin/Documents/projects/SatQuery AI/dataset/ben-ge-8k"

s2_dir = sorted(glob.glob(os.path.join(ROOT, "sentinel-2", "*")))[0]
s1_dir = sorted(glob.glob(os.path.join(ROOT, "sentinel-1", "*")))[0]

print("S2 folder:", os.path.basename(s2_dir))
for f in sorted(os.listdir(s2_dir)):
    if f.endswith(".tif"):
        with rasterio.open(os.path.join(s2_dir, f)) as src:
            arr = src.read(1)
            print(f"  {f[-7:]:8s} shape={arr.shape}  min={arr.min():6d}  max={arr.max():6d}  dtype={arr.dtype}")

print("\nS1 folder:", os.path.basename(s1_dir))
for f in sorted(os.listdir(s1_dir)):
    if f.endswith(".tif"):
        with rasterio.open(os.path.join(s1_dir, f)) as src:
            arr = src.read(1)
            print(f"  {f[-6:]:8s} shape={arr.shape}  min={arr.min():.2f}  max={arr.max():.2f}  dtype={arr.dtype}")