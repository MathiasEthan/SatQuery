# scripts/shortlist_patches.py
import pandas as pd
from src.config import META_CSV
from src.tool import dispatch, warmup


def main():
    warmup()
    ids = pd.read_csv(META_CSV)["patch_id"].head(40)

    for pid in ids:
        r = dispatch("analyze_land_cover", {"patch_id": pid, "top_k": 3})
        if "error" in r:
            continue
        top = ", ".join(f"{c} {r['probabilities'][c]:.2f}" for c in r["labels"])
        print(f"{r['lat']:8.3f} {r['lon']:8.3f}  {pid}\n    {top}\n")


if __name__ == "__main__":
    main()