# scripts/make_demo_bundle.py
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.config import META_CSV, S1_DIR, S2_DIR

OUT = Path("demo_bundle/ben-ge-demo")


def main(patch_ids):
    meta = pd.read_csv(META_CSV)
    subset = meta[meta["patch_id"].isin(patch_ids)]

    missing = set(patch_ids) - set(subset["patch_id"])
    if missing:
        raise SystemExit(f"not in meta.csv: {missing}")

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "sentinel-2").mkdir(parents=True)
    (OUT / "sentinel-1").mkdir(parents=True)

    for _, row in subset.iterrows():
        s2_id, s1_id = row["patch_id"], row["patch_id_s1"]
        shutil.copytree(S2_DIR / s2_id, OUT / "sentinel-2" / s2_id)
        shutil.copytree(S1_DIR / s1_id, OUT / "sentinel-1" / s1_id)
        print(f"copied {s2_id}  <->  {s1_id}")

    subset.to_csv(OUT / "ben-ge-8k_meta.csv", index=False)

    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"\nbundle: {OUT}  ({size/1e6:.1f} MB, {len(subset)} pairs)")

    archive = shutil.make_archive("demo_bundle/ben-ge-demo", "zip",
                                  root_dir="demo_bundle", base_dir="ben-ge-demo")
    print(f"archive: {archive}")

    print("\n--- verifying bundle is self-sufficient ---")
    subprocess.run(
        [sys.executable, "-m", "scripts.test_tool"],
        env={**__import__("os").environ,
             "SATQUERY_DATA_ROOT": str(OUT.resolve())},
        check=True,
    )


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--first":
        n = int(args[1]) if len(args) > 1 else 40
        ids = pd.read_csv(META_CSV)["patch_id"].head(n).tolist()
    elif args:
        ids = args
    else:
        raise SystemExit("usage: python -m scripts.make_demo_bundle [--first N | <patch_id> ...]")
    main(ids)