# src/config.py
import os
from pathlib import Path

_DEFAULT = Path("/Users/burhanuddinamin/Documents/projects/SatQuery AI/dataset/ben-ge-8k")

DATA_ROOT = Path(os.environ.get("SATQUERY_DATA_ROOT", _DEFAULT))

META_CSV   = DATA_ROOT / "ben-ge-8k_meta.csv"
SPLITS_DIR = DATA_ROOT / "splits"
S1_DIR     = DATA_ROOT / "sentinel-1"
S2_DIR     = DATA_ROOT / "sentinel-2"

CKPT_PATH = Path(os.environ.get(
    "SATQUERY_FUSION_CKPT",
    Path(__file__).resolve().parent.parent / "checkpoints" / "fused_seed0.pt",
))