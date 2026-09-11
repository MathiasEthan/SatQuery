# scripts/test_fusion.py
import torch
from src.fusion import FusionClassifier
from src.cached_dataset import CachedFeatures

ds = CachedFeatures("val")
b = {k: v.unsqueeze(0) for k, v in ds[0].items()}

for name, kw in [("s2-only", dict(use_s2=True, use_s1=False)),
                 ("s1-only", dict(use_s2=False, use_s1=True)),
                 ("fused",   dict(use_s2=True, use_s1=True))]:
    m = FusionClassifier(**kw)
    out = m(s2=b["s2"] if kw["use_s2"] else None,
            s1=b["s1"] if kw["use_s1"] else None)
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"{name:8s} out={tuple(out.shape)}  trainable={n/1e6:.2f}M")