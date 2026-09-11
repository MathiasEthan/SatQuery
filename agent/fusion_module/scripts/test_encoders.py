# scripts/test_encoders.py
import torch
from src.encoders import FrozenEncoder
from src.device import get_device

device = get_device()

for modality, n_bands in [("s2", 10), ("s1", 2)]:
    enc = FrozenEncoder(modality).to(device)
    x = torch.randn(2, n_bands, 120, 120, device=device)
    feats = enc(x)
    print(f"{modality}: in {tuple(x.shape)} -> out {tuple(feats.shape)}  dim={enc.feature_dim}")

    tokens = feats.flatten(2).transpose(1, 2)
    print(f"     as tokens: {tuple(tokens.shape)}")

# prove the guard rails fire
enc = FrozenEncoder("s2").to(device)
for bad in [torch.randn(2, 3, 120, 120), torch.randn(2, 10, 64, 64)]:
    try:
        enc(bad.to(device))
        print("!!! NO ERROR — guard failed")
    except AssertionError as e:
        print("caught:", e)