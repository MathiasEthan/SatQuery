# scripts/load_encoder.py
import torch
from reben_publication.BigEarthNetv2_0_ImageClassifier import BigEarthNetv2_0_ImageClassifier
from src.device import get_device

device = get_device()

model = BigEarthNetv2_0_ImageClassifier.from_pretrained(
    "BIFOLD-BigEarthNetv2-0/resnet50-s2-v0.2.0"
).to(device)

print("type:", type(model))
print("params (M):", round(sum(p.numel() for p in model.parameters()) / 1e6, 1))

x = torch.randn(2, 10, 120, 120, device=device)

# training mode: dropout is ACTIVE
model.train()
a1 = model(x)
a2 = model(x)
print("train mode  — two runs identical:", torch.allclose(a1, a2))

# eval mode: dropout is OFF
model.eval()
with torch.no_grad():
    b1 = model(x)
    b2 = model(x)
print("eval  mode  — two runs identical:", torch.allclose(b1, b2))
print("output shape:", tuple(b1.shape))