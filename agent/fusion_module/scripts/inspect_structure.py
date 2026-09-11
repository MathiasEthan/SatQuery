# scripts/inspect_structure.py
import torch
from reben_publication.BigEarthNetv2_0_ImageClassifier import BigEarthNetv2_0_ImageClassifier

model = BigEarthNetv2_0_ImageClassifier.from_pretrained(
    "BIFOLD-BigEarthNetv2-0/resnet50-s2-v0.2.0"
)
model.eval()

def walk(module, prefix="", depth=0, max_depth=2):
    for name, child in module.named_children():
        path = f"{prefix}.{name}" if prefix else name
        n = sum(p.numel() for p in child.parameters())
        print("  " * depth + f"{path:35s} {type(child).__name__:25s} {n/1e6:7.2f}M")
        if depth < max_depth:
            walk(child, path, depth + 1, max_depth)

walk(model)