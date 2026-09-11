# scripts/sanity_check_labels.py
import json
import torch
from reben_publication.BigEarthNetv2_0_ImageClassifier import BigEarthNetv2_0_ImageClassifier
from configilm.extra.BENv2_utils import NEW_LABELS
from src.dataset import BenGeFusionDataset
from src.config import S2_DIR
from src.device import get_device

device = get_device()
clf = BigEarthNetv2_0_ImageClassifier.from_pretrained(
    "BIFOLD-BigEarthNetv2-0/resnet50-s2-v0.2.0").to(device)
clf.eval()

ds = BenGeFusionDataset("train")

for i in range(6):
    s = ds[i]
    with torch.no_grad():
        probs = torch.sigmoid(clf(s["s2"].unsqueeze(0).to(device)))[0].cpu()
    top = torch.topk(probs, 3)

    meta = json.load(open(S2_DIR / s["patch_id"] / f"{s['patch_id']}_labels_metadata.json"))

    print(f"\n{s['patch_id']}")
    print("  true:", meta["labels"])
    print("  pred:", [(NEW_LABELS[j], round(float(p), 2))
                      for p, j in zip(top.values, top.indices)])