# scripts/inspect_config.py
import json
from huggingface_hub import hf_hub_download

for repo in ["BIFOLD-BigEarthNetv2-0/resnet50-s2-v0.2.0",
             "BIFOLD-BigEarthNetv2-0/resnet50-s1-v0.2.0"]:
    path = hf_hub_download(repo, "config.json")
    print("=" * 60)
    print(repo)
    print(json.dumps(json.load(open(path)), indent=2))