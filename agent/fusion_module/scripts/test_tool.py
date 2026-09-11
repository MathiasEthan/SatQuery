# scripts/test_tool.py
from src.tool import dispatch

near = dispatch("find_patches_near", {"lat": 45.54, "lon": 19.43, "k": 3})
print(near)

pid = near["matches"][0]["patch_id"]
print(dispatch("analyze_land_cover", {"patch_id": pid, "top_k": 5}))