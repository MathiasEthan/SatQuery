# scripts/find_band_order.py
import configilm.extra.BENv2_utils as u

names = [n for n in dir(u) if not n.startswith("_")]
print(names)

for n in names:
    if "band" in n.lower() or "BAND" in n:
        print("\n---", n, "---")
        print(getattr(u, n))