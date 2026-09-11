# scripts/find_mapping.py
import bigearthnet_common.constants as c

names = [n for n in dir(c) if not n.startswith("_")]
print(names)

for n in names:
    if any(k in n.lower() for k in ["43", "19", "old", "new", "map"]):
        v = getattr(c, n)
        print(f"\n--- {n} ({type(v).__name__}) ---")
        print(v)