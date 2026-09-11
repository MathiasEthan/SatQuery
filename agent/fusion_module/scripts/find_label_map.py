# scripts/find_label_map.py
import pkgutil, importlib
import configilm.extra as extra

for m in pkgutil.iter_modules(extra.__path__):
    try:
        mod = importlib.import_module(f"configilm.extra.{m.name}")
    except Exception as e:
        print(m.name, "-> import failed:", type(e).__name__)
        continue
    hits = [n for n in dir(mod)
            if not n.startswith("_")
            and any(k in n.lower() for k in ["label", "map", "old", "new", "43", "19"])]
    if hits:
        print(f"{m.name}: {hits}")