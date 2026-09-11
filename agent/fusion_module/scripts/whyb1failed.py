# scripts/why_benv1_failed.py
import importlib, traceback

try:
    importlib.import_module("configilm.extra.BENv1_utils")
except Exception:
    traceback.print_exc()