# src/tool.py
"""LLM-callable optical + SAR fusion tool.

Setup:
    export SATQUERY_DATA_ROOT=/path/to/ben-ge-8k
    export SATQUERY_FUSION_CKPT=/path/to/fused_seed0.pt   # optional

The agent passes a Sentinel-2 patch id or a coordinate. It never handles
Sentinel-1 ids, band ordering, or normalization: a mismatched optical/radar
pair produces confident wrong answers with no error, so pairing is resolved
internally from the dataset metadata.
"""

from src.config import CKPT_PATH, DATA_ROOT, META_CSV, S1_DIR, S2_DIR

_predictor = None


def check_setup():
    """Verify everything is in place. Returns {"ok": True} or a list of problems."""
    problems = []
    for label, path in [("dataset root", DATA_ROOT), ("metadata csv", META_CSV),
                        ("sentinel-1 dir", S1_DIR), ("sentinel-2 dir", S2_DIR)]:
        if not path.exists():
            problems.append(f"missing {label}: {path}  (set SATQUERY_DATA_ROOT)")
    if not CKPT_PATH.exists():
        problems.append(f"missing checkpoint: {CKPT_PATH}  (set SATQUERY_FUSION_CKPT)")
    return {"ok": True} if not problems else {"ok": False, "problems": problems}


def warmup():
    """Load the model. Call once at service startup — first load takes ~10s."""
    return get_predictor().test_metrics


def get_predictor():
    global _predictor
    if _predictor is None:
        status = check_setup()
        if not status["ok"]:
            raise RuntimeError("fusion tool not configured:\n  " +
                               "\n  ".join(status["problems"]))
        from src.inference import FusionPredictor
        _predictor = FusionPredictor(ckpt_path=CKPT_PATH)
    return _predictor


def analyze_land_cover(patch_id: str, top_k: int = 5) -> dict:
    try:
        return get_predictor().predict(patch_id, top_k=top_k)
    except KeyError as e:
        return {"error": "unknown_patch_id", "detail": str(e),
                "hint": "use find_patches_near to get a valid patch_id"}
    except FileNotFoundError as e:
        return {"error": "patch_files_missing", "detail": str(e)}
    except Exception as e:
        return {"error": "prediction_failed", "detail": f"{type(e).__name__}: {e}"}


def find_patches_near(lat: float, lon: float, k: int = 5) -> dict:
    try:
        return {"query": {"lat": lat, "lon": lon},
                "matches": get_predictor().find_nearest(lat, lon, k=k)}
    except Exception as e:
        return {"error": "lookup_failed", "detail": f"{type(e).__name__}: {e}"}


TOOLS = {
    "analyze_land_cover": analyze_land_cover,
    "find_patches_near": find_patches_near,
}

TOOL_SPECS = [
    {
        "name": "analyze_land_cover",
        "description": (
            "Identify land-cover types in a satellite patch by fusing Sentinel-2 "
            "optical imagery with Sentinel-1 radar. Radar penetrates cloud and works "
            "at night, so this is more reliable than optical alone, particularly for "
            "water, wetlands and built-up areas. Returns the 19 BigEarthNet land-cover "
            "classes ranked by probability. Requires a known Sentinel-2 patch id; call "
            "find_patches_near first if you only have a place name or coordinates."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "patch_id": {
                    "type": "string",
                    "description": "Sentinel-2 patch id, e.g. 'S2A_MSIL2A_20171002T094031_64_46'. Must come from find_patches_near or from the user — do not construct one.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "How many highest-probability classes to list. Default 5.",
                },
            },
            "required": ["patch_id"],
        },
    },
    {
        "name": "find_patches_near",
        "description": (
            "Find satellite patches closest to a latitude/longitude, with their distance "
            "in km. Use this to turn a coordinate into patch ids that analyze_land_cover "
            "accepts. Coverage is limited to the ben-ge-8k dataset, so the nearest patch "
            "may still be far from the query — check approx_km before relying on it."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "lat": {"type": "number", "description": "Latitude in degrees."},
                "lon": {"type": "number", "description": "Longitude in degrees."},
                "k": {"type": "integer", "description": "How many patches to return. Default 5."},
            },
            "required": ["lat", "lon"],
        },
    },
]


def dispatch(name: str, arguments: dict) -> dict:
    """Route a tool call from the agent. Always returns a JSON-serializable dict."""
    if name not in TOOLS:
        return {"error": "unknown_tool", "detail": name,
                "available": list(TOOLS)}
    try:
        return TOOLS[name](**arguments)
    except TypeError as e:
        return {"error": "bad_arguments", "detail": str(e)}