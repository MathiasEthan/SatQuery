# Optical + SAR fusion tool

Land-cover classification from paired Sentinel-2 (optical) and Sentinel-1 (radar)
imagery. Exposed as two agent-callable tools.

## Setup

    pip install -r requirements.txt
    export SATQUERY_DATA_ROOT=/path/to/ben-ge-8k

Requires the ben-ge-8k dataset on disk (~4.2 GB, Zenodo record 8121208).
The two pretrained encoders download automatically from HuggingFace on first
run (~190 MB, cached in ~/.cache/huggingface).

Verify:

    python -c "from src.tool import check_setup; print(check_setup())"

## Integration

    from src.tool import TOOL_SPECS, dispatch, warmup

    warmup()                      # once at startup, ~10s
    result = dispatch(name, args) # per tool call

`TOOL_SPECS` goes into the `tools` parameter of the LLM call.
`dispatch` always returns a JSON-serializable dict, never raises.

## Tools

**find_patches_near(lat, lon, k=5)** — coordinate to patch ids, with approx_km.
**analyze_land_cover(patch_id, top_k=5)** — ranked land-cover classes.

Typical agent flow: place name → coordinates → find_patches_near →
analyze_land_cover.

## Notes for the agent prompt

- Never invent a patch_id. Get one from find_patches_near or the user.
- Coverage is only the 7,991 ben-ge-8k patches, spread across Europe. If
  approx_km is large, say so rather than presenting the result as the
  queried location.
- Every result carries the full `probabilities` dict. Prefer it over `labels`
  — the 0.5 threshold is untuned, and "Inland wetlands 0.47" is useful
  information that the label list silently drops.
- `model_test_mAP` (0.81) rides along in every result. Use it to caveat.

## Performance

Test mAP, mean of 3 seeds:

| Configuration | mAP |
|---|---|
| Radar only | 0.731 |
| Optical only | 0.787 |
| Optical + radar, **shuffled control** | 0.784 |
| **Optical + radar (this model)** | **0.810** |

The shuffled control uses the identical architecture and parameter count but
pairs each optical patch with a random patch's radar. It recovers optical-only
performance, confirming the gain comes from genuine cross-modal correspondence
rather than added model capacity.