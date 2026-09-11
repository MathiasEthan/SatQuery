from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableLambda
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langchain.tools import tool
from typing import Literal
import os
import sys
import shutil
import re
import math
import uuid
import requests
import pandas as pd
from PIL import Image, ImageDraw
from change_agent.change_agent import run_model
from visualizer import process_and_visualize
from vqa_agent import run_vqa
from captioning_agent import run_captioning

FUSION_MODULE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "fusion_module"
)
if FUSION_MODULE_DIR not in sys.path:
    sys.path.append(FUSION_MODULE_DIR)

try:
    from fusion_module.src.tool import analyze_land_cover, find_patches_near
    from fusion_module.src.config import S2_DIR, DATA_ROOT, META_CSV
except ImportError:
    analyze_land_cover = None
    find_patches_near = None
    S2_DIR = None
    DATA_ROOT = None
    META_CSV = None


def resolve_patch_id(candidate: str) -> str:
    """Helper to extract or map a patch ID from a filepath, string, or S1 ID."""
    if not candidate:
        return ""
    name = os.path.basename(candidate).strip()
    name = os.path.splitext(name)[0]
    # Remove standard band / channel suffixes
    cleaned = re.sub(
        r"_(rgb|B01|B02|B03|B04|B05|B06|B07|B08|B09|B11|B12|B8A|VV|VH|labels_metadata)$",
        "",
        name,
        flags=re.IGNORECASE,
    )
    # Check for S2 patch ID pattern
    s2_match = re.search(r"(S2[AB]_MSIL2A_[0-9A-Za-z_]+)", cleaned)
    if s2_match:
        matched = s2_match.group(1)
        return re.sub(
            r"_(rgb|B01|B02|B03|B04|B05|B06|B07|B08|B09|B11|B12|B8A)$",
            "",
            matched,
            flags=re.IGNORECASE,
        )
    # Check for S1 patch ID pattern
    s1_match = re.search(r"(S1[AB]_IW_GRDH_[0-9A-Za-z_]+)", cleaned)
    if s1_match:
        s1_id = re.sub(r"_(VV|VH)$", "", s1_match.group(1), flags=re.IGNORECASE)
        try:
            if META_CSV and os.path.exists(META_CSV):
                df = pd.read_csv(META_CSV)
                match = df[df["patch_id_s1"] == s1_id]
                if not match.empty:
                    return str(match.iloc[0]["patch_id"])
        except Exception:
            pass

    # If candidate is a file on disk (e.g. uploaded .tif or GeoTIFF)
    if candidate and os.path.exists(candidate) and candidate.lower().endswith((".tif", ".tiff")):
        try:
            import rasterio
            from rasterio.warp import transform_bounds
            with rasterio.open(candidate) as src:
                if src.crs is not None and src.bounds is not None:
                    bounds_4326 = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
                    center_lon = (bounds_4326[0] + bounds_4326[2]) / 2.0
                    center_lat = (bounds_4326[1] + bounds_4326[3]) / 2.0
                    if find_patches_near is not None:
                        res_near = find_patches_near(center_lat, center_lon, k=1)
                        if "matches" in res_near and len(res_near["matches"]) > 0:
                            match = res_near["matches"][0]
                            if match.get("approx_km", 999) < 5.0:
                                return match["patch_id"]
        except Exception:
            pass

    return cleaned


def parse_bboxes(raw_text: str, image_width: int, image_height: int):
    """
    Parse GeoChat grounding output into plottable boxes.

    GeoChat encodes each box as {<x0><y0><x1><y1>|<angle>} on a 0-100 scale,
    where angle (degrees) rotates the box about its own center - these are
    oriented bounding boxes, not axis-aligned ones. Grounded-caption style
    output wraps the referring phrase in <p>...</p> immediately before its
    box(es); plain grounding/referring output has no <p> tags at all.

    Each returned box has:
      - "label": the referring phrase, or None if untagged
      - "polygon": the 4 rotated corners [(x, y), (x, y), (x, y), (x, y)],
                    in pixel coords, in order TL, TR, BR, BL (pre-rotation)
      - "box_2d": [xmin, ymin, xmax, ymax] axis-aligned box around the
                   rotated polygon (handy for cropping / simple boxes)
      - "angle": the raw angle in degrees, kept for reference
    """

    def decode_blocks(text):
        decoded = []
        for block in re.findall(r"\{<[^}]+>\}", text):
            integers = [int(x) for x in re.findall(r"-?\d+", block)]
            if len(integers) < 4:
                continue
            x0, y0, x1, y1 = integers[:4]
            angle = integers[4] if len(integers) > 4 else 0

            # scale percent -> pixels
            xmin = (x0 / 100.0) * image_width
            xmax = (x1 / 100.0) * image_width
            ymin = (y0 / 100.0) * image_height
            ymax = (y1 / 100.0) * image_height
            xmin, xmax = min(xmin, xmax), max(xmin, xmax)
            ymin, ymax = min(ymin, ymax), max(ymin, ymax)

            cx = xmin + (xmax - xmin) / 2.0
            cy = ymin + (ymax - ymin) / 2.0

            corners = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]

            angle_rad = math.radians(angle)
            cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)

            polygon = []
            for x, y in corners:
                tx, ty = x - cx, y - cy
                rx = tx * cos_a - ty * sin_a + cx
                ry = tx * sin_a + ty * cos_a + cy
                polygon.append((rx, ry))

            xs = [p[0] for p in polygon]
            ys = [p[1] for p in polygon]
            axis_aligned = [min(xs), min(ys), max(xs), max(ys)]

            decoded.append((polygon, axis_aligned, angle))
        return decoded

    boxes = []
    entity_matches = re.findall(
        r"<p>(.*?)</p>(.*?)(?=(?:<p>|$))", raw_text, flags=re.DOTALL
    )

    if entity_matches:
        for label, bbox_part in entity_matches:
            for polygon, axis_aligned, angle in decode_blocks(bbox_part):
                boxes.append(
                    {
                        "label": label.strip(),
                        "polygon": polygon,
                        "box_2d": axis_aligned,
                        "angle": angle,
                    }
                )
        return boxes

    for polygon, axis_aligned, angle in decode_blocks(raw_text):
        boxes.append(
            {
                "label": None,
                "polygon": polygon,
                "box_2d": axis_aligned,
                "angle": angle,
            }
        )
    return boxes


load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
geochat_url = os.getenv("geochat_url")
backend_url = os.getenv("backend_url")


@tool
def vqa_tool(image_path: str, query: str) -> dict:
    """Executes visual question answering on a single optical, multispectral, or SAR image to answer natural-language queries about land cover, objects, or features."""
    try:
        answer = run_vqa(image_path, query)
    except RuntimeError as e:
        return {"analysis": str(e), "img": None}
    return {"analysis": answer, "img": None}


@tool
def grounding_tool(image_path: str, query: str) -> dict:
    """Performs text-guided region grounding on church.png to locate and outline specific objects or areas mentioned in the query."""
    with open(image_path, "rb") as f:
        files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
        data = {"text_prompt": "[grounding]" + query}
        response = requests.post(f"{geochat_url}/grounding", files=files, data=data)
    res_json = response.json()
    raw_text = res_json.get("text", "")
    print(raw_text)
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    boxes = parse_bboxes(raw_text, width, height)

    draw = ImageDraw.Draw(image)
    for box in boxes:
        draw.polygon(box["polygon"], outline="red", width=3)
        if box["label"]:
            x0, y0 = box["polygon"][0]
            draw.text((x0, y0 - 10), box["label"], fill="red")
    req_id = uuid.uuid4().hex[:8]
    output_path = "tmp/" + req_id + "_annotated.png"
    image.save(output_path)

    return {
        "analysis": "here are the groundings yuo requested",
        "img": backend_url + "/outputs/" + req_id + "_annotated.png",
    }


@tool
def captioning_tool(image_path: str, query: str) -> dict:
    """Generates a comprehensive scene description and land-cover summary for a single remote-sensing image."""
    try:
        caption = run_captioning(image_path, query)
    except RuntimeError as e:
        return {"analysis": str(e), "img": None}
    return {"analysis": caption, "img": None}


@tool
def change_analysis_tool(image_path_t1: str, image_path_t2: str, query: str) -> dict:
    """Analyzes a bi-temporal image pair to detect surface changes, describe alterations over time, and answer change-based visual questions with spatial change map outputs where applicable."""
    # 1. Full change analysis with bounding boxes & prior land-cover:
    req_id = uuid.uuid4().hex[:8]
    analysis = run_model(image_path_t1, image_path_t2, task="analyze")
    output_path = f"{req_id}_comparision.png"
    process_and_visualize(
        analysis, image_path_t2, image_path_t1, output_path=output_path
    )
    return {"analysis": analysis, "img": backend_url + "/outputs/" + output_path}


@tool
def cross_modal_fusion_tool(
    optical_image_path: str, sar_image_path: str, query: str
) -> dict:
    """Extracts and combines complementary structural and spectral information from a co-registered optical and SAR image pair for joint region identification."""
    if analyze_land_cover is None:
        return {
            "analysis": "Fusion module is not available. Please verify the fusion_module setup.",
            "img": None,
        }

    # 1. Resolve patch ID from query, optical path, or SAR path
    patch_id = ""
    # Try finding an S2 patch ID in the query first
    s2_query_match = re.search(r"(S2[AB]_MSIL2A_[0-9A-Za-z_]+)", query)
    if s2_query_match:
        patch_id = resolve_patch_id(s2_query_match.group(1))
    elif optical_image_path:
        patch_id = resolve_patch_id(optical_image_path)
    elif sar_image_path:
        patch_id = resolve_patch_id(sar_image_path)

    # If not found yet, check for S1 patch ID in query
    if not patch_id:
        s1_query_match = re.search(r"(S1[AB]_IW_GRDH_[0-9A-Za-z_]+)", query)
        if s1_query_match:
            patch_id = resolve_patch_id(s1_query_match.group(1))

    # If still not found, check if coordinates exist in query (lat, lon)
    if not patch_id and find_patches_near is not None:
        coord_match = re.search(r"(-?\d+\.\d+)[,\s]+(-?\d+\.\d+)", query)
        if coord_match:
            try:
                lat, lon = float(coord_match.group(1)), float(coord_match.group(2))
                res_near = find_patches_near(lat, lon, k=1)
                if "matches" in res_near and len(res_near["matches"]) > 0:
                    patch_id = res_near["matches"][0]["patch_id"]
            except Exception:
                pass

    if not patch_id:
        return {
            "analysis": (
                "Could not identify a valid Sentinel-2 patch ID from the input images or query.\n"
                "Please provide a patch ID (e.g., S2A_MSIL2A_20170803T094031_30_11) "
                "or geographic coordinates (e.g., 45.909, 18.893) in your query, "
                "or upload a patch image from the dataset."
            ),
            "img": None,
        }

    # 2. Run fusion model inference
    try:
        prediction = analyze_land_cover(patch_id, top_k=5)
    except Exception as e:
        return {"analysis": f"Error running fusion inference: {e}", "img": None}

    if "error" in prediction:
        return {
            "analysis": f"Fusion tool error ({prediction.get('error')}): {prediction.get('detail', '')}. {prediction.get('hint', '')}",
            "img": None,
        }

    # 3. Format predictions into human-readable analysis
    s1_id = prediction.get("s1_patch_id", "Unknown")
    lat = prediction.get("lat", "N/A")
    lon = prediction.get("lon", "N/A")
    labels = prediction.get("labels", [])
    probs = prediction.get("probabilities", {})
    modalities = ", ".join(prediction.get("modalities_used", ["s2", "s1"])).upper()

    prob_lines = []
    for rank, (cls_name, score) in enumerate(probs.items(), 1):
        percentage = f"{score * 100:.2f}%"
        is_top = " (Detected)" if cls_name in labels else ""
        prob_lines.append(f"{rank}. {cls_name}: {percentage}{is_top}")
        if rank >= 5:
            break

    classes_text = "\n".join(prob_lines)

    analysis_text = (
        f"Cross-Modal Fusion Land-Cover Analysis\n\n"
        f"- Sentinel-2 Optical Patch: {patch_id}\n"
        f"- Sentinel-1 SAR Patch: {s1_id}\n"
        f"- Coordinates: Lat {lat}, Lon {lon}\n"
        f"- Modalities Fused: {modalities} (Optical 12-band + Radar Dual-Pol VV/VH)\n\n"
        f"Top Predicted Land Cover Classes:\n"
        f"{classes_text}\n"
    )

    # 4. Preview image
    img_url = None
    tmp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    # Check for rgb preview in S2_DIR
    rgb_src = None
    if S2_DIR:
        candidate_rgb = S2_DIR / patch_id / f"{patch_id}_rgb.png"
        if candidate_rgb.exists():
            rgb_src = candidate_rgb

    if rgb_src and rgb_src.exists():
        req_id = uuid.uuid4().hex[:8]
        out_filename = f"{req_id}_{patch_id}_rgb.png"
        out_dest = os.path.join(tmp_dir, out_filename)
        shutil.copyfile(str(rgb_src), out_dest)
        img_url = f"{backend_url}/outputs/{out_filename}"
    elif (
        optical_image_path
        and os.path.exists(optical_image_path)
        and optical_image_path.lower().endswith((".png", ".jpg", ".jpeg"))
    ):
        req_id = uuid.uuid4().hex[:8]
        ext = os.path.splitext(optical_image_path)[1]
        out_filename = f"{req_id}_{patch_id}{ext}"
        out_dest = os.path.join(tmp_dir, out_filename)
        shutil.copyfile(optical_image_path, out_dest)
        img_url = f"{backend_url}/outputs/{out_filename}"

    return {"analysis": analysis_text, "img": img_url}


@tool
def compatibility_check_tool(
    image_paths: list[str], expected_modalities: list[str]
) -> dict:
    """Validates the format, metadata, spatial correspondence, and modality compatibility of input GeoTIFF or TIFF files before execution."""
    pass


TOOL_MAP = {
    "vqa": vqa_tool,
    "grounding": grounding_tool,
    "captioning": captioning_tool,
    "change_analysis": change_analysis_tool,
    "cross_modal_fusion": cross_modal_fusion_tool,
    "compatibility_check": compatibility_check_tool,
}

# Initialize Groq LLM
llm = ChatGroq(model_name="qwen/qwen3.8-27b", temperature=0.7, max_tokens=20)


class RouteDecision(TypedDict):
    category: Literal["billing", "technical", "general"]


router_llm = llm.with_structured_output(RouteDecision)


def route_and_execute(inputs: dict) -> str:
    user_query = inputs["query"]
    image_paths = inputs.get("image_paths", [])
    history = inputs.get("history", [])

    if history:
        history_str = "\n".join(
            [f"User: {h['user']}\nAgent: {h['agent']}" for h in history[-3:]]
        )
        rewrite_prompt = (
            f"Conversation History:\n{history_str}\n\n"
            f"User's follow-up question: {user_query}\n\n"
            "Rewrite the follow-up question to be a standalone query that can be understood without the history. Do not answer the question, just rewrite it. Standalone query:"
        )
        user_query = llm.invoke(rewrite_prompt).content.strip()
        print(f"-> Rewritten Query: {user_query}")

    prompt = (
        "Classify this user query and its inputs into exactly one category: vqa, grounding, captioning, change_analysis, cross_modal_fusion, or compatibility_check.\n"
        "Respond with ONLY the category name. Do not write anything else.\n"
        f"Query: {user_query}\n"
        "Category:"
    )

    response = llm.invoke(prompt)
    chosen_category = response.content.strip().lower()

    if chosen_category not in TOOL_MAP:
        print(
            f"-> Unrecognized category '{chosen_category}', falling back to vqa tool."
        )
        chosen_category = "vqa"

    print(f"-> Detected Query Type: {chosen_category.upper()}")

    selected_tool = TOOL_MAP[chosen_category]

    if len(image_paths) == 0:
        if chosen_category == "cross_modal_fusion":
            return selected_tool.invoke(
                {
                    "optical_image_path": "",
                    "sar_image_path": "",
                    "query": user_query,
                }
            )
        return {"analysis": "Please upload at least one image.", "img": None}

    if chosen_category == "change_analysis":
        return selected_tool.invoke(
            {
                "image_path_t1": image_paths[0],
                "image_path_t2": (
                    image_paths[1] if len(image_paths) > 1 else image_paths[0]
                ),
                "query": user_query,
            }
        )
    elif chosen_category == "cross_modal_fusion":
        return selected_tool.invoke(
            {
                "optical_image_path": image_paths[0],
                "sar_image_path": (
                    image_paths[1] if len(image_paths) > 1 else image_paths[0]
                ),
                "query": user_query,
            }
        )
    elif chosen_category == "compatibility_check":
        return selected_tool.invoke(
            {"image_paths": image_paths, "expected_modalities": ["optical", "sar"]}
        )
    else:
        # For VQA, Grounding, Captioning, use the last uploaded image if there is one
        return selected_tool.invoke(
            {"image_path": image_paths[-1], "query": user_query}
        )


router_agent = RunnableLambda(route_and_execute)


def run_agent(query, image_paths=None, history=None):
    if image_paths is None:
        image_paths = []
    if history is None:
        history = []
    output = router_agent.invoke(
        {
            "image_paths": image_paths,
            "query": query,
            "history": history,
        }
    )
    return output
