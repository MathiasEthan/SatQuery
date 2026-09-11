from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableLambda
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langchain.tools import tool
from typing import Literal
import os
import requests
import requests
from PIL import Image, ImageDraw
import re
from change_agent.change_agent import run_model
from visualizer import process_and_visualize
import uuid

from vqa_agent import run_vqa


import re
import math


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
def captioning_tool(image_path: str, query: str) -> str:
    """Generates a comprehensive scene description and land-cover summary for a single remote-sensing image."""
    pass


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
) -> str:
    """Extracts and combines complementary structural and spectral information from a co-registered optical and SAR image pair for joint region identification."""
    pass


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
        history_str = "\n".join([f"User: {h['user']}\nAgent: {h['agent']}" for h in history[-3:]])
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
        return selected_tool.invoke({"image_path": image_paths[-1], "query": user_query})


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
