from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableLambda
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langchain.tools import tool
from typing import Literal
import os
import requests

load_dotenv()
geochat_url = os.getenv("geochat_url")


@tool
def vqa_tool(image_path: str, query: str) -> str:
    """Executes visual question answering on a single optical, multispectral, or SAR image to answer natural-language queries about land cover, objects, or features."""
    pass


@tool
def grounding_tool(image_path: str, query: str) -> dict:
    """Performs text-guided region grounding on a single image to locate and outline specific objects or areas mentioned in the query."""
    response = requests.post(f"{geochat_url}/agent", json={"query": query})
    return response.json()


@tool
def captioning_tool(image_path: str, query: str) -> str:
    """Generates a comprehensive scene description and land-cover summary for a single remote-sensing image."""
    pass


@tool
def change_analysis_tool(image_path_t1: str, image_path_t2: str, query: str) -> dict:
    """Analyzes a bi-temporal image pair to detect surface changes, describe alterations over time, and answer change-based visual questions with spatial change map outputs where applicable."""
    pass


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
llm = ChatGroq(model_name="qwen/qwen3.8-27b", temperature=0.7)


class RouteDecision(TypedDict):
    category: Literal["billing", "technical", "general"]


router_llm = llm.with_structured_output(RouteDecision)


def route_and_execute(inputs: dict) -> str:
    user_query = inputs["query"]
    image_paths = inputs.get("image_paths", ["/"])

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
        return selected_tool.invoke({"image_path": image_paths[0], "query": user_query})


router_agent = RunnableLambda(route_and_execute)

output = router_agent.invoke(
    {"image_paths": ["/", "/"], "query": "draw box around the church"}
)
print(output)
