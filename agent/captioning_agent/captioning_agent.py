"""
Captioning specialist tool - same shape as vqa_agent: owns prompt
construction and the HTTP call to the hosted GeoChat model, hands the
orchestrator back a plain-text answer.
"""

import os
import requests
from dotenv import load_dotenv

from .prompts import build_captioning_prompt

_ENV_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"
)
load_dotenv(_ENV_PATH, override=False)

CAPTIONING_ENDPOINT_PATH = os.getenv("captioning_endpoint_path", "/captioning")


def run_captioning(image_path: str, query: str = "") -> str:
    """
    Runs captioning on a single image against the hosted GeoChat model.

    Returns the model's raw text answer. Raises RuntimeError on failure.
    """
    geochat_url = os.getenv("geochat_url")
    if not geochat_url:
        raise RuntimeError("geochat_url is not set - check agent/.env")

    text_prompt = build_captioning_prompt(query)

    with open(image_path, "rb") as f:
        files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
        data = {"text_prompt": text_prompt}
        response = requests.post(
            f"{geochat_url}{CAPTIONING_ENDPOINT_PATH}", files=files, data=data
        )

    if response.status_code != 200:
        raise RuntimeError(f"Error from GeoChat: {response.status_code} - {response.text}")

    res_json = response.json()
    return res_json.get("text", "No text returned")