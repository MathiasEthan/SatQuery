"""
VQA specialist tool - talks to the hosted GeoChat model (run by you + your
friend) and hands the orchestrator back a plain-text answer.

This module owns everything except the actual model forward pass: prompt
construction, the HTTP call to the hosted model, and turning the raw
response into the string the orchestrator expects. It does NOT use the
friend's predict()/serving pipeline - we send our own finished prompt and
expect a bare generation back.
"""

import os
import requests
from dotenv import load_dotenv

from .prompts import build_vqa_prompt

# Load agent/.env if it hasn't been loaded yet - safe no-op if agent.py
# (which imports this module) already loaded it first.
_ENV_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"
)
load_dotenv(_ENV_PATH, override=False)

# Path only - base URL (geochat_url) still comes from .env, same as the
# rest of the repo. Override via VQA_ENDPOINT_PATH in .env if the friend
# ends up naming the route something else - no code change needed then.
VQA_ENDPOINT_PATH = os.getenv("vqa_endpoint_path", "/vqa")


def run_vqa(image_path: str, query: str) -> str:
    """
    Runs VQA on a single image against the hosted GeoChat model.

    Returns the model's raw text answer. Raises RuntimeError on failure so
    the caller (vqa_tool) decides how to surface it to the orchestrator.
    """
    geochat_url = os.getenv("geochat_url")
    if not geochat_url:
        raise RuntimeError("geochat_url is not set - check agent/.env")

    text_prompt = build_vqa_prompt(query)

    with open(image_path, "rb") as f:
        files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
        data = {"text_prompt": text_prompt}
        response = requests.post(
            f"{geochat_url}{VQA_ENDPOINT_PATH}", files=files, data=data
        )

    if response.status_code != 200:
        raise RuntimeError(f"Error from GeoChat: {response.status_code} - {response.text}")

    res_json = response.json()
    return res_json.get("text", "No text returned")