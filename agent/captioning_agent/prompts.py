"""
Prompt construction for the captioning specialist tool.
"""


def build_captioning_prompt(query: str = "") -> str:
    """
    Builds the text_prompt sent to the hosted GeoChat model for captioning.

    NOTE: unlike "[vqa]" and "[grounding]", there's no confirmed task prefix
    for captioning in this codebase - GeoChat's captioning ability wasn't
    part of any training round here, so this is a plain instruction, not a
    verified convention. Confirm against GeoChat's own eval scripts (or
    however you end up training/testing this) before trusting it blindly.
    """
    if query and query.strip():
        return f"Describe this image, focusing on: {query.strip()}"
    return "Provide a detailed description of this satellite image, including land cover, notable objects, and layout."