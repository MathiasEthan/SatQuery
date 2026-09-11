"""
Prompt construction for the VQA specialist tool.

Everything that turns a raw user query into the exact string sent to the
hosted GeoChat model lives here, kept separate from vqa_agent.py so the
"how we ask the model" logic is easy to find and tune on its own.
"""


def build_vqa_prompt(query: str) -> str:
    """
    Builds the final text_prompt sent to the hosted GeoChat VQA endpoint.

    Currently applies the "[vqa]" task prefix, matching the convention
    already used elsewhere in this repo (see grounding_tool's "[grounding]"
    prefix).

    TODO: drop in the actual prompt-engineering template here.
    TODO: drop in the second boosting technique (the one you forgot) here.
    """
    return f"[vqa] {query.strip()}"