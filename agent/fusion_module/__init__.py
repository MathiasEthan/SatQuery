"""Optical + SAR fusion tool. See README.md for setup."""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from src.tool import TOOL_SPECS, check_setup, dispatch, warmup  # noqa: E402

__all__ = ["TOOL_SPECS", "dispatch", "warmup", "check_setup"]
