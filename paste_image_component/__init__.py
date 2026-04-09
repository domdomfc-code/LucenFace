"""Vùng dán ảnh clipboard — custom component Streamlit (frontend/index.html)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit.components.v1 as components

_frontend = Path(__file__).resolve().parent / "frontend"
_paste_image = components.declare_component("p2c_paste_image_v1", path=str(_frontend))


def paste_image_from_clipboard(*, enable_global_paste: bool = True, key: str = "p2c_paste_clipboard") -> Any:
    """
    Trả về chuỗi data URL (base64) khi người dùng dán ảnh, hoặc None.
    Lưu ý: tùy trình duyệt, có thể cần HTTPS/permission; component có nút fallback `clipboard.read()`.
    """
    return _paste_image(enable_global_paste=bool(enable_global_paste), default=None, key=key, tab_index=0)
