"""Gọi PortraitProcessor.process — không phụ thuộc Streamlit."""
from __future__ import annotations

import inspect
from typing import Any, Dict

from PIL import Image


def run_portrait_process(
    processor: Any,
    pil: Image.Image,
    *,
    prefer_face_crop: bool,
    replace_blue_bg: bool,
    skip_rembg_if_uniform_background: bool = True,
    auto_orient: bool = True,
    crop_center_mode: str = "nose",
    letterbox_smart_framing: bool = True,
    check_only: bool = False,
) -> Any:
    """
    Gọi `PortraitProcessor.process` — chỉ truyền các kwarg có trong chữ ký
    (tránh TypeError khi worker chạy bản `backend` cũ hơn).
    """
    params = inspect.signature(processor.process).parameters
    kw: Dict[str, object] = {}
    if "replace_background" in params:
        kw["replace_background"] = replace_blue_bg
    if "skip_rembg_if_uniform_background" in params:
        kw["skip_rembg_if_uniform_background"] = skip_rembg_if_uniform_background
    if "auto_orient" in params:
        kw["auto_orient"] = bool(auto_orient)
    if "crop_center_mode" in params:
        kw["crop_center_mode"] = crop_center_mode
    if "letterbox_smart_framing" in params:
        kw["letterbox_smart_framing"] = bool(letterbox_smart_framing)
    if "check_only" in params:
        kw["check_only"] = bool(check_only)
    return processor.process(pil, prefer_face_crop=prefer_face_crop, **kw)
