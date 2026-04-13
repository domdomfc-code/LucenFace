"""Khởi tạo PortraitProcessor (cache) và gọi pipeline xử lý ảnh."""
from __future__ import annotations

import inspect
from typing import Any, Dict, Tuple

import streamlit as st
from PIL import Image

import frontend.backend_lazy as backend_lazy
from frontend.backend_lazy import ensure_image_backend
from frontend.config import APP_BUILD


@st.cache_resource(show_spinner=False)
def get_cached_portrait_processor(
    ratio: str,
    blue_rgb: Tuple[int, int, int],
    min_face_conf: float,
    *,
    rembg_engine: str,
    rembg_model: str,
    remove_bg_api_key: str | None,
    cache_version: str = APP_BUILD,
) -> Any:
    """
    `cache_version` đổi khi deploy (APP_BUILD) để tránh giữ PortraitProcessor cũ
    không khớp chữ ký `process(..., replace_background=...)` → TypeError trên Cloud.
    """
    ensure_image_backend()
    _ = cache_version
    PC = backend_lazy.PortraitProcessor
    assert PC is not None
    return PC(
        ratio=ratio,
        blue_rgb=blue_rgb,
        min_face_conf=min_face_conf,
        rembg_engine=rembg_engine,
        rembg_model=rembg_model,
        remove_bg_api_key=remove_bg_api_key,
    )


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
) -> Any:
    """
    Gọi `PortraitProcessor.process` — chỉ truyền các kwarg có trong chữ ký
    (tránh TypeError khi worker Cloud chạy bản `backend` cũ hơn `frontend`).
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
    return processor.process(pil, prefer_face_crop=prefer_face_crop, **kw)
