"""Khởi tạo PortraitProcessor (cache) và gọi pipeline xử lý ảnh."""
from __future__ import annotations

from typing import Any, Tuple

import streamlit as st

import frontend.backend_lazy as backend_lazy
from frontend.backend_lazy import ensure_image_backend
from frontend.config import APP_BUILD
from frontend.processing_core import run_portrait_process


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


# re-export cho code import từ processor_service
__all__ = ("get_cached_portrait_processor", "run_portrait_process")
