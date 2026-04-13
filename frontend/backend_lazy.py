"""
Lazy-import `backend.image_utils` — tránh tải OpenCV/MediaPipe/rembg khi chỉ cần UI (healthz, v.v.).
"""
from __future__ import annotations

from typing import Any, Optional, Type

PortraitProcessor: Optional[Type[Any]] = None  # type: ignore[assignment,misc]
ProcessResult: Optional[Type[Any]] = None  # type: ignore[assignment,misc]
pil_to_jpeg_bytes: Any = None  # type: ignore[assignment,misc]
_image_backend_loaded = False


def ensure_image_backend() -> None:
    """Gọi trước khi dùng PortraitProcessor / pil_to_jpeg_bytes."""
    global PortraitProcessor, ProcessResult, pil_to_jpeg_bytes, _image_backend_loaded
    if _image_backend_loaded:
        return
    from backend.image_utils import PortraitProcessor as _PC
    from backend.image_utils import ProcessResult as _PR
    from backend.image_utils import pil_to_jpeg_bytes as _pj

    PortraitProcessor = _PC
    ProcessResult = _PR
    pil_to_jpeg_bytes = _pj
    _image_backend_loaded = True
