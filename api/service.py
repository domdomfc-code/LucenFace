"""Logic audit/process dùng chung cho route FastAPI."""
from __future__ import annotations

import base64
import io
import os
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageOps

import frontend.bootstrap  # noqa: F401 — sys.path
from frontend.backend_lazy import ensure_image_backend
from frontend.config import APP_BUILD
from frontend.image_io import (
    looks_like_heic,
    normalize_filename_hint,
    sniff_image_kind,
)
from frontend.processing_core import run_portrait_process

from api.schemas import ProcessConfig

MAX_FILES = 50
MAX_BYTES = 12 * 1024 * 1024

_processor_cache: Dict[tuple, Any] = {}


def read_remove_bg_api_key() -> Optional[str]:
    v = os.environ.get("REMOVEBG_API_KEY", "").strip()
    return v or None


def get_portrait_processor(config: ProcessConfig) -> Any:
    ensure_image_backend()
    from frontend.backend_lazy import PortraitProcessor

    assert PortraitProcessor is not None
    key = config.cache_key()
    if key not in _processor_cache:
        rem_eng = config.effective_rembg_engine()
        api_key = read_remove_bg_api_key() if rem_eng == "remove_bg_api" else None
        _processor_cache[key] = PortraitProcessor(
            ratio=config.ratio,
            blue_rgb=config.blue_rgb(),
            min_face_conf=config.min_face_conf,
            rembg_engine=rem_eng,
            rembg_model=config.rembg_model,
            remove_bg_api_key=api_key,
        )
    return _processor_cache[key]


def _pil_from_raw(raw: bytes) -> Optional[Image.Image]:
    try:
        pil = Image.open(io.BytesIO(raw))
        try:
            pil = ImageOps.exif_transpose(pil)
        except Exception:
            pass
        return pil.convert("RGB")
    except Exception:
        return None


def result_to_checks_dict(res: Any) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for k, v in res.checks.items():
        out[k] = {"ok": bool(v.ok), "message": str(v.message)}
    return out


def validate_and_stage(
    filenames: List[str], raw_list: List[bytes]
) -> Tuple[List[Tuple[str, bytes]], List[Dict[str, str]]]:
    """Trả work_items và rejected (giống Streamlit app)."""
    rejected: List[Dict[str, str]] = []
    work_items: List[Tuple[str, bytes]] = []
    for fname, raw in zip(filenames, raw_list):
        fname_norm = normalize_filename_hint(fname)
        if looks_like_heic(fname_norm, raw):
            rejected.append(
                {
                    "file": fname_norm,
                    "reason": "Định dạng HEIC/HEIF chưa hỗ trợ — export JPG.",
                }
            )
            continue
        kind = sniff_image_kind(raw)
        if kind is None:
            rejected.append(
                {"file": fname_norm, "reason": "File không phải ảnh JPG/PNG hợp lệ."}
            )
            continue
        if len(raw) > MAX_BYTES:
            rejected.append(
                {
                    "file": fname_norm,
                    "reason": f"File quá nặng ({len(raw)/1024/1024:.1f} MB).",
                }
            )
            continue
        try:
            im = Image.open(io.BytesIO(raw))
            mode = (im.mode or "").upper()
            if mode in ("CMYK", "I;16", "I;16B", "I;16L", "I;16S", "I"):
                rejected.append(
                    {
                        "file": fname_norm,
                        "reason": f"Chế độ {mode} — xuất lại RGB JPG.",
                    }
                )
                continue
            im.verify()
        except Exception:
            rejected.append({"file": fname_norm, "reason": "Không đọc được ảnh."})
            continue
        work_items.append((fname_norm, raw))
    return work_items, rejected


def run_audit(
    work_items: List[Tuple[str, bytes]], config: ProcessConfig
) -> List[Dict[str, Any]]:
    processor = get_portrait_processor(config)
    rows_out: List[Dict[str, Any]] = []
    for work_idx, (filename, raw) in enumerate(work_items):
        pil = _pil_from_raw(raw)
        if pil is None:
            rows_out.append(
                {
                    "work_idx": work_idx,
                    "filename": filename,
                    "status": "FAILED",
                    "checks": {"Đọc ảnh": {"ok": False, "message": "Không mở được file ảnh."}},
                    "errors": ["Không đọc được ảnh."],
                    "warnings": [],
                    "n_ok": 0,
                    "n_tot": 1,
                }
            )
            continue
        try:
            res = run_portrait_process(
                processor,
                pil,
                prefer_face_crop=config.prefer_face_crop,
                replace_blue_bg=config.replace_blue_bg,
                skip_rembg_if_uniform_background=not config.force_blue_despite_uniform,
                auto_orient=config.auto_orient,
                crop_center_mode=config.crop_center_mode,
                letterbox_smart_framing=config.letterbox_smart_framing,
                check_only=True,
            )
        except RuntimeError as e:
            rows_out.append(
                {
                    "work_idx": work_idx,
                    "filename": filename,
                    "status": "FAILED",
                    "checks": {"Lỗi": {"ok": False, "message": str(e)}},
                    "errors": [str(e)],
                    "warnings": [],
                    "n_ok": 0,
                    "n_tot": 1,
                }
            )
            continue
        cd = result_to_checks_dict(res)
        n_ok = sum(1 for v in cd.values() if v.get("ok"))
        rows_out.append(
            {
                "work_idx": work_idx,
                "filename": filename,
                "status": res.status,
                "checks": cd,
                "errors": list(res.errors),
                "warnings": list(res.warnings),
                "n_ok": n_ok,
                "n_tot": len(cd),
            }
        )
    return rows_out


def run_process_indices(
    work_items: List[Tuple[str, bytes]],
    config: ProcessConfig,
    indices: List[int],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    if config.replace_blue_bg and config.rembg_engine == "remove_bg_api":
        if not read_remove_bg_api_key():
            raise ValueError(
                "remove.bg cần biến môi trường REMOVEBG_API_KEY"
            )

    processor = get_portrait_processor(config)
    ensure_image_backend()
    from frontend.backend_lazy import pil_to_jpeg_bytes

    assert pil_to_jpeg_bytes is not None

    results: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []
    for j, work_idx in enumerate(indices):
        if work_idx < 0 or work_idx >= len(work_items):
            failures.append({"file": f"#{work_idx}", "reason": "Chỉ số không hợp lệ"})
            continue
        filename, raw = work_items[work_idx]
        pil = _pil_from_raw(raw)
        if pil is None:
            msg = "Không đọc được ảnh."
            failures.append({"file": filename, "reason": msg})
            results.append(
                {
                    "work_idx": work_idx,
                    "filename": filename,
                    "jpg_base64": None,
                    "download_name": None,
                    "error": msg,
                }
            )
            continue
        try:
            res = run_portrait_process(
                processor,
                pil,
                prefer_face_crop=config.prefer_face_crop,
                replace_blue_bg=config.replace_blue_bg,
                skip_rembg_if_uniform_background=not config.force_blue_despite_uniform,
                auto_orient=config.auto_orient,
                crop_center_mode=config.crop_center_mode,
                letterbox_smart_framing=config.letterbox_smart_framing,
                check_only=False,
            )
        except RuntimeError as e:
            failures.append({"file": filename, "reason": str(e)})
            results.append(
                {
                    "work_idx": work_idx,
                    "filename": filename,
                    "jpg_base64": None,
                    "download_name": None,
                    "error": str(e),
                }
            )
            continue
        if res.processed_image is None:
            msg = "Không tạo được ảnh đầu ra."
            failures.append({"file": filename, "reason": msg})
            results.append(
                {
                    "work_idx": work_idx,
                    "filename": filename,
                    "jpg_base64": None,
                    "download_name": None,
                    "error": msg,
                }
            )
            continue
        out_bytes = pil_to_jpeg_bytes(res.processed_image, quality=95)
        base = filename.rsplit(".", 1)[0] if "." in filename else filename
        dl_name = f"{base}_chuanhoa.jpg"
        b64 = base64.standard_b64encode(out_bytes).decode("ascii")
        results.append(
            {
                "work_idx": work_idx,
                "filename": filename,
                "jpg_base64": b64,
                "download_name": dl_name,
                "error": None,
            }
        )
    return results, failures


def build_meta() -> Dict[str, str]:
    return {"app_build": APP_BUILD}
