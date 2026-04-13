"""Tiện ích Streamlit dùng chung: secrets, checklist, ZIP, gợi ý cài OpenCV."""
from __future__ import annotations

import io
import os
import platform
import zipfile
from typing import Any, Dict, List, Tuple

import streamlit as st


def read_remove_bg_api_key() -> str | None:
    """API key remove.bg: Streamlit Secrets hoặc biến môi trường REMOVEBG_API_KEY."""
    try:
        k = st.secrets.get("REMOVEBG_API_KEY")
        if k is not None and str(k).strip():
            return str(k).strip()
    except Exception:
        pass
    v = os.environ.get("REMOVEBG_API_KEY", "").strip()
    return v or None


def cv2_troubleshoot_markdown() -> str:
    if platform.system() == "Windows":
        return (
            "**Windows (máy bạn):**\n\n"
            "1. PowerShell trong thư mục dự án: `python -m venv .venv`\n"
            "2. Bật venv: `.\\.venv\\Scripts\\Activate.ps1`  \n"
            "   (nếu bị chặn: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`)\n"
            "3. Cài: `pip install -U pip` rồi `pip install -r requirements.txt`\n"
            "4. Chạy: `streamlit run app.py`\n\n"
            "Chỉ thiếu OpenCV: `pip install opencv-python-headless` trong venv.\n\n"
            "Lỗi kiểu `libgthread` / `.so` là của **Linux** (Cloud), không áp dụng Windows — không cần `apt` hay `packages.txt` trên máy bạn."
        )
    return (
        "**macOS / Linux (local):** `python3 -m venv .venv` → `source .venv/bin/activate` "
        "→ `pip install -r requirements.txt` → `streamlit run app.py`.\n\n"
        "**Streamlit Cloud (Linux):** thêm `libgl1` và `libglib2.0-0t64` vào `packages.txt` "
        "nếu thiếu `libGL.so.1` hoặc `libgthread-2.0.so.0`. Không dùng `libglib2.0-0` (tên cũ)."
    )


def render_checklist(checks: Dict[str, Dict[str, str]]) -> None:
    """Bảng dữ liệu — không qua Markdown/HTML (tránh lộ thẻ hoặc ký tự bị hiểu nhầm)."""
    if not checks:
        return
    rows = [
        {
            "Đạt": "Có" if payload["ok"] else "Không",
            "Tiêu chí": str(name),
            "Chi tiết": str(payload["message"]).strip(),
        }
        for name, payload in checks.items()
    ]
    st.dataframe(rows, width="stretch", hide_index=True)


def result_to_checks_dict(res: Any) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for k, v in res.checks.items():
        out[k] = {"ok": bool(v.ok), "message": str(v.message)}
    return out


def make_zip(files: List[Tuple[str, bytes]]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, data in files:
            z.writestr(name, data)
    return buf.getvalue()
