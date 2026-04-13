"""Xem trước ảnh đã stage trong Streamlit."""
from __future__ import annotations

import io
from typing import Dict, List, Tuple

import streamlit as st
from PIL import Image, ImageOps


def render_image_thumbnails(
    staged: List[Tuple[str, bytes]],
    selected: Dict[str, bool],
    *,
    cols_per_row: int = 6,
    width_px: int = 112,
) -> None:
    st.markdown("### Xem trước")
    st.caption(f"**{len(staged)}** ảnh — kiểm tra nhanh trước khi xử lý.")
    for row_start in range(0, len(staged), cols_per_row):
        cols = st.columns(cols_per_row)
        for ci, col in enumerate(cols):
            idx = row_start + ci
            if idx >= len(staged):
                break
            fname, raw_b = staged[idx]
            key = f"sel::{idx}::{fname}"
            with col:
                selected[key] = st.checkbox(
                    "Chọn",
                    value=selected.get(key, True),
                    key=f"thumb_cb_{idx}_{hash(fname)}",
                    label_visibility="collapsed",
                )
                try:
                    p = Image.open(io.BytesIO(raw_b))
                    try:
                        p = ImageOps.exif_transpose(p)
                    except Exception:
                        pass
                    p = p.convert("RGB")
                    cap = fname if len(fname) <= 32 else fname[:29] + "…"
                    st.image(p, caption=cap, width=width_px)
                except Exception:
                    short = fname[:22] + "…" if len(fname) > 22 else fname
                    st.markdown(
                        '<div style="display:inline-block;padding:2px 8px;border-radius:999px;background:#fee2e2;color:#991b1b;font-weight:900;font-size:12px;">Lỗi</div>',
                        unsafe_allow_html=True,
                    )
                    st.caption(f"`{short}`")
                    st.caption("Không xem trước được.")
