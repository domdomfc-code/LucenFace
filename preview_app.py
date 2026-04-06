"""
Bản PREVIEW UI — chỉ để xem giao diện trên Streamlit Cloud (build nhanh).
Không import OpenCV / MediaPipe / rembg.

Chạy local:
  streamlit run preview_app.py

Deploy Cloud:
  Main file path: preview_app.py
  Requirements: requirements-preview.txt
"""
from __future__ import annotations

import io
from typing import Dict, List, Tuple

import streamlit as st
from PIL import Image, ImageDraw

APP_TITLE = "Chuẩn hóa ảnh chân dung học sinh"
APP_BUILD = "preview-ui-only"
BLUE = "#005BC4"
BG = "#F6F9FF"


def _inject_css() -> None:
    st.markdown(
        f"""
        <style>
          :root {{
            --blue: {BLUE};
            --bg: {BG};
            --card: #ffffff;
            --text: #0f172a;
            --muted: #64748b;
            --border: rgba(2, 6, 23, 0.10);
            --shadow: 0 10px 30px rgba(2, 6, 23, 0.06);
          }}

          .stApp {{
            background:
              radial-gradient(1000px 500px at 20% 0%, rgba(0, 91, 196, 0.14), rgba(0,0,0,0) 60%),
              radial-gradient(900px 500px at 90% 10%, rgba(56, 189, 248, 0.14), rgba(0,0,0,0) 60%),
              var(--bg);
          }}

          section.main > div {{
            padding-top: 1.25rem;
          }}

          header, footer {{
            visibility: hidden;
            height: 0px;
          }}

          .topbar {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            background: rgba(255,255,255,0.75);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 12px 14px;
            box-shadow: var(--shadow);
            backdrop-filter: blur(10px);
          }}
          .brand {{
            display: flex;
            align-items: center;
            gap: 10px;
            font-weight: 900;
            color: var(--text);
            letter-spacing: -0.02em;
          }}
          .brand-badge {{
            width: 34px;
            height: 34px;
            border-radius: 10px;
            background: linear-gradient(135deg, var(--blue), #38bdf8);
            box-shadow: 0 10px 22px rgba(0,91,196,0.20);
          }}
          .brand-title {{
            font-size: 1.05rem;
            line-height: 1.1;
          }}
          .brand-sub {{
            font-size: 0.8rem;
            color: var(--muted);
            font-weight: 700;
            margin-top: 1px;
          }}
          .top-actions {{
            display: flex;
            gap: 8px;
            align-items: center;
            color: var(--muted);
            font-weight: 700;
          }}
          .pill {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 7px 10px;
            border: 1px solid var(--border);
            border-radius: 999px;
            background: rgba(255,255,255,0.9);
          }}

          .app-title {{
            font-size: 1.8rem;
            font-weight: 900;
            color: var(--text);
            margin: 0.35rem 0 0.2rem 0;
            letter-spacing: -0.03em;
          }}
          .app-subtitle {{
            color: var(--muted);
            margin-bottom: 1.0rem;
            font-weight: 600;
          }}
          .card {{
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 14px 14px 10px 14px;
            box-shadow: var(--shadow);
          }}
          .card-soft {{
            background: rgba(255,255,255,0.85);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 14px;
            box-shadow: var(--shadow);
            backdrop-filter: blur(10px);
          }}
          .badge-ok {{
            display: inline-block;
            padding: 2px 10px;
            border-radius: 999px;
            background: rgba(22,163,74,0.12);
            color: #166534;
            font-weight: 700;
            font-size: 0.85rem;
          }}
          .checklist {{
            margin-top: 6px;
            padding-left: 0px;
          }}
          .check-item {{
            display: flex;
            gap: 8px;
            margin: 4px 0px;
            align-items: baseline;
          }}
          .check-name {{
            font-weight: 700;
            color: #111827;
            min-width: 160px;
          }}
          .check-msg {{
            color: #374151;
            opacity: 0.95;
          }}
          .muted {{
            color: var(--muted);
          }}

          [data-testid="stFileUploader"] {{
            background: transparent;
            border: none;
          }}
          [data-testid="stFileUploader"] > div {{
            border: 2px dashed rgba(0, 91, 196, 0.35);
            background: rgba(255,255,255,0.80);
            border-radius: 18px;
            padding: 18px 16px;
            box-shadow: var(--shadow);
          }}
          [data-testid="stFileUploader"] label {{
            font-weight: 900;
            color: var(--text);
          }}
          [data-testid="stFileUploader"] small {{
            color: var(--muted);
            font-weight: 600;
          }}

          .stButton > button {{
            border-radius: 12px;
            font-weight: 800;
            border: 1px solid var(--border);
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _checklist_html(checks: Dict[str, Dict[str, str]]) -> str:
    parts = ['<div class="checklist">']
    for name, payload in checks.items():
        ok = payload["ok"]
        msg = payload["message"]
        icon = "✅" if ok else "❌"
        parts.append(
            f"""
            <div class="check-item">
              <div>{icon}</div>
              <div class="check-name">{name}</div>
              <div class="check-msg">{msg}</div>
            </div>
            """
        )
    parts.append("</div>")
    return "\n".join(parts)


def _fake_processed_preview(pil: Image.Image, blue_hex: str) -> Image.Image:
    """Ảnh demo: nền xanh + khung trắng (không xử lý CV thật)."""
    w, h = pil.size
    out = Image.new("RGB", (w, h), blue_hex)
    draw = ImageDraw.Draw(out)
    pad = int(min(w, h) * 0.06)
    draw.rectangle([pad, pad, w - pad, h - pad], outline=(255, 255, 255), width=max(2, int(min(w, h) * 0.004)))
    thumb = pil.convert("RGB").resize((max(1, w - 2 * pad), max(1, h - 2 * pad)))
    out.paste(thumb, (pad, pad))
    return out


def main() -> None:
    st.set_page_config(page_title=f"{APP_TITLE} (Preview)", page_icon="👀", layout="wide")
    _inject_css()

    st.markdown(
        f"""
        <div class="topbar">
          <div class="brand">
            <div class="brand-badge"></div>
            <div>
              <div class="brand-title">LucenFace</div>
              <div class="brand-sub">UI Preview • Build {APP_BUILD}</div>
            </div>
          </div>
          <div class="top-actions">
            <span class="pill">Preview</span>
            <span class="pill">Không xử lý ảnh thật</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(f'<div class="app-title">{APP_TITLE}</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">Đây là bản <b>preview giao diện</b> (build nhanh). Upload ảnh chỉ để xem layout — không chạy OpenCV/MediaPipe/rembg.</div>',
        unsafe_allow_html=True,
    )
    st.warning("Đây là **PREVIEW UI**. App đầy đủ: `app.py` + `requirements.txt`.")

    with st.sidebar:
        st.markdown("### Cài đặt (demo)")
        ratio = st.selectbox("Tỷ lệ ảnh đầu ra", ["3x4", "4x6"], index=0)
        st.caption("Tối đa 50 ảnh/lần (demo).")
        st.markdown("---")
        st.markdown("### Nền xanh (demo)")
        blue_hex = st.color_picker("Chọn màu nền", value=BLUE)

    rgb = tuple(int(blue_hex.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))

    st.markdown("### Kéo & thả ảnh")
    st.markdown(
        '<div class="card-soft muted">Bản preview: không kiểm tra tiêu chuẩn thật — chỉ hiển thị checklist mẫu.</div>',
        unsafe_allow_html=True,
    )
    uploads = st.file_uploader(
        "Kéo và thả tệp vào đây hoặc bấm để chọn",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    if not uploads:
        st.info("Upload ít nhất 1 ảnh để xem layout 3 cột.")
        return

    if len(uploads) > 50:
        st.error("Tối đa 50 ảnh.")
        return

    if st.button("Xem preview layout", type="primary"):
        for up in uploads:
            raw = up.read()
            try:
                pil = Image.open(io.BytesIO(raw)).convert("RGB")
            except Exception:
                st.error(f"Không đọc được: `{up.name}`")
                continue

            fake_checks: Dict[str, Dict[str, str]] = {
                "Khuôn mặt": {"ok": True, "message": "Demo checklist (không phát hiện thật)."},
                "Vị trí (giữa khung)": {"ok": True, "message": "Demo."},
                "Tỷ lệ khuôn mặt": {"ok": False, "message": "Demo cảnh báo."},
                "Ánh sáng & Tương phản": {"ok": True, "message": "Demo."},
                "Nền đơn sắc": {"ok": True, "message": "Demo."},
                "Thay nền xanh": {"ok": False, "message": "Preview không có rembg."},
            }

            out = _fake_processed_preview(pil, rgb)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            c1, c2, c3 = st.columns([1.05, 1.05, 1.25], gap="large")
            with c1:
                st.markdown("**Original**")
                st.image(pil, caption=up.name, use_container_width=True)
            with c2:
                st.markdown("**Trạng thái / Checklist (demo)**")
                st.markdown('<span class="badge-ok">PREVIEW</span>', unsafe_allow_html=True)
                st.markdown("**Checklist**")
                st.markdown(_checklist_html(fake_checks), unsafe_allow_html=True)
            with c3:
                st.markdown("**Processed (demo)**")
                st.image(out, use_container_width=True)
                st.caption(f"Tỷ lệ chọn: **{ratio}** (chỉ hiển thị, không crop thật).")
            st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
