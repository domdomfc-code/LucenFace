from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

# Đảm bảo import được package `backend` khi chạy: streamlit run frontend/app.py
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend.image_utils import PortraitProcessor, ProcessResult, pil_to_jpeg_bytes


APP_TITLE = "Chuẩn hóa ảnh chân dung học sinh"
# Đổi số khi deploy để kiểm tra Streamlit Cloud đã build bản mới (sidebar hiển thị).
APP_BUILD = "3.2-studio-highkey"
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

          /* Không ẩn cả header — Streamlit cần vùng này để mở lại sidebar khi thu gọn */
          header[data-testid="stHeader"] {{
            background: rgba(255, 255, 255, 0.55) !important;
            backdrop-filter: blur(10px);
            border-bottom: 1px solid var(--border);
          }}
          .stDeployButton {{
            display: none !important;
          }}
          footer {{
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
          .badge-fail {{
            display: inline-block;
            padding: 2px 10px;
            border-radius: 999px;
            background: rgba(220,38,38,0.12);
            color: #991b1b;
            font-weight: 700;
            font-size: 0.85rem;
          }}
          .small-note {{
            font-size: 0.9rem;
            opacity: 0.85;
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
          .stDownloadButton > button {{
            border-radius: 12px;
            font-weight: 800;
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _sidebar_reopen_button() -> None:
    """Nút cố định góc trái: mở lại sidebar khi đã thu nhỏ (backup nếu khó tìm nút mặc định)."""
    components.html(
        """
        <div style="position:fixed;top:52px;left:8px;z-index:999999;">
        <button type="button" title="Mở cài đặt (sidebar)"
          onclick="(() => {
            try {
              const d = window.parent.document;
              const q = (s) => d.querySelector(s);
              (q('[data-testid="collapsedControl"]')
                || q('[data-testid="stSidebarCollapsedControl"]')
                || q('button[data-testid="baseButton-header"]')
                || q('header button[kind="header"]'))?.click();
            } catch (e) {}
          })()"
          style="font-size:1.05rem;line-height:1;padding:0.45rem 0.55rem;border-radius:10px;
                 border:1px solid rgba(15,23,42,0.12);background:rgba(255,255,255,0.96);
                 cursor:pointer;box-shadow:0 4px 14px rgba(15,23,42,0.12);color:#0f172a;">
          ☰
        </button>
        </div>
        """,
        height=52,
    )


def _render_checklist(checks: Dict[str, Dict[str, str]]) -> None:
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
    st.dataframe(rows, use_container_width=True, hide_index=True)


def _result_to_checks_dict(res: ProcessResult) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for k, v in res.checks.items():
        out[k] = {"ok": bool(v.ok), "message": str(v.message)}
    return out


def _make_zip(files: List[Tuple[str, bytes]]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, data in files:
            z.writestr(name, data)
    return buf.getvalue()


@st.cache_resource(show_spinner=False)
def _get_processor(ratio: str, blue_rgb: Tuple[int, int, int], min_face_conf: float) -> PortraitProcessor:
    # Cache tài nguyên nặng: MediaPipe detector + rembg session
    return PortraitProcessor(ratio=ratio, blue_rgb=blue_rgb, min_face_conf=min_face_conf)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🪪", layout="wide")
    _inject_css()
    _sidebar_reopen_button()

    st.markdown(
        f"""
        <div class="topbar">
          <div class="brand">
            <div class="brand-badge"></div>
            <div>
              <div class="brand-title">LucenFace</div>
              <div class="brand-sub">Portrait Standardizer • Build {APP_BUILD}</div>
            </div>
          </div>
          <div class="top-actions">
            <span class="pill">Batch ≤ 50 ảnh</span>
            <span class="pill">Nền xanh</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(f'<div class="app-title">{APP_TITLE}</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">Kéo & thả ảnh vào vùng bên dưới để tự động chuẩn hóa (crop, cân bằng sáng, thay nền xanh, tải ZIP).</div>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.caption(f"**Build:** `{APP_BUILD}` — UI `frontend/`, xử lý `backend/`")
        with st.expander("Kiểm tra thư viện", expanded=False):
            import platform

            st.write(f"**Python:** `{platform.python_version()}`")
            try:
                import cv2  # type: ignore

                st.success(f"OpenCV OK (`cv2`): {getattr(cv2, '__version__', 'unknown')}")
            except Exception as e:
                st.error("Thiếu OpenCV (`cv2`).")
                st.code(str(e))
                st.caption("Gợi ý (Python 3.13): `pip install -r requirements-local-py313.txt` trong venv.")

        st.markdown("### Hướng dẫn nhanh")
        st.markdown(
            """
            - **Ảnh hợp lệ**: JPG/PNG, chân dung rõ mặt, **chỉ 1 khuôn mặt**.
            - **Tiêu chuẩn kiểm tra**:
              - Vị trí mặt gần trung tâm ngang
              - Tỷ lệ khuôn mặt hợp lý
              - Độ sáng & tương phản đủ
              - Nền tương đối đơn sắc
            - **Tự động xử lý**: crop theo chuẩn 3x4/4x6, cân bằng sáng, thay nền xanh.
            """
        )
        st.markdown("---")
        ratio = st.selectbox("Tỷ lệ ảnh đầu ra", ["3x4", "4x6"], index=0)
        max_files = 50
        st.caption(f"Tối đa {max_files} ảnh/lần.")
        st.markdown("---")
        st.markdown("### Nâng cao")
        min_face_conf = st.slider("Độ tin cậy phát hiện mặt", min_value=0.3, max_value=0.9, value=0.6, step=0.05)
        lazy_init = st.toggle("Khởi tạo engine khi bấm xử lý", value=True, help="Tránh đứng UI do MediaPipe/rembg load model.")
        st.markdown("---")
        st.markdown("### Thông số nền xanh")
        st.caption("Chuẩn mặc định: `#005BC4` (RGB 0, 91, 196).")
        blue_hex = st.color_picker("Chọn màu nền", value=BLUE)

    blue_rgb = tuple(int(blue_hex.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))

    st.markdown("### Kéo & thả ảnh")
    st.markdown(
        '<div class="card-soft muted">Mẹo: ảnh rõ mặt, thẳng góc; hệ thống sẽ tự phát hiện 1 khuôn mặt để crop đúng chuẩn.</div>',
        unsafe_allow_html=True,
    )
    uploads = st.file_uploader(
        "Kéo và thả tệp vào đây hoặc bấm để chọn",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    if not uploads:
        st.info("Hãy upload ít nhất 1 ảnh để bắt đầu.")
        return

    if len(uploads) > 50:
        st.error("Bạn đã chọn quá 50 ảnh. Vui lòng giảm số lượng và thử lại.")
        return

    st.markdown("### Xử lý hàng loạt")
    start = st.button("Bắt đầu xử lý", type="primary", use_container_width=False)
    if not start:
        st.caption("Bấm **Bắt đầu xử lý** để chạy pipeline cho toàn bộ ảnh.")
        return

    # Preflight: ensure OpenCV is importable before creating cached processor.
    # If cv2 is missing, `st.cache_resource` would cache the exception and keep failing.
    try:
        import cv2  # type: ignore  # noqa: F401
    except Exception as e:
        st.error("Thiếu OpenCV (`cv2`) hoặc OpenCV không import được trong môi trường hiện tại.")
        st.code(str(e))
        st.info("Hãy cài `opencv-python-headless` (local: dùng venv) rồi chạy lại.")
        st.stop()

    # Lazy init processor to avoid UI freeze on first load (mediapipe/rembg can take time).
    if lazy_init:
        with st.spinner("Đang khởi tạo engine (MediaPipe + rembg)… lần đầu có thể mất 10–60 giây."):
            processor = _get_processor(ratio=ratio, blue_rgb=blue_rgb, min_face_conf=min_face_conf)
    else:
        processor = _get_processor(ratio=ratio, blue_rgb=blue_rgb, min_face_conf=min_face_conf)

    progress = st.progress(0)
    processed_zip_items: List[Tuple[str, bytes]] = []

    for idx, up in enumerate(uploads, start=1):
        progress.progress(min(100, int((idx - 1) / max(len(uploads), 1) * 100)))

        raw = up.read()
        filename = up.name

        try:
            pil = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            st.error(f"Không đọc được ảnh: `{filename}`")
            continue

        with st.spinner(f"Đang xử lý: {filename}"):
            try:
                res = processor.process(pil)
            except RuntimeError as e:
                st.error(str(e))
                continue

        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            c1, c2, c3 = st.columns([1.05, 1.05, 1.25], gap="large")

            with c1:
                st.markdown("**Original**")
                st.image(pil, caption=filename, use_container_width=True)

            with c2:
                st.markdown("**Trạng thái / Cảnh báo**")
                if res.status == "OK":
                    st.markdown('<span class="badge-ok">OK</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="badge-fail">FAILED</span>', unsafe_allow_html=True)

                if res.errors:
                    st.error("\n".join(res.errors))
                if res.warnings:
                    st.warning("\n".join(res.warnings))
                if not res.errors and not res.warnings:
                    st.markdown('<div class="small-note muted">Không có cảnh báo.</div>', unsafe_allow_html=True)

                checks_dict = _result_to_checks_dict(res)
                st.markdown("**Checklist**")
                _render_checklist(checks_dict)

            with c3:
                st.markdown("**Processed**")
                if res.processed_image is None:
                    st.info("Không xử lý được do lỗi phát hiện khuôn mặt.")
                else:
                    st.image(res.processed_image, use_container_width=True)
                    out_bytes = pil_to_jpeg_bytes(res.processed_image, quality=95)
                    base = filename.rsplit(".", 1)[0] if "." in filename else filename
                    zip_name = f"{idx:03d}_{base}_chuanhoa.jpg"
                    dl_name = f"{base}_chuanhoa.jpg"
                    processed_zip_items.append((zip_name, out_bytes))
                    st.download_button(
                        label="Download ảnh này (JPG)",
                        data=out_bytes,
                        file_name=dl_name,
                        mime="image/jpeg",
                        use_container_width=True,
                        key=f"dl_single_{idx}",
                    )

            st.markdown("</div>", unsafe_allow_html=True)

    progress.progress(100)

    st.markdown("### Tải về toàn bộ")
    if processed_zip_items:
        zip_bytes = _make_zip(processed_zip_items)
        st.download_button(
            label=f"Download ZIP ({len(processed_zip_items)} ảnh)",
            data=zip_bytes,
            file_name="anh_chan_dung_chuan_hoa.zip",
            mime="application/zip",
            type="primary",
            key="dl_zip_all",
        )
    else:
        st.warning("Không có ảnh nào được xử lý thành công để đóng gói ZIP.")


if __name__ == "__main__":
    main()
