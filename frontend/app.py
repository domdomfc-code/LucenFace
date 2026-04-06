from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import streamlit as st
from PIL import Image

# Đảm bảo import được package `backend` khi chạy: streamlit run frontend/app.py
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend.image_utils import PortraitProcessor, ProcessResult, pil_to_jpeg_bytes


APP_TITLE = "Chuẩn hóa ảnh chân dung học sinh"
# Đổi số khi deploy để kiểm tra Streamlit Cloud đã build bản mới (sidebar hiển thị).
APP_BUILD = "2.3-mediapipe-0.10.33"
BLUE = "#005BC4"
BG = "#F6F9FF"


def _inject_css() -> None:
    st.markdown(
        f"""
        <style>
          .stApp {{
            background: {BG};
          }}
          .app-title {{
            font-size: 1.65rem;
            font-weight: 800;
            color: {BLUE};
            margin-bottom: 0.25rem;
          }}
          .app-subtitle {{
            color: #2b2f38;
            opacity: 0.75;
            margin-bottom: 1rem;
          }}
          .card {{
            background: white;
            border: 1px solid rgba(0,0,0,0.06);
            border-radius: 14px;
            padding: 14px 14px 10px 14px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.05);
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
            color: #6b7280;
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

    st.markdown(f'<div class="app-title">{APP_TITLE}</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">Upload ảnh → kiểm tra tiêu chuẩn → auto-crop/cân bằng sáng → thay nền xanh → tải về ZIP.</div>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.caption(f"**Build:** `{APP_BUILD}` — UI `frontend/`, xử lý `backend/`")
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

    st.markdown("### Upload ảnh")
    uploads = st.file_uploader(
        "Kéo-thả hoặc bấm để chọn (JPG/PNG, tối đa 50 ảnh).",
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
                st.markdown(_checklist_html(checks_dict), unsafe_allow_html=True)

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
