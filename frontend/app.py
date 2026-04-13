"""
Ứng dụng Streamlit chính: UI + luồng batch ảnh.
Chạy từ gốc dự án: `streamlit run app.py` (khuyến nghị) hoặc `streamlit run frontend/app.py`.
"""
from __future__ import annotations

import io
import platform
from typing import Any, Dict, List, Tuple

import streamlit as st
from PIL import Image, ImageOps

import frontend.bootstrap  # noqa: F401 — đưa thư mục gốc dự án vào sys.path
from frontend import backend_lazy
from frontend.backend_lazy import ensure_image_backend
from frontend.config import APP_BUILD, APP_TITLE, BLUE
from frontend.image_io import (
    decode_data_url_image,
    decode_data_url_image_verbose,
    gather_staged_images,
    looks_like_heic,
    normalize_filename_hint,
    sniff_image_kind,
)
from frontend.processor_service import get_cached_portrait_processor, run_portrait_process
from frontend.streamlit_helpers import (
    cv2_troubleshoot_markdown,
    make_zip,
    read_remove_bg_api_key,
    render_checklist,
    result_to_checks_dict,
)
from frontend.styling import inject_app_css, render_sidebar_reopen_button
from frontend.thumbnails import render_image_thumbnails
from paste_image_component import paste_image_from_clipboard


def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🪪",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_app_css()
    render_sidebar_reopen_button()

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
            <span class="pill">Nền xanh (tùy chọn)</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(f'<div class="app-title">{APP_TITLE}</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">Kéo & thả ảnh để chuẩn hóa (khung tùy chọn, cân bằng sáng; có thể ghép nền xanh hoặc chỉ giữ nền gốc — tải ZIP).</div>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.caption(f"**Build:** `{APP_BUILD}` — UI `frontend/`, xử lý `backend/`")
        with st.expander("Kiểm tra thư viện", expanded=False):
            st.write(f"**Python:** `{platform.python_version()}` — **HĐH:** `{platform.system()}`")
            try:
                import cv2  # type: ignore

                st.success(f"OpenCV OK (`cv2`): {getattr(cv2, '__version__', 'unknown')}")
            except Exception as e:
                st.error("Thiếu OpenCV (`cv2`).")
                st.code(str(e))
                if platform.system() == "Windows":
                    st.caption(
                        "Windows: `python -m venv .venv` → `.\\.venv\\Scripts\\Activate.ps1` "
                        "→ `pip install -r requirements.txt` hoặc `pip install opencv-python-headless`."
                    )
                else:
                    st.caption(
                        "Gợi ý (Python 3.13): `pip install -r requirements-local-py313.txt` trong venv."
                    )

        st.markdown("### Hướng dẫn nhanh")
        st.markdown(
            """
            - **Ảnh hợp lệ**: JPG/PNG, chân dung rõ mặt, **chỉ 1 khuôn mặt**.
            - **Tiêu chuẩn kiểm tra**:
              - Vị trí mặt gần trung tâm ngang
              - Tỷ lệ khuôn mặt hợp lý
              - Độ sáng & tương phản đủ
              - **Phông nền**: ảnh gốc & khung đầu ra — viền gần **một màu** (đạt chuẩn) hay không
            - **Khung**: mặc định giữ ảnh gốc (scale); tự cắt theo mặt nếu mặt quá nhỏ trong ảnh (dưới ~44% chiều cao), hoặc nền không đơn sắc, hoặc bạn bật cắt mặt.
            - **Ghép nền xanh**: khi bật rembg, nếu khung đầu ra **đã có phông một màu** thì **không** ghép thêm (trừ khi bạn bật “luôn ghép”).
            """
        )
        st.markdown("---")
        ratio = st.selectbox("Tỷ lệ ảnh đầu ra", ["3x4", "4x6"], index=0)
        prefer_face_crop = st.toggle(
            "Cắt theo khuôn mặt (chuẩn chân dung)",
            value=False,
            help=(
                "Tắt: giữ khung gốc và chỉ scale — trừ khi nền không đơn sắc hoặc mặt quá nhỏ trong ảnh "
                "(dưới ~44% chiều cao), lúc đó sẽ tự crop để chủ thể lấp khung hơn. Bật: luôn crop theo mặt."
            ),
        )
        replace_blue_bg = st.toggle(
            "Tự động ghép nền xanh",
            value=True,
            help="Bật: tách nền (rembg local hoặc remove.bg API) rồi ghép màu. Tắt: chỉ crop/scale, giữ nền gốc.",
        )
        force_blue_despite_uniform = False
        rembg_engine = "none"
        rembg_model = "u2net_human_seg"
        if replace_blue_bg:
            force_blue_despite_uniform = st.toggle(
                "Luôn ghép nền xanh (kể cả phông đã một màu)",
                value=False,
                help=(
                    "Tắt (mặc định): nền đơn sắc thì **không** rembg (an toàn với studio trắng/xanh). "
                    "Bật: **luôn** ghép màu nền bạn chọn kể cả khi phông đã một màu (đổi màu nền; có thể viền artefact trên áo tối)."
                ),
            )
            st.markdown("### Thông số nền xanh")
            st.caption("Chuẩn mặc định: `#005BC4` (RGB 0, 91, 196).")
            blue_hex = st.color_picker("Chọn màu nền", value=BLUE)
        else:
            st.caption("Ghép nền xanh đang **tắt** — không dùng rembg; màu nền bên dưới không áp dụng.")
        max_files = 50
        st.caption(f"Tối đa {max_files} ảnh/lần.")
        st.markdown("---")
        st.markdown("### Nâng cao")
        min_face_conf = st.slider("Độ tin cậy phát hiện mặt", min_value=0.3, max_value=0.9, value=0.9, step=0.05)
        auto_orient = st.toggle(
            "Kiểm tra hướng ảnh (không xoay output)",
            value=True,
            help="Bật: hệ thống sẽ thử xoay khi kiểm tra checklist để cảnh báo ảnh bị xoay/lật, nhưng ảnh xuất ra vẫn giữ hướng gốc.",
        )
        enable_global_paste = st.toggle(
            "Bắt Ctrl+V toàn trang",
            value=True,
            help="Bật: Ctrl+V/⌘+V ở bất kỳ đâu sẽ dán ảnh (trừ khi đang gõ trong ô chữ). Tắt nếu bạn không muốn hotkey global.",
        )
        lazy_init = st.toggle(
            "Khởi tạo engine khi bấm xử lý",
            value=True,
            help="Tránh đứng UI khi tải MediaPipe (và rembg nếu bật ghép nền xanh).",
        )
        if replace_blue_bg:
            st.markdown("---")
            st.markdown("### Engine tách nền")
            _eng_pick = st.radio(
                "Nguồn tách nền",
                options=["rembg (local, miễn phí)", "remove.bg (API — gần chất lượng web upload)"],
                index=0,
                help="remove.bg tương đương trang [remove.bg/upload](https://www.remove.bg/vi/upload) — cần API key.",
            )
            if _eng_pick.startswith("remove.bg"):
                rembg_engine = "remove_bg_api"
                st.caption("[remove.bg — lấy API key](https://www.remove.bg/api)")
                if read_remove_bg_api_key():
                    st.success("Đã có `REMOVEBG_API_KEY` (Secrets / môi trường).")
                else:
                    st.warning("Thêm `REMOVEBG_API_KEY` trong **Streamlit Secrets** hoặc biến môi trường.")
            else:
                rembg_engine = "local"
                _rembg_models = ["u2net", "isnet-general-use", "u2net_human_seg", "silueta"]
                rembg_model = st.selectbox(
                    "Model rembg (ONNX)",
                    options=_rembg_models,
                    index=_rembg_models.index("u2net_human_seg"),
                    help="**u2net_human_seg** (mặc định): tốt cho người. **u2net**: ổn định với pymatting. ISNet: không dùng pymatting (tránh viền mờ kép).",
                )

    if replace_blue_bg:
        blue_rgb = tuple(int(blue_hex.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
    else:
        blue_rgb = tuple(int(BLUE.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))

    st.caption(
        "Mẹo: ảnh rõ mặt, thẳng góc — tùy chọn ghép nền xanh và rembg nằm trong **sidebar** (☰)."
    )
    _g_left, _g_mid, _g_right = st.columns([1, 3.2, 1])
    with _g_mid:
        st.markdown(
            """
<div class="p2c-upload-hero-top">
  <div class="p2c-upload-icon" aria-hidden="true">
    <svg width="38" height="38" viewBox="0 0 56 56" fill="none" xmlns="http://www.w3.org/2000/svg">
      <rect x="6" y="10" width="44" height="36" rx="6" fill="#4b5563"/>
      <path d="M12 38 L22 26 L30 34 L38 22 L44 28 V38 H12 Z" fill="#9ca3af"/>
      <circle cx="40" cy="18" r="5" fill="#d1d5db"/>
    </svg>
  </div>
  <div class="p2c-upload-tagline-scroll"><p class="p2c-upload-tagline">Kéo hình ảnh của bạn bất cứ nơi nào trên trang này hoặc nhấn <strong style="color:rgba(255,255,255,0.95)">Ctrl</strong> / <strong style="color:rgba(255,255,255,0.95)">⌘</strong> + <strong style="color:rgba(255,255,255,0.95)">V</strong> để dán hình ảnh.</p></div>
</div>
""",
            unsafe_allow_html=True,
        )
        uploads = st.file_uploader(
            "Chọn ảnh",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            help="JPG, PNG — tối đa ~20MB mỗi file; tối đa 50 ảnh mỗi lần.",
            label_visibility="collapsed",
        )
        nonce = int(st.session_state.get("p2c_clipboard_paste_nonce", 0))
        pasted_data_url = paste_image_from_clipboard(
            enable_global_paste=enable_global_paste,
            key=f"p2c_clipboard_paste_{nonce}",
        )

        upload_list = list(uploads) if uploads else []
        if not upload_list and not pasted_data_url:
            st.markdown(
                '<div class="p2c-upload-empty">Chưa có ảnh — hãy <strong>chọn tệp</strong> hoặc <strong>dán</strong> từ clipboard để bắt đầu.</div>',
                unsafe_allow_html=True,
            )
            return

    staged = gather_staged_images(upload_list, pasted_data_url)
    if not staged:
        if pasted_data_url and not upload_list:
            _dec, reason = decode_data_url_image_verbose(pasted_data_url)
            if reason:
                st.warning(f"Không nhận được ảnh từ clipboard. Lý do: **{reason}**")
        st.warning("Không đọc được ảnh hợp lệ — chỉ hỗ trợ **JPG/PNG** (HEIC/HEIF cần export sang JPG).")
        return

    if len(staged) > 50:
        st.error("Tối đa 50 ảnh mỗi lần (upload + dán tính chung). Vui lòng giảm số lượng và thử lại.")
        return

    c1, c2 = st.columns([1, 2], gap="small")
    with c1:
        if st.button("Xóa ảnh clipboard", width="content"):
            st.session_state["p2c_clipboard_paste_nonce"] = int(st.session_state.get("p2c_clipboard_paste_nonce", 0)) + 1
            st.rerun()
    with c2:
        if pasted_data_url:
            dec = decode_data_url_image(str(pasted_data_url))
            if dec:
                blob, fn = dec
                st.success(f"Đã nhận ảnh từ clipboard: `{fn}` ({len(blob)/1024:.0f} KB).")

    if "p2c_selected" not in st.session_state:
        st.session_state["p2c_selected"] = {}
    selected: Dict[str, bool] = st.session_state["p2c_selected"]

    a1, a2, a3 = st.columns([1, 1, 2], gap="small")
    with a1:
        if st.button("Chọn tất cả", width="content"):
            for i, (fname, _) in enumerate(staged):
                selected[f"sel::{i}::{fname}"] = True
    with a2:
        if st.button("Bỏ chọn", width="content"):
            for i, (fname, _) in enumerate(staged):
                selected[f"sel::{i}::{fname}"] = False
    with a3:
        st.caption("Bạn có thể bỏ chọn ảnh không muốn xử lý.")

    render_image_thumbnails(staged, selected)

    st.markdown("### Xử lý hàng loạt")
    selected_n = sum(1 for i, (fname, _raw) in enumerate(staged) if selected.get(f"sel::{i}::{fname}", True))
    st.caption(f"Đã chọn **{selected_n}/{len(staged)}** ảnh (giới hạn tối đa **50**).")
    b1, b2 = st.columns([1, 1], gap="small")
    with b1:
        start = st.button("Bắt đầu xử lý", type="primary", width="content")
    with b2:
        stop_now = st.button("Dừng", width="content")
        if stop_now:
            st.session_state["p2c_stop"] = True
    if not start:
        st.caption("Bấm **Bắt đầu xử lý** để chạy pipeline cho toàn bộ ảnh.")
        return
    st.session_state["p2c_stop"] = False

    if replace_blue_bg and rembg_engine == "remove_bg_api" and not read_remove_bg_api_key():
        st.error(
            "Chế độ **remove.bg** cần API key. Trên Streamlit Cloud: **Settings → Secrets** thêm "
            "`REMOVEBG_API_KEY = \"...\"`. Local: biến môi trường cùng tên. "
            "[remove.bg API](https://www.remove.bg/api)"
        )
        st.stop()

    remove_bg_key_for_processor = (
        read_remove_bg_api_key() if (replace_blue_bg and rembg_engine == "remove_bg_api") else None
    )

    try:
        import cv2  # type: ignore  # noqa: F401
    except Exception as e:
        st.error("Thiếu OpenCV (`cv2`) hoặc OpenCV không import được trong môi trường hiện tại.")
        st.code(str(e))
        st.markdown(cv2_troubleshoot_markdown())
        st.stop()

    if not replace_blue_bg:
        _spin_engine = "MediaPipe"
    elif rembg_engine == "remove_bg_api":
        _spin_engine = "MediaPipe + remove.bg API"
    else:
        _spin_engine = f"MediaPipe + rembg ({rembg_model})"
    if lazy_init:
        with st.spinner(f"Đang khởi tạo engine ({_spin_engine})… lần đầu có thể mất 10–60 giây."):
            processor = get_cached_portrait_processor(
                ratio=ratio,
                blue_rgb=blue_rgb,
                min_face_conf=min_face_conf,
                rembg_engine=rembg_engine,
                rembg_model=rembg_model,
                remove_bg_api_key=remove_bg_key_for_processor,
                cache_version=APP_BUILD,
            )
    else:
        processor = get_cached_portrait_processor(
            ratio=ratio,
            blue_rgb=blue_rgb,
            min_face_conf=min_face_conf,
            rembg_engine=rembg_engine,
            rembg_model=rembg_model,
            remove_bg_api_key=remove_bg_key_for_processor,
            cache_version=APP_BUILD,
        )

    MAX_BYTES = 12 * 1024 * 1024
    work_items: List[Tuple[str, bytes]] = []
    rejected: List[Tuple[str, str]] = []
    for i, (fname, raw) in enumerate(staged):
        if not selected.get(f"sel::{i}::{fname}", True):
            continue
        fname_norm = normalize_filename_hint(fname)
        if looks_like_heic(fname_norm, raw):
            rejected.append((fname_norm, "Định dạng HEIC/HEIF chưa hỗ trợ — hãy Export/Save As **JPG** rồi upload lại."))
            continue
        kind = sniff_image_kind(raw)
        if kind is None:
            rejected.append((fname_norm, "File không giống ảnh hợp lệ (magic bytes không khớp JPG/PNG/GIF/WebP)."))
            continue
        if len(raw) > MAX_BYTES:
            rejected.append((fname_norm, f"File quá nặng ({len(raw)/1024/1024:.1f} MB) — hãy upload ảnh nhỏ hơn."))
            continue
        try:
            im = Image.open(io.BytesIO(raw))
            mode = (im.mode or "").upper()
            if mode in ("CMYK", "I;16", "I;16B", "I;16L", "I;16S", "I"):
                rejected.append((fname_norm, f"Ảnh ở chế độ {mode} (có thể lệch màu). Hãy xuất lại **RGB JPG** nếu bị sai."))
                continue
            im.verify()
        except Exception:
            rejected.append((fname_norm, "Không đọc được ảnh hoặc file không hợp lệ."))
            continue
        work_items.append((fname_norm, raw))

    if rejected:
        st.warning("Một số ảnh bị loại do không hợp lệ.")
        st.dataframe([{"file": f, "reason": r} for f, r in rejected], use_container_width=True, hide_index=True)

    if not work_items:
        st.error("Không có ảnh hợp lệ (hoặc bạn đã bỏ chọn hết).")
        return

    status_line = st.empty()
    progress = st.progress(0)
    processed_zip_items: List[Tuple[str, bytes]] = []
    failed_items: List[Dict[str, str]] = []

    for idx, (filename, raw) in enumerate(work_items, start=1):
        if st.session_state.get("p2c_stop"):
            st.warning("Đã dừng theo yêu cầu. Bạn có thể chỉnh lựa chọn rồi bấm **Bắt đầu xử lý** lại.")
            break
        status_line.caption(f"Đang xử lý **{idx}/{len(work_items)}**: `{filename}`")
        progress.progress(min(100, int((idx - 1) / max(len(work_items), 1) * 100)))

        try:
            pil = Image.open(io.BytesIO(raw))
            try:
                pil = ImageOps.exif_transpose(pil)
            except Exception:
                pass
            pil = pil.convert("RGB")
        except Exception:
            st.error(f"Không đọc được ảnh: `{filename}`")
            failed_items.append({"file": filename, "reason": "Không đọc được ảnh."})
            continue

        with st.spinner(f"Đang xử lý: {filename}"):
            try:
                res = run_portrait_process(
                    processor,
                    pil,
                    prefer_face_crop=prefer_face_crop,
                    replace_blue_bg=replace_blue_bg,
                    skip_rembg_if_uniform_background=not force_blue_despite_uniform,
                    auto_orient=auto_orient,
                )
            except RuntimeError as e:
                st.error(str(e))
                failed_items.append({"file": filename, "reason": str(e)})
                continue

        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            c1, c2, c3 = st.columns([1.05, 1.05, 1.25], gap="large")

            with c1:
                st.markdown("**Original**")
                st.image(pil, caption=filename, width="stretch")

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

                checks_dict = result_to_checks_dict(res)
                st.markdown("**Checklist**")
                render_checklist(checks_dict)

            with c3:
                st.markdown("**Processed**")
                if res.processed_image is None:
                    st.info("Không xử lý được do lỗi phát hiện khuôn mặt.")
                    failed_items.append({"file": filename, "reason": "Không tạo được ảnh đầu ra."})
                else:
                    st.image(res.processed_image, width="stretch")
                    ensure_image_backend()
                    pj = backend_lazy.pil_to_jpeg_bytes
                    assert pj is not None
                    out_bytes = pj(res.processed_image, quality=95)
                    base = filename.rsplit(".", 1)[0] if "." in filename else filename
                    zip_name = f"{idx:03d}_{base}_chuanhoa.jpg"
                    dl_name = f"{base}_chuanhoa.jpg"
                    processed_zip_items.append((zip_name, out_bytes))
                    st.download_button(
                        label="Download ảnh này (JPG)",
                        data=out_bytes,
                        file_name=dl_name,
                        mime="image/jpeg",
                        width="stretch",
                        key=f"dl_single_{idx}",
                    )

            st.markdown("</div>", unsafe_allow_html=True)

    progress.progress(100)

    st.markdown("### Tải về toàn bộ")
    ok_n = len(processed_zip_items)
    fail_n = len(failed_items) + len(rejected)
    st.markdown(f"**Kết quả:** OK **{ok_n}**, FAILED/loại **{fail_n}**.")
    if failed_items:
        st.markdown("### Danh sách lỗi")
        st.dataframe(failed_items, use_container_width=True, hide_index=True)
    if processed_zip_items:
        zip_bytes = make_zip(processed_zip_items)
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
