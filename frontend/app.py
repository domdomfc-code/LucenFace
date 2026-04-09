from __future__ import annotations

import base64
import inspect
import io
import os
import platform
import re
import sys
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import streamlit as st
from PIL import Image, ImageOps

# Đảm bảo import được package `backend` khi chạy: streamlit run frontend/app.py
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paste_image_component import paste_image_from_clipboard

# Không import `backend.image_utils` lúc load module — tránh crash/OOM/timeout healthz trên Streamlit Cloud
# (OpenCV + MediaPipe + rembg + ONNX rất nặng). Chỉ tải khi `_ensure_image_backend()` chạy.
if TYPE_CHECKING:
    from backend.image_utils import PortraitProcessor, ProcessResult

PortraitProcessor = None  # type: ignore[assignment,misc]
ProcessResult = None  # type: ignore[assignment,misc]
pil_to_jpeg_bytes = None  # type: ignore[assignment,misc]
_image_backend_loaded = False


def _ensure_image_backend() -> None:
    """Lazy-import pipeline ảnh — gọi trước khi dùng PortraitProcessor / pil_to_jpeg_bytes."""
    global PortraitProcessor, ProcessResult, pil_to_jpeg_bytes, _image_backend_loaded
    if _image_backend_loaded:
        return
    from backend.image_utils import PortraitProcessor as _PC, ProcessResult as _PR, pil_to_jpeg_bytes as _pj

    PortraitProcessor = _PC
    ProcessResult = _PR
    pil_to_jpeg_bytes = _pj
    _image_backend_loaded = True


def _decode_data_url_image(data_url: str) -> Optional[Tuple[bytes, str]]:
    """Parse data:image/...;base64,... → (bytes, filename gợi ý)."""
    if not data_url or not isinstance(data_url, str) or not data_url.startswith("data:"):
        return None
    m = re.match(
        r"data:image/([\w.+-]+);base64,(.+)",
        data_url.strip(),
        re.DOTALL | re.IGNORECASE,
    )
    if not m:
        return None
    mime = m.group(1).lower().replace("jpg", "jpeg")
    b64 = m.group(2).strip()
    try:
        raw = base64.b64decode(b64, validate=False)
    except Exception:
        return None
    if not raw:
        return None
    ext = "png"
    if "jpeg" in mime or mime == "jpeg":
        ext = "jpg"
    elif "webp" in mime:
        ext = "webp"
    elif "gif" in mime:
        ext = "gif"
    return raw, f"clipboard.{ext}"


def _decode_data_url_image_verbose(data_url: Any) -> Tuple[Optional[Tuple[bytes, str]], str | None]:
    """
    Decode data URL từ clipboard, trả về (bytes, filename) hoặc (None, reason).
    """
    if not data_url:
        return None, "Clipboard rỗng."
    if not isinstance(data_url, str):
        return None, "Dữ liệu clipboard không hợp lệ."
    if not data_url.startswith("data:"):
        return None, "Clipboard không phải data URL."
    if not data_url.lower().startswith("data:image/"):
        return None, "Clipboard không phải ảnh (data:image/...)."
    dec = _decode_data_url_image(data_url)
    if not dec:
        return None, "Không giải mã được ảnh từ clipboard (base64 lỗi hoặc định dạng lạ)."
    return dec, None


def _sniff_image_kind(raw: bytes) -> str | None:
    """
    Sniff magic bytes để chặn file giả mạo không phải ảnh.
    Trả về: 'jpeg' | 'png' | 'gif' | 'webp' hoặc None.
    """
    if not raw or len(raw) < 12:
        return None
    if raw[:2] == b"\xff\xd8":
        return "jpeg"
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if raw[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "webp"
    return None


def _normalize_filename_hint(name: str) -> str:
    return (name or "").strip().replace("\\", "/").split("/")[-1]


def _looks_like_heic(name: str, raw: bytes) -> bool:
    n = _normalize_filename_hint(name).lower()
    if n.endswith(".heic") or n.endswith(".heif"):
        return True
    # HEIF/HEIC brand in ftyp box
    if len(raw) >= 12 and raw[4:8] == b"ftyp":
        brand = raw[8:12]
        if brand in (b"heic", b"heix", b"hevc", b"hevx", b"mif1", b"msf1"):
            return True
    return False


def _gather_staged_images(upload_list: List[Any], pasted_data_url: Any) -> List[Tuple[str, bytes]]:
    """Đọc bytes từ upload + clipboard (một lần) để xem trước và xử lý."""
    out: List[Tuple[str, bytes]] = []
    for up in upload_list:
        try:
            up.seek(0)
        except Exception:
            pass
        out.append((up.name, up.read()))
    if pasted_data_url:
        dec, _reason = _decode_data_url_image_verbose(pasted_data_url)
        if dec:
            blob, fn = dec
            out.append((fn, blob))
    return out


def _render_image_thumbnails(
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


def _read_remove_bg_api_key() -> str | None:
    """API key remove.bg: Streamlit Secrets hoặc biến môi trường REMOVEBG_API_KEY."""
    try:
        k = st.secrets.get("REMOVEBG_API_KEY")
        if k is not None and str(k).strip():
            return str(k).strip()
    except Exception:
        pass
    v = os.environ.get("REMOVEBG_API_KEY", "").strip()
    return v or None


def _cv2_troubleshoot_markdown() -> str:
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


APP_TITLE = "Chuẩn hóa ảnh chân dung học sinh"
# Đổi số khi deploy để kiểm tra Streamlit Cloud đã build bản mới (sidebar hiển thị).
APP_BUILD = "3.10.4-sidebar-advanced-before-engine"
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

          /* Sidebar mặc định mở + rộng tối đa hợp lý (clamp tránh tràn màn hình nhỏ). */
          [data-testid="stSidebar"] {{
            min-width: clamp(280px, 30rem, min(560px, 52vw)) !important;
            width: clamp(280px, 30rem, min(560px, 52vw)) !important;
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
    html = """
<!DOCTYPE html><html><head><meta charset="utf-8"></head>
<body style="margin:0;padding:0;background:transparent;">
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
</body></html>
"""
    iframe = getattr(st, "iframe", None)
    if iframe is not None:
        iframe(html, height=52)
    else:
        import streamlit.components.v1 as components

        components.html(html, height=52)


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
    st.dataframe(rows, width="stretch", hide_index=True)


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
def _get_processor(
    ratio: str,
    blue_rgb: Tuple[int, int, int],
    min_face_conf: float,
    *,
    rembg_engine: str,
    rembg_model: str,
    remove_bg_api_key: str | None,
    cache_version: str = APP_BUILD,
) -> PortraitProcessor:
    """
    `cache_version` đổi khi deploy (APP_BUILD) để tránh giữ PortraitProcessor cũ
    không khớp chữ ký `process(..., replace_background=...)` → TypeError trên Cloud.
    """
    _ensure_image_backend()
    _ = cache_version  # phân vùng cache theo APP_BUILD
    return PortraitProcessor(
        ratio=ratio,
        blue_rgb=blue_rgb,
        min_face_conf=min_face_conf,
        rembg_engine=rembg_engine,
        rembg_model=rembg_model,
        remove_bg_api_key=remove_bg_api_key,
    )


def _run_processor(
    processor: PortraitProcessor,
    pil: Image.Image,
    *,
    prefer_face_crop: bool,
    replace_blue_bg: bool,
    skip_rembg_if_uniform_background: bool = True,
    auto_orient: bool = True,
) -> ProcessResult:
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
    return processor.process(pil, prefer_face_crop=prefer_face_crop, **kw)


def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🪪",
        layout="wide",
        initial_sidebar_state="expanded",
    )
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
                if _read_remove_bg_api_key():
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

    st.markdown("### Kéo & thả ảnh / Dán từ clipboard")
    st.markdown(
        '<div class="card-soft muted">Mẹo: ảnh rõ mặt, thẳng góc. Bật/tắt ghép nền xanh trong sidebar trước khi xử lý. Cắt theo mặt khi bạn bật hoặc nền không đơn sắc. '
        "Có thể <strong>dán ảnh</strong>: copy ảnh hoặc chụp màn hình, rồi <strong>Ctrl+V / ⌘+V</strong> bất kỳ đâu trên trang (trừ khi đang gõ trong ô chữ).</div>",
        unsafe_allow_html=True,
    )
    uploads = st.file_uploader(
        "Kéo và thả tệp vào đây hoặc bấm để chọn",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )
    st.caption(
        "**Dán ảnh:** **Ctrl+V** / **⌘+V** ở bất kỳ đâu trên trang (không cần nhấp ô) — không áp dụng khi focus đang ở ô nhập chữ/sidebar text. "
        "Hoặc bấm **đọc clipboard** trong khung component. Ảnh được nén trước khi gửi."
    )
    nonce = int(st.session_state.get("p2c_clipboard_paste_nonce", 0))
    pasted_data_url = paste_image_from_clipboard(
        enable_global_paste=enable_global_paste,
        key=f"p2c_clipboard_paste_{nonce}",
    )

    upload_list = list(uploads) if uploads else []
    if not upload_list and not pasted_data_url:
        st.info("Hãy **upload** ít nhất một ảnh, hoặc **dán** ảnh từ clipboard để bắt đầu.")
        return

    # Stage bytes into session_state so previews don't re-read unpredictably across reruns.
    staged = _gather_staged_images(upload_list, pasted_data_url)
    if not staged:
        if pasted_data_url and not upload_list:
            _dec, reason = _decode_data_url_image_verbose(pasted_data_url)
            if reason:
                st.warning(f"Không nhận được ảnh từ clipboard. Lý do: **{reason}**")
        st.warning("Không đọc được ảnh hợp lệ — chỉ hỗ trợ **JPG/PNG** (HEIC/HEIF cần export sang JPG).")
        return

    if len(staged) > 50:
        st.error("Tối đa 50 ảnh mỗi lần (upload + dán tính chung). Vui lòng giảm số lượng và thử lại.")
        return

    # Clipboard utilities
    c1, c2 = st.columns([1, 2], gap="small")
    with c1:
        if st.button("Xóa ảnh clipboard", width="content"):
            # Reset by changing component key
            st.session_state["p2c_clipboard_paste_nonce"] = int(st.session_state.get("p2c_clipboard_paste_nonce", 0)) + 1
            st.rerun()
    with c2:
        if pasted_data_url:
            dec = _decode_data_url_image(str(pasted_data_url))
            if dec:
                blob, fn = dec
                st.success(f"Đã nhận ảnh từ clipboard: `{fn}` ({len(blob)/1024:.0f} KB).")

    # Selection
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

    _render_image_thumbnails(staged, selected)

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

    if replace_blue_bg and rembg_engine == "remove_bg_api" and not _read_remove_bg_api_key():
        st.error(
            "Chế độ **remove.bg** cần API key. Trên Streamlit Cloud: **Settings → Secrets** thêm "
            "`REMOVEBG_API_KEY = \"...\"`. Local: biến môi trường cùng tên. "
            "[remove.bg API](https://www.remove.bg/api)"
        )
        st.stop()

    remove_bg_key_for_processor = (
        _read_remove_bg_api_key() if (replace_blue_bg and rembg_engine == "remove_bg_api") else None
    )

    # Preflight: ensure OpenCV is importable before creating cached processor.
    # If cv2 is missing, `st.cache_resource` would cache the exception and keep failing.
    try:
        import cv2  # type: ignore  # noqa: F401
    except Exception as e:
        st.error("Thiếu OpenCV (`cv2`) hoặc OpenCV không import được trong môi trường hiện tại.")
        st.code(str(e))
        st.markdown(_cv2_troubleshoot_markdown())
        st.stop()

    # Lazy init processor to avoid UI freeze on first load (mediapipe/rembg can take time).
    if not replace_blue_bg:
        _spin_engine = "MediaPipe"
    elif rembg_engine == "remove_bg_api":
        _spin_engine = "MediaPipe + remove.bg API"
    else:
        _spin_engine = f"MediaPipe + rembg ({rembg_model})"
    if lazy_init:
        with st.spinner(f"Đang khởi tạo engine ({_spin_engine})… lần đầu có thể mất 10–60 giây."):
            processor = _get_processor(
                ratio=ratio,
                blue_rgb=blue_rgb,
                min_face_conf=min_face_conf,
                rembg_engine=rembg_engine,
                rembg_model=rembg_model,
                remove_bg_api_key=remove_bg_key_for_processor,
                cache_version=APP_BUILD,
            )
    else:
        processor = _get_processor(
            ratio=ratio,
            blue_rgb=blue_rgb,
            min_face_conf=min_face_conf,
            rembg_engine=rembg_engine,
            rembg_model=rembg_model,
            remove_bg_api_key=remove_bg_key_for_processor,
            cache_version=APP_BUILD,
        )

    # Filter selected items + basic validation limits
    MAX_BYTES = 12 * 1024 * 1024  # ~12MB per image staged (post-clipboard compression typically smaller)
    # Không chặn ảnh nhỏ: một số ảnh crop/scan vẫn dùng được; checklist sẽ tự đánh giá thêm.
    work_items: List[Tuple[str, bytes]] = []
    rejected: List[Tuple[str, str]] = []
    for i, (fname, raw) in enumerate(staged):
        if not selected.get(f"sel::{i}::{fname}", True):
            continue
        fname_norm = _normalize_filename_hint(fname)
        if _looks_like_heic(fname_norm, raw):
            rejected.append((fname_norm, "Định dạng HEIC/HEIF chưa hỗ trợ — hãy Export/Save As **JPG** rồi upload lại."))
            continue
        kind = _sniff_image_kind(raw)
        if kind is None:
            rejected.append((fname_norm, "File không giống ảnh hợp lệ (magic bytes không khớp JPG/PNG/GIF/WebP)."))
            continue
        if len(raw) > MAX_BYTES:
            rejected.append((fname_norm, f"File quá nặng ({len(raw)/1024/1024:.1f} MB) — hãy upload ảnh nhỏ hơn."))
            continue
        try:
            im = Image.open(io.BytesIO(raw))
            w, h = im.size
            # Không reject theo kích thước; nếu quá nhỏ, face detection/checklist có thể FAIL sau.
            mode = (im.mode or "").upper()
            if mode in ("CMYK", "I;16", "I;16B", "I;16L", "I;16S", "I"):
                # Vẫn xử lý được khi convert RGB, nhưng cảnh báo sớm để tránh màu sai.
                rejected.append((fname_norm, f"Ảnh ở chế độ {mode} (có thể lệch màu). Hãy xuất lại **RGB JPG** nếu bị sai."))
                continue
            # verify then reopen later in pipeline
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
                res = _run_processor(
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

                checks_dict = _result_to_checks_dict(res)
                st.markdown("**Checklist**")
                _render_checklist(checks_dict)

            with c3:
                st.markdown("**Processed**")
                if res.processed_image is None:
                    st.info("Không xử lý được do lỗi phát hiện khuôn mặt.")
                    failed_items.append({"file": filename, "reason": "Không tạo được ảnh đầu ra."})
                else:
                    st.image(res.processed_image, width="stretch")
                    _ensure_image_backend()
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
