"""
Điểm vào Streamlit (Streamlit Cloud / `streamlit run app.py` từ thư mục gốc dự án).
UI nằm trong `frontend/`, logic xử lý ảnh trong `backend/`.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Đăng ký package trước submodule — giảm KeyError ('frontend'/'backend') khi import đồng thời trên Cloud.
try:
    import backend  # noqa: F401
    import frontend  # noqa: F401
except Exception:  # pragma: no cover
    pass

try:
    from frontend.app import main
except Exception as e:
    import streamlit as st

    st.set_page_config(page_title="Lỗi phụ thuộc", layout="centered")
    st.error(
        "Không tải được thư viện xử lý ảnh (OpenCV, MediaPipe, rembg, …). "
        "Trên Windows: bật venv rồi `pip install -r requirements.txt`. "
        "Trên Streamlit Cloud: xem logs và `requirements.txt`."
    )
    st.code(str(e))
    st.stop()
else:
    main()
