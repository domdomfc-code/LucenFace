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

try:
    from frontend.app import main
except ImportError as e:
    import streamlit as st

    st.set_page_config(page_title="Lỗi phụ thuộc", layout="centered")
    st.error("Không tải được thư viện xử lý ảnh (OpenCV, MediaPipe, rembg, …). Kiểm tra `requirements.txt` và logs trên Streamlit Cloud.")
    st.code(str(e))
    st.stop()
else:
    main()
