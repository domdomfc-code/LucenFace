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

from frontend.app import main

main()
