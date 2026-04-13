"""
Đảm bảo thư mục gốc dự án có trên sys.path khi chạy `streamlit run frontend/app.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
