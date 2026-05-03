"""Ảnh mẫu một bấm — URL hoặc file trong repo (xem SAMPLE_DEMOS)."""
from __future__ import annotations

from pathlib import Path
from typing import List, TypedDict

import requests

_ROOT = Path(__file__).resolve().parent.parent


class SampleDemo(TypedDict, total=False):
    """Mỗi mục cần `label`, `filename` và một trong hai: `url` hoặc `path`."""

    label: str
    filename: str
    url: str
    # Đường dẫn tương đối từ thư mục gốc repo, ví dụ: "assets/sample_portraits/mau1.jpg"
    path: str


# --- Chỉnh ảnh mẫu của bạn tại đây -------------------------------------------
# • Để dùng link: đặt "url": "https://..."
# • Để dùng file trong repo: đặt "path": "assets/sample_portraits/ten_anh.jpg"
#   (tạo thư mục, bỏ ảnh JPG/PNG vào, commit cùng repo — không cần mạng khi chạy)
SAMPLE_DEMOS: List[SampleDemo] = [
    {
        "label": "Thử 1",
        "filename": "mau_01.png",
        "path": "assets/sample_portraits/sample_01.png",
    },
    {
        "label": "Thử 2",
        "filename": "mau_02.png",
        "path": "assets/sample_portraits/sample_02.png",
    },
    {
        "label": "Thử 3",
        "filename": "mau_03.png",
        "path": "assets/sample_portraits/sample_03.png",
    },
    {
        "label": "Thử 4",
        "filename": "mau_04.png",
        "path": "assets/sample_portraits/sample_04.png",
    },
    {
        "label": "Thử 5",
        "filename": "mau_05.png",
        "path": "assets/sample_portraits/sample_05.png",
    },
]


def _resolved_path(row: SampleDemo) -> Path | None:
    rel = (row.get("path") or "").strip()
    if not rel:
        return None
    return (_ROOT / rel).resolve()


def sample_image_for_display(row: SampleDemo) -> str:
    """Tham số cho `st.image`: đường dẫn file hoặc URL."""
    p = _resolved_path(row)
    if p is not None and p.is_file():
        return str(p)
    url = row.get("url")
    if url:
        return url
    raise ValueError(f"Mẫu '{row.get('label')}' cần 'url' hoặc 'path' trỏ tới file tồn tại.")


def load_demo_image_bytes(row: SampleDemo, timeout: float = 25.0) -> bytes:
    p = _resolved_path(row)
    if p is not None:
        if not p.is_file():
            raise FileNotFoundError(f"Không thấy file mẫu: {p}")
        return p.read_bytes()
    url = row.get("url")
    if not url:
        raise ValueError("Cần 'url' hoặc 'path' trong SAMPLE_DEMOS.")
    return _fetch_url_bytes(url, timeout=timeout)


def _fetch_url_bytes(url: str, timeout: float = 25.0) -> bytes:
    headers = {
        "User-Agent": "LucenFace/1.0 (portrait demo; educational use)",
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    }
    r = requests.get(url, timeout=timeout, headers=headers)
    r.raise_for_status()
    return bytes(r.content)


# Tương thích code cũ gọi fetch_demo_bytes(url)
def fetch_demo_bytes(url: str, timeout: float = 25.0) -> bytes:
    return _fetch_url_bytes(url, timeout=timeout)
