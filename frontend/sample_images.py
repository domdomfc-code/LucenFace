"""Ảnh mẫu một bấm (demo kiểu remove.bg) — URL công khai, tải qua requests."""
from __future__ import annotations

import requests

# Unsplash: giấy phép Unsplash; chỉ dùng để demo trong app.
SAMPLE_DEMOS: list[dict[str, str]] = [
    {
        "label": "Thử 1",
        "filename": "mau_chan_dung_1.jpg",
        "url": "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=480&auto=format&fit=crop&q=85",
    },
    {
        "label": "Thử 2",
        "filename": "mau_chan_dung_2.jpg",
        "url": "https://images.unsplash.com/photo-1506794778202-cad84cf45f1d?w=480&auto=format&fit=crop&q=85",
    },
    {
        "label": "Thử 3",
        "filename": "mau_chan_dung_3.jpg",
        "url": "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=480&auto=format&fit=crop&q=85",
    },
    {
        "label": "Thử 4",
        "filename": "mau_chan_dung_4.jpg",
        "url": "https://images.unsplash.com/photo-1529626455594-4ff0802cfb7e?w=480&auto=format&fit=crop&q=85",
    },
]


def fetch_demo_bytes(url: str, timeout: float = 25.0) -> bytes:
    headers = {
        "User-Agent": "LucenFace/1.0 (portrait demo; educational use)",
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    }
    r = requests.get(url, timeout=timeout, headers=headers)
    r.raise_for_status()
    return bytes(r.content)
