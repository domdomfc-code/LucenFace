"""Đọc / kiểm tra bytes ảnh từ upload và clipboard (không phụ thuộc backend nặng)."""
from __future__ import annotations

import base64
import re
from typing import Any, List, Optional, Tuple


def decode_data_url_image(data_url: str) -> Optional[Tuple[bytes, str]]:
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


def decode_data_url_image_verbose(data_url: Any) -> Tuple[Optional[Tuple[bytes, str]], str | None]:
    """Decode data URL từ clipboard, trả về (bytes, filename) hoặc (None, reason)."""
    if not data_url:
        return None, "Clipboard rỗng."
    if not isinstance(data_url, str):
        return None, "Dữ liệu clipboard không hợp lệ."
    if not data_url.startswith("data:"):
        return None, "Clipboard không phải data URL."
    if not data_url.lower().startswith("data:image/"):
        return None, "Clipboard không phải ảnh (data:image/...)."
    dec = decode_data_url_image(data_url)
    if not dec:
        return None, "Không giải mã được ảnh từ clipboard (base64 lỗi hoặc định dạng lạ)."
    return dec, None


def sniff_image_kind(raw: bytes) -> str | None:
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


def normalize_filename_hint(name: str) -> str:
    return (name or "").strip().replace("\\", "/").split("/")[-1]


def looks_like_heic(name: str, raw: bytes) -> bool:
    n = normalize_filename_hint(name).lower()
    if n.endswith(".heic") or n.endswith(".heif"):
        return True
    if len(raw) >= 12 and raw[4:8] == b"ftyp":
        brand = raw[8:12]
        if brand in (b"heic", b"heix", b"hevc", b"hevx", b"mif1", b"msf1"):
            return True
    return False


def gather_staged_images(upload_list: List[Any], pasted_data_url: Any) -> List[Tuple[str, bytes]]:
    """Đọc bytes từ upload + clipboard (một lần) để xem trước và xử lý."""
    out: List[Tuple[str, bytes]] = []
    for up in upload_list:
        try:
            up.seek(0)
        except Exception:
            pass
        out.append((up.name, up.read()))
    if pasted_data_url:
        dec, _reason = decode_data_url_image_verbose(pasted_data_url)
        if dec:
            blob, fn = dec
            out.append((fn, blob))
    return out
