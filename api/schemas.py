"""Cấu hình xử lý ảnh (JSON gửi kèm multipart)."""
from __future__ import annotations

from typing import Literal, Tuple

from pydantic import BaseModel, Field, field_validator

Ratio = Literal["3x4", "4x6"]
RembgEngine = Literal["none", "local", "remove_bg_api"]


class ProcessConfig(BaseModel):
    ratio: Ratio = "3x4"
    prefer_face_crop: bool = False
    replace_blue_bg: bool = True
    force_blue_despite_uniform: bool = False
    blue_hex: str = Field(default="#005BC4", description="Màu nền khi ghép")
    min_face_conf: float = Field(default=0.9, ge=0.3, le=0.9)
    auto_orient: bool = True
    crop_center_mode: Literal["nose", "face"] = "nose"
    letterbox_smart_framing: bool = True
    rembg_engine: RembgEngine = "remove_bg_api"
    rembg_model: str = "u2net_human_seg"

    @field_validator("blue_hex")
    @classmethod
    def hex_ok(cls, v: str) -> str:
        s = (v or "").strip()
        if not s.startswith("#") or len(s) != 7:
            raise ValueError("blue_hex phải dạng #RRGGBB")
        try:
            int(s[1:], 16)
        except ValueError as e:
            raise ValueError("blue_hex không hợp lệ") from e
        return s

    def blue_rgb(self) -> Tuple[int, int, int]:
        h = self.blue_hex.lstrip("#")
        return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]

    def effective_rembg_engine(self) -> str:
        if not self.replace_blue_bg:
            return "none"
        return self.rembg_engine

    def cache_key(self) -> tuple:
        """Khóa cache PortraitProcessor."""
        return (
            self.ratio,
            self.blue_rgb(),
            round(self.min_face_conf, 2),
            self.effective_rembg_engine(),
            self.rembg_model,
        )
