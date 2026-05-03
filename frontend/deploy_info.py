"""Thông tin đối chiếu bản đang chạy (local hoặc Streamlit Cloud)."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def git_short_sha() -> str | None:
    """7 ký tự đầu của HEAD nếu có thư mục .git (thường có trên Streamlit Cloud)."""
    for key in (
        "STREAMLIT_CLOUD_COMMIT_SHA",
        "SOURCE_VERSION",
        "GIT_COMMIT",
        "COMMIT_SHA",
    ):
        v = (os.environ.get(key) or "").strip()
        if v:
            return v[:7] if len(v) >= 7 else v
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(_ROOT),
            capture_output=True,
            text=True,
            timeout=4,
            shell=False,
        )
        if r.returncode == 0:
            s = (r.stdout or "").strip()
            return s or None
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return None


def is_streamlit_cloud() -> bool:
    """Heuristic: môi trường share.streamlit.io thường có biến này."""
    return bool(os.environ.get("STREAMLIT_SHARING_BASE_URL"))
