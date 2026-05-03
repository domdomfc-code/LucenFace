"""
Chạy API: từ thư mục gốc repo:
  pip install -r requirements.txt
  uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import base64
import io
import json
import zipfile
from pathlib import Path
from typing import Annotated, List

import sys

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

import frontend.bootstrap  # noqa: F401

from api.schemas import ProcessConfig
from api.service import (
    MAX_FILES,
    build_meta,
    read_remove_bg_api_key,
    run_audit,
    run_process_indices,
    validate_and_stage,
)

def _cors_origins() -> List[str]:
    import os

    raw = os.environ.get("WEB_CORS_ORIGINS", "http://localhost:3000")
    return [o.strip() for o in raw.split(",") if o.strip()]


app = FastAPI(title="LucenFace API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    ok_cv2 = True
    err = None
    try:
        import cv2  # noqa: F401

        _ = cv2.__version__
    except Exception as e:
        ok_cv2 = False
        err = repr(e)
    return {
        "ok": ok_cv2,
        "opencv": ok_cv2,
        "opencv_error": err,
        **build_meta(),
    }


def _parse_config(config_raw: str) -> ProcessConfig:
    try:
        data = json.loads(config_raw)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"config JSON không hợp lệ: {e}") from e
    try:
        return ProcessConfig.model_validate(data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


async def _read_uploads(files: List[UploadFile]) -> tuple[List[str], List[bytes]]:
    names: List[str] = []
    blobs: List[bytes] = []
    for f in files:
        raw = await f.read()
        names.append(f.filename or "upload.jpg")
        blobs.append(raw)
    return names, blobs


@app.post("/api/audit")
async def audit_endpoint(
    files: Annotated[List[UploadFile], File(description="Ảnh JPG/PNG")],
    config: Annotated[str, Form(description="JSON ProcessConfig")],
) -> JSONResponse:
    if not files:
        raise HTTPException(status_code=400, detail="Cần ít nhất một file.")
    if len(files) > MAX_FILES:
        raise HTTPException(
            status_code=400, detail=f"Tối đa {MAX_FILES} ảnh mỗi lần."
        )
    cfg = _parse_config(config)
    names, blobs = await _read_uploads(files)
    work_items, rejected = validate_and_stage(names, blobs)
    if not work_items:
        return JSONResponse(
            {
                "rejected": rejected,
                "rows": [],
                "message": "Không có ảnh hợp lệ.",
            }
        )
    try:
        rows = run_audit(work_items, cfg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý: {e}") from e
    return JSONResponse({"rejected": rejected, "rows": rows})


@app.post("/api/process")
async def process_endpoint(
    files: Annotated[List[UploadFile], File()],
    config: Annotated[str, Form()],
    indices: Annotated[str, Form(description='JSON mảng số, ví dụ [0,1,2]')],
) -> JSONResponse:
    if not files:
        raise HTTPException(status_code=400, detail="Cần ít nhất một file.")
    if len(files) > MAX_FILES:
        raise HTTPException(status_code=400, detail=f"Tối đa {MAX_FILES} ảnh.")
    cfg = _parse_config(config)
    try:
        idx_list = json.loads(indices)
        if not isinstance(idx_list, list):
            raise ValueError("indices phải là mảng")
        idx_int = [int(x) for x in idx_list]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"indices không hợp lệ: {e}") from e
    names, blobs = await _read_uploads(files)
    work_items, rejected = validate_and_stage(names, blobs)
    if rejected and not work_items:
        raise HTTPException(
            status_code=400,
            detail="Không có ảnh hợp lệ sau khi lọc.",
        )
    try:
        results, failures = run_process_indices(work_items, cfg, idx_int)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý: {e}") from e
    return JSONResponse({"results": results, "failures": failures, "rejected": rejected})


@app.post("/api/process-zip")
async def process_zip_endpoint(
    files: Annotated[List[UploadFile], File()],
    config: Annotated[str, Form()],
    indices: Annotated[str, Form()],
) -> Response:
    """Trả ZIP các ảnh JPG đã xử lý (chỉ ảnh thành công)."""
    if not files:
        raise HTTPException(status_code=400, detail="Cần ít nhất một file.")
    cfg = _parse_config(config)
    try:
        idx_list = json.loads(indices)
        idx_int = [int(x) for x in idx_list]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"indices không hợp lệ: {e}") from e
    names, blobs = await _read_uploads(files)
    work_items, _rej = validate_and_stage(names, blobs)
    if not work_items:
        raise HTTPException(status_code=400, detail="Không có ảnh hợp lệ.")
    try:
        results, _failures = run_process_indices(work_items, cfg, idx_int)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    buf = io.BytesIO()
    n = 0
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        for item in results:
            b64 = item.get("jpg_base64")
            name = item.get("download_name")
            if b64 and name:
                raw = base64.standard_b64decode(b64)
                z.writestr(name, raw)
                n += 1
    if n == 0:
        raise HTTPException(status_code=400, detail="Không có ảnh output để nén ZIP.")
    buf.seek(0)
    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={
            "Content-Disposition": 'attachment; filename="anh_chan_dung_chuan_hoa.zip"'
        },
    )


@app.get("/api/config-hints")
def config_hints() -> dict:
    return {
        "remove_bg_api_configured": bool(read_remove_bg_api_key()),
        "max_files": MAX_FILES,
        "max_bytes_per_file": 12 * 1024 * 1024,
    }
