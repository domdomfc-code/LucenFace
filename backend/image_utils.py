from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# OpenCV is required for processing; keep import optional so UI can still load
# and show a friendly error instead of crashing at import-time.
try:
    import cv2  # type: ignore
    _CV2_IMPORT_ERROR: str | None = None
except Exception as e:  # pragma: no cover
    cv2 = None  # type: ignore
    _CV2_IMPORT_ERROR = repr(e)
import numpy as np
from PIL import Image

# Optional heavy deps (often not available on Python 3.13+ on Windows)
try:
    import mediapipe as mp  # type: ignore
except Exception:  # pragma: no cover
    mp = None  # type: ignore

try:
    from rembg import new_session, remove  # type: ignore
except Exception:  # pragma: no cover
    new_session = None  # type: ignore
    remove = None  # type: ignore


@dataclass
class CheckResult:
    ok: bool
    message: str


@dataclass
class ProcessResult:
    status: str  # "OK" | "FAILED"
    errors: List[str]
    warnings: List[str]
    checks: Dict[str, CheckResult]
    processed_image: Optional[Image.Image]


def _pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    _require_cv2()
    if pil_img.mode not in ("RGB", "RGBA"):
        pil_img = pil_img.convert("RGB")
    arr = np.array(pil_img)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        return arr
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    _require_cv2()
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _box_area_xyxy(box: Tuple[int, int, int, int]) -> float:
    return float(max(0, box[2] - box[0]) * max(0, box[3] - box[1]))


def _intersection_area_xyxy(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    return float(iw * ih)


def _iou_xyxy(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    inter = _intersection_area_xyxy(a, b)
    if inter <= 0.0:
        return 0.0
    aa = _box_area_xyxy(a)
    bb = _box_area_xyxy(b)
    union = aa + bb - inter
    return float(inter / union) if union > 0 else 0.0


def _fraction_of_a_inside_b(
    inner: Tuple[int, int, int, int], outer: Tuple[int, int, int, int]
) -> float:
    """Tỷ lệ diện tích inner giao với outer / diện tích inner (≈1 → inner gần như nằm trong outer)."""
    ia = _box_area_xyxy(inner)
    if ia <= 0:
        return 0.0
    return _intersection_area_xyxy(inner, outer) / ia


def _suppress_contained_duplicates(
    faces: List[Tuple[int, int, int, int, float]],
    contain_frac: float = 0.86,
    area_ratio_max: float = 0.52,
) -> List[Tuple[int, int, int, int, float]]:
    """
    Bỏ box nhỏ gần như nằm trọn trong box lớn hơn (MediaPipe đôi khi trả thêm bbox phụ trên tóc/mái).
    """
    if len(faces) <= 1:
        return faces
    faces = sorted(faces, key=lambda f: _box_area_xyxy(f[:4]), reverse=True)
    kept: List[Tuple[int, int, int, int, float]] = []
    for f in faces:
        fx = f[:4]
        fa = _box_area_xyxy(fx)
        drop = False
        for k in kept:
            kx = k[:4]
            ka = _box_area_xyxy(kx)
            if ka <= fa * 1.02:
                continue
            if fa > ka * area_ratio_max:
                continue
            if _fraction_of_a_inside_b(fx, kx) >= contain_frac:
                drop = True
                break
        if not drop:
            kept.append(f)
    return kept


def _nms_face_boxes(
    faces: List[Tuple[int, int, int, int, float]],
    iou_threshold: float = 0.38,
) -> List[Tuple[int, int, int, int, float]]:
    """
    Giữ một bbox cho mỗi khuôn mặt: loại trùng IoU cao (cùng người, nhiều detection).
    """
    if len(faces) <= 1:
        return faces
    faces = sorted(faces, key=lambda f: f[4], reverse=True)
    kept: List[Tuple[int, int, int, int, float]] = []
    for f in faces:
        if any(_iou_xyxy(f[:4], k[:4]) >= iou_threshold for k in kept):
            continue
        kept.append(f)
    return kept


def _dedupe_face_detections(
    faces: List[Tuple[int, int, int, int, float]],
) -> List[Tuple[int, int, int, int, float]]:
    """Chuỗi lọc: box lồng nhau → NMS IoU."""
    if len(faces) <= 1:
        return faces
    faces = _suppress_contained_duplicates(faces)
    return _nms_face_boxes(faces, iou_threshold=0.38)


def _compute_brightness_contrast(bgr: np.ndarray) -> Tuple[float, float]:
    _require_cv2()
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    return brightness, contrast


def _background_uniformity_check(bgr: np.ndarray, border_pct: float = 0.08) -> Tuple[bool, str]:
    """
    Heuristic: sample border pixels and check color variance.
    If standard deviation of RGB channels is small, background is likely solid.
    """
    h, w = bgr.shape[:2]
    bw = max(2, int(w * border_pct))
    bh = max(2, int(h * border_pct))

    top = bgr[0:bh, :, :]
    bottom = bgr[h - bh : h, :, :]
    left = bgr[:, 0:bw, :]
    right = bgr[:, w - bw : w, :]

    border = np.concatenate(
        [top.reshape(-1, 3), bottom.reshape(-1, 3), left.reshape(-1, 3), right.reshape(-1, 3)],
        axis=0,
    )
    std = border.astype(np.float32).std(axis=0).mean()

    if std < 18.0:
        return True, f"Nền tương đối đơn sắc (độ lệch chuẩn màu ~{std:.1f})."
    return False, f"Nền có thể không đơn sắc (độ lệch chuẩn màu ~{std:.1f})."


def _boost_bgr_for_face_detection(bgr: np.ndarray) -> np.ndarray:
    """
    Tăng sáng / CLAHE vừa phải để phát hiện mặt (MediaPipe, Haar) trên ảnh thiếu sáng.
    Giữ kích thước; bbox áp dụng lên ảnh gốc.
    """
    _require_cv2()
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    mean_y = float(np.mean(y))
    if mean_y >= 118:
        return bgr
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    y_adj = clahe.apply(y)
    strength = _clamp((100.0 - mean_y) / 95.0, 0.18, 0.58)
    y_mix = cv2.addWeighted(y, 1.0 - strength, y_adj, strength, 0).astype(np.float32)
    y_out = np.clip(((y_mix / 255.0) ** 0.92) * 255.0, 0, 255).astype(np.uint8)
    ycrcb_out = cv2.merge([y_out, cr, cb])
    return cv2.cvtColor(ycrcb_out, cv2.COLOR_YCrCb2BGR)


def _boost_bgr_for_segmentation(bgr: np.ndarray) -> np.ndarray:
    """
    Tăng sáng/tương phản mạnh hơn bước CLAHE thường — chỉ dùng làm *đầu vào* rembg
    (giúp phân tách chủ thể/nền khi ảnh tối). Màu đầu ra cuối cùng vẫn lấy từ pipeline CLAHE nhẹ.
    """
    _require_cv2()
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    mean_y = float(np.mean(y))
    clahe = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(8, 8))
    y_adj = clahe.apply(y)
    strength = _clamp((92.0 - mean_y) / 88.0, 0.35, 0.72)
    y_mix = cv2.addWeighted(y, 1.0 - strength, y_adj, strength, 0).astype(np.float32)
    # Gamma nhẹ (<1) làm sáng vùng tối
    y_out = np.clip(((y_mix / 255.0) ** 0.88) * 255.0, 0, 255).astype(np.uint8)
    ycrcb_out = cv2.merge([y_out, cr, cb])
    return cv2.cvtColor(ycrcb_out, cv2.COLOR_YCrCb2BGR)


def _refine_alpha_u8(alpha: np.ndarray) -> np.ndarray:
    """Lấp lỗ nhỏ trên mặt nạ, mép mềm hơn — giảm hiện rách / lỗ trống trên ảnh tối."""
    _require_cv2()
    if alpha.ndim != 2:
        alpha = alpha.squeeze()
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    x = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, k5, iterations=1)
    x = cv2.GaussianBlur(x, (3, 3), 0)
    return x


def _mask_subject_ok(alpha: np.ndarray) -> Tuple[bool, str]:
    """Heuristic: mặt nạ toàn nền / toàn chủ thể / trống giữa → không tin cậy."""
    h, w = alpha.shape[:2]
    if h < 8 or w < 8:
        return False, "Kích thước mặt nạ không hợp lệ."
    fg = float(np.mean(alpha > 128))
    if fg < 0.055:
        return False, "Diện tích chủ thể quá nhỏ sau tách nền."
    if fg > 0.93:
        return False, "Mặt nạ gần như phủ kín ảnh — tách nền không tin cậy."
    cy, cx = h // 2, w // 2
    ch, cw = max(4, h // 2), max(4, w // 2)
    y1, y2 = max(0, cy - ch // 2), min(h, cy + ch // 2)
    x1, x2 = max(0, cx - cw // 2), min(w, cx + cw // 2)
    center_mean = float(np.mean(alpha[y1:y2, x1:x2]))
    if center_mean < 52.0:
        return False, "Vùng trung tâm gần như trong suốt — tách nền có thể sai."
    return True, ""


def _enhance_luminance_y_channel(
    bgr: np.ndarray,
    force_skip: bool = False,
    *,
    dark_boost: bool = False,
) -> np.ndarray:
    """
    Chỉnh sáng: CLAHE + trộn kênh Y. `dark_boost`: ảnh/crop thiếu sáng — trộn mạnh hơn trước rembg.
    """
    if force_skip:
        return bgr
    _require_cv2()
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    mean_y = float(np.mean(y))

    # Crop đã sáng — không chỉnh
    if mean_y >= 100:
        return bgr

    if dark_boost:
        clahe = cv2.createCLAHE(clipLimit=1.38, tileGridSize=(8, 8))
        y_adj = clahe.apply(y)
        strength = _clamp((102.0 - mean_y) / 90.0, 0.0, 1.0) * 0.36
    else:
        clahe = cv2.createCLAHE(clipLimit=1.12, tileGridSize=(8, 8))
        y_adj = clahe.apply(y)
        strength = _clamp((98.0 - mean_y) / 95.0, 0.0, 1.0) * 0.17
    if strength < 0.008:
        return bgr
    w_orig = 1.0 - strength
    y_out = cv2.addWeighted(y, w_orig, y_adj, strength, 0).astype(np.uint8)
    ycrcb_out = cv2.merge([y_out, cr, cb])
    return cv2.cvtColor(ycrcb_out, cv2.COLOR_YCrCb2BGR)


def _face_center_h_check(face_xyxy: Tuple[int, int, int, int], img_w: int, tolerance: float = 0.12) -> Tuple[bool, str]:
    x1, y1, x2, y2 = face_xyxy
    face_cx = (x1 + x2) / 2.0
    img_cx = img_w / 2.0
    delta = abs(face_cx - img_cx) / img_w
    if delta <= tolerance:
        return True, "Khuôn mặt nằm gần trung tâm theo chiều ngang."
    return False, "Khuôn mặt lệch khỏi trung tâm theo chiều ngang."


def _face_area_ratio_check(face_xyxy: Tuple[int, int, int, int], img_w: int, img_h: int) -> Tuple[bool, str, float]:
    """
    Theo tiêu chuẩn ảnh thẻ thường gặp, có thể kiểm tra theo *chiều cao khuôn mặt* so với chiều cao ảnh.
    Yêu cầu bài toán: khuôn mặt chiếm khoảng 50–70% khung hình (diễn giải theo chiều cao).
    """
    x1, y1, x2, y2 = face_xyxy
    face_h = max(1, (y2 - y1))
    ratio = float(face_h / max(1, img_h))
    if 0.44 <= ratio <= 0.74:
        return True, f"Khuôn mặt chiếm ≈{ratio*100:.1f}% chiều cao ảnh (đạt).", ratio
    return False, f"Khuôn mặt chiếm ≈{ratio*100:.1f}% chiều cao ảnh (chưa đạt).", ratio


def _brightness_contrast_check(brightness: float, contrast: float) -> Tuple[bool, List[str]]:
    warnings: List[str] = []
    ok = True
    if brightness < 80:
        ok = False
        warnings.append("Ảnh quá tối (độ sáng thấp).")
    elif brightness > 225:
        # Studio nền trắng thường ~170–210 — không gắn nhãn “cháy” quá sớm
        ok = False
        warnings.append("Ảnh rất sáng / có thể bị cháy vùng sáng (độ sáng rất cao).")

    if contrast < 25:
        ok = False
        warnings.append("Ảnh bị mờ / tương phản thấp.")
    return ok, warnings


def _expand_face_bbox_for_portrait(
    face_xyxy: Tuple[int, int, int, int],
    img_w: int,
    img_h: int,
    *,
    pad_scale: float = 1.0,
) -> Tuple[int, int, int, int]:
    """
    Mở bbox mặt từ detector (thường bó sát) để crop gồm tóc mái, phần đầu và vai nhẹ.
    `pad_scale` > 1 khi ảnh tối — bbox detector thường lệch/bó; cần thêm biên.
    Chỉ dùng cho bước crop; các check nghiệp vụ vẫn dùng bbox gốc.
    """
    x1, y1, x2, y2 = face_xyxy
    fw = max(1, x2 - x1)
    fh = max(1, y2 - y1)
    ps = float(_clamp(pad_scale, 1.0, 1.28))
    pad_top = int(0.38 * fh * ps)
    pad_side = int(0.14 * fw * ps)
    pad_bottom = int(0.12 * fh * ps)
    nx1 = x1 - pad_side
    ny1 = y1 - pad_top
    nx2 = x2 + pad_side
    ny2 = y2 + pad_bottom
    nx1 = int(_clamp(nx1, 0, img_w - 1))
    ny1 = int(_clamp(ny1, 0, img_h - 1))
    nx2 = int(_clamp(nx2, nx1 + 1, img_w))
    ny2 = int(_clamp(ny2, ny1 + 1, img_h))
    return nx1, ny1, nx2, ny2


def _compute_crop_rect(
    img_w: int,
    img_h: int,
    face_xyxy: Tuple[int, int, int, int],
    aspect: float,
    target_face_height_frac: float = 0.50,
    headroom_frac: float = 0.28,
) -> Tuple[int, int, int, int]:
    """
    Compute crop rectangle (x1,y1,x2,y2) with desired aspect ratio,
    trying to keep the face centered and occupying target fraction of crop height.
    """
    fx1, fy1, fx2, fy2 = face_xyxy
    face_h = max(1, fy2 - fy1)
    face_cx = (fx1 + fx2) / 2.0
    face_cy = (fy1 + fy2) / 2.0

    crop_h = int(face_h / _clamp(target_face_height_frac, 0.45, 0.75))
    crop_w = int(crop_h * aspect)

    desired_top = int(face_cy - (0.5 - headroom_frac) * crop_h)
    desired_left = int(face_cx - crop_w / 2)

    x1 = desired_left
    y1 = desired_top
    x2 = x1 + crop_w
    y2 = y1 + crop_h

    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > img_w:
        dx = x2 - img_w
        x1 -= dx
        x2 = img_w
    if y2 > img_h:
        dy = y2 - img_h
        y1 -= dy
        y2 = img_h

    x1 = int(_clamp(x1, 0, img_w - 1))
    y1 = int(_clamp(y1, 0, img_h - 1))
    x2 = int(_clamp(x2, x1 + 1, img_w))
    y2 = int(_clamp(y2, y1 + 1, img_h))
    return x1, y1, x2, y2


def _safe_crop_with_pad(bgr: np.ndarray, rect: Tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = rect
    h, w = bgr.shape[:2]
    x1c, y1c, x2c, y2c = int(_clamp(x1, 0, w)), int(_clamp(y1, 0, h)), int(_clamp(x2, 0, w)), int(_clamp(y2, 0, h))
    crop = bgr[y1c:y2c, x1c:x2c].copy()
    if crop.size == 0:
        return bgr.copy()
    return crop


def _resize_to_standard(bgr: np.ndarray, ratio_name: str) -> np.ndarray:
    _require_cv2()
    if ratio_name == "3x4":
        out_w, out_h = 600, 800
    else:
        out_w, out_h = 600, 900
    return cv2.resize(bgr, (out_w, out_h), interpolation=cv2.INTER_CUBIC)


def _resize_cover_to_standard(bgr: np.ndarray, ratio_name: str) -> np.ndarray:
    """Scale toàn ảnh để phủ khung 3:4 hoặc 2:3, cắt giữa — không crop theo mặt."""
    _require_cv2()
    if ratio_name == "3x4":
        out_w, out_h = 600, 800
    else:
        out_w, out_h = 600, 900
    h, w = bgr.shape[:2]
    if w < 1 or h < 1:
        return _resize_to_standard(bgr, ratio_name)
    scale = max(out_w / float(w), out_h / float(h))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_CUBIC)
    x0 = max(0, (nw - out_w) // 2)
    y0 = max(0, (nh - out_h) // 2)
    patch = resized[y0 : y0 + out_h, x0 : x0 + out_w]
    if patch.shape[0] != out_h or patch.shape[1] != out_w:
        return cv2.resize(bgr, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
    return patch.copy()


def _remove_bg_and_compose_blue(
    pil_rgb: Image.Image,
    blue_rgb: Tuple[int, int, int],
    session: Any,
    high_key_photo: bool = False,
    low_light: bool = False,
    pil_segmentation: Image.Image | None = None,
) -> Tuple[Image.Image, bool]:
    """
    rembg → RGBA, ghép lên nền xanh. Luôn dùng màu từ `pil_rgb`; kênh alpha có thể suy từ ảnh boost
    (`pil_segmentation`) khi ảnh tối.

    Ảnh sáng: tắt matting (tránh tối da). Ảnh tối: tắt matting + boost đầu vào — alpha matting
    dễ làm rách/lỗ mặt nạ khi tương phản thấp.
    """
    if remove is None:
        raise RuntimeError("Thiếu thư viện `rembg` trong môi trường hiện tại.")
    inp_pil = pil_segmentation if pil_segmentation is not None else pil_rgb
    buf = io.BytesIO()
    inp_pil.save(buf, format="PNG")
    inp = buf.getvalue()

    if high_key_photo:
        # Alpha matting mạnh dễ làm tối vùng da/tóc trên ảnh studio sáng
        out = remove(inp, session=session, alpha_matting=False)
    elif low_light:
        # Ảnh tối: chỉ dùng mask u2net; matting thường phá hỏng mép
        out = remove(inp, session=session, alpha_matting=False)
    else:
        out = remove(
            inp,
            session=session,
            alpha_matting=True,
            alpha_matting_foreground_threshold=235,
            alpha_matting_background_threshold=10,
            alpha_matting_erode_size=4,
        )
    fg = Image.open(io.BytesIO(out)).convert("RGBA")
    if fg.size != pil_rgb.size:
        _resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.LANCZOS)
        fg = fg.resize(pil_rgb.size, _resample)

    r_src, g_src, b_src = pil_rgb.convert("RGB").split()
    _, _, _, a_raw = fg.split()
    a_np = np.array(a_raw)
    a_np = _refine_alpha_u8(a_np)
    mask_ok, _ = _mask_subject_ok(a_np)
    a = Image.fromarray(a_np, mode="L")
    fg2 = Image.merge("RGBA", (r_src, g_src, b_src, a))
    bg = Image.new("RGBA", fg2.size, blue_rgb + (255,))
    comp = Image.alpha_composite(bg, fg2).convert("RGB")
    return comp, mask_ok


def detect_faces_mediapipe(bgr: np.ndarray, min_confidence: float = 0.6) -> List[Tuple[int, int, int, int, float]]:
    """Trả về danh sách khuôn mặt (x1,y1,x2,y2,score) theo pixel (không tái sử dụng detector — dùng PortraitProcessor khi batch)."""
    return _detect_faces_with_detector(bgr, min_confidence=min_confidence, detector=None)


def _get_mediapipe_face_detector(min_confidence: float = 0.6) -> Any:
    """
    Trả về instance FaceDetection của MediaPipe (có thể tái sử dụng).
    Có fallback import path cho một số môi trường deploy.
    """
    if mp is None:
        raise RuntimeError("Thiếu thư viện `mediapipe` trong môi trường hiện tại.")
    try:
        mp_fd = mp.solutions.face_detection  # type: ignore[attr-defined]
    except Exception:
        try:
            from mediapipe.python.solutions import face_detection as mp_fd  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Không thể khởi tạo MediaPipe Face Detection. "
                "Vui lòng kiểm tra cài đặt thư viện `mediapipe` trong môi trường chạy."
            ) from e
    return mp_fd.FaceDetection(model_selection=1, min_detection_confidence=float(min_confidence))


def _try_get_mediapipe_face_detector(min_confidence: float = 0.6) -> Any | None:
    """Khởi tạo Face Detection nếu được; lỗi (Cloud/protobuf/path) → None, pipeline dùng Haar."""
    if mp is None:
        return None
    try:
        return _get_mediapipe_face_detector(min_confidence=min_confidence)
    except Exception:
        return None


def _try_get_selfie_segmentation() -> Any | None:
    """Segmentation người/nền (MediaPipe) — dùng tinh chỉnh crop tóc/vai khi thiếu sáng."""
    if mp is None:
        return None
    try:
        mp_seg = mp.solutions.selfie_segmentation  # type: ignore[attr-defined]
    except Exception:
        try:
            from mediapipe.python.solutions import selfie_segmentation as mp_seg  # type: ignore
        except Exception:
            return None
    try:
        return mp_seg.SelfieSegmentation(model_selection=1)
    except Exception:
        return None


def _refine_expanded_face_bbox_with_selfie(
    bgr: np.ndarray,
    face_xyxy: Tuple[int, int, int, int],
    expanded_xyxy: Tuple[int, int, int, int],
    selfie: Any,
) -> Tuple[int, int, int, int]:
    """
    Mở bbox đã pad theo mặt bằng mặt nạ người (tóc phía trên, vai/trang phục phía dưới).
    Chạy trên ảnh đã boost sáng để ổn định khi thiếu sáng.
    """
    _require_cv2()
    h, w = bgr.shape[:2]
    fx1, fy1, fx2, fy2 = face_xyxy
    ex1, ey1, ex2, ey2 = expanded_xyxy
    fw = max(1, fx2 - fx1)
    fh = max(1, fy2 - fy1)
    fcx = int((fx1 + fx2) / 2)
    fcy = int((fy1 + fy2) / 2)

    bgr_boost = _boost_bgr_for_face_detection(bgr)
    rgb = cv2.cvtColor(bgr_boost, cv2.COLOR_BGR2RGB)
    try:
        res = selfie.process(rgb)
    except Exception:
        return expanded_xyxy
    if res is None or getattr(res, "segmentation_mask", None) is None:
        return expanded_xyxy

    m = np.asarray(res.segmentation_mask, dtype=np.float32)
    if m.ndim != 2:
        return expanded_xyxy
    mh, mw = m.shape[:2]
    if mh != h or mw != w:
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)

    # Ngưỡng thấp hơn một chút khi ảnh tối — vẫn lọc nhiễu bằng morphology
    binm = (m > 0.38).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binm = cv2.morphologyEx(binm, cv2.MORPH_CLOSE, kernel, iterations=1)
    binm = cv2.dilate(binm, kernel, iterations=1)

    # Cửa sổ chân dung quanh mặt (không lấy cả chân)
    win_y1 = max(0, fy1 - int(1.2 * fh))
    win_y2 = min(h, fy2 + int(2.45 * fh))
    win_x1 = max(0, fcx - int(1.55 * fw))
    win_x2 = min(w, fcx + int(1.55 * fw))
    win_mask = np.zeros((h, w), dtype=np.uint8)
    win_mask[win_y1:win_y2, win_x1:win_x2] = 255
    masked = cv2.bitwise_and(binm, win_mask)

    num, labels, stats, _ = cv2.connectedComponentsWithStats((masked > 127).astype(np.uint8), connectivity=8)
    if num <= 1:
        return expanded_xyxy

    fcx = int(_clamp(float(fcx), 0.0, float(w - 1)))
    fcy = int(_clamp(float(fcy), 0.0, float(h - 1)))
    lbl = int(labels[fcy, fcx])
    if lbl == 0:
        # Điểm giữa mặt đôi khi rơi nền — thử lệch nhẹ trong bbox mặt
        found = False
        for dy in (0, int(-0.12 * fh), int(0.08 * fh)):
            for dx in (0, int(-0.1 * fw), int(0.1 * fw)):
                cy = int(_clamp(float(fcy + dy), 0.0, float(h - 1)))
                cx = int(_clamp(float(fcx + dx), 0.0, float(w - 1)))
                t = int(labels[cy, cx])
                if t != 0:
                    lbl = t
                    found = True
                    break
            if found:
                break
    if lbl == 0:
        return expanded_xyxy

    x_, y_, cw_, ch_, area = stats[lbl]
    if area < max(200, int(0.012 * w * h)):
        return expanded_xyxy

    px1, py1, px2, py2 = int(x_), int(y_), int(x_ + cw_), int(y_ + ch_)

    new_x1 = min(ex1, px1)
    new_y1 = min(ey1, py1)
    new_x2 = max(ex2, px2)
    new_y2 = max(ey2, py2)

    # Giới hạn mở ngang (tránh kéo full khung khi nền lẫn)
    max_span = int(2.7 * fw)
    if new_x2 - new_x1 > max_span:
        cx = (fx1 + fx2) // 2
        new_x1 = max(0, cx - max_span // 2)
        new_x2 = min(w, cx + max_span // 2)

    # Giới hạn dưới: không kéo quá xa so với đáy mặt (tránh lấy hết thân)
    bottom_cap = min(h, fy2 + int(2.65 * fh))
    if new_y2 > bottom_cap:
        new_y2 = bottom_cap

    new_x1 = int(_clamp(new_x1, 0, w - 1))
    new_y1 = int(_clamp(new_y1, 0, h - 1))
    new_x2 = int(_clamp(new_x2, new_x1 + 1, w))
    new_y2 = int(_clamp(new_y2, new_y1 + 1, h))

    return (new_x1, new_y1, new_x2, new_y2)


class PortraitProcessor:
    """
    Bộ xử lý ảnh chân dung có cache tài nguyên nặng (MediaPipe detector, rembg session).

    Dùng class này cho batch processing để nhanh hơn nhiều so với tạo mới cho mỗi ảnh.
    """

    def __init__(
        self,
        ratio: str = "3x4",
        blue_rgb: Tuple[int, int, int] = (0, 91, 196),
        min_face_conf: float = 0.6,
        rembg_model: str = "u2net",
    ) -> None:
        self.ratio = ratio
        self.blue_rgb = blue_rgb
        self.min_face_conf = float(min_face_conf)
        _require_cv2()
        self._fd = _try_get_mediapipe_face_detector(min_confidence=self.min_face_conf)
        self._selfie = _try_get_selfie_segmentation()
        self._rembg_session = None
        if new_session is not None:
            try:
                self._rembg_session = new_session(rembg_model)
            except Exception:
                self._rembg_session = None

    def process(self, pil_img: Image.Image, *, prefer_face_crop: bool = False) -> ProcessResult:
        return process_portrait_image(
            pil_img,
            ratio=self.ratio,
            blue_rgb=self.blue_rgb,
            min_face_conf=self.min_face_conf,
            prefer_face_crop=prefer_face_crop,
            _mp_face_detector=self._fd,
            _rembg_session=self._rembg_session,
            _selfie_segmentation=self._selfie,
        )


def process_portrait_image(
    pil_img: Image.Image,
    ratio: str = "3x4",
    blue_rgb: Tuple[int, int, int] = (0, 91, 196),
    min_face_conf: float = 0.6,
    prefer_face_crop: bool = False,
    _mp_face_detector: Any | None = None,
    _rembg_session: Any | None = None,
    _selfie_segmentation: Any | None = None,
) -> ProcessResult:
    """
    Pipeline backend:
    - Phát hiện khuôn mặt (đúng 1 mặt)
    - Validation: vị trí, tỷ lệ, sáng/tương phản, nền
    - Crop theo mặt chỉ khi: prefer_face_crop hoặc nền không đơn sắc; không thì scale toàn ảnh (cover).
    - Ảnh thiếu sáng: có thể tinh chỉnh bbox crop bằng MediaPipe Selfie Segmentation (tóc/vai).
    - Chỉnh sáng nhẹ (CLAHE + trộn Y) khi cần
    - Thay nền: rembg + nền xanh
    """
    errors: List[str] = []
    warnings: List[str] = []
    checks: Dict[str, CheckResult] = {}

    try:
        _require_cv2()
    except RuntimeError as e:
        errors.append(str(e))
        checks["Thư viện OpenCV"] = CheckResult(False, str(e))
        return ProcessResult(status="FAILED", errors=errors, warnings=warnings, checks=checks, processed_image=None)

    bgr0 = _pil_to_bgr(pil_img)
    h0, w0 = bgr0.shape[:2]

    brightness, contrast = _compute_brightness_contrast(bgr0)
    use_det_boost = brightness < 106 or contrast < 33
    bgr_for_faces = _boost_bgr_for_face_detection(bgr0) if use_det_boost else bgr0
    faces = _detect_faces_with_detector(bgr_for_faces, min_confidence=min_face_conf, detector=_mp_face_detector)
    if len(faces) == 0 and not use_det_boost:
        faces = _detect_faces_with_detector(
            _boost_bgr_for_face_detection(bgr0), min_confidence=min_face_conf, detector=_mp_face_detector
        )
    if len(faces) == 0:
        errors.append("Không tìm thấy khuôn mặt.")
        checks["Khuôn mặt"] = CheckResult(False, "Không phát hiện được khuôn mặt.")
        return ProcessResult(status="FAILED", errors=errors, warnings=warnings, checks=checks, processed_image=None)
    if len(faces) > 1:
        errors.append("Có nhiều hơn 1 khuôn mặt trong ảnh.")
        checks["Khuôn mặt"] = CheckResult(False, f"Phát hiện {len(faces)} khuôn mặt.")
        return ProcessResult(status="FAILED", errors=errors, warnings=warnings, checks=checks, processed_image=None)

    fx1, fy1, fx2, fy2, _score = faces[0]
    checks["Khuôn mặt"] = CheckResult(True, "Phát hiện đúng 1 khuôn mặt.")

    ok_center, msg_center = _face_center_h_check((fx1, fy1, fx2, fy2), w0)
    checks["Vị trí (giữa khung)"] = CheckResult(ok_center, msg_center)
    if not ok_center:
        warnings.append(msg_center)

    ok_ratio, msg_ratio, _ = _face_area_ratio_check((fx1, fy1, fx2, fy2), w0, h0)
    checks["Tỷ lệ khuôn mặt"] = CheckResult(ok_ratio, msg_ratio)
    if not ok_ratio:
        warnings.append(msg_ratio)

    ok_bc, bc_warns = _brightness_contrast_check(brightness, contrast)
    checks["Ánh sáng & Tương phản"] = CheckResult(
        ok_bc, f"Độ sáng ~{brightness:.0f}, tương phản ~{contrast:.0f}."
    )
    warnings.extend(bc_warns)

    ok_bg, msg_bg = _background_uniformity_check(bgr0)
    checks["Nền đơn sắc"] = CheckResult(ok_bg, msg_bg)
    if not ok_bg:
        warnings.append(msg_bg)

    should_face_crop = bool(prefer_face_crop) or (not ok_bg)
    # Ảnh thiếu sáng: bbox mặt có thể nhỏ/lệch → thêm padding + headroom trên crop
    pad_scale = 1.0 + (1.0 - min(brightness, 108.0) / 108.0) * 0.17
    if brightness >= 108:
        pad_scale = 1.0
    headroom_crop = 0.28 if brightness >= 100 else 0.32

    if should_face_crop:
        checks["Khung ảnh"] = CheckResult(
            True,
            "Cắt theo khuôn mặt (bạn bật hoặc nền không đơn sắc).",
        )
        aspect = 3 / 4 if ratio == "3x4" else 2 / 3
        crop_face = _expand_face_bbox_for_portrait(
            (fx1, fy1, fx2, fy2), w0, h0, pad_scale=pad_scale
        )
        if _selfie_segmentation is not None and (
            brightness < 114 or contrast < 36 or not ok_bc
        ):
            crop_face = _refine_expanded_face_bbox_with_selfie(
                bgr0,
                (fx1, fy1, fx2, fy2),
                crop_face,
                _selfie_segmentation,
            )
        crop_rect = _compute_crop_rect(
            w0, h0, crop_face, aspect=aspect, headroom_frac=headroom_crop
        )
        cropped = _safe_crop_with_pad(bgr0, crop_rect)
    else:
        checks["Khung ảnh"] = CheckResult(
            True,
            "Giữ khung gốc — chỉ scale về khung chuẩn (nền đơn sắc, không bật cắt mặt).",
        )
        cropped = bgr0.copy()

    br_crop, _ = _compute_brightness_contrast(cropped)
    # Ảnh gốc hoặc crop đã sáng → không CLAHE (tránh lệch tông trước rembg)
    skip_luma = brightness >= 118 or br_crop >= 102
    cropped_eq = _enhance_luminance_y_channel(
        cropped,
        force_skip=skip_luma,
        dark_boost=(not skip_luma) and (brightness < 108 or br_crop < 88),
    )

    if should_face_crop:
        out_bgr = _resize_to_standard(cropped_eq, ratio_name=ratio)
    else:
        out_bgr = _resize_cover_to_standard(cropped_eq, ratio_name=ratio)
    out_pil = _bgr_to_pil(out_bgr).convert("RGB")

    # Studio / nền trắng: rembg matting dễ tối màu → tắt matting, giữ màu gần gốc hơn
    high_key = brightness >= 128 or br_crop >= 108
    # Ảnh tối / tương phản thấp: matting + rembg trên ảnh chưa boost dễ rách mặt nạ → boost đầu vào + tắt matting
    low_light = (not high_key) and (brightness < 108 or br_crop < 88 or not ok_bc)
    pil_seg = _bgr_to_pil(_boost_bgr_for_segmentation(out_bgr)) if low_light else None

    try:
        if new_session is None or remove is None:
            raise RuntimeError("Thiếu rembg")
        session = _rembg_session if _rembg_session is not None else new_session("u2net")
        out_pil, mask_ok = _remove_bg_and_compose_blue(
            out_pil,
            blue_rgb=blue_rgb,
            session=session,
            high_key_photo=high_key,
            low_light=low_light,
            pil_segmentation=pil_seg,
        )
        if high_key:
            msg_rembg = "Đã thay nền xanh (ảnh sáng: tách nền không matting để giữ màu)."
        elif low_light:
            msg_rembg = "Đã thay nền xanh (ảnh tối: tách nền với boost + không matting)."
        else:
            msg_rembg = "Đã thay nền xanh chuẩn."
        if not mask_ok:
            msg_rembg = "Tách nền không đủ tin cậy — nên chụp lại ảnh sáng, tương phản hơn."
            warnings.append(msg_rembg)
        checks["Thay nền xanh"] = CheckResult(mask_ok, msg_rembg)
    except Exception:
        warnings.append("Không thể xóa nền tự động (rembg). Ảnh vẫn được scale/cân bằng sáng.")
        checks["Thay nền xanh"] = CheckResult(False, "Thiếu/lỗi rembg, bỏ qua thay nền.")

    return ProcessResult(status="OK", errors=errors, warnings=warnings, checks=checks, processed_image=out_pil)


def pil_to_jpeg_bytes(pil_img: Image.Image, quality: int = 95) -> bytes:
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=int(_clamp(quality, 60, 100)), optimize=True)
    return buf.getvalue()

def _require_cv2() -> None:
    if cv2 is None:
        detail = f" Chi tiết: {_CV2_IMPORT_ERROR}" if _CV2_IMPORT_ERROR else ""
        raise RuntimeError("Thiếu OpenCV (`cv2`). Hãy cài `opencv-python-headless` rồi chạy lại." + detail)


def _detect_faces_with_detector(
    bgr: np.ndarray,
    min_confidence: float,
    detector: Any | None,
) -> List[Tuple[int, int, int, int, float]]:
    """
    Nếu có detector (FaceDetection) thì dùng lại; nếu không thì tạo tạm (chậm hơn).
    """
    _require_cv2()
    h, w = bgr.shape[:2]
    fd = detector
    if fd is None:
        fd = _try_get_mediapipe_face_detector(min_confidence=min_confidence)
    if fd is not None:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = fd.process(rgb)

        faces: List[Tuple[int, int, int, int, float]] = []
        if not getattr(res, "detections", None):
            return faces

        for det in res.detections:
            score = float(det.score[0]) if det.score else 0.0
            box = det.location_data.relative_bounding_box
            x1 = int(box.xmin * w)
            y1 = int(box.ymin * h)
            bw = int(box.width * w)
            bh = int(box.height * h)
            x2 = x1 + bw
            y2 = y1 + bh
            x1 = int(_clamp(x1, 0, w - 1))
            y1 = int(_clamp(y1, 0, h - 1))
            x2 = int(_clamp(x2, x1 + 1, w))
            y2 = int(_clamp(y2, y1 + 1, h))
            faces.append((x1, y1, x2, y2, score))

        faces.sort(key=lambda f: f[4], reverse=True)
        return _dedupe_face_detections(faces)

    # Fallback: OpenCV Haar Cascade (MediaPipe không khởi tạo được hoặc không cài)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    dets = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    faces2: List[Tuple[int, int, int, int, float]] = []
    for (x, y, ww, hh) in dets:
        x1 = int(_clamp(x, 0, w - 1))
        y1 = int(_clamp(y, 0, h - 1))
        x2 = int(_clamp(x + ww, x1 + 1, w))
        y2 = int(_clamp(y + hh, y1 + 1, h))
        faces2.append((x1, y1, x2, y2, 0.50))  # heuristic score
    faces2.sort(key=lambda f: _box_area_xyxy(f[:4]), reverse=True)
    return _dedupe_face_detections(faces2)
