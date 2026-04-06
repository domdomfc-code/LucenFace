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


def _enhance_luminance_y_channel(bgr: np.ndarray, force_skip: bool = False) -> np.ndarray:
    """
    Chỉnh sáng rất nhẹ: CLAHE clip thấp + tối đa ~17% trộn với Y gốc.
    Ảnh đã sáng (studio, nền trắng) — force_skip để không làm tối/lệch tông.
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
) -> Tuple[int, int, int, int]:
    """
    Mở bbox mặt từ detector (thường bó sát) để crop gồm tóc mái, phần đầu và vai nhẹ.
    Chỉ dùng cho bước crop; các check nghiệp vụ vẫn dùng bbox gốc.
    """
    x1, y1, x2, y2 = face_xyxy
    fw = max(1, x2 - x1)
    fh = max(1, y2 - y1)
    pad_top = int(0.38 * fh)
    pad_side = int(0.14 * fw)
    pad_bottom = int(0.12 * fh)
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


def _remove_bg_and_compose_blue(
    pil_rgb: Image.Image,
    blue_rgb: Tuple[int, int, int],
    session: Any,
    high_key_photo: bool = False,
) -> Image.Image:
    """rembg → RGBA, ghép lên nền xanh. Ảnh sáng/high-key: matting nhẹ hoặc tắt để tránh tối màu & mép cứng."""
    if remove is None:
        raise RuntimeError("Thiếu thư viện `rembg` trong môi trường hiện tại.")
    buf = io.BytesIO()
    pil_rgb.save(buf, format="PNG")
    inp = buf.getvalue()

    if high_key_photo:
        # Alpha matting mạnh dễ làm tối vùng da/tóc trên ảnh studio sáng
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
    bg = Image.new("RGBA", fg.size, blue_rgb + (255,))
    comp = Image.alpha_composite(bg, fg).convert("RGB")
    return comp


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
        self._rembg_session = None
        if new_session is not None:
            try:
                self._rembg_session = new_session(rembg_model)
            except Exception:
                self._rembg_session = None

    def process(self, pil_img: Image.Image) -> ProcessResult:
        return process_portrait_image(
            pil_img,
            ratio=self.ratio,
            blue_rgb=self.blue_rgb,
            min_face_conf=self.min_face_conf,
            _mp_face_detector=self._fd,
            _rembg_session=self._rembg_session,
        )


def process_portrait_image(
    pil_img: Image.Image,
    ratio: str = "3x4",
    blue_rgb: Tuple[int, int, int] = (0, 91, 196),
    min_face_conf: float = 0.6,
    _mp_face_detector: Any | None = None,
    _rembg_session: Any | None = None,
) -> ProcessResult:
    """
    Pipeline backend:
    - Phát hiện khuôn mặt (đúng 1 mặt)
    - Validation: vị trí, tỷ lệ, sáng/tương phản, nền
    - Auto-fix: crop tỷ lệ, chỉnh sáng nhẹ (CLAHE + trộn Y, không equalize toàn cục)
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

    faces = _detect_faces_with_detector(bgr0, min_confidence=min_face_conf, detector=_mp_face_detector)
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

    brightness, contrast = _compute_brightness_contrast(bgr0)
    ok_bc, bc_warns = _brightness_contrast_check(brightness, contrast)
    checks["Ánh sáng & Tương phản"] = CheckResult(
        ok_bc, f"Độ sáng ~{brightness:.0f}, tương phản ~{contrast:.0f}."
    )
    warnings.extend(bc_warns)

    ok_bg, msg_bg = _background_uniformity_check(bgr0)
    checks["Nền đơn sắc"] = CheckResult(ok_bg, msg_bg)
    if not ok_bg:
        warnings.append(msg_bg)

    aspect = 3 / 4 if ratio == "3x4" else 2 / 3
    crop_face = _expand_face_bbox_for_portrait((fx1, fy1, fx2, fy2), w0, h0)
    crop_rect = _compute_crop_rect(w0, h0, crop_face, aspect=aspect)
    cropped = _safe_crop_with_pad(bgr0, crop_rect)

    br_crop, _ = _compute_brightness_contrast(cropped)
    # Ảnh gốc hoặc crop đã sáng → không CLAHE (tránh lệch tông trước rembg)
    skip_luma = brightness >= 118 or br_crop >= 102
    cropped_eq = _enhance_luminance_y_channel(cropped, force_skip=skip_luma)

    out_bgr = _resize_to_standard(cropped_eq, ratio_name=ratio)
    out_pil = _bgr_to_pil(out_bgr).convert("RGB")

    # Studio / nền trắng: rembg matting dễ tối màu → tắt matting, giữ màu gần gốc hơn
    high_key = brightness >= 128 or br_crop >= 108

    try:
        if new_session is None or remove is None:
            raise RuntimeError("Thiếu rembg")
        session = _rembg_session if _rembg_session is not None else new_session("u2net")
        out_pil = _remove_bg_and_compose_blue(
            out_pil,
            blue_rgb=blue_rgb,
            session=session,
            high_key_photo=high_key,
        )
        msg_rembg = (
            "Đã thay nền xanh (ảnh sáng: tách nền không matting để giữ màu)."
            if high_key
            else "Đã thay nền xanh chuẩn."
        )
        checks["Thay nền xanh"] = CheckResult(True, msg_rembg)
    except Exception:
        warnings.append("Không thể xóa nền tự động (rembg). Ảnh vẫn được crop/cân bằng sáng.")
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
        return faces

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
    return faces2
