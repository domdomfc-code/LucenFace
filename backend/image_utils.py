from __future__ import annotations

import io
import math
import os
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
from PIL import Image, ImageOps

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


class CheckResult:
    """Kết quả một mục checklist (không dùng @dataclass — tránh lỗi import trên Python 3.12/Streamlit)."""

    __slots__ = ("ok", "message")

    def __init__(self, ok: bool, message: str) -> None:
        self.ok = ok
        self.message = message


class ProcessResult:
    """Kết quả pipeline (không dùng @dataclass — tránh lỗi import trên Python 3.12/Streamlit)."""

    __slots__ = ("status", "errors", "warnings", "checks", "processed_image")

    def __init__(
        self,
        status: str,
        errors: List[str],
        warnings: List[str],
        checks: Dict[str, CheckResult],
        processed_image: Optional[Image.Image],
    ) -> None:
        self.status = status
        self.errors = errors
        self.warnings = warnings
        self.checks = checks
        self.processed_image = processed_image


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


def _pil_apply_exif_transpose(pil_img: Image.Image) -> Image.Image:
    """Áp dụng tag Orientation trong EXIF (ảnh từ điện thoại thường bị xoay/lật trong pixel)."""
    try:
        out = ImageOps.exif_transpose(pil_img)
        return out if out is not None else pil_img
    except Exception:
        return pil_img


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
    iou_threshold: float = 0.43,
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


def _filter_spurious_secondary_faces(
    faces: List[Tuple[int, int, int, int, float]],
    img_w: int,
    img_h: int,
) -> List[Tuple[int, int, int, int, float]]:
    """
    Ảnh thẻ 1 người: detector đôi khi trả thêm bbox nhỏ (cà vạt, ve áo, nền) hoặc hai box
    chồng nhẹ cho cùng một mặt. Giữ bbox chính (điểm + diện tích lớn nhất).
    Hai người thật: thường hai box tương đương, xa nhau — không lọc.
    """
    if len(faces) <= 1:
        return faces
    faces = sorted(faces, key=lambda f: (f[4], _box_area_xyxy(f[:4])), reverse=True)
    primary = faces[0]
    px1, py1, px2, py2 = primary[:4]
    pa = _box_area_xyxy(primary[:4])
    if pa <= 0:
        return faces
    diag = max(1.0, math.hypot(float(px2 - px1), float(py2 - py1)))
    pcx = (px1 + px2) / 2.0
    pcy = (py1 + py2) / 2.0
    img_area = float(max(1, img_w * img_h))

    kept: List[Tuple[int, int, int, int, float]] = [primary]
    for f in faces[1:]:
        fx1, fy1, fx2, fy2 = f[:4]
        fa = _box_area_xyxy(f[:4])
        fcx = (fx1 + fx2) / 2.0
        fcy = (fy1 + fy2) / 2.0
        iou = _iou_xyxy(f[:4], primary[:4])
        dist = math.hypot(fcx - pcx, fcy - pcy)

        # Trùng / gần trùng cùng một mặt
        if iou >= 0.09:
            continue
        # Box phụ rất nhỏ trên ảnh hoặc nhỏ hơn nhiều so với mặt chính, gần tâm mặt
        if fa < 0.012 * img_area and (fa < 0.48 * pa or iou > 0.035 or dist < 0.52 * diag):
            continue
        # Hai box chồng nhẹ: box nhỏ hơn rõ, điểm không hơn mặt chính
        if fa < 0.44 * pa and dist < 0.65 * diag and f[4] <= primary[4] * 0.96:
            continue
        # Nhiễu cực nhỏ so với mặt chính (dù xa tâm một chút)
        if fa < 0.20 * pa and f[4] < primary[4] * 0.90:
            continue

        kept.append(f)

    return kept


def _pick_primary_portrait_face(
    faces: List[Tuple[int, int, int, int, float]],
    img_w: int,
    img_h: int,
) -> List[Tuple[int, int, int, int, float]]:
    """
    Chỉ giữ một mặt: selfie/chân dung thường 1 người; nền (poster, ảnh treo, kệ) tạo thêm bbox.
    Chọn mặt lớn + gần trung tâm khung + điểm detector cao.
    Bỏ bbox quá nhỏ (<~5.5% chiều cao) nếu còn ứng viên khác — tránh texture/tóc nhận nhầm là mặt.
    """
    if len(faces) <= 1:
        return faces
    img_area = float(max(1, img_w * img_h))
    img_h_f = float(max(1, img_h))
    min_h_frac = 0.055
    eligible = [f for f in faces if (f[3] - f[1]) / img_h_f >= min_h_frac]
    if not eligible:
        eligible = faces
    icx = img_w / 2.0
    icy = img_h / 2.0
    best_f = eligible[0]
    best_c = -1.0
    for f in eligible:
        x1, y1, x2, y2, sc = f
        fa = _box_area_xyxy((x1, y1, x2, y2))
        if fa <= 0:
            continue
        fh = (y2 - y1) / img_h_f
        fcx = (x1 + x2) / 2.0
        fcy = (y1 + y2) / 2.0
        dx = (fcx - icx) / max(1.0, float(img_w))
        dy = (fcy - icy) / max(1.0, float(img_h))
        dist = math.hypot(dx, dy)
        center = math.exp(-3.0 * dist * dist)
        size_norm = math.sqrt(fa) / math.sqrt(max(0.008 * img_area, 1.0))
        size_norm = min(2.8, max(0.35, size_norm))
        combined = float(sc) * size_norm * center
        if fh < 0.10:
            combined *= 0.42
        elif fh < 0.14:
            combined *= 0.78
        if combined > best_c:
            best_c = combined
            best_f = f
    return [best_f]


def _dedupe_face_detections(
    faces: List[Tuple[int, int, int, int, float]],
    img_w: int,
    img_h: int,
) -> Tuple[List[Tuple[int, int, int, int, float]], int]:
    """
    Chuỗi lọc: box lồng nhau → NMS IoU → bỏ bbox phụ → chọn 1 mặt chủ thể.
    Trả về (danh sách mặt, số mặt trước bước chọn chủ thể) để cảnh báo UI.
    """
    if len(faces) == 0:
        return [], 0
    if len(faces) == 1:
        return faces, 1
    faces = _suppress_contained_duplicates(faces)
    faces = _nms_face_boxes(faces, iou_threshold=0.43)
    faces = _filter_spurious_secondary_faces(faces, img_w, img_h)
    n_before = len(faces)
    faces = _pick_primary_portrait_face(faces, img_w, img_h)
    return faces, n_before


def _compute_brightness_contrast(bgr: np.ndarray) -> Tuple[float, float]:
    _require_cv2()
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    return brightness, contrast


def _mean_std_rgb(pixels_flat_bgr: np.ndarray) -> float:
    """Trung bình σ của 3 kênh BGR trên mảng (N, 3)."""
    if pixels_flat_bgr.size < 9:
        return 999.0
    return float(pixels_flat_bgr.astype(np.float32).std(axis=0).mean())


def _lab_ab_std_mean(bgr_region: np.ndarray) -> float:
    """
    Độ lệch trên kênh a,b (Lab) — ít nhạy với gradient sáng/tối trên nền phẳng một màu
    so với chỉ dùng RGB trên toàn viền (hay lẫn tóc/vai).
    """
    if bgr_region.size < 9:
        return 999.0
    _require_cv2()
    lab = cv2.cvtColor(bgr_region, cv2.COLOR_BGR2LAB)
    a = lab[..., 1].astype(np.float32).ravel()
    b_ = lab[..., 2].astype(np.float32).ravel()
    return float(0.5 * (np.std(a) + np.std(b_)))


def _background_uniformity_check(
    bgr: np.ndarray,
    border_pct: float = 0.08,
    *,
    standard_wording: bool = False,
) -> Tuple[bool, str]:
    """
    Heuristic: phông một màu — không tin viền đủ 4 cạnh (hay lẫn chủ thể: tóc, vai, cổ áo).

    Dùng thêm: bốn góc + dải phía trên giữa (thường chỉ nền), và σ a,b trong Lab.
    Giữ `σ toàn viền` trong thông báo để debug.
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
    std_border_full = _mean_std_rgb(border)

    # Góc: vùng nhỏ ít bị người che (ảnh thẻ chân dung)
    m = min(w, h)
    cw = max(2, int(m * 0.10))
    ch = max(2, int(m * 0.10))
    corner_patches = (
        bgr[0:ch, 0:cw, :],
        bgr[0:ch, w - cw : w, :],
        bgr[h - ch : h, 0:cw, :],
        bgr[h - ch : h, w - cw : w, :],
    )
    corner_flat = np.concatenate([p.reshape(-1, 3) for p in corner_patches], axis=0)
    std_corner_rgb = _mean_std_rgb(corner_flat)
    std_corner_ab = _lab_ab_std_mean(np.concatenate([p for p in corner_patches], axis=0))

    # Dải trên giữa (phía trên đầu — thường chỉ phông)
    tw = max(cw, int(w * 0.42))
    th = max(2, int(h * 0.06))
    x0 = max(0, (w - tw) // 2)
    x1 = min(w, x0 + tw)
    top_mid = bgr[0:th, x0:x1, :]
    std_top_rgb = _mean_std_rgb(top_mid.reshape(-1, 3))
    std_top_ab = _lab_ab_std_mean(top_mid)

    std_rgb_clean = min(std_corner_rgb, std_top_rgb)
    std_ab_clean = min(std_corner_ab, std_top_ab)

    # Ngưỡng: RGB trên vùng “sạch”; Lab a,b bắt nền một sắc độ khi có gradient sáng
    rgb_ok = 20.0
    rgb_soft = 34.0
    ab_ok = 12.0
    ok = (std_rgb_clean < rgb_ok) or (std_rgb_clean < rgb_soft and std_ab_clean < ab_ok)

    detail = (
        f"σ RGB (góc/trên giữa) ~{std_rgb_clean:.1f}, σ Lab(a,b) ~{std_ab_clean:.1f}; "
        f"viền đủ cạnh ~{std_border_full:.1f}"
    )

    if ok:
        if standard_wording:
            return True, f"Đạt tiêu chuẩn — phông gần một màu ({detail})."
        return True, f"Nền tương đối đơn sắc ({detail})."
    if standard_wording:
        return False, f"Chưa đạt — phông không đồng nhất ({detail}); có thể ghép nền xanh nếu cần."
    return False, f"Nền có thể không đơn sắc ({detail})."


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


def _refine_alpha_u8(
    alpha: np.ndarray,
    *,
    strong: bool = False,
    blur_edges: bool = True,
) -> np.ndarray:
    """
    Lấp lỗ nhỏ trên mặt nạ. Gaussian trên alpha dễ làm **viền mờ rộng / halo** với ISNet
    hoặc ảnh studio sáng — tắt bằng `blur_edges=False`.
    """
    _require_cv2()
    if alpha.ndim != 2:
        alpha = alpha.squeeze()
    ksz = 7 if strong else 5
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
    x = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, k, iterations=1)
    if blur_edges:
        x = cv2.GaussianBlur(x, (3, 3), 0)
    return x


def _rembg_model_skip_pymatting(model: str) -> bool:
    """
    Chỉ `u2net` gốc hưởng lợi rõ từ pymatting (trimap). ISNet / u2net_human_seg / silueta …
    đã ra alpha mềm — thêm matting dễ **đôi mép, halo trắng** khi ghép nền xanh.
    """
    m = (model or "u2net").lower().strip()
    return m != "u2net"


def _decontaminate_straight_alpha_rgb(
    r: np.ndarray,
    g: np.ndarray,
    b: np.ndarray,
    a: np.ndarray,
    bg_rgb: Tuple[int, int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    rembg trả RGBA straight alpha: mép thường nhiễm màu nền (xanh/viền halo).
    Ước lượng màu foreground thật: Cf = (C - Cbg*(1-a)) / max(a, eps).
    Chỉ áp vùng alpha > 12 để tránh nổ số.
    """
    br, bg_c, bb = float(bg_rgb[0]), float(bg_rgb[1]), float(bg_rgb[2])
    rf = r.astype(np.float32)
    gf = g.astype(np.float32)
    bf_c = b.astype(np.float32)
    af = np.maximum(a.astype(np.float32) / 255.0, 0.07)
    r2 = (rf - br * (1.0 - af)) / af
    g2 = (gf - bg_c * (1.0 - af)) / af
    b2 = (bf_c - bb * (1.0 - af)) / af
    r2 = np.clip(r2, 0, 255)
    g2 = np.clip(g2, 0, 255)
    b2 = np.clip(b2, 0, 255)
    safe = a > 12
    r_out = np.where(safe, r2, rf).astype(np.uint8)
    g_out = np.where(safe, g2, gf).astype(np.uint8)
    b_out = np.where(safe, b2, bf_c).astype(np.uint8)
    return r_out, g_out, b_out


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


def _merge_rembg_alpha_with_selfie(
    alpha_u8: np.ndarray,
    pil_rgb: Image.Image,
    selfie: Any,
) -> np.ndarray:
    """
    rembg trên ảnh tối hay mất tóc/áo (tương phản nền thấp). Lấy max(alpha_rembg, alpha_selfie)
    trong vùng chân dung (trên ~82% khung, dải giữa) để giữ silhouette người.
    """
    _require_cv2()
    bgr = _pil_to_bgr(pil_rgb)
    bgr_in = _boost_bgr_for_segmentation(bgr)
    rgb = cv2.cvtColor(bgr_in, cv2.COLOR_BGR2RGB)
    try:
        res = selfie.process(rgb)
    except Exception:
        return alpha_u8
    if res is None or getattr(res, "segmentation_mask", None) is None:
        return alpha_u8

    m = np.asarray(res.segmentation_mask, dtype=np.float32)
    h, w = alpha_u8.shape[:2]
    if m.shape[0] != h or m.shape[1] != w:
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)

    hh, ww = h, w
    y_lim = min(hh - 1, int(hh * 0.82))
    xc = ww // 2
    bw = max(8, int(ww * 0.76))
    x1 = max(0, xc - bw // 2)
    x2 = min(ww, xc + bw // 2)
    spatial = np.zeros((hh, ww), dtype=np.float32)
    spatial[:y_lim, x1:x2] = 1.0
    bf = min(int(hh * 0.11), hh - y_lim)
    for i in range(bf):
        y = y_lim + i
        if y >= hh:
            break
        wgt = 1.0 - (i + 1) / float(bf + 1)
        spatial[y, x1:x2] = wgt

    # Làm mềm ngưỡng để không gắn cứng 0.5
    person = np.clip((m - 0.35) / 0.55, 0.0, 1.0)
    person = person * spatial
    boost = (person * 255.0).astype(np.float32)
    a_out = np.maximum(alpha_u8.astype(np.float32), boost * 0.91)
    a_out = np.clip(a_out, 0, 255).astype(np.uint8)
    a_out = cv2.GaussianBlur(a_out, (3, 3), 0)
    return _refine_alpha_u8(a_out, strong=True, blur_edges=False)


def _remove_bg_via_remove_bg_api(pil_rgb: Image.Image, api_key: str) -> Image.Image:
    """
    Dịch vụ remove.bg (https://www.remove.bg/api) — chất lượng tách nền gần trang upload,
    cần API key (Streamlit Secrets / biến môi trường REMOVEBG_API_KEY).
    """
    try:
        import requests
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("Thiếu gói `requests` — thêm vào requirements.txt để dùng remove.bg API.") from e
    buf = io.BytesIO()
    pil_rgb.convert("RGB").save(buf, format="PNG")
    png_bytes = buf.getvalue()
    resp = requests.post(
        "https://api.remove.bg/v1.0/removeBg",
        files={"image_file": ("portrait.png", png_bytes, "image/png")},
        data={"size": "auto", "type": "person"},
        headers={"X-Api-Key": api_key.strip()},
        timeout=120,
    )
    if resp.status_code != 200:
        hint = resp.text[:300] if resp.text else ""
        raise RuntimeError(f"remove.bg API trả lỗi HTTP {resp.status_code}. {hint}".strip())
    return Image.open(io.BytesIO(resp.content)).convert("RGBA")


def _remove_bg_and_compose_blue(
    pil_rgb: Image.Image,
    blue_rgb: Tuple[int, int, int],
    session: Any,
    high_key_photo: bool = False,
    low_light: bool = False,
    pil_segmentation: Image.Image | None = None,
    selfie_for_alpha: Any | None = None,
    remove_bg_api_key: str | None = None,
    rembg_model_name: str = "u2net",
) -> Tuple[Image.Image, bool]:
    """
    rembg hoặc remove.bg API → RGBA, ghép lên nền xanh. Luôn dùng màu từ `pil_rgb`; kênh alpha có thể
    suy từ ảnh boost (`pil_segmentation`) khi ảnh tối (chỉ bản local rembg).

    Ảnh sáng: tắt matting (tránh tối da). Ảnh tối: tắt matting + boost đầu vào — alpha matting
    dễ làm rách/lỗ mặt nạ khi tương phản thấp.
    Nếu có `selfie_for_alpha`: kết hợp max alpha với MediaPipe Selfie (chỉ rembg local).
    """
    inp_pil = pil_segmentation if pil_segmentation is not None else pil_rgb
    api_mode = bool(remove_bg_api_key and str(remove_bg_api_key).strip())

    if api_mode:
        fg = _remove_bg_via_remove_bg_api(inp_pil, str(remove_bg_api_key).strip())
    else:
        if remove is None:
            raise RuntimeError("Thiếu thư viện `rembg` trong môi trường hiện tại.")
        buf = io.BytesIO()
        inp_pil.save(buf, format="PNG")
        inp = buf.getvalue()

        use_pymatting = (
            (not high_key_photo)
            and (not low_light)
            and (not _rembg_model_skip_pymatting(rembg_model_name))
        )
        if high_key_photo or low_light or not use_pymatting:
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
    # Blur alpha chỉ khi u2net + pymatting + điều kiện “bình thường” — còn lại dễ halo
    blur_alpha = (
        (not api_mode)
        and (not high_key_photo)
        and (not low_light)
        and (not _rembg_model_skip_pymatting(rembg_model_name))
    )
    a_np = _refine_alpha_u8(
        a_np,
        strong=low_light and not api_mode,
        blur_edges=blur_alpha,
    )
    if low_light and selfie_for_alpha is not None and not api_mode:
        a_np = _merge_rembg_alpha_with_selfie(a_np, pil_rgb, selfie_for_alpha)
    mask_ok, _ = _mask_subject_ok(a_np)
    r_arr = np.array(r_src)
    g_arr = np.array(g_src)
    b_arr = np.array(b_src)
    r_arr, g_arr, b_arr = _decontaminate_straight_alpha_rgb(r_arr, g_arr, b_arr, a_np, blue_rgb)
    r_src = Image.fromarray(r_arr, mode="L")
    g_src = Image.fromarray(g_arr, mode="L")
    b_src = Image.fromarray(b_arr, mode="L")
    a = Image.fromarray(a_np, mode="L")
    fg2 = Image.merge("RGBA", (r_src, g_src, b_src, a))
    bg = Image.new("RGBA", fg2.size, blue_rgb + (255,))
    comp = Image.alpha_composite(bg, fg2).convert("RGB")
    return comp, mask_ok


def detect_faces_mediapipe(bgr: np.ndarray, min_confidence: float = 0.6) -> List[Tuple[int, int, int, int, float]]:
    """Trả về danh sách khuôn mặt (x1,y1,x2,y2,score) theo pixel (không tái sử dụng detector — dùng PortraitProcessor khi batch)."""
    faces, _ = _detect_faces_with_detector(bgr, min_confidence=min_confidence, detector=None)
    return faces


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
        rembg_engine: str = "local",
        rembg_model: str = "u2net",
        remove_bg_api_key: str | None = None,
    ) -> None:
        self.ratio = ratio
        self.blue_rgb = blue_rgb
        self.min_face_conf = float(min_face_conf)
        self._rembg_engine = rembg_engine
        self._rembg_model = rembg_model
        if rembg_engine == "remove_bg_api":
            k = (remove_bg_api_key or os.environ.get("REMOVEBG_API_KEY") or "").strip()
            self._remove_bg_api_key = k or None
        else:
            self._remove_bg_api_key = None
        _require_cv2()
        self._fd = _try_get_mediapipe_face_detector(min_confidence=self.min_face_conf)
        self._selfie = _try_get_selfie_segmentation()
        self._rembg_session = None
        if rembg_engine == "local" and new_session is not None:
            for m in (rembg_model, "u2net"):
                try:
                    self._rembg_session = new_session(m)
                    self._rembg_model = m
                    break
                except Exception:
                    continue

    def process(
        self,
        pil_img: Image.Image,
        *,
        prefer_face_crop: bool = False,
        replace_background: bool = True,
        skip_rembg_if_uniform_background: bool = True,
        auto_orient: bool = True,
    ) -> ProcessResult:
        return process_portrait_image(
            pil_img,
            ratio=self.ratio,
            blue_rgb=self.blue_rgb,
            min_face_conf=self.min_face_conf,
            prefer_face_crop=prefer_face_crop,
            replace_background=replace_background,
            skip_rembg_if_uniform_background=skip_rembg_if_uniform_background,
            auto_orient=auto_orient,
            _mp_face_detector=self._fd,
            _rembg_session=self._rembg_session,
            _selfie_segmentation=self._selfie,
            _rembg_engine=self._rembg_engine,
            _rembg_model=self._rembg_model,
            _remove_bg_api_key=self._remove_bg_api_key,
        )


def _score_portrait_orientation_quality(
    bgr: np.ndarray,
    faces: List[Tuple[int, int, int, int, float]],
) -> float:
    """
    Điểm ưu tiên hướng chân dung: mặt chiếm tỷ lệ cao theo chiều dọc ảnh; bbox quá “dẹt ngang”
    (thường gặp khi ảnh bị xoay 90°) bị phạt nhẹ.
    """
    if not faces:
        return -1.0
    fx1, fy1, fx2, fy2, _ = max(faces, key=lambda f: (f[2] - f[0]) * (f[3] - f[1]))
    fh = float(fy2 - fy1)
    fw = float(fx2 - fx1)
    H = float(max(bgr.shape[0], 1))
    score = fh / H
    ar = fh / max(fw, 1.0)
    if ar < 0.72:
        score *= 0.82
    return score


def _primary_face_box_largest(
    faces: List[Tuple[int, int, int, int, float]],
) -> Tuple[int, int, int, int, float]:
    return max(faces, key=lambda f: (f[2] - f[0]) * (f[3] - f[1]))


def _select_bgr_orientation_for_portrait(
    bgr0: np.ndarray,
    min_face_conf: float,
    detector: Any | None,
    *,
    min_improve_vs_identity: float = 1.14,
    min_abs_score_gain: float = 0.024,
) -> Tuple[np.ndarray, Optional[str], List[Tuple[int, int, int, int, float]], int]:
    """
    Thử 4 hướng (gốc + 90°/180°/270°), chọn điểm cao nhất — nếu gốc đã có mặt:
    chỉ giữ gốc khi mặt đủ lớn theo chiều dọc (≥ ~8.2% chiều cao ảnh) và bbox “đứng”; nếu mặt quá nhỏ
    thì vẫn so với các hướng xoay (xoay 90° thường làm tăng điểm). Khi id thấp, cần margin tốt hơn
    mới chấp nhận xoay để hạn chế xoay nhầm ảnh thẳng toàn thân.
    """
    _require_cv2()
    candidates: Tuple[Tuple[np.ndarray, Optional[str]], ...] = (
        (bgr0, None),
        (cv2.rotate(bgr0, cv2.ROTATE_90_CLOCKWISE), "xoay 90° theo chiều kim đồng hồ"),
        (cv2.rotate(bgr0, cv2.ROTATE_180), "xoay 180°"),
        (cv2.rotate(bgr0, cv2.ROTATE_90_COUNTERCLOCKWISE), "xoay 90° ngược chiều kim đồng hồ"),
    )

    best: Optional[Tuple[np.ndarray, Optional[str], List[Tuple[int, int, int, int, float]], int, float]] = None
    id_score = -1.0
    id_pack: Optional[Tuple[np.ndarray, List[Tuple[int, int, int, int, float]], int]] = None

    for b, label in candidates:
        faces, n = _detect_faces_bgr_with_boost(b, min_face_conf, detector)
        if len(faces) == 0:
            continue
        score = _score_portrait_orientation_quality(b, faces)
        if label is None:
            id_score = score
            id_pack = (b, faces, n)
        if best is None or score > best[4] + 1e-9:
            best = (b, label, faces, n, score)

    if best is None:
        return bgr0, None, [], 0

    b, label, faces, n, best_score = best

    if label is not None and id_pack is not None and id_score >= 0:
        b0, f0, n0 = id_pack
        H0, W0 = int(bgr0.shape[0]), int(bgr0.shape[1])
        fx1, fy1, fx2, fy2, _ = _primary_face_box_largest(f0)
        fh = float(fy2 - fy1)
        fw = float(fx2 - fx1)
        ar_id = fh / max(fw, 1.0)

        # Chỉ “tin ảnh gốc đã thẳng” khi mặt chiếm đủ chiều cao khung; nếu quá nhỏ (~ảnh xoay 90°
        # vẫn bắt được mặt) thì không được giữ gốc — phải so với các hướng xoay.
        min_rel = float(min_improve_vs_identity)
        min_abs = float(min_abs_score_gain)
        trust_upright_min = 0.082
        if id_score < trust_upright_min:
            min_rel = max(min_rel, 1.20)
            min_abs = max(min_abs, 0.032)

        # Khung dọc/vuông + bbox không dẹt + mặt đủ lớn theo chiều dọc → giữ gốc.
        if H0 >= W0 * 0.94 and ar_id >= 0.64 and id_score >= trust_upright_min:
            return b0, None, f0, n0

        # Landscape + mặt đứng trong khung + mặt đủ lớn → giữ gốc.
        if W0 > H0 and ar_id >= 0.80 and id_score >= trust_upright_min:
            return b0, None, f0, n0

        rel_ok = best_score >= id_score * min_rel
        abs_ok = (best_score - id_score) >= min_abs
        if not (rel_ok and abs_ok):
            return b0, None, f0, n0

    return b, label, faces, n


def process_portrait_image(
    pil_img: Image.Image,
    ratio: str = "3x4",
    blue_rgb: Tuple[int, int, int] = (0, 91, 196),
    min_face_conf: float = 0.6,
    prefer_face_crop: bool = False,
    replace_background: bool = True,
    skip_rembg_if_uniform_background: bool = True,
    auto_orient: bool = True,
    _mp_face_detector: Any | None = None,
    _rembg_session: Any | None = None,
    _selfie_segmentation: Any | None = None,
    _rembg_engine: str = "local",
    _rembg_model: str = "u2net",
    _remove_bg_api_key: str | None = None,
) -> ProcessResult:
    """
    Pipeline backend:
    - Phát hiện khuôn mặt; nhiều ứng viên (nền/poster) → chọn một mặt chủ thể
    - Validation: vị trí, tỷ lệ, sáng/tương phản, nền
    - Crop theo mặt chỉ khi: prefer_face_crop hoặc nền không đơn sắc; không thì scale toàn ảnh (cover).
    - Ảnh thiếu sáng: có thể tinh chỉnh bbox crop bằng MediaPipe Selfie Segmentation (tóc/vai).
    - Chỉnh sáng nhẹ (CLAHE + trộn Y) khi cần
    - `replace_background`: True → rembg + nền xanh; False → chỉ crop/scale theo tỷ lệ, giữ nền gốc.
    - Khi `replace_background` và `skip_rembg_if_uniform_background`: nếu viền khung đầu ra gần một màu
      (tiêu chuẩn phông), bỏ qua rembg và giữ nền gốc.
    - `_rembg_engine`: `"local"` (rembg + ONNX), `"remove_bg_api"` (API remove.bg), `"none"` (không tải model).
    - EXIF trước; so 4 hướng nhưng chỉ tự xoay khi gốc nghiêng rõ (bbox dẹt / điểm thấp) hoặc xoay hơn hẳn gốc;
      ảnh dọc hoặc landscape có mặt đứng thường giữ nguyên.
      Nếu không hướng nào có mặt nhưng lật ngang/dọc lại có → ảnh không hợp lệ.
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

    if auto_orient:
        pil_img = _pil_apply_exif_transpose(pil_img)
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    orient_msg = "Hướng ảnh phù hợp."
    if auto_orient:
        orient_msg = "Hướng ảnh phù hợp (đã áp dụng EXIF nếu có)."
    bgr0 = _pil_to_bgr(pil_img)

    # Luôn xử lý theo hướng gốc (sau EXIF nếu bật). Phần xoay chỉ dùng cho checklist, không ảnh hưởng output.
    faces, n_face_candidates = _detect_faces_bgr_with_boost(bgr0, min_face_conf, _mp_face_detector)
    rot_label: Optional[str] = None
    rot_faces: List[Tuple[int, int, int, int, float]] = []
    rot_candidates = 0
    if auto_orient:
        _bgr_best, rot_label, rot_faces, rot_candidates = _select_bgr_orientation_for_portrait(
            bgr0, min_face_conf, _mp_face_detector
        )
        if rot_label is not None:
            # Không xoay output; chỉ cảnh báo/đánh dấu checklist.
            warnings.append(
                f"Ảnh có vẻ bị xoay — nên xoay đúng hướng trước khi upload (gợi ý: {rot_label})."
            )

    if len(faces) == 0:
        if auto_orient:
            # Nếu chỉ xoay mới thấy mặt: ảnh bị xoay → fail sớm.
            if rot_label is not None and rot_faces:
                errors.append(
                    f"Ảnh không hợp lệ: có dấu hiệu bị xoay ({rot_label}). Vui lòng upload ảnh khác (đúng hướng)."
                )
                checks["Định hướng ảnh"] = CheckResult(
                    False,
                    "Ảnh bị xoay 90°/180°/270° — không chấp nhận. Hãy upload ảnh khác.",
                )
                checks["Khuôn mặt"] = CheckResult(
                    False,
                    "Chỉ phát hiện được mặt khi xoay ảnh; cần file gốc đúng hướng.",
                )
                return ProcessResult(status="FAILED", errors=errors, warnings=warnings, checks=checks, processed_image=None)

            # Nếu chỉ lật mới thấy mặt: ảnh bị lật → fail sớm.
            b_h = cv2.flip(bgr0, 1)
            b_v = cv2.flip(bgr0, 0)
            fh, _ = _detect_faces_bgr_with_boost(b_h, min_face_conf, _mp_face_detector)
            fv, _ = _detect_faces_bgr_with_boost(b_v, min_face_conf, _mp_face_detector)
            if len(fh) > 0 or len(fv) > 0:
                if len(fh) > 0 and len(fv) > 0:
                    kind = "lật ngang và/hoặc lật dọc"
                elif len(fh) > 0:
                    kind = "lật ngang (ảnh gương)"
                else:
                    kind = "lật dọc (ngược chiều dọc)"
                errors.append(
                    "Ảnh không hợp lệ: có dấu hiệu "
                    + kind
                    + ". Vui lòng tải lên ảnh khác, đúng hướng chụp (không gương, không lật)."
                )
                checks["Định hướng ảnh"] = CheckResult(
                    False,
                    "Ảnh bị lật ngang hoặc lật ngược — không chấp nhận. Hãy upload ảnh khác.",
                )
                checks["Khuôn mặt"] = CheckResult(
                    False,
                    "Chỉ phát hiện được mặt khi lật ảnh; cần file gốc đúng hướng.",
                )
                return ProcessResult(
                    status="FAILED", errors=errors, warnings=warnings, checks=checks, processed_image=None
                )

    if len(faces) == 0:
        errors.append("Không tìm thấy khuôn mặt.")
        checks["Định hướng ảnh"] = CheckResult(
            False,
            "Không phát hiện mặt — hãy chụp lại rõ mặt, đúng hướng.",
        )
        checks["Khuôn mặt"] = CheckResult(False, "Không phát hiện được khuôn mặt.")
        return ProcessResult(status="FAILED", errors=errors, warnings=warnings, checks=checks, processed_image=None)

    if auto_orient and rot_label is not None:
        checks["Định hướng ảnh"] = CheckResult(
            False,
            f"Nghi ảnh bị xoay ({rot_label}). Output giữ nguyên hướng gốc; nên xoay đúng hướng trước khi upload.",
        )
    else:
        checks["Định hướng ảnh"] = CheckResult(True, orient_msg)

    h0, w0 = bgr0.shape[:2]
    brightness, contrast = _compute_brightness_contrast(bgr0)
    use_det_boost = brightness < 106 or contrast < 33

    if len(faces) == 1:
        sx1, sy1, sx2, sy2, _ = faces[0]
        fh_rel = (sy2 - sy1) / float(max(1, h0))
        if fh_rel < 0.092:
            if use_det_boost:
                fb, nc = _detect_faces_with_detector(
                    bgr0, min_confidence=min_face_conf, detector=_mp_face_detector
                )
            else:
                fb, nc = _detect_faces_with_detector(
                    _boost_bgr_for_face_detection(bgr0),
                    min_confidence=min_face_conf,
                    detector=_mp_face_detector,
                )
            if len(fb) >= 1:
                alt = fb[0]
                fh_alt = (alt[3] - alt[1]) / float(max(1, h0))
                if fh_alt > fh_rel + 0.015:
                    faces = fb
                    n_face_candidates = max(n_face_candidates, nc)

    fx1, fy1, fx2, fy2, _score = faces[0]
    if n_face_candidates > 1:
        warnings.append(
            "Phát hiện nhiều vùng giống mặt (nền, poster, ảnh nhỏ) — chỉ dùng mặt chính (lớn, gần giữa khung)."
        )
        checks["Khuôn mặt"] = CheckResult(
            True,
            f"Đã chọn 1 mặt chủ thể (lọc từ {n_face_candidates} vùng phát hiện).",
        )
    else:
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

    ok_bg_final, msg_bg_final = _background_uniformity_check(out_bgr, standard_wording=True)
    checks["Phông nền (khung đầu ra)"] = CheckResult(ok_bg_final, msg_bg_final)

    if not replace_background:
        checks["Thay nền xanh"] = CheckResult(
            True,
            "Đã tắt — chỉ chuẩn hóa khung theo tỷ lệ đã chọn và cân bằng sáng (giữ nền gốc).",
        )
        return ProcessResult(status="OK", errors=errors, warnings=warnings, checks=checks, processed_image=out_pil)

    # Cả hai đều đơn sắc + user không ép ghép → không rembg (studio trắng/xanh dễ artefact trên áo tối).
    # Bật "Luôn ghép nền…" → skip_rembg_if_uniform_background=False → vẫn chạy rembg.
    if skip_rembg_if_uniform_background and ok_bg and ok_bg_final:
        checks["Thay nền xanh"] = CheckResult(
            True,
            "Bỏ qua ghép nền xanh — nền gốc và khung đầu ra đều đơn sắc (giữ phông, tránh artefact rembg).",
        )
        return ProcessResult(status="OK", errors=errors, warnings=warnings, checks=checks, processed_image=out_pil)

    if skip_rembg_if_uniform_background and (ok_bg_final or ok_bg):
        checks["Thay nền xanh"] = CheckResult(
            True,
            "Bỏ qua ghép nền xanh — phông đạt chuẩn một màu (ảnh gốc hoặc khung đầu ra, giữ nguyên nền).",
        )
        return ProcessResult(status="OK", errors=errors, warnings=warnings, checks=checks, processed_image=out_pil)

    # Studio / nền trắng: rembg matting dễ tối màu → tắt matting, giữ màu gần gốc hơn
    high_key = brightness >= 128 or br_crop >= 108
    # Ảnh tối / tương phản thấp: matting + rembg trên ảnh chưa boost dễ rách mặt nạ → boost đầu vào + tắt matting
    low_light = (not high_key) and (brightness < 108 or br_crop < 88 or not ok_bc)
    pil_seg = _bgr_to_pil(_boost_bgr_for_segmentation(out_bgr)) if low_light else None

    use_api = _rembg_engine == "remove_bg_api"
    api_key_stripped = (_remove_bg_api_key or "").strip()

    try:
        if use_api and not api_key_stripped:
            raise RuntimeError(
                "Chế độ remove.bg: chưa có API key. Đặt REMOVEBG_API_KEY trong Streamlit Secrets "
                "hoặc biến môi trường — xem https://www.remove.bg/api"
            )

        session: Any = None
        if not use_api:
            if new_session is None or remove is None:
                raise RuntimeError("Thiếu rembg")
            session = _rembg_session if _rembg_session is not None else new_session(_rembg_model)
            if session is None:
                try:
                    session = new_session("u2net")
                except Exception as e:
                    raise RuntimeError(
                        "Không tải được model rembg. Kiểm tra mạng hoặc onnxruntime."
                    ) from e

        out_pil, mask_ok = _remove_bg_and_compose_blue(
            out_pil,
            blue_rgb=blue_rgb,
            session=session,
            high_key_photo=high_key,
            low_light=low_light,
            pil_segmentation=pil_seg,
            selfie_for_alpha=_selfie_segmentation if low_light else None,
            remove_bg_api_key=api_key_stripped if use_api else None,
            rembg_model_name=_rembg_model if not use_api else "u2net",
        )
        if use_api:
            msg_rembg = "Đã thay nền xanh (remove.bg API — gần chất lượng trang upload)."
        elif high_key:
            msg_rembg = "Đã thay nền xanh (ảnh sáng: tách nền không matting để giữ màu)."
        elif low_light:
            if _selfie_segmentation is not None:
                msg_rembg = (
                    "Đã thay nền xanh (ảnh tối: rembg + selfie giữ tóc/vai; nên chụp sáng hơn nếu cần)."
                )
            else:
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


def _detect_faces_bgr_with_boost(
    bgr: np.ndarray,
    min_face_conf: float,
    detector: Any | None,
) -> Tuple[List[Tuple[int, int, int, int, float]], int]:
    brightness, contrast = _compute_brightness_contrast(bgr)
    use_det_boost = brightness < 106 or contrast < 33
    bgr_for_faces = _boost_bgr_for_face_detection(bgr) if use_det_boost else bgr
    faces, n_face_candidates = _detect_faces_with_detector(
        bgr_for_faces, min_confidence=min_face_conf, detector=detector
    )
    if len(faces) == 0 and not use_det_boost:
        faces, n_face_candidates = _detect_faces_with_detector(
            _boost_bgr_for_face_detection(bgr), min_confidence=min_face_conf, detector=detector
        )
    return faces, n_face_candidates


def _detect_faces_with_detector(
    bgr: np.ndarray,
    min_confidence: float,
    detector: Any | None,
) -> Tuple[List[Tuple[int, int, int, int, float]], int]:
    """
    Nếu có detector (FaceDetection) thì dùng lại; nếu không thì tạo tạm (chậm hơn).
    Trả về (danh sách mặt, số ứng viên trước khi chọn 1 mặt chủ thể).
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
            return [], 0

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
        return _dedupe_face_detections(faces, w, h)

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
    return _dedupe_face_detections(faces2, w, h)
