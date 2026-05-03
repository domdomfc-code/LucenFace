"""Hằng số giao diện và phiên bản build (đổi APP_BUILD khi deploy để vô hiệu cache processor)."""

APP_TITLE = "Chuẩn hóa ảnh chân dung học sinh"

# --- Nội dung hiển thị trang chủ (sửa tại đây) --------------------------------
APP_BRAND_NAME = "LucenFace"
APP_BRAND_TAGLINE = "Chuẩn hóa chân dung · batch & ZIP"
APP_TOPBAR_PILL = "≤ 50 ảnh · Cài đặt trong sidebar >>"
# Đoạn dưới tiêu đề lớn — cho phép HTML nhẹ (vd. <strong>…</strong>).
APP_INTRO_HTML = (
    "Upload hoặc dán <strong>JPG/PNG</strong>, kiểm tra rồi xử lý — tải <strong>ZIP</strong>. "
    "Tỷ lệ khung, nền xanh, rembg và tùy chọn nâng cao nằm trong <strong>sidebar (>>)</strong>."
)
# `st.caption` — Markdown (vd. **đậm**).
APP_TIP_CAPTION = (
    "Mẹo: ảnh rõ mặt, thẳng góc — tùy chọn ghép nền xanh và rembg nằm trong **sidebar** (>>)."
)

# Tăng / đổi chuỗi này khi deploy để dễ thấy trên Cloud (topbar + sidebar).
APP_BUILD = "3.18.6-mobile-thumb-row-js"
BLUE = "#005BC4"
BG = "#F6F9FF"
# Màu chữ nội dung / tiêu đề tùy chỉnh (CSS --text)
TEXT = "#54616C"
