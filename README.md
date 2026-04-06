# Chuẩn hóa ảnh chân dung học sinh (Streamlit)

Ứng dụng web chuẩn hóa ảnh chân dung theo chuẩn ảnh thẻ:
- Upload tối đa **50 ảnh** (JPG/PNG)
- Phát hiện **đúng 1 khuôn mặt** (MediaPipe)
- Checklist tiêu chí (vị trí, tỷ lệ, ánh sáng/tương phản, nền)
- Auto-fix: crop theo **3x4 / 4x6**, cân bằng sáng
- Xóa nền & thay nền **xanh chuẩn** bằng `rembg`
- Tải về từng ảnh hoặc **ZIP** toàn bộ ảnh xử lý thành công

## Cấu trúc
- `frontend/`: Streamlit UI
- `backend/`: xử lý ảnh (OpenCV/MediaPipe/rembg)
- `app.py`: entrypoint cho Streamlit Cloud

## Cài đặt & chạy

```bash
pip install -r requirements.txt
streamlit run app.py
```

Hoặc chạy trực tiếp UI:

```bash
streamlit run frontend/app.py
```

## Ghi chú kỹ thuật
- Batch processing dùng cache tài nguyên nặng (MediaPipe detector + rembg session) để xử lý nhanh hơn.
- Theme xanh-trắng cấu hình tại `.streamlit/config.toml`.

## Đẩy lên GitHub

1. Tạo repository mới trên GitHub (repo trống, **không** tick “Add README” nếu bạn sẽ push code từ máy để tránh conflict lần đầu).
2. Trong thư mục dự án (`p2c`), chạy:

```bash
git init
git branch -M main
git add .
git commit -m "Initial commit: Streamlit portrait standardization app"
git remote add origin https://github.com/<TAI_KHOAN>/<TEN_REPO>.git
git push -u origin main
```

Thay `<TAI_KHOAN>` và `<TEN_REPO>` bằng đúng của bạn. Nếu dùng SSH, đổi URL `remote` sang dạng `git@github.com:...`.

**Lưu ý:** `.gitignore` đã bỏ qua `venv/`, `__pycache__/` — không đẩy môi trường ảo lên GitHub.

## Chạy trên Streamlit Community Cloud

1. Vào [Streamlit Community Cloud](https://share.streamlit.io/) và đăng nhập GitHub.
2. **New app** → chọn repo vừa push → branch `main`.
3. **Main file path**: `app.py` (file ở thư mục gốc dự án).
4. **Deploy**. Cloud sẽ cài đặt từ `requirements.txt`.
5. Lần chạy đầu, `rembg` có thể tải model (hơi lâu); nếu timeout, bấm **Reboot** trong app settings.

Không cần `secrets.toml` trừ khi bạn thêm API key sau này.

### Lỗi `ImportError` khi `import cv2` trên Streamlit Cloud
Repo đã kèm **`packages.txt`** (apt) cho OpenCV trên Linux (`libgl1`, `libglib2.0-0t64`, …). Trên **Debian Trixie** (image Cloud mới), dùng **`libglib2.0-0t64`** để có `libgthread-2.0.so.0`; **không** dùng `libglib2.0-0` (tên cũ, dễ lỗi apt). **Không ghi comment** trong `packages.txt`. Sau khi push: **Manage app** → **Reboot**.

Streamlit Cloud có thể dùng **Python 3.14**; `mediapipe==0.10.14` **không còn wheel** — repo dùng **`mediapipe==0.10.33`**. Nếu pip vẫn báo không có wheel: **Settings** → **Python version** chọn **3.11** hoặc **3.12** (Advanced settings).

