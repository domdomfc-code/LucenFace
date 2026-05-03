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
- `api/`: FastAPI (dùng chung pipeline với Streamlit)
- `web/`: Next.js UI (gọi API)
- `app.py`: entrypoint cho Streamlit Cloud

## Giao diện Next.js + API (khuyến nghị khi cần UX chuyên nghiệp)

**Terminal 1 — Python (từ thư mục gốc `p2c`):**

```bash
pip install -r requirements.txt
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Tương đương: `npm run dev:api` (lắng nghe `127.0.0.1:8000`).

**Terminal 2 — frontend:**

```bash
cd web
npm install
copy .env.local.example .env.local   # Windows; Linux/macOS: cp ...
npm run dev
```

Hoặc **một lệnh từ thư mục gốc** (sau khi đã `cd web && npm install` ít nhất một lần):

```bash
npm run dev:web
```

Mở [http://127.0.0.1:3000](http://127.0.0.1:3000) (hoặc `http://localhost:3000`). **Nếu trình duyệt báo “từ chối kết nối”** (`ERR_CONNECTION_REFUSED`), nghĩa là chưa chạy `npm run dev` / `npm run dev:web` — static như `/lucenface-logo.png` chỉ có khi Next đang chạy.

API: [http://127.0.0.1:8000](http://127.0.0.1:8000) với `npm run dev:api` từ gốc repo (cần `pip install -r requirements.txt`). Đổi URL API qua `NEXT_PUBLIC_API_URL` trong `web/.env.local`.

remove.bg: đặt biến môi trường `REMOVEBG_API_KEY` trước khi chạy `uvicorn`. CORS: mặc định cho phép `http://localhost:3000`; danh sách khác qua `WEB_CORS_ORIGINS` (phân tách bằng dấu phẩy).

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

## Kiểm tra app Streamlit đã đúng bản code mới

1. Mở app → **sidebar** xem **Build** (`APP_BUILD` trong `frontend/config.py`) và **Git** (7 ký tự đầu của commit, nếu Cloud có `.git`).
2. So sánh Git đó với repo GitHub (commit mới nhất trên nhánh deploy) — khớp thì đang chạy đúng commit.
3. Trên [Streamlit Cloud](https://share.streamlit.io/) → **Manage app** → xem lịch sử deploy / nút **Reboot** khi nghi bản cũ.
4. Trình duyệt: **Ctrl+F5** (hard refresh) tránh cache.
5. Mỗi lần muốn “thấy rõ” có deploy mới: tăng / đổi chuỗi **`APP_BUILD`** trong `frontend/config.py` rồi push (topbar và sidebar đều hiện).

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

