# LucenFace

**LucenFace** là ứng dụng chuẩn hóa ảnh chân dung (ảnh thẻ): phát hiện khuôn mặt, kiểm tra tiêu chí, crop tỷ lệ, cân bằng sáng, xóa/thay nền và xuất batch (ZIP). Repo gồm **giao diện Streamlit**, **API FastAPI** và **web Next.js** dùng chung pipeline xử lý.

| Thành phần | Mô tả |
|------------|--------|
| Streamlit | `app.py` — chạy local hoặc [Streamlit Community Cloud](https://share.streamlit.io/) |
| Next.js + API | UX hiện đại; Next gọi FastAPI (`web/` + `api/`) |
| `host/` | Script chạy nhanh (Windows/macOS/Linux) và Docker |

## Tính năng chính

- Upload tối đa **50 ảnh** (JPG/PNG), hỗ trợ dán ảnh
- Phát hiện mặt (**MediaPipe**, có fallback)
- Checklist tiêu chí (vị trí, tỷ lệ, ánh sáng, nền, …)
- Auto-fix: crop **3×4 / 4×6**, cân bằng sáng
- Xóa nền & nền xanh chuẩn với **rembg**
- Tải từng ảnh hoặc **ZIP** toàn bộ ảnh xử lý thành công

## Yêu cầu hệ thống

- **Python 3.11 hoặc 3.12** (khuyến nghị). Python 3.13: thử `requirements-local-py313.txt` nếu thiếu wheel.
- **Git** (clone / pull).
- Giao diện Next.js: **Node.js 18+** và npm.

---

## Cài đặt (Python)

Từ **thư mục gốc repo** (chứa `requirements.txt`):

```bash
python -m venv .venv
```

**Windows (cmd):**

```bat
.venv\Scripts\activate
```

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
source .venv/bin/activate
```

Cài phụ thuộc:

```bash
pip install -r requirements.txt
```

---

## Cách chạy

### 1) Streamlit (đơn giản nhất)

```bash
streamlit run app.py
```

Hoặc chạy trực tiếp module UI:

```bash
streamlit run frontend/app.py
```

Mở trình duyệt tại URL hiển thị (thường **http://localhost:8501**).

### 2) Script có sẵn (`host/`)

Cách nhanh trên **Windows** (không cần đổi ExecutionPolicy nếu dùng `.cmd`):

```powershell
cd đường\dẫn\tới\repo
.\host\run-local.cmd
```

Hoặc:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\host\run-local.ps1
```

**macOS / Linux:**

```bash
chmod +x host/run-local.sh
./host/run-local.sh
```

Lần đầu script có thể tạo `.venv` và `pip install` (vài phút). Dừng server: **Ctrl+C**.

### 3) Docker

Từ **gốc repo**:

```bash
docker compose -f host/docker-compose.yml up --build
```

Trình duyệt: **http://localhost:8501**.

### 4) Next.js + API (khuyến nghị khi cần UX web chuyên nghiệp)

**Cửa sổ terminal 1 — API (từ gốc repo, venv đã kích hoạt):**

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Hoặc:

```bash
npm run dev:api
```

**Terminal 2 — frontend:**

```bash
cd web
npm install
copy .env.local.example .env.local
```

Trên macOS/Linux thay `copy` bằng `cp .env.local.example .env.local`.

```bash
npm run dev
```

Hoặc từ gốc repo (sau khi đã `cd web && npm install` ít nhất một lần):

```bash
npm run dev:web
```

- Web: [http://127.0.0.1:3000](http://127.0.0.1:3000)
- API: [http://127.0.0.1:8000](http://127.0.0.1:8000)

Nếu trình duyệt báo **ERR_CONNECTION_REFUSED** khi mở trang web, hãy đảm bảo `npm run dev` / `npm run dev:web` đang chạy.

---

## Cấu hình tùy chọn

| Biến môi trường | Mục đích |
|-----------------|----------|
| `REMOVEBG_API_KEY` | Tích hợp remove.bg (đặt trước khi chạy `uvicorn` hoặc trong `.env` khi dùng Docker) |
| `NEXT_PUBLIC_API_URL` | URL API cho Next.js (file `web/.env.local`, mặc định `http://localhost:8000`) |
| `WEB_CORS_ORIGINS` | Danh sách origin CORS cho API, phân tách bằng dấu phẩy (mặc định cho phép `http://localhost:3000`) |

**Docker + remove.bg:** tạo `.env` ở thư mục bạn chạy `docker compose` với `REMOVEBG_API_KEY=...`, hoặc `export REMOVEBG_API_KEY=...` trước khi `docker compose up`.

**Giới hạn upload (local):** chỉnh `.streamlit/config.toml` (ví dụ `maxUploadSize`).

---

## Cấu trúc thư mục

- `frontend/` — UI Streamlit, cấu hình thương hiệu (`APP_BRAND_NAME`, `APP_BUILD` trong `frontend/config.py`)
- `backend/` — xử lý ảnh (OpenCV, MediaPipe, rembg)
- `api/` — FastAPI
- `web/` — Next.js
- `app.py` — entrypoint Streamlit (dùng cho Cloud)
- `preview_app.py` — **chỉ UI**, không tải OpenCV/MediaPipe (build nhanh trên Cloud để xem layout)
- `host/` — `run-local.*`, `Dockerfile`, `docker-compose.yml`

---

## Streamlit Community Cloud

1. Đăng nhập [share.streamlit.io](https://share.streamlit.io/) bằng GitHub.
2. **New app** → chọn repo → branch `main`.
3. **Main file path:** `app.py`.
4. **Deploy** — Cloud cài từ `requirements.txt`.

Repo có **`packages.txt`** (apt) phục vụ OpenCV trên Linux. **Không ghi comment** trong `packages.txt`. Nếu lỗi apt trên image mới: **Manage app** → **Reboot** sau khi push.

Nếu `mediapipe` không có wheel cho phiên bản Python Cloud: **Settings** → **Python version** chọn **3.11** hoặc **3.12**.

### Bản preview UI (không xử lý ảnh thật)

- Main file: `preview_app.py`
- Requirements: `requirements-preview.txt`

---

## Kiểm tra bản deploy đúng code

1. Trên Streamlit: sidebar xem **Build** (`APP_BUILD` trong `frontend/config.py`) và hash Git (nếu có `.git`).
2. So sánh với commit mới nhất trên GitHub.
3. Hard refresh trình duyệt (**Ctrl+F5**).
4. Khi cần “đánh dấu” deploy mới: tăng/đổi **`APP_BUILD`** trong `frontend/config.py` rồi push.

---

## Đẩy lên GitHub

1. Tạo repository mới (trống, **không** tick “Add README” nếu bạn push code lần đầu từ máy).
2. Trong thư mục dự án:

```bash
git init
git branch -M main
git add .
git commit -m "Initial commit: LucenFace — chuẩn hóa ảnh chân dung"
git remote add origin https://github.com/<TAI_KHOAN>/<TEN_REPO>.git
git push -u origin main
```

Thay `<TAI_KHOAN>` và `<TEN_REPO>`. SSH: `git@github.com:<TAI_KHOAN>/<TEN_REPO>.git`.

`.gitignore` đã bỏ qua `venv/`, `__pycache__/`, … — không đẩy môi trường ảo lên GitHub.

---

## Gỡ lỗi nhanh

| Vấn đề | Gợi ý |
|--------|--------|
| `python` không nhận | Cài Python 3.12, bật PATH, mở lại terminal. |
| Thiếu OpenCV | `pip install opencv-python-headless` trong venv. |
| MediaPipe / rembg / numpy (Python 3.13) | `pip install -r requirements.txt` lại; ưu tiên Python 3.12. |
| Port **8501** đã dùng | `streamlit run app.py --server.port 8502` |

- Batch xử lý dùng cache detector / rembg để nhanh hơn.
- Theme Streamlit: `.streamlit/config.toml`.

---

*LucenFace — chuẩn hóa chân dung · batch & ZIP*
