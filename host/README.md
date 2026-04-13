# Chạy LucenFace trên máy (localhost)

Thư mục `host/` chứa **script và Docker** để chạy app trên máy bạn (không giới hạn Streamlit Cloud).

**Mã nguồn** (`app.py`, `frontend/`, `backend/`) luôn ở **gốc repo**. Sau mỗi `git pull`, bạn chạy lại như dưới — không cần copy file vào `host/`.

Phiên bản hiển thị trên UI xem `APP_BUILD` trong `frontend/config.py`.

## Yêu cầu

- **Windows / macOS / Linux**
- **Python 3.11 hoặc 3.12** (khuyến nghị). Python 3.13 có thể thiếu wheel một số gói — thử `requirements-local-py313.txt` ở gốc repo nếu cần.
- Git (để clone/pull). Trên Windows cần [Python](https://www.python.org/downloads/) và tùy chọn “Add to PATH”.

---

## Cách nhanh: Windows (PowerShell)

1. Mở **PowerShell**.
2. Vào thư mục repo (ví dụ đã clone `LucenFace`):

   ```powershell
   cd E:\Code\p2c
   ```

3. (Khuyến nghị) Cập nhật code mới nhất:

   ```powershell
   git pull
   ```

4. Chạy:

   ```powershell
   .\host\run-local.ps1
   ```

5. Trình duyệt mở **http://localhost:8501** (hoặc mở tay URL đó).

6. Dừng server: trong cửa sổ PowerShell nhấn **Ctrl+C**.

**Lần đầu:** script tạo `.venv` ở **gốc repo** và `pip install -r requirements.txt` (có thể vài phút).

---

## macOS / Linux (Bash)

```bash
cd /đường/dẫn/tới/p2c    # hoặc tên repo của bạn
git pull                 # tùy chọn
chmod +x host/run-local.sh
./host/run-local.sh
```

Mở **http://localhost:8501** — dừng bằng **Ctrl+C**.

---

## Docker (mọi OS)

Từ **gốc repo** (thư mục cha của `host/`):

```bash
git pull
docker compose -f host/docker-compose.yml up --build
```

Trình duyệt: **http://localhost:8501**.

**remove.bg (tùy chọn):** tạo file `.env` cùng thư mục bạn chạy lệnh (thường là gốc repo) với:

```env
REMOVEBG_API_KEY=your_key_here
```

Hoặc:

```bash
export REMOVEBG_API_KEY="your_key"
docker compose -f host/docker-compose.yml up --build
```

---

## Tăng giới hạn upload (local)

Trong repo có `.streamlit/config.toml` (`maxUploadSize`, …). Trên máy bạn có thể tạo `.streamlit/config.local.toml` hoặc sửa trực tiếp để tăng `maxUploadSize` (MB).

---

## Gỡ lỗi nhanh

| Vấn đề | Gợi ý |
|--------|--------|
| `python` không nhận | Cài Python 3.12, bật PATH, mở lại terminal. |
| Thiếu OpenCV | Trong venv: `pip install opencv-python-headless` |
| MediaPipe / rembg lỗi trên 3.13 | Dùng Python 3.12 hoặc `requirements-local-py313.txt` |
| Port 8501 đã dùng | Đổi port: `streamlit run app.py --server.port 8502` (hoặc sửa tương tự trong `run-local.ps1` / `run-local.sh`) |

---

## Nội dung thư mục `host/`

| File | Mục đích |
|------|----------|
| `run-local.ps1` | Windows: venv + cài deps + Streamlit |
| `run-local.sh` | macOS/Linux: tương tự |
| `Dockerfile` | Ảnh Docker Python 3.12 + deps |
| `docker-compose.yml` | `docker compose` từ gốc repo |
