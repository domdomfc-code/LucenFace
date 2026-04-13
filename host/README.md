# Chạy LucenFace trên máy (localhost)

Thư mục này chứa **cách chạy đầy đủ** app trên máy bạn (RAM/GPU theo máy, không giới hạn Streamlit Cloud).

Mã nguồn app vẫn nằm ở **gốc repo** (`app.py`, `frontend/`, `backend/`). Script trong `host/` chỉ **khởi chạy** từ đúng thư mục đó.

## Yêu cầu

- Python **3.11 hoặc 3.12** (khuyến nghị; 3.13 có thể thiếu wheel cho một số gói — xem `requirements-local-py313.txt` ở gốc repo nếu cần).
- Windows / macOS / Linux.

## Cách 1: PowerShell (Windows)

Trong PowerShell, từ thư mục `host/`:

```powershell
.\run-local.ps1
```

Hoặc từ gốc repo:

```powershell
.\host\run-local.ps1
```

Lần đầu: tạo `.venv` ở **gốc repo**, cài `requirements.txt`, rồi mở <http://localhost:8501>.

## Cách 2: Bash (macOS / Linux)

```bash
chmod +x host/run-local.sh
./host/run-local.sh
```

## Cách 3: Docker

Từ **gốc repo** (không phải trong `host/`):

```bash
docker compose -f host/docker-compose.yml up --build
```

Trình duyệt: <http://localhost:8501>.

Biến môi trường tùy chọn (remove.bg):

```bash
export REMOVEBG_API_KEY="your_key"
docker compose -f host/docker-compose.yml up --build
```

## Tăng giới hạn upload (local)

Mặc định `.streamlit/config.toml` đặt `maxUploadSize = 20` (MB). Trên máy bạn có thể tạo `.streamlit/config.local.toml` (hoặc sửa trực tiếp `config.toml`) và tăng `maxUploadSize` (ví dụ `200`).

## Gỡ lỗi nhanh

- Thiếu OpenCV: `pip install opencv-python-headless` trong venv.
- MediaPipe / rembg lỗi trên Python 3.13: thử 3.12 hoặc file `requirements-local-py313.txt`.
