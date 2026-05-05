# Báo cáo dự án LucenFace

**Phiên bản tài liệu:** 1.0  
**Ngày:** tháng 5 năm 2026  
**Phạm vi:** mã nguồn repository LucenFace (chuẩn hóa ảnh chân dung)

---

## 1. Tóm tắt điều hành

**LucenFace** là hệ thống phần mềm hỗ trợ chuẩn hóa ảnh chân dung phục vụ quy trình ảnh thẻ / hồ sơ: kiểm tra chất lượng đầu vào, phát hiện khuôn mặt, crop theo tỷ lệ chuẩn, xử lý nền và xuất kết quả dạng đơn lẻ hoặc gói nén ZIP. Sản phẩm triển khai theo mô hình **đa giao diện — một pipeline xử lý**: ứng dụng web Streamlit, API REST (FastAPI) kết hợp giao diện Next.js, cùng tùy chọn container hóa và script chạy cục bộ.

**Giá trị chính:** giảm thao tác thủ công khi xử lý hàng loạt ảnh, thống nhất tiêu chí kiểm tra, và cung cấp lộ trình mở rộng từ demo nội bộ tới tích hợp qua API.

---

## 2. Bối cảnh và mục tiêu dự án

### 2.1 Bối cảnh

Ảnh chân dung từ nhiều nguồn (điện thoại, máy ảnh, scan) thường khác nhau về tỷ lệ, góc chụp, ánh sáng và nền. Việc chuẩn hóa thủ công tốn thời gian và dễ không đồng nhất giữa các bộ hồ sơ.

### 2.2 Mục tiêu

| Mục tiêu | Mô tả |
|----------|--------|
| **Tự động hóa** | Phát hiện mặt, crop, cân chỉnh và nền theo cấu hình có thể lặp lại. |
| **Kiểm soát chất lượng** | Checklist tiêu chí trước / trong xử lý, phản hồi rõ ràng cho từng ảnh. |
| **Xử lý batch** | Hỗ trợ nhiều ảnh trong một phiên, giới hạn rõ ràng để ổn định hệ thống. |
| **Linh hoạt triển khai** | Chạy cục bộ, cloud (Streamlit), hoặc tích hợp qua HTTP API và SPA. |

### 2.3 Phạm vi ngoài dự án (giới hạn có chủ đích)

- Không thay thế quyết định pháp lý về tiêu chuẩn ảnh hộ chiếu / CMND của từng cơ quan.
- Định dạng **HEIC/HEIF** được từ chối có thông báo; người dùng cần xuất JPG/PNG.
- Một số chế độ màu ảnh đặc thù (ví dụ CMYK) được yêu cầu chuyển về RGB trước khi xử lý.

---

## 3. Mô tả sản phẩm và chức năng

### 3.1 Luồng nghiệp vụ tổng quát

1. Người dùng tải lên hoặc cung cấp ảnh (JPG/PNG), tối đa **50 file** mỗi lần (theo cấu hình API/dịch vụ).
2. Hệ thống **kiểm tra (audit)** định dạng, kích thước và khả năng đọc ảnh; từ chối có lý do.
3. Pipeline phát hiện khuôn mặt (ưu tiên **MediaPipe**), áp dụng heuristic crop theo tỷ lệ **3×4** hoặc **4×6**, tùy chọn căn tâm theo **mũi** hoặc **bbox mặt**.
4. Tùy cấu hình: thay nền, dùng **rembg** cục bộ hoặc API **remove.bg**, cân bằng sáng và các bước chuẩn hóa khác trong `PortraitProcessor`.
5. Xuất ảnh đã xử lý (từng file hoặc **ZIP** qua API).

### 3.2 Nhóm chức năng chính

- **Upload & kiểm soát đầu vào:** giới hạn kích thước file (ví dụ tối đa **12 MB** mỗi ảnh trên API), sniff định dạng, xử lý EXIF orientation.
- **Phát hiện và crop chân dung:** tích hợp logic dedupe/NMS và chọn mặt chính phù hợp ảnh chân dung.
- **Checklist chất lượng:** kết quả kiểm tra có trạng thái và thông điệp (JSON qua API).
- **Xuất batch:** endpoint ZIP với nén DEFLATE, tên file tải về thống nhất.

---

## 4. Kiến trúc kỹ thuật

### 4.1 Sơ đồ logic tầng

```text
┌─────────────────────────────────────────────────────────────┐
│  Lớp trình bày                                                │
│  • Streamlit (`frontend/app.py`, `app.py`)                   │
│  • Next.js (`web/`)                                          │
└───────────────────────────┬─────────────────────────────────┘
                            │ HTTP / multipart
┌───────────────────────────▼─────────────────────────────────┐
│  Lớp API (tùy chọn)                                          │
│  • FastAPI `api/main.py` — audit, process, process-zip       │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│  Lớp nghiệp vụ / orchestration                               │
│  • `api/service.py` — validate, cache processor              │
│  • `frontend/processing_core.py` — gọi processor           │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│  Lớp xử lý ảnh                                               │
│  • `backend/image_utils.py` — PortraitProcessor, CV, MP      │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Ngăn xếp công nghệ

| Thành phần | Công nghệ |
|------------|-----------|
| Ngôn ngữ lõi | Python 3.11–3.12 (khuyến nghị) |
| Xử lý ảnh | OpenCV (headless), Pillow, NumPy |
| Phát hiện mặt | MediaPipe Face Detection (có đường dự phòng) |
| Tách nền | rembg, ONNX Runtime; tùy chọn remove.bg API |
| UI nhanh / nội bộ | Streamlit |
| API | FastAPI, Pydantic (`ProcessConfig`) |
| SPA | Next.js, Tailwind CSS |
| Container | Docker (`host/Dockerfile`, `docker-compose`) |

### 4.3 Tái sử dụng pipeline

- `PortraitProcessor` tập trung tại `backend/image_utils.py`.
- Streamlit và FastAPI đều đi qua lớp `run_portrait_process` / `get_portrait_processor` để tránh trùng lặp logic và dễ bảo trì.

### 4.4 Hiệu năng

- **Cache theo khóa cấu hình:** `api/service.py` giữ instance processor theo `ProcessConfig.cache_key()` (tỷ lệ crop, màu nền, ngưỡng confidence, engine rembg, model), giảm chi phí khởi tạo lặp lại khi xử lý batch.

---

## 5. API và hợp đồng dữ liệu

### 5.1 Endpoint chính (FastAPI)

| Phương thức | Đường dẫn | Mục đích |
|-------------|-----------|----------|
| GET | `/api/health` | Kiểm tra sức khỏe dịch vụ, OpenCV, metadata build |
| POST | `/api/audit` | Kiểm tra ảnh + checklist không ghi file output đầy đủ |
| POST | `/api/process` | Xử lý theo chỉ số ảnh được chọn, trả JSON (base64 JPG, …) |
| POST | `/api/process-zip` | Trả file ZIP các ảnh JPG thành công |
| GET | `/api/config-hints` | Gợi ý cấu hình (API remove.bg, giới hạn file) |

### 5.2 Cấu hình xử lý (`ProcessConfig`)

Được gửi kèm dạng JSON trong form multipart, gồm các trường tiêu biểu: `ratio` (3x4 | 4x6), `prefer_face_crop`, `replace_blue_bg`, `blue_hex`, `min_face_conf`, `auto_orient`, `crop_center_mode` (nose | face), `letterbox_smart_framing`, `rembg_engine` (none | local | remove_bg_api), `rembg_model`.

Validation phía server (Pydantic) đảm bảo `blue_hex` đúng dạng `#RRGGBB`.

### 5.3 CORS và biến môi trường

- CORS mặc định cho phép `http://localhost:3000`; danh sách mở rộng qua **`WEB_CORS_ORIGINS`** (phân tách bằng dấu phẩy).
- **`REMOVEBG_API_KEY`:** bắt buộc khi chọn engine remove.bg qua API bên thứ ba.
- **`NEXT_PUBLIC_API_URL`:** URL backend cho ứng dụng Next.js.

---

## 6. Triển khai và vận hành

### 6.1 Môi trường cục bộ

- Cài đặt Python, tạo venv, `pip install -r requirements.txt`.
- Script tiện dụng: `host/run-local.ps1`, `run-local.cmd`, `run-local.sh`.
- Giao diện web đầy đủ: chạy `uvicorn` và `npm run dev` trong `web/` (xem README repository).

### 6.2 Streamlit Community Cloud

- Entrypoint khuyến nghị: **`app.py`**.
- File **`packages.txt`** cung cấp thư viện hệ thống cho OpenCV trên Linux (không chứa comment, tuân thủ yêu cầu nền tảng).
- Cần chú ý phiên bản Python trên cloud (khuyến nghị 3.11–3.12) để tương thích wheel MediaPipe.

### 6.3 Docker

- Build từ `host/Dockerfile`, orchestration qua `host/docker-compose.yml`.
- Có thể inject `REMOVEBG_API_KEY` qua biến môi trường hoặc file `.env` tùy cách triển khai.

### 6.4 Theo dõi phiên bản triển khai

- Hằng số **`APP_BUILD`** trong `frontend/config.py` dùng để nhận diện bản build trên UI (hữu ích khi cache hoặc nhiều môi trường).

---

## 7. Bảo mật, quyền riêng tư và tuân thủ vận hành

| Chủ đề | Thực tiễn trong dự án |
|--------|------------------------|
| **Khóa API** | Không hardcode; đọc từ biến môi trường (`REMOVEBG_API_KEY`). Không commit `.env` chứa secret (`.gitignore`). |
| **CORS** | Giới hạn origin có cấu hình; tránh `*` trong môi trường production nếu triển khai công khai. |
| **Đầu vào người dùng** | Giới hạn số file, kích thước, định dạng; từ chối có thông báo để giảm abuse và OOM. |
| **Dữ liệu cá nhân** | Ảnh chân dung là dữ liệu nhạy cảm; tổ chức triển khai cần chính sách lưu trữ, log và TLS riêng — nằm ngoài phạm vi “mặc định” của repo demo. |

---

## 8. Rủi ro và hạn chế

1. **Phụ thuộc mô hình ML:** chất lượng tách nền và phát hiện mặt phụ thuộc ảnh đầu vào (góc chụp, che khuất, ánh sáng cực đoan).
2. **Chi phí & độ trễ API:** remove.bg tính phí theo chính sách nhà cung cấp; xử lý batch lớn cần giám sát quota.
3. **Python 3.13+:** một gói có thể thiếu wheel; repo có file hỗ trợ thử nghiệm `requirements-local-py313.txt`.
4. **Định dạng ảnh:** HEIC và một số mode đặc biệt không được hỗ trợ trực tiếp — cần chuẩn hóa đầu vào phía người dùng.

---

## 9. Bảo trì và mở rộng đề xuất

- **Theo dõi & logging:** tích hợp structured logging và metrics (latency, tỷ lệ lỗi theo mã) khi đưa API ra production.
- **Xác thực API:** thêm API key hoặc OAuth nếu endpoint mở trên Internet.
- **Hàng đợi xử lý:** với khối lượng lớn, tách worker xử lý ảnh khỏi process HTTP đồng bộ.
- **Kiểm thử tự động:** unit test cho `validate_and_stage`, schema `ProcessConfig`, và golden tests ảnh mẫu trong `assets/sample_portraits/`.

---

## 10. Tổng Kết

LucenFace là giải pháp **chuyên biệt cho chuẩn hóa ảnh chân dung**, có kiến trúc rõ ràng (tách UI / API / xử lý ảnh), hỗ trợ nhiều kênh triển khai và mở rộng tích hợp qua FastAPI. Tài liệu này phục vụ mục đích nội bộ và trình bày dự án; chi tiết cài đặt và lệnh chạy tham chiếu **`README.md`** tại gốc repository.

---

*Tài liệu được soạn dựa trên trạng thái mã nguồn tại thời điểm lập báo cáo.*
