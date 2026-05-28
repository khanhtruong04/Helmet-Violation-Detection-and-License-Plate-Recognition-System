# Helmet Detection & License Plate Recognition System

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-orange) ![UI](https://img.shields.io/badge/UI-Gradio-yellow)

> Hệ thống AI tự động phát hiện người tham gia giao thông không đội mũ bảo hiểm và trích xuất biển số xe vi phạm từ video giao thông theo thời gian thực.

Dự án tích hợp **YOLOv8** và **Faster R-CNN** để phát hiện mũ bảo hiểm, **YOLO Plate Detector** để phát hiện vị trí biển số, và **4 phương pháp OCR** (CNN, EasyOCR, CRNN, Transformer) để nhận dạng ký tự biển số. Giao diện web trực quan sử dụng **Gradio** cho phép xử lý video theo thời gian thực với streaming.

##  Mục Lục

- [Tính Năng Nổi Bật](#tính-năng-nổi-bật)
- [Cấu Trúc Dự Án](#cấu-trúc-dự-án)
- [Các Mô Hình AI](#các-mô-hình-ai)
- [Yêu Cầu Hệ Thống](#yêu-cầu-hệ-thống)
- [Hướng Dẫn Cài Đặt](#hướng-dẫn-cài-đặt)
- [Hướng Dẫn Chạy](#hướng-dẫn-chạy)
---

## Tính Năng Nổi Bật

- **Đa Mô Hình Phát Hiện Mũ:** Hỗ trợ **YOLOv8** (tốc độ cao) và **Faster R-CNN** (độ chính xác cao) để lựa chọn linh hoạt.
- **4 Phương Pháp OCR:** Lựa chọn từ **CNN Classifier**, **EasyOCR**, **CRNN**, hoặc **Transformer OCR** để nhận dạng biển số với độ chính xác tối ưu.
- **Test Time Augmentation (TTA):** Transformer OCR sử dụng 5 biến thể ảnh với voting để tăng độ chính xác.
- **Giao Diện Web Tương Tác:** Gradio UI cho phép:
  - Upload video giao thông trực tiếp
  - Lựa chọn mô hình và phương pháp OCR
  - Xem stream xử lý video theo thời gian thực
  - Tải xuống video đã annotate
- **GPU Acceleration:** Pipeline tối ưu hóa cho NVIDIA CUDA.

---

## Cấu Trúc Dự Án

```
C:\Python\project\
├── main.py                          # Entry point - Gradio web application
├── requirements.txt                 # Danh sách thư viện phụ thuộc
├── README.md                        # Tài liệu dự án
├── .venv/                           # Virtual environment
│
├── weights/                         # Thư mục chứa file trọng số (.pt, .pth, .h5)
│   ├── helmet_detection_yolov8.pt
│   ├── helmet_detection_fasterrcnn.pth
│   ├── best_lp_detector.pt
│   ├── character_classifier.h5
│   ├── crnn_ocr_best.pth
│   └── transformer_ocr_best.pt
│
├── modules/                         # Các module xử lý chính
│   ├── __init__.py
│   ├── helmet_detector.py           # Phát hiện mũ bảo hiểm
│   ├── plate_detector.py            # Phát hiện vị trí biển số
│   ├── ocr/
│   │   ├── __init__.py
│   │   ├── cnn_ocr.py               # CNN Character Classifier
│   │   ├── easyocr_module.py        # EasyOCR integration
│   │   ├── crnn_ocr.py              # CRNN network
│   │   └── transformer_ocr.py       # Transformer-based OCR
│   └── video_processor.py           # Pipeline xử lý video
│
├── utils/
│   ├── __init__.py
│   ├── drawing.py                   # Vẽ bounding box, annotation
│   └── helpers.py                   # Các hàm tiện ích
│
├── input/                           # Thư mục video đầu vào
│   └── sample_traffic.mp4
│
└── output/                          # Thư mục lưu video đầu ra
    └── result_video.mp4
```

---

## Các Mô Hình AI

### 1. Phát Hiện Mũ Bảo Hiểm

| Model | File | Framework | Đặc điểm |
|-------|------|-----------|----------|
| YOLOv8 | `helmet_detection_yolov8.pt` | Ultralytics | Tốc độ cao, real-time |
| Faster R-CNN | `helmet_detection_fasterrcnn.pth` | PyTorch | Độ chính xác cao, MobileNetV3 backbone |

**Output:**
- `helmet` → Box `
- `no_helmet` → Box `
- `rider` → Box `

### 2. Phát Hiện Biển Số Xe

| Model | File | Framework |
|-------|------|-----------|
| YOLOv8 (Plate) | `best_lp_detector.pt` | Ultralytics |

**Output:** Box  vàng 

### 3. Nhận Dạng Biển Số (4 Lựa Chọn)

| Phương Pháp | File |
|------------|------|
| CNN Classifier | `character_classifier.h5` |
| EasyOCR | `easyocr` library |
| CRNN | `crnn_ocr_best.pth` |
| Transformer OCR | `transformer_ocr_best.pt` |

---

## Yêu Cầu Hệ Thống

| Thành phần | Tối thiểu | Khuyến nghị |
|-----------|-----------|------------|
| OS | Windows 10 | Windows 10/11 |
| Python | 3.9+ | 3.10+ |
| RAM | 8 GB | 16 GB |
| GPU | Không bắt buộc | NVIDIA GPU (CUDA) 4GB+ |
| Disk | 5 GB | 10 GB |

---

## Hướng Dẫn Cài Đặt

### Bước 1: Clone hoặc Tải Dự Án

```bash
cd C:\Python\project
```

### Bước 2: Tạo Virtual Environment

```bash
python -m venv .venv
```

### Bước 3: Kích Hoạt Virtual Environment

```bash
# Windows CMD
.venv\Scripts\activate.bat

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

### Bước 4: Cài Đặt Phụ Thuộc

```bash
pip install -r requirements.txt
```

**`requirements.txt`:**

### Bước 5: Kiểm Tra File Trọng Số

Đảm bảo các file sau tồn tại trong `weights/`:

```
helmet_detection_yolov8.pt
helmet_detection_fasterrcnn.pth
best_lp_detector.pt
character_classifier.h5
crnn_ocr_best.pth
transformer_ocr_best.pt
```

---

## Hướng Dẫn Chạy

### 1. Khởi Chạy Ứng Dụng

```bash
# Kích hoạt virtual environment
.venv\Scripts\activate.bat

# Chạy ứng dụng
python main.py
```

### 2. Truy Cập Giao Diện

Mở trình duyệt: **http://localhost:7860/**

### 3. Sử Dụng Giao Diện

1. **Upload Video:** Tải lên video giao thông
2. **Chọn Mô Hình:** YOLOv8 hoặc Faster R-CNN
3. **Chọn OCR Method:** CNN / EasyOCR / CRNN / Transformer
4. **Bấm "Chạy":** Bắt đầu xử lý
5. **Xem Kết Quả:** Stream trực tiếp hoặc tải video

---

Tải các file trọng số mô hình:
https://drive.google.com/drive/folders/1J3O6DSZ58M5O2H5fAJMGUyTb8zbAFXCd?usp=sharing

---



