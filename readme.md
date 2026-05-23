# Helmet Detection & License Plate OCR System
**Python | PyTorch | YOLOv8 | Gradio**

Hệ thống phát hiện mũ bảo hiểm và trích xuất thông tin biển số xe tự động, được tối ưu hóa cho xử lý video thời gian thực. Dự án tích hợp YOLOv8 để phát hiện đối tượng, Faster R-CNN cho phát hiện chính xác, và nhiều phương pháp OCR tiên tiến (EasyOCR, CNN, CRNN, Transformer) để trích xuất và phân loại ký tự biển số.

## Tính Năng Nổi Bật

###  Phát Hiện Đa Mô Hình
- **YOLOv8**: Phát hiện nhanh, tối ưu cho xử lý video thời gian thực
- **Faster R-CNN**: Phát hiện chính xác cao, lý tưởng cho phân tích chi tiết
- Cả hai mô hình được tối ưu hóa cho GPU (CUDA)

###  Phát Hiện Mũ Bảo Hiểm
Nhận diện chính xác:
- Người đội mũ bảo hiểm
- Người không đội mũ bảo hiểm
- Hỗ trợ các góc nhìn khác nhau và điều kiện ánh sáng đa dạng

###  Trích Xuất Biển Số Xe
Tìm kiếm và phân loại biển số trong video:
- Phát hiện vị trí biển số (Plate Detection)
- Xử lý tiền xử lý (góc, độ sáng, nhiễu)

###  Nhận Dạng Ký Tự OCR - 4 Phương Pháp
1. **CNN Classifier**: Phân loại nhanh, tối ưu hóa cho hiệu suất
2. **EasyOCR**: OCR đơn giản, hỗ trợ nhiều ngôn ngữ
3. **CRNN**: Mạng nơ-ron tái quy, phù hợp với chuỗi ký tự
4. **Transformer OCR**: Mô hình tiên tiến, độ chính xác cao nhất

###  Giao Diện Tương Tác (Web UI)
- **Gradio Interface**: Giao diện trực quan, không cần kiến thức lập trình
- Tải video từ file
- Chọn mô hình phát hiện mũ (YOLOv8 hoặc Faster R-CNN)
- Chọn phương pháp OCR (CNN, EasyOCR, CRNN, Transformer)
- Xem kết quả stream và thống kê thời gian xử lý

###  Tối Ưu Hóa GPU
- Pipeline xử lý được tối ưu hóa cho GPU (CUDA)
- Hỗ trợ CPU fallback nếu GPU không khả dụng
- Batch processing cho hiệu suất cực đại

###  Nhật Ký & Thống Kê
- Lưu trữ kết quả nhận diện
- Thống kê hiệu suất xử lý (FPS, thời gian/frame)
- Export kết quả dưới dạng JSON/CSV

## Cấu Trúc Dự Án

```
helmet_ocr_project/
├── main.py                     # File chạy chính (Gradio UI)
├── test.py                     # Script kiểm tra cài đặt
├── requirements.txt            # Các thư viện phụ thuộc
├── INSTALLATION.md             # Hướng dẫn cài đặt chi tiết
├── IMPLEMENTATION_COMPLETE.md  # Thông tin triển khai
│
├── modules/                    # Các module xử lý
│   ├── helmet_detector.py      # Logic phát hiện mũ bảo hiểm
│   ├── plate_detector.py       # Logic phát hiện biển số xe
│   ├── video_processor.py      # Vòng lặp xử lý video chính
│   │
│   └── ocr/                    # Các thuật toán OCR
│       ├── cnn_ocr.py          # CNN Classifier
│       ├── easyocr_module.py   # EasyOCR wrapper
│       ├── crnn_ocr.py         # CRNN model
│       └── transformer_ocr.py  # Transformer-based OCR
│
├── utils/                      # Các tiện ích hỗ trợ
│   ├── drawing.py              # Vẽ bounding box, thông tin lên frame
│   ├── helpers.py              # Hàm trợ giúp chung
│   └── __init__.py
│
├── weights/                    # Thư mục trọng số mô hình
│   ├── helmet_detection_yolov8.pt
│   ├── helmet_detection_fasterrcnn.pt
│   ├── best_lp_detector.pt
│   ├── crnn_ocr_best.pth
│   ├── character_classifier.h5
│   └── transformer_ocr_best.pt
│
├── input/                      # Thư mục input video
│   └── (đặt video ở đây)
│
├── output/                     # Thư mục output video/kết quả
│   └── (kết quả được lưu ở đây)
│
└── notebooks/                  # Jupyter Notebooks cho phát triển
    ├── lp_detection_cnn_ocr.ipynb
    ├── lp_detection_easyocr.ipynb
    └── test_gpu.ipynb

```

## Hướng Dẫn Cài Đặt

### Yêu Cầu Hệ Thống
- **OS**: Windows 10+, Linux, macOS
- **Python**: 3.10+
- **RAM**: 8 GB (khuyến nghị 16 GB)
- **Disk**: 5-10 GB (tùy thuộc vào trọng số mô hình)
- **GPU** *(tùy chọn)*: NVIDIA GPU + CUDA 11.8+ (khuyến nghị)

### Bước 1: Clone Repository
```bash
git clone https://github.com/your-username/helmet-ocr-system.git
cd helmet-ocr-system
```

### Bước 2: Tạo Môi Trường Ảo & Cài Đặt
```bash
# Tạo virtual environment
python -m venv venv

# Activate (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Activate (Windows CMD)
.venv\Scripts\activate.bat

# Activate (Linux/macOS)
source venv/bin/activate
```

### Bước 3: Cài Đặt Phụ Thuộc
```bash
# Cài đặt PyTorch với CUDA support (tùy chọn nhưng khuyến nghị)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Cài đặt các thư viện còn lại
pip install -r requirements.txt
```

### Bước 4: Tải Trọng Số Mô Hình
Tải các file trọng số từ: [Google Drive Link](https://your-drive-link)

Hoặc tải riêng từ nguồn:
- **YOLOv8 Helmet Detection**: Tự động tải từ Ultralytics
- **Faster R-CNN**: Tải từ torchvision hoặc custom weights
- **License Plate Detector**: Custom trained weights
- **OCR Models**: Tùy theo phương pháp

### Bước 5: Kiểm Tra Cài Đặt
```bash
# Chạy script kiểm tra
python test.py

# Kết quả mong đợi:
# ✓ GPU available: Yes (nếu có GPU)
# ✓ PyTorch version: 2.0.0+
# ✓ YOLOv8 loaded successfully
# ✓ All dependencies installed
```

## Hướng Dẫn Chạy

### Chạy Giao Diện Web
```bash
python main.py
```

Khi chạy thành công, bạn sẽ thấy:
```
Running on local URL:  http://127.0.0.1:7860
```

Mở trình duyệt và truy cập: **http://localhost:7860**

### Sử Dụng Giao Diện

#### 1. **Tải Video**
- Click vào mục "Upload Video"
- Chọn file video từ máy tính
- Định dạng hỗ trợ: MP4, AVI, MOV, MKV

#### 2. **Chọn Mô Hình Phát Hiện Mũ**
- **YOLOv8 (Nhanh)**: Tốc độ cao, thích hợp xử lý thời gian thực
- **Faster R-CNN (Chính xác)**: Độ chính xác cao, xử lý chậm hơn

#### 3. **Chọn Phương Pháp OCR**
- **CNN Classifier**: Nhanh, tối ưu cho biển số tiêu chuẩn
- **EasyOCR**: Linh hoạt, hỗ trợ nhiều ngôn ngữ
- **CRNN**: Cân bằng tốc độ và độ chính xác
- **Transformer OCR**: Độ chính xác cao nhất, chậm hơn

#### 4. **Bắt Đầu Xử Lý**
- Click nút **"Start Processing"**
- Chờ kết quả (tùy theo độ dài video)
- Video output được lưu tại thư mục `output/`

#### 5. **Xem Kết Quả**
- **Video với annotation**: Bounding box, nhãn phát hiện
- **Thống kê**: FPS, thời gian xử lý, số frame
- **Export**: Lưu kết quả dưới dạng JSON/CSV
