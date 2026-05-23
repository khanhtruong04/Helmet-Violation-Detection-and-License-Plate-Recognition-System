"""
Hàm tiện ích cơ bản cho dự án
"""
import os
import cv2
import torch
import numpy as np
from pathlib import Path


def get_weight_path(weight_name):
    """Lấy đường dẫn đầy đủ đến file trọng số"""
    weight_dir = Path(__file__).parent.parent / "weights"
    return str(weight_dir / weight_name)


def ensure_dir_exists(directory):
    """Tạo thư mục nếu chưa tồn tại"""
    os.makedirs(directory, exist_ok=True)
    return directory


def draw_box(frame, box, label, color, thickness=2, draw_label=False):
    """
    Vẽ bounding box lên frame
    
    Args:
        frame: ảnh frame
        box: (x1, y1, x2, y2)
        label: tên nhãn (string)
        color: (B, G, R)
        thickness: độ dày của box
        draw_label: có vẽ label hay không
    
    Returns:
        frame với box đã vẽ
    """
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    return frame


def draw_text(frame, text, position, font_scale=0.7, color=(255, 255, 255), 
              thickness=2, bg_color=(0, 0, 0)):
    """
    Vẽ text trên frame với background
    
    Args:
        frame: ảnh frame
        text: nội dung text
        position: (x, y)
        font_scale: kích thước font
        color: màu text (B, G, R)
        thickness: độ dày text
        bg_color: màu background
    
    Returns:
        frame với text đã vẽ
    """
    x, y = position
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Tính kích thước text để vẽ background
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )
    
    # Vẽ background
    cv2.rectangle(
        frame,
        (x - 5, y - text_height - 10),
        (x + text_width + 5, y + baseline + 5),
        bg_color,
        -1
    )
    
    # Vẽ text
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)
    return frame


def get_fps_counter(start_time):
    """Tính FPS từ thời gian bắt đầu"""
    import time
    elapsed = time.time() - start_time
    return 1 / elapsed if elapsed > 0 else 0


def normalize_plate_text(text):
    """
    Chuẩn hóa text biển số Việt Nam
    - Loại bỏ khoảng trắng
    - Convert thành uppercase
    
    Args:
        text: text từ OCR
    
    Returns:
        text chuẩn hóa
    """
    text = text.strip()
    text = text.upper()
    text = text.replace(" ", "")
    return text


def crop_region(frame, box):
    """
    Cắt vùng từ frame theo bounding box
    
    Args:
        frame: ảnh frame
        box: (x1, y1, x2, y2)
    
    Returns:
        ảnh đã cắt
    """
    x1, y1, x2, y2 = map(int, box)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)
    return frame[y1:y2, x1:x2]


def resize_frame(frame, max_width=1280):
    """
    Resize frame để phù hợp nếu quá lớn
    
    Args:
        frame: ảnh frame
        max_width: chiều rộng tối đa
    
    Returns:
        frame đã resize (nếu cần)
    """
    if frame.shape[1] > max_width:
        ratio = max_width / frame.shape[1]
        new_height = int(frame.shape[0] * ratio)
        return cv2.resize(frame, (max_width, new_height))
    return frame


def get_device_info():
    """
    Kiểm tra thiết bị (GPU/CPU) và trả về thông tin chi tiết
    
    Returns:
        dict: {
            'device': 'cuda' hoặc 'cpu',
            'device_name': tên GPU/CPU,
            'message': thông báo tiếng Việt
        }
    """
    if torch.cuda.is_available():
        device = 'cuda'
        device_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
        message = f"✅ Sử dụng GPU: {device_name} ({gpu_memory:.1f}GB)"
        return {
            'device': device,
            'device_name': device_name,
            'memory_gb': gpu_memory,
            'message': message
        }
    else:
        device = 'cpu'
        import platform
        processor = platform.processor() or "CPU"
        message = "⚠️  Không tìm thấy GPU, sử dụng CPU (tốc độ xử lý sẽ chậm hơn)"
        return {
            'device': device,
            'device_name': processor,
            'memory_gb': None,
            'message': message
        }

