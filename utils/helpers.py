"""
Helper functions cho hệ thống phát hiện mũ bảo hiểm & biển số xe
"""

import os
import cv2
import torch
import numpy as np


def get_weight_path(filename):
    """
    Lấy đường dẫn đầy đủ tới file trọng số
    
    Args:
        filename: tên file (e.g. 'helmet_detection_yolov8.pt')
    
    Returns:
        đường dẫn tuyệt đối tới file
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    weight_path = os.path.join(base_dir, 'weights', filename)
    
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Không tìm thấy file trọng số: {weight_path}")
    
    return weight_path


def get_device_info():
    """
    Lấy thông tin về thiết bị (CPU/GPU)
    
    Returns:
        dict chứa:
            - device: 'cuda' hoặc 'cpu'
            - memory_gb: dung lượng VRAM (nếu GPU)
            - message: chuỗi thông báo
    """
    if torch.cuda.is_available():
        device = 'cuda'
        gpu_name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        message = f"✓ GPU: {gpu_name} ({memory_gb:.1f}GB VRAM)"
        return {
            'device': device,
            'memory_gb': memory_gb,
            'message': message
        }
    else:
        device = 'cpu'
        message = "✓ CPU mode (GPU không khả dụng)"
        return {
            'device': device,
            'memory_gb': None,
            'message': message
        }


def ensure_dir_exists(directory):
    """
    Tạo thư mục nếu chưa tồn tại
    
    Args:
        directory: đường dẫn thư mục
    """
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def draw_box(frame, box, label, color, thickness=2, draw_label=True, font_scale=0.6):
    """
    Vẽ bounding box lên frame
    
    Args:
        frame: ảnh frame (H x W x 3)
        box: [x1, y1, x2, y2] hoặc [[x1,y1], [x2,y2]]
        label: chuỗi nhãn
        color: màu BGR tuple (B, G, R)
        thickness: độ dày đường
        draw_label: có vẽ label không
        font_scale: kích thước font
    
    Returns:
        frame đã vẽ
    """
    # Chuyển box về format [x1, y1, x2, y2]
    if isinstance(box, list) and len(box) == 2:
        # Format [[x1,y1], [x2,y2]]
        x1, y1 = int(box[0][0]), int(box[0][1])
        x2, y2 = int(box[1][0]), int(box[1][1])
    else:
        # Format [x1, y1, x2, y2]
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    
    # Vẽ box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    
    # Vẽ label nếu yêu cầu
    if draw_label and label:
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
        label_y = max(y1 - 5, 15)
        
        # Background cho label
        cv2.rectangle(
            frame,
            (x1, label_y - label_size[1] - 5),
            (x1 + label_size[0], label_y),
            color,
            -1
        )
        
        # Text label
        cv2.putText(
            frame,
            label,
            (x1, label_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
    
    return frame


def draw_text(frame, text, position, font_scale=0.7, color=(0, 0, 0), 
              thickness=1, bg_color=None, font=cv2.FONT_HERSHEY_SIMPLEX):
    """
    Vẽ text lên frame với option background
    
    Args:
        frame: ảnh frame
        text: chuỗi text
        position: (x, y) vị trí vẽ
        font_scale: kích thước font
        color: màu text BGR
        thickness: độ dày chữ
        bg_color: màu background BGR (hoặc None để không vẽ background)
        font: font type
    
    Returns:
        frame đã vẽ
    """
    x, y = position
    x, y = int(x), int(y)
    
    # Tính kích thước text
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_width = text_size[0]
    text_height = text_size[1]
    
    # Vẽ background nếu có
    if bg_color is not None:
        padding = 2
        cv2.rectangle(
            frame,
            (x - padding, y - text_height - padding),
            (x + text_width + padding, y + padding),
            bg_color,
            -1
        )
    
    # Vẽ text
    cv2.putText(
        frame,
        text,
        (x, y),
        font,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA
    )
    
    return frame


def normalize_plate_text(text):
    """
    Chuẩn hóa text biển số xe Việt Nam
    
    Args:
        text: chuỗi text thô từ OCR
    
    Returns:
        chuỗi text chuẩn hóa
    """
    if not text:
        return ""
    
    # Chuyển thành uppercase
    text = text.upper().strip()
    
    # Loại bỏ khoảng trắng thừa
    text = ' '.join(text.split())
    
    # Thay thế các ký tự nhầm lẫn thường gặp
    replacements = {
        'O': '0',  # Chữ O → số 0
        'I': '1',  # Chữ I → số 1
        'S': '5',  # Chữ S → số 5
        'L': '1',  # Chữ L → số 1
        'Z': '2',  # Chữ Z → số 2
    }
    
    # Chỉ thay thế những ký tự không phải chữ cái hợp lệ
    for old, new in replacements.items():
        # Giữ lại chữ cái nếu nó có thể là phần của biển số
        pass
    
    return text


def get_center_point(box):
    """
    Lấy điểm tâm của bounding box
    
    Args:
        box: [x1, y1, x2, y2]
    
    Returns:
        (cx, cy)
    """
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return (cx, cy)


def get_box_area(box):
    """
    Tính diện tích của bounding box
    
    Args:
        box: [x1, y1, x2, y2]
    
    Returns:
        diện tích (int)
    """
    x1, y1, x2, y2 = box
    return (x2 - x1) * (y2 - y1)


def crop_box_from_frame(frame, box, padding=0):
    """
    Cắt vùng ảnh từ bounding box
    
    Args:
        frame: ảnh frame
        box: [x1, y1, x2, y2]
        padding: thêm padding xung quanh crop
    
    Returns:
        ảnh crop
    """
    x1, y1, x2, y2 = box
    
    h, w = frame.shape[:2]
    
    # Thêm padding
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    
    return frame[int(y1):int(y2), int(x1):int(x2)]
