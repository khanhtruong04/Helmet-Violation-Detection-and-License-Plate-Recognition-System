"""
Drawing utilities cho annotation
"""

import cv2
from utils.helpers import draw_box, draw_text


def annotate_frame_with_helmet_detections(frame, detections):
    """Vẽ helmet detections lên frame (không hiển thị label)"""
    for det in detections:
        box = det['box']
        label = det['label']
        
        # Đặt màu theo label (match Colab matplotlib colors → BGR)
        if label == 'helmet':
            color = (80, 175, 76)        # Xanh lá: #4CAF50
        elif label == 'no_helmet':
            color = (54, 67, 244)        # Đỏ: #F44336
        elif label == 'rider':
            color = (243, 150, 33)       # Xanh dương: #2196F3
        else:
            color = (255, 255, 255)
        
        # Vẽ box (KHÔNG hiển thị label)
        frame = draw_box(frame, box, '', color, thickness=2, draw_label=False)
    
    return frame


def annotate_frame_with_plate_detections(frame, detections):
    """Vẽ plate detections lên frame (không hiển thị label)"""
    for det in detections:
        box = det['box']
        color = (7, 193, 255)  # Vàng: #FFC107 (BGR format)
        # Vẽ box (KHÔNG hiển thị label 'plate')
        frame = draw_box(frame, box, '', color, thickness=2, draw_label=False)
    
    return frame


def annotate_frame_with_ocr_text(frame, plate_texts):
    """Vẽ OCR text lên frame (nền vàng, chữ đen)"""
    for plate_text in plate_texts:
        box = plate_text['box']
        text = plate_text['text']
        
        x1, y1, x2, y2 = box
        
        # Vẽ text phía trên box (giống như ví dụ Colab)
        # Đảm bảo text không bị cắt phía trên màn hình
        text_y = max(30, int(y1) - 5)
        
        frame = draw_text(
            frame,
            text,
            (int(x1), text_y),
            font_scale=0.8,
            color=(0, 0, 0),           # Chữ đen
            thickness=2,
            bg_color=(7, 193, 255)     # Nền vàng: #FFC107 (BGR format)
        )
    
    return frame
