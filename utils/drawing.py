"""
Drawing utilities cho annotation
"""

import cv2
from utils.helpers import draw_box, draw_text


def annotate_frame_with_helmet_detections(frame, detections):
    """Vẽ helmet detections lên frame"""
    for det in detections:
        box = det['box']
        label = det['label']
        conf = det['confidence']
        
        color = (0, 255, 0) if label == 'helmet' else (0, 0, 255)
        frame = draw_box(frame, box, label, color)
        
        x1, y1, _, _ = box
        text = f"{label} {conf:.2f}"
        frame = draw_text(frame, text, (int(x1), int(y1) - 10), color=color)
    
    return frame


def annotate_frame_with_plate_detections(frame, detections):
    """Vẽ plate detections lên frame"""
    for det in detections:
        box = det['box']
        conf = det['confidence']
        
        color = (0, 255, 255)  # Vàng
        frame = draw_box(frame, box, 'plate', color)
        # Không vẽ text confidence để tránh đè lên OCR text
    
    return frame


def annotate_frame_with_ocr_text(frame, plate_texts):
    """Vẽ OCR text lên frame"""
    for plate_text in plate_texts:
        box = plate_text['box']
        text = plate_text['text']
        
        x1, y1, _, _ = box
        # Đảm bảo y position không âm - nếu quá cao thì vẽ phía dưới box
        text_y = max(20, int(y1) - 30)  # Tối thiểu 20px từ top
        frame = draw_text(
            frame,
            text,
            (int(x1), text_y),
            color=(0, 0, 0),
            bg_color=(0, 255, 255)
        )
    
    return frame
