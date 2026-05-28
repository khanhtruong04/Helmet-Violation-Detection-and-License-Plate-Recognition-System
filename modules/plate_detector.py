"""
Module phát hiện biển số xe

Sử dụng YOLOv8 model để phát hiện vị trí biển số
"""

import cv2
import torch
import numpy as np
from ultralytics import YOLO
from utils.helpers import get_weight_path, crop_box_from_frame


class PlateDetector:
    """Phát hiện vị trí biển số xe"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Khởi tạo detector
        
        Args:
            device: 'cuda' hoặc 'cpu'
        """
        self.device = device
        self.model = None
        self.conf_threshold = 0.5
        
        self._load_model()
    
    def _load_model(self):
        """Load mô hình YOLOv8 phát hiện biển số"""
        try:
            weight_path = get_weight_path('best_lp_detector.pt')
            self.model = YOLO(weight_path)
            self.model.to(self.device)
            print(f"✓ License Plate Detector (YOLOv8) loaded from {weight_path}")
        except Exception as e:
            print(f"❌ Lỗi load plate detector: {e}")
            raise
    
    def detect(self, frame, conf_threshold=None):
        """
        Phát hiện biển số xe từ frame
        
        Args:
            frame: ảnh frame (H x W x 3)
            conf_threshold: ngưỡng confidence (mặc định 0.5)
        
        Returns:
            list các detection dạng:
            [{
                'box': [x1, y1, x2, y2],
                'confidence': float,
                'crop': ảnh biển số đã cắt
            }, ...]
        """
        if self.model is None:
            return []
        
        if conf_threshold is None:
            conf_threshold = self.conf_threshold
        
        try:
            # Chạy inference
            results = self.model(frame, conf=conf_threshold, device=self.device, verbose=False)
            
            detections = []
            
            for result in results:
                if result.boxes is None or len(result.boxes) == 0:
                    continue
                
                for box in result.boxes:
                    # Lấy tọa độ box
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = xyxy
                    
                    # Lấy confidence
                    conf = float(box.conf[0].cpu().numpy())
                    
                    # Cắt vùng biển số
                    plate_box = [x1, y1, x2, y2]
                    plate_crop = crop_box_from_frame(frame, plate_box, padding=5)
                    
                    detections.append({
                        'box': plate_box,
                        'confidence': conf,
                        'crop': plate_crop
                    })
            
            return detections
            
        except Exception as e:
            print(f"❌ Lỗi phát hiện biển số: {e}")
            return []
