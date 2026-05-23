"""
Module phát hiện biển số xe

Sử dụng YOLOv8 để phát hiện vị trí biển số trong frame
"""

import cv2
import torch
from ultralytics import YOLO
from utils.helpers import get_weight_path, draw_box, draw_text, crop_region


class PlateDetector:
    """Phát hiện vị trí biển số xe"""
    
    # Màu sắc theo quy ước
    COLOR = (0, 255, 255)  # Vàng
    
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
        """Load mô hình YOLO phát hiện biển số"""
        try:
            weight_path = get_weight_path('best_lp_detector.pt')
            self.model = YOLO(weight_path)
            self.model.to(self.device)
            print(f"✓ Plate Detector model loaded from {weight_path}")
        except Exception as e:
            print(f"❌ Lỗi load plate detector: {e}")
            raise
    
    def detect(self, frame):
        """
        Phát hiện biển số trong frame
        
        Args:
            frame: ảnh frame (numpy array, H x W x 3)
        
        Returns:
            list của detections: [
                {
                    'box': (x1, y1, x2, y2),
                    'confidence': float,
                    'crop': ảnh crop biển số
                },
                ...
            ]
        """
        if self.model is None:
            return []
        
        detections = []
        
        try:
            # Chạy inference
            results = self.model(frame, conf=self.conf_threshold, verbose=False)
            
            # Parse results
            for result in results:
                if result.boxes is not None:
                    for box, conf in zip(
                        result.boxes.xyxy,
                        result.boxes.conf
                    ):
                        x1, y1, x2, y2 = box.cpu().numpy()
                        conf_score = conf.cpu().item()
                        
                        # Crop vùng biển số
                        plate_crop = crop_region(frame, (x1, y1, x2, y2))
                        
                        detections.append({
                            'box': (x1, y1, x2, y2),
                            'confidence': conf_score,
                            'crop': plate_crop
                        })
        except Exception as e:
            print(f"❌ Lỗi inference plate detector: {e}")
        
        return detections
    
    def draw_detections(self, frame, detections, draw_crop_area=False):
        """
        Vẽ bounding box biển số lên frame (DEPRECATED)
        
        Sử dụng drawing.annotate_frame_with_plate_detections() thay vào đó
        để tránh vẽ box trùng lặp
        
        Args:
            frame: ảnh frame
            detections: danh sách detection từ detect()
            draw_crop_area: có vẽ vùng crop hay không
        
        Returns:
            frame với annotations
        """
        # Đã được thay thế bởi drawing.annotate_frame_with_plate_detections()
        return frame
    
    def set_confidence_threshold(self, threshold):
        """Đặt ngưỡng confidence"""
        self.conf_threshold = threshold
