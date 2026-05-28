"""
Module phát hiện mũ bảo hiểm

Hỗ trợ 2 mô hình:
1. YOLOv8 - tốc độ cao, phù hợp real-time
2. Faster R-CNN - độ chính xác cao
"""

import cv2
import torch
import numpy as np
from ultralytics import YOLO
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from utils.helpers import get_weight_path, draw_box, draw_text


class HelmetDetector:
    """Phát hiện mũ bảo hiểm và motorbike"""
    
    # Màu sắc theo quy ước (BGR) - Match Colab colors
    COLORS = {
        'helmet': (80, 175, 76),        # Xanh lá: #4CAF50
        'no_helmet': (54, 67, 244),     # Đỏ: #F44336
        'rider': (243, 150, 33),        # Xanh dương: #2196F3
        'motorbike': (243, 150, 33),    # Xanh dương: #2196F3
    }
    
    def __init__(self, model_type='yolov8', device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Khởi tạo detector
        
        Args:
            model_type: 'yolov8' hoặc 'faster_rcnn'
            device: 'cuda' hoặc 'cpu'
        """
        self.model_type = model_type.lower()
        self.device = device
        self.model = None
        self.conf_threshold = 0.5
        
        self._load_model()
    
    def _load_model(self):
        """Load mô hình dựa vào model_type"""
        if self.model_type == 'yolov8':
            self._load_yolov8()
        elif self.model_type == 'faster_rcnn':
            self._load_faster_rcnn()
        else:
            raise ValueError(f"Model type {self.model_type} không được hỗ trợ")
    
    def _load_yolov8(self):
        """Load mô hình YOLOv8"""
        try:
            weight_path = get_weight_path('helmet_detection_yolov8.pt')
            self.model = YOLO(weight_path)
            self.model.to(self.device)
            print(f"✓ YOLOv8 model loaded from {weight_path}")
        except Exception as e:
            print(f"❌ Lỗi load YOLOv8: {e}")
            raise
    
    def _load_faster_rcnn(self):
        """Load mô hình Faster R-CNN với MobileNetV3 backbone từ file .pth"""
        try:
            # Build model với MobileNetV3 backbone (4 classes: background, helmet, no_helmet, rider)
            # Tương tự notebook lp-detection-helmet-fasterrcnn-ipynb.ipynb
            self.model = fasterrcnn_mobilenet_v3_large_fpn(
                weights='DEFAULT',
                weights_backbone='DEFAULT'
            )
            
            # Thay đổi FC layer để match số class (4: background, helmet, no_helmet, rider/motorbike)
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=4)
            
            # Load weights từ file .pth
            try:
                weight_path = get_weight_path('helmet_detection_fasterrcnn.pth')
                checkpoint = torch.load(weight_path, map_location=self.device)
                
                # Load state_dict
                incompatible_keys = self.model.load_state_dict(checkpoint, strict=False)
                
                if incompatible_keys.missing_keys:
                    print(f"⚠ Missing keys ({len(incompatible_keys.missing_keys)} keys)")
                
                if incompatible_keys.unexpected_keys:
                    print(f"⚠ Unexpected keys ({len(incompatible_keys.unexpected_keys)} keys)")
                
                print(f"✓ Faster R-CNN (MobileNetV3) weights loaded from {weight_path}")
                    
            except FileNotFoundError:
                print(f"⚠ Faster R-CNN weights không tìm thấy tại {weight_path}")
                print("  → Dùng pre-trained MobileNetV3 từ ImageNet")
            except Exception as e:
                print(f"⚠ Lỗi load Faster R-CNN weights: {str(e)[:100]}...")
                print("  → Dùng pre-trained MobileNetV3 từ ImageNet")
            
            self.model.to(self.device)
            self.model.eval()
            print("✓ Faster R-CNN (MobileNetV3) model loaded")
        except Exception as e:
            print(f"❌ Lỗi load Faster R-CNN: {e}")
            raise
    
    def detect(self, frame):
        """
        Phát hiện mũ bảo hiểm trong frame
        
        Args:
            frame: ảnh frame (numpy array, H x W x 3)
        
        Returns:
            list của detections: [
                {
                    'box': (x1, y1, x2, y2),
                    'label': 'helmet' hoặc 'no_helmet',
                    'confidence': float,
                    'class_id': int
                },
                ...
            ]
        """
        if self.model is None:
            return []
        
        if self.model_type == 'yolov8':
            return self._detect_yolov8(frame)
        else:
            return self._detect_faster_rcnn(frame)
    
    def _detect_yolov8(self, frame):
        """Phát hiện sử dụng YOLOv8"""
        detections = []
        
        try:
            # Chạy inference
            results = self.model(frame, conf=self.conf_threshold, verbose=False)
            
            # Parse results
            for result in results:
                if result.boxes is not None:
                    for box, conf, cls in zip(
                        result.boxes.xyxy,
                        result.boxes.conf,
                        result.boxes.cls
                    ):
                        x1, y1, x2, y2 = box.cpu().numpy()
                        conf_score = conf.cpu().item()
                        class_id = int(cls.cpu().item())
                        
                        # Map class ID to label
                        label = 'helmet' if class_id == 0 else 'no_helmet'
                        
                        detections.append({
                            'box': (x1, y1, x2, y2),
                            'label': label,
                            'confidence': conf_score,
                            'class_id': class_id
                        })
        except Exception as e:
            print(f" Lỗi inference YOLOv8: {e}")
        
        return detections
    
    def _detect_faster_rcnn(self, frame):
        """Phát hiện sử dụng Faster R-CNN (4 classes: helmet, no_helmet, rider/motorbike)"""
        detections = []
        
        try:
            # Prepare input
            img_tensor = F.to_tensor(frame).unsqueeze(0).to(self.device)
            
            # Chạy inference
            with torch.no_grad():
                predictions = self.model(img_tensor)
            
            # Parse results
            pred = predictions[0]
            boxes = pred['boxes'].cpu().numpy()
            scores = pred['scores'].cpu().numpy()
            labels = pred['labels'].cpu().numpy()
            
            # Filter by confidence threshold
            mask = scores >= self.conf_threshold
            boxes = boxes[mask]
            scores = scores[mask]
            labels = labels[mask]
            
            for box, score, label_id in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box
                
                # Map label ID (1=helmet, 2=no_helmet, 3=rider/motorbike)
                if label_id == 1:
                    label = 'helmet'
                elif label_id == 2:
                    label = 'no_helmet'
                elif label_id == 3:
                    label = 'rider'
                else:
                    continue  # Skip background
                
                detections.append({
                    'box': (x1, y1, x2, y2),
                    'label': label,
                    'confidence': float(score),
                    'class_id': int(label_id)
                })
        except Exception as e:
            print(f"❌ Lỗi inference Faster R-CNN: {e}")
        
        return detections
    
    def draw_detections(self, frame, detections):
        """
        Vẽ bounding box lên frame (không hiển thị label)
        
        Args:
            frame: ảnh frame
            detections: danh sách detection từ detect()
        
        Returns:
            frame với annotations
        """
        for det in detections:
            box = det['box']
            label = det['label']
            color = self.COLORS.get(label, (255, 255, 255))
            
            # Vẽ box chỉ (KHÔNG hiển thị label)
            frame = draw_box(frame, box, '', color, thickness=2, draw_label=False)
        
        return frame
    
    def set_confidence_threshold(self, threshold):
        """Đặt ngưỡng confidence"""
        self.conf_threshold = threshold
