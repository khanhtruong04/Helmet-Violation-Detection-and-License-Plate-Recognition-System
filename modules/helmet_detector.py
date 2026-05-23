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
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models import ResNet101_Weights
from torchvision.ops import MultiScaleRoIAlign
from torchvision.transforms import functional as F
from utils.helpers import get_weight_path, draw_box, draw_text


class HelmetDetector:
    """Phát hiện mũ bảo hiểm"""
    
    # Màu sắc theo quy ước
    COLORS = {
        'helmet': (0, 255, 0),      # Xanh lá
        'no_helmet': (0, 0, 255),   # Đỏ
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
        """Load mô hình Faster R-CNN với ResNet-101 backbone (giống file training)"""
        try:
            # Build backbone ResNet-101 + FPN (giống như fasterrcnn.ipynb)
            backbone = resnet_fpn_backbone(
                backbone_name='resnet101',
                weights=ResNet101_Weights.IMAGENET1K_V2,
                trainable_layers=3
            )
            
            # Build RPN anchor generator (giống training)
            anchor_generator = AnchorGenerator(
                sizes=((32,), (64,), (128,), (256,), (512,)),
                aspect_ratios=((0.5, 1.0, 2.0),) * 5
            )
            
            # Build ROI pooler
            roi_pooler = MultiScaleRoIAlign(
                featmap_names=['0', '1', '2', '3'],
                output_size=7,
                sampling_ratio=2
            )
            
            # Build Faster R-CNN với ResNet-101
            self.model = FasterRCNN(
                backbone=backbone,
                num_classes=3,  # helmet, no_helmet, background
                rpn_anchor_generator=anchor_generator,
                box_roi_pool=roi_pooler
            )
            
            # Load weights nếu có file
            try:
                weight_path = get_weight_path('helmet_detection_fasterrcnn.pt')
                checkpoint = torch.load(weight_path, map_location=self.device)
                
                # Load state_dict
                incompatible_keys = self.model.load_state_dict(checkpoint, strict=False)
                
                if incompatible_keys.missing_keys:
                    print(f"⚠ Missing keys ({len(incompatible_keys.missing_keys)} keys):")
                    for key in list(incompatible_keys.missing_keys)[:3]:
                        print(f"    - {key}")
                    if len(incompatible_keys.missing_keys) > 3:
                        print(f"    ... và {len(incompatible_keys.missing_keys) - 3} keys khác")
                
                if incompatible_keys.unexpected_keys:
                    print(f"⚠ Unexpected keys ({len(incompatible_keys.unexpected_keys)} keys):")
                    for key in list(incompatible_keys.unexpected_keys)[:3]:
                        print(f"    - {key}")
                    if len(incompatible_keys.unexpected_keys) > 3:
                        print(f"    ... và {len(incompatible_keys.unexpected_keys) - 3} keys khác")
                
                if not incompatible_keys.missing_keys and not incompatible_keys.unexpected_keys:
                    print(f"✓ Faster R-CNN weights loaded successfully from {weight_path}")
                else:
                    print(f"✓ Faster R-CNN weights loaded (ResNet-101) from {weight_path}")
                    
            except FileNotFoundError:
                print(f"⚠ Faster R-CNN weights không tìm thấy tại {weight_path}")
                print("  → Dùng pre-trained ResNet-101 từ ImageNet-1K")
            except Exception as e:
                print(f"⚠ Lỗi load Faster R-CNN weights: {str(e)[:100]}...")
                print("  → Dùng pre-trained ResNet-101 từ ImageNet-1K")
            
            self.model.to(self.device)
            self.model.eval()
            print("✓ Faster R-CNN (ResNet-101) model loaded")
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
            print(f"❌ Lỗi inference YOLOv8: {e}")
        
        return detections
    
    def _detect_faster_rcnn(self, frame):
        """Phát hiện sử dụng Faster R-CNN"""
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
                
                # Map label ID (1=helmet, 2=no_helmet)
                label = 'helmet' if label_id == 1 else 'no_helmet'
                
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
        Vẽ bounding box lên frame
        
        Args:
            frame: ảnh frame
            detections: danh sách detection từ detect()
        
        Returns:
            frame với annotations
        """
        for det in detections:
            box = det['box']
            label = det['label']
            conf = det['confidence']
            color = self.COLORS[label]
            
            # Vẽ box
            frame = draw_box(frame, box, label, color, thickness=2)
            
            # Vẽ text nhãn + confidence
            x1, y1, _, _ = box
            text = f"{label} {conf:.2f}"
            frame = draw_text(frame, text, (int(x1), int(y1) - 10), 
                            color=color, bg_color=(0, 0, 0))
        
        return frame
    
    def set_confidence_threshold(self, threshold):
        """Đặt ngưỡng confidence"""
        self.conf_threshold = threshold
