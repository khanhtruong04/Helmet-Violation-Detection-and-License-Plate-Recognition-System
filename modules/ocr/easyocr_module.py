"""
OCR Module 2: EasyOCR

Phương pháp: Nhận dạng toàn bộ vùng biển số trực tiếp
"""

import easyocr
from utils.helpers import normalize_plate_text


class EasyOCRModule:
    """OCR sử dụng EasyOCR"""
    
    def __init__(self, languages=['en']):
        """
        Khởi tạo EasyOCR reader
        
        Args:
            languages: danh sách ngôn ngữ hỗ trợ
        """
        self.reader = None
        self.languages = languages
        self._init_reader()
    
    def _init_reader(self):
        """Khởi tạo reader EasyOCR"""
        try:
            self.reader = easyocr.Reader(self.languages)
            print(f"[OK] EasyOCR Reader initialized with languages: {self.languages}")
        except Exception as e:
            print(f"[ERROR] EasyOCR init failed: {e}")
            self.reader = None  # Mark as failed
    
    def preprocess_plate(self, plate_crop):
        """
        Tiền xử lý ảnh biển số
        
        Args:
            plate_crop: ảnh biển số đã crop
        
        Returns:
            ảnh đã xử lý
        """
        import cv2
        import numpy as np
        
        # Nếu ảnh quá nhỏ, phóng to
        h, w = plate_crop.shape[:2]
        if w < 100:
            scale = max(1, 100 / w)
            plate_crop = cv2.resize(plate_crop, None, fx=scale, fy=scale)
        
        return plate_crop
    
    def recognize(self, plate_crop):
        """
        Nhận dạng biển số từ ảnh crop
        
        Args:
            plate_crop: ảnh biển số đã crop
        
        Returns:
            chuỗi biển số
        """
        if self.reader is None:
            return ""
        
        try:
            # Tiền xử lý
            processed = self.preprocess_plate(plate_crop)
            
            # Chạy OCR
            results = self.reader.readtext(processed)
            
            if not results:
                return ""
            
            # Lấy text từ kết quả có confidence cao nhất
            plate_text = ""
            for (bbox, text, conf) in results:
                if conf > 0.3:  # Filter by confidence
                    plate_text += text
            
            # Chuẩn hóa
            return normalize_plate_text(plate_text) if plate_text else "???"
        except Exception as e:
            print(f"[WARN] EasyOCR recognize error: {e}")
            return ""
