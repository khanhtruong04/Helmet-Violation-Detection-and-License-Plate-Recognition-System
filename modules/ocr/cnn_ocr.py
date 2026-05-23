"""
OCR Module 1: CNN Character Classifier

Phương pháp: Tách từng ký tự → phân loại từng ký tự → ghép lại
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from utils.helpers import get_weight_path, normalize_plate_text


class CNNCharacterClassifier:
    """OCR sử dụng CNN phân loại ký tự"""
    
    def __init__(self):
        """Khởi tạo classifier"""
        self.model = None
        self.char_list = None
        self._load_model()
    
    def _load_model(self):
        """Load mô hình Keras .h5"""
        try:
            weight_path = get_weight_path('character_classifier.h5')
            self.model = keras.models.load_model(weight_path)
            print(f"✓ CNN Classifier loaded from {weight_path}")
            
            # Định nghĩa danh sách ký tự biển số Việt Nam
            self.char_list = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        except Exception as e:
            print(f" Lỗi load CNN Classifier: {e}")
            raise
    
    def preprocess_plate(self, plate_crop):
        """
        Tiền xử lý ảnh biển số
        
        Args:
            plate_crop: ảnh biển số đã crop
        
        Returns:
            ảnh đã xử lý
        """
        # Resize về chiều cao chuẩn
        h, w = plate_crop.shape[:2]
        target_h = 50
        new_w = int(w * target_h / h)
        resized = cv2.resize(plate_crop, (new_w, target_h))
        
        # Chuyển sang grayscale
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
        # Cân bằng histogram
        equalized = cv2.equalizeHist(gray)
        
        return equalized
    
    def segment_characters(self, plate_image):
        """
        Tách từng ký tự từ ảnh biển số
        
        Args:
            plate_image: ảnh biển số grayscale đã xử lý
        
        Returns:
            list của ảnh ký tự
        """
        # Binary threshold
        _, binary = cv2.threshold(plate_image, 150, 255, cv2.THRESH_BINARY)
        
        # Tìm contour
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        characters = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 5 and h > 10:  # Filter by size
                char_img = plate_image[y:y+h, x:x+w]
                characters.append((x, char_img))
        
        # Sort by x coordinate
        characters.sort(key=lambda c: c[0])
        return [c[1] for c in characters]
    
    def classify_character(self, char_image):
        """
        Phân loại một ký tự
        
        Args:
            char_image: ảnh ký tự
        
        Returns:
            ký tự được phân loại (string)
        """
        # Resize to model input size (e.g., 28x28)
        char_resized = cv2.resize(char_image, (28, 28))
        char_normalized = char_resized.astype('float32') / 255.0
        char_input = np.expand_dims(np.expand_dims(char_normalized, axis=0), axis=-1)
        
        # Predict
        prediction = self.model.predict(char_input, verbose=0)
        class_id = np.argmax(prediction[0])
        
        return self.char_list[class_id] if class_id < len(self.char_list) else '?'
    
    def recognize(self, plate_crop):
        """
        Nhận dạng biển số từ ảnh crop
        
        Args:
            plate_crop: ảnh biển số đã crop
        
        Returns:
            chuỗi biển số
        """
        try:
            # Tiền xử lý
            processed = self.preprocess_plate(plate_crop)
            
            # Tách ký tự
            characters = self.segment_characters(processed)
            
            if not characters:
                return ""
            
            # Phân loại từng ký tự
            plate_text = ""
            for char_img in characters:
                char = self.classify_character(char_img)
                plate_text += char
            
            # Chuẩn hóa
            return normalize_plate_text(plate_text)
        except Exception as e:
            print(f"❌ Lỗi recognize CNN: {e}")
            return ""
