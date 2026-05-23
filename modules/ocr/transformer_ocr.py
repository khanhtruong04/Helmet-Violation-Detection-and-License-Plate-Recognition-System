"""
OCR Module 4: Transformer OCR

Phương pháp: Attention-based Transformer sequence recognition
"""

import cv2
import torch
import numpy as np
from torch import nn
from utils.helpers import get_weight_path, normalize_plate_text


class TransformerOCRModel(nn.Module):
    """Kiến trúc Transformer cho OCR"""
    
    def __init__(self, num_classes=37, d_model=512, nhead=8, num_layers=6):
        """
        Args:
            num_classes: số class output (37 = 10 digits + 26 letters + blank)
            d_model: dimensionality của model
            nhead: số attention heads
            num_layers: số transformer layers
        """
        super(TransformerOCRModel, self).__init__()
        self.d_model = d_model
        
        # CNN feature extraction
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),
            
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2048,
            batch_first=True,
            dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output layer
        self.fc = nn.Linear(d_model, num_classes)
    
    def forward(self, input):
        """Forward pass"""
        # CNN extraction
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        
        # Reshape for transformer
        conv = conv.permute(0, 2, 3, 1).reshape(b, -1, c)  # (b, h*w, c)
        
        # Pad or project to d_model if needed
        if c != self.d_model:
            conv = nn.Linear(c, self.d_model)(conv)
        
        # Transformer
        transformer_out = self.transformer_encoder(conv)
        
        # Output
        output = self.fc(transformer_out)
        
        return output


class TransformerOCRModule:
    """OCR sử dụng Transformer"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """Khởi tạo Transformer OCR module"""
        self.device = device
        self.model = None
        self.char_list = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        self._load_model()
    
    def _load_model(self):
        """Load mô hình Transformer"""
        try:
            self.model = TransformerOCRModel()
            weight_path = get_weight_path('transformer_ocr_best.pt')
            checkpoint = torch.load(weight_path, map_location=self.device)
            
            # Handle wrapped checkpoint format
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                checkpoint = checkpoint['model_state_dict']
            
            self.model.load_state_dict(checkpoint, strict=False)
            self.model.to(self.device)
            self.model.eval()
            print(f"✓ Transformer OCR model loaded from {weight_path}")
        except Exception as e:
            print(f"⚠ Lỗi load Transformer model: {e}")
            print(f"  Fallback: Transformer disabled, use EasyOCR instead")
            self.model = None  # Mark as failed
    
    def preprocess_plate(self, plate_crop):
        """
        Tiền xử lý ảnh biển số
        
        Args:
            plate_crop: ảnh biển số đã crop
        
        Returns:
            tensor đã xử lý
        """
        # Resize to model input size
        imgH = 32
        h, w = plate_crop.shape[:2]
        imgW = int(w * imgH / h)
        
        resized = cv2.resize(plate_crop, (imgW, imgH))
        
        # Normalize to [0, 1]
        img_tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        return img_tensor
    
    def decode_prediction(self, output):
        """
        Decode Transformer output to text
        
        Args:
            output: model output (batch_size, seq_len, num_class)
        
        Returns:
            text
        """
        output = output.cpu().data.numpy()
        argmax_indices = np.argmax(output[0], axis=1)
        
        text = ""
        for idx in argmax_indices:
            if idx > 0 and idx < len(self.char_list):  # 0 is blank/padding
                text += self.char_list[idx]
        
        return text
    
    def recognize(self, plate_crop):
        """
        Nhận dạng biển số từ ảnh crop
        
        Args:
            plate_crop: ảnh biển số đã crop
        
        Returns:
            chuỗi biển số
        """
        if self.model is None:
            return ""  # Fallback: return empty if model failed to load
        
        try:
            # Tiền xử lý
            img_tensor = self.preprocess_plate(plate_crop)
            
            # Inference
            with torch.no_grad():
                output = self.model(img_tensor)
            
            # Decode
            text = self.decode_prediction(output)
            
            # Chuẩn hóa
            return normalize_plate_text(text)
        except Exception as e:
            print(f"⚠ Lỗi recognize Transformer: {e}")
            return ""
