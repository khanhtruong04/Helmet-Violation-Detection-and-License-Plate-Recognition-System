"""
OCR Module 3: CRNN (Convolutional Recurrent Neural Network)

Phương pháp: CNN + BiLSTM + CTC - phù hợp với chuỗi ký tự liên tục
"""

import cv2
import torch
import numpy as np
from torch import nn
from utils.helpers import get_weight_path, normalize_plate_text


class CRNNModel(nn.Module):
    """Kiến trúc CRNN cho OCR - match với notebook training"""
    
    def __init__(self, imgH=32, nc=3, nclass=37, nh=256):
        """
        Args:
            imgH: chiều cao ảnh input
            nc: số channel input (3 cho RGB)
            nclass: số class output (37 = 10 digits + 26 letters + blank)
            nh: số hidden units trong LSTM
        """
        super(CRNNModel, self).__init__()
        self.imgH = imgH
        
        # CNN feature extraction (match training notebook)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),
            
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),
            
            nn.Conv2d(512, 512, 2, 1, 0),
            nn.ReLU(inplace=True),
        )
        
        # RNN (2x BiLSTM layers)
        self.rnn1 = nn.LSTM(512, nh, bidirectional=True, batch_first=False)
        self.rnn2 = nn.LSTM(nh * 2, nh, bidirectional=True, batch_first=False)
        
        # Output layer
        self.fc = nn.Linear(nh * 2, nclass)
    
    def forward(self, input):
        """Forward pass"""
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, f"Height must be 1, got {h}"
        
        # Reshape for RNN: (b, c, w) -> (w, b, c)
        conv = conv.squeeze(2)  # (b, c, w)
        conv = conv.permute(2, 0, 1)  # (w, b, c) - time-first for LSTM
        
        # 2 LSTM layers
        rnn_out, _ = self.rnn1(conv)
        rnn_out, _ = self.rnn2(rnn_out)
        
        # Output: (w, b, num_classes)
        output = self.fc(rnn_out)
        
        return output


class CRNNOCRModule:
    """OCR sử dụng CRNN"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """Khởi tạo CRNN module"""
        self.device = device
        self.model = None
        self.char_list = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        self._load_model()
    
    def _load_model(self):
        """Load mô hình CRNN"""
        try:
            self.model = CRNNModel()
            weight_path = get_weight_path('crnn_ocr_best.pth')
            checkpoint = torch.load(weight_path, map_location=self.device)
            
            # Handle wrapped checkpoint format
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                checkpoint = checkpoint['model_state_dict']
            
            self.model.load_state_dict(checkpoint, strict=True)
            self.model.to(self.device)
            self.model.eval()
            print(f"✓ CRNN model loaded from {weight_path}")
        except Exception as e:
            print(f"⚠ Lỗi load CRNN model: {e}")
            print(f"  Fallback: CRNN disabled, use EasyOCR instead")
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
        Decode CTC output to text
        
        Args:
            output: model output (batch_size, seq_len, num_class)
        
        Returns:
            text
        """
        output = output.cpu().data.numpy()
        argmax_indices = np.argmax(output[0], axis=1)
        
        text = ""
        last_idx = 0
        for idx in argmax_indices:
            if idx != 0 and idx != last_idx:  # 0 is blank
                if idx < len(self.char_list):
                    text += self.char_list[idx]
            last_idx = idx
        
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
            print(f"⚠ Lỗi recognize CRNN: {e}")
            return ""
