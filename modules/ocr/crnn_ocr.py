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
        # MATCH COLAB TRAINING: 36 chars + blank = 37 classes
        self.char_list = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        # Create idx2char mapping: idx 0 = blank, idx 1-36 = chars
        self.idx2char = {i + 1: c for i, c in enumerate(self.char_list)}
        self._load_model()
    
    def _load_model(self):
        """Load mô hình CRNN"""
        try:
            self.model = CRNNModel()
            weight_path = get_weight_path('crnn_ocr_best.pth')
            checkpoint = torch.load(weight_path, map_location=self.device)
            
            # Load model state dict
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
                # Load idx2char từ checkpoint (IMPORTANT!)
                if 'idx2char' in checkpoint:
                    self.idx2char = checkpoint['idx2char']
                    print(f"[OK] Loaded idx2char from checkpoint: {len(self.idx2char)} classes")
            else:
                self.model.load_state_dict(checkpoint, strict=True)
            
            self.model.to(self.device)
            self.model.eval()
            print(f"[OK] CRNN model loaded from {weight_path}")
        except Exception as e:
            print(f"[ERROR] Load CRNN model failed: {e}")
            print(f"  Fallback: CRNN disabled, use EasyOCR instead")
            self.model = None  # Mark as failed
    
    def preprocess_plate(self, plate_crop):
        """
        Tiền xử lý ảnh biển số - Match với Colab training
        
        Args:
            plate_crop: ảnh biển số đã crop (BGR format)
        
        Returns:
            tensor đã xử lý (normalized to [-1, 1])
        """
        # Convert BGR → RGB
        img_rgb = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)
        
        h, w = img_rgb.shape[:2]
        
        # Nếu tỷ lệ cao > 0.6 → split thành 2 nửa (top/bottom) rồi ghép lại
        if h / w > 0.6:
            mid = h // 2
            top, bot = img_rgb[:mid], img_rgb[mid:]
            th = max(top.shape[0], bot.shape[0])
            top = cv2.resize(top, (int(top.shape[1] * th / top.shape[0]), th))
            bot = cv2.resize(bot, (int(bot.shape[1] * th / bot.shape[0]), th))
            img_rgb = np.hstack([top, bot])
        
        # Convert RGB → Grayscale
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        
        # CLAHE: Contrast Limited Adaptive Histogram Equalization
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        gray = clahe.apply(gray)
        
        # Denoising
        gray = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
        
        # Adaptive Threshold
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, 8
        )
        
        # Resize to model input (32, 128)
        imgH, imgW = 32, 128
        h, w = binary.shape[:2]
        new_w = int(w * imgH / h)
        
        if new_w > imgW:
            binary = cv2.resize(binary, (imgW, imgH))
        else:
            binary = cv2.resize(binary, (new_w, imgH))
            # Padding với trắng (255)
            binary = cv2.copyMakeBorder(
                binary, 0, 0, 0, imgW - new_w,
                cv2.BORDER_CONSTANT, value=255
            )
        
        # Stack 3 channels (binary → 3 channel)
        binary_3ch = np.stack([binary, binary, binary], axis=-1)
        
        # Convert to tensor: [0, 1]
        img_tensor = torch.from_numpy(binary_3ch).float() / 255.0
        
        # Normalize to [-1, 1] (MATCH TRAINING)
        img_tensor = (img_tensor - 0.5) / 0.5
        
        # Add batch dimension
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        return img_tensor
    
    def decode_prediction(self, output):
        """
        Decode CTC output to text - MATCH COLAB EXACTLY
        
        Args:
            output: model output (seq_len, batch_size, num_class)
        
        Returns:
            text
        """
        # Apply log softmax and argmax (same as Colab)
        import torch.nn.functional as F
        log_probs = F.log_softmax(output, dim=-1)
        preds = log_probs.argmax(dim=-1).squeeze(1).cpu().numpy()  # (seq_len,)
        
        # Decode: CTC logic from Colab
        # Skip blank (0) and consecutive duplicates
        chars = []
        prev = None
        BLANK_IDX = 0
        
        for p in preds:
            p_int = int(p)
            # If not blank AND not same as previous → add char
            if p_int != prev and p_int != BLANK_IDX:
                char = self.idx2char.get(p_int, '')
                if char:
                    chars.append(char)
            prev = p_int
        
        return ''.join(chars) if chars else '???'
    
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
