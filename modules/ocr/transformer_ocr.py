"""
OCR Module 4: Transformer OCR

Phương pháp: Attention-based Transformer sequence recognition
"""

import cv2
import math
import torch
import numpy as np
from torch import nn
from torchvision import transforms
from PIL import Image
from utils.helpers import get_weight_path, normalize_plate_text

# Constants từ notebook
CHARS = '0123456789ABCDEFGHKLMNPRSTUVXYZ'
PAD_TOKEN = '<PAD>'
SOS_TOKEN = '<SOS>'
EOS_TOKEN = '<EOS>'
VOCAB = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN] + list(CHARS)
char2idx = {c: i for i, c in enumerate(VOCAB)}
idx2char_notebook = {i: c for i, c in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)
PAD_IDX = char2idx[PAD_TOKEN]
SOS_IDX = char2idx[SOS_TOKEN]
EOS_IDX = char2idx[EOS_TOKEN]
MAX_LEN = 12


class CNNBackbone(nn.Module):
    """CNN Backbone từ notebook"""
    
    def __init__(self, d_model=256):
        super().__init__()
        
        def block(in_c, out_c, pool_h=True):
            layers = [
                nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
                nn.MaxPool2d((2,2)) if pool_h else nn.MaxPool2d((2,1))
            ]
            return nn.Sequential(*layers)
        
        self.features = nn.Sequential(
            block(3, 64, True),
            block(64, 128, True),
            block(128, 256, False),
            block(256, 256, False),
            nn.Conv2d(256, d_model, 3, padding=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, None))
        )
    
    def forward(self, x):
        x = self.features(x)
        b, c, h, w = x.shape
        return x.squeeze(2).permute(2, 0, 1)


class PositionalEncoding(nn.Module):
    """Positional Encoding từ notebook"""
    
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(1))
    
    def forward(self, x):
        return self.dropout(x + self.pe[:x.size(0)])


class TransformerOCRModel(nn.Module):
    """Kiến trúc Transformer OCR - khớp với notebook"""
    
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=256, nhead=8,
                 num_enc_layers=4, num_dec_layers=4, dim_ff=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.PAD_IDX = PAD_IDX
        
        # CNN Backbone
        self.backbone = CNNBackbone(d_model)
        
        # Positional Encodings
        self.src_pos_enc = PositionalEncoding(d_model, dropout)
        self.tgt_pos_enc = PositionalEncoding(d_model, dropout)
        
        # Target embedding
        self.tgt_embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        
        # Transformer Encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, dropout,
            batch_first=False, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_enc_layers,
                                            nn.LayerNorm(d_model))
        
        # Transformer Decoder
        dec_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_ff, dropout,
            batch_first=False, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_dec_layers,
                                            nn.LayerNorm(d_model))
        
        # Output layer
        self.fc_out = nn.Linear(d_model, vocab_size)
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    @staticmethod
    def _causal_mask(sz, device):
        return torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
    
    def encode(self, src_img):
        src = self.src_pos_enc(self.backbone(src_img))
        return self.encoder(src)
    
    def decode(self, tgt, memory):
        tgt_mask = self._causal_mask(tgt.size(0), tgt.device)
        tgt_key_padding_mask = (tgt == PAD_IDX).T
        tgt_emb = self.tgt_pos_enc(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        out = self.decoder(tgt_emb, memory,
                          tgt_mask=tgt_mask,
                          tgt_key_padding_mask=tgt_key_padding_mask)
        return self.fc_out(out)
    
    def forward(self, src_img, tgt=None):
        """Forward pass - tương thích với notebook"""
        memory = self.encode(src_img)
        if tgt is not None:
            tgt = tgt.permute(1, 0)
            return self.decode(tgt[:-1], memory)
        return memory
    
    @torch.no_grad()
    def greedy_decode(self, src_img, max_len=MAX_LEN, device='cpu'):
        """Greedy decoding"""
        self.eval()
        B = src_img.size(0)
        memory = self.encode(src_img)
        ys = torch.full((1, B), SOS_IDX, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        result = [[] for _ in range(B)]
        
        for _ in range(max_len):
            logits = self.decode(ys, memory)
            next_token = logits[-1].argmax(dim=-1)
            ys = torch.cat([ys, next_token.unsqueeze(0)], dim=0)
            for b in range(B):
                if finished[b]: continue
                t = next_token[b].item()
                if t == EOS_IDX:
                    finished[b] = True
                elif t != PAD_IDX:
                    result[b].append(idx2char_notebook.get(t, ''))
            if finished.all(): break
        
        return [''.join(r) for r in result]


class TransformerOCRModule:
    """OCR sử dụng Transformer"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """Khởi tạo Transformer OCR module"""
        self.device = device
        self.model = None
        self.char_list = CHARS
        self.idx2char = idx2char_notebook.copy()
        self._load_model()
    
    def _load_model(self):
        """Load mô hình Transformer"""
        try:
            self.model = TransformerOCRModel(
                vocab_size=VOCAB_SIZE,
                d_model=256,
                nhead=8,
                num_enc_layers=4,
                num_dec_layers=4,
                dim_ff=512,
                dropout=0.1
            )
            
            weight_path = get_weight_path('transformer_ocr_best.pt')
            checkpoint = torch.load(weight_path, map_location=self.device)
            
            # Load model state dict
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                    if 'idx2char' in checkpoint:
                        self.idx2char = checkpoint['idx2char']
                        print(f"[OK] Loaded idx2char from checkpoint: {len(self.idx2char)} classes")
                else:
                    # Trực tiếp là state_dict
                    self.model.load_state_dict(checkpoint, strict=False)
            else:
                self.model.load_state_dict(checkpoint, strict=False)
            
            self.model.to(self.device)
            self.model.eval()
            print(f"[OK] Transformer OCR model loaded from {weight_path}")
        except Exception as e:
            print(f"[WARN] Transformer model load failed: {e}")
            print(f"  Fallback: Transformer disabled, use EasyOCR instead")
            self.model = None  # Mark as failed
    
    def preprocess_plate(self, plate_crop):
        """
        Tiền xử lý ảnh biển số - KHỚP 100% với notebook
        
        Args:
            plate_crop: ảnh biển số đã crop (H, W, 3) BGR
        
        Returns:
            tensor đã xử lý (1, 3, 32, 128)
        """
        # Convert BGR → RGB (PIL yêu cầu RGB)
        rgb = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(rgb)
        
        # Sử dụng Torchvision Transform giống notebook
        infer_transform = transforms.Compose([
            transforms.Resize((32, 128)),  # Direct resize to (H, W)
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        img_tensor = infer_transform(img_pil).unsqueeze(0).to(self.device)
        return img_tensor
    
    def decode_prediction_greedy(self, logits):
        """
        Decode Transformer output to text - Greedy decoding
        
        Args:
            logits: model output (seq_len, batch_size, vocab_size)
        
        Returns:
            text
        """
        preds = logits.argmax(dim=-1).permute(1, 0).cpu().numpy()  # (batch, seq_len)
        
        results = []
        for seq in preds:
            chars = []
            prev = None
            for idx in seq:
                idx = int(idx)
                if idx == EOS_IDX:
                    break
                if idx not in [PAD_IDX, SOS_IDX]:
                    # Skip consecutive duplicates (CTC style)
                    if idx != prev:
                        char = self.idx2char.get(idx, '')
                        if char:
                            chars.append(char)
                prev = idx
            results.append(''.join(chars))
        
        return results[0] if results else '???'
    
    def recognize(self, plate_crop):
        """
        Nhận dạng biển số từ ảnh crop
        
        Args:
            plate_crop: ảnh biển số đã crop (H, W, 3) BGR
        
        Returns:
            chuỗi biển số
        """
        if self.model is None:
            return ""
        
        try:
            img_tensor = self.preprocess_plate(plate_crop)
            if img_tensor is None:
                return ""
            
            with torch.no_grad():
                texts = self.model.greedy_decode(img_tensor, max_len=MAX_LEN,
                                                device=self.device)
                text = texts[0] if texts else '???'
            
            return normalize_plate_text(text)
        except Exception as e:
            return ""
