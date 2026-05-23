"""
Pipeline xử lý video

Xử lý video frame-by-frame:
1. Phát hiện mũ bảo hiểm
2. Phát hiện biển số
3. OCR biển số
4. Vẽ annotations
5. Ghi video đầu ra
"""

import cv2
import time
import torch
import numpy as np
from tqdm import tqdm
from modules.helmet_detector import HelmetDetector
from modules.plate_detector import PlateDetector
from utils.helpers import ensure_dir_exists
from utils.drawing import (
    annotate_frame_with_helmet_detections,
    annotate_frame_with_plate_detections,
    annotate_frame_with_ocr_text
)


class VideoProcessor:
    """Pipeline xử lý video"""
    
    def __init__(self, helmet_model='yolov8', ocr_method='easyocr', device=None):
        """
        Khởi tạo processor
        
        Args:
            helmet_model: 'yolov8' hoặc 'faster_rcnn'
            ocr_method: 'cnn', 'easyocr', 'crnn', 'transformer'
            device: 'cuda' hoặc 'cpu' (mặc định tự động phát hiện)
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.device = device
        self.helmet_detector = None
        self.plate_detector = None
        self.ocr_module = None
        
        self._init_detectors(helmet_model, ocr_method)
    
    def _init_detectors(self, helmet_model, ocr_method):
        """Khởi tạo các detector"""
        try:
            print("Initializing detectors...")
            
            # Helmet detector - truyền device
            self.helmet_detector = HelmetDetector(model_type=helmet_model, device=self.device)
            
            # Plate detector - truyền device
            self.plate_detector = PlateDetector(device=self.device)
            
            # OCR module - lazy load khi cần
            self._init_ocr_module(ocr_method)
            
            print("✓ All detectors initialized successfully")
        except Exception as e:
            print(f"❌ Lỗi khởi tạo detector: {e}")
            raise
    
    def _init_ocr_module(self, ocr_method):
        """Lazy load OCR module - chỉ import khi cần"""
        try:
            if ocr_method.lower() == 'cnn':
                from modules.ocr.cnn_ocr import CNNCharacterClassifier
                try:
                    self.ocr_module = CNNCharacterClassifier()
                except:
                    print("  ⚠ CNN failed, fallback to EasyOCR")
                    from modules.ocr.easyocr_module import EasyOCRModule
                    self.ocr_module = EasyOCRModule()
                    
            elif ocr_method.lower() == 'crnn':
                from modules.ocr.crnn_ocr import CRNNOCRModule
                self.ocr_module = CRNNOCRModule()
                if self.ocr_module.model is None:
                    print("  ⚠ CRNN failed, fallback to EasyOCR")
                    from modules.ocr.easyocr_module import EasyOCRModule
                    self.ocr_module = EasyOCRModule()
                    
            elif ocr_method.lower() == 'transformer':
                from modules.ocr.transformer_ocr import TransformerOCRModule
                self.ocr_module = TransformerOCRModule()
                if self.ocr_module.model is None:
                    print("  ⚠ Transformer failed, fallback to EasyOCR")
                    from modules.ocr.easyocr_module import EasyOCRModule
                    self.ocr_module = EasyOCRModule()
                    
            else:  # Default: easyocr
                from modules.ocr.easyocr_module import EasyOCRModule
                self.ocr_module = EasyOCRModule()
        except Exception as e:
            print(f"  ⚠ Error loading OCR {ocr_method}: {e}")
            # Final fallback to EasyOCR
            from modules.ocr.easyocr_module import EasyOCRModule
            self.ocr_module = EasyOCRModule()
    
    def process_frame(self, frame):
        """
        Xử lý một frame
        
        Args:
            frame: ảnh frame (H x W x 3)
        
        Returns:
            dict chứa:
                - frame: frame đã xử lý
                - detections: {helmet, plate}
        """
        # Resize frame nếu quá lớn
        h, w = frame.shape[:2]
        if w > 1280:
            scale = 1280 / w
            frame = cv2.resize(frame, (1280, int(h * scale)))
        
        # Helmet detection + annotations
        helmet_results = self.helmet_detector.detect(frame)
        frame = annotate_frame_with_helmet_detections(frame, helmet_results)
        
        # Plate detection + annotations
        plate_results = self.plate_detector.detect(frame)
        frame = annotate_frame_with_plate_detections(frame, plate_results)
        
        # OCR for each plate
        plate_texts = []
        for plate_det in plate_results:
            plate_crop = plate_det['crop']
            
            try:
                # Nhận dạng biển số
                plate_text = self.ocr_module.recognize(plate_crop)
                plate_texts.append({
                    'box': plate_det['box'],
                    'text': plate_text
                })
            except Exception as e:
                print(f"⚠ Lỗi OCR: {e}")
        
        # Vẽ OCR text annotations (xử lý y position đúng từ drawing.py)
        frame = annotate_frame_with_ocr_text(frame, plate_texts)
        
        return {
            'frame': frame,
            'helmet_detections': helmet_results,
            'plate_detections': plate_results,
            'plate_texts': plate_texts
        }
    
    def process_video(self, input_path, output_path=None, show_fps=True):
        """
        Xử lý video đầu vào
        
        Args:
            input_path: đường dẫn video input
            output_path: đường dẫn video output (nếu None, không ghi file)
            show_fps: hiển thị FPS lên frame
        
        Returns:
            dict chứa thông tin xử lý
        """
        ensure_dir_exists('output')
        
        if output_path is None:
            output_path = 'output/result_video.mp4'
        
        try:
            # Mở video
            cap = cv2.VideoCapture(input_path)
            
            if not cap.isOpened():
                print(f"❌ Không thể mở video: {input_path}")
                return None
            
            # Lấy thông tin video
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"\n📹 Video: {input_path}")
            print(f"   FPS: {fps}, Resolution: {width}x{height}, Frames: {total_frames}")
            
            # Hiển thị thông tin thiết bị
            device_name = "GPU (CUDA)" if self.device == 'cuda' else "CPU"
            if self.device == 'cuda':
                print(f"   💻 Sử dụng: {device_name} - {torch.cuda.get_device_name(0)}")
            else:
                print(f"   💻 Sử dụng: {device_name}")
            
            # Khởi tạo VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Xử lý từng frame
            frame_count = 0
            statistics = {
                'helmet_detected': 0,
                'no_helmet_detected': 0,
                'plates_detected': 0,
                'plates_with_text': 0
            }
            
            print(f"\n⏳ Đang xử lý video...")
            pbar = tqdm(total=total_frames, desc="Progress")
            
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Xử lý frame
                start_time = time.time()
                result = self.process_frame(frame)
                elapsed_time = time.time() - start_time
                
                processed_frame = result['frame']
                
                # Cập nhật thống kê
                for det in result['helmet_detections']:
                    if det['label'] == 'helmet':
                        statistics['helmet_detected'] += 1
                    else:
                        statistics['no_helmet_detected'] += 1
                
                statistics['plates_detected'] += len(result['plate_detections'])
                statistics['plates_with_text'] += len(result['plate_texts'])
                
                # Vẽ FPS
                if show_fps:
                    fps_value = 1 / elapsed_time if elapsed_time > 0 else 0
                    cv2.putText(
                        processed_frame,
                        f"FPS: {fps_value:.1f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )
                
                # Ghi frame
                out.write(processed_frame)
                
                frame_count += 1
                pbar.update(1)
            
            pbar.close()
            
            # Đóng files
            cap.release()
            out.release()
            
            print(f"\n✓ Video xử lý thành công")
            print(f"   📁 Đầu ra: {output_path}")
            print(f"   📊 Frames xử lý: {frame_count}")
            print(f"   💻 Thiết bị sử dụng: {'GPU (CUDA)' if self.device == 'cuda' else 'CPU'}")
            print(f"   📈 Thống kê:")
            print(f"     - Có đội mũ: {statistics['helmet_detected']}")
            print(f"     - Không đội mũ (vi phạm): {statistics['no_helmet_detected']}")
            print(f"     - Biển số phát hiện: {statistics['plates_detected']}")
            print(f"     - Biển số có text: {statistics['plates_with_text']}")
            
            return {
                'success': True,
                'output_path': output_path,
                'frames_processed': frame_count,
                'statistics': statistics
            }
        
        except Exception as e:
            print(f"❌ Lỗi xử lý video: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def process_video_streaming(self, input_path, callback=None):
        """
        Xử lý video với streaming callback
        
        Args:
            input_path: đường dẫn video input
            callback: function(frame, result) được gọi cho mỗi frame
        
        Yields:
            frame đã xử lý
        """
        try:
            cap = cv2.VideoCapture(input_path)
            
            if not cap.isOpened():
                print(f"❌ Không thể mở video: {input_path}")
                return
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Xử lý frame
                result = self.process_frame(frame)
                processed_frame = result['frame']
                
                # Call callback nếu có
                if callback:
                    callback(processed_frame, result)
                
                frame_count += 1
                yield processed_frame
            
            cap.release()
            
        except Exception as e:
            print(f"❌ Lỗi streaming video: {e}")
