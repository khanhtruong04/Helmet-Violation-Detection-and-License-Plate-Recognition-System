"""
Pipeline xử lý video - Triển khai 7-STEP VIDEO PROCESSING PIPELINE

📊 7-STEP PIPELINE (từ README - Chi Tiết Luồng Xử Lý Video):
═══════════════════════════════════════════════════════════════

1️⃣ VIDEO ĐẦU VÀO
   - Input: video.mp4 (giao thông)
   - Sử dụng: cv2.VideoCapture()

2️⃣ FRAME-BY-FRAME LOOP
   - Đọc từng frame từ video
   - Sử dụng: while cap.read()

3️⃣ DETECT HELMET + PLATE (YOLO)
   - Helmet Detector: YOLOv8 hoặc Faster R-CNN
     → Detect: helmet (✓), no_helmet (❌)
   - Plate Detector: YOLO
     → Detect: vị trí biển số (bounding box)

4️⃣ CROP BIỂN SỐ TỪ FRAME (Kích thước tự nhiên)
   - plate_crop = frame[y1:y2, x1:x2]
   - Shape: (H, W, 3) BGR - tự nhiên, không resize cố định
   - Ví dụ: (60×250×3), (80×300×3), v.v.

5️⃣ TRANSFORMER OCR INFERENCE (Upgraded!)
   - Text = ocr_module.recognize(plate_crop)
   - Bên trong:
     • Preprocess: Aspect-ratio-preserving resize + Normalize
     • CNN backbone: Xử lý size gốc tự động
     • TTA: 5 augmentations → voting → confidence score
     • Rules: O→0, I→1, S→5, B→8
     • Output: "27D7251" (recognized text)

6️⃣ ANNOTATION LÊN FRAME
   - Vẽ box vàng (0, 255, 255) quanh biển số
   - Vẽ text OCR phía trên (nền vàng + chữ đen)
   - Vẽ box xanh (0, 255, 0) cho helmet
   - Vẽ box đỏ (0, 0, 255) cho no_helmet
   - Vẽ box xanh dương (219, 152, 52) cho motorbike

7️⃣ WRITE FRAME TO OUTPUT VIDEO
   - Frame đã annotated → cv2.VideoWriter
   - Output: video_output.mp4 (đã xử lý)

═══════════════════════════════════════════════════════════════

⏰ LOOP: Repeat bước 2-7 cho frame tiếp theo
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

            self.helmet_detector = HelmetDetector(model_type=helmet_model, device=self.device)
            self.plate_detector = PlateDetector(device=self.device)
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
            from modules.ocr.easyocr_module import EasyOCRModule
            self.ocr_module = EasyOCRModule()

    def process_frame(self, frame):
        """
        Xử lý một frame - Theo 7 bước pipeline từ README:
        
        📊 7-STEP VIDEO PROCESSING PIPELINE:
        
        🔹 Step 1: VIDEO ĐẦU VÀO
           Input frame từ video stream (H x W x 3) BGR format
        
        🔹 Step 2: FRAME-BY-FRAME LOOP
           Xử lý từng frame tuần tự
        
        🔹 Step 3: DETECT HELMET + PLATE (YOLO)
           - Helmet Detector: detect người có/không mũ
           - Plate Detector: detect vị trí biển số (bounding box)
        
        🔹 Step 4: CROP BIỂN SỐ (Kích thước tự nhiên)
           plate_crop = frame[y1:y2, x1:x2]
           → Shape: (H, W, 3) BGR - kích thước tự nhiên, không resize cố định
           → Ví dụ: (60×250×3), (80×300×3), v.v.
        
        🔹 Step 5: TRANSFORMER OCR INFERENCE (Upgraded!)
           text = ocr_module.recognize(plate_crop)
           → Bên trong xử lý:
             - Preprocess: Aspect-ratio-preserving resize + Normalize
             - CNN backbone: Tự động xử lý kích thước tự nhiên
             - TTA: 5 augmentations → voting → confidence score
             - Post-processing: O→0, I→1, S→5, B→8
             - Output: "27D7251" (recognized text)
        
        🔹 Step 6: ANNOTATION LÊN FRAME
           - Vẽ box vàng (0, 255, 255) quanh biển số
           - Vẽ text OCR phía trên box với nền vàng + chữ đen
           - Vẽ box xanh (0, 255, 0) cho helmet
           - Vẽ box đỏ (0, 0, 255) cho no_helmet
           - Vẽ box xanh dương (219, 152, 52) cho motorbike
        
        🔹 Step 7: WRITE FRAME TO OUTPUT VIDEO
           → frame đã annotated được ghi vào video output
           → Tiếp tục loop cho frame tiếp theo
        
        Args:
            frame: ảnh frame input (H x W x 3) BGR format

        Returns:
            dict chứa:
                - frame: frame đã xử lý với annotations
                - helmet_detections: list helmet detections
                - plate_detections: list plate detections
                - plate_texts: list OCR results (text + box)
        """
        # ========== STEP 1+2: VIDEO INPUT + FRAME LOOP ==========
        # Frame đã được đọc từ cv2.VideoCapture trong process_video()
        
        # Resize frame nếu quá lớn (optimization)
        h, w = frame.shape[:2]
        if w > 1280:
            scale = 1280 / w
            frame = cv2.resize(frame, (1280, int(h * scale)))

        # ========== STEP 3: DETECT HELMET + PLATE (YOLO) ==========
        # 🎯 HELMET DETECTION: YOLOv8 hoặc Faster R-CNN
        helmet_results = self.helmet_detector.detect(frame)
        frame = annotate_frame_with_helmet_detections(frame, helmet_results)
        
        # 🎯 PLATE DETECTION: YOLO
        plate_results = self.plate_detector.detect(frame)
        frame = annotate_frame_with_plate_detections(frame, plate_results)

        # ========== STEP 4+5: CROP BIỂN SỐ + TRANSFORMER OCR ==========
        # Xử lý từng biển số phát hiện được
        plate_texts = []
        for plate_det in plate_results:
            # STEP 4: CROP BIỂN SỐ (kích thước tự nhiên)
            # plate_det['crop'] đã được crop với kích thước tự nhiên
            # Shape: (H, W, 3) BGR - không resize cố định
            plate_crop = plate_det['crop']
            
            try:
                # STEP 5: TRANSFORMER OCR INFERENCE
                # recognize() xử lý:
                # - Aspect-ratio-preserving resize (không méo)
                # - TTA: 5 augmentations + voting
                # - Post-processing: quy tắc biển số Việt Nam
                plate_text = self.ocr_module.recognize(plate_crop)
                plate_texts.append({
                    'box': plate_det['box'],
                    'text': plate_text
                })
            except Exception as e:
                print(f"⚠ Lỗi OCR: {e}")

        # ========== STEP 6: ANNOTATION LÊN FRAME ==========
        # Vẽ text OCR lên frame với:
        # - Box vàng (0, 255, 255) quanh biển số
        # - Text phía trên (nền vàng + chữ đen)
        frame = annotate_frame_with_ocr_text(frame, plate_texts)

        # ========== STEP 7: READY FOR VIDEO OUTPUT ==========
        # Frame đã annotated sẽ được ghi vào output video
        # trong process_video() → out.write(processed_frame)

        return {
            'frame': frame,  # Frame đã annotated - ready for output
            'helmet_detections': helmet_results,
            'plate_detections': plate_results,
            'plate_texts': plate_texts
        }

    def process_video(self, input_path, output_path=None, show_fps=True):
        """
        Xử lý video đầu vào theo 7-STEP PIPELINE
        
        📊 PIPELINE IMPLEMENTATION:
        ├─ STEP 1️⃣: VIDEO ĐẦU VÀO
        │   cv2.VideoCapture(input_path) - mở video input
        │
        ├─ STEP 2️⃣: FRAME-BY-FRAME LOOP
        │   while cap.read(): 
        │     for each frame → process_frame()
        │
        ├─ STEP 3️⃣-6️⃣: Xử lý trong process_frame()
        │   - DETECT HELMET + PLATE (YOLO)
        │   - CROP BIỂN SỐ (kích thước tự nhiên)
        │   - TRANSFORMER OCR INFERENCE
        │   - ANNOTATION LÊN FRAME
        │
        └─ STEP 7️⃣: WRITE FRAME TO OUTPUT VIDEO
            cv2.VideoWriter: ghi frame đã annotated

        Args:
            input_path: đường dẫn video input
            output_path: đường dẫn video output (nếu None, mặc định 'output/result_video.mp4')
            show_fps: hiển thị FPS lên frame

        Returns:
            dict chứa thông tin xử lý
        """
        ensure_dir_exists('output')

        if output_path is None:
            output_path = 'output/result_video.mp4'

        try:
            # ========== STEP 1️⃣: VIDEO ĐẦU VÀO ==========
            cap = cv2.VideoCapture(input_path)

            if not cap.isOpened():
                print(f"❌ Không thể mở video: {input_path}")
                return None

            fps      = cap.get(cv2.CAP_PROP_FPS)
            width    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            print(f"\n📹 Video: {input_path}")
            print(f"   FPS: {fps}, Resolution: {width}x{height}, Frames: {total_frames}")

            device_name = "GPU (CUDA)" if self.device == 'cuda' else "CPU"
            if self.device == 'cuda':
                print(f"   💻 Sử dụng: {device_name} - {torch.cuda.get_device_name(0)}")
            else:
                print(f"   💻 Sử dụng: {device_name}")

            # ========== STEP 7️⃣: SETUP OUTPUT VIDEO WRITER ==========
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            frame_count = 0
            statistics = {
                'helmet_detected': 0,
                'no_helmet_detected': 0,
                'plates_detected': 0,
                'plates_with_text': 0
            }

            print(f"\n⏳ Đang xử lý video...")
            pbar = tqdm(total=total_frames, desc="Progress")

            # ========== STEP 2️⃣: FRAME-BY-FRAME LOOP ==========
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                start_time = time.time()
                
                # ========== STEP 3️⃣-6️⃣: DETECT + CROP + OCR + ANNOTATE ==========
                # process_frame() xử lý tất cả bước này:
                # - DETECT HELMET + PLATE (YOLO)
                # - CROP BIỂN SỐ (kích thước tự nhiên - no fixed resize)
                # - TRANSFORMER OCR (Aspect-ratio-preserving + TTA)
                # - ANNOTATION (vẽ boxes + text)
                result = self.process_frame(frame)
                elapsed_time = time.time() - start_time

                processed_frame = result['frame']  # Frame đã annotated

                # Thống kê
                for det in result['helmet_detections']:
                    if det['label'] == 'helmet':
                        statistics['helmet_detected'] += 1
                    else:
                        statistics['no_helmet_detected'] += 1

                statistics['plates_detected']   += len(result['plate_detections'])
                statistics['plates_with_text']  += len(result['plate_texts'])

                # Vẽ FPS góc dưới trái để không che box vàng biển số
                if show_fps:
                    fps_value = 1 / elapsed_time if elapsed_time > 0 else 0
                    cv2.putText(
                        processed_frame,
                        f"FPS: {fps_value:.1f}",
                        (10, processed_frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

                # ========== STEP 7️⃣: WRITE FRAME TO OUTPUT VIDEO ==========
                # Ghi frame đã annotated vào output video
                out.write(processed_frame)
                
                frame_count += 1
                pbar.update(1)

            pbar.close()
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

            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                result = self.process_frame(frame)
                processed_frame = result['frame']

                if callback:
                    callback(processed_frame, result)

                frame_count += 1
                yield processed_frame

            cap.release()

        except Exception as e:
            print(f"❌ Lỗi streaming video: {e}")