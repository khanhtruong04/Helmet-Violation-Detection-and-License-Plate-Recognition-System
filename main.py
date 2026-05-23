"""
Main application - Gradio Web Interface

Giao diện web để:
1. Upload video
2. Chọn mô hình phát hiện mũ
3. Chọn phương pháp OCR
4. Xem kết quả stream
"""

import os
import gradio as gr
import torch
from modules.video_processor import VideoProcessor
from utils.helpers import ensure_dir_exists, get_device_info


# Biến global để lưu processor
current_processor = None


def process_video_ui(video_input, helmet_model, ocr_method):
    """
    Callback xử lý video từ giao diện
    
    Args:
        video_input: file video từ upload
        helmet_model: model phát hiện mũ ('yolov8' hoặc 'faster_rcnn')
        ocr_method: phương pháp OCR ('cnn', 'easyocr', 'crnn', 'transformer')
    
    Returns:
        dict chứa video output và thống kê
    """
    global current_processor
    
    if video_input is None:
        return None, "❌ Vui lòng upload video trước"
    
    try:
        # Map UI labels to internal names
        helmet_model_map = {
            'YOLOv8 (Nhanh)': 'yolov8',
            'Faster R-CNN (Chính xác)': 'faster_rcnn'
        }
        
        ocr_method_map = {
            'CNN Classifier': 'cnn',
            'EasyOCR': 'easyocr',
            'CRNN': 'crnn',
            'Transformer OCR': 'transformer'
        }
        
        helmet_model_internal = helmet_model_map.get(helmet_model, 'yolov8')
        ocr_method_internal = ocr_method_map.get(ocr_method, 'easyocr')
        
        # Lấy thông tin GPU/CPU
        device_info = get_device_info()
        
        print(f"\n🔄 Đang xử lý video:")
        print(f"   🎯 Mô hình phát hiện mũ: {helmet_model}")
        print(f"   📝 Phương pháp OCR: {ocr_method}")
        print(f"   💻 {device_info['message']}")
        
        # Khởi tạo processor
        current_processor = VideoProcessor(
            helmet_model=helmet_model_internal,
            ocr_method=ocr_method_internal,
            device=device_info['device']
        )
        
        # Xử lý video
        result = current_processor.process_video(
            input_path=video_input,
            output_path='output/result_video.mp4',
            show_fps=False
        )
        
        if result and result['success']:
            stats_text = f"""
✓ **Video xử lý thành công!**

📊 **Thống kê:**
- Có đội mũ: {result['statistics']['helmet_detected']}
- Không đội mũ (vi phạm): {result['statistics']['no_helmet_detected']}
- Biển số phát hiện: {result['statistics']['plates_detected']}
- Biển số có text: {result['statistics']['plates_with_text']}
- Tổng frames: {result['frames_processed']}

💾 **Đầu ra:** {result['output_path']}

💻 **Thiết bị sử dụng:** {device_info['message']}
            """
            return result['output_path'], stats_text
        else:
            error_msg = result.get('error', 'Unknown error') if result else 'Unknown error'
            return None, f"❌ Lỗi: {error_msg}"
    
    except Exception as e:
        import traceback
        print(f"❌ Lỗi: {e}")
        print(traceback.format_exc())
        return None, f"❌ Lỗi: {str(e)}"


def create_interface():
    """Tạo giao diện Gradio"""
    
    with gr.Blocks(title="🪖 Helmet Detection & License Plate Recognition") as demo:
        gr.Markdown("""
        # 🪖 Hệ Thống Phát Hiện Vi Phạm Mũ Bảo Hiểm & Nhận Dạng Biển Số
        
        > Phát hiện người không đội mũ bảo hiểm và trích xuất biển số xe từ video giao thông theo thời gian thực
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## ⚙️ Cấu Hình")
                
                video_input = gr.Video(
                    label="📹 Upload Video",
                    format="mp4"
                )
                
                helmet_model = gr.Dropdown(
                    choices=[
                        'YOLOv8 (Nhanh)',
                        'Faster R-CNN (Chính xác)'
                    ],
                    value='YOLOv8 (Nhanh)',
                    label="🎯 Mô hình Phát hiện Mũ"
                )
                
                ocr_method = gr.Dropdown(
                    choices=[
                        'CNN Classifier',
                        'EasyOCR',
                        'CRNN',
                        'Transformer OCR'
                    ],
                    value='EasyOCR',
                    label="📝 Phương pháp OCR"
                )
                
                process_btn = gr.Button(
                    "▶️ Chạy Xử Lý Video",
                    scale=1,
                    size='lg',
                    variant='primary'
                )
            
            with gr.Column(scale=1):
                gr.Markdown("## 📊 Kết Quả")
                
                video_output = gr.Video(
                    label="🎬 Video Đầu Ra",
                    format="mp4"
                )
                
                stats_output = gr.Markdown(
                    "Chọn video và cấu hình, sau đó bấm 'Chạy Xử Lý Video'"
                )
        
        # Event handler
        process_btn.click(
            fn=process_video_ui,
            inputs=[video_input, helmet_model, ocr_method],
            outputs=[video_output, stats_output]
        )
        
        gr.Markdown("""
        ---
        
        ## 📋 Hướng Dẫn
        
        1. **Upload video**: Tải lên video giao thông cần xử lý
        2. **Chọn mô hình phát hiện mũ**:
           - **YOLOv8** (Nhanh, phù hợp real-time)
           - **Faster R-CNN** (Chính xác cao)
        3. **Chọn phương pháp OCR** để nhận dạng biển số:
           - **CNN Classifier**: Phân loại từng ký tự
           - **EasyOCR**: Dễ tích hợp, hỗ trợ nhiều ngôn ngữ
           - **CRNN**: Nhận dạng chuỗi ký tự liên tục
           - **Transformer OCR**: Độ chính xác cao nhất
        4. **Bấm "Chạy Xử Lý Video"** để bắt đầu
        5. **Xem kết quả**: Video đầu ra với annotations + thống kê
        
        ## 🎨 Màu sắc Annotations
        
        - 🟩 **Xanh lá**: Người đội mũ bảo hiểm ✅
        - 🟥 **Đỏ**: Người không đội mũ (vi phạm) ❌
        - 🟨 **Vàng**: Vị trí biển số xe
        - Text **đen**: Ký tự biển số đọc được
        """)
    
    return demo


def main():
    """Chạy ứng dụng"""
    # Tạo các thư mục cần thiết
    ensure_dir_exists('input')
    ensure_dir_exists('output')
    
    # Lấy thông tin thiết bị
    device_info = get_device_info()
    
    print("\n" + "="*60)
    print("🪖 Hệ Thống Phát Hiện Vi Phạm Mũ Bảo Hiểm & Biển Số Xe")
    print("="*60)
    print(f"\n💻 Thông tin thiết bị:")
    print(f"   {device_info['message']}")
    if device_info['device'] == 'cuda':
        print(f"   📊 VRAM: {device_info['memory_gb']:.1f}GB")
    print()
    print("🚀 Khởi động giao diện Gradio...")
    print("📍 Mở trình duyệt: http://localhost:7860/")
    print("\n" + "="*60 + "\n")
    
    # Tạo và chạy giao diện
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    main()
