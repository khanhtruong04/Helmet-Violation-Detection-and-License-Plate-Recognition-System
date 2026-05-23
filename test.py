"""
Test cơ bản cho các modules

Kiểm tra xem các module có tải và hoạt động đúng không
"""

import sys
import os

# Thêm dự án vào path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_imports():
    """Test import các modules"""
    print("\n" + "="*60)
    print("🧪 Testing Module Imports")
    print("="*60 + "\n")
    
    tests_passed = 0
    tests_failed = 0
    
    # Test imports
    test_cases = [
        ("utils.helpers", lambda: __import__('utils.helpers')),
        ("modules.helmet_detector", lambda: __import__('modules.helmet_detector')),
        ("modules.plate_detector", lambda: __import__('modules.plate_detector')),
        ("modules.video_processor", lambda: __import__('modules.video_processor')),
        ("modules.ocr.easyocr_module", lambda: __import__('modules.ocr.easyocr_module')),
        ("modules.ocr.cnn_ocr", lambda: __import__('modules.ocr.cnn_ocr')),
        ("modules.ocr.crnn_ocr", lambda: __import__('modules.ocr.crnn_ocr')),
        ("modules.ocr.transformer_ocr", lambda: __import__('modules.ocr.transformer_ocr')),
    ]
    
    for test_name, test_func in test_cases:
        try:
            test_func()
            print(f"✓ {test_name}")
            tests_passed += 1
        except Exception as e:
            print(f"✗ {test_name}: {str(e)[:60]}")
            tests_failed += 1
    
    print(f"\n📊 Import Tests: {tests_passed} passed, {tests_failed} failed\n")
    return tests_failed == 0


def test_weight_files():
    """Kiểm tra file trọng số"""
    print("="*60)
    print("🔍 Checking Weight Files")
    print("="*60 + "\n")
    
    weight_dir = "weights"
    required_files = [
        "best_lp_detector.pt",
        "best-helmet-detection.pt",
        "character_classifier.h5",
        "crnn_ocr_best.pth",
        "transformer_ocr_best.pt"
    ]
    
    print(f"Weight directory: {os.path.abspath(weight_dir)}\n")
    
    found_count = 0
    for filename in required_files:
        filepath = os.path.join(weight_dir, filename)
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024*1024)
            print(f"✓ {filename} ({size_mb:.1f} MB)")
            found_count += 1
        else:
            print(f"✗ {filename} - NOT FOUND")
    
    print(f"\n✓ Found {found_count}/{len(required_files)} weight files\n")
    return found_count > 0


def test_helpers():
    """Test các hàm helper"""
    print("="*60)
    print("✨ Testing Helper Functions")
    print("="*60 + "\n")
    
    from utils.helpers import (
        get_weight_path,
        ensure_dir_exists,
        normalize_plate_text,
        resize_frame
    )
    
    try:
        # Test get_weight_path
        path = get_weight_path("test.pt")
        assert "weights" in path
        print("✓ get_weight_path works")
        
        # Test ensure_dir_exists
        test_dir = "test_dir"
        ensure_dir_exists(test_dir)
        assert os.path.exists(test_dir)
        os.rmdir(test_dir)
        print("✓ ensure_dir_exists works")
        
        # Test normalize_plate_text
        result = normalize_plate_text("  aB 12  cD 34  ")
        assert result == "AB12CD34"
        print("✓ normalize_plate_text works")
        
        # Test resize_frame
        import numpy as np
        frame = np.zeros((480, 1920, 3), dtype=np.uint8)
        resized = resize_frame(frame, max_width=1280)
        assert resized.shape[1] <= 1280
        print("✓ resize_frame works")
        
        print("\n✓ All helper functions work correctly\n")
        return True
    except Exception as e:
        print(f"\n✗ Error: {e}\n")
        return False


def main():
    """Chạy tất cả tests"""
    print("\n" + "="*60)
    print("🪖 Helmet Detection & License Plate Recognition")
    print("PROJECT TEST SUITE")
    print("="*60)
    
    # Run tests
    imports_ok = test_imports()
    weights_ok = test_weight_files()
    helpers_ok = test_helpers()
    
    # Summary
    print("="*60)
    print("📋 TEST SUMMARY")
    print("="*60)
    print(f"Import tests: {'✓ PASSED' if imports_ok else '✗ FAILED'}")
    print(f"Weight files: {'✓ OK' if weights_ok else '⚠ MISSING FILES'}")
    print(f"Helper functions: {'✓ PASSED' if helpers_ok else '✗ FAILED'}")
    print("\n" + "="*60)
    
    if imports_ok and helpers_ok:
        print("✓ Basic tests passed! Project structure is correct.")
        print("\n🚀 Ready to run: python main.py")
    else:
        print("✗ Some tests failed. Please check the errors above.")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
