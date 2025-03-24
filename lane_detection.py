from ultralytics import YOLO
import cv2

# Load model YOLOv8 Instance Segmentation
model = YOLO("best.pt")

def detect_rail_lane(image_path):
    """Mendeteksi jalur rel menggunakan YOLOv8 Instance Segmentation"""
    results = model(image_path, show=True)
    results[0].save("lane_detection_result.jpg")

# Contoh penggunaan
detect_rail_lane("rail_segmentation2/test/images/20250321_40508PMByGPSMapCamera_jpg.rf.d86dc03d2ca9b1c8aa336ac4d7c43df9.jpg")