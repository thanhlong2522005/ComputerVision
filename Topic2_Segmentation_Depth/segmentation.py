import cv2
from ultralytics import YOLO

class SegmentationModule:
    def __init__(self, model_path="yolov8n-seg.pt", device="cpu"):
        self.device = device
        print("Đang tải YOLOv8n-seg (Ép xung tốc độ cao)...")
        self.model = YOLO(model_path)
        self.vehicle_classes = [2, 3, 5, 7]

    def predict(self, img_rgb):
        # TỐI ƯU 4: Ép tham số imgsz=320 để AI chạy cực nhanh trên CPU
        results = self.model(img_rgb, 
                            classes=self.vehicle_classes,
                            verbose=False,
                            device=self.device,
                            imgsz=320)
        return results[0]