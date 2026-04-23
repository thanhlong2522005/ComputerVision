import cv2
import numpy as np
import torch
from depth import DepthModule # Gọi hàm tính khoảng cách của TV2

class CoreLogicAnalyzer:
    def __init__(self, danger_threshold=5.0):
        self.danger_threshold = danger_threshold # Ngưỡng nguy hiểm (dưới 5m)

    def analyze_scene(self, unet_tensor, midas_tensor, target_shape):
        h, w = target_shape

        # 1. POST-PROCESSING: Upscale về kích thước gốc của Video
        seg_mask = torch.nn.functional.interpolate(unet_tensor, size=(h, w), mode="bilinear", align_corners=False)
        seg_mask = torch.argmax(seg_mask, dim=1).squeeze().cpu().numpy()

        depth_map = torch.nn.functional.interpolate(midas_tensor.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False).squeeze().cpu().numpy()

        # 2. OBJECT EXTRACTION: Trích xuất mảng chứa xe (Class 2)
        car_mask = (seg_mask == 2).astype(np.uint8)
        contours, _ = cv2.findContours(car_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_cars = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 800: # Lọc nhiễu
                x, y, bw, bh = cv2.boundingRect(cnt)
                
                # Trích riêng từng chiếc xe để đo Depth
                single_car_mask = np.zeros_like(car_mask)
                cv2.drawContours(single_car_mask, [cnt], -1, 1, thickness=cv2.FILLED)
                
                # Tính độ sâu trung bình và gọi hàm TV2 để quy đổi ra mét
                mean_depth = np.mean(depth_map[single_car_mask == 1])
                distance = DepthModule.estimate_distance(mean_depth)
                
                detected_cars.append({'bbox': (x, y, bw, bh), 'distance': distance})

        # 3. DECISION MAKING: Quyết định có nguy hiểm không
        danger_flag = False
        min_distance = 999.0
        if detected_cars:
            min_distance = min(car['distance'] for car in detected_cars)
            if min_distance < self.danger_threshold:
                danger_flag = True

        # Trả về gói dữ liệu đã xử lý cho TV5 vẽ
        return seg_mask, detected_cars, danger_flag, min_distance