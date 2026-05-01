import cv2
import numpy as np
import torch
from depth import DepthModule 

class CoreLogicAnalyzer:
    def __init__(self, danger_threshold=5.0):
        self.danger_threshold = danger_threshold 
        # Biến nhớ để làm mượt số liệu (Chống nhảy loạn xạ)
        self.smoothed_min_dist = None 

    def analyze_scene(self, yolo_result, midas_tensor, target_shape):
        h, w = target_shape

        depth_map = torch.nn.functional.interpolate(
            midas_tensor.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False
        ).squeeze().cpu().numpy()

        # Làn đường an toàn (ROI 30% - 70%)
        safe_zone_left = w * 0.3
        safe_zone_right = w * 0.7

        detected_cars = []
        danger_flag = False
        raw_min_distance = 999.0
        
        if yolo_result.masks is not None and yolo_result.boxes is not None:
            masks = yolo_result.masks.data.cpu().numpy() 
            boxes = yolo_result.boxes.xyxy.cpu().numpy() 
            
            for i in range(len(masks)):
                x1, y1, x2, y2 = map(int, boxes[i])
                bw, bh = x2 - x1, y2 - y1
                
                # 1. Bỏ qua các đốm nhiễu quá nhỏ
                if bw * bh <= 400: 
                    continue

                # 2. Bỏ qua mui xe của chính mình (Ego-vehicle hood)
                if y2 > h * 0.95 and bw > w * 0.6:
                    continue

                # 3. Lấy Depth ở phần gầm xe (Chính xác hơn nóc xe)
                mask = cv2.resize(masks[i], (w, h), interpolation=cv2.INTER_NEAREST)
                
                y_bottom_start = int(y1 + bh * 0.8) # Lấy 20% dưới cùng
                bottom_mask = mask.copy()
                bottom_mask[:y_bottom_start, :] = 0 
                
                valid_depths = depth_map[bottom_mask == 1]
                if len(valid_depths) > 0:
                    mean_depth = np.mean(valid_depths)
                else:
                    mean_depth = np.mean(depth_map[mask == 1])
                
                # 4. Truyền thêm chiều cao xe (bh) vào hàm để tính toán Vật lý Pinhole
                distance = DepthModule.estimate_distance(mean_depth, bh)
                
                # 5. Phân loại xe cản đường (ROI)
                center_x = x1 + (bw / 2)
                is_in_path = safe_zone_left < center_x < safe_zone_right
                
                detected_cars.append({
                    'bbox': (x1, y1, bw, bh),
                    'distance': distance,
                    'mask': mask,
                    'is_in_path': is_in_path 
                })

                # Chỉ xét khoảng cách cảnh báo nếu xe đang ngáng đường
                if is_in_path and distance < raw_min_distance:
                    raw_min_distance = distance

        # 6. THUẬT TOÁN EMA (LÀM MƯỢT SỐ NHẢY)
        if raw_min_distance != 999.0:
            if self.smoothed_min_dist is None or self.smoothed_min_dist == 999.0:
                self.smoothed_min_dist = raw_min_distance
            else:
                self.smoothed_min_dist = (0.3 * raw_min_distance) + (0.7 * self.smoothed_min_dist)
        else:
            self.smoothed_min_dist = 999.0

        # Kích hoạt báo động dựa trên con số đã được làm mượt
        if self.smoothed_min_dist < self.danger_threshold:
            danger_flag = True

        return None, detected_cars, danger_flag, self.smoothed_min_dist