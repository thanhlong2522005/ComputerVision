import cv2
import numpy as np
import torch
from depth import DepthModule 

class CoreLogicAnalyzer:
    def __init__(self, danger_threshold=7.0):
        self.danger_threshold = danger_threshold 
        
        # Làm mượt khoảng cách
        self.smoothed_min_dist = None 
        
        # NEW: lưu khoảng cách frame trước
        self.prev_min_dist = None

    def analyze_scene(self, yolo_result, midas_tensor, target_shape):
        h, w = target_shape

        depth_map = torch.nn.functional.interpolate(
            midas_tensor.unsqueeze(1),
            size=(h, w),
            mode="bilinear",
            align_corners=False
        ).squeeze().cpu().numpy()

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
                
                if bw * bh <= 400:
                    continue

                if y2 > h * 0.95 and bw > w * 0.6:
                    continue

                mask = cv2.resize(masks[i], (w, h), interpolation=cv2.INTER_NEAREST)
                
                y_bottom_start = int(y1 + bh * 0.8)
                bottom_mask = mask.copy()
                bottom_mask[:y_bottom_start, :] = 0 
                
                valid_depths = depth_map[bottom_mask == 1]
                if len(valid_depths) > 0:
                    mean_depth = np.mean(valid_depths)
                else:
                    mean_depth = np.mean(depth_map[mask == 1])
                
                distance = DepthModule.estimate_distance(mean_depth, bh)
                
                center_x = x1 + (bw / 2)
                is_in_path = safe_zone_left < center_x < safe_zone_right
                
                detected_cars.append({
                    'bbox': (x1, y1, bw, bh),
                    'distance': distance,
                    'mask': mask,
                    'is_in_path': is_in_path 
                })

                if is_in_path and distance < raw_min_distance:
                    raw_min_distance = distance

        # ===== EMA =====
        if raw_min_distance != 999.0:
            if self.smoothed_min_dist is None:
                self.smoothed_min_dist = raw_min_distance
            else:
                self.smoothed_min_dist = (
                    0.3 * raw_min_distance + 0.7 * self.smoothed_min_dist
                )
        else:
            self.smoothed_min_dist = 999.0

        # phát hiện đang tiến lại gần
        is_closing = False
        if self.prev_min_dist is not None and self.smoothed_min_dist != 999.0:
            delta = self.prev_min_dist - self.smoothed_min_dist
            
            # Nếu khoảng cách giảm đủ lớn → đang tiến lại
            if delta > 0.1:
                is_closing = True

        # cập nhật cho frame sau
        self.prev_min_dist = self.smoothed_min_dist

        # CHỈ cảnh báo khi: gần + đang tiến lại
        if self.smoothed_min_dist < self.danger_threshold and is_closing:
            danger_flag = True

        return None, detected_cars, danger_flag, self.smoothed_min_dist, is_closing