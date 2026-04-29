import cv2
import numpy as np
import torch
from depth import DepthModule 

class CoreLogicAnalyzer:
    def __init__(self, danger_threshold=5.0):
        self.danger_threshold = danger_threshold 

    def analyze_scene(self, yolo_result, midas_tensor, target_shape):
        h, w = target_shape

        # TỐI ƯU 2: Đổi "bicubic" thành "bilinear" (tốn ít tính toán CPU hơn)
        depth_map = torch.nn.functional.interpolate(
            midas_tensor.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False
        ).squeeze().cpu().numpy()

        detected_cars = []
        
        if yolo_result.masks is not None and yolo_result.boxes is not None:
            masks = yolo_result.masks.data.cpu().numpy() 
            boxes = yolo_result.boxes.xyxy.cpu().numpy() 
            
            for i in range(len(masks)):
                x1, y1, x2, y2 = map(int, boxes[i])
                bw, bh = x2 - x1, y2 - y1
                
                if bw * bh <= 150: 
                    continue

                if y2 > h * 0.95 and bw > w * 0.6:
                    continue

                center_x = x1 + (bw / 2)
                
                is_on_the_edge = (center_x < w * 0.20 or center_x > w * 0.80)
                
                if is_on_the_edge:
                    if bw * bh < 4000:
                        continue

                mask = cv2.resize(masks[i], (w, h), interpolation=cv2.INTER_NEAREST)
                
                mask = cv2.resize(masks[i], (w, h), interpolation=cv2.INTER_NEAREST)
                
                y_bottom_start = int(y1 + bh * 0.8)
                
                bottom_mask = mask.copy()
                bottom_mask[:y_bottom_start, :] = 0 
                
                valid_depths = depth_map[bottom_mask == 1]
                if len(valid_depths) > 0:
                    mean_depth = np.mean(valid_depths)
                else:
                    mean_depth = np.mean(depth_map[mask == 1])
                
                distance = DepthModule.estimate_distance(mean_depth)
                # -----------------------------------------------------
                
                detected_cars.append({
                    'bbox': (x1, y1, bw, bh),
                    'distance': distance,
                    'mask': mask 
                })

        danger_flag = False
        min_distance = 999.0
        if detected_cars:
            min_distance = min(car['distance'] for car in detected_cars)
            if min_distance < self.danger_threshold:
                danger_flag = True

        return None, detected_cars, danger_flag, min_distance