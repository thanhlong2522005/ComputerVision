import cv2
import numpy as np
import torch
from depth import DepthModule

_MIN_CAR_AREA = 800  # px² — filters segmentation noise


class CoreLogicAnalyzer:
    def __init__(self, danger_threshold: float = 5.0):
        self.danger_threshold = danger_threshold

    def analyze_scene(
        self,
        unet_tensor: torch.Tensor,
        midas_tensor: torch.Tensor,
        target_shape: tuple,
    ):
        h, w = target_shape

        seg_mask = torch.nn.functional.interpolate(
            unet_tensor, size=(h, w), mode="bilinear", align_corners=False
        )
        seg_mask = torch.argmax(seg_mask, dim=1).squeeze().cpu().numpy()

        depth_map = torch.nn.functional.interpolate(
            midas_tensor.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False
        ).squeeze().cpu().numpy()

        car_mask = (seg_mask == 2).astype(np.uint8)
        contours, _ = cv2.findContours(car_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_cars = []
        for cnt in contours:
            if cv2.contourArea(cnt) < _MIN_CAR_AREA:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            single_mask = np.zeros_like(car_mask)
            cv2.drawContours(single_mask, [cnt], -1, 1, thickness=cv2.FILLED)
            mean_depth = np.mean(depth_map[single_mask == 1])
            detected_cars.append({
                "bbox": (x, y, bw, bh),
                "distance": DepthModule.estimate_distance(mean_depth),
            })

        min_distance = min((c["distance"] for c in detected_cars), default=999.0)
        danger_flag = bool(detected_cars) and min_distance < self.danger_threshold

        return seg_mask, detected_cars, danger_flag, min_distance
