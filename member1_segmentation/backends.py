from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


LABELS: Dict[int, str] = {
    0: "background",
    1: "road",
    2: "sky",
    3: "vehicle",
    4: "pedestrian",
}

PALETTE: Dict[int, Tuple[int, int, int]] = {
    0: (0, 0, 0),
    1: (60, 160, 60),
    2: (200, 120, 40),
    3: (40, 40, 220),
    4: (220, 220, 40),
}

CITYSCAPES_TO_PROJECT = {
    0: 1,   # road
    10: 2,  # sky
    11: 4,  # person
    12: 4,  # rider
    13: 3,  # car
    14: 3,  # truck
    15: 3,  # bus
    16: 3,  # train
    17: 3,  # motorcycle
    18: 3,  # bicycle
}


@dataclass
class SegmentationResult:
    seg_map: np.ndarray
    stats: Dict[str, float]


def compute_stats(seg_map: np.ndarray) -> Dict[str, float]:
    total = float(seg_map.size)
    return {
        label_name: round(float(np.count_nonzero(seg_map == label_id)) / total, 4)
        for label_id, label_name in LABELS.items()
    }


def normalize_device(torch_module, device: Optional[str]) -> str:
    if device:
        return device
    return "cuda" if torch_module.cuda.is_available() else "cpu"


def remap_labels(pred: np.ndarray, mapping: Dict[int, int]) -> np.ndarray:
    seg_map = np.zeros_like(pred, dtype=np.uint8)
    for source_id, target_id in mapping.items():
        seg_map[pred == source_id] = target_id
    return seg_map


class SegFormerCityscapesSegmenter:
    def __init__(
        self,
        device: Optional[str] = None,
        model_name: str = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
    ) -> None:
        try:
            import torch
            from PIL import Image
            from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
        except ImportError as exc:
            raise RuntimeError(
                "This project requires torch, torchvision, pillow, and transformers."
            ) from exc

        self.torch = torch
        self.Image = Image
        self.device = normalize_device(torch, device)
        self.processor = SegformerImageProcessor.from_pretrained(model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def predict(self, frame_bgr: np.ndarray) -> SegmentationResult:
        height, width = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_pil = self.Image.fromarray(frame_rgb)
        inputs = self.processor(images=frame_pil, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with self.torch.inference_mode():
            outputs = self.model(**inputs)
            pred = self.processor.post_process_semantic_segmentation(
                outputs,
                target_sizes=[(height, width)],
            )[0].detach().cpu().numpy().astype(np.uint8)

        seg_map = remap_labels(pred, CITYSCAPES_TO_PROJECT)
        return SegmentationResult(seg_map=seg_map, stats=compute_stats(seg_map))

    def __call__(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Direct call to get segmentation map."""
        return self.predict(frame_bgr).seg_map


def create_segmenter(device: Optional[str] = None, model_name: Optional[str] = None) -> SegFormerCityscapesSegmenter:
    return SegFormerCityscapesSegmenter(
        device=device,
        model_name=model_name or "nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
    )
