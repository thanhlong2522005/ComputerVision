import torch


class DepthModule:
    def __init__(self, device: torch.device):
        self.device = device
        print("[Depth] Loading MiDaS model...")
        self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").to(device).eval()
        self.transforms = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

    def predict(self, img_rgb) -> torch.Tensor:
        batch = self.transforms(img_rgb).to(self.device)
        with torch.no_grad():
            return self.model(batch)

    @staticmethod
    def estimate_distance(depth_value: float) -> float:
        # Inverse-depth heuristic calibrated for dashcam footage
        return 2000.0 / (depth_value + 1e-6)
