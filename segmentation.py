import torch
import cv2
import segmentation_models_pytorch as smp


class SegmentationModule:
    def __init__(self, model_path: str, device: torch.device):
        self.device = device
        print("[Seg] Loading U-Net model...")
        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=6,
        )
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device).eval()

    def predict(self, img_rgb) -> torch.Tensor:
        img = cv2.resize(img_rgb, (512, 512))
        tensor = (
            torch.tensor(img / 255.0, dtype=torch.float32)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device)
        )
        with torch.no_grad():
            return self.model(tensor)
