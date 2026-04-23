import torch
import cv2
import segmentation_models_pytorch as smp

class SegmentationModule:
    def __init__(self, model_path, device):
        self.device = device
        print("Đang tải mô hình U-Net...")
        # Cấu hình kiến trúc U-Net
        self.model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=6)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()

    def predict(self, img_rgb):
        # Tiền xử lý riêng cho U-Net (Resize 512x512)
        img_resized = cv2.resize(img_rgb, (512, 512))
        input_tensor = torch.tensor(img_resized / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        # Suy luận (Inference)
        with torch.no_grad():
            output = self.model(input_tensor)
            
        return output