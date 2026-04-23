import torch

class DepthModule:
    def __init__(self, device):
        self.device = device
        print("[TV2] Đang tải mô hình MiDaS...")
        self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").to(device)
        self.model.eval()
        self.transforms = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

    def predict(self, img_rgb):
        # Tiền xử lý bằng bộ transform chuẩn của MiDaS
        input_batch = self.transforms(img_rgb).to(self.device)
        
        # Suy luận
        with torch.no_grad():
            prediction = self.model(input_batch)
            
        return prediction # Trả về Tensor thô

    @staticmethod
    def estimate_distance(depth_value):
        # Công thức chuẩn hóa relative depth sang mét (mô phỏng)
        return 2000 / (depth_value + 1e-6)