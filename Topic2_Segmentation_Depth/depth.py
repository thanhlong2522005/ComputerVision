import torch

class DepthModule:
    def __init__(self, device):
        self.device = device
        print("Đang tải mô hình MiDaS...")
        self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").to(device)
        self.model.eval()
        self.transforms = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

    def predict(self, img_rgb):
        input_batch = self.transforms(img_rgb).to(self.device)
        with torch.no_grad():
            prediction = self.model(input_batch)
        return prediction 

    @staticmethod
    def estimate_distance(depth_value, bbox_height):
        # 1. Khoảng cách dựa theo AI (MiDaS)
        midas_dist = 2000 / (depth_value + 1e-6)
        
        # 2. Khoảng cách Vật lý (Thuật toán Pinhole Camera Model)
        # Giả định xe ô tô cao trung bình 1.5m, tiêu cự camera (focal length) ~ 700
        # Xe trên màn hình (bbox_height) càng to thì khoảng cách càng nhỏ
        physics_dist = (1.5 * 700) / (bbox_height + 1e-6)
        
        # Lấy trung bình cộng (Lai ghép) để bù trừ sai số cho nhau
        final_dist = (midas_dist * 0.4) + (physics_dist * 0.6)
        
        # Giới hạn khoảng cách tối đa 50m để số không bị nhảy lên mấy trăm mét
        return min(final_dist, 50.0)