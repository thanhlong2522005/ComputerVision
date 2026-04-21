import cv2
import torch
import numpy as np

class MidasDepthEstimator:
    def __init__(self, model_type="MiDaS_small"):
        """
        Khởi tạo mô hình MiDaS.
        Nên dùng "MiDaS_small" cho realtime vì nó nhẹ, tốc độ cao. 
        Nếu muốn chính xác hơn nhưng chậm, dùng "DPT_Hybrid" hoặc "DPT_Large".
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Đang tải model MiDaS ({model_type}) trên {self.device}...")
        
        # Load model từ PyTorch Hub
        self.midas = torch.hub.load("intel-isl/MiDaS", model_type)
        self.midas.to(self.device)
        self.midas.eval()

        # Load transforms tương ứng
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if model_type in ["DPT_Large", "DPT_Hybrid"]:
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform
            
        print("Model tải xong và sẵn sàng!")

    def process_frame(self, frame):
        """
        Đầu vào: 1 frame (numpy array) từ OpenCV[cite: 59].
        Đầu ra: depth_map đã chuẩn hóa (0-255)[cite: 55].
        """
        # Chuyển BGR (OpenCV) sang RGB
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Tiền xử lý
        input_batch = self.transform(img).to(self.device)

        # Chạy model
        with torch.no_grad():
            prediction = self.midas(input_batch)

            # Resize kết quả về đúng kích thước ảnh gốc [cite: 55]
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        # Chuyển về numpy array
        depth_map = prediction.cpu().numpy()
        
        # Chuẩn hóa (Normalize) về dải 0 - 255 để dễ visualize và đặt ngưỡng [cite: 55]
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        if depth_max - depth_min > 0:
            depth_normalized = 255 * (depth_map - depth_min) / (depth_max - depth_min)
        else:
            depth_normalized = np.zeros_like(depth_map)
            
        return depth_normalized.astype(np.uint8)

    def classify_distance(self, depth_normalized, danger_ratio=0.85, safe_ratio=0.60):
        """
        Chiến thuật Ngưỡng Động (Adaptive Threshold).
        - danger_ratio = 0.85: Top 15% giá trị sáng nhất sẽ bị đánh dấu 'near'
        - safe_ratio = 0.60: Từ 60% đến 85% sẽ là 'medium'
        """
        # Dùng percentile (phân vị) thay vì max() để lọc nhiễu.
        # Nó sẽ bỏ qua những điểm ảnh sáng chói bất thường (do lỗi cam hoặc phản sáng).
        # Lấy mốc 95% coi như là "khoảng cách gần nhất thực tế" của khung hình.
        v_max = np.percentile(depth_normalized, 95)
        
        # Tự động nội suy ra ngưỡng cho frame hiện tại
        thresh_near = v_max * danger_ratio
        thresh_far = v_max * safe_ratio
        
        # Phân loại
        distance_level = np.full(depth_normalized.shape, "far", dtype=object)
        distance_level[depth_normalized >= thresh_near] = "near"
        distance_level[(depth_normalized < thresh_near) & (depth_normalized >= thresh_far)] = "medium"
        
        return distance_level