import cv2
import numpy as np
import sys
from pathlib import Path

# 1. Tu dong tim duong dan den thu muc cua TV1
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import module tu TV1 va TV2
try:
    # Luu y: Dam bao thu muc 'member1_segmentation' co file '__init__.py'
    from member1_segmentation.backends import create_segmenter, LABELS
    from depth_estimation import MidasDepthEstimator
    print(">>> KET NOI THANH CONG: Da nhan dien duoc module Segmentation va Depth")
except Exception as e:
    print(f">>> LOI KET NOI: {e}")
    print("Hay dam bao thu muc member1_segmentation nam cung cho voi file nay.")

class CoreLogic:
    def __init__(self):
        # Khoi tao AI cua TV1 (Nhan dien vat the)
        # Neu may co card do hoa Nvidia, no se tu dung CUDA cho nhanh
        self.segmenter = create_segmenter()
        
        # Khoi tao AI cua TV2 (Tinh do sau)
        self.depth_model = MidasDepthEstimator(model_type="MiDaS_small")
        
        # Thong so logic
        self.RESIZE_W = 512      # Kich thuoc chuan de 2 ben khop nhau
        self.DANGER_THRESH = 170 # Nguong canh bao nguy hiem (Dua tren ket qua hom qua)
        
        # ID doi tuong theo quy dinh cua TV1 trong backends.py
        self.ID_VEHICLE = 3 
        self.ID_PEDESTRIAN = 4

    def process_logic(self, frame):
        # Buoc 1: Resize ve size chuan de TV1 va TV2 tinh toan tren cung 1 ma tran
        h_orig, w_orig = frame.shape[:2]
        scale = self.RESIZE_W / float(w_orig)
        new_h = int(h_orig * scale)
        frame_sm = cv2.resize(frame, (self.RESIZE_W, new_h))

        # Buoc 2: Lay ket qua tu 2 module
        seg_map = self.segmenter(frame_sm)      # Tra ve map chua ID 3 (xe), 4 (nguoi)
        depth_map = self.depth_model.process_frame(frame_sm) # Tra ve map do sau (0-255)

        # Buoc 3: LOGIC TRICH XUAT DOI TUONG (Object Extraction)
        # Tim tat ca cac diem anh la XE hoac NGUOI
        target_pixels = np.where((seg_map == self.ID_VEHICLE) | (seg_map == self.ID_PEDESTRIAN))
        
        is_danger = False
        avg_depth = 0
        
        if len(target_pixels[0]) > 0:
            # Tính độ sâu trung bình CHỈ TẠI VÙNG có xe/người         
            avg_depth = np.mean(depth_map[target_pixels])
            if avg_depth < self.DANGER_THRESH:
                is_danger = True
        return is_danger, avg_depth, depth_map, seg_map, frame_sm
if __name__ == "__main__":
    core = CoreLogic()
    video_path = "test_traffic.mp4"
    
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"!!! LOI: Khong tim thay file {video_path} trong thu muc.")
    else:
        print(">>> DA KET NOI VIDEO OFFLINE. He thong bat dau chay...")

    print("--- DANG CHAY HE THONG HOP NHAT (Nhan 'q' de thoat) ---")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        danger, dist, d_map, s_map, frame_sm = core.process_logic(frame)
        
        # Tao hinh anh hien thi (Ket hop mau nhiet do sau)
        depth_color = cv2.applyColorMap(d_map, cv2.COLORMAP_INFERNO)
        
        # Ve canh bao len man hinh
        msg = "NGUY HIEM - PHANH GAP!" if danger else "AN TOAN"
        color = (0, 0, 255) if danger else (0, 255, 0)
        
        cv2.putText(depth_color, msg, (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
        cv2.putText(depth_color, f"Distance Score: {dist:.1f}", (20, 90), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)

        cv2.imshow("HE THONG CANH BAO VA CHAM THONG MINH", depth_color)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()