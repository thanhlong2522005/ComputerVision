import torch
import cv2

from segmentation import SegmentationModule
from depth import DepthModule
from performance import PerformanceTracker
from core_logic import CoreLogicAnalyzer
from ui_ux import UIAndVoiceManager

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Bắt đầu chạy LIVE REAL-TIME trên: {device}")

    # 1. Khởi tạo AI
    model_path = "unet_road_car_sky(class6).pth"
    seg_module = SegmentationModule(model_path, device)
    depth_module = DepthModule(device)
    perf_tracker = PerformanceTracker()
    logic_analyzer = CoreLogicAnalyzer(danger_threshold=5.0)
    ui_manager = UIAndVoiceManager()

    # 2. Mở Video
    video_input = "test_video.mp4" # Tên video của bạn
    cap = cv2.VideoCapture(video_input)
    
    if not cap.isOpened():
        print("Lỗi: Không đọc được video!")
        return

    # Resize để chạy nhẹ hơn trên màn hình
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    target_w = 1024
    target_h = int((1024 / orig_w) * orig_h)

    print("Hệ thống sẵn sàng! Đang mở cửa sổ Video... (Bấm phím 'q' để thoát)")

    # 3. VÒNG LẶP CHẠY TRỰC TIẾP LÊN MÀN HÌNH
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Đã phát hết video!")
            break

        # Tiền xử lý
        frame = cv2.resize(frame, (target_w, target_h))
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Chạy AI
        unet_tensor = seg_module.predict(img_rgb)
        midas_tensor = depth_module.predict(img_rgb)

        # Phân tích Logic
        seg_mask, cars, danger_flag, min_dist = logic_analyzer.analyze_scene(
            unet_tensor, midas_tensor, (target_h, target_w)
        )

        # Đo FPS
        current_fps = perf_tracker.update()

        # Vẽ UI và kích hoạt loa
        final_frame = ui_manager.render_ui(frame, seg_mask, cars, danger_flag, min_dist, current_fps)

        cv2.imshow("Hệ thống ADAS (Cảnh báo va chạm)", final_frame)
        
        # Bấm phím 'q' trên bàn phím để tắt video giữa chừng
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Đã thoát chương trình!")
            break

    # Dọn dẹp cửa sổ khi chạy xong
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()