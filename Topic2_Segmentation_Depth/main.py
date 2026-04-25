import torch
import cv2

from segmentation import SegmentationModule
from depth import DepthModule
from performance import PerformanceTracker
from core_logic import CoreLogicAnalyzer
from ui_ux import UIAndVoiceManager

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Bắt đầu chạy LIVE REAL-TIME (Tối ưu hóa CPU) trên: {device}")

    seg_module = SegmentationModule(model_path="yolov8n-seg.pt", device=device)
    depth_module = DepthModule(device)
    perf_tracker = PerformanceTracker()
    logic_analyzer = CoreLogicAnalyzer(danger_threshold=5.0)
    ui_manager = UIAndVoiceManager()

    video_input = "test_video.mp4" 
    cap = cv2.VideoCapture(video_input)
    
    if not cap.isOpened():
        print("Lỗi: Không đọc được video!")
        return

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    target_w = 640
    target_h = int((640 / orig_w) * orig_h)

    print("Hệ thống sẵn sàng! Đang mở cửa sổ Video...")

    FRAME_SKIP = 3  # Chỉ chạy AI mỗi 3 frame (Tăng số này lên FPS càng cao)
    frame_counter = 0
    
    # Biến lưu trữ (Cache) kết quả của Frame trước đó
    cached_cars = []
    cached_danger = False
    cached_min_dist = 999.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (target_w, target_h))
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # CHỈ GỌI AI KHI BỘ ĐẾM CHIA HẾT CHO FRAME_SKIP
        if frame_counter % FRAME_SKIP == 0:
            yolo_result = seg_module.predict(img_rgb)
            midas_tensor = depth_module.predict(img_rgb)

            _, cached_cars, cached_danger, cached_min_dist = logic_analyzer.analyze_scene(
                yolo_result, midas_tensor, (target_h, target_w)
            )

        # Lấy Cache vẽ ra UI liên tục ở mọi Frame -> FPS rất cao
        current_fps = perf_tracker.update()
        final_frame = ui_manager.render_ui(frame, None, cached_cars, cached_danger, cached_min_dist, current_fps)

        cv2.imshow("He thong ADAS (CPU Optimized)", final_frame)
        
        # Tăng bộ đếm
        frame_counter += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()