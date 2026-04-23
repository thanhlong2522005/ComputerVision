import torch
import cv2

from segmentation import SegmentationModule
from depth import DepthModule
from performance import PerformanceTracker
from core_logic import CoreLogicAnalyzer
from ui_ux import UIAndVoiceManager


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    seg_module    = SegmentationModule("unet_road_car_sky(class6).pth", device)
    depth_module  = DepthModule(device)
    perf_tracker  = PerformanceTracker()
    logic_analyzer = CoreLogicAnalyzer(danger_threshold=5.0)
    ui_manager    = UIAndVoiceManager(output_path="output_demo.mp4")

    cap = cv2.VideoCapture("test_video.mp4")
    if not cap.isOpened():
        print("Error: cannot open video.")
        return

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    target_w = 1024
    target_h = int((target_w / orig_w) * orig_h)

    ui_manager.init_video_writer(target_w, target_h, src_fps)
    print("Ready. Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video.")
                break

            frame = cv2.resize(frame, (target_w, target_h))
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            unet_tensor  = seg_module.predict(img_rgb)
            midas_tensor = depth_module.predict(img_rgb)

            seg_mask, cars, danger_flag, min_dist = logic_analyzer.analyze_scene(
                unet_tensor, midas_tensor, (target_h, target_w)
            )

            final_frame = ui_manager.render_ui(
                frame, seg_mask, cars, danger_flag, min_dist, perf_tracker.update()
            )

            cv2.imshow("ADAS – Collision Warning", final_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        ui_manager.release()
        cv2.destroyAllWindows()
        ui_manager.merge_audio_into_video()


if __name__ == "__main__":
    main()
