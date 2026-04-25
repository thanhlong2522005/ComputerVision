import cv2
import numpy as np
import winsound  # Thư viện âm thanh siêu nhẹ có sẵn của Windows (Không cần pip install)
import threading
import time

class UIAndVoiceManager:
    def __init__(self):
        print("Khởi tạo hệ thống Âm thanh & UI...")
        self.last_alert_time = 0
        self.cooldown = 1.0 # Giảm cooldown xuống 1s vì tiếng Beep rất ngắn và dứt khoát

    def trigger_warning_sound(self):
        current_time = time.time()
        if current_time - self.last_alert_time > self.cooldown:
            self.last_alert_time = current_time
            
            # Vẫn cho vào Thread để tiếng Beep không làm đứng khung hình video
            def beep_task():
                try:
                    # Phát tiếng Beep cảnh báo (Tần số 1500Hz, Kéo dài 400ms)
                    # Nghe y hệt tiếng radar cảnh báo trên xe thật!
                    winsound.Beep(1500, 400)
                except: pass
                
            threading.Thread(target=beep_task, daemon=True).start()

    def render_ui(self, frame, seg_mask, cars, danger_flag, min_dist, fps):
        display = frame.copy()
        
        # Layer overlay để vẽ mask trộn màu siêu tốc
        overlay = display.copy()

        for car in cars:
            x, y, w, h = car['bbox']
            dist = car['distance']
            mask = car['mask']
            
            # Chỉ tô màu lên layer overlay
            overlay[mask == 1] = [0, 0, 255] # BGR
            
            # Vẽ Box và Text lên ảnh gốc
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(display, f"{dist:.1f}m", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        # Trộn màu 1 LẦN DUY NHẤT cho tất cả các xe
        if cars:
            cv2.addWeighted(overlay, 0.4, display, 0.6, 0, display)

        # Logic Dashboard & Cảnh báo toàn màn hình
        status_color = (0, 255, 0)
        status_text = "SAFE"

        if danger_flag:
            status_color = (0, 0, 255)
            status_text = "DANGER! BRAKE!"
            # Chớp viền đỏ
            cv2.rectangle(display, (0, 0), display.shape[1::-1], (0, 0, 255), 10)
            # Kích hoạt tiếng Beep của Windows
            self.trigger_warning_sound()
        elif min_dist < 15.0:
            status_color = (0, 165, 255)
            status_text = "WARNING"

        # Vẽ Bảng điều khiển (Dashboard)
        cv2.rectangle(display, (10, 10), (320, 100), (0, 0, 0), -1)
        cv2.putText(display, f"FPS: {fps:.1f}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display, f"Closest: {min_dist:.1f}m" if min_dist != 999.0 else "Closest: None", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display, f"Action: {status_text}", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        return display