import cv2
import numpy as np
import winsound 
import threading
import time

class UIAndVoiceManager:
    def __init__(self):
        print("Khởi tạo hệ thống Âm thanh & UI...")
        self.last_alert_time = 0
        self.cooldown = 0.5

    def trigger_warning_sound(self):
        current_time = time.time()
        if current_time - self.last_alert_time > self.cooldown:
            self.last_alert_time = current_time
            def beep_task():
                try: winsound.Beep(1500, 400)
                except: pass
            threading.Thread(target=beep_task, daemon=True).start()

    def render_ui(self, frame, seg_mask, cars, danger_flag, min_dist, fps, is_closing):
        display = frame.copy()
        h, w = display.shape[:2]
        
        # Vẽ 2 vạch giả lập "Làn đường của mình" (ROI Corridor)
        # cv2.line(display, (int(w*0.3), h), (int(w*0.45), int(h*0.6)), (255, 255, 255), 2, cv2.LINE_AA)
        # cv2.line(display, (int(w*0.7), h), (int(w*0.55), int(h*0.6)), (255, 255, 255), 2, cv2.LINE_AA)

        overlay = display.copy()

        for car in cars:
            x, y, bw, bh = car['bbox']
            dist = car['distance']
            mask = car['mask']
            is_in_path = car['is_in_path']
            
            # Phân loại màu sắc: Xe cản đường (Đỏ) - Xe bên lề (Cyan/Xanh nhạt)
            box_color = (0, 0, 255) if is_in_path else (255, 255, 0)
            
            overlay[mask == 1] = box_color # Tô màu mask
            
            cv2.rectangle(display, (x, y), (x + bw, y + bh), box_color, 2)
            cv2.putText(display, f"{dist:.1f}m", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2, cv2.LINE_AA)

        if cars:
            cv2.addWeighted(overlay, 0.4, display, 0.6, 0, display)

        status_color = (0, 255, 0)
        status_text = "SAFE"

        if danger_flag and is_closing:
            status_color = (0, 0, 255)
            status_text = "DANGER! BRAKE!"
            cv2.rectangle(display, (0, 0), (w, h), (0, 0, 255), 10)
            self.trigger_warning_sound()
        elif min_dist < 15.0 and min_dist != 999.0:
            status_color = (0, 165, 255)
            status_text = "WARNING"
        elif min_dist < 6.0 and not is_closing:
            status_color = (255, 255, 0)
            status_text = "STOPPED (SAFE)"

        cv2.rectangle(display, (10, 10), (320, 100), (0, 0, 0), -1)
        cv2.putText(display, f"FPS: {fps:.1f}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display, f"In Path: {min_dist:.1f}m" if min_dist != 999.0 else "In Path: Clear", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display, f"Action: {status_text}", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        return display