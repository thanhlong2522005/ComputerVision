import cv2
import numpy as np
import pyttsx3
import threading
import time

class UIAndVoiceManager:
    def __init__(self):
        print("Khởi tạo hệ thống Âm thanh & UI...")
        self.last_alert_time = 0
        self.cooldown = 3.0 # Chống spam loa

    def trigger_voice_warning(self):
        current_time = time.time()
        if current_time - self.last_alert_time > self.cooldown:
            self.last_alert_time = current_time
            
            def tts_task():
                try:
                    engine = pyttsx3.init()
                    engine.setProperty('rate', 150)
                    engine.say("Danger! Brake now!")
                    engine.runAndWait()
                except: pass
                
            threading.Thread(target=tts_task, daemon=True).start()

    def render_ui(self, frame, seg_mask, cars, danger_flag, min_dist, fps):
        display = frame.copy()

        # 1. Overlay màu (Đường -> Xanh lá, Trời -> Xanh dương)
        road_mask = (seg_mask == 1)
        sky_mask = (seg_mask == 3)
        display[road_mask] = display[road_mask] * 0.6 + np.array([0, 255, 0]) * 0.4
        display[sky_mask] = display[sky_mask] * 0.7 + np.array([255, 0, 0]) * 0.3

        # 2. Vẽ xe và khoảng cách
        for car in cars:
            x, y, w, h = car['bbox']
            dist = car['distance']
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(display, f"{dist:.1f}m", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        # 3. Logic Dashboard & Cảnh báo toàn màn hình
        status_color = (0, 255, 0)
        status_text = "SAFE"

        if danger_flag:
            status_color = (0, 0, 255)
            status_text = "DANGER! BRAKE!"
            # Chớp đỏ viền màn hình và phát loa
            cv2.rectangle(display, (0, 0), display.shape[1::-1], (0, 0, 255), 10)
            self.trigger_voice_warning()
        elif min_dist < 15.0:
            status_color = (0, 165, 255)
            status_text = "WARNING"

        # 4. Vẽ Bảng điều khiển (Dashboard)
        cv2.rectangle(display, (10, 10), (400, 120), (0, 0, 0), -1)
        cv2.putText(display, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display, f"Closest: {min_dist:.1f}m" if min_dist != 999.0 else "Closest: None", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display, f"Action: {status_text}", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        return display