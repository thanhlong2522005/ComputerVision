import cv2
import numpy as np
import pyttsx3
import threading
import time
import os

_SEG_COLORS = {
    1: (50, 205, 50),
    3: (205, 133, 63),
    4: (0, 215, 215),
    5: (215, 0, 215),
}


def _dist_color(dist: float):
    if dist < 5.0:
        return (0, 0, 255)
    if dist < 15.0:
        return (0, 165, 255)
    return (50, 220, 50)


class UIAndVoiceManager:
    def __init__(self, output_path: str = "output_demo.mp4"):
        self.output_path = output_path
        self._writer = None
        self._src_fps = 25.0
        self._frame_count = 0

        self.last_alert_time = 0.0
        self.alert_cooldown = 3.0
        self.flash_end_time = 0.0

        self._warning_timestamps: list[float] = []
        self._audio_clip_path = "_warning_sound.wav"
        self._generate_warning_audio()
        print("[UI] System ready.")

    # ------------------------------------------------------------------ #
    #  Audio                                                               #
    # ------------------------------------------------------------------ #
    def _generate_warning_audio(self):
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 140)
            engine.setProperty("volume", 1.0)
            engine.save_to_file("Danger! Brake now!", self._audio_clip_path)
            engine.runAndWait()
            engine.stop()
            if os.path.exists(self._audio_clip_path):
                print(f"[TTS] Warning audio ready: {self._audio_clip_path}")
            else:
                print("[TTS] WARNING: WAV file was not created.")
        except Exception as e:
            print(f"[TTS] Could not pre-generate WAV: {e}")

    def _trigger_voice(self, current_time_sec: float):
        now = time.time()
        if now - self.last_alert_time < self.alert_cooldown:
            return
        self.last_alert_time = now
        self.flash_end_time = now + 0.6
        self._warning_timestamps.append(current_time_sec)
        print(f"[ALERT] Danger at {current_time_sec:.1f}s")

        # winsound.PlaySound + SND_ASYNC: non-blocking, no COM/thread issues on Windows
        try:
            import winsound
            if os.path.exists(self._audio_clip_path):
                winsound.PlaySound(
                    self._audio_clip_path,
                    winsound.SND_FILENAME | winsound.SND_ASYNC,
                )
            else:
                winsound.Beep(880, 600)
        except Exception:
            # Fallback for non-Windows
            def _speak():
                try:
                    engine = pyttsx3.init()
                    engine.setProperty("rate", 140)
                    engine.setProperty("volume", 1.0)
                    engine.say("Danger! Brake now!")
                    engine.runAndWait()
                    engine.stop()
                except Exception as e:
                    print(f"[TTS] {e}")
            threading.Thread(target=_speak, daemon=True).start()

    def merge_audio_into_video(self):
        if not self._warning_timestamps or not os.path.exists(self._audio_clip_path):
            return

        try:
            from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
        except ImportError:
            print("[Audio] Install moviepy: pip install moviepy")
            return

        try:
            print(f"[Audio] Merging {len(self._warning_timestamps)} alert(s) into video...")
            video = VideoFileClip(self.output_path)
            clip = AudioFileClip(self._audio_clip_path)

            audio_clips = [
                clip.copy().set_start(ts)
                for ts in self._warning_timestamps
                if ts < video.duration
            ]

            if audio_clips:
                out_path = self.output_path.replace(".mp4", "_with_audio.mp4")
                (video
                 .set_audio(CompositeAudioClip(audio_clips))
                 .write_videofile(out_path, codec="libx264", audio_codec="aac", logger=None))
                print(f"[Audio] Saved: {out_path}")

            video.close()
            clip.close()
            os.remove(self._audio_clip_path)
        except Exception as e:
            print(f"[Audio] Error: {e}")

    # ------------------------------------------------------------------ #
    #  Video recording                                                     #
    # ------------------------------------------------------------------ #
    def init_video_writer(self, frame_w: int, frame_h: int, fps: float):
        self._src_fps = fps
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(self.output_path, fourcc, fps, (frame_w, frame_h))
        if self._writer.isOpened():
            print(f"[Recorder] Recording -> {self.output_path} ({frame_w}x{frame_h} @ {fps:.1f}fps)")
        else:
            print("[Recorder] WARNING: Could not open VideoWriter.")
            self._writer = None

    def _record(self, frame: np.ndarray):
        if self._writer and self._writer.isOpened():
            self._writer.write(frame)

    def release(self):
        if self._writer and self._writer.isOpened():
            self._writer.release()
            print(f"[Recorder] Saved: {self.output_path}")

    # ------------------------------------------------------------------ #
    #  Drawing                                                             #
    # ------------------------------------------------------------------ #
    def _draw_seg_overlay(self, canvas: np.ndarray, seg_mask: np.ndarray):
        overlay = canvas.copy()
        for cls_id, color in _SEG_COLORS.items():
            mask = seg_mask == cls_id
            if mask.any():
                overlay[mask] = color
        cv2.addWeighted(overlay, 0.30, canvas, 0.70, 0, canvas)

    def _draw_car_boxes(self, canvas: np.ndarray, cars: list):
        for car in cars:
            x, y, w, h = car["bbox"]
            dist = car["distance"]
            color = _dist_color(dist)
            thickness = 3 if dist < 5.0 else 2

            cv2.rectangle(canvas, (x, y), (x + w, y + h), color, thickness)

            cl = max(14, min(w, h) // 5)
            for cx, cy, sx, sy in [(x, y, 1, 1), (x+w, y, -1, 1),
                                    (x, y+h, 1, -1), (x+w, y+h, -1, -1)]:
                cv2.line(canvas, (cx, cy), (cx + sx * cl, cy), color, 3)
                cv2.line(canvas, (cx, cy), (cx, cy + sy * cl), color, 3)

            label = f"{dist:.1f}m"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            ly = max(y - 6, th + 6)
            cv2.rectangle(canvas, (x - 2, ly - th - 4), (x + tw + 6, ly + 4), color, -1)
            cv2.putText(canvas, label, (x + 2, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)

            if dist < 5.0:
                cv2.putText(canvas, "!", (x + w // 2 - 8, y + h // 2 + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, cv2.LINE_AA)

    def _draw_danger_flash(self, canvas: np.ndarray):
        remaining = self.flash_end_time - time.time()
        if remaining <= 0:
            return
        thick = max(4, int(14 * remaining / 0.6))
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1] - 1, canvas.shape[0] - 1),
                      (0, 0, 255), thick)

    def _draw_warning_banner(self, canvas: np.ndarray):
        h, w = canvas.shape[:2]
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, h - 48), (w, h), (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.82, canvas, 0.18, 0, canvas)
        if int(time.time() * 2) % 2 == 0:
            msg = "!!! CANH BAO: QUA GAN - GIAM TOC NGAY !!!"
            (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2)
            cv2.putText(canvas, msg, (max(0, (w - tw) // 2), h - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)

    def _draw_dashboard(self, canvas: np.ndarray, cars: list,
                        danger_flag: bool, min_dist: float, fps: float):
        px, py, pw, ph = 12, 12, 330, 138
        overlay = canvas.copy()
        cv2.rectangle(overlay, (px, py), (px + pw, py + ph), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0, canvas)
        cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (90, 90, 90), 1)
        cv2.rectangle(canvas, (px, py), (px + pw, py + 26), (30, 30, 100), -1)
        cv2.putText(canvas, "ADAS  |  Collision Warning", (px + 8, py + 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 255), 1, cv2.LINE_AA)

        rows = [py + 52, py + 78, py + 104, py + 130]
        fps_color = (50, 220, 50) if fps >= 15 else (0, 165, 255)
        cv2.putText(canvas, f"FPS      : {fps:.1f}", (px + 10, rows[0]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, fps_color, 2, cv2.LINE_AA)
        cv2.putText(canvas, f"Vehicles : {len(cars)}", (px + 10, rows[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (220, 220, 220), 2, cv2.LINE_AA)

        dist_txt = f"Closest  : {min_dist:.1f} m" if min_dist < 999.0 else "Closest  : --"
        cv2.putText(canvas, dist_txt, (px + 10, rows[2]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, _dist_color(min_dist) if min_dist < 999.0 else (180, 180, 180), 2, cv2.LINE_AA)

        if danger_flag:
            st, sc = "! DANGER - BRAKE !", (0, 0, 255)
        elif min_dist < 15.0:
            st, sc = "WARNING", (0, 165, 255)
        else:
            st, sc = "SAFE", (50, 220, 50)
        cv2.putText(canvas, f"Status   : {st}", (px + 10, rows[3]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, sc, 2, cv2.LINE_AA)

    # ------------------------------------------------------------------ #
    #  Main entry point                                                    #
    # ------------------------------------------------------------------ #
    def render_ui(self, frame: np.ndarray, seg_mask: np.ndarray,
                  cars: list, danger_flag: bool, min_dist: float, fps: float) -> np.ndarray:
        canvas = frame.copy()
        current_time_sec = self._frame_count / max(self._src_fps, 1.0)
        self._frame_count += 1

        self._draw_seg_overlay(canvas, seg_mask)
        self._draw_car_boxes(canvas, cars)

        if danger_flag:
            self._draw_danger_flash(canvas)
            self._draw_warning_banner(canvas)
            self._trigger_voice(current_time_sec)

        self._draw_dashboard(canvas, cars, danger_flag, min_dist, fps)
        self._record(canvas)
        return canvas
