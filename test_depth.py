import cv2
import time
from depth_estimation import MidasDepthEstimator

# --- Hàm bắt sự kiện click chuột ---
def get_depth_value(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Lấy giá trị tại điểm click (y là hàng, x là cột)
        # Giá trị trả về từ 0 (xa nhất) đến 255 (gần nhất)
        depth_val = depth_map[y, x]
        print(f"👉 Điểm vừa click (x={x}, y={y}) có mức độ nguy hiểm (Depth) là: {depth_val}")

# Khởi tạo model
depth_model = MidasDepthEstimator(model_type="MiDaS_small")
cap = cv2.VideoCapture("test_video1.mp4")

# Mở cửa sổ trước để gắn tính năng click chuột
cv2.namedWindow('Depth Map')
cv2.setMouseCallback('Depth Map', get_depth_value)

frame_count = 0
prev_time = time.time()

# Biến global để hàm click chuột đọc được dữ liệu
global depth_map 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    frame_count += 1
    if frame_count % 3 != 0: # Skip frame để video chạy mượt hơn
        continue
        
    frame = cv2.resize(frame, (480, 360))

    # Xử lý mô hình
    depth_map = depth_model.process_frame(frame)
    distance_map = depth_model.classify_distance(depth_map, danger_ratio=0.85, safe_ratio=0.60)
    
    # Áp màu hiển thị
    depth_colormap = cv2.applyColorMap(depth_map, cv2.COLORMAP_INFERNO)

    # Hiển thị FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Original Frame', frame)
    cv2.imshow('Depth Map', depth_colormap)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()