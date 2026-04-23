# ComputerVision
Dự án học phần Xử lý ảnh và thị giác máy tính

TOPIC 2: Segmentation + Depth => Scene understanding
Mục tiêu: ứng dụng sử dụng camera hành trình để phát hiện phương tiện phía trước, ước lượng khoảng cách và đưa ra cảnh báo nguy hiểm kịp thời.

![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=flat&logo=opencv&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-success)

Dự án Đồ án môn học Computer Vision: Xây dựng hệ thống Hỗ trợ Lái xe Tiên tiến (ADAS) sử dụng Video từ Camera hành trình (Dashcam). Hệ thống kết hợp mạng học sâu **U-Net** (Semantic Segmentation) và **MiDaS** (Monocular Depth Estimation) để phát hiện phương tiện, ước lượng khoảng cách và đưa ra cảnh báo nguy hiểm bằng giọng nói.

---

**Tính năng nổi bật**

- Phân vùng Ngữ nghĩa (Semantic Segmentation)
- Ước lượng Độ sâu (Depth Estimation)
- Trích xuất Đối tượng & Khoảng cách
- Cảnh báo Giọng nói (Voice Warning)
- Xử lý Thời gian thực (Real-time Pipeline)


Dự án được áp dụng mô hình thiết kế hướng đối tượng (OOP) và chia thành 6 module độc lập giúp tối ưu việc quản lý source code và làm việc nhóm:


`segmentation.py` - Khởi tạo và chạy nội suy mạng U-Net. Cung cấp Tensor thô chứa Mask của đường/xe/trời/người.
`depth.py` - Tích hợp MiDaS, tạo Depth Map và cung cấp hàm toán học ước lượng khoảng cách tương đối.
`performance.py` - Theo dõi và đo lường FPS, tối ưu hóa tài nguyên tính toán (Inference time).
`core_logic.py` - Xử lý Upscale, tìm viền (Contours) đối tượng, Mapping ma trận và quyết định trạng thái (SAFE/DANGER).
`ui_ux.py` - Vẽ Bounding Box, tô màu Segmentation, render Dashboard và kích hoạt luồng Âm thanh cảnh báo (pyttsx3).
`main.py` - khởi tạo luồng đọc Video, gọi các module theo đúng Workflow và xuất file kết quả.

*(Ngoài ra repo còn đính kèm file `unet_road_car_sky(class4).pth và unet_road_car_sky(class4).pth` là các file model đã được huấn luyện (trained model)).*

---
## Cài đặt các thư viện yêu cầu
pip install torch torchvision
pip install opencv-python numpy
pip install segmentation-models-pytorch
pip install pyttsx3 tqdm


