# ComputerVision
Dự án học phần Xử lý ảnh và thị giác máy tính

## Cai dat moi truong tu requirements.txt

### 1) Tao va kich hoat virtual environment (khuyen nghi)

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Cai thu vien tu requirements.txt

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3) Kiem tra nhanh

```bash
python -c "import torch, cv2, numpy; print('OK')"
```

Neu in ra `OK` la cai dat thanh cong.
