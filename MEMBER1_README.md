# Member 1: Semantic Segmentation

Semantic segmentation module using SegFormer for video/camera stream processing.

## Overview

- **Model**: SegFormer (pretrained on Cityscapes)
- **Input**: Video file or webcam stream
- **Output**: Pixel-level segmentation maps (H × W)
- **Labels**: background, road, sky, vehicle, pedestrian

## Structure

```
tv1-segmentation/
├── run_segmentation.py          # Main entry point
├── requirements.txt             # Dependencies
└── src/member1_segmentation/
    ├── __init__.py
    ├── backends.py              # SegFormer implementation
```

## Installation

**Step 1: Navigate to TV1 folder**

```bash
cd tv1-segmentation
```

**Step 2: Create virtual environment (nếu chưa có)**

```bash
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

**Step 3: Install dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Step 4: Verify installation**

```bash
python -c "import torch, cv2, numpy, transformers; print('OK')"
```

**Dependencies:**

- torch, torchvision
- transformers
- opencv-python
- numpy, pillow

## Usage

**Quick test (webcam):**

```bash
python run_segmentation.py --source 0 --display
```

Press `q` to quit.

**Video file:**

```bash
python run_segmentation.py --source path/to/video.mp4 --device cuda
```

**With all options:**

```bash
python run_segmentation.py ^
  --source video.mp4 ^
  --device cuda ^
  --resize-width 512 ^
  --skip-frames 2 ^
  --max-frames 300 ^
  --output-dir outputs/member1_segmentation ^
  --display
```

**Arguments:**

- `--source`: Video file path or `0` for webcam (default: `0`)
- `--device`: `cuda` or `cpu` (auto-detect if not specified)
- `--resize-width`: Resize input to width (default: `512`)
- `--skip-frames`: Process every Nth frame (default: `0`)
- `--max-frames`: Stop after N frames (default: `0` = all)
- `--output-dir`: Output directory (default: `outputs/member1_segmentation`)
- `--display`: Show live preview window

## Output Format

**Files saved to `outputs/member1_segmentation/`:**

- `segmentation_overlay.mp4` - Demo video with color overlay
- `masks/seg_map_000000.png` - Raw segmentation masks (one per frame)
- `run_metadata.json` - Processing metadata and label statistics

**Using seg_map in Python code:**

**Note:** Import path depends on working directory:

- **From workspace root**: `from src.member1_segmentation.backends import create_segmenter`
- **From `tv1-segmentation/` folder**: `from member1_segmentation.backends import create_segmenter`

```python
# If running from tv1-segmentation/ folder:
from member1_segmentation.backends import create_segmenter
# Or from workspace root:
# from src.member1_segmentation.backends import create_segmenter

import cv2
import numpy as np

# Initialize segmenter
segmenter = create_segmenter(device="cuda")  # or "cpu"

# Read a frame
frame = cv2.imread("image.png")

# Get segmentation map
seg_map = segmenter(frame)

# Properties:
print(f"Shape: {seg_map.shape}")      # (H, W)
print(f"Dtype: {seg_map.dtype}")      # uint8
print(f"Unique labels: {np.unique(seg_map)}")  # [0, 1, 2, 3, 4]
```

**Label meanings:**

- `0`: background
- `1`: road
- `2`: sky
- `3`: vehicle (car, truck, bus, etc.)
- `4`: pedestrian (person, rider)

## Class Labels

| ID  | Label      | Color (BGR)    |
| --- | ---------- | -------------- |
| 0   | background | (0, 0, 0)      |
| 1   | road       | (60, 160, 60)  |
| 2   | sky        | (200, 120, 40) |
| 3   | vehicle    | (40, 40, 220)  |
| 4   | pedestrian | (220, 220, 40) |

## Features

✅ Output format: `seg_map = segmentation_model(frame)`  
✅ Size: Output matches input (H × W)  
✅ Noise filtering: Morphological operations (opening/closing)  
✅ GPU support: CUDA acceleration with fallback to CPU  
✅ Metadata export: Frame stats and processing info

## Integration

- **Member 4** (Depth Estimation): Reads `seg_map` for scene understanding
- **Member 6** (3D Fusion): Uses masks for coordinate projection
- **Metadata**: Saved in `run_metadata.json` for logging

## Troubleshooting

**Error: `ModuleNotFoundError: No module named 'torch'`**

- Check if virtual environment is activated
- Run: `pip install -r requirements.txt` again

**Error: `CUDA out of memory`**

- Use CPU instead: `--device cpu`
- Or reduce input size: `--resize-width 256`

**Slow processing (CPU mode)**

- Use GPU if available: `--device cuda`
- Reduce resolution: `--resize-width 256`
- Skip frames: `--skip-frames 2`

**No output video generated**

- Check `outputs/member1_segmentation/` folder exists
- Verify input video format (supports `.mp4`, `.avi`, `.mov`)
- Check console for error messages
