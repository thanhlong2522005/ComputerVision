# tv1-segmentation

Semantic segmentation for Member 1.

## What this repo contains

- `run_segmentation.py`: main script to run inference on video or camera
- `src/member1_segmentation/backends.py`: SegFormer Cityscapes backend

## Model

This repo uses:

- `nvidia/segformer-b0-finetuned-cityscapes-1024-1024`

Project labels:

- `0`: background
- `1`: road
- `2`: sky
- `3`: vehicle
- `4`: person

## Install

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

## Run

CPU:

```bash
python run_segmentation.py --source path/to/video.mp4 --resize-width 512 --skip-frames 2 --display
```

GPU:

```bash
python run_segmentation.py --source path/to/video.mp4 --device cuda --resize-width 512 --skip-frames 2 --display
```
