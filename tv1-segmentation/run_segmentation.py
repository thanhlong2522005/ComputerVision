from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterator, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from member1_segmentation.backends import LABELS, PALETTE, create_segmenter


WINDOW_NAME = "member1_segmentation"
OUTPUT_VIDEO = "segmentation_overlay.mp4"
OUTPUT_MASK_DIR = "masks"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TV1 semantic segmentation for video or camera.")
    parser.add_argument("--source", default="0", help="Video path or camera index.")
    parser.add_argument("--backend", choices=("segformer", "classical"), default="segformer", help="Segmentation backend: 'segformer' uses SegFormer, 'classical' uses classical CV methods.")
    parser.add_argument("--device", default=None, help="Device: cpu or cuda.")
    parser.add_argument("--model-name", default=None, help="Optional SegFormer model id.")
    parser.add_argument("--output-dir", default="outputs/member1_segmentation", help="Output directory.")
    parser.add_argument("--resize-width", type=int, default=512, help="Resize width before inference.")
    parser.add_argument("--skip-frames", type=int, default=0, help="Skip N frames between processed frames.")
    parser.add_argument("--max-frames", type=int, default=0, help="Maximum number of frames to process.")
    parser.add_argument("--display", action="store_true", help="Show live preview.")
    return parser


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def source_to_capture_value(source: str):
    return int(source) if source.isdigit() else source


def resize_frame(frame: np.ndarray, resize_width: int) -> np.ndarray:
    if resize_width <= 0 or frame.shape[1] == resize_width:
        return frame
    scale = resize_width / float(frame.shape[1])
    new_height = int(frame.shape[0] * scale)
    return cv2.resize(frame, (resize_width, new_height), interpolation=cv2.INTER_LINEAR)


def frame_iterator(cap: cv2.VideoCapture, skip_frames: int) -> Iterator[Tuple[int, np.ndarray]]:
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if skip_frames and frame_idx % (skip_frames + 1) != 0:
            frame_idx += 1
            continue
        yield frame_idx, frame
        frame_idx += 1


def refine_seg_map(seg_map: np.ndarray) -> np.ndarray:
    refined = seg_map.copy()
    kernel_open = np.ones((3, 3), np.uint8)
    kernel_close = np.ones((5, 5), np.uint8)

    for label_id in (1, 2, 3, 4):
        mask = (refined == label_id).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        refined[mask > 0] = label_id

    height = refined.shape[0]
    refined[height // 2 :, :][refined[height // 2 :, :] == 2] = 0
    return refined


def overlay_mask(frame: np.ndarray, seg_map: np.ndarray) -> np.ndarray:
    color_mask = np.zeros_like(frame)
    for label_id, color in PALETTE.items():
        color_mask[seg_map == label_id] = color
    return cv2.addWeighted(frame, 0.55, color_mask, 0.45, 0.0)


def draw_legend(frame: np.ndarray) -> np.ndarray:
    x, y = 16, 18
    for label_id, label_name in LABELS.items():
        cv2.rectangle(frame, (x, y), (x + 18, y + 18), PALETTE[label_id], -1)
        cv2.putText(frame, label_name, (x + 28, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        y += 24
    return frame


def draw_footer(frame: np.ndarray, frame_idx: int, fps: float, backend: str = "segformer-cityscapes") -> np.ndarray:
    cv2.putText(
        frame,
        f"backend={backend} frame={frame_idx} fps={fps:.2f}",
        (16, frame.shape[0] - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


def save_metadata(output_dir: Path, source: str, processed_frames: int, source_size: Tuple[int, int], backend: str = "segformer") -> None:
    metadata = {
        "source": source,
        "backend": backend,
        "labels": LABELS,
        "processed_frames": processed_frames,
        "source_resolution": list(source_size),
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = Path(args.output_dir)
    masks_dir = output_dir / OUTPUT_MASK_DIR
    ensure_dir(output_dir)
    ensure_dir(masks_dir)

    if args.backend == "classical":
        raise NotImplementedError("Classical backend not yet implemented. Use 'segformer' instead.")
    
    segmenter = create_segmenter(device=args.device, model_name=args.model_name)
    capture = cv2.VideoCapture(source_to_capture_value(args.source))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open source: {args.source}")

    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    source_fps = source_fps if source_fps and source_fps > 1 else 20.0

    if args.display:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    writer = None
    started_at = time.perf_counter()
    processed = 0

    try:
        for frame_idx, frame in frame_iterator(capture, args.skip_frames):
            if args.max_frames and processed >= args.max_frames:
                break

            frame = resize_frame(frame, args.resize_width)
            if writer is None:
                writer = cv2.VideoWriter(
                    str(output_dir / OUTPUT_VIDEO),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    source_fps,
                    (frame.shape[1], frame.shape[0]),
                )

            seg_map = segmenter(frame)
            seg_map = refine_seg_map(seg_map)
            overlay = draw_footer(draw_legend(overlay_mask(frame, seg_map)), frame_idx, processed / max(time.perf_counter() - started_at, 1e-6), backend="segformer-cityscapes")

            cv2.imwrite(str(masks_dir / f"seg_map_{frame_idx:06d}.png"), seg_map)
            writer.write(overlay)

            if args.display:
                cv2.resizeWindow(WINDOW_NAME, frame.shape[1], frame.shape[0])
                cv2.imshow(WINDOW_NAME, overlay)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            processed += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    save_metadata(output_dir, args.source, processed, (source_height, source_width), backend=args.backend)
    print(f"Processed {processed} frame(s). Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
