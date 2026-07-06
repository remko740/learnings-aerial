"""
Generate demo video: run detector on eval.mp4, save annotated output.

Draws per-detection:
  - Bounding box coloured by distance band (green=0-200m, cyan=200-400m)
  - Confidence score + distance estimate

Draws per-frame overlay:
  - Detection count
  - Model name
"""

import cv2
import numpy as np
from pathlib import Path
from math import tan, radians
from ultralytics import YOLO

ROOT      = Path(__file__).parent.parent
VIDEO_IN  = ROOT / "data" / "videos" / "eval.mp4"
VIDEO_OUT = ROOT / "data" / "demo_eval.mp4"
MODEL_PT  = ROOT / "runs" / "train_v3" / "weights" / "best.pt"  # round 3: best on eval

CONF      = 0.25
IMGSZ     = 640

CAR_LENGTH_M  = 4.5
VFOV_HALF_DEG = 21.0

# band colours in BGR
BAND_COLOUR = {
    "0-200m":   (0, 220, 80),    # green
    "200-400m": (220, 200, 0),   # cyan-yellow
    None:       (120, 120, 120), # grey — outside bands
}


def focal_px(img_h):
    return (img_h / 2) / tan(radians(VFOV_HALF_DEG))


def estimate_dist(bh_norm, img_h):
    px = bh_norm * img_h
    return (CAR_LENGTH_M * focal_px(img_h)) / px if px > 0 else float("inf")


def band_for(dist):
    if dist <= 200:
        return "0-200m"
    elif dist <= 400:
        return "200-400m"
    return None


def draw_box(img, x1, y1, x2, y2, conf, dist, colour):
    cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
    band = band_for(dist)
    label = f"{conf:.2f}  {dist:.0f}m" if dist < 500 else f"{conf:.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    lx, ly = x1, max(y1 - 4, th + 2)
    cv2.rectangle(img, (lx, ly - th - 2), (lx + tw + 2, ly + 2), colour, -1)
    cv2.putText(img, label, (lx + 1, ly),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)


def draw_overlay(img, n_det, frame_idx, fps):
    h, w = img.shape[:2]
    ts = frame_idx / fps
    cv2.rectangle(img, (0, 0), (260, 62), (0, 0, 0), -1)
    cv2.rectangle(img, (0, 0), (260, 62), (60, 60, 60), 1)
    cv2.putText(img, f"YOLOv8s  aerial-detect", (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(img, f"Detections: {n_det}", (8, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(img, f"t = {ts:.1f}s", (8, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    # legend
    cv2.rectangle(img, (w - 170, 0), (w, 50), (0, 0, 0), -1)
    cv2.rectangle(img, (w - 170, 0), (w, 50), (60, 60, 60), 1)
    cv2.rectangle(img, (w - 162, 8), (w - 148, 20), BAND_COLOUR["0-200m"], -1)
    cv2.putText(img, "0-200 m", (w - 143, 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
    cv2.rectangle(img, (w - 162, 28), (w - 148, 40), BAND_COLOUR["200-400m"], -1)
    cv2.putText(img, "200-400 m", (w - 143, 39),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)


def main():
    print(f"Model:  {MODEL_PT}")
    print(f"Input:  {VIDEO_IN}")
    print(f"Output: {VIDEO_OUT}")

    model = YOLO(str(MODEL_PT))

    cap = cv2.VideoCapture(str(VIDEO_IN))
    fps     = cap.get(cv2.CAP_PROP_FPS)
    width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(VIDEO_OUT), fourcc, fps, (width, height))

    print(f"Processing {n_total} frames at {fps:.0f} fps ({width}×{height})...")
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=CONF, imgsz=IMGSZ, verbose=False)[0]
        boxes = results.boxes.xyxy.cpu().numpy()  if len(results.boxes) else []
        confs = results.boxes.conf.cpu().numpy()  if len(results.boxes) else []

        for (x1, y1, x2, y2), c in zip(boxes, confs):
            bh_norm = (y2 - y1) / height
            dist    = estimate_dist(bh_norm, height)
            colour  = BAND_COLOUR[band_for(dist)]
            draw_box(frame, int(x1), int(y1), int(x2), int(y2), c, dist, colour)

        draw_overlay(frame, len(boxes), frame_idx, fps)
        out.write(frame)
        frame_idx += 1

        if frame_idx % 50 == 0:
            print(f"  {frame_idx}/{n_total} frames  ({frame_idx/n_total*100:.0f}%)")

    cap.release()
    out.release()
    print(f"\nDone → {VIDEO_OUT}")
    print(f"Play:   open {VIDEO_OUT}")


if __name__ == "__main__":
    main()
