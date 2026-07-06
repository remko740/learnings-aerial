"""
Quick experiment: compare yolov8s-worldv2 vs yolov8l-worldv2 on train_a.
Only processes train_a (83 frames) to keep runtime short (~3 min total).
Saves side-by-side previews so differences are easy to spot visually.
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from torchvision.ops import nms
from ultralytics import YOLOWorld

FRAMES_DIR  = Path(__file__).parent.parent / "data" / "frames" / "train_a"
OUT_DIR     = Path(__file__).parent.parent / "data" / "experiment_model_size"
CONF        = 0.2
NMS_IOU     = 0.4
IMGSZ       = 1280
PREVIEW_N   = 5
MODELS      = ["yolov8s-worldv2.pt", "yolov8l-worldv2.pt"]
CLASSES     = ["car", "truck", "bus", "van", "vehicle"]


def apply_nms(boxes, confs):
    if len(boxes) == 0:
        return np.array([], dtype=int)
    keep = nms(
        torch.tensor(boxes, dtype=torch.float32),
        torch.tensor(confs, dtype=torch.float32),
        iou_threshold=NMS_IOU,
    )
    return keep.numpy()


def run_model(model_name: str, frame_paths: list) -> dict:
    print(f"\nLoading {model_name} ...")
    model = YOLOWorld(model_name)
    model.set_classes(CLASSES)

    out_dir = OUT_DIR / model_name.replace(".pt", "")
    out_dir.mkdir(parents=True, exist_ok=True)

    preview_idx = set(np.linspace(0, len(frame_paths) - 1, PREVIEW_N, dtype=int))
    total, empty = 0, 0

    for i, fp in enumerate(frame_paths):
        img = cv2.imread(str(fp))
        h, w = img.shape[:2]

        results = model.predict(img, conf=CONF, imgsz=IMGSZ, verbose=False)[0]
        boxes = results.boxes.xyxy.cpu().numpy() if len(results.boxes) else np.empty((0, 4))
        confs = results.boxes.conf.cpu().numpy() if len(results.boxes) else np.array([])

        keep = apply_nms(boxes, confs)
        boxes, confs = boxes[keep], confs[keep]
        total += len(boxes)
        if len(boxes) == 0:
            empty += 1

        if i in preview_idx:
            preview = img.copy()
            for (x1, y1, x2, y2), c in zip(boxes, confs):
                cv2.rectangle(preview, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(preview, f"{c:.2f}", (int(x1), int(y1) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            cv2.imwrite(str(out_dir / fp.name), preview)

    return {"boxes": total, "empty": empty, "avg": total / len(frame_paths)}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame_paths = sorted(FRAMES_DIR.glob("*.jpg"))
    print(f"Frames: {len(frame_paths)}  |  imgsz={IMGSZ}  |  conf={CONF}")

    results = {}
    for model_name in MODELS:
        results[model_name] = run_model(model_name, frame_paths)

    print("\n── Results on train_a ──────────────────────────")
    print(f"{'Model':<25} {'Boxes':>6} {'Avg/frame':>10} {'Empty frames':>13}")
    print("─" * 58)
    for name, r in results.items():
        print(f"{name:<25} {r['boxes']:>6} {r['avg']:>10.1f} {r['empty']:>13}")

    print(f"\nPreviews saved to: {OUT_DIR}")
    print("Open both subfolders and compare the same frame side by side.")


if __name__ == "__main__":
    main()
