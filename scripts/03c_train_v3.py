"""
Round 3 fine-tune: start from Round 1 best.pt, train on manually-cleaned labels_mixed.

Changes vs Round 1:
  - Base model: runs/train/weights/best.pt  (already aerial-adapted)
  - Labels:     labels_mixed after manual cleanup (5191 boxes, fewer FP)
  - lr0:        0.0005  (lower LR — we nudge a fine-tuned model, not COCO weights)
  - epochs:     80 with patience=20
  - Output:     runs/train_v3/
"""

import shutil
import yaml
from pathlib import Path
from ultralytics import YOLO

ROOT       = Path(__file__).parent.parent
FRAMES_DIR = ROOT / "data" / "frames"
LABELS_DIR = ROOT / "data" / "labels_mixed"
DATASET    = ROOT / "data" / "dataset_v3"
RUNS_DIR   = ROOT / "runs"

VAL_FRACTION = 0.2
EPOCHS       = 80
IMGSZ        = 640
BATCH        = 16
DEVICE       = "mps"
MODEL        = str(ROOT / "runs" / "train" / "weights" / "best.pt")


def build_dataset():
    if DATASET.exists():
        shutil.rmtree(DATASET)

    for split in ("train", "val"):
        (DATASET / "images" / split).mkdir(parents=True)
        (DATASET / "labels" / split).mkdir(parents=True)

    # exclude eval frames from training
    video_ids = sorted(d.name for d in FRAMES_DIR.iterdir()
                       if d.is_dir() and d.name != "eval")
    counts = {"train": 0, "val": 0}

    for vid in video_ids:
        frames = sorted((FRAMES_DIR / vid).glob("*.jpg"))
        labels = sorted((LABELS_DIR / vid).glob("*.txt"))

        n_val   = max(1, int(len(frames) * VAL_FRACTION))
        n_train = len(frames) - n_val
        splits  = ["train"] * n_train + ["val"] * n_val
        label_map = {lf.stem: lf for lf in labels}

        for frame, split in zip(frames, splits):
            new_stem = f"{vid}_{frame.stem}"
            shutil.copy(frame, DATASET / "images" / split / f"{new_stem}.jpg")
            lbl = label_map.get(frame.stem)
            dst = DATASET / "labels" / split / f"{new_stem}.txt"
            if lbl and lbl.stat().st_size > 0:
                shutil.copy(lbl, dst)
            else:
                dst.write_text("")
            counts[split] += 1

    cfg = {
        "path":  str(DATASET),
        "train": "images/train",
        "val":   "images/val",
        "nc":    1,
        "names": ["vehicle"],
    }
    with open(DATASET / "dataset.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    print(f"Dataset ready: {counts['train']} train / {counts['val']} val frames")


def train():
    model = YOLO(MODEL)
    model.train(
        data     = str(DATASET / "dataset.yaml"),
        epochs   = EPOCHS,
        imgsz    = IMGSZ,
        batch    = BATCH,
        device   = DEVICE,
        project  = str(RUNS_DIR),
        name     = "train_v3",
        patience = 20,
        lr0      = 0.0005,
        exist_ok = True,
        verbose  = True,
    )
    best = RUNS_DIR / "train_v3" / "weights" / "best.pt"
    print(f"\nTraining complete. Best weights → {best}")


if __name__ == "__main__":
    print("=== Round 3: fine-tune on manually-cleaned labels ===")
    print(f"Base model: {MODEL}")
    print(f"Labels:     {LABELS_DIR}  (manually cleaned)")
    build_dataset()
    train()
