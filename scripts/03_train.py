"""
Fine-tune YOLOv8s on pseudo-labeled aerial frames.

What fine-tuning means:
  We start from yolov8s.pt — a model pretrained on COCO (80 classes, street-level).
  It already recognises cars, trucks, and buses. We continue training it on our
  aerial-perspective frames so it learns:
    - top-down and oblique viewpoints
    - small vehicle sizes (10–40 px)
    - our single unified class "vehicle"

  The alternative — training from scratch — would need ~10× more data and time.
  Fine-tuning converges in 50–100 epochs because the backbone already knows edges,
  wheels, and car shapes; we only need to adapt the head to our viewpoint.

Dataset split strategy:
  Frames in each video are ordered chronologically. We take the LAST 20% of each
  video as validation. This is a temporal split: the model trains on earlier frames
  and validates on later frames, which avoids the subtle leakage that random splits
  cause (adjacent frames look nearly identical — a random val frame would be "seen"
  by its neighbors in the train set).

  train_a: 83 frames → train: 0–66, val: 67–82
  train_b: 125 frames → train: 0–99, val: 100–124
  train_c: 68 frames → train: 0–54, val: 55–67
  train_d: 105 frames → train: 0–83, val: 84–104

Training config (M1 Pro):
  model:   yolov8s.pt  (small, fast, good starting point for fine-tuning)
  epochs:  100         (enough epochs; early stopping will halt if val plateaus)
  imgsz:   640         (standard YOLO resolution; fast on MPS)
  batch:   16          (fits M1 Pro 16 GB unified memory comfortably)
  device:  mps         (Apple Metal Performance Shaders — GPU on M1)
  patience: 20         (stop early if val metric doesn't improve for 20 epochs)
"""

import shutil
import yaml
from pathlib import Path
from ultralytics import YOLO

# ── paths ─────────────────────────────────────────────────────────────────────

ROOT       = Path(__file__).parent.parent
FRAMES_DIR = ROOT / "data" / "frames"
LABELS_DIR = ROOT / "data" / "labels_mixed"   # final pseudo-labels
DATASET    = ROOT / "data" / "dataset"         # assembled train/val structure
RUNS_DIR   = ROOT / "runs"

# ── config ────────────────────────────────────────────────────────────────────

VAL_FRACTION = 0.2   # last 20% of each video → validation
EPOCHS       = 100
IMGSZ        = 640
BATCH        = 16
DEVICE       = "mps"
MODEL        = "yolov8s.pt"


# ── step 1: build dataset directory ───────────────────────────────────────────

def build_dataset() -> None:
    """
    Assemble the dataset/ directory that YOLOv8 expects:

      dataset/
        images/train/   ← .jpg frames
        images/val/
        labels/train/   ← matching .txt label files
        labels/val/
        dataset.yaml    ← tells YOLO where to look and what classes we have

    Files are COPIED (not symlinked) so the dataset is self-contained.
    Filenames are prefixed with the video id to avoid collisions:
      train_a/0001.jpg → dataset/images/train/train_a_0001.jpg
    """
    # Clean and recreate
    if DATASET.exists():
        shutil.rmtree(DATASET)

    for split in ("train", "val"):
        (DATASET / "images" / split).mkdir(parents=True)
        (DATASET / "labels" / split).mkdir(parents=True)

    video_ids = sorted(d.name for d in FRAMES_DIR.iterdir() if d.is_dir())
    counts = {"train": 0, "val": 0}

    for vid in video_ids:
        frames = sorted((FRAMES_DIR / vid).glob("*.jpg"))
        labels = sorted((LABELS_DIR / vid).glob("*.txt"))

        # Temporal split: last VAL_FRACTION frames → val
        n_val   = max(1, int(len(frames) * VAL_FRACTION))
        n_train = len(frames) - n_val

        splits = ["train"] * n_train + ["val"] * n_val

        label_map = {lf.stem: lf for lf in labels}

        for frame, split in zip(frames, splits):
            new_stem = f"{vid}_{frame.stem}"

            # Copy image
            shutil.copy(frame, DATASET / "images" / split / f"{new_stem}.jpg")

            # Copy label (may not exist for empty frames → write empty .txt)
            lbl = label_map.get(frame.stem)
            dst_lbl = DATASET / "labels" / split / f"{new_stem}.txt"
            if lbl and lbl.stat().st_size > 0:
                shutil.copy(lbl, dst_lbl)
            else:
                dst_lbl.write_text("")   # empty label → no objects in this frame

            counts[split] += 1

    # Write dataset.yaml
    cfg = {
        "path": str(DATASET),
        "train": "images/train",
        "val":   "images/val",
        "nc":    1,
        "names": ["vehicle"],
    }
    with open(DATASET / "dataset.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    print(f"Dataset ready: {counts['train']} train / {counts['val']} val frames")
    print(f"  → {DATASET / 'dataset.yaml'}")


# ── step 2: train ─────────────────────────────────────────────────────────────

def train() -> None:
    """
    Fine-tune YOLOv8s on the assembled dataset.

    Key hyperparameters explained:
      epochs    — full passes over the training data; more = better up to a point
      imgsz     — resize all images to this before feeding to the model;
                  640 is the YOLO standard (same as our hires detection at inference)
      batch     — images per gradient update; larger = more stable gradients,
                  more memory; 16 fits M1 Pro comfortably
      device    — 'mps' uses Apple Metal GPU; falls back to 'cpu' if unavailable
      patience  — early stopping: halt if val/mAP50 doesn't improve for N epochs;
                  prevents wasting time after the model has converged
      lr0       — initial learning rate; 0.001 is lower than default (0.01) which
                  is appropriate for fine-tuning — we nudge existing weights, not
                  reset them
      augment   — YOLOv8 applies mosaic, flip, HSV jitter, scale, translate by
                  default; this is crucial because our dataset is small (~300 train
                  frames) and augmentation artificially expands variety
    """
    model = YOLO(MODEL)

    model.train(
        data    = str(DATASET / "dataset.yaml"),
        epochs  = EPOCHS,
        imgsz   = IMGSZ,
        batch   = BATCH,
        device  = DEVICE,
        project = str(RUNS_DIR),
        name    = "train",
        patience= 20,
        lr0     = 0.001,
        exist_ok= True,
        verbose = True,
    )

    best = RUNS_DIR / "train" / "weights" / "best.pt"
    print(f"\nTraining complete.")
    print(f"Best weights → {best}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=== Step 1: build dataset ===")
    build_dataset()

    print("\n=== Step 2: fine-tune YOLOv8s ===")
    print(f"Model:   {MODEL}")
    print(f"Epochs:  {EPOCHS}  (early stopping patience={20})")
    print(f"Imgsz:   {IMGSZ}")
    print(f"Batch:   {BATCH}")
    print(f"Device:  {DEVICE}")
    print()
    train()


if __name__ == "__main__":
    main()
