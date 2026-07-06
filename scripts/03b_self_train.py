"""
Self-training: re-label training frames with our fine-tuned model, then retrain.

Why this helps:
  Round 1 labels came from YOLO-World — a general zero-shot model that was never
  trained on aerial footage. It missed vehicles in unusual viewpoints (moving
  toward/away from camera, very small objects at high altitude).

  Our fine-tuned model has seen hundreds of aerial frames during training and
  has adapted to this domain. It will find some of the vehicles YOLO-World missed.
  Training on these improved labels = round 2 model with higher recall.

Risk:
  If our model has false positives (detects road markings as vehicles), those
  will be reinforced in round 2. We use a higher confidence threshold (0.30 vs 0.20)
  to keep precision high. After relabeling, we compare box counts per video to
  catch obvious regressions before starting the second training run.

Pipeline:
  1. Run best.pt on all 381 train frames → new pseudo-labels (labels_self_trained)
  2. Print comparison: labels_mixed vs labels_self_trained per video
  3. Assemble new dataset/ directory from the new labels
  4. Fine-tune from best.pt for 60 more epochs on improved labels
     Produces: runs/train_v2/weights/best.pt
"""

import shutil
import yaml
import cv2
import numpy as np
import torch
from pathlib import Path
from torchvision.ops import nms
from ultralytics import YOLO

# ── paths ─────────────────────────────────────────────────────────────────────

ROOT          = Path(__file__).parent.parent
FRAMES_DIR    = ROOT / "data" / "frames"
ORIG_LABELS   = ROOT / "data" / "labels_mixed"
NEW_LABELS    = ROOT / "data" / "labels_self_trained"
DATASET_V2    = ROOT / "data" / "dataset_v2"
MODEL_PT      = ROOT / "runs" / "train" / "weights" / "best.pt"
RUNS_DIR      = ROOT / "runs"

# ── config ────────────────────────────────────────────────────────────────────

CONF_THRESHOLD = 0.30   # higher than original 0.20 — our model can be noisier than YOLO-World
NMS_IOU        = 0.30
IMGSZ          = 1280   # hires inference for better small object recall
MAX_BOX_AREA   = 0.03   # max normalised w×h; road marking outlines are ~0.13, real vehicles ≤0.007
VAL_FRACTION   = 0.20

# round 2 training — shorter since we start from best.pt, not COCO
EPOCHS_V2      = 60
BATCH          = 16
DEVICE         = "mps"
PREVIEW_N      = 5


# ── step 1: re-label with best.pt ─────────────────────────────────────────────

def apply_nms(boxes, confs):
    if len(boxes) == 0:
        return np.array([], dtype=int)
    keep = nms(
        torch.tensor(boxes, dtype=torch.float32),
        torch.tensor(confs, dtype=torch.float32),
        iou_threshold=NMS_IOU,
    )
    return keep.numpy()


def remove_contained_boxes(boxes, confs, containment_thr=0.70):
    """
    Remove boxes that are mostly contained within a larger box.
    Fixes the truck cab+trailer problem: cab box is ~70-90% inside
    the full-truck box, but standard NMS IoU (~0.30) won't merge them.
    containment = intersection / area_of_smaller_box
    """
    if len(boxes) < 2:
        return np.arange(len(boxes))
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    suppress = np.zeros(len(boxes), dtype=bool)
    for i in range(len(boxes)):
        for j in range(len(boxes)):
            if i == j or suppress[j]:
                continue
            ix1 = max(boxes[i, 0], boxes[j, 0])
            iy1 = max(boxes[i, 1], boxes[j, 1])
            ix2 = min(boxes[i, 2], boxes[j, 2])
            iy2 = min(boxes[i, 3], boxes[j, 3])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            # if box i is mostly inside box j → suppress i (keep the larger j)
            if areas[i] > 0 and inter / areas[i] >= containment_thr and areas[i] < areas[j]:
                suppress[i] = True
                break
    return np.where(~suppress)[0]


def relabel(video_id: str, model) -> dict:
    src_dir = FRAMES_DIR / video_id
    dst_dir = NEW_LABELS / video_id
    dst_dir.mkdir(parents=True, exist_ok=True)

    prev_dir = ROOT / "data" / "preview_self_trained" / video_id
    prev_dir.mkdir(parents=True, exist_ok=True)

    frame_paths  = sorted(src_dir.glob("*.jpg"))
    preview_idx  = set(np.linspace(0, len(frame_paths) - 1, PREVIEW_N, dtype=int))
    total, empty = 0, 0

    for i, fp in enumerate(frame_paths):
        img  = cv2.imread(str(fp))
        h, w = img.shape[:2]

        results = model.predict(img, conf=CONF_THRESHOLD, imgsz=IMGSZ, verbose=False)[0]
        boxes   = results.boxes.xyxy.cpu().numpy() if len(results.boxes) else np.empty((0, 4))
        confs_  = results.boxes.conf.cpu().numpy() if len(results.boxes) else np.array([])

        keep  = apply_nms(boxes, confs_)
        boxes = boxes[keep]
        confs_ = confs_[keep]
        keep2 = remove_contained_boxes(boxes, confs_)
        boxes = boxes[keep2]
        confs_ = confs_[keep2]

        # drop oversized boxes (road markings, building outlines)
        bw_norm = (boxes[:, 2] - boxes[:, 0]) / w
        bh_norm = (boxes[:, 3] - boxes[:, 1]) / h
        size_mask = (bw_norm * bh_norm) <= MAX_BOX_AREA
        boxes = boxes[size_mask]

        lines = []
        for x1, y1, x2, y2 in boxes:
            xc = ((x1 + x2) / 2) / w
            yc = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        (dst_dir / fp.name.replace(".jpg", ".txt")).write_text("\n".join(lines))
        total += len(lines)
        if len(lines) == 0:
            empty += 1

        if i in preview_idx:
            preview = img.copy()
            for x1, y1, x2, y2 in boxes:
                cv2.rectangle(preview, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 255), 2)
            cv2.imwrite(str(prev_dir / fp.name), preview)

    return {"boxes": total, "empty": empty, "avg": total / max(len(frame_paths), 1)}


def count_labels(labels_dir: Path) -> dict:
    result = {}
    for vid_dir in sorted(labels_dir.iterdir()):
        if not vid_dir.is_dir():
            continue
        boxes = sum(
            len([l for l in f.read_text().splitlines() if l.strip()])
            for f in vid_dir.glob("*.txt")
        )
        result[vid_dir.name] = boxes
    return result


# ── step 2: compare old vs new labels ─────────────────────────────────────────

def compare_labels() -> None:
    old = count_labels(ORIG_LABELS)
    new = count_labels(NEW_LABELS)

    print(f"\n{'Video':<10} {'labels_mixed':>14} {'self_trained':>14} {'Δ':>8}")
    print("─" * 50)
    total_old, total_new = 0, 0
    for vid in sorted(old):
        o, n = old.get(vid, 0), new.get(vid, 0)
        total_old += o
        total_new += n
        marker = "↑" if n > o else ("↓" if n < o else "=")
        print(f"  {vid:<10} {o:>14} {n:>14} {n-o:>+7}  {marker}")
    print("─" * 50)
    print(f"  {'TOTAL':<10} {total_old:>14} {total_new:>14} {total_new-total_old:>+7}")
    print()

    if total_new < total_old * 0.8:
        print("WARNING: new labels have >20% fewer boxes — check previews before training.")
    elif total_new > total_old * 1.5:
        print("WARNING: new labels have >50% more boxes — model may have high FP rate.")
    else:
        print("Label counts look reasonable. Review previews, then proceed to training.")


# ── step 3: build dataset v2 ──────────────────────────────────────────────────

def build_dataset_v2() -> None:
    if DATASET_V2.exists():
        shutil.rmtree(DATASET_V2)

    for split in ("train", "val"):
        (DATASET_V2 / "images" / split).mkdir(parents=True)
        (DATASET_V2 / "labels" / split).mkdir(parents=True)

    video_ids = sorted(d.name for d in FRAMES_DIR.iterdir()
                       if d.is_dir() and d.name != "eval")
    counts = {"train": 0, "val": 0}

    for vid in video_ids:
        frames    = sorted((FRAMES_DIR / vid).glob("*.jpg"))
        labels    = sorted((NEW_LABELS / vid).glob("*.txt"))
        n_val     = max(1, int(len(frames) * VAL_FRACTION))
        splits    = ["train"] * (len(frames) - n_val) + ["val"] * n_val
        label_map = {lf.stem: lf for lf in labels}

        for frame, split in zip(frames, splits):
            new_stem = f"{vid}_{frame.stem}"
            shutil.copy(frame, DATASET_V2 / "images" / split / f"{new_stem}.jpg")
            lbl     = label_map.get(frame.stem)
            dst_lbl = DATASET_V2 / "labels" / split / f"{new_stem}.txt"
            if lbl and lbl.stat().st_size > 0:
                shutil.copy(lbl, dst_lbl)
            else:
                dst_lbl.write_text("")
            counts[split] += 1

    cfg = {"path": str(DATASET_V2), "train": "images/train",
           "val": "images/val", "nc": 1, "names": ["vehicle"]}
    with open(DATASET_V2 / "dataset.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    print(f"Dataset v2 ready: {counts['train']} train / {counts['val']} val frames")


# ── step 4: retrain from best.pt ──────────────────────────────────────────────

def retrain() -> None:
    """
    Fine-tune from best.pt (not COCO weights).
    Starting from an already-adapted model means we need fewer epochs.
    lr0 is lower (0.0005) to avoid undoing what round 1 learned.
    """
    model = YOLO(str(MODEL_PT))

    model.train(
        data    = str(DATASET_V2 / "dataset.yaml"),
        epochs  = EPOCHS_V2,
        imgsz   = 640,
        batch   = BATCH,
        device  = DEVICE,
        project = str(RUNS_DIR),
        name    = "train_v2",
        patience= 15,
        lr0     = 0.0005,   # lower LR — nudging an already-good model
        exist_ok= True,
        verbose = True,
    )

    best = RUNS_DIR / "train_v2" / "weights" / "best.pt"
    print(f"\nRound 2 training complete.")
    print(f"Best weights → {best}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    if not MODEL_PT.exists():
        raise FileNotFoundError(f"best.pt not found: {MODEL_PT}\nRun 03_train.py first.")

    print("=== Step 1: re-label training frames with best.pt ===")
    print(f"Model:  {MODEL_PT}")
    print(f"Conf:   {CONF_THRESHOLD}  NMS: {NMS_IOU}  imgsz: {IMGSZ}\n")

    model    = YOLO(str(MODEL_PT))
    video_ids = sorted(d.name for d in FRAMES_DIR.iterdir()
                       if d.is_dir() and d.name != "eval")

    for vid in video_ids:
        print(f"  {vid} ...", end=" ", flush=True)
        r = relabel(vid, model)
        print(f"{r['boxes']} boxes  ({r['avg']:.1f}/frame,  {r['empty']} empty)")

    print("\n=== Step 2: compare labels_mixed vs labels_self_trained ===")
    compare_labels()

    print("Review previews at: data/preview_self_trained/")

    print("\n=== Step 3: build dataset v2 ===")
    build_dataset_v2()

    print("\n=== Step 4: fine-tune from best.pt (round 2) ===")
    print(f"Base:    {MODEL_PT.name}  (not COCO weights)")
    print(f"Epochs:  {EPOCHS_V2}  lr0=0.0005  patience=15")
    retrain()


if __name__ == "__main__":
    main()
