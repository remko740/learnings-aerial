"""
Tune confidence threshold on the validation set, then re-evaluate on eval.mp4.

Why this is allowed:
  The task forbids threshold tuning ON eval.mp4. We tune on the validation split
  (last 20% of each training video) which is separate from eval. This is standard
  practice — pick threshold on held-out val, report results on test (eval).

Two filters tuned jointly:
  1. conf_threshold — minimum model confidence to accept a detection
  2. min_box_area   — minimum bounding box area (normalized, 0–1) to accept a detection
                      removes lamp posts, road signs, road markings that are smaller
                      than any real vehicle

Method:
  For each (conf_threshold, min_box_area) pair, compute F1 on val set.
  Pick the pair that maximises F1. Apply it to eval predictions and reprint metrics.
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from itertools import product
from torchvision.ops import nms, box_iou
from ultralytics import YOLO

ROOT       = Path(__file__).parent.parent
FRAMES_DIR = ROOT / "data" / "frames"
DATASET    = ROOT / "data" / "dataset"
MODEL_PT   = ROOT / "runs" / "train" / "weights" / "best.pt"
PRED_DIR   = ROOT / "data" / "labels_eval_pred"
GT_DIR     = ROOT / "data" / "labels_eval_gt"

MATCH_IOU  = 0.5
EVAL_FPS   = 4
IMG_W, IMG_H = 1280, 720

CAR_LENGTH_M    = 4.5
FOCAL_LENGTH_PX = 938
BANDS = {"0-200m": (0, 200), "200-400m": (200, 400)}

# Grid of values to try
CONF_THRESHOLDS  = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
MIN_BOX_AREAS    = [0.0, 0.0005, 0.001, 0.002]  # normalised w×h


# ── helpers ───────────────────────────────────────────────────────────────────

def load_labels(path: Path, with_conf: bool = False):
    """Load YOLO label file. If with_conf, expects 6th field = confidence."""
    if not path.exists() or path.stat().st_size == 0:
        return np.empty((0, 5)) if not with_conf else (np.empty((0, 4)), np.array([]))
    rows, confs = [], []
    for line in path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) >= 5:
            rows.append(list(map(float, parts[1:5])))  # xc yc bw bh
            confs.append(float(parts[5]) if len(parts) > 5 else 1.0)
    if not rows:
        return (np.empty((0, 4)), np.array([])) if with_conf else np.empty((0, 5))
    return (np.array(rows), np.array(confs)) if with_conf else np.array(rows)


def xywhn_to_xyxy(boxes, w=IMG_W, h=IMG_H):
    if len(boxes) == 0:
        return np.empty((0, 4))
    out = np.empty_like(boxes)
    out[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * w
    out[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * h
    out[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * w
    out[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * h
    return out


def match(pred_xyxy, gt_xyxy):
    if len(pred_xyxy) == 0 and len(gt_xyxy) == 0:
        return 0, 0, 0
    if len(pred_xyxy) == 0:
        return 0, 0, len(gt_xyxy)
    if len(gt_xyxy) == 0:
        return 0, len(pred_xyxy), 0
    iou = box_iou(
        torch.tensor(pred_xyxy, dtype=torch.float32),
        torch.tensor(gt_xyxy,  dtype=torch.float32),
    ).numpy()
    matched_p, matched_g = set(), set()
    for i, j, v in sorted(
        [(i, j, iou[i, j]) for i in range(len(pred_xyxy)) for j in range(len(gt_xyxy))],
        key=lambda x: -x[2]
    ):
        if v < MATCH_IOU:
            break
        if i not in matched_p and j not in matched_g:
            matched_p.add(i); matched_g.add(j)
    tp = len(matched_p)
    fp = len(pred_xyxy) - tp
    fn = len(gt_xyxy)  - tp
    return tp, fp, fn


def apply_filters(boxes_norm, confs, conf_thr, min_area):
    """Apply confidence + minimum box area filter."""
    if len(boxes_norm) == 0:
        return np.empty((0, 4))
    mask = (confs >= conf_thr) & ((boxes_norm[:, 2] * boxes_norm[:, 3]) >= min_area)
    return boxes_norm[mask]


def estimate_distance(bh_norm):
    px = bh_norm * IMG_H
    return (CAR_LENGTH_M * FOCAL_LENGTH_PX) / px if px > 0 else float("inf")


def band_for(dist):
    for name, (lo, hi) in BANDS.items():
        if lo < dist <= hi:
            return name
    return None


# ── step 1: run inference on val set ──────────────────────────────────────────

def run_val_inference():
    """Run model on val images and save predictions with confidence scores."""
    val_dir  = DATASET / "images" / "val"
    pred_dir = ROOT / "data" / "val_predictions"
    pred_dir.mkdir(exist_ok=True)

    if len(list(pred_dir.glob("*.txt"))) == len(list(val_dir.glob("*.jpg"))):
        print(f"  Val predictions already exist ({len(list(pred_dir.glob('*.txt')))} files)")
        return pred_dir

    print(f"  Running inference on {len(list(val_dir.glob('*.jpg')))} val frames...")
    model = YOLO(str(MODEL_PT))

    for fp in sorted(val_dir.glob("*.jpg")):
        img = cv2.imread(str(fp))
        h, w = img.shape[:2]
        results = model.predict(img, conf=0.15, imgsz=640, verbose=False)[0]
        boxes = results.boxes.xyxy.cpu().numpy() if len(results.boxes) else np.empty((0, 4))
        confs = results.boxes.conf.cpu().numpy() if len(results.boxes) else np.array([])
        lines = []
        for (x1, y1, x2, y2), c in zip(boxes, confs):
            xc = ((x1+x2)/2)/w; yc = ((y1+y2)/2)/h
            bw = (x2-x1)/w;     bh = (y2-y1)/h
            lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f} {c:.4f}")
        (pred_dir / fp.name.replace(".jpg", ".txt")).write_text("\n".join(lines))

    print(f"  Done → {pred_dir}")
    return pred_dir


# ── step 2: grid search on val set ───────────────────────────────────────────

def evaluate_on_val(pred_dir, conf_thr, min_area):
    val_label_dir = DATASET / "labels" / "val"
    tp_total = fp_total = fn_total = 0

    for lbl_path in sorted(val_label_dir.glob("*.txt")):
        pred_path = pred_dir / lbl_path.name
        gt_norm   = load_labels(lbl_path)
        boxes_norm, confs = load_labels(pred_path, with_conf=True)
        filtered = apply_filters(boxes_norm, confs, conf_thr, min_area)
        tp, fp, fn = match(xywhn_to_xyxy(filtered), xywhn_to_xyxy(gt_norm))
        tp_total += tp; fp_total += fp; fn_total += fn

    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
    recall    = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp_total, "fp": fp_total, "fn": fn_total}


def grid_search(pred_dir):
    print(f"\n{'Conf':>6} {'MinArea':>9} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("─" * 46)
    best, best_params = {"f1": 0}, None
    for conf, area in product(CONF_THRESHOLDS, MIN_BOX_AREAS):
        r = evaluate_on_val(pred_dir, conf, area)
        marker = " ←" if r["f1"] > best["f1"] else ""
        print(f"  {conf:.2f}  {area:.4f}  {r['precision']:>9.3f}  {r['recall']:>7.3f}  {r['f1']:>7.3f}{marker}")
        if r["f1"] > best["f1"]:
            best = r; best_params = (conf, area)
    print(f"\nBest: conf={best_params[0]}  min_area={best_params[1]}  "
          f"→  precision={best['precision']:.3f}  recall={best['recall']:.3f}  F1={best['f1']:.3f}")
    return best_params


# ── step 3: re-evaluate eval with best params ─────────────────────────────────

def recompute_eval_metrics(conf_thr, min_area):
    gt_files   = sorted(GT_DIR.glob("*.txt"))
    pred_files = sorted(PRED_DIR.glob("*.txt"))
    n_frames   = len(gt_files)
    duration_s = n_frames / EVAL_FPS

    band_stats = {b: {"tp": 0, "fp": 0, "fn": 0, "first_tp_frame": None}
                  for b in BANDS}

    for idx, (gf, pf) in enumerate(zip(gt_files, pred_files)):
        gt_norm = load_labels(gf)
        boxes_norm, confs = load_labels(pf, with_conf=True)
        filtered = apply_filters(boxes_norm, confs, conf_thr, min_area)

        gt_xyxy   = xywhn_to_xyxy(gt_norm)
        pred_xyxy = xywhn_to_xyxy(filtered)
        tp_idx, fp_idx, fn_idx = set(), set(), set()

        if len(pred_xyxy) > 0 and len(gt_xyxy) > 0:
            iou = box_iou(
                torch.tensor(pred_xyxy, dtype=torch.float32),
                torch.tensor(gt_xyxy,  dtype=torch.float32),
            ).numpy()
            matched_p, matched_g = set(), set()
            for i, j, v in sorted(
                [(i, j, iou[i, j]) for i in range(len(pred_xyxy)) for j in range(len(gt_xyxy))],
                key=lambda x: -x[2]
            ):
                if v < MATCH_IOU:
                    break
                if i not in matched_p and j not in matched_g:
                    matched_p.add(i); matched_g.add(j)
            tp_idx = matched_g
            fp_idx = set(range(len(pred_xyxy))) - matched_p
            fn_idx = set(range(len(gt_xyxy)))  - matched_g
        elif len(pred_xyxy) == 0:
            fn_idx = set(range(len(gt_xyxy)))
        else:
            fp_idx = set(range(len(pred_xyxy)))

        gt_bands = [band_for(estimate_distance(r[3])) for r in gt_norm] if len(gt_norm) else []
        for j in tp_idx:
            b = gt_bands[j] if j < len(gt_bands) else None
            if b:
                band_stats[b]["tp"] += 1
                if band_stats[b]["first_tp_frame"] is None:
                    band_stats[b]["first_tp_frame"] = idx
        for j in fn_idx:
            b = gt_bands[j] if j < len(gt_bands) else None
            if b:
                band_stats[b]["fn"] += 1
        for i in fp_idx:
            if i < len(filtered):
                b = band_for(estimate_distance(filtered[i, 3]))
                if b:
                    band_stats[b]["fp"] += 1

    print(f"\n{'═'*62}")
    print(f"  Threshold tuned: conf≥{conf_thr}  min_box_area≥{min_area}")
    print(f"{'─'*62}")
    print(f"{'Metric':<32} {'0–200 m':>12} {'200–400 m':>12}")
    print(f"{'─'*62}")
    bands = list(BANDS.keys())
    rows = {}
    for band in bands:
        s = band_stats[band]
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        det  = tp / (tp+fn) if (tp+fn) > 0 else 0
        prec = tp / (tp+fp) if (tp+fp) > 0 else 0
        fa   = fp * 60 / duration_s if duration_s > 0 else 0
        tf   = (s["first_tp_frame"] / EVAL_FPS) if s["first_tp_frame"] is not None else float("nan")
        rows[band] = dict(det=det, prec=prec, fa=fa, tf=tf, tp=tp, fp=fp, fn=fn)
    r0, r1 = rows[bands[0]], rows[bands[1]]
    print(f"{'Detection rate  TP/(TP+FN)':<32} {r0['det']:>12.3f} {r1['det']:>12.3f}")
    print(f"{'Precision  TP/(TP+FP)':<32} {r0['prec']:>12.3f} {r1['prec']:>12.3f}")
    print(f"{'False alarms / min':<32} {r0['fa']:>12.1f} {r1['fa']:>12.1f}")
    print(f"{'Time to first detection (s)':<32} {r0['tf']:>12.2f} {r1['tf']:>12.2f}")
    print(f"{'─'*62}")
    print(f"{'TP / FP / FN':<32} {r0['tp']}/{r0['fp']}/{r0['fn']}  {r1['tp']}/{r1['fp']}/{r1['fn']}")
    print(f"{'═'*62}")
    return rows


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=== Step 1: val inference (conf≥0.15 to keep all candidates) ===")
    pred_dir = run_val_inference()

    print("\n=== Step 2: grid search on val set ===")
    best_conf, best_area = grid_search(pred_dir)

    print("\n=== Step 3: re-evaluate eval.mp4 with tuned threshold ===")
    recompute_eval_metrics(best_conf, best_area)


if __name__ == "__main__":
    main()
