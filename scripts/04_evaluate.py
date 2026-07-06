"""
Evaluate the fine-tuned detector on held-out eval clips.

Supports multiple eval videos so both distance bands (0-200 m, 200-400 m)
are covered even if a single clip does not contain vehicles at all ranges.

Per-clip pipeline:
  1. Extract frames at EVAL_FPS
  2. YOLO-World → pseudo ground-truth labels
  3. best.pt → predictions
  4. IoU matching → TP / FP / FN per frame
  5. Distance estimation → assign each box to a band
  6. Save example images

Distance estimation:
  distance_m = (CAR_LENGTH_M × focal_px) / box_height_px

  focal_px is derived per-clip from image height and assumed vertical FOV:
    focal_px = (img_h / 2) / tan(VFOV_HALF_DEG)
  Assumptions: car ≈ 4.5 m, camera ~70° HFOV → ~42° VFOV (16:9).
  Same normalised band thresholds apply regardless of resolution.
"""

import cv2
import numpy as np
import torch
from dataclasses import dataclass, field
from pathlib import Path
from math import tan, radians
from torchvision.ops import nms, box_iou
from ultralytics import YOLO, YOLOWorld

# ── paths ─────────────────────────────────────────────────────────────────────

ROOT      = Path(__file__).parent.parent
VIDEO_DIR = ROOT / "data" / "videos"
MODEL_PT  = ROOT / "runs" / "train_v3" / "weights" / "best.pt"

# fallback to round 1 if round 2 doesn't exist yet
if not MODEL_PT.exists():
    MODEL_PT = ROOT / "runs" / "train" / "weights" / "best.pt"

# ── eval clips ────────────────────────────────────────────────────────────────

# Add or remove clips here. tag is used for directory names.
EVAL_CLIPS = [
    {"tag": "eval",   "file": "eval.mp4"},
    {"tag": "eval_b", "file": "eval_b.mp4"},
]

# ── config ────────────────────────────────────────────────────────────────────

EVAL_FPS     = 4
GT_CONF      = 0.20
GT_NMS_IOU   = 0.30
PRED_CONF    = 0.25
PRED_NMS_IOU = 0.30
MATCH_IOU    = 0.50
IMGSZ        = 1280
CLASSES      = ["car", "truck", "bus", "van", "vehicle"]
PREVIEW_N    = 10

CAR_LENGTH_M  = 4.5
VFOV_HALF_DEG = 21.0   # ~42° vertical FOV, 16:9, ~70° HFOV

BANDS = {"0-200m": (0, 200), "200-400m": (200, 400)}


# ── distance helpers ──────────────────────────────────────────────────────────

def focal_from_height(img_h: int) -> float:
    return (img_h / 2) / tan(radians(VFOV_HALF_DEG))


def estimate_distance_m(bh_norm: float, focal_px: float, img_h: int) -> float:
    px = bh_norm * img_h
    return (CAR_LENGTH_M * focal_px) / px if px > 0 else float("inf")


def band_for(dist_m: float) -> str | None:
    for name, (lo, hi) in BANDS.items():
        if lo < dist_m <= hi:
            return name
    return None


# ── box helpers ───────────────────────────────────────────────────────────────

def apply_nms(boxes, confs, iou):
    if len(boxes) == 0:
        return np.array([], dtype=int)
    return nms(
        torch.tensor(boxes, dtype=torch.float32),
        torch.tensor(confs, dtype=torch.float32),
        iou_threshold=iou,
    ).numpy()


def xywhn_to_xyxy(boxes_norm, w, h):
    out = np.empty_like(boxes_norm)
    out[:, 0] = (boxes_norm[:, 0] - boxes_norm[:, 2] / 2) * w
    out[:, 1] = (boxes_norm[:, 1] - boxes_norm[:, 3] / 2) * h
    out[:, 2] = (boxes_norm[:, 0] + boxes_norm[:, 2] / 2) * w
    out[:, 3] = (boxes_norm[:, 1] + boxes_norm[:, 3] / 2) * h
    return out


def match_preds_to_gt(pred_xyxy, gt_xyxy):
    if len(pred_xyxy) == 0 and len(gt_xyxy) == 0:
        return set(), set(), set()
    if len(pred_xyxy) == 0:
        return set(), set(), set(range(len(gt_xyxy)))
    if len(gt_xyxy) == 0:
        return set(), set(range(len(pred_xyxy))), set()
    iou_mat = box_iou(
        torch.tensor(pred_xyxy, dtype=torch.float32),
        torch.tensor(gt_xyxy,  dtype=torch.float32),
    ).numpy()
    matched_p, matched_g = set(), set()
    for i, j, v in sorted(
        [(i, j, iou_mat[i, j]) for i in range(len(pred_xyxy)) for j in range(len(gt_xyxy))],
        key=lambda x: -x[2],
    ):
        if v < MATCH_IOU:
            break
        if i not in matched_p and j not in matched_g:
            matched_p.add(i); matched_g.add(j)
    return matched_g, set(range(len(pred_xyxy))) - matched_p, set(range(len(gt_xyxy))) - matched_g


# ── per-clip pipeline ─────────────────────────────────────────────────────────

def extract_frames(video_path: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob("*.jpg"))
    if existing:
        print(f"  Frames already exist: {len(existing)}")
        return existing
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    step = max(1, round(fps / EVAL_FPS))
    saved = idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            cv2.imwrite(str(out_dir / f"{saved:04d}.jpg"), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
            saved += 1
        idx += 1
    cap.release()
    paths = sorted(out_dir.glob("*.jpg"))
    print(f"  Extracted {len(paths)} frames at {EVAL_FPS} fps")
    return paths


def generate_gt(frame_paths, gt_dir, yw_model):
    gt_dir.mkdir(parents=True, exist_ok=True)
    if len(list(gt_dir.glob("*.txt"))) == len(frame_paths):
        print(f"  GT labels already exist ({len(frame_paths)} files)")
        return
    total = 0
    for fp in frame_paths:
        img = cv2.imread(str(fp))
        h, w = img.shape[:2]
        res = yw_model.predict(img, conf=GT_CONF, imgsz=IMGSZ, verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy() if len(res.boxes) else np.empty((0, 4))
        confs = res.boxes.conf.cpu().numpy() if len(res.boxes) else np.array([])
        keep  = apply_nms(boxes, confs, GT_NMS_IOU)
        boxes = boxes[keep]
        lines = []
        for x1, y1, x2, y2 in boxes:
            xc = ((x1+x2)/2)/w; yc = ((y1+y2)/2)/h
            bw = (x2-x1)/w;     bh = (y2-y1)/h
            lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
        (gt_dir / fp.name.replace(".jpg", ".txt")).write_text("\n".join(lines))
        total += len(lines)
    print(f"  GT: {total} boxes across {len(frame_paths)} frames")


def run_predictions(frame_paths, pred_dir, det_model):
    pred_dir.mkdir(parents=True, exist_ok=True)
    if len(list(pred_dir.glob("*.txt"))) == len(frame_paths):
        print(f"  Predictions already exist ({len(frame_paths)} files)")
        return
    total = 0
    for fp in frame_paths:
        img = cv2.imread(str(fp))
        h, w = img.shape[:2]
        res  = det_model.predict(img, conf=PRED_CONF, imgsz=640, verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy() if len(res.boxes) else np.empty((0, 4))
        confs = res.boxes.conf.cpu().numpy() if len(res.boxes) else np.array([])
        keep  = apply_nms(boxes, confs, PRED_NMS_IOU)
        boxes = boxes[keep]; confs = confs[keep]
        lines = []
        for (x1, y1, x2, y2), c in zip(boxes, confs):
            xc = ((x1+x2)/2)/w; yc = ((y1+y2)/2)/h
            bw = (x2-x1)/w;     bh = (y2-y1)/h
            lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f} {c:.4f}")
        (pred_dir / fp.name.replace(".jpg", ".txt")).write_text("\n".join(lines))
        total += len(lines)
    print(f"  Predictions: {total} boxes across {len(frame_paths)} frames")


def compute_metrics(frame_paths, gt_dir, pred_dir):
    band_stats = {b: {"tp": 0, "fp": 0, "fn": 0, "first_tp_frame": None}
                  for b in BANDS}
    for idx, fp in enumerate(frame_paths):
        img = cv2.imread(str(fp))
        img_h, img_w = img.shape[:2]
        focal = focal_from_height(img_h)

        def load_norm(path, with_conf=False):
            if not path.exists() or path.stat().st_size == 0:
                return (np.empty((0, 4)), np.array([])) if with_conf else np.empty((0, 4))
            rows, cs = [], []
            for line in path.read_text().strip().splitlines():
                p = line.split()
                if len(p) >= 5:
                    rows.append(list(map(float, p[1:5])))
                    cs.append(float(p[5]) if len(p) > 5 else 1.0)
            if not rows:
                return (np.empty((0, 4)), np.array([])) if with_conf else np.empty((0, 4))
            return (np.array(rows), np.array(cs)) if with_conf else np.array(rows)

        gt_norm   = load_norm(gt_dir   / fp.name.replace(".jpg", ".txt"))
        pred_norm, _ = load_norm(pred_dir / fp.name.replace(".jpg", ".txt"), with_conf=True)

        gt_xyxy   = xywhn_to_xyxy(gt_norm,   img_w, img_h) if len(gt_norm)   else np.empty((0, 4))
        pred_xyxy = xywhn_to_xyxy(pred_norm, img_w, img_h) if len(pred_norm) else np.empty((0, 4))

        matched_gt, fp_idx, fn_idx = match_preds_to_gt(pred_xyxy, gt_xyxy)

        gt_bands = [band_for(estimate_distance_m(r[3], focal, img_h)) for r in gt_norm] \
                   if len(gt_norm) else []

        for j in matched_gt:
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
            if i < len(pred_norm):
                b = band_for(estimate_distance_m(pred_norm[i, 3], focal, img_h))
                if b:
                    band_stats[b]["fp"] += 1

    return band_stats


def save_examples(frame_paths, gt_dir, pred_dir, viz_dir, img_h_default):
    viz_dir.mkdir(parents=True, exist_ok=True)
    for idx in np.linspace(0, len(frame_paths) - 1, PREVIEW_N, dtype=int):
        fp  = frame_paths[idx]
        img = cv2.imread(str(fp))
        h, w = img.shape[:2]
        focal = focal_from_height(h)

        gt_path = gt_dir / fp.name.replace(".jpg", ".txt")
        if gt_path.exists():
            for line in gt_path.read_text().strip().splitlines():
                p = line.split()
                if len(p) == 5:
                    xc, yc, bw, bh = map(float, p[1:])
                    x1 = int((xc-bw/2)*w); y1 = int((yc-bh/2)*h)
                    x2 = int((xc+bw/2)*w); y2 = int((yc+bh/2)*h)
                    cv2.rectangle(img, (x1,y1),(x2,y2),(0,200,0),2)
                    d = estimate_distance_m(bh, focal, h)
                    cv2.putText(img, f"GT {d:.0f}m", (x1, y1-4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,200,0), 1)

        pred_path = pred_dir / fp.name.replace(".jpg", ".txt")
        if pred_path.exists():
            for line in pred_path.read_text().strip().splitlines():
                p = line.split()
                if len(p) >= 5:
                    xc, yc, bw, bh = map(float, p[1:5])
                    conf = float(p[5]) if len(p) > 5 else 0.0
                    x1 = int((xc-bw/2)*w); y1 = int((yc-bh/2)*h)
                    x2 = int((xc+bw/2)*w); y2 = int((yc+bh/2)*h)
                    cv2.rectangle(img, (x1,y1),(x2,y2),(200,80,0),2)
                    cv2.putText(img, f"{conf:.2f}", (x1, y2+12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,80,0), 1)

        cv2.imwrite(str(viz_dir / fp.name), img)
    print(f"  Saved {PREVIEW_N} examples → {viz_dir}")


# ── print helpers ─────────────────────────────────────────────────────────────

def print_table(label, band_stats, n_frames):
    dur = n_frames / EVAL_FPS
    print(f"\n── {label} ({'×'.join(str(n_frames)+'fr')}) ──")
    print(f"{'Metric':<32} {'0–200 m':>10} {'200–400 m':>10}")
    print("─" * 54)
    rows = {}
    for band in BANDS:
        s = band_stats[band]
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        dr   = tp/(tp+fn) if (tp+fn) > 0 else 0
        prec = tp/(tp+fp) if (tp+fp) > 0 else 0
        fa   = fp*60/dur  if dur > 0    else 0
        tf   = s["first_tp_frame"]/EVAL_FPS if s["first_tp_frame"] is not None else float("nan")
        rows[band] = dict(dr=dr, prec=prec, fa=fa, tf=tf, tp=tp, fp=fp, fn=fn)
    r0, r1 = rows["0-200m"], rows["200-400m"]
    print(f"{'Detection rate  TP/(TP+FN)':<32} {r0['dr']:>10.3f} {r1['dr']:>10.3f}")
    print(f"{'Precision  TP/(TP+FP)':<32} {r0['prec']:>10.3f} {r1['prec']:>10.3f}")
    print(f"{'False alarms / min':<32} {r0['fa']:>10.1f} {r1['fa']:>10.1f}")
    print(f"{'Time to first detection (s)':<32} {r0['tf']:>10.2f} {r1['tf']:>10.2f}")
    print(f"{'TP / FP / FN':<32}   {r0['tp']}/{r0['fp']}/{r0['fn']}      {r1['tp']}/{r1['fp']}/{r1['fn']}")
    return rows


def merge_stats(all_stats):
    merged = {b: {"tp": 0, "fp": 0, "fn": 0, "first_tp_frame": None} for b in BANDS}
    for stats in all_stats:
        for b in BANDS:
            merged[b]["tp"] += stats[b]["tp"]
            merged[b]["fp"] += stats[b]["fp"]
            merged[b]["fn"] += stats[b]["fn"]
    return merged


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Model: {MODEL_PT}")

    print("\n=== Loading models ===")
    yw_model = YOLOWorld("yolov8l-worldv2.pt")
    yw_model.set_classes(CLASSES)
    det_model = YOLO(str(MODEL_PT))

    all_stats  = []
    all_frames = 0

    for clip in EVAL_CLIPS:
        tag  = clip["tag"]
        vpath = VIDEO_DIR / clip["file"]
        if not vpath.exists():
            print(f"\nSkipping {clip['file']} — not found")
            continue

        print(f"\n{'='*54}")
        print(f"  Clip: {clip['file']} [{tag}]")
        print(f"{'='*54}")

        frames_dir = ROOT / "data" / "frames"   / tag
        gt_dir     = ROOT / "data" / f"labels_{tag}_gt"
        pred_dir   = ROOT / "data" / f"labels_{tag}_pred"
        viz_dir    = ROOT / "data" / f"{tag}_preview"

        print("\n--- Step 1: extract frames ---")
        frame_paths = extract_frames(vpath, frames_dir)

        print("\n--- Step 2: GT labels (YOLO-World) ---")
        generate_gt(frame_paths, gt_dir, yw_model)

        print("\n--- Step 3: predictions (best.pt) ---")
        run_predictions(frame_paths, pred_dir, det_model)

        print("\n--- Step 4: metrics ---")
        band_stats = compute_metrics(frame_paths, gt_dir, pred_dir)
        all_stats.append(band_stats)
        all_frames += len(frame_paths)

        print("\n--- Step 5: example images ---")
        # pass first frame height for label sizing
        sample_img = cv2.imread(str(frame_paths[0]))
        save_examples(frame_paths, gt_dir, pred_dir, viz_dir, sample_img.shape[0])

        print_table(clip["file"], band_stats, len(frame_paths))

    # Combined across all clips
    if len(all_stats) > 1:
        print(f"\n{'═'*54}")
        print("  COMBINED (all eval clips)")
        print(f"{'═'*54}")
        merged = merge_stats(all_stats)
        print_table("combined", merged, all_frames)

    print(f"\nAssumptions: car≈{CAR_LENGTH_M}m, VFOV≈{VFOV_HALF_DEG*2}°, focal scales with resolution")
    print("GT = YOLO-World pseudo-labels  |  green = GT  |  blue = predictions")


if __name__ == "__main__":
    main()
