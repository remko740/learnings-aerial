"""
Auto-label training frames using YOLO-World (zero-shot detector).

Supports five detection modes for comparison, plus a mixed mode:
  standard  — default 640px input size
  hires     — 1280px input (imgsz=1280); same model, bigger input
  tiled     — SAHI-style slicing: split image into overlapping tiles,
              detect on each, merge back; best for small/distant objects
  clahe     — hires + CLAHE contrast enhancement before detection;
              helps for low-contrast scenes (haze, early morning lighting)
  rotated   — hires at 0°+90°+180°+270°, boxes mapped back to original space,
              merged with NMS; catches vehicles moving toward/away from camera
  mixed     — per-video mode selected based on visual inspection results:
              train_a/b/c → rotated (improved recall without noise)
              train_d     → hires   (rotated added false positives on asphalt)

Run a specific mode:
  python 02_autolabel.py standard
  python 02_autolabel.py hires
  python 02_autolabel.py tiled
  python 02_autolabel.py clahe
  python 02_autolabel.py rotated
  python 02_autolabel.py mixed    ← use this for final training labels

Results land in:
  data/labels_<mode>/   ← YOLO .txt files
  data/preview_<mode>/  ← sample frames with boxes drawn
"""

import sys
import cv2
import numpy as np
import torch
from pathlib import Path
from torchvision.ops import nms
from ultralytics import YOLOWorld

FRAMES_DIR = Path(__file__).parent.parent / "data" / "frames"

CLASSES        = ["car", "truck", "bus", "van", "vehicle"]
CONF_THRESHOLD = 0.2
IMGSZ          = 1280  # inference resolution for hires/rotated modes

# CLAHE config — used only in "clahe" mode
# clipLimit: max contrast amplification per tile (higher = more aggressive)
# tileGridSize: image is divided into this many tiles for local equalization
CLAHE_CLIP   = 2.0
CLAHE_GRID   = (8, 8)
NMS_IOU        = 0.3  # lowered from 0.4 — trucks often get cab+body detected
                      # separately; IoU between them is ~0.2-0.35 which 0.4 misses
PREVIEW_COUNT  = 5  # preview frames saved per video

# Tiling config — used only in "tiled" mode
TILE_SIZE    = 640   # each tile fed to the model at this resolution
TILE_OVERLAP = 0.2   # 20% overlap between tiles so objects at edges aren't cut


# ── helpers ──────────────────────────────────────────────────────────────────

def apply_nms(boxes: np.ndarray, confs: np.ndarray) -> np.ndarray:
    """
    Non-Maximum Suppression: remove duplicate boxes that describe the same object.

    Algorithm:
      1. Sort boxes by confidence (highest first)
      2. Keep the top box, remove any other box whose IoU with it exceeds
         NMS_IOU (they overlap too much → same object → duplicate)
      3. Repeat on remaining boxes

    IoU (Intersection over Union) = overlap_area / union_area
      0.0 = no overlap at all
      1.0 = identical boxes
      0.4 = 40% overlap → we treat this as a duplicate

    Returns the indices of boxes to keep.
    """
    if len(boxes) == 0:
        return np.array([], dtype=int)
    keep = nms(
        torch.tensor(boxes, dtype=torch.float32),
        torch.tensor(confs, dtype=torch.float32),
        iou_threshold=NMS_IOU,
    )
    return keep.numpy()


def save_yolo_labels(boxes: np.ndarray, img_w: int, img_h: int, path: Path) -> int:
    """
    Write boxes to a YOLO-format .txt file.

    YOLO format: <class_id> <x_center> <y_center> <width> <height>
    All values normalized 0–1. class_id = 0 (single class: vehicle).
    """
    lines = []
    for x1, y1, x2, y2 in boxes:
        xc = ((x1 + x2) / 2) / img_w
        yc = ((y1 + y2) / 2) / img_h
        w  = (x2 - x1) / img_w
        h  = (y2 - y1) / img_h
        lines.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    path.write_text("\n".join(lines))
    return len(lines)


def draw_boxes(img: np.ndarray, boxes: np.ndarray, confs: np.ndarray) -> np.ndarray:
    out = img.copy()
    for (x1, y1, x2, y2), c in zip(boxes, confs):
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(out, f"{c:.2f}", (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    return out


# ── preprocessing ────────────────────────────────────────────────────────────

def apply_clahe(img: np.ndarray) -> np.ndarray:
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Why standard histogram equalization isn't enough:
      Regular equalization looks at the WHOLE image histogram and stretches it.
      If 80% of the image is bright sky, the remaining 20% (road + vehicles)
      gets compressed into a narrow dark range — vehicles become even harder to see.

    CLAHE solves this by:
      1. Dividing the image into small tiles (CLAHE_GRID = 8×8 = 64 tiles)
      2. Running histogram equalization INDEPENDENTLY on each tile
         → the road tile gets its own stretch, regardless of the sky tiles
      3. Applying a clip limit (CLAHE_CLIP) — if a tile's histogram bar exceeds
         this limit, the excess contrast is redistributed rather than amplified,
         which prevents noise amplification in uniform regions

    Applied to the L (lightness) channel in LAB colorspace so that colors
    are not distorted — only the perceived brightness is enhanced.
    """
    # Convert BGR (OpenCV default) → LAB colorspace
    # LAB separates lightness (L) from color (A=green-red, B=blue-yellow)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE only to the L (lightness) channel
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
    l_enhanced = clahe.apply(l)

    # Merge back and convert to BGR
    enhanced = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


# ── detection modes ───────────────────────────────────────────────────────────

def detect_standard(model, img: np.ndarray, imgsz: int = 640):
    """
    Standard single-pass detection.
    imgsz controls what resolution the image is resized to before inference.
    Default is 640; hires mode passes 1280 here.
    """
    results = model.predict(img, conf=CONF_THRESHOLD, imgsz=imgsz, verbose=False)[0]
    boxes = results.boxes.xyxy.cpu().numpy() if len(results.boxes) else np.empty((0, 4))
    confs = results.boxes.conf.cpu().numpy() if len(results.boxes) else np.array([])
    return boxes, confs


def detect_tiled(model, img: np.ndarray):
    """
    Sliced / tiled detection — the core idea of SAHI.

    Why this helps for small objects:
      A car that is 12px tall in a 720p image gets scaled to ~5px when the
      whole image is resized to 640. The model can barely see it.
      If we instead cut the image into 640px tiles and run the model on each
      tile, that same car is now 24px tall → much easier to detect.

    Step by step:
      1. Compute how many tiles fit horizontally and vertically, with overlap
      2. For each tile: crop → run detector → convert box coords back to the
         full-image coordinate space
      3. Pool all detections from all tiles
      4. Run NMS across the pool to remove duplicates at tile boundaries
         (the same car can appear in two adjacent tiles)

    Overlap (TILE_OVERLAP=0.2) prevents objects sitting exactly on a tile edge
    from being cut in half and missed.
    """
    h, w = img.shape[:2]

    # Step size = tile size × (1 - overlap). With 20% overlap a 640px tile
    # steps 512px, so each tile shares 128px with its neighbor.
    step = int(TILE_SIZE * (1 - TILE_OVERLAP))

    all_boxes, all_confs = [], []

    y = 0
    while y < h:
        x = 0
        while x < w:
            # Clamp tile to image boundaries
            x2 = min(x + TILE_SIZE, w)
            y2 = min(y + TILE_SIZE, h)
            x1_t, y1_t = x2 - TILE_SIZE, y2 - TILE_SIZE
            x1_t = max(x1_t, 0)
            y1_t = max(y1_t, 0)

            tile = img[y1_t:y2, x1_t:x2]
            results = model.predict(tile, conf=CONF_THRESHOLD,
                                    imgsz=TILE_SIZE, verbose=False)[0]

            if len(results.boxes):
                # Boxes are in tile-local coords → shift to full-image coords
                boxes = results.boxes.xyxy.cpu().numpy().copy()
                boxes[:, 0] += x1_t  # x1
                boxes[:, 2] += x1_t  # x2
                boxes[:, 1] += y1_t  # y1
                boxes[:, 3] += y1_t  # y2
                all_boxes.append(boxes)
                all_confs.append(results.boxes.conf.cpu().numpy())

            if x2 == w:
                break
            x += step
        if y2 == h:
            break
        y += step

    if not all_boxes:
        return np.empty((0, 4)), np.array([])

    boxes = np.vstack(all_boxes)
    confs = np.concatenate(all_confs)
    return boxes, confs


def rotate_boxes_back(boxes: np.ndarray, rotation: int, H: int, W: int) -> np.ndarray:
    """
    Convert bounding boxes detected in a rotated image back to original coordinates.

    Each rotation changes which side of the vehicle the camera "sees":
      0°   — original, no change needed
      90°  — image rotated CW; vertical-road vehicles now appear horizontal
      180° — image flipped; same perspective, mirrored
      270° — image rotated CCW; same as 90° but other direction

    Formulas derived from the rotation inverse transform:
      If point (x, y) in original maps to rotated image as:
        90° CW:  (H-1-y, x)  → image shape (W, H)
        180°:    (W-1-x, H-1-y) → image shape (H, W)
        270° CW: (y, W-1-x)  → image shape (W, H)
      then the inverse (rotated → original) for a box (x1,y1,x2,y2) is:
        90° CW:  x1_o=y1_r,      y1_o=H-x2_r, x2_o=y2_r,      y2_o=H-x1_r
        180°:    x1_o=W-x2_r,    y1_o=H-y2_r, x2_o=W-x1_r,    y2_o=H-y1_r
        270° CW: x1_o=W-y2_r,    y1_o=x1_r,   x2_o=W-y1_r,    y2_o=x2_r
    """
    if len(boxes) == 0:
        return boxes
    out = np.empty_like(boxes)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    if rotation == 90:
        out[:, 0] = y1;     out[:, 1] = H - x2
        out[:, 2] = y2;     out[:, 3] = H - x1
    elif rotation == 180:
        out[:, 0] = W - x2; out[:, 1] = H - y2
        out[:, 2] = W - x1; out[:, 3] = H - y1
    elif rotation == 270:
        out[:, 0] = W - y2; out[:, 1] = x1
        out[:, 2] = W - y1; out[:, 3] = x2
    else:
        out = boxes.copy()
    # Clamp to image bounds (floating point safety)
    out[:, 0] = np.clip(out[:, 0], 0, W)
    out[:, 2] = np.clip(out[:, 2], 0, W)
    out[:, 1] = np.clip(out[:, 1], 0, H)
    out[:, 3] = np.clip(out[:, 3], 0, H)
    return out


# OpenCV rotation codes indexed by degrees
_CV2_ROTATIONS = {
    90:  cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def detect_rotated(model, img: np.ndarray):
    """
    Run hires detection at 0°, 90°, 180°, 270°, map all boxes back to
    the original image coordinate space, then apply NMS globally.

    Why four rotations?
      Each rotation changes which face of the vehicle is visible to the model:
      - 0°/180°: catches vehicles moving left-right (horizontal roads)
      - 90°/270°: catches vehicles moving up-down (vertical roads, toward/away)
      The four together give near-omnidirectional coverage.
    """
    H, W = img.shape[:2]
    all_boxes, all_confs = [], []

    for angle in [0, 90, 180, 270]:
        if angle == 0:
            rotated_img = img
        else:
            rotated_img = cv2.rotate(img, _CV2_ROTATIONS[angle])

        results = model.predict(rotated_img, conf=CONF_THRESHOLD,
                                imgsz=IMGSZ, verbose=False)[0]
        if len(results.boxes) == 0:
            continue

        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        boxes_orig = rotate_boxes_back(boxes, angle, H, W)
        all_boxes.append(boxes_orig)
        all_confs.append(confs)

    if not all_boxes:
        return np.empty((0, 4)), np.array([])
    return np.vstack(all_boxes), np.concatenate(all_confs)


# ── main pipeline ─────────────────────────────────────────────────────────────

def process_video(model, video_id: str, mode: str,
                  labels_dir: Path, preview_dir: Path) -> dict:
    frames_dir = FRAMES_DIR / video_id
    (labels_dir / video_id).mkdir(parents=True, exist_ok=True)
    (preview_dir / video_id).mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(frames_dir.glob("*.jpg"))
    preview_idx = set(np.linspace(0, len(frame_paths) - 1,
                                  PREVIEW_COUNT, dtype=int))
    total_boxes, empty = 0, 0

    for i, fp in enumerate(frame_paths):
        img = cv2.imread(str(fp))
        h, w = img.shape[:2]

        if mode == "tiled":
            boxes, confs = detect_tiled(model, img)
        elif mode == "clahe":
            img_enhanced = apply_clahe(img)
            boxes, confs = detect_standard(model, img_enhanced, imgsz=1280)
        elif mode == "rotated":
            boxes, confs = detect_rotated(model, img)
        elif mode == "hires":
            boxes, confs = detect_standard(model, img, imgsz=1280)
        else:
            boxes, confs = detect_standard(model, img, imgsz=640)

        # Always deduplicate regardless of mode
        keep = apply_nms(boxes, confs)
        boxes, confs = boxes[keep], confs[keep]

        n = save_yolo_labels(boxes, w, h,
                             labels_dir / video_id / fp.name.replace(".jpg", ".txt"))
        total_boxes += n
        if n == 0:
            empty += 1

        if i in preview_idx:
            preview = draw_boxes(img, boxes, confs)
            cv2.imwrite(str(preview_dir / video_id / fp.name), preview)

    return {"frames": len(frame_paths), "boxes": total_boxes,
            "empty": empty, "avg": total_boxes / max(len(frame_paths), 1)}


def main():
    # Per-video mode overrides used in "mixed" mode.
    # Chosen by visual inspection: rotated improved recall on a/b/c without noise;
    # on train_d rotated added false-positive boxes on asphalt → use hires there.
    VIDEO_MODES = {
        "train_a": "rotated",
        "train_b": "rotated",
        "train_c": "rotated",
        "train_d": "hires",
    }

    mode = sys.argv[1] if len(sys.argv) > 1 else "standard"
    if mode not in ("standard", "hires", "tiled", "clahe", "rotated", "mixed"):
        print("Usage: python 02_autolabel.py [standard|hires|tiled|clahe|rotated|mixed]")
        sys.exit(1)

    labels_dir  = Path(__file__).parent.parent / f"data/labels_{mode}"
    preview_dir = Path(__file__).parent.parent / f"data/preview_{mode}"

    print(f"Mode: {mode}")
    print(f"Loading YOLO-World ...")
    model = YOLOWorld("yolov8l-worldv2.pt")
    model.set_classes(CLASSES)
    print(f"Classes: {CLASSES}\n")

    video_ids = sorted(d.name for d in FRAMES_DIR.iterdir() if d.is_dir())
    totals = {"frames": 0, "boxes": 0, "empty": 0}

    for vid in video_ids:
        vid_mode = VIDEO_MODES.get(vid, "hires") if mode == "mixed" else mode
        print(f"  {vid} [{vid_mode}] ...", end=" ", flush=True)
        s = process_video(model, vid, vid_mode, labels_dir, preview_dir)
        print(f"{s['boxes']} boxes  ({s['avg']:.1f}/frame,  {s['empty']} empty frames)")
        for k in totals:
            totals[k] += s[k]

    print(f"\nTotal: {totals['boxes']} boxes across {totals['frames']} frames")
    print(f"Labels  → {labels_dir}")
    print(f"Preview → {preview_dir}")


if __name__ == "__main__":
    main()
