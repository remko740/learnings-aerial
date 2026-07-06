"""
Temporal consistency post-processing for auto-labels.

Problem:
  The labeler processes each frame independently — it has no memory of what
  it saw in the previous frame. This creates "blinking" labels: a vehicle
  clearly visible in frames N-1 and N+1 may be missed in frame N because the
  model happened to be slightly less confident on that frame.

  Two variants of this problem exist:

  (A) Completely empty frame:
      frame N-1: [car_A, car_B]
      frame N:   []               ← model missed everything
      frame N+1: [car_A, car_B]

  (B) Partially detected frame — the harder case:
      frame N-1: [car_A, car_B]
      frame N:   [car_B]          ← car_A missed; frame is NOT empty
      frame N+1: [car_A, car_B]

  Our first version only caught case (A). Visual inspection of train_a showed
  case (B) is also common — the highway has many vehicles and the model
  sporadically misses individual ones in otherwise well-detected frames.

Solution: track-aware interpolation
  For each frame N, we look at what vehicles "pass through" — meaning they
  are present in BOTH frame N-1 and frame N+1. For each such vehicle, we
  check if it also appears in frame N. If not → add an interpolated box.

  Step 1: Match boxes in frame N-1 to boxes in frame N+1 by center proximity
          (greedy nearest-neighbor). These are "pass-through vehicles".
  Step 2: For each pass-through vehicle, compute its expected position in
          frame N (linear midpoint between N-1 and N+1 positions).
  Step 3: Check if frame N already has a box near that expected position.
          If yes → already detected, nothing to do.
          If no  → add the interpolated box to frame N.

Matching threshold:
  At 4 fps and ~100 m drone altitude, a vehicle at 60 km/h moves ~100 px
  between frames. MAX_CENTER_DIST = 200 px gives comfortable headroom while
  preventing unrelated detections from being linked.
"""

import cv2
import numpy as np
from pathlib import Path

SOURCE_DIR  = Path(__file__).parent.parent / "data" / "labels_mixed"
OUTPUT_DIR  = Path(__file__).parent.parent / "data" / "labels_temporal"
FRAMES_DIR  = Path(__file__).parent.parent / "data" / "frames"
PREVIEW_DIR = Path(__file__).parent.parent / "data" / "preview_temporal"
PREVIEW_N   = 8  # preview frames per video

MAX_CENTER_DIST = 200  # pixels — max distance to consider two boxes the same vehicle
IMG_W, IMG_H = 1280, 720


# ── label I/O ─────────────────────────────────────────────────────────────────

def read_labels(path: Path) -> np.ndarray:
    """Load YOLO label file → (N, 5) array [class, xc, yc, w, h] normalized."""
    if not path.exists() or path.stat().st_size == 0:
        return np.empty((0, 5))
    rows = []
    for line in path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) == 5:
            rows.append(list(map(float, parts)))
    return np.array(rows) if rows else np.empty((0, 5))


def write_labels(boxes: np.ndarray, path: Path) -> None:
    """Write (N, 5) array to YOLO format .txt file."""
    lines = [
        f"{int(r[0])} {r[1]:.6f} {r[2]:.6f} {r[3]:.6f} {r[4]:.6f}"
        for r in boxes
    ]
    path.write_text("\n".join(lines))


# ── geometry helpers ──────────────────────────────────────────────────────────

def center_dist_px(a: np.ndarray, b: np.ndarray) -> float:
    """Pixel distance between centers of two normalized boxes."""
    return float(np.sqrt(
        ((a[1] - b[1]) * IMG_W) ** 2 +
        ((a[2] - b[2]) * IMG_H) ** 2
    ))


def interpolate(prev_box: np.ndarray, next_box: np.ndarray) -> np.ndarray:
    """Linear midpoint between two matched boxes."""
    return np.array([
        prev_box[0],
        (prev_box[1] + next_box[1]) / 2,
        (prev_box[2] + next_box[2]) / 2,
        (prev_box[3] + next_box[3]) / 2,
        (prev_box[4] + next_box[4]) / 2,
    ])


# ── core: find vehicles missing in current frame ──────────────────────────────

def find_missing(prev_boxes: np.ndarray,
                 curr_boxes: np.ndarray,
                 next_boxes: np.ndarray) -> np.ndarray:
    """
    Find vehicles that "pass through" (present in prev AND next) but are
    absent from curr. Returns interpolated boxes to add to curr.

    Algorithm:
      1. Match prev → next (greedy nearest-neighbor): these are pass-through vehicles.
      2. For each matched pair, compute expected position in curr (midpoint).
      3. If curr has no box within MAX_CENTER_DIST of that expected position
         → the vehicle was missed → add the interpolated box.
    """
    if len(prev_boxes) == 0 or len(next_boxes) == 0:
        return np.empty((0, 5))

    # Step 1: match prev to next
    pass_through = []
    used_next = set()

    for pb in prev_boxes:
        best_dist = MAX_CENTER_DIST
        best_j = -1
        for j, nb in enumerate(next_boxes):
            if j in used_next:
                continue
            d = center_dist_px(pb, nb)
            if d < best_dist:
                best_dist = d
                best_j = j
        if best_j >= 0:
            used_next.add(best_j)
            pass_through.append((pb, next_boxes[best_j]))

    if not pass_through:
        return np.empty((0, 5))

    # Step 2 + 3: for each pass-through pair, check if curr already has it
    to_add = []
    for pb, nb in pass_through:
        expected = interpolate(pb, nb)

        already_present = any(
            center_dist_px(expected, cb) < MAX_CENTER_DIST
            for cb in curr_boxes
        ) if len(curr_boxes) > 0 else False

        if not already_present:
            to_add.append(expected)

    return np.array(to_add) if to_add else np.empty((0, 5))


# ── preview helpers ───────────────────────────────────────────────────────────

def draw_preview(frame_path: Path, original_boxes: np.ndarray,
                 added_boxes: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """
    Draw boxes on a frame image.
    Original (kept) boxes → green.
    Newly interpolated boxes → yellow, thicker border.
    This makes it easy to spot at a glance what temporal consistency added.
    """
    img = cv2.imread(str(frame_path))

    for row in original_boxes:
        xc, yc, w, h = row[1] * img_w, row[2] * img_h, row[3] * img_w, row[4] * img_h
        x1, y1 = int(xc - w / 2), int(yc - h / 2)
        x2, y2 = int(xc + w / 2), int(yc + h / 2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # green

    for row in added_boxes:
        xc, yc, w, h = row[1] * img_w, row[2] * img_h, row[3] * img_w, row[4] * img_h
        x1, y1 = int(xc - w / 2), int(yc - h / 2)
        x2, y2 = int(xc + w / 2), int(yc + h / 2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 220, 255), 3)  # yellow, thicker
        cv2.putText(img, "interp", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 255), 1)

    return img


# ── per-video processing ──────────────────────────────────────────────────────

def process_video(video_id: str) -> dict:
    src_dir     = SOURCE_DIR / video_id
    dst_dir     = OUTPUT_DIR / video_id
    preview_dir = PREVIEW_DIR / video_id
    dst_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    label_files  = sorted(src_dir.glob("*.txt"))
    frame_files  = sorted((FRAMES_DIR / video_id).glob("*.jpg"))
    all_labels   = [read_labels(f) for f in label_files]

    # Determine which frames get a preview:
    # Always include patched frames (so we can verify the interpolated boxes),
    # plus a few evenly-spaced frames for context.
    boxes_before   = sum(len(b) for b in all_labels)
    added_total    = 0
    frames_patched = 0

    # Build updated labels + track which boxes were added per frame
    updated    = [arr.copy() for arr in all_labels]
    added_per  = [np.empty((0, 5)) for _ in all_labels]  # added boxes per frame

    for i in range(1, len(label_files) - 1):
        new_boxes = find_missing(all_labels[i - 1], all_labels[i], all_labels[i + 1])
        if len(new_boxes) > 0:
            updated[i] = (
                np.vstack([all_labels[i], new_boxes])
                if len(all_labels[i]) > 0
                else new_boxes
            )
            added_per[i]    = new_boxes
            added_total    += len(new_boxes)
            frames_patched += 1

    # Write all label files
    for fp, boxes in zip(label_files, updated):
        write_labels(boxes, dst_dir / fp.name)

    # Generate previews: patched frames + evenly spaced sample
    patched_idx = {i for i, a in enumerate(added_per) if len(a) > 0}
    sample_idx  = set(np.linspace(0, len(frame_files) - 1, PREVIEW_N, dtype=int))
    preview_idx = patched_idx | sample_idx

    for i in preview_idx:
        if i >= len(frame_files):
            continue
        img_w = IMG_W  # assume consistent resolution
        img_h = IMG_H
        preview = draw_preview(
            frame_files[i],
            all_labels[i],   # original boxes → green
            added_per[i],    # interpolated boxes → yellow
            img_w, img_h,
        )
        cv2.imwrite(str(preview_dir / frame_files[i].name), preview)

    return {
        "frames": len(label_files),
        "boxes_before": boxes_before,
        "added": added_total,
        "frames_patched": frames_patched,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    video_ids = sorted(d.name for d in SOURCE_DIR.iterdir() if d.is_dir())

    print(f"Source: {SOURCE_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"MAX_CENTER_DIST: {MAX_CENTER_DIST} px\n")

    grand_before  = 0
    grand_added   = 0
    grand_patched = 0

    for vid in video_ids:
        r = process_video(vid)
        grand_before  += r["boxes_before"]
        grand_added   += r["added"]
        grand_patched += r["frames_patched"]
        print(
            f"  {vid}: +{r['added']:3d} boxes  "
            f"({r['frames_patched']} frames patched)"
        )

    print(f"\nBefore: {grand_before} boxes")
    print(f"Added:  +{grand_added} boxes  ({grand_patched} frames patched)")
    print(f"After:  {grand_before + grand_added} boxes")
    print(f"\nLabels → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
