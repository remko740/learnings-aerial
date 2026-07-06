"""
Manual label cleanup tool.

Controls:
  Left click       — select nearest box (turns red)
  D                — delete selected box
  Right click+drag — draw new box
  U                — undo last action
  S / Space        — save and next frame
  A                — previous frame (no save)
  Q / Esc          — quit
"""

import cv2
import numpy as np
from pathlib import Path
import sys

ROOT       = Path(__file__).parent.parent
FRAMES_DIR = ROOT / "data" / "frames"
LABELS_DIR = ROOT / "data" / "labels_mixed"
IMG_W, IMG_H = 1280, 720

VIDEO      = sys.argv[1] if len(sys.argv) > 1 else "train_d"
PROGRESS_F = ROOT / "data" / ".cleanup_progress.txt"


def load_progress():
    if not PROGRESS_F.exists():
        return {}
    out = {}
    for line in PROGRESS_F.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = int(v.strip())
    return out


def save_progress(video, frame_idx):
    p = load_progress()
    p[video] = frame_idx
    PROGRESS_F.write_text("\n".join(f"{k}={v}" for k, v in p.items()))


def load_boxes(txt_path):
    if not txt_path.exists() or txt_path.stat().st_size == 0:
        return []
    boxes = []
    for line in txt_path.read_text().strip().splitlines():
        p = line.split()
        if len(p) >= 5:
            boxes.append(list(map(float, p[1:5])))
    return boxes


def save_boxes(txt_path, boxes):
    lines = [f"0 {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}" for b in boxes]
    txt_path.write_text("\n".join(lines))


def to_xyxy(box, w, h):
    xc, yc, bw, bh = box
    return (int((xc - bw/2)*w), int((yc - bh/2)*h),
            int((xc + bw/2)*w), int((yc + bh/2)*h))


def to_xywhn(x1, y1, x2, y2, w, h):
    xc = ((x1 + x2) / 2) / w
    yc = ((y1 + y2) / 2) / h
    bw = abs(x2 - x1) / w
    bh = abs(y2 - y1) / h
    return [xc, yc, bw, bh]


def nearest_box(boxes, mx, my, w, h):
    best, best_d = -1, float("inf")
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = to_xyxy(b, w, h)
        cx, cy = (x1+x2)//2, (y1+y2)//2
        d = (mx-cx)**2 + (my-cy)**2
        if d < best_d:
            best_d, best = d, i
    return best


def draw_frame(img, boxes, selected, draw_start, draw_cur):
    out = img.copy()

    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = to_xyxy(b, IMG_W, IMG_H)
        color = (0, 0, 220) if i == selected else (0, 220, 80)
        thick = 3 if i == selected else 2
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
        cv2.putText(out, str(i), (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # box being drawn right now
    if draw_start and draw_cur:
        cv2.rectangle(out, draw_start, draw_cur, (0, 180, 255), 2)

    mode = "DRAW" if draw_start else "SELECT"
    color_mode = (0, 180, 255) if draw_start else (255, 255, 255)
    info = (f"{VIDEO}  |  frame {frame_counter}  |  {len(boxes)} boxes  |  "
            f"[{mode}]  LClick=select  RDrag=draw  D=del  U=undo  S=save+next  A=prev  Q=quit")
    cv2.putText(out, info, (8, IMG_H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_mode, 1)
    return out


# shared mouse state
draw_start = None
draw_cur   = None
selected   = -1
boxes      = []
frame_counter = 0


def on_mouse(event, mx, my, flags, param):
    global draw_start, draw_cur, selected, boxes

    if event == cv2.EVENT_LBUTTONDOWN:
        # left click — select nearest existing box
        if draw_start is None:
            selected = nearest_box(boxes, mx, my, IMG_W, IMG_H)

    elif event == cv2.EVENT_RBUTTONDOWN:
        # right button start — begin drawing new box
        draw_start = (mx, my)
        draw_cur   = (mx, my)
        selected   = -1

    elif event == cv2.EVENT_MOUSEMOVE and draw_start:
        draw_cur = (mx, my)

    elif event == cv2.EVENT_RBUTTONUP and draw_start:
        # right button release — finish drawing
        x1, y1 = draw_start
        x2, y2 = mx, my
        if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:
            new_box = to_xywhn(x1, y1, x2, y2, IMG_W, IMG_H)
            boxes.append(new_box)
            selected = len(boxes) - 1
        draw_start = None
        draw_cur   = None


def main():
    global boxes, selected, draw_start, draw_cur, frame_counter

    frame_paths = sorted((FRAMES_DIR / VIDEO).glob("*.jpg"))
    progress = load_progress()
    idx      = progress.get(VIDEO, 0)
    print(f"Loaded {len(frame_paths)} frames from {VIDEO}, resuming from frame {idx}")
    print("Left click = select  |  Right drag = draw new box  |  D = delete  |  S = save+next")

    history  = {}

    cv2.namedWindow("cleanup", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("cleanup", IMG_W, IMG_H)
    cv2.setMouseCallback("cleanup", on_mouse)

    while 0 <= idx < len(frame_paths):
        fp  = frame_paths[idx]
        txt = LABELS_DIR / VIDEO / fp.name.replace(".jpg", ".txt")
        img = cv2.imread(str(fp))
        img = cv2.resize(img, (IMG_W, IMG_H))

        boxes       = load_boxes(txt)
        selected    = -1
        draw_start  = None
        draw_cur    = None
        frame_counter = idx

        while True:
            cv2.imshow("cleanup", draw_frame(img, boxes, selected, draw_start, draw_cur))
            key = cv2.waitKey(20) & 0xFF

            if key in (ord('q'), 27):
                cv2.destroyAllWindows()
                return

            elif key == ord('d') and selected >= 0:
                history[idx] = boxes[:]
                boxes.pop(selected)
                selected = -1

            elif key == ord('u') and idx in history:
                boxes = history.pop(idx)
                selected = -1

            elif key in (ord('s'), ord(' ')):
                save_boxes(txt, boxes)
                save_progress(VIDEO, idx + 1)
                print(f"  saved {fp.name}: {len(boxes)} boxes")
                idx += 1
                break

            elif key == ord('a'):
                idx -= 1
                break

    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
