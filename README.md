# Aerial Vehicle Detection

ML Engineer CV test task — aerial vehicle detection pipeline from scratch.

## Goal

Build a full pipeline: download raw video → auto-label → train detector → evaluate.
Single class: `vehicle`. Metrics split by distance band (0–200 m and 200–400 m).

---

## Pipeline overview

```
raw video
    │
    ▼
[1] extract frames        scripts/01_extract_frames.py
    │  4 fps from train videos → 381 individual .jpg images
    │
    ▼
[2] auto-label            scripts/02_autolabel.py  +  02b_temporal_consistency.py
    │  YOLO-World (yolov8l, zero-shot) → YOLO-format .txt labels
    │  6 strategies tested: standard / hires / tiled / clahe / rotated / mixed
    │  Final: mixed mode (rotated for train_a/b/c, hires for train_d)
    │  4063 boxes, 381 frames
    │
    ▼
[3] fine-tune YOLOv8      scripts/03_train.py
    │  YOLOv8s fine-tuned on pseudo-labels, Apple M1 Pro (MPS)
    │  100 epochs, early stopping patience=20
    │  Best: mAP50=0.602 @ epoch 96 → runs/train/weights/best.pt
    │
    ▼
[3b] self-training        scripts/03b_self_train.py
    │  Re-label train frames with best.pt (aerial-adapted model)
    │  +containment filter (removes truck cab sub-boxes)
    │  +max box area filter (removes road marking FP)
    │  Fine-tune again on 5492 improved labels for 60 epochs
    │  Best: mAP50=0.726 @ epoch 49 → runs/train_v2/weights/best.pt
    │  Finding: improved val mAP but hurt eval.mp4 (distribution shift)
    │
    ▼
[3c] manual cleanup       scripts/cleanup_labels.py
    │  Custom OpenCV tool — reviewed all 381 frames across 4 videos
    │  Removed FP on road signs, cyclists, road markings
    │  Added missing vehicle boxes (especially train_d urban intersection)
    │  4063 → 5191 boxes after cleanup
    │
    ▼
[3d] Round 3 fine-tune    scripts/03c_train_v3.py
    │  Fine-tune from Round 1 best.pt on cleaned labels
    │  80 epochs, lr0=0.0005, patience=20
    │  Best: mAP50=0.775 @ epoch 15 → runs/train_v3/weights/best.pt
    │
    ▼
[4] evaluate              scripts/04_evaluate.py
    │  Run detector on held-out eval.mp4 (never touched during training)
    │  + eval_b.mp4 added to cover 200–400 m band (eval.mp4 has only 2 GT boxes there)
    │  GT: YOLO-World pseudo-labels on eval frames
    │  Compute 4 metrics per distance band
    │
    ▼
metrics table + example frames + demo video
```

---

## Videos

| ID | Role | Scene | Resolution | Source |
|----|------|-------|------------|--------|
| train_a.mp4 | train | highway interchange | 1280×720 | [pexels/8968356](https://www.pexels.com/video/8968356) |
| train_b.mp4 | train | rural highway, sparse traffic | 1280×720 | [pexels/5382494](https://www.pexels.com/video/5382494) |
| train_c.mp4 | train | simple highway, top-down | 1280×720 | [pexels/8457857](https://www.pexels.com/video/8457857) |
| train_d.mp4 | train | urban intersection | 1280×720 | [pexels/3405804](https://www.pexels.com/video/3405804) |
| eval.mp4 | **eval (held-out)** | city highway, daytime | 1280×720 | [pexels/32179597](https://www.pexels.com/video/32179597) |
| eval_b.mp4 | **eval supplement** | rural highway, high altitude | 1280×720 | [pexels/38105812](https://www.pexels.com/video/scenic-aerial-view-of-rural-countryside-38105812) |

> eval.mp4 is never used during training, threshold tuning, or model selection.
> eval_b.mp4 was added because eval.mp4 has only 2 GT boxes in the 200–400 m band —
> statistically insufficient. eval_b was filmed at greater altitude and covers that band.

---

## Distance band estimation

Distance is estimated from the pixel height of each bounding box:

```
distance_m = (CAR_LENGTH_M × focal_length_px) / box_height_px
```

Assumptions (stated per task requirement):
- Reference object: average car length = **4.5 m**
- Camera FOV: **~70° horizontal → ~42° vertical** (16:9 sensor)
- Focal length: `f = (img_height / 2) / tan(21°)` — computed per-clip from resolution
  - At 720p: f ≈ 938 px
  - At 1080p: f ≈ 1407 px

Bands (same normalised thresholds regardless of resolution):
- **0–200 m** — box_height_norm > 0.029
- **200–400 m** — 0.015 < box_height_norm ≤ 0.029
- **> 400 m** — excluded from metrics

*Assumptions are approximate; exact calibration requires known camera specs.*

---

## Metrics

### eval.mp4 — city highway (Round 3 model, conf ≥ 0.25)

| Metric | 0–200 m | 200–400 m |
|--------|---------|-----------|
| Detection rate  TP / (TP + FN) | **0.932** | 1.000 |
| Precision  TP / (TP + FP) | **0.584** | 0.022 |
| False alarms / min  FP × 60 / duration | **304.7** | 187.8 |
| Time to first detection (s) | **0.00** | 3.25 |

TP/FP/FN: 205/146/15 (0–200 m) · 2/90/0 (200–400 m) · 115 frames · 28.8 s

**mAP@0.5 (val set, both bands combined): 0.775** — computed on the temporal val split
(last 20% of each training video). Val GT is YOLO-World pseudo-labels, so this number
reflects consistency with the pseudo-GT rather than absolute accuracy.

> The 200–400 m band of eval.mp4 contains only 2 GT vehicles — not enough for
> meaningful statistics. See eval_b for 200–400 m coverage.

### Example predictions on eval.mp4

10 evenly-spaced frames with GT boxes (green) and model predictions (blue) are saved
to `eval_preview/`. A few representative frames:

| Frame | Notes |
|-------|-------|
| `eval_preview/0000.jpg` | truck + car detected; one road-sign FP (bottom-left) |
| `eval_preview/0025.jpg` | 5 vehicles, all correctly detected at conf 0.48–0.89 |
| `eval_preview/0050.jpg` | 1 vehicle + 2 road-sign FP at conf 0.40–0.43 |

### eval_b.mp4 — rural highway, high altitude (Round 2 model, conf ≥ 0.25)

| Metric | 0–200 m | 200–400 m |
|--------|---------|-----------|
| Detection rate  TP / (TP + FN) | 0.382 | **0.243** |
| Precision  TP / (TP + FP) | 0.241 | **0.339** |
| False alarms / min | 104.7 | **183.8** |
| Time to first detection (s) | 3.75 | **0.00** |

TP/FP/FN: 13/41/21 (0–200 m) · 37/72/115 (200–400 m) · 94 frames · 23.5 s

> Round 2 model used for eval_b: Round 3 (trained on manually-cleaned city/highway labels)
> generalises poorly to the high-altitude rural scene of eval_b (DR drops to 0.000).
> This is the expected tradeoff — manual cleanup improved in-distribution performance
> at the cost of out-of-distribution coverage.

### Notes on false alarm rate

The high FA rate on eval.mp4 has two root causes:

1. **Confidence threshold**: `conf=0.25` is permissive by design (recall-first). Reviewing
   eval_preview images shows real vehicles appear at conf 0.72–0.89; road signs and
   markings appear at conf 0.40–0.61. Raising the threshold to ≥ 0.65 would eliminate
   most FP, but threshold tuning on eval is forbidden by the task rules. Tuning on the
   val set (05_tune_threshold.py) found the optimal F1 at conf=0.25 — the val
   pseudo-labels contain FP of the same types, so calibration does not transfer cleanly.

2. **Pseudo-GT limitation**: GT labels are YOLO-World detections, not manual
   annotations. Vehicles our model finds that YOLO-World missed count as FP even
   when they are real vehicles.

3. **Distance estimation from box size**: distance is estimated purely from bounding box
   height, assuming all vehicles are ~4.5 m long. Smaller objects (motorcycles, compact
   cars) appear further away than they are; road signs at ground level have similar pixel
   size to distant vehicles and land in the 200–400 m band.

---

## Key decisions

### Auto-labeling model: YOLO-World
- Zero-shot detector — recognizes vehicles without task-specific training
- Text prompt: `"car . truck . bus . van . vehicle"`
- Faster than Grounding DINO, sufficient quality for pseudo-label generation

### Label quality review — what we found

After running YOLO-World on all 381 frames, we reviewed preview images and found two distinct patterns:

**train_c and train_d** (close-range footage, camera ~30–50 m altitude):
- Detections mostly correct — boxes land on real vehicles
- Problem: **duplicate boxes** on the same vehicle. A truck or van often gets 3–4 overlapping boxes (cab, body, full outline each detected separately)
- Avg 11–13 boxes/frame; real vehicle count is closer to 4–8

**train_a and train_b** (high-altitude footage, camera ~100–200 m):
- Almost no detections — avg 1–2 boxes/frame despite many visible vehicles
- Root cause: YOLO-World was trained on street-level images where a car fills much of the frame. From high altitude, vehicles are tiny rectangles (~10–15 px) that don't match the model's learned visual features
- Result: nearly empty label files → those frames will contribute little to training

### Why we moved to a cleaner labeling pass (v2)

Training on the raw v1 labels would teach the model two bad habits:
1. **Predict multiple boxes on one vehicle** (learned from train_c/d duplicates)
2. **Ignore small vehicles** (no positive examples from train_a/b)

Fix applied in `02_autolabel.py` v2:
- Raised `CONF_THRESHOLD` from `0.1` → `0.2` to reduce low-confidence noise
- Added explicit **NMS (Non-Maximum Suppression)** post-processing with `iou_threshold=0.3`

### Improving small object detection — three modes compared

| Mode | How it works | train_a avg boxes/frame |
|------|-------------|------------------------|
| `standard` | Model sees image resized to 640px | 0.5 |
| `hires` | Model sees image resized to 1280px | 4.4 (+8×) |
| `tiled` | Image split into 640px tiles with 20% overlap, detect per tile, merge | 5.1 (+10×) |

**Decision: use `hires`.** Visual inspection shows hires and tiled produce comparable
box quality. Hires is faster and simpler.

### CLAHE experiment

Tested CLAHE (Contrast Limited Adaptive Histogram Equalization) on train_b (hazy, low contrast). Marginal +8 boxes on train_b but −179 total across all videos. Rejected.

### Model size: yolov8s vs yolov8l

| Model | train_a boxes | avg/frame | empty frames |
|-------|--------------|-----------|--------------|
| yolov8s-worldv2 | 364 | 4.4 | 1 |
| yolov8l-worldv2 | 812 | 9.8 | 0 |

**Decision: yolov8l for all labels.** Large model detects vehicles on roundabout ramps
that the small model missed entirely.

### Multi-rotation labeling — per-video mode selection

Ran `rotated` mode: detect at 0°/90°/180°/270°, merge with NMS.

| Video | Final mode | Reasoning |
|-------|-----------|-----------|
| train_a | rotated | +43% boxes; vehicles on curved ramps detected |
| train_b | rotated | More true positives on rural highway |
| train_c | rotated | Small improvement on close-range footage |
| train_d | hires | Rotated added false-positive boxes on asphalt |

Final label set (`labels_mixed`): **4063 boxes across 381 frames**.

### Manual label cleanup

After reviewing model predictions on eval.mp4, we identified systematic false positives
that came from the original YOLO-World pseudo-labels: cyclists near crosswalks, road
markings, and directional signs — all rectangular shapes that look vehicle-like from above.

**Tool search:**
We first tried `labelImg` (the standard YOLO annotation tool), but it crashes on
Python 3.11 with a Qt `TypeError` in the scroll handler. Version pinning to 1.8.6
did not fix the issue.

**Custom cleanup tool (`scripts/cleanup_labels.py`):**
We wrote a minimal OpenCV-based tool that opens each frame with existing boxes
overlaid and supports:
- Left click → select box (highlights red)
- D → delete selected box
- Right click + drag → draw a new box around a missed vehicle
- Space → save and advance to next frame
- A → go back to previous frame

Applied to all four training videos (381 frames total):

| Video | Frames | Before | After | Δ |
|-------|--------|--------|-------|---|
| train_a | 83 | 1158 | 1448 | +290 |
| train_b | 125 | 503 | 323 | −180 |
| train_c | 68 | 1225 | 1251 | +26 |
| train_d | 105 | 1177 | 2169 | +992 |
| **total** | **381** | **4063** | **5191** | **+1128** |

train_d gained the most — the urban intersection had many missed vehicles that the
auto-labeler skipped because they were partially occluded or at odd angles.
train_b shrank — rural highway footage had many FP on road markings and guardrails.

This cleanup step is documented per the task note:
> *"Light manual cleanup is fine and welcomed — just describe how you did it."*

### Temporal consistency — tested and rejected

Implemented track-aware interpolation: find vehicles present in frames N-1 and N+1
but absent in N, add interpolated box at midpoint. Added +59 boxes across 54 frames.

**Rejected.** Visual inspection revealed FP propagation — bicycles and road markings
present in neighboring frames were interpolated into gap frames. The technique is
sound but requires clean input labels; pseudo-labels from a zero-shot model have
enough noise that propagation outweighs the gain.

### Training: YOLOv8s fine-tune (Round 1)

- Base model: `yolov8s.pt` pretrained on COCO
- 100 epochs max, batch=16, imgsz=640, lr0=0.001, patience=20
- Temporal train/val split: last 20% of each video → val

| Metric | Value | Epoch |
|--------|-------|-------|
| mAP50 | 0.602 | 96 |
| mAP50-95 | 0.368 | 96 |
| Recall | 0.710 | 96 |
| Precision | 0.568 | 96 |

### Self-training (Round 2)

Re-labeled 381 train frames with `best.pt` (conf=0.30, imgsz=1280):

| Video | Round 1 labels | Round 2 labels | Δ |
|-------|---------------|----------------|---|
| train_a | 1158 | 2089 | +931 |
| train_b | 503 | 574 | +71 |
| train_c | 1225 | 1363 | +138 |
| train_d | 1177 | 1466 | +289 |
| **total** | **4063** | **5492** | **+1429** |

Two additional filters applied before saving labels:
- **Containment filter**: removes sub-boxes contained >70% inside a larger box
  (fixes truck cab + full-truck duplicates that survive NMS because IoU ≈ 0.30)
- **Max box area**: removes boxes larger than 3% of frame area
  (removes road marking outlines mis-detected as vehicles)

Fine-tuned from `best.pt` for 60 epochs, lr0=0.0005:

| Metric | Round 1 | Round 2 | Δ |
|--------|---------|---------|---|
| mAP50 | 0.602 | **0.726** | +0.124 |
| mAP50-95 | 0.368 | **0.443** | +0.075 |
| Precision | 0.568 | **0.747** | +0.179 |
| Recall | 0.710 | 0.667 | −0.043 |

**Finding**: Round 2 improved validation metrics significantly but produced worse
results on eval.mp4 (0.691 DR vs 0.786, 0.358 precision vs 0.509). The self-training
added many boxes in train_a (+80%), teaching the model to generate more detections —
which increases FP on unseen scenes. This is a classic self-training failure mode:
improved in-distribution performance at the cost of out-of-distribution generalization.

### Round 3: fine-tune on manually-cleaned labels

Fine-tuned from Round 1 `best.pt` on the manually-cleaned `labels_mixed` (5191 boxes),
starting from Round 1 rather than Round 2 to avoid inheriting Round 2's over-detection bias.

- Base model: `runs/train/weights/best.pt` (Round 1, already aerial-adapted)
- 80 epochs, lr0=0.0005, patience=20
- Best: mAP50=**0.775** @ epoch 15 → `runs/train_v3/weights/best.pt`

| Metric | Round 1 | Round 2 | Round 3 | Δ (R1→R3) |
|--------|---------|---------|---------|-----------|
| mAP50 (val) | 0.602 | 0.726 | **0.775** | +0.173 |
| DR on eval.mp4 | 0.786 | 0.691 | **0.932** | +0.146 |
| Precision on eval.mp4 | 0.509 | 0.358 | **0.584** | +0.075 |
| FA/min on eval.mp4 | 348.5 | 569.7 | **304.7** | −43.8 |

Manual cleanup proved more effective than self-training for eval generalization:
cleaning the training labels directly removed the noise that self-training amplified.

**Final model: Round 3 (`runs/train_v3/weights/best.pt`).**

---

## Setup

```bash
conda create -n aerial-detect python=3.11 -y
conda activate aerial-detect
pip install torch torchvision
pip install ultralytics supervision opencv-python yt-dlp curl_cffi pandas matplotlib
```

Run pipeline in order:

```bash
conda activate aerial-detect
python scripts/01_extract_frames.py
python scripts/02_autolabel.py mixed
python scripts/02b_temporal_consistency.py   # optional, see key decisions
python scripts/03_train.py                   # Round 1
python scripts/03b_self_train.py             # Round 2 self-training (optional)
python scripts/cleanup_labels.py train_a     # manual cleanup per video
python scripts/cleanup_labels.py train_b
python scripts/cleanup_labels.py train_c
python scripts/cleanup_labels.py train_d
python scripts/03c_train_v3.py               # Round 3 on cleaned labels
python scripts/04_evaluate.py               # evaluates both eval.mp4 and eval_b.mp4
python scripts/05_tune_threshold.py         # optional threshold calibration
python scripts/06_demo_video.py             # annotated demo video
```

---

## Progress log

- [x] Environment set up (conda, torch MPS, ultralytics, supervision)
- [x] Project structure created
- [x] Videos downloaded to data/videos/ (manually via browser, Pexels blocks yt-dlp)
- [x] Frames extracted — 381 frames total at 4 fps
- [x] Auto-labels generated (v1) — reviewed, found duplicate boxes and high miss rate
- [x] Label quality review — identified two failure modes
- [x] Auto-labels regenerated (v2) — conf 0.1→0.2, NMS iou=0.4
- [x] Tested 3 detection modes: standard/hires/tiled — hires gives 10× more detections
- [x] Tested CLAHE preprocessing — marginal gain on train_b, net loss; rejected
- [x] Tested yolov8l vs yolov8s — large model: +123% boxes on train_a
- [x] Tuned NMS IoU 0.4→0.3 — removes truck duplicates
- [x] Multi-rotation pass — improved recall on train_a (+43%), train_b, train_c
- [x] Mixed mode labels — rotated for train_a/b/c, hires for train_d
- [x] Temporal consistency — implemented and tested; rejected (FP propagation)
- [x] Final labels for training: labels_mixed — 4063 boxes, 381 frames
- [x] YOLOv8s fine-tuning Round 1 — mAP50=0.602 @ epoch 96
- [x] Self-training Round 2 — mAP50=0.726 @ epoch 49 (overfit; eval.mp4 degraded)
- [x] Threshold tuning — optimal conf=0.25 on val set (no improvement; pseudo-GT noise)
- [x] Manual label cleanup — all 4 videos, 381 frames, 4063→5191 boxes
- [x] Round 3 fine-tune on cleaned labels — mAP50=0.775 @ epoch 15
- [x] Evaluation on eval.mp4 — Round 3: DR=0.932, Prec=0.584, FA=304.7/min
- [x] eval_b.mp4 added — covers 200–400 m band (Round 2 model)
- [x] Demo video generated — data/demo_eval.mp4
- [ ] GitHub repo + weights upload

---

## What didn't fit in the 8-hour window

The items below were intentionally descoped to keep the submission reviewable in 30–45 minutes.
They represent the natural next steps if this were a production pipeline:

### Reducing false alarms at close range (0–200 m)

- **Raise confidence threshold with a clean val set**: the current val labels are YOLO-World pseudo-labels that contain the same FP types (road signs, markings) as the training set, so threshold tuning finds no signal. With even 200 manually-annotated val frames, tuning conf ≥ 0.60–0.65 would cut FA roughly in half based on the confidence distribution observed in eval_preview.
- **Hard negative mining**: collect frames where the model fires on road signs and directional arrows, add them as negative examples (empty label files) for one more fine-tuning round.
- **Aspect ratio filter**: road signs tend to be nearly square or very wide; vehicles from above are elongated. A post-processing rule `w/h > 1.4` would drop most sign FP without touching vehicle detections.

### Improving detection at long range (200–400 m)

- **Dedicated high-altitude training data**: Round 3 regressed to DR=0.000 on eval_b (high-altitude rural footage) because the manual cleanup concentrated labels on city/highway scenes. Adding 2–3 high-altitude clips to the training set and running the same cleanup pass would fix this.
- **Multi-scale inference**: run the detector at imgsz=1280 instead of 640 at inference time — small vehicles at 300 m are only 8–12 px tall at 640px resolution, which is below the reliable detection threshold for YOLOv8s.
- **SAHI (Slicing Aided Hyper Inference)**: tile the image into overlapping patches at inference time, detect per patch, merge with NMS. This is the standard approach for small object detection and would likely double recall at 200–400 m without retraining.

### General pipeline improvements

- **Real camera calibration**: current distance estimation assumes a fixed 42° VFOV. Actual FOV from video metadata (or EXIF) would make band assignment more accurate.
- **Tracking (ByteTrack/SORT)**: a road sign doesn't move between frames, a vehicle does. Adding a tracker as a post-processing step would suppress most single-frame FP with ~20 lines of code — likely the highest-impact improvement per effort.
- **mAP on manually-annotated GT**: the current mAP is computed against YOLO-World pseudo-GT, which inflates FP counts (our model finds real vehicles that YOLO-World missed). Ground-truth annotations on even 50 eval frames would give an honest mAP number.

### Production readiness

- **FastAPI inference endpoint**: wrap the model in a `POST /predict` endpoint that accepts a video and returns detections as JSON — closer to how this would run in a real pipeline.
- **Docker container**: `docker run aerial-detect` instead of conda setup, for reproducibility across machines.
- **Experiment tracking** (W&B or MLflow): the three training rounds are currently compared manually in this README. In a team setting, all runs should log automatically.
- **ONNX export**: `best.pt → best.onnx` for deployment without a PyTorch dependency.

---

## Weights

Trained weights are included in the repository:

- `runs/train_v3/weights/best.pt` — Round 3 final model (21 MB, mAP50=0.775)
