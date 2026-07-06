"""
Extract frames from training videos at a fixed rate (default: 2 fps).

Why 2 fps and not every frame?
  - Consecutive frames are nearly identical — training on all of them wastes time
    and makes the dataset biased toward static scenes.
  - 2 fps gives enough variety while keeping the dataset manageable (~300-600
    frames per video on a typical 2-3 minute clip).

Output: data/frames/train_a/0001.jpg, 0002.jpg, ...
        Only train videos are extracted — eval video stays untouched.
"""

import cv2
from pathlib import Path

VIDEOS_DIR = Path(__file__).parent.parent / "data" / "videos"
FRAMES_DIR = Path(__file__).parent.parent / "data" / "frames"

# Only extract from train videos — eval is held out
TRAIN_VIDEOS = ["train_a", "train_b", "train_c", "train_d"]

SAMPLE_FPS = 4  # how many frames to keep per second of video


def extract_frames(video_id: str) -> int:
    """Extract frames from one video. Returns number of frames saved."""
    video_path = next(VIDEOS_DIR.glob(f"{video_id}.*"), None)
    if video_path is None:
        print(f"  ERROR: {video_id} not found in {VIDEOS_DIR}")
        return 0

    out_dir = FRAMES_DIR / video_id
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS)          # original fps of the video (e.g. 30)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_frames / video_fps

    # Keep every Nth frame so that SAMPLE_FPS frames per second are saved.
    # E.g. video is 30fps, we want 2fps → keep every 15th frame.
    step = max(1, round(video_fps / SAMPLE_FPS))

    print(f"  {video_id}: {duration_s:.1f}s @ {video_fps:.0f}fps → "
          f"saving every {step}th frame (~{total_frames // step} frames)")

    saved = 0
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            out_path = out_dir / f"{saved:04d}.jpg"
            cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            saved += 1
        frame_idx += 1

    cap.release()
    return saved


def main():
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    total = 0
    for video_id in TRAIN_VIDEOS:
        count = extract_frames(video_id)
        total += count
        print(f"  → saved {count} frames\n")

    print(f"Total frames extracted: {total}")
    print(f"Saved to: {FRAMES_DIR}")


if __name__ == "__main__":
    main()
