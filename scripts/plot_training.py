"""
Plot training metrics from runs/train/results.csv.

Usage:
  python scripts/plot_training.py          # save PNG and open it
  python scripts/plot_training.py --live   # auto-refresh every 30s in a window
"""

import sys
import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

CSV_PATH  = Path(__file__).parent.parent / "runs" / "train" / "results.csv"
PNG_PATH  = Path(__file__).parent.parent / "runs" / "train" / "progress.png"
REFRESH_S = 30


def load() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.strip()
    return df


def draw(df: pd.DataFrame, fig, axes) -> None:
    epochs = df["epoch"]
    best_epoch = df["metrics/mAP50(B)"].idxmax()
    best_map   = df["metrics/mAP50(B)"].max()

    colors = {
        "train": "#4C9BE8",
        "val":   "#E8754C",
        "map":   "#2ECC71",
        "prec":  "#9B59B6",
        "rec":   "#F39C12",
    }

    for ax in axes.flat:
        ax.clear()

    # ── top-left: mAP50 ──────────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(epochs, df["metrics/mAP50(B)"], color=colors["map"], lw=2, label="mAP50")
    ax.axvline(df.loc[best_epoch, "epoch"], color=colors["map"],
               lw=1, ls="--", alpha=0.5)
    ax.annotate(f"best {best_map:.3f}\nepoch {df.loc[best_epoch,'epoch']:.0f}",
                xy=(df.loc[best_epoch, "epoch"], best_map),
                xytext=(10, -20), textcoords="offset points",
                fontsize=8, color=colors["map"])
    ax.set_title("mAP50", fontweight="bold")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # ── top-right: precision & recall ────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(epochs, df["metrics/precision(B)"], color=colors["prec"],
            lw=2, label="Precision")
    ax.plot(epochs, df["metrics/recall(B)"],    color=colors["rec"],
            lw=2, label="Recall")
    ax.set_title("Precision & Recall", fontweight="bold")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # ── bottom-left: box loss ─────────────────────────────────────────────────
    ax = axes[1, 0]
    ax.plot(epochs, df["train/box_loss"], color=colors["train"],
            lw=2, label="train")
    ax.plot(epochs, df["val/box_loss"],   color=colors["val"],
            lw=2, label="val")
    ax.set_title("Box Loss  ↓ better", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # ── bottom-right: cls loss ────────────────────────────────────────────────
    ax = axes[1, 1]
    ax.plot(epochs, df["train/cls_loss"], color=colors["train"],
            lw=2, label="train")
    ax.plot(epochs, df["val/cls_loss"],   color=colors["val"],
            lw=2, label="val")
    ax.set_title("Class Loss  ↓ better", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # ── shared styling ────────────────────────────────────────────────────────
    last = df.iloc[-1]
    fig.suptitle(
        f"Training progress  —  epoch {last['epoch']:.0f}  |  "
        f"mAP50 {last['metrics/mAP50(B)']:.3f}  |  "
        f"recall {last['metrics/recall(B)']:.3f}  |  "
        f"val/box_loss {last['val/box_loss']:.3f}",
        fontsize=11, fontweight="bold", y=1.01,
    )
    fig.tight_layout()


def save_png(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    draw(df, fig, axes)
    fig.savefig(PNG_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {PNG_PATH}")


def live(df_init: pd.DataFrame) -> None:
    plt.ion()
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    plt.show()

    df = df_init
    while plt.fignum_exists(fig.number):
        draw(df, fig, axes)
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        print(f"  epoch {df.iloc[-1]['epoch']:.0f}  mAP50={df.iloc[-1]['metrics/mAP50(B)']:.3f}"
              f"  — refreshing in {REFRESH_S}s  (close window to stop)")
        for _ in range(REFRESH_S * 10):
            if not plt.fignum_exists(fig.number):
                break
            plt.pause(0.1)
        df = load()

    print("Window closed.")


def main():
    if not CSV_PATH.exists():
        print(f"Not found: {CSV_PATH}")
        print("Training hasn't started yet or runs/train/ path is wrong.")
        return

    df = load()

    if "--live" in sys.argv:
        print(f"Live mode — refreshes every {REFRESH_S}s. Close the window to stop.")
        live(df)
    else:
        save_png(df)
        import subprocess, os
        subprocess.run(["open", str(PNG_PATH)])


if __name__ == "__main__":
    main()
