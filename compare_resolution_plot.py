"""640x360 vs 4K 해상도별 Baseline vs I/P-frame 스케줄러 비교 막대 그래프 생성."""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    "font.family": ["Malgun Gothic", "AppleGothic", "Noto Sans CJK KR", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})

COLOR_BASE = "#E74C3C"
COLOR_OURS = "#2E86C1"


def load_total_ms(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return np.array([float(r["total_ms"]) for r in rows])


DATASETS = [
    ("640x360", "baseline_log.csv", "latency_log.csv"),
    ("4K",      "baseline_4k.csv",  "latency_4k.csv"),
]


def main():
    labels = []
    base_means, ours_means = [], []
    base_fps, ours_fps = [], []

    for label, base_csv, ours_csv in DATASETS:
        bt = load_total_ms(base_csv)
        ot = load_total_ms(ours_csv)
        labels.append(label)
        base_means.append(bt.mean())
        ours_means.append(ot.mean())
        base_fps.append(1000 / bt.mean())
        ours_fps.append(1000 / ot.mean())
        print(f"[{label}] Baseline mean={bt.mean():.2f}ms ({1000/bt.mean():.1f} FPS) "
              f"| Scheduler mean={ot.mean():.2f}ms ({1000/ot.mean():.1f} FPS)")

    x = np.arange(len(labels))
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    ax = axes[0]
    bars1 = ax.bar(x - w/2, base_means, w, label="Baseline (All I-frame)", color=COLOR_BASE, alpha=0.85)
    bars2 = ax.bar(x + w/2, ours_means, w, label="I/P-frame Scheduler",   color=COLOR_OURS, alpha=0.85)
    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"{bar.get_height():.1f}ms", ha="center", va="bottom", fontsize=11, fontweight="bold")
    for i in range(len(labels)):
        ratio = base_means[i] / max(ours_means[i], 0.01)
        ax.text(x[i], max(base_means[i], ours_means[i]) * 1.12, f"{ratio:.1f}x↓",
                ha="center", fontsize=12, color="darkgreen", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean Latency (ms)")
    ax.set_title("Mean Latency by Resolution")
    ax.set_ylim(0, max(base_means) * 1.35)
    ax.legend(fontsize=10)

    ax2 = axes[1]
    bars3 = ax2.bar(x - w/2, base_fps, w, label="Baseline (All I-frame)", color=COLOR_BASE, alpha=0.85)
    bars4 = ax2.bar(x + w/2, ours_fps, w, label="I/P-frame Scheduler",   color=COLOR_OURS, alpha=0.85)
    for bar in list(bars3) + list(bars4):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    for i in range(len(labels)):
        ratio = ours_fps[i] / max(base_fps[i], 0.01)
        ax2.text(x[i], max(base_fps[i], ours_fps[i]) * 1.12, f"{ratio:.1f}x↑",
                 ha="center", fontsize=12, color="darkgreen", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("FPS")
    ax2.set_title("Throughput (FPS) by Resolution")
    ax2.set_ylim(0, max(ours_fps) * 1.35)
    ax2.legend(fontsize=10)

    fig.suptitle("Baseline vs I/P-frame Scheduler: 640x360 vs 4K", fontsize=15, fontweight="bold")
    plt.tight_layout()

    out_path = Path("10_resolution_compare.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[Plot] saved -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
