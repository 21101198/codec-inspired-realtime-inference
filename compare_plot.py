"""baseline_log.csv vs latency_log.csv 비교 그래프 4개 생성."""
import csv
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
COLOR_I    = "#E67E22"
COLOR_P    = "#27AE60"


def load_csv(path, has_frame_type=False):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            r = {
                "frame_idx":     int(row["frame_idx"]),
                "preprocess_ms": float(row["preprocess_ms"]),
                "htod_ms":       float(row["htod_ms"]),
                "infer_ms":      float(row["infer_ms"]),
                "dtoh_ms":       float(row["dtoh_ms"]),
                "total_ms":      float(row["total_ms"]),
            }
            if has_frame_type:
                r["frame_type"] = row["frame_type"]
                r["alert"] = row["alert"].lower() == "true"
            rows.append(r)
    return rows


def stats(values):
    arr = np.array(values)
    return {
        "p50": np.percentile(arr, 50),
        "p90": np.percentile(arr, 90),
        "p99": np.percentile(arr, 99),
        "mean": np.mean(arr),
        "std": np.std(arr),
        "fps": 1000 / np.mean(arr),
    }


def plot_distribution(base_rows, ours_rows, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    base_totals = [r["total_ms"] for r in base_rows]
    ours_totals = [r["total_ms"] for r in ours_rows]

    for ax, data, color, label in [
        (axes[0], base_totals, COLOR_BASE, "Baseline (All I-frame)"),
        (axes[1], ours_totals, COLOR_OURS, "Ours (I/P-frame Scheduler)")
    ]:
        p50 = np.percentile(data, 50)
        p90 = np.percentile(data, 90)
        p99 = np.percentile(data, 99)

        ax.hist(data, bins=60, color=color, alpha=0.7, edgecolor="white", linewidth=0.4)
        ax.axvline(p50, color="green",  linestyle="--", linewidth=2,
                   label=f"P50 = {p50:.1f}ms")
        ax.axvline(p90, color="orange", linestyle="--", linewidth=2,
                   label=f"P90 = {p90:.1f}ms")
        ax.axvline(p99, color="red",    linestyle="--", linewidth=2,
                   label=f"P99 = {p99:.1f}ms")
        ax.set_title(label)
        ax.set_xlabel("Total Latency (ms)")
        ax.set_ylabel("Frame Count")
        ax.legend(fontsize=10)

    fig.suptitle("Tail Latency Distribution: Baseline vs Proposed", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] 저장: {out_path}")


def plot_percentile_bar(base_rows, ours_rows, out_path):
    base_totals = [r["total_ms"] for r in base_rows]
    ours_totals = [r["total_ms"] for r in ours_rows]

    labels = ["P50", "P90", "P99 (Tail)"]
    base_vals = [np.percentile(base_totals, p) for p in [50, 90, 99]]
    ours_vals = [np.percentile(ours_totals, p) for p in [50, 90, 99]]

    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(9, 6))
    bars1 = ax.bar(x - w/2, base_vals, w, label="Baseline", color=COLOR_BASE, alpha=0.85)
    bars2 = ax.bar(x + w/2, ours_vals, w, label="Ours (I/P-frame)", color=COLOR_OURS, alpha=0.85)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f"{bar.get_height():.1f}ms", ha="center", va="bottom", fontsize=11, fontweight="bold")
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f"{bar.get_height():.1f}ms", ha="center", va="bottom", fontsize=11, fontweight="bold")

    for i, (bv, ov) in enumerate(zip(base_vals, ours_vals)):
        ratio = bv / max(ov, 0.01)
        ax.text(x[i], max(bv, ov) + 1.5, f"{ratio:.1f}x↓",
                ha="center", fontsize=12, color="darkgreen", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Latency (ms)", fontsize=12)
    ax.set_title("Latency Percentile Comparison: Baseline vs Proposed", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] 저장: {out_path}")


def plot_timeline_compare(base_rows, ours_rows, out_path):
    n = min(200, len(base_rows), len(ours_rows))
    frames = list(range(n))

    base_totals = [r["total_ms"] for r in base_rows[:n]]
    ours_totals = [r["total_ms"] for r in ours_rows[:n]]
    ours_types  = [r.get("frame_type", "I") for r in ours_rows[:n]]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(frames, base_totals, color=COLOR_BASE, linewidth=0.8, alpha=0.9)
    axes[0].axhline(np.percentile(base_totals, 99), color="red",
                    linestyle="--", linewidth=1.5, label=f"P99={np.percentile(base_totals,99):.1f}ms")
    axes[0].axhline(np.mean(base_totals), color="green",
                    linestyle=":", linewidth=1.5, label=f"Mean={np.mean(base_totals):.1f}ms")
    axes[0].set_title("Baseline (All frames: Full Inference)", fontsize=13)
    axes[0].set_ylabel("Latency (ms)")
    axes[0].legend(fontsize=10)

    for i, (f, t, ft) in enumerate(zip(frames, ours_totals, ours_types)):
        if ft == "I":
            axes[1].axvspan(f-0.5, f+0.5, alpha=0.15, color=COLOR_I)
    axes[1].plot(frames, ours_totals, color=COLOR_OURS, linewidth=0.8, alpha=0.9)
    axes[1].axhline(np.percentile(ours_totals, 99), color="red",
                    linestyle="--", linewidth=1.5, label=f"P99={np.percentile(ours_totals,99):.1f}ms")
    axes[1].axhline(np.mean(ours_totals), color="green",
                    linestyle=":", linewidth=1.5, label=f"Mean={np.mean(ours_totals):.1f}ms")
    i_patch = mpatches.Patch(color=COLOR_I, alpha=0.3, label="I-frame (전체 추론)")
    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].legend(handles + [i_patch], labels + ["I-frame"], fontsize=10)
    axes[1].set_title("Proposed: Codec-Inspired I/P-frame Scheduler", fontsize=13)
    axes[1].set_ylabel("Latency (ms)")
    axes[1].set_xlabel("Frame Index")

    fig.suptitle("End-to-End Latency Timeline Comparison", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] 저장: {out_path}")


def plot_fps_and_segments(base_rows, ours_rows, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    base_fps = 1000 / np.mean([r["total_ms"] for r in base_rows])
    ours_fps = 1000 / np.mean([r["total_ms"] for r in ours_rows])

    ax = axes[0]
    bars = ax.bar(["Baseline", "Proposed"], [base_fps, ours_fps],
                  color=[COLOR_BASE, COLOR_OURS], alpha=0.85, width=0.5)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f"{bar.get_height():.1f}", ha="center", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(base_fps, ours_fps) * 1.3)
    ax.text(0.5, 0.92, f"{ours_fps/base_fps:.1f}x faster",
            ha="center", fontsize=16, color="darkgreen", fontweight="bold",
            transform=ax.transAxes)
    ax.set_ylabel("FPS", fontsize=12)
    ax.set_title("Throughput (FPS)", fontsize=13, fontweight="bold")

    ax2 = axes[1]
    segments = ["Preprocess", "HtoD", "Infer", "DtoH"]
    keys = ["preprocess_ms", "htod_ms", "infer_ms", "dtoh_ms"]

    base_seg = [np.mean([r[k] for r in base_rows]) for k in keys]
    ours_seg = [np.mean([r[k] for r in ours_rows]) for k in keys]

    x = np.arange(len(segments))
    w = 0.35
    ax2.bar(x - w/2, base_seg, w, label="Baseline", color=COLOR_BASE, alpha=0.85)
    ax2.bar(x + w/2, ours_seg, w, label="Proposed", color=COLOR_OURS, alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(segments, fontsize=11)
    ax2.set_ylabel("Mean Latency (ms)", fontsize=12)
    ax2.set_title("Segment-wise Latency Breakdown", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=11)

    fig.suptitle("Performance Comparison: Baseline vs Proposed", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] 저장: {out_path}")


def print_summary(base_rows, ours_rows):
    bt = [r["total_ms"] for r in base_rows]
    ot = [r["total_ms"] for r in ours_rows]

    print("\n" + "="*65)
    print("  Final Comparison: Baseline vs Proposed")
    print("="*65)
    print(f"  {'지표':<25} {'Baseline':>12} {'Proposed':>12} {'향상':>10}")
    print("-"*65)

    metrics = [
        ("Mean Latency (ms)",  np.mean(bt),           np.mean(ot),           "x↓"),
        ("P50 Latency (ms)",   np.percentile(bt, 50), np.percentile(ot, 50), "x↓"),
        ("P90 Latency (ms)",   np.percentile(bt, 90), np.percentile(ot, 90), "x↓"),
        ("P99 Latency (ms)",   np.percentile(bt, 99), np.percentile(ot, 99), "x↓"),
        ("Std Dev (ms)",       np.std(bt),            np.std(ot),            "x↓"),
        ("FPS",                1000/np.mean(bt),      1000/np.mean(ot),      "x↑"),
    ]

    for name, bv, ov, direction in metrics:
        if "↓" in direction:
            ratio = bv / max(ov, 0.001)
        else:
            ratio = ov / max(bv, 0.001)
        print(f"  {name:<25} {bv:>12.2f} {ov:>12.2f} {ratio:>8.1f}{direction}")

    if "frame_type" in ours_rows[0]:
        i_cnt = sum(1 for r in ours_rows if r.get("frame_type") == "I")
        p_cnt = len(ours_rows) - i_cnt
        print("-"*65)
        print(f"  {'I-frame 비율':<25} {'100.0%':>12} {i_cnt/len(ours_rows)*100:>11.1f}%")
        print(f"  {'P-frame 비율':<25} {'0.0%':>12} {p_cnt/len(ours_rows)*100:>11.1f}%")
        i_infers = [r["infer_ms"] for r in ours_rows if r.get("frame_type") == "I"]
        p_infers = [r["infer_ms"] for r in ours_rows if r.get("frame_type") == "P"]
        if i_infers and p_infers:
            speedup = np.mean(i_infers) / max(np.mean(p_infers), 0.001)
            print(f"  {'P-frame 추론 속도 향상':<25} {'-':>12} {speedup:>10.0f}x↑")
    print("="*65)


if __name__ == "__main__":
    base_csv = sys.argv[1] if len(sys.argv) > 1 else "baseline_log.csv"
    ours_csv = sys.argv[2] if len(sys.argv) > 2 else "latency_log.csv"
    out_dir  = Path(sys.argv[3]) if len(sys.argv) > 3 else Path(".")

    print(f"[Compare] 베이스라인: {base_csv}")
    print(f"[Compare] 우리 방법: {ours_csv}")

    base_rows = load_csv(base_csv, has_frame_type=False)
    ours_rows = load_csv(ours_csv, has_frame_type=True)

    print(f"[Compare] 베이스라인 {len(base_rows)}프레임 / 우리 {len(ours_rows)}프레임 분석 중...\n")

    plot_distribution(   base_rows, ours_rows, str(out_dir / "01_tail_distribution.png"))
    plot_percentile_bar( base_rows, ours_rows, str(out_dir / "02_percentile_bar.png"))
    plot_timeline_compare(base_rows, ours_rows, str(out_dir / "03_timeline_compare.png"))
    plot_fps_and_segments(base_rows, ours_rows, str(out_dir / "04_fps_segments.png"))
    print_summary(base_rows, ours_rows)

    print(f"\n[Compare] 완료! 그래프 4개 저장됨:")
    print(f"  -> {out_dir}/01_tail_distribution.png")
    print(f"  -> {out_dir}/02_percentile_bar.png")
    print(f"  -> {out_dir}/03_timeline_compare.png")
    print(f"  -> {out_dir}/04_fps_segments.png")