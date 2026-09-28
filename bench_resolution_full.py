"""360p/1080p/4K 해상도별로 Baseline vs I/P-frame 스케줄러를 동일 조건에서 측정한다."""
import csv
import gc
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core import (
    FrameLatency,
    FrameProfiler,
    SceneChangeDetector,
    CodecInspiredScheduler,
)

VIDEO_PATH = r"C:\Users\USER\OneDrive\Desktop\proj2\test_4k.mp4"
MODEL_PATH = "yolov5s.pt"
CONF_THRESHOLD = 0.4
N_FRAMES = 500

RESOLUTIONS = [
    ("360p (640x360)",   (640, 360),   "360"),
    ("1080p (1920x1080)", (1920, 1080), "1080"),
    ("4K (3840x2160)",    (3840, 2160), "2160"),
]

ALERT_THRESHOLD_MS = 50.0


def load_model():
    print("[Model] loading...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = torch.hub.load("ultralytics/yolov5", "custom",
                                path=MODEL_PATH, verbose=False)
    except Exception:
        model = torch.hub.load("ultralytics/yolov5", "yolov5s", verbose=False)
    model.to(device)
    model.conf = CONF_THRESHOLD
    print(f"[Model] device={device}")
    return model, device


def load_frames(resize, n=N_FRAMES):
    cap = cv2.VideoCapture(VIDEO_PATH)
    frames = []
    while len(frames) < n:
        ret, f = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        frames.append(cv2.resize(f, resize))
    cap.release()
    return frames


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def run_baseline(frames, model, device):
    log = []
    for idx, frame_bgr in enumerate(frames):
        lat = FrameLatency(frame_idx=idx)

        t = time.perf_counter()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        lat.preprocess_ms = (time.perf_counter() - t) * 1000.0

        t = time.perf_counter()
        if device == "cuda":
            _ = torch.from_numpy(frame_rgb.copy()).to(device)
        lat.htod_ms = (time.perf_counter() - t) * 1000.0

        sync(device)
        t = time.perf_counter()
        results = model(frame_rgb)
        sync(device)
        lat.infer_ms = (time.perf_counter() - t) * 1000.0

        t = time.perf_counter()
        _ = results.xyxy[0].cpu().numpy()
        lat.dtoh_ms = (time.perf_counter() - t) * 1000.0

        lat.compute_total()
        log.append(lat)
    return log


def run_scheduler(frames, model, device):
    detector = SceneChangeDetector(threshold=0.60)
    scheduler = CodecInspiredScheduler(max_p_streak=4)
    profiler = FrameProfiler()
    log = []

    for idx, frame_bgr in enumerate(frames):
        lat = FrameLatency(frame_idx=idx)

        profiler.start("pre", gpu_sync=False)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        lat.preprocess_ms = profiler.end("pre", gpu_sync=False)

        scene_score = detector.compute_score(frame_bgr, mean_conf=scheduler.get_last_conf())
        lat.scene_score = round(scene_score, 4)

        alert = False
        for past in reversed(log[-10:]):
            if past.frame_type == "I":
                if past.infer_ms > ALERT_THRESHOLD_MS:
                    alert = True
                break
        lat.alert = alert

        frame_type = scheduler.decide_frame_type(scene_score, alert)
        lat.frame_type = frame_type

        profiler.start("htod", gpu_sync=False)
        if device == "cuda":
            _ = torch.from_numpy(frame_rgb.copy()).to(device)
        lat.htod_ms = profiler.end("htod", gpu_sync=False)

        if frame_type == "I":
            profiler.start("infer", gpu_sync=True)
            results = model(frame_rgb)
            dets = results.xyxy[0].cpu().numpy()
            boxes = dets[:, :4].tolist() if len(dets) else []
            confs = dets[:, 4].tolist() if len(dets) else [1.0]
            mean_conf = float(np.mean(confs)) if confs else 1.0
            lat.infer_ms = profiler.end("infer", gpu_sync=True)
            scheduler.update_last_result(boxes, mean_conf)
        else:
            profiler.start("infer", gpu_sync=False)
            boxes = scheduler.get_last_boxes()
            lat.infer_ms = profiler.end("infer", gpu_sync=False)

        profiler.start("dtoh", gpu_sync=False)
        _ = boxes
        lat.dtoh_ms = profiler.end("dtoh", gpu_sync=False)

        lat.compute_total()
        log.append(lat)
    return log


def save_baseline_csv(log, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame_idx", "preprocess_ms", "htod_ms", "infer_ms", "dtoh_ms", "total_ms"])
        for lat in log:
            w.writerow([lat.frame_idx, round(lat.preprocess_ms, 3), round(lat.htod_ms, 3),
                        round(lat.infer_ms, 3), round(lat.dtoh_ms, 3), round(lat.total_ms, 3)])


def save_scheduler_csv(log, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame_idx", "frame_type", "scene_score", "alert",
                    "preprocess_ms", "htod_ms", "infer_ms", "dtoh_ms", "total_ms"])
        for lat in log:
            w.writerow([lat.frame_idx, lat.frame_type, lat.scene_score, lat.alert,
                        round(lat.preprocess_ms, 3), round(lat.htod_ms, 3),
                        round(lat.infer_ms, 3), round(lat.dtoh_ms, 3), round(lat.total_ms, 3)])


def summarize(log):
    totals = np.array([l.total_ms for l in log])
    return {
        "mean": totals.mean(),
        "p50": np.percentile(totals, 50),
        "p90": np.percentile(totals, 90),
        "p99": np.percentile(totals, 99),
        "fps": 1000 / totals.mean(),
    }


def main():
    model, device = load_model()
    results = []

    for label, resize, suffix in RESOLUTIONS:
        print(f"\n{'='*60}\n[{label}] loading {N_FRAMES} frames...")
        frames = load_frames(resize)
        print(f"[{label}] {len(frames)} frames loaded, running baseline...")

        base_log = run_baseline(frames, model, device)
        save_baseline_csv(base_log, f"baseline_{suffix}.csv")
        base_stats = summarize(base_log)
        print(f"[{label}] Baseline  mean={base_stats['mean']:.2f}ms  fps={base_stats['fps']:.1f}")

        print(f"[{label}] running I/P scheduler...")
        sched_log = run_scheduler(frames, model, device)
        save_scheduler_csv(sched_log, f"latency_{suffix}.csv")
        sched_stats = summarize(sched_log)
        i_count = sum(1 for l in sched_log if l.frame_type == "I")
        print(f"[{label}] Scheduler mean={sched_stats['mean']:.2f}ms  fps={sched_stats['fps']:.1f}  "
              f"I-ratio={i_count/len(sched_log)*100:.1f}%")

        results.append({
            "label": label, "suffix": suffix,
            "base": base_stats, "sched": sched_stats,
            "i_ratio": i_count / len(sched_log) * 100,
        })

        del frames, base_log, sched_log
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    plot_results(results)
    print_summary_table(results)


def plot_results(results):
    plt.rcParams.update({
        "font.family": ["Malgun Gothic", "DejaVu Sans"],
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

    labels = [r["label"] for r in results]
    base_means = [r["base"]["mean"] for r in results]
    sched_means = [r["sched"]["mean"] for r in results]
    base_fps = [r["base"]["fps"] for r in results]
    sched_fps = [r["sched"]["fps"] for r in results]

    x = np.arange(len(labels))
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    ax = axes[0]
    bars1 = ax.bar(x - w/2, base_means, w, label="Baseline (All I-frame)", color=COLOR_BASE, alpha=0.85)
    bars2 = ax.bar(x + w/2, sched_means, w, label="I/P-frame Scheduler", color=COLOR_OURS, alpha=0.85)
    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"{bar.get_height():.1f}ms", ha="center", va="bottom", fontsize=10, fontweight="bold")
    for i in range(len(labels)):
        ratio = base_means[i] / max(sched_means[i], 0.01)
        ax.text(x[i], max(base_means[i], sched_means[i]) * 1.12, f"{ratio:.1f}x",
                ha="center", fontsize=12, color="darkgreen", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Mean Latency (ms)")
    ax.set_title("Mean Latency by Resolution")
    ax.set_ylim(0, max(base_means) * 1.35)
    ax.legend(fontsize=10)

    ax2 = axes[1]
    bars3 = ax2.bar(x - w/2, base_fps, w, label="Baseline (All I-frame)", color=COLOR_BASE, alpha=0.85)
    bars4 = ax2.bar(x + w/2, sched_fps, w, label="I/P-frame Scheduler", color=COLOR_OURS, alpha=0.85)
    for bar in list(bars3) + list(bars4):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    for i in range(len(labels)):
        ratio = sched_fps[i] / max(base_fps[i], 0.01)
        ax2.text(x[i], max(base_fps[i], sched_fps[i]) * 1.12, f"{ratio:.1f}x",
                 ha="center", fontsize=12, color="darkgreen", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_ylabel("FPS")
    ax2.set_title("Throughput (FPS) by Resolution")
    ax2.set_ylim(0, max(sched_fps) * 1.35)
    ax2.legend(fontsize=10)

    fig.suptitle("Baseline vs I/P-frame Scheduler across Resolutions (same video, same model)",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()

    out_path = Path("11_resolution_full_compare.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[Plot] saved -> {out_path.resolve()}")


def print_summary_table(results):
    print("\n" + "="*90)
    print(f"  {'Resolution':<22} {'Baseline(ms)':>13} {'Sched(ms)':>11} {'Speedup':>9} "
          f"{'Base FPS':>9} {'Sched FPS':>10} {'I-ratio':>8}")
    print("-"*90)
    for r in results:
        speedup = r["base"]["mean"] / max(r["sched"]["mean"], 0.001)
        print(f"  {r['label']:<22} {r['base']['mean']:>13.2f} {r['sched']['mean']:>11.2f} "
              f"{speedup:>8.2f}x {r['base']['fps']:>9.1f} {r['sched']['fps']:>10.1f} "
              f"{r['i_ratio']:>7.1f}%")
    print("="*90)


if __name__ == "__main__":
    main()
