"""YOLOv5s -> YOLOv8s 모델 교체 비교: 기존 bench_resolution_full.py와 동일 조건
(360p/1080p/4K, Baseline vs I/P-frame 스케줄러)에서 두 모델을 나란히 측정한다."""
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

VIDEO_PATH = "test_4k.mp4"
MODEL_PATH_V5 = "yolov5s.pt"
MODEL_PATH_V8 = "yolov8s.pt"
CONF_THRESHOLD = 0.4
N_FRAMES = 300
ALERT_THRESHOLD_MS = 50.0

RESOLUTIONS = [
    ("360p (640x360)",    (640, 360),   "360"),
    ("1080p (1920x1080)", (1920, 1080), "1080"),
    ("4K (3840x2160)",    (3840, 2160), "2160"),
]

MODELS = ["yolov5", "yolov8"]

HERE = Path(__file__).resolve().parent
PLOT_PATH = HERE / "14_yolov8_vs_yolov5_compare.png"
SUMMARY_CSV = HERE / "yolov8_compare_summary.csv"


def load_model(model_type):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Model] loading {model_type}...")
    if model_type == "yolov8":
        from ultralytics import YOLO
        model = YOLO(MODEL_PATH_V8)
        model.to(device)
    else:
        try:
            model = torch.hub.load("ultralytics/yolov5", "custom",
                                    path=MODEL_PATH_V5, verbose=False)
        except Exception:
            model = torch.hub.load("ultralytics/yolov5", "yolov5s", verbose=False)
        model.to(device)
        model.conf = CONF_THRESHOLD
    print(f"[Model] {model_type} ready on {device}")
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


def infer_full(model, model_type, frame_rgb):
    """모델 종류에 무관하게 (boxes, mean_conf)를 반환. dtoh 시간은 호출부에서 별도로 잰다."""
    if model_type == "yolov8":
        results = model(frame_rgb, verbose=False, conf=CONF_THRESHOLD)
        r = results[0]
        if len(r.boxes):
            boxes_t, confs_t = r.boxes.xyxy, r.boxes.conf
        else:
            boxes_t, confs_t = None, None
        return boxes_t, confs_t

    results = model(frame_rgb)
    dets = results.xyxy[0]
    return dets, None


def to_boxes_confs(model_type, raw):
    boxes_t, confs_t = raw
    if model_type == "yolov8":
        if boxes_t is None:
            return [], [1.0]
        boxes = boxes_t.cpu().numpy().tolist()
        confs = confs_t.cpu().numpy().tolist()
        return boxes, confs
    dets = boxes_t.cpu().numpy()
    boxes = dets[:, :4].tolist() if len(dets) else []
    confs = dets[:, 4].tolist() if len(dets) else [1.0]
    return boxes, confs


def run_baseline(frames, model, model_type, device):
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
        raw = infer_full(model, model_type, frame_rgb)
        sync(device)
        lat.infer_ms = (time.perf_counter() - t) * 1000.0

        t = time.perf_counter()
        _ = to_boxes_confs(model_type, raw)
        lat.dtoh_ms = (time.perf_counter() - t) * 1000.0

        lat.compute_total()
        log.append(lat)
    return log


def run_scheduler(frames, model, model_type, device):
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
            raw = infer_full(model, model_type, frame_rgb)
            boxes, confs = to_boxes_confs(model_type, raw)
            lat.infer_ms = profiler.end("infer", gpu_sync=True)
            mean_conf = float(np.mean(confs)) if confs else 1.0
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
    results = []  # list of dicts: {label, suffix, model_type, base, sched, i_ratio}

    for model_type in MODELS:
        model, device = load_model(model_type)

        for label, resize, suffix in RESOLUTIONS:
            print(f"\n{'='*60}\n[{model_type} | {label}] loading {N_FRAMES} frames...")
            frames = load_frames(resize)

            # warmup (JIT/cuDNN autotune 등 첫 호출 오버헤드 제외)
            warm_rgb = cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB)
            for _ in range(3):
                infer_full(model, model_type, warm_rgb)
            sync(device)

            print(f"[{model_type} | {label}] running baseline...")
            base_log = run_baseline(frames, model, model_type, device)
            save_baseline_csv(base_log, HERE / f"baseline_{suffix}_{model_type}.csv")
            base_stats = summarize(base_log)
            print(f"[{model_type} | {label}] Baseline  mean={base_stats['mean']:.2f}ms  fps={base_stats['fps']:.1f}")

            print(f"[{model_type} | {label}] running I/P scheduler...")
            sched_log = run_scheduler(frames, model, model_type, device)
            save_scheduler_csv(sched_log, HERE / f"latency_{suffix}_{model_type}.csv")
            sched_stats = summarize(sched_log)
            i_count = sum(1 for l in sched_log if l.frame_type == "I")
            print(f"[{model_type} | {label}] Scheduler mean={sched_stats['mean']:.2f}ms  fps={sched_stats['fps']:.1f}  "
                  f"I-ratio={i_count/len(sched_log)*100:.1f}%")

            results.append({
                "model_type": model_type, "label": label, "suffix": suffix,
                "base": base_stats, "sched": sched_stats,
                "i_ratio": i_count / len(sched_log) * 100,
            })

            del frames, base_log, sched_log
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_summary_csv(results)
    plot_results(results)
    print_summary_table(results)


def save_summary_csv(results):
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model_type", "resolution", "base_mean_ms", "base_fps",
                    "sched_mean_ms", "sched_fps", "i_ratio_pct", "sched_speedup_vs_base"])
        for r in results:
            speedup = r["base"]["mean"] / max(r["sched"]["mean"], 0.001)
            w.writerow([r["model_type"], r["label"], round(r["base"]["mean"], 2),
                        round(r["base"]["fps"], 2), round(r["sched"]["mean"], 2),
                        round(r["sched"]["fps"], 2), round(r["i_ratio"], 1), round(speedup, 2)])
    print(f"\n[CSV] summary saved -> {SUMMARY_CSV.resolve()}")


def plot_results(results):
    plt.rcParams.update({
        "font.family": ["Malgun Gothic", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "figure.dpi": 150,
    })
    COLOR_V5 = "#7F8C8D"
    COLOR_V8 = "#8E44AD"

    labels = [suffix for (_, _, suffix) in RESOLUTIONS]
    res_labels = [lbl for (lbl, _, _) in RESOLUTIONS]

    def get(model_type, mode, field):
        return [next(r[mode][field] for r in results
                     if r["model_type"] == model_type and r["suffix"] == s)
                for s in labels]

    v5_base_ms  = get("yolov5", "base", "mean")
    v8_base_ms  = get("yolov8", "base", "mean")
    v5_sched_ms = get("yolov5", "sched", "mean")
    v8_sched_ms = get("yolov8", "sched", "mean")

    v5_base_fps  = get("yolov5", "base", "fps")
    v8_base_fps  = get("yolov8", "base", "fps")
    v5_sched_fps = get("yolov5", "sched", "fps")
    v8_sched_fps = get("yolov8", "sched", "fps")

    x = np.arange(len(labels))
    w = 0.35

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    def bar_panel(ax, v5_vals, v8_vals, ylabel, title, unit_fmt):
        b1 = ax.bar(x - w/2, v5_vals, w, label="YOLOv5s", color=COLOR_V5, alpha=0.85)
        b2 = ax.bar(x + w/2, v8_vals, w, label="YOLOv8s", color=COLOR_V8, alpha=0.85)
        for bar in list(b1) + list(b2):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    unit_fmt.format(bar.get_height()), ha="center", va="bottom",
                    fontsize=9, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(res_labels, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(0, max(v5_vals + v8_vals) * 1.3)
        ax.legend(fontsize=9)

    bar_panel(axes[0, 0], v5_base_ms, v8_base_ms, "Mean Latency (ms)",
              "Baseline (All I-frame): Mean Latency", "{:.1f}ms")
    bar_panel(axes[0, 1], v5_sched_ms, v8_sched_ms, "Mean Latency (ms)",
              "I/P-frame Scheduler: Mean Latency", "{:.1f}ms")
    bar_panel(axes[1, 0], v5_base_fps, v8_base_fps, "FPS",
              "Baseline (All I-frame): Throughput", "{:.1f}")
    bar_panel(axes[1, 1], v5_sched_fps, v8_sched_fps, "FPS",
              "I/P-frame Scheduler: Throughput", "{:.1f}")

    fig.suptitle("YOLOv5s vs YOLOv8s across Resolutions (same video, same I/P-frame scheduler)",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()

    plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] saved -> {PLOT_PATH.resolve()}")


def print_summary_table(results):
    print("\n" + "="*100)
    print(f"  {'Model':<8} {'Resolution':<20} {'Base(ms)':>9} {'Base FPS':>9} "
          f"{'Sched(ms)':>10} {'Sched FPS':>10} {'I-ratio':>8} {'Speedup':>9}")
    print("-"*100)
    for r in results:
        speedup = r["base"]["mean"] / max(r["sched"]["mean"], 0.001)
        print(f"  {r['model_type']:<8} {r['label']:<20} {r['base']['mean']:>9.2f} "
              f"{r['base']['fps']:>9.1f} {r['sched']['mean']:>10.2f} {r['sched']['fps']:>10.1f} "
              f"{r['i_ratio']:>7.1f}% {speedup:>8.2f}x")
    print("-"*100)

    for suffix_label in [lbl for (lbl, _, _) in RESOLUTIONS]:
        v5 = next(r for r in results if r["model_type"] == "yolov5" and r["label"] == suffix_label)
        v8 = next(r for r in results if r["model_type"] == "yolov8" and r["label"] == suffix_label)
        base_ratio = v5["base"]["fps"] / max(v8["base"]["fps"], 0.001)
        sched_ratio = v5["sched"]["fps"] / max(v8["sched"]["fps"], 0.001)
        print(f"  [{suffix_label}] YOLOv5s/YOLOv8s FPS ratio -> Baseline: {base_ratio:.2f}x  "
              f"Scheduler: {sched_ratio:.2f}x  (>1 means YOLOv5s faster)")
    print("="*100)


if __name__ == "__main__":
    main()
