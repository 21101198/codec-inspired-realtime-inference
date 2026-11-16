"""스마트 팩토리 시나리오: 카메라 N대를 동시에 처리하는 다중 스트림 벤치마크.

test_4k.mp4 하나를 N개의 카메라 소스로 사용(서로 다른 구간에서 시작)하고,
기존 I/P-frame 스케줄러(core.py)를 각 스트림에 독립적으로 적용해 스레드로 동시 실행한다.
GPU 추론 구간만 락으로 직렬화(실제 단일 GPU 컨텍스트를 모사)하고, 전처리/스킵된
P-frame 구간은 락 없이 겹쳐 실행되어 진짜 멀티스트림 상황의 GPU 처리량을 측정한다.

비교 대상 4가지:
  1) Baseline(All I-frame), 단일 스트림
  2) Baseline(All I-frame), N 스트림 동시
  3) I/P-frame Scheduler,  단일 스트림
  4) I/P-frame Scheduler,  N 스트림 동시
"""
import argparse
import csv
import gc
import threading
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
MODEL_PATH = "yolov5s.pt"
CONF_THRESHOLD = 0.4
ALERT_THRESHOLD_MS = 50.0

HERE = Path(__file__).resolve().parent
PLOT_PATH = HERE / "13_multistream_compare.png"
SUMMARY_CSV = HERE / "multistream_bench_summary.csv"


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


def load_camera_frames(cam_id, n_streams, n_frames, resize):
    """test_4k.mp4를 카메라 cam_id 몫의 구간에서부터 읽어 독립된 스트림처럼 흉내낸다."""
    cap = cv2.VideoCapture(VIDEO_PATH)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start = (total // max(n_streams, 1)) * cam_id
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    frames = []
    while len(frames) < n_frames:
        ret, f = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        frames.append(cv2.resize(f, resize))
    cap.release()
    return frames


def run_stream_baseline(cam_id, frames, model, device, gpu_lock, out):
    """Baseline: 매 프레임 전체 추론. GPU를 만지는 구간만 락으로 직렬화."""
    log = []
    for idx, frame_bgr in enumerate(frames):
        lat = FrameLatency(frame_idx=idx)

        t = time.perf_counter()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        lat.preprocess_ms = (time.perf_counter() - t) * 1000.0

        with gpu_lock:
            t = time.perf_counter()
            if device == "cuda":
                _ = torch.from_numpy(frame_rgb.copy()).to(device)
            lat.htod_ms = (time.perf_counter() - t) * 1000.0

            if device == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter()
            results = model(frame_rgb)
            if device == "cuda":
                torch.cuda.synchronize()
            lat.infer_ms = (time.perf_counter() - t) * 1000.0

            t = time.perf_counter()
            _ = results.xyxy[0].cpu().numpy()
            lat.dtoh_ms = (time.perf_counter() - t) * 1000.0

        lat.compute_total()
        log.append(lat)
    out[cam_id] = log


def run_stream_scheduler(cam_id, frames, model, device, gpu_lock, out):
    """I/P-frame 스케줄러: I-frame만 GPU 락을 잡는다. P-frame은 GPU를 아예 건드리지 않아
    다른 스트림과 완전히 겹쳐 실행된다."""
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

        if frame_type == "I":
            with gpu_lock:
                profiler.start("htod", gpu_sync=False)
                if device == "cuda":
                    _ = torch.from_numpy(frame_rgb.copy()).to(device)
                lat.htod_ms = profiler.end("htod", gpu_sync=False)

                profiler.start("infer", gpu_sync=True)
                results = model(frame_rgb)
                dets = results.xyxy[0].cpu().numpy()
                boxes = dets[:, :4].tolist() if len(dets) else []
                confs = dets[:, 4].tolist() if len(dets) else [1.0]
                mean_conf = float(np.mean(confs)) if confs else 1.0
                lat.infer_ms = profiler.end("infer", gpu_sync=True)

                profiler.start("dtoh", gpu_sync=False)
                _ = boxes
                lat.dtoh_ms = profiler.end("dtoh", gpu_sync=False)
            scheduler.update_last_result(boxes, mean_conf)
        else:
            profiler.start("htod", gpu_sync=False)
            lat.htod_ms = profiler.end("htod", gpu_sync=False)

            profiler.start("infer", gpu_sync=False)
            boxes = scheduler.get_last_boxes()
            lat.infer_ms = profiler.end("infer", gpu_sync=False)

            profiler.start("dtoh", gpu_sync=False)
            _ = boxes
            lat.dtoh_ms = profiler.end("dtoh", gpu_sync=False)

        lat.compute_total()
        log.append(lat)
    out[cam_id] = log


def run_parallel(target, cam_frames, model, device, gpu_lock):
    """cam_frames의 각 스트림을 별도 스레드로 동시에 돌리고, 전체 wall time을 측정한다."""
    out = {}
    threads = [
        threading.Thread(target=target, args=(i, frames, model, device, gpu_lock, out))
        for i, frames in enumerate(cam_frames)
    ]
    t0 = time.perf_counter()
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    wall_s = time.perf_counter() - t0
    logs = [out[i] for i in range(len(cam_frames))]
    return logs, wall_s


def save_stream_csvs(logs, prefix):
    for cam_id, log in enumerate(logs):
        path = HERE / f"{prefix}_cam{cam_id}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["frame_idx", "frame_type", "scene_score", "alert",
                        "preprocess_ms", "htod_ms", "infer_ms", "dtoh_ms", "total_ms"])
            for lat in log:
                w.writerow([lat.frame_idx, lat.frame_type, lat.scene_score, lat.alert,
                            round(lat.preprocess_ms, 3), round(lat.htod_ms, 3),
                            round(lat.infer_ms, 3), round(lat.dtoh_ms, 3), round(lat.total_ms, 3)])


def i_ratio(logs):
    total = sum(len(l) for l in logs)
    i_cnt = sum(1 for l in logs for lat in l if lat.frame_type == "I")
    return i_cnt / total * 100 if total else 0.0


def main():
    ap = argparse.ArgumentParser(description="스마트 팩토리: 다중 카메라 스트림 I/P-frame 스케줄러 벤치마크")
    ap.add_argument("--streams", type=int, default=4, help="동시 처리할 카메라 스트림 수")
    ap.add_argument("--frames", type=int, default=200, help="스트림당 처리 프레임 수")
    ap.add_argument("--width", type=int, default=1280, help="카메라 리사이즈 폭")
    ap.add_argument("--height", type=int, default=720, help="카메라 리사이즈 높이")
    args = ap.parse_args()

    n = args.streams
    resize = (args.width, args.height)

    model, device = load_model()

    print(f"\n[Video] {VIDEO_PATH} 에서 카메라 {n}대 분량({args.frames}프레임씩, "
          f"{resize[0]}x{resize[1]}) 로딩...")
    cam_frames = [load_camera_frames(i, n, args.frames, resize) for i in range(n)]
    print(f"[Video] 로딩 완료: 스트림당 {len(cam_frames[0])}프레임")

    print("[Model] GPU warmup...")
    warm_rgb = cv2.cvtColor(cam_frames[0][0], cv2.COLOR_BGR2RGB)
    for _ in range(5):
        model(warm_rgb)
    if device == "cuda":
        torch.cuda.synchronize()

    gpu_lock = threading.Lock()

    print(f"\n{'='*60}\n[1/4] 단일 스트림 Baseline (All I-frame)...")
    out = {}
    t0 = time.perf_counter()
    run_stream_baseline(0, cam_frames[0], model, device, gpu_lock, out)
    wall_base_1 = time.perf_counter() - t0
    base1_log = out[0]
    print(f"  wall={wall_base_1:.2f}s  fps={args.frames/wall_base_1:.1f}")

    print(f"\n[2/4] {n}개 스트림 동시 Baseline (All I-frame)...")
    base4_logs, wall_base_4 = run_parallel(run_stream_baseline, cam_frames, model, device, gpu_lock)
    print(f"  wall={wall_base_4:.2f}s  aggregate fps={n*args.frames/wall_base_4:.1f}  "
          f"per-cam fps={n*args.frames/wall_base_4/n:.1f}")

    print("\n[3/4] 단일 스트림 I/P-frame 스케줄러...")
    out = {}
    t0 = time.perf_counter()
    run_stream_scheduler(0, cam_frames[0], model, device, gpu_lock, out)
    wall_sched_1 = time.perf_counter() - t0
    sched1_log = out[0]
    print(f"  wall={wall_sched_1:.2f}s  fps={args.frames/wall_sched_1:.1f}  "
          f"I-ratio={i_ratio([sched1_log]):.1f}%")

    print(f"\n[4/4] {n}개 스트림 동시 I/P-frame 스케줄러...")
    sched4_logs, wall_sched_4 = run_parallel(run_stream_scheduler, cam_frames, model, device, gpu_lock)
    print(f"  wall={wall_sched_4:.2f}s  aggregate fps={n*args.frames/wall_sched_4:.1f}  "
          f"per-cam fps={n*args.frames/wall_sched_4/n:.1f}  I-ratio={i_ratio(sched4_logs):.1f}%")

    save_stream_csvs([base1_log], "multistream_baseline_1stream")
    save_stream_csvs(base4_logs, "multistream_baseline_4stream")
    save_stream_csvs([sched1_log], "multistream_scheduler_1stream")
    save_stream_csvs(sched4_logs, "multistream_scheduler_4stream")

    results = {
        "streams": n,
        "frames_per_stream": args.frames,
        "base_1": {"wall_s": wall_base_1, "fps_agg": args.frames / wall_base_1,
                    "fps_per_cam": args.frames / wall_base_1},
        "base_n": {"wall_s": wall_base_4, "fps_agg": n * args.frames / wall_base_4,
                    "fps_per_cam": n * args.frames / wall_base_4 / n},
        "sched_1": {"wall_s": wall_sched_1, "fps_agg": args.frames / wall_sched_1,
                     "fps_per_cam": args.frames / wall_sched_1, "i_ratio": i_ratio([sched1_log])},
        "sched_n": {"wall_s": wall_sched_4, "fps_agg": n * args.frames / wall_sched_4,
                     "fps_per_cam": n * args.frames / wall_sched_4 / n, "i_ratio": i_ratio(sched4_logs)},
    }
    results["base_scaling"] = results["base_n"]["fps_agg"] / results["base_1"]["fps_agg"]
    results["sched_scaling"] = results["sched_n"]["fps_agg"] / results["sched_1"]["fps_agg"]
    results["base_efficiency"] = results["base_scaling"] / n * 100
    results["sched_efficiency"] = results["sched_scaling"] / n * 100
    results["gpu_throughput_gain"] = results["sched_n"]["fps_agg"] / results["base_n"]["fps_agg"]

    save_summary_csv(results)
    plot_results(results)
    print_summary_table(results)

    del cam_frames, base1_log, base4_logs, sched1_log, sched4_logs
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


def save_summary_csv(r):
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["config", "streams", "frames_per_stream", "wall_s",
                    "fps_aggregate", "fps_per_camera", "i_ratio_pct"])
        w.writerow(["baseline_1stream", 1, r["frames_per_stream"], round(r["base_1"]["wall_s"], 3),
                    round(r["base_1"]["fps_agg"], 2), round(r["base_1"]["fps_per_cam"], 2), ""])
        w.writerow(["baseline_Nstream", r["streams"], r["frames_per_stream"], round(r["base_n"]["wall_s"], 3),
                    round(r["base_n"]["fps_agg"], 2), round(r["base_n"]["fps_per_cam"], 2), ""])
        w.writerow(["scheduler_1stream", 1, r["frames_per_stream"], round(r["sched_1"]["wall_s"], 3),
                    round(r["sched_1"]["fps_agg"], 2), round(r["sched_1"]["fps_per_cam"], 2),
                    round(r["sched_1"]["i_ratio"], 1)])
        w.writerow(["scheduler_Nstream", r["streams"], r["frames_per_stream"], round(r["sched_n"]["wall_s"], 3),
                    round(r["sched_n"]["fps_agg"], 2), round(r["sched_n"]["fps_per_cam"], 2),
                    round(r["sched_n"]["i_ratio"], 1)])
    print(f"\n[CSV] summary saved -> {SUMMARY_CSV.resolve()}")


def plot_results(r):
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
    n = r["streams"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    ax = axes[0]
    labels = ["1 stream", f"{n} streams"]
    base_vals = [r["base_1"]["fps_agg"], r["base_n"]["fps_agg"]]
    sched_vals = [r["sched_1"]["fps_agg"], r["sched_n"]["fps_agg"]]
    x = np.arange(len(labels))
    w = 0.35
    bars1 = ax.bar(x - w/2, base_vals, w, label="Baseline (All I-frame)", color=COLOR_BASE, alpha=0.85)
    bars2 = ax.bar(x + w/2, sched_vals, w, label="I/P-frame Scheduler", color=COLOR_OURS, alpha=0.85)
    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Aggregate GPU Throughput (FPS)")
    ax.set_title("GPU Throughput: 1 stream vs N streams")
    ax.set_ylim(0, max(base_vals + sched_vals) * 1.3)
    ax.legend(fontsize=10)

    ax2 = axes[1]
    labels2 = ["Baseline\n(All I-frame)", "I/P-frame\nScheduler"]
    scaling = [r["base_scaling"], r["sched_scaling"]]
    eff = [r["base_efficiency"], r["sched_efficiency"]]
    x2 = np.arange(len(labels2))
    bars3 = ax2.bar(x2, scaling, 0.5, color=[COLOR_BASE, COLOR_OURS], alpha=0.85)
    ax2.axhline(n, color="darkgreen", linestyle="--", linewidth=1.5,
                label=f"{n} independent GPUs (hypothetical, {n}x)")
    for bar, e in zip(bars3, eff):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f"{bar.get_height():.2f}x\n({e:.0f}% of {n}-GPU)", ha="center", va="bottom",
                 fontsize=10, fontweight="bold")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(labels2, fontsize=10)
    ax2.set_ylabel("Throughput Scaling (x of 1-stream, 1 GPU)")
    ax2.set_title(f"{n}-Stream Scaling on a Single Shared GPU")
    ax2.set_ylim(0, n * 1.35)
    ax2.legend(fontsize=10)

    fig.suptitle(f"Smart Factory: {n}-Camera Concurrent GPU Throughput "
                 f"(Baseline vs I/P-frame Scheduler)", fontsize=15, fontweight="bold")
    plt.tight_layout()

    plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] saved -> {PLOT_PATH.resolve()}")


def print_summary_table(r):
    n = r["streams"]
    print("\n" + "="*90)
    print(f"  Smart Factory {n}-Camera Multi-stream Benchmark")
    print("-"*90)
    print(f"  {'Config':<22} {'Wall(s)':>9} {'Agg FPS':>9} {'Per-cam FPS':>12} {'I-ratio':>9}")
    print(f"  {'Baseline 1-stream':<22} {r['base_1']['wall_s']:>9.2f} {r['base_1']['fps_agg']:>9.1f} "
          f"{r['base_1']['fps_per_cam']:>12.1f} {'-':>9}")
    print(f"  {'Baseline '+str(n)+'-stream':<22} {r['base_n']['wall_s']:>9.2f} {r['base_n']['fps_agg']:>9.1f} "
          f"{r['base_n']['fps_per_cam']:>12.1f} {'-':>9}")
    print(f"  {'Scheduler 1-stream':<22} {r['sched_1']['wall_s']:>9.2f} {r['sched_1']['fps_agg']:>9.1f} "
          f"{r['sched_1']['fps_per_cam']:>12.1f} {r['sched_1']['i_ratio']:>8.1f}%")
    print(f"  {'Scheduler '+str(n)+'-stream':<22} {r['sched_n']['wall_s']:>9.2f} {r['sched_n']['fps_agg']:>9.1f} "
          f"{r['sched_n']['fps_per_cam']:>12.1f} {r['sched_n']['i_ratio']:>8.1f}%")
    print("-"*90)
    print(f"  {n}-stream scaling   : Baseline {r['base_scaling']:.2f}x (ideal {n}x, "
          f"{r['base_efficiency']:.0f}%)  |  Scheduler {r['sched_scaling']:.2f}x "
          f"(ideal {n}x, {r['sched_efficiency']:.0f}%)")
    print(f"  GPU throughput gain at {n} streams (Scheduler vs Baseline): "
          f"{r['gpu_throughput_gain']:.2f}x")
    print("="*90)


if __name__ == "__main__":
    main()
