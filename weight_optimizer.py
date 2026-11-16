"""w1, w2, threshold를 바꿔가며 최적 SCS 파라미터를 찾는 grid search 실험 코드."""

import cv2
import torch
import numpy as np
import time
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def load_model(device):
    print("[Optimizer] 모델 로딩 중...")
    try:
        model = torch.hub.load("ultralytics/yolov5", "yolov5s", verbose=False)
    except Exception:
        model = torch.hub.load("ultralytics/yolov5", "yolov5s", verbose=False)
    model.to(device)
    model.conf = 0.4
    return model


def run_experiment(source, model, device, w1, w2, threshold,
                   max_frames=200, max_p_streak=8):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"영상 열기 실패: {source}")

    prev_gray = None
    p_streak = 0
    last_conf = 1.0

    i_count = 0
    p_count = 0
    total_latency = []
    infer_latency_i = []

    frame_idx = 0
    while frame_idx < max_frames:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        t_start = time.perf_counter()

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # SCS score = w1*pixel_change + w2*(1-conf)
        if prev_gray is None:
            scene_score = 1.0
        else:
            diff = cv2.absdiff(gray, prev_gray)
            pixel_change = np.count_nonzero(diff > 15) / diff.size
            scene_score = float(np.clip(
                w1 * pixel_change + w2 * (1.0 - last_conf), 0.0, 1.0
            ))
        prev_gray = gray

        force_i = p_streak >= max_p_streak
        if force_i or scene_score >= threshold:
            if device == "cuda":
                torch.cuda.synchronize()
            t_infer = time.perf_counter()
            results = model(frame_rgb)
            if device == "cuda":
                torch.cuda.synchronize()
            infer_ms = (time.perf_counter() - t_infer) * 1000.0

            dets = results.xyxy[0].cpu().numpy()
            confs = dets[:, 4].tolist() if len(dets) else [1.0]
            last_conf = float(np.mean(confs)) if confs else 1.0

            p_streak = 0
            i_count += 1
            infer_latency_i.append(infer_ms)
        else:
            infer_ms = 0.0
            p_streak += 1
            p_count += 1

        total_ms = (time.perf_counter() - t_start) * 1000.0
        total_latency.append(total_ms)
        frame_idx += 1

    cap.release()

    total = i_count + p_count
    if total == 0:
        return None

    return {
        "w1": w1, "w2": w2, "threshold": threshold,
        "i_ratio": i_count / total * 100,
        "p_ratio": p_count / total * 100,
        "mean_ms": float(np.mean(total_latency)),
        "p99_ms": float(np.percentile(total_latency, 99)),
        "fps": 1000.0 / float(np.mean(total_latency)),
        "i_mean_infer": float(np.mean(infer_latency_i)) if infer_latency_i else 0,
        "i_count": i_count,
        "p_count": p_count,
    }


def grid_search(source, model, device, max_frames):
    configs = [
        (0.3, 0.7, 0.30), (0.3, 0.7, 0.35), (0.3, 0.7, 0.40),
        (0.5, 0.5, 0.30), (0.5, 0.5, 0.35), (0.5, 0.5, 0.40),
        (0.6, 0.4, 0.30), (0.6, 0.4, 0.35), (0.6, 0.4, 0.40),
        (0.7, 0.3, 0.30), (0.7, 0.3, 0.35), (0.7, 0.3, 0.40),
        (0.8, 0.2, 0.30), (0.8, 0.2, 0.35), (0.8, 0.2, 0.40),
    ]

    results = []
    total = len(configs)
    for i, (w1, w2, thr) in enumerate(configs):
        print(f"[{i+1:2d}/{total}] w1={w1}, w2={w2}, threshold={thr} ...", end=" ")
        res = run_experiment(source, model, device, w1, w2, thr, max_frames)
        if res:
            results.append(res)
            print(f"FPS={res['fps']:.1f}, I-frame={res['i_ratio']:.1f}%")
        else:
            print("실패")

    results.sort(key=lambda x: -x["fps"])
    return results


def plot_results(results, out_dir):
    fixed = [r for r in results if abs(r["w1"] - 0.6) < 0.01]
    fixed.sort(key=lambda x: x["threshold"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    thresholds = [r["threshold"] for r in fixed]
    fps_vals   = [r["fps"] for r in fixed]
    i_ratios   = [r["i_ratio"] for r in fixed]

    ax1 = axes[0]
    ax2 = ax1.twinx()
    ax1.plot(thresholds, fps_vals, "o-", color="#3B82F6", lw=2.5, ms=9, label="FPS")
    ax2.plot(thresholds, i_ratios, "s--", color="#EF4444", lw=2.5, ms=9, label="I-frame %")
    ax1.set_xlabel("Threshold", fontsize=12)
    ax1.set_ylabel("FPS", fontsize=12, color="#3B82F6")
    ax2.set_ylabel("I-frame 비율 (%)", fontsize=12, color="#EF4444")
    ax1.tick_params(axis="y", labelcolor="#3B82F6")
    ax2.tick_params(axis="y", labelcolor="#EF4444")
    for thr, fps in zip(thresholds, fps_vals):
        ax1.annotate(f"{fps:.0f}", (thr, fps), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=10)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, fontsize=10)
    ax1.set_title("Threshold별 FPS vs I-frame 비율 (w1=0.6, w2=0.4)", fontsize=12, fontweight="bold")

    fixed2 = [r for r in results if abs(r["threshold"] - 0.35) < 0.01]
    fixed2.sort(key=lambda x: x["w1"])
    w1_vals  = [r["w1"] for r in fixed2]
    fps_vals2 = [r["fps"] for r in fixed2]
    ir_vals2  = [r["i_ratio"] for r in fixed2]

    x = np.arange(len(w1_vals))
    w = 0.35
    ax = axes[1]
    bars1 = ax.bar(x - w/2, fps_vals2, w, label="FPS", color="#3B82F6", alpha=0.85)
    ax3 = ax.twinx()
    ax3.bar(x + w/2, ir_vals2, w, label="I-frame %", color="#EF4444", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"w1={v}" for v in w1_vals], fontsize=10)
    ax.set_ylabel("FPS", fontsize=12, color="#3B82F6")
    ax3.set_ylabel("I-frame 비율 (%)", fontsize=12, color="#EF4444")
    ax.tick_params(axis="y", labelcolor="#3B82F6")
    ax3.tick_params(axis="y", labelcolor="#EF4444")
    for bar in bars1:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                f"{bar.get_height():.0f}", ha="center", fontsize=9)
    ax.set_title("w1 가중치별 FPS vs I-frame 비율 (threshold=0.35)", fontsize=12, fontweight="bold")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax3.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2, fontsize=10)

    plt.tight_layout()
    out = str(Path(out_dir) / "05_weight_optimization.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Optimizer] 저장: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0")
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--output", type=str, default=".")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Optimizer] 디바이스: {device.upper()}")

    model = load_model(device)

    print(f"\n[Optimizer] Grid Search 시작 ({args.frames}프레임 × 15조합)\n")
    results = grid_search(source, model, device, args.frames)

    print("\n" + "="*68)
    print("  Grid Search 결과 — 전체 (FPS 기준 정렬)")
    print("="*68)
    print(f"  {'w1':>4} {'w2':>4} {'thr':>5} {'FPS':>8} {'I%':>7} {'P%':>7} {'P99':>8}")
    print("-"*68)
    for r in results:
        print(f"  {r['w1']:>4.1f} {r['w2']:>4.1f} {r['threshold']:>5.2f} "
              f"{r['fps']:>8.1f} {r['i_ratio']:>6.1f}% {r['p_ratio']:>6.1f}% "
              f"{r['p99_ms']:>6.1f}ms")

    best = results[0]
    curr = next((r for r in results if abs(r["w1"]-0.6)<0.01
                 and abs(r["w2"]-0.4)<0.01 and abs(r["threshold"]-0.35)<0.01), None)

    print("="*68)
    print(f"\n  최적 조합: w1={best['w1']}, w2={best['w2']}, threshold={best['threshold']}")
    print(f"  → FPS={best['fps']:.1f}, I-frame={best['i_ratio']:.1f}%, P99={best['p99_ms']:.1f}ms")
    if curr:
        imp = (best['fps'] - curr['fps']) / curr['fps'] * 100
        print(f"\n  현재 설정(w1=0.6, w2=0.4, thr=0.35): FPS={curr['fps']:.1f}")
        print(f"  최적화 시 FPS 추가 향상: {imp:+.1f}%")
    print("="*68)

    plot_results(results, args.output)
    print("\n[Optimizer] 완료!")