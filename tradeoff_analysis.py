"""max_p_streak을 변화시키며 FPS vs IoU 관계를 측정하는 Speed-Accuracy Tradeoff 분석."""

import cv2
import torch
import numpy as np
import time
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = area1 + area2 - inter
    return inter/union if union > 0 else 0.0


def match_boxes(gt_boxes, pred_boxes, iou_threshold=0.5):
    if len(gt_boxes) == 0 and len(pred_boxes) == 0:
        return 1.0
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return 0.0
    matched = sum(
        1 for gt in gt_boxes
        if max((compute_iou(gt, p) for p in pred_boxes), default=0) >= iou_threshold
    )
    return matched / max(len(gt_boxes), len(pred_boxes))


def run_single(source, model, device, max_p_streak, max_frames=200,
               threshold=0.35, conf=0.4):
    cap = cv2.VideoCapture(source)
    model.conf = conf

    prev_gray = None
    p_streak = 0
    last_boxes = []
    last_conf_mean = 1.0

    latencies = []
    iou_scores = []
    i_count = 0
    p_count = 0

    frame_idx = 0
    while frame_idx < max_frames:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        t_start = time.perf_counter()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        if prev_gray is None:
            scene_score = 1.0
        else:
            diff = cv2.absdiff(gray, prev_gray)
            pixel_change = np.count_nonzero(diff > 15) / diff.size
            scene_score = float(np.clip(
                0.6 * pixel_change + 0.4 * (1.0 - last_conf_mean), 0, 1
            ))
        prev_gray = gray

        if device == "cuda":
            torch.cuda.synchronize()
        results_gt = model(frame_rgb)
        if device == "cuda":
            torch.cuda.synchronize()

        dets_gt = results_gt.xyxy[0].cpu().numpy()
        gt_boxes = dets_gt[:, :4].tolist() if len(dets_gt) else []
        gt_confs = dets_gt[:, 4].tolist() if len(dets_gt) else [1.0]
        last_conf_mean = float(np.mean(gt_confs)) if gt_confs else 1.0

        force_i = p_streak >= max_p_streak
        if force_i or scene_score >= threshold:
            pred_boxes = gt_boxes
            last_boxes = gt_boxes
            p_streak = 0
            i_count += 1
        else:
            pred_boxes = last_boxes
            p_streak += 1
            p_count += 1

        iou = match_boxes(gt_boxes, pred_boxes)
        iou_scores.append(iou)
        latencies.append((time.perf_counter() - t_start) * 1000.0)
        frame_idx += 1

    cap.release()
    total = i_count + p_count

    return {
        "max_p_streak": max_p_streak,
        "i_ratio": i_count / total * 100 if total > 0 else 0,
        "p_ratio": p_count / total * 100 if total > 0 else 0,
        "mean_ms": float(np.mean(latencies)),
        "fps": 1000.0 / float(np.mean(latencies)),
        "mean_iou": float(np.mean(iou_scores)) * 100,
        "p99_ms": float(np.percentile(latencies, 99)),
    }


def plot_tradeoff(results, baseline_fps, baseline_iou, out_dir="."):
    streaks  = [r["max_p_streak"] for r in results]
    fps_vals = [r["fps"] for r in results]
    iou_vals = [r["mean_iou"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    sc = ax.scatter(fps_vals, iou_vals, c=streaks, cmap="RdYlGn_r",
                    s=120, zorder=5, edgecolors="black", linewidths=0.8)
    ax.plot(fps_vals, iou_vals, "b--", alpha=0.4, linewidth=1.2)

    for r in results:
        ax.annotate(f"streak={r['max_p_streak']}\n({r['i_ratio']:.0f}% I-frame)",
                    (r["fps"], r["mean_iou"]),
                    textcoords="offset points", xytext=(8, 4),
                    fontsize=8, color="black")

    ax.scatter([baseline_fps], [baseline_iou], marker="*", s=300,
               color="gold", edgecolors="black", linewidths=1.2,
               zorder=6, label=f"Baseline\n(all I-frame)\nFPS={baseline_fps:.0f}")
    ax.annotate("Baseline\n(모든 프레임 추론)",
                (baseline_fps, baseline_iou),
                textcoords="offset points", xytext=(8, -20),
                fontsize=8, color="darkred", fontweight="bold")

    plt.colorbar(sc, ax=ax, label="max_p_streak")
    ax.set_xlabel("FPS (처리 속도)", fontsize=12)
    ax.set_ylabel("Mean IoU Match Rate (%)", fontsize=12)
    ax.set_title("Speed-Accuracy Tradeoff\n(max_p_streak 변화에 따른 FPS vs 정확도)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    x = np.arange(len(streaks))
    w = 0.35

    bars1 = ax2.bar(x - w/2, fps_vals, w, label="FPS", color="#3B82F6", alpha=0.85)
    ax3 = ax2.twinx()
    bars2 = ax3.bar(x + w/2, iou_vals, w, label="IoU Match (%)", color="#10B981", alpha=0.85)

    ax2.set_xticks(x)
    ax2.set_xticklabels([f"streak={s}" for s in streaks], fontsize=10)
    ax2.set_ylabel("FPS", fontsize=12, color="#3B82F6")
    ax3.set_ylabel("IoU Match Rate (%)", fontsize=12, color="#10B981")
    ax2.tick_params(axis="y", labelcolor="#3B82F6")
    ax3.tick_params(axis="y", labelcolor="#10B981")

    for bar in bars1:
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                f"{bar.get_height():.0f}", ha="center", fontsize=9, color="#1D4ED8")
    for bar in bars2:
        ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                f"{bar.get_height():.1f}%", ha="center", fontsize=9, color="#065F46")

    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax3.get_legend_handles_labels()
    ax2.legend(h1+h2, l1+l2, fontsize=10)
    ax2.set_title("max_p_streak별 FPS vs 탐지 정확도", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = str(Path(out_dir) / "09_speed_accuracy_tradeoff.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Tradeoff] 저장: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0")
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--output", type=str, default=".")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Tradeoff] 디바이스: {device.upper()}")

    print("[Tradeoff] 모델 로딩 중...")
    model = torch.hub.load("ultralytics/yolov5", "yolov5s", verbose=False)
    model.to(device)

    streak_values = [1, 2, 4, 6, 8]
    results = []

    print(f"\n[Tradeoff] {len(streak_values)}가지 설정 × {args.frames}프레임 실험 시작\n")
    for streak in streak_values:
        print(f"  max_p_streak={streak} 실험 중...", end=" ", flush=True)
        res = run_single(source, model, device, streak, args.frames)
        results.append(res)
        print(f"FPS={res['fps']:.1f}, IoU={res['mean_iou']:.1f}%, I-frame={res['i_ratio']:.1f}%")

    print("  Baseline (all I-frame) 실험 중...", end=" ", flush=True)
    baseline = run_single(source, model, device, 0, args.frames)
    print(f"FPS={baseline['fps']:.1f}, IoU={baseline['mean_iou']:.1f}%")

    print("\n" + "="*65)
    print("  Speed-Accuracy Tradeoff 결과")
    print("="*65)
    print(f"  {'설정':<20} {'FPS':>8} {'IoU(%)':>8} {'I-frame%':>10} {'P99':>8}")
    print("-"*65)
    print(f"  {'Baseline(all I-frame)':<20} {baseline['fps']:>8.1f} "
          f"{baseline['mean_iou']:>7.1f}% {100.0:>9.1f}% {baseline['p99_ms']:>6.1f}ms")
    for r in results:
        print(f"  {'streak='+str(r['max_p_streak']):<20} {r['fps']:>8.1f} "
              f"{r['mean_iou']:>7.1f}% {r['i_ratio']:>9.1f}% {r['p99_ms']:>6.1f}ms")
    print("="*65)

    best = max(results, key=lambda x: x["fps"] * x["mean_iou"])
    print(f"\n  최적 조합 (FPS × IoU 조화): max_p_streak={best['max_p_streak']}")
    print(f"  → FPS={best['fps']:.1f}, IoU={best['mean_iou']:.1f}%")

    plot_tradeoff(results, baseline["fps"], baseline["mean_iou"], args.output)
    print("\n[Tradeoff] 완료!")
