"""I-frame(전체 추론) vs P-frame(결과 재사용) 탐지 정확도 비교."""

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
    """두 박스 간 IoU 계산"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


def match_boxes(gt_boxes, pred_boxes, iou_threshold=0.5):
    """gt_boxes와 pred_boxes 간 IoU>=threshold 매칭 비율 반환."""
    if len(gt_boxes) == 0 and len(pred_boxes) == 0:
        return 1.0
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return 0.0

    matched = 0
    for gt in gt_boxes:
        best_iou = max((compute_iou(gt, p) for p in pred_boxes), default=0)
        if best_iou >= iou_threshold:
            matched += 1

    return matched / max(len(gt_boxes), len(pred_boxes))


def evaluate(source, model, device, max_frames=300,
             threshold=0.35, max_p_streak=8, conf=0.4):

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"영상 열기 실패: {source}")

    model.conf = conf

    prev_gray = None
    p_streak = 0
    last_boxes = []
    last_conf_mean = 1.0

    records = []

    frame_idx = 0
    print(f"[Eval] 평가 시작 ({max_frames}프레임)...")

    while frame_idx < max_frames:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # SCS score = w1*pixel_change + w2*(1-conf)
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
        gt_conf_mean = float(np.mean(gt_confs)) if gt_confs else 1.0

        force_i = p_streak >= max_p_streak
        if force_i or scene_score >= threshold:
            frame_type = "I"
            pred_boxes = gt_boxes
            last_boxes = gt_boxes
            last_conf_mean = gt_conf_mean
            p_streak = 0
        else:
            frame_type = "P"
            pred_boxes = last_boxes
            p_streak += 1

        iou_match = match_boxes(gt_boxes, pred_boxes, iou_threshold=0.5)

        records.append({
            "frame_idx": frame_idx,
            "frame_type": frame_type,
            "scene_score": scene_score,
            "gt_count": len(gt_boxes),
            "pred_count": len(pred_boxes),
            "iou_match": iou_match,
            "gt_conf": gt_conf_mean,
        })

        if frame_idx % 50 == 0:
            print(f"  Frame {frame_idx:4d} | {frame_type}-frame | "
                  f"GT={len(gt_boxes):2d}개 Pred={len(pred_boxes):2d}개 | "
                  f"IoU match={iou_match:.2f} | scene={scene_score:.2f}")

        frame_idx += 1

    cap.release()
    return records


def analyze_and_plot(records, out_dir="."):
    i_records = [r for r in records if r["frame_type"] == "I"]
    p_records = [r for r in records if r["frame_type"] == "P"]

    all_iou    = [r["iou_match"] for r in records]
    p_iou      = [r["iou_match"] for r in p_records]
    i_iou      = [r["iou_match"] for r in i_records]

    p_gt_counts   = [r["gt_count"] for r in p_records]
    p_pred_counts = [r["pred_count"] for r in p_records]

    print("\n" + "="*60)
    print("  정확도 분석 결과")
    print("="*60)
    print(f"  총 프레임       : {len(records)}")
    print(f"  I-frame 횟수    : {len(i_records)} ({len(i_records)/len(records)*100:.1f}%)")
    print(f"  P-frame 횟수    : {len(p_records)} ({len(p_records)/len(records)*100:.1f}%)")
    print("-"*60)
    print(f"  전체 IoU Match  : {np.mean(all_iou):.4f}  ({np.mean(all_iou)*100:.1f}%)")
    print(f"  I-frame IoU     : {np.mean(i_iou):.4f}  ({np.mean(i_iou)*100:.1f}%)")
    print(f"  P-frame IoU     : {np.mean(p_iou):.4f}  ({np.mean(p_iou)*100:.1f}%)")
    print("-"*60)

    if p_gt_counts and p_pred_counts:
        gt_arr   = np.array(p_gt_counts)
        pred_arr = np.array(p_pred_counts)
        same = np.sum(gt_arr == pred_arr)
        print(f"  P-frame 탐지 수 일치율: {same/len(p_gt_counts)*100:.1f}%")
        print(f"  P-frame GT 평균 객체 수  : {gt_arr.mean():.2f}")
        print(f"  P-frame Pred 평균 객체 수: {pred_arr.mean():.2f}")

    perfect = sum(1 for r in p_records if r["iou_match"] >= 0.999)
    print(f"  P-frame 완벽 일치 비율  : {perfect/max(len(p_records),1)*100:.1f}%")
    print(f"  P-frame IoU >= 0.5 비율 : {sum(1 for r in p_records if r['iou_match']>=0.5)/max(len(p_records),1)*100:.1f}%")
    print("="*60)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    frames = [r["frame_idx"] for r in records]
    iou_vals = [r["iou_match"] for r in records]
    types = [r["frame_type"] for r in records]

    for r in records:
        if r["frame_type"] == "I":
            axes[0].axvspan(r["frame_idx"]-0.5, r["frame_idx"]+0.5,
                           alpha=0.2, color="red")

    axes[0].plot(frames, iou_vals, color="#2E75B6", linewidth=0.8, alpha=0.8)
    axes[0].axhline(np.mean(iou_vals), color="green", linestyle="--",
                    linewidth=1.5, label=f"평균 IoU = {np.mean(iou_vals):.3f}")
    axes[0].axhline(0.5, color="orange", linestyle=":", linewidth=1.5, label="IoU=0.5 기준선")
    axes[0].set_ylabel("IoU Match Rate", fontsize=11)
    axes[0].set_title("Detection Consistency: I-frame(전체 추론) vs P-frame(재사용)", fontsize=13, fontweight="bold")
    axes[0].legend(fontsize=10)
    axes[0].set_ylim(-0.05, 1.1)

    import matplotlib.patches as mpatches
    i_patch = mpatches.Patch(color="red", alpha=0.3, label="I-frame (전체 추론)")
    axes[0].legend(fontsize=9, handles=axes[0].get_legend_handles_labels()[0] + [i_patch])

    p_frames = [r["frame_idx"] for r in p_records]
    axes[1].plot(p_frames, [r["gt_count"] for r in p_records],
                color="red", linewidth=0.8, label="실제 탐지 수 (전체 추론)", alpha=0.8)
    axes[1].plot(p_frames, [r["pred_count"] for r in p_records],
                color="blue", linewidth=0.8, linestyle="--", label="P-frame 재사용 수", alpha=0.8)
    axes[1].set_xlabel("Frame Index", fontsize=11)
    axes[1].set_ylabel("Detection Count", fontsize=11)
    axes[1].set_title("P-frame 구간: 실제 탐지 수 vs 재사용 탐지 수", fontsize=13, fontweight="bold")
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    out1 = str(Path(out_dir) / "07_accuracy_timeline.png")
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[Eval] 저장: {out1}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist([r["iou_match"] for r in p_records], bins=20,
                color="#2E75B6", alpha=0.8, edgecolor="white")
    axes[0].axvline(np.mean(p_iou), color="red", linestyle="--",
                   linewidth=2, label=f"평균 = {np.mean(p_iou):.3f}")
    axes[0].axvline(0.5, color="orange", linestyle=":", linewidth=2, label="IoU=0.5")
    axes[0].set_xlabel("IoU Match Rate", fontsize=11)
    axes[0].set_ylabel("Frame Count", fontsize=11)
    axes[0].set_title("P-frame IoU Match 분포", fontsize=12, fontweight="bold")
    axes[0].legend(fontsize=10)

    categories = ["완벽 일치\n(IoU=1.0)", "IoU ≥ 0.5\n(허용 범위)", "IoU < 0.5\n(불일치)"]
    perfect_pct = sum(1 for r in p_records if r["iou_match"] >= 0.999) / max(len(p_records),1) * 100
    ok_pct      = sum(1 for r in p_records if 0.5 <= r["iou_match"] < 0.999) / max(len(p_records),1) * 100
    bad_pct     = sum(1 for r in p_records if r["iou_match"] < 0.5) / max(len(p_records),1) * 100

    colors = ["#10B981", "#3B82F6", "#EF4444"]
    bars = axes[1].bar(categories, [perfect_pct, ok_pct, bad_pct],
                      color=colors, alpha=0.85)
    for bar, val in zip(bars, [perfect_pct, ok_pct, bad_pct]):
        axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                    f"{val:.1f}%", ha="center", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("비율 (%)", fontsize=11)
    axes[1].set_title("P-frame 탐지 일치도 분류", fontsize=12, fontweight="bold")
    axes[1].set_ylim(0, 110)

    plt.tight_layout()
    out2 = str(Path(out_dir) / "08_accuracy_distribution.png")
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Eval] 저장: {out2}")

    return {
        "mean_iou": round(float(np.mean(all_iou)), 4),
        "p_frame_mean_iou": round(float(np.mean(p_iou)), 4),
        "perfect_pct": round(perfect_pct, 1),
        "ok_pct": round(ok_pct, 1),
        "bad_pct": round(bad_pct, 1),
    }


def evaluate_modes(source, max_frames=300, conf=0.4, max_p_streak=4):
    """frame 모드(CodecInspiredScheduler) vs object 모드(ObjectSelectiveScheduler).

    매 프레임 전체 추론 결과를 GT로 두고, 두 스케줄러가 같은 프레임에서 내놓는 박스를
    각각 GT와 비교한다. 모델/스케줄러/ROI 추론 경로는 pipeline.py와 동일하다.
    경보(alert)는 실행 시간에 따라 달라지므로 두 모드 모두 끈다.
    """
    from pipeline import RealtimePipeline, SceneChangeDetector, CodecInspiredScheduler
    from object_level_scheduler import ObjectSelectiveScheduler

    pl = RealtimePipeline(source=source, max_frames=max_frames,
                          conf_threshold=conf, mode="object")
    pl.cap.release()
    cap = cv2.VideoCapture(source)

    f_det, f_sched = SceneChangeDetector(), CodecInspiredScheduler(max_p_streak=max_p_streak)
    o_det, o_sched = SceneChangeDetector(), ObjectSelectiveScheduler(max_p_streak=max_p_streak)

    records = []
    print(f"[Eval] 모드 비교 시작 ({max_frames}프레임)...")
    for frame_idx in range(max_frames):
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        d = pl._detect(frame_rgb)[0]
        gt_dets = [(r[:4].tolist(), float(r[4]), int(r[5])) for r in d]
        gt_boxes = [b for b, _, _ in gt_dets]

        # frame 모드: I면 전체 추론 결과, P면 직전 I-frame 결과 재사용
        score = f_det.compute_score(frame_bgr, mean_conf=f_sched.get_last_conf())
        f_type = f_sched.decide_frame_type(score)
        if f_type == "I":
            confs = [c for _, c, _ in gt_dets]
            f_sched.update_last_result(gt_boxes, float(np.mean(confs)) if confs else 1.0)
        f_boxes = f_sched.get_last_boxes()

        # object 모드: I면 전체 추론, S면 빠른 객체 ROI만 추론, P면 재사용
        score = o_det.compute_score(frame_bgr, mean_conf=o_sched.get_last_conf())
        plan = o_sched.plan(frame_bgr, score, gray=o_det.last_gray)
        if plan.frame_type == "I":
            o_sched.commit(plan, gt_dets)
        elif plan.frame_type == "S":
            o_sched.commit(plan, pl._infer_rois(frame_rgb, plan.rois))
        o_boxes = o_sched.get_last_boxes()

        records.append({
            "frame_idx": frame_idx,
            "gt_count": len(gt_boxes),
            "frame_type": f_type,
            "frame_iou": match_boxes(gt_boxes, f_boxes),
            "frame_pred": len(f_boxes),
            "object_type": plan.frame_type,
            "object_iou": match_boxes(gt_boxes, o_boxes),
            "object_pred": len(o_boxes),
            "n_fast": plan.n_fast,
        })

        if frame_idx % 50 == 0:
            r = records[-1]
            print(f"  Frame {frame_idx:4d} | GT={r['gt_count']:2d} | "
                  f"frame {f_type} IoU={r['frame_iou']:.2f} | "
                  f"object {plan.frame_type} IoU={r['object_iou']:.2f}")

    cap.release()
    return records


def analyze_modes(records, csv_path=None):
    import csv
    n = len(records)

    def summary(mode):
        iou = np.array([r[f"{mode}_iou"] for r in records])
        types = np.array([r[f"{mode}_type"] for r in records])
        reuse = types != "I"
        return {
            "mean_iou": float(iou.mean()),
            "reuse_iou": float(iou[reuse].mean()) if reuse.any() else float("nan"),
            "bad_pct": float((iou < 0.5).mean() * 100),
            "infer_pct": float(np.isin(types, ["I", "S"]).mean() * 100),
            "by_type": {t: (int((types == t).sum()), float(iou[types == t].mean()))
                        for t in ("I", "S", "P") if (types == t).any()},
        }

    f, o = summary("frame"), summary("object")
    print("\n" + "="*64)
    print("  frame 모드 vs object 모드 IoU Match Rate (GT = 매 프레임 전체 추론)")
    print("="*64)
    print(f"  {'':24s}{'frame':>12s}{'object':>12s}")
    print(f"  {'전체 IoU Match':24s}{f['mean_iou']*100:11.1f}%{o['mean_iou']*100:11.1f}%")
    print(f"  {'재사용/부분추론 프레임':22s}{f['reuse_iou']*100:11.1f}%{o['reuse_iou']*100:11.1f}%")
    print(f"  {'IoU Match < 0.5 프레임':23s}{f['bad_pct']:11.1f}%{o['bad_pct']:11.1f}%")
    print(f"  {'추론 수행 프레임(I+S)':22s}{f['infer_pct']:11.1f}%{o['infer_pct']:11.1f}%")
    print("-"*64)
    for name, s in (("frame", f), ("object", o)):
        parts = [f"{t}={cnt}({cnt/n*100:.0f}%) IoU {m*100:.1f}%" for t, (cnt, m) in s["by_type"].items()]
        print(f"  {name:7s}: " + " | ".join(parts))

    # 두 모드가 갈리는 지점: object 모드가 빠른 객체를 다시 추론한 S-frame
    s_rows = [r for r in records if r["object_type"] == "S"]
    if s_rows:
        print("-"*64)
        print(f"  object S-frame {len(s_rows)}개에서: "
              f"frame 모드 IoU {np.mean([r['frame_iou'] for r in s_rows])*100:.1f}% vs "
              f"object 모드 IoU {np.mean([r['object_iou'] for r in s_rows])*100:.1f}%")
    print("="*64)

    if csv_path:
        with open(csv_path, "w", newline="", encoding="utf-8") as fp:
            w = csv.DictWriter(fp, fieldnames=list(records[0].keys()))
            w.writeheader()
            w.writerows(records)
        print(f"[Eval] 저장: {csv_path}")
    return f, o


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--output", type=str, default=".")
    parser.add_argument("--conf",   type=float, default=0.4)
    parser.add_argument("--compare-modes", action="store_true",
                        help="pipeline.py의 frame 모드 vs object 모드 IoU Match Rate 비교")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    if args.compare_modes:
        records = evaluate_modes(source, max_frames=args.frames, conf=args.conf)
        stem = Path(args.source).stem if isinstance(source, str) else f"cam{source}"
        analyze_modes(records, csv_path=str(Path(args.output) / f"accuracy_modes_{stem}.csv"))
        raise SystemExit(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Eval] 디바이스: {device.upper()}")

    print("[Eval] 모델 로딩 중...")
    model = torch.hub.load("ultralytics/yolov5", "yolov5s", verbose=False)
    model.to(device)

    records = evaluate(source, model, device,
                      max_frames=args.frames,
                      conf=args.conf)

    stats = analyze_and_plot(records, out_dir=args.output)

    print(f"\n[Eval] 핵심 결과:")
    print(f"  P-frame 평균 IoU Match: {stats['p_frame_mean_iou']*100:.1f}%")
    print(f"  완벽 일치 비율:          {stats['perfect_pct']:.1f}%")
    print(f"  허용 범위(IoU≥0.5):      {stats['ok_pct']+stats['perfect_pct']:.1f}%")
    print("[Eval] 완료!")
