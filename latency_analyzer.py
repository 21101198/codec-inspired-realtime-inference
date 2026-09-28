
import csv
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


def load_csv(path: str) -> list:
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "frame_idx":     int(row["frame_idx"]),
                "frame_type":    row["frame_type"],
                "scene_score":   float(row["scene_score"]),
                "alert":         row["alert"].lower() == "true",
                "preprocess_ms": float(row["preprocess_ms"]),
                "htod_ms":       float(row["htod_ms"]),
                "infer_ms":      float(row["infer_ms"]),
                "dtoh_ms":       float(row["dtoh_ms"]),
                "total_ms":      float(row["total_ms"]),
            })
    return rows


def compute_stats(values: list) -> dict:
    arr = np.array(values)
    return {
        "p50":  round(float(np.percentile(arr, 50)),  2),
        "p90":  round(float(np.percentile(arr, 90)),  2),
        "p99":  round(float(np.percentile(arr, 99)),  2),
        "mean": round(float(np.mean(arr)),             2),
        "std":  round(float(np.std(arr)),              2),
        "fps":  round(1000 / float(np.mean(arr)),      1),
    }


def plot_timeline(rows: list, out_path: str):
    frames = [r["frame_idx"] for r in rows]
    totals = [r["total_ms"]  for r in rows]
    types  = [r["frame_type"] for r in rows]
    alerts = [r["alert"]      for r in rows]

    fig, ax = plt.subplots(figsize=(14, 5))

    for r in rows:
        if r["frame_type"] == "I":
            ax.axvspan(r["frame_idx"] - 0.5, r["frame_idx"] + 0.5,
                       alpha=0.08, color="#FF4444")

    ax.plot(frames, totals, color="#2E75B6", linewidth=0.8, label="Total Latency (ms)")

    p99 = np.percentile(totals, 99)
    ax.axhline(p99, color="red", linestyle="--", linewidth=1.2,
               label=f"P99 = {p99:.1f} ms")

    mean_v = np.mean(totals)
    ax.axhline(mean_v, color="green", linestyle=":", linewidth=1.2,
               label=f"Mean = {mean_v:.1f} ms")

    alert_frames = [r["frame_idx"] for r in rows if r["alert"]]
    alert_vals   = [r["total_ms"]  for r in rows if r["alert"]]
    if alert_frames:
        ax.scatter(alert_frames, alert_vals, color="orange", s=30, zorder=5,
                   label="Tail Latency Alert")

    ax.set_xlabel("Frame Index", fontsize=11)
    ax.set_ylabel("Latency (ms)", fontsize=11)
    ax.set_title("End-to-End Latency Timeline (I/P-frame + Alert)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    i_patch = mpatches.Patch(color="#FF4444", alpha=0.15, label="I-frame (Full Inference)")
    p_patch = mpatches.Patch(color="#2E75B6", alpha=0.3,  label="P-frame (Light Inference)")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [i_patch, p_patch], labels + ["I-frame", "P-frame"], fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[Analyzer] 저장: {out_path}")


def plot_breakdown(rows: list, out_path: str):
    i_rows = [r for r in rows if r["frame_type"] == "I"]
    p_rows = [r for r in rows if r["frame_type"] == "P"]

    def seg_mean(rs, key):
        return np.mean([r[key] for r in rs]) if rs else 0

    categories = ["I-frame\n(전체 추론)", "P-frame\n(경량 추론)"]
    pre   = [seg_mean(i_rows, "preprocess_ms"), seg_mean(p_rows, "preprocess_ms")]
    htod  = [seg_mean(i_rows, "htod_ms"),       seg_mean(p_rows, "htod_ms")]
    infer = [seg_mean(i_rows, "infer_ms"),       seg_mean(p_rows, "infer_ms")]
    dtoh  = [seg_mean(i_rows, "dtoh_ms"),        seg_mean(p_rows, "dtoh_ms")]

    x = np.arange(len(categories))
    w = 0.5
    fig, ax = plt.subplots(figsize=(7, 5))

    b1 = ax.bar(x, pre,   w, label="전처리 (Preprocess)",   color="#5B9BD5")
    b2 = ax.bar(x, htod,  w, bottom=pre,                    label="HtoD 전송",          color="#ED7D31")
    b3 = ax.bar(x, infer, w, bottom=np.add(pre, htod),      label="GPU 추론 (Infer)",   color="#A9D18E")
    b4 = ax.bar(x, dtoh,  w, bottom=np.add(np.add(pre, htod), infer), label="DtoH 전송", color="#FFC000")

    totals_bar = [p+h+i+d for p,h,i,d in zip(pre, htod, infer, dtoh)]
    for xi, tv in zip(x, totals_bar):
        ax.text(xi, tv + 0.2, f"{tv:.1f}ms", ha="center", va="bottom",
                fontsize=11, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylabel("Latency (ms)", fontsize=11)
    ax.set_title("구간별 지연 시간 비교: I-frame vs P-frame", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[Analyzer] 저장: {out_path}")


def plot_tail_distribution(rows: list, out_path: str):
    totals = [r["total_ms"] for r in rows]
    p50 = np.percentile(totals, 50)
    p90 = np.percentile(totals, 90)
    p99 = np.percentile(totals, 99)

    fig, ax = plt.subplots(figsize=(9, 5))
    n, bins, patches = ax.hist(totals, bins=50, color="#2E75B6",
                                alpha=0.7, edgecolor="white", linewidth=0.5)

    ax.axvline(p50, color="green",  linestyle="--", linewidth=2, label=f"P50 = {p50:.1f} ms")
    ax.axvline(p90, color="orange", linestyle="--", linewidth=2, label=f"P90 = {p90:.1f} ms")
    ax.axvline(p99, color="red",    linestyle="--", linewidth=2, label=f"P99 = {p99:.1f} ms  ← Tail Latency")

    ax.set_xlabel("Total Latency (ms)", fontsize=11)
    ax.set_ylabel("Frame Count", fontsize=11)
    ax.set_title("Tail Latency 분포 (P50 / P90 / P99)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[Analyzer] 저장: {out_path}")


def print_ablation(rows: list):
    all_total  = [r["total_ms"]  for r in rows]
    i_infer    = [r["infer_ms"]  for r in rows if r["frame_type"] == "I"]
    p_infer    = [r["infer_ms"]  for r in rows if r["frame_type"] == "P"]
    all_infer  = [r["infer_ms"]  for r in rows]

    total_s = compute_stats(all_total)
    infer_s = compute_stats(all_infer)

    i_pct = sum(1 for r in rows if r["frame_type"] == "I") / len(rows) * 100
    p_pct = 100 - i_pct
    alert_cnt = sum(1 for r in rows if r["alert"])

    print("\n" + "="*65)
    print("  Ablation Study 결과 요약")
    print("="*65)
    print(f"  {'항목':<30} {'값':>15}")
    print("-"*65)
    print(f"  {'총 프레임':<30} {len(rows):>15}")
    print(f"  {'I-frame 비율':<30} {i_pct:>14.1f}%")
    print(f"  {'P-frame 비율':<30} {p_pct:>14.1f}%")
    print(f"  {'Tail Latency 경보 횟수':<30} {alert_cnt:>15}")
    print("-"*65)
    print(f"  {'Total Latency P50':<30} {total_s['p50']:>13.2f} ms")
    print(f"  {'Total Latency P90':<30} {total_s['p90']:>13.2f} ms")
    print(f"  {'Total Latency P99 (Tail)':<30} {total_s['p99']:>13.2f} ms")
    print(f"  {'Total Latency Mean':<30} {total_s['mean']:>13.2f} ms")
    print(f"  {'Total Latency Std':<30} {total_s['std']:>13.2f} ms")
    print(f"  {'평균 FPS':<30} {total_s['fps']:>13.1f}")
    print("-"*65)
    print(f"  {'Infer P50':<30} {infer_s['p50']:>13.2f} ms")
    print(f"  {'Infer P99':<30} {infer_s['p99']:>13.2f} ms")
    if i_infer:
        print(f"  {'I-frame 평균 추론 시간':<30} {np.mean(i_infer):>13.2f} ms")
    if p_infer:
        print(f"  {'P-frame 평균 추론 시간':<30} {np.mean(p_infer):>13.2f} ms")
        if i_infer:
            speedup = np.mean(i_infer) / np.mean(p_infer)
            print(f"  {'P-frame 추론 속도 향상':<30} {speedup:>13.2f}x")
    print("="*65)


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "latency_log.csv"

    if not Path(csv_path).exists():
        print(f"[Analyzer] 파일을 찾을 수 없습니다: {csv_path}")
        print("  먼저 pipeline.py를 실행해 latency_log.csv를 생성하세요.")
        sys.exit(1)

    print(f"[Analyzer] 데이터 로드: {csv_path}")
    rows = load_csv(csv_path)
    print(f"[Analyzer] 총 {len(rows)} 프레임 분석 중...\n")

    out_dir = Path(csv_path).parent

    plot_timeline(rows,          str(out_dir / "01_latency_timeline.png"))
    plot_breakdown(rows,         str(out_dir / "02_segment_breakdown.png"))
    plot_tail_distribution(rows, str(out_dir / "03_tail_distribution.png"))
    print_ablation(rows)

    print(f"\n[Analyzer] 완료! 그래프 3개가 저장되었습니다.")
    print(f"  → {out_dir}/01_latency_timeline.png")
    print(f"  → {out_dir}/02_segment_breakdown.png")
    print(f"  → {out_dir}/03_tail_distribution.png")
