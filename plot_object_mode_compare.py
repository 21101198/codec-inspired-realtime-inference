"""frame 모드 vs object 모드 비교 그래프 (object_mode_compare_summary.csv -> 15_object_vs_frame_compare.png)."""

import csv
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams.update({
    "font.family": ["Malgun Gothic", "AppleGothic", "Noto Sans CJK KR", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "font.size": 11,
})

SCENARIOS = ["highspeed", "crowd", "lowlight"]
MODES = ["frame", "object"]
COLORS = {"frame": "#2a78d6", "object": "#eb6834"}
LABELS = {"frame": "frame 모드 (프레임 단위 I/P)", "object": "object 모드 (객체별 I/S/P)"}
TEXT, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def load(path):
    with open(path, encoding="utf-8") as f:
        return {(r["scenario"], r["mode"]): {k: (float(v) if k not in ("scenario", "mode") else v)
                                             for k, v in r.items()}
                for r in csv.DictReader(f)}


def style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, fontsize=12, fontweight="bold", color=TEXT, loc="left")
    ax.set_ylabel(ylabel, color=MUTED)
    ax.set_xticks(np.arange(len(SCENARIOS)))
    ax.set_xticklabels(SCENARIOS, color=TEXT)
    ax.tick_params(axis="y", colors=MUTED, length=0)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)


def label_bars(ax, bars, values, fmt):
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(v),
                ha="center", va="bottom", fontsize=9, color=TEXT)


def main(summary, out):
    d = load(summary)
    x = np.arange(len(SCENARIOS))
    w = 0.36
    # 막대 사이 2px 간격이 보이도록 흰 테두리
    bar_kw = dict(width=w - 0.02, edgecolor=SURFACE, linewidth=1.5)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), facecolor=SURFACE)

    ax = axes[0]
    for i, m in enumerate(MODES):
        rows = [d[(s, m)] for s in SCENARIOS]
        mean = [r["fps_mean"] for r in rows]
        lo = [r["fps_mean"] - min(r["fps_run1"], r["fps_run2"]) for r in rows]
        hi = [max(r["fps_run1"], r["fps_run2"]) - r["fps_mean"] for r in rows]
        bars = ax.bar(x + (i - 0.5) * w, mean, color=COLORS[m], **bar_kw)
        ax.errorbar(x + (i - 0.5) * w, mean, yerr=[lo, hi], fmt="none",
                    ecolor=MUTED, elinewidth=1, capsize=3)
        for b, v, h in zip(bars, mean, hi):
            ax.text(b.get_x() + b.get_width() / 2, v + h + 3, f"{v:.0f}",
                    ha="center", va="bottom", fontsize=9, color=TEXT)
    style(ax, "처리 속도 (높을수록 좋음)", "FPS (2회 평균)")
    ax.set_ylim(0, 240)

    ax = axes[1]
    for i, m in enumerate(MODES):
        vals = [d[(s, m)]["iou_match_pct"] for s in SCENARIOS]
        bars = ax.bar(x + (i - 0.5) * w, vals, color=COLORS[m], **bar_kw)
        label_bars(ax, bars, vals, "{:.1f}")
    style(ax, "IoU Match Rate (높을수록 좋음)", "매 프레임 전체 추론과 일치율 (%)")
    ax.set_ylim(0, 100)

    ax = axes[2]
    for i, m in enumerate(MODES):
        i_pct = [d[(s, m)]["i_pct"] for s in SCENARIOS]
        s_pct = [d[(s, m)]["s_pct"] for s in SCENARIOS]
        pos = x + (i - 0.5) * w
        ax.bar(pos, i_pct, color=COLORS[m], **bar_kw)
        bars = ax.bar(pos, s_pct, bottom=i_pct, color=COLORS[m], alpha=0.45,
                      hatch="///", **bar_kw)
        for b, a, s in zip(bars, i_pct, s_pct):
            ax.text(b.get_x() + b.get_width() / 2, a + s, f"{a + s:.0f}",
                    ha="center", va="bottom", fontsize=9, color=TEXT)
    style(ax, "추론 수행 프레임 비율 (낮을수록 GPU 부하 적음)", "전체 프레임 중 비율 (%)")
    ax.set_ylim(0, 60)

    handles = [Patch(facecolor=COLORS[m], label=LABELS[m]) for m in MODES]
    handles += [Patch(facecolor="white", edgecolor=MUTED, hatch="///",
                      label="S-frame (빠른 객체 ROI만 추론) — 3번째 그래프")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.0), fontsize=10)
    fig.suptitle("frame 모드 vs object 모드: 속도 · 정확도 · 추론 비율 (720p, YOLOv5s, 300프레임)",
                 fontsize=14, fontweight="bold", color=TEXT, y=1.07)
    fig.text(0.01, -0.04,
             "IoU Match Rate: 같은 프레임을 매번 전체 추론한 결과를 기준으로 IoU≥0.5 박스 일치 비율 (클래스 무관). "
             "FPS 오차 막대 = 두 실행의 범위. FPS는 실행 간 편차가 커서 같은 실행 묶음 안에서의 상대 비교로만 해석.",
             fontsize=9, color=MUTED)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close()
    print(f"저장: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="object_mode_compare_summary.csv")
    parser.add_argument("--output", default="15_object_vs_frame_compare.png")
    args = parser.parse_args()
    main(args.summary, args.output)
