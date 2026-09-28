"""고속동작/저조도/군중 시나리오 영상을 내려받아 Baseline vs I/P-frame 스케줄러를 벤치마크한다."""
import argparse
import csv
import gc
import importlib.util
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bench_resolution_full import (
    load_model,
    run_baseline,
    run_scheduler,
    save_baseline_csv,
    save_scheduler_csv,
    summarize,
)

HERE = Path(__file__).resolve().parent
VIDEO_DIR = HERE / "videos"

SCENARIOS = [
    {
        "key": "highspeed",
        "label": "고속 동작 (High-speed Motion)",
        "desc": "빠른 카메라/피사체 이동 → 장면 변화가 잦아 I-frame 비율이 올라가는 케이스",
        "yt_query": "ytsearch8:rally car onboard pov 60fps",
        "yt_fallback_urls": [
            "https://www.youtube.com/watch?v=SkVqJ1SGeL0",
        ],
        "direct_urls": [
            "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/720/Big_Buck_Bunny_720_10s_30MB.mp4",
        ],
    },
    {
        "key": "lowlight",
        "label": "저조도 (Low-light)",
        "desc": "야간/어두운 환경 → 노이즈로 픽셀 diff 가 커지고 detection confidence 가 낮아지는 케이스",
        "yt_query": "ytsearch8:night city street walk neon rain 4k",
        "yt_fallback_urls": [
            "https://www.youtube.com/watch?v=R6MlUcmOul8",
        ],
        "direct_urls": [
            "https://test-videos.co.uk/vids/sintel/mp4/h264/720/Sintel_720_10s_30MB.mp4",
        ],
    },
    {
        "key": "crowd",
        "label": "군중 (Crowd)",
        "desc": "보행자 밀집 → 다수 객체로 추론 부하가 크고 P-frame 재사용 시 정확도 위험이 큰 케이스",
        "yt_query": "ytsearch8:crowded street pedestrians walking timelapse 4k",
        "yt_fallback_urls": [
            "https://www.youtube.com/watch?v=aqz-KE-bpKQ",
        ],
        "direct_urls": [
            "https://test-videos.co.uk/vids/jellyfish/mp4/h264/720/Jellyfish_720_10s_30MB.mp4",
        ],
    },
]

PLOT_PATH = HERE / "12_scenario_compare.png"
SUMMARY_CSV = HERE / "scenario_bench_summary.csv"


MIN_BYTES = 200_000


def _ytdlp_cmd():
    """yt-dlp 실행 프리픽스 반환 (exe 우선, 없으면 python -m yt_dlp)."""
    exe = shutil.which("yt-dlp")
    if exe:
        return [exe]
    if importlib.util.find_spec("yt_dlp") is not None:
        return [sys.executable, "-m", "yt_dlp"]
    return None


def _ytdlp_format(max_height):
    return (f"bv*[height<=?{max_height}][ext=mp4][vcodec^=avc1]/"
            f"bv*[height<=?{max_height}][ext=mp4]/"
            f"bv*[height<=?{max_height}]/"
            f"b[height<=?{max_height}]/bv*/b")


def _ytdlp_download_one(prefix, url, out_path: Path, max_height):
    for cand in out_path.parent.glob(out_path.stem + ".*"):
        cand.unlink()
    cmd = prefix + [
        "-f", _ytdlp_format(max_height),
        "--no-playlist", "--no-part", "--retries", "3", "--force-overwrites",
        "--match-filter", "duration>10 & duration<2400",
        "-o", str(out_path.with_suffix(".%(ext)s")),
        url,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        print("    타임아웃")
        return False
    if r.returncode != 0:
        tail = [l for l in (r.stderr or r.stdout).strip().splitlines() if l][-2:]
        print("    " + " | ".join(tail))
        return False
    got = next((c for c in out_path.parent.glob(out_path.stem + ".*")), None)
    if got and got != out_path:
        got.rename(out_path)
    ok = out_path.exists() and out_path.stat().st_size > MIN_BYTES
    if ok:
        print(f"    완료 ({out_path.stat().st_size / 1e6:.1f} MB)")
    return ok


def _ytsearch_ids(prefix, query, n=8):
    cmd = prefix + ["--flat-playlist", "--no-warnings", "--print", "id",
                    query if query.startswith("ytsearch") else f"ytsearch{n}:{query}"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return []
    return [x.strip() for x in r.stdout.splitlines() if x.strip()]


def download_with_ytdlp(target, out_path: Path, max_height=720):
    """target 이 ytsearch* 면 후보를 순회하며 첫 성공분을 받는다. 아니면 단일 URL."""
    prefix = _ytdlp_cmd()
    if prefix is None:
        print("  [yt-dlp] 미설치 → 건너뜀")
        return False

    if target.startswith("ytsearch"):
        ids = _ytsearch_ids(prefix, target)
        if not ids:
            print(f"  [yt-dlp] 검색 결과 없음: {target}")
            return False
        print(f"  [yt-dlp] 검색 후보 {len(ids)}개: {target}")
        for vid in ids:
            print(f"  [yt-dlp] 시도 https://youtu.be/{vid}")
            if _ytdlp_download_one(prefix, f"https://www.youtube.com/watch?v={vid}",
                                   out_path, max_height):
                return True
        return False

    print(f"  [yt-dlp] {target}")
    return _ytdlp_download_one(prefix, target, out_path, max_height)


def download_direct(url, out_path: Path):
    print(f"  [direct] {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp, open(out_path, "wb") as f:
            total = int(resp.headers.get("Content-Length", 0))
            done = 0
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r    {done/1e6:6.1f} / {total/1e6:.1f} MB", end="")
            print()
    except Exception as e:
        print(f"  [direct] 실패: {e}")
        if out_path.exists():
            out_path.unlink()
        return False
    ok = out_path.exists() and out_path.stat().st_size > MIN_BYTES
    print(f"  [direct] {'완료' if ok else '실패'}")
    return ok


def acquire_video(sc, prefer="ytdlp", yt_override=None, local_override=None,
                  no_download=False, max_height=720) -> Path:
    key = sc["key"]
    out_path = VIDEO_DIR / f"{key}.mp4"

    if local_override:
        p = Path(local_override).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"[{key}] --local 경로 없음: {p}")
        print(f"[{key}] 로컬 파일 사용: {p}")
        return p

    if yt_override and out_path.exists():
        print(f"[{key}] --yt 지정됨 → 기존 {out_path.name} 교체")
        out_path.unlink()

    if not yt_override and out_path.exists() and out_path.stat().st_size > MIN_BYTES:
        print(f"[{key}] 기존 파일 재사용: {out_path.name} "
              f"({out_path.stat().st_size/1e6:.1f} MB)")
        return out_path

    if no_download:
        raise FileNotFoundError(f"[{key}] --no-download 인데 {out_path} 가 없음")

    VIDEO_DIR.mkdir(exist_ok=True)

    if yt_override:
        yt_targets = [yt_override]
    else:
        yt_targets = [sc["yt_query"]] + list(sc.get("yt_fallback_urls", []))
    yt_steps = [("ytdlp", t) for t in yt_targets]
    dir_steps = [("direct", u) for u in sc.get("direct_urls", [])]
    attempts = dir_steps + yt_steps if prefer == "direct" else yt_steps + dir_steps

    print(f"[{key}] {sc['label']} 다운로드 (소스 {len(attempts)}순위)")
    for kind, target in attempts:
        ok = (download_with_ytdlp(target, out_path, max_height) if kind == "ytdlp"
              else download_direct(target, out_path))
        if ok:
            return out_path

    raise RuntimeError(f"[{key}] 모든 소스에서 다운로드 실패")


def load_frames(video_path: Path, resize, n):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열 수 없음: {video_path}")
    frames = []
    empty_reads = 0
    while len(frames) < n:
        ret, f = cap.read()
        if not ret:
            empty_reads += 1
            if empty_reads > 3:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        empty_reads = 0
        frames.append(cv2.resize(f, resize))
    cap.release()
    if not frames:
        raise RuntimeError(f"프레임을 못 읽음: {video_path}")
    return frames


def plot_results(results):
    plt.rcParams.update({
        "font.family": ["Malgun Gothic", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
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
    b1 = ax.bar(x - w/2, base_means, w, label="Baseline (All I-frame)", color=COLOR_BASE, alpha=0.85)
    b2 = ax.bar(x + w/2, sched_means, w, label="I/P-frame Scheduler", color=COLOR_OURS, alpha=0.85)
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"{bar.get_height():.1f}ms", ha="center", va="bottom", fontsize=10, fontweight="bold")
    for i in range(len(labels)):
        ratio = base_means[i] / max(sched_means[i], 0.01)
        ax.text(x[i], max(base_means[i], sched_means[i]) * 1.12, f"{ratio:.1f}x↓",
                ha="center", fontsize=12, color="darkgreen", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Mean Latency (ms)")
    ax.set_title("Mean Latency by Scenario")
    ax.set_ylim(0, max(base_means) * 1.35)
    ax.legend(fontsize=10)

    ax2 = axes[1]
    b3 = ax2.bar(x - w/2, base_fps, w, label="Baseline (All I-frame)", color=COLOR_BASE, alpha=0.85)
    b4 = ax2.bar(x + w/2, sched_fps, w, label="I/P-frame Scheduler", color=COLOR_OURS, alpha=0.85)
    for bar in list(b3) + list(b4):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    for i in range(len(labels)):
        ratio = sched_fps[i] / max(base_fps[i], 0.01)
        ax2.text(x[i], max(base_fps[i], sched_fps[i]) * 1.12, f"{ratio:.1f}x↑",
                 ha="center", fontsize=12, color="darkgreen", fontweight="bold")
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_ylabel("FPS")
    ax2.set_title("Throughput (FPS) by Scenario")
    ax2.set_ylim(0, max(sched_fps) * 1.35)
    ax2.legend(fontsize=10)

    fig.suptitle("Baseline vs I/P-frame Scheduler across Scenarios (same model, same frame count)",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[Plot] saved -> {PLOT_PATH}")


def write_summary_csv(results):
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "key", "frames",
                    "base_mean_ms", "base_p50", "base_p90", "base_p99", "base_fps",
                    "sched_mean_ms", "sched_p50", "sched_p90", "sched_p99", "sched_fps",
                    "speedup_x", "i_ratio_pct"])
        for r in results:
            b, s = r["base"], r["sched"]
            speedup = b["mean"] / max(s["mean"], 1e-3)
            w.writerow([r["label"], r["key"], r["frames"],
                        f"{b['mean']:.3f}", f"{b['p50']:.3f}", f"{b['p90']:.3f}", f"{b['p99']:.3f}", f"{b['fps']:.2f}",
                        f"{s['mean']:.3f}", f"{s['p50']:.3f}", f"{s['p90']:.3f}", f"{s['p99']:.3f}", f"{s['fps']:.2f}",
                        f"{speedup:.3f}", f"{r['i_ratio']:.1f}"])
    print(f"[CSV]  saved -> {SUMMARY_CSV}")


def print_summary_table(results):
    print("\n" + "=" * 96)
    print(f"  {'Scenario':<26} {'Base(ms)':>9} {'Sched(ms)':>10} {'Speedup':>8} "
          f"{'Base FPS':>9} {'Sched FPS':>10} {'I-ratio':>8} {'P99 base':>9} {'P99 sched':>10}")
    print("-" * 96)
    for r in results:
        b, s = r["base"], r["sched"]
        speedup = b["mean"] / max(s["mean"], 1e-3)
        print(f"  {r['label']:<26} {b['mean']:>9.2f} {s['mean']:>10.2f} {speedup:>7.2f}x "
              f"{b['fps']:>9.1f} {s['fps']:>10.1f} {r['i_ratio']:>7.1f}% "
              f"{b['p99']:>9.2f} {s['p99']:>10.2f}")
    print("=" * 96)


def parse_kv(pairs):
    out = {}
    for item in pairs or []:
        if "=" not in item:
            raise SystemExit(f"'key=value' 형식이어야 함: {item}")
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=500, help="시나리오당 처리 프레임 수")
    ap.add_argument("--width", type=int, default=640, help="리사이즈 폭 (기존 실험=640)")
    ap.add_argument("--height", type=int, default=360, help="리사이즈 높이 (기존 실험=360)")
    ap.add_argument("--scenarios", type=str, default="",
                    help="쉼표로 부분 선택 (예: highspeed,crowd). 비우면 전체")
    ap.add_argument("--prefer", choices=["ytdlp", "direct"], default="ytdlp",
                    help="1순위 소스 (기본 ytdlp)")
    ap.add_argument("--max-height", type=int, default=720, help="yt-dlp 최대 세로 해상도")
    ap.add_argument("--no-download", action="store_true",
                    help="다운로드 없이 videos/<key>.mp4 만 사용")
    ap.add_argument("--yt", action="append", metavar="key=URL",
                    help="특정 시나리오의 YouTube URL 지정 (반복 가능)")
    ap.add_argument("--local", action="append", metavar="key=PATH",
                    help="특정 시나리오를 로컬 영상 파일로 대체 (반복 가능)")
    args = ap.parse_args()

    yt_over = parse_kv(args.yt)
    local_over = parse_kv(args.local)
    resize = (args.width, args.height)

    selected = SCENARIOS
    if args.scenarios.strip():
        want = {s.strip() for s in args.scenarios.split(",")}
        selected = [s for s in SCENARIOS if s["key"] in want]
        if not selected:
            raise SystemExit(f"매칭되는 시나리오 없음: {want}")

    print("=" * 70)
    print("  다중 영상 벤치마크: Baseline vs I/P-frame Scheduler")
    print(f"  시나리오: {[s['key'] for s in selected]}")
    print(f"  프레임: {args.frames} | 리사이즈: {resize[0]}x{resize[1]} | 1순위 소스: {args.prefer}")
    print("=" * 70)

    model, device = load_model()
    results = []

    for sc in selected:
        key = sc["key"]
        print(f"\n{'='*70}\n[{key}] {sc['label']}\n  {sc['desc']}\n{'='*70}")

        video = acquire_video(
            sc, prefer=args.prefer,
            yt_override=yt_over.get(key),
            local_override=local_over.get(key),
            no_download=args.no_download,
            max_height=args.max_height,
        )

        print(f"[{key}] 프레임 로딩 중...")
        frames = load_frames(video, resize, args.frames)
        print(f"[{key}] {len(frames)} 프레임 로드 완료")

        base_log = run_baseline(frames, model, device)
        save_baseline_csv(base_log, str(HERE / f"baseline_{key}.csv"))
        base_stats = summarize(base_log)
        print(f"[{key}] Baseline  mean={base_stats['mean']:.2f}ms  fps={base_stats['fps']:.1f}")

        sched_log = run_scheduler(frames, model, device)
        save_scheduler_csv(sched_log, str(HERE / f"latency_{key}.csv"))
        sched_stats = summarize(sched_log)
        i_count = sum(1 for l in sched_log if l.frame_type == "I")
        i_ratio = i_count / len(sched_log) * 100
        print(f"[{key}] Scheduler mean={sched_stats['mean']:.2f}ms  fps={sched_stats['fps']:.1f}  "
              f"I-ratio={i_ratio:.1f}%")

        results.append({
            "key": key, "label": sc["label"], "frames": len(frames),
            "base": base_stats, "sched": sched_stats, "i_ratio": i_ratio,
        })

        del frames, base_log, sched_log
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    if not results:
        raise SystemExit("결과 없음")

    plot_results(results)
    write_summary_csv(results)
    print_summary_table(results)


if __name__ == "__main__":
    main()
