"""onnxruntime-gpu로 FP32 / TensorRT FP16 / TensorRT INT8 추론 속도 비교."""

import argparse
import os
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import cv2
import numpy as np

from trt_ep_setup import enable_gpu_dll_search, trt_providers

enable_gpu_dll_search()
import onnxruntime as ort  # noqa: E402

INPUT_NAME = "images"


def preprocess(frame_bgr):
    img = cv2.resize(frame_bgr, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.ascontiguousarray(img.transpose(2, 0, 1)[None])


def load_frames(video, n):
    frames = []
    if video and os.path.exists(video):
        cap = cv2.VideoCapture(video)
        while len(frames) < n:
            ok, f = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            frames.append(preprocess(f))
        cap.release()
    if not frames:
        frames = [np.random.rand(1, 3, 640, 640).astype(np.float32) for _ in range(n)]
    return frames


def bench(session, frames, label, warmup=15):
    for f in frames[:warmup]:
        session.run(None, {INPUT_NAME: f})
    lat = []
    for f in frames:
        t = time.perf_counter()
        session.run(None, {INPUT_NAME: f})
        lat.append((time.perf_counter() - t) * 1000.0)
    a = np.array(lat)
    row = dict(label=label, mean=a.mean(), p50=np.percentile(a, 50),
              p90=np.percentile(a, 90), p99=np.percentile(a, 99), fps=1000.0 / a.mean())
    print(f"  [{label}]  mean {row['mean']:6.2f} ms | p50 {row['p50']:6.2f} | "
          f"p90 {row['p90']:6.2f} | p99 {row['p99']:6.2f} | {row['fps']:6.1f} FPS")
    return row


def make_session(onnx_path, providers, tag):
    try:
        s = ort.InferenceSession(onnx_path, providers=providers)
        got = s.get_providers()[0]
        want = providers[0] if isinstance(providers[0], str) else providers[0][0]
        if got != want:
            print(f"  ! {tag}: {want} 로딩 실패 → {got} 로 폴백")
        return s
    except Exception as e:  # noqa
        print(f"  ! {tag}: 세션 생성 실패: {e!r}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp32-onnx", default="yolov5s_static.onnx")
    ap.add_argument("--int8-onnx", default="yolov5s_int8.onnx")
    ap.add_argument("--calib-table", default="yolov5s_calib.flatbuffers")
    ap.add_argument("--video", default="videos/highspeed.mp4")
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--cache", default="trt_cache")
    ap.add_argument("--int8-cache", default="trt_cache_int8")
    ap.add_argument("--cuda-int8", action="store_true", help="CUDA EP INT8(QDQ) 도 측정")
    args = ap.parse_args()

    print(f"ORT {ort.__version__} | providers: {ort.get_available_providers()}")
    print(f"프레임 로딩: {args.video} x{args.frames}")
    frames = load_frames(args.video, args.frames)

    rows = []

    print("\n[1] CUDA EP  FP32")
    s = make_session(args.fp32_onnx, ["CUDAExecutionProvider", "CPUExecutionProvider"], "CUDA FP32")
    if s:
        rows.append(bench(s, frames, "CUDA FP32"))

    print("\n[2] TensorRT EP  FP16   (첫 실행 시 엔진 빌드 1~3분)")
    s = make_session(args.fp32_onnx,
                     trt_providers(args.cache, fp16=True), "TRT FP16")
    if s:
        rows.append(bench(s, frames, "TRT FP16"))

    if os.path.exists(args.int8_onnx):
        print("\n[3] TensorRT EP  INT8 (QDQ)   (첫 실행 시 엔진 빌드 2~5분)")
        s = make_session(args.int8_onnx,
                         trt_providers(args.int8_cache, fp16=True, int8=True), "TRT INT8 QDQ")
        if s:
            rows.append(bench(s, frames, "TRT INT8 (QDQ)"))

        if args.cuda_int8:
            print("\n[3b] CUDA EP  INT8 (QDQ)")
            s = make_session(args.int8_onnx, ["CUDAExecutionProvider", "CPUExecutionProvider"], "CUDA INT8")
            if s:
                rows.append(bench(s, frames, "CUDA INT8 (QDQ)"))
    else:
        print(f"\n[3] 건너뜀: {args.int8_onnx} 없음 → 먼저 `python ort_int8_quantize.py` 실행")

    if os.path.exists(args.calib_table):
        print("\n[4] TensorRT EP  INT8 (calibration table + 원본 FP32 모델)")
        # TRT EP 는 테이블 파일을 CWD 또는 engine cache 경로에서 파일명으로 찾는다.
        tbl_cache = args.int8_cache + "_tbl"
        os.makedirs(tbl_cache, exist_ok=True)
        import shutil
        shutil.copy(args.calib_table, os.path.join(tbl_cache, os.path.basename(args.calib_table)))
        s = make_session(
            args.fp32_onnx,
            trt_providers(tbl_cache, fp16=True, int8=True,
                          int8_calib_table=os.path.basename(args.calib_table)),
            "TRT INT8 table")
        if s:
            rows.append(bench(s, frames, "TRT INT8 (calib table)"))

    if rows:
        base = next((r for r in rows if r["label"] == "CUDA FP32"), rows[0])
        print("\n" + "=" * 74)
        print(f"  {'설정':<26}{'Mean(ms)':>10}{'P99(ms)':>10}{'FPS':>9}{'vs FP32':>10}")
        print("  " + "-" * 70)
        for r in rows:
            print(f"  {r['label']:<26}{r['mean']:>10.2f}{r['p99']:>10.2f}"
                  f"{r['fps']:>9.1f}{base['mean'] / r['mean']:>9.2f}x")
        print("=" * 74)


if __name__ == "__main__":
    main()
