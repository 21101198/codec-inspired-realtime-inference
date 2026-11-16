"""베이스라인: 스케줄러 없이 모든 프레임 전체 추론."""
import cv2
import torch
import numpy as np
import time
import csv
import argparse
from dataclasses import dataclass
from typing import List


@dataclass
class FrameLatency:
    frame_idx: int
    preprocess_ms: float = 0.0
    htod_ms: float = 0.0
    infer_ms: float = 0.0
    dtoh_ms: float = 0.0
    total_ms: float = 0.0

    def compute_total(self):
        self.total_ms = (self.preprocess_ms + self.htod_ms +
                         self.infer_ms + self.dtoh_ms)


class BaselinePipeline:
    """모든 프레임을 YOLO 전체 추론으로 처리하는 베이스라인 파이프라인."""
    def __init__(self, model_path="yolov5s.pt", source=0,
                 max_frames=500, output_csv="baseline_log.csv",
                 conf_threshold=0.4):
        self.max_frames = max_frames
        self.output_csv = output_csv

        print("[Baseline] 모델 로딩 중...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Baseline] 디바이스: {self.device.upper()}")

        # 제안 방법(pipeline.py)과 같은 로더 사용: yolov5/ 폴더가 있으면 오프라인 로드
        from model_loader import load_yolov5
        self.model = load_yolov5(model_path)
        self.model.to(self.device)
        self.model.conf = conf_threshold

        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"[Baseline] 소스를 열 수 없음: {source}")

        self.log: List[FrameLatency] = []

    def _sync(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def run(self):
        print(f"\n[Baseline] 실행 시작 | 최대 프레임: {self.max_frames}\n")
        frame_idx = 0

        while frame_idx < self.max_frames:
            ret, frame_bgr = self.cap.read()
            if not ret:
                break

            lat = FrameLatency(frame_idx=frame_idx)

            t = time.perf_counter()
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            lat.preprocess_ms = (time.perf_counter() - t) * 1000.0

            t = time.perf_counter()
            if self.device == "cuda":
                _ = torch.from_numpy(frame_rgb.copy()).to(self.device)
            lat.htod_ms = (time.perf_counter() - t) * 1000.0

            self._sync()
            t = time.perf_counter()
            results = self.model(frame_rgb)
            self._sync()
            lat.infer_ms = (time.perf_counter() - t) * 1000.0

            t = time.perf_counter()
            _ = results.xyxy[0].cpu().numpy()
            lat.dtoh_ms = (time.perf_counter() - t) * 1000.0

            lat.compute_total()
            self.log.append(lat)

            if frame_idx % 10 == 0:
                print(f"Frame {frame_idx:4d} | ALL I-frame | "
                      f"Total={lat.total_ms:7.2f}ms | "
                      f"Infer={lat.infer_ms:6.2f}ms")

            frame_idx += 1

        self._finalize()

    def _finalize(self):
        self.cap.release()
        cv2.destroyAllWindows()

        if not self.log:
            print("[Baseline] 기록 없음")
            return

        with open(self.output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["frame_idx", "preprocess_ms", "htod_ms",
                             "infer_ms", "dtoh_ms", "total_ms"])
            for lat in self.log:
                writer.writerow([lat.frame_idx,
                                  round(lat.preprocess_ms, 3),
                                  round(lat.htod_ms, 3),
                                  round(lat.infer_ms, 3),
                                  round(lat.dtoh_ms, 3),
                                  round(lat.total_ms, 3)])

        print(f"\n[Baseline] CSV 저장 완료 -> {self.output_csv}")

        totals = [l.total_ms for l in self.log]
        infers = [l.infer_ms for l in self.log]

        print("\n" + "="*60)
        print("  베이스라인 결과 (스케줄러 없음 / 전체 추론)")
        print("="*60)
        print(f"  총 프레임       : {len(self.log)}")
        print("  모든 프레임     : I-frame (전체 추론)")
        print("-"*60)
        print("  Total Latency")
        print(f"    P50  : {np.percentile(totals,50):.2f} ms")
        print(f"    P90  : {np.percentile(totals,90):.2f} ms")
        print(f"    P99  : {np.percentile(totals,99):.2f} ms  <- Tail Latency")
        print(f"    Mean : {np.mean(totals):.2f} ms")
        print(f"    Std  : {np.std(totals):.2f} ms")
        print(f"    FPS  : {1000/np.mean(totals):.1f}")
        print("-"*60)
        print("  Infer Latency")
        print(f"    P50  : {np.percentile(infers,50):.2f} ms")
        print(f"    P99  : {np.percentile(infers,99):.2f} ms")
        print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0")
    parser.add_argument("--model",  type=str, default="yolov5s.pt")
    parser.add_argument("--frames", type=int, default=500)
    parser.add_argument("--output", type=str, default="baseline_log.csv")
    parser.add_argument("--conf",   type=float, default=0.4)
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source

    pipeline = BaselinePipeline(
        model_path=args.model,
        source=source,
        max_frames=args.frames,
        output_csv=args.output,
        conf_threshold=args.conf
    )
    pipeline.run()
