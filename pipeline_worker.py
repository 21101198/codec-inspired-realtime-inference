"""pipeline.py의 RealtimePipeline.run() 루프를 PyQt6 QThread 워커로 이식한 모듈."""

import csv
from typing import List, Optional

import cv2
import numpy as np
import torch

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage

from core import (
    FrameLatency,
    FrameProfiler,
    SceneChangeDetector,
    CodecInspiredScheduler,
    TailLatencyPredictor,
)

ALERT_THRESHOLD_MS = 50.0


class PipelineWorker(QThread):
    frame_ready = pyqtSignal(QImage, FrameLatency)
    stats_updated = pyqtSignal(dict)
    finished_run = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, model_path="yolov5s.pt", source=0,
                 max_frames=500, output_csv="latency_log.csv",
                 conf_threshold=0.4, parent=None):
        super().__init__(parent)
        self.model_path = model_path
        self.source = source
        self.max_frames = max_frames
        self.output_csv = output_csv
        self.conf_threshold = conf_threshold

        self._stop_requested = False

        self.model = None
        self.device = "cpu"
        self.cap: Optional[cv2.VideoCapture] = None

        self.profiler = FrameProfiler()
        self.detector = SceneChangeDetector(threshold=0.60)
        self.scheduler = CodecInspiredScheduler(max_p_streak=4)
        self.predictor = TailLatencyPredictor(window=30, calibration_frames=30)

        self.log: List[FrameLatency] = []

    def stop(self):
        """UI 스레드에서 호출. 다음 루프 반복 시작 시 안전하게 종료된다."""
        self._stop_requested = True

    def _load_model(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        from model_loader import load_yolov5
        self.model = load_yolov5(self.model_path)
        self.model.to(self.device)
        self.model.conf = self.conf_threshold

    def _infer_full(self, frame_rgb):
        results = self.model(frame_rgb)
        dets = results.xyxy[0].cpu().numpy()
        boxes = dets[:, :4].tolist() if len(dets) else []
        confs = dets[:, 4].tolist() if len(dets) else [1.0]
        mean_conf = float(np.mean(confs)) if confs else 1.0
        return boxes, mean_conf

    def _infer_light(self, prev_boxes):
        """P-frame: 직전 I-frame 결과 재사용 (GPU 연산 없음)"""
        if not prev_boxes:
            return [], self.scheduler.get_last_conf()
        return prev_boxes, self.scheduler.get_last_conf()

    @staticmethod
    def _to_qimage(frame_bgr: np.ndarray) -> QImage:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        qimg = QImage(frame_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        return qimg.copy()

    def run(self):
        try:
            self._load_model()
        except Exception as e:
            self.error.emit(f"모델 로딩 실패: {e}")
            return

        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            self.error.emit(f"소스를 열 수 없음: {self.source}")
            return

        frame_idx = 0
        i_count = 0
        p_count = 0
        alert_count = 0

        while frame_idx < self.max_frames and not self._stop_requested:
            ret, frame_bgr = self.cap.read()
            if not ret:
                break

            lat = FrameLatency(frame_idx=frame_idx)

            self.profiler.start("pre", gpu_sync=False)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            lat.preprocess_ms = self.profiler.end("pre", gpu_sync=False)

            scene_score = self.detector.compute_score(
                frame_bgr, mean_conf=self.scheduler.get_last_conf()
            )
            lat.scene_score = round(scene_score, 4)

            alert = False
            for past in reversed(self.log[-10:]):
                if past.frame_type == "I":
                    if past.infer_ms > ALERT_THRESHOLD_MS:
                        alert = True
                    break
            lat.alert = alert
            if alert:
                alert_count += 1

            frame_type = self.scheduler.decide_frame_type(scene_score, alert)
            lat.frame_type = frame_type

            self.profiler.start("htod", gpu_sync=False)
            if self.device == "cuda":
                _ = torch.from_numpy(frame_rgb.copy()).to(self.device)
            lat.htod_ms = self.profiler.end("htod", gpu_sync=False)

            if frame_type == "I":
                self.profiler.start("infer", gpu_sync=True)
                boxes, mean_conf = self._infer_full(frame_rgb)
                lat.infer_ms = self.profiler.end("infer", gpu_sync=True)
                self.scheduler.update_last_result(boxes, mean_conf)
                i_count += 1
            else:
                self.profiler.start("infer", gpu_sync=False)
                boxes, mean_conf = self._infer_light(self.scheduler.get_last_boxes())
                lat.infer_ms = self.profiler.end("infer", gpu_sync=False)
                p_count += 1

            self.profiler.start("dtoh", gpu_sync=False)
            _ = boxes
            lat.dtoh_ms = self.profiler.end("dtoh", gpu_sync=False)

            lat.compute_total()
            self.log.append(lat)
            self.predictor.update(lat.total_ms)

            # 탐지 박스 표시 (지연 측정 이후라 측정값에 영향 없음)
            # I-frame: 새로 추론한 박스(주황) / P-frame: 직전 결과 재사용 박스(초록)
            box_color = (0, 200, 0) if frame_type == "P" else (0, 140, 255)
            for b in boxes:
                x0, y0, x1, y1 = (int(v) for v in b[:4])
                cv2.rectangle(frame_bgr, (x0, y0), (x1, y1), box_color, 2)

            color = (0, 200, 0) if frame_type == "P" else (0, 80, 255)
            cv2.putText(frame_bgr, f"{frame_type}-frame {lat.total_ms:.1f}ms",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            if alert:
                cv2.putText(frame_bgr, "ALERT: Tail Latency!",
                            (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            self.frame_ready.emit(self._to_qimage(frame_bgr), lat)

            if frame_idx % 10 == 0:
                stats = self.predictor.get_stats()
                stats.update({
                    "frame_idx": frame_idx,
                    "frame_type": frame_type,
                    "i_count": i_count,
                    "p_count": p_count,
                    "alert_count": alert_count,
                    "total_ms": round(lat.total_ms, 2),
                    "infer_ms": round(lat.infer_ms, 2),
                    "scene_score": lat.scene_score,
                })
                self.stats_updated.emit(stats)

            frame_idx += 1

        summary = self._finalize(i_count, p_count, alert_count)
        self.finished_run.emit(summary)

    def _finalize(self, i_count, p_count, alert_count) -> dict:
        if self.cap is not None:
            self.cap.release()

        if not self.log:
            return {"frame_count": 0}

        with open(self.output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["frame_idx", "frame_type", "scene_score", "alert",
                             "preprocess_ms", "htod_ms", "infer_ms", "dtoh_ms", "total_ms"])
            for lat in self.log:
                writer.writerow([lat.frame_idx, lat.frame_type,
                                  lat.scene_score, lat.alert,
                                  round(lat.preprocess_ms, 3), round(lat.htod_ms, 3),
                                  round(lat.infer_ms, 3), round(lat.dtoh_ms, 3),
                                  round(lat.total_ms, 3)])

        totals = [l.total_ms for l in self.log]
        i_infers = [l.infer_ms for l in self.log if l.frame_type == "I"]
        p_infers = [l.infer_ms for l in self.log if l.frame_type == "P"]

        summary = {
            "frame_count": len(self.log),
            "i_count": i_count,
            "p_count": p_count,
            "alert_count": alert_count,
            "p50_ms": float(np.percentile(totals, 50)),
            "p90_ms": float(np.percentile(totals, 90)),
            "p99_ms": float(np.percentile(totals, 99)),
            "mean_ms": float(np.mean(totals)),
            "std_ms": float(np.std(totals)),
            "fps": float(1000 / np.mean(totals)) if np.mean(totals) > 0 else 0.0,
            "i_infer_mean_ms": float(np.mean(i_infers)) if i_infers else None,
            "p_infer_mean_ms": float(np.mean(p_infers)) if p_infers else None,
            "output_csv": self.output_csv,
        }
        if i_infers and p_infers:
            summary["p_frame_speedup"] = summary["i_infer_mean_ms"] / max(summary["p_infer_mean_ms"], 0.001)

        return summary
