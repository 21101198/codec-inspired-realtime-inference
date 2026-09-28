"""pipeline.py에서 분리한 UI-비의존 순수 로직 모듈."""

import time
import collections
import numpy as np
import cv2
import torch
from dataclasses import dataclass
from typing import List, Deque


@dataclass
class FrameLatency:
    frame_idx: int
    preprocess_ms: float = 0.0
    htod_ms: float = 0.0
    infer_ms: float = 0.0
    dtoh_ms: float = 0.0
    total_ms: float = 0.0
    frame_type: str = "I"
    scene_score: float = 0.0
    alert: bool = False

    def compute_total(self):
        self.total_ms = (self.preprocess_ms + self.htod_ms +
                         self.infer_ms + self.dtoh_ms)


class FrameProfiler:
    def __init__(self):
        self._t = {}

    def start(self, tag: str, gpu_sync: bool = True):
        if gpu_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        self._t[tag] = time.perf_counter()

    def end(self, tag: str, gpu_sync: bool = True) -> float:
        if gpu_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        return (time.perf_counter() - self._t[tag]) * 1000.0


class SceneChangeDetector:
    def __init__(self, pixel_weight=0.6, conf_weight=0.4, threshold=0.35):
        self.w1 = pixel_weight
        self.w2 = conf_weight
        self.threshold = threshold
        self._prev_gray = None

    def compute_score(self, frame_bgr, mean_conf=1.0):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self._prev_gray is None:
            self._prev_gray = gray
            return 1.0
        diff = cv2.absdiff(gray, self._prev_gray)
        pixel_change_rate = np.count_nonzero(diff > 15) / diff.size
        self._prev_gray = gray
        score = self.w1 * pixel_change_rate + self.w2 * (1.0 - mean_conf)
        return float(np.clip(score, 0.0, 1.0))

    def is_scene_change(self, score):
        return score >= self.threshold


class CodecInspiredScheduler:
    def __init__(self, max_p_streak=8):
        self.max_p_streak = max_p_streak
        self._p_streak = 0
        self._last_boxes = []
        self._last_conf = 1.0

    def decide_frame_type(self, scene_score, alert=False):
        force_i = (self._p_streak >= self.max_p_streak or alert)
        if force_i or scene_score >= 0.35:
            self._p_streak = 0
            return "I"
        else:
            self._p_streak += 1
            return "P"

    def update_last_result(self, boxes, mean_conf):
        self._last_boxes = boxes
        self._last_conf = mean_conf

    def get_last_conf(self):
        return self._last_conf

    def get_last_boxes(self):
        return self._last_boxes


class TailLatencyPredictor:
    def __init__(self, window=30, calibration_frames=30):
        self.window = window
        self.calibration_frames = calibration_frames
        self._history: Deque[float] = collections.deque(maxlen=window)
        self._calib_data: List[float] = []
        self._baseline_std = None
        self._threshold_ms = None

    def update(self, total_ms):
        if self._baseline_std is None:
            self._calib_data.append(total_ms)
            if len(self._calib_data) >= self.calibration_frames:
                arr = np.array(self._calib_data)
                self._baseline_std = float(np.std(arr))
                baseline_mean = float(np.mean(arr))
                self._threshold_ms = baseline_mean + self._baseline_std
                print(f"[Predictor] Calibration 완료 | mean={baseline_mean:.1f}ms std={self._baseline_std:.1f}ms threshold={self._threshold_ms:.1f}ms")
            return False

        self._history.append(total_ms)
        return total_ms > self._threshold_ms

    def get_stats(self):
        h = list(self._history)
        if not h:
            return {"p50_ms": 0, "p90_ms": 0, "p99_ms": 0, "mean_ms": 0}
        return {
            "p50_ms":  round(float(np.percentile(h, 50)), 2),
            "p90_ms":  round(float(np.percentile(h, 90)), 2),
            "p99_ms":  round(float(np.percentile(h, 99)), 2),
            "mean_ms": round(float(np.mean(h)), 2),
        }
