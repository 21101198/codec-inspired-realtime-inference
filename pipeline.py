
import cv2
import torch
import numpy as np
import time
import collections
import csv
import argparse
from dataclasses import dataclass
from typing import List, Deque

from object_level_scheduler import ObjectSelectiveScheduler


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
    n_objects: int = 0
    n_fast: int = 0
    roi_ratio: float = 0.0

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
        self.last_gray = None   # 객체 스케줄러가 재사용

    def compute_score(self, frame_bgr, mean_conf=1.0):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        self.last_gray = gray
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


def infer_model_type(model_path):
    """--model-type이 명시되지 않았을 때 파일명으로 yolov5/yolov8 자동 판별."""
    name = model_path.lower()
    if "yolov8" in name or "yolo11" in name or "yolov9" in name or "yolov10" in name:
        return "yolov8"
    return "yolov5"


class RealtimePipeline:
    ROI_SIZES = (320, 640)

    def __init__(self, model_path="yolov5s.pt", source=0,
                 max_frames=500, output_csv="latency_log.csv", conf_threshold=0.4,
                 model_type=None, mode="frame", record_path=None, show=True):
        self.max_frames = max_frames
        self.record_path = record_path   # 시연 영상 저장 경로 (None이면 저장 안 함)
        self.show = show
        self.writer = None
        self.mode = mode
        self.output_csv = output_csv
        self.conf_threshold = conf_threshold
        self.model_type = model_type or infer_model_type(model_path)

        print(f"[Pipeline] 모델 로딩 중... (type={self.model_type})")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Pipeline] 디바이스: {self.device.upper()}")
        if self.device == "cpu":
            print("[Pipeline] NVIDIA GPU(CUDA)를 찾지 못해 CPU 모드로 실행합니다. 속도가 매우 느릴 수 있습니다.")

        if self.model_type == "yolov8":
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.model.to(self.device)
        else:
            from model_loader import load_yolov5
            self.model = load_yolov5(model_path)
            self.model.to(self.device)
            self.model.conf = conf_threshold

        self.profiler  = FrameProfiler()
        self.detector  = SceneChangeDetector(threshold=0.60)
        if mode == "object":
            self.scheduler = ObjectSelectiveScheduler(max_p_streak=4)
        else:
            self.scheduler = CodecInspiredScheduler(max_p_streak=4)
        print(f"[Pipeline] 스케줄링 모드: {mode}")
        self.predictor = TailLatencyPredictor(window=30, calibration_frames=30)

        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"[Pipeline] 소스를 열 수 없음: {source}")

        self.log: List[FrameLatency] = []

    def _detect(self, images, size=None):
        """이미지 리스트 추론 -> 이미지별 (N, 6) 배열 [x1, y1, x2, y2, conf, cls]"""
        if self.model_type == "yolov8":
            kw = {"imgsz": size} if size else {}
            results = self.model(images, verbose=False, conf=self.conf_threshold, **kw)
            return [r.boxes.data.cpu().numpy() for r in results]
        results = self.model(images, size=size) if size else self.model(images)
        return [d.cpu().numpy() for d in results.xyxy]

    def _infer_full(self, frame_rgb):
        dets = self._detect(frame_rgb)[0]
        boxes = dets[:, :4].tolist() if len(dets) else []
        confs = dets[:, 4].tolist() if len(dets) else [1.0]
        mean_conf = float(np.mean(confs)) if confs else 1.0
        return boxes, mean_conf

    def _infer_rois(self, frame_rgb, rois):
        """S-frame: ROI 크롭만 추론 후 원본 좌표로 복원.

        전체 추론(640)과 같은 축척을 유지해 결과가 I-frame과 일관되게 하고,
        크롭들은 같은 크기의 정사각형 캔버스에 붙여 한 번의 배치로 추론한다.
        입력 크기는 ROI_SIZES 두 가지로 고정해 새 shape의 첫 호출 지연(~20ms)을 없앤다.
        """
        scale = 640 / max(frame_rgb.shape[:2])
        longest = max(max(y1 - y0, x1 - x0) for x0, y0, x1, y1 in rois)
        size = next(s for s in self.ROI_SIZES if longest * scale <= s or s == self.ROI_SIZES[-1])
        side = int(round(size / scale))
        canvases = []
        for x0, y0, x1, y1 in rois:
            canvas = np.full((side, side, 3), 114, np.uint8)
            crop = frame_rgb[y0:min(y1, y0 + side), x0:min(x1, x0 + side)]
            canvas[:crop.shape[0], :crop.shape[1]] = crop
            canvases.append(canvas)
        out = []
        for (x0, y0, x1, y1), dets in zip(rois, self._detect(canvases, size=size)):
            w, h = min(x1 - x0, side), min(y1 - y0, side)
            for d in dets:
                bx = [min(d[0], w) + x0, min(d[1], h) + y0, min(d[2], w) + x0, min(d[3], h) + y0]
                out.append((bx, float(d[4]), int(d[5])))
        return out

    def _warmup_rois(self, frame_rgb, max_batch=4):
        """S-frame에서 쓰일 (크기, 배치) 조합을 미리 한 번씩 실행해 첫 호출 지연 제거.

        빈 캔버스는 검출 후보가 없어 NMS 경로가 워밍업되지 않으므로 실제 프레임 크롭을 쓴다.
        """
        H, W = frame_rgb.shape[:2]
        scale = 640 / max(H, W)
        for size in self.ROI_SIZES:
            side = int(min(size / scale * 0.9, H, W))
            tiles = [[x, y, x + side, y + side]
                     for y in (0, H - side) for x in (0, W - side)]
            for n in range(1, max_batch + 1):
                self._infer_rois(frame_rgb, tiles[:n])
        if self.device == "cuda":
            torch.cuda.synchronize()

    def _infer_light(self, prev_boxes):
        """P-frame: 직전 I-frame 결과 재사용 (GPU 연산 없음)"""
        if not prev_boxes:
            return [], self.scheduler.get_last_conf()
        return prev_boxes, self.scheduler.get_last_conf()

    def run(self):
        print(f"\n[Pipeline] 실행 시작 | 최대 프레임: {self.max_frames}\n")
        frame_idx = 0
        i_count = 0
        s_count = 0
        p_count = 0
        alert_count = 0

        while frame_idx < self.max_frames:
            ret, frame_bgr = self.cap.read()
            if not ret:
                break

            lat = FrameLatency(frame_idx=frame_idx)

            self.profiler.start("pre", gpu_sync=False)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            lat.preprocess_ms = self.profiler.end("pre", gpu_sync=False)

            if frame_idx == 0 and self.mode == "object":
                # 측정 구간 밖에서 첫 프레임으로 ROI 추론 경로 워밍업
                self._warmup_rois(frame_rgb)

            scene_score = self.detector.compute_score(
                frame_bgr, mean_conf=self.scheduler.get_last_conf()
            )
            lat.scene_score = round(scene_score, 4)

            ALERT_THRESHOLD_MS = 50.0
            alert = False
            for past in reversed(self.log[-10:]):
                if past.frame_type == "I":
                    if past.infer_ms > ALERT_THRESHOLD_MS:
                        alert = True
                    break
            lat.alert = alert
            if alert:
                alert_count += 1

            plan = None
            if self.mode == "object":
                # 박스별 광류 속도 추정 비용은 전처리 시간에 포함
                self.profiler.start("plan", gpu_sync=False)
                plan = self.scheduler.plan(frame_bgr, scene_score, alert,
                                           gray=self.detector.last_gray)
                lat.preprocess_ms += self.profiler.end("plan", gpu_sync=False)
                frame_type = plan.frame_type
                lat.n_fast = plan.n_fast
                lat.roi_ratio = round(plan.roi_ratio, 4)
            else:
                frame_type = self.scheduler.decide_frame_type(scene_score, alert)
            lat.frame_type = frame_type

            self.profiler.start("htod", gpu_sync=False)
            if self.device == "cuda":
                _ = torch.from_numpy(frame_rgb.copy()).to(self.device)
            lat.htod_ms = self.profiler.end("htod", gpu_sync=False)

            if plan is not None:
                self.profiler.start("infer", gpu_sync=plan.frame_type != "P")
                if plan.frame_type == "I":
                    d = self._detect(frame_rgb)[0]
                    dets = [(r[:4].tolist(), float(r[4]), int(r[5])) for r in d]
                    self.scheduler.commit(plan, dets)
                    i_count += 1
                elif plan.frame_type == "S":
                    self.scheduler.commit(plan, self._infer_rois(frame_rgb, plan.rois))
                    s_count += 1
                else:
                    p_count += 1
                lat.infer_ms = self.profiler.end("infer", gpu_sync=plan.frame_type != "P")
                boxes = self.scheduler.get_last_boxes()
            elif frame_type == "I":
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

            lat.n_objects = len(boxes)
            lat.compute_total()
            self.log.append(lat)

            if frame_idx % 10 == 0:
                stats = self.predictor.get_stats()
                print(
                    f"Frame {frame_idx:4d} | {frame_type}-frame | "
                    f"Total={lat.total_ms:7.2f}ms | "
                    f"Infer={lat.infer_ms:6.2f}ms | "
                    f"Scene={scene_score:.2f} | "
                    f"Obj={lat.n_fast}/{lat.n_objects} fast | "
                    f"Alert={'YES' if alert else 'no ':3s} | "
                    f"P99={stats.get('p99_ms', 0):.1f}ms"
                )

            # 탐지 박스 표시 (지연 측정 이후라 측정값에 영향 없음)
            # I-frame: 새로 추론한 박스(주황) / P-frame: 직전 결과 재사용 박스(초록)
            box_color = (0, 200, 0) if frame_type == "P" else (0, 140, 255)
            for b in boxes:
                x0, y0, x1, y1 = (int(v) for v in b[:4])
                cv2.rectangle(frame_bgr, (x0, y0), (x1, y1), box_color, 2)
            if plan is not None and frame_type == "S":
                for x0, y0, x1, y1 in plan.rois:
                    cv2.rectangle(frame_bgr, (x0, y0), (x1, y1), (0, 165, 255), 2)
            color = {"P": (0, 200, 0), "S": (0, 165, 255)}.get(frame_type, (0, 80, 255))
            cv2.putText(frame_bgr, f"{frame_type}-frame {lat.total_ms:.1f}ms",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            if alert:
                cv2.putText(frame_bgr, "ALERT: Tail Latency!",
                            (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if self.record_path:
                self._record_frame(frame_bgr, lat, i_count, p_count, s_count, alert_count)
            if self.show:
                cv2.imshow("Pipeline", frame_bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1

        self._finalize(i_count, p_count, alert_count, s_count)

    def _record_frame(self, frame_bgr, lat, i_count, p_count, s_count, alert_count):
        """시연 영상용: 처리한 프레임 옆에 실시간 통계 패널을 붙여 저장.
        지연 측정이 끝난 뒤 호출되므로 측정값에는 영향이 없다."""
        h, w = frame_bgr.shape[:2]
        pw = 420
        panel = np.full((h, pw, 3), 30, dtype=np.uint8)
        totals = [l.total_ms for l in self.log]
        n = len(totals)
        p50 = float(np.percentile(totals, 50)) if n else 0.0
        p99 = float(np.percentile(totals, 99)) if n >= 2 else 0.0
        fps = 1000.0 / float(np.mean(totals)) if n else 0.0
        ft = lat.frame_type
        ft_color = {"P": (0, 200, 0), "S": (0, 165, 255)}.get(ft, (0, 140, 255))
        rows = [
            ("Frame", f"{lat.frame_idx}", (230, 230, 230)),
            ("Frame type", f"{ft}-frame", ft_color),
            ("Latency", f"{lat.total_ms:.2f} ms", (230, 230, 230)),
            ("Infer", f"{lat.infer_ms:.2f} ms", (230, 230, 230)),
            ("P50", f"{p50:.2f} ms", (230, 230, 230)),
            ("P99 (tail)", f"{p99:.2f} ms", (80, 80, 255)),
            ("Processing FPS", f"{fps:.1f}", (255, 200, 80)),
            ("I / P" + (" / S" if self.mode == "object" else ""),
             f"{i_count} / {p_count}" + (f" / {s_count}" if self.mode == "object" else ""), (230, 230, 230)),
            ("Alerts", f"{alert_count}", (230, 230, 230)),
        ]
        y = 50
        cv2.putText(panel, "Codec-Inspired Pipeline", (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        y += 20
        cv2.line(panel, (18, y), (pw - 18, y), (90, 90, 90), 1)
        for label, val, col in rows:
            y += 62
            cv2.putText(panel, label, (18, y - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1)
            cv2.putText(panel, val, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.95, col, 2)
        cv2.putText(panel, f"mode: {self.mode}", (18, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1)
        cv2.putText(panel, "orange=I (new)  green=P (reuse)", (18, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        out = np.hstack([frame_bgr, panel])
        if self.writer is None:
            src_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            if not (1.0 <= src_fps <= 120.0):
                src_fps = 30.0
            self.writer = cv2.VideoWriter(self.record_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                          src_fps, (out.shape[1], out.shape[0]))
            print(f"[Pipeline] 시연 영상 저장 시작 -> {self.record_path} ({src_fps:.1f} fps)")
        self.writer.write(out)

    def _finalize(self, i_count, p_count, alert_count, s_count=0):
        self.cap.release()
        if self.writer is not None:
            self.writer.release()
            print(f"[Pipeline] 시연 영상 저장 완료 -> {self.record_path}")
        cv2.destroyAllWindows()

        if not self.log:
            print("[Pipeline] 기록 없음")
            return

        with open(self.output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["frame_idx", "frame_type", "scene_score", "alert",
                             "preprocess_ms", "htod_ms", "infer_ms", "dtoh_ms", "total_ms",
                             "n_objects", "n_fast", "roi_ratio"])
            for lat in self.log:
                writer.writerow([lat.frame_idx, lat.frame_type,
                                  lat.scene_score, lat.alert,
                                  round(lat.preprocess_ms, 3), round(lat.htod_ms, 3),
                                  round(lat.infer_ms, 3), round(lat.dtoh_ms, 3),
                                  round(lat.total_ms, 3),
                                  lat.n_objects, lat.n_fast, lat.roi_ratio])

        print(f"\n[Pipeline] CSV 저장 완료 -> {self.output_csv}")

        totals = [l.total_ms for l in self.log]
        i_infers = [l.infer_ms for l in self.log if l.frame_type == "I"]
        s_infers = [l.infer_ms for l in self.log if l.frame_type == "S"]
        p_infers = [l.infer_ms for l in self.log if l.frame_type == "P"]

        print("\n" + "="*60)
        print("  최종 결과 요약")
        print("="*60)
        print(f"  총 프레임       : {len(self.log)}")
        print(f"  I-frame 횟수    : {i_count}  ({i_count/len(self.log)*100:.1f}%)")
        if self.mode == "object":
            print(f"  S-frame 횟수    : {s_count}  ({s_count/len(self.log)*100:.1f}%)  <- 빠른 객체 ROI만 추론")
        print(f"  P-frame 횟수    : {p_count}  ({p_count/len(self.log)*100:.1f}%)")
        print(f"  경보 발령 횟수   : {alert_count}")
        print("-"*60)
        print(f"  Total Latency")
        print(f"    P50  : {np.percentile(totals,50):.2f} ms")
        print(f"    P90  : {np.percentile(totals,90):.2f} ms")
        print(f"    P99  : {np.percentile(totals,99):.2f} ms  <- Tail Latency")
        print(f"    Mean : {np.mean(totals):.2f} ms")
        print(f"    Std  : {np.std(totals):.2f} ms")
        print(f"    FPS  : {1000/np.mean(totals):.1f}")
        print("-"*60)
        if i_infers:
            print(f"  I-frame 평균 추론 : {np.mean(i_infers):.2f} ms")
        if s_infers:
            roi = [l.roi_ratio for l in self.log if l.frame_type == "S"]
            print(f"  S-frame 평균 추론 : {np.mean(s_infers):.2f} ms  (평균 ROI 면적 {np.mean(roi)*100:.1f}%)")
        if p_infers:
            print(f"  P-frame 평균 추론 : {np.mean(p_infers):.2f} ms")
        if i_infers and p_infers:
            speedup = np.mean(i_infers) / max(np.mean(p_infers), 0.001)
            print(f"  P-frame 속도 향상 : {speedup:.1f}x")
        print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0")
    parser.add_argument("--model",  type=str, default="yolov5s.pt")
    parser.add_argument("--model-type", type=str, choices=["yolov5", "yolov8"], default=None,
                         help="모델 종류 (생략 시 --model 파일명에서 자동 추론)")
    parser.add_argument("--frames", type=int, default=500)
    parser.add_argument("--output", type=str, default="latency_log.csv")
    parser.add_argument("--conf",   type=float, default=0.4)
    parser.add_argument("--mode",   type=str, choices=["object", "frame"], default="frame",
                         help="object: 객체별 속도 기반 선택 추론, frame: 프레임 단위 I/P")
    parser.add_argument("--record", type=str, default=None,
                         help="처리한 프레임과 실시간 통계를 mp4로 저장 (시연 영상용, 측정값에 영향 없음)")
    parser.add_argument("--no-show", action="store_true", help="영상 창을 띄우지 않음")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source

    model_type = args.model_type or infer_model_type(args.model)
    model_path = args.model
    if model_type == "yolov8" and model_path == "yolov5s.pt":
        model_path = "yolov8s.pt"

    pipeline = RealtimePipeline(
        model_path=model_path,
        source=source,
        max_frames=args.frames,
        output_csv=args.output,
        conf_threshold=args.conf,
        model_type=model_type,
        mode=args.mode,
        record_path=args.record,
        show=not args.no_show,
    )
    pipeline.run()