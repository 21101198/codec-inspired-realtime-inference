"""캡스톤디자인 통합 실행기 (영상 플레이어 형태).

- 오른쪽 목록에서 영상을 고르면 YOLOv5 탐지 결과를 박스로 그리며 재생한다.
- 재생 중 [제안 방식 | 기존 방식]을 바꿔 FPS/지연 변화를 바로 비교할 수 있다.
  제안 방식: 장면 변화가 작으면 GPU 추론을 건너뛰고 직전 결과를 재사용 (I/P-frame)
  기존 방식: 모든 프레임을 GPU로 추론
- [비교 실험]은 baseline.py / pipeline.py / compare_plot.py 를 그대로 실행해
  같은 영상을 두 방식으로 측정하고 결과표와 그래프를 보여준다.

지연 측정 구간은 pipeline.py 와 같다 (전처리 + 전송 + 추론 + 결과 처리).
박스와 화면 표시는 측정이 끝난 뒤에 하므로 측정값에 포함되지 않는다.
"""

import collections
import csv
import os
import re
import subprocess
import sys
import threading
import time

import cv2
import numpy as np

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize, QRectF, QPointF
from PyQt6.QtGui import (QImage, QPixmap, QIcon, QPainter, QColor, QPainterPath,
                         QFont, QPen)
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QSlider, QListWidget,
    QListWidgetItem, QHBoxLayout, QVBoxLayout, QFrame, QFileDialog,
    QDialog, QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView, QCheckBox,
    QSizePolicy, QPlainTextEdit, QButtonGroup,
)

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "yolov5s.pt")
CONF = 0.4
ALERT_THRESHOLD_MS = 50.0
STATS_WINDOW = 300          # 통계(FPS, P50, P99)는 최근 300프레임 기준
EXPERIMENT_FRAMES = 600     # 비교 실험 최대 프레임 수 (영상이 더 짧으면 영상 길이까지)

ACCENT = "#22d3ee"
ORANGE = "#fb923c"
GREEN = "#4ade80"
BG = "#0b0f14"
PANEL = "#121821"
PANEL2 = "#1a2230"
TEXT = "#e6edf3"
MUTED = "#8b98a9"


# ─────────────────────────────────────────────────────────────
# 영상 목록
# ─────────────────────────────────────────────────────────────
def find_original_video_dir():
    """USB 제출본의 원본실험영상 폴더를 찾는다 (run_experiment_long.bat 과 같은 규칙)."""
    cands = [os.environ.get("VIDEO_DIR", "")]
    for up in range(0, 4):
        cands.append(os.path.join(HERE, *([".."] * up), "원본실험영상"))
    drive = os.path.splitdrive(HERE)[0]
    if drive:
        cands.append(drive + os.sep + "원본실험영상")
    for c in cands:
        if c and os.path.isdir(c) and any(os.path.exists(os.path.join(c, f"{k}.mp4"))
                                           for k in ("crowd", "lowlight", "highspeed")):
            return os.path.normpath(c)
    return None


def default_playlist():
    items = []
    sample = os.path.join(HERE, "sample_video.mp4")
    if os.path.exists(sample):
        items.append(("교차로 샘플", sample))
    vdir = find_original_video_dir()
    if vdir:
        for key, name in (("crowd", "교차로 (원본)"), ("lowlight", "야간 저조도"),
                          ("highspeed", "고속 동작")):
            p = os.path.join(vdir, f"{key}.mp4")
            if os.path.exists(p):
                items.append((name, p))
    return items


def video_info(path):
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if not (1.0 <= fps <= 240.0):
        fps = 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(int(fps), max(n - 1, 0)))
    ok, frame = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()
    cap.release()
    thumb = None
    if ok:
        h, w = frame.shape[:2]
        tw, th = 176, 99
        s = max(tw / w, th / h)
        frame = cv2.resize(frame, (int(w * s) + 1, int(h * s) + 1))
        y0 = (frame.shape[0] - th) // 2
        x0 = (frame.shape[1] - tw) // 2
        thumb = bgr_to_qimage(frame[y0:y0 + th, x0:x0 + tw].copy())
    return n, fps, thumb


def safe_folder_name(path):
    """결과 폴더 이름: Windows에서 쓸 수 없는 문자와 끝의 공백·마침표를 정리."""
    stem = os.path.splitext(os.path.basename(path))[0]
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", stem).strip().rstrip(". ")
    return stem or "video"


def fmt_time(sec):
    sec = max(0, int(round(sec)))
    return f"{sec // 60:02d}:{sec % 60:02d}"


def bgr_to_qimage(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()


# ─────────────────────────────────────────────────────────────
# 아이콘 (QPainter로 직접 그림)
# ─────────────────────────────────────────────────────────────
def make_icon(kind, color=TEXT, size=40):
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(color))
    s = size
    if kind == "play":
        path = QPainterPath()
        path.moveTo(s * 0.34, s * 0.24)
        path.lineTo(s * 0.78, s * 0.50)
        path.lineTo(s * 0.34, s * 0.76)
        path.closeSubpath()
        p.drawPath(path)
    elif kind == "pause":
        p.drawRoundedRect(QRectF(s * 0.28, s * 0.24, s * 0.15, s * 0.52), 2, 2)
        p.drawRoundedRect(QRectF(s * 0.57, s * 0.24, s * 0.15, s * 0.52), 2, 2)
    elif kind == "restart":
        p.drawRoundedRect(QRectF(s * 0.24, s * 0.26, s * 0.08, s * 0.48), 1.5, 1.5)
        path = QPainterPath()
        path.moveTo(s * 0.74, s * 0.26)
        path.lineTo(s * 0.36, s * 0.50)
        path.lineTo(s * 0.74, s * 0.74)
        path.closeSubpath()
        p.drawPath(path)
    elif kind == "add":
        pen = QPen(QColor(color), s * 0.08)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        p.drawLine(QPointF(s * 0.5, s * 0.26), QPointF(s * 0.5, s * 0.74))
        p.drawLine(QPointF(s * 0.26, s * 0.5), QPointF(s * 0.74, s * 0.5))
    elif kind == "cam":
        p.drawRoundedRect(QRectF(s * 0.14, s * 0.30, s * 0.50, s * 0.40), 4, 4)
        path = QPainterPath()
        path.moveTo(s * 0.66, s * 0.44)
        path.lineTo(s * 0.86, s * 0.32)
        path.lineTo(s * 0.86, s * 0.68)
        path.lineTo(s * 0.66, s * 0.56)
        path.closeSubpath()
        p.drawPath(path)
    elif kind == "chart":
        for i, h in enumerate((0.30, 0.50, 0.40, 0.62)):
            x = s * (0.18 + i * 0.17)
            p.drawRoundedRect(QRectF(x, s * (0.80 - h), s * 0.11, s * h), 2, 2)
    p.end()
    return QIcon(pm)


def placeholder_thumb(kind):
    pm = QPixmap(176, 99)
    pm.fill(QColor(PANEL2))
    p = QPainter(pm)
    p.drawPixmap(68, 30, make_icon(kind, MUTED, 40).pixmap(40, 40))
    p.end()
    return pm


# ─────────────────────────────────────────────────────────────
# 추론 엔진 (별도 스레드)
# ─────────────────────────────────────────────────────────────
class Engine(QThread):
    model_ready = pyqtSignal(str)
    opened = pyqtSignal(dict)
    frame_ready = pyqtSignal(QImage, dict)
    ended = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._lock = threading.Lock()
        self._cmds = collections.deque()
        self._quit = False
        self.playing = False
        self.mode = "proposed"
        self.realtime = True
        self.is_camera = False
        self.cap = None
        self.src_fps = 30.0
        self.total_frames = 0
        self.frame_idx = 0
        self.model = None
        self.device = "cpu"
        self.torch = None

    # UI 스레드에서 호출하는 명령
    def command(self, name, value=None):
        with self._lock:
            self._cmds.append((name, value))

    def quit_engine(self):
        self._quit = True

    # ── 내부 ──
    def _reset_state(self):
        from core import SceneChangeDetector, CodecInspiredScheduler
        self.detector = SceneChangeDetector(threshold=0.60)
        self.scheduler = CodecInspiredScheduler(max_p_streak=4)
        self.boxes = []
        self.recent_i_infer = collections.deque(maxlen=10)
        self.totals = collections.deque(maxlen=STATS_WINDOW)
        self.types = collections.deque(maxlen=STATS_WINDOW)

    def _load_model(self):
        import torch
        self.torch = torch
        from model_loader import load_yolov5
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = load_yolov5(MODEL_PATH)
        self.model.to(self.device)
        self.model.conf = CONF
        if self.device == "cuda":
            name = torch.cuda.get_device_name(0)
        else:
            name = "CPU (GPU 없음, 느림)"
        # 첫 추론은 느리므로 측정 전에 한 번 실행 (워밍업)
        self.model(np.zeros((360, 640, 3), dtype=np.uint8))
        return name

    def _sync(self):
        if self.device == "cuda":
            self.torch.cuda.synchronize()

    def _open(self, source):
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            self.cap = None
            self.error.emit(f"영상을 열 수 없습니다: {source}")
            return
        self.is_camera = isinstance(source, int)
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.src_fps = fps if 1.0 <= fps <= 240.0 else 30.0
        self.total_frames = 0 if self.is_camera else int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.frame_idx = 0
        self._reset_state()
        self.opened.emit({"total_frames": self.total_frames, "fps": self.src_fps,
                          "camera": self.is_camera})
        self.playing = True

    def _seek(self, frame_no):
        if self.cap is None or self.is_camera:
            return
        frame_no = int(np.clip(frame_no, 0, max(self.total_frames - 1, 0)))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        self.frame_idx = frame_no
        self._reset_state()
        if not self.playing:
            self._step()

    def _handle_commands(self):
        while True:
            with self._lock:
                if not self._cmds:
                    return
                name, value = self._cmds.popleft()
            if name == "open":
                self._open(value)
            elif name == "play":
                if self.cap is not None:
                    if not self.is_camera and self.frame_idx >= self.total_frames - 1:
                        self._seek(0)
                    self.playing = True
            elif name == "pause":
                self.playing = False
            elif name == "toggle":
                self.command("pause" if self.playing else "play")
            elif name == "seek":
                self._seek(value)
            elif name == "mode":
                self.mode = value
                self._reset_state()
            elif name == "realtime":
                self.realtime = bool(value)
            elif name == "close":
                if self.cap is not None:
                    self.cap.release()
                self.cap = None
                self.playing = False

    def _step(self):
        ok, frame_bgr = self.cap.read()
        if not ok:
            self.playing = False
            self.ended.emit()
            return False
        lat = self._process(frame_bgr)
        self.frame_ready.emit(bgr_to_qimage(frame_bgr), lat)
        self.frame_idx += 1
        return True

    def _process(self, frame_bgr):
        t = time.perf_counter
        t0 = t()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pre = (t() - t0) * 1000

        if self.mode == "proposed":
            score = self.detector.compute_score(frame_bgr, mean_conf=self.scheduler.get_last_conf())
            alert = bool(self.recent_i_infer) and self.recent_i_infer[-1] > ALERT_THRESHOLD_MS
            ftype = self.scheduler.decide_frame_type(score, alert)
        else:
            score, alert, ftype = 1.0, False, "I"

        t0 = t()
        if self.device == "cuda":
            _ = self.torch.from_numpy(frame_rgb.copy()).to(self.device)
        htod = (t() - t0) * 1000

        if ftype == "I":
            self._sync()
            t0 = t()
            dets = self.model(frame_rgb).xyxy[0].cpu().numpy()
            self._sync()
            infer = (t() - t0) * 1000
            self.boxes = dets[:, :4].tolist() if len(dets) else []
            confs = dets[:, 4].tolist() if len(dets) else [1.0]
            self.scheduler.update_last_result(self.boxes, float(np.mean(confs)))
            self.recent_i_infer.append(infer)
        else:
            t0 = t()
            self.boxes = self.scheduler.get_last_boxes()
            infer = (t() - t0) * 1000

        t0 = t()
        _ = self.boxes
        dtoh = (t() - t0) * 1000
        total = pre + htod + infer + dtoh

        self.totals.append(total)
        self.types.append(ftype)
        arr = np.fromiter(self.totals, dtype=float)
        n_i = sum(1 for x in self.types if x == "I")

        # 측정이 끝난 뒤 박스 표시 (I: 주황 = 새로 추론, P: 초록 = 직전 결과 재사용)
        color = (60, 146, 251) if ftype == "I" else (128, 222, 74)
        th = max(2, frame_bgr.shape[1] // 640)
        for b in self.boxes:
            x0, y0, x1, y1 = (int(v) for v in b[:4])
            cv2.rectangle(frame_bgr, (x0, y0), (x1, y1), color, th)

        return {
            "frame_idx": self.frame_idx,
            "type": ftype,
            "total_ms": total,
            "infer_ms": infer,
            "score": score,
            "fps": 1000.0 / max(arr.mean(), 1e-6),
            "p50": float(np.percentile(arr, 50)),
            "p99": float(np.percentile(arr, 99)),
            "i_ratio": n_i / len(self.types),
            "n": len(self.types),
            "objects": len(self.boxes),
            "mode": self.mode,
        }

    def run(self):
        try:
            name = self._load_model()
        except Exception as e:  # noqa: BLE001
            self.error.emit(f"모델 로딩 실패: {e}")
            return
        self._reset_state()
        self.model_ready.emit(name)
        while not self._quit:
            self._handle_commands()
            if not self.playing or self.cap is None:
                self.msleep(15)
                continue
            t_start = time.perf_counter()
            if not self._step():
                continue
            if self.realtime:
                wait = 1.0 / self.src_fps - (time.perf_counter() - t_start)
                if wait > 0:
                    time.sleep(wait)
        if self.cap is not None:
            self.cap.release()


# ─────────────────────────────────────────────────────────────
# 비교 실험 (스크립트를 그대로 실행)
# ─────────────────────────────────────────────────────────────
class ExperimentRunner(QThread):
    progress = pyqtSignal(str)
    done = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, video_path, frames, out_dir):
        super().__init__()
        self.video_path = video_path
        self.frames = frames
        self.out_dir = out_dir
        self.proc = None
        self._cancel = False

    def cancel(self):
        self._cancel = True
        if self.proc is not None and self.proc.poll() is None:
            self.proc.kill()

    def _run(self, label, args):
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", MPLBACKEND="Agg")
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self.proc = subprocess.Popen([sys.executable] + args, cwd=HERE, env=env,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, encoding="utf-8", errors="replace",
                                     creationflags=flags)
        tail = collections.deque(maxlen=30)
        for line in self.proc.stdout:
            line = line.rstrip()
            tail.append(line)
            s = line.strip()
            if s.startswith("Frame"):
                try:
                    k = int(s.split()[1])
                    self.progress.emit(f"{label}  {k + 1} / {self.frames} 프레임")
                except (ValueError, IndexError):
                    pass
        code = self.proc.wait()
        if self._cancel:
            raise RuntimeError("취소됨")
        if code != 0:
            raise RuntimeError(f"{label} 실패 (종료 코드 {code})\n" + "\n".join(tail))

    def run(self):
        try:
            os.makedirs(self.out_dir, exist_ok=True)
            b = os.path.join(self.out_dir, "baseline_log.csv")
            o = os.path.join(self.out_dir, "latency_log.csv")
            self._run("[1/3] 기존 방식 (모든 프레임 추론)",
                      ["baseline.py", "--source", self.video_path, "--frames", str(self.frames), "--output", b])
            self._run("[2/3] 제안 방식 (I/P 스케줄러)",
                      ["pipeline.py", "--source", self.video_path, "--frames", str(self.frames),
                       "--output", o, "--no-show"])
            self.progress.emit("[3/3] 비교 그래프 생성 중...")
            self._run("[3/3] 비교 그래프", ["compare_plot.py", b, o, self.out_dir])
            self.done.emit(self.out_dir)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


def read_latency_csv(path):
    totals, types = [], []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            totals.append(float(r["total_ms"]))
            types.append(r.get("frame_type", "I"))
    arr = np.array(totals)
    return {
        "frames": len(arr),
        "fps": 1000.0 / arr.mean(),
        "mean": arr.mean(),
        "p50": np.percentile(arr, 50),
        "p90": np.percentile(arr, 90),
        "p99": np.percentile(arr, 99),
        "i_ratio": sum(1 for t in types if t == "I") / max(len(types), 1),
    }


class ResultDialog(QDialog):
    def __init__(self, out_dir, video_name, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"비교 실험 결과 - {video_name}")
        avail = QApplication.primaryScreen().availableGeometry()
        self.resize(min(1100, int(avail.width() * 0.92)), min(820, int(avail.height() * 0.9)))
        b = read_latency_csv(os.path.join(out_dir, "baseline_log.csv"))
        o = read_latency_csv(os.path.join(out_dir, "latency_log.csv"))

        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 18, 20, 18)
        lay.setSpacing(14)
        title = QLabel(f"{video_name} · {o['frames']}프레임 · 기존 방식 vs 제안 방식")
        title.setObjectName("dlgTitle")
        lay.addWidget(title)

        cards = QHBoxLayout()
        cards.setSpacing(12)
        def change(before, after, higher_better):
            r = after / max(before, 1e-6) if higher_better else before / max(after, 1e-6)
            if r >= 1.05:
                return (f"{r:.1f}배 빠름" if higher_better else f"{r:.1f}배 감소"), GREEN
            if r <= 0.95:
                return (f"{1 / r:.1f}배 느림" if higher_better else f"{1 / r:.1f}배 증가"), "#f87171"
            return "비슷함", MUTED
        cards.addWidget(self._card("처리 속도 (FPS)", f"{b['fps']:.0f} → {o['fps']:.0f}",
                                   *change(b['fps'], o['fps'], True)))
        cards.addWidget(self._card("P50 지연 (보통 프레임)", f"{b['p50']:.1f} → {o['p50']:.1f} ms",
                                   *change(b['p50'], o['p50'], False)))
        cards.addWidget(self._card("P99 지연 (가장 느린 1%)", f"{b['p99']:.1f} → {o['p99']:.1f} ms",
                                   *change(b['p99'], o['p99'], False)))
        cards.addWidget(self._card("GPU 추론한 프레임 (I-frame)", f"100% → {o['i_ratio'] * 100:.0f}%",
                                   f"{(1 - o['i_ratio']) * 100:.0f}%는 직전 결과 재사용", ORANGE))
        lay.addLayout(cards)

        tabs = QTabWidget()
        table = QTableWidget(6, 3)
        table.setHorizontalHeaderLabels(["지표", "기존 방식", "제안 방식"])
        rows = [("FPS", "fps", "{:.1f}"), ("평균 지연 (ms)", "mean", "{:.2f}"),
                ("P50 지연 (ms)", "p50", "{:.2f}"), ("P90 지연 (ms)", "p90", "{:.2f}"),
                ("P99 지연 (ms)", "p99", "{:.2f}"), ("I-frame 비율", "i_ratio", "{:.1%}")]
        for i, (name, key, f) in enumerate(rows):
            table.setItem(i, 0, QTableWidgetItem(name))
            table.setItem(i, 1, QTableWidgetItem(f.format(b[key])))
            table.setItem(i, 2, QTableWidgetItem(f.format(o[key])))
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        tabs.addTab(table, "결과표")
        for fname, label in (("02_percentile_bar.png", "P50/P90/P99"),
                             ("01_tail_distribution.png", "지연 분포"),
                             ("03_timeline_compare.png", "프레임별 지연"),
                             ("04_fps_segments.png", "구간별 FPS")):
            p = os.path.join(out_dir, fname)
            if os.path.exists(p):
                tabs.addTab(self._image_tab(p), label)
        tabs.setCurrentIndex(1 if tabs.count() > 1 else 0)
        lay.addWidget(tabs, 1)

        bottom = QHBoxLayout()
        note = QLabel(f"저장 위치: {out_dir}")
        note.setObjectName("muted")
        note.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        bottom.addWidget(note, 1)
        open_btn = QPushButton("결과 폴더 열기")
        open_btn.clicked.connect(lambda: open_folder(out_dir))
        close_btn = QPushButton("닫기")
        close_btn.setObjectName("primary")
        close_btn.clicked.connect(self.accept)
        bottom.addWidget(open_btn)
        bottom.addWidget(close_btn)
        lay.addLayout(bottom)

    @staticmethod
    def _card(title, value, sub, color):
        f = QFrame()
        f.setObjectName("card")
        v = QVBoxLayout(f)
        v.setContentsMargins(16, 12, 16, 12)
        a = QLabel(title)
        a.setObjectName("muted")
        b = QLabel(value)
        b.setObjectName("cardValue")
        c = QLabel(sub)
        c.setStyleSheet(f"color:{color}; font-weight:600;")
        v.addWidget(a)
        v.addWidget(b)
        v.addWidget(c)
        return f

    @staticmethod
    def _image_tab(path):
        lbl = QLabel()
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pm = QPixmap(path)
        lbl.setPixmap(pm.scaled(1000, 560, Qt.AspectRatioMode.KeepAspectRatio,
                                Qt.TransformationMode.SmoothTransformation))
        lbl.setStyleSheet("background:white; border-radius:8px;")
        return lbl


def open_folder(path):
    if os.name == "nt":
        os.startfile(path)  # noqa: S606
    else:
        subprocess.Popen(["xdg-open", path])


# ─────────────────────────────────────────────────────────────
# 화면 구성 요소
# ─────────────────────────────────────────────────────────────
class VideoView(QWidget):
    """영상 + 좌상단 프레임 종류 배지 + 우상단 FPS 배지."""

    def __init__(self):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(640, 360)
        self.image = None
        self.info = None
        self.message = "모델을 불러오는 중입니다..."

    def set_frame(self, img, info):
        self.image = img
        self.info = info
        self.message = None
        self.update()

    def set_message(self, text):
        self.message = text
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        r = self.rect()
        clip = QPainterPath()
        clip.addRoundedRect(QRectF(r), 14, 14)
        p.setClipPath(clip)
        p.fillRect(r, QColor("#05070a"))
        if self.image is not None:
            img = self.image.scaled(r.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                    Qt.TransformationMode.SmoothTransformation)
            x = (r.width() - img.width()) // 2
            y = (r.height() - img.height()) // 2
            p.drawImage(x, y, img)
        if self.info is not None:
            i = self.info
            is_i = i["type"] == "I"
            if i["mode"] == "proposed":
                txt = (f"I-frame  GPU 추론  {i['total_ms']:.1f} ms" if is_i
                       else f"P-frame  결과 재사용  {i['total_ms']:.1f} ms")
                col = QColor(ORANGE if is_i else GREEN)
            else:
                txt = f"기존 방식  매 프레임 추론  {i['total_ms']:.1f} ms"
                col = QColor(ORANGE)
            self._badge(p, 18, 18, txt, col)
            self._badge(p, None, 18, f"{i['fps']:.0f} FPS", QColor(ACCENT), big=True)
        if self.message:
            p.setPen(QColor(MUTED))
            f = QFont(self.font())
            f.setPointSize(13)
            p.setFont(f)
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, self.message)
        p.end()

    def _badge(self, p, x, y, text, color, big=False):
        f = QFont(self.font())
        f.setPointSize(15 if big else 11)
        f.setBold(True)
        p.setFont(f)
        fm = p.fontMetrics()
        dot = not big
        w = fm.horizontalAdvance(text) + (44 if dot else 28)
        h = fm.height() + 14
        if x is None:
            x = self.width() - w - 18
        rect = QRectF(x, y, w, h)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(5, 7, 10, 190))
        p.drawRoundedRect(rect, h / 2, h / 2)
        p.setBrush(color)
        if dot:
            p.drawEllipse(QPointF(x + 18, y + h / 2), 4.5, 4.5)
            rect = QRectF(x + 30, y, w - 42, h)
        p.setPen(color)
        p.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)


class StatTile(QFrame):
    def __init__(self, title, color=TEXT):
        super().__init__()
        self.setObjectName("tile")
        v = QVBoxLayout(self)
        v.setContentsMargins(16, 10, 16, 10)
        v.setSpacing(2)
        t = QLabel(title)
        t.setObjectName("muted")
        self.value = QLabel("-")
        self.value.setObjectName("tileValue")
        self.value.setStyleSheet(f"color:{color};")
        v.addWidget(t)
        v.addWidget(self.value)

    def set(self, text):
        self.value.setText(text)


class SeekSlider(QSlider):
    """클릭한 위치로 바로 이동하는 슬라이더."""

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton and self.maximum() > 0:
            v = QStyleHelper.value_at(self, e.position().x())
            self.setValue(v)
            self.sliderMoved.emit(v)
            self.sliderReleased.emit()
            e.accept()
            return
        super().mousePressEvent(e)


class QStyleHelper:
    @staticmethod
    def value_at(slider, x):
        w = max(slider.width(), 1)
        return int(round(slider.minimum() + (slider.maximum() - slider.minimum()) * x / w))


# ─────────────────────────────────────────────────────────────
# 메인 창
# ─────────────────────────────────────────────────────────────
class PlayerWindow(QMainWindow):
    def __init__(self, engine=None):
        super().__init__()
        self.setWindowTitle("Codec-Inspired 실시간 영상 추론 · 캡스톤디자인 (암갈란투르)")
        self.resize(1440, 860)
        self.engine = engine or Engine()
        self.current = None          # (name, path or int)
        self.total_frames = 0
        self.src_fps = 30.0
        self.playing = False
        self.model_loaded = False
        self.runner = None
        self._seeking = False

        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(22, 16, 22, 18)
        outer.setSpacing(14)

        # 상단 바
        top = QHBoxLayout()
        logo = QLabel("</>")
        logo.setObjectName("logo")
        title = QLabel("Codec-Inspired Realtime Inference")
        title.setObjectName("appTitle")
        sub = QLabel("I/P-frame 적응형 추론 스케줄링 · YOLOv5s")
        sub.setObjectName("muted")
        top.addWidget(logo)
        top.addSpacing(6)
        top.addWidget(title)
        top.addSpacing(10)
        top.addWidget(sub)
        top.addStretch(1)
        self.device_label = QLabel("모델 로딩 중...")
        self.device_label.setObjectName("chip")
        top.addWidget(self.device_label)
        outer.addLayout(top)

        body = QHBoxLayout()
        body.setSpacing(16)
        outer.addLayout(body, 1)

        # 왼쪽: 영상 + 컨트롤
        left = QFrame()
        left.setObjectName("panel")
        lv = QVBoxLayout(left)
        lv.setContentsMargins(14, 14, 14, 12)
        lv.setSpacing(10)
        self.view = VideoView()
        lv.addWidget(self.view, 1)

        seek_row = QHBoxLayout()
        self.time_now = QLabel("00:00")
        self.time_now.setObjectName("time")
        self.slider = SeekSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.sliderPressed.connect(lambda: setattr(self, "_seeking", True))
        self.slider.sliderReleased.connect(self._on_seek)
        self.time_total = QLabel("00:00")
        self.time_total.setObjectName("time")
        seek_row.addWidget(self.time_now)
        seek_row.addWidget(self.slider, 1)
        seek_row.addWidget(self.time_total)
        lv.addLayout(seek_row)

        ctrl = QHBoxLayout()
        ctrl.setSpacing(10)
        self.btn_restart = self._icon_button("restart", "처음으로")
        self.btn_restart.clicked.connect(lambda: self.engine.command("seek", 0))
        self.btn_play = self._icon_button("play", "재생 / 일시정지", big=True)
        self.btn_play.clicked.connect(self._toggle_play)
        ctrl.addWidget(self.btn_restart)
        ctrl.addWidget(self.btn_play)
        ctrl.addSpacing(12)
        self.frame_label = QLabel("프레임 -")
        self.frame_label.setObjectName("muted")
        ctrl.addWidget(self.frame_label)
        ctrl.addStretch(1)

        self.realtime_box = QCheckBox("원래 속도로 보기")
        self.realtime_box.setChecked(True)
        self.realtime_box.setToolTip("끄면 영상 속도 제한 없이 최대 속도로 처리합니다 (FPS 값은 같은 방식으로 계산)")
        self.realtime_box.toggled.connect(lambda v: self.engine.command("realtime", v))
        ctrl.addWidget(self.realtime_box)
        ctrl.addSpacing(14)

        seg = QFrame()
        seg.setObjectName("segment")
        sl = QHBoxLayout(seg)
        sl.setContentsMargins(4, 4, 4, 4)
        sl.setSpacing(4)
        self.btn_prop = QPushButton("제안 방식  (I/P 스케줄러)")
        self.btn_base = QPushButton("기존 방식  (매 프레임 추론)")
        self.mode_group = QButtonGroup(self)
        for b in (self.btn_prop, self.btn_base):
            b.setCheckable(True)
            b.setObjectName("segBtn")
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            self.mode_group.addButton(b)
            sl.addWidget(b)
        self.btn_prop.setChecked(True)
        self.btn_prop.clicked.connect(lambda: self._set_mode("proposed"))
        self.btn_base.clicked.connect(lambda: self._set_mode("baseline"))
        ctrl.addWidget(seg)
        lv.addLayout(ctrl)
        body.addWidget(left, 1)

        # 오른쪽: 영상 목록 + 비교 실험
        right = QFrame()
        right.setObjectName("panel")
        right.setFixedWidth(340)
        rv = QVBoxLayout(right)
        rv.setContentsMargins(14, 14, 14, 14)
        rv.setSpacing(10)
        h = QLabel("영상 목록")
        h.setObjectName("section")
        rv.addWidget(h)
        self.playlist = QListWidget()
        self.playlist.setObjectName("playlist")
        self.playlist.setIconSize(QSize(120, 68))
        self.playlist.setSpacing(4)
        self.playlist.itemClicked.connect(self._on_item)
        rv.addWidget(self.playlist, 1)

        add_row = QHBoxLayout()
        self.btn_add = QPushButton("  파일 추가")
        self.btn_add.setIcon(make_icon("add", TEXT, 32))
        self.btn_add.clicked.connect(self._add_file)
        self.btn_cam = QPushButton("  웹캠")
        self.btn_cam.setIcon(make_icon("cam", TEXT, 32))
        self.btn_cam.clicked.connect(lambda: self._play_source("웹캠", 0))
        add_row.addWidget(self.btn_add)
        add_row.addWidget(self.btn_cam)
        rv.addLayout(add_row)

        exp = QFrame()
        exp.setObjectName("expBox")
        ev = QVBoxLayout(exp)
        ev.setContentsMargins(14, 12, 14, 14)
        ev.setSpacing(8)
        et = QLabel("비교 실험")
        et.setObjectName("section")
        self.exp_desc = QLabel("선택한 영상을 기존 방식과 제안 방식으로\n각각 측정해 결과표와 그래프를 보여줍니다.")
        self.exp_desc.setObjectName("muted")
        self.exp_desc.setWordWrap(True)
        self.btn_exp = QPushButton("  비교 실험 시작")
        self.btn_exp.setObjectName("primary")
        self.btn_exp.setIcon(make_icon("chart", "#04131a", 32))
        self.btn_exp.clicked.connect(self._start_experiment)
        ev.addWidget(et)
        ev.addWidget(self.exp_desc)
        ev.addWidget(self.btn_exp)
        rv.addWidget(exp)
        body.addWidget(right)

        # 하단 통계
        stats = QHBoxLayout()
        stats.setSpacing(12)
        self.t_fps = StatTile("처리 속도", ACCENT)
        self.t_p50 = StatTile("P50 지연 (보통 프레임)")
        self.t_p99 = StatTile("P99 지연 (가장 느린 1%)")
        self.t_ratio = StatTile("GPU 추론 / 결과 재사용")
        self.t_obj = StatTile("탐지 객체")
        for t in (self.t_fps, self.t_p50, self.t_p99, self.t_ratio, self.t_obj):
            stats.addWidget(t, 1)
        legend = QLabel(f"<span style='color:{ORANGE}'>■</span> I-frame: GPU로 새로 추론<br>"
                        f"<span style='color:{GREEN}'>■</span> P-frame: 직전 결과 재사용<br>"
                        f"<span style='color:{MUTED}'>통계는 최근 {STATS_WINDOW}프레임 기준</span>")
        legend.setObjectName("legend")
        stats.addWidget(legend)
        outer.addLayout(stats)

        self.setStyleSheet(STYLE)
        self._fill_playlist()

        e = self.engine
        e.model_ready.connect(self._on_model_ready)
        e.opened.connect(self._on_opened)
        e.frame_ready.connect(self._on_frame)
        e.ended.connect(self._on_ended)
        e.error.connect(self._on_error)
        e.start()

    # ── 구성 ──
    def _icon_button(self, kind, tip, big=False):
        b = QPushButton()
        b.setObjectName("roundBig" if big else "round")
        b.setIcon(make_icon(kind, "#04131a" if big else TEXT, 48))
        b.setIconSize(QSize(30, 30) if big else QSize(22, 22))
        b.setToolTip(tip)
        b.setCursor(Qt.CursorShape.PointingHandCursor)
        return b

    def _fill_playlist(self):
        self.playlist.clear()
        for name, path in default_playlist():
            self._add_playlist_item(name, path)
        if self.playlist.count() == 0:
            self.view.set_message("영상이 없습니다. [파일 추가]로 영상을 선택하세요.")

    def _add_playlist_item(self, name, path):
        n, fps, thumb = video_info(path)
        dur = fmt_time(n / fps) if n else "--:--"
        it = QListWidgetItem(f"{name}\n{dur}  ·  {n}프레임")
        it.setData(Qt.ItemDataRole.UserRole, (name, path))
        it.setIcon(QIcon(QPixmap.fromImage(thumb)) if thumb is not None else QIcon(placeholder_thumb("play")))
        it.setSizeHint(QSize(300, 80))
        self.playlist.addItem(it)
        return it

    # ── 동작 ──
    def _on_model_ready(self, device_name):
        self.model_loaded = True
        self.device_label.setText(f"●  {device_name}")
        if self.playlist.count():
            self.playlist.setCurrentRow(0)
            self._on_item(self.playlist.item(0))
        else:
            self.view.set_message("영상이 없습니다. [파일 추가]로 영상을 선택하세요.")

    def _on_item(self, item):
        name, path = item.data(Qt.ItemDataRole.UserRole)
        self._play_source(name, path)

    def _play_source(self, name, source):
        if not self.model_loaded:
            return
        if self.runner is not None:
            return
        self.current = (name, source)
        self.view.set_message("영상을 여는 중...")
        self.engine.command("open", source)
        self.btn_exp.setEnabled(not isinstance(source, int))

    def _add_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "영상 파일 선택", HERE,
                                              "영상 (*.mp4 *.avi *.mov *.mkv);;모든 파일 (*.*)")
        if path:
            it = self._add_playlist_item(os.path.splitext(os.path.basename(path))[0], path)
            self.playlist.setCurrentItem(it)
            self._on_item(it)

    def _toggle_play(self):
        if self.current is None:
            return
        self.engine.command("pause" if self.playing else "play")
        self._set_playing(not self.playing)

    def _set_playing(self, playing):
        self.playing = playing
        self.btn_play.setIcon(make_icon("pause" if playing else "play", "#04131a", 48))

    def _set_mode(self, mode):
        self.engine.command("mode", mode)

    def _on_seek(self):
        self._seeking = False
        self.engine.command("seek", self.slider.value())

    def _on_opened(self, info):
        self.total_frames = info["total_frames"]
        self.src_fps = info["fps"]
        self.slider.setEnabled(not info["camera"])
        self.slider.setRange(0, max(self.total_frames - 1, 0))
        self.time_total.setText("LIVE" if info["camera"] else fmt_time(self.total_frames / self.src_fps))
        self._set_playing(True)

    def _on_frame(self, img, info):
        self.view.set_frame(img, info)
        k = info["frame_idx"]
        if not self._seeking:
            self.slider.setValue(k)
        self.time_now.setText(fmt_time(k / self.src_fps))
        total = f" / {self.total_frames}" if self.total_frames else ""
        self.frame_label.setText(f"프레임 {k + 1}{total}")
        self.t_fps.set(f"{info['fps']:.0f} FPS")
        self.t_p50.set(f"{info['p50']:.1f} ms")
        self.t_p99.set(f"{info['p99']:.1f} ms")
        if info["mode"] == "proposed":
            self.t_ratio.set(f"{info['i_ratio'] * 100:.0f}% / {(1 - info['i_ratio']) * 100:.0f}%")
        else:
            self.t_ratio.set("100% / 0%")
        self.t_obj.set(f"{info['objects']}개")

    def _on_ended(self):
        self._set_playing(False)
        if self.view.info is not None:
            self.view.set_message(None)

    def _on_error(self, msg):
        self.view.set_message(msg)
        self.device_label.setText("오류")

    # ── 비교 실험 ──
    def _start_experiment(self):
        if self.current is None or isinstance(self.current[1], int) or self.runner is not None:
            return
        name, path = self.current
        frames = min(EXPERIMENT_FRAMES, self.total_frames) if self.total_frames else EXPERIMENT_FRAMES
        out_dir = os.path.join(HERE, "실험결과", safe_folder_name(path))
        self.engine.command("pause")
        self._set_playing(False)
        self.runner = ExperimentRunner(path, frames, out_dir)
        self.runner.progress.connect(self._on_exp_progress)
        self.runner.done.connect(lambda d: self._on_exp_done(d, name))
        self.runner.failed.connect(self._on_exp_failed)
        self.btn_exp.setEnabled(False)
        self.btn_exp.setText("  실험 중...")
        self.exp_desc.setText(f"{name} · {frames}프레임\n준비 중...")
        self.runner.start()

    def _on_exp_progress(self, text):
        self.exp_desc.setText(f"{self.current[0]}\n{text}")

    def _finish_exp(self):
        self.runner = None
        self.btn_exp.setEnabled(True)
        self.btn_exp.setText("  비교 실험 시작")
        self.exp_desc.setText("선택한 영상을 기존 방식과 제안 방식으로\n각각 측정해 결과표와 그래프를 보여줍니다.")

    def _on_exp_done(self, out_dir, name):
        self._finish_exp()
        ResultDialog(out_dir, name, self).exec()

    def _on_exp_failed(self, msg):
        self._finish_exp()
        dlg = QDialog(self)
        dlg.setWindowTitle("비교 실험 실패")
        v = QVBoxLayout(dlg)
        t = QPlainTextEdit(msg)
        t.setReadOnly(True)
        v.addWidget(t)
        dlg.resize(760, 420)
        dlg.exec()

    def closeEvent(self, e):
        if self.runner is not None:
            self.runner.cancel()
            self.runner.wait(3000)
        self.engine.quit_engine()
        self.engine.wait(3000)
        super().closeEvent(e)


STYLE = f"""
QWidget#root, QMainWindow, QDialog {{ background:{BG}; }}
QWidget {{ color:{TEXT}; font-family:'Malgun Gothic','Apple SD Gothic Neo','Noto Sans CJK KR',sans-serif; font-size:10.5pt; }}
QLabel#logo {{ color:{ACCENT}; font-size:17pt; font-weight:800; }}
QLabel#appTitle {{ font-size:15pt; font-weight:700; }}
QLabel#muted, QLabel#time {{ color:{MUTED}; }}
QLabel#time {{ font-family:Consolas,'DejaVu Sans Mono',monospace; }}
QLabel#chip {{ background:{PANEL2}; color:{GREEN}; border-radius:13px; padding:5px 14px; font-weight:600; }}
QLabel#section {{ font-size:12pt; font-weight:700; }}
QLabel#dlgTitle {{ font-size:15pt; font-weight:700; }}
QLabel#legend {{ color:{TEXT}; padding:0 6px; }}
QFrame#panel {{ background:{PANEL}; border-radius:18px; }}
QFrame#tile, QFrame#card {{ background:{PANEL}; border-radius:14px; }}
QLabel#tileValue {{ font-size:19pt; font-weight:700; }}
QLabel#cardValue {{ font-size:17pt; font-weight:700; }}
QFrame#expBox {{ background:{PANEL2}; border-radius:14px; }}
QFrame#segment {{ background:{PANEL2}; border-radius:12px; }}
QPushButton {{ background:{PANEL2}; border:none; border-radius:10px; padding:9px 14px; font-weight:600; }}
QPushButton:hover {{ background:#243044; }}
QPushButton:disabled {{ color:{MUTED}; }}
QPushButton#primary {{ background:{ACCENT}; color:#04131a; padding:11px 14px; }}
QPushButton#primary:hover {{ background:#67e8f9; }}
QPushButton#primary:disabled {{ background:#155e6b; color:#9fdbe6; }}
QPushButton#segBtn {{ background:transparent; color:{MUTED}; padding:8px 14px; border-radius:9px; }}
QPushButton#segBtn:checked {{ background:{ACCENT}; color:#04131a; }}
QPushButton#round {{ background:{PANEL2}; border-radius:21px; min-width:42px; max-width:42px; min-height:42px; max-height:42px; padding:0; }}
QPushButton#roundBig {{ background:{ACCENT}; border-radius:26px; min-width:52px; max-width:52px; min-height:52px; max-height:52px; padding:0; }}
QPushButton#roundBig:hover {{ background:#67e8f9; }}
QSlider::groove:horizontal {{ height:6px; background:#263142; border-radius:3px; }}
QSlider::sub-page:horizontal {{ background:{ACCENT}; border-radius:3px; }}
QSlider::handle:horizontal {{ background:white; width:16px; height:16px; margin:-5px 0; border-radius:8px; }}
QListWidget#playlist {{ background:transparent; border:none; outline:none; }}
QListWidget#playlist::item {{ background:{PANEL2}; border-radius:12px; padding:6px; margin:2px 0; }}
QListWidget#playlist::item:selected {{ background:#173a46; border:1px solid {ACCENT}; color:{TEXT}; }}
QListWidget#playlist::item:hover {{ background:#202b3b; }}
QCheckBox {{ color:{MUTED}; spacing:8px; }}
QCheckBox::indicator {{ width:16px; height:16px; border-radius:4px; background:{PANEL2}; border:1px solid #33415a; }}
QCheckBox::indicator:checked {{ background:{ACCENT}; border-color:{ACCENT}; }}
QTabWidget::pane {{ border:none; background:{PANEL}; border-radius:12px; }}
QTabBar::tab {{ background:transparent; color:{MUTED}; padding:9px 16px; font-weight:600; }}
QTabBar::tab:selected {{ color:{ACCENT}; border-bottom:2px solid {ACCENT}; }}
QTableWidget {{ background:{PANEL}; border:none; gridline-color:#263142; font-size:11.5pt; }}
QHeaderView::section {{ background:{PANEL2}; color:{MUTED}; border:none; padding:8px; font-weight:700; }}
QPlainTextEdit {{ background:{PANEL}; border:none; font-family:Consolas,monospace; }}
QToolTip {{ background:{PANEL2}; color:{TEXT}; border:none; padding:6px; }}
"""


def main():
    os.chdir(HERE)
    sys.path.insert(0, HERE)
    app = QApplication(sys.argv)
    app.setApplicationName("Codec-Inspired Player")
    w = PlayerWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
