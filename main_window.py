"""PipelineWorker를 구동하는 PyQt6 실시간 비디오/벤치마크 대시보드 메인 윈도우."""
import os
import sys

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap, QCloseEvent
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QDoubleSpinBox,
    QSpinBox, QPushButton, QHBoxLayout, QVBoxLayout, QGridLayout, QGroupBox,
    QStatusBar, QMessageBox, QFrame,
)

from core import FrameLatency
from pipeline_worker import PipelineWorker


class StatTile(QFrame):
    """라벨 + 값 한 쌍을 보여주는 작은 통계 타일."""

    def __init__(self, title: str):
        super().__init__()
        self.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)

        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("color: #888; font-size: 11px;")
        self.value_label = QLabel("-")
        self.value_label.setStyleSheet("font-size: 18px; font-weight: 600;")

        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)

    def set_value(self, text: str):
        self.value_label.setText(text)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Realtime Pipeline Dashboard")
        self.resize(1200, 700)

        self.worker: PipelineWorker | None = None

        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        root.addLayout(self._build_controls())

        body = QHBoxLayout()
        body.addWidget(self._build_video_panel(), stretch=3)
        body.addWidget(self._build_stats_panel(), stretch=1)
        root.addLayout(body)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("대기 중")

    def _build_controls(self) -> QHBoxLayout:
        layout = QHBoxLayout()

        # 샘플 영상이 있으면 기본 입력으로, 없으면 웹캠(0)
        sample = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_video.mp4")
        self.source_input = QLineEdit("sample_video.mp4" if os.path.exists(sample) else "0")
        self.source_input.setPlaceholderText("웹캠 index 또는 영상 경로")
        self.model_input = QLineEdit("yolov5s.pt")

        self.conf_input = QDoubleSpinBox()
        self.conf_input.setRange(0.01, 1.0)
        self.conf_input.setSingleStep(0.05)
        self.conf_input.setValue(0.4)

        self.frames_input = QSpinBox()
        self.frames_input.setRange(1, 100000)
        self.frames_input.setValue(500)

        self.start_btn = QPushButton("시작")
        self.stop_btn = QPushButton("정지")
        self.stop_btn.setEnabled(False)
        self.start_btn.clicked.connect(self._on_start)
        self.stop_btn.clicked.connect(self._on_stop)

        for label_text, widget in [
            ("소스", self.source_input),
            ("모델", self.model_input),
            ("conf", self.conf_input),
            ("frames", self.frames_input),
        ]:
            layout.addWidget(QLabel(label_text))
            layout.addWidget(widget)

        layout.addStretch()
        layout.addWidget(self.start_btn)
        layout.addWidget(self.stop_btn)
        return layout

    def _build_video_panel(self) -> QGroupBox:
        box = QGroupBox("비디오")
        layout = QVBoxLayout(box)

        self.video_label = QLabel("영상 없음")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("background-color: #111; color: #666;")
        layout.addWidget(self.video_label)
        return box

    def _build_stats_panel(self) -> QGroupBox:
        box = QGroupBox("실시간 통계")
        grid = QGridLayout(box)

        self.tiles = {
            "frame_idx":   StatTile("Frame"),
            "frame_type":  StatTile("Frame Type"),
            "total_ms":    StatTile("Total (ms)"),
            "infer_ms":    StatTile("Infer (ms)"),
            "scene_score": StatTile("Scene Score"),
            "p50_ms":      StatTile("P50 (ms)"),
            "p90_ms":      StatTile("P90 (ms)"),
            "p99_ms":      StatTile("P99 (ms) - Tail"),
            "i_count":     StatTile("I-frame 수"),
            "p_count":     StatTile("P-frame 수"),
            "alert_count": StatTile("Alert 횟수"),
        }

        positions = list(self.tiles.keys())
        for i, key in enumerate(positions):
            grid.addWidget(self.tiles[key], i // 2, i % 2)

        self.alert_banner = QLabel("")
        self.alert_banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.alert_banner.setStyleSheet(
            "background-color: #400; color: #f66; font-weight: 700; padding: 6px;"
        )
        self.alert_banner.hide()
        grid.addWidget(self.alert_banner, (len(positions) + 1) // 2, 0, 1, 2)

        return box

    def _on_start(self):
        if self.worker is not None and self.worker.isRunning():
            return

        source_text = self.source_input.text().strip()
        source = int(source_text) if source_text.isdigit() else source_text

        self.worker = PipelineWorker(
            model_path=self.model_input.text().strip() or "yolov5s.pt",
            source=source,
            max_frames=self.frames_input.value(),
            output_csv="latency_log.csv",
            conf_threshold=self.conf_input.value(),
        )
        self.worker.frame_ready.connect(self._on_frame_ready)
        self.worker.stats_updated.connect(self._on_stats_updated)
        self.worker.finished_run.connect(self._on_finished)
        self.worker.error.connect(self._on_error)

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_bar.showMessage("실행 중...")
        self.worker.start()

    def _on_stop(self):
        if self.worker is not None:
            self.worker.stop()
        self.stop_btn.setEnabled(False)
        self.status_bar.showMessage("정지 요청됨...")

    def _on_frame_ready(self, qimg: QImage, lat: FrameLatency):
        pixmap = QPixmap.fromImage(qimg).scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video_label.setPixmap(pixmap)

        self.tiles["frame_idx"].set_value(str(lat.frame_idx))
        self.tiles["frame_type"].set_value(lat.frame_type)
        self.tiles["total_ms"].set_value(f"{lat.total_ms:.2f}")
        self.tiles["infer_ms"].set_value(f"{lat.infer_ms:.2f}")
        self.tiles["scene_score"].set_value(f"{lat.scene_score:.2f}")

        if lat.alert:
            self.alert_banner.setText("ALERT: Tail Latency!")
            self.alert_banner.show()
        else:
            self.alert_banner.hide()

    def _on_stats_updated(self, stats: dict):
        self.tiles["p50_ms"].set_value(f"{stats.get('p50_ms', 0):.1f}")
        self.tiles["p90_ms"].set_value(f"{stats.get('p90_ms', 0):.1f}")
        self.tiles["p99_ms"].set_value(f"{stats.get('p99_ms', 0):.1f}")
        self.tiles["i_count"].set_value(str(stats.get("i_count", 0)))
        self.tiles["p_count"].set_value(str(stats.get("p_count", 0)))
        self.tiles["alert_count"].set_value(str(stats.get("alert_count", 0)))

    def _on_finished(self, summary: dict):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

        if summary.get("frame_count", 0) == 0:
            self.status_bar.showMessage("기록 없음")
            return

        fps = summary.get("fps", 0)
        p99 = summary.get("p99_ms", 0)
        self.status_bar.showMessage(
            f"완료 | {summary['frame_count']} frames | "
            f"FPS={fps:.1f} | P99={p99:.1f}ms | CSV={summary.get('output_csv')}"
        )

    def _on_error(self, message: str):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_bar.showMessage(f"오류: {message}")
        QMessageBox.critical(self, "오류", message)

    def closeEvent(self, event: QCloseEvent):
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(3000)
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
