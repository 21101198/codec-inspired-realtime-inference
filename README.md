# Realtime 4K YOLOv5 Pipeline

YOLOv5 기반 실시간 4K 영상 객체 탐지 파이프라인. 비디오 코덱의 I-frame/P-frame
개념을 추론 스케줄링에 적용해서, 장면 변화가 거의 없는 프레임은 GPU 추론을
스킵하고 직전 결과를 재사용함으로써 지연시간(latency)을 줄이는 실험 프로젝트입니다.
CLI 배치 도구와 PyQt6 실시간 대시보드 두 가지 실행 방식을 제공합니다.

**시연 영상**: https://youtu.be/_5iaqo70-LU (교차로·야간·고속 영상 처리 + PyQt6 대시보드, 61초)

## 주요 기능

- **Scene Change Detection**: 픽셀 diff + 이전 confidence 기반으로 장면 변화 점수 계산
- **Codec-Inspired Scheduler**: 장면 변화 점수에 따라 프레임을 I(풀 추론)/P(결과 재사용)로 분류
- **Tail Latency Predictor**: 초반 프레임으로 baseline mean/std를 캘리브레이션하고 임계값 초과 시 경보
- **Latency Profiling**: 전처리/HtoD/추론/DtoH 구간별 소요시간 측정, P50/P90/P99 통계 및 CSV 로깅
- **TensorRT / ONNX 변환·벤치마크**: FP16 TensorRT 엔진 빌드 및 ONNXRuntime 대비 속도 비교
- **PyQt6 실시간 대시보드**: 비디오 프리뷰 + 실시간 통계 패널 (I/P 카운트, P50/P90/P99, alert)

## 프로젝트 구조

```
core.py              # 순수 로직 (FrameLatency, SceneChangeDetector, Scheduler, TailLatencyPredictor)
pipeline.py           # CLI 배치 실행 (cv2.imshow 기반 실시간 파이프라인)
pipeline_worker.py    # core.py 로직을 감싸는 QThread 워커 (PyQt6 대시보드용)
main_window.py         # PyQt6 메인 윈도우 (비디오 프리뷰 + 실시간 통계 대시보드)
player.py              # 영상 플레이어 형태의 통합 실행 화면 (방식 전환 · 비교 실험)
model_loader.py        # YOLOv5 로더 (같은 폴더에 yolov5/ 코드가 있으면 오프라인으로 로드)
object_level_scheduler.py  # 객체 단위 선택적 추론(OL-SCS, pipeline.py --mode object)

baseline.py           # 기준(baseline) 성능 측정
bench_4k.py           # 해상도별(360p/1080p/4K) 추론 속도 벤치마크
bench_resolution_full.py  # 해상도별 Baseline vs I/P 스케줄러 동일조건 비교
bench_scenarios.py    # 고속동작/저조도/군중 3개 시나리오 영상 다운로드 + Baseline vs I/P 벤치마크
bench_multistream.py  # GPU 1개로 1/4 스트림 동시 처리 비교 (스마트팩토리 시나리오)
bench_yolov8_compare.py   # YOLOv5s vs YOLOv8s 모델 독립성 비교
optimize_4k.py         # scene-change threshold(theta) 스윕 최적화
weight_optimizer.py   # SceneChangeDetector 가중치(w1/w2) 최적화
accuracy_eval.py       # I/P 프레임 스케줄링의 탐지 정확도 영향 평가
latency_analyzer.py    # latency 로그 CSV 분석
tradeoff_analysis.py   # 속도-정확도 트레이드오프 분석
compare_plot.py         # 결과 비교 시각화
compare_resolution_plot.py  # 해상도별 결과 그래프
plot_object_mode_compare.py # 프레임 단위 vs 객체 단위 스케줄링 비교 그래프

tensorrt_export.py     # PyTorch -> ONNX -> TensorRT 변환
trt_build.py           # TensorRT 엔진 빌드
trt_ort_bench.py       # TensorRT vs ONNXRuntime 벤치마크
trt_ep_setup.py        # onnxruntime-gpu TensorRT EP DLL 경로 헬퍼 (Build Tools 불필요)
ort_int8_quantize.py   # YOLOv5s ONNX -> INT8 (QDQ 모델 + 캘리브레이션 테이블) 생성
ort_int8_bench.py      # FP32 / TRT FP16 / TRT INT8 추론 속도 비교
TENSORRT_INT8_ANALYSIS.md  # Build Tools 없이 TensorRT INT8 구현 분석 + 결과

*.csv                  # 실행 로그 (latency, baseline 등)
*.png                  # 벤치마크 결과 그래프 (01~09번)
```

> `*.pt` / `*.onnx` / `*.engine` 모델 파일과 `test_4k.mp4` 등 대용량 미디어 파일은
> 저장소 용량 문제로 `.gitignore`에서 제외되어 있습니다. 아래 설치 방법을 참고해
> 직접 준비해주세요. 4K 벤치마크 스크립트(`bench_4k.py`, `bench_resolution_full.py` 등)는
> 저장소 폴더에 있는 `test_4k.mp4`를 입력으로 사용합니다.

## 설치 방법

**요구 사항**: Python 3.11+, (선택) CUDA 지원 NVIDIA GPU

```bash
pip install torch opencv-python numpy PyQt6
```

TensorRT 관련 스크립트(`tensorrt_export.py`, `trt_build.py`, `trt_ort_bench.py`)를
사용하려면 추가로 필요합니다.

```bash
pip install onnx onnxruntime-gpu tensorrt
```

YOLOv5 모델은 최초 실행 시 `torch.hub`를 통해 자동으로 내려받아 캐시됩니다
(`~/.cache/torch/hub`). 직접 준비한 가중치(`yolov5s.pt`)가 있다면 프로젝트 루트에
두고 `--model`/`model_path` 인자로 지정할 수 있습니다.

### TensorRT INT8 (Visual C++ Build Tools 불필요)

`onnxruntime-gpu` + `tensorrt`(둘 다 prebuilt wheel)만으로 TensorRT INT8 추론을
돌릴 수 있습니다. 컴파일이나 `pycuda`가 필요 없습니다. 자세한 분석은
[`TENSORRT_INT8_ANALYSIS.md`](TENSORRT_INT8_ANALYSIS.md) 참고.

```bash
python ort_int8_quantize.py --calib-video videos/highspeed.mp4 --calib-frames 80
python ort_int8_bench.py --video videos/highspeed.mp4 --frames 200 --cuda-int8
```

RTX 4060 / YOLOv5s 640 / batch 1 기준: FP32 대비 **TRT INT8(캘리브레이션 테이블) 약 2.1x**
(FP16과 비슷하거나 약간 빠름). QDQ 정적 양자화 경로는 이 모델·배치에서는 FP16보다 느립니다.

## 실행 방법

### CLI 배치 실행

```bash
python pipeline.py --source 0 --model yolov5s.pt --frames 500 --output latency_log.csv --conf 0.4
```

- `--source`: 웹캠 인덱스(숫자) 또는 영상 파일 경로
- `--frames`: 처리할 최대 프레임 수
- `--output`: 결과 CSV 저장 경로
- `--conf`: confidence 임계값

`q` 키로 중간 종료할 수 있으며, 종료 시 CSV 저장과 함께 P50/P90/P99/FPS 등
콘솔 요약이 출력됩니다.

`--record demo.mp4`를 붙이면 처리한 모든 프레임을 실시간 통계 패널(I/P 종류, 지연, P50/P99,
처리 FPS)과 함께 영상으로 저장합니다. 시연 영상용 옵션이며, 저장은 지연 측정 이후에 하므로
프레임별 측정값에는 포함되지 않습니다(단, 인코딩 부하로 전체 FPS는 다소 낮아질 수 있음).
`--no-show`를 붙이면 영상 창 없이 실행합니다.

### 통합 플레이어 (권장)

```bash
python player.py
```

영상 플레이어 형태의 실행 화면입니다. 오른쪽 목록에서 영상을 고르면 탐지 박스를 그리며 재생하고,
재생 중에 **제안 방식(I/P 스케줄러) / 기존 방식(매 프레임 추론)** 을 바꿔 FPS·P50·P99 변화를 바로
비교할 수 있습니다. **[비교 실험 시작]** 은 아래 비교 실험(baseline.py → pipeline.py → compare_plot.py)을
그대로 실행하고 결과표와 그래프를 창에 보여줍니다. 같은 폴더(또는 상위 폴더)의 `sample_video.mp4`,
`원본실험영상/` 이 있으면 목록에 자동으로 추가됩니다.

### Baseline vs 제안 방식 비교 실험

```bash
python baseline.py --source sample_video.mp4 --frames 200 --output baseline_log.csv
python pipeline.py --source sample_video.mp4 --frames 200 --output latency_log.csv --no-show
python compare_plot.py baseline_log.csv latency_log.csv .
```

같은 영상을 모든 프레임 추론(Baseline)과 I/P 스케줄러로 각각 처리한 뒤, 지연 분포·P50/P90/P99·
타임라인·FPS 비교 그래프(01~04_*.png)와 비교표를 출력합니다.

### PyQt6 대시보드 실행

```bash
python main_window.py
```

창에서 소스(웹캠 인덱스 또는 영상 경로)/모델 경로/conf/frames를 입력하고
**시작**을 누르면 비디오 프리뷰와 실시간 통계(P50/P90/P99, I/P 프레임 수,
alert 횟수)가 갱신됩니다. **정지**로 언제든 중단할 수 있습니다.

영상에는 탐지 박스가 프레임 종류별 색으로 표시됩니다. I-frame에서 새로 추론한
박스는 **주황색**, P-frame에서 직전 결과를 재사용한 박스는 **초록색**입니다.
박스는 지연 측정이 끝난 뒤에 그리므로 측정값에는 영향이 없습니다.

### 오프라인 실행 (인터넷 없는 PC)

기본적으로 YOLOv5 코드는 `torch.hub`로 GitHub에서 내려받습니다. 인터넷이 없는
PC에서 실행하려면, 인터넷이 되는 PC에서 한 번 실행한 뒤
`~/.cache/torch/hub/ultralytics_yolov5_master` 폴더를 이 저장소 폴더에
`yolov5`라는 이름으로 복사하고 `yolov5s.pt`를 같이 두면 됩니다.
`model_loader.py`가 `yolov5/` 폴더를 먼저 찾아 오프라인으로 로드합니다.

### 벤치마크 / 분석 스크립트

```bash
python bench_4k.py           # 해상도별 추론 속도 비교
python optimize_4k.py         # scene-change threshold 최적화
python tradeoff_analysis.py   # 속도-정확도 트레이드오프 분석
```

#### 다중 영상(시나리오) 벤치마크

```bash
pip install yt-dlp                        # YouTube 다운로드용 (ffmpeg 불필요)
python bench_scenarios.py                 # 고속동작/저조도/군중 3개 영상 자동 다운로드 후 벤치마크
python bench_scenarios.py --frames 800 --max-height 360
python bench_scenarios.py --scenarios highspeed,crowd
python bench_scenarios.py --yt lowlight=https://youtu.be/XXXX      # 특정 영상 고정
python bench_scenarios.py --local crowd=D:\clips\market.mp4        # 로컬 파일 사용
python bench_scenarios.py --no-download                           # videos/<key>.mp4 재사용
```

- 영상 취득: YouTube 검색(`yt-dlp`, video-only 스트림) → CC-BY 공개 영상 → 공개 mp4 직링크 순으로 자동 폴백
- 출력: `videos/<key>.mp4`, `baseline_<key>.csv` / `latency_<key>.csv`,
  `12_scenario_compare.png`, `scenario_bench_summary.csv`
- 스케줄링 로직은 `core.py` / `bench_resolution_full.py` 를 그대로 재사용 (기존 코드 무수정)
