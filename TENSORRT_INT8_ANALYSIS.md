# TensorRT INT8 (onnxruntime-gpu) — Visual C++ Build Tools 없이 구현

## 결론 요약

**가능하다. Build Tools 도, pycuda 도 필요 없다.**
이 저장소에서 실제로 구현·측정 완료 (`ort_int8_quantize.py`, `ort_int8_bench.py`, `trt_ep_setup.py`).

| 설정 | Mean (ms) | P99 (ms) | FPS | vs FP32 |
|---|---|---|---|---|
| CUDA EP FP32 | 5.07 | 6.48 | 197 | 1.00x |
| TensorRT EP FP16 | 2.59 | 3.38 | 387 | 1.96x |
| **TensorRT EP INT8 (calibration table)** | **2.43** | **3.39** | **411** | **2.08x** |
| TensorRT EP INT8 (QDQ, Conv 한정) — 부분 폴백 | 3.78 | 5.73 | 264 | 1.34x |
| CUDA EP INT8 (QDQ) | 8.03 | 9.10 | 125 | 0.63x |

> RTX 4060 (sm_89) / YOLOv5s 640×640 / batch 1 / onnxruntime-gpu 1.26.0 / TensorRT 10.15 / videos/highspeed.mp4 200프레임
> 3회 반복 측정 (INT8 calib table): 2.43 / 2.86 / 2.49 ms — 모두 FP32 대비 1.9~2.1x, FP16 과 동급.
>
> **요점:** TensorRT INT8 은 onnxruntime-gpu 만으로 정상 동작하며 FP32 대비 약 2x.
> 이 작은 모델·batch 1 에서는 FP16 과 사실상 동급이고, 실질 이득은
> **엔진 크기 절반(17.5MB → 9MB)** 과 배치·해상도가 커질 때 나온다.

---

## 1. 왜 Build Tools 가 필요 없는가

Visual C++ Build Tools 는 **네이티브 소스를 컴파일할 때만** 필요하다.
TensorRT INT8 를 onnxruntime-gpu 경유로 쓰는 경로에는 컴파일 단계가 하나도 없다.

| 구성요소 | 형태 | 컴파일 필요? |
|---|---|---|
| `onnxruntime-gpu` (TensorRT EP 포함) | prebuilt wheel (`onnxruntime_providers_tensorrt.dll` 동봉) | X |
| `tensorrt` / `tensorrt_libs` | prebuilt wheel (`nvinfer_10.dll` 등 DLL 만) | X |
| `torch` (CUDA 12 런타임 DLL 동봉) | prebuilt wheel | X |
| `onnxruntime.quantization` (양자화·캘리브레이션) | 순수 Python + NumPy | X |
| INT8 엔진 빌드 / 캘리브레이션 | TensorRT 런타임이 GPU 에서 수행 | X |

`pycuda` 는 Windows 에서 MSVC 로 빌드해야 하는 대표적인 패키지지만,
**TensorRT EP 경로에서는 필요 없다.** GPU 메모리 할당·H2D/D2H·CUDA 컨텍스트를
전부 onnxruntime 의 TensorRT EP 가 내부에서 처리하기 때문이다.
(기존 `trt_build.py` 는 `IInt8EntropyCalibrator2` 를 직접 구현하면서 pycuda 대신
PyTorch 텐서로 GPU 버퍼를 잡는 우회를 썼지만, EP 경로에서는 그 코드 자체가 불필요하다.)

---

## 2. 실제로 부딪힌 문제는 Build Tools 가 아니라 DLL 탐색 경로였다

onnxruntime-gpu 1.26 의 TensorRT EP 는 로드시 아래를 요구한다.

- `nvinfer_10.dll`, `nvonnxparser_10.dll` … → `site-packages/tensorrt_libs/`
- `cublas64_12.dll`, `cublasLt64_12.dll`, `cudart64_12.dll`, `cudnn64_9.dll` (CUDA 12) → `site-packages/torch/lib/`

두 묶음이 서로 다른 폴더에 있어 기본 상태에서는
`Error loading onnxruntime_providers_tensorrt.dll ... nvinfer_10.dll is missing` 으로 실패한다.

**해결 (컴파일 없이 `pip` 설치본만으로):** 세션 생성 전에 두 폴더를
프로세스 DLL 검색 경로에 추가한다 — `trt_ep_setup.enable_gpu_dll_search()`.

```python
import os, torch, tensorrt_libs
os.add_dll_directory(os.path.join(os.path.dirname(torch.__file__), "lib"))       # CUDA 12 런타임
os.add_dll_directory(os.path.dirname(tensorrt_libs.__file__))                    # TensorRT 10
```

주의: **`import tensorrt` 는 하지 않는다.** 설치된 파이썬 패키지가 CUDA 13 빌드
(`tensorrt-cu13`)라, import 하면 CUDA 13 cuBLAS 를 검색 경로에 얹어 오히려
onnxruntime 의 CUDA 12 EP 까지 깨뜨린다. DLL 폴더 참조용으로 `tensorrt_libs`
(순수 데이터 패키지) 만 쓴다.

---

## 3. 두 가지 INT8 구현 경로

### (A) 캘리브레이션 테이블 + 원본 FP32 ONNX  ← **권장**

`onnxruntime.quantization` 으로 MinMax 캘리브레이션만 수행해
ORT 형식 테이블(`yolov5s_calib.flatbuffers`)을 만들고,
TensorRT EP 에 원본 모델 그대로 넘긴다.

```python
providers = [("TensorrtExecutionProvider", {
    "trt_int8_enable": True,
    "trt_fp16_enable": True,                                   # INT8 불가 레이어는 FP16 폴백
    "trt_int8_calibration_table_name": "yolov5s_calib.flatbuffers",
    "trt_int8_use_native_calibration_table": False,            # ORT 포맷
    "trt_engine_cache_enable": True,
    "trt_engine_cache_path": "trt_cache_int8",
}), "CUDAExecutionProvider"]
```

- ONNX 그래프에 QDQ 노드를 삽입하지 않으므로 TensorRT ONNX 파서와 100% 호환.
- TensorRT 가 레이어별로 INT8/FP16 을 자동 선택 → reformat 오버헤드 최소.
- 결과: **2.43 ms / 411 FPS / FP32 대비 2.08x, FP16 대비도 소폭 빠름.**

### (B) QDQ 정적 양자화 (`quantize_static`)

`quantize_static(..., quant_format=QuantFormat.QDQ)` 로 INT8 QDQ 모델을 생성.
CUDA EP 로도 바로 INT8 추론이 되고, TensorRT EP 는 QDQ 를 읽어 INT8 엔진을 만든다.

YOLOv5 + 이 ORT/TRT 버전에서 TensorRT ONNX 파서가 거부하는 패턴이 있어 처리가 필요하다.

| 증상 | 원인 | 대응 |
|---|---|---|
| `DequantizeLinear: axis must be in range [0,0]` | opset 11 모델에 `axis` 속성 DQ 삽입 | 먼저 opset 13 으로 변환 |
| Detect 헤드(`model.24`) 상수 DQ 파서 에러 | 헤드의 anchor/grid 상수 양자화 | `op_types_to_quantize=["Conv"]` 로 백본만 |
| TRT 는 비대칭 INT8 미지원 | 기본이 비대칭 | `ActivationSymmetric/WeightSymmetric = True` |
| `DequantizeLayer can only run in INT8…` (`*_bias_dq`, INT32, axis 0) | 모든 Conv bias 에 int32 DQ 부착 | `QuantizeBias:False` **는 이 ORT 버전에서 무시됨** → `fold_bias_dq()` 로 상수 폴딩 |

**그래도 QDQ→TRT 는 이 조합(onnxruntime-gpu 1.26 + TensorRT 10.15)에서 완전히 되지 않는다.**
모델에서 bias DQ 를 다 제거해도 **ORT 의 TensorRT EP 가 서브그래프 변환 단계에서
bias Q/DQ(int32, axis 0)를 다시 삽입**해 TRT 파서가 또 거부하고, 해당 노드들이
CUDA EP 로 폴백된다(로그에 `60 Memcpy nodes ... for CUDAExecutionProvider`).
그래서 3.78 ms — FP16 보다 느림. reformat + 폴백 오버헤드 때문.
CUDA EP 의 INT8(QDQ)은 8.03 ms 로 더 느리다 (ORT CUDA INT8 커널이 이 형태에 최적화 안 됨).

`fold_bias_dq()` 로 정리한 `yolov5s_int8.onnx` 는 여전히 유효하다 — CUDA EP 및
다른 런타임(예: 네이티브 TensorRT `trtexec`)이 소비할 때는 문제없다.

> INT8 QDQ 가 이득이 되는 구간: 큰 배치 / 고해상도 / 더 큰 백본 / 전 구간 양자화.
> 이 파이프라인(실시간, batch 1, 640)에서는 **경로 (A)** 가 정답.

---

## 4. 사용법

```bash
# 1) INT8 산출물 생성 (컴파일 없음, 1~2분)
python ort_int8_quantize.py --calib-video videos/highspeed.mp4 --calib-frames 80
#   -> yolov5s_int8.onnx            (QDQ, 경로 B)
#   -> yolov5s_calib.flatbuffers    (캘리브레이션 테이블, 경로 A)

# 2) FP32 / FP16 / INT8 벤치마크 (첫 실행 시 엔진 빌드 수 분)
python ort_int8_bench.py --video videos/highspeed.mp4 --frames 200 --cuda-int8
```

`trt_ep_setup.py` 는 두 스크립트가 공용으로 import 한다. 다른 코드에서 TensorRT EP
를 쓸 때도 맨 위에서 `from trt_ep_setup import enable_gpu_dll_search; enable_gpu_dll_search()`
만 호출하면 된다.

## 5. 환경 (측정 시점)

```
Windows 11 / Python 3.11.9
onnxruntime-gpu 1.26.0   torch 2.5.1+cu121   tensorrt 10.15.1.29
NVIDIA RTX 4060  driver 595.71
Visual C++ Build Tools: 미설치   pycuda: 미설치
```
