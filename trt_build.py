"""YOLOv5s ONNX -> TensorRT FP16/INT8 엔진 빌드 + 벤치마크 (pycuda 없이 PyTorch CUDA 텐서 사용)."""
import torch
import numpy as np
import time
import os
import argparse


def export_onnx(save_path="yolov5s_static.onnx"):
    print("[1] YOLOv5s 모델 로딩...")
    model = torch.hub.load("ultralytics/yolov5", "yolov5s", verbose=False)
    model.eval().cuda()
    inner = model.model
    inner.eval()

    dummy = torch.zeros(1, 3, 640, 640).cuda()

    print(f"[1] ONNX 내보내기 → {save_path}")
    torch.onnx.export(
        inner, dummy, save_path,
        opset_version=11,
        input_names=["images"],
        output_names=["output"],
        verbose=False
    )
    print(f"[1] ONNX 저장 완료: {save_path}")
    return save_path


def prepare_calib_data(source=None, n_frames=100):
    """INT8 calibration용 이미지 준비: source 있으면 균등 샘플링, 없으면 랜덤 데이터."""
    images = []

    if source is not None:
        import cv2
        cap = cv2.VideoCapture(source)
        if cap.isOpened():
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            step = max(1, total // n_frames)
            for i in range(n_frames):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (640, 640))
                arr = frame_resized.astype(np.float32) / 255.0
                arr = np.transpose(arr, (2, 0, 1))[np.newaxis]
                images.append(arr)
            cap.release()
            print(f"[Calib] 동영상에서 {len(images)}프레임 추출 완료: {source}")

    needed = n_frames - len(images)
    if needed > 0:
        if source is not None:
            print(f"[Calib] 부족분 {needed}장을 랜덤 데이터로 보완")
        else:
            print(f"[Calib] 동영상 소스 없음 → 랜덤 데이터 {needed}장 생성")
        for _ in range(needed):
            images.append(np.random.rand(1, 3, 640, 640).astype(np.float32))

    return images


def make_int8_calibrator(calib_images, cache_file="int8_calib.cache"):
    """TensorRT IInt8EntropyCalibrator2 구현 (pycuda 대신 PyTorch CUDA 텐서 사용)."""
    import tensorrt as trt

    class _Int8Calibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self):
            super().__init__()
            self._images = calib_images
            self._cache_file = cache_file
            self._index = 0
            self._d_input = torch.zeros(
                calib_images[0].shape, dtype=torch.float32, device="cuda"
            )

        def get_batch_size(self):
            return 1

        def get_batch(self, _names):
            if self._index >= len(self._images):
                return None
            img = torch.from_numpy(self._images[self._index].copy()).cuda()
            self._d_input.copy_(img)
            self._index += 1
            if self._index % 20 == 0:
                print(f"[Calib] 진행: {self._index}/{len(self._images)}")
            return [self._d_input.data_ptr()]

        def read_calibration_cache(self):
            if os.path.exists(self._cache_file):
                print(f"[Calib] 캐시 로드: {self._cache_file}")
                with open(self._cache_file, "rb") as f:
                    return f.read()
            return None

        def write_calibration_cache(self, cache):
            with open(self._cache_file, "wb") as f:
                f.write(cache)
            print(f"[Calib] 캐시 저장: {self._cache_file}")

    return _Int8Calibrator()


def build_engine(onnx_path, engine_path, mode="fp16", calib_images=None):
    import tensorrt as trt

    torch.cuda.init()

    print(f"\n[Build] TensorRT 엔진 빌드 시작 (mode={mode.upper()})...")
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)

    if mode == "fp16":
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("[Build] FP16 모드 활성화")
        else:
            print("[Build] 경고: FP16 미지원 GPU — FP32로 빌드")

    elif mode == "int8":
        if not builder.platform_has_fast_int8:
            print("[Build] 경고: INT8 미지원 GPU — FP16으로 폴백")
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
        else:
            config.set_flag(trt.BuilderFlag.INT8)
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
            if calib_images is None:
                print("[Build] calibration 데이터 없음 → 랜덤 100장 사용")
                calib_images = prepare_calib_data(n_frames=100)
            config.int8_calibrator = make_int8_calibrator(calib_images)
            print("[Build] INT8 모드 활성화 (캘리브레이션 포함)")

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"[Build] 파싱 오류: {parser.get_error(i)}")
            return None

    inp = network.get_input(0)
    print(f"[Build] 입력 shape: {inp.shape}")
    if -1 in inp.shape:
        profile = builder.create_optimization_profile()
        profile.set_shape("images",
            (1, 3, 640, 640), (1, 3, 640, 640), (1, 3, 640, 640))
        config.add_optimization_profile(profile)
    else:
        print("[Build] Static shape — profile 불필요")

    print("[Build] 빌드 중... (FP16: 1~3분, INT8: 2~5분)")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print("[Build] 빌드 실패!")
        return None

    with open(engine_path, "wb") as f:
        f.write(serialized)
    size_mb = os.path.getsize(engine_path) // 1024 // 1024
    print(f"[Build] 엔진 저장 완료: {engine_path}  ({size_mb} MB)")
    return engine_path


def benchmark_trt(engine_path, label="TensorRT", n_runs=200):
    import tensorrt as trt

    torch.cuda.init()

    print(f"\n[Benchmark] {label} 엔진 로드: {engine_path}")
    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f:
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    input_shape  = (1, 3, 640, 640)
    output_shape = tuple(engine.get_binding_shape(1))
    print(f"[Benchmark] 출력 shape: {output_shape}")

    d_input  = torch.zeros(input_shape,  dtype=torch.float32, device="cuda")
    d_output = torch.zeros(output_shape, dtype=torch.float32, device="cuda")
    bindings = [d_input.data_ptr(), d_output.data_ptr()]

    h_input = np.random.randn(*input_shape).astype(np.float32)

    for _ in range(20):
        d_input.copy_(torch.from_numpy(h_input))
        context.execute_v2(bindings)
        _ = d_output.cpu().numpy()
    torch.cuda.synchronize()

    latencies = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        t = time.perf_counter()
        d_input.copy_(torch.from_numpy(h_input))
        context.execute_v2(bindings)
        _ = d_output.cpu().numpy()
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t) * 1000.0)

    latencies = np.array(latencies)
    print(f"\n[{label}]")
    print(f"  Mean : {latencies.mean():.2f} ms")
    print(f"  P50  : {np.percentile(latencies, 50):.2f} ms")
    print(f"  P90  : {np.percentile(latencies, 90):.2f} ms")
    print(f"  P99  : {np.percentile(latencies, 99):.2f} ms  <- Tail Latency")
    print(f"  FPS  : {1000/latencies.mean():.1f}")
    return latencies


def benchmark_pytorch(n_runs=200):
    print("\n[Benchmark] PyTorch FP32 베이스라인...")
    model = torch.hub.load("ultralytics/yolov5", "yolov5s", verbose=False)
    model.eval().cuda()
    inner = model.model
    dummy = torch.zeros(1, 3, 640, 640).cuda()

    with torch.no_grad():
        for _ in range(20):
            _ = inner(dummy)
    torch.cuda.synchronize()

    latencies = []
    with torch.no_grad():
        for _ in range(n_runs):
            torch.cuda.synchronize()
            t = time.perf_counter()
            _ = inner(dummy)
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t) * 1000.0)

    latencies = np.array(latencies)
    print("\n[PyTorch FP32]")
    print(f"  Mean : {latencies.mean():.2f} ms")
    print(f"  P50  : {np.percentile(latencies, 50):.2f} ms")
    print(f"  P90  : {np.percentile(latencies, 90):.2f} ms")
    print(f"  P99  : {np.percentile(latencies, 99):.2f} ms")
    print(f"  FPS  : {1000/latencies.mean():.1f}")
    return latencies


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",       type=str, default=None,
                        help="INT8 calibration용 동영상 경로 (없으면 랜덤 데이터)")
    parser.add_argument("--calib-frames", type=int, default=100,
                        help="INT8 calibration 프레임 수 (기본: 100)")
    parser.add_argument("--fp16-engine",  type=str, default="yolov5s_fp16.engine")
    parser.add_argument("--int8-engine",  type=str, default="yolov5s_int8.engine")
    parser.add_argument("--onnx",         type=str, default="yolov5s_static.onnx")
    parser.add_argument("--runs",         type=int, default=200,
                        help="벤치마크 반복 횟수 (기본: 200)")
    args = parser.parse_args()

    if not os.path.exists(args.onnx):
        export_onnx(args.onnx)

    if not os.path.exists(args.fp16_engine):
        build_engine(args.onnx, args.fp16_engine, mode="fp16")
    else:
        print(f"[Main] 기존 FP16 엔진 사용: {args.fp16_engine}")

    if not os.path.exists(args.int8_engine):
        print(f"\n[Main] INT8 calibration 데이터 준비 ({args.calib_frames}프레임)...")
        calib_data = prepare_calib_data(source=args.source, n_frames=args.calib_frames)
        build_engine(args.onnx, args.int8_engine, mode="int8", calib_images=calib_data)
    else:
        print(f"[Main] 기존 INT8 엔진 사용: {args.int8_engine}")

    lat_fp32 = benchmark_pytorch(n_runs=args.runs)
    lat_fp16 = benchmark_trt(args.fp16_engine, label="TensorRT FP16", n_runs=args.runs)
    lat_int8 = benchmark_trt(args.int8_engine, label="TensorRT INT8", n_runs=args.runs)

    ref = lat_fp32.mean()

    def fmt_row(label, lats):
        mean = lats.mean()
        p99  = float(np.percentile(lats, 99))
        fps  = 1000 / mean
        spdup = ref / mean
        return f"  {label:<28} {mean:>8.2f} {p99:>8.2f} {fps:>7.1f} {spdup:>7.2f}x"

    print(f"\n{'='*62}")
    print("  Ablation Study — Table 5 업데이트 (TensorRT INT8 추가)")
    print(f"{'='*62}")
    print(f"  {'설정':<28} {'Mean(ms)':>8} {'P99(ms)':>8} {'FPS':>7} {'vs FP32':>8}")
    print(f"  {'-'*60}")
    print(fmt_row("PyTorch FP32 (Baseline)", lat_fp32))
    print(fmt_row("+ TensorRT FP16",         lat_fp16))
    print(fmt_row("+ TensorRT INT8",         lat_int8))
    print(f"{'='*62}")

    fp16_mean = lat_fp16.mean()
    int8_mean = lat_int8.mean()
    print(f"\n  INT8 vs FP16 추가 개선: {fp16_mean/int8_mean:.2f}x 빠름  "
          f"({(1 - int8_mean/fp16_mean)*100:.1f}% 지연 감소)")
    print(f"{'='*62}")
