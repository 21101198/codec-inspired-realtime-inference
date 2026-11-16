"""YOLOv5s를 TensorRT INT8/FP16으로 변환하고 PyTorch 추론과 속도 비교."""

import torch
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")


def benchmark_model(model, input_tensor, n_warmup=20, n_runs=200, label=""):
    """모델 추론 속도 측정"""
    device = input_tensor.device

    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(input_tensor)
    if device.type == "cuda":
        torch.cuda.synchronize()

    latencies = []
    with torch.no_grad():
        for _ in range(n_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter()
            _ = model(input_tensor)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t) * 1000.0)

    latencies = np.array(latencies)
    print(f"\n  [{label}]")
    print(f"    Mean : {latencies.mean():.2f} ms")
    print(f"    P50  : {np.percentile(latencies, 50):.2f} ms")
    print(f"    P90  : {np.percentile(latencies, 90):.2f} ms")
    print(f"    P99  : {np.percentile(latencies, 99):.2f} ms  <- Tail Latency")
    print(f"    Std  : {latencies.std():.2f} ms")
    print(f"    FPS  : {1000/latencies.mean():.1f}")

    return {
        "label": label,
        "mean": round(float(latencies.mean()), 2),
        "p50":  round(float(np.percentile(latencies, 50)), 2),
        "p90":  round(float(np.percentile(latencies, 90)), 2),
        "p99":  round(float(np.percentile(latencies, 99)), 2),
        "std":  round(float(latencies.std()), 2),
        "fps":  round(1000/float(latencies.mean()), 1),
    }


def export_to_tensorrt_fp16(model, input_shape=(1, 3, 640, 640)):
    """PyTorch 모델을 TensorRT FP16으로 변환"""
    print("\n[TRT] TensorRT FP16 변환 시작...")
    try:
        from torch2trt import torch2trt
        device = next(model.parameters()).device
        dummy_input = torch.randn(input_shape).to(device)
        model_trt = torch2trt(model, [dummy_input], fp16_mode=True)
        print("[TRT] FP16 변환 완료!")
        return model_trt
    except ImportError:
        print("[TRT] torch2trt 없음 → torch.compile로 대체")
        return None
    except Exception as e:
        print(f"[TRT] 변환 실패: {e}")
        return None


def optimize_with_compile(model):
    """torch.compile로 모델 최적화 (PyTorch 2.0+)"""
    print("\n[Compile] torch.compile 최적화 적용 중...")
    try:
        optimized = torch.compile(model, mode="reduce-overhead")
        print("[Compile] 완료!")
        return optimized
    except Exception as e:
        print(f"[Compile] 실패: {e}")
        return model


def export_onnx(model, save_path="yolov5s.onnx", input_shape=(1, 3, 640, 640)):
    """ONNX 형식으로 내보내기"""
    print(f"\n[ONNX] 모델 내보내기 → {save_path}")
    device = next(model.parameters()).device
    dummy = torch.randn(input_shape).to(device)

    try:
        forward_model = model.model if hasattr(model, 'model') else model

        torch.onnx.export(
            forward_model,
            dummy,
            save_path,
            opset_version=12,
            input_names=["images"],
            output_names=["output"],
            dynamic_axes={"images": {0: "batch"}, "output": {0: "batch"}},
            verbose=False
        )
        print(f"[ONNX] 저장 완료: {save_path}")
        return True
    except Exception as e:
        print(f"[ONNX] 실패: {e}")
        return False


def build_tensorrt_from_onnx(onnx_path="yolov5s.onnx",
                              engine_path="yolov5s_fp16.engine",
                              fp16=True):
    """ONNX → TensorRT 엔진 빌드"""
    print(f"\n[TRT] TensorRT 엔진 빌드 중 (FP16={fp16})...")
    try:
        import tensorrt as trt

        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, logger)
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)

        if fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("[TRT] FP16 모드 활성화")

        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(f"[TRT] 파싱 오류: {parser.get_error(i)}")
                return None

        print("[TRT] 엔진 빌드 중... (1~3분 소요)")
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            print("[TRT] 빌드 실패")
            return None

        with open(engine_path, "wb") as f:
            f.write(serialized)
        print(f"[TRT] 엔진 저장 완료: {engine_path}")
        return engine_path

    except Exception as e:
        print(f"[TRT] 빌드 실패: {e}")
        return None


class TRTInference:
    """TensorRT 엔진으로 추론"""
    def __init__(self, engine_path):
        import tensorrt as trt
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        print(f"[TRT] 엔진 로드 완료: {engine_path}")

    def infer(self, input_tensor):
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401  (CUDA 컨텍스트 초기화용)

        input_np = input_tensor.cpu().numpy().astype(np.float32)
        output_shape = tuple(self.engine.get_binding_shape(1))
        output = np.empty(output_shape, dtype=np.float32)

        d_input = cuda.mem_alloc(input_np.nbytes)
        d_output = cuda.mem_alloc(output.nbytes)
        cuda.memcpy_htod(d_input, input_np)
        self.context.execute_v2([int(d_input), int(d_output)])
        cuda.memcpy_dtoh(output, d_output)
        return output


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"디바이스: {device.upper()}")
    print(f"PyTorch: {torch.__version__}")

    print("\n[1] YOLOv5s 모델 로딩...")
    model = torch.hub.load("ultralytics/yolov5", "yolov5s", verbose=False)
    model.to(device).eval()

    dummy_input = torch.randn(1, 3, 640, 640).to(device)

    results = []

    print("\n" + "="*55)
    print("  추론 속도 비교 실험 (200회 측정)")
    print("="*55)

    res1 = benchmark_model(model, dummy_input, label="PyTorch FP32 (Baseline)")
    results.append(res1)

    try:
        model_fp16 = model.half()
        dummy_fp16 = dummy_input.half()
        res2 = benchmark_model(model_fp16, dummy_fp16, label="PyTorch FP16")
        results.append(res2)
        model_fp16 = model_fp16.float()
    except Exception as e:
        print(f"\n  [PyTorch FP16] 실패: {e}")

    try:
        model_compiled = torch.compile(model, mode="reduce-overhead")
        with torch.no_grad():
            for _ in range(5):
                _ = model_compiled(dummy_input)
        if device == "cuda":
            torch.cuda.synchronize()
        res3 = benchmark_model(model_compiled, dummy_input,
                               n_warmup=10, n_runs=100,
                               label="torch.compile (reduce-overhead)")
        results.append(res3)
    except Exception as e:
        print(f"\n  [torch.compile] 실패: {e}")

    print("\n[4] ONNX → TensorRT FP16 시도...")
    onnx_ok = export_onnx(model, "yolov5s.onnx")
    if onnx_ok:
        engine = build_tensorrt_from_onnx("yolov5s.onnx", "yolov5s_fp16.engine", fp16=True)
        if engine:
            try:
                trt_model = TRTInference(engine)
                latencies = []
                for _ in range(200):
                    t = time.perf_counter()
                    trt_model.infer(dummy_input)
                    latencies.append((time.perf_counter() - t) * 1000.0)
                latencies = np.array(latencies)
                res4 = {
                    "label": "TensorRT FP16",
                    "mean": round(float(latencies.mean()), 2),
                    "p50":  round(float(np.percentile(latencies, 50)), 2),
                    "p90":  round(float(np.percentile(latencies, 90)), 2),
                    "p99":  round(float(np.percentile(latencies, 99)), 2),
                    "std":  round(float(latencies.std()), 2),
                    "fps":  round(1000/float(latencies.mean()), 1),
                }
                results.append(res4)
                print("\n  [TensorRT FP16]")
                for k, v in res4.items():
                    if k != "label":
                        print(f"    {k:6}: {v}")
            except Exception as e:
                print(f"  [TRT 추론] 실패: {e}")

    print("\n\n" + "="*65)
    print("  최종 비교 요약")
    print("="*65)
    print(f"  {'방법':<35} {'Mean':>8} {'P99':>8} {'FPS':>8} {'향상'}")
    print("-"*65)
    baseline_fps = results[0]["fps"] if results else 1
    for r in results:
        speedup = r["fps"] / baseline_fps
        marker = " ← 최적" if r == max(results, key=lambda x: x["fps"]) else ""
        print(f"  {r['label']:<35} {r['mean']:>6.2f}ms {r['p99']:>6.2f}ms "
              f"{r['fps']:>7.1f} {speedup:.2f}x{marker}")
    print("="*65)
