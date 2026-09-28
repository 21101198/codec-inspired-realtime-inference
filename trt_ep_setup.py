"""TensorRT/CUDA EP용 DLL 검색 경로 설정 헬퍼."""

import os
import sys
import glob

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

_DONE = False


def enable_gpu_dll_search(verbose: bool = True):
    """torch/lib 와 tensorrt_libs 를 DLL 검색 경로에 추가한다. 여러 번 불러도 안전."""
    global _DONE
    if _DONE:
        return
    dirs = []

    try:
        import torch
        d = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.isdir(d) and glob.glob(os.path.join(d, "cublas64_12.dll")):
            dirs.append(("CUDA 12 runtime (torch)", d))
    except ImportError:
        pass

    try:
        import tensorrt_libs
        d = os.path.dirname(tensorrt_libs.__file__)
        if os.path.isdir(d) and glob.glob(os.path.join(d, "nvinfer_10.dll")):
            dirs.append(("TensorRT 10 (tensorrt_libs)", d))
    except ImportError:
        pass

    for label, d in dirs:
        os.add_dll_directory(d)
        os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
        if verbose:
            print(f"[trt_ep_setup] + {label}: {d}")

    if verbose and not dirs:
        print("[trt_ep_setup] 경고: torch/lib · tensorrt_libs 를 찾지 못함")

    _DONE = True


def trt_providers(cache_dir="trt_cache", fp16=True, int8=False,
                  int8_calib_table=None, workspace_bytes=2 << 30, extra=None):
    """(TensorrtExecutionProvider, CUDAExecutionProvider, CPUExecutionProvider) provider 리스트 생성."""
    os.makedirs(cache_dir, exist_ok=True)
    opts = {
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": cache_dir,
        "trt_max_workspace_size": workspace_bytes,
        "trt_fp16_enable": bool(fp16),
    }
    if int8:
        opts["trt_int8_enable"] = True
        if int8_calib_table:
            opts["trt_int8_calibration_table_name"] = int8_calib_table
            opts["trt_int8_use_native_calibration_table"] = False
    if extra:
        opts.update(extra)
    return [
        ("TensorrtExecutionProvider", opts),
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]


if __name__ == "__main__":
    enable_gpu_dll_search()
    import onnxruntime as ort
    print("available:", ort.get_available_providers())
