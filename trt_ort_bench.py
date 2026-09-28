import onnxruntime as ort
import numpy as np
import time, cv2, os

ONNX_PATH = r"C:\Users\USER\OneDrive\Desktop\proj2\yolov5s_static.onnx"
VIDEO_PATH = r"C:\Users\USER\OneDrive\Desktop\proj1\test_video.mp4"
N_FRAMES = 200

def preprocess(frame):
    img = cv2.resize(frame, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return img.transpose(2, 0, 1)[None]

def bench(session, frames, label):
    for f in frames[:10]:
        session.run(None, {"images": f})
    times = []
    for f in frames:
        t0 = time.perf_counter()
        session.run(None, {"images": f})
        times.append((time.perf_counter() - t0) * 1000)
    arr = np.array(times)
    print(f"  [{label}]")
    print(f"    Mean : {arr.mean():.2f} ms")
    print(f"    P50  : {np.percentile(arr,50):.2f} ms")
    print(f"    P99  : {np.percentile(arr,99):.2f} ms")
    print(f"    FPS  : {1000/arr.mean():.1f}")
    return arr

print("[1] 프레임 로딩...")
cap = cv2.VideoCapture(VIDEO_PATH)
frames = []
while len(frames) < N_FRAMES:
    ret, f = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    frames.append(preprocess(f))
cap.release()
print(f"[1] {len(frames)}프레임 완료")

print("[2] CUDA 세션...")
sess_cuda = ort.InferenceSession(ONNX_PATH, providers=["CUDAExecutionProvider","CPUExecutionProvider"])
t_cuda = bench(sess_cuda, frames, "CUDA ORT")

print("[3] TensorRT FP16 (1~3분 소요)...")
os.makedirs(r"C:\Users\USER\OneDrive\Desktop\proj2\trt_cache", exist_ok=True)
trt_opts = {
    "trt_fp16_enable": True,
    "trt_engine_cache_enable": True,
    "trt_engine_cache_path": r"C:\Users\USER\OneDrive\Desktop\proj2\trt_cache",
    "trt_max_workspace_size": 2147483648,
}
sess_trt = ort.InferenceSession(ONNX_PATH, providers=[("TensorrtExecutionProvider", trt_opts), "CUDAExecutionProvider"])
t_trt = bench(sess_trt, frames, "TensorRT FP16")

print("[4] TensorRT INT8 (1~3분 소요)...")
trt_opts_int8 = {
    "trt_int8_enable": True,
    "trt_fp16_enable": True,
    "trt_engine_cache_enable": True,
    "trt_engine_cache_path": r"C:\Users\USER\OneDrive\Desktop\proj2\trt_cache",
    "trt_max_workspace_size": 2147483648,
}
sess_int8 = ort.InferenceSession(ONNX_PATH, providers=[("TensorrtExecutionProvider", trt_opts_int8), "CUDAExecutionProvider"])
t_int8 = bench(sess_int8, frames, "TensorRT INT8")

print("=" * 50)
print(f"PyTorch FP32 baseline : 3.76ms / 266.1 FPS")
print(f"CUDA ORT              : {t_cuda.mean():.2f}ms / {1000/t_cuda.mean():.1f} FPS")
print(f"TensorRT FP16         : {t_trt.mean():.2f}ms / {1000/t_trt.mean():.1f} FPS")
print(f"TensorRT INT8         : {t_int8.mean():.2f}ms / {1000/t_int8.mean():.1f} FPS")
print("=" * 50)