import cv2, time, numpy as np, torch
from pathlib import Path

VIDEO_4K = r"C:\Users\USER\OneDrive\Desktop\proj2\test_4k.mp4"
N_FRAMES = 500

def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.cuda()
    model.eval()
    return model

def run_bench(video_path, resize_to, label, model, n=N_FRAMES):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < n:
        ret, f = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        frames.append(cv2.resize(f, resize_to))
    cap.release()
    print(f"[{label}] 프레임 로딩 완료: {len(frames)}개")

    for f in frames[:10]:
        img = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        model([img])

    times = []
    for f in frames:
        img = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        t0 = time.perf_counter()
        model([img])
        times.append((time.perf_counter() - t0) * 1000)

    arr = np.array(times)
    print(f"  Mean : {arr.mean():.2f} ms")
    print(f"  P50  : {np.percentile(arr,50):.2f} ms")
    print(f"  P90  : {np.percentile(arr,90):.2f} ms")
    print(f"  P99  : {np.percentile(arr,99):.2f} ms")
    print(f"  FPS  : {1000/arr.mean():.1f}")
    return arr

print("모델 로딩...")
model = load_model()

print("\n" + "="*50)
print("4K vs 1080p Baseline Benchmark")
print("="*50)

print("\n[1] 640x360 (기존 실험 해상도)")
t_360 = run_bench(VIDEO_4K, (640, 360), "640x360", model)

print("\n[2] 1920x1080 (Full HD)")
t_1080 = run_bench(VIDEO_4K, (1920, 1080), "1920x1080", model)

print("\n[3] 3840x2160 (4K)")
t_4k = run_bench(VIDEO_4K, (3840, 2160), "3840x2160", model)

print("\n" + "="*50)
print("결과 요약")
print("="*50)
print(f"640x360  : {t_360.mean():.2f}ms / {1000/t_360.mean():.1f} FPS")
print(f"1920x1080: {t_1080.mean():.2f}ms / {1000/t_1080.mean():.1f} FPS")
print(f"3840x2160: {t_4k.mean():.2f}ms / {1000/t_4k.mean():.1f} FPS")
print("="*50)