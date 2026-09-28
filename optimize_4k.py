import cv2, torch, numpy as np, time, collections
from dataclasses import dataclass
from typing import List, Deque

VIDEO = r"C:\Users\USER\OneDrive\Desktop\proj2\test_4k.mp4"
N_FRAMES = 300

print("Loading model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.hub.load("ultralytics/yolov5", "yolov5s", verbose=False)
model.to(device)
model.conf = 0.4

def load_frames(n=N_FRAMES):
    cap = cv2.VideoCapture(VIDEO)
    frames = []
    while len(frames) < n:
        ret, f = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        frames.append(f)
    cap.release()
    return frames

def infer_full(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    with torch.no_grad():
        results = model([rgb])
    dets = results.xyxy[0]
    if len(dets) == 0:
        return [], 1.0
    confs = dets[:, 4].cpu().numpy()
    return dets.cpu().numpy(), float(np.mean(confs))

def run_scs(frames, theta, w1=0.6, w2=0.4, max_streak=8):
    prev_gray = None
    last_conf = 1.0
    last_boxes = []
    p_streak = 0
    times, i_count, p_count = [], 0, 0

    for f in frames:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        if prev_gray is None:
            score = 1.0
        else:
            diff = cv2.absdiff(gray, prev_gray)
            pcr = np.count_nonzero(diff > 15) / diff.size
            score = w1 * pcr + w2 * (1.0 - last_conf)
        prev_gray = gray

        force_i = (p_streak >= max_streak)
        if force_i or score >= theta:
            t0 = time.perf_counter()
            boxes, last_conf = infer_full(f)
            elapsed = (time.perf_counter() - t0) * 1000
            last_boxes = boxes
            p_streak = 0
            i_count += 1
        else:
            elapsed = 0.0
            p_streak += 1
            p_count += 1
        times.append(elapsed)

    arr = np.array(times)
    total = i_count + p_count
    fps_approx = 1000 / (arr[arr > 0].mean()) * (i_count / total) if i_count > 0 else 0
    i_times = arr[arr > 0]
    total_time_ms = i_times.sum() if len(i_times) > 0 else 1
    real_fps = (total / total_time_ms) * 1000
    p99 = float(np.percentile(i_times, 99)) if len(i_times) > 0 else 0

    return {
        "theta": theta,
        "i_ratio": round(i_count / total * 100, 1),
        "fps": round(real_fps, 1),
        "p99": round(p99, 2),
        "i_count": i_count,
    }

print("Loading frames...")
frames = load_frames()
print(f"{len(frames)} frames loaded\n")

thetas = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

print("="*60)
print(f"{'θ':>6} | {'I-frame%':>9} | {'FPS':>7} | {'P99(ms)':>9} | I수")
print("-"*60)

results = []
for theta in thetas:
    r = run_scs(frames, theta)
    results.append(r)
    print(f"{r['theta']:>6.2f} | {r['i_ratio']:>8.1f}% | {r['fps']:>7.1f} | {r['p99']:>9.2f} | {r['i_count']}")

print("="*60)
best = min(results, key=lambda x: abs(x['i_ratio'] - 11.2))
print(f"\nBest theta (closest to 11.2% I-frame): {best['theta']}")
print(f"  I-frame ratio: {best['i_ratio']}%")
print(f"  FPS: {best['fps']}")
print(f"  P99: {best['p99']}ms")