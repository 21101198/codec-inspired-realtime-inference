"""YOLOv5 모델 로더 (오프라인 실행 지원)

같은 폴더의 yolov5/ 코드를 우선 사용하므로 인터넷 없이도 실행된다.
yolov5/ 폴더가 없을 때만 GitHub(torch.hub)에서 내려받는다.
"""
import os
import warnings

import torch

warnings.filterwarnings("ignore", category=FutureWarning)

HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL_YOLOV5 = os.path.join(HERE, "yolov5")


def _resolve(path):
    """상대 경로면 이 폴더 기준으로 찾는다 (bat을 다른 위치에서 실행해도 동작)."""
    if os.path.isabs(path) or os.path.exists(path):
        return path
    cand = os.path.join(HERE, path)
    return cand if os.path.exists(cand) else path


def load_yolov5(model_path="yolov5s.pt"):
    model_path = _resolve(model_path)
    has_local = os.path.isfile(os.path.join(LOCAL_YOLOV5, "hubconf.py"))
    if not os.path.exists(model_path):
        if has_local:
            raise FileNotFoundError(
                f"모델 파일을 찾을 수 없습니다: {model_path}\n"
                f"yolov5s.pt 파일이 {HERE} 폴더에 있는지 확인하세요.")
        print("[Model] 가중치 파일이 없어 YOLOv5s 기본 가중치를 내려받습니다 (인터넷 필요)")
        return torch.hub.load("ultralytics/yolov5", "yolov5s", verbose=False)
    if has_local:
        print(f"[Model] 로컬 YOLOv5 코드 사용 (오프라인): {LOCAL_YOLOV5}")
        return torch.hub.load(LOCAL_YOLOV5, "custom", path=model_path,
                              source="local", verbose=False)
    print("[Model] 로컬 yolov5 폴더가 없어 GitHub에서 내려받습니다 (인터넷 필요)")
    return torch.hub.load("ultralytics/yolov5", "custom", path=model_path, verbose=False)
