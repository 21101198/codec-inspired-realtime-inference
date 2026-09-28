"""YOLOv5s ONNX -> INT8 양자화: calib-table(권장)과 QDQ INT8 모델 두 산출물 생성."""

import argparse
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import cv2
import numpy as np
import onnx
from onnx import version_converter

from onnxruntime.quantization import (
    quantize_static, CalibrationDataReader, CalibrationMethod,
    QuantType, QuantFormat,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

INPUT_NAME = "images"
INPUT_HW = (640, 640)


def preprocess(frame_bgr):
    img = cv2.resize(frame_bgr, INPUT_HW)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.ascontiguousarray(img.transpose(2, 0, 1)[None])


def load_calib_frames(video, n_frames):
    """동영상에서 균등 샘플링. 없으면 랜덤 데이터로 폴백."""
    frames = []
    if video and os.path.exists(video):
        cap = cv2.VideoCapture(video)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or (n_frames * 4)
        step = max(1, total // n_frames)
        idx = 0
        while len(frames) < n_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, f = cap.read()
            if not ok:
                break
            frames.append(preprocess(f))
            idx += step
        cap.release()
        print(f"[calib] {video} 에서 {len(frames)}프레임 샘플링 (step={step})")
    if len(frames) < n_frames:
        need = n_frames - len(frames)
        print(f"[calib] 랜덤 데이터 {need}장으로 보완")
        frames += [np.random.rand(1, 3, *INPUT_HW).astype(np.float32) for _ in range(need)]
    return frames


class FrameReader(CalibrationDataReader):
    def __init__(self, frames):
        self._it = iter([{INPUT_NAME: f} for f in frames])

    def get_next(self):
        return next(self._it, None)

    def rewind(self):
        raise NotImplementedError


def fold_bias_dq(onnx_path):
    """Conv bias의 INT32 DequantizeLinear를 상수 폴딩(fp32 = q*scale)해 TensorRT 파서 호환시킴."""
    import numpy as _np
    from onnx import numpy_helper

    m = onnx.load(onnx_path)
    g = m.graph
    inits = {i.name: i for i in g.initializer}
    folded = 0
    keep_nodes = []
    new_inits = []
    consumed = set()
    for node in g.node:
        is_bias_dq = (
            node.op_type == "DequantizeLinear"
            and (node.name.endswith("_bias_dq") or node.doc_string == "Bias DQ node")
            and node.input and node.input[0] in inits
            and inits[node.input[0]].data_type == onnx.TensorProto.INT32
        )
        if not is_bias_dq:
            keep_nodes.append(node)
            continue
        q = numpy_helper.to_array(inits[node.input[0]]).astype(_np.float64)
        scale = numpy_helper.to_array(inits[node.input[1]]).astype(_np.float64)
        bias_fp32 = (q * scale).astype(_np.float32)
        t = numpy_helper.from_array(bias_fp32, name=node.output[0])
        new_inits.append(t)
        consumed.update(node.input)
        folded += 1

    if folded:
        still_used = {inp for n in keep_nodes for inp in n.input}
        g.ClearField("node")
        g.node.extend(keep_nodes)
        for t in new_inits:
            g.initializer.append(t)
        for name in list(consumed):
            if name not in still_used and name in inits:
                g.initializer.remove(inits[name])
        onnx.save(m, onnx_path)
        print(f"[prep] bias DQ 상수 폴딩: {folded}개 노드 제거 -> TensorRT 파서 호환")
    return onnx_path


def ensure_opset13(onnx_path, work_path):
    """DequantizeLinear 의 axis 속성은 opset>=13 필요. 낮으면 변환."""
    m = onnx.load(onnx_path)
    v = max(op.version for op in m.opset_import if op.domain in ("", "ai.onnx"))
    if v >= 13:
        return onnx_path
    print(f"[prep] opset {v} -> 13 변환")
    m = version_converter.convert_version(m, 13)
    onnx.save(m, work_path)
    return work_path


def write_trt_calib_table(model_fp32, frames, stem="yolov5s_calib"):
    """MinMax 캘리브레이션으로 TensorRT EP용 calib table(.flatbuffers)을 생성한다."""
    import glob
    import shutil
    from onnxruntime.quantization.calibrate import create_calibrator
    from onnxruntime.quantization.quant_utils import write_calibration_table

    os.makedirs("_calib_aug", exist_ok=True)
    calibrator = create_calibrator(
        model_fp32, op_types_to_calibrate=None,
        augmented_model_path="_calib_aug/aug.onnx",
        calibrate_method=CalibrationMethod.MinMax,
    )
    calibrator.set_execution_providers(["CUDAExecutionProvider", "CPUExecutionProvider"])
    calibrator.collect_data(FrameReader(frames))
    write_calibration_table(calibrator.compute_data())

    for p in glob.glob("calibration.*"):
        dst = stem + os.path.splitext(p)[1]
        shutil.move(p, dst)
        print(f"[calib-table] {dst}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="yolov5s_static.onnx", help="입력 FP32 ONNX (static shape)")
    ap.add_argument("--out", default="yolov5s_int8.onnx", help="출력 QDQ INT8 ONNX")
    ap.add_argument("--calib-video", default="videos/highspeed.mp4")
    ap.add_argument("--calib-frames", type=int, default=100)
    ap.add_argument("--per-channel", action="store_true", default=True)
    ap.add_argument("--op-types", default="Conv",
                    help="양자화할 op 타입(콤마구분). 기본 Conv. "
                         "'all' 로 주면 전체(단, YOLOv5 Detect 헤드에서 TRT 파서 오류 가능)")
    ap.add_argument("--calib-method", default="minmax", choices=["minmax", "entropy", "percentile"])
    ap.add_argument("--no-calib-table", action="store_true",
                    help="QDQ 모델만 만들고 TensorRT 캘리브레이션 테이블은 건너뜀")
    args = ap.parse_args()

    if not os.path.exists(args.onnx):
        raise SystemExit(f"입력 ONNX 없음: {args.onnx}")

    op13 = ensure_opset13(args.onnx, "yolov5s_op13.onnx")

    pre_path = "yolov5s_preproc.onnx"
    print("[prep] quant_pre_process (shape inference + graph 최적화)")
    quant_pre_process(op13, pre_path, skip_symbolic_shape=False)

    frames = load_calib_frames(args.calib_video, args.calib_frames)

    method = {
        "minmax": CalibrationMethod.MinMax,
        "entropy": CalibrationMethod.Entropy,
        "percentile": CalibrationMethod.Percentile,
    }[args.calib_method]

    op_types = None if args.op_types.lower() == "all" else \
        [t.strip() for t in args.op_types.split(",") if t.strip()]

    print(f"[quant] quantize_static  QDQ / INT8 symmetric / per_channel={args.per_channel} / "
          f"{args.calib_method} / op_types={op_types or 'all'}")
    # 대칭 INT8(TRT 요구) + Conv bias 양자화 억제(-> fold_bias_dq 후처리)
    quantize_static(
        pre_path, args.out, FrameReader(frames),
        quant_format=QuantFormat.QDQ,
        per_channel=args.per_channel,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=method,
        op_types_to_quantize=op_types,
        extra_options={
            "ActivationSymmetric": True,
            "WeightSymmetric": True,
            "QuantizeBias": False,
        },
    )
    fold_bias_dq(args.out)
    print(f"[quant] 저장: {args.out}  ({os.path.getsize(args.out) / 1e6:.1f} MB, "
          f"원본 {os.path.getsize(args.onnx) / 1e6:.1f} MB)")

    if not args.no_calib_table:
        print("[calib-table] 원본 FP32 모델용 TensorRT 캘리브레이션 테이블 생성 중...")
        try:
            write_trt_calib_table(pre_path, frames, "yolov5s_calib")
        except Exception as e:  # noqa
            print(f"[calib-table] 실패(무시 가능, QDQ 경로는 정상): {e!r}")

    for tmp in ("yolov5s_op13.onnx", pre_path):
        try:
            if tmp != args.onnx and os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass
    import shutil
    shutil.rmtree("_calib_aug", ignore_errors=True)

    print("\n완료. 다음으로:  python ort_int8_bench.py")


if __name__ == "__main__":
    main()
