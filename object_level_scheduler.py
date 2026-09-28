"""객체 단위 선택적 추론 스케줄러.

프레임 전체를 I/P로 나누는 CodecInspiredScheduler와 달리, 바운딩 박스마다
광류로 이동 속도를 계산해 빠르게 움직이는 객체 영역만 다시 추론하고
정지한 객체는 직전 추론 결과를 그대로 재사용한다.

    I: 전체 프레임 추론
    S: 빠른 객체 영역(ROI)만 크롭 추론
    P: 추론 생략, 모든 객체 결과 재사용
"""

import numpy as np
import cv2
from dataclasses import dataclass
from typing import List


def box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0.0


@dataclass
class TrackedObject:
    box: List[float]          # xyxy, 원본 해상도
    conf: float
    cls: int = -1
    speed: float = 0.0        # 프레임당 이동량 / 박스 대각선
    drift_x: float = 0.0      # 마지막 추론 이후 누적 이동량 (px)
    drift_y: float = 0.0

    @property
    def diag(self):
        return max(float(np.hypot(self.box[2]-self.box[0], self.box[3]-self.box[1])), 1.0)

    @property
    def drift(self):
        return float(np.hypot(self.drift_x, self.drift_y)) / self.diag


@dataclass
class ObjectPlan:
    frame_type: str                 # I: 전체 추론, S: 빠른 객체 ROI만 추론, P: 전부 재사용
    rois: List[List[int]]           # S-frame에서 추론할 영역 (xyxy, 원본 해상도)
    refresh_ids: List[int]          # 이번 추론 결과로 교체되는 객체 인덱스
    n_fast: int = 0                 # 속도/누적이동 기준을 넘은 객체 수
    roi_ratio: float = 0.0          # ROI 면적 합 / 프레임 면적


class ObjectMotionEstimator:
    """저해상도 그레이 영상에서 LK 광류로 박스별 이동 속도를 추정."""

    def __init__(self, work_width=480, diff_thresh=15, max_pts_per_box=20):
        self.work_width = work_width
        self.diff_thresh = diff_thresh
        self.max_pts_per_box = max_pts_per_box
        self._prev_full = self._cur_full = None
        self._prev_small = self._cur_small = None
        self.scale = 1.0

    @property
    def has_prev(self):
        return self._prev_full is not None

    def update(self, gray_full):
        """현재 프레임(원본 해상도 그레이)을 저장. 축소는 pair()에서 필요할 때만 한다."""
        self.scale = min(1.0, self.work_width / gray_full.shape[1])
        self._prev_full, self._cur_full = self._cur_full, gray_full
        self._prev_small, self._cur_small = self._cur_small, None

    def _shrink(self, gray_full):
        if self.scale >= 1.0:
            return gray_full
        return cv2.resize(gray_full, None, fx=self.scale, fy=self.scale,
                          interpolation=cv2.INTER_LINEAR)

    def pair(self):
        """(직전, 현재) 저해상도 그레이 쌍. 직전 프레임이 건너뛰어졌으면 그때 축소한다."""
        if self._prev_small is None:
            self._prev_small = self._shrink(self._prev_full)
        if self._cur_small is None:
            self._cur_small = self._shrink(self._cur_full)
        return self._prev_small, self._cur_small

    def _box_slice(self, box, shape):
        s = self.scale
        h, w = shape
        x0, y0 = max(int(box[0]*s), 0), max(int(box[1]*s), 0)
        x1, y1 = min(int(box[2]*s)+1, w), min(int(box[3]*s)+1, h)
        return x0, y0, x1, y1

    def box_motion(self, prev, gray, boxes):
        """박스별 (dx, dy, speed_px) 반환 (원본 해상도 px/frame). 특징점이 부족하면 None."""
        pts, owner = [], []
        for i, b in enumerate(boxes):
            x0, y0, x1, y1 = self._box_slice(b, gray.shape)
            if x1 - x0 < 4 or y1 - y0 < 4:
                continue
            p = cv2.goodFeaturesToTrack(prev[y0:y1, x0:x1], self.max_pts_per_box,
                                        qualityLevel=0.01, minDistance=3)
            if p is None:
                continue
            p = p.reshape(-1, 2) + (x0, y0)
            pts.append(p)
            owner += [i] * len(p)

        result = [None] * len(boxes)
        if not pts:
            return result
        p0 = np.concatenate(pts).astype(np.float32).reshape(-1, 1, 2)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, gray, p0, None,
                                             winSize=(15, 15), maxLevel=2)
        flow = (p1 - p0).reshape(-1, 2) / self.scale
        ok = st.ravel() == 1
        owner = np.array(owner)
        for i in range(len(boxes)):
            f = flow[(owner == i) & ok]
            if len(f) < 3:
                continue
            dx, dy = np.median(f, axis=0)
            # 박스 안의 배경 점이 속도를 희석하지 않도록 상위 분위수 사용
            speed = float(np.percentile(np.hypot(f[:, 0], f[:, 1]), 70))
            result[i] = (float(dx), float(dy), speed)
        return result

    def motion_mask(self, prev, gray):
        """카메라 전역 움직임(affine)을 보정한 뒤의 픽셀 변화 마스크."""
        h, w = gray.shape
        valid = np.full((h, w), 255, np.uint8)
        p0 = cv2.goodFeaturesToTrack(prev, 300, qualityLevel=0.01, minDistance=8)
        if p0 is not None and len(p0) >= 10:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, gray, p0, None,
                                                 winSize=(15, 15), maxLevel=2)
            ok = st.ravel() == 1
            if ok.sum() >= 10:
                A, _ = cv2.estimateAffinePartial2D(p0[ok], p1[ok], method=cv2.RANSAC,
                                                   ransacReprojThreshold=2.0)
                if A is not None:
                    prev = cv2.warpAffine(prev, A, (w, h), borderMode=cv2.BORDER_REPLICATE)
                    valid = cv2.warpAffine(valid, A, (w, h))
        mask = (cv2.absdiff(gray, prev) > self.diff_thresh).astype(np.uint8) * 255
        mask &= cv2.erode(valid, np.ones((5, 5), np.uint8))
        k = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        return cv2.dilate(mask, k, iterations=2)


class ObjectSelectiveScheduler:
    """객체 단위 I/P 결정: 빠르게 움직이는 객체 영역만 추론, 정지 객체는 직전 결과 재사용.

    - I: 첫 프레임 / 장면 전환 / 경보 / 연속 재사용 한도 / ROI가 너무 넓을 때 전체 추론
    - S: 빠른 객체 영역만 크롭 추론 (detect_new_motion=True면 박스 밖 새 움직임 영역도 포함)
    - P: 움직이는 객체가 없어 추론 생략

    새 객체는 기본적으로 주기적 I-frame(max_p_streak)과 장면 전환 I-frame에서 잡는다.
    """

    def __init__(self, max_p_streak=8, scene_threshold=0.35,
                 speed_threshold=0.03, drift_threshold=0.15,
                 max_roi_ratio=0.5, roi_pad=0.15, min_roi=96,
                 new_motion_min_area=0.002, motion_width=480,
                 detect_new_motion=False):
        self.max_p_streak = max_p_streak
        self.scene_threshold = scene_threshold
        self.speed_threshold = speed_threshold
        self.drift_threshold = drift_threshold
        self.max_roi_ratio = max_roi_ratio
        self.roi_pad = roi_pad
        self.min_roi = min_roi
        self.new_motion_min_area = new_motion_min_area
        self.detect_new_motion = detect_new_motion
        self.motion = ObjectMotionEstimator(work_width=motion_width)
        self.objects: List[TrackedObject] = []
        self._p_streak = 0
        self._frame_hw = (0, 0)

    def plan(self, frame_bgr, scene_score, alert=False, gray=None) -> ObjectPlan:
        """gray: 원본 해상도 그레이 (장면 변화 감지에서 만든 것을 재사용). 없으면 직접 변환."""
        if gray is None:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        self.motion.update(gray)
        H, W = gray.shape[:2]
        self._frame_hw = (H, W)

        if (not self.motion.has_prev or alert or scene_score >= self.scene_threshold
                or self._p_streak >= self.max_p_streak):
            return self._full_plan(W, H)

        # 추적 중인 객체가 없으면 광류를 계산할 대상이 없으므로 바로 재사용
        if not self.objects and not self.detect_new_motion:
            self._p_streak += 1
            return ObjectPlan("P", [], [])

        # 1) 박스별 속도 추정 → 빠른 객체 선별
        prev, gray = self.motion.pair()
        motions = self.motion.box_motion(prev, gray, [o.box for o in self.objects])
        # 움직임 마스크는 비용이 커서(전역 광류 + affine) 필요할 때만 계산
        mask = None
        if self.detect_new_motion or any(m is None for m in motions):
            mask = self.motion.motion_mask(prev, gray)
        fast_ids = []
        for i, (o, m) in enumerate(zip(self.objects, motions)):
            if m is None:
                # 특징점이 부족한 박스는 픽셀 변화율로 판단
                x0, y0, x1, y1 = self.motion._box_slice(o.box, mask.shape)
                patch = mask[y0:y1, x0:x1]
                changed = np.count_nonzero(patch) / max(patch.size, 1)
                o.speed = self.speed_threshold if changed > 0.25 else 0.0
            else:
                dx, dy, speed_px = m
                o.speed = speed_px / o.diag
                o.drift_x += dx
                o.drift_y += dy
            if o.speed >= self.speed_threshold or o.drift >= self.drift_threshold:
                fast_ids.append(i)

        # 2) ROI = 빠른 객체 박스(이동량만큼 여유) (+ 옵션: 기존 박스 밖의 새 움직임)
        rois = []
        for i in fast_ids:
            o = self.objects[i]
            bw, bh = o.box[2]-o.box[0], o.box[3]-o.box[1]
            move = o.speed * o.diag * 2 + np.hypot(o.drift_x, o.drift_y)
            rois.append([o.box[0] - bw*self.roi_pad - move, o.box[1] - bh*self.roi_pad - move,
                         o.box[2] + bw*self.roi_pad + move, o.box[3] + bh*self.roi_pad + move])

        if self.detect_new_motion:
            rois += self._new_motion_rois(mask)

        if not rois:
            self._p_streak += 1
            return ObjectPlan("P", [], [])

        rois = self._merge_rois([self._clip_roi(r, W, H) for r in rois])
        roi_ratio = sum((r[2]-r[0])*(r[3]-r[1]) for r in rois) / float(W * H)
        if roi_ratio >= self.max_roi_ratio:
            return self._full_plan(W, H)

        # ROI에 완전히 포함된 정지 객체도 이번 추론 결과로 교체
        refresh = set(fast_ids)
        for i, o in enumerate(self.objects):
            if any(r[0] <= o.box[0] and r[1] <= o.box[1] and
                   o.box[2] <= r[2] and o.box[3] <= r[3] for r in rois):
                refresh.add(i)
        self._p_streak += 1
        return ObjectPlan("S", rois, sorted(refresh), len(fast_ids), roi_ratio)

    def _new_motion_rois(self, mask):
        """기존 박스 밖의 움직임 영역 → ROI. 배경이 움직이는 영상에선 ROI가 과도해짐."""
        s = self.motion.scale
        for o in self.objects:
            x0, y0, x1, y1 = self.motion._box_slice(o.box, mask.shape)
            mask[y0:y1, x0:x1] = 0
        min_area = self.new_motion_min_area * mask.size
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rois = []
        for c in contours:
            if cv2.contourArea(c) >= min_area:
                x, y, w, h = cv2.boundingRect(c)
                pad = 0.25 * max(w, h)
                rois.append([(x - pad)/s, (y - pad)/s, (x + w + pad)/s, (y + h + pad)/s])
        return rois

    def _full_plan(self, W, H):
        self._p_streak = 0
        n = len(self.objects)
        return ObjectPlan("I", [[0, 0, W, H]], list(range(n)), n, 1.0)

    def _clip_roi(self, r, W, H):
        x0, y0, x1, y1 = r
        # 너무 작은 크롭은 검출이 불안정하므로 최소 크기 보장
        if x1 - x0 < self.min_roi:
            cx = (x0 + x1) / 2
            x0, x1 = cx - self.min_roi / 2, cx + self.min_roi / 2
        if y1 - y0 < self.min_roi:
            cy = (y0 + y1) / 2
            y0, y1 = cy - self.min_roi / 2, cy + self.min_roi / 2
        return [int(max(x0, 0)), int(max(y0, 0)), int(min(x1, W)), int(min(y1, H))]

    @staticmethod
    def _merge_rois(rois):
        """겹치는 ROI를 하나로 합쳐 중복 추론을 막는다."""
        i = 0
        while i < len(rois):
            j = i + 1
            while j < len(rois):
                a, b = rois[i], rois[j]
                if a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]:
                    rois[i] = [min(a[0], b[0]), min(a[1], b[1]),
                               max(a[2], b[2]), max(a[3], b[3])]
                    del rois[j]
                    j = i + 1
                else:
                    j += 1
            i += 1
        return rois

    def commit(self, plan: ObjectPlan, dets):
        """dets: 원본 좌표계의 (box, conf, cls) 리스트. P-frame이면 호출 불필요."""
        if plan.frame_type == "P":
            return
        if plan.frame_type == "I":
            self.objects = [TrackedObject(list(b), c, k) for b, c, k in dets]
            return

        H, W = self._frame_hw
        refresh = set(plan.refresh_ids)
        kept = [o for i, o in enumerate(self.objects) if i not in refresh]
        new = []
        for b, c, k in dets:
            # ROI 내부 경계에 걸린 검출이 정지 객체와 겹치면 잘린 중복으로 보고 버림
            roi = next((r for r in plan.rois if r[0] <= b[0] and r[1] <= b[1]
                        and b[2] <= r[2] and b[3] <= r[3]), None)
            on_inner_edge = roi is not None and (
                (b[0] - roi[0] < 2 and roi[0] > 0) or (b[1] - roi[1] < 2 and roi[1] > 0) or
                (roi[2] - b[2] < 2 and roi[2] < W) or (roi[3] - b[3] < 2 and roi[3] < H))
            if on_inner_edge and any(box_iou(b, o.box) > 0.3 for o in kept):
                continue
            new.append(TrackedObject(list(b), c, k))
        kept = [o for o in kept if all(box_iou(o.box, n.box) < 0.5 for n in new)]
        # ROI 추론에서 다시 잡히지 않은 객체는 사라졌다고 단정하지 않고 직전 결과 유지.
        # 크롭 경계에 잘리거나 문맥이 부족해 놓치는 경우가 많고, 실제로 사라졌다면 다음 I-frame에서 정리된다.
        missed = [o for i, o in enumerate(self.objects)
                  if i in refresh and all(box_iou(o.box, n.box) < 0.1 for n in new)]
        self.objects = kept + missed + new

    def get_last_boxes(self):
        return [o.box for o in self.objects]

    def get_last_conf(self):
        if not self.objects:
            return 1.0
        return float(np.mean([o.conf for o in self.objects]))
