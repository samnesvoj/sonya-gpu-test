"""
real_visual_metrics.py — реальные визуальные метрики вместо YOLO conf × константа.

Заменяет в grok4_teacher._analyze_with_yolo():
    "action_intensity":  float(conf * 0.9)       → optical flow magnitude
    "composition_score": float(avg_conf)          → rule-of-thirds scoring
    "emotional_peaks":   float(avg_conf * 0.85)   → face detection

Подключение (одна строка import):

    from real_visual_metrics import compute_all_frame_metrics

    prev_gray = None
    for frame_idx, frame in enumerate(keyframes):
        # ... YOLO detection код без изменений ...
        metrics = compute_all_frame_metrics(frame, prev_gray, frame_dets)
        prev_gray = metrics["current_gray"]
        frame_confidences.append({
            ...
            "action_intensity":  metrics["action_intensity"],
            "emotional_peaks":   metrics["emotional_peaks"],
            "composition_score": metrics["composition_score"],
        })
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────── optical flow params
_OF_PARAMS: Dict[str, Any] = dict(
    pyr_scale=0.5,
    levels=3,
    winsize=15,
    iterations=3,
    poly_n=5,
    poly_sigma=1.2,
    flags=0,
)
# Максимальное движение (px/frame) соответствующее intensity=1.0
_OF_MAX_MAG: float = 20.0

# ─────────────────────────────────────────────── rule-of-thirds
_ROT_LINES = (1 / 3.0, 2 / 3.0)        # позиции линий (0..1)
_ROT_TOLERANCE = 0.10                   # допуск ± от линии (0..1)

# ─────────────────────────────────────────────── face cascade (ленивый синглтон)
_face_cascade: Optional[cv2.CascadeClassifier] = None
_profile_cascade: Optional[cv2.CascadeClassifier] = None


def _get_face_cascade() -> Optional[cv2.CascadeClassifier]:
    global _face_cascade
    if _face_cascade is None:
        try:
            path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            clf = cv2.CascadeClassifier(path)
            _face_cascade = None if clf.empty() else clf
        except Exception as e:
            logger.debug("Face cascade init failed: %s", e)
    return _face_cascade


def _get_profile_cascade() -> Optional[cv2.CascadeClassifier]:
    global _profile_cascade
    if _profile_cascade is None:
        try:
            path = cv2.data.haarcascades + "haarcascade_profileface.xml"
            clf = cv2.CascadeClassifier(path)
            _profile_cascade = None if clf.empty() else clf
        except Exception as e:
            logger.debug("Profile cascade init failed: %s", e)
    return _profile_cascade


# ═══════════════════════════════════════════════ core functions

def compute_action_intensity(
    prev_gray: Optional[np.ndarray],
    gray: np.ndarray,
) -> float:
    """
    Вычисляет интенсивность движения через Farneback optical flow.

    Args:
        prev_gray:  предыдущий grayscale кадр (None для первого кадра → 0.0)
        gray:       текущий grayscale кадр

    Returns:
        float 0.0..1.0  (0 = статичный кадр, 1.0 = максимальное движение)
    """
    if prev_gray is None:
        return 0.0
    try:
        if prev_gray.shape != gray.shape:
            prev_gray = cv2.resize(prev_gray, (gray.shape[1], gray.shape[0]))
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, **_OF_PARAMS)
        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        return float(min(np.mean(magnitude) / _OF_MAX_MAG, 1.0))
    except Exception as e:
        logger.debug("Optical flow error: %s", e)
        return 0.0


def compute_emotional_peaks(gray: np.ndarray) -> float:
    """
    Детектирует лица (frontal + profile) через Haar Cascade.

    Returns:
        float 0.0..1.0  (0 = нет лиц, 1.0 = 3+ лица)
    """
    cascade = _get_face_cascade()
    if cascade is None:
        return 0.5  # нет каскада — нейтральное значение

    try:
        eq = cv2.equalizeHist(gray)
        faces = cascade.detectMultiScale(
            eq,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(20, 20),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        n = len(faces) if isinstance(faces, np.ndarray) and faces.ndim == 2 else 0

        if n == 0:
            profile = _get_profile_cascade()
            if profile is not None:
                pfaces = profile.detectMultiScale(
                    eq, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20)
                )
                n = len(pfaces) if isinstance(pfaces, np.ndarray) and pfaces.ndim == 2 else 0

        return float(min(n / 3.0, 1.0))
    except Exception as e:
        logger.debug("Face detection error: %s", e)
        return 0.0


def compute_composition_score(
    yolo_detections: List[Dict[str, Any]],
    frame_shape: tuple,
) -> float:
    """
    Оценивает расположение объектов по правилу третей (rule-of-thirds).

    Для каждого bbox вычисляет расстояние от центра до ближайшего узла
    сетки 3×3 и нормирует в 0..1.

    Args:
        yolo_detections:  список {"bbox": [x1,y1,x2,y2], ...} из YOLO
        frame_shape:      (H, W) или (H, W, C)

    Returns:
        float 0.0..1.0  (1.0 = все объекты точно в узлах сетки)
    """
    if not yolo_detections:
        return 0.5  # пустой кадр — нейтральное значение

    H, W = frame_shape[0], frame_shape[1]
    if H <= 0 or W <= 0:
        return 0.5

    nodes_x = [W * r for r in _ROT_LINES]
    nodes_y = [H * r for r in _ROT_LINES]
    tol_x = W * _ROT_TOLERANCE * 3
    tol_y = H * _ROT_TOLERANCE * 3

    scores: List[float] = []
    for det in yolo_detections:
        bbox = det.get("bbox")
        if bbox is None or len(bbox) < 4:
            continue
        x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        min_dx = min(abs(cx - nx) for nx in nodes_x)
        min_dy = min(abs(cy - ny) for ny in nodes_y)

        sx = max(0.0, 1.0 - min_dx / tol_x)
        sy = max(0.0, 1.0 - min_dy / tol_y)
        scores.append((sx + sy) / 2)

    return float(np.mean(scores)) if scores else 0.5


def compute_all_frame_metrics(
    frame: np.ndarray,
    prev_gray: Optional[np.ndarray],
    yolo_detections: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Вычисляет все три визуальные метрики для одного кадра.

    Args:
        frame:            BGR кадр (numpy array H×W×3)
        prev_gray:        предыдущий grayscale кадр (None для первого кадра)
        yolo_detections:  список detections от YOLO для этого кадра

    Returns:
        {
            "action_intensity":  float,        # optical flow
            "emotional_peaks":   float,        # face detection
            "composition_score": float,        # rule-of-thirds
            "current_gray":      np.ndarray,   # передать как prev_gray следующему кадру
        }
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return {
        "action_intensity":  compute_action_intensity(prev_gray, gray),
        "emotional_peaks":   compute_emotional_peaks(gray),
        "composition_score": compute_composition_score(yolo_detections, frame.shape),
        "current_gray":      gray,
    }


# ═══════════════════════════════════════════════ smoke test
if __name__ == "__main__":
    import sys

    print("real_visual_metrics.py — smoke test")

    # Создаём синтетический кадр
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.rectangle(frame, (50, 50), (150, 150), (200, 100, 50), -1)

    metrics = compute_all_frame_metrics(frame, None, [])
    assert metrics["action_intensity"] == 0.0, "First frame should have 0 action"
    assert "current_gray" in metrics

    # Второй кадр с движением
    frame2 = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.rectangle(frame2, (100, 100), (200, 200), (200, 100, 50), -1)
    metrics2 = compute_all_frame_metrics(frame2, metrics["current_gray"], [])
    assert metrics2["action_intensity"] > 0.0, "Second frame should have motion"

    # Rule-of-thirds
    dets = [{"bbox": [100, 70, 140, 110]}]  # центр ~ (120, 90) близко к 1/3 × 320 = 107, 1/3 × 240 = 80
    score = compute_composition_score(dets, (240, 320, 3))
    assert 0.0 <= score <= 1.0

    print(f"  action_intensity (frame1): {metrics['action_intensity']:.3f}")
    print(f"  action_intensity (frame2): {metrics2['action_intensity']:.3f}")
    print(f"  composition_score:         {score:.3f}")
    print("  All assertions passed ✓")
    sys.exit(0)
