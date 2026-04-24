"""
utils.py — общие утилиты SONYA.

Устраняет дублирование в проекте:
  ❌ 4 реализации get_video_duration   → ✅ одна (ffprobe → OpenCV → fallback)
  ❌ 3 копии temporal_nms              → ✅ одна с параметрами
  ❌ 2 подхода к get_ffmpeg_path       → ✅ один кроссплатформенный
  ❌ librosa.load 40× на одном файле   → ✅ load_audio_once() с кэшем

Использование:
    from utils import (
        get_video_duration,
        get_ffmpeg_path, get_ffprobe_path,
        temporal_nms,
        load_audio_once, get_audio_segment, clear_audio_cache,
        safe_float,
        sanitize_filename,
    )
"""

from __future__ import annotations

import logging
import math
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_UNSAFE_FILENAME_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

# ═══════════════════════════════════════════════ audio cache

# key: "path:sr" → (y_array, actual_sr)
_audio_cache: Dict[str, Tuple[Any, int]] = {}


def load_audio_once(path: str, sr: int = 16_000) -> Tuple[Any, int]:
    """
    Загружает аудио через librosa с кэшем в памяти.

    Решает проблему: librosa.load вызывается 40+ раз на одном файле
    в разных функциях проекта. Каждый вызов занимает 2–5 секунд.
    С кэшем — первый вызов медленный, остальные мгновенные.

    Returns:
        (y, actual_sr) — numpy array и реальный sample rate
    """
    import librosa  # type: ignore

    cache_key = f"{path}:{sr}"
    if cache_key not in _audio_cache:
        logger.debug("Loading audio (first time): %s", path)
        y, actual_sr = librosa.load(path, sr=sr)
        _audio_cache[cache_key] = (y, int(actual_sr))
    return _audio_cache[cache_key]


def get_audio_segment(
    path: str,
    start_sec: float,
    end_sec: float,
    sr: int = 16_000,
) -> Any:
    """
    Возвращает numpy array аудио для диапазона [start_sec, end_sec].
    Использует кэш load_audio_once — без повторной загрузки файла.
    """
    y, actual_sr = load_audio_once(path, sr)
    start_idx = int(start_sec * actual_sr)
    end_idx = int(end_sec * actual_sr)
    start_idx = max(0, min(start_idx, len(y)))
    end_idx = max(start_idx, min(end_idx, len(y)))
    return y[start_idx:end_idx]


def clear_audio_cache() -> None:
    """Очищает кэш аудио. Вызывать между видео для экономии RAM."""
    _audio_cache.clear()
    logger.debug("Audio cache cleared")


# ═══════════════════════════════════════════════ video duration

def get_video_duration(video_path: str) -> float:
    """
    Длительность видео в секундах.

    Порядок методов (по надёжности):
      1. ffprobe  — точный, работает со всеми форматами
      2. OpenCV   — быстрый, небольшая погрешность для VFR
      3. Fallback 60.0

    Заменяет 4 независимые реализации в проекте:
      - modes_scoring._get_video_duration_sec
      - modes_scoring_v2._get_video_duration_sec
      - cut_clips_from_result.get_video_duration
      - integrate_topic_segmentation.get_video_duration
    """
    # 1. ffprobe
    try:
        import subprocess
        cmd = [
            get_ffprobe_path(),
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        dur = float(result.stdout.strip())
        if dur > 0:
            return dur
    except Exception:
        pass

    # 2. OpenCV
    try:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        cap.release()
        if fps > 0 and count > 0:
            return float(count / fps)
    except Exception:
        pass

    logger.warning("Cannot determine duration for %s — using 60.0s fallback", video_path)
    return 60.0


# ═══════════════════════════════════════════════ ffmpeg / ffprobe paths

def get_ffmpeg_path() -> str:
    """
    Путь к ffmpeg: сначала PATH, потом tools/ffmpeg в проекте.

    Кроссплатформенный: Windows (.exe), Linux/macOS (без суффикса).
    Исправляет баг оригинала в cut_clips_from_result.py:
    там искался только *.exe — на macOS/Linux не работало.
    """
    exe = shutil.which("ffmpeg")
    if exe:
        return exe

    suffix = ".exe" if sys.platform == "win32" else ""
    tools = _PROJECT_ROOT / "tools" / "ffmpeg"
    if tools.exists():
        for pattern in (
            f"**/bin/ffmpeg{suffix}",
            f"**/ffmpeg{suffix}",
            f"ffmpeg{suffix}",
        ):
            found = list(tools.glob(pattern))
            if found:
                return str(found[0].resolve())

    return "ffmpeg"


def get_ffprobe_path() -> str:
    """
    Путь к ffprobe (ищет рядом с ffmpeg или в PATH).
    """
    exe = shutil.which("ffprobe")
    if exe:
        return exe

    ffmpeg = get_ffmpeg_path()
    if ffmpeg != "ffmpeg":
        parent = Path(ffmpeg).parent
        suffix = ".exe" if sys.platform == "win32" else ""
        probe = parent / f"ffprobe{suffix}"
        if probe.exists():
            return str(probe)

    return "ffprobe"


# ═══════════════════════════════════════════════ temporal NMS

def temporal_nms(
    candidates: List[Dict],
    iou_thresh: float = 0.5,
    start_key: str = "start",
    end_key: str = "end",
    score_key: str = "score",
) -> List[Dict]:
    """
    Temporal Non-Maximum Suppression.

    Убирает сильно перекрывающиеся кандидаты, оставляя с наибольшим score.

    Заменяет три независимые копии:
      - hook_mode_v1._temporal_nms
      - trailer_mode_v1._temporal_nms
      - story_mode_v1 (если появится)

    Args:
        candidates:  список клипов с start/end/score
        iou_thresh:  IoU порог (0.5 = 50% перекрытия → удалить)
        start_key:   ключ начала
        end_key:     ключ конца
        score_key:   ключ score (сортировка по убыванию)

    Returns:
        Отфильтрованный список без значительных перекрытий.
    """
    if not candidates:
        return []

    sorted_cands = sorted(
        candidates, key=lambda c: c.get(score_key, 0.0), reverse=True
    )
    kept: List[Dict] = []

    for cand in sorted_cands:
        s = float(cand.get(start_key, 0))
        e = float(cand.get(end_key, 0))
        if e <= s:
            continue

        overlap = False
        for k in kept:
            ks = float(k.get(start_key, 0))
            ke = float(k.get(end_key, 0))
            inter = max(0.0, min(e, ke) - max(s, ks))
            union = (e - s) + (ke - ks) - inter
            if union > 0 and inter / union >= iou_thresh:
                overlap = True
                break

        if not overlap:
            kept.append(cand)

    return kept


# ═══════════════════════════════════════════════ safe conversions

def safe_float(value: Any) -> Optional[float]:
    """
    Безопасное приведение к float.

    Возвращает None для: None, NaN, Inf, строк которые нельзя распарсить.
    Заменяет паттерн `x != x` (NaN-хак) в нескольких файлах.
    """
    if value is None:
        return None
    try:
        v = float(value)
    except (ValueError, TypeError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def safe_int(value: Any, default: int = 0) -> int:
    """Безопасное приведение к int с fallback."""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


# ═══════════════════════════════════════════════ filename utilities

def sanitize_filename(text: str, max_len: int = 60) -> str:
    """
    Очищает строку для безопасного использования в имени файла.
    Убирает спецсимволы, схлопывает подчёркивания, обрезает по длине.
    """
    safe = _UNSAFE_FILENAME_RE.sub("_", str(text))
    safe = re.sub(r"_+", "_", safe).strip("_")
    return safe[:max_len] if safe else "unnamed"


# ═══════════════════════════════════════════════ smoke test
if __name__ == "__main__":
    # temporal_nms
    cands = [
        {"start": 0, "end": 10, "score": 0.9},
        {"start": 5, "end": 15, "score": 0.7},   # пересекается с первым
        {"start": 20, "end": 30, "score": 0.8},  # не пересекается
    ]
    kept = temporal_nms(cands, iou_thresh=0.3)
    assert len(kept) == 2, f"Expected 2, got {len(kept)}"
    assert kept[0]["start"] == 0
    assert kept[1]["start"] == 20

    # safe_float
    assert safe_float(None) is None
    assert safe_float("bad") is None
    assert safe_float(float("nan")) is None
    assert safe_float(3.14) == 3.14

    # sanitize_filename
    assert sanitize_filename("hello/world:test") == "hello_world_test"

    # get_ffmpeg_path / get_ffprobe_path — just call without crashing
    _ = get_ffmpeg_path()
    _ = get_ffprobe_path()

    print("utils.py smoke test passed ✓")
