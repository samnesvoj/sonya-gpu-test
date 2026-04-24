"""
SONYA GPU Benchmark Runner
==========================
Прогоняет видео через режимы SONYA, сохраняет все debug-артефакты,
runtime-метрики, шаблон ручной оценки и summary.csv.

Запуск:
    python benchmark_runner.py \
        --videos data/input \
        --modes hook story trailer_preview \
        --model yolov8n \
        --output outputs/test_today

Тестирует по 4 уровням:
    1. Технически не падает
    2. Находит нормальные кандидаты
    3. Выбирает правильный top-result
    4. Итоговый клип соответствует режиму

Архитектура адаптеров:
    Каждый адаптер пробует подключить реальный модуль режима.
    Если реальный модуль упал — fallback в _stub_result с логированием ошибки.
    ASR (Whisper) вычисляется один раз на видео и передаётся во все режимы.
    trailer_preview получает кешированные результаты hook/story/viral/educational
    из текущего прогона (если они уже были запущены).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# GPU / VRAM мониторинг
# ─────────────────────────────────────────────────────────────────────────────

def _gpu_info() -> Tuple[bool, str, float]:
    """Возвращает (gpu_available, gpu_name, vram_total_mb)."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode()
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return True, name, mem.total / 1024 / 1024
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            total = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
            return True, name, total
    except Exception:
        pass
    return False, "CPU", 0.0


def _vram_used_mb() -> float:
    """Текущее использование VRAM в МБ."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return mem.used / 1024 / 1024
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated(0) / 1024 / 1024
    except Exception:
        pass
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Метаданные входного видео
# ─────────────────────────────────────────────────────────────────────────────

def extract_input_metadata(video_path: Path) -> Dict[str, Any]:
    """Извлекает технические метаданные входного видео через OpenCV."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    duration_sec = frame_count / fps if fps > 0 else 0
    size_mb = video_path.stat().st_size / 1024 / 1024

    return {
        "filename": video_path.name,
        "path": str(video_path),
        "duration_sec": round(duration_sec, 2),
        "fps": round(fps, 2),
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "size_mb": round(size_mb, 2),
        "aspect_ratio": f"{width}x{height}",
    }


# ─────────────────────────────────────────────────────────────────────────────
# YOLO базовый анализ (общий для всех режимов)
# ─────────────────────────────────────────────────────────────────────────────

def run_yolo_analysis(
    video_path: Path, model_name: str
) -> Tuple[Dict[str, Any], float]:
    """
    Прогоняет YOLOv8 по ключевым кадрам видео.
    Возвращает (base_analysis_dict, elapsed_sec).
    """
    t0 = time.perf_counter()
    try:
        from ultralytics import YOLO

        model_file = model_name if model_name.endswith(".pt") else f"{model_name}.pt"
        scripts_dir = Path(__file__).parent
        model_path = scripts_dir / model_file
        if not model_path.exists():
            model_path = model_file  # позволяем YOLO скачать самостоятельно

        yolo = YOLO(str(model_path))

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        sample_step = max(1, int(fps * 2))  # каждые 2 секунды

        detections = []
        frame_idx = 0
        while frame_idx < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            results = yolo(frame, verbose=False)
            timestamp = frame_idx / fps
            det = {
                "timestamp_sec": round(timestamp, 2),
                "person_count": 0,
                "objects": [],
                "confidence_max": 0.0,
            }
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    cls_name = yolo.names.get(cls_id, str(cls_id))
                    det["objects"].append({"class": cls_name, "confidence": round(conf, 3)})
                    if cls_name == "person":
                        det["person_count"] += 1
                    det["confidence_max"] = max(det["confidence_max"], conf)
            detections.append(det)
            frame_idx += sample_step
        cap.release()

        person_frames = [d for d in detections if d["person_count"] > 0]
        elapsed = time.perf_counter() - t0
        return {
            "model": model_name,
            "total_frames_sampled": len(detections),
            "person_presence_ratio": round(len(person_frames) / max(len(detections), 1), 3),
            "avg_confidence": round(
                np.mean([d["confidence_max"] for d in detections]) if detections else 0, 3
            ),
            "detections": detections,
        }, elapsed

    except Exception as e:
        elapsed = time.perf_counter() - t0
        logger.warning(f"YOLO analysis failed: {e}")
        return {"model": model_name, "error": str(e), "detections": []}, elapsed


# ─────────────────────────────────────────────────────────────────────────────
# АДАПТЕРЫ РЕЖИМОВ
# Сначала пробуем подключить реальный файл режима.
# Если не получилось — используем стаб-заглушку.
# ─────────────────────────────────────────────────────────────────────────────

SCRIPTS_DIR = Path(__file__).parent / "scripts"
if SCRIPTS_DIR.exists():
    sys.path.insert(0, str(SCRIPTS_DIR))


# ─────────────────────────────────────────────────────────────────────────────
# ASR — Whisper, кэш на время процесса (один раз на видео)
# ─────────────────────────────────────────────────────────────────────────────

def run_asr(video_path: Path, whisper_model: str = "base") -> Tuple[List[Dict], float]:
    """
    Транскрибирует видео через Whisper (asr.py из проекта).
    Кэшируется на уровне процесса по (video_path, model).
    Возвращает (segments, elapsed_sec).
    """
    t0 = time.perf_counter()
    try:
        from asr import transcribe_video  # из SONYA-DATASET/scripts/asr.py
        segments = transcribe_video(str(video_path), model_size=whisper_model)
        return segments or [], time.perf_counter() - t0
    except ImportError:
        pass
    try:
        import whisper  # прямой fallback
        model = whisper.load_model(whisper_model)
        result = model.transcribe(str(video_path))
        segments = [
            {"start": s["start"], "end": s["end"], "text": s["text"]}
            for s in result.get("segments", [])
        ]
        return segments, time.perf_counter() - t0
    except Exception as e:
        logger.warning(f"ASR failed ({video_path.name}): {e} — режимы без транскрипта")
        return [], time.perf_counter() - t0


def _stub_result(mode: str, video_path: Path, base_analysis: Dict) -> Dict[str, Any]:
    """
    Универсальная заглушка — имитирует структуру результата режима.
    Используется когда реальный модуль не подключён.
    """
    dur = base_analysis.get("detections", [{}])
    n_cands = max(3, len(dur) // 5)
    candidates = []
    video_dur = 0.0
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    video_dur = total / fps if fps > 0 else 60.0

    step = max(5.0, video_dur / (n_cands + 1))
    for i in range(n_cands):
        start = round(step * (i + 1), 2)
        end = round(min(start + 8.0, video_dur), 2)
        candidates.append({
            "id": i + 1,
            "start_sec": start,
            "end_sec": end,
            "score": round(0.9 - i * 0.12, 3),
            "reason": f"stub_candidate_{mode}_{i+1}",
            "source": "stub",
        })

    ranking = sorted(candidates, key=lambda x: x["score"], reverse=True)
    top1 = ranking[0] if ranking else {}
    export_decision = "manual_review"  # стаб всегда manual_review

    result: Dict[str, Any] = {
        "mode": mode,
        "stub": True,
        "candidates": candidates,
        "ranking": ranking,
        "top1": top1,
        "export_decision": export_decision,
        "boundary_diagnostics": {
            "top1_boundary_quality": "unknown",
            "cut_at_pause": None,
            "cut_at_sentence": None,
        },
        "mode_metrics": {},
    }

    # Режим-специфичные поля
    if mode == "story":
        result["story_events"] = [{"type": "stub_event", "start": top1.get("start_sec", 0)}]
        result["beats"] = []
        result["arcs"] = [{"arc": "stub_arc", "score": 0.5}]
        result["role_probs"] = {"exposition": 0.4, "conflict": 0.3, "resolution": 0.3}
        result["mode_metrics"] = {
            "story_event_recall": None,
            "arc_detected": False,
            "payoff_preserved": None,
            "self_sufficient": None,
            "context_missing_risk": None,
            "story_vs_education_confusion": None,
            "story_boundary_ok": None,
        }

    elif mode == "hook":
        result["mode_metrics"] = {
            "hook_candidate_recall": None,
            "visual_first_hit": None,
            "text_bias_risk": None,
            "non_hook_risk": None,
            "top3_hook_good": None,
            "first_3_sec_strength": None,
            "hook_boundary_ok": None,
        }

    elif mode == "trailer_preview":
        result["themes"] = [{"theme_id": 1, "label": "stub_theme", "candidates": [1, 2]}]
        result["slots"] = [{"slot": "intro", "candidate_id": 1}, {"slot": "climax", "candidate_id": 2}]
        result["transition_graph"] = {"edges": []}
        result["assembly_plan"] = {"sequence": [1, 2, 3], "total_duration_sec": 30}
        result["ui_payload"] = {"editable": True, "slots": result["slots"]}
        result["mode_metrics"] = {
            "sequence_coherence": None,
            "theme_diversity": None,
            "transition_quality": None,
            "spoiler_risk": None,
            "preview_intrigue": None,
            "over_explanation_risk": None,
            "scene_redundancy": None,
            "ending_sting_quality": None,
            "ui_editability": True,
        }

    elif mode == "viral":
        result["mode_metrics"] = {
            "energy_peak_detected": None,
            "reaction_strength": None,
            "visual_spike_score": None,
            "short_form_fit": None,
            "wow_moment_present": None,
            "fake_viral_risk": None,
        }

    elif mode == "educational":
        result["mode_metrics"] = {
            "explanation_detected": None,
            "definition_present": None,
            "step_by_step_quality": None,
            "example_present": None,
            "clarity_score": None,
            "value_density": None,
            "boring_risk": None,
        }

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Нормализаторы выхода реальных режимов → стандартный формат
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_moments(
    moments: List[Dict],
    start_key: str = "start",
    end_key: str = "end",
    score_key: str = "score",
) -> Tuple[List[Dict], List[Dict], Dict]:
    """
    Превращает список моментов реального режима в (candidates, ranking, top1).
    Поля start/end могут называться start_sec/end_sec у некоторых режимов.
    """
    candidates = []
    for i, m in enumerate(moments):
        start = m.get(start_key) or m.get("start_sec") or m.get("start", 0.0)
        end = m.get(end_key) or m.get("end_sec") or m.get("end", start + 10.0)
        score = m.get(score_key) or m.get("virality_score") or m.get("hook_score", 0.0)
        candidates.append({
            "id": i + 1,
            "start_sec": float(start),
            "end_sec": float(end),
            "score": round(float(score), 4),
            "reason": m.get("hook_type") or m.get("story_type") or m.get("segment_type")
                      or m.get("viral_type") or "",
            "source": "real",
            "_raw": m,
        })
    ranking = sorted(candidates, key=lambda x: x["score"], reverse=True)
    top1 = ranking[0] if ranking else {}
    return candidates, ranking, top1


def _export_decision_from_score(top1: Dict, mode: str) -> str:
    """Авто-решение на экспорт по score top-1."""
    score = top1.get("score", 0.0) if top1 else 0.0
    thresholds = {
        "hook": (0.65, 0.40),
        "story": (0.60, 0.35),
        "trailer_preview": (0.55, 0.35),
        "viral": (0.62, 0.38),
        "educational": (0.55, 0.35),
    }
    high, low = thresholds.get(mode, (0.60, 0.35))
    if score >= high:
        return "auto_export"
    if score >= low:
        return "manual_review"
    return "reject"


# ─────────────────────────────────────────────────────────────────────────────
# АДАПТЕРЫ РЕЖИМОВ
# Сигнатура каждого: (video_path, video_duration_sec, base_analysis,
#                     asr_segments, model_name, mode_results_cache) -> (result, elapsed)
# mode_results_cache: Dict[str, Dict] — результаты уже прогнанных режимов для
#                     этого видео (нужно trailer_preview)
# ─────────────────────────────────────────────────────────────────────────────

def run_hook(
    video_path: Path,
    video_duration_sec: float,
    base_analysis: Dict,
    asr_segments: List[Dict],
    model_name: str,
    mode_results_cache: Dict,
) -> Tuple[Dict, float]:
    t0 = time.perf_counter()
    adapter_error: Optional[str] = None
    try:
        from hook_mode_v1 import find_hook_moments  # type: ignore

        raw = find_hook_moments(
            video_path=str(video_path),
            video_duration_sec=video_duration_sec,
            asr_segments=asr_segments or None,
            base_analysis=base_analysis,
            top_k=5,
        )
        moments = raw.get("hook_moments", [])
        candidates, ranking, top1 = _normalize_moments(moments, score_key="score")
        export_decision = _export_decision_from_score(top1, "hook")

        # Метрики из stats
        stats = raw.get("stats", {})
        top3 = ranking[:3]
        mode_metrics = {
            "hook_candidate_recall": len(candidates) > 0,
            "visual_first_hit": stats.get("visual_first_hit"),
            "text_bias_risk": stats.get("text_bias_risk"),
            "non_hook_risk": None,
            "top3_hook_good": None,  # только ручная проверка
            "first_3_sec_strength": (
                top3[0]["start_sec"] < 3.0 if top3 else None
            ),
            "hook_boundary_ok": None,
        }
        boundary_diag = top3[0].get("_raw", {}).get("boundary_diagnostics", {}) if top3 else {}

        result: Dict[str, Any] = {
            "mode": "hook",
            "stub": False,
            "candidates": candidates,
            "ranking": ranking,
            "top1": top1,
            "export_decision": export_decision,
            "boundary_diagnostics": boundary_diag,
            "mode_metrics": mode_metrics,
        }
        return result, time.perf_counter() - t0

    except Exception as e:
        adapter_error = f"{type(e).__name__}: {e}"
        logger.warning(f"[hook] реальный режим упал → stub. Ошибка: {adapter_error}")

    result = _stub_result("hook", video_path, base_analysis)
    result["adapter_error"] = adapter_error
    return result, time.perf_counter() - t0


def run_story(
    video_path: Path,
    video_duration_sec: float,
    base_analysis: Dict,
    asr_segments: List[Dict],
    model_name: str,
    mode_results_cache: Dict,
) -> Tuple[Dict, float]:
    t0 = time.perf_counter()
    adapter_error: Optional[str] = None
    try:
        from story_mode_v1 import find_story_moments  # type: ignore

        raw = find_story_moments(
            video_path=str(video_path),
            video_duration_sec=video_duration_sec,
            asr_segments=asr_segments or None,
            base_analysis=base_analysis,
            top_k=5,
        )
        moments = raw.get("story_moments", [])
        candidates, ranking, top1 = _normalize_moments(moments, score_key="score")
        export_decision = _export_decision_from_score(top1, "story")

        # Story-специфичные артефакты из первого момента
        first_raw = ranking[0]["_raw"] if ranking else {}
        story_events = raw.get("story_events", [
            {"type": m["_raw"].get("story_type", ""), "start": m["start_sec"]}
            for m in candidates
        ])
        beats = raw.get("beats", [
            b for m in candidates
            for b in m["_raw"].get("story_beats", [])
        ])
        arcs = raw.get("arcs", [
            {"arc_id": m["_raw"].get("arc_id"), "arc_pattern": m["_raw"].get("arc_pattern"),
             "arc_confidence": m["_raw"].get("arc_confidence")}
            for m in candidates if m["_raw"].get("arc_id") is not None
        ])
        role_probs_raw = first_raw.get("role_sequence", [])
        role_probs: Dict = {}
        if isinstance(role_probs_raw, list):
            for role in role_probs_raw:
                role_probs[role] = role_probs.get(role, 0) + 1

        mode_metrics = {
            "story_event_recall": len(candidates) > 0,
            "arc_detected": any(m["_raw"].get("arc_id") is not None for m in candidates),
            "payoff_preserved": first_raw.get("payoff_strength", None),
            "self_sufficient": first_raw.get("story_export_safety", None),
            "context_missing_risk": None,
            "story_vs_education_confusion": None,
            "story_boundary_ok": first_raw.get("boundary_diagnostics", {}).get("ok"),
        }
        boundary_diag = first_raw.get("boundary_diagnostics", {})

        result: Dict[str, Any] = {
            "mode": "story",
            "stub": False,
            "candidates": candidates,
            "ranking": ranking,
            "top1": top1,
            "export_decision": export_decision,
            "boundary_diagnostics": boundary_diag,
            "mode_metrics": mode_metrics,
            "story_events": story_events,
            "beats": beats,
            "arcs": arcs,
            "role_probs": role_probs,
        }
        return result, time.perf_counter() - t0

    except Exception as e:
        adapter_error = f"{type(e).__name__}: {e}"
        logger.warning(f"[story] реальный режим упал → stub. Ошибка: {adapter_error}")

    result = _stub_result("story", video_path, base_analysis)
    result["adapter_error"] = adapter_error
    return result, time.perf_counter() - t0


def run_viral(
    video_path: Path,
    video_duration_sec: float,
    base_analysis: Dict,
    asr_segments: List[Dict],
    model_name: str,
    mode_results_cache: Dict,
) -> Tuple[Dict, float]:
    t0 = time.perf_counter()
    adapter_error: Optional[str] = None
    try:
        from modes_scoring import find_viral_moments_v3  # type: ignore

        moments = find_viral_moments_v3(
            base_analysis=base_analysis,
            video_duration_sec=video_duration_sec,
            video_path=str(video_path),
            asr_segments=asr_segments or None,
            top_k=5,
            threshold=0.5,
        )
        # find_viral_moments_v3 возвращает список напрямую
        if not isinstance(moments, list):
            moments = moments.get("viral_moments", []) if isinstance(moments, dict) else []

        candidates, ranking, top1 = _normalize_moments(
            moments, start_key="start", end_key="end", score_key="virality_score"
        )
        export_decision = _export_decision_from_score(top1, "viral")

        top1_raw = ranking[0]["_raw"] if ranking else {}
        mode_metrics = {
            "energy_peak_detected": len(candidates) > 0,
            "reaction_strength": top1_raw.get("semantic_score"),
            "visual_spike_score": top1_raw.get("visual_score"),
            "short_form_fit": None,
            "wow_moment_present": None,
            "fake_viral_risk": None,
        }

        result: Dict[str, Any] = {
            "mode": "viral",
            "stub": False,
            "candidates": candidates,
            "ranking": ranking,
            "top1": top1,
            "export_decision": export_decision,
            "boundary_diagnostics": {},
            "mode_metrics": mode_metrics,
        }
        return result, time.perf_counter() - t0

    except Exception as e:
        adapter_error = f"{type(e).__name__}: {e}"
        logger.warning(f"[viral] реальный режим упал → stub. Ошибка: {adapter_error}")

    result = _stub_result("viral", video_path, base_analysis)
    result["adapter_error"] = adapter_error
    return result, time.perf_counter() - t0


def run_educational(
    video_path: Path,
    video_duration_sec: float,
    base_analysis: Dict,
    asr_segments: List[Dict],
    model_name: str,
    mode_results_cache: Dict,
) -> Tuple[Dict, float]:
    t0 = time.perf_counter()
    adapter_error: Optional[str] = None
    try:
        from educational_mode_v5 import run_educational_mode_v5  # type: ignore

        raw = run_educational_mode_v5(
            video_path=str(video_path),
            asr_segments=asr_segments or None,
            topic_segments=None,   # auto-degraded inside the mode
            base_analysis=base_analysis,
            mode="educational",
        )
        moments = raw.get("educational_moments", [])
        candidates, ranking, top1 = _normalize_moments(moments, score_key="score")
        export_decision = _export_decision_from_score(top1, "educational")

        top1_raw = ranking[0]["_raw"] if ranking else {}
        mode_metrics = {
            "explanation_detected": len(candidates) > 0,
            "definition_present": top1_raw.get("has_definition"),
            "step_by_step_quality": top1_raw.get("step_score"),
            "example_present": top1_raw.get("has_example"),
            "clarity_score": top1_raw.get("clarity_score"),
            "value_density": top1_raw.get("pedagogical_density"),
            "boring_risk": top1_raw.get("confusion_risk"),
        }
        boundary_diag = top1_raw.get("boundary_diagnostics", {})

        result: Dict[str, Any] = {
            "mode": "educational",
            "stub": False,
            "candidates": candidates,
            "ranking": ranking,
            "top1": top1,
            "export_decision": export_decision,
            "boundary_diagnostics": boundary_diag,
            "mode_metrics": mode_metrics,
        }
        return result, time.perf_counter() - t0

    except Exception as e:
        adapter_error = f"{type(e).__name__}: {e}"
        logger.warning(f"[educational] реальный режим упал → stub. Ошибка: {adapter_error}")

    result = _stub_result("educational", video_path, base_analysis)
    result["adapter_error"] = adapter_error
    return result, time.perf_counter() - t0


def run_trailer_preview(
    video_path: Path,
    video_duration_sec: float,
    base_analysis: Dict,
    asr_segments: List[Dict],
    model_name: str,
    mode_results_cache: Dict,
) -> Tuple[Dict, float]:
    """
    Trailer/Preview — зависит от результатов других режимов.
    Берёт hook/story/viral/educational из mode_results_cache если они уже
    прогонялись для этого видео. Иначе реальный режим работает с None-ами.
    """
    t0 = time.perf_counter()
    adapter_error: Optional[str] = None
    try:
        from trailer_mode_v3 import find_trailer_clips, TrailerModeConfig  # type: ignore

        hook_result = mode_results_cache.get("hook", {}).get("_raw_mode_output")
        story_result = mode_results_cache.get("story", {}).get("_raw_mode_output")
        viral_result = mode_results_cache.get("viral", {}).get("_raw_mode_output")
        edu_result = mode_results_cache.get("educational", {}).get("_raw_mode_output")

        raw = find_trailer_clips(
            video_path=str(video_path),
            video_duration_sec=video_duration_sec,
            hook_result=hook_result,
            story_result=story_result,
            viral_result=viral_result,
            educational_result=edu_result,
            base_analysis=base_analysis,
            asr_segments=asr_segments or None,
        )

        clips = raw.get("trailer_clips", [])
        candidates, ranking, top1 = _normalize_moments(clips, score_key="score")
        export_decision = _export_decision_from_score(top1, "trailer_preview")

        theme_blocks_raw = raw.get("theme_blocks", [])
        themes = [
            {
                "theme_id": b.get("theme_id"),
                "label": b.get("label"),
                "start": b.get("start"),
                "end": b.get("end"),
                "importance": b.get("importance"),
            }
            for b in theme_blocks_raw
        ]

        render_instructions = raw.get("render_instructions", [])
        assembly_plan = {
            "sequence": [c.get("id") for c in candidates],
            "total_duration_sec": sum(
                (c["end_sec"] - c["start_sec"]) for c in candidates
            ),
            "render_instructions": render_instructions,
        }

        stats = raw.get("stats", {})
        mode_metrics = {
            "sequence_coherence": stats.get("avg_transition_score"),
            "theme_diversity": stats.get("num_themes"),
            "transition_quality": stats.get("avg_transition_score"),
            "spoiler_risk": stats.get("spoiler_agg"),
            "preview_intrigue": stats.get("curiosity_agg"),
            "over_explanation_risk": None,
            "scene_redundancy": None,
            "ending_sting_quality": stats.get("tease_strength"),
            "ui_editability": True,
        }

        result: Dict[str, Any] = {
            "mode": "trailer_preview",
            "stub": False,
            "candidates": candidates,
            "ranking": ranking,
            "top1": top1,
            "export_decision": export_decision,
            "boundary_diagnostics": {},
            "mode_metrics": mode_metrics,
            "themes": themes,
            "slots": render_instructions,
            "transition_graph": {"edges": raw.get("transition_edges", [])},
            "assembly_plan": assembly_plan,
            "ui_payload": raw.get("ui_payload", {}),
            "_raw_mode_output": raw,
        }
        return result, time.perf_counter() - t0

    except Exception as e:
        adapter_error = f"{type(e).__name__}: {e}"
        logger.warning(f"[trailer_preview] реальный режим упал → stub. Ошибка: {adapter_error}")

    result = _stub_result("trailer_preview", video_path, base_analysis)
    result["adapter_error"] = adapter_error
    return result, time.perf_counter() - t0


MODE_RUNNERS = {
    "hook": run_hook,
    "story": run_story,
    "trailer_preview": run_trailer_preview,
    "viral": run_viral,
    "educational": run_educational,
}


# ─────────────────────────────────────────────────────────────────────────────
# Экспорт клипа
# ─────────────────────────────────────────────────────────────────────────────

def export_clip(
    video_path: Path,
    start_sec: float,
    end_sec: float,
    output_path: Path,
) -> Tuple[bool, float]:
    """Нарезает клип через ffmpeg-python. Возвращает (success, elapsed_sec)."""
    t0 = time.perf_counter()
    try:
        import ffmpeg

        (
            ffmpeg
            .input(str(video_path), ss=start_sec, to=end_sec)
            .output(str(output_path), vcodec="libx264", acodec="aac", loglevel="quiet")
            .overwrite_output()
            .run()
        )
        elapsed = time.perf_counter() - t0
        return True, elapsed
    except Exception as e:
        logger.warning(f"ffmpeg export failed: {e}")
        elapsed = time.perf_counter() - t0
        return False, elapsed


# ─────────────────────────────────────────────────────────────────────────────
# Шаблон ручной оценки
# ─────────────────────────────────────────────────────────────────────────────

HUMAN_REVIEW_TEMPLATE: Dict[str, Any] = {
    "human_rating": None,          # 1–5: общая оценка клипа
    "mode_match": None,            # true / false / "partial"
    "top3_contains_good": None,    # true / false
    "boundary_ok": None,           # true / false — нормальна ли обрезка
    "would_publish": None,         # true / false — можно ли публиковать
    "failure_reason": None,        # proposer / ranking / boundary / quality_gate /
                                   # transition / assembly / runtime / bad_input
    "notes": "",
}

# ─────────────────────────────────────────────────────────────────────────────
# Русские метки метрик
# ─────────────────────────────────────────────────────────────────────────────

METRIC_LABELS_RU: Dict[str, str] = {
    # Общие runtime-метрики
    "runtime_total_sec":      "Общее время обработки (сек)",
    "yolo_sec":               "Время YOLO-анализа (сек)",
    "mode_logic_sec":         "Время логики режима (сек)",
    "export_sec":             "Время экспорта клипа (сек)",
    "vram_peak_mb":           "Пик VRAM (МБ)",
    "gpu_available":          "GPU доступна",
    "gpu_name":               "Имя GPU",
    "num_candidates":         "Количество кандидатов",
    "export_decision":        "Решение на экспорт (auto/manual/reject)",
    "top1_score":             "Оценка лучшего кандидата",
    "top3_contains_good":     "Хороший вариант есть в top-3",
    "boundary_ok":            "Качество границ (нет обрезки фразы)",
    # Hook
    "hook_candidate_recall":  "Hook: нашёл ли реальный hook вообще",
    "visual_first_hit":       "Hook: нашёл визуальный hook без текста",
    "text_bias_risk":         "Hook: риск победы текстового мусора",
    "non_hook_risk":          "Hook: риск выбора случайного громкого момента",
    "top3_hook_good":         "Hook: хороший hook в top-3",
    "first_3_sec_strength":   "Hook: цепляют ли первые 3 секунды",
    "hook_boundary_ok":       "Hook: нормальная ли обрезка начала",
    # Story
    "story_event_recall":     "Story: нашёл ли важные события истории",
    "arc_detected":           "Story: есть ли дуга начало→конфликт→payoff",
    "payoff_preserved":       "Story: сохранился ли финальный смысл",
    "self_sufficient":        "Story: понятен ли клип без полного видео",
    "context_missing_risk":   "Story: риск потери важного контекста",
    "story_vs_education_confusion": "Story: риск спутать историю с объяснением",
    "story_boundary_ok":      "Story: не режет ли фразу/сцену",
    # Trailer/Preview
    "sequence_coherence":     "Trailer: логика между кусками",
    "theme_diversity":        "Trailer: разнообразие тем (не повтор)",
    "transition_quality":     "Trailer: качество переходов",
    "spoiler_risk":           "Trailer: риск раскрыть финал",
    "preview_intrigue":       "Trailer: хочется ли смотреть полное видео",
    "over_explanation_risk":  "Trailer: риск превратиться в пересказ",
    "scene_redundancy":       "Trailer: нет ли одинаковых сцен подряд",
    "ending_sting_quality":   "Trailer: сила финального крючка",
    "ui_editability":         "Trailer: можно ли редактировать через UI payload",
    # Viral
    "energy_peak_detected":   "Viral: найден ли пик энергии",
    "reaction_strength":      "Viral: есть ли реакция/эмоция",
    "visual_spike_score":     "Viral: есть ли визуальный всплеск",
    "short_form_fit":         "Viral: подходит ли для Shorts",
    "wow_moment_present":     "Viral: есть ли \"вау\"-момент",
    "fake_viral_risk":        "Viral: риск выбрать просто шум",
    # Educational
    "explanation_detected":   "Edu: найден ли объяснительный блок",
    "definition_present":     "Edu: есть ли определение/понятие",
    "step_by_step_quality":   "Edu: есть ли пошаговость",
    "example_present":        "Edu: есть ли пример",
    "clarity_score":          "Edu: понятно ли без контекста",
    "value_density":          "Edu: сколько пользы на минуту",
    "boring_risk":            "Edu: риск скучного монтажа",
    # Человеческая оценка
    "human_rating":           "Оценка человека (1–5)",
    "mode_match":             "Соответствует ли режиму (true/false/partial)",
    "would_publish":          "Можно ли публиковать",
    "failure_reason":         "Что сломалось (proposer/ranking/boundary/…)",
    "notes":                  "Заметки",
}

# Уровни зрелости режима
MATURITY_LEVELS = {
    1: "🔴 Уровень 1 — технически запустился",
    2: "🟡 Уровень 2 — находит кандидатов",
    3: "🟡 Уровень 3 — правильно ранжирует",
    4: "🟢 Уровень 4 — хорошо монтирует",
    5: "🔵 Уровень 5 — guarded production",
}

# MVP-нормы для авто-оценки зрелости (без ручной проверки)
MVP_NORMS = {
    "hook":            {"top3_contains_good": 0.70, "boundary_ok": 0.75},
    "story":           {"top3_contains_good": 0.60, "boundary_ok": 0.75},
    "trailer_preview": {"top3_contains_good": 0.55, "boundary_ok": 0.75},
    "viral":           {"top3_contains_good": 0.65, "boundary_ok": 0.75},
    "educational":     {"top3_contains_good": 0.60, "boundary_ok": 0.75},
}


# ─────────────────────────────────────────────────────────────────────────────
# Вспомогательные функции
# ─────────────────────────────────────────────────────────────────────────────

def _rebuild_raw(mode: str, result: Dict) -> Dict:
    """
    Строит raw_mode_output совместимый с trailer_mode_v3 из нормализованного
    результата. Используется когда реальный режим не вернул _raw_mode_output.
    """
    candidates_raw = [c.get("_raw", c) for c in result.get("candidates", [])]
    key_map = {
        "hook": "hook_moments",
        "story": "story_moments",
        "viral": "viral_moments",
        "educational": "educational_moments",
    }
    key = key_map.get(mode, f"{mode}_moments")
    return {key: candidates_raw, "mode": mode}


# ─────────────────────────────────────────────────────────────────────────────
# Главный прогон одного run
# ─────────────────────────────────────────────────────────────────────────────

def run_single(
    video_path: Path,
    mode: str,
    model_name: str,
    output_dir: Path,
    *,
    video_duration_sec: float = 0.0,
    base_analysis: Optional[Dict] = None,
    yolo_sec_precomputed: float = 0.0,
    asr_segments: Optional[List[Dict]] = None,
    mode_results_cache: Optional[Dict] = None,
    skip_export: bool = False,
) -> Dict[str, Any]:
    """
    Прогоняет одно видео через один режим.
    Сохраняет все JSON-артефакты в output_dir.
    Возвращает строку для summary.csv.

    Параметры pre-computed принимаются из main() чтобы не пересчитывать
    YOLO и ASR для каждого режима на одном видео.
    """
    run_id = f"{video_path.stem}__{mode}__{model_name}__{uuid.uuid4().hex[:6]}"
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"▶ Старт: {run_id}")
    t_total_start = time.perf_counter()

    gpu_available, gpu_name, _ = _gpu_info()
    vram_before = _vram_used_mb()

    # ── 1. Метаданные входного видео ─────────────────────────────────────────
    input_meta = extract_input_metadata(video_path)
    _save_json(run_dir / "input_metadata.json", input_meta)

    # ── 2. YOLO анализ (повторный только если не передан) ────────────────────
    if base_analysis is None:
        base_analysis, yolo_sec = run_yolo_analysis(video_path, model_name)
    else:
        yolo_sec = yolo_sec_precomputed
    vram_after_yolo = _vram_used_mb()

    # ── 3. Режим ──────────────────────────────────────────────────────────────
    runner = MODE_RUNNERS.get(mode)
    if runner is None:
        raise ValueError(f"Неизвестный режим: {mode}")

    mode_result, mode_logic_sec = runner(
        video_path,
        video_duration_sec or input_meta["duration_sec"],
        base_analysis,
        asr_segments or [],
        model_name,
        mode_results_cache or {},
    )
    # Сохраняем raw output в cache для trailer
    if mode_results_cache is not None:
        # Сохраняем как _raw_mode_output чтобы trailer мог получить
        mode_results_cache[mode] = {"_raw_mode_output": mode_result.get("_raw_mode_output") or _rebuild_raw(mode, mode_result)}

    candidates = mode_result.get("candidates", [])
    ranking = mode_result.get("ranking", [])
    top1 = mode_result.get("top1", {})
    export_decision = mode_result.get("export_decision", "manual_review")
    boundary_diag = mode_result.get("boundary_diagnostics", {})
    mode_metrics = mode_result.get("mode_metrics", {})
    is_stub = mode_result.get("stub", False)

    _save_json(run_dir / "candidates.json", candidates)
    _save_json(run_dir / "ranking.json", ranking)
    _save_json(run_dir / "export_decision.json", {"decision": export_decision})
    _save_json(run_dir / "boundary_diagnostics.json", boundary_diag)

    # Режим-специфичные артефакты
    if mode == "story":
        _save_json(run_dir / "story_events.json", mode_result.get("story_events", []))
        _save_json(run_dir / "beats.json", mode_result.get("beats", []))
        _save_json(run_dir / "arcs.json", mode_result.get("arcs", []))
        _save_json(run_dir / "role_probs.json", mode_result.get("role_probs", {}))

    if mode == "trailer_preview":
        _save_json(run_dir / "themes.json", mode_result.get("themes", []))
        _save_json(run_dir / "slots.json", mode_result.get("slots", []))
        _save_json(run_dir / "transition_graph.json", mode_result.get("transition_graph", {}))
        _save_json(run_dir / "assembly_plan.json", mode_result.get("assembly_plan", {}))
        _save_json(run_dir / "ui_payload.json", mode_result.get("ui_payload", {}))

    # ── 4. Экспорт клипа ──────────────────────────────────────────────────────
    export_sec = 0.0
    output_mp4 = run_dir / "output.mp4"
    clip_exported = False

    if not skip_export and top1 and export_decision != "reject":
        start = top1.get("start_sec", 0.0)
        end = top1.get("end_sec", start + 10.0)
        clip_exported, export_sec = export_clip(video_path, start, end, output_mp4)

    vram_peak = max(_vram_used_mb(), vram_after_yolo) - vram_before

    t_total_end = time.perf_counter()
    runtime_total_sec = round(t_total_end - t_total_start, 2)

    # ── 5. Runtime metrics ────────────────────────────────────────────────────
    top1_score = top1.get("score", None) if top1 else None
    num_candidates = len(candidates)

    adapter_error = mode_result.get("adapter_error")

    runtime_metrics = {
        "run_id": run_id,
        "video": video_path.name,
        "mode": mode,
        "model": model_name,
        "stub_mode": is_stub,
        "adapter_error": adapter_error,
        # Время
        "runtime_total_sec": runtime_total_sec,
        "yolo_sec": round(yolo_sec, 2),
        "mode_logic_sec": round(mode_logic_sec, 2),
        "export_sec": round(export_sec, 2),
        # GPU
        "vram_peak_mb": round(vram_peak, 1),
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        # Результаты
        "num_candidates": num_candidates,
        "top1_score": top1_score,
        "export_decision": export_decision,
        "clip_exported": clip_exported,
        # Режим-специфичные метрики
        **mode_metrics,
        # Русские лейблы (для удобства чтения)
        "_labels_ru": {k: METRIC_LABELS_RU.get(k, k) for k in [
            "runtime_total_sec", "yolo_sec", "mode_logic_sec", "export_sec",
            "vram_peak_mb", "gpu_available", "gpu_name", "num_candidates",
            "top1_score", "export_decision", *mode_metrics.keys()
        ]},
        # Уровень зрелости (авто)
        "_maturity_level": _estimate_maturity_level(mode, num_candidates, export_decision, is_stub),
    }

    _save_json(run_dir / "runtime_metrics.json", runtime_metrics)

    # ── 6. Шаблон ручной оценки ───────────────────────────────────────────────
    human_review = {
        **HUMAN_REVIEW_TEMPLATE,
        "_run_id": run_id,
        "_video": video_path.name,
        "_mode": mode,
        "_model": model_name,
        "_instructions_ru": {
            "human_rating": "Оцени клип от 1 до 5",
            "mode_match": "true если клип соответствует режиму, false если нет, 'partial' если частично",
            "top3_contains_good": "true если хороший момент есть среди первых 3 кандидатов",
            "boundary_ok": "true если начало/конец клипа не режет фразу или сцену",
            "would_publish": "true если клип можно публиковать",
            "failure_reason": (
                "proposer — режим не нашёл нужные моменты\n"
                "ranking — нашёл, но выбрал плохой\n"
                "boundary — смысл хороший, но обрезка кривая\n"
                "quality_gate — режим отсеял хороший момент\n"
                "transition — плохие переходы (trailer)\n"
                "assembly — плохая сборка sequence (trailer)\n"
                "runtime — слишком медленно\n"
                "bad_input — плохое исходное видео"
            ),
        },
    }
    _save_json(run_dir / "human_review_template.json", human_review)

    logger.success(
        f"✓ {run_id} | кандидатов={num_candidates} | "
        f"top1_score={top1_score} | решение={export_decision} | "
        f"{runtime_total_sec}s | {'STUB' if is_stub else 'REAL'}"
    )

    return {
        "run_id": run_id,
        "video": video_path.name,
        "mode": mode,
        "model": model_name,
        "stub": is_stub,
        "runtime_total_sec": runtime_total_sec,
        "yolo_sec": round(yolo_sec, 2),
        "mode_logic_sec": round(mode_logic_sec, 2),
        "export_sec": round(export_sec, 2),
        "vram_peak_mb": round(vram_peak, 1),
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "num_candidates": num_candidates,
        "top1_score": top1_score,
        "export_decision": export_decision,
        "clip_exported": clip_exported,
        "maturity_level": runtime_metrics["_maturity_level"],
        "output_dir": str(run_dir),
    }


def _estimate_maturity_level(
    mode: str, num_candidates: int, export_decision: str, is_stub: bool
) -> int:
    """
    Авто-оценка уровня зрелости прогона (без ручной оценки).
    1 — технически запустился
    2 — нашёл кандидатов (>0)
    3 — export_decision != reject (ранжирование что-то выбрало)
    """
    if is_stub:
        return 1
    if num_candidates == 0:
        return 1
    if export_decision == "reject":
        return 2
    return 3  # выше 3 только после ручной оценки


# ─────────────────────────────────────────────────────────────────────────────
# Утилиты
# ─────────────────────────────────────────────────────────────────────────────

def _save_json(path: Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_summary_csv(rows: List[Dict], output_dir: Path) -> Path:
    if not rows:
        return output_dir / "summary.csv"

    fieldnames = list(rows[0].keys())
    path = output_dir / "summary.csv"
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def save_failure_breakdown(rows: List[Dict], output_dir: Path) -> Path:
    """Разбивка по полю failure_reason из human_review."""
    path = output_dir / "failure_breakdown_template.csv"
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "video", "mode", "model",
            "human_rating", "top3_contains_good", "boundary_ok", "mode_match",
            "would_publish", "failure_reason", "notes"
        ])
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "video": r.get("video", ""),
                "mode": r.get("mode", ""),
                "model": r.get("model", ""),
                "human_rating": None,
                "top3_contains_good": None,
                "boundary_ok": None,
                "mode_match": None,
                "would_publish": None,
                "failure_reason": None,
                "notes": "",
            })
    return path


def save_runtime_breakdown(rows: List[Dict], output_dir: Path) -> Path:
    path = output_dir / "runtime_breakdown.csv"
    fieldnames = [
        "video", "mode", "model", "stub",
        "runtime_total_sec", "yolo_sec", "mode_logic_sec", "export_sec",
        "vram_peak_mb", "gpu_name", "num_candidates", "export_decision"
    ]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def save_model_comparison(rows: List[Dict], output_dir: Path) -> Path:
    path = output_dir / "model_comparison.csv"
    fieldnames = [
        "video", "mode", "model", "runtime_total_sec",
        "yolo_sec", "vram_peak_mb", "num_candidates", "export_decision"
    ]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def save_progress_report(rows: List[Dict], output_dir: Path) -> None:
    """Текстовый отчёт с прогрессом по уровням для каждого режима."""
    by_mode: Dict[str, List[Dict]] = {}
    for r in rows:
        by_mode.setdefault(r["mode"], []).append(r)

    lines = ["=" * 70, "SONYA BENCHMARK — ПРОГРЕСС ПО УРОВНЯМ", "=" * 70, ""]

    for mode, mode_rows in by_mode.items():
        lines.append(f"Режим: {mode.upper()}")
        lines.append(f"  Прогонов: {len(mode_rows)}")

        avg_cands = np.mean([r["num_candidates"] for r in mode_rows])
        auto_export = sum(1 for r in mode_rows if r["export_decision"] == "auto_export")
        manual_review = sum(1 for r in mode_rows if r["export_decision"] == "manual_review")
        reject = sum(1 for r in mode_rows if r["export_decision"] == "reject")
        avg_runtime = np.mean([r["runtime_total_sec"] for r in mode_rows])

        lines.append(f"  Среднее кандидатов: {avg_cands:.1f}")
        lines.append(f"  Среднее время: {avg_runtime:.1f}s")
        lines.append(f"  auto_export: {auto_export} | manual_review: {manual_review} | reject: {reject}")

        stub_count = sum(1 for r in mode_rows if r.get("stub", True))
        if stub_count == len(mode_rows):
            lines.append("  ⚠️  Все прогоны через заглушку — подключи реальный модуль")

        lines.append("")

    lines += [
        "─" * 70,
        "Уровни зрелости режима:",
        "  🔴 Уровень 1 — технически запустился (stub или 0 кандидатов)",
        "  🟡 Уровень 2 — находит кандидатов",
        "  🟡 Уровень 3 — ранжирует, не reject",
        "  🟢 Уровень 4 — после ручной проверки: нормальные границы",
        "  🔵 Уровень 5 — guarded production",
        "",
        "MVP-нормы:",
        "  Hook:     top3_contains_good ≥ 70%, boundary_ok ≥ 75%",
        "  Story:    top3_contains_good ≥ 60%, boundary_ok ≥ 75%",
        "  Trailer:  top3_contains_good ≥ 55%, boundary_ok ≥ 75%",
        "  Viral:    top3_contains_good ≥ 65%, boundary_ok ≥ 75%",
        "  Edu:      top3_contains_good ≥ 60%, boundary_ok ≥ 75%",
    ]

    path = output_dir / "progress_report.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"📊 Отчёт прогресса: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Точка входа
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SONYA GPU Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--videos", default="data/input",
                        help="Папка с входными видео (.mp4/.mov/.avi)")
    parser.add_argument("--modes", nargs="+",
                        default=["hook", "story", "trailer_preview"],
                        choices=list(MODE_RUNNERS.keys()),
                        help="Список режимов для тестирования")
    parser.add_argument("--model", default="yolov8n",
                        choices=["yolov8n", "yolov8s", "yolov8m", "yolov8x"],
                        help="Модель YOLOv8")
    parser.add_argument("--output", default="outputs/test_today",
                        help="Папка для результатов")
    parser.add_argument("--skip-export", action="store_true",
                        help="Не нарезать клипы (только JSON-артефакты)")
    parser.add_argument("--whisper-model", default="base",
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Размер Whisper: tiny/base = быстро, small/medium = качественнее")
    args = parser.parse_args()

    videos_dir = Path(args.videos)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_extensions = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    video_files = sorted([
        f for f in videos_dir.iterdir()
        if f.suffix.lower() in video_extensions
    ])

    if not video_files:
        logger.error(f"Нет видео в папке: {videos_dir}")
        sys.exit(1)

    gpu_available, gpu_name, vram_total = _gpu_info()
    logger.info(f"GPU: {gpu_name} | VRAM: {vram_total:.0f} MB | Доступна: {gpu_available}")
    logger.info(f"Видео: {len(video_files)} | Режимы: {args.modes} | Модель: {args.model}")
    logger.info(f"Выход: {output_dir}")

    all_rows: List[Dict] = []
    total_runs = len(video_files) * len(args.modes)

    # Whisper размер: base для быстрого теста, small/medium для качества
    whisper_model = getattr(args, "whisper_model", "base")

    with tqdm(total=total_runs, desc="Прогоны") as pbar:
        for video_path in video_files:
            pbar.set_description(f"{video_path.name} — подготовка")

            # ── Предвычисления на видео (один раз, общие для всех режимов) ──
            logger.info(f"── Видео: {video_path.name} ──")

            pre_base_analysis, pre_yolo_sec = run_yolo_analysis(video_path, args.model)
            pre_input_meta = extract_input_metadata(video_path)
            pre_duration = pre_input_meta["duration_sec"]

            logger.info(f"  YOLO готов за {pre_yolo_sec:.1f}s, длина видео {pre_duration:.1f}s")

            pre_asr_segments: List[Dict] = []
            needs_asr = any(m in args.modes for m in ("hook", "story", "educational", "trailer_preview"))
            if needs_asr:
                logger.info("  Запуск ASR (Whisper)...")
                pre_asr_segments, asr_elapsed = run_asr(video_path, whisper_model)
                logger.info(f"  ASR готов за {asr_elapsed:.1f}s, сегментов: {len(pre_asr_segments)}")

            # Кеш результатов режимов для этого видео (нужен trailer)
            mode_results_cache: Dict = {}

            for mode in args.modes:
                pbar.set_description(f"{video_path.name} / {mode}")
                try:
                    row = run_single(
                        video_path, mode, args.model, output_dir,
                        video_duration_sec=pre_duration,
                        base_analysis=pre_base_analysis,
                        yolo_sec_precomputed=pre_yolo_sec,
                        asr_segments=pre_asr_segments,
                        mode_results_cache=mode_results_cache,
                        skip_export=args.skip_export,
                    )
                    all_rows.append(row)
                except Exception as e:
                    logger.error(f"ОШИБКА {video_path.name}/{mode}: {e}")
                    logger.debug(traceback.format_exc())
                    all_rows.append({
                        "run_id": f"{video_path.stem}__{mode}__ERROR",
                        "video": video_path.name,
                        "mode": mode,
                        "model": args.model,
                        "error": str(e),
                        "stub": True,
                        "runtime_total_sec": 0,
                        "yolo_sec": 0,
                        "mode_logic_sec": 0,
                        "export_sec": 0,
                        "vram_peak_mb": 0,
                        "gpu_available": gpu_available,
                        "gpu_name": gpu_name,
                        "num_candidates": 0,
                        "top1_score": None,
                        "export_decision": "error",
                        "clip_exported": False,
                        "maturity_level": 0,
                        "output_dir": str(output_dir),
                    })
                pbar.update(1)

    # ── Итоговые файлы ────────────────────────────────────────────────────────
    csv_path = save_summary_csv(all_rows, output_dir)
    fb_path = save_failure_breakdown(all_rows, output_dir)
    rt_path = save_runtime_breakdown(all_rows, output_dir)
    mc_path = save_model_comparison(all_rows, output_dir)
    save_progress_report(all_rows, output_dir)

    logger.success(f"\n{'='*60}")
    logger.success(f"✅ Готово! Прогонов: {len(all_rows)}")
    logger.success(f"📁 Результаты: {output_dir}")
    logger.success(f"📋 summary.csv:              {csv_path}")
    logger.success(f"📋 failure_breakdown:        {fb_path}")
    logger.success(f"📋 runtime_breakdown.csv:    {rt_path}")
    logger.success(f"📋 model_comparison.csv:     {mc_path}")
    logger.success(f"📋 progress_report.txt:      {output_dir / 'progress_report.txt'}")
    logger.success(f"\nДальше: заполни human_review_template.json в каждой run-папке")
    logger.success(f"{'='*60}")


if __name__ == "__main__":
    main()
