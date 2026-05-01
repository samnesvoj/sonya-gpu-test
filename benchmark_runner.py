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
import datetime
import hashlib
import json
import os
import platform
import subprocess
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

# ── Shared audio cache ────────────────────────────────────────────────────────
_scripts_path = Path(__file__).parent / "scripts"
if str(_scripts_path) not in sys.path:
    sys.path.insert(0, str(_scripts_path))

try:
    from audio_cache import get_audio_cache_manifest as _get_audio_cache_manifest
    _HAS_AUDIO_CACHE = True
except ImportError:
    _get_audio_cache_manifest = None  # type: ignore
    _HAS_AUDIO_CACHE = False

# ─────────────────────────────────────────────────────────────────────────────
# Run manifest helpers
# ─────────────────────────────────────────────────────────────────────────────

def _git_commit() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _command_version(cmd: List[str]) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
        out = (r.stdout or r.stderr or "").strip()
        return out.splitlines()[0] if out else "unavailable"
    except Exception:
        return "unavailable"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            while True:
                chunk = f.read(8 * 1024 * 1024)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return "error"


def _get_torch_info() -> Dict[str, Any]:
    try:
        import torch
        return {
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except Exception:
        return {"torch_version": "unavailable", "cuda_available": False, "cuda_device_name": None}


def _now_iso() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"


def _build_initial_manifest(
    output_dir: Path,
    video_files: List[Path],
    modes: List[str],
    model: str,
    whisper_model: str,
    strict_real: bool,
    skip_export: bool,
) -> Dict[str, Any]:
    torch_info = _get_torch_info()
    gpu_available, gpu_name, _ = _gpu_info()
    video_entries = []
    for vp in video_files:
        try:
            size_mb = round(vp.stat().st_size / 1024 / 1024, 2)
        except Exception:
            size_mb = 0.0
        video_entries.append({
            "filename": vp.name,
            "path": str(vp),
            "sha256": None,   # filled lazily after YOLO
            "size_mb": size_mb,
            "duration_sec": None,
        })
    return {
        "run_id": output_dir.name,
        "git_commit": _git_commit(),
        "started_at": _now_iso(),
        "finished_at": None,
        "status": "running",
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch_info["torch_version"],
        "cuda_available": torch_info["cuda_available"],
        "gpu_name": gpu_name,
        "ffmpeg_version": _command_version(["ffmpeg", "-version"]),
        "ffprobe_version": _command_version(["ffprobe", "-version"]),
        "input_videos": video_entries,
        "modes": modes,
        "model": model,
        "whisper_model": whisper_model,
        "strict_real": strict_real,
        "skip_export": skip_export,
        "summary": {
            "total_runs": 0,
            "stub_count": 0,
            "error_count": 0,
            "exported_count": 0,
        },
    }


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
# Long-video safe mode: sparse sampling, ffprobe metadata, ffmpeg fallback
# ─────────────────────────────────────────────────────────────────────────────

def _ffprobe_metadata(video_path: Path) -> Dict[str, Any]:
    """
    Получает точные метаданные видео через ffprobe.
    Возвращает dict с duration_sec, fps, frame_count, width, height.
    OpenCV используется как fallback если ffprobe недоступен.
    """
    meta: Dict[str, Any] = {
        "duration_sec": 0.0, "fps": 25.0,
        "frame_count": 0, "width": 0, "height": 0,
        "source": "opencv_fallback",
    }
    try:
        import subprocess as _sp, json as _json
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,nb_frames:format=duration",
            "-of", "json", str(video_path),
        ]
        r = _sp.run(cmd, capture_output=True, text=True, timeout=15)
        if r.returncode == 0:
            data = _json.loads(r.stdout)
            fmt = data.get("format", {})
            streams = data.get("streams", [{}])
            st = streams[0] if streams else {}
            dur = float(fmt.get("duration") or st.get("duration") or 0)
            rfr = st.get("r_frame_rate", "25/1")
            try:
                num, den = rfr.split("/")
                fps = float(num) / max(float(den), 1e-6)
            except Exception:
                fps = 25.0
            nb = st.get("nb_frames")
            frame_count = int(nb) if nb and nb != "N/A" else int(dur * fps)
            meta.update({
                "duration_sec": round(dur, 3),
                "fps": round(fps, 3),
                "frame_count": frame_count,
                "width": int(st.get("width", 0)),
                "height": int(st.get("height", 0)),
                "source": "ffprobe",
            })
            return meta
    except Exception:
        pass

    # OpenCV fallback
    try:
        cap = cv2.VideoCapture(str(video_path))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()
        meta.update({
            "duration_sec": round(fc / fps if fps > 0 else 0.0, 3),
            "fps": round(fps, 3),
            "frame_count": fc,
            "width": w,
            "height": h,
            "source": "opencv_fallback",
        })
    except Exception:
        pass
    return meta


def _video_analysis_params(duration_sec: float) -> Dict[str, Any]:
    """
    Определяет параметры семплирования в зависимости от длины видео.
    Возвращает: video_analysis_mode, max_frames, sampling_interval_sec
    """
    if duration_sec <= 600:
        return {
            "video_analysis_mode": "standard",
            "max_frames": 300,
            "sampling_interval_sec": 2.0,
        }
    elif duration_sec <= 1200:
        max_frames = 300
        interval = max(5.0, duration_sec / max_frames)
        return {
            "video_analysis_mode": "sparse_long_video",
            "max_frames": max_frames,
            "sampling_interval_sec": round(interval, 2),
        }
    else:
        max_frames = 200
        interval = max(5.0, duration_sec / max_frames)
        return {
            "video_analysis_mode": "sparse_long_video",
            "max_frames": max_frames,
            "sampling_interval_sec": round(interval, 2),
        }


def _run_yolo_on_frame(
    yolo: Any,
    frame: np.ndarray,
    timestamp_sec: float,
    _real_visual_fn: Any,
    _prev_gray: Any,
) -> Tuple[Dict[str, Any], Any]:
    """Run YOLO + optional real_visual on one frame. Returns (det, new_prev_gray)."""
    det: Dict[str, Any] = {
        "timestamp_sec": round(timestamp_sec, 2),
        "person_count": 0,
        "objects": [],
        "confidence_max": 0.0,
    }
    yolo_dets_for_visual: List[Dict] = []
    try:
        results = yolo(frame, verbose=False)
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = yolo.names.get(cls_id, str(cls_id))
                det["objects"].append({"class": cls_name, "confidence": round(conf, 3)})
                if cls_name == "person":
                    det["person_count"] += 1
                det["confidence_max"] = max(det["confidence_max"], conf)
                if hasattr(box, "xyxy"):
                    xy = box.xyxy[0].cpu().numpy().tolist()
                    yolo_dets_for_visual.append({"bbox": xy})
    except Exception:
        pass

    new_prev_gray = _prev_gray
    if _real_visual_fn is not None:
        try:
            vm = _real_visual_fn(frame, _prev_gray, yolo_dets_for_visual)
            det["action_intensity"] = round(vm["action_intensity"], 4)
            det["emotional_peaks"] = round(vm["emotional_peaks"], 4)
            det["composition_score"] = round(vm["composition_score"], 4)
            new_prev_gray = vm["current_gray"]
        except Exception:
            new_prev_gray = None

    return det, new_prev_gray


def run_yolo_analysis(
    video_path: Path, model_name: str
) -> Tuple[Dict[str, Any], float]:
    """
    Long-video safe YOLO analysis.

    Sampling strategy:
      ≤ 600s  → standard mode, every 2s (≤300 frames)
      ≤1200s  → sparse mode, every max(5s, dur/300) (≤300 frames)
      >1200s  → sparse mode, every max(5s, dur/200) (≤200 frames)

    Metadata: ffprobe first, OpenCV fallback.
    Frame read: sampled timestamps via cap.set(POS_MSEC), not sequential.
    If OpenCV fails (< 10% expected frames read): ffmpeg jpg fallback.
    """
    t0 = time.perf_counter()

    try:
        from real_visual_metrics import compute_all_frame_metrics as _real_visual_fn
        _has_real_visual = True
    except ImportError:
        _real_visual_fn = None
        _has_real_visual = False

    # ── Step 1: video metadata ─────────────────────────────────────────────────
    vmeta = _ffprobe_metadata(video_path)
    duration_sec = vmeta["duration_sec"]
    source_fps = vmeta["fps"]
    source_frame_count = vmeta["frame_count"]

    # ── Step 2: sampling params ────────────────────────────────────────────────
    params = _video_analysis_params(duration_sec)
    mode_label = params["video_analysis_mode"]
    max_frames = params["max_frames"]
    interval_sec = params["sampling_interval_sec"]

    # Build list of sample timestamps (seconds)
    sample_timestamps: List[float] = []
    t = 0.0
    while t < duration_sec and len(sample_timestamps) < max_frames:
        sample_timestamps.append(round(t, 3))
        t += interval_sec
    target_count = len(sample_timestamps)

    logger.info(f"  Video duration: {duration_sec:.1f}s | mode={mode_label} | "
                f"interval={interval_sec}s | target={target_count} frames")

    # ── Step 3: load YOLO ──────────────────────────────────────────────────────
    try:
        from ultralytics import YOLO
        model_file = model_name if model_name.endswith(".pt") else f"{model_name}.pt"
        scripts_dir = Path(__file__).parent
        model_path = scripts_dir / model_file
        if not model_path.exists():
            model_path = Path(model_file)
        yolo = YOLO(str(model_path))
        # detect device
        try:
            import torch
            yolo_device = "0" if torch.cuda.is_available() else "cpu"
        except Exception:
            yolo_device = "cpu"
    except Exception as e:
        elapsed = time.perf_counter() - t0
        logger.warning(f"YOLO load failed: {e}")
        return {"model": model_name, "error": str(e), "detections": [],
                "video_analysis_mode": mode_label, "video_duration_sec": duration_sec}, elapsed

    # ── Step 4: OpenCV sparse seek ─────────────────────────────────────────────
    detections: List[Dict] = []
    read_failures = 0
    opencv_used = True
    ffmpeg_fallback_used = False
    _prev_gray = None
    log_interval = max(1, target_count // 4)

    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError("cv2.VideoCapture failed to open")

        for i, ts in enumerate(sample_timestamps):
            cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
            ret, frame = cap.read()
            if not ret or frame is None:
                read_failures += 1
                continue

            det, _prev_gray = _run_yolo_on_frame(yolo, frame, ts, _real_visual_fn, _prev_gray)
            detections.append(det)

            if (i + 1) % log_interval == 0 or (i + 1) == target_count:
                logger.info(f"  Sampled frames: {i + 1}/{target_count} "
                            f"(read_ok={len(detections)}, failures={read_failures})")

        cap.release()
    except Exception as cap_exc:
        logger.warning(f"  OpenCV read error: {cap_exc} — will try ffmpeg fallback")
        detections = []
        read_failures = target_count

    # ── Step 5: ffmpeg fallback (if OpenCV gave < 10% expected frames) ─────────
    if len(detections) < max(1, target_count * 0.10):
        logger.warning(
            f"  OpenCV read rate too low ({len(detections)}/{target_count}) "
            f"— switching to ffmpeg frame extraction fallback"
        )
        ffmpeg_fallback_used = True
        opencv_used = False
        detections = []
        _prev_gray = None
        read_failures = 0

        import tempfile
        import shutil
        import subprocess as _sp
        tmp_dir = Path(tempfile.mkdtemp(prefix="sonya_yolo_"))
        try:
            # extract sparse frames to jpg
            vf_filter = f"fps=1/{interval_sec:.1f},scale=1280:-2"
            ffmpeg_cmd = [
                "ffmpeg", "-y", "-v", "error",
                "-fflags", "+genpts", "-err_detect", "ignore_err",
                "-i", str(video_path),
                "-vf", vf_filter,
                "-frames:v", str(max_frames),
                "-q:v", "3",
                str(tmp_dir / "frame_%06d.jpg"),
            ]
            _sp.run(ffmpeg_cmd, timeout=300, check=False,
                    stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)

            jpg_files = sorted(tmp_dir.glob("frame_*.jpg"))
            logger.info(f"  ffmpeg extracted {len(jpg_files)} frames to {tmp_dir}")

            for i, jpg_path in enumerate(jpg_files):
                frame = cv2.imread(str(jpg_path))
                if frame is None:
                    read_failures += 1
                    continue
                ts = i * interval_sec
                det, _prev_gray = _run_yolo_on_frame(yolo, frame, ts, _real_visual_fn, _prev_gray)
                detections.append(det)

                if (i + 1) % log_interval == 0 or (i + 1) == len(jpg_files):
                    logger.info(f"  [ffmpeg] Processed frames: {i + 1}/{len(jpg_files)}")
        except Exception as ffmpeg_exc:
            logger.error(f"  ffmpeg fallback also failed: {ffmpeg_exc}")
        finally:
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass

    # ── Step 6: aggregate results ──────────────────────────────────────────────
    person_frames = [d for d in detections if d["person_count"] > 0]
    elapsed = time.perf_counter() - t0

    logger.info(
        f"  YOLO done: {len(detections)} frames in {elapsed:.1f}s | "
        f"person_ratio={round(len(person_frames)/max(len(detections),1),3)} | "
        f"opencv={opencv_used} ffmpeg_fallback={ffmpeg_fallback_used}"
    )

    return {
        # Legacy fields (unchanged for mode compatibility)
        "model": model_name,
        "total_frames_sampled": len(detections),
        "person_presence_ratio": round(len(person_frames) / max(len(detections), 1), 3),
        "avg_confidence": round(
            float(np.mean([d["confidence_max"] for d in detections])) if detections else 0.0, 3
        ),
        "has_real_visual_metrics": _has_real_visual,
        "detections": detections,
        # New diagnostic fields (v3.4)
        "video_analysis_mode": mode_label,
        "video_duration_sec": round(duration_sec, 2),
        "source_fps": round(source_fps, 3),
        "source_frame_count": source_frame_count,
        "target_sampled_frames": target_count,
        "sampled_frames_count": len(detections),
        "sampling_interval_sec": interval_sec,
        "opencv_used": opencv_used,
        "ffmpeg_fallback_used": ffmpeg_fallback_used,
        "opencv_frame_read_failures": read_failures,
        "yolo_device": yolo_device,
        "metadata_source": vmeta.get("source", "unknown"),
    }, elapsed


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
    Диагностическая заглушка — module absent or import failed.
    stub=True = diagnostic error, NOT a result.
    candidates=[] — NO fake candidates ever generated.
    export_decision="reject" — stub output is never exported.
    """
    candidates: List[Dict] = []
    ranking: List[Dict] = []
    top1: Dict = {}
    export_decision = "reject"  # stub output is always rejected

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


def _error_result(
    mode: str,
    adapter_error: str,
    error_traceback: str,
) -> Dict[str, Any]:
    """
    Runtime crash in a real mode module.
    stub=True, candidates=[], export_decision=reject.
    Full traceback stored for writing to error_traceback.txt in run_single.
    NEVER generates fake candidates.
    """
    result: Dict[str, Any] = {
        "mode": mode,
        "stub": True,
        "candidates": [],
        "ranking": [],
        "top1": {},
        "export_decision": "reject",
        "main_reject_reason": f"{mode}_real_module_error",
        "adapter_error": adapter_error,
        "error_traceback": error_traceback,
        "boundary_diagnostics": {},
        "mode_metrics": {},
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


def _classify_export_decision(
    mode: str,
    raw: Dict,
    top1: Dict,
    candidates: List[Dict],
    asr_segments_count: int = 0,
) -> Tuple[str, List[str]]:
    """Учитывает сигналы из самих режимов (weak_fallback, degraded_preview,
    topic_source). Возвращает (decision, reasons)."""
    score_based = _export_decision_from_score(top1, mode)
    reasons: List[str] = [f"score_based={score_based}"]

    if len(candidates) == 0:
        reasons.append("no_final_candidates")
        return "reject", reasons

    if mode == "hook":
        if raw.get("weak_hook_fallback_used"):
            reasons.append("weak_hook_fallback_used")
            return "manual_review", reasons
        return score_based, reasons

    if mode == "story":
        if raw.get("weak_story_fallback_used"):
            reasons.append("weak_story_fallback_used")
            return "manual_review", reasons
        return score_based, reasons

    if mode == "viral":
        return score_based, reasons

    if mode == "educational":
        topic_source = raw.get("topic_source")
        topics = raw.get("topic_segments") or []
        if (
            topic_source == "fallback_single_topic"
            and asr_segments_count > 10
            and score_based == "auto_export"
        ):
            reasons.append("single_topic_fallback_with_many_asr")
            return "manual_review", reasons
        if len(topics) <= 1 and asr_segments_count > 10 and score_based == "auto_export":
            reasons.append("only_one_topic_despite_many_asr")
            return "manual_review", reasons
        return score_based, reasons

    if mode == "trailer_preview":
        # Режим уже сам классифицирует и передаёт причины
        mode_decision = raw.get("export_decision")
        mode_reasons = raw.get("export_decision_reasons") or []
        if mode_decision in ("auto_export", "manual_review", "reject"):
            return mode_decision, list(mode_reasons) + reasons
        return score_based, reasons

    return score_based, reasons


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
        export_decision, _export_reasons = _classify_export_decision(
            "hook", raw, top1, candidates
        )

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
            "_raw_mode_output": raw,
        }
        return result, time.perf_counter() - t0

    except Exception as e:
        adapter_error = f"{type(e).__name__}: {e}"
        _tb = traceback.format_exc()
        logger.error(f"[hook] REAL MODULE CRASHED — stub=True, candidates=[]\n{_tb}")
        return _error_result("hook", adapter_error, _tb), time.perf_counter() - t0


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
        export_decision, _export_reasons = _classify_export_decision(
            "story", raw, top1, candidates
        )

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
            "_raw_mode_output": raw,
        }
        return result, time.perf_counter() - t0

    except Exception as e:
        adapter_error = f"{type(e).__name__}: {e}"
        _tb = traceback.format_exc()
        logger.error(f"[story] REAL MODULE CRASHED — stub=True, candidates=[]\n{_tb}")
        return _error_result("story", adapter_error, _tb), time.perf_counter() - t0


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
        viral_raw_dict = moments if isinstance(moments, dict) else {"viral_moments": moments}
        export_decision, _export_reasons = _classify_export_decision(
            "viral", viral_raw_dict, top1, candidates
        )

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
            "_raw_mode_output": moments if isinstance(moments, dict) else {"viral_moments": moments},
        }
        return result, time.perf_counter() - t0

    except Exception as e:
        adapter_error = f"{type(e).__name__}: {e}"
        _tb = traceback.format_exc()
        logger.error(f"[viral] REAL MODULE CRASHED — stub=True, candidates=[]\n{_tb}")
        return _error_result("viral", adapter_error, _tb), time.perf_counter() - t0


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
        export_decision, _export_reasons = _classify_export_decision(
            "educational", raw, top1, candidates,
            asr_segments_count=len(asr_segments or []),
        )

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
            "_raw_mode_output": raw,
        }
        return result, time.perf_counter() - t0

    except Exception as e:
        adapter_error = f"{type(e).__name__}: {e}"
        _tb = traceback.format_exc()
        logger.error(f"[educational] REAL MODULE CRASHED — stub=True, candidates=[]\n{_tb}")
        return _error_result("educational", adapter_error, _tb), time.perf_counter() - t0


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
        export_decision, _export_reasons = _classify_export_decision(
            "trailer_preview", raw, top1, candidates
        )

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
        _tb = traceback.format_exc()
        logger.error(f"[trailer_preview] REAL MODULE CRASHED — stub=True, candidates=[]\n{_tb}")
        return _error_result("trailer_preview", adapter_error, _tb), time.perf_counter() - t0


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
    # Debug-сигналы фильтрации
    "raw_proposals_count":    "Предложений до фильтрации (raw proposals)",
    "rejected_count":         "Отфильтровано кандидатов",
    "main_reject_reason":     "Главная причина отсева",
    "had_raw_signal":         "Режим что-то нашёл до финального фильтра",
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
# Debug artifact extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_debug_artifacts(
    mode: str,
    raw: Dict,
    final_candidates: List[Dict],
    asr_segments: List[Dict],
) -> Dict[str, Any]:
    """
    Извлекает debug-артефакты из сырого вывода режима.
    Цель: если final_candidates пусто — понять, что именно пошло не так.

    Возвращает словарь с ключами:
        raw_proposals       — сырые предложения до финального фильтра (если доступны)
        rejected_candidates — кандидаты, отфильтрованные до top-k
        filter_reasons      — причины отсева по каждому кандидату / агрегированные счётчики
        pipeline_trace      — этапы пайплайна и количество элементов на каждом
        raw_proposals_count — int или None
        rejected_count      — int или None
        main_reject_reason  — str или None
        had_raw_signal      — True если режим вообще что-то нашёл до финального фильтра
        # режим-специфичные ключи добавляются ниже
    """
    stats: Dict = raw.get("stats", {}) or {}
    error: Optional[str] = raw.get("error")

    raw_proposals: List[Dict] = []
    rejected_candidates: List[Dict] = []
    filter_reasons: Dict = {}
    raw_proposals_count: Optional[int] = None
    rejected_count: Optional[int] = None
    main_reject_reason: Optional[str] = error
    had_raw_signal: bool = len(final_candidates) > 0

    # ── Общий pipeline trace ─────────────────────────────────────────────────
    pipeline_trace: Dict[str, Any] = {
        "mode": mode,
        "asr_segments_count": len(asr_segments),
        "final_candidates_count": len(final_candidates),
        "early_exit_error": error,
        "stats_snapshot": stats,
    }

    # ── Hook-специфичная диагностика ─────────────────────────────────────────
    hook_proposals: List[Dict] = []
    hook_filter_reasons: Dict = {}
    visual_first_candidates: List[Dict] = []
    non_hook_risks: List[Dict] = []
    rejected_hooks: List[Dict] = []
    weak_hook_fallback_used: bool = False
    micro_hook_active: bool = False

    if mode == "hook":
        n_before_nms = stats.get("n_candidates_before_nms")
        raw_proposals_count = (
            raw.get("hook_proposals_count")
            or stats.get("hook_proposals_count")
            or n_before_nms
        )
        had_raw_signal = bool(raw_proposals_count and raw_proposals_count > 0) or had_raw_signal

        proposal_sources = stats.get("proposal_sources_detail") or stats.get("proposal_sources", {})

        # Приоритет: реальные rejected_hooks из режима (v1.1)
        rejected_hooks = raw.get("rejected_hooks") or stats.get("rejected_hooks") or []
        # raw_proposals — либо список из mode, либо агрегат
        if rejected_hooks:
            hook_proposals = [
                {"source": src, "count": cnt}
                for src, cnt in proposal_sources.items()
            ]
            raw_proposals = list(rejected_hooks)
        else:
            hook_proposals = [
                {"source": src, "count": cnt}
                for src, cnt in proposal_sources.items()
            ]
            raw_proposals = hook_proposals

        if raw_proposals_count is not None:
            rejected_count = max(0, raw_proposals_count - len(final_candidates))

        weak_hook_fallback_used = bool(
            raw.get("weak_hook_fallback_used") or stats.get("weak_hook_fallback_used")
        )
        micro_hook_active = bool(
            raw.get("micro_hook_active") or stats.get("micro_hook_active")
        )

        # filter_reasons: из режима v1.1 + backward-compat
        hook_filter_reasons = raw.get("hook_filter_reasons") or stats.get("hook_filter_reasons") or {}
        export_decisions = stats.get("export_decisions", {})
        hook_filter_reasons = {
            **hook_filter_reasons,
            "export_decisions_in_pool": export_decisions,
            "avg_false_positive_risk": stats.get("avg_false_positive_risk"),
            "loose_hook_mode": stats.get("loose_hook_mode"),
            "threshold_effective": stats.get("threshold_effective"),
            "min_hook_score_effective": stats.get("min_hook_score_effective"),
            "weak_hook_fallback_used": weak_hook_fallback_used,
            "micro_hook_active": micro_hook_active,
        }
        filter_reasons = hook_filter_reasons

        # visual_first_candidates: финальные кандидаты с visual proposal_source
        visual_sources = {"face_reaction", "audio_burst", "visual_onset",
                          "face_onset", "visual_burst", "motion_onset", "prosody_burst"}
        visual_first_candidates = [
            c for c in final_candidates
            if c.get("_raw", {}).get("proposal_source", "") in visual_sources
        ]
        # non_hook_risks: кандидаты с high false_positive_risk
        non_hook_risks = [
            {
                "id": c.get("id"),
                "start_sec": c.get("start_sec"),
                "end_sec": c.get("end_sec"),
                "false_positive_risk": c.get("_raw", {}).get("false_positive_risk"),
                "hook_type": c.get("_raw", {}).get("hook_type"),
            }
            for c in final_candidates
            if (c.get("_raw", {}).get("false_positive_risk") or 0) > 0.5
        ]

        if not had_raw_signal and error:
            main_reject_reason = error
        elif not had_raw_signal:
            main_reject_reason = "no_proposals_generated"
        elif len(final_candidates) == 0:
            main_reject_reason = "all_filtered_by_nms_or_score_threshold"
        elif weak_hook_fallback_used:
            main_reject_reason = "weak_hook_manual_review"

        pipeline_trace.update({
            "stage_proposals": raw_proposals_count,
            "stage_after_nms": len(final_candidates),
            "stage_rejected_hooks": len(rejected_hooks),
            "proposal_sources": proposal_sources,
            "weak_hook_fallback_used": weak_hook_fallback_used,
            "micro_hook_active": micro_hook_active,
        })

    # ── Story-специфичная диагностика ────────────────────────────────────────
    story_filter_reasons: Dict = {}
    rejected_arcs: List[Dict] = []
    beat_to_arc_trace: List[Dict] = []
    weak_story_pool: List[Dict] = []
    weak_story_fallback_used: bool = False

    if mode == "story":
        filter_counts: Dict = stats.get("filter_counts", {})
        story_filter_reasons = filter_counts

        total_filtered = sum(filter_counts.values()) if filter_counts else 0
        rejected_count = total_filtered
        had_raw_signal = total_filtered > 0 or len(final_candidates) > 0
        raw_proposals_count = total_filtered + len(final_candidates) if filter_counts else None

        # v1.1: новые сигналы из режима
        rejected_arcs_from_mode = raw.get("rejected_arcs") or stats.get("rejected_arcs") or []
        beat_to_arc_trace = raw.get("beat_to_arc_trace") or stats.get("beat_to_arc_trace") or []
        weak_story_pool = raw.get("weak_story_pool") or stats.get("weak_story_pool") or []
        weak_story_fallback_used = bool(
            raw.get("weak_story_fallback_used") or stats.get("weak_story_fallback_used")
        )

        if filter_counts:
            main_reject_reason = max(filter_counts, key=lambda k: filter_counts[k]) \
                if any(filter_counts.values()) else error
        elif error:
            main_reject_reason = error
        elif len(final_candidates) == 0:
            main_reject_reason = "no_arcs_built_or_no_moments_passed_threshold"
        elif weak_story_fallback_used:
            main_reject_reason = "weak_story_manual_review"

        arc_pattern_dist = stats.get("arc_pattern_distribution", {})
        narrative_dist = stats.get("narrative_score_distribution", {})
        story_filter_reasons = {
            **filter_counts,
            "_arc_pattern_distribution": arc_pattern_dist,
            "_narrative_score_distribution": narrative_dist,
            "_pipeline_mode": stats.get("pipeline_mode"),
            "_threshold": stats.get("threshold"),
            "_min_narrative_threshold": stats.get("min_narrative_threshold"),
            "weak_story_fallback_used": weak_story_fallback_used,
            "n_rejected_arcs": len(rejected_arcs_from_mode),
            "n_weak_story_pool": len(weak_story_pool),
        }
        filter_reasons = story_filter_reasons

        # Объединяем rejected_arcs из режима + финальные reject export_decisions
        rejected_arcs = list(rejected_arcs_from_mode)
        rejected_arcs.extend([
            {
                "id": c.get("id"),
                "start_sec": c.get("start_sec"),
                "end_sec": c.get("end_sec"),
                "score": c.get("score"),
                "story_type": c.get("_raw", {}).get("story_type"),
                "export_decision": c.get("_raw", {}).get("export_decision"),
                "export_reject_reasons": c.get("_raw", {}).get("export_reject_reasons", []),
                "active_story_failures": c.get("_raw", {}).get("active_story_failures", []),
            }
            for c in final_candidates
            if c.get("_raw", {}).get("export_decision") == "reject"
        ])

        pipeline_trace.update({
            "stage_events": stats.get("stage_events_count"),
            "stage_beats": stats.get("stage_beats_count"),
            "stage_arcs": stats.get("stage_arcs_count"),
            "stage_moments_all": raw_proposals_count,
            "stage_top_k": len(final_candidates),
            "stage_weak_story_pool": len(weak_story_pool),
            "stage_rejected_arcs": len(rejected_arcs_from_mode),
            "pipeline_mode": stats.get("pipeline_mode"),
            "filter_counts": filter_counts,
            "weak_story_fallback_used": weak_story_fallback_used,
        })

    # ── Trailer-специфичная диагностика ──────────────────────────────────────
    input_candidate_library: List[Dict] = []
    rejected_trailer_candidates: List[Dict] = []
    trailer_filter_reasons: Dict = {}
    theme_blocks: List[Dict] = []
    transition_scores: List[Dict] = []
    slot_assignment_before_nms: List[Dict] = []
    slot_assignment_after_nms: List[Dict] = []
    duplicate_removed: List[Dict] = []
    overlap_removed: List[Dict] = []
    duration_cap_trace: Dict = {}
    final_sequence_validation: Dict = {}
    is_degraded_preview: bool = False

    if mode == "trailer_preview":
        # v3.1: режим теперь сам отдаёт input_candidate_library
        input_candidate_library = raw.get("input_candidate_library") or []
        if not input_candidate_library:
            # Fallback: собираем из mode-специфичных полей
            for mode_key in ("hook_moments", "story_moments", "viral_moments", "educational_moments"):
                for m in raw.get(mode_key, []):
                    input_candidate_library.append({**m, "_source_mode": mode_key.replace("_moments", "")})

        raw_proposals_count = len(input_candidate_library)
        had_raw_signal = raw_proposals_count > 0

        # v3.1: новые debug artifacts
        slot_assignment_before_nms = raw.get("slot_assignment_before_nms") or []
        slot_assignment_after_nms = raw.get("slot_assignment_after_nms") or []
        duplicate_removed = raw.get("duplicate_removed") or []
        overlap_removed = raw.get("overlap_removed") or []
        duration_cap_trace = raw.get("duration_cap_trace") or {}
        final_sequence_validation = raw.get("final_sequence_validation") or {}
        is_degraded_preview = bool(raw.get("is_degraded_preview"))

        # rejected: всё из input_library, чего нет в final_candidates
        rejected_trailer_candidates = [
            c for c in input_candidate_library
            if not any(
                abs(c.get("start", c.get("start_sec", 0)) - f.get("start_sec", 0)) < 1.0
                for f in final_candidates
            )
        ]
        rejected_count = len(rejected_trailer_candidates)

        trailer_filter_reasons = {
            "n_input_candidates": raw_proposals_count,
            "n_rejected": rejected_count,
            "n_duplicates_removed": len(duplicate_removed),
            "n_overlaps_removed": len(overlap_removed),
            "avg_transition_score": stats.get("avg_transition_score"),
            "num_themes": stats.get("num_themes"),
            "spoiler_agg": stats.get("spoiler_agg"),
            "is_degraded_preview": is_degraded_preview,
            "effective_target_duration": duration_cap_trace.get("effective_target_duration"),
            "assembled_total_duration": final_sequence_validation.get("total_duration"),
            "video_duration": duration_cap_trace.get("video_duration"),
        }
        filter_reasons = trailer_filter_reasons

        if len(final_candidates) == 0:
            if raw_proposals_count == 0:
                main_reject_reason = "no_input_candidates_from_other_modes"
            else:
                main_reject_reason = "all_trailer_candidates_filtered"
        elif is_degraded_preview:
            main_reject_reason = "degraded_preview_hook_or_story_missing"
        elif final_sequence_validation.get("total_duration", 0) > (
            duration_cap_trace.get("effective_target_duration") or float("inf")
        ):
            main_reject_reason = "assembled_duration_exceeds_target"
        elif not main_reject_reason:
            main_reject_reason = None

        theme_blocks = raw.get("theme_blocks", [])
        transition_scores = [
            {"from": e.get("from"), "to": e.get("to"), "score": e.get("score")}
            for e in raw.get("transition_edges", [])
        ]

        pipeline_trace.update({
            "stage_input_library": raw_proposals_count,
            "stage_slots_before_nms": len(slot_assignment_before_nms),
            "stage_slots_after_nms": len(slot_assignment_after_nms),
            "stage_duplicates_removed": len(duplicate_removed),
            "stage_overlaps_removed": len(overlap_removed),
            "stage_final": len(final_candidates),
            "theme_blocks_count": len(theme_blocks),
            "is_degraded_preview": is_degraded_preview,
            "effective_target_duration": duration_cap_trace.get("effective_target_duration"),
            "assembled_duration": final_sequence_validation.get("total_duration"),
        })

    # ── Viral-специфичная диагностика ────────────────────────────────────────
    viral_feature_breakdown: List[Dict] = []
    if mode == "viral":
        moments_key = "viral_moments"
        all_raw_moments = raw.get(moments_key, []) if isinstance(raw, dict) else []
        if not raw_proposals_count:
            raw_proposals_count = len(all_raw_moments)
            raw_proposals = list(all_raw_moments)
            had_raw_signal = raw_proposals_count > 0
        if len(final_candidates) == 0:
            main_reject_reason = error or "no_moments_passed_threshold"

        viral_feature_breakdown = [
            {
                "id": c.get("id"),
                "start_sec": c.get("start_sec"),
                "end_sec": c.get("end_sec"),
                "virality_score": c.get("score"),
                "breakdown": c.get("_raw", {}).get("viral_feature_breakdown", {}),
                "semantic_score": c.get("_raw", {}).get("semantic_score"),
                "visual_score": c.get("_raw", {}).get("visual_score"),
                "audio_score": c.get("_raw", {}).get("audio_score"),
            }
            for c in final_candidates
        ]

        pipeline_trace.update({
            "stage_raw_moments": raw_proposals_count,
            "stage_top_k": len(final_candidates),
            "n_with_feature_breakdown": sum(
                1 for v in viral_feature_breakdown if v.get("breakdown")
            ),
        })

    # ── Educational-специфичная диагностика ──────────────────────────────────
    topic_segments_out: List[Dict] = []
    educational_windows: List[Dict] = []
    educational_scores: List[Dict] = []
    topic_source: Optional[str] = None
    if mode == "educational":
        moments_key = "educational_moments"
        all_raw_moments = raw.get(moments_key, []) if isinstance(raw, dict) else []
        if not raw_proposals_count:
            raw_proposals_count = len(all_raw_moments)
            raw_proposals = list(all_raw_moments)
            had_raw_signal = raw_proposals_count > 0
        if len(final_candidates) == 0:
            main_reject_reason = error or "no_moments_passed_threshold"

        topic_segments_out = raw.get("topic_segments") or []
        educational_windows = raw.get("educational_windows") or []
        educational_scores = raw.get("educational_scores") or []
        topic_source = raw.get("topic_source")

        pipeline_trace.update({
            "stage_raw_moments": raw_proposals_count,
            "stage_top_k": len(final_candidates),
            "stage_topics": len(topic_segments_out),
            "stage_windows": len(educational_windows),
            "topic_source": topic_source,
        })

    return {
        # Универсальные
        "raw_proposals": raw_proposals,
        "rejected_candidates": rejected_candidates,
        "filter_reasons": filter_reasons,
        "pipeline_trace": pipeline_trace,
        "raw_proposals_count": raw_proposals_count,
        "rejected_count": rejected_count,
        "main_reject_reason": main_reject_reason,
        "had_raw_signal": had_raw_signal,
        # Hook-специфичные
        "hook_proposals": hook_proposals,
        "hook_filter_reasons": hook_filter_reasons,
        "visual_first_candidates": visual_first_candidates,
        "non_hook_risks": non_hook_risks,
        "rejected_hooks": rejected_hooks,
        "weak_hook_fallback_used": weak_hook_fallback_used,
        "micro_hook_active": micro_hook_active,
        # Story-специфичные
        "story_filter_reasons": story_filter_reasons,
        "rejected_arcs": rejected_arcs,
        "beat_to_arc_trace": beat_to_arc_trace,
        "weak_story_pool": weak_story_pool,
        "weak_story_fallback_used": weak_story_fallback_used,
        # Viral-специфичные
        "viral_feature_breakdown": viral_feature_breakdown,
        # Educational-специфичные
        "topic_segments": topic_segments_out,
        "educational_windows": educational_windows,
        "educational_scores": educational_scores,
        "topic_source": topic_source,
        # Trailer-специфичные
        "input_candidate_library": input_candidate_library,
        "rejected_trailer_candidates": rejected_trailer_candidates,
        "trailer_filter_reasons": trailer_filter_reasons,
        "theme_blocks": theme_blocks,
        "transition_scores": transition_scores,
        "slot_assignment_before_nms": slot_assignment_before_nms,
        "slot_assignment_after_nms": slot_assignment_after_nms,
        "duplicate_removed": duplicate_removed,
        "overlap_removed": overlap_removed,
        "duration_cap_trace": duration_cap_trace,
        "final_sequence_validation": final_sequence_validation,
        "is_degraded_preview": is_degraded_preview,
    }


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
    # Сохраняем base_analysis в output run folder для диагностики
    _save_json(run_dir / "base_analysis.json", base_analysis)

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

    # ── Error traceback: persist immediately if real module crashed ───────────
    _error_tb = mode_result.get("error_traceback")
    if _error_tb:
        try:
            (run_dir / "error_traceback.txt").write_text(
                f"mode: {mode}\nadapter_error: {mode_result.get('adapter_error')}\n\n{_error_tb}",
                encoding="utf-8",
            )
        except Exception:
            pass

    _save_json(run_dir / "candidates.json", candidates)
    _save_json(run_dir / "ranking.json", ranking)
    _save_json(run_dir / "export_decision.json", {"decision": export_decision})
    _save_json(run_dir / "boundary_diagnostics.json", boundary_diag)

    # ── Debug artifacts ───────────────────────────────────────────────────────
    raw_output = mode_result.get("_raw_mode_output") or _rebuild_raw(mode, mode_result)
    dbg = _extract_debug_artifacts(mode, raw_output, candidates, asr_segments or [])

    _save_json(run_dir / "raw_proposals.json", dbg["raw_proposals"])
    _save_json(run_dir / "rejected_candidates.json", dbg["rejected_candidates"])
    _save_json(run_dir / "filter_reasons.json", dbg["filter_reasons"])
    # Inject error info + base_analysis diagnostics into pipeline_trace
    _pipeline_trace = dict(dbg["pipeline_trace"])
    if _error_tb:
        _pipeline_trace["stub_mode"] = True
        _pipeline_trace["adapter_error"] = mode_result.get("adapter_error")
        _pipeline_trace["error_traceback_saved"] = "error_traceback.txt"
        _pipeline_trace["main_reject_reason"] = mode_result.get("main_reject_reason")
    # Embed base_analysis diagnostics so every mode's pipeline_trace is self-contained
    _pipeline_trace["base_analysis"] = {
        "video_analysis_mode": base_analysis.get("video_analysis_mode", "unknown"),
        "video_duration_sec": base_analysis.get("video_duration_sec", 0.0),
        "target_sampled_frames": base_analysis.get("target_sampled_frames", 0),
        "sampled_frames_count": base_analysis.get("sampled_frames_count",
                                                    base_analysis.get("total_frames_sampled", 0)),
        "sampling_interval_sec": base_analysis.get("sampling_interval_sec", 2.0),
        "opencv_used": base_analysis.get("opencv_used", True),
        "ffmpeg_fallback_used": base_analysis.get("ffmpeg_fallback_used", False),
        "opencv_frame_read_failures": base_analysis.get("opencv_frame_read_failures", 0),
        "yolo_device": base_analysis.get("yolo_device", "cpu"),
    }
    # Audio cache diagnostics for viral and educational modes
    if mode in ("viral", "educational") and _HAS_AUDIO_CACHE and _get_audio_cache_manifest:
        try:
            _manifest = _get_audio_cache_manifest()
            _pipeline_trace["audio_cache"] = {
                "enabled": bool(_manifest.get("enabled", False)),
                "items_count": len(_manifest.get("items", [])),
                "source": "cached_wav",
            }
        except Exception:
            pass
    _save_json(run_dir / "pipeline_trace.json", _pipeline_trace)

    # Режим-специфичные артефакты
    if mode == "hook":
        _save_json(run_dir / "hook_proposals.json", dbg["hook_proposals"])
        _save_json(run_dir / "hook_filter_reasons.json", dbg["hook_filter_reasons"])
        _save_json(run_dir / "visual_first_candidates.json", dbg["visual_first_candidates"])
        _save_json(run_dir / "non_hook_risks.json", dbg["non_hook_risks"])
        _save_json(run_dir / "rejected_hooks.json", dbg["rejected_hooks"])

    if mode == "story":
        _save_json(run_dir / "story_events.json", mode_result.get("story_events", []))
        _save_json(run_dir / "beats.json", mode_result.get("beats", []))
        _save_json(run_dir / "arcs.json", mode_result.get("arcs", []))
        _save_json(run_dir / "role_probs.json", mode_result.get("role_probs", {}))
        _save_json(run_dir / "rejected_arcs.json", dbg["rejected_arcs"])
        _save_json(run_dir / "story_filter_reasons.json", dbg["story_filter_reasons"])
        _save_json(run_dir / "beat_to_arc_trace.json", dbg["beat_to_arc_trace"])
        _save_json(run_dir / "weak_story_pool.json", dbg["weak_story_pool"])

    if mode == "viral":
        _save_json(run_dir / "viral_feature_breakdown.json", dbg["viral_feature_breakdown"])

    if mode == "educational":
        _save_json(run_dir / "topic_segments.json", dbg["topic_segments"])
        _save_json(run_dir / "educational_windows.json", dbg["educational_windows"])
        _save_json(run_dir / "educational_scores.json", dbg["educational_scores"])
        _save_json(run_dir / "topic_source.json", {"topic_source": dbg["topic_source"]})

    if mode == "trailer_preview":
        _save_json(run_dir / "themes.json", mode_result.get("themes", []))
        _save_json(run_dir / "slots.json", mode_result.get("slots", []))
        _save_json(run_dir / "transition_graph.json", mode_result.get("transition_graph", {}))
        _save_json(run_dir / "assembly_plan.json", mode_result.get("assembly_plan", {}))
        _save_json(run_dir / "ui_payload.json", mode_result.get("ui_payload", {}))
        _save_json(run_dir / "input_candidate_library.json", dbg["input_candidate_library"])
        _save_json(run_dir / "rejected_trailer_candidates.json", dbg["rejected_trailer_candidates"])
        _save_json(run_dir / "trailer_filter_reasons.json", dbg["trailer_filter_reasons"])
        _save_json(run_dir / "theme_blocks.json", dbg["theme_blocks"])
        _save_json(run_dir / "transition_scores.json", dbg["transition_scores"])
        # v3.1: новые артефакты
        _save_json(run_dir / "slot_assignment_before_nms.json", dbg["slot_assignment_before_nms"])
        _save_json(run_dir / "slot_assignment_after_nms.json", dbg["slot_assignment_after_nms"])
        _save_json(run_dir / "duplicate_removed.json", dbg["duplicate_removed"])
        _save_json(run_dir / "overlap_removed.json", dbg["overlap_removed"])
        _save_json(run_dir / "duration_cap_trace.json", dbg["duration_cap_trace"])
        _save_json(run_dir / "final_sequence_validation.json", dbg["final_sequence_validation"])
        _save_json(
            run_dir / "degraded_preview_flag.json",
            {"is_degraded_preview": dbg["is_degraded_preview"]},
        )
        # v3.3: hard repair artifacts
        _raw_out = mode_result.get("_raw_mode_output") or {}
        if _raw_out.get("final_repair_trace"):
            _save_json(run_dir / "final_repair_trace.json", _raw_out["final_repair_trace"])
        if _raw_out.get("overlap_repair_trace") is not None:
            _save_json(run_dir / "overlap_repair_trace.json", _raw_out["overlap_repair_trace"])
        if _raw_out.get("duration_trimmed") is not None:
            _save_json(run_dir / "duration_trimmed.json", _raw_out["duration_trimmed"])

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

    raw_proposals_count = dbg["raw_proposals_count"]
    rejected_count = dbg["rejected_count"]
    main_reject_reason = dbg["main_reject_reason"]
    had_raw_signal = dbg["had_raw_signal"]

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
        # Debug-сигналы фильтрации
        "raw_proposals_count": raw_proposals_count,
        "rejected_count": rejected_count,
        "main_reject_reason": main_reject_reason,
        "had_raw_signal": had_raw_signal,
        # Режим-специфичные метрики
        **mode_metrics,
        # Русские лейблы (для удобства чтения)
        "_labels_ru": {k: METRIC_LABELS_RU.get(k, k) for k in [
            "runtime_total_sec", "yolo_sec", "mode_logic_sec", "export_sec",
            "vram_peak_mb", "gpu_available", "gpu_name", "num_candidates",
            "top1_score", "export_decision",
            "raw_proposals_count", "rejected_count", "main_reject_reason", "had_raw_signal",
            *mode_metrics.keys()
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
        "raw_proposals_count": raw_proposals_count,
        "rejected_count": rejected_count,
        "main_reject_reason": main_reject_reason,
        "had_raw_signal": had_raw_signal,
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
        "vram_peak_mb", "gpu_name", "num_candidates", "export_decision",
        "raw_proposals_count", "rejected_count", "main_reject_reason", "had_raw_signal",
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

        # Debug-сигналы фильтрации
        zero_cands = [r for r in mode_rows if r["num_candidates"] == 0]
        had_signal = sum(1 for r in zero_cands if r.get("had_raw_signal"))
        no_signal = sum(1 for r in zero_cands if not r.get("had_raw_signal"))
        avg_raw = np.mean([r["raw_proposals_count"] for r in mode_rows
                           if r.get("raw_proposals_count") is not None]) \
            if any(r.get("raw_proposals_count") is not None for r in mode_rows) else None

        reject_reasons: Dict[str, int] = {}
        for r in mode_rows:
            reason = r.get("main_reject_reason")
            if reason:
                reject_reasons[reason] = reject_reasons.get(reason, 0) + 1

        lines.append(f"  Среднее кандидатов: {avg_cands:.1f}")
        lines.append(f"  Среднее time: {avg_runtime:.1f}s")
        lines.append(f"  auto_export: {auto_export} | manual_review: {manual_review} | reject: {reject}")
        if avg_raw is not None:
            lines.append(f"  Среднее raw proposals: {avg_raw:.1f}")
        if zero_cands:
            lines.append(
                f"  ⚠️  Прогонов с 0 кандидатов: {len(zero_cands)} "
                f"(был сигнал: {had_signal}, пусто совсем: {no_signal})"
            )
        if reject_reasons:
            top_reason = max(reject_reasons, key=reject_reasons.__getitem__)
            lines.append(f"  Главная причина отсева: {top_reason} ({reject_reasons[top_reason]}x)")

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
    parser.add_argument("--strict-real", action="store_true",
                        help="Fail with exit code 1 if any mode ran as stub (real module crashed)")
    parser.add_argument("--whisper-model", default="base",
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Размер Whisper: tiny/base = быстро, small/medium = качественнее")
    parser.add_argument("--resume", action="store_true",
                        help="Resume: reuse cached shared/base_analysis.json and shared/asr_segments.json "
                             "from the same output dir (skip YOLO/ASR if cached)")
    args = parser.parse_args()

    videos_dir = Path(args.videos)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── shared/ artifacts directory ──────────────────────────────────────────
    shared_dir = output_dir / "shared"
    shared_dir.mkdir(parents=True, exist_ok=True)

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
    if getattr(args, "resume", False):
        logger.info("  --resume: will reuse cached shared/ artifacts if available")

    all_rows: List[Dict] = []
    total_runs = len(video_files) * len(args.modes)
    whisper_model = getattr(args, "whisper_model", "base")

    # ── run_manifest.json: initial write ─────────────────────────────────────
    manifest = _build_initial_manifest(
        output_dir=output_dir,
        video_files=video_files,
        modes=args.modes,
        model=args.model,
        whisper_model=whisper_model,
        strict_real=getattr(args, "strict_real", False),
        skip_export=getattr(args, "skip_export", False),
    )
    _save_json(output_dir / "run_manifest.json", manifest)

    try:
        with tqdm(total=total_runs, desc="Прогоны") as pbar:
            for vid_idx, video_path in enumerate(video_files):
                pbar.set_description(f"{video_path.name} — подготовка")
                logger.info(f"── Видео: {video_path.name} ──")

                pre_input_meta = extract_input_metadata(video_path)
                pre_duration = pre_input_meta["duration_sec"]

                # ── Update manifest video entry with duration ─────────────
                if vid_idx < len(manifest["input_videos"]):
                    manifest["input_videos"][vid_idx]["duration_sec"] = pre_duration

                # ── YOLO: resume or compute ───────────────────────────────
                _resume = getattr(args, "resume", False)
                pre_base_analysis = None
                pre_yolo_sec = 0.0
                _shared_ba = shared_dir / "base_analysis.json"

                if _resume and _shared_ba.exists():
                    try:
                        with open(_shared_ba, encoding="utf-8") as _f:
                            pre_base_analysis = json.load(_f)
                        logger.info("Resume: using cached shared/base_analysis.json")
                    except Exception as _e:
                        logger.warning(f"Resume base_analysis invalid ({_e}), recalculating")
                        pre_base_analysis = None

                if pre_base_analysis is None:
                    pre_base_analysis, pre_yolo_sec = run_yolo_analysis(video_path, args.model)
                    logger.info(f"  YOLO готов за {pre_yolo_sec:.1f}s, длина видео {pre_duration:.1f}s")

                # ── Save shared base_analysis.json ────────────────────────
                _save_json(shared_dir / "base_analysis.json", pre_base_analysis)

                # ── Save shared input_metadata.json ──────────────────────
                _shared_input_meta = {
                    "filename": video_path.name,
                    "path": str(video_path),
                    "duration_sec": pre_duration,
                    "fps": pre_input_meta.get("fps", pre_base_analysis.get("source_fps")),
                    "frame_count": pre_input_meta.get("frame_count",
                                                       pre_base_analysis.get("source_frame_count")),
                    "width": pre_input_meta.get("width", pre_base_analysis.get("source_width")),
                    "height": pre_input_meta.get("height", pre_base_analysis.get("source_height")),
                    "metadata_source": pre_base_analysis.get("metadata_source", "unknown"),
                }
                _save_json(shared_dir / "input_metadata.json", _shared_input_meta)

                # ── ASR: resume or compute ────────────────────────────────
                pre_asr_segments: List[Dict] = []
                _shared_asr = shared_dir / "asr_segments.json"
                needs_asr = any(m in args.modes for m in ("hook", "story", "educational", "trailer_preview"))

                if _resume and _shared_asr.exists() and needs_asr:
                    try:
                        with open(_shared_asr, encoding="utf-8") as _f:
                            _loaded_asr = json.load(_f)
                        if isinstance(_loaded_asr, list):
                            pre_asr_segments = _loaded_asr
                            logger.info(f"Resume: using cached shared/asr_segments.json "
                                        f"({len(pre_asr_segments)} segments)")
                        else:
                            raise ValueError("not a list")
                    except Exception as _e:
                        logger.warning(f"Resume asr_segments invalid ({_e}), recalculating")
                        pre_asr_segments = []

                if not pre_asr_segments and needs_asr:
                    logger.info("  Запуск ASR (Whisper)...")
                    pre_asr_segments, asr_elapsed = run_asr(video_path, whisper_model)
                    logger.info(f"  ASR готов за {asr_elapsed:.1f}s, сегментов: {len(pre_asr_segments)}")

                # ── Save shared asr_segments.json ─────────────────────────
                _save_json(shared_dir / "asr_segments.json", pre_asr_segments)

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

                # ── Save shared audio_cache_manifest.json ─────────────────
                try:
                    _audio_manifest = (
                        _get_audio_cache_manifest()
                        if (_HAS_AUDIO_CACHE and _get_audio_cache_manifest)
                        else {"enabled": False, "items": []}
                    )
                    _save_json(shared_dir / "audio_cache_manifest.json", _audio_manifest)
                except Exception:
                    _save_json(shared_dir / "audio_cache_manifest.json", {"enabled": False, "items": []})

        # ── run_manifest: update summary on success ───────────────────────────
        stub_count = sum(1 for r in all_rows if r.get("stub"))
        error_count = sum(1 for r in all_rows if r.get("export_decision") == "error")
        exported_count = sum(1 for r in all_rows if r.get("clip_exported"))
        manifest["finished_at"] = _now_iso()
        manifest["status"] = "success"
        manifest["summary"] = {
            "total_runs": len(all_rows),
            "stub_count": stub_count,
            "error_count": error_count,
            "exported_count": exported_count,
        }
        _save_json(output_dir / "run_manifest.json", manifest)

    except Exception as _main_exc:
        # ── run_manifest: mark failed ──────────────────────────────────────
        manifest["finished_at"] = _now_iso()
        manifest["status"] = "failed"
        manifest["error"] = str(_main_exc)
        try:
            _save_json(output_dir / "run_manifest.json", manifest)
        except Exception:
            pass
        raise

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
    logger.success(f"📋 run_manifest.json:        {output_dir / 'run_manifest.json'}")
    logger.success(f"📋 shared/:                  {shared_dir}")
    logger.success(f"\nДальше: заполни human_review_template.json в каждой run-папке")
    logger.success(f"{'='*60}")

    # ── --strict-real: fail if any mode ran as stub ───────────────────────────
    if getattr(args, "strict_real", False):
        stub_rows = [r for r in all_rows if r.get("stub")]
        if stub_rows:
            stub_summary = ", ".join(
                f"{r['video']}/{r['mode']}" for r in stub_rows
            )
            msg = (
                f"\n{'='*60}\n"
                f"REAL TEST FAILED: {len(stub_rows)} stub(s) detected\n"
                f"  {stub_summary}\n"
                f"Check error_traceback.txt in each run folder.\n"
                f"{'='*60}"
            )
            logger.error(msg)
            pr_path = output_dir / "progress_report.txt"
            try:
                with open(pr_path, "a", encoding="utf-8") as _f:
                    _f.write(msg + "\n")
            except Exception:
                pass
            sys.exit(1)


if __name__ == "__main__":
    main()
