"""
Архитектура 5 режимов SONYA: формулы скоров и заглушки алгоритмов.

Режимы: viral, educational, stories, hooks, trailer.
Использует YOLOv8x (Grok4Teacher); ASR / Audio / LLM — заглушки до интеграции.

Version: 3.0 (Production)
"""

import sys
import json
import hashlib as _hashlib
import tempfile as _tempfile
import subprocess as _subprocess
import shutil as _shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

# Audio/ML imports (опциональны, с fallback)
try:
    import librosa
    import numpy as np
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False
    np = None  # type: ignore
    print("Warning: librosa/numpy not installed. Audio features will use fallback values.")

# ─── Module-level audio caches (keyed by video_path string) ─────────────────
# _AUDIO_WAV_PATH_CACHE: video_path → Path to extracted .wav
# _AUDIO_LOADED_CACHE:  video_path → (y_full: np.ndarray, sr: int)
_AUDIO_WAV_PATH_CACHE: Dict[str, Path] = {}
_AUDIO_LOADED_CACHE: Dict[str, Any] = {}


def get_cached_audio_wav(
    video_path: str,
    cache_dir: Optional[Path] = None,
) -> Optional[Path]:
    """
    Extract mono 16kHz WAV from video via ffmpeg, with file-system caching.
    Returns Path to WAV, or None on failure.
    Avoids PySoundFile/audioread issues with mp4 by producing a plain WAV.
    """
    video_str = str(video_path)

    # Check in-process cache first
    if video_str in _AUDIO_WAV_PATH_CACHE:
        cached = _AUDIO_WAV_PATH_CACHE[video_str]
        if Path(cached).exists():
            print(f"  [audio_cache] reused (in-process): {Path(cached).name}")
            return Path(cached)

    # Determine cache directory
    if cache_dir is None:
        cache_dir = Path(_tempfile.gettempdir()) / "sonya_audio_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Stable filename based on video path hash
    path_hash = _hashlib.md5(video_str.encode("utf-8")).hexdigest()[:14]
    wav_path = cache_dir / f"audio_{path_hash}.wav"

    if wav_path.exists() and wav_path.stat().st_size > 0:
        print(f"  [audio_cache] reused WAV on disk: {wav_path.name}")
        _AUDIO_WAV_PATH_CACHE[video_str] = wav_path
        return wav_path

    # Extract via ffmpeg
    try:
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-i", video_str,
            "-vn", "-ac", "1", "-ar", "16000",
            "-acodec", "pcm_s16le",
            str(wav_path),
        ]
        result = _subprocess.run(cmd, timeout=180, capture_output=True)
        if result.returncode == 0 and wav_path.exists() and wav_path.stat().st_size > 0:
            size_kb = wav_path.stat().st_size // 1024
            print(f"  [audio_cache] audio_cache_created: {wav_path.name} ({size_kb} KB)")
            _AUDIO_WAV_PATH_CACHE[video_str] = wav_path
            return wav_path
        else:
            err = result.stderr.decode(errors="replace")[:300]
            print(f"  [audio_cache] ffmpeg failed rc={result.returncode}: {err}")
    except Exception as exc:
        print(f"  [audio_cache] exception extracting WAV: {exc}")

    return None


def load_full_cached_audio(
    video_path: str,
    sr: int = 16000,
) -> Tuple[Any, int]:
    """
    Load full audio for video_path as (y_full, sr) — exactly once per process.
    Prefers cached WAV to avoid PySoundFile/audioread on mp4.
    Returns (np.ndarray, sr) or (None, sr) if unavailable.
    """
    if not HAS_AUDIO:
        return None, sr

    video_str = str(video_path)
    if video_str in _AUDIO_LOADED_CACHE:
        return _AUDIO_LOADED_CACHE[video_str]

    # Prefer WAV cache; fall back to direct path
    wav_path = get_cached_audio_wav(video_str)
    load_path = str(wav_path) if (wav_path and Path(wav_path).exists()) else video_str
    audio_source = "cached_wav" if (wav_path and Path(wav_path).exists()) else "direct_mp4"

    try:
        y_full, loaded_sr = librosa.load(load_path, sr=sr, mono=True)
        dur_s = len(y_full) / max(loaded_sr, 1)
        print(f"  [audio_cache] audio_loaded_once: {len(y_full)} samples "
              f"({dur_s:.1f}s) | source={audio_source}")
        result = (y_full, loaded_sr)
    except Exception as exc:
        print(f"  [audio_cache] load failed ({audio_source}): {exc}")
        result = (None, sr)

    _AUDIO_LOADED_CACHE[video_str] = result
    return result

# LLM imports (опциональны)
HAS_LLM = False
try:
    # Добавляем scripts в путь для импорта llm_segment_analysis
    _scripts_dir = Path(__file__).resolve().parent
    if str(_scripts_dir) not in sys.path:
        sys.path.insert(0, str(_scripts_dir))
    
    from llm_segment_analysis import _call_llm, _parse_json_from_response
    HAS_LLM = True
except ImportError:
    _call_llm = None
    _parse_json_from_response = None
    print("Warning: llm_segment_analysis not available. Semantic features will use fallback values.")

# Grok4Teacher imports (для time-series visual features)
try:
    from grok4_teacher import get_visual_features_for_window
    HAS_TIMESERIES_VISUAL = True
except ImportError:
    get_visual_features_for_window = None
    HAS_TIMESERIES_VISUAL = False
    print("Warning: get_visual_features_for_window not available. Using global visual features.")

# Загрузка конфига (PyYAML опционален)
_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "modes_architecture.yaml"
_MODES_CONFIG: Optional[Dict] = None


def get_modes_config() -> Dict:
    global _MODES_CONFIG
    if _MODES_CONFIG is None and _CONFIG_PATH.exists():
        try:
            import yaml
            with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
                _MODES_CONFIG = yaml.safe_load(f) or {}
        except Exception:
            _MODES_CONFIG = {}
    return _MODES_CONFIG or {"modes": {}, "feature_extraction": {}}


# --- Формулы скоров (по MODES_ARCHITECTURE.md) ---


def virality_score(
    hook_quality: float = 0.5,
    emotion_peak: float = 0.5,
    action_intensity: float = 0.5,
    visual_salience: float = 0.5,
    audio_energy: float = 0.5,
    shareability_score: float = 0.5,
    trending_format_match: float = 0.0,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """Скор виральности для сегмента (Режим 1)."""
    cfg = get_modes_config()
    w = weights or cfg.get("modes", {}).get("viral", {}).get("virality_weights", {})
    return (
        w.get("hook_quality", 0.25) * hook_quality
        + w.get("emotion_peak", 0.20) * emotion_peak
        + w.get("action_intensity", 0.15) * action_intensity
        + w.get("visual_salience", 0.15) * visual_salience
        + w.get("audio_energy", 0.10) * audio_energy
        + w.get("shareability_score", 0.10) * shareability_score
        + w.get("trending_format_match", 0.05) * trending_format_match
    )


def educational_score(
    insight_quality: float = 0.5,
    explanation_clarity: float = 0.5,
    practical_value: float = 0.5,
    clarity_score: float = 0.5,
    structure_score: float = 0.5,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """Скор образовательной ценности (Режим 2)."""
    cfg = get_modes_config()
    w = weights or cfg.get("modes", {}).get("educational", {}).get("edu_weights", {})
    return (
        w.get("insight_quality", 0.30) * insight_quality
        + w.get("explanation_clarity", 0.25) * explanation_clarity
        + w.get("practical_value", 0.20) * practical_value
        + w.get("clarity_score", 0.15) * clarity_score
        + w.get("structure_score", 0.10) * structure_score
    )


def story_score(
    setup_quality: float = 0.5,
    conflict_intensity: float = 0.5,
    resolution_quality: float = 0.5,
    has_twist_bonus: float = 0.0,
    emotional_arc_score: float = 0.5,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """Скор полноты story arc (Режим 3)."""
    cfg = get_modes_config()
    w = weights or cfg.get("modes", {}).get("stories", {}).get("story_weights", {})
    return (
        w.get("setup_quality", 0.25) * setup_quality
        + w.get("conflict_intensity", 0.25) * conflict_intensity
        + w.get("resolution_quality", 0.20) * resolution_quality
        + w.get("has_twist_bonus", 0.15) * has_twist_bonus
        + w.get("emotional_arc_score", 0.15) * emotional_arc_score
    )


def hook_score(
    intrigue_score: float = 0.5,
    visual_salience: float = 0.5,
    emotional_intensity: float = 0.5,
    audio_energy: float = 0.5,
    action_intensity: float = 0.5,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """Скор хука (первые 1–5 сек) (Режим 4)."""
    cfg = get_modes_config()
    w = weights or cfg.get("modes", {}).get("hooks", {}).get("hook_weights", {})
    return (
        w.get("intrigue_score", 0.30) * intrigue_score
        + w.get("visual_salience", 0.25) * visual_salience
        + w.get("emotional_intensity", 0.20) * emotional_intensity
        + w.get("audio_energy", 0.15) * audio_energy
        + w.get("action_intensity", 0.10) * action_intensity
    )


def compute_audio_score(features: Dict[str, float]) -> float:
    """
    Audio sub-score для VIRAL (late fusion).
    
    Фичи:
        - audio_energy (0.4): громкость
        - pitch_variance (0.3): эмоциональность голоса
        - silence_ratio (0.2): тишина (инвертируем)
        - speech_rate (0.1): темп речи
    """
    return (
        0.4 * features.get("audio_energy", 0.5)
        + 0.3 * features.get("pitch_variance", 0.5)
        + 0.2 * (1.0 - features.get("silence_ratio", 0.5))  # Инвертируем: меньше тишины = лучше
        + 0.1 * features.get("speech_rate", 0.5)
    )


def compute_visual_score(features: Dict[str, float]) -> float:
    """
    Visual sub-score для VIRAL (late fusion).
    
    Фичи:
        - action_intensity (0.3): движение
        - visual_salience (0.25): заметность
        - composition_score (0.25): композиция
        - emotional_peaks (0.2): эмоции на лицах
    """
    return (
        0.3 * features.get("action_intensity", 0.5)
        + 0.25 * features.get("visual_salience", 0.5)
        + 0.25 * features.get("composition_score", 0.5)
        + 0.2 * features.get("emotional_peaks", 0.5)
    )


def compute_semantic_score(features: Dict[str, float]) -> float:
    """
    Semantic sub-score для VIRAL (late fusion).
    
    Фичи:
        - hook_strength (0.4): зацепка
        - tension_level (0.3): конфликт, драма
        - payoff_presence (0.3): развязка, punchline
    """
    return (
        0.4 * features.get("hook_strength", 0.5)
        + 0.3 * features.get("tension_level", 0.5)
        + 0.3 * features.get("payoff_presence", 0.5)
    )


def virality_score_v3(features: Dict[str, float], weights: Optional[Dict[str, float]] = None) -> float:
    """
    VIRAL v3.0 PRODUCTION: Late fusion трёх модальностей.
    
    Architecture:
        audio_score (30%) ← энергия, pitch, тишина, темп речи
        visual_score (30%) ← движение, заметность, композиция, эмоции
        semantic_score (40%) ← hook, tension, payoff
        
    Args:
        features: 11 фичей
        weights: опционально переопределить веса (audio, visual, semantic)
    
    Returns:
        Финальный virality_score (0-1)
    """
    # Вычисляем суб-скоры
    audio = compute_audio_score(features)
    visual = compute_visual_score(features)
    semantic = compute_semantic_score(features)
    
    # Late fusion с настраиваемыми весами
    w = weights or {"audio": 0.30, "visual": 0.30, "semantic": 0.40}
    
    final_score = (
        w.get("audio", 0.30) * audio
        + w.get("visual", 0.30) * visual
        + w.get("semantic", 0.40) * semantic
    )
    
    return final_score


def virality_score_v2(features: Dict[str, float]) -> float:
    """
    Финальная формула VIRAL v2.0 по 11 фичам (БЕЗ заглушек).
    DEPRECATED: используйте virality_score_v3 для production.
    
    Веса оптимизированы для viral content:
        - hook_strength: 0.20 (самое важное - зацепка)
        - tension_level: 0.15 (конфликт, драма)
        - action_intensity: 0.15 (движение)
        - emotional_peaks: 0.15 (эмоции)
        - visual_salience: 0.10 (заметность)
        - audio_energy: 0.10 (энергия звука)
        - payoff_presence: 0.10 (развязка)
        - speech_rate: 0.05 (темп речи)
    
    Не используется:
        - silence_ratio, pitch_variance (корреляция низкая)
    """
    return (
        0.20 * features.get("hook_strength", 0.5)
        + 0.15 * features.get("tension_level", 0.5)
        + 0.15 * features.get("action_intensity", 0.5)
        + 0.15 * features.get("emotional_peaks", 0.5)
        + 0.10 * features.get("visual_salience", 0.5)
        + 0.10 * features.get("audio_energy", 0.5)
        + 0.10 * features.get("payoff_presence", 0.5)
        + 0.05 * features.get("speech_rate", 0.5)
    )


# --- Алгоритмы (заглушки: только визуал от teacher; ASR/Audio/LLM — заглушки) ---


def find_viral_moments_v3(
    base_analysis: Dict,
    video_duration_sec: float,
    video_path: str,
    asr_segments: Optional[List[Dict]] = None,
    topic_segments: Optional[List[Dict]] = None,
    top_k: int = 5,
    threshold: float = 0.6,
    min_per_topic: int = 1
) -> List[Dict]:
    """
    VIRAL v3.0 PRODUCTION: Late fusion + topic segmentation.
    
    Architecture:
        1. Разбивка по topic segments (если есть) или sliding windows
        2. Для каждого окна: compute_viral_features_for_window (11 фичей)
        3. Late fusion: audio_score + visual_score + semantic_score
        4. Гарантируем min_per_topic моментов на тему
    
    Args:
        base_analysis: визуальные метрики от YOLOв8x
        video_duration_sec: длительность видео
        video_path: путь к видео (обязательно)
        asr_segments: транскрипт из Whisper
        topic_segments: тематические границы из audio_topic_segmentation
        top_k: количество топ-моментов
        threshold: минимальный порог virality_score
        min_per_topic: минимум моментов на тему
    """
    viral_moments = []
    
    # Если есть topic segments - работаем по темам
    if topic_segments:
        print(f"  Using topic segmentation: {len(topic_segments)} topics")
        
        for topic_idx, topic in enumerate(topic_segments):
            # Создаём окна ВНУТРИ темы
            topic_duration = topic["end"] - topic["start"]
            windows = create_sliding_windows(
                duration_sec=topic_duration,
                window_size=min(30, topic_duration),
                step=15,
                min_duration=5.0
            )
            
            topic_moments = []
            
            for w in windows:
                # Смещаем окна относительно начала темы
                absolute_start = topic["start"] + w["start"]
                absolute_end = topic["start"] + w["end"]
                
                # Вычисляем фичи
                features = compute_viral_features_for_window(
                    video_path=video_path,
                    start_sec=absolute_start,
                    end_sec=absolute_end,
                    base_analysis=base_analysis,
                    asr_segments=asr_segments
                )
                
                # Late fusion score + суб-скоры
                score = virality_score_v3(features)
                audio_subscore = compute_audio_score(features)
                visual_subscore = compute_visual_score(features)
                semantic_subscore = compute_semantic_score(features)
                
                if score >= threshold:
                    topic_moments.append({
                        "start": absolute_start,
                        "end": absolute_end,
                        "virality_score": round(score, 4),
                        "audio_score": round(audio_subscore, 3),
                        "visual_score": round(visual_subscore, 3),
                        "semantic_score": round(semantic_subscore, 3),
                        "topic_id": topic_idx,
                        "topic_confidence": round(topic.get("confidence", 0.5), 3),
                        "viral_feature_breakdown": features.get("viral_feature_breakdown", {}),
                        **{k: (round(v, 3) if isinstance(v, (int, float)) else v)
                           for k, v in features.items()
                           if not k.startswith("_") and k != "viral_feature_breakdown"}
                    })
            
            # Гарантируем минимум моментов на тему
            if len(topic_moments) < min_per_topic and windows:
                # Берём лучший момент из темы даже если ниже threshold
                all_moments_in_topic = []
                for w in windows:
                    absolute_start = topic["start"] + w["start"]
                    absolute_end = topic["start"] + w["end"]
                    features = compute_viral_features_for_window(
                        video_path, absolute_start, absolute_end, base_analysis, asr_segments
                    )
                    score = virality_score_v3(features)
                    all_moments_in_topic.append((score, absolute_start, absolute_end, features))
                
                # Берём лучший
                if all_moments_in_topic:
                    best = max(all_moments_in_topic, key=lambda x: x[0])
                    score, start, end, features = best
                    topic_moments.append({
                        "start": start,
                        "end": end,
                        "virality_score": round(score, 4),
                        "audio_score": round(compute_audio_score(features), 3),
                        "visual_score": round(compute_visual_score(features), 3),
                        "semantic_score": round(compute_semantic_score(features), 3),
                        "topic_id": topic_idx,
                        "forced": True,  # Помечаем что это forced выбор
                        "viral_feature_breakdown": features.get("viral_feature_breakdown", {}),
                        **{k: (round(v, 3) if isinstance(v, (int, float)) else v)
                           for k, v in features.items()
                           if not k.startswith("_") and k != "viral_feature_breakdown"}
                    })
            
            viral_moments.extend(topic_moments)
    
    else:
        # Fallback: sliding windows по всему видео
        print(f"  Using sliding windows (no topics)")
        windows = create_sliding_windows(
            duration_sec=video_duration_sec,
            window_size=30,
            step=15,
            min_duration=5.0
        )
        
        for w in windows:
            features = compute_viral_features_for_window(
                video_path=video_path,
                start_sec=w["start"],
                end_sec=w["end"],
                base_analysis=base_analysis,
                asr_segments=asr_segments
            )
            
            # Late fusion score + суб-скоры
            score = virality_score_v3(features)
            audio_subscore = compute_audio_score(features)
            visual_subscore = compute_visual_score(features)
            semantic_subscore = compute_semantic_score(features)
            
            if score >= threshold:
                viral_moments.append({
                    "start": w["start"],
                    "end": w["end"],
                    "virality_score": round(score, 4),
                    "audio_score": round(audio_subscore, 3),
                    "visual_score": round(visual_subscore, 3),
                    "semantic_score": round(semantic_subscore, 3),
                    "viral_feature_breakdown": features.get("viral_feature_breakdown", {}),
                    **{k: (round(v, 3) if isinstance(v, (int, float)) else v)
                       for k, v in features.items()
                       if not k.startswith("_") and k != "viral_feature_breakdown"}
                })
    
    # Сортируем по скору и берём топ-K
    viral_moments.sort(key=lambda x: x["virality_score"], reverse=True)
    return viral_moments[:top_k]


def find_viral_moments(
    base_analysis: Dict,
    video_duration_sec: float = 60.0,
    video_path: Optional[str] = None,
    top_k: int = 5,
    threshold: float = 0.55,
) -> List[Dict]:
    """
    Режим 1: сегменты с высоким virality_score.
    Использует sliding windows для поиска лучших моментов.
    
    Args:
        base_analysis: визуальные метрики от YOLOv8x
        video_duration_sec: длительность видео
        video_path: путь к видео (для audio_energy)
        top_k: количество топ-моментов
        threshold: минимальный порог virality_score
    """
    # Создаём скользящие окна (30s window, 15s step = 50% overlap)
    windows = create_sliding_windows(
        duration_sec=video_duration_sec,
        window_size=30,
        step=15,
        min_duration=5.0
    )
    
    # Базовые визуальные метрики (усредненные по всему видео)
    base_composition = base_analysis.get("composition_score", 0.5)
    base_emotion = base_analysis.get("emotional_peaks", 0.5)
    base_action = base_analysis.get("action_intensity", 0.5)
    base_salience = base_analysis.get("visual_salience", 0.5)
    
    viral_moments = []
    
    for w in windows:
        # Вычисляем audio_energy для окна (если есть путь к видео)
        if video_path:
            audio_energy = compute_audio_energy(video_path, w["start"], w["end"])
        else:
            audio_energy = 0.5  # Fallback
        
        # Вычисляем virality_score для окна
        score = virality_score(
            hook_quality=base_composition,
            emotion_peak=base_emotion,
            action_intensity=base_action,
            visual_salience=base_salience,
            audio_energy=audio_energy,
            shareability_score=0.6,  # TODO: добавить LLM анализ
            trending_format_match=0.0,  # TODO: добавить trending detector
        )
        
        # Фильтруем по порогу
        if score >= threshold:
            viral_moments.append({
                "start": w["start"],
                "end": w["end"],
                "virality_score": round(score, 4),
                "hook_quality": round(base_composition, 3),
                "emotion_peak": round(base_emotion, 3),
                "audio_energy": round(audio_energy, 3),  # Для отладки
            })
    
    # Сортируем по скору и берём топ-K
    viral_moments.sort(key=lambda x: x["virality_score"], reverse=True)
    return viral_moments[:top_k]


def find_educational_moments(
    base_analysis: Dict,
    transcript_segments: Optional[List[Dict]] = None,
    edu_score_threshold: float = 0.6,
) -> List[Dict]:
    """
    Режим 2: сегменты с образовательной ценностью.
    transcript_segments: [{"start", "end", "text", "insight_quality", ...}]. Если None — заглушка по визуалу.
    """
    clarity = base_analysis.get("clarity_score", 0.5)
    comp = base_analysis.get("composition_score", 0.5)
    if transcript_segments:
        out = []
        for seg in transcript_segments:
            edu = educational_score(
                insight_quality=seg.get("insight_quality", 0.5),
                explanation_clarity=seg.get("explanation_clarity", 0.5),
                practical_value=seg.get("practical_value", 0.5),
                clarity_score=clarity,
                structure_score=seg.get("structure_score", 0.5),
            )
            if edu >= edu_score_threshold:
                out.append({"start": seg["start"], "end": seg["end"], "edu_score": round(edu, 4), "insight": seg.get("key_takeaway", "")})
        return out
    edu = educational_score(clarity_score=clarity, structure_score=comp)
    if edu >= edu_score_threshold:
        return [{"start": 0, "end": 60, "edu_score": round(edu, 4), "insight": ""}]
    return []


def find_story_moments(
    base_analysis: Dict,
    transcript_windows: Optional[List[Dict]] = None,
    video_duration_sec: float = 60.0,
) -> List[Dict]:
    """
    Режим 3: сегменты с полным story arc.
    transcript_windows: [{"start", "end", "text", "setup_quality", "conflict_intensity", ...}]. Если None — заглушка.
    """
    if transcript_windows:
        out = []
        for w in transcript_windows:
            sc = story_score(
                setup_quality=w.get("setup_quality", 0.5),
                conflict_intensity=w.get("conflict_intensity", 0.5),
                resolution_quality=w.get("resolution_quality", 0.5),
                has_twist_bonus=0.2 if w.get("has_twist") else 0.0,
                emotional_arc_score=w.get("emotional_arc_score", 0.5),
            )
            out.append({"start": w["start"], "end": w["end"], "story_score": round(sc, 4), "arc_type": w.get("arc_type", "unknown"), "summary": w.get("story_summary", "")})
        return sorted(out, key=lambda x: x["story_score"], reverse=True)
    emo = base_analysis.get("emotional_peaks", 0.5)
    sc = story_score(emotional_arc_score=emo)
    end_sec = max(1.0, video_duration_sec)
    return [{"start": 0, "end": end_sec, "story_score": round(sc, 4), "arc_type": "unknown", "summary": ""}]


def find_hooks(
    base_analysis: Dict,
    potential_clips: Optional[List[Dict]] = None,
    hook_score_threshold: float = 0.7,
    first_5s_hook_analysis: Optional[Dict] = None,
    first_5s_text: Optional[str] = None,
) -> List[Dict]:
    """
    Режим 4: сильные первые 1–5 сек.
    first_5s_hook_analysis: от LLM (intrigue_score, emotional_intensity, hook_type) — если есть, используем в скоре.
    """
    sal = base_analysis.get("visual_salience", 0.5)
    act = base_analysis.get("action_intensity", 0.5)
    if first_5s_hook_analysis:
        intrigue = first_5s_hook_analysis.get("intrigue_score", 0.5)
        emo = first_5s_hook_analysis.get("emotional_intensity", 0.5)
        hook_type = first_5s_hook_analysis.get("hook_type", "unknown")
    else:
        intrigue, emo, hook_type = 0.5, 0.5, "unknown"
    sc = hook_score(intrigue_score=intrigue, visual_salience=sal, emotional_intensity=emo, audio_energy=0.5, action_intensity=act)
    if sc < hook_score_threshold:
        return []
    clips = potential_clips or [{"start": 0, "end": 30}]
    out = []
    opening = (first_5s_text or "")[:100] if first_5s_text else None
    for c in clips[:5]:
        out.append({"clip_start": c.get("start", 0), "hook_type": hook_type, "hook_score": round(sc, 4), "opening_line": opening})
    return out


def create_trailer(
    viral_moments: List[Dict],
    educational_moments: List[Dict],
    story_moments: List[Dict],
    hooks: List[Dict],
    target_length_sec: int = 60,
    diversity_weight: float = 0.3,
    quality_weight: float = 0.7,
    video_duration_sec: float = 0.0,
    profile: Optional[str] = None,
    base_analysis: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Mode 5: thin wrapper over trailer_mode_v1.find_trailer_clips().

    Legacy parameters are preserved for backward compatibility.
    New parameters:
        profile (str | None):
            Profile name ("short" / "long" / "promo" / "universal") or path to YAML.
            Overrides target_length_sec when provided.
            Uses TrailerModeConfig.from_yaml_or_profile() [v1.2].
        base_analysis (dict | None):
            base_analysis from Grok4Teacher for energy_bonus computation [v1.2].
    """
    try:
        from trailer_mode_v1 import find_trailer_clips, TrailerModeConfig

        # --- Load config via from_yaml_or_profile [v1.2] ---
        if profile is not None:
            cfg = TrailerModeConfig.from_yaml_or_profile(profile)
        else:
            cfg = TrailerModeConfig(target_trailer_duration=float(target_length_sec))

        hook_result        = {"hook_moments":         hooks}               if hooks               else None
        story_result       = {"story_moments":        story_moments}       if story_moments       else None
        viral_result       = {"viral_moments":        viral_moments}       if viral_moments       else None
        educational_result = {"educational_segments": educational_moments} if educational_moments else None

        return find_trailer_clips(
            video_path="",
            video_duration_sec=video_duration_sec or float(target_length_sec) * 2,
            hook_result=hook_result,
            story_result=story_result,
            viral_result=viral_result,
            educational_result=educational_result,
            base_analysis=base_analysis,
            config=cfg,
        )
    except ImportError:
        # Fallback на старую логику если trailer_mode_v1 недоступен
        _cfg = get_modes_config().get("modes", {}).get("trailer", {})
        limits = _cfg.get("moments_per_mode", {"viral": 3, "educational": 2, "stories": 2, "hooks": 2})
        all_moments = []
        for m in viral_moments[: limits.get("viral", 3)]:
            all_moments.append({**m, "type": "viral", "priority": 1.0})
        for m in educational_moments[: limits.get("educational", 2)]:
            all_moments.append({**m, "type": "educational", "priority": 0.7})
        for m in story_moments[: limits.get("stories", 2)]:
            all_moments.append({**m, "type": "story", "priority": 0.8})
        for m in hooks[: limits.get("hooks", 2)]:
            all_moments.append({**m, "type": "hook", "priority": 1.0})
        all_moments.sort(key=lambda x: x.get("priority", 0.5), reverse=True)
        selected = all_moments[: max(5, target_length_sec // 15)]
        total_dur = sum(m.get("end", 10) - m.get("start", 0) for m in selected)
        return {
            "sequence": selected,
            "total_length": total_dur,
            "diversity_score": len(set(m.get("type", "") for m in selected)) / 4.0,
            "render_instructions": [
                {"start": m.get("start", 0), "end": m.get("end", m.get("start", 0) + 10),
                 "type": m.get("type", "unknown")}
                for m in selected
            ],
        }


# --- Точка входа: анализ по режимам (использует teacher) ---


def analyze_modes(
    base_analysis: Dict,
    video_duration_sec: float = 60.0,
    video_path: Optional[str] = None,
    transcript_segments: Optional[List[Dict]] = None,
    transcript_windows: Optional[List[Dict]] = None,
    first_5s_hook_analysis: Optional[Dict] = None,
    first_5s_text: Optional[str] = None,
    use_viral_v2: bool = False,
    use_viral_v3: bool = True,
    topic_segments: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """
    Запуск всех 5 режимов по base_analysis (от Grok4Teacher).
    
    Args:
        base_analysis: визуальные метрики от YOLOv8x
        video_duration_sec: длительность видео
        video_path: путь к видео (для audio_energy)
        transcript_segments: транскрипт из Whisper
        topic_segments: тематические границы (опционально)
        use_viral_v2: использовать VIRAL v2.0 (deprecated)
        use_viral_v3: использовать VIRAL v3.0 PRODUCTION (по умолчанию)
    """
    # VIRAL режим: v3 (production) > v2 (deprecated) > v1 (legacy)
    if use_viral_v3 and video_path:
        viral_moments = find_viral_moments_v3(
            base_analysis=base_analysis,
            video_duration_sec=video_duration_sec,
            video_path=video_path,
            asr_segments=transcript_segments,
            topic_segments=topic_segments
        )
        version = "v3.0"
    elif use_viral_v2 and video_path:
        viral_moments = find_viral_moments_v2(
            base_analysis=base_analysis,
            video_duration_sec=video_duration_sec,
            video_path=video_path,
            asr_segments=transcript_segments
        )
        version = "v2.0"
    else:
        viral_moments = find_viral_moments(
            base_analysis,
            video_duration_sec=video_duration_sec,
            video_path=video_path
        )
        version = "v1.0"
    
    educational_moments = find_educational_moments(base_analysis, transcript_segments=transcript_segments)
    story_moments = find_story_moments(base_analysis, transcript_windows=transcript_windows, video_duration_sec=video_duration_sec)
    hooks = find_hooks(
        base_analysis,
        potential_clips=viral_moments if viral_moments else None,
        first_5s_hook_analysis=first_5s_hook_analysis,
        first_5s_text=first_5s_text,
    )
    trailer = create_trailer(viral_moments, educational_moments, story_moments, hooks, target_length_sec=60)
    return {
        "viral": viral_moments,
        "educational": educational_moments,
        "stories": story_moments,
        "hooks": hooks,
        "trailer": trailer,
        "version": version
    }


def find_viral_moments_v2(
    base_analysis: Dict,
    video_duration_sec: float,
    video_path: str,
    asr_segments: Optional[List[Dict]] = None,
    top_k: int = 5,
    threshold: float = 0.6
) -> List[Dict]:
    """
    VIRAL v2.0: все фичи реальные, формула финальная.
    DEPRECATED: используйте find_viral_moments_v3 для production.
    
    Использует:
        - 4 визуальные фичи (YOLOv8x)
        - 4 аудио фичи (librosa)
        - 3 семантические фичи (LLM)
    
    Args:
        base_analysis: визуальные метрики от YOLOv8x
        video_duration_sec: длительность видео
        video_path: путь к видео (обязательно)
        asr_segments: транскрипт из Whisper (опционально)
        top_k: количество топ-моментов
        threshold: минимальный порог virality_score
    """
    # Создаём скользящие окна (30s window, 15s step)
    windows = create_sliding_windows(
        duration_sec=video_duration_sec,
        window_size=30,
        step=15,
        min_duration=5.0
    )
    
    viral_moments = []
    
    for w in windows:
        # Вычисляем ВСЕ 11 фич для окна
        features = compute_viral_features_for_window(
            video_path=video_path,
            start_sec=w["start"],
            end_sec=w["end"],
            base_analysis=base_analysis,
            asr_segments=asr_segments
        )
        
        # Финальный скор по новой формуле
        score = virality_score_v2(features)
        
        # Фильтруем по порогу
        if score >= threshold:
            viral_moments.append({
                "start": w["start"],
                "end": w["end"],
                "virality_score": round(score, 4),
                "viral_feature_breakdown": features.get("viral_feature_breakdown", {}),
                **{k: (round(v, 3) if isinstance(v, (int, float)) else v)
                   for k, v in features.items()
                   if not k.startswith("_") and k != "viral_feature_breakdown"}
            })
    
    # Сортируем по скору и берём топ-K
    viral_moments.sort(key=lambda x: x["virality_score"], reverse=True)
    return viral_moments[:top_k]


def compute_audio_energy(video_path: str, start_sec: float, end_sec: float) -> float:
    """
    Вычисляет энергию аудио в сегменте (RMS amplitude).
    Возвращает нормированное значение 0-1.
    Использует module-level cache: загружает полное аудио один раз на видео.
    """
    if not HAS_AUDIO:
        return 0.5

    try:
        y_full, sr = load_full_cached_audio(video_path, sr=16000)
        if y_full is None or len(y_full) == 0:
            return 0.5

        start_sample = int(start_sec * sr)
        end_sample = int(end_sec * sr)
        y = y_full[start_sample:end_sample]
        if len(y) == 0:
            return 0.5

        rms = librosa.feature.rms(y=y)[0]
        energy = float(np.mean(rms))
        return min(energy * 20, 1.0)
    except Exception:
        return 0.5


def compute_audio_features(
    video_path: str,
    start_sec: float,
    end_sec: float,
    asr_segments: Optional[List[Dict]] = None,
) -> Dict[str, float]:
    """
    Вычисляет аудио-фичи для одного окна (VIRAL режим).

    Оптимизация long-video: загружает полное аудио один раз на видео (module-level cache).
    WAV-кэш через ffmpeg устраняет повторные PySoundFile/audioread предупреждения.

    Возвращает: audio_energy, speech_rate, silence_ratio, pitch_variance
    """
    _fallback = {
        "audio_energy": 0.5,
        "speech_rate": 0.5,
        "silence_ratio": 0.5,
        "pitch_variance": 0.5,
        "_source": "fallback_no_audio",
    }

    if not HAS_AUDIO:
        return _fallback

    try:
        # Load full audio once per video (cached in _AUDIO_LOADED_CACHE)
        y_full, sr = load_full_cached_audio(video_path, sr=16000)
        if y_full is None or len(y_full) == 0:
            return {**_fallback, "_source": "fallback_load_failed"}

        # Slice to window
        start_sample = int(start_sec * sr)
        end_sample = int(end_sec * sr)
        y = y_full[start_sample:end_sample]
        if len(y) == 0:
            return {**_fallback, "_source": "fallback_empty_slice"}

        # 1. Audio energy (RMS)
        rms = librosa.feature.rms(y=y)[0]
        audio_energy = min(float(np.mean(rms)) * 20, 1.0)

        # 2. Speech rate (from ASR, no audio needed)
        speech_rate = 0.5
        if asr_segments:
            words_in_window = []
            for seg in asr_segments:
                if seg["start"] < end_sec and seg["end"] > start_sec:
                    words_in_window.extend(seg["text"].split())
            duration = end_sec - start_sec
            if duration > 0 and words_in_window:
                speech_rate = min(len(words_in_window) / (duration * 3.0), 1.0)

        # 3. Silence ratio
        silence_ratio = float(np.sum(np.abs(y) < 0.01) / max(len(y), 1))

        # 4. Pitch variance
        pitch_variance = 0.0
        try:
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr, fmin=50, fmax=400)
            pitch_values = []
            for t in range(pitches.shape[1]):
                idx = magnitudes[:, t].argmax()
                pitch = pitches[idx, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            if len(pitch_values) > 1:
                pitch_variance = min(float(np.std(pitch_values)) / 100.0, 1.0)
        except Exception:
            pitch_variance = 0.0

        return {
            "audio_energy": audio_energy,
            "speech_rate": speech_rate,
            "silence_ratio": silence_ratio,
            "pitch_variance": pitch_variance,
            "_source": "cached_wav",
        }

    except Exception:
        return {**_fallback, "_source": "fallback_exception"}


def create_sliding_windows(
    duration_sec: float,
    window_size: int = 30,
    step: int = 15,
    min_duration: float = 5.0
) -> List[Dict]:
    """
    Создаёт скользящие окна с перекрытием (50% overlap по умолчанию).
    
    Args:
        duration_sec: длительность видео
        window_size: размер окна (сек)
        step: сдвиг окна (сек)
        min_duration: минимальная длина окна
    
    Returns:
        [{"start": 0, "end": 30}, {"start": 15, "end": 45}, ...]
    """
    windows = []
    
    for start in range(0, int(duration_sec), step):
        end = min(start + window_size, duration_sec)
        
        # Пропускаем слишком короткие окна
        if end - start >= min_duration:
            windows.append({
                "start": float(start),
                "end": float(end)
            })
    
    return windows


# v3.1: LEXICAL FALLBACK — used when LLM unavailable or API key missing
_VIRAL_LEXICAL_MARKERS = {
    "strong_claims": [
        "все", "никто", "никогда", "всегда", "главное", "ключевое",
        "нужно только", "работает всегда", "гарантирую", "100%",
        "every", "never", "always", "only thing", "the key",
    ],
    "emotion_words": [
        "шок", "шокирован", "удивительно", "невероятно", "потрясающе",
        "обалдеть", "вау", "восторг", "ужас", "страх", "ненависть",
        "любовь", "боль", "ярость", "радость",
        "shocking", "amazing", "incredible", "insane", "crazy", "wild",
    ],
    "contrast_markers": [
        "но", "однако", "хотя", "зато", "наоборот", "противоположность",
        "в отличие", "с другой стороны", "тем не менее",
        "but", "however", "instead", "on the contrary", "yet",
    ],
    "novelty_markers": [
        "впервые", "новое", "новейшее", "секрет", "скрывают", "раскрываю",
        "никто не знает", "мало кто знает", "малоизвестный",
        "first time", "secret", "hidden", "nobody knows", "unknown",
    ],
    "first_person_conflict": [
        "я столкнулся", "моя проблема", "когда я", "я понял", "у меня была",
        "мне пришлось", "я не знал", "я думал", "я боялся", "я ошибся",
        "i faced", "my problem", "when i", "i realized", "i thought",
    ],
}


def _count_lexical_markers(text_lower: str, markers: List[str]) -> int:
    return sum(1 for m in markers if m in text_lower)


def analyze_viral_semantics_lexical(text: str) -> Dict[str, float]:
    """
    Лексикалный анализ текста БЕЗ LLM для VIRAL режима.
    Выдаёт те же 3 скора, что и LLM-версия, но на основе маркеров.

    Шкала:
      hook_strength   — questions + novelty + first-person conflict
      tension_level   — contrast markers + emotion words + first-person conflict
      payoff_presence — strong_claims + exclamations + (conclusion markers)
    """
    if not text or len(text.strip()) < 10:
        return {"hook_strength": 0.3, "tension_level": 0.3, "payoff_presence": 0.3}

    text_lower = text.lower()
    n_questions = text.count("?")
    n_excl = text.count("!")

    strong_claims = _count_lexical_markers(text_lower, _VIRAL_LEXICAL_MARKERS["strong_claims"])
    emotion = _count_lexical_markers(text_lower, _VIRAL_LEXICAL_MARKERS["emotion_words"])
    contrast = _count_lexical_markers(text_lower, _VIRAL_LEXICAL_MARKERS["contrast_markers"])
    novelty = _count_lexical_markers(text_lower, _VIRAL_LEXICAL_MARKERS["novelty_markers"])
    fp_conflict = _count_lexical_markers(text_lower, _VIRAL_LEXICAL_MARKERS["first_person_conflict"])

    # Hook strength: есть ли вопрос, обещание новизны, первое лицо
    hook_raw = (
        0.30 * min(n_questions / 2.0, 1.0)
        + 0.35 * min(novelty / 2.0, 1.0)
        + 0.20 * min(fp_conflict / 2.0, 1.0)
        + 0.15 * min(emotion / 3.0, 1.0)
    )

    # Tension: контраст, эмоции, первое лицо с конфликтом
    tension_raw = (
        0.40 * min(contrast / 2.0, 1.0)
        + 0.30 * min(emotion / 3.0, 1.0)
        + 0.30 * min(fp_conflict / 2.0, 1.0)
    )

    # Payoff: claims + exclamations + conclusion markers
    payoff_raw = (
        0.50 * min(strong_claims / 2.0, 1.0)
        + 0.30 * min(n_excl / 2.0, 1.0)
        + 0.20 * min(emotion / 4.0, 1.0)
    )

    return {
        "hook_strength": float(min(max(hook_raw, 0.15), 0.95)),
        "tension_level": float(min(max(tension_raw, 0.15), 0.95)),
        "payoff_presence": float(min(max(payoff_raw, 0.15), 0.95)),
        # raw counts for debugging
        "_lex_strong_claims": strong_claims,
        "_lex_emotion": emotion,
        "_lex_contrast": contrast,
        "_lex_novelty": novelty,
        "_lex_fp_conflict": fp_conflict,
        "_lex_questions": n_questions,
        "_lex_exclamations": n_excl,
    }


def get_visual_features_for_window_from_yolo(
    base_analysis: Dict,
    start_sec: float,
    end_sec: float,
) -> Dict[str, float]:
    """
    Window-level visual features, вычисленные из YOLO detections.
    Формат detections (benchmark): [{"timestamp_sec", "person_count",
    "objects", "confidence_max"}].

    Возвращает стандартные 4 viral-фичи + расширенные метрики
    (person_presence_ratio, object_density, confidence_peaks,
     scene_changes, motion_proxy).
    """
    detections = base_analysis.get("detections", []) or []
    if not detections:
        return {
            "action_intensity": base_analysis.get("action_intensity", 0.5),
            "visual_salience": base_analysis.get("visual_salience", 0.5),
            "composition_score": base_analysis.get("composition_score", 0.5),
            "emotional_peaks": base_analysis.get("emotional_peaks", 0.5),
            "person_presence_ratio": base_analysis.get("person_presence_ratio", 0.0),
            "object_density": 0.0,
            "confidence_peaks": 0.0,
            "scene_changes": 0.0,
            "motion_proxy": 0.0,
            "_source": "global_fallback",
        }

    # Совместимость: поддерживаем и "timestamp_sec" (benchmark), и "timestamp" (grok4_teacher)
    def _ts(d: Dict) -> float:
        return float(d.get("timestamp_sec", d.get("timestamp", 0.0)))

    in_window = [d for d in detections if start_sec <= _ts(d) <= end_sec]
    if not in_window:
        # Берём ближайший кадр
        mid = (start_sec + end_sec) / 2.0
        in_window = [min(detections, key=lambda d: abs(_ts(d) - mid))]

    person_counts = [int(d.get("person_count", 0)) for d in in_window]
    confidences = [float(d.get("confidence_max", 0.0)) for d in in_window]
    obj_counts = [len(d.get("objects", []) or []) for d in in_window]

    np_arr_conf = np.asarray(confidences) if confidences else np.asarray([0.0])
    np_arr_persons = np.asarray(person_counts) if person_counts else np.asarray([0.0])
    np_arr_objs = np.asarray(obj_counts) if obj_counts else np.asarray([0.0])

    person_presence_ratio = float(np.mean((np_arr_persons > 0).astype(float)))
    object_density = float(np.clip(np.mean(np_arr_objs) / 5.0, 0.0, 1.0))
    mean_conf = float(np.mean(np_arr_conf))
    confidence_peaks = float(np.clip((np.max(np_arr_conf) - mean_conf), 0.0, 1.0))

    # Motion proxy: frame-to-frame изменения в person_count + confidence
    if len(np_arr_conf) >= 2:
        conf_diff = float(np.mean(np.abs(np.diff(np_arr_conf))))
        person_diff = float(np.mean(np.abs(np.diff(np_arr_persons.astype(float)))))
        motion_proxy = float(np.clip(conf_diff * 2.5 + person_diff * 0.5, 0.0, 1.0))
    else:
        motion_proxy = 0.0

    # Scene changes: скачки в object set между кадрами
    if len(in_window) >= 2:
        scene_change_events = 0
        prev_objs = {o.get("class") for o in in_window[0].get("objects", [])}
        for d in in_window[1:]:
            cur_objs = {o.get("class") for o in d.get("objects", [])}
            overlap = len(prev_objs & cur_objs) / max(len(prev_objs | cur_objs), 1)
            if overlap < 0.4:
                scene_change_events += 1
            prev_objs = cur_objs
        scene_changes = float(np.clip(scene_change_events / max(len(in_window) - 1, 1), 0.0, 1.0))
    else:
        scene_changes = 0.0

    # Главные 4 фичи, совместимые с формулой virality_score:
    action_intensity = float(np.clip(motion_proxy * 0.6 + object_density * 0.4, 0.0, 1.0))
    visual_salience = float(np.clip(mean_conf * 0.5 + confidence_peaks * 0.5, 0.0, 1.0))
    composition_score = float(np.clip(person_presence_ratio * 0.7 + (1 - scene_changes) * 0.3, 0.0, 1.0))
    emotional_peaks = float(np.clip(confidence_peaks * 0.4 + motion_proxy * 0.3 + person_presence_ratio * 0.3, 0.0, 1.0))

    return {
        "action_intensity": action_intensity,
        "visual_salience": visual_salience,
        "composition_score": composition_score,
        "emotional_peaks": emotional_peaks,
        "person_presence_ratio": person_presence_ratio,
        "object_density": object_density,
        "confidence_peaks": confidence_peaks,
        "scene_changes": scene_changes,
        "motion_proxy": motion_proxy,
        "_source": "yolo_window",
        "_n_frames": len(in_window),
    }


def analyze_viral_semantics(text: str) -> Dict[str, float]:
    """
    LLM анализ текста для VIRAL режима.
    
    Анализирует:
        - hook_strength: неожиданность, вопрос, интрига
        - tension_level: конфликт, спор, сильная эмоция
        - payoff_presence: punchline, вывод, резолюция
    
    Returns:
        Dict с 3 скорами (0-1)
    """
    # v3.1: если LLM недоступен — используем lexical fallback (а не тупое 0.5)
    if not HAS_LLM or not _call_llm:
        return analyze_viral_semantics_lexical(text)
    
    if not text or len(text.strip()) < 10:
        return analyze_viral_semantics_lexical(text)
    
    prompt = f'''Analyze this video segment for virality. Reply with ONLY a JSON object.

Text: "{text[:1000]}"

JSON format:
{{
  "hook_strength": 0.0 to 1.0,
  "tension_level": 0.0 to 1.0,
  "payoff_presence": 0.0 to 1.0
}}

Scoring guide:
- hook_strength: Rate high (0.7-1.0) if: "You won't believe...", shocking statement, strong question, unexpected reveal
- tension_level: Rate high (0.7-1.0) if: argument, debate, high emotion, suspense, conflict, drama
- payoff_presence: Rate high (0.7-1.0) if: punchline, reveal, conclusion, satisfying moment, resolution

Rate low (0.0-0.3) if: boring, no hook, calm, no conflict, no conclusion.'''

    try:
        raw = _call_llm(prompt, max_tokens=200)
        data = _parse_json_from_response(raw) if raw else None
        
        if not data:
            # v3.1: вместо тупого 0.5 — lexical fallback
            return analyze_viral_semantics_lexical(text)
        
        return {
            "hook_strength": float(data.get("hook_strength", 0.5)),
            "tension_level": float(data.get("tension_level", 0.5)),
            "payoff_presence": float(data.get("payoff_presence", 0.5))
        }
    except Exception as e:
        return analyze_viral_semantics_lexical(text)


def compute_viral_features_for_window(
    video_path: str,
    start_sec: float,
    end_sec: float,
    base_analysis: Dict,
    asr_segments: Optional[List[Dict]] = None
) -> Dict[str, float]:
    """
    Вычисляет ВСЕ фичи для одного окна (VIRAL режим).
    
    Возвращает 11 фич без заглушек:
        ВИЗУАЛ (4): action_intensity, visual_salience, composition_score, emotional_peaks
        АУДИО (4): audio_energy, speech_rate, silence_ratio, pitch_variance
        СЕМАНТИКА (3): hook_strength, tension_level, payoff_presence
    """
    # 1. Визуальные фичи — приоритет:
    #    (a) YOLO detections из benchmark → window-aware
    #    (b) grok4_teacher frames_data → window-aware
    #    (c) global fallback (deprecated)
    detections = base_analysis.get("detections") or []
    has_yolo_timeseries = (
        detections
        and any("timestamp_sec" in d or "timestamp" in d for d in detections)
    )

    if has_yolo_timeseries:
        visual_features = get_visual_features_for_window_from_yolo(
            base_analysis, start_sec, end_sec,
        )
    elif HAS_TIMESERIES_VISUAL and get_visual_features_for_window:
        visual_features = get_visual_features_for_window(base_analysis, start_sec, end_sec)
        visual_features["_source"] = "frames_data_window"
    else:
        visual_features = {
            "action_intensity": base_analysis.get("action_intensity", 0.5),
            "visual_salience": base_analysis.get("visual_salience", 0.5),
            "composition_score": base_analysis.get("composition_score", 0.5),
            "emotional_peaks": base_analysis.get("emotional_peaks", 0.5),
            "_source": "global_fallback_deprecated",
        }
    
    # 2. Аудио фичи (реальные для окна)
    audio_features = compute_audio_features(video_path, start_sec, end_sec, asr_segments)
    
    # 3. Семантические фичи — LLM если доступен, иначе lexical fallback
    semantic_features = {
        "hook_strength": 0.3,
        "tension_level": 0.3,
        "payoff_presence": 0.3,
        "_source": "no_text",
    }
    
    if asr_segments:
        # Собираем текст из ASR сегментов в этом окне
        window_text_parts = []
        for seg in asr_segments:
            # Проверяем пересечение сегмента с окном
            if seg["start"] < end_sec and seg["end"] > start_sec:
                window_text_parts.append(seg["text"])
        
        window_text = " ".join(window_text_parts).strip()
        
        if window_text and len(window_text) > 20:
            # v3.1: LLM + lexical fallback inside analyze_viral_semantics
            semantic_features = analyze_viral_semantics(window_text)
            # Помечаем источник
            if "_source" not in semantic_features:
                semantic_features["_source"] = "llm" if HAS_LLM else "lexical"
        else:
            # Короткий текст — тоже lexical (поставит минимальные значения)
            semantic_features = analyze_viral_semantics_lexical(window_text or "")
            semantic_features["_source"] = "lexical_short_text"
    
    # v3.1: объединяем все фичи + возвращаем структурированный feature breakdown
    merged = {
        **visual_features,
        **audio_features,
        **semantic_features,
    }
    # Прячем служебные "_source" в отдельный ключ, но оставляем для debug
    merged["viral_feature_breakdown"] = {
        "visual": {k: v for k, v in visual_features.items() if not k.startswith("_")},
        "audio":  {k: v for k, v in audio_features.items() if not k.startswith("_")},
        "semantic": {k: v for k, v in semantic_features.items() if not k.startswith("_")},
        "sources": {
            "visual": visual_features.get("_source", "unknown"),
            "semantic": semantic_features.get("_source", "unknown"),
        },
    }
    return merged


def _get_video_duration_sec(video_path: str) -> float:
    """Длительность видео в секундах через OpenCV."""
    try:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        cap.release()
        return count / fps if fps > 0 else 60.0
    except Exception:
        return 60.0


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="5 режимов: viral / educational / stories / hooks / trailer")
    parser.add_argument("--video", type=str, default=None, help="Путь к видео (реальный тест через Grok4Teacher)")
    parser.add_argument("--no-grok", action="store_true", help="Не вызывать Grok для сцены (только YOLO)")
    parser.add_argument("--no-asr", action="store_true", help="Не запускать ASR+LLM (только визуал)")
    parser.add_argument("--use-topics", action="store_true", help="Использовать topic segmentation")
    parser.add_argument("--version", type=str, default="v3", choices=["v1", "v2", "v3"], 
                        help="Версия VIRAL: v3=production (default), v2=deprecated, v1=legacy")
    parser.add_argument("--asr", action="store_true", help="Включить ASR (по умолчанию выключен)")
    args = parser.parse_args()

    if args.video:
        video_path = Path(args.video)
        if not video_path.exists():
            print(f"File not found: {video_path}")
            exit(1)
        print(f"Video: {video_path}")
        print("Running Grok4Teacher (YOLOv8x + optional Grok scene)...")
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from grok4_teacher import Grok4Teacher
        teacher = Grok4Teacher(fallback_to_local=True, enable_grok_scene=not args.no_grok)
        base_analysis = teacher.analyze_video(str(video_path))
        duration_sec = _get_video_duration_sec(str(video_path))
        print(f"Duration: {duration_sec:.1f}s")

        transcript_segments = None
        transcript_windows = None
        first_5s_hook_analysis = None
        first_5s_text = None
        topic_segments = None

        # Topic Segmentation (если запрошено)
        if args.use_topics:
            print("Running Topic Segmentation...")
            try:
                from audio_topic_segmentation import TopicSegmenter
                segmenter = TopicSegmenter()
                topics = segmenter.segment_video(str(video_path))
                if topics:
                    topic_segments = topics
                    print(f"  Topics: {len(topic_segments)} segments")
                else:
                    print("  Topics: no segments found")
            except Exception as e:
                print(f"  Topic segmentation error: {e}")

        # ASR: по умолчанию выключен (медленно), включается флагом --asr
        if args.asr:
            print("Running ASR (Whisper)...")
            try:
                from asr_transcribe import transcribe_video, segments_to_windows, get_first_n_seconds_text
                segments = transcribe_video(str(video_path))
                if segments:
                    print(f"  ASR: {len(segments)} segments")
                    from llm_segment_analysis import analyze_educational, analyze_story_arc, analyze_hook
                    # Educational: LLM по каждому сегменту (лимит 5 для экономии)
                    transcript_segments = []
                    for s in segments[:5]:
                        llm_edu = analyze_educational(s["text"])
                        transcript_segments.append({
                            "start": s["start"], "end": s["end"], "text": s["text"],
                            "insight_quality": llm_edu["insight_quality"],
                            "explanation_clarity": llm_edu["explanation_clarity"],
                            "practical_value": llm_edu["practical_value"],
                            "structure_score": llm_edu["structure_score"],
                            "key_takeaway": llm_edu["key_takeaway"],
                        })
                    # Story: окна 60–180 сек
                    windows = segments_to_windows(segments, min_len_sec=30, max_len_sec=120, step_sec=40)
                    transcript_windows = []
                    for w in windows[:3]:
                        llm_story = analyze_story_arc(w["text"])
                        transcript_windows.append({
                            "start": w["start"], "end": w["end"], "text": w["text"],
                            "setup_quality": llm_story["setup_quality"],
                            "conflict_intensity": llm_story["conflict_intensity"],
                            "resolution_quality": llm_story["resolution_quality"],
                            "has_twist": llm_story["has_twist"],
                            "arc_type": llm_story["arc_type"],
                            "story_summary": llm_story["story_summary"],
                            "emotional_arc_score": llm_story["emotional_arc_score"],
                        })
                    # Hook: первые 5 сек
                    first_5s_text = get_first_n_seconds_text(segments, 5)
                    first_5s_hook_analysis = analyze_hook(first_5s_text)
                else:
                    print("  ASR: no speech segments (silent or Whisper failed)")
            except Exception as e:
                print(f"  ASR/LLM skip: {e}")

        # Определяем версию VIRAL
        use_v2 = (args.version == "v2")
        use_v3 = (args.version == "v3")
        viral_version_name = {
            "v1": "v1.0 (6 фич, legacy)",
            "v2": "v2.0 (11 фич, deprecated)",
            "v3": "v3.0 PRODUCTION (late fusion)"
        }.get(args.version, "v3.0")
        
        print(f"Running 5 modes (viral / educational / stories / hooks / trailer)...")
        print(f"  VIRAL version: {viral_version_name}")
        if topic_segments:
            print(f"  Topic segmentation: ENABLED ({len(topic_segments)} topics)")
        
        result = analyze_modes(
            base_analysis,
            video_duration_sec=duration_sec,
            video_path=str(video_path),
            transcript_segments=transcript_segments,
            transcript_windows=transcript_windows,
            first_5s_hook_analysis=first_5s_hook_analysis,
            first_5s_text=first_5s_text,
            use_viral_v2=use_v2,
            use_viral_v3=use_v3,
            topic_segments=topic_segments,
        )
        print("\n✅ RESULT:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        # Демо на тестовых данных
        base = {
            "object_confidence": 0.7,
            "visual_salience": 0.8,
            "action_intensity": 0.6,
            "composition_score": 0.75,
            "emotional_peaks": 0.5,
            "clarity_score": 0.9,
            "cut_points": [10, 25, 40],
        }
        result = analyze_modes(base, video_duration_sec=120)
        print(json.dumps(result, indent=2, ensure_ascii=False))
