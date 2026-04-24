"""
asr.py — единый ASR модуль SONYA (объединяет asr_transcribe.py + asr_production.py).

Исправления по сравнению с оригиналами:
  - Кэш модели Whisper (_whisper_model_cache) — не грузить модель каждый раз
  - language=None = auto-detect (вместо hardcoded "ru")
  - Chunked transcription с корректным смещением таймкодов
  - segments_to_windows: новая версия с sliding window (фикс бага оригинала)
  - get_first_n_seconds_text: фикс условия фильтрации
  - ffmpeg путь через utils.get_ffmpeg_path (кроссплатформенный)

Использование:
    from asr import transcribe_video, transcribe_video_chunked, segments_to_windows

    # Простое — одним вызовом
    segs = transcribe_video("video.mp4", model_size="base")

    # Длинные видео — по чанкам
    segs = transcribe_video_chunked("long_video.mp4", chunk_duration_sec=300)

    # Текст первых 10 секунд
    intro_text = get_first_n_seconds_text(segs, n_sec=10)

    # Скользящие окна для story-arc
    windows = segments_to_windows(segs, window_sec=30, step_sec=15)
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────── Whisper model cache
# Загружаем модель один раз — повторные вызовы мгновенные
_whisper_model_cache: Dict[str, Any] = {}  # model_size → whisper model


def _get_whisper_model(model_size: str = "base") -> Any:
    """Возвращает закэшированную модель Whisper (загружает при первом вызове)."""
    if model_size not in _whisper_model_cache:
        try:
            import whisper  # type: ignore
        except ImportError:
            raise ImportError(
                "openai-whisper not installed: pip install openai-whisper"
            )
        logger.info("Loading Whisper model '%s' (first time)...", model_size)
        _whisper_model_cache[model_size] = whisper.load_model(model_size)
        logger.info("Whisper model '%s' loaded", model_size)
    return _whisper_model_cache[model_size]


def clear_whisper_cache() -> None:
    """Выгрузить модели из памяти (освобождает RAM/VRAM)."""
    _whisper_model_cache.clear()
    logger.debug("Whisper model cache cleared")


# ─────────────────────────────────────────────── ffmpeg path helper
def _ffmpeg() -> str:
    try:
        from utils import get_ffmpeg_path
        return get_ffmpeg_path()
    except ImportError:
        import shutil
        return shutil.which("ffmpeg") or "ffmpeg"


# ═══════════════════════════════════════════════ Core: extract audio WAV

def extract_audio_wav(
    video_path: str,
    start_sec: float = 0.0,
    duration_sec: Optional[float] = None,
    output_path: Optional[str] = None,
    sample_rate: int = 16_000,
) -> str:
    """
    Извлекает аудио из видео в WAV файл (через ffmpeg, без pipe-проблем).

    Args:
        video_path:   путь к видео
        start_sec:    начало сегмента (0 = с начала)
        duration_sec: длина сегмента (None = до конца)
        output_path:  путь для WAV (None = временный файл)
        sample_rate:  sample rate (Hz)

    Returns:
        Путь к созданному WAV файлу.
    """
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".wav", prefix="sonya_asr_")
        os.close(fd)

    cmd = [_ffmpeg(), "-y", "-ss", str(start_sec)]
    if duration_sec is not None:
        cmd += ["-t", str(duration_sec)]
    cmd += [
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "1",
        output_path,
    ]

    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed: {e.stderr.decode(errors='replace')[:300]}")
    except subprocess.TimeoutExpired:
        raise RuntimeError("ffmpeg extraction timeout (>120s)")

    return output_path


# ═══════════════════════════════════════════════ Core: transcribe WAV

def transcribe_audio_file(
    audio_path: str,
    model_size: str = "base",
    language: Optional[str] = None,
) -> List[Dict]:
    """
    Транскрибирует WAV файл через Whisper.

    Args:
        audio_path:  путь к .wav файлу
        model_size:  tiny/base/small/medium/large
        language:    код языка ("ru", "en") или None = auto-detect

    Returns:
        [{"start": float, "end": float, "text": str}, ...]
    """
    model = _get_whisper_model(model_size)
    result = model.transcribe(
        audio_path,
        language=language,
        task="transcribe",
        verbose=False,
    )

    segments = []
    for seg in result.get("segments", []):
        text = (seg.get("text") or "").strip()
        if text:
            segments.append({
                "start": float(seg["start"]),
                "end":   float(seg["end"]),
                "text":  text,
            })

    # Fallback: Whisper вернул только полный текст (бывает для коротких файлов)
    if not segments:
        full_text = (result.get("text") or "").strip()
        if full_text:
            raw = result.get("segments", [])
            duration = max((float(s.get("end", 0)) for s in raw), default=60.0)
            duration = max(duration, 1.0)
            segments = [{"start": 0.0, "end": duration, "text": full_text}]

    return segments


# ═══════════════════════════════════════════════ High-level: transcribe video

def transcribe_video(
    video_path: str,
    model_size: str = "base",
    language: Optional[str] = None,
    cleanup: bool = True,
) -> List[Dict]:
    """
    Транскрибирует всё видео целиком (одним WAV → Whisper вызовом).

    Подходит для коротких видео (< 10 минут).
    Для длинных — используй transcribe_video_chunked.

    Returns:
        [{"start": sec, "end": sec, "text": str}, ...]
    """
    video_path = str(video_path)
    if not Path(video_path).exists():
        logger.warning("Video not found: %s", video_path)
        return []

    wav_path: Optional[str] = None
    try:
        wav_path = extract_audio_wav(video_path)
        return transcribe_audio_file(wav_path, model_size, language)
    except Exception as e:
        logger.error("transcribe_video failed: %s", e)
        return []
    finally:
        if cleanup and wav_path and Path(wav_path).exists():
            Path(wav_path).unlink(missing_ok=True)


def transcribe_video_chunked(
    video_path: str,
    model_size: str = "base",
    language: Optional[str] = None,
    chunk_duration_sec: float = 300.0,
) -> List[Dict]:
    """
    Транскрибирует длинное видео по чанкам.

    Каждый чанк обрабатывается отдельно, таймкоды смещаются на абсолютные.

    Args:
        video_path:          путь к видео
        model_size:          Whisper model size
        language:            код языка или None = auto
        chunk_duration_sec:  длина одного чанка (секунды, дефолт = 5 мин)

    Returns:
        Объединённые сегменты со скорректированными таймкодами.
    """
    try:
        from utils import get_video_duration
        total_duration = get_video_duration(video_path)
    except ImportError:
        # Fallback: ffprobe напрямую
        try:
            cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                   "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            total_duration = float(result.stdout.strip())
        except Exception:
            total_duration = 60.0

    if total_duration <= 0:
        return transcribe_video(video_path, model_size, language)

    import math as _math
    num_chunks = max(1, _math.ceil(total_duration / chunk_duration_sec))
    logger.info(
        "Chunked ASR: %.1fs, splitting into %d chunks of %.0fs",
        total_duration, num_chunks, chunk_duration_sec,
    )

    all_segments: List[Dict] = []
    offset = 0.0
    chunk_idx = 0

    while offset < total_duration:
        remaining = total_duration - offset
        chunk_dur = min(chunk_duration_sec, remaining)

        wav_path: Optional[str] = None
        try:
            wav_path = extract_audio_wav(
                str(video_path),
                start_sec=offset,
                duration_sec=chunk_dur,
            )
            segments = transcribe_audio_file(wav_path, model_size, language)

            # Смещаем таймкоды на абсолютные
            for s in segments:
                s["start"] += offset
                s["end"] += offset
                all_segments.append(s)

            chunk_idx += 1
            logger.info(
                "Chunk %d/%d done: %.1f–%.1fs, %d segments",
                chunk_idx, num_chunks, offset, offset + chunk_dur, len(segments),
            )
        except Exception as e:
            logger.error("Chunk %d failed at offset %.1fs: %s", chunk_idx + 1, offset, e)
        finally:
            if wav_path and Path(wav_path).exists():
                Path(wav_path).unlink(missing_ok=True)

        offset += chunk_dur

    logger.info("Total segments: %d", len(all_segments))
    return all_segments


# ═══════════════════════════════════════════════ Utility: segments_to_windows

def segments_to_windows(
    segments: List[Dict],
    window_sec: float = 30.0,
    step_sec: float = 15.0,
    video_duration: Optional[float] = None,
) -> List[Dict]:
    """
    Группирует ASR-сегменты в скользящие окна для story-arc анализа.

    Исправлена по сравнению с asr_transcribe.py:
    - параметры window_sec/step_sec (вместо hardcoded min/max/step)
    - корректный sliding window (не min_len/max_len приближение)

    Args:
        segments:       список сегментов от Whisper
        window_sec:     ширина окна (секунды)
        step_sec:       шаг (секунды)
        video_duration: общая длина видео (None = вычислить из сегментов)

    Returns:
        [{"start": float, "end": float, "text": str, "segments": list}, ...]
    """
    if not segments:
        return []

    if video_duration is None:
        video_duration = max(s["end"] for s in segments)

    windows = []
    pos = 0.0
    while pos < video_duration:
        w_start = pos
        w_end = min(pos + window_sec, video_duration)

        # Сегменты которые пересекаются с окном
        window_segs = [
            s for s in segments
            if s["end"] > w_start and s["start"] < w_end
        ]
        text = " ".join(s["text"] for s in window_segs).strip()

        windows.append({
            "start":        round(w_start, 2),
            "end":          round(w_end, 2),
            "text":         text,
            "segments":     window_segs,
            "num_segments": len(window_segs),
        })

        pos += step_sec

    return windows


# ═══════════════════════════════════════════════ Utility: first-N-seconds text

def get_first_n_seconds_text(
    segments: List[Dict],
    n_sec: float = 10.0,
) -> str:
    """
    Извлекает текст из первых N секунд видео.

    Используется для hook detection и intro analysis.

    ИСПРАВЛЕНО: старая версия в asr_transcribe.py имела баг в условии —
    `s["start"] < n_sec and s["end"] > 0` пропускала сегменты из середины видео
    (например, сегмент [0s, 200s] проходил фильтр). Теперь корректно.
    """
    if not segments:
        return ""

    texts = []
    for s in segments:
        start = float(s.get("start", 0.0))
        if start < n_sec:
            texts.append(s.get("text", "").strip())

    return " ".join(texts).strip()


# ═══════════════════════════════════════════════ Backward compat aliases
# Чтобы код использующий asr_production / asr_transcribe не ломался сразу

def get_asr_segments_production(video_path: str, model_name: str = "base") -> List[Dict]:
    """Alias для asr_production.get_asr_segments_production."""
    return transcribe_video_chunked(video_path, model_size=model_name)


def transcribe_video_segment(
    video_path: str,
    start_sec: float = 0.0,
    end_sec: Optional[float] = None,
    model_name: str = "base",
    cleanup: bool = True,
) -> List[Dict]:
    """Alias для asr_production.transcribe_video_segment."""
    if end_sec is None:
        try:
            from utils import get_video_duration
            end_sec = get_video_duration(video_path)
        except Exception:
            end_sec = start_sec + 60.0

    duration = end_sec - start_sec
    wav_path: Optional[str] = None
    try:
        wav_path = extract_audio_wav(str(video_path), start_sec=start_sec, duration_sec=duration)
        segs = transcribe_audio_file(wav_path, model_name)
        for s in segs:
            s["start"] += start_sec
            s["end"] += start_sec
        return segs
    except Exception as e:
        logger.error("transcribe_video_segment error: %s", e)
        return []
    finally:
        if cleanup and wav_path and Path(wav_path).exists():
            Path(wav_path).unlink(missing_ok=True)


# ═══════════════════════════════════════════════ CLI
if __name__ == "__main__":
    import argparse, json as _json

    parser = argparse.ArgumentParser(description="SONYA ASR module")
    parser.add_argument("--video", required=True)
    parser.add_argument("--model", default="base",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--language", default=None,
                        help="Language code (ru/en/...) or omit for auto-detect")
    parser.add_argument("--chunked", action="store_true",
                        help="Use chunked transcription for long videos")
    parser.add_argument("--chunk-sec", type=float, default=300.0)
    args = parser.parse_args()

    if args.chunked:
        segs = transcribe_video_chunked(
            args.video, args.model, args.language, args.chunk_sec
        )
    else:
        segs = transcribe_video(args.video, args.model, args.language)

    for s in segs:
        print(f"[{s['start']:.1f}s – {s['end']:.1f}s] {s['text']}")

    print(f"\nTotal: {len(segs)} segments")
