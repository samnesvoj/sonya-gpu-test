"""
asr_v2.py — optional ASR adapter (openai-whisper + faster-whisper).

Modes consume legacy segment list: [{"start", "end", "text"}, ...].
Additional artifacts are written under shared/ when requested.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

ASR_ENGINES = ("openai-whisper", "faster-whisper")
COMPUTE_TYPES = ("float16", "int8_float16", "int8")


class AsrV2Error(RuntimeError):
    """ASR v2 failed; caller should not treat as success."""


@dataclass
class AsrV2Config:
    engine: str = "openai-whisper"
    whisper_model: str = "base"
    language: Optional[str] = None
    word_timestamps: bool = False
    vad: bool = False
    compute_type: str = "float16"
    batch_size: int = 8
    strict_real: bool = False
    audio_cache_dir: Optional[Path] = None


@dataclass
class AsrV2Result:
    segments: List[Dict[str, Any]]
    words: List[Dict[str, Any]] = field(default_factory=list)
    speech_segments: List[Dict[str, Any]] = field(default_factory=list)
    quality: Dict[str, Any] = field(default_factory=dict)
    elapsed_sec: float = 0.0


def _round_ts(value: float) -> float:
    return round(float(value), 2)


def legacy_segments(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Strip to the format expected by existing modes."""
    out: List[Dict[str, Any]] = []
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        out.append({
            "start": _round_ts(seg["start"]),
            "end": _round_ts(seg["end"]),
            "text": text,
        })
    return out


def _get_audio_wav(video_path: Path, cache_dir: Optional[Path]) -> Path:
    try:
        from audio_cache import get_cached_audio_wav
        return get_cached_audio_wav(video_path, cache_dir=cache_dir)
    except ImportError:
        pass
    from asr import extract_audio_wav
    return Path(extract_audio_wav(str(video_path)))


def _video_duration_sec(video_path: Path, segments: List[Dict[str, Any]]) -> float:
    if segments:
        return max(float(s.get("end", 0.0)) for s in segments)
    try:
        from utils import get_video_duration
        return float(get_video_duration(str(video_path)))
    except Exception:
        return 0.0


def _avg_word_confidence(words: List[Dict[str, Any]]) -> Optional[float]:
    vals = [float(w["confidence"]) for w in words if w.get("confidence") is not None]
    if not vals:
        return None
    return round(sum(vals) / len(vals), 4)


def _merge_speech_ranges(
    segments: List[Dict[str, Any]],
    source: str = "faster-whisper-vad",
    gap_sec: float = 0.35,
) -> List[Dict[str, Any]]:
    if not segments:
        return []
    ordered = sorted(segments, key=lambda s: float(s["start"]))
    ranges: List[Dict[str, Any]] = []
    cur_start = float(ordered[0]["start"])
    cur_end = float(ordered[0]["end"])
    for seg in ordered[1:]:
        start = float(seg["start"])
        end = float(seg["end"])
        if start - cur_end <= gap_sec:
            cur_end = max(cur_end, end)
        else:
            ranges.append({
                "start": _round_ts(cur_start),
                "end": _round_ts(cur_end),
                "confidence": None,
                "source": source,
            })
            cur_start, cur_end = start, end
    ranges.append({
        "start": _round_ts(cur_start),
        "end": _round_ts(cur_end),
        "confidence": None,
        "source": source,
    })
    return ranges


def _faster_whisper_available() -> bool:
    try:
        import faster_whisper  # noqa: F401
        return True
    except ImportError:
        return False


def _transcribe_openai_whisper(video_path: Path, config: AsrV2Config) -> AsrV2Result:
    from asr import transcribe_video

    t0 = time.perf_counter()
    segments = transcribe_video(
        str(video_path),
        model_size=config.whisper_model,
        language=config.language,
    )
    elapsed = time.perf_counter() - t0
    legacy = legacy_segments(segments)
    if config.vad:
        logger.info("[asr_v2] --asr-vad ignored for openai-whisper backend")

    quality = {
        "engine": "openai-whisper",
        "model": config.whisper_model,
        "language": config.language,
        "language_probability": None,
        "word_timestamps": False,
        "vad_enabled": bool(config.vad),
        "segments_count": len(legacy),
        "words_count": 0,
        "avg_word_confidence": None,
        "duration_sec": _round_ts(_video_duration_sec(video_path, legacy)),
        "runtime_sec": round(elapsed, 3),
        "fallback_used": False,
        "fallback_reason": None,
        "source": "openai-whisper",
    }
    return AsrV2Result(
        segments=legacy,
        words=[],
        speech_segments=[],
        quality=quality,
        elapsed_sec=elapsed,
    )


def _transcribe_faster_whisper(video_path: Path, config: AsrV2Config) -> AsrV2Result:
    from faster_whisper import WhisperModel

    wav_path = _get_audio_wav(video_path, config.audio_cache_dir)

    device = "cuda"
    try:
        import torch
        if not torch.cuda.is_available():
            device = "cpu"
    except Exception:
        device = "cpu"

    compute_type = config.compute_type
    if device == "cpu" and compute_type == "float16":
        compute_type = "int8"

    t0 = time.perf_counter()
    model = WhisperModel(
        config.whisper_model,
        device=device,
        compute_type=compute_type,
    )

    if config.batch_size and int(config.batch_size) != 1:
        logger.info(
            "[asr_v2] batch_size=%s ignored for WhisperModel.transcribe "
            "(use BatchedInferencePipeline if needed)",
            config.batch_size,
        )

    transcribe_kwargs: Dict[str, Any] = {
        "language": config.language,
        "word_timestamps": config.word_timestamps,
        "vad_filter": config.vad,
    }
    segments_iter, info = model.transcribe(str(wav_path), **transcribe_kwargs)

    legacy: List[Dict[str, Any]] = []
    words: List[Dict[str, Any]] = []
    raw_segments: List[Dict[str, Any]] = []

    for segment_id, segment in enumerate(segments_iter):
        text = (segment.text or "").strip()
        if text:
            legacy.append({
                "start": _round_ts(segment.start),
                "end": _round_ts(segment.end),
                "text": text,
            })
        raw_segments.append({
            "start": float(segment.start),
            "end": float(segment.end),
            "text": text,
        })
        if config.word_timestamps and segment.words:
            for word in segment.words:
                word_text = (word.word or "").strip()
                if not word_text:
                    continue
                prob = getattr(word, "probability", None)
                words.append({
                    "word": word_text,
                    "start": _round_ts(word.start),
                    "end": _round_ts(word.end),
                    "confidence": round(float(prob), 4) if prob is not None else None,
                    "segment_id": segment_id,
                    "speaker": None,
                    "source": "faster-whisper",
                })

    elapsed = time.perf_counter() - t0
    speech_segments: List[Dict[str, Any]] = []
    if config.vad and raw_segments:
        speech_segments = _merge_speech_ranges(raw_segments, source="faster-whisper-vad")

    lang = getattr(info, "language", None)
    lang_prob = getattr(info, "language_probability", None)

    quality = {
        "engine": "faster-whisper",
        "model": config.whisper_model,
        "language": lang,
        "language_probability": round(float(lang_prob), 4) if lang_prob is not None else None,
        "word_timestamps": bool(config.word_timestamps),
        "vad_enabled": bool(config.vad),
        "segments_count": len(legacy),
        "words_count": len(words),
        "avg_word_confidence": _avg_word_confidence(words),
        "duration_sec": _round_ts(_video_duration_sec(video_path, legacy)),
        "runtime_sec": round(elapsed, 3),
        "fallback_used": False,
        "fallback_reason": None,
        "source": "faster-whisper",
        "device": device,
        "compute_type": compute_type,
    }

    return AsrV2Result(
        segments=legacy,
        words=words,
        speech_segments=speech_segments,
        quality=quality,
        elapsed_sec=elapsed,
    )


def run_asr_v2(video_path: Path, config: AsrV2Config) -> AsrV2Result:
    """
    Run ASR with the selected engine. May fall back to openai-whisper when allowed.
    Raises AsrV2Error on failure (no silent empty success).
    """
    engine = config.engine
    if engine not in ASR_ENGINES:
        raise AsrV2Error(f"Unknown asr engine: {engine!r}. Use one of {ASR_ENGINES}")

    if engine == "openai-whisper":
        result = _transcribe_openai_whisper(video_path, config)
        if not result.segments:
            raise AsrV2Error(f"openai-whisper returned no segments for {video_path.name}")
        return result

    if not _faster_whisper_available():
        msg = "faster-whisper is not installed. Run: pip install -U faster-whisper"
        if config.strict_real:
            raise AsrV2Error(msg)
        return _fallback_openai_from_faster(video_path, config, msg)

    try:
        result = _transcribe_faster_whisper(video_path, config)
        if not result.segments:
            raise AsrV2Error(f"faster-whisper returned no segments for {video_path.name}")
        return result
    except AsrV2Error:
        raise
    except Exception as exc:
        if config.strict_real:
            raise AsrV2Error(f"faster-whisper failed: {exc}") from exc
        return _fallback_openai_from_faster(video_path, config, str(exc))


def _fallback_openai_from_faster(
    video_path: Path,
    config: AsrV2Config,
    reason: str,
) -> AsrV2Result:
    logger.warning(
        "[asr_v2] faster-whisper unavailable (%s), falling back to openai-whisper", reason
    )
    fallback = _transcribe_openai_whisper(video_path, config)
    if not fallback.segments:
        raise AsrV2Error(
            f"faster-whisper failed ({reason}) and openai-whisper fallback produced no segments"
        )
    # openai-whisper backend does not produce word timestamps
    fallback.quality["fallback_used"] = True
    fallback.quality["fallback_reason"] = reason
    fallback.quality["engine"] = "openai-whisper"
    fallback.quality["requested_engine"] = "faster-whisper"
    fallback.quality["word_timestamps"] = False
    # If VAD was requested, approximate speech ranges from legacy segments
    if config.vad and fallback.segments:
        fallback.speech_segments = _merge_speech_ranges(
            fallback.segments,
            source="openai-whisper-segments",
        )
    return fallback


def asr_cache_sufficient(shared_dir: Path, config: AsrV2Config) -> bool:
    """Return True if cached shared ASR artifacts match the requested config."""
    if not (shared_dir / "asr_segments.json").exists():
        return False

    if config.engine == "openai-whisper" and not config.word_timestamps and not config.vad:
        return True

    quality_path = shared_dir / "asr_quality.json"
    if not quality_path.exists():
        return False

    try:
        with open(quality_path, encoding="utf-8") as f:
            cached_quality = json.load(f)
    except Exception:
        return False

    cached_engine = cached_quality.get("engine")
    if config.engine == "faster-whisper":
        if cached_quality.get("fallback_used"):
            return False
        if cached_engine != "faster-whisper":
            return False
        if cached_quality.get("model") != config.whisper_model:
            return False
        if bool(cached_quality.get("word_timestamps")) != bool(config.word_timestamps):
            return False
        if bool(cached_quality.get("vad_enabled")) != bool(config.vad):
            return False

    if config.word_timestamps and not (shared_dir / "asr_words.json").exists():
        return False

    if config.vad and not (shared_dir / "speech_segments.json").exists():
        return False

    return True


def load_asr_from_cache(shared_dir: Path) -> AsrV2Result:
    """Load ASR result from shared/ cache files."""
    with open(shared_dir / "asr_segments.json", encoding="utf-8") as f:
        segments = json.load(f)
    if not isinstance(segments, list):
        raise ValueError("asr_segments.json is not a list")

    words: List[Dict[str, Any]] = []
    words_path = shared_dir / "asr_words.json"
    if words_path.exists():
        with open(words_path, encoding="utf-8") as f:
            loaded_words = json.load(f)
            if isinstance(loaded_words, list):
                words = loaded_words

    speech_segments: List[Dict[str, Any]] = []
    speech_path = shared_dir / "speech_segments.json"
    if speech_path.exists():
        with open(speech_path, encoding="utf-8") as f:
            loaded_speech = json.load(f)
            if isinstance(loaded_speech, list):
                speech_segments = loaded_speech

    quality: Dict[str, Any] = {}
    quality_path = shared_dir / "asr_quality.json"
    if quality_path.exists():
        with open(quality_path, encoding="utf-8") as f:
            loaded_quality = json.load(f)
            if isinstance(loaded_quality, dict):
                quality = loaded_quality

    elapsed = float(quality.get("runtime_sec") or 0.0)
    return AsrV2Result(
        segments=segments,
        words=words,
        speech_segments=speech_segments,
        quality=quality,
        elapsed_sec=elapsed,
    )


def save_asr_artifacts(shared_dir: Path, result: AsrV2Result) -> None:
    """Persist shared ASR artifacts."""
    shared_dir.mkdir(parents=True, exist_ok=True)

    with open(shared_dir / "asr_segments.json", "w", encoding="utf-8") as f:
        json.dump(result.segments, f, ensure_ascii=False, indent=2)

    with open(shared_dir / "asr_quality.json", "w", encoding="utf-8") as f:
        json.dump(result.quality, f, ensure_ascii=False, indent=2)

    if result.quality.get("word_timestamps") or result.words:
        with open(shared_dir / "asr_words.json", "w", encoding="utf-8") as f:
            json.dump(result.words, f, ensure_ascii=False, indent=2)

    if result.quality.get("vad_enabled") or result.speech_segments:
        with open(shared_dir / "speech_segments.json", "w", encoding="utf-8") as f:
            json.dump(result.speech_segments, f, ensure_ascii=False, indent=2)
