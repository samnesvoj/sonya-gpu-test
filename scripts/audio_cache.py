"""
SONYA Production Audio Cache
=============================
Unified WAV extraction + in-memory audio cache for all SONYA modes.

Usage:
    from audio_cache import get_cached_audio_wav, load_full_cached_audio, get_audio_window

Flow per video:
    1. get_cached_audio_wav()  → extracts mono 16kHz WAV via ffmpeg, caches on disk
    2. load_full_cached_audio() → loads WAV once into RAM, caches in _AUDIO_LOADED_CACHE
    3. get_audio_window()       → returns numpy slice (start_sec, end_sec) from cached array

All functions are safe to call multiple times — cache lookups are O(1).
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Optional numpy / librosa ────────────────────────────────────────────────
try:
    import numpy as np
    import librosa
    _HAS_AUDIO = True
except ImportError:
    np = None  # type: ignore
    librosa = None  # type: ignore
    _HAS_AUDIO = False
    logger.warning("[audio_cache] librosa/numpy not available — audio features will be empty")

# ── Module-level caches ─────────────────────────────────────────────────────
# _AUDIO_WAV_PATH_CACHE: cache_key → Path to WAV file
# _AUDIO_LOADED_CACHE:   cache_key → (y_full: np.ndarray, sr: int)
# _AUDIO_MANIFEST_ITEMS: track all accessed items for diagnostics
_AUDIO_WAV_PATH_CACHE: Dict[str, Path] = {}
_AUDIO_LOADED_CACHE: Dict[str, Any] = {}
_AUDIO_MANIFEST_ITEMS: Dict[str, Dict] = {}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_cache_dir(cache_dir: Optional[Any] = None) -> Path:
    if cache_dir is not None:
        return Path(cache_dir)
    env_dir = os.environ.get("SONYA_AUDIO_CACHE_DIR")
    if env_dir:
        return Path(env_dir)
    return Path(tempfile.gettempdir()) / "sonya_audio_cache"


def _make_cache_key(video_path: Path, sample_rate: int) -> str:
    """Stable cache key from absolute path + file size + mtime + sample_rate."""
    try:
        stat = video_path.stat()
        raw = f"{video_path.resolve()}:{stat.st_size}:{stat.st_mtime:.3f}:{sample_rate}"
    except OSError:
        raw = f"{video_path.resolve()}:{sample_rate}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _get_ffmpeg_path() -> str:
    try:
        from utils import get_ffmpeg_path  # type: ignore
        return get_ffmpeg_path()
    except Exception:
        pass
    which = shutil.which("ffmpeg")
    return which if which else "ffmpeg"


# ── Public API ───────────────────────────────────────────────────────────────

def get_cached_audio_wav(
    video_path: "str | Path",
    cache_dir: "str | Path | None" = None,
    sample_rate: int = 16000,
    timeout_sec: int = 180,
) -> Path:
    """
    Extract mono WAV from video via ffmpeg, with file-system caching.

    Returns Path to .wav file.
    Raises RuntimeError if ffmpeg extraction fails.
    """
    vp = Path(video_path).resolve()
    if not vp.exists():
        raise FileNotFoundError(f"[audio_cache] video not found: {vp}")

    cache_key = _make_cache_key(vp, sample_rate)

    # Check in-process memory cache
    if cache_key in _AUDIO_WAV_PATH_CACHE:
        cached_wav = _AUDIO_WAV_PATH_CACHE[cache_key]
        if cached_wav.exists() and cached_wav.stat().st_size > 0:
            logger.debug("[audio_cache] reused (in-process): %s", cached_wav.name)
            _AUDIO_MANIFEST_ITEMS.setdefault(cache_key, {
                "video_path": str(vp), "wav_path": str(cached_wav),
                "sample_rate": sample_rate, "created": False,
            })["reused"] = True
            return cached_wav

    # Check disk cache
    cd = _get_cache_dir(cache_dir)
    cd.mkdir(parents=True, exist_ok=True)
    wav_path = cd / f"audio_{cache_key}.wav"

    if wav_path.exists() and wav_path.stat().st_size > 0:
        logger.info("[audio_cache] reused WAV on disk: %s", wav_path.name)
        _AUDIO_WAV_PATH_CACHE[cache_key] = wav_path
        _AUDIO_MANIFEST_ITEMS[cache_key] = {
            "video_path": str(vp), "wav_path": str(wav_path),
            "sample_rate": sample_rate, "created": False, "reused": True,
        }
        return wav_path

    # Extract via ffmpeg
    ffmpeg = _get_ffmpeg_path()
    cmd = [
        ffmpeg, "-y", "-v", "error",
        "-i", str(vp),
        "-vn", "-ac", "1", "-ar", str(sample_rate),
        "-acodec", "pcm_s16le",
        str(wav_path),
    ]
    logger.info("[audio_cache] extracting WAV: %s → %s", vp.name, wav_path.name)
    try:
        result = subprocess.run(
            cmd, timeout=timeout_sec, capture_output=True,
        )
        if result.returncode != 0 or not wav_path.exists() or wav_path.stat().st_size == 0:
            stderr_short = result.stderr.decode(errors="replace")[:300]
            raise RuntimeError(f"ffmpeg rc={result.returncode}: {stderr_short}")
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"[audio_cache] ffmpeg timed out after {timeout_sec}s")

    size_mb = round(wav_path.stat().st_size / 1024 / 1024, 2)
    logger.info("[audio_cache] audio_cache_created: %s (%.1f MB)", wav_path.name, size_mb)

    _AUDIO_WAV_PATH_CACHE[cache_key] = wav_path
    _AUDIO_MANIFEST_ITEMS[cache_key] = {
        "video_path": str(vp), "wav_path": str(wav_path),
        "sample_rate": sample_rate, "created": True, "reused": False,
        "size_mb": size_mb,
    }
    return wav_path


def load_full_cached_audio(
    video_path: "str | Path",
    sample_rate: int = 16000,
    cache_dir: "str | Path | None" = None,
) -> Tuple[Any, int]:
    """
    Load full audio for video_path as (y_full, sr).
    Audio is extracted to WAV once, then loaded into RAM once per process.

    Returns (np.ndarray, sr) or raises RuntimeError.
    """
    if not _HAS_AUDIO:
        raise RuntimeError("[audio_cache] librosa/numpy not available")

    vp = Path(video_path).resolve()
    cache_key = _make_cache_key(vp, sample_rate)

    if cache_key in _AUDIO_LOADED_CACHE:
        y_full, loaded_sr = _AUDIO_LOADED_CACHE[cache_key]
        logger.debug("[audio_cache] in-memory cache hit for %s", vp.name)
        return y_full, loaded_sr

    wav_path = get_cached_audio_wav(vp, cache_dir=cache_dir, sample_rate=sample_rate)

    logger.info("[audio_cache] loading WAV into memory: %s", wav_path.name)
    y_full, loaded_sr = librosa.load(str(wav_path), sr=sample_rate, mono=True)

    dur_s = len(y_full) / max(loaded_sr, 1)
    logger.info(
        "[audio_cache] audio_loaded_once: %d samples (%.1fs) @ %d Hz | source=cached_wav",
        len(y_full), dur_s, loaded_sr,
    )

    _AUDIO_LOADED_CACHE[cache_key] = (y_full, loaded_sr)

    # Update manifest
    item = _AUDIO_MANIFEST_ITEMS.setdefault(cache_key, {
        "video_path": str(vp), "wav_path": str(wav_path),
        "sample_rate": sample_rate,
    })
    item["loaded_once"] = True
    item["duration_sec"] = round(dur_s, 2)

    return y_full, loaded_sr


def get_audio_window(
    video_path: "str | Path",
    start_sec: float,
    end_sec: float,
    sample_rate: int = 16000,
    cache_dir: "str | Path | None" = None,
) -> Tuple[Any, int]:
    """
    Return a numpy slice of the full audio for [start_sec, end_sec).

    Uses load_full_cached_audio() — loads full audio once, then slices in-memory.
    Returns (y_window: np.ndarray, sr: int).
    """
    y_full, sr = load_full_cached_audio(video_path, sample_rate=sample_rate, cache_dir=cache_dir)

    total_samples = len(y_full)
    start_sample = int(max(0.0, start_sec) * sr)
    end_sample = int(max(start_sec, end_sec) * sr)
    # Clamp
    start_sample = min(start_sample, total_samples)
    end_sample = min(end_sample, total_samples)

    return y_full[start_sample:end_sample], sr


def get_audio_cache_manifest() -> Dict:
    """
    Return diagnostic manifest of all audio cache activity in this process.

    Schema:
        {
            "enabled": true,
            "items": [
                {
                    "video_path": "...",
                    "wav_path": "...",
                    "sample_rate": 16000,
                    "created": true/false,
                    "reused": true/false,
                    "loaded_once": true/false,
                    "duration_sec": 1800.0,
                    "size_mb": 56.2
                }
            ]
        }
    """
    items = []
    for key, item in _AUDIO_MANIFEST_ITEMS.items():
        entry: Dict = {
            "video_path": item.get("video_path", ""),
            "wav_path": item.get("wav_path", ""),
            "sample_rate": item.get("sample_rate", 16000),
            "created": bool(item.get("created", False)),
            "reused": bool(item.get("reused", False)),
            "loaded_once": bool(item.get("loaded_once", False)),
            "duration_sec": item.get("duration_sec"),
            "size_mb": item.get("size_mb"),
        }
        # Fill size_mb from disk if not set
        if entry["size_mb"] is None and entry["wav_path"]:
            try:
                entry["size_mb"] = round(Path(entry["wav_path"]).stat().st_size / 1024 / 1024, 2)
            except OSError:
                pass
        items.append(entry)

    return {"enabled": True, "items": items}


def clear_audio_cache(memory_only: bool = True) -> None:
    """
    Clear audio cache.

    memory_only=True (default): clears only in-process RAM cache.
    memory_only=False: also removes WAV files from disk.
    """
    _AUDIO_LOADED_CACHE.clear()
    if not memory_only:
        for wav_path in list(_AUDIO_WAV_PATH_CACHE.values()):
            try:
                Path(wav_path).unlink(missing_ok=True)
                logger.info("[audio_cache] removed disk WAV: %s", wav_path)
            except Exception as exc:
                logger.warning("[audio_cache] could not remove %s: %s", wav_path, exc)
        _AUDIO_WAV_PATH_CACHE.clear()
        _AUDIO_MANIFEST_ITEMS.clear()
    logger.info("[audio_cache] cache cleared (memory_only=%s)", memory_only)
