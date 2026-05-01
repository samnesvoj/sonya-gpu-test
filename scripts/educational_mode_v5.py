"""
EDUCATIONAL MODE v5.0 — Production-ready intelligent pipeline

Architecture (v5.0 — layered, signal-driven):

    extract_audio_global()              ← AudioGlobalData (one-shot)
         ↓
    build_topic_windows()               ← multi-scale window generation per topic
         ↓
    collect_window_context()            ← transcript + context + visual cache lookup
         ↓
    extract_visual_features_for_window()← base_analysis → VisualWindowFeatures
    compute_audio_features_from_array() ← raw audio → AudioWindowFeatures
    analyze_semantic_window()           ← markers → type → SemanticWindowFeatures
         ↓
    compute_educational_score_v5()      ← eligibility gates → multimodal core →
                                           type-aware bonuses → penalties
         ↓
    deduplicate_windows()               ← NMS on overlap × content similarity
         ↓
    select_windows_with_topic_coverage()← coverage quota: explanation + example/formula + summary
         ↓
    refine_clip_boundaries()            ← snap to sentence/pause/silence boundary
         ↓
    build_educational_output()          ← result + chapters + diagnostics

Public API:
    run_educational_mode_v5()
    export_educational_result()

Changelog v5.0:
- AudioGlobalData dataclass replaces bare tuple return
- AudioWindowFeatures: 11 fields (was 3)
- VisualWindowFeatures: full YOLO8X-derived feature set + derived summary scores
- extract_visual_features_for_window(): real base_analysis adapter with graceful fallback
- Semantic layer split: extract_semantic_markers / classify_segment_type_semantic / score_semantic_window
- SemanticWindowFeatures: 13 fields including pedagogical_density + confusion_risk
- compute_educational_score_v5(): 4-stage pipeline (eligibility → core → type-aware → penalties)
- WindowScoreResult: structured score with per-reason breakdown
- deduplicate_windows(): overlap-ratio + transcript similarity NMS
- select_windows_with_topic_coverage(): quota-based selector (core + support + summary roles)
- refine_clip_boundaries(): pause-aware boundary snapping
- debug_scores.json + window_diagnostics.jsonl + selection_trace.json export
- EDUCATIONAL_SEGMENT_TYPES extended with demo + comparison
"""

from __future__ import annotations

import io
import json
import logging
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Shared audio cache (scripts/audio_cache.py) ──────────────────────────────
_edu_scripts_dir = Path(__file__).resolve().parent
if str(_edu_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_edu_scripts_dir))

try:
    from audio_cache import load_full_cached_audio as _load_full_cached_audio
    _HAS_AUDIO_CACHE = True
except ImportError:
    _load_full_cached_audio = None  # type: ignore
    _HAS_AUDIO_CACHE = False


# =============================================================================
# SEGMENT TYPE REGISTRY
# =============================================================================

EDUCATIONAL_SEGMENT_TYPES: Dict[str, Dict] = {
    "definition": {
        "weight": 1.20, "min_duration": 25, "max_duration": 90,
        "title_template": "Определение: {topic}",
        "priority_fields": ["topic_relevance", "concept_density", "text_readability"],
    },
    "explanation": {
        "weight": 1.15, "min_duration": 45, "max_duration": 120,
        "title_template": "Объяснение: {topic}",
        "priority_fields": ["explanation_quality", "speech_continuity", "pedagogical_density"],
    },
    "example": {
        "weight": 1.10, "min_duration": 30, "max_duration": 90,
        "title_template": "Пример: {topic}",
        "priority_fields": ["has_example", "object_demo_ratio", "contrast_markers"],
    },
    "step_by_step": {
        "weight": 1.25, "min_duration": 60, "max_duration": 180,
        "title_template": "Пошагово: {topic}",
        "priority_fields": ["has_steps", "speech_continuity", "hand_activity_mean", "sequence_markers"],
    },
    "summary": {
        "weight": 1.00, "min_duration": 25, "max_duration": 60,
        "title_template": "Итоги: {topic}",
        "priority_fields": ["has_takeaway", "topic_keywords_overlap", "redundancy_penalty"],
    },
    "formula": {
        "weight": 1.30, "min_duration": 20, "max_duration": 60,
        "title_template": "Формула/Правило: {topic}",
        "priority_fields": ["has_formula", "text_region_ratio_mean", "dense_text_likelihood_mean"],
    },
    "demo": {
        "weight": 1.20, "min_duration": 45, "max_duration": 150,
        "title_template": "Демонстрация: {topic}",
        "priority_fields": ["desk_demo_presence_ratio", "hand_activity_mean", "object_demo_ratio"],
    },
    "comparison": {
        "weight": 1.10, "min_duration": 30, "max_duration": 90,
        "title_template": "Сравнение: {topic}",
        "priority_fields": ["contrast_markers", "insight_value", "structure_score"],
    },
    "off_topic": {
        "weight": 0.30, "min_duration": 0, "max_duration": 30,
        "title_template": "Отступление",
        "priority_fields": [],
    },
}

AUDIO_PROFILES: Dict[str, Dict] = {
    "lecture": {
        "optimal_zcr_range": (0.05, 0.12),
        "optimal_silence_ratio": (0.10, 0.30),
        "min_speech_clarity": 0.60,
        "description": "Академическая лекция",
    },
    "podcast": {
        "optimal_zcr_range": (0.08, 0.15),
        "optimal_silence_ratio": (0.05, 0.20),
        "min_speech_clarity": 0.50,
        "description": "Подкаст/беседа",
    },
    "tutorial": {
        "optimal_zcr_range": (0.06, 0.13),
        "optimal_silence_ratio": (0.15, 0.35),
        "min_speech_clarity": 0.65,
        "description": "Пошаговый туториал",
    },
}

MODE_CONFIGS: Dict[str, Dict] = {
    "educational": {
        "weights": {"semantic": 0.40, "audio": 0.22, "visual": 0.18, "pedagogy": 0.20},
        "window_size_range": (45, 120),
        "min_clip_duration": 25,
        "max_clip_duration": 180,
        "preferred_duration": 75,
        "requires_topic_segmentation": True,
        "export_chapters": True,
        "title_style": "educational",
        "reasons_focus": ["explanation", "clarity", "structure", "slides"],
    },
    "viral": {
        "weights": {"semantic": 0.25, "audio": 0.30, "visual": 0.45, "pedagogy": 0.00},
        "window_size_range": (10, 45),
        "min_clip_duration": 5,
        "max_clip_duration": 60,
        "preferred_duration": 25,
        "requires_topic_segmentation": False,
        "export_chapters": False,
        "title_style": "viral",
        "reasons_focus": ["emotion", "motion", "hook", "laughter"],
    },
}


# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class AudioGlobalData:
    samples: np.ndarray
    sample_rate: int
    duration_sec: float
    source_path: str
    extraction_ok: bool
    error: Optional[str] = None


@dataclass
class AudioWindowFeatures:
    speech_clarity: float = 0.5
    speech_rate_score: float = 0.5
    silence_ratio: float = 0.2
    voice_activity_ratio: float = 0.5
    speech_continuity: float = 0.5
    long_pause_ratio: float = 0.2
    pause_count: int = 0
    prosody_stability: float = 0.5
    energy_mean: float = 0.5
    energy_std: float = 0.2
    audio_quality_flag: str = "ok"  # ok / noisy / weak / clipped


@dataclass
class VisualWindowFeatures:
    # Scene presence ratios
    person_present_ratio: float = 0.5
    single_speaker_ratio: float = 0.5
    multi_person_ratio: float = 0.1
    screen_presence_ratio: float = 0.3
    whiteboard_presence_ratio: float = 0.1
    desk_demo_presence_ratio: float = 0.1
    object_demo_ratio: float = 0.1
    hand_activity_mean: float = 0.2
    # Text/OCR signals
    text_region_ratio_mean: float = 0.2
    dense_text_likelihood_mean: float = 0.3
    # Camera/motion
    scene_stability: float = 0.7
    scene_change_rate: float = 0.05
    camera_motion_mean: float = 0.15
    # Quality flags
    visual_clutter_score: float = 0.3
    blur_ratio: float = 0.1
    low_light_ratio: float = 0.05
    # Derived summary scores (computed by adapter)
    visual_readability: float = 0.6
    instructional_visual_value: float = 0.5
    composition_score: float = 0.6
    visual_stability_score: float = 0.7


@dataclass
class SemanticWindowFeatures:
    segment_type: str = "explanation"
    explanation_quality: float = 0.5
    insight_value: float = 0.5
    structure_score: float = 0.5
    concept_density: float = 0.4
    pedagogical_density: float = 0.4
    topic_relevance: float = 0.5
    redundancy_penalty: float = 0.0
    has_takeaway: bool = False
    has_formula: bool = False
    has_example: bool = False
    has_steps: bool = False
    confusion_risk: float = 0.2
    # Marker counts (used downstream)
    definition_markers: int = 0
    example_markers: int = 0
    sequence_markers: int = 0
    summary_markers: int = 0
    formula_markers: int = 0
    contrast_markers: int = 0
    instruction_markers: int = 0
    topic_keywords_overlap: float = 0.0
    lexical_density: float = 0.5


@dataclass
class WindowScoreResult:
    final_score: float = 0.0
    semantic_core: float = 0.0
    audio_core: float = 0.0
    visual_core: float = 0.0
    pedagogy_core: float = 0.0
    penalties: Dict[str, float] = field(default_factory=dict)
    bonuses: Dict[str, float] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)
    decision_flags: List[str] = field(default_factory=list)
    eligible: bool = True
    reject_reason: Optional[str] = None


@dataclass
class WindowContext:
    start: float
    end: float
    duration: float
    transcript: str
    context_before: str
    context_after: str
    topic_id: int
    topic_title: str
    topic_keywords: List[str] = field(default_factory=list)


# =============================================================================
# LAYER 1 — AUDIO
# =============================================================================

def extract_audio_global(video_path: str) -> AudioGlobalData:
    """
    Load audio once for the full video. Returns AudioGlobalData.

    Uses shared audio_cache (scripts/audio_cache.py):
    - ffmpeg extracts mono 16kHz WAV once (no PySoundFile/audioread on mp4)
    - librosa loads the WAV once per process
    Falls back to direct librosa.load(video_path) if audio_cache is unavailable.
    """
    try:
        if _HAS_AUDIO_CACHE and _load_full_cached_audio is not None:
            logger.info("Extracting full audio via shared audio_cache...")
            y, sr = _load_full_cached_audio(video_path, sample_rate=16000)
        else:
            import librosa as _librosa
            logger.info("Extracting full audio via direct librosa.load (audio_cache unavailable)...")
            y, sr = _librosa.load(video_path, sr=16000, mono=True)

        duration_sec = float(len(y) / max(sr, 1))
        logger.info("Audio extracted from cached WAV: %.1fs @ %d Hz", duration_sec, sr)
        return AudioGlobalData(
            samples=y,
            sample_rate=sr,
            duration_sec=duration_sec,
            source_path=video_path,
            extraction_ok=True,
        )
    except Exception as exc:
        logger.error(f"Audio extraction failed: {exc}")
        return AudioGlobalData(
            samples=np.array([]),
            sample_rate=16000,
            duration_sec=0.0,
            source_path=video_path,
            extraction_ok=False,
            error=str(exc),
        )


def compute_audio_features_from_array(
    y: np.ndarray,
    sr: int,
    start_sec: float,
    end_sec: float,
    profile: str = "lecture",
) -> AudioWindowFeatures:
    """Compute rich audio features for a time window from a pre-loaded array."""
    try:
        import librosa

        start_sample = int(start_sec * sr)
        end_sample = int(end_sec * sr)
        win = y[start_sample:end_sample]

        if len(win) < sr * 0.5:
            return AudioWindowFeatures(audio_quality_flag="weak")

        profile_cfg = AUDIO_PROFILES.get(profile, AUDIO_PROFILES["lecture"])

        # --- speech clarity ---
        spectral_flatness = librosa.feature.spectral_flatness(y=win)[0]
        speech_clarity = float(np.clip(1.0 - np.mean(spectral_flatness), 0, 1))

        # --- speech rate via ZCR ---
        zcr = librosa.feature.zero_crossing_rate(win)[0]
        zcr_mean = float(np.mean(zcr))
        zcr_min, zcr_max = profile_cfg["optimal_zcr_range"]
        if zcr_mean < zcr_min:
            speech_rate_score = float(np.clip(zcr_mean / zcr_min * 0.8, 0, 1))
        elif zcr_mean > zcr_max:
            speech_rate_score = float(np.clip(1.0 - (zcr_mean - zcr_max) / zcr_max * 0.3, 0, 1))
        else:
            speech_rate_score = 1.0

        # --- energy / RMS ---
        frame_length = 2048
        hop_length = 512
        rms = librosa.feature.rms(y=win, frame_length=frame_length, hop_length=hop_length)[0]
        energy_mean = float(np.mean(rms))
        energy_std = float(np.std(rms))

        # --- silence and pauses ---
        threshold_rms = max(0.005, energy_mean * 0.15)
        silent_frames = rms < threshold_rms
        silence_ratio = float(np.mean(silent_frames))

        silence_min, silence_max = profile_cfg["optimal_silence_ratio"]

        # count pause runs (consecutive silent frames)
        pause_count = 0
        long_pause_frames = 0
        in_pause = False
        pause_run = 0
        long_pause_threshold_frames = int(0.5 * sr / hop_length)  # 0.5s
        for sf in silent_frames:
            if sf:
                in_pause = True
                pause_run += 1
                if pause_run == 1:
                    pause_count += 1
                if pause_run >= long_pause_threshold_frames:
                    long_pause_frames += 1
            else:
                in_pause = False
                pause_run = 0
        long_pause_ratio = float(long_pause_frames / max(len(silent_frames), 1))

        # --- voice activity ratio ---
        voice_activity_ratio = 1.0 - silence_ratio

        # --- speech continuity: penalise many short pauses ---
        if pause_count > 0:
            avg_words_between_pauses = voice_activity_ratio / (pause_count / max(len(silent_frames), 1) + 1e-6)
            speech_continuity = float(np.clip(avg_words_between_pauses / 50.0, 0, 1))
        else:
            speech_continuity = voice_activity_ratio

        # --- prosody stability: energy variance relative to mean ---
        prosody_stability = float(np.clip(1.0 - energy_std / (energy_mean + 1e-6), 0, 1))

        # --- audio quality flag ---
        if energy_mean < 0.005:
            audio_quality_flag = "weak"
        elif energy_std / (energy_mean + 1e-6) > 2.0:
            audio_quality_flag = "clipped"
        elif speech_clarity < 0.35:
            audio_quality_flag = "noisy"
        else:
            audio_quality_flag = "ok"

        return AudioWindowFeatures(
            speech_clarity=speech_clarity,
            speech_rate_score=speech_rate_score,
            silence_ratio=silence_ratio,
            voice_activity_ratio=voice_activity_ratio,
            speech_continuity=speech_continuity,
            long_pause_ratio=long_pause_ratio,
            pause_count=pause_count,
            prosody_stability=prosody_stability,
            energy_mean=float(np.clip(energy_mean * 20, 0, 1)),
            energy_std=float(np.clip(energy_std * 20, 0, 1)),
            audio_quality_flag=audio_quality_flag,
        )

    except Exception as exc:
        logger.error(f"Audio feature extraction error: {exc}")
        return AudioWindowFeatures(audio_quality_flag="noisy")


# =============================================================================
# LAYER 2 — VISUAL ADAPTER
# =============================================================================

def extract_visual_features_for_window(
    base_analysis: Dict,
    start_sec: float,
    end_sec: float,
) -> VisualWindowFeatures:
    """
    Adapter: base_analysis (populated by YOLO8X / shot detector) → VisualWindowFeatures.

    Lookup order:
      1. window_visual_cache keyed by "start-end"
      2. Aggregation over frame_events in [start, end]
      3. Nearest shot_segment overlap
      4. global_visual_stats as last resort
      5. Conservative defaults (all neutral) if nothing available
    """
    if not base_analysis:
        return _visual_defaults()

    window_key = f"{start_sec:.1f}-{end_sec:.1f}"

    # ── path 1: pre-computed window cache ──────────────────────────────────
    cache = base_analysis.get("window_visual_cache", {})
    if window_key in cache:
        return _build_visual_features_from_dict(cache[window_key])

    # try nearby keys (±2s tolerance)
    for key, val in cache.items():
        try:
            ks, ke = map(float, key.split("-"))
            if abs(ks - start_sec) <= 2.0 and abs(ke - end_sec) <= 2.0:
                return _build_visual_features_from_dict(val)
        except ValueError:
            pass

    # ── path 2: aggregate frame_events ─────────────────────────────────────
    frame_events = base_analysis.get("frame_events", [])
    window_frames = [e for e in frame_events if start_sec <= e.get("ts", -1) <= end_sec]
    if window_frames:
        return _aggregate_frame_events(window_frames, end_sec - start_sec)

    # ── path 3: shot_segments overlap ──────────────────────────────────────
    shots = base_analysis.get("shot_segments", [])
    overlap_shots = [
        s for s in shots
        if s.get("start", 0) < end_sec and s.get("end", 0) > start_sec
    ]
    if overlap_shots:
        return _aggregate_shot_segments(overlap_shots)

    # ── path 4: global_visual_stats ────────────────────────────────────────
    global_stats = base_analysis.get("global_visual_stats", {})
    if global_stats:
        return _build_visual_features_from_dict(global_stats)

    # ── path 5 (v5.1): benchmark-style YOLO detections ────────────────────
    # Detections formatted as [{"timestamp_sec", "person_count",
    # "objects", "confidence_max"}, ...]
    detections = base_analysis.get("detections", [])
    if detections and isinstance(detections, list) and detections:
        first = detections[0] if isinstance(detections[0], dict) else {}
        if "timestamp_sec" in first or "person_count" in first:
            in_window = [
                d for d in detections
                if start_sec <= float(d.get("timestamp_sec", d.get("timestamp", -1))) <= end_sec
            ]
            if not in_window:
                mid = (start_sec + end_sec) / 2.0
                in_window = [min(
                    detections,
                    key=lambda d: abs(float(d.get("timestamp_sec", d.get("timestamp", 0))) - mid),
                )]
            person_ratio = float(np.mean([1.0 if d.get("person_count", 0) > 0 else 0.0 for d in in_window]))
            multi_person_ratio = float(np.mean([1.0 if d.get("person_count", 0) > 1 else 0.0 for d in in_window]))
            obj_density = float(np.mean([len(d.get("objects", []) or []) for d in in_window]) / 5.0)
            obj_density = float(np.clip(obj_density, 0.0, 1.0))
            conf_mean = float(np.mean([d.get("confidence_max", 0.0) for d in in_window]))
            if len(in_window) >= 2:
                confs = [d.get("confidence_max", 0.0) for d in in_window]
                scene_change_rate = float(np.clip(np.mean(np.abs(np.diff(confs))) * 2.0, 0.0, 1.0))
            else:
                scene_change_rate = 0.05

            derived = {
                "person_present_ratio": person_ratio,
                "single_speaker_ratio": max(0.0, person_ratio - multi_person_ratio),
                "multi_person_ratio": multi_person_ratio,
                "screen_presence_ratio": obj_density * 0.5,
                "whiteboard_presence_ratio": 0.1,
                "desk_demo_presence_ratio": obj_density * 0.3,
                "object_demo_ratio": obj_density,
                "hand_activity_mean": 0.2,
                "text_region_ratio_mean": 0.2,
                "dense_text_likelihood_mean": 0.3,
                "scene_stability": max(0.0, 1.0 - scene_change_rate),
                "scene_change_rate": scene_change_rate,
                "camera_motion_mean": min(scene_change_rate * 0.8, 0.5),
                "visual_clutter_score": min(obj_density * 0.8, 0.7),
                "blur_ratio": 0.1,
                "low_light_ratio": 0.05,
            }
            vf = _build_visual_features_from_dict(derived)
            return vf

    # ── path 6: conservative defaults ──────────────────────────────────────
    return _visual_defaults()


def _visual_defaults() -> VisualWindowFeatures:
    """Neutral/conservative defaults when no visual data is available."""
    vf = VisualWindowFeatures()
    vf.visual_readability = _derive_readability(vf)
    vf.instructional_visual_value = _derive_instructional_value(vf)
    vf.composition_score = _derive_composition(vf)
    vf.visual_stability_score = _derive_stability(vf)
    return vf


def _build_visual_features_from_dict(d: Dict) -> VisualWindowFeatures:
    vf = VisualWindowFeatures(
        person_present_ratio=float(d.get("person_present_ratio", 0.5)),
        single_speaker_ratio=float(d.get("single_speaker_ratio", 0.5)),
        multi_person_ratio=float(d.get("multi_person_ratio", 0.1)),
        screen_presence_ratio=float(d.get("screen_presence_ratio", 0.3)),
        whiteboard_presence_ratio=float(d.get("whiteboard_presence_ratio", 0.1)),
        desk_demo_presence_ratio=float(d.get("desk_demo_presence_ratio", 0.1)),
        object_demo_ratio=float(d.get("object_demo_ratio", 0.1)),
        hand_activity_mean=float(d.get("hand_activity_mean", 0.2)),
        text_region_ratio_mean=float(d.get("text_region_ratio_mean", 0.2)),
        dense_text_likelihood_mean=float(d.get("dense_text_likelihood_mean", 0.3)),
        scene_stability=float(d.get("scene_stability", 0.7)),
        scene_change_rate=float(d.get("scene_change_rate", 0.05)),
        camera_motion_mean=float(d.get("camera_motion_mean", 0.15)),
        visual_clutter_score=float(d.get("visual_clutter_score", 0.3)),
        blur_ratio=float(d.get("blur_ratio", 0.1)),
        low_light_ratio=float(d.get("low_light_ratio", 0.05)),
    )
    # derived scores — either take from cache or compute
    vf.visual_readability = float(d.get("visual_readability", _derive_readability(vf)))
    vf.instructional_visual_value = float(d.get("instructional_visual_value", _derive_instructional_value(vf)))
    vf.composition_score = float(d.get("composition_score", _derive_composition(vf)))
    vf.visual_stability_score = float(d.get("visual_stability_score", _derive_stability(vf)))
    return vf


def _aggregate_frame_events(frames: List[Dict], duration: float) -> VisualWindowFeatures:
    def _ratio(key: str) -> float:
        vals = [f.get("scene_tags", {}).get(key, 0) for f in frames]
        return float(np.mean(vals)) if vals else 0.0

    def _ocr(key: str) -> float:
        vals = [f.get("ocr_proxy", {}).get(key, 0) for f in frames]
        return float(np.mean(vals)) if vals else 0.0

    person_ratio = float(np.mean([
        1.0 if any(d.get("label") == "person" for d in f.get("detections", [])) else 0.0
        for f in frames
    ]))

    vf = VisualWindowFeatures(
        person_present_ratio=person_ratio,
        single_speaker_ratio=_ratio("single_speaker"),
        multi_person_ratio=_ratio("multi_person"),
        screen_presence_ratio=_ratio("screen_present"),
        whiteboard_presence_ratio=_ratio("whiteboard_present"),
        desk_demo_presence_ratio=_ratio("desk_demo_present"),
        object_demo_ratio=float(np.mean([
            1.0 if any(d.get("label") in {"book", "keyboard", "mouse", "laptop"} for d in f.get("detections", [])) else 0.0
            for f in frames
        ])),
        hand_activity_mean=_ratio("hand_activity"),
        text_region_ratio_mean=_ocr("text_region_ratio"),
        dense_text_likelihood_mean=_ocr("dense_text_likelihood"),
        scene_stability=1.0 - _ratio("visual_motion"),
        scene_change_rate=_ratio("visual_motion") * 0.1,
        camera_motion_mean=_ratio("visual_motion") * 0.5,
        visual_clutter_score=max(0, _ratio("multi_person") * 0.5 + _ratio("visual_motion") * 0.3),
        blur_ratio=0.0,
        low_light_ratio=0.0,
    )
    vf.visual_readability = _derive_readability(vf)
    vf.instructional_visual_value = _derive_instructional_value(vf)
    vf.composition_score = _derive_composition(vf)
    vf.visual_stability_score = _derive_stability(vf)
    return vf


def _aggregate_shot_segments(shots: List[Dict]) -> VisualWindowFeatures:
    def _mean(key: str) -> float:
        vals = [s.get(key, 0) for s in shots if key in s]
        return float(np.mean(vals)) if vals else 0.0

    vf = VisualWindowFeatures(
        screen_presence_ratio=_mean("screen_presence_ratio"),
        whiteboard_presence_ratio=_mean("board_presence_ratio"),
        desk_demo_presence_ratio=_mean("demo_presence_ratio"),
        text_region_ratio_mean=_mean("text_region_ratio_mean"),
        scene_stability=_mean("scene_stability"),
        camera_motion_mean=_mean("camera_motion_mean"),
        single_speaker_ratio=float(np.mean([1.0 if s.get("speaker_count_mode", 0) == 1 else 0.0 for s in shots])),
        multi_person_ratio=float(np.mean([1.0 if s.get("speaker_count_mode", 0) > 1 else 0.0 for s in shots])),
    )
    vf.visual_readability = _derive_readability(vf)
    vf.instructional_visual_value = _derive_instructional_value(vf)
    vf.composition_score = _derive_composition(vf)
    vf.visual_stability_score = _derive_stability(vf)
    return vf


def _derive_readability(vf: VisualWindowFeatures) -> float:
    score = (
        0.35 * vf.scene_stability
        + 0.25 * (1.0 - vf.visual_clutter_score)
        + 0.20 * (1.0 - vf.blur_ratio)
        + 0.10 * (1.0 - vf.low_light_ratio)
        + 0.10 * (1.0 - vf.camera_motion_mean)
    )
    return float(np.clip(score, 0, 1))


def _derive_instructional_value(vf: VisualWindowFeatures) -> float:
    score = (
        0.25 * vf.screen_presence_ratio
        + 0.20 * vf.single_speaker_ratio
        + 0.15 * vf.text_region_ratio_mean
        + 0.15 * vf.whiteboard_presence_ratio
        + 0.15 * vf.desk_demo_presence_ratio
        + 0.10 * vf.dense_text_likelihood_mean
    )
    return float(np.clip(score, 0, 1))


def _derive_composition(vf: VisualWindowFeatures) -> float:
    score = (
        0.40 * vf.single_speaker_ratio
        + 0.30 * vf.scene_stability
        + 0.30 * (1.0 - vf.visual_clutter_score)
    )
    return float(np.clip(score, 0, 1))


def _derive_stability(vf: VisualWindowFeatures) -> float:
    score = (
        0.50 * vf.scene_stability
        + 0.30 * (1.0 - vf.camera_motion_mean)
        + 0.20 * (1.0 - vf.scene_change_rate * 10)
    )
    return float(np.clip(score, 0, 1))


# =============================================================================
# LAYER 3 — SEMANTIC
# =============================================================================

# Russian + English keyword banks
_MARKERS = {
    "definition": [
        "определение", "definition", "это значит", "называется", "является",
        "понятие", "термин", "concept", "means", "defined as", "refers to",
    ],
    "example": [
        "например", "example", "допустим", "скажем", "к примеру",
        "рассмотрим", "возьмём", "for instance", "such as", "e.g.", "let's say",
    ],
    "sequence": [
        "во-первых", "во-вторых", "в-третьих", "сначала", "затем", "потом",
        "первый шаг", "второй шаг", "first", "second", "third", "next",
        "then", "after that", "step 1", "step 2", "шаг 1", "шаг 2",
    ],
    "summary": [
        "итак", "подведём", "в итоге", "в заключение", "таким образом",
        "summary", "in conclusion", "to summarize", "in short", "so",
        "главный вывод", "ключевая мысль",
    ],
    "formula": [
        "формула", "formula", "правило", "rule", "закон", "law",
        "равно", "equals", " = ", "уравнение", "equation", "теорема", "theorem",
    ],
    "contrast": [
        "в отличие", "тогда как", "однако", "но", "with contrast", "whereas",
        "however", "unlike", "on the other hand", "зато", "хотя",
    ],
    "takeaway": [
        "главное", "важно", "ключ", "key point", "main", "запомните",
        "remember", "crucial", "critical", "essential", "the takeaway",
    ],
    "instruction": [
        "нужно", "необходимо", "следует", "you should", "you need", "must",
        "important to", "make sure", "don't forget", "следует помнить",
    ],
}


def extract_semantic_markers(
    transcript: str,
    context_before: str = "",
    context_after: str = "",
    topic_keywords: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Pure rule-based marker extraction. No LLM dependency.

    Returns counts and derived ratios used by the classifier and scorer.
    """
    text = transcript.lower()
    words = text.split()
    word_count = max(len(words), 1)

    counts = {
        "definition_markers": sum(1 for kw in _MARKERS["definition"] if kw in text),
        "example_markers": sum(1 for kw in _MARKERS["example"] if kw in text),
        "sequence_markers": sum(1 for kw in _MARKERS["sequence"] if kw in text),
        "summary_markers": sum(1 for kw in _MARKERS["summary"] if kw in text),
        "formula_markers": sum(1 for kw in _MARKERS["formula"] if kw in text),
        "contrast_markers": sum(1 for kw in _MARKERS["contrast"] if kw in text),
        "takeaway_markers": sum(1 for kw in _MARKERS["takeaway"] if kw in text),
        "instruction_markers": sum(1 for kw in _MARKERS["instruction"] if kw in text),
    }

    # topic keyword overlap
    topic_keywords_overlap = 0.0
    if topic_keywords:
        hits = sum(1 for kw in topic_keywords if kw.lower() in text)
        topic_keywords_overlap = float(hits / max(len(topic_keywords), 1))

    # lexical density: unique content words / total words
    stop_words = {"и", "в", "на", "с", "по", "к", "за", "из", "не", "а", "the", "a", "of", "in", "to", "and", "is", "it"}
    content_words = [w for w in words if w not in stop_words and len(w) > 2]
    unique_content = set(content_words)
    lexical_density = float(len(unique_content) / max(len(content_words), 1))

    # concept density: marker hits per 100 words
    total_marker_hits = sum(counts.values())
    concept_density = float(min(total_marker_hits / (word_count / 100.0 + 1e-6), 1.0))

    # repetition: how many words appear 3+ times
    from collections import Counter
    word_freq = Counter(words)
    repeated = sum(1 for w, c in word_freq.items() if c >= 3 and w not in stop_words)
    repetition_ratio = float(repeated / max(len(unique_content), 1))

    return {
        **counts,
        "topic_keywords_overlap": topic_keywords_overlap,
        "lexical_density": lexical_density,
        "concept_density": concept_density,
        "repetition_ratio": repetition_ratio,
        "word_count": word_count,
    }


def classify_segment_type_semantic(
    markers: Dict[str, Any],
    transcript: str,
) -> Tuple[str, float]:
    """
    Determine segment type from markers.

    Returns (segment_type, confidence 0-1).
    """
    scores = {
        "formula": markers["formula_markers"] * 2.0 + (0.5 if "=" in transcript else 0),
        "step_by_step": markers["sequence_markers"] * 1.5 + markers["instruction_markers"] * 0.5,
        "definition": markers["definition_markers"] * 2.0,
        "example": markers["example_markers"] * 1.8,
        "summary": markers["summary_markers"] * 1.5 + markers["takeaway_markers"] * 0.5,
        "comparison": markers["contrast_markers"] * 1.8,
        "explanation": markers["lexical_density"] * 2.0,
        "off_topic": 0.1,
    }

    # penalise off_topic if there are any meaningful markers
    if sum(v for k, v in scores.items() if k != "off_topic") > 0.5:
        scores["off_topic"] = 0.0

    best_type = max(scores, key=lambda k: scores[k])
    best_score = scores[best_type]

    # fallback to explanation if nothing stands out
    if best_score < 0.3:
        best_type = "explanation"
        best_score = 0.3

    confidence = float(np.clip(best_score / (sum(scores.values()) + 1e-6), 0, 1))
    return best_type, confidence


def score_semantic_window(
    markers: Dict[str, Any],
    transcript: str,
    segment_type: str,
    topic_relevance: float = 0.5,
    llm_result: Optional[Dict] = None,
) -> SemanticWindowFeatures:
    """
    Build SemanticWindowFeatures from markers + optional LLM enhancement.
    LLM result is used as optional overrides, not as the sole source of truth.
    """
    word_count = markers.get("word_count", max(len(transcript.split()), 1))

    # explanation quality — length + lexical richness + structure
    explanation_quality = float(np.clip(
        0.35 * min(word_count / 120.0, 1.0)
        + 0.35 * markers["lexical_density"]
        + 0.20 * min(markers["sequence_markers"] / 3.0, 1.0)
        + 0.10 * min(markers["definition_markers"] / 2.0, 1.0),
        0, 1,
    ))

    # insight_value — unique content + concept density + topic overlap
    insight_value = float(np.clip(
        0.40 * markers["concept_density"]
        + 0.35 * markers["topic_keywords_overlap"]
        + 0.25 * markers["lexical_density"],
        0, 1,
    ))

    # structure_score
    structure_score = float(np.clip(
        0.50 * min(markers["sequence_markers"] / 3.0, 1.0)
        + 0.30 * min((markers["summary_markers"] + markers["takeaway_markers"]) / 2.0, 1.0)
        + 0.20 * min(markers["definition_markers"] / 2.0, 1.0),
        0, 1,
    ))

    pedagogical_density = float(np.clip(
        (markers["example_markers"] + markers["formula_markers"] +
         markers["instruction_markers"] + markers["sequence_markers"]) / (word_count / 50.0 + 1.0),
        0, 1,
    ))

    confusion_risk = float(np.clip(
        markers.get("repetition_ratio", 0) * 0.4
        + (1.0 - markers["lexical_density"]) * 0.3
        + max(0, 0.3 - markers["concept_density"]) * 0.3,
        0, 1,
    ))

    redundancy_penalty = float(np.clip(markers.get("repetition_ratio", 0) * 0.5, 0, 0.4))

    # boolean flags
    has_takeaway = markers["takeaway_markers"] > 0 or markers["summary_markers"] > 0
    has_formula = markers["formula_markers"] > 0 or "=" in transcript
    has_example = markers["example_markers"] > 0
    has_steps = markers["sequence_markers"] >= 2

    # apply LLM overrides if provided and plausible
    if llm_result and isinstance(llm_result, dict):
        def _llm_float(key: str, default: float) -> float:
            val = llm_result.get(key)
            if isinstance(val, (int, float)) and 0 <= val <= 1:
                return float(val)
            return default

        explanation_quality = _llm_float("explanation_quality", explanation_quality)
        insight_value = _llm_float("insight_value", insight_value)
        structure_score = _llm_float("structure_score", structure_score)
        if isinstance(llm_result.get("has_takeaway"), bool):
            has_takeaway = llm_result["has_takeaway"]
        if isinstance(llm_result.get("has_formula"), bool):
            has_formula = llm_result["has_formula"]

    return SemanticWindowFeatures(
        segment_type=segment_type,
        explanation_quality=explanation_quality,
        insight_value=insight_value,
        structure_score=structure_score,
        concept_density=markers["concept_density"],
        pedagogical_density=pedagogical_density,
        topic_relevance=topic_relevance,
        redundancy_penalty=redundancy_penalty,
        has_takeaway=has_takeaway,
        has_formula=has_formula,
        has_example=has_example,
        has_steps=has_steps,
        confusion_risk=confusion_risk,
        definition_markers=markers["definition_markers"],
        example_markers=markers["example_markers"],
        sequence_markers=markers["sequence_markers"],
        summary_markers=markers["summary_markers"],
        formula_markers=markers["formula_markers"],
        contrast_markers=markers["contrast_markers"],
        instruction_markers=markers["instruction_markers"],
        topic_keywords_overlap=markers["topic_keywords_overlap"],
        lexical_density=markers["lexical_density"],
    )


def analyze_semantic_window(
    ctx: WindowContext,
    llm_model: Optional[Any] = None,
) -> SemanticWindowFeatures:
    """
    Full semantic analysis: markers → type → features.
    llm_model is called if provided, used as optional enhancer.
    """
    if not ctx.transcript or len(ctx.transcript.strip()) < 20:
        return SemanticWindowFeatures(
            segment_type="off_topic",
            explanation_quality=0.2,
            insight_value=0.2,
            topic_relevance=0.1,
            confusion_risk=0.7,
        )

    markers = extract_semantic_markers(
        ctx.transcript,
        ctx.context_before,
        ctx.context_after,
        ctx.topic_keywords,
    )

    segment_type, _confidence = classify_segment_type_semantic(markers, ctx.transcript)

    # optional LLM call
    llm_result = None
    if llm_model is not None:
        try:
            full_context = (
                f"{ctx.context_before} [MAIN SEGMENT] {ctx.transcript} "
                f"[END SEGMENT] {ctx.context_after}"
            )
            prompt = _build_educational_llm_prompt(full_context, segment_type)
            raw = llm_model.analyze(prompt)
            if isinstance(raw, str):
                raw = json.loads(raw)
            llm_result = raw
        except Exception as exc:
            logger.warning(f"LLM call failed, using heuristics: {exc}")

    topic_relevance = _estimate_topic_relevance(ctx.transcript, ctx.topic_keywords, markers)

    return score_semantic_window(markers, ctx.transcript, segment_type, topic_relevance, llm_result)


def _build_educational_llm_prompt(full_context: str, hint_type: str) -> str:
    return f"""You are analyzing an educational video segment.
Segment type hint: {hint_type}
Context (main segment between tags):
{full_context[:1200]}

Return ONLY valid JSON:
{{
  "explanation_quality": <0-1>,
  "insight_value": <0-1>,
  "structure_score": <0-1>,
  "has_takeaway": <true/false>,
  "has_formula": <true/false>
}}"""


def _estimate_topic_relevance(
    transcript: str,
    topic_keywords: Optional[List[str]],
    markers: Dict[str, Any],
) -> float:
    if not topic_keywords:
        return 0.5
    base = markers["topic_keywords_overlap"]
    # boost by lexical density — more dense text is more likely on-topic
    return float(np.clip(base * 0.7 + markers["lexical_density"] * 0.3, 0, 1))


# =============================================================================
# LAYER 4 — SCORING
# =============================================================================

def compute_educational_score_v5(
    visual: VisualWindowFeatures,
    audio: AudioWindowFeatures,
    semantic: SemanticWindowFeatures,
    mode: str = "educational",
) -> WindowScoreResult:
    """
    4-stage educational scorer:

    Stage 1 — Eligibility gates: hard filters before scoring
    Stage 2 — Core multimodal subscores (semantic / audio / visual / pedagogy)
    Stage 3 — Type-aware bonuses for specific segment types
    Stage 4 — Penalties (redundancy, chaos, off-topic, weak audio)
    """
    weights = MODE_CONFIGS[mode]["weights"]
    penalties: Dict[str, float] = {}
    bonuses: Dict[str, float] = {}
    reasons: List[str] = []
    decision_flags: List[str] = []

    # ── Stage 1: Eligibility ──────────────────────────────────────────────
    reject_reason: Optional[str] = None

    if semantic.topic_relevance < 0.30:
        reject_reason = "low_topic_relevance"
        decision_flags.append("topic_relevance_below_30pct")

    if audio.audio_quality_flag in ("weak", "clipped") and semantic.topic_relevance < 0.5:
        reject_reason = reject_reason or "bad_audio_quality"
        decision_flags.append(f"audio_flag_{audio.audio_quality_flag}")

    if semantic.segment_type == "off_topic" and semantic.insight_value < 0.4:
        reject_reason = reject_reason or "off_topic_low_value"
        decision_flags.append("off_topic_reject")

    if reject_reason:
        return WindowScoreResult(
            final_score=0.05,
            eligible=False,
            reject_reason=reject_reason,
            decision_flags=decision_flags,
            penalties={reject_reason: 0.95},
        )

    eligible = True

    # ── Stage 2: Core subscores ───────────────────────────────────────────

    # Semantic core (0.40 default)
    semantic_core = float(np.clip(
        0.35 * semantic.explanation_quality
        + 0.25 * semantic.insight_value
        + 0.20 * semantic.structure_score
        + 0.20 * semantic.topic_relevance,
        0, 1,
    ))

    # Audio core (0.22 default)
    audio_core = float(np.clip(
        0.35 * audio.speech_clarity
        + 0.25 * audio.speech_rate_score
        + 0.20 * audio.speech_continuity
        + 0.20 * audio.prosody_stability,
        0, 1,
    ))

    # Visual core (0.18 default)
    visual_core = float(np.clip(
        0.40 * visual.visual_readability
        + 0.35 * visual.visual_stability_score
        + 0.25 * visual.instructional_visual_value,
        0, 1,
    ))

    # Pedagogy core (0.20 default)
    pedagogy_core = float(np.clip(
        0.30 * semantic.pedagogical_density
        + 0.25 * semantic.concept_density
        + 0.20 * (1.0 if semantic.has_takeaway else 0.0)
        + 0.15 * (1.0 if semantic.has_example else 0.5)
        + 0.10 * (1.0 if semantic.has_steps else 0.5),
        0, 1,
    ))

    # Weighted combination
    raw_score = (
        weights["semantic"] * semantic_core
        + weights["audio"] * audio_core
        + weights["visual"] * visual_core
        + weights.get("pedagogy", 0.0) * pedagogy_core
    )

    # ── Stage 3: Type-aware bonuses ───────────────────────────────────────
    stype = semantic.segment_type

    if stype == "definition":
        if semantic.concept_density > 0.5:
            bonuses["dense_definition"] = 0.05
        if visual.text_region_ratio_mean > 0.3:
            bonuses["text_on_screen_definition"] = 0.04
        if semantic.topic_relevance > 0.7:
            bonuses["on_topic_definition"] = 0.03
        reasons.append("type_definition")

    elif stype == "formula":
        if visual.text_region_ratio_mean > 0.25 or visual.dense_text_likelihood_mean > 0.5:
            bonuses["formula_on_screen"] = 0.07
        if visual.visual_stability_score > 0.7:
            bonuses["stable_formula_shot"] = 0.03
        if semantic.has_formula:
            bonuses["confirmed_formula"] = 0.05
        reasons.append("type_formula")

    elif stype == "step_by_step":
        if semantic.has_steps:
            bonuses["sequential_structure"] = 0.06
        if audio.speech_continuity > 0.6:
            bonuses["continuous_tutorial_speech"] = 0.04
        if visual.hand_activity_mean > 0.3:
            bonuses["hand_interaction"] = 0.03
        reasons.append("type_step_by_step")

    elif stype == "example":
        if visual.object_demo_ratio > 0.2:
            bonuses["demo_object_present"] = 0.05
        if semantic.contrast_markers > 0:
            bonuses["contrast_example"] = 0.03
        reasons.append("type_example")

    elif stype == "demo":
        if visual.desk_demo_presence_ratio > 0.3:
            bonuses["desk_demo_visible"] = 0.06
        if visual.hand_activity_mean > 0.4:
            bonuses["active_demonstration"] = 0.05
        reasons.append("type_demo")

    elif stype == "summary":
        if semantic.has_takeaway:
            bonuses["clear_takeaway"] = 0.06
        if semantic.topic_keywords_overlap > 0.5:
            bonuses["topic_aligned_summary"] = 0.04
        reasons.append("type_summary")

    elif stype == "explanation":
        if semantic.explanation_quality > 0.7:
            reasons.append("strong_explanation")
        if audio.speech_clarity > 0.7:
            reasons.append("clear_speech")

    # general bonuses
    if visual.screen_presence_ratio > 0.6:
        bonuses["slides_or_screen"] = 0.03
        reasons.append("has_slides")
    if semantic.has_takeaway:
        bonuses["key_takeaway"] = 0.04
        reasons.append("key_takeaway")

    # ── Stage 4: Penalties ────────────────────────────────────────────────
    if semantic.redundancy_penalty > 0.1:
        penalties["redundancy"] = semantic.redundancy_penalty * 0.5

    if visual.scene_change_rate > 0.15:
        penalties["scene_chaos"] = min(0.10, visual.scene_change_rate * 0.4)

    if visual.multi_person_ratio > 0.5 and stype in ("explanation", "definition", "formula"):
        penalties["multi_person_distraction"] = 0.05

    if audio.long_pause_ratio > 0.3 and stype in ("explanation", "step_by_step"):
        penalties["excessive_pauses"] = 0.06

    if stype == "off_topic":
        penalties["off_topic"] = 0.30

    if audio.audio_quality_flag == "noisy":
        penalties["noisy_audio"] = 0.04

    if semantic.confusion_risk > 0.5:
        penalties["confusion_risk"] = semantic.confusion_risk * 0.08

    # Eligibility gates produce soft penalties even when not outright rejecting
    if audio.audio_quality_flag in ("weak", "clipped"):
        penalties["bad_audio_quality"] = 0.08

    if visual.visual_readability < 0.25 and stype in ("definition", "formula"):
        penalties["low_readability_for_text_type"] = 0.07

    total_bonus = sum(bonuses.values())
    total_penalty = sum(penalties.values())

    final_score = float(np.clip(
        raw_score + total_bonus - total_penalty,
        0.0, 1.0,
    ))

    return WindowScoreResult(
        final_score=final_score,
        semantic_core=semantic_core,
        audio_core=audio_core,
        visual_core=visual_core,
        pedagogy_core=pedagogy_core,
        penalties=penalties,
        bonuses=bonuses,
        reasons=reasons,
        decision_flags=decision_flags,
        eligible=eligible,
        reject_reason=reject_reason,
    )


# =============================================================================
# LAYER 5 — WINDOW GENERATION
# =============================================================================

def build_topic_windows(
    topic_segments: List[Dict],
    config: Dict,
) -> List[Dict]:
    """
    Multi-scale window generation per topic.

    Short topic  (≤90s):  single window covering full topic
    Medium topic (≤300s): 45s windows, step=20s
    Long topic   (>300s): multi-scale: 45s + 75s + 120s, step=30s
    """
    min_dur = config.get("min_clip_duration", 25)
    win_range = config.get("window_size_range", (45, 120))

    windows = []
    for t_idx, topic in enumerate(topic_segments):
        t_start = topic.get("start", 0.0)
        t_end = topic.get("end", 0.0)
        t_dur = t_end - t_start

        if t_dur < min_dur:
            continue

        scales: List[Tuple[float, float]] = []

        if t_dur <= 90:
            scales = [(t_dur, t_dur)]
        elif t_dur <= 300:
            scales = [(float(win_range[0]), 20.0)]
        else:
            scales = [
                (float(win_range[0]), 30.0),
                (75.0, 30.0),
                (float(win_range[1]), 30.0),
            ]

        seen: set = set()
        for win_size, step in scales:
            pos = t_start
            while pos < t_end:
                end = min(pos + win_size, t_end)
                if end - pos >= min_dur:
                    key = (round(pos, 1), round(end, 1))
                    if key not in seen:
                        seen.add(key)
                        windows.append({
                            "start": pos,
                            "end": end,
                            "duration": end - pos,
                            "topic_id": t_idx,
                            "topic_title": topic.get("title", f"Topic {t_idx + 1}"),
                            "topic_keywords": topic.get("keywords", []),
                            "window_scale": win_size,
                        })
                pos += step
                if step <= 0:
                    break

    return windows


def merge_short_topics_into_windows(
    topic_segments: List[Dict],
    asr_segments: Optional[List[Dict]] = None,
    min_window: float = 20.0,
    max_window: float = 45.0,
    overlap: float = 7.0,
) -> List[Dict]:
    """
    Fallback 1: merge adjacent short topics into windows of min_window..max_window sec.
    Called when all topics are shorter than min_clip_duration and build_topic_windows → 0.

    Algorithm:
      - Walk sorted topics left-to-right, accumulate until span >= min_window.
      - Cap at max_window. Emit window.
      - Next window starts at (current_end - overlap) to preserve continuity.
    """
    if not topic_segments:
        return []

    sorted_topics = sorted(topic_segments, key=lambda t: float(t.get("start", 0.0)))
    windows: List[Dict] = []
    seen: set = set()
    n = len(sorted_topics)

    i = 0
    while i < n:
        t_start = float(sorted_topics[i].get("start", 0.0))
        t_end = float(sorted_topics[i].get("end", t_start))
        topic_id = int(sorted_topics[i].get("topic_id", i))
        merged_title = str(sorted_topics[i].get("title", f"Topic {i + 1}"))
        merged_kw: List[str] = list(sorted_topics[i].get("keywords", []))

        j = i
        while j < n:
            t_end = float(sorted_topics[j].get("end", t_end))
            merged_kw.extend(sorted_topics[j].get("keywords", []))
            if t_end - t_start >= min_window:
                break
            j += 1

        # cap at max_window
        if t_end - t_start > max_window:
            t_end = t_start + max_window

        dur = t_end - t_start
        if dur >= min_window * 0.6:  # allow slightly shorter at video edges
            key = (round(t_start, 1), round(t_end, 1))
            if key not in seen:
                seen.add(key)
                windows.append({
                    "start": t_start,
                    "end": t_end,
                    "duration": dur,
                    "topic_id": topic_id,
                    "topic_title": merged_title,
                    "topic_keywords": list(dict.fromkeys(merged_kw))[:8],
                    "window_scale": dur,
                    "_source": "merged_short_topics",
                })

        # advance: next window starts at (t_end - overlap)
        next_start = t_end - overlap
        new_i = j + 1
        for k in range(i, n):
            if float(sorted_topics[k].get("end", 0.0)) > next_start:
                new_i = k
                break
        i = max(new_i, i + 1)

    return windows


def build_asr_fallback_windows(
    asr_segments: List[Dict],
    min_window: float = 8.0,
    max_window: float = 35.0,
    step: float = 5.0,
    video_duration: Optional[float] = None,
) -> List[Dict]:
    """
    Fallback 2: build windows directly from ASR segments when topic-based
    approaches yield 0 windows.  Groups consecutive segments into sliding
    windows of min_window..max_window seconds.

    Each returned dict has _source="asr_fallback_windows" so benchmark and
    ChatGPT can identify the source in diagnostics.
    """
    if not asr_segments:
        return []

    sorted_segs = sorted(asr_segments, key=lambda s: float(s.get("start", 0.0)))
    total_end = float(
        video_duration
        or max((s.get("end", 0.0) for s in sorted_segs), default=0.0)
    )
    if total_end <= 0:
        return []

    windows: List[Dict] = []
    seen: set = set()

    pos = float(sorted_segs[0].get("start", 0.0))
    while pos < total_end:
        win_end_target = pos + max_window
        # grab all segments that start within the window (+ small grace)
        in_range = [
            s for s in sorted_segs
            if float(s.get("start", 0.0)) >= pos - 0.5
            and float(s.get("end", 0.0)) <= win_end_target + 3.0
        ]
        if not in_range:
            pos += step
            continue

        actual_end = min(
            float(max(s.get("end", pos) for s in in_range)),
            win_end_target,
        )
        dur = actual_end - pos

        if dur >= min_window:
            key = (round(pos, 1), round(actual_end, 1))
            if key not in seen:
                seen.add(key)
                asr_text_preview = " ".join(
                    str(s.get("text", "")) for s in in_range[:5]
                )[:80]
                windows.append({
                    "start": pos,
                    "end": actual_end,
                    "duration": dur,
                    "topic_id": 0,
                    "topic_title": "ASR Window",
                    "topic_keywords": [],
                    "window_scale": dur,
                    "_source": "asr_fallback_windows",
                    "_asr_preview": asr_text_preview,
                })

        pos += step

    return windows


def collect_window_context(
    window: Dict,
    asr_segments: Optional[List[Dict]],
    topic_segments: Optional[List[Dict]] = None,
) -> WindowContext:
    """Build WindowContext: transcript + ±10s context."""
    start = window["start"]
    end = window["end"]

    transcript = ""
    context_before = ""
    context_after = ""

    if asr_segments:
        in_window = [s for s in asr_segments if s.get("start", 0) <= end and s.get("end", 0) >= start]
        transcript = " ".join(s.get("text", "") for s in in_window)

        before = [s for s in asr_segments if start - 12 <= s.get("start", 0) < start]
        context_before = " ".join(s.get("text", "") for s in before[-3:])

        after = [s for s in asr_segments if end < s.get("start", 0) <= end + 12]
        context_after = " ".join(s.get("text", "") for s in after[:3])

    return WindowContext(
        start=start,
        end=end,
        duration=end - start,
        transcript=transcript,
        context_before=context_before,
        context_after=context_after,
        topic_id=window.get("topic_id", 0),
        topic_title=window.get("topic_title", ""),
        topic_keywords=window.get("topic_keywords", []),
    )


# =============================================================================
# LAYER 6 — DEDUPLICATION
# =============================================================================

def _overlap_ratio(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Intersection over minimum length."""
    overlap = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    min_len = min(a_end - a_start, b_end - b_start)
    return overlap / max(min_len, 1e-6)


def _transcript_similarity(t1: str, t2: str) -> float:
    """Fast token Jaccard similarity."""
    s1 = set(t1.lower().split())
    s2 = set(t2.lower().split())
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


def deduplicate_windows(
    analyzed: List[Dict],
    overlap_threshold: float = 0.60,
    similarity_threshold: float = 0.65,
) -> List[Dict]:
    """
    NMS-style deduplication.

    For each pair where temporal overlap > threshold AND transcript similarity > threshold
    AND both are the same segment_type → suppress the one with the lower score.
    """
    if not analyzed:
        return []

    sorted_items = sorted(analyzed, key=lambda x: x["score"].final_score, reverse=True)
    kept = []
    suppressed_ids: set = set()

    for i, item_i in enumerate(sorted_items):
        if i in suppressed_ids:
            continue
        kept.append(item_i)
        for j, item_j in enumerate(sorted_items):
            if j <= i or j in suppressed_ids:
                continue
            ov = _overlap_ratio(
                item_i["context"].start, item_i["context"].end,
                item_j["context"].start, item_j["context"].end,
            )
            if ov < overlap_threshold:
                continue
            sim = _transcript_similarity(item_i["context"].transcript, item_j["context"].transcript)
            if sim > similarity_threshold:
                suppressed_ids.add(j)
                item_j["suppressed_by"] = i
                item_j["suppression_reason"] = f"overlap={ov:.2f} sim={sim:.2f}"

    logger.info(f"Dedup: {len(analyzed)} → {len(kept)} (suppressed {len(suppressed_ids)})")
    return kept


# =============================================================================
# LAYER 7 — COVERAGE-BASED SELECTOR
# =============================================================================

# Role quotas for each topic
_TOPIC_SELECTION_PLAN = {
    "must_have": ["explanation"],
    "prefer_one_of": ["example", "step_by_step", "formula", "demo"],
    "optional_one_of": ["definition", "summary", "comparison"],
    "max_windows_per_topic": 3,
}


def select_windows_with_topic_coverage(
    candidates: List[Dict],
    topic_segments: List[Dict],
    max_per_topic: int = 3,
    min_coverage_topics: int = 1,
    max_monopoly_ratio: float = 0.60,
) -> List[Dict]:
    """
    Coverage-aware selector.

    For each topic:
      1. Try to fill role quotas (must_have, prefer_one_of, optional_one_of)
      2. Cap at max_per_topic
    Global:
      3. Prevent a single topic from taking > max_monopoly_ratio of all selected clips
    """
    from collections import defaultdict

    by_topic: Dict[int, List[Dict]] = defaultdict(list)
    for item in candidates:
        if item.get("score", WindowScoreResult()).eligible:
            by_topic[item["context"].topic_id].append(item)

    # sort each topic's candidates by score descending
    for tid in by_topic:
        by_topic[tid].sort(key=lambda x: x["score"].final_score, reverse=True)

    selected: List[Dict] = []
    selection_trace: List[Dict] = []

    for t_idx in range(len(topic_segments)):
        pool = by_topic.get(t_idx, [])
        if not pool:
            selection_trace.append({"topic_id": t_idx, "selected": [], "reason": "no_candidates"})
            continue

        chosen_types: List[str] = []
        chosen: List[Dict] = []

        def _pick_by_type(role_types: List[str], exclusive: bool) -> Optional[Dict]:
            for role in role_types:
                for item in pool:
                    stype = item["semantic"].segment_type
                    if stype == role and stype not in chosen_types:
                        return item
            return None

        # must_have
        for role in _TOPIC_SELECTION_PLAN["must_have"]:
            if len(chosen) >= max_per_topic:
                break
            candidate = _pick_by_type([role], exclusive=True)
            if candidate is None:
                # fallback: take best available that hasn't been chosen
                remaining = [x for x in pool if x not in chosen]
                if remaining:
                    candidate = remaining[0]
            if candidate and candidate not in chosen:
                chosen.append(candidate)
                chosen_types.append(candidate["semantic"].segment_type)

        # prefer_one_of
        if len(chosen) < max_per_topic:
            candidate = _pick_by_type(_TOPIC_SELECTION_PLAN["prefer_one_of"], exclusive=True)
            if candidate and candidate not in chosen:
                chosen.append(candidate)
                chosen_types.append(candidate["semantic"].segment_type)

        # optional_one_of
        if len(chosen) < max_per_topic:
            candidate = _pick_by_type(_TOPIC_SELECTION_PLAN["optional_one_of"], exclusive=True)
            if candidate and candidate not in chosen:
                chosen.append(candidate)
                chosen_types.append(candidate["semantic"].segment_type)

        for item in chosen:
            item["selection_role"] = _determine_role(item["semantic"].segment_type)
        selected.extend(chosen)

        selection_trace.append({
            "topic_id": t_idx,
            "selected": [
                {
                    "start": item["context"].start,
                    "end": item["context"].end,
                    "type": item["semantic"].segment_type,
                    "score": round(item["score"].final_score, 4),
                    "role": item.get("selection_role"),
                }
                for item in chosen
            ],
        })

    # global monopoly guard
    if selected:
        from collections import Counter
        topic_counts = Counter(item["context"].topic_id for item in selected)
        total = len(selected)
        dominant_topic = topic_counts.most_common(1)[0]
        if total > 2 and dominant_topic[1] / total > max_monopoly_ratio:
            # trim dominant topic to max_monopoly_ratio fraction
            max_allowed = max(1, int(total * max_monopoly_ratio))
            dominant_id = dominant_topic[0]
            dominant_items = [x for x in selected if x["context"].topic_id == dominant_id]
            other_items = [x for x in selected if x["context"].topic_id != dominant_id]
            trimmed = sorted(dominant_items, key=lambda x: x["score"].final_score, reverse=True)[:max_allowed]
            selected = other_items + trimmed
            logger.info(f"Monopoly guard: trimmed topic {dominant_id} to {max_allowed} clips")

    logger.info(f"Coverage selector: {len(candidates)} → {len(selected)} clips across {len(topic_segments)} topics")
    return selected


def _determine_role(segment_type: str) -> str:
    if segment_type in ("explanation",):
        return "core_explanation"
    if segment_type in ("example", "step_by_step", "formula", "demo"):
        return "support"
    if segment_type in ("definition", "summary", "comparison"):
        return "context"
    return "supplemental"


# =============================================================================
# LAYER 8 — BOUNDARY REFINEMENT
# =============================================================================

def refine_clip_boundaries(
    selected: List[Dict],
    asr_segments: Optional[List[Dict]],
    audio_global: Optional[AudioGlobalData],
    pad_before_sec: float = 2.0,
    pad_after_sec: float = 1.5,
) -> List[Dict]:
    """
    Snap clip boundaries to natural sentence or silence boundaries.

    Strategy:
    - Expand start backward by up to pad_before_sec to nearest sentence start
    - Trim end to nearest sentence end or silence after final sentence
    """
    if not asr_segments:
        return selected

    asr_starts = np.array([s.get("start", 0.0) for s in asr_segments])
    asr_ends = np.array([s.get("end", 0.0) for s in asr_segments])

    for item in selected:
        ctx = item["context"]
        new_start = ctx.start
        new_end = ctx.end

        # snap start: find nearest ASR start within pad_before_sec before current start
        candidates_start = asr_starts[(asr_starts >= ctx.start - pad_before_sec) & (asr_starts <= ctx.start)]
        if len(candidates_start) > 0:
            new_start = float(candidates_start[-1])

        # snap end: find nearest ASR end within pad_after_sec after current end
        candidates_end = asr_ends[(asr_ends >= ctx.end) & (asr_ends <= ctx.end + pad_after_sec)]
        if len(candidates_end) > 0:
            new_end = float(candidates_end[0])

        item["refined_start"] = round(new_start, 2)
        item["refined_end"] = round(new_end, 2)
        item["refined_duration"] = round(new_end - new_start, 2)

    return selected


# =============================================================================
# v5.1 — ASR-BASED TOPIC SEGMENTATION (non-LLM, lightweight)
# =============================================================================

# Structural markers that likely indicate a topic transition
_TOPIC_SPLIT_MARKERS = [
    # Russian
    "важно", "важный момент", "запомни",
    "например", "к примеру", "вот пример",
    "то есть", "иначе говоря", "другими словами",
    "во-первых", "во-вторых", "в-третьих",
    "первое", "второе", "третье",
    "поэтому", "таким образом", "итог",
    "вывод", "подводя итог", "резюмируя",
    "перейдём", "а теперь", "теперь про",
    "следующий момент", "кстати",
    # English
    "important", "for example", "for instance",
    "that is", "in other words", "firstly",
    "secondly", "thirdly", "first", "second", "third",
    "therefore", "in conclusion", "to summarize",
    "let's move on", "now let's", "next",
]


def segment_topics_from_asr(
    asr_segments: List[Dict],
    video_duration_sec: float,
    pause_threshold_sec: float = 2.5,
    min_segs_per_topic: int = 3,
    max_segs_per_topic: int = 6,
    min_topic_duration_sec: float = 6.0,
) -> List[Dict]:
    """
    Non-LLM topic segmentation из ASR сегментов.

    Алгоритм:
      1. Детектим "split points" по паузам (gap > pause_threshold_sec)
         и маркерам-сигналам ("важно", "например", "первое", …).
      2. Группируем сегменты между точками разреза.
      3. Enforce: 3-6 ASR segments per topic — сливаем короткие, дробим длинные.

    Returns:
      [{"start", "end", "duration", "title", "topic_id", "keywords",
        "n_asr_segments", "_source": "asr_topic_segmentation"}]
    """
    if not asr_segments or len(asr_segments) < min_segs_per_topic:
        return []

    segs = sorted(asr_segments, key=lambda s: float(s.get("start", 0.0)))

    # 1. Find split points (indices at which a new topic starts)
    split_indices = [0]
    for i in range(1, len(segs)):
        prev_end = float(segs[i - 1].get("end", 0.0))
        cur_start = float(segs[i].get("start", 0.0))
        cur_text = str(segs[i].get("text", "")).lower()

        pause_split = (cur_start - prev_end) > pause_threshold_sec
        marker_split = any(m in cur_text for m in _TOPIC_SPLIT_MARKERS)

        if pause_split or marker_split:
            split_indices.append(i)

    # Build initial groups
    groups: List[List[int]] = []
    for k in range(len(split_indices)):
        start_i = split_indices[k]
        end_i = split_indices[k + 1] if k + 1 < len(split_indices) else len(segs)
        group = list(range(start_i, end_i))
        if group:
            groups.append(group)

    # 2. Normalise group size: merge small, split large
    merged: List[List[int]] = []
    i = 0
    while i < len(groups):
        g = groups[i]
        # If too small — merge with next
        while len(g) < min_segs_per_topic and i + 1 < len(groups):
            g = g + groups[i + 1]
            i += 1
        merged.append(g)
        i += 1

    final_groups: List[List[int]] = []
    for g in merged:
        if len(g) <= max_segs_per_topic:
            final_groups.append(g)
        else:
            # Split into chunks of ≤ max_segs_per_topic
            step = max_segs_per_topic
            for s in range(0, len(g), step):
                chunk = g[s:s + step]
                if chunk:
                    final_groups.append(chunk)

    # 3. Build topic dicts
    topics: List[Dict] = []
    for topic_id, indices in enumerate(final_groups):
        g_start = float(segs[indices[0]].get("start", 0.0))
        g_end = float(segs[indices[-1]].get("end", g_start + 5.0))
        g_end = min(g_end, video_duration_sec)
        duration = g_end - g_start

        if duration < min_topic_duration_sec:
            continue

        # Keywords: TOP-5 common non-trivial words
        joined_text = " ".join(str(segs[i].get("text", "")) for i in indices).lower()
        tokens = [t for t in joined_text.split() if len(t) > 4]
        from collections import Counter as _C
        top_tokens = [tok for tok, _ in _C(tokens).most_common(5)]

        # Title: first 8 words of the first segment
        first_text = str(segs[indices[0]].get("text", "")).strip()
        title = " ".join(first_text.split()[:8]) or f"Topic {topic_id + 1}"

        topics.append({
            "start": round(g_start, 2),
            "end": round(g_end, 2),
            "duration": round(duration, 2),
            "title": title,
            "topic_id": topic_id,
            "keywords": top_tokens,
            "n_asr_segments": len(indices),
            "asr_indices": indices,
            "confidence": 0.5,  # неизвестно — для совместимости с viral topic API
            "_source": "asr_topic_segmentation",
        })

    return topics


# =============================================================================
# LAYER 9 — ORCHESTRATOR
# =============================================================================

def run_educational_mode_v5(
    video_path: str,
    asr_segments: Optional[List[Dict]],
    topic_segments: Optional[List[Dict]],
    base_analysis: Optional[Dict],
    mode: str = "educational",
    audio_profile: str = "lecture",
    top_k_per_topic: int = 3,
    threshold: float = 0.52,
    adaptive_threshold: bool = True,
    llm_model: Optional[Any] = None,
) -> Dict:
    """
    Full educational pipeline v5.

    Returns:
        {
            "mode": str,
            "topic_segments": [...],
            "educational_moments": [...],
            "chapters": [...],
            "stats": {...},
            "_diagnostics": {...}
        }
    """
    logger.info("=" * 80)
    logger.info(f"EDUCATIONAL MODE v5.0 — {mode.upper()}")
    logger.info("=" * 80)

    config = MODE_CONFIGS[mode]
    base_analysis = base_analysis or {}

    # ── Step 1: audio ─────────────────────────────────────────────────────
    logger.info("Step 1/8: Extracting audio...")
    audio_global = extract_audio_global(video_path)
    if not audio_global.extraction_ok:
        return {"mode": mode, "error": "audio_extraction_failed", "detail": audio_global.error}

    # ── Step 2: topic segments ─────────────────────────────────────────────
    # v5.1: если нет явных topic_segments, но есть >10 ASR сегментов —
    # автоматически сегментируем по паузам + маркерам (non-LLM).
    topic_source = "provided"
    if not topic_segments:
        if asr_segments and len(asr_segments) > 10:
            auto_topics = segment_topics_from_asr(
                asr_segments=asr_segments,
                video_duration_sec=audio_global.duration_sec,
            )
            if auto_topics:
                topic_segments = auto_topics
                topic_source = "asr_auto"
                logger.info(
                    f"Auto-segmented {len(asr_segments)} ASR segments into "
                    f"{len(auto_topics)} topics using pauses + markers"
                )

        if not topic_segments:
            if config["requires_topic_segmentation"]:
                logger.warning(
                    "Educational mode requires topic segmentation — "
                    "degrading to educational-lite (single topic). "
                    "Provide real topic_segments for full intelligent selection."
                )
                topic_segments = [{
                    "start": 0.0,
                    "end": audio_global.duration_sec,
                    "duration": audio_global.duration_sec,
                    "title": "Full Video",
                    "topic_id": 0,
                    "keywords": [],
                    "_fallback": True,
                }]
                topic_source = "fallback_single_topic"
            else:
                topic_segments = []
                topic_source = "none"
    logger.info(f"Step 2/8: {len(topic_segments)} topics (source={topic_source})")

    # ── Step 3: build windows ──────────────────────────────────────────────
    logger.info("Step 3/8: Building windows...")

    # micro_educational: video < 3 min → allow smaller windows (8–45 s)
    _video_dur = audio_global.duration_sec
    _micro_mode = _video_dur < 180.0
    if _micro_mode:
        logger.info(f"  micro_educational mode (video={_video_dur:.1f}s < 180s): lowering min_clip_duration → 8s")
    _build_config = dict(config)
    if _micro_mode:
        _build_config["min_clip_duration"] = 8.0
        _build_config["window_size_range"] = (8, 45)

    windows = build_topic_windows(topic_segments, _build_config)
    _windows_before_fallback = len(windows)
    _window_source = "topic_windows"

    # Fallback 1: merge adjacent short topics into valid-size windows
    if not windows and topic_segments:
        logger.warning(
            f"  build_topic_windows → 0 (all topics shorter than min_clip_duration). "
            f"Trying merge_short_topics_into_windows..."
        )
        _min_win = 10.0 if _micro_mode else 20.0
        _max_win = 45.0
        windows = merge_short_topics_into_windows(
            topic_segments, asr_segments,
            min_window=_min_win, max_window=_max_win, overlap=7.0,
        )
        if windows:
            _window_source = "merged_short_topics"
            logger.info(f"  Merge fallback: {len(windows)} windows (source=merged_short_topics)")

    # Fallback 2: build directly from ASR segments
    if not windows and asr_segments:
        logger.warning(
            f"  merge_short_topics → 0 as well. "
            f"Falling back to ASR-direct windows (source=asr_fallback_windows)..."
        )
        _min_win = 8.0 if _micro_mode else 15.0
        _max_win = 35.0 if _micro_mode else 60.0
        windows = build_asr_fallback_windows(
            asr_segments,
            min_window=_min_win, max_window=_max_win, step=5.0,
            video_duration=_video_dur,
        )
        if windows:
            _window_source = "asr_fallback_windows"
            logger.warning(f"  ASR fallback: {len(windows)} windows")

    _windows_after_fallback = len(windows)
    logger.info(
        f"  Windows: before_fallback={_windows_before_fallback}  "
        f"after_fallback={_windows_after_fallback}  source={_window_source}"
    )

    # ── Step 4: analyze all windows ───────────────────────────────────────
    logger.info("Step 4/8: Analyzing windows...")
    analyzed: List[Dict] = []
    for window in windows:
        ctx = collect_window_context(window, asr_segments, topic_segments)
        visual = extract_visual_features_for_window(base_analysis, ctx.start, ctx.end)
        audio = compute_audio_features_from_array(
            audio_global.samples, audio_global.sample_rate,
            ctx.start, ctx.end, profile=audio_profile,
        )
        semantic = analyze_semantic_window(ctx, llm_model=llm_model)
        score = compute_educational_score_v5(visual, audio, semantic, mode=mode)

        analyzed.append({
            "context": ctx,
            "visual": visual,
            "audio": audio,
            "semantic": semantic,
            "score": score,
            "window_meta": window,
        })

    logger.info(f"  Analyzed {len(analyzed)} windows")

    # ── Step 5: adaptive threshold ─────────────────────────────────────────
    eligible_scores = [a["score"].final_score for a in analyzed if a["score"].eligible]
    if adaptive_threshold and eligible_scores:
        pct75 = float(np.percentile(eligible_scores, 75))
        # don't let the threshold choke everything — min floor
        threshold = float(np.clip(max(threshold, pct75 * 0.90), 0.30, 0.85))
        logger.info(f"Step 5/8: Adaptive threshold = {threshold:.3f}")

    # ── Step 6: filter + dedup ─────────────────────────────────────────────
    logger.info("Step 6/8: Filtering and deduplicating...")
    above_threshold = [a for a in analyzed if a["score"].final_score >= threshold and a["score"].eligible]

    # Fallback: если threshold убил всё, но eligible-окна есть → top-3 как manual_review
    _weak_edu_fallback_used = False
    if not above_threshold:
        eligible_all = [a for a in analyzed if a["score"].eligible]
        if eligible_all:
            weak_pool = sorted(eligible_all, key=lambda a: a["score"].final_score, reverse=True)[:3]
            for a in weak_pool:
                a["_weak_edu_fallback"] = True
            above_threshold = weak_pool
            _weak_edu_fallback_used = True
            logger.warning(
                f"Educational threshold={threshold:.3f} killed all candidates. "
                f"Promoting top-{len(weak_pool)} eligible windows as manual_review fallback."
            )
        else:
            logger.warning("Educational: 0 eligible windows even before threshold — no output.")

    deduped = deduplicate_windows(above_threshold)

    # ── Step 7: coverage-based selection ──────────────────────────────────
    logger.info("Step 7/8: Coverage-based selection...")
    selected = select_windows_with_topic_coverage(
        deduped,
        topic_segments,
        max_per_topic=top_k_per_topic,
    )

    # ── Step 8: boundary refinement ───────────────────────────────────────
    logger.info("Step 8/8: Refining clip boundaries...")
    selected = refine_clip_boundaries(selected, asr_segments, audio_global)

    # ── Build output ───────────────────────────────────────────────────────
    _pipeline_trace = {
        "stage_topics":                 len(topic_segments),
        "stage_windows_before_fallback": _windows_before_fallback,
        "stage_windows_after_fallback":  _windows_after_fallback,
        "window_source":                 _window_source,
        "micro_mode":                    _micro_mode,
        "weak_edu_fallback_used":        _weak_edu_fallback_used,
        "topic_source":                  topic_source,
    }
    out = build_educational_output(
        selected, topic_segments, analyzed, config, threshold,
        pipeline_trace=_pipeline_trace,
    )
    out["topic_source"] = topic_source
    out["weak_edu_fallback_used"] = _weak_edu_fallback_used
    # v5.1: экспонируем topic_segments как отдельное поле для benchmark дампа
    out["topic_segments"] = [
        {k: v for k, v in t.items() if k != "asr_indices"}
        for t in topic_segments
    ]
    # v5.1: educational_windows.json содержит ВСЕ построенные окна (до фильтра)
    out["educational_windows"] = [
        {
            "start": round(a["context"].start, 2),
            "end": round(a["context"].end, 2),
            "duration": round(a["context"].end - a["context"].start, 2),
            "topic_id": a["context"].topic_id,
            "topic_title": a["context"].topic_title,
            "segment_type": a["semantic"].segment_type,
            "transcript_preview": a["context"].transcript[:120],
            "has_takeaway": a["semantic"].has_takeaway,
            "has_formula": a["semantic"].has_formula,
            "has_example": a["semantic"].has_example,
            "has_steps": a["semantic"].has_steps,
            "final_score": round(a["score"].final_score, 4),
            "eligible": a["score"].eligible,
            "reject_reason": a["score"].reject_reason,
        }
        for a in analyzed
    ]
    # v5.1: educational_scores.json — только числовые скоры, компактно
    out["educational_scores"] = [
        {
            "start": round(a["context"].start, 2),
            "end": round(a["context"].end, 2),
            "topic_id": a["context"].topic_id,
            "final": round(a["score"].final_score, 4),
            "semantic": round(a["score"].semantic_core, 4),
            "audio": round(a["score"].audio_core, 4),
            "visual": round(a["score"].visual_core, 4),
            "pedagogy": round(a["score"].pedagogy_core, 4),
            "eligible": a["score"].eligible,
        }
        for a in analyzed
    ]
    return out


# =============================================================================
# LAYER 10 — OUTPUT BUILDER
# =============================================================================

def build_educational_output(
    selected: List[Dict],
    topic_segments: List[Dict],
    all_analyzed: List[Dict],
    config: Dict,
    threshold: float,
    pipeline_trace: Optional[Dict] = None,
) -> Dict:
    moments = []
    for item in selected:
        ctx: WindowContext = item["context"]
        sem: SemanticWindowFeatures = item["semantic"]
        score: WindowScoreResult = item["score"]

        stype = sem.segment_type
        template = EDUCATIONAL_SEGMENT_TYPES.get(stype, {}).get("title_template", "{topic}")
        title = template.format(topic=ctx.topic_title)

        start = item.get("refined_start", ctx.start)
        end = item.get("refined_end", ctx.end)

        moments.append({
            "start": round(start, 2),
            "end": round(end, 2),
            "duration": round(end - start, 2),
            "score": round(score.final_score, 4),
            "subscores": {
                "semantic": round(score.semantic_core, 4),
                "audio": round(score.audio_core, 4),
                "visual": round(score.visual_core, 4),
                "pedagogy": round(score.pedagogy_core, 4),
            },
            "segment_type": stype,
            "selection_role": item.get("selection_role", "supplemental"),
            "reasons": score.reasons,
            "decision_flags": score.decision_flags,
            "bonuses": score.bonuses,
            "penalties": score.penalties,
            "topic_id": ctx.topic_id,
            "topic_title": ctx.topic_title,
            "title": title,
            "transcript_preview": ctx.transcript[:250],
            "has_takeaway": sem.has_takeaway,
            "has_formula": sem.has_formula,
            "has_example": sem.has_example,
            "has_steps": sem.has_steps,
            "export_decision": "manual_review" if item.get("_weak_edu_fallback") else "auto_export",
            "is_weak_edu_fallback": bool(item.get("_weak_edu_fallback")),
        })

    moments.sort(key=lambda m: (m["topic_id"], m["start"]))

    chapters = [
        {
            "start": round(t["start"], 2),
            "end": round(t["end"], 2),
            "title": t.get("title", f"Chapter {i + 1}"),
            "duration": round(t["end"] - t["start"], 2),
        }
        for i, t in enumerate(topic_segments)
    ]

    _pt = pipeline_trace or {}
    scores_list = [a["score"].final_score for a in all_analyzed if a["score"].eligible]
    stats = {
        "num_topics": len(topic_segments),
        "num_windows_analyzed": len(all_analyzed),
        "num_windows_above_threshold": len([a for a in all_analyzed if a["score"].final_score >= threshold]),
        "num_moments_found": len(moments),
        "threshold": round(threshold, 4),
        "mode": config.get("title_style", "educational"),
        "avg_score_all": round(float(np.mean(scores_list)), 4) if scores_list else 0.0,
        "avg_score_selected": round(float(np.mean([m["score"] for m in moments])), 4) if moments else 0.0,
        "topics_covered": len(set(m["topic_id"] for m in moments)),
        "has_real_visual_data": bool(all_analyzed and any(
            a["visual"].person_present_ratio != 0.5 for a in all_analyzed
        )),
        # pipeline trace (v5.2) — passed explicitly, never read from outer scope
        "stage_topics":                  _pt.get("stage_topics", len(topic_segments)),
        "stage_windows_before_fallback": _pt.get("stage_windows_before_fallback", 0),
        "stage_windows_after_fallback":  _pt.get("stage_windows_after_fallback", len(all_analyzed)),
        "window_source":                 _pt.get("window_source", "unknown"),
        "micro_mode":                    _pt.get("micro_mode", False),
        "weak_edu_fallback_used":        _pt.get("weak_edu_fallback_used", False),
        "topic_source":                  _pt.get("topic_source", "unknown"),
    }

    # diagnostics payload (attached as _diagnostics, stripped in export_cleaned)
    diagnostics = {
        "window_diagnostics": [
            {
                "start": round(a["context"].start, 2),
                "end": round(a["context"].end, 2),
                "topic_id": a["context"].topic_id,
                "segment_type": a["semantic"].segment_type,
                "final_score": round(a["score"].final_score, 4),
                "eligible": a["score"].eligible,
                "reject_reason": a["score"].reject_reason,
                "semantic_core": round(a["score"].semantic_core, 4),
                "audio_core": round(a["score"].audio_core, 4),
                "visual_core": round(a["score"].visual_core, 4),
                "pedagogy_core": round(a["score"].pedagogy_core, 4),
                "penalties": a["score"].penalties,
                "bonuses": a["score"].bonuses,
                "audio_flag": a["audio"].audio_quality_flag,
                "suppressed_by": a.get("suppressed_by"),
                "suppression_reason": a.get("suppression_reason"),
            }
            for a in all_analyzed
        ],
    }

    logger.info("=" * 80)
    logger.info("EDUCATIONAL MODE v5.0 COMPLETE")
    logger.info(
        f"  Topics: {stats['num_topics']} | Windows: {stats['num_windows_analyzed']} | "
        f"Selected: {stats['num_moments_found']} | Avg score: {stats['avg_score_selected']}"
    )
    logger.info(f"  Visual data: {'REAL' if stats['has_real_visual_data'] else 'PLACEHOLDER (provide base_analysis!)'}")
    logger.info("=" * 80)

    return {
        "mode": "educational_v5",
        "topic_segments": topic_segments,
        "educational_moments": moments,
        "chapters": chapters,
        "stats": stats,
        "_diagnostics": diagnostics,
    }


# =============================================================================
# EXPORT
# =============================================================================

def export_educational_result(
    result: Dict,
    output_json: str,
    export_chapters: bool = True,
    chapter_formats: Optional[List[str]] = None,
    export_diagnostics: bool = True,
) -> None:
    """
    Save result JSON + optional chapter exports + diagnostic files.

    Exports:
        <output_json>                   — main result (no diagnostics)
        <stem>_debug_scores.json        — flat score table for all windows
        <stem>_window_diagnostics.jsonl — per-window JSONL for streaming analysis
        <stem>_selection_trace.json     — why each moment was selected/rejected
    """
    if chapter_formats is None:
        chapter_formats = ["json", "ffmetadata", "webvtt"]

    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # main result without internal diagnostics
    clean_result = {k: v for k, v in result.items() if k != "_diagnostics"}
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(clean_result, fh, indent=2, ensure_ascii=False)
    logger.info(f"Result saved: {output_path}")

    if export_diagnostics and "_diagnostics" in result:
        diag = result["_diagnostics"]
        stem = output_path.stem

        # debug scores table
        debug_path = output_path.parent / f"{stem}_debug_scores.json"
        with open(debug_path, "w", encoding="utf-8") as fh:
            json.dump(diag.get("window_diagnostics", []), fh, indent=2, ensure_ascii=False)
        logger.info(f"Debug scores saved: {debug_path}")

        # window diagnostics JSONL
        diag_path = output_path.parent / f"{stem}_window_diagnostics.jsonl"
        with open(diag_path, "w", encoding="utf-8") as fh:
            for entry in diag.get("window_diagnostics", []):
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info(f"Window diagnostics saved: {diag_path}")

        # selection trace (selected moments with reason)
        trace_path = output_path.parent / f"{stem}_selection_trace.json"
        trace = []
        for m in result.get("educational_moments", []):
            trace.append({
                "start": m["start"],
                "end": m["end"],
                "topic_title": m["topic_title"],
                "segment_type": m["segment_type"],
                "selection_role": m["selection_role"],
                "score": m["score"],
                "reasons": m["reasons"],
                "decision_flags": m["decision_flags"],
                "bonuses": m["bonuses"],
                "penalties": m["penalties"],
            })
        with open(trace_path, "w", encoding="utf-8") as fh:
            json.dump(trace, fh, indent=2, ensure_ascii=False)
        logger.info(f"Selection trace saved: {trace_path}")

    # optional chapter export
    if export_chapters and result.get("chapters"):
        try:
            from integrations.chapter_export import ChapterExporter
            exporter = ChapterExporter()
            chapters_dir = output_path.parent / "chapters"
            chapters_dir.mkdir(exist_ok=True)
            export_results = exporter.batch_export(
                result["chapters"],
                output_dir=str(chapters_dir),
                basename=output_path.stem,
                formats=chapter_formats,
                metadata={"title": f"Educational Analysis — {output_path.stem}", "mode": result["mode"]},
            )
            ok = sum(1 for v in export_results.values() if v)
            logger.info(f"Chapters exported: {ok}/{len(export_results)} formats")
        except ImportError:
            logger.warning("ChapterExporter not available — chapter export skipped")


# =============================================================================
# CLI ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Educational Mode v5.0")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--mode", choices=["educational", "viral"], default="educational")
    parser.add_argument("--profile", choices=["lecture", "podcast", "tutorial"], default="lecture")
    parser.add_argument("--output", "-o", help="Output JSON path")
    parser.add_argument("--chapters", action="store_true", help="Export chapters")
    parser.add_argument("--no-diagnostics", action="store_true", help="Skip diagnostic file export")
    parser.add_argument("--top-k", type=int, default=3, help="Max clips per topic")
    parser.add_argument("--threshold", type=float, default=0.52, help="Base score threshold")

    args = parser.parse_args()

    # get duration via ffprobe
    ffprobe_cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", args.video,
    ]
    try:
        duration_str = subprocess.run(ffprobe_cmd, capture_output=True, text=True, check=True).stdout.strip()
        duration = float(duration_str)
    except Exception as e:
        logger.error(f"Could not determine video duration: {e}")
        sys.exit(1)

    # NOTE: provide real asr_segments, topic_segments, base_analysis here
    result = run_educational_mode_v5(
        video_path=args.video,
        asr_segments=None,
        topic_segments=None,
        base_analysis={},
        mode=args.mode,
        audio_profile=args.profile,
        top_k_per_topic=args.top_k,
        threshold=args.threshold,
    )

    output_path = args.output or f"educational_result_v5_{Path(args.video).stem}.json"
    export_educational_result(
        result,
        output_json=output_path,
        export_chapters=args.chapters,
        export_diagnostics=not args.no_diagnostics,
    )
