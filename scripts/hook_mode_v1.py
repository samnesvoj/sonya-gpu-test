"""
HOOK MODE v2.0 - Opening Hook Detection
Находит цепляющие хуки в первых секундах видео.

Architecture (v2.0 — layered pipeline):

    propose_hook_candidates()   ← multi-source: sliding window + text-onset events
         ↓
    _extract_hook_features()    ← per-candidate local signals (audio, visual, text)
         ↓
    _compute_hook_subscores()   ← curiosity / surprise / emotional / immediacy /
                                   continuation / delivery
         ↓
    compute_hook_penalties()    ← boring_intro / late_hook / context_dep /
                                   explanation / false_peak
         ↓
    compute_hook_final_score()  ← weighted combination of subscores - penalties
         ↓
    _refine_hook_boundaries()   ← snap to phrase / punctuation / pause boundaries
         ↓
    find_hook_moments()         ← orchestrator (slim); NMS → top-K

Public API (unchanged from v1.x):
    detect_hook_structure()
    detect_hook_type()
    compute_hook_final_score()
    build_hook_reasons()
    find_hook_moments()

Changelog v2.0:
- Layered pipeline: proposal → features → subscores → penalties → refine
- Local intensity per candidate window (was: global hook_end average)
- Multi-subscores: curiosity / surprise / emotional / immediacy / continuation / delivery
- Penalty layer: boring_intro / late_hook / context_dep / explanation / false_peak
- Nuanced viral_compat (replaces binary 0.7 / 0.3)
- Multi-source proposal: sliding window + text-onset events around strong markers
- Boundary refinement: snap to phrase / punctuation / ASR pause boundaries
- New hook types: curiosity_hook / warning_hook / reveal_hook /
                  reaction_hook / contrarian_hook
- Quality output per moment: hook_confidence, false_positive_risk,
  boundary_quality, continuation_tension, subscores, penalties

Changelog v1.4:
- Regex для всех YAML/JSON профилей
- Множественные вхождения: findall вместо search
- step_sec и nms_iou_thresh вынесены в конфиг
- Effective thresholds в stats
"""

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Pattern

import numpy as np

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class HookModeConfig:
    """
    Единый конфиг Hook Mode — YAML/JSON профили под ниши.

    Профили:
    - reels_hooks_v1.yaml       — короткие хуки (3-5s), акцент на вопросы + факты
    - youtube_hooks_v1.yaml     — длинные хуки (5-10s), интрига + обещание ценности
    - educational_hooks_v1.yaml — факты + эмоции (шок от статистики)
    """
    # Метаданные
    mode_name: str = "default"
    profile_version: str = "v2.0"
    locale: str = "ru"

    # Параметры окна
    hook_window_sec: float = 10.0
    min_hook_duration: float = 3.0
    position_boost: float = 1.5      # Бонус для маркеров в первых 3 сек

    # Пороги
    threshold: float = 0.45
    min_hook_score: float = 0.35

    # Режим фильтрации
    loose_hook_mode: bool = False

    # Sliding window
    step_sec: float = 1.0
    nms_iou_thresh: float = 0.6

    # Boundary refinement: snap window edges to phrase boundaries
    # within this tolerance (seconds)
    boundary_snap_sec: float = 0.5

    # Веса компонентов финального score
    weights: Dict[str, float] = field(default_factory=lambda: {
        # Sub-score weights (sum should be ≤ 1.0 — remainder is policy slack)
        "curiosity":     0.22,
        "surprise":      0.18,
        "emotional":     0.15,
        "immediacy":     0.12,
        "continuation":  0.13,
        "delivery":      0.10,
        # Legacy compatibility slot (maps to sub-score blend internally)
        "markers":       0.40,
        "intensity":     0.25,
        "type_match":    0.20,
        "viral_compat":  0.15,
    })

    # Penalty weights (each scales a 0..1 raw penalty → subtracted from score)
    penalty_weights: Dict[str, float] = field(default_factory=lambda: {
        "boring_intro":      0.20,
        "late_hook":         0.15,
        "context_dependency":0.12,
        "explanation_drift": 0.10,
        "false_peak":        0.08,
    })

    # Маркеры хуков
    hook_markers: Dict[str, List[str]] = field(default_factory=lambda: {
        "question_markers": [
            "что если", "как", "почему", "знаешь ли", "угадай",
            "можешь ли", "представь", "а что", "когда", "куда",
        ],
        "intrigue_markers": [
            "секрет", "ошибка", "никто не знает", "шок", "скрывают",
            "не расскажут", "правда", "раскрываю", "впервые",
        ],
        "fact_markers": [
            "раз", "никогда", "всегда", "статистика",
            "исследование", "доказано", "факт", "данные",
            r"\b\d+\s*%", r"\b\d+x\b", r"\b\d+\+\b",
        ],
        "emotion_markers": [
            "удивлен", "шокирован", "не поверишь", "невероятно",
            "обалдеть", "вау", "ужас", "восторг",
        ],
        "promise_markers": [
            "узнаешь", "покажу", "расскажу", "научу", "сэкономишь",
            "заработаешь", "изменит", "результат",
        ],
        "viral_markers": [
            "тренд", "челлендж", "вирусное", "взорвало", "топ",
            "trending", "challenge", "viral",
        ],
        # v2.0: new marker families
        "warning_markers": [
            "осторожно", "не делайте", "стоп", "внимание", "опасно",
            "запрещено", "нельзя", "хватит",
        ],
        "contrarian_markers": [
            "на самом деле", "вопреки", "все думают что", "а вот нет",
            "но это миф", "ошибаются", "неправильно", "заблуждение",
        ],
        "reveal_markers": [
            "сейчас покажу", "вот как", "вот почему", "объясню",
            "раскрою", "открою", "покажу как", "вот что",
        ],
        # Penalty detection markers (negative signals)
        "boring_intro_markers": [
            "привет всем", "всем привет", "добрый день",
            "сегодня поговорим", "в этом видео я", "меня зовут",
            "подписывайтесь", "итак начнём",
        ],
        "explanation_markers": [
            "давайте разберём", "начнём с того что", "для начала",
            "прежде чем", "сначала нужно понять", "по сути",
            "это значит что", "иными словами",
        ],
        "context_dep_markers": [
            r"\bэтот\b", r"\bэто\b", r"\bтот\b", r"\bтакой\b",
            r"\bони\b", r"\bон\b", r"\bона\b",
        ],
        "resolution_markers": [
            "потому что", "так как", "оказывается", "вот почему",
            "следовательно", "значит", "итак",
        ],
        "suspension_markers": [
            r"\bно\b", r"\bоднако\b", "а вот", "но вот что",
            "но есть одно", "и вот",
        ],
    })

    def __post_init__(self) -> None:
        self._marker_patterns: Dict[str, List[Pattern]] = _compile_marker_patterns(
            self.hook_markers
        )

    @classmethod
    def from_yaml(cls, path: str) -> "HookModeConfig":
        if yaml is None:
            raise ImportError("PyYAML не установлен: pip install pyyaml")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls._from_dict(data)

    @classmethod
    def from_json(cls, path: str) -> "HookModeConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: Dict) -> "HookModeConfig":
        import dataclasses
        known = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    def to_yaml(self, path: str) -> None:
        if yaml is None:
            raise ImportError("PyYAML не установлен: pip install pyyaml")
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(asdict(self), f, allow_unicode=True, default_flow_style=False)

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)


# =============================================================================
# HOOK TYPES  (v2.0: extended)
# =============================================================================

HOOK_TYPES: Dict[str, Dict[str, Any]] = {
    "question_hook": {
        "weight": 1.20,
        "description": "Вопрос останавливает скролл",
        "example": "Что если я скажу, что 70% делают это неправильно?",
    },
    "curiosity_hook": {
        "weight": 1.30,
        "description": "Любопытство + незавершённость — зритель хочет узнать ответ",
        "example": "Одна деталь, которую все упускают. И она меняет всё.",
    },
    "intrigue_hook": {
        "weight": 1.35,
        "description": "Интрига/секрет вызывает любопытство",
        "example": "Одна ошибка убивает 70% Reels. Никто об этом не говорит.",
    },
    "fact_bomb": {
        "weight": 1.40,
        "description": "Неожиданный факт/статистика",
        "example": "За 10 часов я увеличил охват в 10 раз. Вот как.",
    },
    "emotional_hook": {
        "weight": 1.10,
        "description": "Эмоция (шок/радость/удивление)",
        "example": "Это изменение сэкономило мне часы — попробуйте!",
    },
    "promise_hook": {
        "weight": 1.25,
        "description": "Обещание ценности",
        "example": "За 5 минут покажу, как удвоить продуктивность",
    },
    "viral_tease": {
        "weight": 1.15,
        "description": "Связь с трендом/челленджем",
        "example": "Этот тренд взорвал TikTok. Пробую сам.",
    },
    "warning_hook": {
        "weight": 1.20,
        "description": "Предупреждение — триггер защитной реакции",
        "example": "Стоп. Не делайте это перед публикацией.",
    },
    "contrarian_hook": {
        "weight": 1.30,
        "description": "Контрарный тезис — опровержение общепринятого",
        "example": "На самом деле, всё что вы знаете об этом — неверно.",
    },
    "reveal_hook": {
        "weight": 1.25,
        "description": "Объявление об обнаружении/раскрытии",
        "example": "Сейчас покажу, почему это работает совсем не так.",
    },
    "reaction_hook": {
        "weight": 1.15,
        "description": "Эмоциональная реакция как открывающий кадр",
        "example": "— (strong face reaction, no words needed)",
    },
    "weak_hook": {
        "weight": 0.60,
        "description": "Слабый хук (недостаточно маркеров)",
        "example": "Привет всем, сегодня расскажу...",
    },
    "non_hook": {
        "weight": 0.30,
        "description": "Нет хука",
        "example": "Просто видео",
    },
}

MAX_HOOK_WEIGHT: float = max(v["weight"] for v in HOOK_TYPES.values())  # 1.4


# =============================================================================
# MARKER PATTERNS
# =============================================================================

def _compile_marker_patterns(markers: Dict[str, List[str]]) -> Dict[str, List[Pattern]]:
    """
    Компиляция regex-паттернов с word boundary.
    Raw-regex (содержит '\\') компилируется as-is.
    Обычные строки экранируются и оборачиваются в \\b...\\b.
    """
    patterns: Dict[str, List[Pattern]] = {}
    for key, words in markers.items():
        pats: List[Pattern] = []
        for w in words:
            if "\\" in w:
                pats.append(re.compile(w, flags=re.IGNORECASE))
            else:
                pats.append(re.compile(rf"\b{re.escape(w)}\b", flags=re.IGNORECASE))
        patterns[key] = pats
    return patterns


_DEFAULT_CFG = HookModeConfig()
_MARKER_PATTERNS: Dict[str, List[Pattern]] = _compile_marker_patterns(
    _DEFAULT_CFG.hook_markers
)


def _count_markers(text: str, patterns: List[Pattern]) -> int:
    """Count total findall matches across all patterns."""
    return sum(len(p.findall(text)) for p in patterns)


def _extra_punctuation_boost(text: str) -> float:
    """Буст за ? ! и CAPS-слова."""
    boost = 0.0
    if "?" in text:
        boost += 0.2
    if "!" in text:
        boost += 0.1
    if any(t.isupper() and len(t) >= 3 for t in text.split()):
        boost += 0.1
    return boost


def _temporal_iou(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    start = max(a[0], b[0])
    end = min(a[1], b[1])
    inter = max(0.0, end - start)
    union = max(a[1], b[1]) - min(a[0], b[0])
    return 0.0 if union <= 0.0 else inter / union


def _temporal_nms(moments: List[Dict], iou_thresh: float = 0.6) -> List[Dict]:
    """Temporal Non-Maximum Suppression — убирает дубликаты пересекающихся окон."""
    if len(moments) <= 1:
        return moments
    kept: List[Dict] = []
    for m in moments:
        cur = (m["start"], m["end"])
        if all(_temporal_iou(cur, (k["start"], k["end"])) <= iou_thresh for k in kept):
            kept.append(m)
    return kept


def _get_effective_thresholds(cfg: HookModeConfig) -> Tuple[float, float]:
    if not cfg.loose_hook_mode:
        return cfg.min_hook_score, cfg.threshold
    return (
        max(0.3, cfg.min_hook_score - 0.15),
        max(0.4, cfg.threshold - 0.15),
    )


# =============================================================================
# DETECT HOOK STRUCTURE  (unchanged public API)
# =============================================================================

def detect_hook_structure(
    transcript: str,
    config: Optional[HookModeConfig] = None,
    asr_segments: Optional[List[Dict]] = None,
    window_start: float = 0.0,
    window_end: Optional[float] = None,
    use_time_aware: bool = True,
) -> Dict:
    """
    Детектирование структуры хука по маркерным семьям.

    Returns:
        {
            "has_hook": bool,
            "hook_score": float,          # backward-compat aggregate
            "has_question": bool,
            "has_intrigue": bool,
            "has_fact": bool,
            "has_emotion": bool,
            "has_promise": bool,
            "has_viral": bool,
            "has_warning": bool,          # v2.0
            "has_contrarian": bool,       # v2.0
            "has_reveal": bool,           # v2.0
            "has_continuation": bool,     # v2.0 — suspension at window end
            "dominant_type": str,
            "raw_counts": Dict[str, float],
        }
    """
    cfg = config or HookModeConfig()
    _window_end = window_end if window_end is not None else cfg.hook_window_sec

    _empty = {
        "has_hook": False, "hook_score": 0.0,
        "has_question": False, "has_intrigue": False,
        "has_fact": False, "has_emotion": False,
        "has_promise": False, "has_viral": False,
        "has_warning": False, "has_contrarian": False,
        "has_reveal": False, "has_continuation": False,
        "dominant_type": "non_hook", "raw_counts": {},
    }

    if not transcript or len(transcript.strip()) < 10:
        return _empty

    text_lower = transcript.lower()
    patterns: Dict[str, List[Pattern]] = (
        getattr(cfg, "_marker_patterns", None) or _MARKER_PATTERNS
    )

    # ── Count markers ─────────────────────────────────────────────────────────
    if use_time_aware and asr_segments:
        window_segs = [
            s for s in asr_segments
            if s.get("start", 0) < _window_end and s.get("end", 0) > window_start
        ]
        counts: Dict[str, float] = {k: 0.0 for k in (
            "question", "intrigue", "fact", "emotion", "promise",
            "viral", "warning", "contrarian", "reveal",
        )}
        for seg in window_segs:
            seg_text = (seg.get("text") or "").lower()
            pos_weight = cfg.position_boost if seg.get("start", 0) < 3.0 else 1.0
            counts["question"]   += _count_markers(seg_text, patterns["question_markers"])   * pos_weight
            counts["intrigue"]   += _count_markers(seg_text, patterns["intrigue_markers"])   * pos_weight
            counts["fact"]       += _count_markers(seg_text, patterns["fact_markers"])       * pos_weight
            counts["emotion"]    += _count_markers(seg_text, patterns["emotion_markers"])    * pos_weight
            counts["promise"]    += _count_markers(seg_text, patterns["promise_markers"])    * pos_weight
            counts["viral"]      += _count_markers(seg_text, patterns["viral_markers"])      * pos_weight
            counts["warning"]    += _count_markers(seg_text, patterns["warning_markers"])    * pos_weight
            counts["contrarian"] += _count_markers(seg_text, patterns["contrarian_markers"]) * pos_weight
            counts["reveal"]     += _count_markers(seg_text, patterns["reveal_markers"])     * pos_weight
            counts["emotion"]    += _extra_punctuation_boost(seg.get("text") or "") * pos_weight
    else:
        counts = {
            "question":   float(_count_markers(text_lower, patterns["question_markers"])),
            "intrigue":   float(_count_markers(text_lower, patterns["intrigue_markers"])),
            "fact":       float(_count_markers(text_lower, patterns["fact_markers"])),
            "emotion":    float(_count_markers(text_lower, patterns["emotion_markers"])),
            "promise":    float(_count_markers(text_lower, patterns["promise_markers"])),
            "viral":      float(_count_markers(text_lower, patterns["viral_markers"])),
            "warning":    float(_count_markers(text_lower, patterns["warning_markers"])),
            "contrarian": float(_count_markers(text_lower, patterns["contrarian_markers"])),
            "reveal":     float(_count_markers(text_lower, patterns["reveal_markers"])),
        }
        counts["emotion"] += _extra_punctuation_boost(transcript)

    # ── Normalise ─────────────────────────────────────────────────────────────
    n_words = max(1, len(text_lower.split()))
    norm_factor = max(1.0, (n_words ** 0.5) / 3.0)
    norm = {k: min(1.0, v / norm_factor) for k, v in counts.items()}

    # ── Continuation: does the window end on an open/suspension note? ─────────
    # Look at the last segment (or last 30 chars) for suspension markers and
    # absence of resolution markers.
    last_text = ""
    if use_time_aware and asr_segments:
        sorted_segs = sorted(
            [s for s in asr_segments if s.get("end", 0) <= _window_end + 0.1],
            key=lambda s: s.get("end", 0),
        )
        last_text = (sorted_segs[-1].get("text") or "") if sorted_segs else ""
    else:
        last_text = transcript[-120:]

    last_lower = last_text.lower()
    has_suspension = bool(
        _count_markers(last_lower, patterns.get("suspension_markers", []))
    )
    has_resolution = bool(
        _count_markers(last_lower, patterns.get("resolution_markers", []))
    )
    has_continuation = has_suspension and not has_resolution

    # ── Boolean flags ─────────────────────────────────────────────────────────
    has_question   = norm["question"]   > 0.1
    has_intrigue   = norm["intrigue"]   > 0.1
    has_fact       = norm["fact"]       > 0.1
    has_emotion    = norm["emotion"]    > 0.1
    has_promise    = norm["promise"]    > 0.1
    has_viral      = norm["viral"]      > 0.05
    has_warning    = norm["warning"]    > 0.1
    has_contrarian = norm["contrarian"] > 0.1
    has_reveal     = norm["reveal"]     > 0.1

    # ── Text dominant family (marker-level argmax) ────────────────────────────
    # This is the STRONGEST MARKER FAMILY in the text — a raw lexical signal.
    # It is NOT the hook type.  The hook type is determined by the soft
    # competitive classifier in detect_hook_type().
    #
    # Named `text_dominant_family` to make its limited authority explicit.
    # A legacy alias `dominant_type` is kept for external callers only; all
    # INTERNAL pipeline code must read `text_dominant_family`.
    type_keys = {
        "question": "question", "intrigue": "intrigue", "fact": "fact",
        "emotion": "emotion", "promise": "promise", "viral": "viral",
        "warning": "warning", "contrarian": "contrarian", "reveal": "reveal",
    }
    dominant_key          = max(norm, key=norm.get)
    text_dominant_family  = type_keys.get(dominant_key, dominant_key) + "_hook"
    if max(norm.values()) < 0.1:
        text_dominant_family = "weak_hook"

    # ── Aggregate hook_score (text evidence summary) ──────────────────────────
    # Renamed role: this is now "text evidence strength", not "hook quality".
    # It is kept as hook_score for backward compatibility with downstream
    # systems that read it.  The final hook type and quality are computed
    # independently by detect_hook_type() and _compute_hook_quality().
    hook_score = float(np.clip(
        0.28 * norm["question"]   +
        0.22 * norm["intrigue"]   +
        0.18 * norm["fact"]       +
        0.10 * norm["emotion"]    +
        0.10 * norm["promise"]    +
        0.06 * norm["contrarian"] +
        0.04 * norm["warning"]    +
        0.02 * norm["viral"],
        0.0, 1.0,
    ))

    eff_min, _ = _get_effective_thresholds(cfg)
    if hook_score < eff_min:
        text_dominant_family = "weak_hook"

    # ── Narrow text evidence fields (Stage 3: replaces coarse hook_score usage) ─
    # These are the canonical text evidence signals for downstream logic.
    # Downstream should read THESE rather than the aggregate hook_score.
    # hook_score is kept only as legacy/debug summary.
    #
    # question_open   — question asked with no answering fact in same window
    # promise_open    — promise made with no reveal in same window
    # contrast_signal — contrarian or warning register active
    # reveal_signal   — reveal language present but tension still open
    # resolution_signal — resolution/closure language detected (tension closed)
    question_open    = has_question and not bool(norm.get("fact", 0) > 0.10)
    promise_open     = has_promise  and not has_reveal
    contrast_signal  = has_contrarian or has_warning
    reveal_signal    = has_reveal   and not bool(norm.get("resolution", 0) > 0.10)
    resolution_signal= bool(
        _count_markers(last_lower, patterns.get("resolution_markers", []))
    )

    return {
        "has_hook":             hook_score >= eff_min,
        "hook_score":           round(hook_score, 4),
        "has_question":         has_question,
        "has_intrigue":         has_intrigue,
        "has_fact":             has_fact,
        "has_emotion":          has_emotion,
        "has_promise":          has_promise,
        "has_viral":            has_viral,
        "has_warning":          has_warning,
        "has_contrarian":       has_contrarian,
        "has_reveal":           has_reveal,
        "has_continuation":     has_continuation,
        "text_dominant_family": text_dominant_family,
        # legacy alias — external callers may still read this
        "dominant_type":        text_dominant_family,
        "raw_counts":           {k: round(v, 3) for k, v in norm.items()},
        # ── Narrow text evidence (Stage 3) ────────────────────────────────────
        # Read these in downstream logic instead of the aggregate hook_score.
        "question_open":    question_open,
        "promise_open":     promise_open,
        "contrast_signal":  contrast_signal,
        "reveal_signal":    reveal_signal,
        "resolution_signal":resolution_signal,
    }


# =============================================================================
# DETECT HOOK TYPE  (v2.2: soft competitive scoring, not priority cascade)
# =============================================================================

def detect_hook_type(
    hook_structure: Dict,
    intensity: float = 0.0,
) -> Tuple[str, float]:
    """
    Classify hook type via SOFT COMPETITIVE SCORING.

    v2.2 replaces the v2.0 top-down if-elif cascade with independent
    per-type score functions.  Every type competes simultaneously.  The
    winner is whichever type accumulates the most evidence — no fixed
    priority ordering.

    Benefits over priority cascade:
    - Mixed-signal candidates (contrarian + reaction + reveal) are resolved
      by the balance of evidence, not by which rule fires first.
    - The score gap between top-1 and top-2 drives `confidence`, so the
      system knows when a call is ambiguous.
    - A purely visual hook (reaction_hook, reveal_hook) can win even with
      zero lexical markers, as long as face_prom / visual_int are high.

    Returns (primary_hook_type, confidence).
    """
    norm         = hook_structure.get("raw_counts", {})
    hook_score   = float(hook_structure.get("hook_score",   0.0))   # legacy: kept for weak_hook / non_hook only
    has_cont     = bool(hook_structure.get("has_continuation"))
    has_emotion  = bool(hook_structure.get("has_emotion"))
    visual_int   = float(hook_structure.get("visual_intensity", 0.0))
    face_prom    = float(hook_structure.get("face_prominence",  0.0))
    visual_trend = float(hook_structure.get("visual_trend",     0.0))

    # ── Narrow text evidence (Stage 3) ────────────────────────────────────────
    # Type classifier reads these INSTEAD of aggregate hook_score wherever possible.
    q_open      = bool(hook_structure.get("question_open",   False))
    p_open      = bool(hook_structure.get("promise_open",    False))
    contrast    = bool(hook_structure.get("contrast_signal", False))
    rev_signal  = bool(hook_structure.get("reveal_signal",   False))
    res_signal  = bool(hook_structure.get("resolution_signal", False))

    # ── Event semantics (Stage 7) ─────────────────────────────────────────────
    # Read event_scores from the semantics bridge layer; give boosts to matching types.
    ev = (hook_structure.get("_event_semantics") or {}).get("event_scores", {})

    def n(key: str) -> float:
        return float(norm.get(key, 0.0))

    def es(key: str) -> float:
        """Event semantics score for a given event label."""
        return float(ev.get(key, 0.0))

    # ── Per-type score functions ──────────────────────────────────────────────
    # PRINCIPLE: each type must be justified by its OWN specific evidence.
    # hook_score (aggregate text) is NO LONGER a catch-all fallback here —
    # it only appears in weak_hook / non_hook where it is semantically correct.
    #
    # Evidence hierarchy per type:
    #   PRIMARY   — the defining signal for this type (high weight)
    #   SECONDARY — supporting evidence from another channel (lower weight)
    #   EVENT_BOOST — bonus when event_semantics agrees (from Stage 7)

    type_scores: Dict[str, float] = {}

    # contrarian_hook: explicit counter-framing
    type_scores["contrarian_hook"] = float(np.clip(
        0.60 * n("contrarian") +
        0.20 * n("fact") +
        0.12 * n("warning") +
        0.08 * es("escalation_event"),     # contrast + building tension
        0.0, 1.0,
    ))

    # fact_bomb: stat / research lead
    type_scores["fact_bomb"] = float(np.clip(
        0.70 * n("fact") +
        0.18 * n("intrigue") +
        0.12 * intensity,                  # delivery matters for facts
        0.0, 1.0,
    ))

    # curiosity_hook: intrigue + open gap
    type_scores["curiosity_hook"] = float(np.clip(
        0.45 * n("intrigue") +
        0.30 * (1.0 if has_cont else 0.0) +
        0.15 * n("question") +
        0.10 * es("unresolved_open_event"), # event-level open tension
        0.0, 1.0,
    ))

    # warning_hook: protective trigger
    type_scores["warning_hook"] = float(np.clip(
        0.65 * n("warning") +
        0.20 * n("emotion") +
        0.10 * intensity +
        0.05 * (1.0 if contrast else 0.0),
        0.0, 1.0,
    ))

    # reveal_hook: explicit reveal language + visual confirmation
    type_scores["reveal_hook"] = float(np.clip(
        0.50 * n("reveal") +
        0.20 * (1.0 if rev_signal else 0.0) +      # narrow evidence: reveal still open
        0.18 * float(np.clip((visual_trend + 1.0) / 2.0, 0.0, 1.0)) +
        0.12 * es("reveal_event"),                   # event semantics confirmation
        0.0, 1.0,
    ))

    # intrigue_hook: mystery/secret WITHOUT a more specific type winning
    # Replaced: was 0.60 * intrigue + 0.40 * hook_score (text bias removed)
    # Now: intrigue raw count + continuation + contrast as secondary evidence
    type_scores["intrigue_hook"] = float(np.clip(
        0.55 * n("intrigue") +
        0.25 * (1.0 if has_cont else 0.0) +
        0.12 * (1.0 if contrast else 0.0) +
        0.08 * es("escalation_event"),
        0.0, 1.0,
    ))

    # promise_hook: commitment to deliver value
    # Replaced: was 0.70 * has_promise + 0.30 * hook_score (text bias removed)
    # Now: promise presence + promise is still OPEN + delivery energy
    type_scores["promise_hook"] = float(np.clip(
        0.60 * (1.0 if hook_structure.get("has_promise") else 0.0) +
        0.25 * (1.0 if p_open else 0.0) +            # promise with no payoff yet
        0.15 * intensity,                              # delivery energy
        0.0, 1.0,
    ))

    # question_hook: open question as entry device
    type_scores["question_hook"] = float(np.clip(
        0.60 * (1.0 if hook_structure.get("has_question") else 0.0) +
        0.25 * n("question") +
        0.15 * (1.0 if q_open else 0.0),              # question with no answer yet
        0.0, 1.0,
    ))

    # reaction_hook: VISUAL-FIRST — face/emotion energy without heavy speech
    type_scores["reaction_hook"] = float(np.clip(
        0.45 * face_prom +
        0.25 * intensity +
        0.18 * visual_int +
        0.12 * es("reaction_event"),                   # event semantics: is this a reaction?
        0.0, 1.0,
    ))

    # emotional_hook: audio-emotional burst, may have lexical support
    type_scores["emotional_hook"] = float(np.clip(
        0.45 * intensity +
        0.30 * n("emotion") +
        0.15 * (1.0 if has_emotion else 0.0) +
        0.10 * es("reaction_event"),
        0.0, 1.0,
    ))

    # viral_tease: trend / shareability association
    # Replaced: was 0.70 * has_viral + 0.30 * hook_score (text bias removed)
    # Now: viral marker + delivery (viral hooks need energy to land)
    type_scores["viral_tease"] = float(np.clip(
        0.70 * (1.0 if hook_structure.get("has_viral") else 0.0) +
        0.30 * intensity,
        0.0, 1.0,
    ))

    # weak_hook / non_hook: legitimate use of hook_score as text-summary fallback
    # These are explicitly text-opinion slots — it's correct for them to use it.
    av_signal = max(face_prom, visual_int, intensity)
    type_scores["weak_hook"] = float(np.clip(
        0.50 * hook_score +
        0.50 * av_signal * 0.4,     # weak visual evidence also contributes
        0.0, 1.0,
    )) * 0.6
    type_scores["non_hook"]  = float(np.clip(
        (1.0 - max(hook_score, av_signal)) * 0.5,
        0.0, 1.0,
    ))

    # ── Competitive selection ─────────────────────────────────────────────────
    sorted_types    = sorted(type_scores.items(), key=lambda kv: kv[1], reverse=True)
    hook_type       = sorted_types[0][0]
    top_score       = sorted_types[0][1]
    runner_up       = sorted_types[1] if len(sorted_types) > 1 else ("non_hook", 0.0)
    runner_up_score = runner_up[1]

    # ── secondary_hook_type + type_ambiguity ──────────────────────────────────
    secondary_hook_type = runner_up[0]
    gap = top_score - runner_up_score
    type_ambiguity = round(
        float(np.clip(1.0 - (gap / max(top_score, 1e-6)), 0.0, 1.0)), 3
    )

    hook_structure["_secondary_hook_type"] = secondary_hook_type
    hook_structure["_type_ambiguity"]      = type_ambiguity

    # ── Confidence ────────────────────────────────────────────────────────────
    # event_clarity from Stage 7 contributes: a high-clarity event call means
    # the semantics are unambiguous, which increases overall type confidence.
    event_clarity = float(hook_structure.get("_event_clarity", 0.5))
    type_weight   = float(HOOK_TYPES.get(hook_type, {}).get("weight", 1.0))
    confidence    = float(np.clip(
        0.32 * (gap / 0.40) +
        0.22 * top_score +
        0.18 * (type_weight / MAX_HOOK_WEIGHT) +
        0.12 * (1.0 if has_cont else 0.0) +
        0.10 * (1.0 - type_ambiguity) +
        0.06 * event_clarity,          # semantic clarity bonus
        0.0, 1.0,
    ))

    return hook_type, confidence


# =============================================================================
# LOCAL INTENSITY  (v2.0: per-candidate window, not global hook_end average)
# =============================================================================

def _compute_local_intensity(
    base_analysis: Optional[Dict],
    win_start: float,
    win_end: float,
    video_duration_sec: float,
) -> Tuple[float, float, float]:
    """
    Compute (intensity, visual_intensity, face_prominence) for a specific window.

    Previously _compute_intensity_from_base() averaged over 0..hook_end.
    Now we slice time_series to the [win_start, win_end] range, so a single
    intense moment in the opening doesn't inflate all subsequent window scores.

    Returns:
        intensity        — audio/emotion signal [0..1], fallback 0.5
        visual_intensity — visual energy [0..1], fallback 0.0
        face_prominence  — mean face presence score [0..1], fallback 0.0
    """
    intensity = 0.5
    visual_intensity = 0.0
    face_prominence = 0.0

    if not base_analysis:
        return intensity, visual_intensity, face_prominence

    ts = base_analysis.get("time_series") or {}
    if not isinstance(ts, dict):
        return intensity, visual_intensity, face_prominence

    dur = max(video_duration_sec, 1e-6)

    def _slice(arr: Any) -> Optional[np.ndarray]:
        """Return the array slice corresponding to [win_start, win_end]."""
        if arr is None:
            return None
        a = np.asarray(arr, dtype=float)
        if len(a) == 0:
            return None
        n = len(a)
        i_start = max(0, int(n * win_start / dur))
        i_end   = max(i_start + 1, int(n * win_end / dur))
        return a[i_start:i_end]

    # Emotion intensity
    for key in ("emotion_intensity", "valence", "arousal"):
        sliced = _slice(ts.get(key))
        if sliced is not None and len(sliced) > 0:
            intensity = float(np.clip(np.nanmean(sliced), 0.0, 1.0))
            break

    # Movement as secondary signal
    movement_sliced = _slice(ts.get("movement_intensity"))
    if movement_sliced is not None and len(movement_sliced) > 0:
        motion = float(np.clip(np.nanmean(movement_sliced), 0.0, 1.0))
        intensity = float(np.clip((intensity + motion) / 2.0, 0.0, 1.0))

    # Visual intensity
    vis_arr = ts.get("visual_intensity") or ts.get("visual_emotion")
    vis_sliced = _slice(vis_arr)
    if vis_sliced is not None and len(vis_sliced) > 0:
        visual_intensity = float(np.clip(np.nanmean(vis_sliced), 0.0, 1.0))

    # Face prominence — boost intensity on large face, track for type detection
    face_sliced = _slice(ts.get("face_presence"))
    if face_sliced is not None and len(face_sliced) > 0:
        face_prominence = float(np.clip(np.nanmean(face_sliced), 0.0, 1.0))
        if face_prominence > 0.7:
            intensity = float(np.clip(intensity * 1.15, 0.0, 1.0))

    return intensity, visual_intensity, face_prominence


# =============================================================================
# VISUAL TREND  (v2.1)
# =============================================================================

def _compute_visual_trend(
    base_analysis: Optional[Dict],
    win_start: float,
    win_end: float,
    video_duration_sec: float,
) -> float:
    """
    Compute whether visual intensity is RISING across the candidate window.

    Returns a signed score in [-1, 1]:
        > 0  — visual energy increases toward the end of the window
               (suggests visual reveal building, continuation pulling forward)
        < 0  — visual energy drops (suggests climax already happened)
        ≈ 0  — flat

    Used in _compute_hook_subscores() to enrich the continuation sub-score:
    a rising visual trend is a first-class signal that the hook is incomplete
    and the viewer should stay to see what comes next.
    """
    if not base_analysis:
        return 0.0

    ts  = base_analysis.get("time_series") or {}
    dur = max(video_duration_sec, 1e-6)

    # Prefer visual_intensity, fall back to movement_intensity
    arr_raw = ts.get("visual_intensity") or ts.get("movement_intensity")
    if arr_raw is None:
        return 0.0

    arr = np.asarray(arr_raw, dtype=float)
    n   = len(arr)
    if n < 4:
        return 0.0

    # Slice to candidate window
    i_start = max(0, int(n * win_start / dur))
    i_end   = max(i_start + 2, int(n * win_end / dur))
    sliced  = arr[i_start:i_end]
    if len(sliced) < 4:
        return 0.0

    # Compare mean of first half vs second half of the window
    mid  = len(sliced) // 2
    mean_first  = float(np.nanmean(sliced[:mid]))
    mean_second = float(np.nanmean(sliced[mid:]))

    # Normalise to [-1, 1] with a modest scale factor
    trend = float(np.clip((mean_second - mean_first) / 0.30, -1.0, 1.0))
    return round(trend, 3)


# =============================================================================
# WINDOW TEMPORAL DESCRIPTORS  (v2.3)
# =============================================================================

def _compute_window_temporal_descriptors(
    base_analysis: Optional[Dict],
    win_start: float,
    win_end: float,
    video_duration_sec: float,
) -> Dict[str, float]:
    """
    Compute temporal shape descriptors for the candidate window.

    These go beyond mean-aggregation and capture HOW intensity behaves
    within and around the window — which is often the defining quality of
    a hook moment.

    onset_jump      — signal level at window START vs. the 1-second period
                      BEFORE the window.  A strong positive jump means the
                      hook has a sharp beginning (good for immediacy/delivery).

    peakiness       — max(signal) / mean(signal) within the window.
                      Values > 1.5 indicate a spike rather than a plateau.
                      High peakiness = the window has a clear peak moment,
                      not just sustained energy.

    prewindow_delta — mean signal level in the window vs. mean in the
                      full pre-window period [0, win_start].  Positive =
                      window is hotter than what came before it.

    All values are in [0, ∞) for onset_jump / peakiness, [-1, 1] for delta.
    Callers clip them as needed before use.
    """
    result = {"onset_jump": 0.0, "peakiness": 1.0, "prewindow_delta": 0.0}

    if not base_analysis:
        return result

    ts  = base_analysis.get("time_series") or {}
    dur = max(video_duration_sec, 1e-6)

    # Prefer emotion_intensity as the primary signal; fall back to visual_intensity
    arr_raw = ts.get("emotion_intensity") or ts.get("visual_intensity")
    if arr_raw is None:
        return result

    arr = np.asarray(arr_raw, dtype=float)
    n   = len(arr)
    if n < 4:
        return result

    def t2i(t: float) -> int:
        return max(0, min(n - 1, int(n * t / dur)))

    i_start = t2i(win_start)
    i_end   = max(i_start + 2, t2i(win_end))
    win_arr = arr[i_start:i_end]

    if len(win_arr) < 2:
        return result

    win_mean = float(np.nanmean(win_arr))
    win_max  = float(np.nanmax(win_arr))

    # peakiness: spike vs plateau
    result["peakiness"] = round(win_max / max(win_mean, 1e-6), 3)

    # onset_jump: compare first 20% of window vs 1s before window
    pre_start = max(0, t2i(win_start - 1.0))
    pre_arr   = arr[pre_start:i_start] if i_start > pre_start else None
    win_first = float(np.nanmean(win_arr[:max(1, len(win_arr) // 5)]))

    if pre_arr is not None and len(pre_arr) > 0:
        pre_mean = float(np.nanmean(pre_arr))
        result["onset_jump"] = round(
            float(np.clip(win_first - pre_mean, -1.0, 1.0)), 3
        )

    # prewindow_delta: window mean vs full pre-window mean
    if i_start > 0:
        full_pre_mean = float(np.nanmean(arr[:i_start]))
        result["prewindow_delta"] = round(
            float(np.clip(win_mean - full_pre_mean, -1.0, 1.0)), 3
        )

    # ── Temporal shape features (v2.4) ────────────────────────────────────────
    # time_to_peak: FRACTION [0,1] of the window at which the max occurs.
    #   0 = peak at very start (sharp, punchy hook start)
    #   1 = peak at very end (slow build; less "grabbing" for hook)
    peak_idx = int(np.argmax(win_arr))
    result["time_to_peak"] = round(
        float(peak_idx) / max(len(win_arr) - 1, 1), 3
    )

    # first_spike: mean signal in the FIRST QUARTER of the window.
    #   High = the window starts hot (good for immediacy / reaction hooks)
    q1 = win_arr[:max(1, len(win_arr) // 4)]
    result["first_spike"] = round(float(np.nanmean(q1)), 3)

    # post_peak_drop: how much does the signal drop AFTER the peak?
    #   Positive = a clear spike followed by decay (peaky event)
    #   Near 0  = sustained plateau (not a spike, but an elevated region)
    post_peak = win_arr[peak_idx + 1:] if peak_idx + 1 < len(win_arr) else np.array([win_max])
    post_mean = float(np.nanmean(post_peak)) if len(post_peak) > 0 else win_max
    result["post_peak_drop"] = round(float(np.clip(win_max - post_mean, 0.0, 1.0)), 3)

    return result


# Legacy wrapper — keeps backward compatibility for callers that pass hook_end
def _compute_intensity_from_base(
    base_analysis: Optional[Dict],
    hook_end: float,
    video_duration_sec: float,
) -> Tuple[float, float]:
    """
    Backward-compatible wrapper for _compute_local_intensity.
    Slices from 0 to hook_end.
    """
    intensity, visual_intensity, _ = _compute_local_intensity(
        base_analysis, 0.0, hook_end, video_duration_sec
    )
    return intensity, visual_intensity


# =============================================================================
# SUB-SCORES  (v2.0)
# =============================================================================

def _compute_hook_subscores(
    hook_structure: Dict,
    local_intensity: float,
    visual_intensity: float,
    face_prominence: float,
    win_start: float,
    hook_window_sec: float,
    visual_trend: float = 0.0,
    win_transcript: str = "",
    onset_jump: float = 0.0,
    peakiness: float = 1.0,
    prewindow_delta: float = 0.0,
) -> Dict[str, float]:
    """
    Compute the six semantic sub-scores for one candidate window.

    curiosity     — question + intrigue + continuation gap
    surprise      — fact + contrarian + unexpected signal
    emotional     — emotion markers + audio intensity
    immediacy     — how early in the hook window the signal appears
    continuation  — RICHER: suspension markers + unresolved Q + incomplete
                    promise + rising visual trend + open-punctuation syntax
    delivery      — audio/visual energy, face prominence

    v2.1 changes:
    - `continuation` now uses visual_trend as a first-class signal
    - Incomplete promise (promise without reveal) adds tension
    - Syntax: window ending on comma / dash / ellipsis is a continuation cue
    - Rising visual trend (visual_trend > 0) means something is building
    """
    n = hook_structure.get("raw_counts", {})

    def nc(k: str) -> float:
        return float(n.get(k, 0.0))

    # ── curiosity ─────────────────────────────────────────────────────────────
    curiosity = float(np.clip(
        0.40 * nc("question") +
        0.35 * nc("intrigue") +
        0.25 * (1.0 if hook_structure.get("has_promise") else 0.0),
        0.0, 1.0,
    ))
    if hook_structure.get("has_continuation"):
        curiosity = float(np.clip(curiosity * 1.20, 0.0, 1.0))

    # ── surprise ──────────────────────────────────────────────────────────────
    surprise = float(np.clip(
        0.50 * nc("fact") +
        0.30 * nc("contrarian") +
        0.20 * nc("warning"),
        0.0, 1.0,
    ))

    # ── emotional ─────────────────────────────────────────────────────────────
    emotional = float(np.clip(
        0.50 * local_intensity +
        0.35 * nc("emotion") +
        0.15 * visual_intensity,
        0.0, 1.0,
    ))

    # ── immediacy (v2.3: +onset_jump, +prewindow_delta) ──────────────────────
    # Original: earlier in window → higher.
    # Now also rewards SHARP STARTS: a window that begins with a sudden signal
    # rise (onset_jump > 0) and is hotter than its context (prewindow_delta > 0)
    # gets a bonus even if it appears slightly later in the overall opening.
    position_score = 1.0 - (win_start / max(hook_window_sec, 1.0))
    jump_bonus     = float(np.clip(onset_jump * 0.30, 0.0, 0.20))
    delta_bonus    = float(np.clip(prewindow_delta * 0.20, 0.0, 0.12))
    immediacy = float(np.clip(position_score + jump_bonus + delta_bonus, 0.0, 1.0))

    # ── continuation (Stage 5: tension-type-aware) ────────────────────────────
    #
    # Distinguishes two fundamentally different states:
    #   OPEN TENSION  — hook promises something; viewer MUST continue to get it
    #   CLOSED HOOK   — hook already delivered its payload; no forward pull
    #
    # Seven tension sub-types, each with independent scoring:
    #  A. Unresolved question         — asked but not answered
    #  B. Unresolved reveal           — reveal language present, no closure
    #  C. Unresolved action           — action/promise initiated, no completion
    #  D. Incomplete explanation      — explanation started, no resolution
    #  E. Suspension markers          — explicit linguistic suspension
    #  F. Open-punctuation syntax     — window ends mid-thought
    #  G. Rising visual signal        — visual energy still building

    cont_score = 0.0
    tension_types: List[str] = []

    # A. Unresolved question (reads narrow evidence field directly)
    if hook_structure.get("question_open"):
        cont_score += 0.25
        tension_types.append("unresolved_question")
    elif hook_structure.get("has_question") and not hook_structure.get("has_fact"):
        cont_score += 0.22
        tension_types.append("unresolved_question")

    # B. Unresolved reveal (reads narrow evidence field)
    if hook_structure.get("reveal_signal"):
        cont_score += 0.22
        tension_types.append("unresolved_reveal")

    # C. Unresolved action / incomplete promise (reads narrow evidence field)
    if hook_structure.get("promise_open"):
        cont_score += 0.20
        tension_types.append("unresolved_action")

    # D. Incomplete explanation: explanation started but no resolution signal
    if hook_structure.get("has_intrigue") and not hook_structure.get("resolution_signal"):
        cont_score += 0.12
        tension_types.append("incomplete_explanation")

    # E. Suspension markers at window end
    if hook_structure.get("has_continuation"):
        cont_score += 0.18
        if "unresolved_question" not in tension_types:
            tension_types.append("suspension_marker")

    # F. Open-punctuation syntax (window literally stops mid-thought)
    last_char = win_transcript.rstrip()[-1:] if win_transcript.rstrip() else ""
    if last_char in (",", "—", "–", "…", "-"):
        cont_score += 0.15
        tension_types.append("open_syntax")

    # G. Rising visual trend — visual-origin continuation signal
    if visual_trend > 0.10:
        cont_score += float(np.clip(visual_trend * 0.25, 0.0, 0.15))
        tension_types.append("rising_visual")

    # Resolution penalty: if the window's own text already resolves tension,
    # the hook is "closed" — viewer has nothing to chase.
    if hook_structure.get("resolution_signal"):
        cont_score *= 0.50   # tension still counts but is partially closed

    continuation = float(np.clip(cont_score, 0.0, 1.0))
    # Store tension types for downstream reasoning / quality gate
    hook_structure["_tension_types"] = tension_types
    hook_structure["_tension_closed"] = bool(hook_structure.get("resolution_signal"))

    # ── delivery (v2.3: +peakiness) ───────────────────────────────────────────
    # Peakiness > 1 means the window has a spike rather than a plateau.
    # Hooks that "punch" rather than "sustain" are more attention-grabbing.
    # Contribution is bounded so a spiky-but-weak signal doesn't dominate.
    peakiness_bonus = float(np.clip((peakiness - 1.0) * 0.10, 0.0, 0.12))
    delivery = float(np.clip(
        0.40 * face_prominence +
        0.28 * visual_intensity +
        0.22 * local_intensity +
        0.10 * peakiness_bonus,
        0.0, 1.0,
    ))

    return {
        "curiosity":    round(curiosity,    3),
        "surprise":     round(surprise,     3),
        "emotional":    round(emotional,    3),
        "immediacy":    round(immediacy,    3),
        "continuation": round(continuation, 3),
        "delivery":     round(delivery,     3),
    }


# =============================================================================
# VISUAL-FIRST HOOK SUBTYPE CLASSIFICATION  (Stage 2)
# =============================================================================

def _classify_visual_hook_subtype(
    face_prominence:  float,
    visual_intensity: float,
    local_intensity:  float,
    onset_jump:       float,
    peakiness:        float,
    visual_trend:     float,
    post_peak_drop:   float = 0.0,
    first_spike:      float = 0.0,
    proposal_source:  str = "",
) -> Dict[str, Any]:
    """
    Classify the VISUAL HOOK SUBTYPE from audio/visual signals alone.

    Replaces the single "strong burst" model with four semantically distinct
    visual hook classes.  Each has its own evidence profile.

    reaction_like    — face reacts suddenly; emotion burst; onset is sharp;
                       the event is human-driven (expression, micro-gesture)
                       Key signals: face_prominence HIGH, onset_jump HIGH,
                       local_intensity HIGH, peakiness HIGH

    reveal_like      — something NEW enters the frame or becomes visible;
                       visual_intensity rises (trend > 0); onset is moderate;
                       the event unfolds over time (not a single spike)
                       Key signals: visual_trend HIGH, visual_intensity HIGH,
                       onset_jump moderate, post_peak_drop LOW (sustained)

    interruption_like — abrupt break in visual continuity; very sharp onset;
                        the frame CHANGES suddenly (cut, motion spike);
                        onset_jump is very high, peakiness very high,
                        but face_prominence may be low
                        Key signals: onset_jump VERY HIGH, peakiness VERY HIGH,
                        visual_intensity moderate

    object_onset_like — object or action BEGINS in frame; moderate onset;
                        sustained visual signal; not face-driven;
                        Key signals: visual_intensity MODERATE-HIGH,
                        visual_trend POSITIVE, face_prominence LOW,
                        first_spike moderate

    Returns:
        {
          "visual_subtype":    primary subtype label
          "subtype_confidence": [0..1]
          "subtype_scores":    {all four subtypes: score}
        }
    """
    # ── Per-subtype scores ───────────────────────────────────────────────────
    _FACE_SOURCES = {"face_onset", "face_reaction"}
    face_source_bonus = 0.15 if proposal_source in _FACE_SOURCES else 0.0

    reaction_score = float(np.clip(
        0.40 * face_prominence +
        0.25 * local_intensity +
        0.20 * float(np.clip(onset_jump, 0.0, 1.0)) +
        0.15 * float(np.clip((peakiness - 1.0) / 1.5, 0.0, 1.0)) +
        face_source_bonus,
        0.0, 1.0,
    ))

    # reveal: trend-driven; sustained (low post_peak_drop); visual rise
    reveal_score = float(np.clip(
        0.35 * float(np.clip((visual_trend + 1.0) / 2.0, 0.0, 1.0)) +
        0.30 * visual_intensity +
        0.20 * (1.0 - post_peak_drop) +   # sustained = not a spike
        0.15 * float(np.clip(onset_jump * 0.5, 0.0, 1.0)),
        0.0, 1.0,
    ))

    # interruption: very sharp onset + high peakiness + NOT face-driven
    not_face = float(np.clip(1.0 - face_prominence, 0.0, 1.0))
    interruption_score = float(np.clip(
        0.40 * float(np.clip(onset_jump, 0.0, 1.0)) +
        0.30 * float(np.clip((peakiness - 1.0) / 2.0, 0.0, 1.0)) +
        0.20 * not_face +
        0.10 * visual_intensity,
        0.0, 1.0,
    ))

    # object onset: moderate onset, sustained visual, not face
    object_onset_score = float(np.clip(
        0.35 * visual_intensity +
        0.25 * float(np.clip((visual_trend + 1.0) / 2.0, 0.0, 1.0)) +
        0.20 * not_face +
        0.20 * float(np.clip(first_spike, 0.0, 1.0)),
        0.0, 1.0,
    ))

    subtype_scores = {
        "reaction_like":      round(reaction_score,      3),
        "reveal_like":        round(reveal_score,        3),
        "interruption_like":  round(interruption_score,  3),
        "object_onset_like":  round(object_onset_score,  3),
    }

    best_subtype = max(subtype_scores, key=subtype_scores.get)
    best_score   = subtype_scores[best_subtype]
    sorted_scores = sorted(subtype_scores.values(), reverse=True)
    gap = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else sorted_scores[0]
    subtype_confidence = round(float(np.clip(
        0.60 * (gap / 0.40) + 0.40 * best_score,
        0.0, 1.0,
    )), 3)

    return {
        "visual_subtype":     best_subtype,
        "subtype_confidence": subtype_confidence,
        "subtype_scores":     subtype_scores,
    }


# =============================================================================
# VISUAL-FIRST HOOK SCORE  (v2.4 → v3.0: subtype-aware)
# =============================================================================

def compute_visual_first_hook_score(
    face_prominence:  float,
    visual_intensity: float,
    local_intensity:  float,
    onset_jump:       float,
    peakiness:        float,
    visual_trend:     float,
    prewindow_delta:  float,
    proposal_source:  str = "",
    source_confidence: float = 0.0,
    supporting_modalities: Optional[List[str]] = None,
    time_to_peak: float = 0.5,   # fraction of window [0=instant, 1=at end]
    post_peak_drop: float = 0.0,
    first_spike: float = 0.0,
) -> Dict[str, float]:
    """
    Compute a hook score built PURELY from visual and audio evidence.

    v3.0 (Stage 2): subtype-aware scoring replaces single "burst" model.
    Now answers TWO questions:
      1. "Would this still look like a hook if muted?" (visual_first_hook_score)
      2. "What KIND of visual hook is this?" (visual_subtype)

    The composite score now weights SUBTYPE-SPECIFIC evidence instead of
    treating all visual events as generic "bursts".

    Components:

    delivery_energy   — face + audio + visual presence (raw a/v strength)
    onset_sharpness   — how sharply the event begins (onset_jump + peakiness)
    temporal_position — how early in the window the peak arrives (time_to_peak)
    trend_signal      — is something building? (visual_trend + prewindow_delta)
    multimodal_bonus  — bonus when proposal provenance confirms multiple a/v sensors
    subtype_bonus     — bonus when subtype classification is confident and clear

    Returns:
        visual_first_hook_score [0..1]  — main output
        visual_subtype                  — reaction_like / reveal_like / ...
        subtype_confidence              — how confident is the subtype call
        subtype_scores                  — per-subtype raw scores
        component scores for traceability
    """
    supp = supporting_modalities or []

    # ── delivery_energy ───────────────────────────────────────────────────────
    delivery_energy = float(np.clip(
        0.45 * face_prominence +
        0.30 * local_intensity +
        0.25 * visual_intensity,
        0.0, 1.0,
    ))

    # ── onset_sharpness ───────────────────────────────────────────────────────
    # An abrupt start (high onset_jump) and a spiky signal (peakiness > 1)
    # together indicate a real event rather than a sustained plateau.
    onset_sharpness = float(np.clip(
        0.60 * float(np.clip(onset_jump,    0.0, 1.0)) +
        0.40 * float(np.clip((peakiness - 1.0) / 1.5, 0.0, 1.0)),
        0.0, 1.0,
    ))

    # ── temporal_position ─────────────────────────────────────────────────────
    # A hook that peaks EARLY in its window is more grabbing than one that
    # slowly builds to a climax.  time_to_peak = 0 → score 1.0; = 1 → 0.4.
    temporal_position = float(np.clip(1.0 - 0.60 * time_to_peak, 0.0, 1.0))

    # ── trend_signal ──────────────────────────────────────────────────────────
    # Positive visual_trend: something is still building at clip end (good for
    # continuation hooks).  Positive prewindow_delta: hotter than context.
    trend_signal = float(np.clip(
        0.55 * float(np.clip(visual_trend,      -1.0, 1.0)) * 0.5 + 0.5 +
        0.45 * float(np.clip(prewindow_delta + 0.5, 0.0, 1.0)),
        0.0, 1.0,
    ))

    # ── multimodal_bonus ──────────────────────────────────────────────────────
    # When the PROPOSAL itself came from a visual-first source OR from a hybrid
    # cluster with visual sensors, the a/v evidence is corroborated at the
    # proposal level — not just at the feature level.
    _VAV_SOURCES = {
        "face_onset", "face_reaction", "visual_burst", "visual_onset",
        "motion_onset", "audio_burst", "prosody_burst", "hybrid",
    }
    source_is_av = proposal_source in _VAV_SOURCES
    n_av_support = sum(1 for s in supp if s in _VAV_SOURCES)
    multimodal_bonus = float(np.clip(
        (source_confidence * 0.50 if source_is_av else 0.0) +
        n_av_support * 0.12,
        0.0, 0.35,
    ))

    # ── Visual subtype classification (Stage 2) ────────────────────────────
    subtype_result = _classify_visual_hook_subtype(
        face_prominence=face_prominence,
        visual_intensity=visual_intensity,
        local_intensity=local_intensity,
        onset_jump=onset_jump,
        peakiness=peakiness,
        visual_trend=visual_trend,
        post_peak_drop=post_peak_drop,
        first_spike=first_spike,
        proposal_source=proposal_source,
    )
    # Subtype bonus: a confident, clear subtype call means the evidence is
    # self-consistent (not just "something happened").  Reward that signal.
    subtype_bonus = float(np.clip(
        subtype_result["subtype_confidence"] * 0.12,
        0.0, 0.12,
    ))

    # ── Composite ─────────────────────────────────────────────────────────────
    vfhs = float(np.clip(
        0.28 * delivery_energy  +
        0.23 * onset_sharpness  +
        0.17 * temporal_position +
        0.16 * trend_signal     +
        0.09 * multimodal_bonus +
        0.07 * subtype_bonus,
        0.0, 1.0,
    ))

    result = {
        "visual_first_hook_score": round(vfhs, 3),
        "delivery_energy":         round(delivery_energy,    3),
        "onset_sharpness":         round(onset_sharpness,    3),
        "temporal_position":       round(temporal_position,  3),
        "trend_signal":            round(trend_signal,       3),
        "multimodal_bonus":        round(multimodal_bonus,   3),
        "subtype_bonus":           round(subtype_bonus,      3),
    }
    result.update(subtype_result)
    return result


# =============================================================================
# ANTI-CLASS RISK LAYER  (v2.4)
# =============================================================================

def compute_nonhook_risks(
    hook_structure: Dict,
    win_transcript: str,
    win_start: float,
    hook_window_sec: float,
    local_intensity: float = 0.0,
    visual_intensity: float = 0.0,
    onset_jump: float = 0.0,
    peakiness: float = 1.0,
    prewindow_delta: float = 0.0,
    visual_first_score: float = 0.0,
    config: Optional[HookModeConfig] = None,
) -> Dict[str, float]:
    """
    Estimate the probability that this candidate belongs to a NON-HOOK class.

    Unlike the penalty layer (which applies heuristic deductions), this
    function asks: "which wrong class is most likely here?"

    Returns a dict of risk scores [0..1] for six anti-classes:

    intro_risk         — this sounds like an intro/greeting, not a hook
    explainer_risk     — this looks like the START of an explanation, not a hook
                         with built-in tension (high discourse density, low
                         ambiguity, early resolution tendency)
    context_missing_risk — the hook depends on prior context the viewer
                           does not have (isolated reference without setup)
    resolved_risk      — the tension is already resolved inside the window
                         (promise + payoff in same window = not a hook)
    flat_peak_risk     — there IS a local signal spike, but it has no
                         multi-modal confirmation (visual, audio, text all
                         point in different directions)
    generic_talking_head_risk — nothing distinguishes this window from any
                         other "talking head" moment (moderate face, moderate
                         speech, no distinctive event)

    These risks are separate from (and lower authority than) the main hook
    scoring.  They feed into quality / confidence as additional safety signals.
    Downstream can use them independently: e.g. suppress auto-export if any
    single risk > 0.80.
    """
    cfg      = config or HookModeConfig()
    patterns = getattr(cfg, "_marker_patterns", None) or _MARKER_PATTERNS
    text_lo  = win_transcript.lower()
    words    = text_lo.split()
    n_words  = max(1, len(words))
    n        = hook_structure.get("raw_counts", {})
    hs       = hook_structure

    def rc(k: str) -> float:
        return float(n.get(k, 0.0))

    # ── intro_risk ─────────────────────────────────────────────────────────────
    # Generic hello / self-introduction / channel opening phrasing
    boring_count = _count_markers(text_lo, patterns.get("boring_intro_markers", []))
    intro_risk   = float(np.clip(boring_count * 0.50, 0.0, 1.0))
    # Mitigate if window also has strong hook markers
    if rc("intrigue") > 0.15 or rc("fact") > 0.15 or rc("contrarian") > 0.10:
        intro_risk *= 0.50

    # ── explainer_risk ─────────────────────────────────────────────────────────
    # High explanation-marker density + low tension + no continuation
    expl_count   = _count_markers(text_lo, patterns.get("explanation_markers", []))
    res_count    = _count_markers(text_lo, patterns.get("resolution_markers",  []))
    has_cont     = bool(hs.get("has_continuation"))
    has_tension  = rc("intrigue") > 0.10 or rc("question") > 0.10 or hs.get("has_promise")
    expl_base    = float(np.clip(expl_count * 0.40, 0.0, 0.80))
    expl_risk    = expl_base
    if has_cont or has_tension:
        expl_risk *= 0.45       # explanation + tension ≠ simple explainer
    if res_count > 1:
        expl_risk = min(1.0, expl_risk + 0.20)  # early resolution = explainer

    # ── context_missing_risk ───────────────────────────────────────────────────
    # Anaphoric references in first third without a co-present referent
    first_third  = " ".join(words[:max(1, n_words // 3)])
    ctx_count    = _count_markers(first_third, patterns.get("context_dep_markers", []))
    # Is there also a noun/named entity that could serve as referent?
    has_noun_proxy = any(
        len(w) > 4 and w[0].isupper()     # rough heuristic: capitalised word
        for w in win_transcript.split()[:n_words // 3]
    )
    context_missing_risk = float(np.clip(
        ctx_count * 0.30 * (0.5 if has_noun_proxy else 1.0),
        0.0, 0.80,
    ))

    # ── resolved_risk ──────────────────────────────────────────────────────────
    # The hook "question" or "promise" is answered inside the same window
    has_q_and_fact    = hs.get("has_question")    and hs.get("has_fact")
    has_prm_and_rev   = hs.get("has_promise")     and hs.get("has_reveal")
    resolution_in_win = _count_markers(text_lo, patterns.get("resolution_markers", []))
    resolved_risk = float(np.clip(
        (0.45 if has_q_and_fact  else 0.0) +
        (0.45 if has_prm_and_rev else 0.0) +
        resolution_in_win * 0.20,
        0.0, 1.0,
    ))

    # ── flat_peak_risk ─────────────────────────────────────────────────────────
    # A local event spike exists but there is a MISMATCH between proposal
    # source and the actual evidence in the window.
    #
    # The old version checked "confirmed by text OR a/v".  The new version
    # uses visual_first_hook_score as the visual confirmation signal.  A
    # visual-first proposer with high visual_first_score is self-confirming;
    # a sliding-window candidate with high peakiness but low vfhs is a flat peak.
    active_text_fams = sum(1 for v in n.values() if v >= 0.12)
    has_text_signal  = active_text_fams >= 2 or hook_structure.get("hook_score", 0) >= 0.38
    # visual_first_score is the now-canonical visual confirmation
    has_visual_confirmation = visual_first_score >= 0.40
    has_onset_event  = onset_jump >= 0.18

    is_confirmed = has_text_signal or has_visual_confirmation or has_onset_event

    # Mismatch penalty: proposal from visual source but visual evidence is weak
    _VISUAL_FIRST_SRCS = {
        "face_onset", "face_reaction", "visual_burst", "visual_onset",
        "motion_onset", "audio_burst", "prosody_burst", "hybrid",
    }
    source_name = hook_structure.get("_proposal_source", "")  # attached below if available
    source_mismatch = (
        source_name in _VISUAL_FIRST_SRCS and
        not has_visual_confirmation and
        visual_first_score < 0.28
    )

    flat_peak_risk = float(np.clip(
        (peakiness - 1.0) * 0.22 * (0.0 if is_confirmed else 1.0) +
        (0.30 if source_mismatch else 0.0),
        0.0, 0.80,
    ))

    # ── generic_talking_head_risk ──────────────────────────────────────────────
    # Nothing distinguishes this window: moderate face, moderate speech,
    # no striking event, nothing that would make a viewer stop scrolling.
    is_visually_distinct = (
        visual_intensity >= 0.55 or onset_jump >= 0.20 or
        float(hook_structure.get("face_prominence", 0.0)) >= 0.65
    )
    is_lexically_distinct = (
        hook_structure.get("hook_score", 0.0) >= 0.40 or active_text_fams >= 2
    )
    is_temporally_early = win_start <= hook_window_sec * 0.35
    generic_risk = float(np.clip(
        (0.0 if is_visually_distinct  else 0.35) +
        (0.0 if is_lexically_distinct else 0.35) +
        (0.0 if is_temporally_early   else 0.15),
        0.0, 1.0,
    ))
    # Prewindow delta: if window IS hotter than its context, reduce generic risk
    if prewindow_delta > 0.10:
        generic_risk *= 0.65

    return {
        "intro_risk":           round(intro_risk,           3),
        "explainer_risk":       round(expl_risk,            3),
        "context_missing_risk": round(context_missing_risk, 3),
        "resolved_risk":        round(resolved_risk,        3),
        "flat_peak_risk":       round(flat_peak_risk,       3),
        "generic_talking_head_risk": round(generic_risk,   3),
    }


# =============================================================================
# PENALTY LAYER  (v2.0)
# =============================================================================

def compute_hook_penalties(
    hook_structure: Dict,
    win_transcript: str,
    win_start: float,
    hook_window_sec: float,
    config: Optional[HookModeConfig] = None,
    local_intensity: float = 0.0,
    visual_intensity: float = 0.0,
) -> Dict[str, float]:
    """
    Compute raw (0..1) penalties for a candidate window.

    boring_intro      — generic greeting / self-intro at the very start
    late_hook         — hook starts too far into the hook window
                        (exempted when upstream audio/visual shows build-up)
    context_dependency— anaphoric references WITHOUT antecedent in the SAME
                        window; only counted in the first third of words to
                        avoid penalising natural Russian conversational style
    explanation_drift — expository / tutorial opening language
                        (reduced for contrarian/warning hooks)
    false_peak        — too few concurrent positive signals at borderline score

    Returns:
        Dict[str, float] — each value in [0..1].
    """
    cfg = config or HookModeConfig()
    patterns: Dict[str, List[Pattern]] = (
        getattr(cfg, "_marker_patterns", None) or _MARKER_PATTERNS
    )
    text_lower = win_transcript.lower()
    words = text_lower.split()
    n_words = max(1, len(words))

    # ── boring_intro ──────────────────────────────────────────────────────────
    boring_count = _count_markers(text_lower, patterns.get("boring_intro_markers", []))
    boring_intro = float(np.clip(boring_count * 0.4, 0.0, 1.0))

    # ── late_hook ─────────────────────────────────────────────────────────────
    # Ramps from 0 at win_start=0 to 1.0 at win_start ≥ 60% of window.
    # Exemption: if there is meaningful audio/visual activity BEFORE this
    # window, it may represent an intentional build-up (e.g., "slow burn"
    # opening), not a genuinely late hook.  In that case, halve the penalty.
    late_ratio = win_start / max(hook_window_sec, 1.0)
    late_raw   = float(np.clip((late_ratio - 0.30) / 0.40, 0.0, 1.0))
    # Build-up signal: high audio or visual signal means the pre-window was active
    buildup_signal = max(local_intensity, visual_intensity)
    late_hook = late_raw * (0.5 if buildup_signal >= 0.55 else 1.0)

    # ── context_dependency ────────────────────────────────────────────────────
    # Problem with the naive approach: Russian conversational speech uses
    # pronouns constantly, so counting all pronouns in the window generates
    # many false positives (e.g. penalising "он показал, что 70% ошибаются").
    #
    # Better heuristic: only count pronouns that appear in the FIRST THIRD of
    # the window text — these are the ones most likely to dangle without a
    # referent.  Pronouns later in the window likely refer to something already
    # established within the same window.  Also use a lower scaling factor.
    first_third = " ".join(words[: max(1, n_words // 3)])
    ctx_count   = _count_markers(first_third, patterns.get("context_dep_markers", []))
    # Normalise against first-third length (not full window) and apply
    # a conservative scale: a single pronoun at the very start = ~0.25 penalty.
    first_third_words = max(1, len(first_third.split()))
    context_dependency = float(np.clip(
        ctx_count * 0.25 / (first_third_words ** 0.3),
        0.0, 0.80,   # hard cap at 0.80 — never fully zero a candidate on this alone
    ))

    # ── explanation_drift ─────────────────────────────────────────────────────
    expl_count = _count_markers(text_lower, patterns.get("explanation_markers", []))
    expl_raw   = float(np.clip(expl_count * 0.45, 0.0, 1.0))
    # Contrarian and warning hooks often START with an educational-sounding
    # premise ("на самом деле все делают...") before the punchline.  Halve the
    # penalty for those hook types so we don't over-penalise them.
    # Use text_dominant_family (internal name) — NOT dominant_type (legacy alias).
    text_fam = hook_structure.get("text_dominant_family", hook_structure.get("dominant_type", ""))
    if any(h in text_fam for h in ("contrarian", "warning", "educational")):
        expl_raw *= 0.5
    explanation_drift = expl_raw

    # ── false_peak ────────────────────────────────────────────────────────────
    # Visual-event proposals can carry weak text signals but strong visual ones.
    # The old criterion (just count active text families) would always flag them.
    # Now: if delivery sub-score (visual/audio energy) is high, reduce the
    # false_peak penalty — visual signal IS the concurrent evidence.
    raw_counts    = hook_structure.get("raw_counts", {})
    active_families = sum(1 for v in raw_counts.values() if v >= 0.12)
    hook_score    = float(hook_structure.get("hook_score", 0.0))
    delivery_proxy = max(local_intensity, visual_intensity)

    if active_families == 0 and delivery_proxy < 0.45:
        false_peak = 1.0
    elif active_families <= 1 and hook_score < 0.55 and delivery_proxy < 0.55:
        false_peak = 0.6
    elif active_families <= 1 and delivery_proxy >= 0.55:
        # Strong visual/audio compensates for weak text signal
        false_peak = 0.15
    else:
        false_peak = 0.0

    return {
        "boring_intro":       round(boring_intro,       3),
        "late_hook":          round(late_hook,           3),
        "context_dependency": round(context_dependency,  3),
        "explanation_drift":  round(explanation_drift,   3),
        "false_peak":         round(false_peak,          3),
    }


# =============================================================================
# VIRAL COMPAT  (v2.0: nuanced, replaces binary 0.7/0.3)
# =============================================================================

def _compute_viral_compat(
    hook_structure: Dict,
    win_transcript: str,
    hook_type: str,
) -> float:
    """
    Nuanced viral compatibility score [0..1].

    Components:
        viral_marker  — explicit trend/challenge markers (was the only signal before)
        type_bonus    — fact_bomb and contrarian_hook are inherently shareable
        numeric_bonus — stats and numbers increase credibility + shareability
        brevity_bonus — shorter, punchier windows share better
        novelty       — intrigue + contrarian suggest unexpected angle
    """
    text_lower = win_transcript.lower()

    # Base: viral marker presence
    viral_markers = _DEFAULT_CFG._marker_patterns.get("viral_markers", [])
    viral_count = _count_markers(text_lower, viral_markers)
    viral_marker_score = float(np.clip(viral_count * 0.35, 0.0, 0.6))

    # Type-based shareability
    type_bonus = {
        "fact_bomb":      0.30,
        "contrarian_hook":0.25,
        "curiosity_hook": 0.20,
        "intrigue_hook":  0.18,
        "warning_hook":   0.15,
        "promise_hook":   0.12,
        "reveal_hook":    0.10,
        "question_hook":  0.08,
        "reaction_hook":  0.08,
        "emotional_hook": 0.06,
        "viral_tease":    0.20,
    }.get(hook_type, 0.0)

    # Numeric content (stats → credibility → shares)
    num_matches = len(re.findall(r"\b\d+[\d,\.]*\s*%?", text_lower))
    numeric_bonus = float(np.clip(num_matches * 0.07, 0.0, 0.20))

    # Brevity: very short windows are easier to share / reshoot as hook
    word_count = max(1, len(text_lower.split()))
    brevity_bonus = float(np.clip(1.0 - word_count / 60.0, 0.0, 0.15))

    # Novelty: intrigue + contrarian suggest unexpected take
    raw = hook_structure.get("raw_counts", {})
    novelty = float(np.clip(
        0.5 * raw.get("contrarian", 0.0) + 0.5 * raw.get("intrigue", 0.0),
        0.0, 0.15,
    ))

    score = float(np.clip(
        viral_marker_score + type_bonus + numeric_bonus + brevity_bonus + novelty,
        0.0, 1.0,
    ))
    return round(score, 3)


# =============================================================================
# FINAL SCORE  (v2.0: sub-score weighted blend - penalties)
# =============================================================================

def compute_hook_final_score(
    hook_structure: Dict,
    hook_type: str,
    intensity: float = 0.0,
    viral_compat: float = 0.0,
    config: Optional[HookModeConfig] = None,
    subscores: Optional[Dict[str, float]] = None,
    penalties: Optional[Dict[str, float]] = None,
) -> float:
    """
    Compute final hook score [0..1].

    v2.0 mode (when subscores provided):
        final = Σ(w_i * subscore_i) - Σ(pw_j * penalty_j)

    Legacy mode (subscores=None):
        Uses the original markers/intensity/type_match/viral_compat formula
        for backward compatibility.

    Returns: float [0..1]
    """
    cfg = config or HookModeConfig()

    if subscores is not None:
        # ── v2.0 path ─────────────────────────────────────────────────────────
        w = cfg.weights
        positive = (
            w.get("curiosity",    0.22) * subscores.get("curiosity",    0.0) +
            w.get("surprise",     0.18) * subscores.get("surprise",     0.0) +
            w.get("emotional",    0.15) * subscores.get("emotional",    0.0) +
            w.get("immediacy",    0.12) * subscores.get("immediacy",    0.0) +
            w.get("continuation", 0.13) * subscores.get("continuation", 0.0) +
            w.get("delivery",     0.10) * subscores.get("delivery",     0.0)
        )

        pw = cfg.penalty_weights
        penalty_total = 0.0
        if penalties:
            penalty_total = (
                pw.get("boring_intro",       0.20) * penalties.get("boring_intro",       0.0) +
                pw.get("late_hook",          0.15) * penalties.get("late_hook",           0.0) +
                pw.get("context_dependency", 0.12) * penalties.get("context_dependency",  0.0) +
                pw.get("explanation_drift",  0.10) * penalties.get("explanation_drift",   0.0) +
                pw.get("false_peak",         0.08) * penalties.get("false_peak",          0.0)
            )

        return float(np.clip(positive - penalty_total, 0.0, 1.0))

    # ── Legacy path (unchanged from v1.x) ─────────────────────────────────────
    w = cfg.weights
    hook_score  = float(hook_structure.get("hook_score", 0.0))
    type_weight = float(HOOK_TYPES.get(hook_type, {}).get("weight", 1.0))
    type_norm   = float(np.clip(type_weight / MAX_HOOK_WEIGHT, 0.0, 1.0))

    return float(np.clip(
        w["markers"]      * hook_score  +
        w["intensity"]    * intensity   +
        w["type_match"]   * type_norm   +
        w["viral_compat"] * viral_compat,
        0.0, 1.0,
    ))


# =============================================================================
# EXPORT DECISION  (v2.2: operational policy gate)
# =============================================================================

def _export_decision(
    quality: Dict[str, float],
    subscores: Optional[Dict[str, float]] = None,
    proposal_source: str = "sliding",
    hook_type: str = "",
) -> str:
    """
    Translate quality metrics into an OPERATIONAL decision.

    Returns one of three machine-readable labels:

    "auto_export"   — high confidence, low FPR; safe to export automatically
    "manual_review" — ambiguous; a human should decide
    "reject"        — below minimum quality threshold; do not export

    These are the thresholds:

    AUTO-EXPORT requires ALL of:
      hook_confidence   >= 0.68
      false_positive_risk <= 0.32
      asr_confidence    >= 0.45  (transcript must be usable)
      evidence_breadth  >= 0.40  (at least two signal dimensions active)

    REJECT if ANY of:
      hook_confidence   <  0.30
      false_positive_risk >= 0.72
      evidence_breadth  <  0.20 AND NOT a visual-first source

    Otherwise: "manual_review"

    Visual-first hook types (reaction_hook, reveal_hook, audio_burst origin)
    are given a slightly relaxed ASR threshold because their primary evidence
    is audio/visual, not textual — low word count is expected.
    """
    conf           = float(quality.get("hook_confidence",          0.0))
    fpr            = float(quality.get("false_positive_risk",      1.0))
    asr            = float(quality.get("asr_confidence",           0.70))
    breadth        = float(quality.get("evidence_breadth",         0.0))
    safety_score   = float(quality.get("auto_export_safety_score", 0.0))
    n_failures     = int(  quality.get("failure_mode_count",       0))

    _VISUAL_FIRST_SOURCES = {
        "face_onset", "face_reaction", "visual_burst",
        "visual_onset", "motion_onset", "audio_burst", "prosody_burst",
    }
    _VISUAL_FIRST_TYPES = {"reaction_hook", "reveal_hook", "emotional_hook"}

    is_visual_first = (
        proposal_source in _VISUAL_FIRST_SOURCES or
        hook_type       in _VISUAL_FIRST_TYPES
    )
    asr_threshold = 0.30 if is_visual_first else 0.45

    # ── REJECT conditions (hard stop) ─────────────────────────────────────────
    if conf < 0.30:
        return "reject"
    if fpr >= 0.72:
        return "reject"
    if breadth < 0.20 and not is_visual_first:
        return "reject"
    # Stage 4: any critical failure mode = immediate reject
    active_failures = quality.get("active_failure_modes", [])
    if "strong_event_closed_tension" in active_failures and fpr >= 0.55:
        return "reject"

    # ── AUTO-EXPORT conditions (Stage 4: safety_score is the primary gate) ────
    cont_preserved = float(quality.get("continuation_preserved", 0.5))
    if (safety_score   >= 0.62 and        # Stage 4: safety score gate
        conf           >= 0.68 and
        fpr            <= 0.32 and
        asr            >= asr_threshold and
        breadth        >= 0.40 and
        cont_preserved >= 0.40 and
        n_failures     == 0):              # Stage 4: zero active failure modes
        return "auto_export"

    # ── Otherwise: manual review ───────────────────────────────────────────────
    return "manual_review"


# =============================================================================
# PER-TYPE AGREEMENT  (v2.3)
# =============================================================================

def _compute_type_agreement(
    hook_structure: Dict,
    hook_type: str,
    intensity: float,
    visual_conf: float,
) -> float:
    """
    Compute modality agreement using per-type expected signal profiles.

    Replaces the coarse 3-family approach (visual-first / text-first / mixed)
    with independent profiles for each of the 12 hook types.

    Model:
      agreement = 0.70 * primary_strength + 0.30 * support_adequacy

    Where:
      primary_strength  — how well the defining signals of this type are present
      support_adequacy  — whether the SUPPORTING modality is at least minimally
                          present (not necessarily strong, just not absent)

    This distinguishes two failure modes that the old approach conflated:
      A) Primary signals are weak  → genuine false-positive risk
      B) Supporting signals absent → one modality is silent (still useful to know)

    For types without explicit visual data (visual_conf < 0.30) this function
    should not be called — the caller returns 0.50 (neutral) in that case.
    """
    n          = hook_structure.get("raw_counts", {})
    hook_score = float(hook_structure.get("hook_score",      0.0))
    face_prom  = float(hook_structure.get("face_prominence",  0.0))
    visual_int = float(hook_structure.get("visual_intensity", 0.0))
    vis_trend  = float(hook_structure.get("visual_trend",     0.0))
    has_cont   = bool(hook_structure.get("has_continuation"))
    has_q      = bool(hook_structure.get("has_question"))
    has_prm    = bool(hook_structure.get("has_promise"))
    has_viral  = bool(hook_structure.get("has_viral"))

    def rc(key: str) -> float:
        return float(n.get(key, 0.0))

    # ── Per-type profiles ─────────────────────────────────────────────────────
    # Each entry: (primary_fn, support_fn, support_threshold)
    #
    # primary_fn:        computes strength of PRIMARY evidence [0,1]
    # support_fn:        computes strength of SUPPORTING evidence [0,1]
    # support_threshold: min support_fn value to count as "adequate" (binary)
    #
    # reaction_hook   — visual-audio primary; text support (minimal ok)
    # reveal_hook     — reveal markers + rising visual; continuation support
    # fact_bomb       — fact markers + delivery; visual adequacy
    # contrarian_hook — contrarian markers + hook_score; some delivery
    # curiosity_hook  — intrigue + continuation; visual presence
    # question_hook   — question present + some delivery; visual presence
    # warning_hook    — warning markers + emotion; delivery
    # emotional_hook  — intensity + emotion; face or text
    # promise_hook    — promise present + hook_score; delivery
    # intrigue_hook   — intrigue + hook_score; visual presence
    # viral_tease     — viral present + hook_score; moderate delivery

    _PROFILES: Dict[str, Tuple] = {
        "reaction_hook": (
            lambda: 0.50 * face_prom + 0.30 * intensity + 0.20 * visual_int,
            lambda: hook_score,
            0.12,   # barely any text signal is ok for a silent reaction hook
        ),
        "reveal_hook": (
            lambda: 0.40 * rc("reveal") + 0.35 * max(0.0, vis_trend) + 0.25 * (1.0 if has_cont else 0.0),
            lambda: visual_int,
            0.20,   # reveal should have at least some visual motion
        ),
        "fact_bomb": (
            lambda: 0.60 * rc("fact") + 0.25 * hook_score + 0.15 * intensity,
            lambda: visual_int,
            0.10,   # visual not required for a pure stats hook
        ),
        "contrarian_hook": (
            lambda: 0.65 * rc("contrarian") + 0.25 * hook_score + 0.10 * intensity,
            lambda: intensity,
            0.20,   # delivery should be at least moderate for contrarian to land
        ),
        "curiosity_hook": (
            lambda: 0.40 * rc("intrigue") + 0.35 * (1.0 if has_cont else 0.0) + 0.25 * rc("question"),
            lambda: visual_int,
            0.10,   # curiosity can be text-only
        ),
        "question_hook": (
            lambda: 0.50 * (1.0 if has_q else 0.0) + 0.30 * intensity + 0.20 * rc("question"),
            lambda: visual_int,
            0.10,   # visual support helpful but not required
        ),
        "warning_hook": (
            lambda: 0.60 * rc("warning") + 0.25 * rc("emotion") + 0.15 * intensity,
            lambda: intensity,
            0.25,   # warning needs some delivery energy to feel urgent
        ),
        "emotional_hook": (
            lambda: 0.45 * intensity + 0.35 * rc("emotion") + 0.20 * (1.0 if hook_structure.get("has_emotion") else 0.0),
            lambda: max(face_prom, hook_score),
            0.18,   # needs either face or text support
        ),
        "promise_hook": (
            lambda: 0.60 * (1.0 if has_prm else 0.0) + 0.25 * hook_score + 0.15 * intensity,
            lambda: intensity,
            0.12,
        ),
        "intrigue_hook": (
            # hook_score removed: read specific intrigue + continuation evidence
            lambda: 0.55 * rc("intrigue") + 0.30 * (1.0 if has_cont else 0.0) + 0.15 * rc("question"),
            lambda: visual_int,
            0.10,
        ),
        "viral_tease": (
            lambda: 0.70 * (1.0 if has_viral else 0.0) + 0.30 * hook_score,
            lambda: intensity,
            0.15,
        ),
    }

    profile = _PROFILES.get(hook_type)
    if profile is None:
        # Unknown or weak/non type: fall back to hook_score as primary
        primary_strength  = float(np.clip(hook_score, 0.0, 1.0))
        support_adequacy  = 1.0 if visual_int >= 0.10 else 0.0
    else:
        primary_fn, support_fn, support_thresh = profile
        primary_strength = float(np.clip(primary_fn(), 0.0, 1.0))
        support_val      = float(np.clip(support_fn(), 0.0, 1.0))
        # Binary adequacy: support is either present (≥ threshold) or not
        support_adequacy = 1.0 if support_val >= support_thresh else (support_val / support_thresh)

    agreement = float(np.clip(
        0.70 * primary_strength + 0.30 * support_adequacy,
        0.0, 1.0,
    ))
    return round(agreement, 3)


# =============================================================================
# MODALITY RELIABILITY  (v2.1)
# =============================================================================

def _compute_modality_reliability(
    base_analysis: Optional[Dict],
    win_transcript: str,
    hook_structure: Dict,
    final_hook_type: str = "",
) -> Dict[str, float]:
    """
    Assess how reliable each modality signal is for this candidate.

    asr_confidence    — how trustworthy the ASR transcript is
                        (proxy: word count, noise markers)
    visual_confidence — how many time_series channels are populated
                        (0 = no visual data; 1 = all key channels present)
    modality_agreement— do text and audio/visual signals broadly agree?
                        High disagreement hints at one being unreliable.

    `final_hook_type` must be the output of detect_hook_type() (soft
    competitive classifier), NOT hook_structure["dominant_type"] which is
    derived from raw marker-family argmax.  Passing the final type ensures
    agreement is evaluated against the correct per-type evidence profile.

    These fields are machine-readable: downstream can suppress auto-export
    when asr_confidence OR visual_confidence falls below a threshold, and
    flag modality_agreement < 0.4 for manual review.
    """
    # ── ASR confidence ────────────────────────────────────────────────────────
    word_count    = len(win_transcript.split())
    noise_markers = any(
        tok in win_transcript.lower()
        for tok in ("[unclear]", "[?]", "[inaudible]", "...", "э-э", "мм-м")
    )
    # At least 6 words to be meaningful; noise markers halve confidence
    asr_conf = float(np.clip(min(1.0, word_count / 8.0), 0.0, 1.0))
    if noise_markers:
        asr_conf *= 0.65

    # ── Visual/audio confidence ───────────────────────────────────────────────
    ts = (base_analysis or {}).get("time_series") or {}
    _KEY_CHANNELS = (
        "emotion_intensity", "face_presence", "visual_intensity",
        "movement_intensity", "valence",
    )
    populated = sum(1 for k in _KEY_CHANNELS if ts.get(k) is not None)
    visual_conf = float(np.clip(populated / len(_KEY_CHANNELS), 0.0, 1.0))

    # ── Modality agreement (per-type profiles, v2.3) ─────────────────────────
    # CRITICAL: use final_hook_type (soft competitive classifier output), NOT
    # hook_structure["dominant_type"] (raw marker-family argmax).  The two can
    # differ — e.g. reaction_hook may win the soft competition while the
    # textual dominant_type is "intrigue" — and downstream agreement must
    # reflect the actual hypothesis being tested, not a stale textual proxy.
    # Prefer final_hook_type (soft classifier); fall back to text_dominant_family
    # (renamed from dominant_type); legacy alias last.
    resolved_type = (
        final_hook_type or
        hook_structure.get("text_dominant_family") or
        hook_structure.get("dominant_type", "")
    )
    intensity_proxy = float(hook_structure.get("visual_intensity", 0.0))

    if visual_conf < 0.30:
        agreement = 0.50   # insufficient data; neutral
    else:
        agreement = _compute_type_agreement(
            hook_structure, resolved_type, intensity_proxy, visual_conf
        )

    return {
        "asr_confidence":     round(asr_conf,   3),
        "visual_confidence":  round(visual_conf, 3),
        "modality_agreement": round(agreement,   3),
    }


# =============================================================================
# QUALITY / CONFIDENCE OUTPUT  (v2.0)
# =============================================================================

def _compute_hook_quality(
    hook_structure: Dict,
    hook_type: str,
    subscores: Dict[str, float],
    penalties: Dict[str, float],
    final_score: float,
    intensity: float,
    viral_compat: float,
    boundary_quality: float,
    modality_reliability: Optional[Dict[str, float]] = None,
    proposal_source: str = "",
    source_confidence: float = 0.0,
    supporting_modalities: Optional[List[str]] = None,
    visual_first_hook_score: float = 0.0,
    type_ambiguity: float = 0.0,
    event_clarity: float = 0.5,
) -> Dict[str, float]:
    """
    Compute quality / confidence fields for one hook candidate.

    hook_confidence     — overall quality [0..1]
    false_positive_risk — risk this is not a real hook [0..1]
    continuation_tension— unresolvedness (continuation sub-score)
    delivery_strength   — audio/visual delivery quality
    boundary_quality    — how clean start/end edges are
    asr_confidence      — trustworthiness of ASR transcript
    visual_confidence   — coverage of time_series channels
    modality_agreement  — text vs audio/visual signal alignment
    evidence_breadth    — fraction of sub-score dimensions meaningfully active

    v2.1: hook_confidence and false_positive_risk now account for modality
    reliability.  A candidate with high final_score but low asr_confidence
    or modality_agreement will correctly show reduced hook_confidence.
    """
    supp = supporting_modalities or []

    # ── Sub-score evidence breadth ────────────────────────────────────────────
    # Old: fraction of subscores that are meaningfully non-zero.
    # These subscores partially share underlying signals, so breadth can be
    # inflated.  We keep it but also compute a separate proposal_evidence_breadth.
    active  = sum(1 for v in subscores.values() if v >= 0.15)
    breadth = float(np.clip(active / 4.0, 0.0, 1.0))

    # ── Proposal provenance breadth ───────────────────────────────────────────
    # Measures how INDEPENDENTLY the candidate was corroborated at PROPOSAL
    # level — not just at the feature-extraction level.
    # A hybrid candidate with 3 modality families has breadth=0.75; a
    # sliding-window candidate with no visual support has breadth≈0.0.
    _FAMILY_MAP = {
        "text_onset": "text", "sliding": "text",
        "face_onset": "face", "face_reaction": "face",
        "visual_burst": "visual", "visual_onset": "visual",
        "motion_onset": "motion",
        "audio_burst": "audio", "prosody_burst": "audio",
    }
    proposal_families = {_FAMILY_MAP.get(proposal_source, "text")}
    proposal_families |= {_FAMILY_MAP.get(s, "other") for s in supp}
    proposal_evidence_breadth = round(
        float(np.clip(len(proposal_families) / 4.0, 0.0, 1.0)), 3
    )
    # proposal_confidence: how strong is the proposer's own evidence signal?
    # Boosted by visual_first_hook_score for visual-first sources.
    _VAV = {"face_onset", "face_reaction", "visual_burst", "visual_onset",
            "motion_onset", "audio_burst", "prosody_burst", "hybrid"}
    proposal_conf_base = float(np.clip(source_confidence, 0.0, 1.0))
    if proposal_source in _VAV:
        proposal_conf_base = float(np.clip(
            0.60 * proposal_conf_base + 0.40 * visual_first_hook_score, 0.0, 1.0
        ))

    # ── Penalty load ──────────────────────────────────────────────────────────
    total_penalty = sum(penalties.values()) / max(len(penalties), 1)

    # ── Modality reliability ──────────────────────────────────────────────────
    mr = modality_reliability or {}
    asr_conf      = float(mr.get("asr_confidence",     0.70))
    visual_conf   = float(mr.get("visual_confidence",  0.50))
    mod_agreement = float(mr.get("modality_agreement", 0.50))
    reliability   = float(np.clip(
        0.40 * asr_conf + 0.30 * visual_conf + 0.30 * mod_agreement, 0.0, 1.0
    ))

    # ── Combined evidence signal ───────────────────────────────────────────────
    # text_evidence_strength is the text hook signal (renamed role)
    text_ev = float(hook_structure.get("hook_score", 0.0))
    # evidence = max of text and visual paths (not average — either can carry)
    combined_evidence = float(np.clip(
        max(text_ev, visual_first_hook_score) * 0.70 +
        min(text_ev, visual_first_hook_score) * 0.30,
        0.0, 1.0,
    ))

    # ── hook_confidence ───────────────────────────────────────────────────────
    hook_confidence = float(np.clip(
        0.28 * final_score +
        0.18 * combined_evidence +
        0.14 * (1.0 - total_penalty) +
        0.12 * reliability +
        0.10 * proposal_evidence_breadth +
        0.10 * proposal_conf_base +
        0.05 * boundary_quality +
        0.03 * (1.0 - type_ambiguity),  # ambiguous type = lower confidence
        0.0, 1.0,
    ))

    # ── false_positive_risk ───────────────────────────────────────────────────
    false_positive_risk = float(np.clip(
        (1.0 - breadth)                  * 0.25 +
        total_penalty                    * 0.20 +
        (1.0 - combined_evidence)        * 0.20 +
        (1.0 - reliability)              * 0.15 +
        (1.0 - proposal_evidence_breadth)* 0.12 +
        type_ambiguity                   * 0.08,
        0.0, 1.0,
    ))

    # ── Explicit failure modes (Stage 4) ─────────────────────────────────────
    # Each failure mode is a named reason the system should NOT auto-export.
    # These are machine-readable and can drive UI warnings / hard gates.
    #
    # weak_proposal_provenance  — the candidate was proposed by a single low-
    #                             confidence source with no cross-modal support
    # high_type_ambiguity       — two hook types are equally plausible; the call
    #                             is unreliable regardless of the score
    # strong_score_weak_agreement — score is high but modality signals disagree
    #                               (one modality is lying, or the other is missing)
    # strong_event_closed_tension — there IS a clear event but its tension is
    #                               already resolved inside the window (it's a
    #                               complete scene, not a hook)
    failure_modes: Dict[str, bool] = {
        "weak_proposal_provenance": (
            proposal_evidence_breadth < 0.26 and
            proposal_conf_base < 0.35
        ),
        "high_type_ambiguity": (
            type_ambiguity > 0.65
        ),
        "strong_score_weak_agreement": (
            final_score >= 0.55 and
            mod_agreement < 0.35 and
            visual_conf >= 0.30
        ),
        "strong_event_closed_tension": (
            final_score >= 0.50 and
            subscores.get("continuation", 0.0) < 0.20
        ),
    }
    active_failures = [k for k, v in failure_modes.items() if v]
    n_failures = len(active_failures)

    # ── auto_export_safety_score (Stage 4) ────────────────────────────────────
    # Single gate signal: [0..1] where 1.0 = fully safe to auto-export.
    # Replaces the set of scattered quality fields as the DEFINITIVE gate.
    # Built from quality dimensions plus failure mode penalties.
    failure_penalty = float(np.clip(n_failures * 0.20, 0.0, 0.60))
    auto_export_safety_score = float(np.clip(
        0.28 * hook_confidence +
        0.18 * (1.0 - false_positive_risk) +
        0.14 * proposal_evidence_breadth +
        0.14 * proposal_conf_base +
        0.10 * mod_agreement +
        0.09 * (1.0 - type_ambiguity) +
        0.07 * event_clarity -              # semantic clarity of the event
        failure_penalty,
        0.0, 1.0,
    ))

    quality: Dict[str, float] = {
        "hook_confidence":          round(hook_confidence,          3),
        "false_positive_risk":      round(false_positive_risk,      3),
        "continuation_tension":     round(subscores.get("continuation", 0.0), 3),
        "delivery_strength":        round(subscores.get("delivery",     0.0), 3),
        "boundary_quality":         round(boundary_quality,         3),
        "evidence_breadth":         round(breadth,                  3),
        "proposal_evidence_breadth":round(proposal_evidence_breadth,3),
        "proposal_confidence":      round(proposal_conf_base,       3),
        "combined_evidence":        round(combined_evidence,        3),
        # Stage 4 additions
        "auto_export_safety_score": round(auto_export_safety_score, 3),
        "failure_mode_count":       n_failures,
        "active_failure_modes":     active_failures,
    }
    if mr:
        quality.update({
            "asr_confidence":     round(asr_conf,      3),
            "visual_confidence":  round(visual_conf,   3),
            "modality_agreement": round(mod_agreement, 3),
        })
    return quality


# =============================================================================
# BOUNDARY REFINEMENT  (v2.3: +visual onset snapping, +prosodic pause)
# =============================================================================

def _refine_hook_boundaries(
    win_start: float,
    win_end: float,
    win_segs: List[Dict],
    min_duration: float,
    snap_sec: float = 0.5,
    base_analysis: Optional[Dict] = None,
    video_duration_sec: float = 0.0,
    tension_types: Optional[List[str]] = None,
    tension_closed: bool = False,
) -> Tuple[float, float, float]:
    """
    Snap window boundaries to phrase / punctuation / ASR-pause / visual-onset
    boundaries.

    v2.3 adds two new snapping sources:

    VISUAL ONSET SNAPPING (start edge)
      Check if a face_presence or visual_intensity onset (rapid rise) falls
      within [win_start - snap_sec, win_start + snap_sec].  If found, prefer
      snapping the start to that onset timestamp — the visual "event" start
      is often a more natural and impactful clip entry than the ASR word
      boundary.  When both ASR and visual onsets are available, prefer the
      one that is earlier (captures more of the buildup).

    PROSODIC PAUSE SNAPPING (start edge)
      ASR segment gaps ≥ 0.25s near win_start signal a natural pause —
      a rhetorically clean entry point.  Prefer starting after such pauses.

    Returns (refined_start, refined_end, boundary_quality [0..1]).
    boundary_quality is now a weighted sum:
      - start snapped to ASR:     +0.25
      - start snapped to visual:  +0.30  (visual onset = stronger montage point)
      - start snapped to pause:   +0.20
      - end snapped to punc.:     +0.25
    Maximum = 1.0 (capped).
    """
    if not win_segs:
        return win_start, win_end, 0.0

    _SENTENCE_END = re.compile(r"[.?!…]+\s*$")

    snapped_start   = win_start
    boundary_q      = 0.0

    # ── 1. Visual onset snapping ──────────────────────────────────────────────
    # Provides a first-class visual event boundary, independent of ASR.
    visual_onset_t: Optional[float] = None
    if base_analysis and video_duration_sec > 0:
        ts  = base_analysis.get("time_series") or {}
        dur = max(video_duration_sec, 1e-6)
        for ts_key in ("face_presence", "visual_intensity"):
            arr_raw = ts.get(ts_key)
            if arr_raw is None:
                continue
            arr = np.asarray(arr_raw, dtype=float)
            n   = len(arr)
            if n < 4:
                continue
            # Look for the onset nearest to win_start within ±snap_sec
            for onset_t in _ts_onsets(
                arr, dur, win_start + snap_sec,
                rise_threshold=0.20, window_samples=2, min_gap_sec=0.5,
            ):
                if abs(onset_t - win_start) <= snap_sec:
                    if visual_onset_t is None or onset_t < visual_onset_t:
                        visual_onset_t = onset_t

    if visual_onset_t is not None:
        snapped_start = max(0.0, visual_onset_t)
        boundary_q   += 0.30

    # ── 2. Prosodic pause snapping ────────────────────────────────────────────
    # Find gaps ≥ 0.25s between consecutive ASR segments near win_start.
    sorted_segs = sorted(win_segs, key=lambda s: s.get("start", 0.0))
    for i in range(len(sorted_segs) - 1):
        gap_start = float(sorted_segs[i].get("end", 0.0))
        gap_end   = float(sorted_segs[i + 1].get("start", gap_start))
        gap_dur   = gap_end - gap_start
        if gap_dur >= 0.25 and abs(gap_end - win_start) <= snap_sec:
            # Start after the pause — clean rhetorical entry
            candidate = gap_end
            # Accept only if it gives an earlier or same start as current
            if visual_onset_t is None and candidate < snapped_start + 0.1:
                snapped_start  = candidate
                boundary_q    += 0.20
            break

    # ── 3. ASR boundary snapping ──────────────────────────────────────────────
    asr_candidates = [
        s for s in win_segs
        if abs(s.get("start", win_start) - win_start) <= snap_sec
    ]
    if asr_candidates and visual_onset_t is None:
        best = min(asr_candidates, key=lambda s: abs(s.get("start", win_start) - win_start))
        asr_snap = float(best.get("start", win_start))
        # Only accept ASR snap if not already visual-snapped
        snapped_start = asr_snap
        boundary_q   += 0.25

    # ── 4. Refine end: latest segment on clean punctuation ────────────────────
    candidates_end = [
        s for s in win_segs
        if s.get("end", win_end) <= win_end + snap_sec
        and _SENTENCE_END.search((s.get("text") or "").strip())
    ]
    snapped_end = win_end
    end_snapped = False
    if candidates_end:
        best_end = max(candidates_end, key=lambda s: s.get("end", 0.0))
        candidate_end = float(best_end.get("end", win_end))
        if candidate_end - snapped_start >= min_duration:
            snapped_end = candidate_end
            end_snapped = True
            boundary_q += 0.25

    # ── Ensure minimum duration ───────────────────────────────────────────────
    if snapped_end - snapped_start < min_duration:
        snapped_end = snapped_start + min_duration

    # ── Boundary diagnostics (v2.4) ───────────────────────────────────────────
    # Return disaggregated quality signals so downstream can reason about
    # WHICH aspect of the boundary is clean and which is not.
    #
    # start_quality       — how confidently the START edge is anchored
    #                       (visual onset > prosodic pause > ASR word boundary)
    # end_quality         — how syntactically clean the END edge is
    #                       (sentence-final punctuation > trailing pause)
    # continuation_preserved — does the clip END before resolving the hook
    #                       tension? (high = viewer still curious at clip end)
    #
    # boundary_quality (scalar) remains for backward compat.

    start_quality = round(min(boundary_q - (0.25 if end_snapped else 0.0), 0.75), 3)
    end_quality   = round(0.25 if end_snapped else 0.0, 3)

    # ── continuation_preserved and end_leakage_risk (Stage 6) ────────────────
    # continuation_preserved: does the window end without resolving tension?
    #   "ends_open" = syntactic signal; but we also look for payoff leakage.
    #
    # end_leakage_risk: the clip ends at a point where the PAYOFF is already
    #   starting to appear — which converts a hook into a mini-payoff clip.
    #   Signals: resolution language in the trailing text, completed question,
    #   reveal marker followed by answer-like language.
    trailing_text = " ".join(s.get("text", "") for s in sorted_segs
                             if s.get("start", 0) >= (snapped_end - 1.5)).strip()
    _RESOLUTION_END = re.compile(r"[.!]+\s*$")
    _OPEN_END       = re.compile(r"[,—–…\-:]+\s*$|$")
    ends_open    = bool(_OPEN_END.search(trailing_text)) and not bool(_RESOLUTION_END.search(trailing_text))

    # Payoff leakage detection: resolution/explanation language near clip end
    trailing_lower = trailing_text.lower()
    _PAYOFF_PATTERNS = re.compile(
        r"\b(потому что|так как|оказывается|вот почему|вот как|следовательно|"
        r"значит|итак|потому|именно поэтому|потому что|правильный ответ|ответ)\b",
        re.IGNORECASE,
    )
    _PAYOFF_COMPLETE = re.compile(r"[.!]\s*$")
    has_payoff_language = bool(_PAYOFF_PATTERNS.search(trailing_lower))
    has_complete_sentence = bool(_PAYOFF_COMPLETE.search(trailing_text))

    # end_leakage_risk: [0..1] — how much payoff is already inside the clip
    end_leakage_risk = float(np.clip(
        (0.40 if has_payoff_language else 0.0) +
        (0.25 if has_complete_sentence and has_payoff_language else 0.0) +
        (0.20 if not ends_open else 0.0),
        0.0, 1.0,
    ))

    # ── continuation_preserved: tension-type-aware (Stage 5 integration) ─────
    # The old proxy: "ends with open punctuation" → 0.8, else 0.3.
    # The new model reads from the _tension_types computed in Stage 5.
    # ACTIVE tension types (unresolved_question, unresolved_reveal, etc.) mean
    # the hook is still pulling the viewer forward — regardless of punctuation.
    # CLOSED tension (tension_closed=True) means the hook resolved itself.
    _tt = tension_types or []
    _HIGH_TENSION = {
        "unresolved_question", "unresolved_reveal", "unresolved_action",
        "incomplete_explanation",
    }
    _LOW_TENSION = {"open_syntax", "rising_visual", "suspension_marker"}

    n_high = sum(1 for t in _tt if t in _HIGH_TENSION)
    n_low  = sum(1 for t in _tt if t in _LOW_TENSION)

    if tension_closed:
        # Hook has already resolved its own tension inside the window
        tension_score = 0.20
    elif n_high >= 2:
        tension_score = 0.92   # strong multi-type open tension
    elif n_high == 1:
        tension_score = 0.75   # one clear unresolved type
    elif n_low >= 1:
        tension_score = 0.55   # syntactic/visual signals only
    elif ends_open:
        tension_score = 0.45   # punctuation-only proxy
    else:
        tension_score = 0.25   # no detectable tension

    # End leakage overrides: payoff language always reduces continuation
    continuation_preserved = round(
        float(np.clip(
            tension_score * (1.0 - end_leakage_risk * 0.70),
            0.10, 0.95,
        )), 3,
    )

    boundary_quality = round(min(boundary_q, 1.0), 3)

    # ── Anchor sources (v2.4) ─────────────────────────────────────────────────
    # Explicit labels for what each edge was snapped to — more informative
    # than the boolean `snapped_to_visual`.
    if visual_onset_t is not None:
        start_anchor_source = "visual_onset"
    elif any(
        abs(gap_end - win_start) <= snap_sec
        for i in range(len(sorted_segs) - 1)
        for gap_end in [float(sorted_segs[i + 1].get("start", win_start))]
        if float(sorted_segs[i].get("end", 0)) - float(sorted_segs[i].get("start", 0)) >= 0
        and float(sorted_segs[i + 1].get("start", gap_end)) - float(sorted_segs[i].get("end", 0)) >= 0.25
    ):
        start_anchor_source = "prosodic_pause"
    elif asr_candidates:
        start_anchor_source = "asr_boundary"
    else:
        start_anchor_source = "none"

    end_anchor_source = "punctuation" if end_snapped else "max_window"

    # visual_snap_quality: if start was snapped to a visual onset, how strong
    # was that onset signal?  Provides a continuous measure vs the boolean.
    if visual_onset_t is not None and base_analysis and video_duration_sec > 0:
        ts  = base_analysis.get("time_series") or {}
        dur = max(video_duration_sec, 1e-6)
        vsnap_strengths = []
        for ts_key in ("face_presence", "visual_intensity"):
            arr_raw = ts.get(ts_key)
            if arr_raw is None:
                continue
            arr = np.asarray(arr_raw, dtype=float)
            n   = len(arr)
            idx = max(0, min(n - 1, int(n * visual_onset_t / dur))) if n > 0 else 0
            vsnap_strengths.append(float(arr[idx]) if n > 0 else 0.0)
        visual_snap_quality = round(
            float(np.nanmean(vsnap_strengths)) if vsnap_strengths else 0.0, 3
        )
    else:
        visual_snap_quality = 0.0

    return snapped_start, snapped_end, boundary_quality, {
        "start_quality":          round(max(start_quality, 0.0), 3),
        "end_quality":            end_quality,
        "continuation_preserved": continuation_preserved,
        "end_leakage_risk":       round(end_leakage_risk, 3),
        "snapped_to_visual":      visual_onset_t is not None,
        "start_anchor_source":    start_anchor_source,
        "end_anchor_source":      end_anchor_source,
        "visual_snap_quality":    visual_snap_quality,
    }


# =============================================================================
# TIME-SERIES PEAK DETECTOR
# =============================================================================

def _ts_peaks(
    ts_arr: Any,
    video_duration_sec: float,
    hook_end: float,
    threshold: float = 0.60,
    min_gap_sec: float = 1.2,
) -> List[float]:
    """
    Find timestamps of local peaks in a time-series array that are:
      - above `threshold`,
      - inside [0, hook_end],
      - at least `min_gap_sec` apart (to avoid near-duplicate proposals).

    Returns list of peak timestamps (seconds).
    Used by _propose_hook_candidates() to generate visual-event proposals.
    """
    if ts_arr is None:
        return []
    arr = np.asarray(ts_arr, dtype=float)
    n = len(arr)
    if n < 3:
        return []

    dur = max(video_duration_sec, 1e-6)
    peaks: List[float] = []

    for i in range(1, n - 1):
        t = i / n * dur
        if t > hook_end:
            break
        # Local max above threshold
        if arr[i] >= threshold and arr[i] >= arr[i - 1] and arr[i] >= arr[i + 1]:
            if not peaks or (t - peaks[-1]) >= min_gap_sec:
                peaks.append(t)

    return peaks


# =============================================================================
# ONSET DETECTOR  (v2.2)
# =============================================================================

def _ts_onsets(
    ts_arr: Any,
    video_duration_sec: float,
    hook_end: float,
    rise_threshold: float = 0.25,
    window_samples: int = 3,
    min_gap_sec: float = 1.2,
) -> List[float]:
    """
    Detect onset timestamps where the time-series signal RISES sharply.

    Unlike _ts_peaks (finds local maxima), _ts_onsets finds the MOMENT a
    signal transitions from low to high — the actual "event onset" that
    captures viewer attention.  It answers the question "when did this
    start happening?" rather than "where is it loudest?".

    Algorithm: forward-backward delta.  For each sample i, compare the mean
    of the next `window_samples` vs the previous `window_samples`.  If the
    rise exceeds `rise_threshold` and no onset was found recently, record i
    as an onset.

    Use cases:
      - face_presence:      face suddenly appears in frame (reaction onset)
      - emotion_intensity:  audio/emotional burst without a lexical marker
      - visual_intensity:   abrupt scene/subject change, reveal
      - movement_intensity: hand-object action starts

    Returns list of onset timestamps in seconds.
    """
    if ts_arr is None:
        return []
    arr = np.asarray(ts_arr, dtype=float)
    n   = len(arr)
    if n < window_samples * 2 + 1:
        return []

    dur    = max(video_duration_sec, 1e-6)
    onsets: List[float] = []

    for i in range(window_samples, n - window_samples):
        t = i / n * dur
        if t > hook_end:
            break
        before = float(np.mean(arr[i - window_samples : i]))
        after  = float(np.mean(arr[i              : i + window_samples]))
        delta  = after - before
        if delta >= rise_threshold:
            if not onsets or (t - onsets[-1]) >= min_gap_sec:
                onsets.append(t)

    return onsets


# =============================================================================
# CANDIDATE PROTO  (v2.4)
# =============================================================================
# Every proposer returns List[Dict] where each dict has this shape.
# Downstream sees both WHERE the candidate came from and WHY.
#
# start / end               — clip window edges
# source                    — proposer identifier
# source_confidence         — how strong is THIS proposer's own evidence [0,1]
# trigger_time              — precise event time inside [start, end]
# trigger_strength          — signal amplitude at trigger_time [0,1]
# supporting_modalities     — list of OTHER sources that fire near trigger_time
# pre_context_stats         — {prewindow_level, onset_jump} at proposal time
#
# Downstream MUST read these fields; backward-compat tuple unpacking is NOT
# supported for new code.

_SOURCE_PRIORITY: Dict[str, int] = {
    "hybrid":        9,
    "face_reaction": 7, "audio_burst":  6, "visual_onset": 6,
    "prosody_burst": 6, "face_onset":   5, "visual_burst": 4,
    "motion_onset":  4, "text_onset":   3, "sliding":      1,
}


def _make_proto(
    start: float,
    end: float,
    source: str,
    trigger_time: float,
    trigger_strength: float,
    source_confidence: float = 0.5,
    supporting_modalities: Optional[List[str]] = None,
    prewindow_level: float = 0.0,
    onset_jump: float = 0.0,
) -> Dict[str, Any]:
    """Construct a rich candidate prototype dict."""
    return {
        "start":                start,
        "end":                  end,
        "source":               source,
        "source_confidence":    round(float(np.clip(source_confidence, 0.0, 1.0)), 3),
        "trigger_time":         round(trigger_time, 3),
        "trigger_strength":     round(float(np.clip(trigger_strength, 0.0, 1.0)), 3),
        "supporting_modalities":supporting_modalities or [],
        "pre_context_stats":    {
            "prewindow_level":  round(prewindow_level, 3),
            "onset_jump":       round(onset_jump, 3),
        },
    }


# =============================================================================
# INDEPENDENT SUB-PROPOSERS  (v2.4)
# =============================================================================

def _text_onset_proposer(
    hook_segs: List[Dict],
    hook_end: float,
    min_hook_duration: float,
    step_sec: float,
    config: HookModeConfig,
) -> List[Dict]:
    """
    Text-driven proposals: sliding windows + segment-anchored onset events.

    Sliding windows are the universal baseline — they cover the opening span
    regardless of what signals exist.  Segment-anchored windows snap to ASR
    segments that contain strong rhetorical markers, giving more precise entry
    points than fixed-step sliding.
    """
    patterns = getattr(config, "_marker_patterns", None) or _MARKER_PATTERNS
    _ONSET_FAMILIES = [
        "intrigue_markers", "fact_markers", "warning_markers",
        "contrarian_markers", "reveal_markers", "emotion_markers",
    ]
    protos: List[Dict] = []

    # Sliding windows
    t = 0.0
    while t + min_hook_duration <= hook_end + 1e-6:
        end_t = min(t + min_hook_duration, hook_end)
        protos.append(_make_proto(
            t, end_t, "sliding",
            trigger_time=t, trigger_strength=0.3, source_confidence=0.3,
        ))
        t += step_sec

    # Marker-anchored onset windows
    for seg in hook_segs:
        seg_text  = (seg.get("text") or "").lower()
        seg_start = float(seg.get("start", 0.0))
        # Compute per-family strengths
        family_strengths = {
            fam: _count_markers(seg_text, patterns.get(fam, []))
            for fam in _ONSET_FAMILIES
        }
        total_strength = sum(family_strengths.values())
        if total_strength == 0:
            continue
        vs = max(0.0, seg_start - 0.25)
        ve = min(vs + min_hook_duration, hook_end)
        if ve - vs < min_hook_duration - 0.1:
            continue
        # source_confidence proportional to number of marker families active
        active_fams = sum(1 for v in family_strengths.values() if v > 0)
        protos.append(_make_proto(
            vs, ve, "text_onset",
            trigger_time=seg_start,
            trigger_strength=float(np.clip(total_strength / 3.0, 0.0, 1.0)),
            source_confidence=float(np.clip(active_fams / 3.0, 0.2, 1.0)),
        ))

    return protos


def _visual_audio_proposer(
    ts: Dict[str, Any],
    video_duration_sec: float,
    hook_end: float,
    min_hook_duration: float,
    base_arr: Optional[Any] = None,
) -> List[Dict]:
    """
    Visual and audio event-driven proposals.

    Two strategies per channel:
      PEAK  — captures the plateau / maximum of a sustained signal
      ONSET — captures the MOMENT the signal rises (the actual event start)

    Onset proposals are generally more informative for hook detection because
    they identify *when something changes* rather than *when it's loudest*.

    Returns proposals with accurate trigger_time and trigger_strength.
    """
    dur = max(video_duration_sec, 1e-6)
    protos: List[Dict] = []

    def _arr_val_at(arr_raw: Any, t: float) -> float:
        """Sample a time-series array at timestamp t."""
        if arr_raw is None:
            return 0.0
        arr = np.asarray(arr_raw, dtype=float)
        idx = max(0, min(len(arr) - 1, int(len(arr) * t / dur)))
        return float(arr[idx])

    # Peak-based: faces, visual intensity, motion
    _PEAK_SOURCES: List[Tuple[str, str, float, float]] = [
        ("face_presence",      "face_onset",   0.65, 0.50),
        ("visual_intensity",   "visual_burst", 0.58, 0.40),
        ("movement_intensity", "motion_onset", 0.60, 0.30),
    ]
    for ts_key, src_name, thresh, pre_roll in _PEAK_SOURCES:
        arr_raw = ts.get(ts_key)
        for peak_t in _ts_peaks(arr_raw, dur, hook_end, threshold=thresh, min_gap_sec=1.2):
            strength = _arr_val_at(arr_raw, peak_t)
            vs = max(0.0, peak_t - pre_roll)
            ve = min(vs + min_hook_duration, hook_end)
            if ve - vs < min_hook_duration - 0.1:
                continue
            # Estimate prewindow level for pre_context_stats
            pre_level = _arr_val_at(arr_raw, max(0.0, vs - 1.0))
            protos.append(_make_proto(
                vs, ve, src_name,
                trigger_time=peak_t, trigger_strength=strength,
                source_confidence=float(np.clip((strength - thresh) / (1.0 - thresh + 1e-6), 0.2, 1.0)),
                prewindow_level=pre_level,
                onset_jump=float(np.clip(strength - pre_level, 0.0, 1.0)),
            ))

    # Onset-based: face reaction, audio/prosody bursts, visual onset
    _ONSET_SOURCES: List[Tuple[str, str, float]] = [
        ("face_presence",     "face_reaction", 0.30),
        ("emotion_intensity", "audio_burst",   0.22),
        ("visual_intensity",  "visual_onset",  0.25),
        ("arousal",           "prosody_burst", 0.20),
    ]
    for ts_key, src_name, rise_thresh in _ONSET_SOURCES:
        arr_raw = ts.get(ts_key)
        for onset_t in _ts_onsets(arr_raw, dur, hook_end,
                                  rise_threshold=rise_thresh, window_samples=3, min_gap_sec=1.2):
            vs = max(0.0, onset_t - 0.15)
            ve = min(vs + min_hook_duration, hook_end)
            if ve - vs < min_hook_duration - 0.1:
                continue
            strength = _arr_val_at(arr_raw, onset_t)
            pre_level = _arr_val_at(arr_raw, max(0.0, onset_t - 1.0))
            protos.append(_make_proto(
                vs, ve, src_name,
                trigger_time=onset_t,
                trigger_strength=float(np.clip(strength, 0.0, 1.0)),
                source_confidence=float(np.clip(rise_thresh * 2.5, 0.2, 0.9)),
                prewindow_level=pre_level,
                onset_jump=float(np.clip(strength - pre_level, 0.0, 1.0)),
            ))

    return protos


def _event_cluster_quality(
    cluster_times: List[float],
    cluster_strengths: List[float],
    cluster_sources: List[str],
    cluster_window_sec: float,
) -> Dict[str, float]:
    """
    Assess the internal quality of a multi-source event cluster.

    Three orthogonal dimensions:

    temporal_tightness  — how close together the events are in time.
                          Tightly packed = likely causally related or
                          part of the same perceptual event.
                          [0=spread across full window, 1=all at same t]

    modality_diversity  — how many DIFFERENT modality families are represented.
                          text + face + audio > text + text + text.
                          Families: {text, face, visual, audio/prosody, motion}
                          [0=single family, 1=all 5 families present]

    strength_consistency— do the events have similar strength?
                          Highly varied = one real event + noise.
                          [0=high variance, 1=all equal strength]

    cluster_quality     — weighted composite [0, 1].
    """
    n = len(cluster_times)
    if n < 2:
        return {"temporal_tightness": 0.5, "modality_diversity": 0.2,
                "strength_consistency": 0.5, "cluster_quality": 0.3}

    # Temporal tightness: 1 when all events are simultaneous
    time_spread = max(cluster_times) - min(cluster_times)
    temporal_tightness = round(
        float(np.clip(1.0 - time_spread / max(cluster_window_sec, 1e-6), 0.0, 1.0)), 3
    )

    # Modality diversity: map source names to families
    _FAMILY_MAP = {
        "text_onset":    "text",
        "sliding":       "text",
        "face_onset":    "face",
        "face_reaction": "face",
        "visual_burst":  "visual",
        "visual_onset":  "visual",
        "motion_onset":  "motion",
        "audio_burst":   "audio",
        "prosody_burst": "audio",
    }
    families = {_FAMILY_MAP.get(s, "other") for s in cluster_sources}
    modality_diversity = round(float(np.clip(len(families) / 4.0, 0.0, 1.0)), 3)

    # Strength consistency: low CV = consistent strengths
    arr = np.array(cluster_strengths, dtype=float)
    mean_s = float(np.mean(arr))
    std_s  = float(np.std(arr)) if len(arr) > 1 else 0.0
    cv     = std_s / max(mean_s, 1e-6)
    strength_consistency = round(float(np.clip(1.0 - cv, 0.0, 1.0)), 3)

    # Composite: tightness and diversity matter most
    cluster_quality = round(float(np.clip(
        0.40 * temporal_tightness +
        0.40 * modality_diversity +
        0.20 * strength_consistency,
        0.0, 1.0,
    )), 3)

    return {
        "temporal_tightness":   temporal_tightness,
        "modality_diversity":   modality_diversity,
        "strength_consistency": strength_consistency,
        "cluster_quality":      cluster_quality,
    }


def _hybrid_event_proposer(
    all_raw_events: List[Dict],
    min_hook_duration: float,
    hook_end: float,
    cluster_window_sec: float = 0.60,
    min_sources: int = 2,
) -> List[Dict]:
    """
    Hybrid proposer: find time points where ≥ min_sources DIFFERENT modalities
    fire within `cluster_window_sec` of each other.

    Each cluster is assessed by `_event_cluster_quality()` — source_confidence
    is scaled by cluster_quality so that two weak coincident events are NOT
    treated the same as one strong face reaction + one strong prosody burst.

    This prevents the hybrid priority (9) from being unconditionally granted
    to low-quality coincidences.

    all_raw_events format: [{t, source, strength}, ...]
    """
    if not all_raw_events:
        return []

    events = sorted(all_raw_events, key=lambda e: e["t"])
    protos: List[Dict] = []
    used_indices: set = set()

    for i, anchor in enumerate(events):
        if i in used_indices:
            continue
        cluster_sources:   List[str]   = [anchor["source"]]
        cluster_times:     List[float] = [anchor["t"]]
        cluster_strengths: List[float] = [anchor["strength"]]
        cluster_indices = {i}

        for j, other in enumerate(events):
            if j <= i or j in used_indices:
                continue
            if abs(other["t"] - anchor["t"]) > cluster_window_sec:
                continue
            if other["source"] not in cluster_sources:
                cluster_sources.append(other["source"])
                cluster_times.append(other["t"])
                cluster_strengths.append(other["strength"])
                cluster_indices.add(j)

        if len(cluster_sources) < min_sources:
            continue

        trigger_t   = float(np.mean(cluster_times))
        vs          = max(0.0, trigger_t - 0.20)
        ve          = min(vs + min_hook_duration, hook_end)
        if ve - vs < min_hook_duration - 0.1:
            continue

        avg_strength = float(np.mean(cluster_strengths))
        cq           = _event_cluster_quality(
            cluster_times, cluster_strengths, cluster_sources, cluster_window_sec
        )
        # source_confidence is gated by cluster_quality:
        # a weak cluster (cq["cluster_quality"] < 0.4) gets lower confidence
        # even if priority=9 in de-dup ordering.
        breadth_bonus = float(np.clip((len(cluster_sources) - 1) * 0.20, 0.0, 0.40))
        src_conf = float(np.clip(
            avg_strength * 0.40 + cq["cluster_quality"] * 0.40 + breadth_bonus,
            0.0, 1.0,
        ))

        proto = _make_proto(
            vs, ve, "hybrid",
            trigger_time=trigger_t,
            trigger_strength=avg_strength,
            source_confidence=src_conf,
            supporting_modalities=cluster_sources,
        )
        proto["event_cluster_quality"] = cq
        protos.append(proto)
        used_indices.update(cluster_indices)

    return protos


# =============================================================================
# PROPOSAL AGGREGATOR  (v2.4: independent sub-proposers + candidate fusion)
# =============================================================================

def _propose_hook_candidates(
    hook_segs: List[Dict],
    hook_end: float,
    min_hook_duration: float,
    step_sec: float,
    config: HookModeConfig,
    base_analysis: Optional[Dict] = None,
    video_duration_sec: float = 0.0,
) -> List[Dict]:
    """
    Aggregate proposals from independent sub-proposers into a de-duplicated
    rich candidate pool.

    Architecture (v2.4):
      1. _text_onset_proposer   → text/discourse-driven candidates
      2. _visual_audio_proposer → visual/audio event-driven candidates
      3. _hybrid_event_proposer → multi-modal coincidence candidates (strongest)

    Candidates from all sources are collected as rich protos (Dicts with full
    provenance fields), then de-duplicated by spatial proximity (< 0.30s start
    AND end overlap), keeping the highest-priority source for each region.

    Returns: List[Dict] with rich prototype fields.
    """
    all_protos: List[Dict] = []

    # ── Sub-proposer 1: text / sliding ───────────────────────────────────────
    all_protos.extend(
        _text_onset_proposer(hook_segs, hook_end, min_hook_duration, step_sec, config)
    )

    # ── Sub-proposer 2: visual / audio ───────────────────────────────────────
    if base_analysis and video_duration_sec > 0:
        ts = base_analysis.get("time_series") or {}
        all_protos.extend(
            _visual_audio_proposer(ts, video_duration_sec, hook_end, min_hook_duration)
        )

        # Collect raw events for hybrid proposer
        dur = max(video_duration_sec, 1e-6)
        raw_events: List[Dict] = []
        _EVENT_CHANNELS = [
            ("face_presence",     "face_reaction", 0.30),
            ("emotion_intensity", "audio_burst",   0.22),
            ("visual_intensity",  "visual_onset",  0.25),
            ("arousal",           "prosody_burst", 0.20),
        ]
        for ts_key, src, rise_thresh in _EVENT_CHANNELS:
            for onset_t in _ts_onsets(
                ts.get(ts_key), dur, hook_end,
                rise_threshold=rise_thresh, window_samples=3, min_gap_sec=0.8,
            ):
                arr_raw = ts.get(ts_key)
                arr = np.asarray(arr_raw, dtype=float) if arr_raw is not None else np.array([])
                n   = len(arr)
                idx = max(0, min(n - 1, int(n * onset_t / dur))) if n > 0 else 0
                strength = float(arr[idx]) if n > 0 else 0.5
                raw_events.append({"t": onset_t, "source": src, "strength": strength})

        # Also add strong text-onset events to the hybrid pool
        patterns = getattr(config, "_marker_patterns", None) or _MARKER_PATTERNS
        for seg in hook_segs:
            seg_text  = (seg.get("text") or "").lower()
            seg_start = float(seg.get("start", 0.0))
            n_markers = sum(
                _count_markers(seg_text, patterns.get(fam, []))
                for fam in ("intrigue_markers", "fact_markers", "warning_markers",
                            "contrarian_markers", "reveal_markers")
            )
            if n_markers > 0:
                raw_events.append({
                    "t": seg_start,
                    "source": "text_onset",
                    "strength": float(np.clip(n_markers / 3.0, 0.0, 1.0)),
                })

        # ── Sub-proposer 3: hybrid coincidences ──────────────────────────────
        all_protos.extend(
            _hybrid_event_proposer(raw_events, min_hook_duration, hook_end)
        )

    # ── De-duplicate: prefer higher-priority sources for overlapping regions ──
    all_protos.sort(
        key=lambda p: _SOURCE_PRIORITY.get(p["source"], 0),
        reverse=True,
    )
    unique: List[Dict] = []
    for p in all_protos:
        if not any(
            abs(p["start"] - u["start"]) < 0.30 and abs(p["end"] - u["end"]) < 0.30
            for u in unique
        ):
            unique.append(p)

    return unique


# =============================================================================
# EVENT-TO-SEMANTICS BRIDGE  (Stage 7)
# =============================================================================

def _compute_event_semantics(
    hook_structure: Dict,
    visual_subtype: str = "",
    subtype_confidence: float = 0.0,
    local_intensity: float = 0.0,
    visual_intensity: float = 0.0,
    face_prominence: float = 0.0,
    visual_trend: float = 0.0,
    onset_jump: float = 0.0,
    proposal_source: str = "",
    source_confidence: float = 0.0,
) -> Dict[str, Any]:
    """
    Translate raw signals into a SEMANTIC EVENT layer.

    This layer sits BETWEEN raw feature extraction and final ranking.
    It answers: "what kind of hook EVENT is this, and why?".

    Instead of passing raw signal values into the type classifier directly,
    we first interpret WHAT IS HAPPENING semantically, then let the type
    classifier and quality layer read those interpretations.

    Event labels (each is a score [0..1], not a binary flag):

    reaction_event        — someone reacts visually/emotionally to something;
                            the event is human-driven, face-present, short.
    reveal_event          — something is being shown or disclosed;
                            rising visual, reveal language, show-don't-tell.
    interruption_event    — visual continuity breaks; abrupt onset; scene cut.
    escalation_event      — tension/energy is building toward something bigger;
                            rising visual trend + contrast or warning signal.
    unresolved_open_event — the clip ends with tension clearly open:
                            question asked, no answer; promise made, no payoff.

    These labels are built from BOTH text evidence (via narrow fields from
    Stage 3) AND visual evidence (via subtype_result from Stage 2).
    No single modality can dominate — both must agree for high confidence.
    """
    n    = hook_structure.get("raw_counts", {})
    hs   = hook_structure

    def rc(k: str) -> float:
        return float(n.get(k, 0.0))

    # ── reaction_event ────────────────────────────────────────────────────────
    # Visual-first: face is dominant, emotion burst, short sharp onset.
    # Text support is optional (reaction hooks can be silent).
    reaction_event = float(np.clip(
        0.40 * face_prominence +
        0.25 * local_intensity +
        0.20 * float(np.clip(onset_jump, 0.0, 1.0)) +
        0.15 * (1.0 if visual_subtype == "reaction_like" and subtype_confidence > 0.4 else 0.0),
        0.0, 1.0,
    ))

    # ── reveal_event ─────────────────────────────────────────────────────────
    # Text reveal markers + visual trend rising + face or visual onset.
    # Both channels must agree for high score.
    text_reveal = float(np.clip(
        0.60 * rc("reveal") + 0.40 * (1.0 if hs.get("reveal_signal") else 0.0),
        0.0, 1.0,
    ))
    visual_reveal = float(np.clip(
        0.50 * float(np.clip((visual_trend + 1.0) / 2.0, 0.0, 1.0)) +
        0.50 * (1.0 if visual_subtype == "reveal_like" and subtype_confidence > 0.3 else 0.0),
        0.0, 1.0,
    ))
    reveal_event = float(np.clip(
        0.45 * text_reveal +
        0.45 * visual_reveal +
        0.10 * source_confidence,
        0.0, 1.0,
    ))

    # ── interruption_event ───────────────────────────────────────────────────
    # Abrupt visual break: high onset_jump, high peakiness, NOT face-driven.
    # No strong text signal expected — this is a montage-level event.
    interruption_event = float(np.clip(
        0.50 * (1.0 if visual_subtype == "interruption_like" and subtype_confidence > 0.35 else 0.0) +
        0.30 * float(np.clip(onset_jump, 0.0, 1.0)) +
        0.20 * visual_intensity,
        0.0, 1.0,
    ))

    # ── escalation_event ─────────────────────────────────────────────────────
    # Energy/tension is building — hook is an opening MOVE, not the peak.
    # Rising visual trend + contrast/warning text signals.
    text_escalation = float(np.clip(
        0.50 * (1.0 if hs.get("contrast_signal") else 0.0) +
        0.30 * rc("intrigue") +
        0.20 * rc("warning"),
        0.0, 1.0,
    ))
    visual_escalation = float(np.clip(
        0.60 * float(np.clip(visual_trend, 0.0, 1.0)) +
        0.40 * float(np.clip(onset_jump * 0.5, 0.0, 1.0)),
        0.0, 1.0,
    ))
    escalation_event = float(np.clip(
        0.50 * text_escalation + 0.50 * visual_escalation,
        0.0, 1.0,
    ))

    # ── unresolved_open_event ─────────────────────────────────────────────────
    # The clip definitively ends with tension OPEN.
    # Sources: narrow text evidence fields (Stage 3) + tension types (Stage 5).
    tension_types = hs.get("_tension_types", [])
    n_tension = len(tension_types)
    tension_closed = bool(hs.get("_tension_closed", False))
    unresolved_open_event = float(np.clip(
        0.35 * (1.0 if hs.get("question_open")  else 0.0) +
        0.25 * (1.0 if hs.get("promise_open")   else 0.0) +
        0.20 * (1.0 if hs.get("reveal_signal")  else 0.0) +
        0.15 * float(np.clip(n_tension / 3.0, 0.0, 1.0)) +
        0.05 * (1.0 if hs.get("has_continuation") else 0.0),
        0.0, 1.0,
    )) * (0.40 if tension_closed else 1.0)

    # ── Dominant event ────────────────────────────────────────────────────────
    event_scores = {
        "reaction_event":     round(reaction_event,     3),
        "reveal_event":       round(reveal_event,       3),
        "interruption_event": round(interruption_event, 3),
        "escalation_event":   round(escalation_event,   3),
        "unresolved_open_event": round(unresolved_open_event, 3),
    }
    dominant_event = max(event_scores, key=event_scores.get)
    dominant_score = event_scores[dominant_event]

    # event_clarity: how clearly does one event label dominate over the others?
    sorted_vals = sorted(event_scores.values(), reverse=True)
    event_clarity = round(float(np.clip(
        (sorted_vals[0] - sorted_vals[1]) / max(sorted_vals[0], 1e-6), 0.0, 1.0
    )), 3) if len(sorted_vals) > 1 else 1.0

    return {
        "event_scores":    event_scores,
        "dominant_event":  dominant_event,
        "dominant_score":  dominant_score,
        "event_clarity":   event_clarity,
    }


# =============================================================================
# FEATURE EXTRACTION  (v2.0)
# =============================================================================

def _extract_hook_features(
    win_start: float,
    win_end: float,
    win_segs: List[Dict],
    win_transcript: str,
    base_analysis: Optional[Dict],
    video_duration_sec: float,
    config: HookModeConfig,
) -> Dict[str, Any]:
    """
    Extract all per-candidate signals in one place.

    Returns:
        hook_structure, local_intensity, visual_intensity, face_prominence,
        visual_trend (v2.1: rising/falling visual signal across the window)
    """
    # Structure
    hook_structure = detect_hook_structure(
        win_transcript,
        config=config,
        asr_segments=win_segs,
        window_start=win_start,
        window_end=win_end,
        use_time_aware=True,
    )

    # Local intensity (per-window, not global average)
    local_intensity, visual_intensity, face_prominence = _compute_local_intensity(
        base_analysis, win_start, win_end, video_duration_sec
    )

    # Visual trend: is the visual signal rising or falling within the window?
    # Positive trend → something is building → strengthens continuation score
    visual_trend = _compute_visual_trend(
        base_analysis, win_start, win_end, video_duration_sec
    )

    # Temporal shape descriptors: onset sharpness, peakiness, prewindow delta
    temporal = _compute_window_temporal_descriptors(
        base_analysis, win_start, win_end, video_duration_sec
    )

    # Attach visual signals to hook_structure for detect_hook_type
    hook_structure["visual_intensity"] = visual_intensity
    hook_structure["face_prominence"]  = face_prominence
    hook_structure["visual_trend"]     = visual_trend

    return {
        "hook_structure":    hook_structure,
        "local_intensity":   local_intensity,
        "visual_intensity":  visual_intensity,
        "face_prominence":   face_prominence,
        "visual_trend":      visual_trend,
        "onset_jump":        temporal["onset_jump"],
        "peakiness":         temporal["peakiness"],
        "prewindow_delta":   temporal["prewindow_delta"],
        "time_to_peak":      temporal.get("time_to_peak",   0.5),
        "first_spike":       temporal.get("first_spike",    0.0),
        "post_peak_drop":    temporal.get("post_peak_drop", 0.0),
    }


# =============================================================================
# REASONS  (v2.0: includes penalty notes)
# =============================================================================

_HOOK_REASONS_FAMILY: Dict[str, str] = {
    "strong_question": "markers",
    "high_intrigue":   "markers",
    "fact_present":    "markers",
    "contrarian":      "markers",
    "warning":         "markers",
    "reveal":          "markers",
    "viral_tie":       "markers",
    "promise_clear":   "semantic",
    "high_intensity":  "audio",
    "visual_peak":     "visual",
    "continuation":    "semantic",
    # Penalties (negative reasons)
    "boring_intro_penalty":  "penalty",
    "late_hook_penalty":     "penalty",
    "explanation_penalty":   "penalty",
    "false_peak_penalty":    "penalty",
}


def build_hook_reasons(
    hook_structure: Dict,
    hook_type: str,
    intensity: float,
    viral_compat: float,
    max_reasons: int = 5,
    subscores: Optional[Dict[str, float]] = None,
    penalties: Optional[Dict[str, float]] = None,
) -> List[Dict]:
    """
    Build structured reason list.  In v2.0 this includes sub-score reasons
    and significant penalties as negative-weight entries.

    Format: [{code, message, weight, family}]
    """
    candidates: List[Dict] = []

    # ── Positive reasons ──────────────────────────────────────────────────────
    if hook_structure.get("has_question"):
        candidates.append({"code": "strong_question", "message": "Сильный вопрос",       "weight": 0.85})
    if hook_structure.get("has_intrigue"):
        candidates.append({"code": "high_intrigue",   "message": "Интрига/секрет",        "weight": 0.90})
    if hook_structure.get("has_contrarian"):
        candidates.append({"code": "contrarian",      "message": "Контрарный тезис",      "weight": 0.88})
    if hook_structure.get("has_warning"):
        candidates.append({"code": "warning",         "message": "Триггер-предупреждение","weight": 0.82})
    if hook_structure.get("has_reveal"):
        candidates.append({"code": "reveal",          "message": "Объявление раскрытия",  "weight": 0.80})
    if hook_structure.get("has_fact"):
        candidates.append({"code": "fact_present",    "message": "Факт/статистика",       "weight": 0.80})
    if hook_structure.get("has_viral"):
        candidates.append({"code": "viral_tie",       "message": "Связь с трендом",       "weight": 0.75})
    if hook_structure.get("has_promise"):
        candidates.append({"code": "promise_clear",   "message": "Обещание ценности",     "weight": 0.70})
    if hook_structure.get("has_continuation"):
        candidates.append({"code": "continuation",    "message": "Незавершённость / gap", "weight": 0.78})

    if intensity > 0.65:
        candidates.append({
            "code": "high_intensity",
            "message": f"Высокая интенсивность ({intensity:.2f})",
            "weight": round(float(np.clip(intensity, 0.0, 1.0)), 2),
        })

    visual_int = float(hook_structure.get("visual_intensity", 0.0))
    if visual_int > 0.6:
        candidates.append({
            "code": "visual_peak",
            "message": f"Визуальный акцент ({visual_int:.2f})",
            "weight": round(float(np.clip(visual_int, 0.0, 1.0)), 2),
        })

    if viral_compat > 0.40:
        candidates.append({
            "code": "viral_compat",
            "message": f"Виральная совместимость ({viral_compat:.2f})",
            "weight": round(float(np.clip(viral_compat, 0.0, 1.0)), 2),
        })

    type_weight = float(HOOK_TYPES.get(hook_type, {}).get("weight", 1.0))
    if type_weight >= 1.25:
        type_desc = HOOK_TYPES[hook_type]["description"]
        candidates.append({
            "code":    f"type_{hook_type}",
            "message": f"{type_desc} (w={type_weight})",
            "weight":  round(type_weight - 1.0, 2),
        })

    # Sub-score highlights
    if subscores:
        if subscores.get("curiosity", 0) > 0.60:
            candidates.append({
                "code": "high_curiosity",
                "message": f"Высокий curiosity ({subscores['curiosity']:.2f})",
                "weight": round(subscores["curiosity"] * 0.7, 2),
            })
        if subscores.get("surprise", 0) > 0.55:
            candidates.append({
                "code": "high_surprise",
                "message": f"Элемент неожиданности ({subscores['surprise']:.2f})",
                "weight": round(subscores["surprise"] * 0.7, 2),
            })

    # ── Penalty reasons (negative weight) ────────────────────────────────────
    if penalties:
        pw = {
            "boring_intro":       ("boring_intro_penalty",  "Скучный интро-паттерн",    -0.15),
            "late_hook":          ("late_hook_penalty",     "Хук поздно в окне",         -0.12),
            "explanation_drift":  ("explanation_penalty",   "Объяснительный старт",      -0.10),
            "false_peak":         ("false_peak_penalty",    "Изолированный маркер",      -0.08),
        }
        for key, (code, msg, base_weight) in pw.items():
            val = float(penalties.get(key, 0.0))
            if val >= 0.3:
                candidates.append({"code": code, "message": msg, "weight": round(base_weight * val, 2)})

    # ── Select top-max_reasons, ≤ 2 per family ────────────────────────────────
    candidates.sort(key=lambda x: abs(x["weight"]), reverse=True)
    family_counts: Dict[str, int] = {}
    result: List[Dict] = []
    markers_added = False

    for c in candidates:
        code = c["code"]
        family = _HOOK_REASONS_FAMILY.get(code, "other")
        if family_counts.get(family, 0) >= 2:
            continue
        # Always include penalty reasons (they inform downstream quality gate)
        result.append({**c, "family": family})
        family_counts[family] = family_counts.get(family, 0) + 1
        if family == "markers":
            markers_added = True
        if len(result) >= max_reasons:
            break

    # Guarantee at least one markers reason if any exist
    if not markers_added:
        for c in candidates:
            fam = _HOOK_REASONS_FAMILY.get(c["code"], "other")
            if fam == "markers" and c not in result:
                result.insert(0, {**c, "family": fam})
                if len(result) > max_reasons:
                    result = result[:max_reasons]
                break

    return result


# =============================================================================
# MOMENT BUILDER  (v2.0)
# =============================================================================

def _make_hook_moment(
    win_start: float,
    win_end: float,
    hook_type: str,
    hook_structure: Dict,
    hook_score: float,
    intensity: float,
    reasons: List[Dict],
    transcript: str,
    config: Optional[HookModeConfig] = None,
    subscores: Optional[Dict[str, float]] = None,
    penalties: Optional[Dict[str, float]] = None,
    quality: Optional[Dict[str, float]] = None,
    viral_compat: float = 0.0,
    proposal_source: str = "sliding",
) -> Dict[str, Any]:
    """Build one hook moment dict."""
    final_score = compute_hook_final_score(
        hook_structure, hook_type, intensity, viral_compat, config,
        subscores=subscores, penalties=penalties,
    )
    summary = transcript[:100] + "..." if len(transcript) > 100 else transcript
    title = f"Хук: {HOOK_TYPES.get(hook_type, {}).get('description', hook_type)}"

    moment: Dict[str, Any] = {
        "start":    float(win_start),
        "end":      float(win_end),
        "duration": float(win_end - win_start),
        "score":    round(final_score, 3),
        "type":     "hook",
        "hook_type":      hook_type,
        "hook_structure": {k: v for k, v in hook_structure.items() if k.startswith("has_")},
        "hook_score":     round(hook_score, 3),
        "intensity":      round(intensity, 3),
        "viral_compat":   round(viral_compat, 3),
        "reasons":        reasons,
        "title":          title,
        "summary":        summary,
        "transcript":     transcript[:300],
        "priority":       "front_hook",
        "priority_score": 1.0,
        "export_title":   f"Hook_{hook_type}_{win_start:.1f}-{win_end:.1f}s_score{final_score:.2f}",
        "proposal_source": proposal_source,
    }

    # v2.0 quality fields (present only when computed)
    if subscores is not None:
        moment["subscores"] = subscores
    if penalties is not None:
        moment["penalties"] = penalties
    if quality is not None:
        moment.update(quality)

    return moment


# =============================================================================
# EMPTY RESULT
# =============================================================================

def _empty_hook_result(
    video_duration_sec: float,
    error: str,
    cfg: Optional[HookModeConfig] = None,
) -> Dict:
    _cfg = cfg or HookModeConfig()
    return {
        "mode": "hook",
        "error": error,
        "hook_moments": [],
        "stats": {
            "total_duration": video_duration_sec,
            "profile_name":   f"{_cfg.mode_name} {_cfg.profile_version}",
            "hook_window_sec": _cfg.hook_window_sec,
            "num_hooks_found": 0,
            "hook_type":      "non_hook",
            "threshold":      _cfg.threshold,
            "avg_score":      0.0,
        },
    }


# =============================================================================
# MAIN ORCHESTRATOR  (v2.0: slim, delegates to pipeline layers)
# =============================================================================

def find_hook_moments(
    video_path: str,
    video_duration_sec: float,
    asr_segments: Optional[List[Dict]] = None,
    base_analysis: Optional[Dict] = None,
    top_k: int = 3,
    config: Optional[HookModeConfig] = None,
) -> Dict:
    """
    Hook Mode v2.0 — finds opening hooks in the first hook_window_sec of video.

    Pipeline:
        1. _propose_hook_candidates()   ← sliding window + text-onset events
        2. _extract_hook_features()     ← per-window local signals
        3. _compute_hook_subscores()    ← 6 semantic dimensions
        4. compute_hook_penalties()     ← 5 penalty signals
        5. compute_hook_final_score()   ← weighted blend - penalties
        6. _refine_hook_boundaries()    ← snap to phrase boundaries
        7. build_hook_reasons()         ← includes sub-scores + penalties
        8. _temporal_nms() → top-K

    Returns:
        {
            "mode": "hook",
            "hook_moments": [
                {
                    start, end, duration, score,
                    type="hook", hook_type,
                    hook_structure, hook_score, intensity,
                    viral_compat, subscores, penalties,
                    hook_confidence, false_positive_risk,
                    continuation_tension, delivery_strength, boundary_quality,
                    reasons, title, summary, transcript,
                    priority, export_title, proposal_source
                },
                ...
            ],
            "stats": { ... }
        }
    """
    logger.info("=" * 70)
    logger.info("HOOK MODE v2.0 — Layered Pipeline")
    logger.info("=" * 70)

    cfg = config or HookModeConfig()
    eff_min_hook_score, eff_threshold = _get_effective_thresholds(cfg)

    # v2.1: AUTO-RELAX для коротких/средних видео (micro-hook)
    #   video < 30s  → min_hook_duration=1.5, threshold -= 0.10  (micro-hook)
    #   30 ≤ dur < 90s → threshold -= 0.05                       (relaxed)
    #   ≥ 90s → стандартные пороги
    micro_hook_active = False
    if video_duration_sec < 30.0:
        micro_hook_active = True
        cfg.min_hook_duration = min(cfg.min_hook_duration, 1.5)
        cfg.hook_window_sec = min(cfg.hook_window_sec, max(video_duration_sec * 0.6, 5.0))
        eff_min_hook_score = max(0.25, eff_min_hook_score - 0.10)
        eff_threshold = max(0.30, eff_threshold - 0.10)
        logger.info(
            f"micro_hook auto-activated (video={video_duration_sec:.1f}s): "
            f"min_duration={cfg.min_hook_duration:.1f}s, "
            f"window={cfg.hook_window_sec:.1f}s, "
            f"min_hook_score={eff_min_hook_score:.2f}, threshold={eff_threshold:.2f}"
        )
    elif video_duration_sec < 90.0:
        eff_min_hook_score = max(0.30, eff_min_hook_score - 0.05)
        eff_threshold = max(0.35, eff_threshold - 0.05)
        logger.info(
            f"medium-video relaxed thresholds: "
            f"min_hook_score={eff_min_hook_score:.2f}, threshold={eff_threshold:.2f}"
        )

    # v2.1: отслеживаем rejected кандидатов для diagnostics
    rejected_hooks: List[Dict] = []
    hook_filter_reasons: Dict[str, int] = {
        "too_short_transcript": 0,
        "low_hook_score": 0,
        "low_final_score": 0,
    }

    if cfg.loose_hook_mode:
        logger.info(
            f"loose_hook_mode: eff_min_hook_score={eff_min_hook_score:.2f}, "
            f"eff_threshold={eff_threshold:.2f}"
        )

    if not asr_segments:
        logger.warning("Hook Mode requires ASR transcription!")
        return _empty_hook_result(video_duration_sec, "no_transcription", cfg)

    hook_end = min(cfg.hook_window_sec, video_duration_sec)

    if hook_end < cfg.min_hook_duration:
        return _empty_hook_result(video_duration_sec, "hook_window_too_short", cfg)

    hook_segs = [s for s in asr_segments if s.get("start", 0) < hook_end]
    if not hook_segs:
        return _empty_hook_result(video_duration_sec, "no_transcript_in_window", cfg)

    full_transcript = " ".join(s.get("text", "") for s in hook_segs).strip()
    if len(full_transcript.split()) < 4:
        return _empty_hook_result(video_duration_sec, "too_short_hook_text", cfg)

    logger.info(f"Profile: {cfg.mode_name} {cfg.profile_version} | Window: 0–{hook_end:.1f}s")

    try:
        # ── 1. Proposals ──────────────────────────────────────────────────────
        proposals = _propose_hook_candidates(
            hook_segs, hook_end, cfg.min_hook_duration, cfg.step_sec, cfg,
            base_analysis=base_analysis,
            video_duration_sec=video_duration_sec,
        )
        # ── Proposal audit (Stage 1) ──────────────────────────────────────────
        # Log per-source counts and raw_events breakdown for audit.
        # This is the primary tool for diagnosing proposer coverage issues:
        # "which event types are being proposed / missed / noisy?"
        src_counts: Dict[str, int] = {}
        for p in proposals:
            src_counts[p["source"]] = src_counts.get(p["source"], 0) + 1
        logger.info(f"Proposals: {len(proposals)} — {src_counts}")
        # Audit: for each visual source, log min/max trigger_strength
        for src_name in ("face_reaction", "audio_burst", "visual_onset",
                         "face_onset", "visual_burst", "motion_onset", "prosody_burst"):
            src_protos = [p for p in proposals if p["source"] == src_name]
            if src_protos:
                strengths = [p.get("trigger_strength", 0.0) for p in src_protos]
                logger.info(
                    f"  [{src_name}] {len(src_protos)} proposals — "
                    f"strength: min={min(strengths):.3f} max={max(strengths):.3f}"
                )
        # Track proposal trigger times for final hook coverage check
        _all_proposal_trigger_times = {
            round(p.get("trigger_time", p["start"]), 2): p["source"]
            for p in proposals
        }

        hook_candidates: List[Dict] = []

        for proto in proposals:
            win_start       = float(proto["start"])
            win_end         = float(proto["end"])
            proposal_source = proto["source"]
            source_conf     = float(proto.get("source_confidence", 0.5))
            trigger_time    = float(proto.get("trigger_time", win_start))
            trigger_strength= float(proto.get("trigger_strength", 0.0))
            support_mods    = proto.get("supporting_modalities", [])
            win_segs = [
                s for s in hook_segs
                if s.get("start", 0) < win_end and s.get("end", 0) > win_start
            ]
            win_transcript = " ".join(s.get("text", "") for s in win_segs).strip()

            if len(win_transcript.split()) < 4:
                hook_filter_reasons["too_short_transcript"] += 1
                rejected_hooks.append({
                    "start": win_start, "end": win_end,
                    "proposal_source": proposal_source,
                    "reject_reason": "too_short_transcript",
                    "word_count": len(win_transcript.split()),
                    "transcript_preview": win_transcript[:80],
                })
                continue

            # ── 2. Feature extraction ─────────────────────────────────────────
            feats = _extract_hook_features(
                win_start, win_end, win_segs, win_transcript,
                base_analysis, video_duration_sec, cfg,
            )
            hook_structure   = feats["hook_structure"]
            local_intensity  = feats["local_intensity"]
            visual_intensity = feats["visual_intensity"]
            face_prominence  = feats["face_prominence"]
            visual_trend     = feats["visual_trend"]
            onset_jump       = feats["onset_jump"]
            peakiness        = feats["peakiness"]
            prewindow_delta  = feats["prewindow_delta"]
            time_to_peak     = feats.get("time_to_peak",   0.5)
            first_spike      = feats.get("first_spike",    0.0)
            post_peak_drop   = feats.get("post_peak_drop", 0.0)

            hook_score = float(hook_structure.get("hook_score", 0.0))

            # Visual-first gate (v2.2)
            # ─────────────────────────────────────────────────────────────────
            # The standard text-score gate kills silent / reaction / audio-
            # burst hooks because detect_hook_structure() is marker-driven.
            # Allow visual-origin proposals to bypass the text gate when their
            # audio/visual delivery signal is strong enough to stand alone.
            #
            # Sources that may produce hooks WITHOUT strong speech:
            _VISUAL_FIRST_SOURCES = {
                "face_onset", "face_reaction", "visual_burst",
                "visual_onset", "motion_onset", "audio_burst", "prosody_burst",
            }
            # Hybrid proposals that contain visual sensors are also visual-first
            is_visual_first = (
                proposal_source in _VISUAL_FIRST_SOURCES or
                proposal_source == "hybrid" and any(
                    m in _VISUAL_FIRST_SOURCES for m in support_mods
                )
            )
            # Delivery proxy: the strongest a/v signal in this window
            delivery_proxy = max(local_intensity, visual_intensity, face_prominence)

            if hook_score < eff_min_hook_score:
                # Let high-delivery visual-first proposals through with a
                # relaxed threshold.  The penalty layer and final scoring will
                # still down-rank them if the full picture is weak.
                if not (is_visual_first and delivery_proxy >= 0.52):
                    hook_filter_reasons["low_hook_score"] += 1
                    rejected_hooks.append({
                        "start": win_start, "end": win_end,
                        "proposal_source": proposal_source,
                        "reject_reason": "low_hook_score",
                        "hook_score": round(float(hook_score), 3),
                        "threshold": round(float(eff_min_hook_score), 3),
                        "delivery_proxy": round(float(delivery_proxy), 3),
                        "transcript_preview": win_transcript[:80],
                    })
                    continue

            # ── 3. Sub-scores ─────────────────────────────────────────────────
            subscores = _compute_hook_subscores(
                hook_structure, local_intensity, visual_intensity,
                face_prominence, win_start, hook_end,
                visual_trend=visual_trend,
                win_transcript=win_transcript,
                onset_jump=onset_jump,
                peakiness=peakiness,
                prewindow_delta=prewindow_delta,
            )

            # ── 3b. Visual-first hook score (Stage 2: subtype-aware) ─────────
            # Computed independently of text — allows visual-only hooks to
            # surface without requiring high hook_score (text evidence).
            vfhs_result = compute_visual_first_hook_score(
                face_prominence=face_prominence,
                visual_intensity=visual_intensity,
                local_intensity=local_intensity,
                onset_jump=onset_jump,
                peakiness=peakiness,
                visual_trend=visual_trend,
                prewindow_delta=prewindow_delta,
                proposal_source=proposal_source,
                source_confidence=source_conf,
                supporting_modalities=support_mods,
                time_to_peak=feats.get("time_to_peak",   0.5),
                post_peak_drop=feats.get("post_peak_drop", 0.0),
                first_spike=feats.get("first_spike",     0.0),
            )
            visual_first_hook_score = vfhs_result["visual_first_hook_score"]
            visual_subtype          = vfhs_result.get("visual_subtype", "")
            subtype_confidence      = vfhs_result.get("subtype_confidence", 0.0)

            # ── 3c. Event semantics (Stage 7) ──────────────────────────────
            # Interpret WHAT IS HAPPENING before classifying hook type.
            # Type classifier and quality layer read from event_semantics.
            event_semantics = _compute_event_semantics(
                hook_structure=hook_structure,
                visual_subtype=visual_subtype,
                subtype_confidence=subtype_confidence,
                local_intensity=local_intensity,
                visual_intensity=visual_intensity,
                face_prominence=face_prominence,
                visual_trend=visual_trend,
                onset_jump=onset_jump,
                proposal_source=proposal_source,
                source_confidence=source_conf,
            )
            # Attach to hook_structure so detect_hook_type can read them
            hook_structure["_event_semantics"]   = event_semantics
            hook_structure["_dominant_event"]     = event_semantics["dominant_event"]
            hook_structure["_event_clarity"]      = event_semantics["event_clarity"]

            # ── 4. Hook type ──────────────────────────────────────────────────
            hook_type, confidence = detect_hook_type(hook_structure, local_intensity)

            # ── 5. Viral compat ───────────────────────────────────────────────
            viral_compat = _compute_viral_compat(hook_structure, win_transcript, hook_type)

            # ── 6. Penalties + anti-class risks ──────────────────────────────
            penalties = compute_hook_penalties(
                hook_structure, win_transcript, win_start, hook_end, cfg,
                local_intensity=local_intensity,
                visual_intensity=visual_intensity,
            )
            nonhook_risks = compute_nonhook_risks(
                hook_structure, win_transcript, win_start, hook_end,
                local_intensity=local_intensity,
                visual_intensity=visual_intensity,
                onset_jump=onset_jump,
                peakiness=peakiness,
                prewindow_delta=prewindow_delta,
                visual_first_score=visual_first_hook_score,
                config=cfg,
            )

            # ── 7. Final score ────────────────────────────────────────────────
            final_score = compute_hook_final_score(
                hook_structure, hook_type, local_intensity, viral_compat, cfg,
                subscores=subscores, penalties=penalties,
            )

            if final_score < eff_threshold:
                hook_filter_reasons["low_final_score"] += 1
                rejected_hooks.append({
                    "start": win_start, "end": win_end,
                    "proposal_source": proposal_source,
                    "reject_reason": "low_final_score",
                    "final_score": round(float(final_score), 3),
                    "hook_score": round(float(hook_score), 3),
                    "threshold": round(float(eff_threshold), 3),
                    "hook_type_soft": str(hook_type) if "hook_type" in dir() else "",
                    "transcript_preview": win_transcript[:80],
                })
                continue

            # ── 8. Boundary refinement ────────────────────────────────────────
            ref_start, ref_end, boundary_quality, boundary_diag = _refine_hook_boundaries(
                win_start, win_end, win_segs,
                cfg.min_hook_duration, cfg.boundary_snap_sec,
                base_analysis=base_analysis,
                video_duration_sec=video_duration_sec,
                tension_types=hook_structure.get("_tension_types", []),
                tension_closed=bool(hook_structure.get("_tension_closed", False)),
            )

            # ── 9. Quality fields ─────────────────────────────────────────────
            # Pass the FINAL hook_type (from soft classifier) so agreement is
            # evaluated against the right per-type profile.
            mod_reliability = _compute_modality_reliability(
                base_analysis, win_transcript, hook_structure,
                final_hook_type=hook_type,
            )
            quality = _compute_hook_quality(
                hook_structure, hook_type, subscores, penalties,
                final_score, local_intensity, viral_compat, boundary_quality,
                modality_reliability=mod_reliability,
                proposal_source=proposal_source,
                source_confidence=source_conf,
                supporting_modalities=support_mods,
                visual_first_hook_score=visual_first_hook_score,
                type_ambiguity=hook_structure.get("_type_ambiguity", 0.0),
                event_clarity=event_semantics.get("event_clarity", 0.5),
            )
            # Fold the dominant nonhook risk into false_positive_risk.
            max_nh_risk = max(nonhook_risks.values()) if nonhook_risks else 0.0
            if max_nh_risk >= 0.50:
                old_fpr = float(quality.get("false_positive_risk", 0.0))
                quality["false_positive_risk"] = round(
                    float(np.clip(old_fpr * 0.60 + max_nh_risk * 0.40, 0.0, 1.0)), 3
                )

            # Fold continuation_preserved into hook_confidence.
            # A clip that has already resolved its tension at the cut point is a
            # weaker hook regardless of its score — it leaves nothing for the
            # viewer to chase.  Penalise hook_confidence accordingly.
            cont_preserved = float(boundary_diag.get("continuation_preserved", 0.5))
            if cont_preserved < 0.45:
                # Tension already resolved — downgrade confidence
                old_conf = float(quality.get("hook_confidence", 0.0))
                quality["hook_confidence"] = round(
                    float(np.clip(old_conf * (0.70 + 0.30 * cont_preserved / 0.45), 0.0, 1.0)), 3
                )
            # Write continuation_preserved into quality for export gate visibility
            quality["continuation_preserved"] = round(cont_preserved, 3)

            # ── 10. Reasons ───────────────────────────────────────────────────
            reasons = build_hook_reasons(
                hook_structure, hook_type, local_intensity, viral_compat,
                subscores=subscores, penalties=penalties,
            )

            # ── 11. Build moment ──────────────────────────────────────────────
            moment = _make_hook_moment(
                ref_start, ref_end, hook_type, hook_structure,
                hook_score, local_intensity, reasons, win_transcript, cfg,
                subscores=subscores, penalties=penalties,
                quality=quality, viral_compat=viral_compat,
                proposal_source=proposal_source,
            )
            # Temporal descriptors for traceability / downstream tuning
            moment["visual_trend"]    = visual_trend
            moment["onset_jump"]      = onset_jump
            moment["peakiness"]       = round(peakiness, 3)
            moment["prewindow_delta"] = prewindow_delta
            moment["time_to_peak"]    = time_to_peak
            moment["first_spike"]     = first_spike
            moment["post_peak_drop"]  = post_peak_drop
            # Rich proposal provenance (v2.4)
            moment["source_confidence"]     = source_conf
            moment["trigger_time"]          = trigger_time
            moment["trigger_strength"]      = trigger_strength
            moment["supporting_modalities"] = support_mods
            # Type classifier outputs (v2.4)
            moment["secondary_hook_type"] = hook_structure.get("_secondary_hook_type", "")
            moment["type_ambiguity"]      = hook_structure.get("_type_ambiguity", 0.0)
            # Visual-first hook score (v2.4)
            moment["visual_first_hook_score"] = visual_first_hook_score
            moment["vfhs_components"]         = {
                k: v for k, v in vfhs_result.items()
                if k != "visual_first_hook_score"
            }
            # Boundary diagnostics (v2.4)
            moment["boundary_diagnostics"] = boundary_diag
            # Anti-class risks (v2.4)
            moment["nonhook_risks"] = nonhook_risks
            # Dominant anti-class: helps debugging ("why was this flagged?")
            max_risk_key = max(nonhook_risks, key=nonhook_risks.get)
            moment["dominant_nonhook_risk"] = (
                max_risk_key if nonhook_risks[max_risk_key] >= 0.40 else "none"
            )

            # ── Event semantics in output (Stage 7) ──────────────────────────
            moment["event_semantics"]  = event_semantics
            moment["dominant_event"]   = event_semantics["dominant_event"]
            moment["event_clarity"]    = event_semantics["event_clarity"]
            # Visual subtype (Stage 2)
            moment["visual_subtype"]     = visual_subtype
            moment["subtype_confidence"] = subtype_confidence
            moment["subtype_scores"]     = vfhs_result.get("subtype_scores", {})
            # Tension types (Stage 5)
            moment["tension_types"]  = hook_structure.get("_tension_types", [])
            moment["tension_closed"] = hook_structure.get("_tension_closed", False)
            # end_leakage_risk (Stage 6)
            moment["end_leakage_risk"] = float(boundary_diag.get("end_leakage_risk", 0.0))
            # Narrow text evidence (Stage 3)
            for _ev_key in ("question_open", "promise_open", "contrast_signal",
                            "reveal_signal", "resolution_signal"):
                moment[_ev_key] = hook_structure.get(_ev_key, False)

            # ── Operational decision ──────────────────────────────────────────
            moment["export_decision"] = _export_decision(
                quality, subscores=subscores,
                proposal_source=proposal_source, hook_type=hook_type,
            )
            hook_candidates.append(moment)

        # ── Sort → NMS → top-K ────────────────────────────────────────────────
        # v2.3: quality-aware utility ranking.
        # The old sort-by-score-only could surface a "loud" candidate that has
        # high raw score but low hook_confidence (weak evidence breadth, high
        # FPR, poor reliability).  A utility function blending score with
        # confidence better reflects "which candidate should we act on first".
        #
        # Utility = 0.60 * final_score + 0.40 * hook_confidence
        #
        # Within equal utility buckets, auto_export candidates are ranked first
        # so that the top-1 result is already "ready for production" when one
        # exists.
        def _candidate_utility(m: Dict) -> Tuple[int, float]:
            conf    = float(m.get("hook_confidence", 0.0))
            score   = float(m.get("score", 0.0))
            utility = 0.60 * score + 0.40 * conf
            # Secondary sort key: auto_export > manual_review > reject
            export_rank = {"auto_export": 2, "manual_review": 1, "reject": 0}.get(
                m.get("export_decision", "manual_review"), 1
            )
            return (export_rank, utility)

        hook_candidates.sort(key=_candidate_utility, reverse=True)
        n_before_nms = len(hook_candidates)
        hook_candidates = _temporal_nms(hook_candidates, iou_thresh=cfg.nms_iou_thresh)
        hook_candidates = hook_candidates[:top_k]

        # v2.1: WEAK-HOOK FALLBACK — если все отфильтровались, но raw сигнал был,
        # вернём top-3 weak-candidates как manual_review (reason=weak_hook_manual_review)
        weak_hook_fallback_used = False
        if not hook_candidates and rejected_hooks:
            # Сортируем по final_score (если есть), иначе по hook_score
            weak_pool = sorted(
                [r for r in rejected_hooks
                 if r.get("reject_reason") in ("low_final_score", "low_hook_score")],
                key=lambda r: float(r.get("final_score", r.get("hook_score", 0.0))),
                reverse=True,
            )[:3]
            for idx, wk in enumerate(weak_pool):
                ws = float(wk.get("start", 0.0))
                we = float(wk.get("end", ws + cfg.min_hook_duration))
                weak_moment = {
                    "start": round(ws, 2),
                    "end": round(we, 2),
                    "duration": round(max(0.0, we - ws), 2),
                    "score": float(wk.get("final_score", wk.get("hook_score", 0.0))),
                    "hook_score": float(wk.get("hook_score", 0.0)),
                    "hook_type": "weak_hook",
                    "hook_confidence": 0.20,
                    "reasons": [{
                        "code": "weak_hook_manual_review",
                        "message": f"Candidate below threshold but had raw signal "
                                   f"(reject_reason={wk.get('reject_reason')})",
                        "weight": 0.4,
                    }],
                    "export_decision": "manual_review",
                    "reject_reason_original": wk.get("reject_reason"),
                    "transcript_preview": wk.get("transcript_preview", ""),
                    "proposal_source": wk.get("proposal_source", "unknown"),
                    "weak_rank": idx + 1,
                    "is_weak_fallback": True,
                }
                hook_candidates.append(weak_moment)
            weak_hook_fallback_used = True
            logger.info(
                f"weak_hook_manual_review fallback: promoted top-{len(weak_pool)} "
                f"rejected hooks to manual_review (from {len(rejected_hooks)} rejected)"
            )

        if not hook_candidates:
            logger.info("No hooks detected after filtering (rejected=%d)", len(rejected_hooks))
            empty = _empty_hook_result(video_duration_sec, "no_hook_detected", cfg)
            empty["rejected_hooks"] = rejected_hooks
            empty["hook_filter_reasons"] = hook_filter_reasons
            empty["hook_proposals_count"] = len(proposals) if proposals else 0
            empty["micro_hook_active"] = micro_hook_active
            empty["stats"]["n_raw_proposals"] = len(proposals) if proposals else 0
            empty["stats"]["n_rejected"] = len(rejected_hooks)
            empty["stats"]["filter_reasons"] = hook_filter_reasons
            empty["stats"]["micro_hook_active"] = micro_hook_active
            return empty

        # ── Proposal audit: was the best hook proposed? (Stage 1) ─────────────
        # The most common proposer failure: the best hook moment was never
        # put into the candidate pool — so no amount of ranking improvement helps.
        # This check makes that failure visible in the debug log.
        best_hook_start  = round(hook_candidates[0].get("start", 0.0), 2)
        best_hook_source = hook_candidates[0].get("proposal_source", "unknown")
        nearest_proposal_t, nearest_proposal_src = None, "none"
        min_dist = float("inf")
        for trig_t, trig_src in _all_proposal_trigger_times.items():
            d = abs(trig_t - best_hook_start)
            if d < min_dist:
                min_dist, nearest_proposal_t, nearest_proposal_src = d, trig_t, trig_src
        if min_dist <= 0.60:
            logger.info(
                f"[Proposal audit] Best hook at {best_hook_start:.2f}s "
                f"(source={best_hook_source}) WAS proposed (nearest proposal: "
                f"{nearest_proposal_t:.2f}s from {nearest_proposal_src})"
            )
        else:
            logger.warning(
                f"[Proposal audit] Best hook at {best_hook_start:.2f}s "
                f"(source={best_hook_source}) WAS NOT CLEARLY PROPOSED — "
                f"nearest proposal was {min_dist:.2f}s away "
                f"({nearest_proposal_src} at {nearest_proposal_t}s). "
                f"Consider tuning the sub-proposer for this event type."
            )

        for i, h in enumerate(hook_candidates):
            logger.info(
                f"  Hook #{i+1}: {h['hook_type']} "
                f"[{h['start']:.1f}–{h['end']:.1f}s] "
                f"score={h['score']:.3f} "
                f"conf={h.get('hook_confidence', 0):.2f} "
                f"fpr={h.get('false_positive_risk', 0):.2f}"
            )

        avg_score      = float(np.mean([m["score"]                      for m in hook_candidates]))
        avg_confidence = float(np.mean([m.get("hook_confidence", 0)     for m in hook_candidates]))

        # Source breakdown (v2.1) — how many moments came from each proposal source
        src_breakdown: Dict[str, int] = {}
        for m in hook_candidates:
            s = m.get("proposal_source", "unknown")
            src_breakdown[s] = src_breakdown.get(s, 0) + 1

        return {
            "mode": "hook",
            "hook_moments": hook_candidates,
            # v2.1: diagnostics
            "rejected_hooks": rejected_hooks,
            "hook_filter_reasons": hook_filter_reasons,
            "hook_proposals_count": len(proposals) if proposals else 0,
            "micro_hook_active": micro_hook_active,
            "weak_hook_fallback_used": weak_hook_fallback_used,
            "stats": {
                "total_duration":           video_duration_sec,
                "profile_name":             f"{cfg.mode_name} {cfg.profile_version}",
                "hook_window_sec":          hook_end,
                "num_hooks_found":          len(hook_candidates),
                "hook_type":                hook_candidates[0]["hook_type"],
                "threshold_original":       cfg.threshold,
                "threshold_effective":      eff_threshold,
                "min_hook_score_original":  cfg.min_hook_score,
                "min_hook_score_effective": eff_min_hook_score,
                "loose_hook_mode":          cfg.loose_hook_mode,
                "micro_hook_active":        micro_hook_active,
                "n_raw_proposals":          len(proposals) if proposals else 0,
                "n_rejected":               len(rejected_hooks),
                "filter_reasons":           hook_filter_reasons,
                "weak_hook_fallback_used":  weak_hook_fallback_used,
                "avg_score":                round(avg_score, 3),
                "avg_confidence":           round(avg_confidence, 3),
                "max_hook_score":           round(max(m["hook_score"] for m in hook_candidates), 3),
                "avg_intensity":            round(float(np.mean([m["intensity"] for m in hook_candidates])), 3),
                "n_candidates_before_nms":  n_before_nms,
                # v2.0 quality summary
                "avg_false_positive_risk":  round(
                    float(np.mean([m.get("false_positive_risk", 0) for m in hook_candidates])), 3
                ),
                "avg_boundary_quality":     round(
                    float(np.mean([m.get("boundary_quality", 0) for m in hook_candidates])), 3
                ),
                # v2.1 reliability + source breakdown
                "avg_asr_confidence":       round(
                    float(np.mean([m.get("asr_confidence",     0.70) for m in hook_candidates])), 3
                ),
                "avg_visual_confidence":    round(
                    float(np.mean([m.get("visual_confidence",  0.50) for m in hook_candidates])), 3
                ),
                "avg_modality_agreement":   round(
                    float(np.mean([m.get("modality_agreement", 0.50) for m in hook_candidates])), 3
                ),
                "proposal_sources":         src_breakdown,
                # Operational breakdown
                "export_decisions":         {
                    dec: sum(1 for m in hook_candidates
                             if m.get("export_decision") == dec)
                    for dec in ("auto_export", "manual_review", "reject")
                },
                # Stage 4: safety score summary
                "avg_auto_export_safety":   round(
                    float(np.mean([m.get("auto_export_safety_score", 0.0)
                                   for m in hook_candidates])), 3
                ),
                "any_failure_modes":        any(
                    m.get("failure_mode_count", 0) > 0 for m in hook_candidates
                ),
                # Stage 7: dominant events in final pool
                "dominant_events": [m.get("dominant_event", "") for m in hook_candidates],
                # Stage 2: visual subtypes in final pool
                "visual_subtypes": [m.get("visual_subtype", "") for m in hook_candidates],
                # Stage 1: proposal source breakdown (expanded)
                "proposal_sources_detail": src_counts,
            },
        }

    except Exception as exc:
        logger.error(f"Error in Hook Mode: {exc}", exc_info=True)
        return _empty_hook_result(video_duration_sec, f"error: {exc}", cfg)


# =============================================================================
# UI HINTS
# =============================================================================

UI_HOOK_BADGES: Dict[str, Any] = {
    "hook_type_labels": {
        "question_hook":   {"text": "Вопрос",      "color": "#3B82F6"},
        "curiosity_hook":  {"text": "Любопытство", "color": "#6366F1"},
        "intrigue_hook":   {"text": "Интрига",     "color": "#EF4444"},
        "fact_bomb":       {"text": "Факт-бомба",  "color": "#F59E0B"},
        "emotional_hook":  {"text": "Эмоция",      "color": "#8B5CF6"},
        "promise_hook":    {"text": "Обещание",    "color": "#10B981"},
        "viral_tease":     {"text": "Тренд",       "color": "#EC4899"},
        "warning_hook":    {"text": "Внимание",    "color": "#F97316"},
        "contrarian_hook": {"text": "Контрарный",  "color": "#DC2626"},
        "reveal_hook":     {"text": "Раскрытие",   "color": "#0EA5E9"},
        "reaction_hook":   {"text": "Реакция",     "color": "#A855F7"},
        "weak_hook":       {"text": "Слабый хук",  "color": "#6B7280"},
    },
    "structure_labels": {
        "has_question":   {"text": "Вопрос",       "color": "#3B82F6"},
        "has_intrigue":   {"text": "Интрига",      "color": "#EF4444"},
        "has_fact":       {"text": "Факт",         "color": "#F59E0B"},
        "has_emotion":    {"text": "Эмоция",       "color": "#8B5CF6"},
        "has_promise":    {"text": "Обещание",     "color": "#10B981"},
        "has_viral":      {"text": "Вирусный",     "color": "#EC4899"},
        "has_warning":    {"text": "Внимание",     "color": "#F97316"},
        "has_contrarian": {"text": "Контрарный",   "color": "#DC2626"},
        "has_reveal":     {"text": "Раскрытие",    "color": "#0EA5E9"},
        "has_continuation":{"text": "Незавершённость","color": "#6366F1"},
    },
    "quality_labels": {
        "hook_confidence":     {"label": "Уверенность",         "good_threshold": 0.60},
        "false_positive_risk": {"label": "Риск ложного хука",   "good_threshold": 0.30},
        "continuation_tension":{"label": "Напряжение/gap",      "good_threshold": 0.50},
        "boundary_quality":    {"label": "Качество границ",     "good_threshold": 0.50},
        "delivery_strength":   {"label": "Сила подачи",         "good_threshold": 0.50},
    },
}


# =============================================================================
# SMOKE TEST
# =============================================================================

if __name__ == "__main__":
    _cfg = HookModeConfig(mode_name="smoke", hook_window_sec=6.0, threshold=0.35)

    _asr = [
        {"start": 0.0, "end": 1.5, "text": "Что если я скажу,"},
        {"start": 1.5, "end": 3.0, "text": "что 70% делают это неправильно?"},
        {"start": 3.0, "end": 5.0, "text": "Шокирующие факты — но есть кое-что ещё..."},
    ]

    # detect_hook_structure
    _hs = detect_hook_structure(
        " ".join(s["text"] for s in _asr), config=_cfg,
        asr_segments=_asr, window_start=0, window_end=6.0,
    )
    print("has_hook:", _hs["has_hook"], "| hook_score:", _hs["hook_score"])
    print("has_continuation:", _hs["has_continuation"])
    print("raw_counts:", _hs["raw_counts"])

    # detect_hook_type
    _ht, _conf = detect_hook_type(_hs, intensity=0.72)
    print(f"hook_type: {_ht}  conf: {_conf:.2f}")

    # subscores
    _ss = _compute_hook_subscores(_hs, 0.72, 0.6, 0.7, 0.0, 6.0)
    print("subscores:", _ss)

    # penalties
    _pen = compute_hook_penalties(_hs, "Что если я скажу что 70% делают это неправильно?",
                                   0.0, 6.0, _cfg)
    print("penalties:", _pen)

    # viral compat
    _vc = _compute_viral_compat(_hs, "70% делают это неправильно? Шокирующие факты", _ht)
    print(f"viral_compat: {_vc:.3f}")

    # full run
    _result = find_hook_moments("smoke.mp4", 30.0, _asr, config=_cfg)
    print(f"\nHooks found: {len(_result['hook_moments'])}")
    for _h in _result["hook_moments"]:
        print(
            f"  {_h['hook_type']} [{_h['start']:.1f}–{_h['end']:.1f}s] "
            f"score={_h['score']:.3f}  conf={_h.get('hook_confidence',0):.2f}  "
            f"fpr={_h.get('false_positive_risk',0):.2f}  "
            f"boundary_q={_h.get('boundary_quality',0):.2f}"
        )

    print("\nALL OK")
