"""
STORY MODE v1.0 - Narrative-Based Video Segmentation
Находит цельные мини-истории с завязкой, конфликтом и развязкой

Inspired by:
- Computationally-Narrativity-Detection (narrativity scoring)
- ARC-Chapter (narrative video chapters)
- NarrativeArc (BUTTER-Tools/NarrativeArc) - emotion/narrative tension curves
- pNarrative (sentiment-based plot arcs, Vonnegut curves)
- narrative_structures (ben-aaron188) - exposition/climax/resolution in vlogs
- StoryTeller (character-based long video narrative)
"""

import sys
import io
import json
import logging
from pathlib import Path
from typing import Any, List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict
import numpy as np

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# STORY MODE CONFIG CLASS (для версионирования и профилей под ниши)
# =============================================================================

@dataclass
class StoryModeConfig:
    """
    Единый конфиг Story Mode. Все режимные параметры — здесь.
    Загружается из YAML/JSON для версионирования профилей под ниши.

    Профили (примеры):
      - profiles/business_stories_v1.yaml  — строгие требования, высокие пороги
      - profiles/podcast_stories_v1.yaml   — loose_story_mode, мягкие пороги
      - profiles/educational_stories_v1.yaml — акцент на has_takeaway

    Поля профиля (все опциональны при загрузке, дефолты ниже):
    ┌──────────────────────────────────┬──────────┬─────────────────────────────────────────────┐
    │ Поле                             │ Тип      │ Смысл                                       │
    ├──────────────────────────────────┼──────────┼─────────────────────────────────────────────┤
    │ mode_name                        │ str      │ Имя профиля (для логов/аналитики)            │
    │ profile_version                  │ str      │ Версия профиля ("v1.1")                     │
    │ locale                           │ str      │ Язык лексиконов ("ru", "en")                │
    │ threshold                        │ float    │ Финальный порог отбора (0..1)               │
    │ min_narrative_threshold          │ float    │ Порог narrative_score (0..1)                │
    │ require_conflict_and_resolution  │ bool     │ Strict: оба обязательны                    │
    │ loose_story_mode                 │ bool     │ Loose: только конфликт + рост sentiment     │
    │ use_reranker                     │ bool     │ Применять re-ranker после baseline          │
    │ reranker_model_path              │ str/None │ Путь к XGBoost-модели re-ranker             │
    │ window_size                      │ float    │ Размер скользящего окна (секунды)           │
    │ step_size                        │ float    │ Шаг окна (секунды)                          │
    │ min_clip_duration                │ float    │ Минимальная длина клипа                     │
    │ max_clip_duration                │ float    │ Максимальная длина клипа                    │
    │ preferred_duration               │ float    │ Идеальная длина (для length_weight)         │
    │ weights                          │ dict     │ Веса visual/audio/semantic/narrative        │
    │ structure_markers                │ dict     │ setup/conflict/resolution маркеры           │
    │ sentiment_markers                │ dict     │ positive/negative/neutral_turn маркеры      │
    │ takeaway_markers                 │ list     │ Маркеры вывода/урока                        │
    └──────────────────────────────────┴──────────┴─────────────────────────────────────────────┘

    Лексиконы structure_markers и sentiment_markers НЕ пересекаются (по ролям).
    Веса хранятся в профиле и передаются в compute_story_score, переопределяя MODE_CONFIGS.
    """
    # --- Метаданные профиля ---
    mode_name: str = "default"
    profile_version: str = "v1.1"
    locale: str = "ru"

    # --- Лексиконы (разделены по ролям, не пересекаются) ---
    structure_markers: Dict[str, List[str]] = field(default_factory=lambda: {
        "setup_markers": [
            "у нас был", "когда я", "было дело", "расскажу", "история",
            "однажды", "случай", "клиент", "проект", "начали", "задача была"
        ],
        "conflict_markers": [
            "проблема", "сложность", "не получилось", "ошибка", "провал", "факап",
            "столкнулись", "трудно", "вызов", "препятствие", "неожиданно"
        ],
        "resolution_markers": [
            "итог", "в итоге", "в конце", "урок", "опыт", "после этого"
        ],
    })

    sentiment_markers: Dict[str, List[str]] = field(default_factory=lambda: {
        "sentiment_positive": [
            "получилось", "удалось", "успех", "результат", "достигли", "отлично",
            "хорошо", "поняли", "научились", "решили", "сделали", "благодаря",
            "рад", "довольны", "выросла", "улучшили"
        ],
        "sentiment_negative": [
            "облажались", "выгорел", "затянулся", "потеряли", "плохо", "упала", "сложно"
        ],
        "sentiment_neutral_turn": ["но", "однако", "вдруг", "тут"],
    })

    takeaway_markers: List[str] = field(default_factory=lambda: [
        "вывод", "урок", "поняли", "научились", "итог", "главное", "результат",
        "что мы поняли", "в итоге", "теперь"
    ])

    # --- Веса модальностей (переопределяют MODE_CONFIGS["story"]["weights"]) ---
    weights: Dict[str, float] = field(default_factory=lambda: {
        "visual": 0.20,
        "audio": 0.25,
        "semantic": 0.35,
        "narrative": 0.20,
    })

    # --- Пороги ---
    threshold: float = 0.65
    min_narrative_threshold: float = 0.5

    # --- Режим фильтрации ---
    require_conflict_and_resolution: bool = True  # strict mode
    loose_story_mode: bool = False                # достаточно конфликта + рост sentiment

    # --- Параметры окна ---
    window_size: float = 75.0
    step_size: float = 30.0
    min_clip_duration: float = 30.0
    max_clip_duration: float = 150.0
    preferred_duration: float = 75.0

    # --- Re-ranker ---
    use_reranker: bool = False
    reranker_model_path: Optional[str] = None

    # --- Переопределение типов историй (опционально) ---
    story_types: Dict[str, Dict] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str) -> "StoryModeConfig":
        """Загрузка профиля из YAML; неизвестные ключи игнорируются."""
        if yaml is None:
            raise ImportError("PyYAML не установлен: pip install pyyaml")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls._from_dict(data)

    @classmethod
    def from_json(cls, path: str) -> "StoryModeConfig":
        """Загрузка профиля из JSON; неизвестные ключи игнорируются."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: Dict) -> "StoryModeConfig":
        """Создание конфига из dict; только известные поля, дефолты для остальных."""
        import dataclasses
        known = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    def to_yaml(self, path: str):
        """Сохранение профиля в YAML."""
        if yaml is None:
            raise ImportError("PyYAML не установлен: pip install pyyaml")
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(asdict(self), f, allow_unicode=True, default_flow_style=False)

    def to_json(self, path: str):
        """Сохранение профиля в JSON."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)

    def merge_config(self) -> Dict:
        """Единый dict (обратная совместимость со старым STORY_CONFIG)."""
        return {
            **self.structure_markers,
            **self.sentiment_markers,
            "takeaway_markers": self.takeaway_markers,
        }


# =============================================================================
# STORY SEGMENT TYPES (Inspired by narrative research)
# =============================================================================
# Критерии выбора типа истории:
# - personal_story: я-форма, личный опыт, нет клиента
# - client_case: упоминание клиента/заказчика, профессиональный контекст
# - failure_story: явные маркеры провала, негативная тональность → позитивная
# - success_story: явные маркеры успеха, позитивная тональность в конце
# - anecdote: короткая байка, неформальный стиль
# - non_story: нет конфликта, нет дуги, низкий narrative_score

STORY_SEGMENT_TYPES = {
    "personal_story": {
        "weight": 1.25,
        "min_duration": 45,
        "max_duration": 120,
        "title_template": "История: {summary}",
        "markers": ["я", "мы", "расскажу", "было дело", "история", "случай", "однажды"]
    },
    "client_case": {
        "weight": 1.30,
        "min_duration": 60,
        "max_duration": 150,
        "title_template": "Кейс: {summary}",
        "markers": ["клиент", "заказчик", "у нас был", "пришёл", "обратился"]
    },
    "failure_story": {
        "weight": 1.35,  # Факапы особенно ценны!
        "min_duration": 45,
        "max_duration": 120,
        "title_template": "Факап: {summary}",
        "markers": ["ошибка", "провал", "не получилось", "облажались", "факап", "проблема"]
    },
    "success_story": {
        "weight": 1.20,
        "min_duration": 45,
        "max_duration": 120,
        "title_template": "Успех: {summary}",
        "markers": ["получилось", "удалось", "успех", "результат", "достигли"]
    },
    "anecdote": {
        "weight": 1.10,
        "min_duration": 30,
        "max_duration": 90,
        "title_template": "Байка: {summary}",
        "markers": ["смешно", "прикол", "забавно", "анекдот", "история про"]
    },
    "lesson_learned": {
        "weight": 1.15,
        "min_duration": 30,
        "max_duration": 90,
        "title_template": "Урок: {summary}",
        "markers": ["вывод", "научились", "урок", "опыт показал", "поняли что"]
    },
    "non_story": {
        "weight": 0.3,
        "min_duration": 0,
        "max_duration": 30,
        "title_template": "Не история",
        "markers": []
    },
    # v2.1: lightweight subtypes for short/medium videos (< 3 min)
    "micro_story": {
        "weight": 0.95,
        "min_duration": 12,
        "max_duration": 45,
        "title_template": "Микро-история: {summary}",
        "markers": ["я", "мы", "было", "у меня", "представь"],
    },
    "case_fragment": {
        "weight": 1.00,
        "min_duration": 15,
        "max_duration": 60,
        "title_template": "Фрагмент кейса: {summary}",
        "markers": ["клиент", "у нас был", "пример", "кейс", "сталкивался"],
    },
    "explanation_story": {
        "weight": 0.90,
        "min_duration": 15,
        "max_duration": 75,
        "title_template": "Разбор: {summary}",
        "markers": ["объясняю", "разберём", "смысл в том", "причина", "потому что", "поэтому"],
    },
}


# =============================================================================
# STORY ARC PATTERNS (Inspired by Kurt Vonnegut & Christopher Booker)
# =============================================================================

# Arc weights вдохновлены «Shapes of Stories» Курта Воннегута:
# ось X = время, ось Y = эмоциональное состояние/удача.
# Мы делим sentiment-кривую на трети и смотрим на transitions part1→part2→part3.
# Источник: https://publicspeakingmasterclasses.com/8-story-shapes-every-storyteller-must-know/
STORY_ARC_PATTERNS = {
    "rags_to_riches": {
        "pattern": "up",           # непрерывный рост (part1 < part2 < part3)
        "description": "От плохого к хорошему",
        "weight": 1.2
    },
    "riches_to_rags": {
        "pattern": "down",         # непрерывное падение
        "description": "От хорошего к плохому (трагедия)",
        "weight": 1.1
    },
    "man_in_hole": {
        "pattern": "down_up",      # падение → подъём (самый частый в кейсах!)
        "description": "Проблема → Решение",
        "weight": 1.35
    },
    "icarus": {
        "pattern": "up_down",      # подъём → падение
        "description": "Успех → Провал",
        "weight": 1.25
    },
    "cinderella": {
        "pattern": "up_down_up",   # рост → падение → рост
        "description": "Успех → Неудача → Успех",
        "weight": 1.30
    },
    "oedipus": {
        "pattern": "down_up_down", # падение → рост → падение
        "description": "Проблема → Решение → Новая проблема",
        "weight": 1.15
    },
    "flat": {
        "pattern": "flat",         # нет явной дуги
        "description": "Без явной дуги",
        "weight": 0.5
    }
}

# Максимальный arc_weight для нормализации arc_score в [0..1].
# man_in_hole имеет вес 1.35 → MAX_ARC_WEIGHT = 1.35.
# arc_score = arc_confidence * arc_weight / MAX_ARC_WEIGHT
MAX_ARC_WEIGHT = max(v["weight"] for v in STORY_ARC_PATTERNS.values())


# =============================================================================
# STORY CONFIG — лексиконы по ролям (без пересечений)
# =============================================================================
# Структура (setup/conflict/resolution) — только для detect_story_structure,
# с учётом позиции в окне. Sentiment — только для sentiment-кривой по ASR.
# Ни одно слово не входит одновременно в структуру и в sentiment.
#
# Калибровка под нишу: добавлять маркеры в соответствующий блок, не дублируя.

# --- Только структура (position-aware: setup в начале, resolution в конце) ---
STRUCTURE_MARKERS = {
    "setup_markers": [
        "у нас был", "когда я", "было дело", "расскажу", "история",
        "однажды", "случай", "клиент", "проект", "начали", "задача была"
    ],
    "conflict_markers": [
        "проблема", "сложность", "не получилось", "ошибка", "провал", "факап",
        "столкнулись", "трудно", "вызов", "препятствие", "неожиданно"
    ],
    "resolution_markers": [
        "итог", "в итоге", "в конце", "урок", "опыт", "после этого"
    ],
}

# --- Только sentiment (для кривой по сегментам, без структуры) ---
SENTIMENT_MARKERS = {
    "sentiment_positive": [
        "получилось", "удалось", "успех", "результат", "достигли", "отлично",
        "хорошо", "поняли", "научились", "решили", "сделали", "благодаря",
        "рад", "довольны", "выросла", "улучшили"
    ],
    "sentiment_negative": [
        "облажались", "выгорел", "затянулся", "потеряли", "плохо", "упала", "сложно"
    ],
    "sentiment_neutral_turn": ["но", "однако", "вдруг", "тут"],
}

# --- Takeaway (вывод/урок в хвосте; может пересекаться по смыслу, роль другая) ---
TAKEAWAY_MARKERS = [
    "вывод", "урок", "поняли", "научились", "итог", "главное", "результат",
    "что мы поняли", "в итоге", "теперь"
]

# Единый конфиг для обратной совместимости (все ключи в одном dict)
STORY_CONFIG = {
    **STRUCTURE_MARKERS,
    **SENTIMENT_MARKERS,
    "takeaway_markers": TAKEAWAY_MARKERS,
}

# =============================================================================
# MODE CONFIGS: STORY vs VIRAL vs EDUCATIONAL
# =============================================================================

MODE_CONFIGS = {
    "story": {
        "weights": {
            "visual": 0.20,      # Визуал вторичен
            "audio": 0.25,       # Эмоции в голосе важны
            "semantic": 0.35,    # Содержание важно
            "narrative": 0.20    # 🆕 Нарративная структура!
        },
        "window_size_range": (45, 120),  # Истории длиннее
        "min_clip_duration": 30,
        "max_clip_duration": 150,
        "preferred_duration": 75,
        "requires_story_structure": True,
        "min_story_score": 0.6,
        "title_style": "story",
        "reasons_focus": ["story_arc", "emotional_change", "resolution", "conflict"]
    },
    "viral": {
        "weights": {
            "visual": 0.45,
            "audio": 0.30,
            "semantic": 0.25
        },
        "window_size_range": (10, 45),
        "title_style": "viral",
        "reasons_focus": ["emotion", "motion", "hook"]
    },
    "educational": {
        "weights": {
            "visual": 0.25,
            "audio": 0.30,
            "semantic": 0.45
        },
        "window_size_range": (45, 120),
        "title_style": "educational",
        "reasons_focus": ["explanation", "clarity", "structure"]
    }
}


# =============================================================================
# NARRATIVE STRUCTURE DETECTION
# =============================================================================

def detect_story_structure(
    transcript: str,
    config: Optional[Dict] = None,
    llm_callback: Optional[callable] = None,
    asr_segments: Optional[List[Dict]] = None,
    window_start: float = 0.0,
    window_end: float = 0.0,
    use_time_aware: bool = True
) -> Dict:
    """
    Детектирование структуры истории (setup/conflict/resolution).
    
    LLM-first: если передан llm_callback(transcript) -> {has_setup, has_conflict, has_resolution, structure_score?},
    используется его результат. Иначе — position-aware лексикон:
    
    Time-aware (если use_time_aware=True и есть asr_segments с таймкодами):
    - setup: первые 30-40% времени окна
    - conflict: средняя часть (20-80% времени)
    - resolution: последние 30-40% времени
    
    Word-based fallback (если нет таймкодов):
    - setup: первые ~40% слов
    - conflict: середина (20-80% слов)
    - resolution: последние ~40% слов
    
    Маркеры только из structure-лексикона (config), без пересечения с sentiment.
    
    Returns:
        { has_setup, has_conflict, has_resolution, story_markers, structure_score }
    """
    if not transcript or len(transcript.strip()) < 30:
        return {
            "has_setup": False,
            "has_conflict": False,
            "has_resolution": False,
            "story_markers": [],
            "structure_score": 0.0
        }
    
    cfg = config or STORY_CONFIG
    setup_markers = cfg.get("setup_markers", STRUCTURE_MARKERS["setup_markers"])
    conflict_markers = cfg.get("conflict_markers", STRUCTURE_MARKERS["conflict_markers"])
    resolution_markers = cfg.get("resolution_markers", STRUCTURE_MARKERS["resolution_markers"])
    
    # LLM-first: если есть колбэк — используем его результат
    if llm_callback and transcript.strip():
        try:
            raw = llm_callback(transcript)
            if isinstance(raw, dict) and any(k in raw for k in ("has_setup", "has_conflict", "has_resolution")):
                has_setup = bool(raw.get("has_setup", False))
                has_conflict = bool(raw.get("has_conflict", False))
                has_resolution = bool(raw.get("has_resolution", False))
                structure_score = float(raw.get("structure_score", 0.0))
                if structure_score <= 0:
                    structure_score = (0.3 * has_setup + 0.4 * has_conflict + 0.3 * has_resolution)
                    if has_setup and has_conflict and has_resolution:
                        structure_score += 0.2
                structure_score = min(1.0, structure_score)
                found_markers = [m for m, v in [("setup", has_setup), ("conflict", has_conflict), ("resolution", has_resolution)] if v]
                return {
                    "has_setup": has_setup,
                    "has_conflict": has_conflict,
                    "has_resolution": has_resolution,
                    "story_markers": found_markers,
                    "structure_score": float(structure_score)
                }
        except Exception as e:
            logger.debug(f"LLM structure callback failed: {e}")
    
    # Time-aware fallback (если есть asr_segments с таймкодами)
    if use_time_aware and asr_segments and window_start < window_end:
        window_duration = window_end - window_start
        setup_end = window_start + window_duration * 0.4
        conflict_start = window_start + window_duration * 0.2
        conflict_end = window_start + window_duration * 0.8
        resolution_start = window_end - window_duration * 0.4
        
        part_setup = " ".join([
            s.get("text", "") for s in asr_segments
            if s.get("start", 0) < setup_end
        ])
        part_conflict = " ".join([
            s.get("text", "") for s in asr_segments
            if conflict_start <= s.get("start", 0) < conflict_end
        ])
        part_resolution = " ".join([
            s.get("text", "") for s in asr_segments
            if s.get("start", 0) >= resolution_start
        ])
        
        has_setup = any(m in part_setup.lower() for m in setup_markers)
        has_conflict = any(m in part_conflict.lower() for m in conflict_markers)
        has_resolution = any(m in part_resolution.lower() for m in resolution_markers)
    else:
        # Word-based fallback
        text_lower = transcript.lower()
        words = text_lower.split()
        n = len(words)
        if n < 3:
            part_setup = part_conflict = part_resolution = text_lower
        else:
            i_setup = max(1, n * 40 // 100)
            i_conflict_start = max(0, n * 20 // 100)
            i_conflict_end = max(i_conflict_start + 1, n * 80 // 100)
            i_resolution = max(1, n * 40 // 100)
            part_setup = " ".join(words[:i_setup])
            part_conflict = " ".join(words[i_conflict_start:i_conflict_end])
            part_resolution = " ".join(words[-i_resolution:])
        
        has_setup = any(m in part_setup for m in setup_markers)
        has_conflict = any(m in part_conflict for m in conflict_markers)
        has_resolution = any(m in part_resolution for m in resolution_markers)
    
    found_markers = [m for m, v in [("setup", has_setup), ("conflict", has_conflict), ("resolution", has_resolution)] if v]
    structure_score = 0.0
    if has_setup:
        structure_score += 0.3
    if has_conflict:
        structure_score += 0.4
    if has_resolution:
        structure_score += 0.3
    if has_setup and has_conflict and has_resolution:
        structure_score += 0.2
    structure_score = min(1.0, structure_score)

    # ── Narrow evidence fields (evidence, not verdict) ─────────────────────────
    # These are the canonical text evidence signals for downstream pipeline.
    # Downstream role labeling and arc assembly should read THESE rather than
    # the aggregate structure_score.
    #
    # causality_evidence         — causal connectives present
    # temporal_progression_evidence — temporal ordering markers present
    # protagonist_evidence       — character/protagonist references present
    text_lower_full = transcript.lower() if transcript else ""
    _CAUSAL = ["поэтому", "потому что", "из-за", "благодаря", "в результате",
               "так как", "следовательно", "что привело", "это позволило"]
    _TEMPORAL = ["сначала", "потом", "затем", "после", "когда", "тогда", "позже", "вдруг"]
    _PROTAGONIST = ["я", "мы", "он", "она", "клиент", "заказчик", "человек"]
    words_full = text_lower_full.split()

    causal_hits = sum(1 for m in _CAUSAL if m in text_lower_full)
    temporal_hits = sum(1 for m in _TEMPORAL if m in text_lower_full)
    protagonist_hits = sum(1 for w in words_full if w in _PROTAGONIST)

    causality_evidence            = bool(causal_hits >= 1)
    temporal_progression_evidence = bool(temporal_hits >= 1)
    protagonist_evidence          = bool(protagonist_hits >= 1)

    # setup_evidence / conflict_evidence / resolution_evidence:
    # continuous signal [0..1] rather than binary flags, for role labeling
    setup_evidence      = round(float(has_setup) * 0.7 + float(temporal_hits > 0) * 0.3, 3)
    conflict_evidence   = round(float(has_conflict) * 0.7 + float(causal_hits > 0) * 0.3, 3)
    resolution_evidence = round(float(has_resolution), 3)

    # ── trigger_evidence: inciting event / запускающий момент ─────────────────
    # Distinct from setup_markers: these signal the moment something CHANGES
    # rather than providing background context.
    _TRIGGER_MARKERS = [
        "вдруг", "неожиданно", "тут", "в тот момент", "именно тогда",
        "всё изменилось", "стало ясно", "понял", "оказалось",
        "и вот", "тогда я", "и тут",
    ]
    trigger_hits = sum(1 for m in _TRIGGER_MARKERS if m in text_lower_full)
    trigger_evidence = round(float(np.clip(trigger_hits / 3.0, 0.0, 1.0)), 3)

    # ── resolution_closure_evidence: degree of actual closure ─────────────────
    # Distinct from resolution_evidence (which only checks for resolution markers).
    # This checks whether the text reads as ACTUALLY CLOSED vs just signalling closure.
    # Signals: finality language, past tense summary, explicit outcome language.
    _CLOSURE_STRONG = [
        "в итоге", "в конечном счёте", "всё закончилось", "получилось",
        "удалось", "разобрались", "это помогло", "после этого всё",
        "теперь", "с тех пор", "закрыли", "завершили", "решили",
    ]
    _CLOSURE_WEAK = [
        "итог", "вывод", "урок", "опыт", "понял", "научился",
    ]
    strong_closure_hits = sum(1 for m in _CLOSURE_STRONG if m in text_lower_full)
    weak_closure_hits   = sum(1 for m in _CLOSURE_WEAK   if m in text_lower_full)

    # Closure is checked in the last 40% of the text (position-aware)
    words_full_n = len(words_full)
    tail_words   = " ".join(words_full[max(0, words_full_n * 6 // 10):])
    tail_strong  = sum(1 for m in _CLOSURE_STRONG if m in tail_words)
    tail_weak    = sum(1 for m in _CLOSURE_WEAK   if m in tail_words)

    resolution_closure_evidence = round(float(np.clip(
        0.50 * float(np.clip(strong_closure_hits / 3.0, 0.0, 1.0)) +
        0.30 * float(np.clip(tail_strong / 2.0,         0.0, 1.0)) +
        0.20 * float(np.clip(weak_closure_hits / 4.0,   0.0, 1.0)) +
        0.10 * float(np.clip(tail_weak / 3.0,           0.0, 1.0)),
        0.0, 1.0,
    )), 3)

    return {
        "has_setup":      has_setup,
        "has_conflict":   has_conflict,
        "has_resolution": has_resolution,
        "story_markers":  found_markers,
        "structure_score": float(structure_score),
        # ── Evidence fields (downstream reads these, not structure_score) ──
        "setup_evidence":                    setup_evidence,
        "conflict_evidence":                 conflict_evidence,
        "resolution_evidence":               resolution_evidence,
        "causality_evidence":                causality_evidence,
        "temporal_progression_evidence":     temporal_progression_evidence,
        "protagonist_evidence":              protagonist_evidence,
        "trigger_evidence":                  trigger_evidence,
        "resolution_closure_evidence":       resolution_closure_evidence,
    }


def detect_narrativity_score(transcript: str) -> float:
    """
    Вычисление степени "нарративности" текста.
    
    Inspired by: Computationally-Narrativity-Detection
    https://github.com/maxsteg/Computationally-Narrativity-Detection
    
    Факторы:
    - Наличие персонажей (я, мы, клиент, он/она)
    - Временные связки (потом, затем, после)
    - Действия/события (глаголы прошедшего времени)
    - Причинно-следственные связи (поэтому, потому что, из-за)
    
    Returns:
        float 0-1, где 1 = высокая нарративность
    """
    if not transcript or len(transcript.strip()) < 30:
        return 0.0
    
    text_lower = transcript.lower()
    words = text_lower.split()
    
    score = 0.0
    
    # 1. Персонажи (25%)
    character_markers = ["я", "мы", "он", "она", "они", "клиент", "заказчик", "человек", "парень", "девушка"]
    char_count = sum(1 for word in words if word in character_markers)
    score += min(0.25, char_count / len(words) * 5)  # Нормализация
    
    # 2. Временные связки (25%)
    temporal_markers = ["потом", "затем", "после", "когда", "тогда", "сначала", "позже", "вдруг"]
    temp_count = sum(1 for marker in temporal_markers if marker in text_lower)
    score += min(0.25, temp_count / 10)
    
    # 3. Причинно-следственные связи (25%)
    causal_markers = ["поэтому", "потому что", "из-за", "благодаря", "в результате", "так что"]
    causal_count = sum(1 for marker in causal_markers if marker in text_lower)
    score += min(0.25, causal_count / 5)
    
    # 4. Story markers (25%)
    story_markers = ["история", "расскажу", "было", "случилось", "произошло", "событие"]
    story_count = sum(1 for marker in story_markers if marker in text_lower)
    score += min(0.25, story_count / 5)
    
    return float(np.clip(score, 0, 1))


def detect_story_arc_pattern(
    sentiment_curve: List[float],
    min_change: float = 0.3
) -> Tuple[str, float]:
    """
    Определение паттерна story arc по кривой sentiment.
    
    Inspired by:
    - pNarrative (arianbarakat) - Vonnegut plot shapes
    - NarrativeArc (BUTTER-Tools) - narrative tension curves
    
    Args:
        sentiment_curve: список sentiment scores по ходу сегмента
        min_change: минимальное изменение для детекции паттерна
    
    Returns:
        (arc_pattern, confidence)
    """
    if not sentiment_curve or len(sentiment_curve) < 3:
        return "flat", 0.0
    
    curve = np.array(sentiment_curve)
    
    # Нормализация к [-1, 1]
    curve_norm = (curve - curve.mean()) / (curve.std() + 1e-8)
    
    # Разбиваем на трети
    third = len(curve) // 3
    part1 = curve_norm[:third].mean()
    part2 = curve_norm[third:2*third].mean()
    part3 = curve_norm[2*third:].mean()
    
    # Детекция паттерна
    change_12 = part2 - part1
    change_23 = part3 - part2
    
    # Man in hole (down → up) - САМЫЙ ЧАСТЫЙ!
    if change_12 < -min_change and change_23 > min_change:
        confidence = min(abs(change_12), abs(change_23))
        return "man_in_hole", float(confidence)
    
    # Icarus (up → down)
    if change_12 > min_change and change_23 < -min_change:
        confidence = min(abs(change_12), abs(change_23))
        return "icarus", float(confidence)
    
    # Rags to riches (continuous up)
    if change_12 > min_change and change_23 > min_change:
        confidence = (abs(change_12) + abs(change_23)) / 2
        return "rags_to_riches", float(confidence)
    
    # Riches to rags (continuous down)
    if change_12 < -min_change and change_23 < -min_change:
        confidence = (abs(change_12) + abs(change_23)) / 2
        return "riches_to_rags", float(confidence)
    
    # Cinderella (up → down → up)
    total_change = abs(change_12) + abs(change_23)
    if total_change > min_change * 2:
        # Проверяем через полное сравнение
        if part1 < part2 and part2 > part3 and part3 > part1:
            return "cinderella", float(total_change / 2)
        if part1 > part2 and part2 < part3 and part3 < part1:
            return "oedipus", float(total_change / 2)
    
    # Flat (no clear pattern)
    return "flat", 0.2


# =============================================================================
# REAL FEATURES: SENTIMENT, EMOTIONS, SEMANTIC
# =============================================================================

def compute_sentiment_curve_from_asr(
    asr_segments: List[Dict],
    start_sec: float,
    end_sec: float,
    min_segments: int = 3,
    config: Optional[Dict] = None,
    llm_sentiment_callback: Optional[callable] = None
) -> List[float]:
    """
    Sentiment-кривая по ASR-сегментам внутри окна.
    
    LLM-first: если передан llm_sentiment_callback(segment_texts) -> List[float],
    используется его результат (значения 0..1). Иначе — лексикон только из
    SENTIMENT_MARKERS (без пересечения со структурой).
    
    Returns:
        List[float] sentiment по сегментам (0..1 для arc detector)
    """
    window_segs = [
        s for s in asr_segments
        if s.get("start", 0) < end_sec and s.get("end", 0) > start_sec
    ]
    if len(window_segs) < min_segments:
        return []
    
    # LLM-first: если есть колбэк — используем его
    if llm_sentiment_callback:
        try:
            texts = [(s.get("text") or "").strip() for s in window_segs]
            raw_curve = llm_sentiment_callback(texts)
            if isinstance(raw_curve, (list, tuple)) and len(raw_curve) == len(window_segs):
                return [float(np.clip(x, 0, 1)) for x in raw_curve]
        except Exception as e:
            logger.debug(f"LLM sentiment callback failed: {e}")
    
    cfg = config or STORY_CONFIG
    pos_list = cfg.get("sentiment_positive", SENTIMENT_MARKERS["sentiment_positive"])
    neg_list = cfg.get("sentiment_negative", SENTIMENT_MARKERS["sentiment_negative"])
    
    curve = []
    for seg in window_segs:
        text = (seg.get("text") or "").lower().strip()
        if not text:
            curve.append(0.5)
            continue
        words = set(text.split())
        pos_count = sum(1 for w in pos_list if w in words or w in text)
        neg_count = sum(1 for w in neg_list if w in words or w in text)
        raw = (pos_count - neg_count) / max(1, pos_count + neg_count + 1)
        curve.append(0.5 + 0.5 * np.clip(raw, -1, 1))
    return curve


def get_visual_audio_from_base_analysis(
    base_analysis: Optional[Dict],
    start_sec: float,
    end_sec: float,
) -> Tuple[Dict, Dict]:
    """
    Story-event visual/audio feature extractor (v2).

    Extracts two categories:

    LEGACY quality features (for backward-compat with compute_story_score):
        visual: clarity_score, composition_score, emotion_intensity, visual_stability
        audio:  speech_clarity, emotion_variance, speech_rate, silence_ratio

    STORY-EVENT features (new; used by arc/beat scoring):
        visual: speaker_change_event, character_entry_exit, scene_change,
                main_subject_continuity, object_state_change
        audio:  pause_before_payoff, prosody_shift, laughter_gasp_impact,
                music_rise_drop, speech_rate_shift

    Story-event features are returned as sub-dicts inside visual/audio dicts
    under the key "story_events", so callers can choose which tier to use.
    """
    dur = max(end_sec - start_sec, 1.0)

    visual: Dict = {
        "clarity_score":     0.65,
        "composition_score": 0.60,
        "emotion_intensity": 0.50,
        "visual_stability":  0.70,
        "story_events": {
            "speaker_change_count":      0,
            "character_entry_exit":      0.0,
            "scene_change_count":        0,
            "main_subject_continuity":   0.70,
            "object_state_change":       0.0,
            "visual_peak_in_window":     0.0,
        },
    }
    audio: Dict = {
        "speech_clarity":  0.75,
        "emotion_variance": 0.50,
        "speech_rate":      0.80,
        "silence_ratio":    0.15,
        "story_events": {
            "pause_before_payoff":  0.0,
            "prosody_shift":        0.0,
            "laughter_gasp_impact": 0.0,
            "music_rise_drop":      0.0,
            "speech_rate_shift":    0.0,
            "emotional_climax":     0.0,
        },
    }

    if not base_analysis:
        return visual, audio

    # ── time_series (primary source for event detection) ───────────────────
    ts = base_analysis.get("time_series") or base_analysis.get("emotion_curve") or {}
    n_ts = 0

    def _ts_window(key: str) -> Optional[np.ndarray]:
        """Slice a time_series array to [start_sec, end_sec]."""
        arr_raw = ts.get(key) if isinstance(ts, dict) else None
        if arr_raw is None or not isinstance(arr_raw, (list, np.ndarray)):
            return None
        arr = np.asarray(arr_raw, dtype=float)
        n   = len(arr)
        if n < 2:
            return None
        total_dur = float(base_analysis.get("duration", end_sec) or end_sec)
        i0 = max(0, int(start_sec / total_dur * n))
        i1 = min(n, int(end_sec   / total_dur * n) + 1)
        return arr[i0:i1] if i1 > i0 else None

    # ── LEGACY: emotion intensity / variance ─────────────────────────────
    for key in ["emotion_intensity", "valence", "arousal"]:
        win = _ts_window(key)
        if win is not None and len(win) > 0:
            visual["emotion_intensity"]  = float(np.clip(np.nanmean(win), 0, 1))
            audio["emotion_variance"]    = float(np.clip(np.nanstd(win),  0, 1)) if len(win) > 1 else 0.0
            n_ts += 1
            break

    for key in ["speech_rate", "speech_clarity"]:
        win = _ts_window(key)
        if win is not None and len(win) > 0:
            audio[key] = float(np.clip(np.nanmean(win), 0, 1))

    silence_win = _ts_window("silence_ratio")
    if silence_win is not None and len(silence_win) > 0:
        audio["silence_ratio"] = float(np.clip(np.nanmean(silence_win), 0, 1))

    # ── STORY-EVENT: visual ───────────────────────────────────────────────

    # Scene changes (shot cuts) — count within window
    shot_cut_arr = _ts_window("shot_cut") or _ts_window("scene_change")
    if shot_cut_arr is not None:
        visual["story_events"]["scene_change_count"] = int(np.sum(shot_cut_arr > 0.5))

    # Main subject continuity (face_presence or person_track)
    face_arr = _ts_window("face_presence") or _ts_window("person_track_confidence")
    if face_arr is not None and len(face_arr) > 0:
        visual["story_events"]["main_subject_continuity"] = float(
            np.clip(np.nanmean(face_arr), 0, 1)
        )
        # Entry/exit events: large drops or rises in face presence
        if len(face_arr) > 3:
            deriv   = np.abs(np.diff(face_arr))
            n_jumps = int(np.sum(deriv > 0.35))
            visual["story_events"]["character_entry_exit"] = float(
                np.clip(n_jumps / max(dur / 10.0, 1.0), 0, 1)
            )

    # Object state change (hand_object_interaction or object_state signal)
    obj_arr = _ts_window("hand_object_interaction") or _ts_window("object_state_change")
    if obj_arr is not None and len(obj_arr) > 0:
        visual["story_events"]["object_state_change"] = float(
            np.clip(np.nanmean(obj_arr), 0, 1)
        )

    # Visual peak within window (maximum emotion_intensity)
    em_arr = _ts_window("emotion_intensity") or _ts_window("valence")
    if em_arr is not None and len(em_arr) > 0:
        visual["story_events"]["visual_peak_in_window"] = float(
            np.clip(np.nanmax(em_arr), 0, 1)
        )

    # Speaker change (diarization or speaker_id signal)
    spk_arr = _ts_window("speaker_change") or _ts_window("speaker_id")
    if spk_arr is not None and len(spk_arr) > 1:
        changes = int(np.sum(np.abs(np.diff(spk_arr.astype(int))) > 0))
        visual["story_events"]["speaker_change_count"] = changes

    # ── STORY-EVENT: audio ────────────────────────────────────────────────

    # Prosody shift (arousal variance or pitch_variance)
    arousal_arr = _ts_window("arousal") or _ts_window("pitch_variance")
    if arousal_arr is not None and len(arousal_arr) > 1:
        audio["story_events"]["prosody_shift"] = float(
            np.clip(np.nanstd(arousal_arr) * 3.0, 0, 1)
        )
        audio["story_events"]["emotional_climax"] = float(
            np.clip(np.nanmax(arousal_arr), 0, 1)
        )

    # Pause before payoff — long silence in the last 30% of the window
    if silence_win is not None and len(silence_win) >= 4:
        last_30 = silence_win[int(len(silence_win) * 0.70):]
        audio["story_events"]["pause_before_payoff"] = float(
            np.clip(np.nanmax(last_30), 0, 1)
        )

    # Speech rate shift (variance in speech_rate signal)
    sr_arr = _ts_window("speech_rate")
    if sr_arr is not None and len(sr_arr) > 1:
        audio["story_events"]["speech_rate_shift"] = float(
            np.clip(np.nanstd(sr_arr) * 4.0, 0, 1)
        )

    # Laughter / gasp / impact events
    laugh_arr = _ts_window("laughter") or _ts_window("impact_sound") or _ts_window("gasp")
    if laugh_arr is not None and len(laugh_arr) > 0:
        audio["story_events"]["laughter_gasp_impact"] = float(
            np.clip(np.nanmax(laugh_arr), 0, 1)
        )

    # Music rise / drop
    music_arr = _ts_window("music_intensity") or _ts_window("background_music")
    if music_arr is not None and len(music_arr) > 1:
        deriv = np.diff(music_arr)
        rise  = float(np.nanmax(deriv))
        drop  = float(np.nanmin(deriv))
        audio["story_events"]["music_rise_drop"] = float(
            np.clip(max(abs(rise), abs(drop)), 0, 1)
        )

    # ── Segment-level fallback (face_emotions / segments) ────────────────
    segments = base_analysis.get("segments") or base_analysis.get("face_emotions") or []
    if segments and n_ts == 0:
        in_window = [
            s for s in segments
            if isinstance(s, dict) and s.get("start", 0) < end_sec and s.get("end", 0) > start_sec
        ]
        if in_window:
            intensities = []
            for s in in_window:
                v = s.get("emotion_intensity") or s.get("valence") or s.get("score")
                if v is not None:
                    try:
                        intensities.append(float(v))
                    except (TypeError, ValueError):
                        pass
            if intensities:
                visual["emotion_intensity"] = float(np.clip(np.mean(intensities), 0, 1))
                if len(intensities) > 1:
                    audio["emotion_variance"] = float(np.clip(np.std(intensities), 0, 1))

    return visual, audio


def compute_semantic_features_story(
    transcript: str,
    llm_callback: Optional[callable] = None,
    asr_segments: Optional[List[Dict]] = None,
) -> Dict:
    """
    Semantic features for story mode (v2.1): coherence, engagement, informativeness, has_takeaway.

    LLM-first: if llm_callback(text) → {coherence, engagement, informativeness, has_takeaway}
    is available, use it. Otherwise: discourse/event-based heuristics replace
    the old marker-count + length approach.

    Changes from v1:
    - coherence: discourse transition density + causal chain density (not just marker count)
    - informativeness: concrete-detail density (outcomes, numbers, names) > word count
    - engagement: first-person narrative intensity + emotional vocabulary + discourse hooks
    - has_takeaway: position-aware with stronger outcome/lesson vocabulary, uses asr_segments
      to verify the takeaway is actually in the last third of speech (not just anywhere)
    """
    out: Dict = {
        "coherence":       0.5,
        "informativeness": 0.5,
        "engagement":      0.5,
        "has_takeaway":    False,
    }

    if not transcript or len(transcript.strip()) < 20:
        return out

    if llm_callback and transcript.strip():
        try:
            raw = llm_callback(transcript)
            if isinstance(raw, dict):
                out["coherence"]       = float(np.clip(raw.get("coherence",       0.5), 0, 1))
                out["informativeness"] = float(np.clip(raw.get("informativeness", 0.5), 0, 1))
                out["engagement"]      = float(np.clip(raw.get("engagement",      0.5), 0, 1))
                out["has_takeaway"]    = bool(raw.get("has_takeaway", False))
            return out
        except Exception as e:
            logger.debug(f"LLM semantic callback failed: {e}")

    text_lower = transcript.lower()
    words      = text_lower.split()
    n_words    = max(len(words), 1)

    # ── COHERENCE: discourse transition density ────────────────────────────
    # Causal links + temporal ordering + contrast transitions = discourse coherence
    # Not just "how many of these words appear" but transition DENSITY per 100 words
    _CAUSAL_TRANS = ["поэтому", "потому что", "из-за", "благодаря", "в результате",
                     "так как", "следовательно", "что привело"]
    _TEMPORAL_TRANS = ["сначала", "затем", "потом", "после этого", "когда", "тогда", "позже"]
    _CONTRAST_TRANS = ["однако", "но", "хотя", "несмотря на", "зато", "при этом", "тем не менее"]
    _ELABORATION    = ["то есть", "иначе говоря", "например", "в частности", "а именно"]

    causal_count   = sum(1 for m in _CAUSAL_TRANS     if m in text_lower)
    temporal_count = sum(1 for m in _TEMPORAL_TRANS   if m in text_lower)
    contrast_count = sum(1 for m in _CONTRAST_TRANS   if m in text_lower)
    elaboration_c  = sum(1 for m in _ELABORATION      if m in text_lower)

    # Density per 100 words (discourse-rich text = coherent text)
    trans_density  = (causal_count + temporal_count + contrast_count + elaboration_c) / (n_words / 100.0)
    # Causal > temporal > contrast for coherence contribution
    causal_contrib = float(np.clip(causal_count   / 3.0, 0.0, 1.0))
    temporal_contrib = float(np.clip(temporal_count / 4.0, 0.0, 1.0))
    density_score  = float(np.clip(trans_density / 5.0, 0.0, 1.0))

    out["coherence"] = round(float(np.clip(
        0.40 * causal_contrib +
        0.25 * temporal_contrib +
        0.20 * density_score +
        0.15 * 0.5,   # residual floor
        0.0, 1.0,
    )), 3)

    # ── INFORMATIVENESS: concrete detail density ───────────────────────────
    # Outcome language + numbers + named/specific references
    _OUTCOME_WORDS  = ["получилось", "удалось", "достигли", "увеличили", "снизили",
                       "выросло", "упало", "решили", "запустили", "закрыли",
                       "нашли", "потеряли", "сократили", "внедрили"]
    _SPECIFIC_WORDS = ["процент", "%", "раз", "человек", "тысяч", "млн",
                       "минут", "недель", "месяц", "год", "рублей", "долларов"]

    has_digits     = any(c.isdigit() for c in transcript)
    outcome_hits   = sum(1 for w in _OUTCOME_WORDS  if w in text_lower)
    specific_hits  = sum(1 for w in _SPECIFIC_WORDS if w in text_lower)

    # Length matters less than concreteness
    length_score   = float(np.clip(n_words / 120.0, 0.3, 1.0))
    concreteness   = float(np.clip(
        float(has_digits) * 0.3 +
        float(np.clip(outcome_hits  / 3.0, 0.0, 0.4)) +
        float(np.clip(specific_hits / 4.0, 0.0, 0.3)),
        0.0, 1.0,
    ))

    out["informativeness"] = round(float(np.clip(
        0.55 * concreteness +
        0.45 * length_score,
        0.0, 1.0,
    )), 3)

    # ── ENGAGEMENT: narrative intensity + emotional vocabulary ─────────────
    # First-person narrative (я/мы говорю), emotional vocabulary density,
    # direct-address hooks ("представьте", "вы"), exclamation/contrast signals
    _FIRST_PERSON   = ["я ", "мы ", "я,", "я.", "мне ", "нас ", "нам ",
                       "расскажу", "помню", "думал", "решил", "понял"]
    _EMOTIONAL_VOCAB = ["страшно", "неожиданно", "вдруг", "счастлив",
                        "провал", "успех", "ошибка", "прорыв", "боялся",
                        "надеялся", "разочарован", "горд", "удивил"]
    _HOOKS          = ["представьте", "например", "и вот", "а теперь",
                       "самое интересное", "важный момент", "главное"]

    first_person_count = sum(1 for m in _FIRST_PERSON   if m in text_lower)
    emotion_count      = sum(1 for m in _EMOTIONAL_VOCAB if m in text_lower)
    hook_count         = sum(1 for m in _HOOKS           if m in text_lower)

    fp_score    = float(np.clip(first_person_count / 4.0, 0.0, 1.0))
    emo_score   = float(np.clip(emotion_count      / 4.0, 0.0, 1.0))
    hook_score  = float(np.clip(hook_count         / 3.0, 0.0, 1.0))

    out["engagement"] = round(float(np.clip(
        0.40 * fp_score +
        0.35 * emo_score +
        0.25 * hook_score,
        0.0, 1.0,
    )), 3)

    # ── HAS TAKEAWAY: position-aware with outcome/lesson vocabulary ────────
    # Takeaway must appear in the LAST THIRD of speech — not just anywhere.
    # If ASR segments available, use temporal position; else use word position.
    _TAKEAWAY_STRONG = ["в итоге", "главный вывод", "урок из этого", "что я понял",
                        "опыт показал", "с тех пор", "после этого я", "теперь я знаю",
                        "если бы я", "совет", "рекомендую", "важно понять", "главное"]
    _TAKEAWAY_WEAK   = ["в общем", "итого", "теперь", "вывод", "напоследок",
                        "подводя итог", "резюмируя"]

    if asr_segments:
        # Use last third of ASR segments by start time
        sorted_segs    = sorted(asr_segments, key=lambda s: s.get("start", 0))
        cutoff_idx     = max(0, len(sorted_segs) * 2 // 3)
        tail_text      = " ".join(s.get("text", "") for s in sorted_segs[cutoff_idx:]).lower()
    else:
        tail_text      = " ".join(words[max(0, n_words * 2 // 3):])

    strong_in_tail = sum(1 for m in _TAKEAWAY_STRONG if m in tail_text)
    weak_in_tail   = sum(1 for m in _TAKEAWAY_WEAK   if m in tail_text)
    strong_anywhere = sum(1 for m in _TAKEAWAY_STRONG if m in text_lower)
    config_markers  = STORY_CONFIG.get("takeaway_markers", [])
    config_in_tail  = sum(1 for m in config_markers if m in tail_text)

    out["has_takeaway"] = bool(
        strong_in_tail >= 1 or
        weak_in_tail   >= 2 or
        config_in_tail >= 1 or
        (strong_anywhere >= 1 and len(tail_text.split()) < 20)  # short clip = whole text is tail
    )

    return out


def detect_has_takeaway(transcript: str, config: Optional[Dict] = None) -> bool:
    """Явный вывод/урок в тексте (маркеры из config или STORY_CONFIG)."""
    if not transcript or len(transcript.strip()) < 20:
        return False
    cfg = config or STORY_CONFIG
    markers = cfg.get("takeaway_markers", STORY_CONFIG["takeaway_markers"])
    text_lower = transcript.lower()
    words = text_lower.split()
    tail = " ".join(words[-max(1, len(words) // 3):])
    return any(m in tail for m in markers) or any(m in text_lower for m in ["в итоге", "теперь"])


def detect_story_type(
    transcript: str,
    arc_pattern: str = "flat",
    has_takeaway: bool = False,
    sentiment_curve: Optional[List[float]] = None,
) -> Tuple[str, float]:
    """
    Определение типа момента с уточнёнными правилами классификации.

    Правила (приоритет убывает вниз):
    ┌──────────────────┬──────────────────────────────────────────────────────────────┐
    │ Тип              │ Критерии                                                     │
    ├──────────────────┼──────────────────────────────────────────────────────────────┤
    │ lesson_learned   │ has_takeaway=True (обязательно) + маркеры вывода             │
    │ failure_story    │ маркеры провала + arc in (riches_to_rags, oedipus, icarus)   │
    │ client_case      │ маркеры клиента + профессиональный контекст (продукт/метрика)│
    │ success_story    │ маркеры успеха + позитивная тональность в конце              │
    │ anecdote         │ маркеры байки/юмора                                          │
    │ personal_story   │ я-форма, нет клиента, нет явного провала/успеха             │
    │ non_story        │ ничего из вышеперечисленного / score < порога                │
    └──────────────────┴──────────────────────────────────────────────────────────────┘

    Конфликт типов: client_case + failure_story → failure_story побеждает, если
    arc явно негативный; иначе client_case.

    Returns:
        (story_type, confidence 0..1)
    """
    if not transcript or len(transcript.strip()) < 30:
        return "non_story", 0.0

    text_lower = transcript.lower()
    words = text_lower.split()
    n = len(words)

    # --- Базовые маркерные скоры ---
    type_scores: Dict[str, float] = {}
    for stype, cfg in STORY_SEGMENT_TYPES.items():
        if stype == "non_story":
            continue
        markers = cfg.get("markers", [])
        score = sum(1 for m in markers if m in text_lower)
        if score > 0:
            type_scores[stype] = float(score)

    if not type_scores:
        return "non_story", 0.0

    # --- Уточняющие правила ---

    # lesson_learned: требует has_takeaway как обязательное условие
    if "lesson_learned" in type_scores and not has_takeaway:
        del type_scores["lesson_learned"]

    # failure_story: требует негативную дугу ИЛИ сильное падение sentiment в начале
    FAILURE_ARCS = {"riches_to_rags", "oedipus", "icarus"}
    if "failure_story" in type_scores:
        arc_ok = arc_pattern in FAILURE_ARCS
        # Дополнительная проверка: если кривая есть — смотрим на первые 30%
        if not arc_ok and sentiment_curve and len(sentiment_curve) >= 3:
            first_third = sentiment_curve[:max(1, len(sentiment_curve) // 3)]
            arc_ok = float(np.mean(first_third)) > 0.55  # начинали с позитива
        if not arc_ok:
            type_scores["failure_story"] *= 0.6  # штраф, но не удаляем

    # client_case: требует профессионального контекста
    CLIENT_PROFESSIONAL = ["продукт", "метрика", "кампания", "бюджет", "проект",
                           "запуск", "команда", "рынок", "клиент", "заказчик"]
    if "client_case" in type_scores:
        prof_count = sum(1 for w in CLIENT_PROFESSIONAL if w in text_lower)
        if prof_count == 0:
            type_scores["client_case"] *= 0.5  # без профконтекста — штраф

    # success_story: позитивный хвост кривой усиливает
    if "success_story" in type_scores and sentiment_curve and len(sentiment_curve) >= 3:
        tail = sentiment_curve[-max(1, len(sentiment_curve) // 3):]
        if float(np.mean(tail)) > 0.65:
            type_scores["success_story"] *= 1.3  # бонус за позитивный конец

    # Конфликт client_case vs failure_story: при явно негативной дуге — failure побеждает
    if "client_case" in type_scores and "failure_story" in type_scores:
        if arc_pattern in FAILURE_ARCS:
            type_scores["client_case"] = min(
                type_scores["client_case"],
                type_scores["failure_story"] * 0.8
            )

    if not type_scores:
        return "non_story", 0.0

    best_type, best_score = max(type_scores.items(), key=lambda x: x[1])
    confidence = float(np.clip(best_score / 5.0, 0, 1))
    return best_type, confidence


def _estimate_arc_compression(arc: Dict) -> float:
    """
    Estimate compression quality directly from arc data (no quality layer needed).
    Used in compute_narrative_score() to avoid chicken-and-egg with compute_story_quality().

    High score = arc is dense and purposeful (low redundancy, good causal density).
    """
    completeness  = float(arc.get("arc_completeness",    0.0))
    causal_density = float(arc.get("causal_density",     0.0))
    redundancy    = float(arc.get("mid_arc_redundancy",  0.5))
    n_beats       = max(int(arc.get("n_beats",           1)), 1)
    seq_score     = float(arc.get("sequence_order_score", 0.5))

    # Prefer arcs with 3-6 beats (not too sparse, not too redundant)
    beat_density_score = float(np.clip(
        1.0 - abs(n_beats - 4) / 6.0, 0.3, 1.0
    ))

    return float(np.clip(
        0.35 * completeness +
        0.25 * (1.0 - redundancy) +
        0.20 * causal_density +
        0.15 * beat_density_score +
        0.05 * seq_score,
        0.0, 1.0,
    ))


def compute_narrative_score(
    story_structure: Dict,
    narrativity_score: float,
    story_arc: str,
    arc_confidence: float,
    story_type: str = "personal_story",
    # ── Arc/role evidence (primary inputs for arc-aware path) ────────────────
    arc_roles: Optional[List[str]] = None,
    arc_coherence: float = 0.0,
    payoff_strength: float = 0.0,
    compression_quality: float = -1.0,   # -1 = auto-estimate from arc
    character_continuity: float = 0.5,
    topic_continuity: float = 0.5,
    type_ambiguity: float = 0.5,
    arc_completeness: float = 0.0,
    # ── Richer arc evidence (v2, passed from build_story_arcs output) ────────
    arc_raw: Optional[Dict] = None,       # full arc dict for auto-compression estimate
    trigger_evidence: float = 0.0,        # inciting-event signal (new in v2)
    resolution_closure: float = 0.0,      # actual closure quality (new in v2)
    type_profile_match: float = 0.5,      # canonical type profile match (new in v2)
) -> float:
    """
    Arc-aware narrative score — primary story engine core (v2.1).

    Design principle: when arc data is present, THIS function is the story
    verdict engine. Text signals (structure, narrativity) are weak priors only.

    Scoring tiers
    -------------
    1. ARC QUALITY          (48%)  — completeness, coherence, payoff, trigger
       The arc IS the story. This is the dominant signal.

    2. NARRATIVE CONTINUITY (24%)  — character/topic continuity, compression,
       type_ambiguity, type_profile_match
       Coherence of the story across its duration.

    3. TEXT EVIDENCE        (18%)  — evidence fields only (NOT structure_score).
       Marker hits are weak signal, not gate. Now includes trigger + closure.

    4. TEXT PRIOR           (10%)  — narrativity_score capped at 0.6, weight 10%.
       Kept only as a soft floor. Lexical richness ≠ story quality.
       Cannot rescue a weak arc or penalise a strong one.

    Backward compat: if arc_completeness=0 AND arc_coherence=0 AND
    payoff_strength=0 (window fallback path), falls back to a deliberately
    weaker formula to enforce ranking arc-based moments higher.
    """
    has_arc_data = arc_completeness > 0.0 or arc_coherence > 0.0 or payoff_strength > 0.0

    if has_arc_data:
        # ── Auto-estimate compression_quality from arc dict if not supplied ──
        if compression_quality < 0.0:
            compression_quality = _estimate_arc_compression(arc_raw or {})
        compression_quality = float(np.clip(compression_quality, 0.0, 1.0))

        # ── Tier 1: Arc quality (48%) ─────────────────────────────────────────
        arc_quality = float(np.clip(
            0.38 * arc_completeness +
            0.32 * arc_coherence +
            0.22 * payoff_strength +
            0.08 * float(np.clip(trigger_evidence, 0.0, 1.0)),
            0.0, 1.0,
        ))

        # Penalty: arc without payoff is narratively incomplete
        if payoff_strength < 0.15:
            arc_quality *= 0.72

        # Penalty: arc with no trigger / inciting event is weaker
        if trigger_evidence < 0.05 and arc_completeness < 0.50:
            arc_quality *= 0.90

        # ── Tier 2: Narrative continuity (24%) ───────────────────────────────
        narrative_continuity = float(np.clip(
            0.28 * character_continuity +
            0.28 * topic_continuity +
            0.22 * compression_quality +
            0.12 * (1.0 - type_ambiguity) +
            0.10 * float(np.clip(type_profile_match, 0.0, 1.0)),
            0.0, 1.0,
        ))

        # ── Tier 3: Text evidence (18%) — evidence fields only ────────────────
        setup_ev      = float(story_structure.get("setup_evidence",              0.0))
        conflict_ev   = float(story_structure.get("conflict_evidence",           0.0))
        resolution_ev = float(story_structure.get("resolution_evidence",         0.0))
        closure_ev    = float(story_structure.get("resolution_closure_evidence", resolution_closure))
        trigger_ev    = float(story_structure.get("trigger_evidence",            trigger_evidence))
        causal_ev     = 1.0 if story_structure.get("causality_evidence", False) else 0.0
        text_evidence = float(np.clip(
            0.20 * setup_ev +
            0.25 * conflict_ev +
            0.20 * resolution_ev +
            0.15 * closure_ev +
            0.12 * trigger_ev +
            0.08 * causal_ev,
            0.0, 1.0,
        ))

        # ── Tier 4: Text narrativity prior (10%) — soft floor only ───────────
        # Cap at 0.6: very high narrativity_score shouldn't rescue weak arc
        text_prior = float(np.clip(narrativity_score, 0.0, 0.6))

        base_score = float(np.clip(
            0.48 * arc_quality +
            0.24 * narrative_continuity +
            0.18 * text_evidence +
            0.10 * text_prior,
            0.0, 1.0,
        ))

        # ── Type × arc micro-bonus (capped at +0.06) ─────────────────────────
        bonus = 0.0
        if story_type == "failure_story" and story_arc in ("riches_to_rags", "man_in_hole", "icarus"):
            bonus = 0.04
        elif story_type == "success_story" and story_arc in ("rags_to_riches", "cinderella"):
            bonus = 0.04
        elif story_type == "client_case" and story_arc in ("man_in_hole", "rags_to_riches", "cinderella"):
            bonus = 0.03

        # Role sequence bonus (inciting_event + payoff = minimal viable arc)
        if arc_roles:
            role_set = set(arc_roles)
            if "payoff" in role_set and ("inciting_event" in role_set or "tension" in role_set):
                bonus += 0.03
            if "setup" in role_set and "payoff" in role_set and "tension" in role_set:
                bonus += 0.02

        return float(np.clip(base_score + bonus, 0.0, 1.0))

    else:
        # ── Fallback path (window-based, no arc data) ─────────────────────────
        # Deliberately weaker ceiling: window candidates should score lower than
        # arc-based ones so quality-aware ranking keeps arc moments on top.
        #
        # Key change from previous version:
        #   narrativity_score weight: 25% → 8%   (lexical prior demotion)
        #   structure_score weight:   35% → 22%  (verdict → evidence proxy)
        #   arc_confidence weight:    30% → 35%  (best proxy signal in fallback)
        #   text_evidence from fields: 0% → 20%  (evidence fields > structure_score)

        setup_ev      = float(story_structure.get("setup_evidence",    0.0))
        conflict_ev   = float(story_structure.get("conflict_evidence", 0.0))
        resolution_ev = float(story_structure.get("resolution_evidence", 0.0))
        text_ev_field = float(np.clip(
            0.25 * setup_ev + 0.45 * conflict_ev + 0.30 * resolution_ev, 0.0, 1.0
        ))

        arc_weight   = STORY_ARC_PATTERNS.get(story_arc, {}).get("weight", 1.0)
        arc_score    = float(np.clip(arc_confidence * arc_weight / MAX_ARC_WEIGHT, 0, 1))

        # Cap narrativity_score at 0.6 before using as prior
        narr_prior   = float(np.clip(narrativity_score, 0.0, 0.6))

        base_score = float(np.clip(
            0.35 * arc_score +
            0.22 * text_ev_field +
            0.20 * float(story_structure.get("structure_score", 0.0)) +
            0.15 * float(np.clip(arc_confidence, 0, 1)) +
            0.08 * narr_prior,
            0.0, 1.0,
        ))

        # Apply ceiling: fallback scores max at 0.80 to preserve arc-based ranking
        base_score = min(base_score, 0.80)

        bonus = 0.0
        if story_type == "failure_story" and story_arc == "riches_to_rags":
            bonus = 0.04
        elif story_type == "success_story" and story_arc == "rags_to_riches":
            bonus = 0.04
        elif story_type == "client_case" and story_arc in ("rags_to_riches", "cinderella"):
            bonus = 0.02

        return float(np.clip(base_score + bonus, 0.0, 0.80))


def compute_length_weight(
    duration_sec: float,
    story_type: str,
    preferred_duration: float = 75.0
) -> float:
    """
    Story-length reward: бонус, когда duration близка к preferred для типа.
    Слишком короткое/длинное окно для типа — штраф.
    """
    config = STORY_SEGMENT_TYPES.get(story_type, {})
    min_d = config.get("min_duration", 30)
    max_d = config.get("max_duration", 120)
    if duration_sec < min_d:
        return 0.7  # штраф: для такого типа мало
    if duration_sec > max_d:
        return 0.85  # слегка штрафуем за слишком длинное
    # Идеал — около preferred_duration
    diff = abs(duration_sec - preferred_duration)
    if diff <= 15:
        return 1.0
    if diff <= 30:
        return 0.95
    return 0.9


_REASONS_FAMILY = {
    # narrative
    "arc_match":             "narrative",
    "strong_narrative":      "narrative",
    "has_conflict":          "narrative",
    "has_resolution":        "narrative",
    "has_setup":             "narrative",
    # sentiment / arc
    "high_emotional_change": "sentiment",
    "positive_tail":         "sentiment",
    # semantic
    "explicit_takeaway":     "semantic",
    "semantic_coherence":    "semantic",
    "high_informativeness":  "semantic",
    # visual/audio
    "audio_emotion":         "audio",
    "visual_peak":           "visual",
    # story type
    "strong_type_match":     "type",
}


def build_story_reasons(
    narrative_features: Dict,
    audio_features: Dict,
    semantic_features: Dict,
    visual_features: Dict,
    story_type: str,
    has_takeaway: bool,
    max_reasons: int = 5,
) -> List[Dict]:
    """
    Формирует структурированный список причин отбора момента.

    Каждая причина: {"code": str, "message": str, "weight": float}
    - code: из фиксированного словаря _REASONS_FAMILY
    - weight: 0..1, используется re-ranker и UI для приоритизации

    Правила:
    - Топ max_reasons по weight, но гарантируется минимум одна narrative-причина
      и минимум одна из групп (sentiment/semantic/audio/visual), если есть.
    - Не более 2 причин из одной family.
    """
    candidates: List[Dict] = []

    arc = narrative_features.get("arc_pattern", "flat")
    arc_conf = float(narrative_features.get("arc_confidence", 0.0))
    narrative_score = float(narrative_features.get("narrative_score", 0.0))

    # --- Narrative ---
    if narrative_features.get("has_conflict"):
        candidates.append({"code": "has_conflict", "message": "Есть конфликт/проблема", "weight": 0.75})
    if narrative_features.get("has_resolution"):
        candidates.append({"code": "has_resolution", "message": "Есть развязка", "weight": 0.65})
    if narrative_features.get("has_setup"):
        candidates.append({"code": "has_setup", "message": "Есть завязка", "weight": 0.55})
    if arc != "flat" and arc_conf > 0.3:
        arc_desc = STORY_ARC_PATTERNS.get(arc, {}).get("description", arc)
        candidates.append({"code": "arc_match", "message": f"Дуга: {arc_desc} ({arc_conf:.2f})", "weight": round(arc_conf, 2)})
    if narrative_score > 0.70:
        candidates.append({"code": "strong_narrative", "message": f"Высокий narrative score ({narrative_score:.2f})", "weight": round(narrative_score, 2)})

    # --- Sentiment / Audio ---
    em_var = float(audio_features.get("emotion_variance", 0.0))
    if em_var > 0.55:
        candidates.append({"code": "high_emotional_change", "message": f"Высокая эмоциональная динамика ({em_var:.2f})", "weight": round(em_var, 2)})
    em_int = float(audio_features.get("emotion_intensity", 0.0))
    if em_int > 0.65:
        candidates.append({"code": "audio_emotion", "message": f"Интенсивные эмоции в голосе ({em_int:.2f})", "weight": round(em_int, 2)})

    # --- Semantic ---
    if has_takeaway:
        candidates.append({"code": "explicit_takeaway", "message": "Явный вывод/урок", "weight": 0.80})
    coherence = float(semantic_features.get("coherence", 0.0))
    if coherence > 0.70:
        candidates.append({"code": "semantic_coherence", "message": f"Высокая связность ({coherence:.2f})", "weight": round(coherence, 2)})
    info = float(semantic_features.get("informativeness", 0.0))
    if info > 0.70:
        candidates.append({"code": "high_informativeness", "message": f"Высокая информативность ({info:.2f})", "weight": round(info, 2)})

    # --- Visual ---
    v_int = float(visual_features.get("emotion_intensity", 0.0))
    if v_int > 0.65:
        candidates.append({"code": "visual_peak", "message": f"Визуальный пик эмоций ({v_int:.2f})", "weight": round(v_int, 2)})

    # --- Story type ---
    type_weight = STORY_SEGMENT_TYPES.get(story_type, {}).get("weight", 1.0)
    if type_weight >= 1.25:
        candidates.append({"code": "strong_type_match", "message": f"Тип: {story_type} (weight={type_weight})", "weight": round((type_weight - 1.0), 2)})

    # --- Отбор: топ max_reasons, не более 2 из одной family, гарантируем narrative ---
    candidates.sort(key=lambda x: x["weight"], reverse=True)
    family_counts: Dict[str, int] = {}
    result: List[Dict] = []
    narrative_added = False

    for c in candidates:
        family = _REASONS_FAMILY.get(c["code"], "other")
        if family_counts.get(family, 0) >= 2:
            continue
        result.append(c)
        family_counts[family] = family_counts.get(family, 0) + 1
        if family == "narrative":
            narrative_added = True
        if len(result) >= max_reasons:
            break

    # Гарантируем хотя бы одну narrative-причину
    if not narrative_added:
        for c in candidates:
            if _REASONS_FAMILY.get(c["code"]) == "narrative" and c not in result:
                result.insert(0, c)
                if len(result) > max_reasons:
                    result = result[:max_reasons]
                break

    return result


def compute_beat_score(beat: Dict) -> float:
    """
    Level 1: Score a single narrative beat.
    Factors: mean_strength, role_confidence, n_sources (multimodal coverage).
    Returns [0..1].
    """
    strength    = float(beat.get("mean_strength",    0.3))
    role_conf   = float(beat.get("role_confidence",  0.3))
    n_src       = int(beat.get("n_sources",          1))
    src_bonus   = float(np.clip((n_src - 1) / 3.0, 0.0, 0.2))
    return round(float(np.clip(0.55 * strength + 0.35 * role_conf + src_bonus, 0.0, 1.0)), 3)


def compute_arc_score(arc: Dict) -> float:
    """
    Level 2: Score a story arc as a narrative sequence.
    Does NOT consider clip packaging — that is clip_score's job.
    Factors: arc_completeness, coherence_score, payoff_strength,
             payoff_dependency_score, causal_density, mid_arc_redundancy.
    Returns [0..1].
    """
    completeness   = float(arc.get("arc_completeness",        0.0))
    coherence      = float(arc.get("coherence_score",         0.3))
    payoff         = float(arc.get("payoff_strength",         0.0))
    payoff_dep     = float(arc.get("payoff_dependency_score", 0.3))
    causal         = float(arc.get("causal_density",          0.0))
    redundancy     = float(arc.get("mid_arc_redundancy",      0.5))
    arc_failures   = len(arc.get("arc_failure_modes",         []))

    base = float(np.clip(
        0.30 * completeness +
        0.25 * coherence +
        0.20 * payoff +
        0.15 * payoff_dep +
        0.10 * float(np.clip(causal * 3.0, 0.0, 1.0)),
        0.0, 1.0,
    ))
    # Redundancy and failure penalties
    base = base * (1.0 - 0.15 * redundancy) - arc_failures * 0.07

    # Bonus: beat-level mean score contribution
    beats = arc.get("beats", [])
    if beats:
        mean_beat = float(np.mean([compute_beat_score(b) for b in beats]))
        base      = base * 0.80 + mean_beat * 0.20

    return round(float(np.clip(base, 0.0, 1.0)), 3)


def compute_clip_score(
    arc_score_val: float,
    visual_features: Dict,
    audio_features: Dict,
    semantic_features: Dict,
    narrative_features: Dict,
    story_type: str,
    duration_sec: float = 60.0,
    has_takeaway: bool = False,
    clip_self_sufficiency: float = 0.5,
    config: Optional["StoryModeConfig"] = None,
) -> Tuple[float, Dict]:
    """
    Level 3: Score how well an arc packages into a clip.
    Combines arc_score with multimodal quality and clip packaging suitability.
    Returns (clip_score, subscores_dict).
    """
    weights       = (config.weights if config else None) or MODE_CONFIGS["story"]["weights"]
    preferred_dur = (config.preferred_duration if config else None) or MODE_CONFIGS["story"].get("preferred_duration", 75.0)

    # ── Visual score (with story-event bonus) ────────────────────────────
    vs_evts  = visual_features.get("story_events", {})
    vis_base = float(np.clip(
        0.30 * visual_features.get("clarity_score",     0.5) +
        0.20 * visual_features.get("composition_score", 0.5) +
        0.25 * visual_features.get("emotion_intensity", 0.5) +
        0.25 * visual_features.get("visual_stability",  0.5),
        0, 1,
    ))
    vis_event_bonus = float(np.clip(
        0.30 * vs_evts.get("visual_peak_in_window",   0.0) +
        0.40 * vs_evts.get("main_subject_continuity", 0.5) +
        0.30 * float(np.clip(1.0 - vs_evts.get("scene_change_count", 0) / 10.0, 0.0, 1.0)),
        0, 1,
    )) if vs_evts else 0.0
    visual_score = float(np.clip(0.70 * vis_base + 0.30 * vis_event_bonus, 0, 1))

    # ── Audio score (with story-event bonus) ────────────────────────────
    au_evts   = audio_features.get("story_events", {})
    aud_base  = float(np.clip(
        0.35 * audio_features.get("speech_clarity",  0.5) +
        0.30 * audio_features.get("emotion_variance", 0.5) +
        0.20 * audio_features.get("speech_rate",      0.5) +
        0.15 * (1.0 - abs(audio_features.get("silence_ratio", 0.15) - 0.15) / 0.85),
        0, 1,
    ))
    aud_event_bonus = float(np.clip(
        0.35 * au_evts.get("emotional_climax",     0.0) +
        0.25 * au_evts.get("prosody_shift",         0.0) +
        0.20 * au_evts.get("pause_before_payoff",   0.0) +
        0.20 * au_evts.get("laughter_gasp_impact",  0.0),
        0, 1,
    )) if au_evts else 0.0
    audio_score = float(np.clip(0.70 * aud_base + 0.30 * aud_event_bonus, 0, 1))

    # ── Semantic score ───────────────────────────────────────────────────
    semantic_score = float(np.clip(
        0.40 * semantic_features.get("coherence",       0.5) +
        0.35 * semantic_features.get("informativeness", 0.5) +
        0.25 * semantic_features.get("engagement",      0.5),
        0, 1,
    ))

    # ── Narrative score (from narrative_features dict) ───────────────────
    narrative_score = float(np.clip(narrative_features.get("narrative_score", 0.3), 0, 1))

    type_weight   = STORY_SEGMENT_TYPES.get(story_type, {}).get("weight", 1.0)
    length_weight = compute_length_weight(duration_sec, story_type, preferred_dur)

    modal_score = float(np.clip(
        weights["visual"]    * visual_score +
        weights["audio"]     * audio_score +
        weights["semantic"]  * semantic_score +
        weights["narrative"] * narrative_score,
        0, 1,
    ))

    # ── Combine arc quality + modal quality + packaging ──────────────────
    final = float(np.clip(
        0.45 * arc_score_val +      # arc quality is primary
        0.35 * modal_score +        # multimodal confirms the arc
        0.20 * clip_self_sufficiency,  # packaging suitability
        0, 1,
    ))
    final = float(np.clip(final * type_weight * length_weight, 0, 1))

    if has_takeaway:
        final = min(1.0, final + 0.05)

    subscores = {
        "visual":             round(visual_score,          3),
        "audio":              round(audio_score,           3),
        "semantic":           round(semantic_score,        3),
        "narrative":          round(narrative_score,       3),
        "arc":                round(arc_score_val,         3),
        "clip_self_sufficiency": round(clip_self_sufficiency, 3),
        "modal":              round(modal_score,           3),
    }
    return round(final, 3), subscores


def compute_story_score(
    visual_features: Dict,
    audio_features: Dict,
    semantic_features: Dict,
    narrative_features: Dict,
    story_type: str,
    duration_sec: float = 60.0,
    has_takeaway: bool = False,
    config: Optional["StoryModeConfig"] = None,
    # ── Arc inputs (optional; from primary pipeline) ──────────────────────
    arc: Optional[Dict] = None,
    clip_self_sufficiency: float = 0.5,
) -> Tuple[float, Dict, List[Dict]]:
    """
    Three-level story score (v2): beat → arc → clip.

    If `arc` is provided (primary pipeline), scores at all three levels.
    If `arc` is None (window fallback), uses legacy modal weighting only.

    Returns (final_score 0..1, subscores dict, reasons list).
    """
    # Level 2: arc score
    arc_score_val = compute_arc_score(arc) if arc is not None else 0.0

    # Level 3: clip score
    final_score, subscores = compute_clip_score(
        arc_score_val=arc_score_val,
        visual_features=visual_features,
        audio_features=audio_features,
        semantic_features=semantic_features,
        narrative_features=narrative_features,
        story_type=story_type,
        duration_sec=duration_sec,
        has_takeaway=has_takeaway,
        clip_self_sufficiency=clip_self_sufficiency,
        config=config,
    )

    reasons = build_story_reasons(
        narrative_features=narrative_features,
        audio_features=audio_features,
        semantic_features=semantic_features,
        visual_features=visual_features,
        story_type=story_type,
        has_takeaway=has_takeaway,
    )

    return float(final_score), subscores, reasons


# =============================================================================
# DISCOURSE DENSITY SCORERS  (text narrativity, event density, causal density)
# =============================================================================

def compute_text_narrativity_prior(transcript: str) -> float:
    """
    Weak text-based narrativity prior (formerly detect_narrativity_score).
    Kept as a soft signal; should NOT be the primary story gate.
    """
    return detect_narrativity_score(transcript)


def compute_event_density_score(
    asr_segments: List[Dict],
    start: float,
    end: float,
) -> float:
    """
    Density of discourse-level events in a window.
    Events = ASR turn boundaries (gaps >= 0.3s) + strong discourse markers.
    Returns [0..1].
    """
    window_segs = [s for s in asr_segments if s.get("start", 0) < end and s.get("end", 0) > start]
    if not window_segs:
        return 0.0

    duration = max(end - start, 1.0)
    sorted_segs = sorted(window_segs, key=lambda s: s.get("start", 0))

    n_turns = 0
    for i in range(len(sorted_segs) - 1):
        gap = float(sorted_segs[i + 1].get("start", 0)) - float(sorted_segs[i].get("end", 0))
        if gap >= 0.30:
            n_turns += 1

    all_text = " ".join(s.get("text", "") for s in window_segs).lower()
    _DISCOURSE = [
        "однако", "но", "вдруг", "поэтому", "потому что", "в результате",
        "тогда", "затем", "после", "в итоге", "оказалось", "неожиданно"
    ]
    n_markers = sum(1 for m in _DISCOURSE if m in all_text)

    turn_density   = float(np.clip(n_turns    / max(duration / 60.0, 0.1) / 10.0, 0.0, 1.0))
    marker_density = float(np.clip(n_markers  / max(duration / 60.0, 0.1) /  8.0, 0.0, 1.0))

    return round(float(np.clip(0.60 * turn_density + 0.40 * marker_density, 0.0, 1.0)), 3)


def compute_causal_density_score(transcript: str) -> float:
    """
    Density of causal connectives — signals that events are causally linked.
    High causal density = story-like rather than list-like.
    Returns [0..1].
    """
    if not transcript or len(transcript) < 20:
        return 0.0
    text_lower = transcript.lower()
    words = text_lower.split()
    n = max(1, len(words))
    _CAUSAL = [
        "потому что", "из-за", "благодаря", "в результате", "поэтому",
        "так как", "следовательно", "что привело", "это привело",
        "причина", "вызвало", "позволило",
    ]
    count = sum(1 for m in _CAUSAL if m in text_lower)
    return round(float(np.clip(count / (n / 30.0), 0.0, 1.0)), 3)


def compute_character_continuity_score(
    asr_segments: List[Dict],
    start: float,
    end: float,
) -> float:
    """
    Estimate how consistently the same character/protagonist is referenced.
    High continuity = one ongoing story about the same subject.
    Returns [0..1].
    """
    window_segs = [s for s in asr_segments if s.get("start", 0) < end and s.get("end", 0) > start]
    if not window_segs:
        return 0.0

    _CHAR_FIRST = {"я", "мы"}
    _CHAR_THIRD = {"он", "она", "они", "клиент", "заказчик", "человек"}

    first_count = third_count = 0
    for seg in window_segs:
        words_set = set((seg.get("text") or "").lower().split())
        first_count += len(words_set & _CHAR_FIRST)
        third_count += len(words_set & _CHAR_THIRD)

    total_chars = first_count + third_count
    if total_chars == 0:
        return 0.20

    consistency = float(max(first_count, third_count)) / total_chars
    duration    = max(end - start, 1.0)
    density     = float(np.clip(total_chars / max(duration / 10.0, 0.1) / 3.0, 0.0, 1.0))

    return round(float(np.clip(0.60 * consistency + 0.40 * density, 0.0, 1.0)), 3)


# =============================================================================
# STAGE A — EVENTIZATION:  detect_story_events → merge_story_events_to_beats
# =============================================================================

def detect_story_events(
    asr_segments: Optional[List[Dict]],
    base_analysis: Optional[Dict],
    video_duration_sec: float,
    config: Optional["StoryModeConfig"] = None,
) -> List[Dict]:
    """
    Stage A, step 1: Build candidate story events with full event contract.

    Full event contract (v2)
    ------------------------
    event_id          — unique int id (0-indexed, assigned at end)
    t                 — timestamp (seconds)
    source            — source label
    strength          — signal strength [0..1]
    event_type        — semantic event type
    source_modalities — list of modalities contributing to this event
    event_role_hints  — list of narrative roles this event hints at
    supports_conflict — bool: event is evidence of conflict/tension
    supports_payoff   — bool: event is evidence of resolution/payoff
    causal_hint       — bool: event carries causal link signal
    object_change     — bool: event involves visual object state change
    topic_shift       — bool: event marks a discourse topic change
    main_character_id — optional speaker/character id if available

    Sources
    -------
    asr_turn          — ASR segment gap >= 0.3s  (topic/turn shift)
    setup_marker      — setup-role lexical marker in ASR segment
    conflict_marker   — conflict-role marker
    resolution_marker — resolution-role marker
    trigger_marker    — inciting-event marker (vdrug, neozhidanno, etc.)
    discourse         — causal / temporal discourse marker
    audio_burst       — emotion_intensity or arousal onset in time_series
    visual_burst      — visual_intensity or face_presence onset
    """
    cfg    = config or StoryModeConfig()
    merged = cfg.merge_config()
    dur    = max(video_duration_sec, 1e-6)
    events: List[Dict] = []

    # Role-hint maps: which event_types hint at which narrative roles
    _ROLE_HINTS: Dict[str, List[str]] = {
        "turn_boundary":        ["inciting_event", "tension"],
        "setup_signal":         ["setup"],
        "conflict_signal":      ["inciting_event", "tension"],
        "resolution_signal":    ["payoff", "reflection"],
        "trigger_signal":       ["inciting_event"],
        "causal_link":          ["attempt", "tension"],
        "temporal_progression": ["setup", "attempt"],
        "emotion_onset":        ["inciting_event", "tension", "payoff"],
        "visual_onset":         ["inciting_event", "payoff"],
        "face_onset":           ["inciting_event", "payoff"],
        "prosody_burst":        ["tension", "payoff"],
    }
    # Which event_types support conflict / payoff / causal_hint
    _CONFLICT_TYPES = {"conflict_signal", "trigger_signal", "turn_boundary",
                       "emotion_onset", "prosody_burst"}
    _PAYOFF_TYPES   = {"resolution_signal", "visual_onset", "face_onset",
                       "emotion_onset", "prosody_burst"}
    _CAUSAL_TYPES   = {"causal_link", "conflict_signal"}
    _OBJECT_TYPES   = {"visual_onset", "face_onset"}
    _TOPIC_TYPES    = {"turn_boundary", "temporal_progression", "setup_signal"}

    def _make_event(t: float, source: str, strength: float, ev_type: str,
                    modalities: Optional[List[str]] = None,
                    speaker_id: Optional[str] = None) -> Dict:
        return {
            "t":                  round(t, 3),
            "source":             source,
            "strength":           round(float(np.clip(strength, 0.0, 1.0)), 3),
            "event_type":         ev_type,
            "source_modalities":  modalities or [source],
            "event_role_hints":   _ROLE_HINTS.get(ev_type, []),
            "supports_conflict":  ev_type in _CONFLICT_TYPES,
            "supports_payoff":    ev_type in _PAYOFF_TYPES,
            "causal_hint":        ev_type in _CAUSAL_TYPES,
            "object_change":      ev_type in _OBJECT_TYPES,
            "topic_shift":        ev_type in _TOPIC_TYPES,
            "main_character_id":  speaker_id,
        }

    # ── ASR turn boundaries ─────────────────────────────────────────────────
    if asr_segments:
        sorted_segs = sorted(asr_segments, key=lambda s: s.get("start", 0))
        for i in range(len(sorted_segs) - 1):
            t_end  = float(sorted_segs[i].get("end",   0))
            t_next = float(sorted_segs[i + 1].get("start", 0))
            gap    = t_next - t_end
            if gap >= 0.30:
                spk = sorted_segs[i + 1].get("speaker") or sorted_segs[i + 1].get("speaker_id")
                events.append(_make_event(
                    t_next, "asr_turn",
                    float(np.clip(gap / 2.0, 0.0, 1.0)),
                    "turn_boundary",
                    modalities=["text"],
                    speaker_id=str(spk) if spk else None,
                ))

    # ── Discourse markers in ASR segments ──────────────────────────────────
    _DISCOURSE_MAP = {
        "setup_markers":      ("setup_marker",      "setup_signal"),
        "conflict_markers":   ("conflict_marker",   "conflict_signal"),
        "resolution_markers": ("resolution_marker", "resolution_signal"),
    }
    _CAUSAL_MARKERS   = ["поэтому", "потому что", "из-за", "благодаря",
                          "в результате", "так что"]
    _TEMPORAL_MARKERS = ["потом", "затем", "после", "когда", "тогда",
                          "сначала", "позже", "вдруг"]
    _TRIGGER_MARKERS  = ["вдруг", "неожиданно", "тут", "в тот момент",
                          "всё изменилось", "оказалось", "и тут", "тогда я"]

    if asr_segments:
        for seg in asr_segments:
            seg_text  = (seg.get("text") or "").lower()
            seg_start = float(seg.get("start", 0))
            spk       = seg.get("speaker") or seg.get("speaker_id")
            if not seg_text:
                continue

            for marker_key, (src_name, ev_type) in _DISCOURSE_MAP.items():
                markers = merged.get(marker_key, [])
                hits    = sum(1 for m in markers if m in seg_text)
                if hits > 0:
                    events.append(_make_event(
                        seg_start, src_name,
                        float(np.clip(hits / 3.0, 0.2, 1.0)),
                        ev_type,
                        modalities=["text"],
                        speaker_id=str(spk) if spk else None,
                    ))

            # Trigger markers (inciting event signals)
            trigger_hits = sum(1 for m in _TRIGGER_MARKERS if m in seg_text)
            if trigger_hits > 0:
                events.append(_make_event(
                    seg_start, "trigger_marker",
                    float(np.clip(trigger_hits / 3.0, 0.2, 1.0)),
                    "trigger_signal",
                    modalities=["text"],
                    speaker_id=str(spk) if spk else None,
                ))

            causal_hits = sum(1 for m in _CAUSAL_MARKERS if m in seg_text)
            if causal_hits > 0:
                events.append(_make_event(
                    seg_start, "discourse",
                    float(np.clip(causal_hits / 3.0, 0.2, 1.0)),
                    "causal_link",
                    modalities=["text"],
                    speaker_id=str(spk) if spk else None,
                ))

            temporal_hits = sum(1 for m in _TEMPORAL_MARKERS if m in seg_text)
            if temporal_hits > 0:
                events.append(_make_event(
                    seg_start, "discourse",
                    float(np.clip(temporal_hits / 3.0, 0.1, 0.8)),
                    "temporal_progression",
                    modalities=["text"],
                    speaker_id=str(spk) if spk else None,
                ))

    # ── Visual / audio events from time_series ─────────────────────────────
    if base_analysis and video_duration_sec > 0:
        ts = base_analysis.get("time_series") or {}

        def _onsets_for(arr_raw: Any, rise_thresh: float,
                        src: str, ev_type: str,
                        modality: str = "visual") -> List[Dict]:
            if arr_raw is None:
                return []
            arr = np.asarray(arr_raw, dtype=float)
            n   = len(arr)
            if n < 6:
                return []
            win    = 3
            out    = []
            last_t = -5.0
            for i in range(win, n - win):
                t      = i / n * dur
                before = float(np.mean(arr[i - win: i]))
                after  = float(np.mean(arr[i:     i + win]))
                delta  = after - before
                if delta >= rise_thresh and (t - last_t) >= 1.5:
                    strength = float(np.clip(float(arr[i]) + delta * 0.5, 0.0, 1.0))
                    out.append(_make_event(
                        t, src, strength, ev_type,
                        modalities=[modality],
                    ))
                    last_t = t
            return out

        events.extend(_onsets_for(ts.get("emotion_intensity"), 0.22,
                                   "audio_burst",  "emotion_onset", "audio"))
        events.extend(_onsets_for(ts.get("visual_intensity"),  0.25,
                                   "visual_burst", "visual_onset",  "visual"))
        events.extend(_onsets_for(ts.get("face_presence"),     0.30,
                                   "visual_burst", "face_onset",    "visual"))
        events.extend(_onsets_for(ts.get("arousal"),           0.20,
                                   "audio_burst",  "prosody_burst", "audio"))

    events.sort(key=lambda e: e["t"])

    # Assign stable event_ids after sorting
    for idx, ev in enumerate(events):
        ev["event_id"] = idx

    return events


def merge_story_events_to_beats(
    events: List[Dict],
    asr_segments: Optional[List[Dict]],
    cluster_window_sec: float = 3.0,
    min_beat_strength: float = 0.20,
) -> List[Dict]:
    """
    Stage A, step 2: Cluster nearby events into story beats.

    Events within `cluster_window_sec` of each other are merged into one beat.

    Each beat:
        start / end          — temporal span
        events               — constituent events
        dominant_source      — most frequent source
        dominant_event_type  — most frequent event_type
        mean_strength        — average strength
        n_sources            — distinct source count
        transcript_snippet   — ASR text covering the beat
    """
    if not events:
        return []

    beats: List[Dict] = []
    cluster: List[Dict] = [events[0]]

    for ev in events[1:]:
        if ev["t"] - cluster[-1]["t"] <= cluster_window_sec:
            cluster.append(ev)
        else:
            beats.append(_make_story_beat(cluster, asr_segments))
            cluster = [ev]
    if cluster:
        beats.append(_make_story_beat(cluster, asr_segments))

    return [b for b in beats if b["mean_strength"] >= min_beat_strength]


def _make_story_beat(
    cluster: List[Dict],
    asr_segments: Optional[List[Dict]],
) -> Dict:
    """Build one beat dict from a cluster of events."""
    times       = [e["t"] for e in cluster]
    t_start     = min(times)
    t_end       = max(times)
    sources     = [e["source"]     for e in cluster]
    event_types = [e["event_type"] for e in cluster]
    strengths   = [e["strength"]   for e in cluster]

    def _most_common(lst: List) -> Any:
        return max(set(lst), key=lst.count) if lst else "unknown"

    snippet = ""
    if asr_segments:
        beat_segs = [
            s for s in asr_segments
            if s.get("start", 0) <= t_end + 1.0 and s.get("end", 0) >= t_start - 0.5
        ]
        snippet = " ".join(s.get("text", "") for s in beat_segs[:6]).strip()

    return {
        "start":               round(t_start, 3),
        "end":                 round(max(t_end, t_start + 0.5), 3),
        "events":              cluster,
        "dominant_source":     _most_common(sources),
        "dominant_event_type": _most_common(event_types),
        "mean_strength":       round(float(np.mean(strengths)), 3),
        "n_sources":           len(set(sources)),
        "transcript_snippet":  snippet[:200],
    }


# =============================================================================
# STAGE B — NARRATIVE ROLE LABELING:  label_narrative_roles
# =============================================================================

_NARRATIVE_ROLES: List[str] = [
    "setup",           # introduces characters, context, status quo
    "inciting_event",  # disrupts status quo
    "tension",         # conflict intensifies, obstacles accumulate
    "attempt",         # protagonist acts on the conflict
    "payoff",          # conflict resolves (positively or negatively)
    "reflection",      # meaning-making, lesson, takeaway
]


def label_narrative_roles(
    beats: List[Dict],
    asr_segments: Optional[List[Dict]],
    config: Optional["StoryModeConfig"] = None,
    sentiment_curve: Optional[List[float]] = None,
) -> List[Dict]:
    """
    Stage B: Assign narrative role probabilities to each beat.

    Evidence sources:
        POSITION  — position in beat sequence (early → setup; late → payoff)
        DISCOURSE — event_type matches (conflict_signal → tension; etc.)
        SENTIMENT — local sentiment value at beat time
        VISUAL    — visual burst strength

    Each beat is enriched with:
        role_probs      — {role: prob} for all 6 roles (sum to 1)
        dominant_role   — highest-prob role
        role_confidence — gap between top-1 and top-2 role probs
    """
    if not beats:
        return []

    n = len(beats)
    for idx, beat in enumerate(beats):
        pos = float(idx) / max(n - 1, 1)  # 0.0 = first beat, 1.0 = last beat

        # ── Position priors (typical position in a story sequence) ─────────
        pos_priors = {
            "setup":          float(np.clip(1.0 - pos * 2.0,          0.0, 1.0)),
            "inciting_event": float(np.clip(1.0 - abs(pos - 0.20) / 0.20, 0.0, 1.0)),
            "tension":        float(np.clip(1.0 - abs(pos - 0.50) / 0.35, 0.0, 1.0)),
            "attempt":        float(np.clip(1.0 - abs(pos - 0.65) / 0.25, 0.0, 1.0)),
            "payoff":         float(np.clip(1.0 - abs(pos - 0.85) / 0.20, 0.0, 1.0)),
            "reflection":     float(np.clip((pos - 0.80)  / 0.20,          0.0, 1.0)),
        }

        # ── Discourse evidence (event_type counts) ─────────────────────────
        ev_types = [e["event_type"] for e in beat["events"]]
        disc = {
            "setup":          float(ev_types.count("setup_signal"))         * 0.40,
            "inciting_event": float(ev_types.count("conflict_signal"))      * 0.35,
            "tension":        float(ev_types.count("conflict_signal"))      * 0.30
                              + float(ev_types.count("turn_boundary"))      * 0.10,
            "attempt":        float(ev_types.count("causal_link"))          * 0.30
                              + float(ev_types.count("temporal_progression"))* 0.20,
            "payoff":         float(ev_types.count("resolution_signal"))    * 0.45,
            "reflection":     float(ev_types.count("resolution_signal"))    * 0.30,
        }
        disc = {k: float(np.clip(v, 0.0, 1.0)) for k, v in disc.items()}

        # ── Sentiment evidence ─────────────────────────────────────────────
        sent_score = _beat_sentiment(beat, asr_segments, config)
        sent_ev = {
            "setup":          0.50,
            "inciting_event": 0.50 - sent_score * 0.30,
            "tension":        0.50 - sent_score * 0.25,
            "attempt":        sent_score * 0.30 + 0.35,
            "payoff":         sent_score * 0.50 + 0.30,
            "reflection":     0.55,
        }
        sent_ev = {k: float(np.clip(v, 0.0, 1.0)) for k, v in sent_ev.items()}

        # ── Visual burst evidence ──────────────────────────────────────────
        visual_strength = beat["mean_strength"] if "visual" in beat["dominant_source"] else 0.0
        vis_ev = {
            "setup":          0.00,
            "inciting_event": visual_strength * 0.40,
            "tension":        visual_strength * 0.30,
            "attempt":        visual_strength * 0.30,
            "payoff":         visual_strength * 0.45,
            "reflection":     0.00,
        }

        # ── Composite role probabilities ───────────────────────────────────
        role_probs: Dict[str, float] = {}
        for role in _NARRATIVE_ROLES:
            score = float(np.clip(
                0.40 * pos_priors[role] +
                0.35 * disc[role] +
                0.15 * sent_ev[role] +
                0.10 * vis_ev[role],
                0.0, 1.0,
            ))
            role_probs[role] = score

        total = sum(role_probs.values()) or 1.0
        role_probs = {k: round(v / total, 3) for k, v in role_probs.items()}

        sorted_roles    = sorted(role_probs.items(), key=lambda kv: kv[1], reverse=True)
        dominant_role   = sorted_roles[0][0]
        role_confidence = round(float(np.clip(
            sorted_roles[0][1] - (sorted_roles[1][1] if len(sorted_roles) > 1 else 0.0),
            0.0, 1.0,
        )), 3)

        beat["role_probs"]      = role_probs
        beat["dominant_role"]   = dominant_role
        beat["role_confidence"] = role_confidence

    return beats


def _beat_sentiment(
    beat: Dict,
    asr_segments: Optional[List[Dict]],
    config: Optional["StoryModeConfig"],
) -> float:
    """Estimate local sentiment for a beat window. Returns [0..1]."""
    if not asr_segments:
        return 0.5
    cfg = config or StoryModeConfig()
    window_segs = [
        s for s in asr_segments
        if s.get("start", 0) <= beat["end"] + 0.5 and s.get("end", 0) >= beat["start"] - 0.5
    ]
    if not window_segs:
        return 0.5
    merged   = cfg.merge_config()
    pos_list = merged.get("sentiment_positive", SENTIMENT_MARKERS["sentiment_positive"])
    neg_list = merged.get("sentiment_negative", SENTIMENT_MARKERS["sentiment_negative"])
    pos_count = neg_count = 0
    for seg in window_segs:
        text = (seg.get("text") or "").lower()
        pos_count += sum(1 for w in pos_list if w in text)
        neg_count += sum(1 for w in neg_list if w in text)
    if pos_count + neg_count == 0:
        return 0.5
    return float(np.clip(0.5 + 0.5 * (pos_count - neg_count) / (pos_count + neg_count + 1), 0.0, 1.0))


# =============================================================================
# STAGE C — ARC DETECTION (improved): tension segments + change points
# =============================================================================

def detect_tension_curve_segments(
    sentiment_curve: List[float],
    min_segment_len: int = 2,
) -> Dict:
    """
    Replace the crude thirds-based arc with local structure analysis.

    Finds:
        local_maxima   — indices of sentiment peaks
        local_minima   — indices of sentiment troughs
        change_points  — indices of rapid direction reversals
        rise_spans     — (start_idx, end_idx) spans of rising sentiment
        fall_spans     — (start_idx, end_idx) spans of falling sentiment
        payoff_candidate — index of the best payoff peak in the last 40%
        smoothed       — smoothed curve (list)
    """
    if not sentiment_curve or len(sentiment_curve) < 4:
        return {
            "local_maxima": [], "local_minima": [], "change_points": [],
            "rise_spans": [], "fall_spans": [],
            "payoff_candidate": None, "smoothed": list(sentiment_curve or []),
        }

    arr = np.array(sentiment_curve, dtype=float)
    smoothed = np.convolve(arr, np.ones(3) / 3.0, mode="same") if len(arr) >= 5 else arr.copy()
    n      = len(smoothed)
    median = float(np.median(smoothed))

    local_maxima = [i for i in range(1, n - 1)
                    if smoothed[i] > smoothed[i - 1] and smoothed[i] > smoothed[i + 1]]
    local_minima = [i for i in range(1, n - 1)
                    if smoothed[i] < smoothed[i - 1] and smoothed[i] < smoothed[i + 1]]

    deriv = np.diff(smoothed)
    change_points = [
        i for i in range(1, len(deriv))
        if abs(deriv[i]) > 0.15 and np.sign(deriv[i]) != np.sign(deriv[i - 1])
    ]

    rise_spans: List[Tuple[int, int]] = []
    fall_spans: List[Tuple[int, int]] = []
    in_rise = in_fall = False
    span_start = 0
    for i in range(1, n):
        delta = smoothed[i] - smoothed[i - 1]
        if delta > 0.05 and not in_rise:
            if in_fall and i - span_start >= min_segment_len:
                fall_spans.append((span_start, i - 1))
            in_fall, in_rise, span_start = False, True, i - 1
        elif delta < -0.05 and not in_fall:
            if in_rise and i - span_start >= min_segment_len:
                rise_spans.append((span_start, i - 1))
            in_rise, in_fall, span_start = False, True, i - 1
    if in_rise and n - span_start >= min_segment_len:
        rise_spans.append((span_start, n - 1))
    if in_fall and n - span_start >= min_segment_len:
        fall_spans.append((span_start, n - 1))

    last_40pct = int(n * 0.60)
    late_maxima = [i for i in local_maxima if i >= last_40pct and smoothed[i] > median]
    payoff_candidate = max(late_maxima, key=lambda i: smoothed[i]) if late_maxima else None

    return {
        "local_maxima":    local_maxima,
        "local_minima":    local_minima,
        "change_points":   change_points,
        "rise_spans":      rise_spans,
        "fall_spans":      fall_spans,
        "payoff_candidate": payoff_candidate,
        "smoothed":        smoothed.tolist(),
    }


def detect_arc_change_points(
    sentiment_curve: List[float],
) -> Tuple[str, float]:
    """
    Improved arc pattern detection using local extrema and change points.
    Returns (arc_pattern, confidence).
    Falls back to detect_story_arc_pattern() if not enough structure.
    """
    if not sentiment_curve or len(sentiment_curve) < 4:
        return "flat", 0.0

    segs      = detect_tension_curve_segments(sentiment_curve)
    smoothed  = np.array(segs["smoothed"])
    local_max = segs["local_maxima"]
    local_min = segs["local_minima"]
    n         = len(smoothed)

    if not local_max and not local_min:
        return detect_story_arc_pattern(sentiment_curve)

    first_quarter = float(np.mean(smoothed[:max(1, n // 4)]))
    last_quarter  = float(np.mean(smoothed[max(0, 3 * n // 4):]))
    overall       = last_quarter - first_quarter

    has_early_max = any(i < n // 2 for i in local_max)
    has_late_max  = any(i >= n // 2 for i in local_max)
    has_early_min = any(i < n // 2 for i in local_min)
    has_late_min  = any(i >= n // 2 for i in local_min)
    n_extrema     = len(local_max) + len(local_min)

    if has_early_min and has_late_max and overall > 0.1:
        depth = float(np.min(smoothed[local_min[0]: local_min[0] + 2])) if local_min else 0.5
        peak  = float(smoothed[local_max[-1]])                            if local_max else 0.5
        return "man_in_hole", round(float(np.clip(peak - depth, 0.0, 1.0)), 3)

    if has_early_max and has_late_min and overall < -0.1:
        peak  = float(smoothed[local_max[0]])                              if local_max else 0.5
        depth = float(np.min(smoothed[local_min[-1]: local_min[-1] + 2])) if local_min else 0.5
        return "icarus", round(float(np.clip(peak - depth, 0.0, 1.0)), 3)

    if overall > 0.20 and not has_early_max:
        return "rags_to_riches", round(float(np.clip(overall, 0.0, 1.0)), 3)

    if overall < -0.20 and not has_early_min:
        return "riches_to_rags", round(float(np.clip(-overall, 0.0, 1.0)), 3)

    if n_extrema >= 3:
        if first_quarter > 0.5 and last_quarter > 0.5:
            return "cinderella", 0.45
        if first_quarter < 0.5 and last_quarter < 0.5:
            return "oedipus",    0.40

    return "flat", round(float(np.clip(1.0 - abs(overall) * 2, 0.0, 0.4)), 3)


# =============================================================================
# STAGE C — ARC ASSEMBLY:  build_story_arcs
# =============================================================================

_VALID_ARC_SEQUENCES: List[List[str]] = [
    # Full arcs
    ["setup", "inciting_event", "tension", "payoff"],
    ["setup", "inciting_event", "tension", "attempt", "payoff"],
    ["setup", "inciting_event", "tension", "attempt", "payoff", "reflection"],
    # Compressed arcs
    ["setup", "tension", "payoff"],
    ["inciting_event", "tension", "payoff"],
    ["setup", "inciting_event", "payoff"],
    # Minimal arcs
    ["inciting_event", "payoff"],
    ["tension", "payoff"],
    ["tension", "payoff", "reflection"],
]


def build_story_arcs(
    role_labeled_beats: List[Dict],
    min_arc_completeness: float = 0.40,
    max_arc_duration: float = 150.0,
) -> List[Dict]:
    """
    Stage C: Assemble role-labeled beats into story arcs with full diagnostics.

    Each arc now carries extended diagnostics:
        arc_id, beats, start, end, duration, n_beats
        role_sequence       — list of dominant_role values
        role_probs_by_beat  — [{role: prob} per beat] for explainability
        arc_completeness    — [0..1] match vs canonical sequences
        payoff_strength     — mean strength of payoff beats
        coherence_score     — internal consistency
        causal_density      — causal link events within arc / total events
        payoff_dependency_score  — how much non-payoff beats set up the payoff
        setup_necessity_score    — how necessary the setup beats are (early role coverage)
        mid_arc_redundancy       — fraction of beats with low role-confidence (ambiguous)
        arc_failure_modes   — list of named quality failure labels
    """
    if not role_labeled_beats:
        return []

    n    = len(role_labeled_beats)
    arcs: List[Dict] = []

    for start_idx in range(n):
        for end_idx in range(start_idx + 1, n + 1):
            sub      = role_labeled_beats[start_idx:end_idx]
            duration = sub[-1]["end"] - sub[0]["start"]
            if duration > max_arc_duration:
                break

            roles        = [b["dominant_role"] for b in sub]
            completeness = _arc_completeness(roles)
            if completeness < min_arc_completeness:
                continue

            payoff_beats    = [b for b in sub if b["dominant_role"] == "payoff"]
            payoff_strength = round(
                float(np.mean([b["mean_strength"] for b in payoff_beats]))
                if payoff_beats else 0.0, 3
            )
            coherence = _arc_coherence(sub)

            # ── Extended diagnostics ───────────────────────────────────────
            diag = _arc_diagnostics(sub, roles)

            arcs.append({
                "arc_id":           len(arcs),
                "beats":            sub,
                "start":            round(sub[0]["start"],  3),
                "end":              round(sub[-1]["end"],   3),
                "duration":         round(duration,         3),
                "role_sequence":    roles,
                "role_probs_by_beat": [b.get("role_probs", {}) for b in sub],
                "arc_completeness": round(completeness,     3),
                "payoff_strength":  payoff_strength,
                "coherence_score":  round(coherence,        3),
                "n_beats":          len(sub),
                # ── New diagnostics ──────────────────────────────────────
                "causal_density":           diag["causal_density"],
                "payoff_dependency_score":  diag["payoff_dependency_score"],
                "setup_necessity_score":    diag["setup_necessity_score"],
                "mid_arc_redundancy":       diag["mid_arc_redundancy"],
                "arc_failure_modes":        diag["arc_failure_modes"],
            })

    if not arcs:
        return []

    arcs.sort(
        key=lambda a: a["arc_completeness"] * max(a["payoff_strength"], 0.01) * a["coherence_score"],
        reverse=True,
    )

    kept: List[Dict] = []
    for arc in arcs:
        overlaps = any(
            not (arc["end"] <= k["start"] or arc["start"] >= k["end"])
            for k in kept
        )
        if not overlaps:
            kept.append(arc)

    for i, arc in enumerate(kept):
        arc["arc_id"] = i

    return kept


def _arc_diagnostics(beats: List[Dict], roles: List[str]) -> Dict:
    """
    Compute extended arc diagnostics.

    causal_density         — fraction of events that are causal_link type
    payoff_dependency_score — how much preceding beats have conflict/tension
                              evidence that the payoff could resolve
    setup_necessity_score  — whether early beats carry unique setup evidence
                             (i.e. removing them would hurt arc completeness)
    mid_arc_redundancy     — fraction of mid-arc beats with role_confidence < 0.15
    arc_failure_modes      — named failure labels
    """
    # ── Causal density ─────────────────────────────────────────────────────
    all_events   = [e for b in beats for e in b.get("events", [])]
    n_events     = max(len(all_events), 1)
    n_causal     = sum(1 for e in all_events if e.get("event_type") == "causal_link")
    causal_density = round(float(n_causal) / n_events, 3)

    # ── Payoff dependency ─────────────────────────────────────────────────
    # Payoff is more "earned" if earlier beats carry conflict/tension evidence
    pre_payoff_roles = [r for r in roles[:-1] if r in ("tension", "inciting_event", "attempt")]
    payoff_dependency = round(
        float(np.clip(len(pre_payoff_roles) / max(len(roles) - 1, 1), 0.0, 1.0)), 3
    )

    # ── Setup necessity ───────────────────────────────────────────────────
    # If removing first beat would reduce arc_completeness, setup is necessary
    if len(roles) > 1:
        completeness_without_first = _arc_completeness(roles[1:])
        completeness_full          = _arc_completeness(roles)
        setup_necessity = round(
            float(np.clip(completeness_full - completeness_without_first + 0.3, 0.0, 1.0)), 3
        )
    else:
        setup_necessity = 0.5

    # ── Mid-arc redundancy ────────────────────────────────────────────────
    # Mid-arc = beats [1:-1] (skip first and last)
    mid_beats   = beats[1:-1] if len(beats) > 2 else []
    n_ambiguous = sum(
        1 for b in mid_beats if b.get("role_confidence", 1.0) < 0.15
    )
    mid_arc_redundancy = round(
        float(n_ambiguous) / max(len(mid_beats), 1), 3
    ) if mid_beats else 0.0

    # ── Arc failure modes ─────────────────────────────────────────────────
    failures: List[str] = []
    payoff_beats = [b for b in beats if b.get("dominant_role") == "payoff"]
    if not payoff_beats:
        failures.append("no_payoff_beat")
    if "inciting_event" not in roles and "tension" not in roles:
        failures.append("no_conflict_evidence")
    if causal_density < 0.05 and len(all_events) >= 5:
        failures.append("low_causal_density")
    if mid_arc_redundancy > 0.60:
        failures.append("high_mid_arc_redundancy")
    if payoff_dependency < 0.20 and payoff_beats:
        failures.append("unearned_payoff")

    return {
        "causal_density":          causal_density,
        "payoff_dependency_score": payoff_dependency,
        "setup_necessity_score":   setup_necessity,
        "mid_arc_redundancy":      mid_arc_redundancy,
        "arc_failure_modes":       failures,
    }


def _arc_completeness(roles: List[str]) -> float:
    """[0..1]: how well a role sequence matches a canonical arc pattern."""
    if not roles:
        return 0.0
    role_set = set(roles)
    best = 0.0
    for canon in _VALID_ARC_SEQUENCES:
        coverage     = len(role_set & set(canon)) / len(canon)
        order_score  = _sequence_order_score(roles, canon)
        best         = max(best, 0.60 * coverage + 0.40 * order_score)
    return float(np.clip(best, 0.0, 1.0))


def _sequence_order_score(actual: List[str], canon: List[str]) -> float:
    """LCS-based ordering score: how much `actual` respects `canon` order."""
    m, n = len(actual), len(canon)
    if m == 0 or n == 0:
        return 0.0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if actual[i - 1] == canon[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return float(dp[m][n]) / n


def _arc_coherence(beats: List[Dict]) -> float:
    """Internal coherence: consistent strength + role confidence + no huge gaps."""
    if len(beats) < 2:
        return 0.5
    strengths   = [b["mean_strength"]   for b in beats]
    confidences = [b.get("role_confidence", 0.3) for b in beats]
    gaps        = [beats[i + 1]["start"] - beats[i]["end"] for i in range(len(beats) - 1)]
    gap_penalty = float(np.clip(max(gaps) / 30.0, 0.0, 0.5)) if gaps else 0.0
    return float(np.clip(
        0.40 * float(np.mean(strengths)) +
        0.40 * float(np.mean(confidences)) -
        0.20 * gap_penalty,
        0.0, 1.0,
    ))


# =============================================================================
# SOFT STORY TYPE CLASSIFIER  (replaces rule-based detect_story_type cascade)
# =============================================================================

def classify_story_type_soft(
    transcript: str,
    arc_pattern: str = "flat",
    has_takeaway: bool = False,
    sentiment_curve: Optional[List[float]] = None,
    arc_roles: Optional[List[str]] = None,
    semantic_features: Optional[Dict] = None,
) -> Tuple[str, float, str, float, Dict[str, float], float]:
    """
    Soft competitive story type classifier — all types compete simultaneously.

    Returns (story_type, confidence, secondary_type, type_ambiguity,
             type_scores, type_profile_match).

    type_scores        — {type: score} for all types (for explainability)
    type_profile_match — [0..1] how well the winning type matches a canonical
                         profile (role sequence + arc + evidence).
                         Low value = "technically best type but weak match".
    """
    if not transcript or len(transcript.strip()) < 30:
        return "non_story", 0.0, "non_story", 1.0, {"non_story": 0.0}, 0.0

    text_lower = transcript.lower()
    sem        = semantic_features or {}
    roles      = set(arc_roles or [])

    def _marker_score(markers: List[str]) -> float:
        hits = sum(1 for m in markers if m in text_lower)
        return float(np.clip(hits / max(len(markers) * 0.3, 1.0), 0.0, 1.0))

    _TYPE_MARKERS = {
        "personal_story": ["я", "мы", "расскажу", "было дело", "история",
                           "случай", "однажды"],
        "client_case":    ["клиент", "заказчик", "у нас был", "пришёл",
                           "обратился", "продукт", "метрика", "кампания",
                           "бюджет", "проект"],
        "failure_story":  ["ошибка", "провал", "не получилось", "облажались",
                           "факап", "проблема", "потеряли", "неудача"],
        "success_story":  ["получилось", "удалось", "успех", "результат",
                           "достигли", "выросла", "улучшили"],
        "anecdote":       ["смешно", "прикол", "забавно", "анекдот",
                           "история про", "однажды", "случай был"],
        "lesson_learned": ["вывод", "научились", "урок", "опыт показал",
                           "поняли что", "главное", "важно понять"],
    }
    _FAILURE_ARCS = {"riches_to_rags", "oedipus", "icarus", "man_in_hole"}
    _SUCCESS_ARCS = {"rags_to_riches", "cinderella"}

    type_scores: Dict[str, float] = {}

    type_scores["personal_story"] = float(np.clip(
        0.50 * _marker_score(_TYPE_MARKERS["personal_story"]) +
        0.30 * (1.0 if "я" in text_lower.split() else 0.0) +
        0.20 * float(sem.get("coherence", 0.5)),
        0.0, 1.0,
    ))

    type_scores["client_case"] = float(np.clip(
        0.60 * _marker_score(_TYPE_MARKERS["client_case"]) +
        0.25 * float(sem.get("informativeness", 0.5)) +
        0.15 * (1.0 if arc_pattern in ("man_in_hole", "rags_to_riches", "cinderella") else 0.0),
        0.0, 1.0,
    ))

    type_scores["failure_story"] = float(np.clip(
        0.50 * _marker_score(_TYPE_MARKERS["failure_story"]) +
        0.30 * (1.0 if arc_pattern in _FAILURE_ARCS else 0.0) +
        0.20 * (1.0 if "tension" in roles or "inciting_event" in roles else 0.0),
        0.0, 1.0,
    ))

    tail_positive = 0.0
    if sentiment_curve and len(sentiment_curve) >= 3:
        tail         = sentiment_curve[-max(1, len(sentiment_curve) // 3):]
        tail_positive = float(np.clip(float(np.mean(tail)) - 0.5, 0.0, 0.5) * 2)
    type_scores["success_story"] = float(np.clip(
        0.40 * _marker_score(_TYPE_MARKERS["success_story"]) +
        0.30 * (1.0 if arc_pattern in _SUCCESS_ARCS else 0.0) +
        0.30 * tail_positive,
        0.0, 1.0,
    ))

    type_scores["anecdote"] = float(np.clip(
        0.70 * _marker_score(_TYPE_MARKERS["anecdote"]) +
        0.30 * float(sem.get("engagement", 0.5)),
        0.0, 1.0,
    ))

    type_scores["lesson_learned"] = float(np.clip(
        0.50 * (1.0 if has_takeaway else 0.0) +
        0.35 * _marker_score(_TYPE_MARKERS["lesson_learned"]) +
        0.15 * (1.0 if "reflection" in roles or "payoff" in roles else 0.0),
        0.0, 1.0,
    ))

    max_positive = max(type_scores.values()) if type_scores else 0.0
    type_scores["non_story"] = float(np.clip(1.0 - max_positive, 0.0, 0.6)) * 0.5

    sorted_types   = sorted(type_scores.items(), key=lambda kv: kv[1], reverse=True)
    best_type      = sorted_types[0][0]
    best_score     = sorted_types[0][1]
    runner_up      = sorted_types[1] if len(sorted_types) > 1 else ("non_story", 0.0)
    gap            = best_score - runner_up[1]

    confidence     = round(float(np.clip(0.40 * (gap / 0.30) + 0.60 * best_score, 0.0, 1.0)), 3)
    type_ambiguity = round(float(np.clip(1.0 - gap / max(best_score, 1e-6), 0.0, 1.0)), 3)

    # ── type_profile_match: canonical profile match for winning type ──────
    # Each type has a "canonical profile" — which arc roles and arc patterns
    # are ideal for it.  type_profile_match measures how well the candidate
    # satisfies that canonical profile beyond just lexical marker hits.
    _TYPE_CANONICAL_ROLES: Dict[str, List[str]] = {
        "personal_story": ["setup", "inciting_event", "tension", "payoff"],
        "client_case":    ["setup", "tension", "attempt", "payoff"],
        "failure_story":  ["setup", "inciting_event", "tension", "payoff"],
        "success_story":  ["setup", "inciting_event", "attempt", "payoff"],
        "anecdote":       ["inciting_event", "payoff"],
        "lesson_learned": ["tension", "payoff", "reflection"],
        "non_story":      [],
    }
    canonical_roles = _TYPE_CANONICAL_ROLES.get(best_type, [])
    if canonical_roles and arc_roles:
        role_set     = set(arc_roles)
        canon_set    = set(canonical_roles)
        role_overlap = len(role_set & canon_set) / max(len(canon_set), 1)
    else:
        role_overlap = 0.0

    canonical_arc_ok = float({
        "personal_story": arc_pattern not in ("flat",),
        "client_case":    arc_pattern in ("man_in_hole", "rags_to_riches", "cinderella"),
        "failure_story":  arc_pattern in _FAILURE_ARCS,
        "success_story":  arc_pattern in _SUCCESS_ARCS,
        "anecdote":       True,
        "lesson_learned": has_takeaway,
        "non_story":      False,
    }.get(best_type, False))

    type_profile_match = round(float(np.clip(
        0.50 * role_overlap +
        0.30 * canonical_arc_ok +
        0.20 * best_score,
        0.0, 1.0,
    )), 3)

    # Round all type_scores for clean output
    type_scores_out = {k: round(v, 3) for k, v in type_scores.items()}

    return best_type, confidence, runner_up[0], type_ambiguity, type_scores_out, type_profile_match


# =============================================================================
# STORY QUALITY LAYER:  compute_story_quality
# =============================================================================

def _compute_character_continuity_from_arc(
    arc: Optional[Dict],
    semantic_coherence: float,
) -> float:
    """
    Derive character_continuity from arc beat evidence, not just semantic coherence.

    Signals (priority order):
    1. Fraction of beats with consistent main_character_id events
    2. Arc role sequence stability (setup → inciting → ... → payoff with same actor)
    3. Semantic coherence as low-weight fallback only
    """
    if arc is None:
        return float(np.clip(semantic_coherence * 0.7, 0.0, 1.0))

    beats = arc.get("beats", [])
    if not beats:
        return float(np.clip(semantic_coherence * 0.7, 0.0, 1.0))

    # Collect main_character_id from events in each beat
    beat_characters: List[Optional[str]] = []
    for beat in beats:
        char_ids = [
            ev.get("main_character_id")
            for ev in beat.get("events", [])
            if ev.get("main_character_id")
        ]
        beat_characters.append(char_ids[0] if char_ids else None)

    identified    = [c for c in beat_characters if c is not None]
    n_beats       = len(beats)

    if identified:
        from collections import Counter
        most_common   = Counter(identified).most_common(1)[0][1]
        id_consistency = most_common / len(identified)
        id_coverage    = len(identified) / n_beats
        char_continuity_from_ids = float(np.clip(
            0.60 * id_consistency + 0.40 * id_coverage, 0.0, 1.0
        ))
    else:
        char_continuity_from_ids = 0.0

    # Role-sequence continuity: do roles escalate coherently (setup→...→payoff)?
    # Proxy: are there no "orphan" beats (low role_confidence in a sequence)?
    role_confidences = [
        float(b.get("role_confidence", 0.5)) for b in beats
    ]
    role_seq_stability = float(np.mean(role_confidences)) if role_confidences else 0.5

    # Combine: character IDs (when available), role stability, semantic coherence
    if char_continuity_from_ids > 0.0:
        return round(float(np.clip(
            0.50 * char_continuity_from_ids +
            0.30 * role_seq_stability +
            0.20 * semantic_coherence,
            0.0, 1.0,
        )), 3)
    else:
        # No explicit character IDs: fall back to role stability + coherence
        return round(float(np.clip(
            0.55 * role_seq_stability +
            0.45 * semantic_coherence,
            0.0, 1.0,
        )), 3)


def _compute_topic_continuity_from_arc(
    arc: Optional[Dict],
    narrative_features: Dict,
    semantic_coherence: float,
) -> float:
    """
    Derive topic_continuity from arc structure, not from has_conflict/has_resolution verdicts.

    Signals:
    1. arc causal_density — causal chains indicate topically connected discourse
    2. arc mid_arc_redundancy — low redundancy = focused topic
    3. resolution_closure_evidence — story closes on same topic it opened on
    4. trigger_evidence — explicit topic initiating event
    5. Semantic coherence as residual signal only
    """
    if arc is None:
        # Window fallback: use evidence fields as proxy
        causal_ev   = float(narrative_features.get("causality_evidence",            0.0))
        trigger_ev  = float(narrative_features.get("trigger_evidence",              0.0))
        closure_ev  = float(narrative_features.get("resolution_closure_evidence",   0.0))
        return round(float(np.clip(
            0.35 * semantic_coherence +
            0.30 * (1.0 if causal_ev else 0.5) +
            0.20 * closure_ev +
            0.15 * trigger_ev,
            0.0, 1.0,
        )), 3)

    causal_density  = float(arc.get("causal_density",      0.0))
    redundancy      = float(arc.get("mid_arc_redundancy",  0.5))
    closure_ev      = float(narrative_features.get("resolution_closure_evidence", 0.0))
    trigger_ev      = float(narrative_features.get("trigger_evidence",            0.0))
    temp_ev         = float(narrative_features.get("temporal_progression_evidence", False))

    return round(float(np.clip(
        0.30 * causal_density +
        0.25 * (1.0 - redundancy) +
        0.20 * closure_ev +
        0.15 * semantic_coherence +
        0.10 * float(bool(temp_ev)),
        0.0, 1.0,
    )), 3)


def _compute_non_story_risk(
    arc: Optional[Dict],
    arc_event_recall: float,
    type_profile_match: float,
    arc_completeness: float,
    arc_coherence: float,
    narrative_features: Dict,
) -> float:
    """
    Non-story risk derived from arc/event evidence, not from structure_score/narrativity_score.

    High non-story risk = the candidate lacks real story structure.

    Signals (all arc/event-based):
    1. arc_completeness — does the arc have canonical narrative roles?
    2. arc_event_recall — is the arc backed by real events, not just text markers?
    3. type_profile_match — does it match any known story type profile?
    4. arc_coherence — internally coherent sequence?
    5. trigger_evidence — is there a clear inciting event?

    Lexical signals (structure_score, narrativity_score) deliberately excluded.
    """
    if arc is None:
        # Window fallback: limited evidence, assume moderate risk
        trigger_ev = float(narrative_features.get("trigger_evidence",    0.0))
        causal_ev  = float(narrative_features.get("causality_evidence",  False))
        conflict_ev = float(narrative_features.get("conflict_evidence",  0.0))
        return round(float(np.clip(
            0.40 * (1.0 - arc_completeness) +
            0.25 * (1.0 - type_profile_match) +
            0.20 * (1.0 - conflict_ev) +
            0.15 * (1.0 - float(bool(causal_ev))),
            0.0, 1.0,
        )), 3)

    failure_modes    = set(arc.get("arc_failure_modes", []))
    n_failure_modes  = len(failure_modes)
    trigger_ev       = float(narrative_features.get("trigger_evidence",  0.0))

    base_risk = float(np.clip(
        0.30 * (1.0 - arc_completeness) +
        0.25 * (1.0 - arc_event_recall) +
        0.20 * (1.0 - type_profile_match) +
        0.15 * (1.0 - arc_coherence) +
        0.10 * (1.0 - float(np.clip(trigger_ev, 0.0, 1.0))),
        0.0, 1.0,
    ))

    # Explicit arc failures raise risk
    risk_boost = min(n_failure_modes * 0.08, 0.25)
    return round(float(np.clip(base_risk + risk_boost, 0.0, 1.0)), 3)


def compute_story_quality(
    arc: Optional[Dict],
    narrative_features: Dict,
    semantic_features: Dict,
    sentiment_curve: Optional[List[float]],
    type_ambiguity: float = 0.5,
    window_duration: float = 60.0,
    # ── New arc/event-aware inputs (v2.1) ─────────────────────────────────────
    arc_event_recall: float = 0.0,
    type_profile_match: float = 0.5,
    boundary_diagnostics: Optional[Dict] = None,
) -> Dict:
    """
    Quality / reliability signals for a story candidate (v2.1).

    What changed from v1:
    - character_continuity: arc beat character IDs + role stability (not coherence proxy)
    - topic_continuity: causal_density + closure + trigger (not has_conflict/has_resolution)
    - non_story_risk: arc/event signals only (structure_score/narrativity_score removed)
    - story_export_safety: now incorporates arc_event_recall and clip_self_sufficiency

    Outputs
    -------
    arc_coherence        — internal coherence of the arc
    payoff_strength      — how strong is the resolution
    arc_completeness     — fraction of canonical roles covered
    compression_quality  — arc duration fits story requirements
    character_continuity — consistent protagonist through clip (arc-derived)
    topic_continuity     — same topic maintained through clip (arc-derived)
    false_story_risk     — probability this looks like a story but isn't
    non_story_risk       — probability there is no story structure (arc/event-derived)
    story_export_safety  — single gating score [0..1]
    active_story_failures — list of named failure mode labels
    """
    if arc is not None:
        arc_coherence    = float(arc.get("coherence_score",   0.5))
        payoff_strength  = float(arc.get("payoff_strength",   0.0))
        arc_completeness = float(arc.get("arc_completeness",  0.0))
        arc_duration     = float(arc.get("duration",          60.0))
    else:
        arc_coherence    = 0.30
        payoff_strength  = 0.0
        arc_completeness = 0.0
        arc_duration     = window_duration

    # Compression quality based on arc duration
    if arc_duration < 30.0:
        compression_quality = float(np.clip(arc_duration / 30.0, 0.0, 1.0)) * 0.7
    elif arc_duration > 150.0:
        compression_quality = float(np.clip(1.0 - (arc_duration - 150.0) / 100.0, 0.0, 1.0)) * 0.8
    else:
        compression_quality = 1.0

    semantic_coherence = float(semantic_features.get("coherence", 0.5))

    # ── character_continuity: arc-derived, semantic coherence as residual ────
    character_continuity = _compute_character_continuity_from_arc(arc, semantic_coherence)

    # ── topic_continuity: causal/temporal/closure arc signals ────────────────
    topic_continuity = _compute_topic_continuity_from_arc(arc, narrative_features, semantic_coherence)

    # ── Payoff augmented from sentiment curve ─────────────────────────────────
    payoff_from_curve = 0.0
    if sentiment_curve and len(sentiment_curve) >= 4:
        segs = detect_tension_curve_segments(sentiment_curve)
        if segs["payoff_candidate"] is not None:
            payoff_val        = float(segs["smoothed"][segs["payoff_candidate"]])
            payoff_from_curve = float(np.clip((payoff_val - 0.5) * 2, 0.0, 1.0))
    payoff_strength = max(payoff_strength, payoff_from_curve)

    # ── false_story_risk: arc coherence / payoff / ambiguity ─────────────────
    false_story_risk = round(float(np.clip(
        (1.0 - arc_coherence)    * 0.30 +
        (1.0 - payoff_strength)  * 0.25 +
        type_ambiguity           * 0.25 +
        (1.0 - arc_completeness) * 0.20,
        0.0, 1.0,
    )), 3)

    # ── non_story_risk: arc/event signals only, no lexical priors ────────────
    non_story_risk = _compute_non_story_risk(
        arc=arc,
        arc_event_recall=arc_event_recall,
        type_profile_match=type_profile_match,
        arc_completeness=arc_completeness,
        arc_coherence=arc_coherence,
        narrative_features=narrative_features,
    )

    # ── clip packaging quality ────────────────────────────────────────────────
    clip_sufficiency = float(
        (boundary_diagnostics or {}).get("clip_self_sufficiency", 0.5)
    )

    # ── Failure modes ─────────────────────────────────────────────────────────
    failures: List[str] = []
    if arc_coherence    < 0.30: failures.append("low_arc_coherence")
    if payoff_strength  < 0.20: failures.append("weak_payoff")
    if arc_completeness < 0.30: failures.append("incomplete_arc")
    if false_story_risk > 0.65: failures.append("high_false_story_risk")
    if non_story_risk   > 0.70: failures.append("non_story_structure")
    if type_ambiguity   > 0.70: failures.append("ambiguous_story_type")
    if arc_event_recall < 0.20 and arc is not None:
        failures.append("low_event_recall")   # text-only arc, low confidence

    n_failures = len(failures)

    # ── story_export_safety: now includes event recall + clip sufficiency ─────
    story_export_safety = float(np.clip(
        0.22 * arc_coherence +
        0.20 * payoff_strength +
        0.18 * arc_completeness +
        0.13 * compression_quality +
        0.10 * clip_sufficiency +
        0.09 * arc_event_recall +
        0.08 * (1.0 - false_story_risk) -
        n_failures * 0.10,
        0.0, 1.0,
    ))

    return {
        "arc_coherence":          round(arc_coherence,        3),
        "payoff_strength":        round(payoff_strength,       3),
        "arc_completeness":       round(arc_completeness,      3),
        "compression_quality":    round(compression_quality,   3),
        "character_continuity":   character_continuity,
        "topic_continuity":       topic_continuity,
        "false_story_risk":       false_story_risk,
        "non_story_risk":         round(non_story_risk,        3),
        "story_export_safety":    round(story_export_safety,   3),
        "active_story_failures":  failures,
        "failure_mode_count":     n_failures,
    }


# =============================================================================
# STAGE D — CLIP REFINEMENT:  refine_story_clip_boundaries
# =============================================================================

def refine_story_clip_boundaries(
    arc: Optional[Dict],
    asr_segments: Optional[List[Dict]],
    min_duration: float = 30.0,
    max_duration: float = 150.0,
    snap_sec: float = 1.0,
) -> Tuple[float, float, Dict]:
    """
    Story-specific clip boundary refiner (v2.1).

    Anchors
    -------
    setup_anchor      — start of the earliest setup/inciting_event beat
    payoff_anchor     — start of the first payoff beat (must be preserved)
    reflection_anchor — start of reflection beat (optional extension)

    START logic (priority order):
        1. setup_anchor − adaptive_context_pad to include natural lead-in
           Adaptive pad: extends back to the nearest prosodic pause/sentence
           boundary within a 5s window before setup_anchor.
        2. Snap to prosodic pause (ASR gap >= 0.25s) near refined_start
        3. Snap to sentence boundary near refined_start

    END logic (priority order):
        1. After reflection_anchor end (if fits in max_duration)
        2. After payoff_anchor_end (sentence-snapped, +1s buffer)
        3. High-strength event proxy payoff (if no explicit payoff beat)
        4. Snap to sentence boundary near raw arc end

    Payoff coverage:
        payoff_coverage_score — fraction of payoff beat duration captured in clip.
        0.0 = payoff fully cut. 1.0 = payoff fully included.

    Returns (refined_start, refined_end, diagnostics).
    """
    import re as _re
    _SENTENCE_END = _re.compile(r"[.?!…]+\s*$")
    _SETUP_ROLES  = {"setup", "inciting_event"}

    if arc is None:
        return 0.0, min(min_duration, max_duration), {"source": "fallback"}

    raw_start = float(arc.get("start", 0.0))
    raw_end   = float(arc.get("end",   raw_start + min_duration))
    beats     = arc.get("beats", [])

    # ── Identify role anchors ─────────────────────────────────────────────
    setup_beats      = [b for b in beats if b.get("dominant_role") in _SETUP_ROLES]
    payoff_beats     = [b for b in beats if b.get("dominant_role") == "payoff"]
    reflection_beats = [b for b in beats if b.get("dominant_role") == "reflection"]

    setup_anchor      = setup_beats[0]["start"]                         if setup_beats      else raw_start
    payoff_anchor     = payoff_beats[0]["start"]                        if payoff_beats     else None
    payoff_anchor_end = max(b["end"] for b in payoff_beats)             if payoff_beats     else raw_end
    reflection_anchor = reflection_beats[0]["start"]                    if reflection_beats else None
    reflection_end    = max(b["end"] for b in reflection_beats)         if reflection_beats else raw_end

    # Proxy payoff: if no explicit payoff beat, use the last high-strength event
    proxy_payoff_anchor = None
    if not payoff_beats:
        all_events = []
        for b in beats:
            all_events.extend(b.get("events", []))
        if all_events:
            strong_events = sorted(
                [e for e in all_events if float(e.get("strength", 0)) >= 0.5],
                key=lambda e: e.get("t", 0),
                reverse=True,
            )
            if strong_events:
                proxy_payoff_anchor = float(strong_events[0].get("t", raw_end))
                payoff_anchor_end   = proxy_payoff_anchor + 3.0  # rough estimate

    # ── Adaptive context pad (START) ──────────────────────────────────────
    # Default: 2s; extend up to 5s if a natural pause/sentence is available
    adaptive_pad    = 2.0
    start_source    = "setup_anchor"

    if asr_segments:
        sorted_segs = sorted(asr_segments, key=lambda s: s.get("start", 0))
        # Look for a prosodic pause in the 5s window before setup_anchor
        search_from = max(0.0, setup_anchor - 5.0)
        for i in range(len(sorted_segs) - 1):
            s_end   = float(sorted_segs[i].get("end", 0))
            s_next  = float(sorted_segs[i + 1].get("start", s_end))
            gap     = s_next - s_end
            if search_from <= s_end <= setup_anchor and gap >= 0.25:
                # Use this pause as the natural entry point
                adaptive_pad = setup_anchor - s_next
                start_source = "adaptive_prosodic_pad"
                break
            # Sentence boundary fallback
            if search_from <= float(sorted_segs[i].get("start", 0)) <= setup_anchor:
                if _SENTENCE_END.search((sorted_segs[i].get("text") or "").strip()):
                    candidate_pad = setup_anchor - float(sorted_segs[i].get("start", 0))
                    if candidate_pad <= 5.0:
                        adaptive_pad = candidate_pad
                        start_source = "adaptive_sentence_pad"

    refined_start = max(0.0, setup_anchor - adaptive_pad)

    # ── Compute refined end ───────────────────────────────────────────────
    refined_end = payoff_anchor_end + 1.0   # +1s buffer after payoff
    end_source  = "payoff_anchor_end"

    # Extend to include reflection if it fits
    if reflection_beats and reflection_end - refined_start <= max_duration:
        refined_end = reflection_end
        end_source  = "reflection_anchor_end"

    # Use proxy payoff if no explicit payoff
    if not payoff_beats and proxy_payoff_anchor is not None:
        refined_end = payoff_anchor_end
        end_source  = "proxy_payoff_anchor"

    # ── ASR end snapping ─────────────────────────────────────────────────
    if asr_segments:
        sorted_segs = sorted(asr_segments, key=lambda s: s.get("start", 0))

        # Fine-tune start: prosodic pause or sentence boundary near refined_start
        if start_source not in ("adaptive_prosodic_pad", "adaptive_sentence_pad"):
            for i in range(len(sorted_segs) - 1):
                s        = sorted_segs[i]
                s_end    = float(s.get("end", 0))
                s_next   = float(sorted_segs[i + 1].get("start", s_end))
                gap      = s_next - s_end
                if abs(s_next - refined_start) <= snap_sec and gap >= 0.25:
                    refined_start = s_next
                    start_source  = "prosodic_pause"
                    break
                if abs(float(s.get("start", 0)) - refined_start) <= snap_sec:
                    if _SENTENCE_END.search((s.get("text") or "").strip()):
                        refined_start = float(s.get("start", refined_start))
                        start_source  = "sentence_boundary"
                        break

        # Snap end to sentence boundary
        for s in sorted_segs:
            if abs(float(s.get("end", 0)) - refined_end) <= snap_sec:
                if _SENTENCE_END.search((s.get("text") or "").strip()):
                    refined_end = float(s.get("end", refined_end))
                    end_source  = end_source + "_sentence_snapped"
                    break

    # ── Enforce duration constraints ──────────────────────────────────────
    if refined_end - refined_start < min_duration:
        refined_end = refined_start + min_duration
    if refined_end - refined_start > max_duration:
        # Preserve payoff end if possible, else trim from start
        if payoff_anchor_end - refined_start <= max_duration:
            refined_end = refined_start + max_duration
        else:
            refined_start = payoff_anchor_end - max_duration
            refined_end   = payoff_anchor_end

    clip_dur = max(refined_end - refined_start, 1.0)

    # ── Diagnostics ───────────────────────────────────────────────────────

    # start_context_necessity: setup relative to clip start
    if setup_beats:
        setup_offset   = setup_beats[0]["start"] - refined_start
        context_ratio  = setup_offset / clip_dur
        start_context_necessity = round(float(np.clip(context_ratio * 2.0, 0.0, 1.0)), 3)
    else:
        start_context_necessity = 0.5

    # end_payoff_leakage: fraction of payoff beat that falls outside clip
    if payoff_anchor is not None or proxy_payoff_anchor is not None:
        p_start       = payoff_anchor if payoff_anchor is not None else proxy_payoff_anchor
        payoff_covered = (
            min(payoff_anchor_end, refined_end) - max(p_start, refined_start)
        )
        payoff_dur     = max(payoff_anchor_end - p_start, 0.5)
        end_payoff_leakage = round(float(np.clip(1.0 - payoff_covered / payoff_dur, 0.0, 1.0)), 3)
    else:
        end_payoff_leakage = 0.85   # no payoff anchor at all = high leakage risk

    # payoff_coverage_score: opposite of leakage (how much payoff is captured)
    payoff_coverage_score = round(1.0 - end_payoff_leakage, 3)

    # clip_self_sufficiency (v2.1):
    # Adds payoff_coverage_score, penalises when payoff leaks heavily
    has_payoff     = float(1.0 if payoff_beats      else (0.3 if proxy_payoff_anchor else 0.0))
    has_setup      = float(1.0 if setup_beats        else 0.0)
    has_reflection = float(0.8 if reflection_beats   else 0.3)
    no_hard_trim   = float(1.0 if clip_dur <= max_duration * 0.95 else 0.7)

    clip_self_sufficiency = round(float(np.clip(
        0.30 * has_payoff +
        0.25 * payoff_coverage_score +
        0.20 * has_setup +
        0.12 * has_reflection +
        0.08 * no_hard_trim +
        0.05 * (1.0 - float(end_source in ("arc_boundary", "fallback"))),
        0.0, 1.0,
    )), 3)

    return round(refined_start, 3), round(refined_end, 3), {
        "start_source":             start_source,
        "end_source":               end_source,
        "raw_start":                raw_start,
        "raw_end":                  raw_end,
        "setup_anchor":             round(setup_anchor, 3),
        "payoff_anchor":            round(payoff_anchor, 3) if payoff_anchor is not None else None,
        "proxy_payoff_anchor":      round(proxy_payoff_anchor, 3) if proxy_payoff_anchor is not None else None,
        "reflection_anchor":        round(reflection_anchor, 3) if reflection_anchor is not None else None,
        "has_payoff_anchor":        bool(payoff_beats),
        "has_proxy_payoff":         proxy_payoff_anchor is not None,
        "has_reflection_anchor":    bool(reflection_beats),
        "adaptive_context_pad":     round(adaptive_pad, 3),
        "start_context_necessity":  start_context_necessity,
        "end_payoff_leakage":       end_payoff_leakage,
        "payoff_coverage_score":    payoff_coverage_score,
        "clip_self_sufficiency":    clip_self_sufficiency,
    }


# =============================================================================
# EVENT RECALL AUDIT
# =============================================================================

def compute_arc_event_recall(arc: Dict) -> Dict:
    """
    Audit: was this arc genuinely proposed by the event layer, or did it
    emerge by accident from text-marker-only evidence?

    Returns
    -------
    {
        arc_event_recall          — float [0..1]: multimodal event support
        arc_proposed_by_event_layer — bool: dominated by real events (not just text)
        event_support_by_beat     — list of per-beat event support scores
        modal_event_sources       — set of modality strings seen across arc
        text_only_beats           — int: beats with only text-source events
        multimodal_beats          — int: beats with >=2 modality sources
        audit_summary             — short string for logging
    }
    """
    beats          = arc.get("beats", [])
    if not beats:
        return {
            "arc_event_recall":            0.0,
            "arc_proposed_by_event_layer": False,
            "event_support_by_beat":       [],
            "modal_event_sources":         [],
            "text_only_beats":             0,
            "multimodal_beats":            0,
            "audit_summary":               "no_beats",
        }

    per_beat_scores: List[float] = []
    modal_sources:   set         = set()
    text_only_beats  = 0
    multimodal_beats = 0

    for beat in beats:
        events     = beat.get("events", [])
        if not events:
            per_beat_scores.append(0.0)
            text_only_beats += 1
            continue

        beat_modalities: set = set()
        beat_strength        = 0.0

        for ev in events:
            mods     = ev.get("source_modalities", [ev.get("source", "text")])
            strength = float(ev.get("strength", 0.5))
            beat_modalities.update(mods)
            modal_sources.update(mods)
            beat_strength = max(beat_strength, strength)

        # Event recall score for this beat:
        # - base: max event strength
        # - bonus: number of distinct modalities (multimodal = higher recall)
        # - bonus: any non-text modality (real events, not just text markers)
        has_non_text  = any(m not in ("text", "asr_turn", "discourse",
                                       "setup_marker", "conflict_marker",
                                       "resolution_marker", "trigger_marker")
                            for m in beat_modalities)
        n_modalities  = len(beat_modalities)

        beat_recall = float(np.clip(
            0.50 * beat_strength +
            0.30 * float(np.clip((n_modalities - 1) / 2.0, 0.0, 1.0)) +
            0.20 * float(has_non_text),
            0.0, 1.0,
        ))
        per_beat_scores.append(round(beat_recall, 3))

        if beat_modalities <= {"text"}:
            text_only_beats += 1
        if n_modalities >= 2:
            multimodal_beats += 1

    arc_event_recall = round(float(np.mean(per_beat_scores)) if per_beat_scores else 0.0, 3)

    # Proposed by event layer if:
    # - arc_event_recall >= 0.4, AND
    # - at least 1 multimodal beat, OR at least 1 non-text event source
    non_text_sources = modal_sources - {"text", "asr_turn", "discourse",
                                         "setup_marker", "conflict_marker",
                                         "resolution_marker", "trigger_marker"}
    arc_proposed_by_event_layer = (
        arc_event_recall >= 0.40 and
        (multimodal_beats >= 1 or bool(non_text_sources))
    )

    if arc_proposed_by_event_layer:
        audit_summary = f"event_layer: recall={arc_event_recall:.2f} multimodal={multimodal_beats}/{len(beats)}"
    elif arc_event_recall >= 0.20:
        audit_summary = f"partial_event: recall={arc_event_recall:.2f} text_only={text_only_beats}/{len(beats)}"
    else:
        audit_summary = f"text_markers_only: recall={arc_event_recall:.2f}"

    return {
        "arc_event_recall":            arc_event_recall,
        "arc_proposed_by_event_layer": arc_proposed_by_event_layer,
        "event_support_by_beat":       per_beat_scores,
        "modal_event_sources":         sorted(modal_sources),
        "text_only_beats":             text_only_beats,
        "multimodal_beats":            multimodal_beats,
        "audit_summary":               audit_summary,
    }


# =============================================================================
# STORY EXPORT DECISION GATE
# =============================================================================

_EXPORT_DECISION_AUTO_EXPORT   = "auto_export"
_EXPORT_DECISION_MANUAL_REVIEW = "manual_review"
_EXPORT_DECISION_REJECT        = "reject"

def story_export_decision(
    story_export_safety: float,
    compression_quality: float,
    payoff_strength: float,
    type_ambiguity: float,
    type_profile_match: float,
    active_story_failures: Optional[List[str]],
    boundary_diagnostics: Optional[Dict],
    clip_self_sufficiency: float,
    candidate_origin: str = "event_beat_arc",
    arc_completeness: float = 0.0,
) -> Dict:
    """
    Production export gate: decides auto_export / manual_review / reject.

    Inputs come directly from compute_story_quality() and
    refine_story_clip_boundaries() outputs.

    Returns
    -------
    {
        "decision":        "auto_export" | "manual_review" | "reject",
        "export_score":    float [0..1]  — overall export readiness,
        "reject_reasons":  List[str]     — reasons for reject / manual_review,
        "auto_export_ok":  bool,
        "needs_manual_review": bool,
    }
    """
    failures       = set(active_story_failures or [])
    bd             = boundary_diagnostics or {}
    reject_reasons: List[str] = []

    # ── Hard reject conditions ──────────────────────────────────────────────
    # These immediately prevent export regardless of scores.
    HARD_REJECT_FAILURES = {
        "no_payoff_beat",
        "no_conflict_evidence",
        "unearned_payoff",
    }
    hard_failures = failures & HARD_REJECT_FAILURES
    if hard_failures:
        reject_reasons.append(f"hard_failures: {sorted(hard_failures)}")

    if story_export_safety < 0.18:
        reject_reasons.append(f"story_export_safety={story_export_safety:.2f} < 0.18")

    if payoff_strength < 0.10:
        reject_reasons.append(f"payoff_strength={payoff_strength:.2f} < 0.10")

    if reject_reasons:
        return {
            "decision":            _EXPORT_DECISION_REJECT,
            "export_score":        round(story_export_safety * 0.5, 3),
            "reject_reasons":      reject_reasons,
            "auto_export_ok":      False,
            "needs_manual_review": False,
        }

    # ── Soft warning conditions → manual review ─────────────────────────────
    manual_reasons: List[str] = []

    if story_export_safety < 0.40:
        manual_reasons.append(f"story_export_safety={story_export_safety:.2f} < 0.40")

    if type_ambiguity > 0.70:
        manual_reasons.append(f"type_ambiguity={type_ambiguity:.2f} > 0.70")

    if type_profile_match < 0.25:
        manual_reasons.append(f"type_profile_match={type_profile_match:.2f} < 0.25")

    if compression_quality < 0.30:
        manual_reasons.append(f"compression_quality={compression_quality:.2f} < 0.30")

    if clip_self_sufficiency < 0.30:
        manual_reasons.append(f"clip_self_sufficiency={clip_self_sufficiency:.2f} < 0.30")

    SOFT_WARN_FAILURES = {"low_causal_density", "high_mid_arc_redundancy"}
    if failures & SOFT_WARN_FAILURES:
        manual_reasons.append(f"soft_failures: {sorted(failures & SOFT_WARN_FAILURES)}")

    if candidate_origin == "window_fallback":
        manual_reasons.append("candidate_origin=window_fallback (lower confidence)")

    end_leakage = bd.get("end_payoff_leakage", 0.0)
    if end_leakage > 0.5:
        manual_reasons.append(f"end_payoff_leakage={end_leakage:.2f} > 0.50")

    # ── Export score ─────────────────────────────────────────────────────────
    export_score = float(np.clip(
        0.30 * story_export_safety +
        0.20 * payoff_strength +
        0.15 * compression_quality +
        0.15 * clip_self_sufficiency +
        0.10 * arc_completeness +
        0.10 * (1.0 - type_ambiguity),
        0.0, 1.0,
    ))

    # Origin penalty already baked in score if fallback, but add explicit flag
    if candidate_origin == "window_fallback":
        export_score = float(np.clip(export_score * 0.88, 0.0, 1.0))

    # ── Decision ─────────────────────────────────────────────────────────────
    if manual_reasons or export_score < 0.55:
        if export_score < 0.35:
            return {
                "decision":            _EXPORT_DECISION_REJECT,
                "export_score":        round(export_score, 3),
                "reject_reasons":      manual_reasons,
                "auto_export_ok":      False,
                "needs_manual_review": False,
            }
        return {
            "decision":            _EXPORT_DECISION_MANUAL_REVIEW,
            "export_score":        round(export_score, 3),
            "reject_reasons":      manual_reasons,
            "auto_export_ok":      False,
            "needs_manual_review": True,
        }

    return {
        "decision":            _EXPORT_DECISION_AUTO_EXPORT,
        "export_score":        round(export_score, 3),
        "reject_reasons":      [],
        "auto_export_ok":      True,
        "needs_manual_review": False,
    }


# =============================================================================
# MAIN STORY MODE FUNCTION
# =============================================================================

# Минимальный порог по narrative_score: гарантия "это история"
MIN_NARRATIVE_THRESHOLD = 0.5


def find_story_moments(
    video_path: str,
    video_duration_sec: float,
    asr_segments: Optional[List[Dict]] = None,
    base_analysis: Optional[Dict] = None,
    audio_array: Optional[np.ndarray] = None,
    sr: int = 16000,
    top_k: int = 3,
    # --- Режимные параметры (deprecated: передавайте через config) ---
    window_size: Optional[float] = None,
    step_size: Optional[float] = None,
    threshold: Optional[float] = None,
    min_narrative_threshold: Optional[float] = None,
    require_conflict_and_resolution: Optional[bool] = None,
    # --- Конфиг ---
    config: Optional[StoryModeConfig] = None,
    # --- LLM-колбэки ---
    llm_semantic_callback: Optional[callable] = None,
    llm_structure_callback: Optional[callable] = None,
    llm_sentiment_callback: Optional[callable] = None,
) -> Dict:
    """
    Story Mode v2.0 — event → beat → role → arc → clip pipeline.

    Pipeline (dual: event-based primary, window fallback):
      Stage A  detect_story_events → merge_story_events_to_beats
      Stage B  label_narrative_roles (per beat)
      Stage C  build_story_arcs (beat sequences → arcs)
      Stage D  For each arc:
               - refine_story_clip_boundaries
               - compute sentiment curve + detect_arc_change_points
               - classify_story_type_soft
               - compute_story_quality
               - compute_story_score (final)
      Fallback if no arcs: legacy sliding-window pipeline (backward-compat).

    Extended output contract per moment:
        start, end, duration, score, type="story"
        story_type, secondary_story_type, type_ambiguity
        subscores, reasons
        arc_id, arc_roles, role_sequence
        story_beats (list of beat summaries)
        story_structure (has_setup/conflict/resolution + evidence fields)
        arc_pattern, arc_confidence
        coherence_score, payoff_strength, compression_quality
        character_continuity, topic_continuity
        story_export_safety, active_story_failures
        narrativity_score, has_takeaway
        modal_contrib (visual/audio/semantic/narrative subscores)
        story_graph_summary
        boundary_diagnostics
        transcript (первые 300 символов)
    """
    logger.info("=" * 70)
    logger.info("STORY MODE v2.0 — Event → Beat → Arc → Clip")
    logger.info("=" * 70)

    cfg = config or StoryModeConfig()
    _window_size = window_size if window_size is not None else cfg.window_size
    _step_size   = step_size   if step_size   is not None else cfg.step_size
    _threshold   = threshold   if threshold   is not None else cfg.threshold
    _min_narrative = (min_narrative_threshold if min_narrative_threshold is not None
                      else cfg.min_narrative_threshold)
    _require_c_r = (require_conflict_and_resolution
                    if require_conflict_and_resolution is not None
                    else cfg.require_conflict_and_resolution)

    if not asr_segments:
        logger.warning("Story Mode requires ASR transcription!")
        return _empty_story_result(video_duration_sec, "no_transcription", cfg)

    merged_cfg = cfg.merge_config()

    all_moments: List[Dict] = []
    filter_counts = {
        "too_short_transcript": 0,
        "no_conflict": 0,
        "no_resolution": 0,
        "loose_rejected": 0,
        "non_story": 0,
        "short_for_type": 0,
        "low_narrative": 0,
        "low_export_safety": 0,
        "weak_promoted_to_manual_review": 0,
    }
    narrative_scores_list: List[float] = []
    arc_patterns_list:     List[str]   = []
    pipeline_mode = "unknown"

    # v2.1: diagnostics storage
    rejected_arcs: List[Dict] = []
    beat_to_arc_trace: List[Dict] = []
    weak_story_pool: List[Dict] = []   # weak candidates for manual_review fallback

    # v2.1: ease short_for_type для видео до 3 минут
    short_video_duration_ease = video_duration_sec < 180.0
    short_for_type_ease_factor = 0.50 if short_video_duration_ease else 1.0
    if short_video_duration_ease:
        logger.info(
            f"short_video_duration_ease: video={video_duration_sec:.1f}s < 180s, "
            f"type_min_duration × {short_for_type_ease_factor:.2f}"
        )

    # =========================================================================
    # PRIMARY PIPELINE: event → beat → role → arc
    # =========================================================================
    try:
        # Stage A: eventization
        story_events = detect_story_events(
            asr_segments, base_analysis, video_duration_sec, config=cfg
        )
        beats = merge_story_events_to_beats(
            story_events, asr_segments,
            cluster_window_sec=3.0,
            min_beat_strength=0.18,
        )
        logger.info(f"  Stage A: {len(story_events)} events → {len(beats)} beats")

        if beats:
            # Stage B: role labeling
            # Pre-compute full-video sentiment curve for role labeling context
            full_sentiment = compute_sentiment_curve_from_asr(
                asr_segments, 0.0, video_duration_sec,
                min_segments=3, config=merged_cfg,
                llm_sentiment_callback=llm_sentiment_callback,
            )
            beats = label_narrative_roles(
                beats, asr_segments, config=cfg, sentiment_curve=full_sentiment
            )

            # Stage C: arc assembly
            arcs = build_story_arcs(beats, min_arc_completeness=0.35,
                                    max_arc_duration=cfg.max_clip_duration)
            logger.info(f"  Stage C: {len(arcs)} arc(s) assembled from {len(beats)} beats")

            # v2.1: beat→arc trace for diagnostics
            for arc in arcs:
                beat_to_arc_trace.append({
                    "arc_id": arc.get("arc_id"),
                    "start": arc.get("start"),
                    "end": arc.get("end"),
                    "n_beats": arc.get("n_beats", 0),
                    "arc_completeness": arc.get("arc_completeness", 0.0),
                    "role_sequence": arc.get("role_sequence", []),
                    "beat_starts": [b.get("start") for b in arc.get("beats", [])],
                })

            pipeline_mode = "event_beat_arc"

            # Stage D: per-arc scoring → moments
            for arc in arcs:
                try:
                    # Refine boundaries
                    refined_start, refined_end, boundary_diag = refine_story_clip_boundaries(
                        arc, asr_segments,
                        min_duration=cfg.min_clip_duration,
                        max_duration=cfg.max_clip_duration,
                    )
                    duration_sec = refined_end - refined_start

                    # Transcript for this arc
                    arc_segs = [
                        s for s in asr_segments
                        if s.get("start", 0) <= refined_end and s.get("end", 0) >= refined_start
                    ]
                    transcript = " ".join(s.get("text", "") for s in arc_segs).strip()
                    if len(transcript) < 50:
                        filter_counts["too_short_transcript"] += 1
                        continue

                    # Sentiment curve for arc window
                    sentiment_curve = compute_sentiment_curve_from_asr(
                        asr_segments, refined_start, refined_end,
                        min_segments=3, config=merged_cfg,
                        llm_sentiment_callback=llm_sentiment_callback,
                    )
                    if len(sentiment_curve) >= 4:
                        arc_pattern, arc_confidence = detect_arc_change_points(sentiment_curve)
                    elif len(sentiment_curve) >= 3:
                        arc_pattern, arc_confidence = detect_story_arc_pattern(sentiment_curve)
                    else:
                        arc_pattern, arc_confidence = "flat", 0.0

                    # Story structure (evidence fields, not verdict)
                    story_structure = detect_story_structure(
                        transcript, config=merged_cfg,
                        llm_callback=llm_structure_callback,
                        asr_segments=arc_segs,
                        window_start=refined_start,
                        window_end=refined_end,
                        use_time_aware=True,
                    )

                    # Semantic features
                    semantic_features = compute_semantic_features_story(
                        transcript,
                        llm_callback=llm_semantic_callback,
                        asr_segments=arc_segs,
                    )
                    has_takeaway = (
                        semantic_features.get("has_takeaway", False)
                        or detect_has_takeaway(transcript, config=merged_cfg)
                    )

                    # Narrativity prior (weak, not gating)
                    narrativity_score = detect_narrativity_score(transcript)

                    # Soft story type classification
                    arc_roles = arc["role_sequence"]
                    (story_type, type_conf, secondary_type, type_ambiguity,
                     type_scores_all, type_profile_match) = classify_story_type_soft(
                        transcript,
                        arc_pattern=arc_pattern,
                        has_takeaway=has_takeaway,
                        sentiment_curve=sentiment_curve,
                        arc_roles=arc_roles,
                        semantic_features=semantic_features,
                    )

                    if story_type == "non_story":
                        filter_counts["non_story"] += 1
                        rejected_arcs.append({
                            "arc_id": arc.get("arc_id"),
                            "start": refined_start, "end": refined_end,
                            "duration_sec": round(duration_sec, 2),
                            "reject_reason": "non_story",
                            "story_type": story_type,
                            "arc_pattern": arc_pattern,
                            "transcript_preview": transcript[:80],
                        })
                        continue

                    type_min_dur = STORY_SEGMENT_TYPES.get(story_type, {}).get("min_duration", 30)
                    eff_type_min = type_min_dur * short_for_type_ease_factor

                    if duration_sec < eff_type_min:
                        # Hard reject — слишком короткий даже с relaxed порогом
                        filter_counts["short_for_type"] += 1
                        rejected_arcs.append({
                            "arc_id": arc.get("arc_id"),
                            "start": refined_start, "end": refined_end,
                            "duration_sec": round(duration_sec, 2),
                            "reject_reason": "short_for_type",
                            "story_type": story_type,
                            "type_min_duration": type_min_dur,
                            "eff_type_min": round(eff_type_min, 1),
                            "arc_pattern": arc_pattern,
                            "transcript_preview": transcript[:80],
                        })
                        # v2.1: всё равно сохраняем как weak для manual_review
                        weak_story_pool.append({
                            "start": refined_start, "end": refined_end,
                            "duration_sec": round(duration_sec, 2),
                            "arc_id": arc.get("arc_id"),
                            "story_type": story_type,
                            "arc_pattern": arc_pattern,
                            "arc_completeness": arc.get("arc_completeness", 0.0),
                            "reason_code": "weak_story_manual_review",
                            "reject_reason_original": "short_for_type",
                            "transcript_preview": transcript[:120],
                            "arc_roles": arc.get("role_sequence", []),
                        })
                        continue
                    elif duration_sec < type_min_dur:
                        # Soft pass (в диапазоне [eff_min, type_min]) — помечаем как относительно короткий,
                        # но не режем
                        logger.debug(
                            f"  arc={arc.get('arc_id')} short-but-passed: "
                            f"{duration_sec:.1f}s < type_min={type_min_dur}s "
                            f"(eased to {eff_type_min:.1f}s)"
                        )

                    # Narrative score — arc-aware path
                    narrative_score = compute_narrative_score(
                        story_structure, narrativity_score,
                        arc_pattern, arc_confidence,
                        story_type=story_type,
                        arc_roles=arc_roles,
                        arc_coherence=arc.get("coherence_score", 0.0),
                        payoff_strength=arc.get("payoff_strength", 0.0),
                        compression_quality=-1.0,   # auto-estimate from arc_raw
                        character_continuity=float(semantic_features.get("character_continuity",
                                                    semantic_features.get("coherence", 0.5))),
                        topic_continuity=float(semantic_features.get("topic_continuity",
                                                semantic_features.get("coherence", 0.5))),
                        type_ambiguity=type_ambiguity,
                        arc_completeness=arc.get("arc_completeness", 0.0),
                        arc_raw=arc,
                        trigger_evidence=float(story_structure.get("trigger_evidence", 0.0)),
                        resolution_closure=float(story_structure.get("resolution_closure_evidence", 0.0)),
                        type_profile_match=type_profile_match,
                    )
                    narrative_scores_list.append(narrative_score)
                    arc_patterns_list.append(arc_pattern)

                    # narrative threshold is relaxed for arc-based path
                    # (arc_quality already gates via export_safety below)
                    if narrative_score < _min_narrative * 0.70:
                        filter_counts["low_narrative"] += 1
                        continue

                    # Quality layer
                    # Pre-compute event recall for quality layer (arc beats available)
                    _pre_event_recall = compute_arc_event_recall(arc)
                    quality = compute_story_quality(
                        arc=arc,
                        narrative_features={
                            **story_structure,
                            "narrative_score":   narrative_score,
                            "narrativity_score": narrativity_score,
                        },
                        semantic_features=semantic_features,
                        sentiment_curve=sentiment_curve,
                        type_ambiguity=type_ambiguity,
                        window_duration=duration_sec,
                        arc_event_recall=_pre_event_recall["arc_event_recall"],
                        type_profile_match=type_profile_match,
                        boundary_diagnostics=boundary_diag,
                    )

                    if quality["story_export_safety"] < 0.20:
                        filter_counts["low_export_safety"] += 1
                        logger.debug(
                            f"  Arc {arc['arc_id']} rejected: "
                            f"export_safety={quality['story_export_safety']:.2f}"
                        )
                        continue

                    # Visual / audio features (story-event aware)
                    visual_features, audio_features = get_visual_audio_from_base_analysis(
                        base_analysis, refined_start, refined_end
                    )
                    narrative_features = {
                        **story_structure,
                        "narrative_score":   narrative_score,
                        "narrativity_score": narrativity_score,
                        "arc_pattern":       arc_pattern,
                        "arc_confidence":    arc_confidence,
                    }

                    final_score, subscores, reasons = compute_story_score(
                        visual_features, audio_features, semantic_features,
                        narrative_features, story_type,
                        duration_sec=duration_sec,
                        has_takeaway=has_takeaway,
                        config=cfg,
                        arc=arc,
                        clip_self_sufficiency=boundary_diag.get("clip_self_sufficiency", 0.5),
                    )

                    if final_score < _threshold:
                        continue

                    # Story graph summary
                    story_graph_summary = {
                        "n_beats":         arc["n_beats"],
                        "role_sequence":   arc_roles,
                        "arc_completeness": arc["arc_completeness"],
                        "dominant_sources": list({b["dominant_source"] for b in arc["beats"]}),
                    }

                    # Beat summaries (include role_probs for explainability)
                    beat_summaries = [
                        {
                            "start":           b["start"],
                            "end":             b["end"],
                            "dominant_role":   b["dominant_role"],
                            "role_confidence": b.get("role_confidence", 0.0),
                            "role_probs":      b.get("role_probs", {}),
                            "mean_strength":   b["mean_strength"],
                            "n_sources":       b.get("n_sources", 1),
                            "beat_score":      compute_beat_score(b),
                            "snippet":         b.get("transcript_snippet", "")[:80],
                        }
                        for b in arc["beats"]
                    ]

                    summary   = transcript[:100] + "..." if len(transcript) > 100 else transcript
                    title_tpl = STORY_SEGMENT_TYPES.get(story_type, {}).get(
                        "title_template", "{summary}"
                    )
                    title = title_tpl.format(summary=summary[:50])

                    moment = {
                        "start":                 refined_start,
                        "end":                   refined_end,
                        "duration":              round(duration_sec, 3),
                        "score":                 round(final_score, 3),
                        "type":                  "story",
                        # ── Story type ──────────────────────────────────────
                        "story_type":            story_type,
                        "secondary_story_type":  secondary_type,
                        "type_ambiguity":        round(type_ambiguity, 3),
                        "type_scores":           type_scores_all,
                        "type_profile_match":    type_profile_match,
                        # ── Arc / beats ─────────────────────────────────────
                        "arc_id":                arc["arc_id"],
                        "arc_roles":             arc_roles,
                        "story_beats":           beat_summaries,
                        "story_graph_summary":   story_graph_summary,
                        # ── Narrative ────────────────────────────────────────
                        "story_structure": {
                            "has_setup":                     story_structure["has_setup"],
                            "has_conflict":                  story_structure["has_conflict"],
                            "has_resolution":                story_structure["has_resolution"],
                            "setup_evidence":                story_structure.get("setup_evidence", 0.0),
                            "conflict_evidence":             story_structure.get("conflict_evidence", 0.0),
                            "resolution_evidence":           story_structure.get("resolution_evidence", 0.0),
                            "causality_evidence":            story_structure.get("causality_evidence", False),
                            "temporal_progression_evidence": story_structure.get("temporal_progression_evidence", False),
                            "protagonist_evidence":          story_structure.get("protagonist_evidence", False),
                        },
                        "arc_pattern":           arc_pattern,
                        "arc_confidence":        round(arc_confidence, 3),
                        "narrativity_score":     round(narrativity_score, 3),
                        # ── Quality ──────────────────────────────────────────
                        "coherence_score":       quality["arc_coherence"],
                        "payoff_strength":       quality["payoff_strength"],
                        "compression_quality":   quality["compression_quality"],
                        "character_continuity":  quality["character_continuity"],
                        "topic_continuity":      quality["topic_continuity"],
                        "story_export_safety":   quality["story_export_safety"],
                        "active_story_failures": quality["active_story_failures"],
                        # ── Arc diagnostics ──────────────────────────────────
                        "arc_diagnostics": {
                            "arc_completeness":        arc.get("arc_completeness",        0.0),
                            "causal_density":          arc.get("causal_density",          0.0),
                            "payoff_dependency_score": arc.get("payoff_dependency_score", 0.0),
                            "setup_necessity_score":   arc.get("setup_necessity_score",   0.0),
                            "mid_arc_redundancy":      arc.get("mid_arc_redundancy",      0.0),
                            "arc_failure_modes":       arc.get("arc_failure_modes",       []),
                            "arc_score":               subscores.get("arc", 0.0),
                        },
                        # ── Scores / reasons ────────────────────────────────
                        "subscores":     subscores,
                        "modal_contrib": subscores,
                        "reasons":       reasons,
                        # ── Meta ─────────────────────────────────────────────
                        "has_takeaway":         has_takeaway,
                        "title":                title,
                        "summary":              summary,
                        "transcript":           transcript[:300],
                        "boundary_diagnostics": boundary_diag,
                        "pipeline":             "event_beat_arc",
                        "needs_manual_review":  quality["failure_mode_count"] >= 2,
                    }

                    # ── Event recall audit ────────────────────────────────────
                    event_recall_audit = compute_arc_event_recall(arc)
                    moment["arc_event_recall"]            = event_recall_audit["arc_event_recall"]
                    moment["arc_proposed_by_event_layer"] = event_recall_audit["arc_proposed_by_event_layer"]
                    moment["event_recall_audit"]          = event_recall_audit

                    # ── Production export gate ────────────────────────────────
                    export_gate = story_export_decision(
                        story_export_safety   = quality["story_export_safety"],
                        compression_quality   = quality["compression_quality"],
                        payoff_strength       = quality["payoff_strength"],
                        type_ambiguity        = type_ambiguity,
                        type_profile_match    = type_profile_match,
                        active_story_failures = quality["active_story_failures"],
                        boundary_diagnostics  = boundary_diag,
                        clip_self_sufficiency = boundary_diag.get("clip_self_sufficiency", 0.5),
                        candidate_origin      = "event_beat_arc",
                        arc_completeness      = arc.get("arc_completeness", 0.0),
                    )
                    moment["export_decision"]      = export_gate["decision"]
                    moment["export_score"]         = export_gate["export_score"]
                    moment["export_reject_reasons"]= export_gate["reject_reasons"]
                    # Merge auto_export_ok / needs_manual_review from gate
                    moment["auto_export_ok"]       = export_gate["auto_export_ok"]
                    if export_gate["needs_manual_review"]:
                        moment["needs_manual_review"] = True

                    all_moments.append(moment)
                    logger.info(
                        f"  ✓ arc={arc['arc_id']} {refined_start:.1f}s-{refined_end:.1f}s | "
                        f"{story_type} | score={final_score:.3f} | arc={arc_pattern} | "
                        f"safety={quality['story_export_safety']:.2f} | "
                        f"export={export_gate['decision']} | "
                        f"roles={arc_roles}"
                    )

                except Exception as exc:
                    logger.error(f"  Arc {arc.get('arc_id', '?')} scoring error: {exc}")
                    continue

    except Exception as exc:
        logger.warning(f"  Primary pipeline failed ({exc}); falling back to window mode")
        arcs = []

    # =========================================================================
    # FALLBACK PIPELINE: sliding windows (backward-compatible)
    # =========================================================================
    if not all_moments:
        pipeline_mode = "window_fallback"
        logger.info(f"  Fallback: sliding windows | size={_window_size}s | step={_step_size}s")

        windows: List[Dict] = []
        pos = 0.0
        while pos < video_duration_sec:
            end = min(pos + _window_size, video_duration_sec)
            if end - pos >= cfg.min_clip_duration:
                windows.append({"start": pos, "end": end})
            pos += _step_size

        for window in windows:
            start, end = window["start"], window["end"]
            duration_sec = end - start

            window_segs = [
                s for s in asr_segments
                if s.get("start", 0) <= end and s.get("end", 0) >= start
            ]
            transcript = " ".join(s.get("text", "") for s in window_segs).strip()

            if len(transcript) < 50:
                filter_counts["too_short_transcript"] += 1
                continue

            try:
                story_structure = detect_story_structure(
                    transcript, config=merged_cfg,
                    llm_callback=llm_structure_callback,
                    asr_segments=window_segs,
                    window_start=start, window_end=end,
                    use_time_aware=True,
                )
                has_conflict   = story_structure.get("has_conflict",   False)
                has_resolution = story_structure.get("has_resolution", False)

                if _require_c_r:
                    if not has_conflict:
                        filter_counts["no_conflict"]   += 1
                        # v2.1: weak_story_pool если хотя бы какой-то evidence
                        _setup_ev = float(story_structure.get("setup_evidence", 0.0))
                        _progress_ev = story_structure.get("temporal_progression_evidence", False)
                        if _setup_ev > 0.25 or _progress_ev:
                            weak_story_pool.append({
                                "start": start, "end": end,
                                "duration_sec": round(duration_sec, 2),
                                "story_type": "weak_fragment",
                                "reason_code": "weak_story_manual_review",
                                "reject_reason_original": "no_conflict",
                                "setup_evidence": round(_setup_ev, 3),
                                "has_temporal_progression": bool(_progress_ev),
                                "transcript_preview": transcript[:120],
                            })
                        continue
                    if not has_resolution:
                        filter_counts["no_resolution"] += 1; continue
                else:
                    if not has_conflict:
                        filter_counts["no_conflict"] += 1
                        _setup_ev = float(story_structure.get("setup_evidence", 0.0))
                        if _setup_ev > 0.25:
                            weak_story_pool.append({
                                "start": start, "end": end,
                                "duration_sec": round(duration_sec, 2),
                                "story_type": "weak_fragment",
                                "reason_code": "weak_story_manual_review",
                                "reject_reason_original": "no_conflict",
                                "setup_evidence": round(_setup_ev, 3),
                                "transcript_preview": transcript[:120],
                            })
                        continue

                sentiment_curve = compute_sentiment_curve_from_asr(
                    asr_segments, start, end, min_segments=3,
                    config=merged_cfg,
                    llm_sentiment_callback=llm_sentiment_callback,
                )

                if len(sentiment_curve) >= 4:
                    arc_pattern, arc_confidence = detect_arc_change_points(sentiment_curve)
                elif len(sentiment_curve) >= 3:
                    arc_pattern, arc_confidence = detect_story_arc_pattern(sentiment_curve)
                else:
                    arc_pattern, arc_confidence = "flat", 0.0

                if cfg.loose_story_mode and not has_resolution:
                    if len(sentiment_curve) < 3:
                        filter_counts["loose_rejected"] += 1; continue
                    tail = sentiment_curve[-max(1, len(sentiment_curve) // 3):]
                    head = sentiment_curve[:max(1, len(sentiment_curve) // 3)]
                    if float(np.mean(tail)) - float(np.mean(head)) < 0.15 and float(np.mean(tail)) < 0.70:
                        filter_counts["loose_rejected"] += 1; continue

                narrativity_score = detect_narrativity_score(transcript)
                semantic_features = compute_semantic_features_story(
                    transcript,
                    llm_callback=llm_semantic_callback,
                    asr_segments=asr_segments,
                )
                has_takeaway = (
                    semantic_features.get("has_takeaway", False)
                    or detect_has_takeaway(transcript, config=merged_cfg)
                )

                (story_type, type_conf, secondary_type, type_ambiguity,
                 type_scores_all, type_profile_match) = classify_story_type_soft(
                    transcript, arc_pattern=arc_pattern,
                    has_takeaway=has_takeaway,
                    sentiment_curve=sentiment_curve,
                    semantic_features=semantic_features,
                )

                if story_type == "non_story":
                    filter_counts["non_story"] += 1; continue

                type_min_dur = STORY_SEGMENT_TYPES.get(story_type, {}).get("min_duration", 30)
                if duration_sec < type_min_dur:
                    filter_counts["short_for_type"] += 1; continue

                narrative_score = compute_narrative_score(
                    story_structure, narrativity_score, arc_pattern, arc_confidence,
                    story_type=story_type,
                )
                narrative_scores_list.append(narrative_score)
                arc_patterns_list.append(arc_pattern)

                if narrative_score < _min_narrative:
                    filter_counts["low_narrative"] += 1; continue

                quality = compute_story_quality(
                    arc=None,
                    narrative_features={
                        **story_structure,
                        "narrative_score":  narrative_score,
                        "narrativity_score": narrativity_score,
                    },
                    semantic_features=semantic_features,
                    sentiment_curve=sentiment_curve,
                    type_ambiguity=type_ambiguity,
                    window_duration=duration_sec,
                    arc_event_recall=0.0,        # no arc events in fallback
                    type_profile_match=type_profile_match,
                    boundary_diagnostics=None,
                )

                visual_features, audio_features = get_visual_audio_from_base_analysis(
                    base_analysis, start, end
                )
                narrative_features = {
                    **story_structure,
                    "narrative_score":  narrative_score,
                    "narrativity_score": narrativity_score,
                    "arc_pattern":      arc_pattern,
                    "arc_confidence":   arc_confidence,
                }
                final_score, subscores, reasons = compute_story_score(
                    visual_features, audio_features, semantic_features,
                    narrative_features, story_type,
                    duration_sec=duration_sec,
                    has_takeaway=has_takeaway,
                    config=cfg,
                    arc=None,
                    clip_self_sufficiency=0.35,  # window fallback: no explicit clip anchors
                )

                # ── Fallback is lower-confidence: apply score penalty ─────────
                # Window-based candidates have no arc evidence; penalise by 15%
                # so arc-based moments always rank higher at equal quality.
                _FALLBACK_PENALTY = 0.15
                fallback_score    = float(np.clip(final_score * (1.0 - _FALLBACK_PENALTY), 0, 1))
                # Require higher threshold for fallback candidates
                fallback_threshold = min(1.0, _threshold + 0.05)

                if fallback_score >= fallback_threshold:
                    summary   = transcript[:100] + "..." if len(transcript) > 100 else transcript
                    title_tpl = STORY_SEGMENT_TYPES.get(story_type, {}).get("title_template", "{summary}")
                    title     = title_tpl.format(summary=summary[:50])

                    moment = {
                        "start": start, "end": end, "duration": duration_sec,
                        # Penalised score keeps fallback moments ranked below arc-based ones
                        "score":                round(fallback_score, 3),
                        "raw_score_before_penalty": round(final_score, 3),
                        "type":                 "story",
                        "story_type":           story_type,
                        "secondary_story_type": secondary_type,
                        "type_ambiguity":       round(type_ambiguity, 3),
                        "type_scores":          type_scores_all,
                        "type_profile_match":   type_profile_match,
                        "arc_id":               None,
                        "arc_roles":            [],
                        "story_beats":          [],
                        "story_graph_summary":  {},
                        "arc_diagnostics":      {},
                        "story_structure": {
                            "has_setup":      story_structure["has_setup"],
                            "has_conflict":   story_structure["has_conflict"],
                            "has_resolution": story_structure["has_resolution"],
                            "setup_evidence":       story_structure.get("setup_evidence",      0.0),
                            "conflict_evidence":    story_structure.get("conflict_evidence",   0.0),
                            "resolution_evidence":  story_structure.get("resolution_evidence", 0.0),
                            "causality_evidence":   story_structure.get("causality_evidence",  False),
                            "temporal_progression_evidence": story_structure.get("temporal_progression_evidence", False),
                            "protagonist_evidence": story_structure.get("protagonist_evidence", False),
                        },
                        "arc_pattern":          arc_pattern,
                        "arc_confidence":       round(arc_confidence, 3),
                        "narrativity_score":    round(narrativity_score, 3),
                        "coherence_score":      quality["arc_coherence"],
                        "payoff_strength":      quality["payoff_strength"],
                        "compression_quality":  quality["compression_quality"],
                        "character_continuity": quality["character_continuity"],
                        "topic_continuity":     quality["topic_continuity"],
                        "story_export_safety":  quality["story_export_safety"],
                        "active_story_failures": quality["active_story_failures"],
                        "subscores":            subscores,
                        "modal_contrib":        subscores,
                        "reasons":              reasons,
                        "has_takeaway":         has_takeaway,
                        "title": title, "summary": summary,
                        "transcript": transcript[:300],
                        "boundary_diagnostics": {},
                        "pipeline":             "window_fallback",
                        "needs_manual_review":  True,  # fallback always needs review
                    }

                    # ── Fallback export gate (always starts at manual_review) ──
                    fallback_gate = story_export_decision(
                        story_export_safety   = quality["story_export_safety"],
                        compression_quality   = quality["compression_quality"],
                        payoff_strength       = quality["payoff_strength"],
                        type_ambiguity        = type_ambiguity,
                        type_profile_match    = type_profile_match,
                        active_story_failures = quality["active_story_failures"],
                        boundary_diagnostics  = {},
                        clip_self_sufficiency = 0.4,
                        candidate_origin      = "window_fallback",
                        arc_completeness      = 0.0,
                    )
                    moment["export_decision"]       = fallback_gate["decision"]
                    moment["export_score"]          = fallback_gate["export_score"]
                    moment["export_reject_reasons"] = fallback_gate["reject_reasons"]
                    moment["auto_export_ok"]        = fallback_gate["auto_export_ok"]

                    all_moments.append(moment)
                    logger.info(
                        f"  ✓ [win↓] {start:.1f}s-{end:.1f}s | {story_type} | "
                        f"score={fallback_score:.3f} (raw={final_score:.3f}) | arc={arc_pattern} | "
                        f"export={fallback_gate['decision']}"
                    )

            except Exception as exc:
                logger.error(f"  Error in window {start:.1f}-{end:.1f}s: {exc}")
                continue

    # =========================================================================
    # POST-PROCESSING: quality-aware re-rank, then top_k
    # =========================================================================

    # Pre-compute best arc-based candidate quality for text-only demotion logic
    _best_arc_export = max(
        (m.get("export_score", 0.0) for m in all_moments
         if m.get("arc_proposed_by_event_layer") or m.get("pipeline") == "event_beat_arc"),
        default=0.0,
    )

    def _quality_rank_key(m: Dict) -> float:
        """
        Composite ranking key — not just score.

        Weights:
            score                 40%  — base model score
            export_score          25%  — production export readiness
            story_export_safety   15%  — arc/quality safety
            clip_self_sufficiency 10%  — how well-packaged the clip is
            arc_event_recall       5%  — multimodal evidence support
            pipeline_bonus         5%  — arc-based candidates preferred

        text_markers_only penalty:
            When a candidate's origin is text_markers_only (no real events,
            only lexical markers) AND there's a comparable arc-based candidate
            with export_score within 0.25 of this one, apply a −0.12 demotion.
            This ensures arc-based moments consistently outrank text-only ones
            when quality is comparable, regardless of raw score.
        """
        raw_score       = float(m.get("score",              0.0))
        export_sc       = float(m.get("export_score",       0.0))
        safety          = float(m.get("story_export_safety",
                                 m.get("coherence_score",   0.0)))
        sufficiency     = float(
            m.get("boundary_diagnostics", {}).get("clip_self_sufficiency", 0.5)
            if m.get("boundary_diagnostics")
            else 0.4
        )
        event_recall    = float(m.get("arc_event_recall",   0.0))
        pipeline_bonus  = 0.05 if m.get("pipeline") == "event_beat_arc" else 0.0

        # Hard penalty: reject should never rank high
        if m.get("export_decision") == "reject":
            return -1.0

        base = float(np.clip(
            0.40 * raw_score +
            0.25 * export_sc +
            0.15 * safety +
            0.10 * sufficiency +
            0.05 * event_recall +
            pipeline_bonus,
            0.0, 1.0,
        ))

        # text_markers_only demotion:
        # When this candidate has no real event support AND an arc-based
        # candidate of comparable quality exists, demote it explicitly.
        is_text_only = (
            not m.get("arc_proposed_by_event_layer", False) and
            m.get("pipeline") != "event_beat_arc" or
            (m.get("pipeline") == "event_beat_arc" and event_recall < 0.15)
        )
        arc_candidate_nearby = _best_arc_export >= (export_sc - 0.25)

        if is_text_only and arc_candidate_nearby and _best_arc_export > 0.0:
            base -= 0.12

        return float(np.clip(base, -1.0, 1.0))

    all_moments.sort(key=_quality_rank_key, reverse=True)

    # Attach rank_score + origin label to each moment for transparency
    for m in all_moments:
        m["rank_score"] = round(_quality_rank_key(m), 4)
        if not m.get("arc_proposed_by_event_layer") and m.get("pipeline") == "event_beat_arc":
            m["arc_origin_label"] = "text_markers_only"
        elif m.get("arc_proposed_by_event_layer"):
            m["arc_origin_label"] = "event_layer"
        else:
            m["arc_origin_label"] = "window_fallback"

    if cfg.use_reranker and cfg.reranker_model_path:
        try:
            all_moments = apply_story_reranker(all_moments, cfg.reranker_model_path, top_k=top_k)
        except Exception as exc:
            logger.warning(f"Re-ranker failed, using baseline sort: {exc}")
            all_moments = all_moments[:top_k]
    else:
        all_moments = all_moments[:top_k]

    # ── Stats ─────────────────────────────────────────────────────────────────
    narrative_score_distribution: Dict[str, int] = {}
    if narrative_scores_list:
        for lo, hi in zip([0.0, 0.2, 0.4, 0.6, 0.8], [0.2, 0.4, 0.6, 0.8, 1.01]):
            key = f"{lo:.1f}-{hi:.1f}" if hi < 1.01 else f"{lo:.1f}-1.0"
            narrative_score_distribution[key] = sum(1 for s in narrative_scores_list if lo <= s < hi)

    arc_pattern_distribution: Dict[str, int] = {}
    for ap in arc_patterns_list:
        arc_pattern_distribution[ap] = arc_pattern_distribution.get(ap, 0) + 1

    # Track whether the top result came from the fallback pipeline
    fallback_origin_top_result = (
        bool(all_moments) and
        all_moments[0].get("pipeline") == "window_fallback"
    )
    export_decision_distribution: Dict[str, int] = {}
    for m in all_moments:
        d = m.get("export_decision", "unknown")
        export_decision_distribution[d] = export_decision_distribution.get(d, 0) + 1

    # Event recall stats
    event_recall_top_result = (
        all_moments[0].get("arc_event_recall", 0.0) if all_moments else 0.0
    )
    arc_proposal_origin_top = (
        "event_layer"     if all_moments and all_moments[0].get("arc_proposed_by_event_layer")
        else "window_fallback" if all_moments and all_moments[0].get("pipeline") == "window_fallback"
        else "text_markers_only"
    )

    # =========================================================================
    # v2.1: WEAK-STORY FALLBACK — если all_moments пуст, но есть weak_story_pool
    # (rejected arcs / no_conflict windows с evidence), промоутируем top-3
    # в manual_review.
    # =========================================================================
    weak_story_fallback_used = False
    if not all_moments and weak_story_pool:
        def _weak_rank(w: Dict) -> float:
            return (
                float(w.get("arc_completeness", 0.0)) * 0.60
                + float(w.get("setup_evidence", 0.0)) * 0.25
                + float(w.get("duration_sec", 0.0)) / 60.0 * 0.15
            )
        top_weak = sorted(weak_story_pool, key=_weak_rank, reverse=True)[:3]
        for idx, wk in enumerate(top_weak):
            ws = float(wk.get("start", 0.0))
            we = float(wk.get("end", ws + cfg.min_clip_duration))
            weak_moment = {
                "start": round(ws, 2),
                "end": round(we, 2),
                "duration": round(max(0.0, we - ws), 2),
                "score": round(_weak_rank(wk), 3),
                "type": "story",
                "story_type": wk.get("story_type", "weak_fragment"),
                "arc_pattern": wk.get("arc_pattern", "flat"),
                "arc_id": wk.get("arc_id"),
                "arc_roles": wk.get("arc_roles", []),
                "reasons": [{
                    "code": "weak_story_manual_review",
                    "message": f"Arc/fragment had signal but failed gate "
                               f"(reject_reason_original={wk.get('reject_reason_original')})",
                    "weight": 0.4,
                }],
                "export_decision": "manual_review",
                "reject_reason_original": wk.get("reject_reason_original"),
                "transcript": wk.get("transcript_preview", ""),
                "summary": wk.get("transcript_preview", "")[:100],
                "title": f"[Manual] {wk.get('story_type', 'Фрагмент')}",
                "pipeline": "weak_story_fallback",
                "is_weak_fallback": True,
                "weak_rank": idx + 1,
                "needs_manual_review": True,
            }
            all_moments.append(weak_moment)
        weak_story_fallback_used = True
        filter_counts["weak_promoted_to_manual_review"] = len(top_weak)
        logger.info(
            f"weak_story_manual_review fallback: promoted top-{len(top_weak)} "
            f"weak stories to manual_review (from {len(weak_story_pool)} in pool)"
        )

    logger.info(f"Result: {len(all_moments)} moment(s) | pipeline={pipeline_mode} | "
                f"filters: {filter_counts} | fallback_top={fallback_origin_top_result} | "
                f"weak_fb={weak_story_fallback_used}")
    logger.info("=" * 70)

    return {
        "mode":          "story",
        "story_moments": all_moments,
        "rejected_arcs":         rejected_arcs,
        "beat_to_arc_trace":     beat_to_arc_trace,
        "weak_story_pool":       weak_story_pool,
        "story_filter_reasons":  filter_counts,
        "weak_story_fallback_used": weak_story_fallback_used,
        "stats": {
            "total_duration":                video_duration_sec,
            "profile_name":                  f"{cfg.mode_name} {cfg.profile_version}",
            "pipeline_mode":                 pipeline_mode,
            "num_windows_analyzed":          len(locals().get("windows", [])),
            "num_stories_found":             len(all_moments),
            "threshold":                     _threshold,
            "min_narrative_threshold":       _min_narrative,
            "avg_score":                     round(float(np.mean([m["score"] for m in all_moments])), 3)
                                             if all_moments else 0.0,
            "story_types": {
                st: sum(1 for m in all_moments if m["story_type"] == st)
                for st in {m["story_type"] for m in all_moments}
            },
            "filter_counts":                 filter_counts,
            "narrative_score_distribution":  narrative_score_distribution,
            "arc_pattern_distribution":      arc_pattern_distribution,
            "fallback_origin_top_result":    fallback_origin_top_result,
            "export_decision_distribution":  export_decision_distribution,
            "event_recall_top_result":       event_recall_top_result,
            "arc_proposal_origin_top":       arc_proposal_origin_top,
            "n_rejected_arcs":               len(rejected_arcs),
            "n_weak_pool":                   len(weak_story_pool),
            "weak_story_fallback_used":      weak_story_fallback_used,
            "short_video_duration_ease":     short_video_duration_ease,
        },
    }


def _empty_story_result(
    video_duration_sec: float,
    error: str,
    cfg: Optional[StoryModeConfig] = None,
) -> Dict:
    """Пустой результат story mode (совместимый формат)."""
    _cfg = cfg or StoryModeConfig()
    return {
        "mode": "story",
        "error": error,
        "story_moments": [],
        "stats": {
            "total_duration": video_duration_sec,
            "profile_name": f"{_cfg.mode_name} {_cfg.profile_version}",
            "num_windows_analyzed": 0,
            "num_stories_found": 0,
            "threshold": _cfg.threshold,
            "min_narrative_threshold": _cfg.min_narrative_threshold,
            "avg_score": 0.0,
            "story_types": {},
            "filter_counts": {},
            "narrative_score_distribution": {},
            "arc_pattern_distribution": {},
        },
    }


def get_story_for_trailer(
    story_result: Dict,
    n: int = 1,
    min_score: float = 0.70,
    min_distance_from_start: float = 30.0,
    min_distance_from_end: float = 30.0,
    prefer_mode: Optional[str] = None,
) -> List[Dict]:
    """
    Выбор story-клипов для trailer.

    Стратегия: score * type_weight → топ, с учётом min_distance от границ видео.
    IoU (пересечение по времени) считается в get_trailer_clips() при NMS.

    Args:
        story_result: результат find_story_moments()
        n: количество клипов
        min_score: минимальный score для включения (по умолчанию строже, чем threshold)
        min_distance_from_start: не брать из первых X секунд
        min_distance_from_end: не брать из последних X секунд
        prefer_mode: если "story" — story-клипы получают приоритет в NMS

    Returns:
        List[Dict] до n клипов для трейлера.
    """
    moments = story_result.get("story_moments") or []
    video_dur = story_result.get("stats", {}).get("total_duration", 0.0)

    candidates = [
        m for m in moments
        if m.get("score", 0) >= min_score
        and m.get("start", 0) > min_distance_from_start
        and m.get("end", 0) < (video_dur - min_distance_from_end if video_dur > 0 else float("inf"))
    ]

    # Сортируем по score * type_weight
    for c in candidates:
        tw = STORY_SEGMENT_TYPES.get(c.get("story_type", ""), {}).get("weight", 1.0)
        c["_trailer_score"] = c["score"] * tw

    candidates.sort(key=lambda x: x["_trailer_score"], reverse=True)

    # Убираем служебное поле перед возвратом
    result = []
    for c in candidates[:n]:
        c = dict(c)
        c.pop("_trailer_score", None)
        result.append(c)

    # Если нет кандидатов с min_score — берём просто топ-n
    if not result:
        result = moments[:n]

    return result


def _iou_time(a: Dict, b: Dict) -> float:
    """IoU по времени между двумя клипами (overlap / union)."""
    inter_start = max(a["start"], b["start"])
    inter_end = min(a["end"], b["end"])
    if inter_end <= inter_start:
        return 0.0
    inter = inter_end - inter_start
    union = (a["end"] - a["start"]) + (b["end"] - b["start"]) - inter
    return inter / union if union > 0 else 0.0


def get_trailer_clips(
    story_result: Optional[Dict] = None,
    viral_result: Optional[Dict] = None,
    educational_result: Optional[Dict] = None,
    n_story: int = 1,
    n_viral: int = 2,
    n_edu: int = 1,
    iou_threshold: float = 0.20,
    mode_priority: Optional[Dict[str, int]] = None,
) -> List[Dict]:
    """
    Объединение клипов из трёх режимов для трейлера с multi-mode NMS.

    NMS (Non-Maximum Suppression) по IoU времени:
    - IoU считается по времени (пересечение / объединение временных отрезков).
    - При пересечении побеждает клип с большим приоритетом мода; при равном — с большим score.
    - По умолчанию: story > educational > viral.
    - Настраивается через mode_priority (например, {"story": 3, "educational": 2, "viral": 1}).

    Returns:
        List[Dict] клипы, отсортированные по start (для монтажа).
    """
    _priority = mode_priority or {"story": 3, "educational": 2, "viral": 1}

    candidates: List[Dict] = []
    story_moments = (story_result or {}).get("story_moments") or []
    viral_moments = (
        (viral_result or {}).get("viral_moments")
        or (viral_result or {}).get("moments") or []
    )
    edu_moments = (educational_result or {}).get("educational_moments") or []

    for m in story_moments[:n_story]:
        c = dict(m)
        c["type"] = "story"
        candidates.append(c)
    for m in viral_moments[:n_viral]:
        c = dict(m)
        c["type"] = "viral"
        candidates.append(c)
    for m in edu_moments[:n_edu]:
        c = dict(m)
        c["type"] = "educational"
        candidates.append(c)

    # NMS: сортируем по (mode_priority, score), убираем пересечения
    candidates.sort(
        key=lambda x: (_priority.get(x.get("type", ""), 0), x.get("score", 0)),
        reverse=True,
    )
    kept: List[Dict] = []
    for clip in candidates:
        overlaps = any(_iou_time(clip, k) > iou_threshold for k in kept)
        if not overlaps:
            kept.append(clip)

    # Сортировка по времени для монтажа
    kept.sort(key=lambda x: x.get("start", 0))
    return kept


# =============================================================================
# UI HINTS — что показывать для Story Mode (визуально отличить от Viral/Edu)
# =============================================================================
# В UI для каждого story-клипа показывать бейджи без изменений в бэкенде:
#
# - story_structure: бейджи Setup / Conflict / Resolution (по полям has_setup, has_conflict, has_resolution)
# - story_type: тип истории — факап / кейс / личная / успех / байка / урок
# - arc_pattern: дуга — man_in_hole, icarus, rags_to_riches и т.д. (см. STORY_ARC_PATTERNS для описаний)
# - has_takeaway: бейдж «Есть вывод» / «Key takeaway»
#
# Пример: бейджи [Setup] [Conflict] [Resolution] [Кейс] [man_in_hole] [Вывод]

UI_STORY_BADGES = {
    "structure": ["Setup", "Conflict", "Resolution"],
    "story_type_labels": {
        "personal_story": "Личная история",
        "client_case": "Кейс",
        "failure_story": "Факап",
        "success_story": "Успех",
        "anecdote": "Байка",
        "lesson_learned": "Урок",
        "non_story": "Не история",
    },
    "arc_labels": {
        "man_in_hole": "Проблема → Решение",
        "icarus": "Успех → Провал",
        "rags_to_riches": "От плохого к хорошему",
        "riches_to_rags": "От хорошего к плохому",
        "cinderella": "Успех → Неудача → Успех",
        "oedipus": "Проблема → Решение → Новая проблема",
        "flat": "Без явной дуги",
    },
    "takeaway_label": "Вывод",
}


# =============================================================================
# FEEDBACK LOOP — логирование и re-ranker
# =============================================================================

DEFAULT_FEEDBACK_PATH = Path(__file__).parent / "story_feedback.json"


def log_story_feedback(
    moment: Dict,
    kept: bool,
    video_id: Optional[str] = None,
    feedback_path: Optional[Path] = None
) -> None:
    """
    Логировать, какой story_moment пользователь оставил (kept=True) или удалил (kept=False) в UI.
    Данные пишутся в JSON для последующего обучения re-ranker (XGBoost по фичам).
    """
    path = Path(feedback_path) if feedback_path is not None else DEFAULT_FEEDBACK_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    
    entry = {
        "video_id": video_id,
        "start": moment.get("start"),
        "end": moment.get("end"),
        "duration": moment.get("duration"),
        "score": moment.get("score"),
        "story_type": moment.get("story_type"),
        "arc_pattern": moment.get("arc_pattern"),
        "narrativity_score": moment.get("narrativity_score"),
        "has_setup": moment.get("story_structure", {}).get("has_setup"),
        "has_conflict": moment.get("story_structure", {}).get("has_conflict"),
        "has_resolution": moment.get("story_structure", {}).get("has_resolution"),
        "has_takeaway": moment.get("has_takeaway"),
        "subscores": moment.get("subscores"),
        "kept": kept,
        "ts": __import__("time").time(),
    }
    
    try:
        rows = []
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                rows = json.load(f)
        rows.append(entry)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
        logger.info(f"Story feedback logged: kept={kept} -> {path}")
    except Exception as e:
        logger.warning(f"Failed to log story feedback: {e}")


def _moment_to_features(m: Dict) -> Dict:
    """Извлечь плоские фичи из story_moment для re-ranker."""
    ss = m.get("subscores") or {}
    return {
        "score": m.get("score", 0),
        "narrative": ss.get("narrative", 0),
        "semantic": ss.get("semantic", 0),
        "audio": ss.get("audio", 0),
        "visual": ss.get("visual", 0),
        "duration": m.get("duration", 0),
        "has_setup": 1 if m.get("story_structure", {}).get("has_setup") else 0,
        "has_conflict": 1 if m.get("story_structure", {}).get("has_conflict") else 0,
        "has_resolution": 1 if m.get("story_structure", {}).get("has_resolution") else 0,
        "has_takeaway": 1 if m.get("has_takeaway") else 0,
        "narrativity_score": m.get("narrativity_score", 0),
        "arc_flat": 1 if m.get("arc_pattern") == "flat" else 0,
        "arc_man_in_hole": 1 if m.get("arc_pattern") == "man_in_hole" else 0,
    }


def train_story_reranker(
    feedback_path: Optional[Path] = None,
    model_path: Optional[Path] = None,
    min_samples: int = 20
) -> Optional[str]:
    """
    Обучить лёгкий re-ranker (XGBoost) по логам feedback: цель — kept.
    Возвращает путь к сохранённой модели или None при ошибке/мало данных.
    """
    path = Path(feedback_path) if feedback_path is not None else DEFAULT_FEEDBACK_PATH
    if not path.exists():
        logger.warning("No feedback file for story re-ranker")
        return None
    
    try:
        import xgboost as xgb
    except ImportError:
        logger.warning("xgboost not installed; pip install xgboost for story re-ranker")
        return None
    
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    
    if len(rows) < min_samples:
        logger.warning(f"Not enough feedback samples: {len(rows)} < {min_samples}")
        return None
    
    feature_keys = [
        "score", "narrative", "semantic", "audio", "visual", "duration",
        "has_setup", "has_conflict", "has_resolution", "has_takeaway",
        "narrativity_score", "arc_flat", "arc_man_in_hole",
    ]
    X = np.array([[r.get(k, 0) for k in feature_keys] for r in rows])
    y = np.array([1 if r.get("kept") else 0 for r in rows], dtype=np.float32)
    
    model = xgb.XGBClassifier(n_estimators=50, max_depth=4, use_label_encoder=False, eval_metric="logloss")
    model.fit(X, y)
    
    out_path = Path(model_path) if model_path is not None else path.parent / "story_reranker.json"
    model.save_model(str(out_path))
    logger.info(f"Story re-ranker trained and saved to {out_path} (samples={len(rows)})")
    return str(out_path)


def apply_story_reranker(
    story_moments: List[Dict],
    model_path: str,
    top_k: Optional[int] = None
) -> List[Dict]:
    """
    Применить обученный re-ranker к story_moments: добавить rerank_score и отсортировать.
    Если top_k задан — вернуть только top_k после переранжирования.
    """
    try:
        import xgboost as xgb
    except ImportError:
        return story_moments
    
    model = xgb.XGBClassifier(use_label_encoder=False)
    model.load_model(model_path)
    
    feature_keys = [
        "score", "narrative", "semantic", "audio", "visual", "duration",
        "has_setup", "has_conflict", "has_resolution", "has_takeaway",
        "narrativity_score", "arc_flat", "arc_man_in_hole",
    ]
    feats = [_moment_to_features(m) for m in story_moments]
    X = np.array([[f.get(k, 0) for k in feature_keys] for f in feats])
    pred = model.predict_proba(X)
    proba_kept = pred[:, 1] if pred.shape[1] > 1 else pred.ravel()
    
    out = []
    for m, p in zip(story_moments, proba_kept):
        m = dict(m)
        m["rerank_score"] = round(float(p), 4)
        out.append(m)
    out.sort(key=lambda x: x["rerank_score"], reverse=True)
    if top_k is not None:
        out = out[:top_k]
    return out


if __name__ == "__main__":
    # Test
    print("Story Mode v1.0 loaded successfully!")
    print(f"Story types: {list(STORY_SEGMENT_TYPES.keys())}")
    print(f"Arc patterns: {list(STORY_ARC_PATTERNS.keys())}")
