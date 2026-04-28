"""
TRAILER MODE v3.0 — Smart Auto-Preview Director

Dual assembly_mode: "trailer" | "preview"

Director pipeline (v3.0):

    run_candidate_pipeline()
        collect_candidates()                ← flatten mode outputs
        compute_slot_eligibility()          ← rule engine: hard + soft constraints
        enrich_candidates()                 ← 4-part spoiler / suitability /
                                               preview_value / premise_value / ending_strength

    refine_preview_boundaries()             ← scene-safe cutting:
                                               shot boundary / ASR sentence / prosodic pause /
                                               cut_naturalness_score / dialogue_cut_safety

    run_theme_pipeline()
        extract_theme_features()            ← per-candidate: speaker / scene / visual / semantic
        compute_theme_similarity()          ← pairwise float similarity matrix
        merge_candidates_into_themes()      ← cluster → ThemeBlock
        build_theme_blocks()                ← orchestrates above 3

    build_transition_graph()
        compute_transition_score()          ← full pairwise: scene + audio + narrative_handoff +
                                               cut compatibility + hard_break_penalty

    run_slot_pipeline()
        build_trailer_plan()                ← SlotPolicy → slots (primary / secondary / filler)
        assign_candidates_to_slots()        ← sequence-aware: rank_candidates_for_slot()
                                               / compute_assignment_utility()
        force_tease_end()                   ← score_tease_end_candidate() multi-fallback

    run_optimizer_pipeline()
        optimize_trailer_plan()
            spoiler_swap_pass()
            monotony_guard_pass()
            theme_diversity_pass()
            transition_quality_pass()
            preview_coherence_pass()        ← NEW: "does the sequence explain the video?"
            ending_sting_pass()             ← NEW: "does the ending invite watching the full video?"
            over_explanation_pass()         ← NEW: "did we become a mini-summary?"
            scene_redundancy_pass()         ← NEW: "are two adjacent clips too similar?"

    run_ui_payload_builder()
        build_ui_payload()
            why_this_transition             ← NEW: per-transition explanation
            spoiler_meter_per_slot          ← NEW: 0-1 for each slot
            preview_intent_per_slot         ← NEW: hook/context/escalation/tease
            replace_with_same_theme         ← NEW edit command
            replace_with_more_visual        ← NEW edit command
            replace_with_more_context       ← NEW edit command
            make_less_spoilery              ← NEW edit command

Public API (backward-compatible with v1, v2, v2.1):
    find_trailer_clips()

New in v3.0:
    result["assembly_mode"]         — "trailer" | "preview"
    result["boundary_diagnostics"]  — per-clip cut quality
    result["preview_intent"]        — per-slot role in preview
    result["transition_graph"]      — enriched with scene/audio/narrative sub-scores
    result["edit_contract"]         — extended edit commands
"""

from __future__ import annotations

import json
import logging
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# POLICY DATACLASSES
# =============================================================================

@dataclass
class ScoringPolicy:
    # Stage 1: suitability
    w_base_score: float = 0.30
    w_hook_strength: float = 0.18
    w_curiosity_gap: float = 0.16
    w_self_containedness: float = 0.14
    w_transition_flexibility: float = 0.08
    w_theme_value: float = 0.06
    w_preview_value: float = 0.08      # NEW: preview-specific quality
    # Stage 2: risk
    w_spoiler_penalty: float = 0.20
    w_context_penalty: float = 0.10
    # Slot fit added to suitability
    w_slot_fit: float = 0.12
    # Multiplier flags
    use_mode_weight: bool = True
    use_energy_bonus: bool = True
    use_dialogue_bonus: bool = True
    use_diversity_bonus: bool = True
    use_length_factor: bool = True


@dataclass
class SpoilerPolicy:
    anti_spoiler_tail_pct: float = 0.10
    max_trailer_spoiler_budget: float = 0.40
    w_outcome: float = 0.40
    w_reveal: float = 0.30
    w_payoff: float = 0.20
    w_explanation: float = 0.10
    spoiler_lexical_extra: List[str] = field(default_factory=list)
    resolution_source_types: List[str] = field(default_factory=lambda: [
        "resolution", "payoff", "conclusion", "final_reveal", "summary",
        "story_resolution", "happy_ending", "downer_ending",
    ])


@dataclass
class ThemePolicy:
    theme_gap_sec: float = 30.0
    min_lexical_overlap_to_merge: float = 0.18
    source_type_match_bonus: float = 0.15
    narrative_match_bonus: float = 0.10
    speaker_match_weight: float = 0.25      # NEW
    scene_match_weight: float = 0.30        # NEW
    semantic_phrase_weight: float = 0.45    # NEW
    min_candidates_per_theme: int = 1
    max_themes: int = 12


@dataclass
class BoundaryPolicy:
    """Controls scene-safe boundary refinement."""
    enabled: bool = True
    snap_asr_before_sec: float = 2.5   # max look-back for sentence start
    snap_asr_after_sec: float = 1.5    # max look-ahead for sentence end
    min_cut_naturalness: float = 0.30  # reject clips that score below this
    penalize_mid_word_cut: bool = True
    penalize_mid_sentence_cut: bool = True
    # v3.1 — action completion & prosodic pause
    motion_window_sec: float = 1.5     # scan window for motion minima near boundary
    motion_silence_thresh: float = 0.20  # movement_intensity below this = "action complete"
    prosodic_gap_min_sec: float = 0.30   # min silence gap between ASR segments to count as pause
    prosodic_snap_window_sec: float = 1.2  # max distance from original boundary to snap to pause
    # snap priority: "best" picks highest-ranked candidate regardless of signal type
    # "asr_first" prefers ASR; "shot_first" prefers shot boundary
    snap_priority: str = "best"  # "best" | "asr_first" | "shot_first"


@dataclass
class SlotPolicy:
    template: str = "youtube_standard"
    assembly_mode: str = "trailer"      # "trailer" | "preview"
    strict_eligibility: bool = True
    fallback_allowed: bool = True
    fallback_penalty: float = 0.15
    max_alternatives_per_slot: int = 3
    tease_end_required: bool = True
    tease_end_fallback_modes: List[str] = field(default_factory=lambda: ["hook", "viral", "story"])
    tease_end_min_curiosity: float = 0.40
    tease_end_max_spoiler: float = 0.45
    # Sequence-awareness
    sequence_aware_assignment: bool = True
    # v3.1: raised from 0.25 → 0.40 so transition quality drives slot ranking, not just breaks ties
    transition_weight_in_assignment: float = 0.40


@dataclass
class OptimizationPolicy:
    enable_spoiler_swap: bool = True
    enable_monotony_guard: bool = True
    enable_theme_diversity: bool = True
    enable_transition_optimization: bool = True
    enable_preview_coherence: bool = True
    enable_ending_sting: bool = True
    enable_over_explanation: bool = True
    enable_scene_redundancy: bool = True
    # v3.1: lowered from 0.35 → 0.28 so more weak transitions get fixed
    transition_quality_floor: float = 0.28
    # v3.1: run transition optimizer twice (before and after other passes)
    transition_optimizer_passes: int = 2
    max_explanation_slots: int = 2          # for over_explanation_pass
    # v3.1: duration-based cap for preview mode (explanation clips as share of total)
    max_explanation_duration_share: float = 0.45
    scene_redundancy_similarity_threshold: float = 0.55
    # v3.1: minimum average transition score required before accepting the plan
    min_avg_transition_score: float = 0.40
    # v3.2 — over_explanation: mixed-video tuning
    # premise/context slots are discounted in explanation burden (keep one “about” beat)
    over_explanation_context_slot_discount: float = 0.62
    # if True, rank swaps by graded burden, not raw trailer_score_final
    over_explanation_use_graded_burden: bool = True
    # v3.2 — scene redundancy multi-signal weights (visual, lexical, scene tags, theme, speaker, mode)
    redundancy_w_visual: float = 0.28
    redundancy_w_lexical: float = 0.22
    redundancy_w_scene: float = 0.18
    redundancy_w_theme: float = 0.14
    redundancy_w_speaker: float = 0.10
    redundancy_w_mode: float = 0.08


# =============================================================================
# CORE DATACLASSES
# =============================================================================

@dataclass
class EligibilityResult:
    eligible: bool
    hard_fail_reasons: List[str] = field(default_factory=list)
    soft_fit: float = 0.0               # 0–1: how well it fits even if eligible
    eligibility_confidence: float = 0.0 # 0–1: how certain the gate is


@dataclass
class TransitionEdge:
    from_candidate_id: str
    to_candidate_id: str
    from_slot_id: str
    to_slot_id: str
    score: float = 0.0
    reasons: List[str] = field(default_factory=list)
    # sub-scores v2.1
    temporal_contrast_score: float = 0.0
    mode_contrast_score: float = 0.0
    topic_continuity_score: float = 0.0
    rhythm_compatibility: float = 0.0
    spoiler_escalation_safe: float = 0.0
    # sub-scores v3.0
    scene_cut_score: float = 0.0        # how clean is the visual cut
    audio_carry_score: float = 0.0      # is audio continuous or jarring
    narrative_handoff_score: float = 0.0  # does story flow make sense
    visual_match_score: float = 0.0     # visual style continuity
    boundary_handoff_score: float = 0.0 # clean entry on B + dialogue safety
    hard_break_penalty: float = 0.0     # mid-sentence / mid-action cut


@dataclass
class TrailerEditCommand:
    action: str
    target_slot_id: Optional[str] = None
    target_theme_id: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    description: str = ""


@dataclass
class TrailerCandidate:
    candidate_id: str
    start: float
    end: float
    duration: float
    source_mode: str
    source_type: str
    title: Optional[str]
    summary: Optional[str]
    reasons: List[Any]
    base_score: float

    # two-stage score
    suitability_score: float = 0.0
    risk_penalty: float = 0.0
    slot_fit_score: float = 0.0
    trailer_score_raw: float = 0.0
    trailer_score_final: float = 0.0

    # thematic assignment
    theme_id: Optional[str] = None
    theme_label: Optional[str] = None
    theme_role: Optional[str] = None

    # suitability signals
    hook_strength: float = 0.0
    curiosity_gap_score: float = 0.0
    context_dependency: float = 0.0
    self_containedness: float = 0.0
    transition_flexibility: float = 0.0
    theme_value: float = 0.0

    # v3 preview signals
    preview_value: float = 0.0          # how useful for preview (not just raw quality)
    premise_value: float = 0.0          # how well it explains topic / context
    ending_strength: float = 0.0        # strength as tease-ending candidate
    assembly_value: float = 0.0         # slot-agnostic usefulness in a sequence

    # 4-part spoiler
    outcome_spoiler_risk: float = 0.0
    explanation_spoiler_risk: float = 0.0
    reveal_spoiler_risk: float = 0.0
    payoff_spoiler_risk: float = 0.0
    spoiler_risk: float = 0.0
    spoiler_category: Optional[str] = None

    # multipliers
    energy_bonus: float = 1.0
    dialogue_bonus: float = 1.0
    diversity_bonus: float = 1.0
    length_factor: float = 1.0

    # semantic enrichment
    semantic_markers: List[str] = field(default_factory=list)
    narrative_role_hint: Optional[str] = None
    topic_confidence: float = 0.0

    # theme features (from extract_theme_features)
    speaker_signature: Optional[str] = None   # speaker id proxy
    scene_signature: List[str] = field(default_factory=list)  # dominant scene labels
    visual_signature: List[str] = field(default_factory=list) # visual feature tokens

    # slot assignment
    eligible_slots: List[str] = field(default_factory=list)
    eligibility_results: Dict[str, EligibilityResult] = field(default_factory=dict)
    assigned_slot: Optional[str] = None
    slot_fit_scores: Dict[str, float] = field(default_factory=dict)
    replacement_candidates: List[str] = field(default_factory=list)
    is_fallback_pick: bool = False
    selection_priority: float = 0.0

    # boundary refinement (from refine_preview_boundaries)
    refined_start: Optional[float] = None
    refined_end: Optional[float] = None
    scene_safe_start: Optional[float] = None
    scene_safe_end: Optional[float] = None
    cut_naturalness_score: float = 0.5
    dialogue_cut_safety: float = 0.5
    scene_cut_safety: float = 0.5
    boundary_diagnostics: Dict[str, Any] = field(default_factory=dict)

    # transition
    transition_out_scores: Dict[str, float] = field(default_factory=dict)
    transition_in_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class ThemeBlock:
    theme_id: str
    label: str
    start: float
    end: float
    importance: float
    spoiler_level: float
    candidate_ids: List[str] = field(default_factory=list)
    role: str = "body"

    # v2.1
    semantic_signature: List[str] = field(default_factory=list)
    dominant_source_modes: List[str] = field(default_factory=list)
    representative_candidate_id: Optional[str] = None
    best_candidate_ids: List[str] = field(default_factory=list)
    alternative_theme_ids: List[str] = field(default_factory=list)
    constructor_label: Optional[str] = None
    replaceable: bool = True

    # v3.0
    speaker_signature: Optional[str] = None  # dominant speaker in cluster
    scene_signature: List[str] = field(default_factory=list)
    visual_signature: List[str] = field(default_factory=list)
    topic_confidence: float = 0.5
    coherence_with_neighbors: float = 0.5   # avg similarity to adjacent themes


@dataclass
class TrailerSlot:
    slot_id: str
    role: str
    min_duration: float
    max_duration: float
    required: bool = True
    preferred_modes: List[str] = field(default_factory=list)
    preferred_source_types: List[str] = field(default_factory=list)
    preferred_theme_roles: List[str] = field(default_factory=list)
    avoid_spoiler_above: float = 0.6
    curiosity_target: float = 0.5
    description: str = ""
    strict_eligibility: bool = True
    fallback_allowed: bool = True
    fallback_penalty: float = 0.15
    slot_group: str = "primary"
    target_share_of_total: float = 0.0
    must_fill_if_possible: bool = False
    # v3: preview intent label
    preview_intent: str = ""  # hook / context / escalation / tease / depth


@dataclass
class SlotAssignment:
    slot: TrailerSlot
    selected: Optional[TrailerCandidate]
    alternatives: List[TrailerCandidate]
    selection_reason: str
    rejection_reasons: Dict[str, str]
    used_fallback: bool = False
    eligibility_fail_count: int = 0


# =============================================================================
# TRAILER MODE CONFIG
# =============================================================================

@dataclass
class TrailerModeConfig:
    mode_name: str = "default"
    profile_version: str = "v3.0"
    locale: str = "ru"
    assembly_mode: str = "trailer"      # "trailer" | "preview"

    target_trailer_duration: float = 90.0
    min_clip_duration: float = 5.0
    max_clip_duration: float = 25.0
    min_gap_between_clips: float = 0.5
    nms_iou_thresh: float = 0.4
    min_base_score: float = 0.3
    max_clips_total: int = 0
    fill_target_tolerance: float = 1.3

    include_educational: bool = True
    include_viral: bool = True

    mode_weights: Dict[str, float] = field(default_factory=lambda: {
        "hook": 1.3, "story": 1.1, "viral": 1.0, "educational": 0.8,
    })

    scoring: ScoringPolicy = field(default_factory=ScoringPolicy)
    spoiler: SpoilerPolicy = field(default_factory=SpoilerPolicy)
    theme: ThemePolicy = field(default_factory=ThemePolicy)
    boundary: BoundaryPolicy = field(default_factory=BoundaryPolicy)
    slot: SlotPolicy = field(default_factory=SlotPolicy)
    optimization: OptimizationPolicy = field(default_factory=OptimizationPolicy)

    enable_ui_payload: bool = True
    enable_transition_graph: bool = True
    enable_boundary_refinement: bool = True

    _BUILTIN_PROFILES: Dict[str, str] = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def for_preview(cls, **kwargs) -> "TrailerModeConfig":
        """Factory for preview-optimised config."""
        cfg = cls(assembly_mode="preview", **kwargs)
        cfg.slot.template = "preview_standard"
        cfg.slot.assembly_mode = "preview"
        cfg.spoiler.max_trailer_spoiler_budget = 0.30
        cfg.slot.tease_end_max_spoiler = 0.35
        cfg.scoring.w_preview_value = 0.14
        cfg.scoring.w_hook_strength = 0.12
        cfg.optimization.enable_over_explanation = True
        cfg.optimization.enable_preview_coherence = True
        return cfg

    @classmethod
    def from_json(cls, path: str) -> "TrailerModeConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        import dataclasses
        known = {f.name for f in dataclasses.fields(cls) if not f.name.startswith("_")}
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in asdict(self).items() if not k.startswith("_")},
                      f, ensure_ascii=False, indent=2)


# =============================================================================
# SLOT TEMPLATES  (trailer + preview)
# =============================================================================

_SLOT_TEMPLATES: Dict[str, List[Dict]] = {
    # ── TRAILER ──────────────────────────────────────────────────────────
    "youtube_standard": [
        {"slot_id": "open_hook",  "role": "open_hook",  "slot_group": "primary",
         "required": True,  "min_duration": 3.0, "max_duration": 10.0,
         "preferred_modes": ["hook", "viral"],
         "preferred_source_types": ["question_hook", "curiosity_hook", "bold_claim",
                                    "contrarian_hook", "mystery_hook"],
         "preferred_theme_roles": ["intro"],
         "avoid_spoiler_above": 0.50, "curiosity_target": 0.65,
         "must_fill_if_possible": True, "preview_intent": "hook",
         "description": "Первые секунды — интрига или неожиданный вопрос"},
        {"slot_id": "premise",     "role": "premise",    "slot_group": "primary",
         "required": True,  "min_duration": 5.0, "max_duration": 15.0,
         "preferred_modes": ["story", "educational"],
         "preferred_source_types": ["setup", "exposition", "explanation", "definition"],
         "preferred_theme_roles": ["setup", "intro"],
         "avoid_spoiler_above": 0.70, "curiosity_target": 0.40,
         "preview_intent": "context",
         "description": "Контекст: о чём это видео"},
        {"slot_id": "escalation",  "role": "escalation", "slot_group": "primary",
         "required": True,  "min_duration": 6.0, "max_duration": 18.0,
         "preferred_modes": ["story", "viral"],
         "preferred_source_types": ["man_in_hole", "conflict", "escalation", "rising_action"],
         "preferred_theme_roles": ["conflict", "escalation"],
         "avoid_spoiler_above": 0.75, "curiosity_target": 0.55,
         "preview_intent": "escalation",
         "description": "Напряжение / нарастание"},
        {"slot_id": "proof_moment","role": "proof_moment","slot_group": "primary",
         "required": False, "min_duration": 4.0, "max_duration": 12.0,
         "preferred_modes": ["viral", "educational", "hook"],
         "preferred_source_types": ["climax", "wow_moment", "viral", "example", "formula"],
         "preferred_theme_roles": ["climax", "payoff"],
         "avoid_spoiler_above": 0.80, "curiosity_target": 0.45,
         "preview_intent": "escalation",
         "description": "Один сильный момент — доказательство ценности"},
        {"slot_id": "tease_end",   "role": "tease_end",  "slot_group": "primary",
         "required": True,  "min_duration": 3.0, "max_duration": 10.0,
         "preferred_modes": ["hook", "story", "viral"],
         "preferred_source_types": ["cliffhanger", "open_loop", "curiosity_hook",
                                    "revelation_tease", "question_hook"],
         "preferred_theme_roles": ["reveal"],
         "avoid_spoiler_above": 0.45, "curiosity_target": 0.70,
         "must_fill_if_possible": True, "preview_intent": "tease",
         "description": "Финал: тизер без ответа"},
        {"slot_id": "supporting_escalation","role": "escalation","slot_group": "secondary",
         "required": False, "min_duration": 5.0, "max_duration": 15.0,
         "preferred_modes": ["story", "viral"],
         "preferred_source_types": ["conflict", "rising_action", "escalation"],
         "avoid_spoiler_above": 0.75, "curiosity_target": 0.50,
         "preview_intent": "escalation",
         "description": "Дополнительный блок нагнетания"},
        {"slot_id": "backup_proof","role": "proof_moment","slot_group": "secondary",
         "required": False, "min_duration": 4.0, "max_duration": 14.0,
         "preferred_modes": ["viral", "educational"],
         "preferred_source_types": ["wow_moment", "example", "step_by_step"],
         "avoid_spoiler_above": 0.75, "curiosity_target": 0.40,
         "preview_intent": "escalation",
         "description": "Резервный доказательный момент"},
        {"slot_id": "micro_tease", "role": "tease_end",  "slot_group": "filler",
         "required": False, "min_duration": 2.0, "max_duration": 7.0,
         "preferred_modes": ["hook", "viral"],
         "preferred_source_types": ["curiosity_hook", "wow_moment"],
         "avoid_spoiler_above": 0.55, "curiosity_target": 0.60,
         "preview_intent": "tease",
         "description": "Мини-тизер для добивки длины"},
    ],
    # ── PREVIEW ──────────────────────────────────────────────────────────
    "preview_standard": [
        {"slot_id": "preview_hook",  "role": "open_hook",  "slot_group": "primary",
         "required": True,  "min_duration": 3.0, "max_duration": 8.0,
         "preferred_modes": ["hook", "viral"],
         "preferred_source_types": ["question_hook", "bold_claim", "wow_moment"],
         "avoid_spoiler_above": 0.40, "curiosity_target": 0.70,
         "must_fill_if_possible": True, "preview_intent": "hook",
         "description": "Мгновенный захват внимания"},
        {"slot_id": "preview_context","role": "premise",   "slot_group": "primary",
         "required": True,  "min_duration": 5.0, "max_duration": 14.0,
         "preferred_modes": ["educational", "story"],
         "preferred_source_types": ["explanation", "setup", "definition"],
         "avoid_spoiler_above": 0.65, "curiosity_target": 0.35,
         "preview_intent": "context",
         "description": "Краткий смысловой контекст"},
        {"slot_id": "preview_value","role": "escalation", "slot_group": "primary",
         "required": True,  "min_duration": 6.0, "max_duration": 16.0,
         "preferred_modes": ["story", "viral", "educational"],
         "preferred_source_types": ["man_in_hole", "example", "climax", "wow_moment"],
         "avoid_spoiler_above": 0.70, "curiosity_target": 0.50,
         "preview_intent": "escalation",
         "description": "Главная ценность видео без разгадки"},
        {"slot_id": "preview_tease","role": "tease_end",  "slot_group": "primary",
         "required": True,  "min_duration": 3.0, "max_duration": 8.0,
         "preferred_modes": ["hook", "story"],
         "preferred_source_types": ["cliffhanger", "curiosity_hook", "open_loop"],
         "avoid_spoiler_above": 0.35, "curiosity_target": 0.75,
         "must_fill_if_possible": True, "preview_intent": "tease",
         "description": "Уход на незакрытом вопросе"},
        {"slot_id": "preview_depth","role": "optional_depth","slot_group": "secondary",
         "required": False, "min_duration": 6.0, "max_duration": 18.0,
         "preferred_modes": ["educational", "story"],
         "preferred_source_types": ["step_by_step", "explanation", "demo"],
         "avoid_spoiler_above": 0.65, "curiosity_target": 0.35,
         "preview_intent": "depth",
         "description": "Дополнительный смысловой блок для длинного формата"},
    ],
    "preview_talking_head": [
        {"slot_id": "th_hook",    "role": "open_hook",   "slot_group": "primary",
         "required": True,  "min_duration": 3.0, "max_duration": 7.0,
         "preferred_modes": ["hook"], "avoid_spoiler_above": 0.40, "curiosity_target": 0.70,
         "must_fill_if_possible": True, "preview_intent": "hook",
         "description": "Открывающий hook спикера"},
        {"slot_id": "th_claim",   "role": "premise",     "slot_group": "primary",
         "required": True,  "min_duration": 5.0, "max_duration": 12.0,
         "preferred_modes": ["story", "educational"],
         "preferred_source_types": ["bold_claim", "explanation", "setup"],
         "avoid_spoiler_above": 0.60, "curiosity_target": 0.40,
         "preview_intent": "context",
         "description": "Главный тезис без доказательства"},
        {"slot_id": "th_tease",   "role": "tease_end",   "slot_group": "primary",
         "required": True,  "min_duration": 3.0, "max_duration": 8.0,
         "preferred_modes": ["hook", "story"],
         "avoid_spoiler_above": 0.35, "curiosity_target": 0.75,
         "must_fill_if_possible": True, "preview_intent": "tease",
         "description": "Финальный обрыв"},
    ],
    "preview_educational": [
        {"slot_id": "edu_hook",   "role": "open_hook",   "slot_group": "primary",
         "required": True,  "min_duration": 3.0, "max_duration": 8.0,
         "preferred_modes": ["hook", "educational"],
         "preferred_source_types": ["question_hook", "bold_claim"],
         "avoid_spoiler_above": 0.45, "curiosity_target": 0.65,
         "must_fill_if_possible": True, "preview_intent": "hook",
         "description": "Образовательный хук — главный вопрос"},
        {"slot_id": "edu_premise","role": "premise",     "slot_group": "primary",
         "required": True,  "min_duration": 7.0, "max_duration": 18.0,
         "preferred_modes": ["educational"],
         "preferred_source_types": ["explanation", "definition", "step_by_step"],
         "avoid_spoiler_above": 0.65, "curiosity_target": 0.35,
         "preview_intent": "context",
         "description": "Объяснение без финального ответа"},
        {"slot_id": "edu_example","role": "proof_moment","slot_group": "primary",
         "required": False, "min_duration": 5.0, "max_duration": 14.0,
         "preferred_modes": ["educational", "viral"],
         "preferred_source_types": ["example", "formula", "demo"],
         "avoid_spoiler_above": 0.70, "curiosity_target": 0.40,
         "preview_intent": "escalation",
         "description": "Конкретный пример / демонстрация"},
        {"slot_id": "edu_tease",  "role": "tease_end",   "slot_group": "primary",
         "required": True,  "min_duration": 3.0, "max_duration": 8.0,
         "preferred_modes": ["hook", "educational"],
         "preferred_source_types": ["cliffhanger", "curiosity_hook"],
         "avoid_spoiler_above": 0.40, "curiosity_target": 0.70,
         "must_fill_if_possible": True, "preview_intent": "tease",
         "description": "Финал: зачем смотреть полное видео"},
    ],
    "short_teaser": [
        {"slot_id": "open_hook","role":"open_hook","slot_group":"primary",
         "required":True,"min_duration":2.0,"max_duration":6.0,
         "preferred_modes":["hook","viral"],"avoid_spoiler_above":0.40,
         "curiosity_target":0.75,"must_fill_if_possible":True,
         "preview_intent":"hook","description":"Мгновенный хук"},
        {"slot_id": "escalation","role":"escalation","slot_group":"primary",
         "required":True,"min_duration":4.0,"max_duration":10.0,
         "preferred_modes":["story","viral"],"avoid_spoiler_above":0.60,
         "curiosity_target":0.55,"preview_intent":"escalation","description":"Нагнетание"},
        {"slot_id": "tease_end","role":"tease_end","slot_group":"primary",
         "required":True,"min_duration":2.0,"max_duration":6.0,
         "preferred_modes":["hook","viral"],"avoid_spoiler_above":0.35,
         "curiosity_target":0.80,"must_fill_if_possible":True,
         "preview_intent":"tease","description":"Обрыв"},
    ],
}


# =============================================================================
# LEXICAL BANKS
# =============================================================================

_SPOILER_OUTCOME = ["в итоге","оказалось","решение найдено","финал","вот почему",
    "итог","вывод","разгадка","in the end","the answer is","it turns out","the result",
    "conclusion","so that's why","the truth is"]
_SPOILER_REVEAL = ["раскрыть секрет","секрет раскрыт","настоящая причина","оказалось что",
    "revealed","the real reason","secret revealed","finally revealed","hidden truth"]
_SPOILER_PAYOFF = ["счастливый конец","всё обошлось","победа","успех","справился",
    "the payoff","it worked","success","happy ending","they won"]
_SPOILER_EXPLANATION = ["и вот как это работает","вот механизм","полное объяснение",
    "and here's how","full explanation"]
_CURIOSITY_LEXICAL = ["а что если","а вы знали","вы не поверите","тайна","секрет",
    "мало кто знает","главный вопрос","это изменит","did you know","what if",
    "nobody talks about","you won't believe","why does","the truth about",
    "most people don't","the hidden"]
_OPEN_LOOP_TYPES = frozenset({"question_hook","curiosity_hook","cliffhanger",
    "revelation_tease","open_loop","man_in_hole","mystery_hook","contrarian_hook"})
_RESOLUTION_TYPES = frozenset({"resolution","payoff","conclusion","final_reveal",
    "summary","story_resolution","happy_ending","downer_ending"})
_REVEAL_TYPES = frozenset({"revelation_tease","final_reveal","plot_twist","reveal"})
_PAYOFF_TYPES = frozenset({"payoff","happy_ending","climax","downer_ending"})
_EXPLANATION_TYPES = frozenset({"explanation","step_by_step","definition","tutorial","demo"})
_CONTEXT_TYPES = frozenset({"setup","exposition","explanation","definition","premise"})


# =============================================================================
# TEMPORAL NMS + UTILITIES
# =============================================================================

def _iou(a: Tuple[float,float], b: Tuple[float,float]) -> float:
    inter = max(0.0, min(a[1],b[1]) - max(a[0],b[0]))
    union = max(a[1],b[1]) - min(a[0],b[0])
    return 0.0 if union <= 0 else inter / union


def _containment(a: Tuple[float,float], b: Tuple[float,float]) -> float:
    """Fraction of the *shorter* clip that is covered by the other clip (0..1)."""
    inter = max(0.0, min(a[1],b[1]) - max(a[0],b[0]))
    shorter = min(a[1]-a[0], b[1]-b[0])
    return 0.0 if shorter <= 0 else inter / shorter


def _is_conflict(
    a: Tuple[float,float],
    b: Tuple[float,float],
    min_overlap_sec: float = 2.0,
    iou_thresh: float = 0.25,
    containment_thresh: float = 0.60,
) -> bool:
    """
    True if clips a and b should NOT coexist in the final sequence.
    Conflict if: overlap > min_overlap_sec AND (iou > iou_thresh OR containment > containment_thresh).
    This catches fully-nested short clips that pure IoU misses.
    """
    inter = max(0.0, min(a[1],b[1]) - max(a[0],b[0]))
    if inter <= min_overlap_sec:
        return False
    return _iou(a, b) > iou_thresh or _containment(a, b) > containment_thresh


def _nms(candidates: List[TrailerCandidate], thresh: float) -> List[TrailerCandidate]:
    s = sorted(candidates, key=lambda c: c.trailer_score_final, reverse=True)
    kept: List[TrailerCandidate] = []
    for c in s:
        if all(_iou((c.start,c.end),(k.start,k.end)) <= thresh for k in kept):
            kept.append(c)
    return kept


def _min_gap_ok(new: TrailerCandidate, existing: List[TrailerCandidate], gap: float) -> bool:
    return all(new.end+gap <= c.start or new.start >= c.end+gap for c in existing)


def _zone(start: float, end: float, dur: float) -> str:
    p = ((start+end)/2.0)/max(dur,1e-6)
    return "early" if p<=1/3 else "middle" if p<=2/3 else "late"


def _text_tokens(c: TrailerCandidate) -> List[str]:
    blob = " ".join(filter(None,[
        c.title or "",
        c.summary or "",
        " ".join(r.get("message","") if isinstance(r,dict) else str(r) for r in c.reasons),
    ]))
    stop = {"и","в","на","с","по","к","за","из","не","а","the","a","of","in",
            "to","and","is","it","for","это","что","как","но"}
    return [t for t in blob.lower().split() if t not in stop and len(t) > 2]


def _lex_score(tokens: List[str], bank: List[str]) -> float:
    text = " ".join(tokens)
    hits = sum(1 for kw in bank if kw in text)
    return float(np.clip(hits / max(len(bank)*0.12, 1.0), 0, 1))


def _jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    return 0.0 if not sa or not sb else len(sa&sb)/len(sa|sb)


def _energy_bonus(c: TrailerCandidate, base_analysis: Optional[Dict], dur: float) -> float:
    if not base_analysis:
        return 1.0
    ts = base_analysis.get("time_series") or {}
    signals = ("emotion_intensity","movement_intensity","audio_energy",
               "music_intensity","loudness_peaks")
    means = []
    for key in signals:
        raw = ts.get(key)
        if not raw or not isinstance(raw,(list,np.ndarray)):
            continue
        arr = np.asarray(raw, dtype=float)
        n = len(arr)
        if n == 0:
            continue
        i0 = max(0,min(int(n*c.start/max(dur,1e-6)),n-1))
        i1 = max(i0+1,min(int(n*c.end/max(dur,1e-6)),n))
        seg = arr[i0:i1]
        if seg.size > 0:
            means.append(float(np.clip(np.nanmean(seg),0,1)))
    return 1.0 if not means else 0.8+0.4*float(np.mean(means))


def _dialogue_bonus(c: TrailerCandidate) -> float:
    for r in c.reasons:
        if isinstance(r,dict):
            for key in ("dialogue_strength","text_score","speech_score","dialogue_score"):
                v = r.get(key)
                if v is not None:
                    try:
                        return float(np.clip(0.85+0.4*float(v),0.85,1.25))
                    except (TypeError,ValueError):
                        pass
    stype = (c.source_type or "").lower()
    if any(kw in stype for kw in ("question","fact","stat","bold_claim","dialogue")):
        return 1.15
    if any(kw in stype for kw in ("hook","man_in_hole","climax")):
        return 1.05
    if len(c.reasons) >= 3:
        return 1.10
    if len(c.reasons) >= 1:
        return 1.02
    return 1.0


# =============================================================================
# STAGE 1a — COLLECT CANDIDATES
# =============================================================================

def collect_candidates(hook_result, story_result, viral_result, educational_result
                       ) -> List[TrailerCandidate]:
    candidates: List[TrailerCandidate] = []
    def _make(m, mode, type_key, score_keys, items_key):
        for result, mk in [(hook_result,"hook"),(story_result,"story"),
                           (viral_result,"viral"),(educational_result,"educational")]:
            pass  # not used here
        start,end = float(m.get("start",0)), float(m.get("end",0))
        if end<=start: return None
        base=0.5
        for k in score_keys:
            v=m.get(k)
            if v is not None:
                try: base=float(v); break
                except: pass
        return TrailerCandidate(
            candidate_id=f"{mode}_{str(uuid.uuid4())[:8]}",
            start=start, end=end,
            duration=float(m.get("duration",end-start)),
            source_mode=mode, source_type=str(m.get(type_key,mode)),
            title=m.get("title"), summary=m.get("summary"),
            reasons=m.get("reasons",[]), base_score=float(np.clip(base,0,1)),
        )
    _pairs = [
        (hook_result,"hook","hook_type",("score","hook_score"),"hook_moments"),
        (story_result,"story","story_type",("score","narrative_score","story_score"),"story_moments"),
        (viral_result,"viral","viral_type",("score","virality_score"),"viral_moments"),
        (educational_result,"educational","segment_type",
         ("score","edu_score","educational_score"),"educational_moments"),
    ]
    for result, mode, type_key, score_keys, mk in _pairs:
        if result is None: continue
        items = result.get(mk, result.get("educational_segments", result.get("viral",[])))
        for m in items:
            c = _make(m, mode, type_key, score_keys, mk)
            if c: candidates.append(c)
    return candidates


# =============================================================================
# STAGE 1b — SLOT ELIGIBILITY RULE ENGINE
# =============================================================================

def compute_slot_eligibility(
    c: TrailerCandidate,
    slots: List[TrailerSlot],
    cfg: TrailerModeConfig,
) -> Dict[str, EligibilityResult]:
    """
    Rule engine for slot eligibility.

    Returns per-slot EligibilityResult with:
    - eligible (bool): hard gates pass
    - hard_fail_reasons: list of blocking rule names
    - soft_fit (float): 0-1 affinity even if technically eligible
    - eligibility_confidence (float): certainty of gate decision
    """
    results: Dict[str, EligibilityResult] = {}

    for slot in slots:
        hard_fails = []
        soft_bonuses = []
        soft_penalties = []

        stype = (c.source_type or "").lower()
        role = slot.role

        # ── Hard gate 1: spoiler ────────────────────────────────────────
        if c.spoiler_risk > slot.avoid_spoiler_above:
            hard_fails.append(
                f"spoiler_risk={c.spoiler_risk:.2f}>limit={slot.avoid_spoiler_above:.2f}"
            )

        # ── Hard gate 2: minimum duration ──────────────────────────────
        if c.duration < slot.min_duration * 0.5:
            hard_fails.append(f"too_short: {c.duration:.1f}s < {slot.min_duration*0.5:.1f}s")

        # ── Hard gate 3: role-specific hard constraints ─────────────────
        if role == "open_hook":
            if c.self_containedness < 0.30:
                hard_fails.append(f"open_hook_requires_self_containedness≥0.30: got {c.self_containedness:.2f}")
            if c.outcome_spoiler_risk > 0.35:
                hard_fails.append(f"open_hook_outcome_spoiler={c.outcome_spoiler_risk:.2f}")

        if role == "tease_end":
            if c.outcome_spoiler_risk > 0.40:
                hard_fails.append(f"tease_end_outcome_spoiler={c.outcome_spoiler_risk:.2f}")
            if c.payoff_spoiler_risk > 0.35:
                hard_fails.append(f"tease_end_payoff_spoiler={c.payoff_spoiler_risk:.2f}")
            # must NOT be a closed outcome
            if any(rt in stype for rt in _RESOLUTION_TYPES):
                hard_fails.append("tease_end_cannot_be_resolution_type")

        if role == "premise":
            # needs to have some contextual signal
            has_context = (
                c.context_dependency > 0.10 or
                any(ct in stype for ct in _CONTEXT_TYPES) or
                c.source_mode == "educational"
            )
            if not has_context:
                hard_fails.append("premise_needs_contextual_signal")

        # ── Soft fit scoring ────────────────────────────────────────────
        if c.source_mode in slot.preferred_modes:
            soft_bonuses.append(0.25)
        if any(pt in stype for pt in slot.preferred_source_types):
            soft_bonuses.append(0.20)
        if c.theme_role and c.theme_role in slot.preferred_theme_roles:
            soft_bonuses.append(0.15)

        # curiosity gap alignment
        gap_delta = c.curiosity_gap_score - slot.curiosity_target
        soft_bonuses.append(float(np.clip(gap_delta * 0.15, -0.10, 0.10)))

        # narrative role match
        if role == "open_hook" and c.narrative_role_hint in ("intro", "tease"):
            soft_bonuses.append(0.10)
        elif role == "escalation" and c.narrative_role_hint == "conflict":
            soft_bonuses.append(0.10)
        elif role == "tease_end" and c.narrative_role_hint == "tease":
            soft_bonuses.append(0.15)
        elif role == "premise" and c.narrative_role_hint == "intro":
            soft_bonuses.append(0.10)

        # duration fit
        if slot.min_duration <= c.duration <= slot.max_duration:
            soft_bonuses.append(0.10)
        elif c.duration > slot.max_duration:
            soft_penalties.append(0.08)

        soft_fit = float(np.clip(0.50 + sum(soft_bonuses) - sum(soft_penalties), 0, 1))
        eligible = len(hard_fails) == 0
        confidence = 0.90 if hard_fails else float(np.clip(soft_fit * 1.1, 0.50, 0.95))

        results[slot.slot_id] = EligibilityResult(
            eligible=eligible,
            hard_fail_reasons=hard_fails,
            soft_fit=soft_fit,
            eligibility_confidence=confidence,
        )

    return results


# =============================================================================
# STAGE 2 — ENRICH CANDIDATES
# =============================================================================

def _composite_spoiler(c, video_dur, policy, tokens):
    def pos_risk():
        if policy.anti_spoiler_tail_pct <= 0: return 0.0
        pos = ((c.start+c.end)/2.0)/max(video_dur,1e-6)
        tail = 1.0 - policy.anti_spoiler_tail_pct
        if pos < tail: return 0.0
        return float(np.clip((pos-tail)/max(policy.anti_spoiler_tail_pct,1e-6)*0.50,0,0.50))

    stype = (c.source_type or "").lower()

    # outcome
    type_r = 0.50 if any(r in stype for r in policy.resolution_source_types) else 0.0
    c.outcome_spoiler_risk = float(np.clip(
        type_r*0.60 + _lex_score(tokens,_SPOILER_OUTCOME)*0.40 + pos_risk(), 0, 1))

    # reveal
    c.reveal_spoiler_risk = float(np.clip(
        (0.55 if any(r in stype for r in _REVEAL_TYPES) else 0)*0.65
        + _lex_score(tokens,_SPOILER_REVEAL)*0.35, 0, 1))

    # payoff
    c.payoff_spoiler_risk = float(np.clip(
        (0.45 if any(p in stype for p in _PAYOFF_TYPES) else 0)*0.60
        + _lex_score(tokens,_SPOILER_PAYOFF)*0.40, 0, 1))

    # explanation
    c.explanation_spoiler_risk = float(np.clip(
        (0.30 if any(e in stype for e in _EXPLANATION_TYPES) else 0)*0.55
        + _lex_score(tokens,_SPOILER_EXPLANATION)*0.45, 0, 1))

    composite = (policy.w_outcome*c.outcome_spoiler_risk
                 + policy.w_reveal*c.reveal_spoiler_risk
                 + policy.w_payoff*c.payoff_spoiler_risk
                 + policy.w_explanation*c.explanation_spoiler_risk)

    subtypes = {"outcome":c.outcome_spoiler_risk,"reveal":c.reveal_spoiler_risk,
                "payoff":c.payoff_spoiler_risk,"explanation":c.explanation_spoiler_risk}
    c.spoiler_category = max(subtypes,key=subtypes.get) if max(subtypes.values())>0.2 else None
    return float(np.clip(composite,0,1))


def _narrative_hint(c):
    stype=(c.source_type or "").lower()
    if any(k in stype for k in ("intro","setup","exposition")): return "intro"
    if any(k in stype for k in ("man_in_hole","conflict","escalation","rising_action")): return "conflict"
    if any(k in stype for k in ("climax","payoff","wow_moment")): return "climax"
    if any(k in stype for k in _RESOLUTION_TYPES): return "resolution"
    if any(k in stype for k in _OPEN_LOOP_TYPES): return "tease"
    return None


def _compute_preview_value(c: TrailerCandidate) -> float:
    """
    Preview-specific value: how well this clip contributes to a preview
    (not just raw quality — also coherence, non-spoiliness, intrigue).
    """
    val = (
        0.30 * c.curiosity_gap_score
        + 0.25 * c.self_containedness
        + 0.20 * (1.0 - c.spoiler_risk)
        + 0.15 * c.hook_strength
        + 0.10 * c.transition_flexibility
    )
    return float(np.clip(val, 0, 1))


def _compute_premise_value(c: TrailerCandidate) -> float:
    """How well this clip explains context without spoiling."""
    stype=(c.source_type or "").lower()
    val = 0.0
    if any(ct in stype for ct in _CONTEXT_TYPES): val += 0.40
    if c.source_mode == "educational": val += 0.20
    val += 0.20 * c.self_containedness
    val += 0.20 * (1.0 - c.outcome_spoiler_risk)
    return float(np.clip(val, 0, 1))


def _compute_ending_strength(c: TrailerCandidate) -> float:
    """How strong this candidate is as a tease-ending."""
    val = (
        0.40 * c.curiosity_gap_score
        + 0.30 * (1.0 - c.outcome_spoiler_risk)
        + 0.20 * (1.0 - c.payoff_spoiler_risk)
        + 0.10 * c.self_containedness
    )
    # bonus for open-loop types
    stype=(c.source_type or "").lower()
    if any(olt in stype for olt in _OPEN_LOOP_TYPES): val += 0.15
    return float(np.clip(val, 0, 1))


def enrich_candidates(candidates, video_dur, cfg, base_analysis, slots):
    zone_counts: Dict[str,int] = defaultdict(int)
    for c in candidates:
        zone_counts[_zone(c.start,c.end,video_dur)] += 1
    total_zones = max(1,sum(zone_counts.values()))

    for c in candidates:
        tokens = _text_tokens(c)
        c.spoiler_risk = _composite_spoiler(c, video_dur, cfg.spoiler, tokens)
        c.semantic_markers = [t for t in dict.fromkeys(tokens) if len(t)>=4][:12]
        c.narrative_role_hint = _narrative_hint(c)

        stype=(c.source_type or "").lower()
        high_dep = {"resolution","payoff","commentary","follow_up","step_by_step"}
        low_dep = {"question_hook","bold_claim","curiosity_hook","viral","wow_moment"}
        dep = (0.40 if any(t in stype for t in high_dep) else 0)
        dep -= (0.20 if any(t in stype for t in low_dep) else 0)
        if c.duration>20: dep-=0.15
        elif c.duration<8: dep+=0.10
        if c.source_mode=="educational": dep+=0.15
        c.context_dependency = float(np.clip(dep,0,1))

        c.self_containedness = float(np.clip(
            1.0-c.context_dependency
            +(0.05 if c.title else 0)
            +(0.05 if c.summary and len(c.summary)>20 else 0)
            +(0.10 if c.source_mode in("hook","viral") else 0),
            0,1))

        c.curiosity_gap_score = float(np.clip(
            (0.50 if any(olt in stype for olt in _OPEN_LOOP_TYPES) else 0)
            + 0.30*_lex_score(tokens,_CURIOSITY_LEXICAL)
            + (0.20 if c.source_mode=="hook" else 0)
            - (c.spoiler_risk*0.30 if c.spoiler_risk>0.5 else 0),
            0,1))

        c.hook_strength = float(np.clip(
            c.curiosity_gap_score*0.60
            +(0.25 if c.source_mode=="hook" else 0.15 if c.source_mode=="viral" else 0)
            +c.base_score*0.15, 0,1))

        tf_dur = 1.0 if 6<=c.duration<=20 else (c.duration/6 if c.duration<6 else max(0.5,20/c.duration))
        c.transition_flexibility = float(np.clip(tf_dur*0.60+(1.0-c.context_dependency)*0.40,0,1))

        # preview-specific signals (computed after base signals)
        c.preview_value = _compute_preview_value(c)
        c.premise_value = _compute_premise_value(c)
        c.ending_strength = _compute_ending_strength(c)
        c.assembly_value = float(np.clip(
            0.35*c.preview_value + 0.30*c.self_containedness
            + 0.20*c.transition_flexibility + 0.15*(1-c.spoiler_risk), 0,1))

        c.energy_bonus = _energy_bonus(c, base_analysis, video_dur)
        c.dialogue_bonus = _dialogue_bonus(c)
        share = zone_counts[_zone(c.start,c.end,video_dur)]/total_zones
        c.diversity_bonus = float(np.clip(1.3-share,0.7,1.3))
        c.length_factor = 1.0 if c.duration<=cfg.max_clip_duration else max(0.5,cfg.max_clip_duration/max(c.duration,1e-6))

        # compute slot eligibility rule engine
        c.eligibility_results = compute_slot_eligibility(c, slots, cfg)
        c.eligible_slots = [sid for sid,er in c.eligibility_results.items() if er.eligible]

    return candidates


# =============================================================================
# STAGE 2b — REFINE PREVIEW BOUNDARIES
# =============================================================================

def _motion_minima_near(
    t: float,
    window: float,
    motion_arr: np.ndarray,
    video_dur: float,
    thresh: float,
) -> List[float]:
    """
    Return timestamps near `t` (±window) where movement_intensity is below `thresh`.
    Used to find "action-complete" safe cut points.
    """
    if motion_arr is None or len(motion_arr) == 0:
        return []
    n = len(motion_arr)
    fps_proxy = n / max(video_dur, 1e-6)   # frames per second (proxy)
    i_low = max(0, int((t - window) * fps_proxy))
    i_high = min(n - 1, int((t + window) * fps_proxy))
    if i_low >= i_high:
        return []
    seg = motion_arr[i_low:i_high+1]
    below_mask = seg < thresh
    if not np.any(below_mask):
        return []
    # return timestamps of all below-threshold frames, sorted by |ts - t|
    rel_idx = np.where(below_mask)[0]
    timestamps = [(i_low + ri) / fps_proxy for ri in rel_idx]
    return sorted(timestamps, key=lambda ts: abs(ts - t))


def _prosodic_pauses_near(
    t: float,
    window: float,
    asr_starts: np.ndarray,
    asr_ends: np.ndarray,
    min_gap: float,
) -> List[float]:
    """
    Return midpoints of silence gaps between consecutive ASR segments
    that fall within ±window of `t` and are longer than `min_gap`.
    """
    pauses = []
    for i in range(len(asr_starts) - 1):
        gap_start = float(asr_ends[i])
        gap_end = float(asr_starts[i + 1])
        gap_dur = gap_end - gap_start
        if gap_dur >= min_gap:
            pause_mid = (gap_start + gap_end) / 2.0
            if abs(pause_mid - t) <= window:
                pauses.append(pause_mid)
    return sorted(pauses, key=lambda p: abs(p - t))


def _best_snap(
    original: float,
    candidates_with_source: List[Tuple[float, str]],  # (timestamp, source_label)
    priority: str,
    max_shift: float,
) -> Tuple[float, str]:
    """
    Pick best snap point from a list of (timestamp, source_label) candidates.

    priority="best"       → pick closest (any source)
    priority="asr_first"  → rank ASR candidates first, then others by proximity
    priority="shot_first" → rank shot candidates first, then others

    Returns (chosen_timestamp, source_label).
    """
    if not candidates_with_source:
        return original, "original"
    # filter to max_shift window
    valid = [(ts, src) for ts, src in candidates_with_source if abs(ts - original) <= max_shift]
    if not valid:
        return original, "original"

    if priority == "asr_first":
        asr = [(ts, src) for ts, src in valid if "asr" in src]
        others = [(ts, src) for ts, src in valid if "asr" not in src]
        ranked = sorted(asr, key=lambda x: abs(x[0]-original)) + sorted(others, key=lambda x: abs(x[0]-original))
    elif priority == "shot_first":
        shot = [(ts, src) for ts, src in valid if "shot" in src]
        others = [(ts, src) for ts, src in valid if "shot" not in src]
        ranked = sorted(shot, key=lambda x: abs(x[0]-original)) + sorted(others, key=lambda x: abs(x[0]-original))
    else:  # "best"
        # give a bonus to "asr+shot" coincidence — best of all worlds
        coincident = [
            (ts, src) for ts, src in valid
            if "asr" in src and "shot" in src
        ]
        if coincident:
            ranked = sorted(coincident, key=lambda x: abs(x[0]-original))
        else:
            ranked = sorted(valid, key=lambda x: abs(x[0]-original))

    return ranked[0]


def refine_preview_boundaries(
    candidates: List[TrailerCandidate],
    asr_segments: Optional[List[Dict]],
    base_analysis: Optional[Dict],
    cfg: TrailerModeConfig,
) -> List[TrailerCandidate]:
    """
    Montage-safe boundary refinement for each candidate.

    Snap priority (each boundary gathers all candidate points, then picks best):
    ┌──────────────────────────────────────────┬────────────────────┐
    │  Signal                                  │  Source label      │
    ├──────────────────────────────────────────┼────────────────────┤
    │  ASR sentence start/end                  │  "asr"             │
    │  Shot boundary                           │  "shot"            │
    │  ASR sentence end coinciding shot bound. │  "asr+shot"        │
    │  Prosodic pause (silence gap ≥ 0.3s)     │  "pause"           │
    │  Visual action complete (motion minimum) │  "motion"          │
    └──────────────────────────────────────────┴────────────────────┘

    Scores computed:
    - dialogue_cut_safety      — not mid-word after snapping
    - scene_cut_safety         — proximity to shot boundary
    - visual_action_completion — how well we land on a motion minimum
    - prosodic_cut_safety      — landed at or near a silence gap
    - cut_naturalness_score    — weighted composite of all four
    """
    if not cfg.enable_boundary_refinement:
        for c in candidates:
            c.refined_start = c.start
            c.refined_end = c.end
        return candidates

    bp = cfg.boundary
    asr_segs = asr_segments or []

    asr_starts = np.array([s.get("start", 0.0) for s in asr_segs], dtype=float)
    asr_ends = np.array([s.get("end", 0.0) for s in asr_segs], dtype=float)

    # shot boundaries
    shot_boundaries: List[float] = []
    if base_analysis:
        for shot in base_analysis.get("shot_segments", []):
            shot_boundaries.append(float(shot.get("start", 0)))
            shot_boundaries.append(float(shot.get("end", 0)))
    shot_boundaries = sorted(set(shot_boundaries))
    shot_arr = np.array(shot_boundaries) if shot_boundaries else np.array([], dtype=float)

    # motion intensity array for action-completion detection
    motion_arr: Optional[np.ndarray] = None
    video_dur_proxy = 1.0
    if base_analysis:
        ts = base_analysis.get("time_series") or {}
        raw = ts.get("movement_intensity") or ts.get("motion_intensity")
        if raw and isinstance(raw, (list, np.ndarray)):
            motion_arr = np.asarray(raw, dtype=float)
            # infer video duration from the array length + last clip end
            if len(candidates) > 0:
                video_dur_proxy = max(c.end for c in candidates)

    for c in candidates:
        diag: Dict[str, Any] = {}

        # ── Build snap-candidate pools for start and end ─────────────────
        start_cands: List[Tuple[float, str]] = []
        end_cands: List[Tuple[float, str]] = []
        snap_win_start = bp.snap_asr_before_sec + 1.0
        snap_win_end = bp.snap_asr_after_sec + 1.0

        # ASR sentence boundaries
        if len(asr_starts) > 0:
            near_s = asr_starts[
                (asr_starts >= c.start - bp.snap_asr_before_sec)
                & (asr_starts <= c.start + 1.0)
            ]
            for ts_v in near_s:
                lbl = "asr"
                # mark as coincident if also near a shot boundary
                if len(shot_arr) > 0 and np.min(np.abs(shot_arr - ts_v)) < 0.25:
                    lbl = "asr+shot"
                start_cands.append((float(ts_v), lbl))

        if len(asr_ends) > 0:
            near_e = asr_ends[
                (asr_ends >= c.end - 0.5)
                & (asr_ends <= c.end + bp.snap_asr_after_sec)
            ]
            for ts_v in near_e:
                lbl = "asr"
                if len(shot_arr) > 0 and np.min(np.abs(shot_arr - ts_v)) < 0.25:
                    lbl = "asr+shot"
                end_cands.append((float(ts_v), lbl))

        # shot boundaries
        if len(shot_arr) > 0:
            near_ss = shot_arr[
                (shot_arr >= c.start - 1.0) & (shot_arr <= c.start + 2.0)
            ]
            for ts_v in near_ss:
                if not any(s == "asr+shot" and abs(cv - ts_v) < 0.1
                           for cv, s in start_cands):
                    start_cands.append((float(ts_v), "shot"))

            near_se = shot_arr[
                (shot_arr >= c.end - 0.5) & (shot_arr <= c.end + 1.5)
            ]
            for ts_v in near_se:
                if not any(s == "asr+shot" and abs(cv - ts_v) < 0.1
                           for cv, s in end_cands):
                    end_cands.append((float(ts_v), "shot"))

        # prosodic pauses (silence gaps between ASR segments)
        for pause_ts in _prosodic_pauses_near(
            c.start, bp.prosodic_snap_window_sec,
            asr_starts, asr_ends, bp.prosodic_gap_min_sec
        ):
            start_cands.append((pause_ts, "pause"))
        for pause_ts in _prosodic_pauses_near(
            c.end, bp.prosodic_snap_window_sec,
            asr_starts, asr_ends, bp.prosodic_gap_min_sec
        ):
            end_cands.append((pause_ts, "pause"))

        # visual action completion (motion minima)
        if motion_arr is not None:
            for m_ts in _motion_minima_near(
                c.start, bp.motion_window_sec, motion_arr,
                video_dur_proxy, bp.motion_silence_thresh
            )[:3]:
                start_cands.append((m_ts, "motion"))
            for m_ts in _motion_minima_near(
                c.end, bp.motion_window_sec, motion_arr,
                video_dur_proxy, bp.motion_silence_thresh
            )[:3]:
                end_cands.append((m_ts, "motion"))

        # ── Pick best snap point ─────────────────────────────────────────
        new_start, start_src = _best_snap(c.start, start_cands, bp.snap_priority, snap_win_start)
        new_end, end_src = _best_snap(c.end, end_cands, bp.snap_priority, snap_win_end)

        # guard: don't shrink clip below half its original duration
        min_dur = max(cfg.min_clip_duration * 0.5, c.duration * 0.50)
        if new_end - new_start < min_dur:
            new_start = c.start
            new_end = c.end
            start_src = "original"
            end_src = "original"
            diag["snap_rejected_too_short"] = True

        if start_src != "original":
            diag[f"start_snapped_to_{start_src}"] = round(new_start, 3)
        if end_src != "original":
            diag[f"end_snapped_to_{end_src}"] = round(new_end, 3)

        # ── Score all four cut-quality axes ──────────────────────────────

        # 1. dialogue_cut_safety: are we still mid-sentence after snapping?
        dialogue_cut_safety = 1.0
        if len(asr_starts) > 0 and len(asr_ends) > 0:
            mid_start = any(
                s_s < new_start < s_e
                for s_s, s_e in zip(asr_starts, asr_ends)
            )
            mid_end = any(
                s_s < new_end < s_e
                for s_s, s_e in zip(asr_starts, asr_ends)
            )
            if mid_start: dialogue_cut_safety -= 0.35
            if mid_end: dialogue_cut_safety -= 0.25
        dialogue_cut_safety = float(np.clip(dialogue_cut_safety, 0, 1))

        # 2. scene_cut_safety: proximity to shot boundary
        scene_cut_safety = 0.5
        if len(shot_arr) > 0:
            dist_s = float(np.min(np.abs(shot_arr - new_start)))
            dist_e = float(np.min(np.abs(shot_arr - new_end)))
            scene_cut_safety = float(np.clip(1.0 - (dist_s + dist_e) / 3.0, 0, 1))

        # 3. visual_action_completion: did we land on a motion minimum?
        visual_action_completion = 0.5
        if motion_arr is not None:
            n = len(motion_arr)
            fps_p = n / max(video_dur_proxy, 1e-6)
            def _motion_at(t):
                idx = int(np.clip(t * fps_p, 0, n - 1))
                return float(motion_arr[idx])
            m_start = _motion_at(new_start)
            m_end = _motion_at(new_end)
            # lower motion = better action completion
            visual_action_completion = float(np.clip(
                1.0 - (m_start + m_end) / 2.0, 0, 1
            ))
            diag["motion_at_start"] = round(m_start, 3)
            diag["motion_at_end"] = round(m_end, 3)

        # 4. prosodic_cut_safety: did end land near a silence gap?
        prosodic_cut_safety = 0.5
        pauses_at_end = _prosodic_pauses_near(
            new_end, bp.prosodic_snap_window_sec,
            asr_starts, asr_ends, bp.prosodic_gap_min_sec
        )
        if pauses_at_end:
            dist_pause = abs(pauses_at_end[0] - new_end)
            prosodic_cut_safety = float(np.clip(
                1.0 - dist_pause / max(bp.prosodic_snap_window_sec, 1e-6), 0, 1
            ))

        # composite naturalness — weighted by source labels
        src_bonus = 0.0
        if "asr+shot" in (start_src, end_src): src_bonus += 0.10
        elif "asr" in (start_src, end_src): src_bonus += 0.05
        elif "shot" in (start_src, end_src): src_bonus += 0.03

        cut_naturalness = float(np.clip(
            0.30 * dialogue_cut_safety
            + 0.25 * scene_cut_safety
            + 0.20 * visual_action_completion
            + 0.15 * prosodic_cut_safety
            + src_bonus,
            0, 1,
        ))

        c.refined_start = round(new_start, 2)
        c.refined_end = round(new_end, 2)
        c.scene_safe_start = c.refined_start
        c.scene_safe_end = c.refined_end
        c.cut_naturalness_score = round(cut_naturalness, 3)
        c.dialogue_cut_safety = round(dialogue_cut_safety, 3)
        c.scene_cut_safety = round(scene_cut_safety, 3)
        c.boundary_diagnostics = {
            **diag,
            "visual_action_completion": round(visual_action_completion, 3),
            "prosodic_cut_safety": round(prosodic_cut_safety, 3),
            "start_src": start_src,
            "end_src": end_src,
        }

    return candidates


# =============================================================================
# STAGE 3 — THEME BUILDING (extract_theme_features + compute_theme_similarity)
# =============================================================================

def extract_theme_features(
    c: TrailerCandidate,
    base_analysis: Optional[Dict] = None,
    asr_segments: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """
    Extract per-candidate theme features covering the FULL clip duration
    (not just the midpoint ±5s window used in v3.0).

    Signals extracted:
    - speaker_signature   — dominant speaker type over the full clip
    - scene_signature     — top-5 scene tags by frequency across all frames
    - visual_signature    — top-5 detection labels by frequency
    - dominant_emotion    — top emotion tag if available
    - location_tag        — indoor/outdoor/studio proxy from scene_tags
    - semantic_tokens     — from existing semantic_markers
    """
    tokens = c.semantic_markers
    speaker_sig = "unknown"
    scene_sig: List[str] = []
    visual_sig: List[str] = []
    dominant_emotion: Optional[str] = None
    location_tag: Optional[str] = None

    if base_analysis:
        frame_events = base_analysis.get("frame_events", [])
        # v3.1: scan ALL frames within [start, end] instead of midpoint ±5s
        clip_frames = [
            e for e in frame_events
            if c.start <= e.get("ts", -1) <= c.end
        ]
        # fallback: if no events in range, widen to ±3s around midpoint
        if not clip_frames:
            mid = (c.start + c.end) / 2.0
            clip_frames = [e for e in frame_events if abs(e.get("ts", 0) - mid) < 3.0]

        if clip_frames:
            n_frames = len(clip_frames)

            # scene tags — aggregate weighted by frame count
            tag_counts: Counter = Counter()
            for e in clip_frames:
                for tag, val in e.get("scene_tags", {}).items():
                    if isinstance(val, (int, float)) and val > 0.45:
                        tag_counts[tag] += 1
            scene_sig = [t for t, _ in tag_counts.most_common(6)]

            # visual detections
            det_counts: Counter = Counter()
            for e in clip_frames:
                for d in e.get("detections", []):
                    if d.get("conf", 0) > 0.45:
                        det_counts[d.get("label", "")] += 1
            visual_sig = [lbl for lbl, _ in det_counts.most_common(6) if lbl]

            # speaker — use frame majority vote across full clip
            single_frames = sum(
                1 for e in clip_frames
                if e.get("scene_tags", {}).get("single_speaker", 0) > 0.55
            )
            multi_frames = sum(
                1 for e in clip_frames
                if e.get("scene_tags", {}).get("multi_person", 0) > 0.55
            )
            no_person_frames = sum(
                1 for e in clip_frames
                if e.get("scene_tags", {}).get("no_person", 0) > 0.60
            )
            if single_frames > n_frames * 0.50:
                speaker_sig = f"single_speaker_{c.source_mode}"
            elif multi_frames > n_frames * 0.40:
                speaker_sig = "multi_speaker"
            elif no_person_frames > n_frames * 0.60:
                speaker_sig = "no_person"

            # dominant emotion
            emotion_counts: Counter = Counter()
            for e in clip_frames:
                for em, v in e.get("emotion_tags", {}).items():
                    if isinstance(v, (int, float)) and v > 0.50:
                        emotion_counts[em] += 1
            if emotion_counts:
                dominant_emotion = emotion_counts.most_common(1)[0][0]

            # location proxy
            indoor = tag_counts.get("indoor", 0)
            outdoor = tag_counts.get("outdoor", 0)
            studio = tag_counts.get("studio", 0)
            if studio > n_frames * 0.30:
                location_tag = "studio"
            elif indoor > outdoor:
                location_tag = "indoor"
            elif outdoor > indoor:
                location_tag = "outdoor"

    return {
        "speaker_signature": speaker_sig,
        "scene_signature": scene_sig,
        "visual_signature": visual_sig,
        "dominant_emotion": dominant_emotion,
        "location_tag": location_tag,
        "semantic_tokens": tokens,
    }


def compute_theme_similarity(
    a: TrailerCandidate,
    b: TrailerCandidate,
    policy: ThemePolicy,
) -> float:
    """
    Multi-signal pairwise similarity for theme clustering.

    v3.1 additions:
    - temporal proximity bonus: clips within theme_gap/2 get a +0.10 boost
    - location / emotion match: same indoor/outdoor/studio or same dominant emotion
    - scene overlap weighted by detection confidence (via visual_signature richer in v3.1)

    Returns 0–1 float.
    """
    # semantic token overlap
    lex = _jaccard(a.semantic_markers, b.semantic_markers)
    sem_score = lex * policy.semantic_phrase_weight

    # speaker continuity
    spk_score = 0.0
    if (a.speaker_signature and b.speaker_signature
            and a.speaker_signature != "unknown"
            and b.speaker_signature != "unknown"):
        if a.speaker_signature == b.speaker_signature:
            spk_score = policy.speaker_match_weight
        elif (a.speaker_signature.startswith("single") and
              b.speaker_signature.startswith("single")):
            spk_score = policy.speaker_match_weight * 0.50

    # scene/visual continuity
    scene_j = _jaccard(a.scene_signature, b.scene_signature)
    visual_j = _jaccard(a.visual_signature, b.visual_signature)
    scene_score = (scene_j * 0.60 + visual_j * 0.40) * policy.scene_match_weight

    # source_type similarity
    type_bonus = policy.source_type_match_bonus if a.source_type == b.source_type else 0.0

    # narrative role match
    narr_bonus = policy.narrative_match_bonus if (
        a.narrative_role_hint and a.narrative_role_hint == b.narrative_role_hint
    ) else 0.0

    # v3.1: temporal proximity bonus
    gap = abs(b.start - a.end) if b.start >= a.end else abs(a.start - b.end)
    temporal_bonus = 0.10 if gap <= policy.theme_gap_sec * 0.50 else 0.0

    # v3.1: location match (from boundary diagnostics or scene_sig heuristic)
    location_a = _infer_location(a)
    location_b = _infer_location(b)
    location_bonus = 0.05 if (location_a and location_b and location_a == location_b) else 0.0

    total = (sem_score + spk_score + scene_score + type_bonus
             + narr_bonus + temporal_bonus + location_bonus)
    return float(np.clip(total, 0, 1))


def _infer_location(c: TrailerCandidate) -> Optional[str]:
    """Heuristic: extract location from scene_signature tokens."""
    for tag in c.scene_signature:
        if tag in ("studio", "indoor", "outdoor", "nature", "office", "street"):
            return tag
    # fallback from boundary_diagnostics if extract_theme_features stored it
    return c.boundary_diagnostics.get("location_tag") if c.boundary_diagnostics else None


def merge_candidates_into_themes(
    candidates: List[TrailerCandidate],
    policy: ThemePolicy,
    video_dur: float,
) -> List[List[TrailerCandidate]]:
    """
    Build clusters using temporal gap + semantic similarity.
    """
    if not candidates:
        return []
    sorted_c = sorted(candidates, key=lambda x: x.start)
    clusters: List[List[TrailerCandidate]] = [[sorted_c[0]]]

    for c in sorted_c[1:]:
        cluster = clusters[-1]
        cluster_end = max(x.end for x in cluster)
        temporal_ok = (c.start - cluster_end) <= policy.theme_gap_sec
        semantic_ok = any(
            compute_theme_similarity(c, x, policy) >= policy.min_lexical_overlap_to_merge
            for x in cluster
        )
        # merge if temporally close OR semantically similar (within extended window)
        if temporal_ok or (semantic_ok and c.start - cluster_end <= policy.theme_gap_sec * 2.0):
            clusters[-1].append(c)
        else:
            clusters.append([c])

    return clusters


def _intra_cluster_coherence(cluster: List[TrailerCandidate], policy: ThemePolicy) -> float:
    """
    Average pairwise similarity of all candidates within a cluster.
    Used as topic_confidence: a tightly coherent cluster scores high,
    a loose 'miscellaneous' cluster scores low.
    """
    if len(cluster) <= 1:
        return 1.0  # single-item cluster is perfectly coherent by definition
    pairs = [
        compute_theme_similarity(cluster[i], cluster[j], policy)
        for i in range(len(cluster))
        for j in range(i + 1, len(cluster))
    ]
    return float(np.mean(pairs)) if pairs else 0.5


def build_theme_blocks(
    candidates: List[TrailerCandidate],
    video_dur: float,
    cfg: TrailerModeConfig,
    base_analysis: Optional[Dict] = None,
    asr_segments: Optional[List[Dict]] = None,
) -> List[ThemeBlock]:
    """
    Full theme pipeline: extract features → similarity → cluster → ThemeBlock.

    v3.1 changes:
    - Full-clip feature extraction (not midpoint ±5s)
    - topic_confidence = intra-cluster pairwise coherence (not avg base_score)
    - Dominant emotion + location stored in ThemeBlock constructor_label
    - coherence_with_neighbors uses multi-signal similarity, not just Jaccard
    """
    if not candidates:
        return []

    # extract theme features into candidates
    for c in candidates:
        feats = extract_theme_features(c, base_analysis, asr_segments)
        c.speaker_signature = feats["speaker_signature"]
        c.scene_signature = feats["scene_signature"]
        c.visual_signature = feats["visual_signature"]
        # store extra signals in boundary_diagnostics for downstream use
        if feats.get("dominant_emotion") or feats.get("location_tag"):
            c.boundary_diagnostics = {
                **c.boundary_diagnostics,
                "dominant_emotion": feats.get("dominant_emotion"),
                "location_tag": feats.get("location_tag"),
            }

    clusters = merge_candidates_into_themes(candidates, cfg.theme, video_dur)

    theme_blocks: List[ThemeBlock] = []
    for i, cluster in enumerate(clusters):
        t_start = min(c.start for c in cluster)
        t_end = max(c.end for c in cluster)
        center_norm = ((t_start + t_end) / 2.0) / max(video_dur, 1e-6)
        importance = float(np.mean([c.base_score for c in cluster]))
        spoiler_level = float(np.mean([c.spoiler_risk for c in cluster]))

        role = ("intro" if center_norm <= 0.20 else "setup" if center_norm <= 0.45
                else "climax" if center_norm <= 0.70 else "reveal" if center_norm <= 0.88
                else "conclusion")

        # aggregate signatures
        all_tokens: List[str] = []
        for c in cluster: all_tokens.extend(c.semantic_markers)
        semantic_sig = [t for t, _ in Counter(all_tokens).most_common(8)]

        all_scenes: List[str] = []
        for c in cluster: all_scenes.extend(c.scene_signature)
        scene_sig = [t for t, _ in Counter(all_scenes).most_common(5)]

        all_visual: List[str] = []
        for c in cluster: all_visual.extend(c.visual_signature)
        visual_sig = [t for t, _ in Counter(all_visual).most_common(5)]

        spk_counter = Counter(c.speaker_signature for c in cluster if c.speaker_signature)
        dom_speaker = spk_counter.most_common(1)[0][0] if spk_counter else None

        # v3.1: dominant emotion + location
        emo_counter = Counter(
            c.boundary_diagnostics.get("dominant_emotion")
            for c in cluster
            if c.boundary_diagnostics and c.boundary_diagnostics.get("dominant_emotion")
        )
        dom_emotion = emo_counter.most_common(1)[0][0] if emo_counter else None

        loc_counter = Counter(
            c.boundary_diagnostics.get("location_tag")
            for c in cluster
            if c.boundary_diagnostics and c.boundary_diagnostics.get("location_tag")
        )
        dom_location = loc_counter.most_common(1)[0][0] if loc_counter else None

        dominant_modes = [m for m, _ in Counter(c.source_mode for c in cluster).most_common(3)]
        top3 = sorted(cluster, key=lambda c: c.base_score, reverse=True)[:3]

        theme_id = f"theme_{i:02d}"
        label = f"Theme {i+1} — {role}"
        parts = semantic_sig[:3]
        mode_label = "/".join(dominant_modes[:2]) if dominant_modes else "mixed"
        extra_ctx = " ".join(filter(None, [dom_emotion, dom_location]))
        constructor_label = (
            f"[{role}] {' · '.join(parts)} ({mode_label})"
            + (f" [{extra_ctx}]" if extra_ctx else "")
            if parts else f"[{role}] ({mode_label})"
        )

        # v3.1: topic_confidence = intra-cluster pairwise coherence
        topic_confidence = _intra_cluster_coherence(cluster, cfg.theme)

        block = ThemeBlock(
            theme_id=theme_id, label=label, start=t_start, end=t_end,
            importance=importance, spoiler_level=spoiler_level,
            candidate_ids=[c.candidate_id for c in cluster],
            role=role, semantic_signature=semantic_sig,
            dominant_source_modes=dominant_modes,
            representative_candidate_id=top3[0].candidate_id if top3 else None,
            best_candidate_ids=[c.candidate_id for c in top3],
            constructor_label=constructor_label, replaceable=spoiler_level < 0.75,
            speaker_signature=dom_speaker,
            scene_signature=scene_sig, visual_signature=visual_sig,
            topic_confidence=topic_confidence,
        )
        theme_blocks.append(block)

        for c in cluster:
            c.theme_id = theme_id
            c.theme_label = label
            c.theme_role = role
            c.topic_confidence = topic_confidence   # intra-cluster coherence, not raw importance
            c.theme_value = importance              # importance stays base_score average

    # cross-reference alternative themes
    for bi in theme_blocks:
        for bj in theme_blocks:
            if bi.theme_id == bj.theme_id: continue
            sim = _jaccard(bi.semantic_signature, bj.semantic_signature)
            if sim >= 0.20 and bj.theme_id not in bi.alternative_theme_ids:
                bi.alternative_theme_ids.append(bj.theme_id)

    # coherence_with_neighbors: multi-signal, not just Jaccard of semantic tokens
    for i, b in enumerate(theme_blocks):
        sims = []
        if i > 0:
            prev_b = theme_blocks[i - 1]
            lex_s = _jaccard(b.semantic_signature, prev_b.semantic_signature)
            vis_s = _jaccard(b.visual_signature, prev_b.visual_signature)
            sims.append(lex_s * 0.60 + vis_s * 0.40)
        if i < len(theme_blocks) - 1:
            next_b = theme_blocks[i + 1]
            lex_s = _jaccard(b.semantic_signature, next_b.semantic_signature)
            vis_s = _jaccard(b.visual_signature, next_b.visual_signature)
            sims.append(lex_s * 0.60 + vis_s * 0.40)
        b.coherence_with_neighbors = float(np.mean(sims)) if sims else 0.5

    logger.info(
        f"ThemeBlocks: {len(theme_blocks)} from {len(candidates)} candidates "
        f"(avg topic_confidence="
        f"{float(np.mean([b.topic_confidence for b in theme_blocks])):.2f})"
    )
    return theme_blocks


# =============================================================================
# STAGE 4 — SLOT PLAN
# =============================================================================

def build_trailer_plan(cfg: TrailerModeConfig) -> List[TrailerSlot]:
    template_key = cfg.slot.template
    defs = _SLOT_TEMPLATES.get(template_key, _SLOT_TEMPLATES["youtube_standard"])
    slots = []
    for d in defs:
        slots.append(TrailerSlot(
            slot_id=d["slot_id"], role=d["role"],
            min_duration=float(d.get("min_duration",5.0)),
            max_duration=float(d.get("max_duration",20.0)),
            required=bool(d.get("required",True)),
            preferred_modes=list(d.get("preferred_modes",[])),
            preferred_source_types=list(d.get("preferred_source_types",[])),
            preferred_theme_roles=list(d.get("preferred_theme_roles",[])),
            avoid_spoiler_above=float(d.get("avoid_spoiler_above",0.6)),
            curiosity_target=float(d.get("curiosity_target",0.4)),
            description=d.get("description",""),
            strict_eligibility=cfg.slot.strict_eligibility,
            fallback_allowed=cfg.slot.fallback_allowed,
            fallback_penalty=cfg.slot.fallback_penalty,
            slot_group=d.get("slot_group","primary"),
            target_share_of_total=float(d.get("target_share_of_total",0.0)),
            must_fill_if_possible=bool(d.get("must_fill_if_possible",False)),
            preview_intent=d.get("preview_intent",""),
        ))
    return slots


# =============================================================================
# STAGE 5 — TWO-STAGE SCORING  (preview-aware)
# =============================================================================

def _score_for_slot(c: TrailerCandidate, slot: TrailerSlot) -> float:
    if c.spoiler_risk > slot.avoid_spoiler_above: return -1.0
    if c.duration < slot.min_duration*0.5 or c.duration > slot.max_duration*3.0: return -1.0
    s = 0.5
    if c.source_mode in slot.preferred_modes: s += 0.20
    stype=(c.source_type or "").lower()
    if any(pt in stype for pt in slot.preferred_source_types): s += 0.15
    if c.theme_role and c.theme_role in slot.preferred_theme_roles: s += 0.10
    s += 0.08 * float(np.clip(c.curiosity_gap_score - slot.curiosity_target, -1, 0.5))
    # bonus for role-specific signals
    if slot.role == "tease_end": s += 0.12 * c.ending_strength
    if slot.role == "premise":   s += 0.10 * c.premise_value
    if slot.role == "open_hook": s += 0.10 * c.hook_strength
    return float(np.clip(s, 0, 1))


def compute_trailer_score(
    c: TrailerCandidate,
    cfg: TrailerModeConfig,
    video_dur: float,
    slot: Optional[TrailerSlot] = None,
) -> TrailerCandidate:
    """
    Two-stage preview-aware scorer.

    Stage 1 — Suitability:
        base_score + hook_strength + curiosity_gap + self_containedness +
        transition_flexibility + theme_value + preview_value

    Stage 2 — Risk-adjusted:
        suitability + slot_fit - spoiler_penalty - context_penalty

    Final = (raw) × multipliers
    """
    sp = cfg.scoring
    mw = cfg.mode_weights.get(c.source_mode, 1.0)

    # Stage 1
    c.suitability_score = float(np.clip(
        sp.w_base_score * c.base_score
        + sp.w_hook_strength * c.hook_strength
        + sp.w_curiosity_gap * c.curiosity_gap_score
        + sp.w_self_containedness * c.self_containedness
        + sp.w_transition_flexibility * c.transition_flexibility
        + sp.w_theme_value * c.theme_value
        + sp.w_preview_value * c.preview_value,
        0, 1,
    ))

    # slot fit
    if slot is not None:
        c.slot_fit_scores[slot.slot_id] = _score_for_slot(c, slot)
        c.slot_fit_score = c.slot_fit_scores[slot.slot_id]
    else:
        c.slot_fit_score = 0.0

    # Stage 2: risk
    c.risk_penalty = float(np.clip(
        sp.w_spoiler_penalty * c.spoiler_risk
        + sp.w_context_penalty * c.context_dependency,
        0, 0.80,
    ))

    raw = c.suitability_score + sp.w_slot_fit * c.slot_fit_score - c.risk_penalty
    c.trailer_score_raw = float(raw)

    mult = mw
    if sp.use_energy_bonus: mult *= c.energy_bonus
    if sp.use_dialogue_bonus: mult *= c.dialogue_bonus
    if sp.use_diversity_bonus: mult *= c.diversity_bonus
    if sp.use_length_factor: mult *= c.length_factor

    c.trailer_score_final = float(np.clip(raw * mult, 0, 2.0))
    return c


# =============================================================================
# STAGE 6 — FULL TRANSITION GRAPH
# =============================================================================

def compute_audio_transition(
    a: TrailerCandidate,
    b: TrailerCandidate,
    base_analysis: Optional[Dict],
    video_dur: float = 0.0,
) -> float:
    """
    Compare loudness at the tail of clip A vs head of clip B.
    Uses video_dur to map time→bin (fixes v3 bug that used b.end in denominator).
    Averages a few bins at each boundary for stability.
    """
    if not base_analysis:
        return 0.5
    ts = base_analysis.get("time_series", {})
    audio_arr = ts.get("audio_energy")
    if not audio_arr or not isinstance(audio_arr, (list, np.ndarray)):
        return 0.5
    arr = np.asarray(audio_arr, dtype=float)
    n = len(arr)
    if n == 0:
        return 0.5
    dur = max(video_dur, a.end, b.start, b.end, 1.0)

    def _mean_near(t_sec: float, backward: bool, span_sec: float) -> float:
        """Mean of bins in [t-span, t] if backward else [t, t+span]."""
        span = max(span_sec / dur * n, 1.0)
        center = int(np.clip((t_sec / dur) * n, 0, n - 1))
        if backward:
            i0 = max(0, int(center - span))
            i1 = center + 1
        else:
            i0 = center
            i1 = min(n, int(center + span) + 1)
        seg = arr[i0:i1]
        return float(np.nanmean(seg)) if seg.size > 0 else float(arr[center])

    e_a = _mean_near(a.end, backward=True, span_sec=0.35)
    e_b = _mean_near(b.start, backward=False, span_sec=0.35)
    jump = abs(e_b - e_a)
    return float(np.clip(1.0 - jump * 1.5, 0, 1))


def compute_scene_transition(a: TrailerCandidate, b: TrailerCandidate,
                              base_analysis: Optional[Dict]) -> float:
    if not base_analysis: return 0.5
    shots = base_analysis.get("shot_segments", [])
    # check if there's a shot boundary between a.end and b.start
    has_boundary = any(
        a.end - 1.0 <= s.get("start", 0) <= b.start + 1.0
        for s in shots
    )
    visual_j = _jaccard(a.visual_signature, b.visual_signature)
    base = 0.70 if has_boundary else 0.30
    return float(np.clip(base * 0.60 + visual_j * 0.40, 0, 1))


def compute_narrative_handoff(a: TrailerCandidate, b: TrailerCandidate) -> float:
    """
    Does the story logic flow from a to b?

    Narrative order: intro → conflict → climax → tease (ideal for trailer)
    """
    order = {"intro": 0, "conflict": 1, "climax": 2, "tease": 3, "resolution": 4}
    ra = order.get(a.narrative_role_hint or "", 2)
    rb = order.get(b.narrative_role_hint or "", 2)
    if rb >= ra:
        return 0.85  # forward flow
    if rb == ra:
        return 0.60  # same level
    return float(np.clip(0.40 - (ra - rb) * 0.10, 0.1, 0.5))  # backward flow penalty


def compute_transition_score(
    a: TrailerCandidate, b: TrailerCandidate,
    from_slot_id: str, to_slot_id: str,
    base_analysis: Optional[Dict] = None,
    video_dur: float = 0.0,
) -> TransitionEdge:
    """
    Full pairwise transition score. v3.2: audio uses real timeline; visual_match
    and entry-boundary quality enter the composite (were partly unused before).
    """
    vd = max(video_dur, a.end, b.end, 1.0)
    gap = b.start - a.end
    temporal_contrast = float(np.clip(gap / max(a.duration, 1e-6), 0, 1))
    mode_contrast = 0.0 if a.source_mode == b.source_mode else 0.80
    topic_continuity = 0.70 if a.theme_id == b.theme_id else 0.40
    dur_ratio = min(a.duration, b.duration) / max(max(a.duration, b.duration), 1e-6)
    rhythm_compat = float(np.clip(1.0 - abs(dur_ratio - 0.6), 0, 1))
    spoiler_safe = float(np.clip(1.0 - abs(b.spoiler_risk - a.spoiler_risk) * 1.5, 0, 1))

    scene_cut = compute_scene_transition(a, b, base_analysis)
    audio_carry = compute_audio_transition(a, b, base_analysis, vd)
    narrative_handoff = compute_narrative_handoff(a, b)
    visual_match = _jaccard(a.visual_signature, b.visual_signature)
    # Entry feels like a deliberate cut: clean in-point on B + exit on A
    boundary_handoff = float(np.clip(
        0.55 * b.cut_naturalness_score + 0.45 * b.dialogue_cut_safety, 0, 1
    ))
    # hard_break_penalty: if b starts mid-sentence / bad exit from a
    hard_break = 0.0
    if b.dialogue_cut_safety < 0.5: hard_break += 0.20
    if a.scene_cut_safety < 0.4: hard_break += 0.10
    if a.cut_naturalness_score < 0.35: hard_break += 0.06

    # v3.2 weights: scene/audio/narrative/boundary/visual drive “feels edited”
    score = (
        0.08 * temporal_contrast
        + 0.08 * mode_contrast
        + 0.08 * topic_continuity
        + 0.06 * rhythm_compat
        + 0.08 * spoiler_safe
        + 0.18 * scene_cut
        + 0.16 * audio_carry
        + 0.14 * narrative_handoff
        + 0.08 * visual_match
        + 0.06 * boundary_handoff
        - hard_break
    )

    reasons = []
    if mode_contrast > 0.5: reasons.append(f"mode_variety:{a.source_mode}→{b.source_mode}")
    if scene_cut > 0.7: reasons.append("clean_scene_cut")
    if audio_carry < 0.4: reasons.append("audio_jump")
    if narrative_handoff > 0.7: reasons.append("narrative_flow")
    if visual_match > 0.55: reasons.append("visual_continuity")
    if boundary_handoff > 0.72: reasons.append("clean_entry_boundary")
    if hard_break > 0.15: reasons.append("hard_break_penalty")

    return TransitionEdge(
        from_candidate_id=a.candidate_id, to_candidate_id=b.candidate_id,
        from_slot_id=from_slot_id, to_slot_id=to_slot_id,
        score=round(float(np.clip(score, 0, 1)), 4), reasons=reasons,
        temporal_contrast_score=round(temporal_contrast,4),
        mode_contrast_score=round(mode_contrast,4),
        topic_continuity_score=round(topic_continuity,4),
        rhythm_compatibility=round(rhythm_compat,4),
        spoiler_escalation_safe=round(spoiler_safe,4),
        scene_cut_score=round(scene_cut,4),
        audio_carry_score=round(audio_carry,4),
        narrative_handoff_score=round(narrative_handoff,4),
        visual_match_score=round(visual_match,4),
        boundary_handoff_score=round(boundary_handoff,4),
        hard_break_penalty=round(hard_break,4),
    )


def build_transition_graph(
    assignments: List[SlotAssignment],
    base_analysis: Optional[Dict] = None,
    video_dur: float = 0.0,
) -> List[TransitionEdge]:
    filled = sorted(
        [(a.slot, a.selected) for a in assignments if a.selected],
        key=lambda x: x[1].start,
    )
    edges: List[TransitionEdge] = []
    for i in range(len(filled)-1):
        sa, ca = filled[i]
        sb, cb = filled[i+1]
        edge = compute_transition_score(
            ca, cb, sa.slot_id, sb.slot_id, base_analysis, video_dur
        )
        edges.append(edge)
        ca.transition_out_scores[cb.candidate_id] = edge.score
        cb.transition_in_scores[ca.candidate_id] = edge.score
    return edges


# =============================================================================
# STAGE 6b — TRANSITION MATRIX PRECOMPUTE
# =============================================================================

def precompute_transition_matrix(
    candidates: List[TrailerCandidate],
    base_analysis: Optional[Dict] = None,
    top_n: int = 60,
    video_dur: float = 0.0,
) -> Dict[Tuple[str, str], float]:
    """
    Pre-compute pairwise transition scores for top_n candidates
    so that compute_assignment_utility() can look them up in O(1)
    instead of recomputing them for every slot × candidate pair.

    Returns dict keyed by (from_candidate_id, to_candidate_id) → score.
    """
    # limit to top-N by base_score to avoid O(N²) blowup
    pool = sorted(candidates, key=lambda c: c.base_score, reverse=True)[:top_n]
    matrix: Dict[Tuple[str, str], float] = {}
    for a in pool:
        for b in pool:
            if a.candidate_id == b.candidate_id:
                continue
            edge = compute_transition_score(a, b, "", "", base_analysis, video_dur)
            matrix[(a.candidate_id, b.candidate_id)] = edge.score
    logger.info(
        f"Transition matrix precomputed: {len(pool)} candidates, "
        f"{len(matrix)} pairs"
    )
    return matrix


# =============================================================================
# STAGE 7 — SEQUENCE-AWARE SLOT ASSIGNMENT
# =============================================================================

def compute_assignment_utility(
    c: TrailerCandidate,
    slot: TrailerSlot,
    partial_plan: List[Optional[TrailerCandidate]],
    cfg: TrailerModeConfig,
    base_analysis: Optional[Dict] = None,
    transition_matrix: Optional[Dict[Tuple[str, str], float]] = None,
    video_dur: float = 0.0,
) -> float:
    """
    Slot fit + expected transition quality with previous filled candidate.

    transition_matrix (optional pre-computed dict) is used for O(1) lookup
    instead of re-computing the pairwise score on every call.
    """
    fit = _score_for_slot(c, slot)
    if fit < 0: return -1.0

    # expected transition with last filled slot
    prev = next((p for p in reversed(partial_plan) if p is not None), None)
    transition_component = 0.5  # neutral if no previous
    if prev is not None:
        if transition_matrix is not None:
            key = (prev.candidate_id, c.candidate_id)
            transition_component = transition_matrix.get(key, 0.5)
        else:
            edge = compute_transition_score(
                prev, c, "", slot.slot_id, base_analysis, video_dur
            )
            transition_component = edge.score

    tw = cfg.slot.transition_weight_in_assignment
    utility = (1.0 - tw) * fit + tw * transition_component

    # theme diversity: penalise if same theme as any already used
    used_themes = {p.theme_id for p in partial_plan if p is not None and p.theme_id}
    if c.theme_id and c.theme_id in used_themes:
        utility -= 0.12

    return float(np.clip(utility, 0, 1))


def rank_candidates_for_slot(
    slot: TrailerSlot,
    candidates: List[TrailerCandidate],
    used_ids: set,
    used_theme_ids: set,
    partial_plan: List[Optional[TrailerCandidate]],
    cfg: TrailerModeConfig,
    base_analysis: Optional[Dict] = None,
    transition_matrix: Optional[Dict[Tuple[str, str], float]] = None,
    video_dur: float = 0.0,
) -> List[Tuple[float, TrailerCandidate]]:
    ranked = []
    for c in candidates:
        if c.candidate_id in used_ids: continue
        if slot.strict_eligibility and slot.slot_id not in c.eligible_slots: continue
        utility = compute_assignment_utility(
            c, slot, partial_plan, cfg, base_analysis, transition_matrix, video_dur
        )
        if utility < 0: continue
        ranked.append((utility, c))
    ranked.sort(key=lambda x: x[0], reverse=True)
    return ranked


def assign_candidates_to_slots(
    candidates: List[TrailerCandidate],
    slots: List[TrailerSlot],
    cfg: TrailerModeConfig,
    base_analysis: Optional[Dict] = None,
    transition_matrix: Optional[Dict[Tuple[str, str], float]] = None,
    video_dur: float = 0.0,
) -> List[SlotAssignment]:
    assignments: List[SlotAssignment] = []
    used_ids: set = set()
    used_theme_ids: set = set()
    partial_plan: List[Optional[TrailerCandidate]] = []

    for slot in slots:
        rejection_reasons: Dict[str,str] = {}
        ineligible_pool = [c for c in candidates
                           if c.candidate_id not in used_ids
                           and slot.slot_id not in c.eligible_slots]

        ranked = rank_candidates_for_slot(
            slot, candidates, used_ids, used_theme_ids, partial_plan,
            cfg, base_analysis, transition_matrix, video_dur
        )
        used_fallback = False
        eligibility_fail = len(ineligible_pool)

        # fallback path
        if not ranked and slot.required and slot.fallback_allowed:
            fallback_ranked = []
            for c in ineligible_pool:
                utility = compute_assignment_utility(
                    c, slot, partial_plan, cfg, base_analysis, transition_matrix, video_dur
                )
                if utility >= 0:
                    fallback_ranked.append((utility - slot.fallback_penalty, c))
            fallback_ranked.sort(key=lambda x: x[0], reverse=True)
            if fallback_ranked:
                ranked = fallback_ranked
                used_fallback = True

        for c in candidates:
            if c.candidate_id in used_ids:
                rejection_reasons[c.candidate_id] = "already_assigned"
            elif slot.strict_eligibility and slot.slot_id not in c.eligible_slots:
                er = c.eligibility_results.get(slot.slot_id)
                if er:
                    rejection_reasons[c.candidate_id] = "; ".join(er.hard_fail_reasons)

        selected: Optional[TrailerCandidate] = None
        alternatives: List[TrailerCandidate] = []
        sel_reason = "no_eligible_candidate"

        if ranked:
            _, best = ranked[0]
            best = compute_trailer_score(best, cfg, 0, slot)
            selected = best
            selected.is_fallback_pick = used_fallback
            selected.assigned_slot = slot.slot_id
            selected.selection_priority = ranked[0][0]
            used_ids.add(selected.candidate_id)
            if selected.theme_id: used_theme_ids.add(selected.theme_id)
            sel_reason = (
                f"utility={ranked[0][0]:.3f} "
                f"mode={selected.source_mode} type={selected.source_type}"
                + (" [FALLBACK]" if used_fallback else "")
            )
            for _, alt in ranked[1:cfg.slot.max_alternatives_per_slot+1]:
                alternatives.append(alt)

        partial_plan.append(selected)
        assignments.append(SlotAssignment(
            slot=slot, selected=selected, alternatives=alternatives,
            selection_reason=sel_reason, rejection_reasons=rejection_reasons,
            used_fallback=used_fallback, eligibility_fail_count=eligibility_fail,
        ))

    return assignments


# =============================================================================
# STAGE 7b — FORCE TEASE END  (score_tease_end_candidate)
# =============================================================================

def score_tease_end_candidate(c: TrailerCandidate, cfg: TrailerModeConfig) -> float:
    """Composite score specifically for tease_end suitability."""
    sp = cfg.slot
    if c.outcome_spoiler_risk > sp.tease_end_max_spoiler: return -1.0
    if c.payoff_spoiler_risk > sp.tease_end_max_spoiler + 0.05: return -1.0
    # require some minimum self-containedness (can't be fully context-dependent)
    if c.context_dependency > 0.70: return -0.5
    # naturalness if boundary refinement ran
    if c.cut_naturalness_score < 0.25: return -0.5

    score = (
        0.40 * c.ending_strength
        + 0.25 * c.curiosity_gap_score
        + 0.20 * (1.0 - c.outcome_spoiler_risk)
        + 0.10 * c.self_containedness
        + 0.05 * c.cut_naturalness_score
    )
    return float(np.clip(score, 0, 1))


def force_tease_end(
    assignments: List[SlotAssignment],
    candidates: List[TrailerCandidate],
    cfg: TrailerModeConfig,
) -> List[SlotAssignment]:
    tease_assn = next((a for a in assignments if "tease" in a.slot.slot_id
                       and a.slot.must_fill_if_possible), None)
    if tease_assn is None or tease_assn.selected is not None:
        return assignments

    used_ids = {a.selected.candidate_id for a in assignments if a.selected}
    unassigned = [c for c in candidates if c.candidate_id not in used_ids]

    # pass 1: unassigned with proper score
    scored = [(score_tease_end_candidate(c, cfg), c)
              for c in unassigned]
    scored = [(s,c) for s,c in scored if s > 0]
    scored.sort(key=lambda x: x[0], reverse=True)

    if scored:
        winner = scored[0][1]
        winner.assigned_slot = tease_assn.slot.slot_id
        tease_assn.selected = winner
        tease_assn.selection_reason = f"force_tease_end: score={scored[0][0]:.3f}"
        logger.info(f"force_tease_end: {winner.candidate_id} score={scored[0][0]:.3f}")
        return assignments

    # pass 2: swap from proof_moment
    proof_a = next((a for a in assignments
                    if a.slot.role == "proof_moment" and a.selected is not None
                    and a.alternatives), None)
    if proof_a:
        ts = score_tease_end_candidate(proof_a.selected, cfg)
        if ts > 0:
            winner = proof_a.selected
            proof_a.selected = proof_a.alternatives.pop(0)
            winner.assigned_slot = tease_assn.slot.slot_id
            tease_assn.selected = winner
            tease_assn.selection_reason = "force_tease_end: swap_from_proof_moment"
            return assignments

    # pass 3: relax outcome spoiler threshold
    relaxed = [(score_tease_end_candidate(c, cfg) * 0.85, c) for c in candidates]
    relaxed = sorted([(s,c) for s,c in relaxed if s > 0], key=lambda x: x[0], reverse=True)
    if relaxed:
        winner = relaxed[0][1]
        winner.assigned_slot = tease_assn.slot.slot_id
        winner.is_fallback_pick = True
        tease_assn.selected = winner
        tease_assn.used_fallback = True
        tease_assn.selection_reason = "force_tease_end: relaxed_threshold"
        logger.info(f"force_tease_end (relaxed): {winner.candidate_id}")

    return assignments


# =============================================================================
# STAGE 8 — OPTIMIZER (new passes)
# =============================================================================

def _selected(assignments: List[SlotAssignment]) -> List[Tuple[SlotAssignment, TrailerCandidate]]:
    return [(a, a.selected) for a in assignments if a.selected is not None]


def preview_coherence_pass(
    assignments: List[SlotAssignment],
    all_candidates: Optional[List[TrailerCandidate]] = None,
) -> List[SlotAssignment]:
    """
    Hard enforcement: the assembled sequence MUST contain sufficient context.

    Gate 1 — premise slot filled?
        At least one slot with role "premise" must be filled.
    Gate 2 — adequate total premise_value?
        Sum of premise_value across all filled clips must exceed
        MIN_TOTAL_PREMISE = 0.40 * n_filled_slots.
    Gate 3 — spoiler creep?
        If avg outcome_spoiler_risk of filled clips > 0.55, the sequence
        is leaking too much — force-swap the worst offender.

    Repair strategy:
        Step 1: try alternatives in the weakest non-premise slot.
        Step 2: if Step 1 fails, search all_candidates (full pool)
                for the best premise candidate and inject it into the
                slot with the lowest utility.
    """
    filled = [(a, a.selected) for a in assignments if a.selected]
    if not filled:
        return assignments

    def _needs_repair() -> bool:
        has_premise = any(a.slot.role in ("premise", "open_hook") for a, _ in filled)
        if not has_premise:
            return True
        total_pv = sum(c.premise_value for _, c in filled)
        min_pv = 0.40 * len(filled)
        return total_pv < min_pv

    # Gate 3: spoiler creep — independent of premise check
    avg_outcome_spoiler = float(np.mean([c.outcome_spoiler_risk for _, c in filled]))
    if avg_outcome_spoiler > 0.55:
        # swap out the clip with the highest outcome_spoiler_risk
        worst_a = max(
            [a for a in assignments if a.selected],
            key=lambda a: a.selected.outcome_spoiler_risk if a.selected else 0,
        )
        for alt in worst_a.alternatives:
            if alt.outcome_spoiler_risk < worst_a.selected.outcome_spoiler_risk - 0.15:
                logger.warning(
                    f"preview_coherence [spoiler_creep]: slot '{worst_a.slot.slot_id}' "
                    f"outcome_spoiler {worst_a.selected.outcome_spoiler_risk:.2f}"
                    f"→{alt.outcome_spoiler_risk:.2f}"
                )
                worst_a.alternatives.insert(0, worst_a.selected)
                worst_a.selected = alt
                alt.assigned_slot = worst_a.slot.slot_id
                break

    if not _needs_repair():
        return assignments

    logger.warning("preview_coherence: insufficient context — initiating repair")

    # Step 1: inject premise from alternatives of any non-tease slot
    for a in assignments:
        if a.slot.role == "tease_end": continue
        for alt in a.alternatives:
            if alt.premise_value >= 0.45:
                logger.info(
                    f"preview_coherence [step1]: inject premise into slot '{a.slot.slot_id}' "
                    f"premise_value={alt.premise_value:.2f}"
                )
                if a.selected:
                    a.alternatives.insert(0, a.selected)
                a.selected = alt
                alt.assigned_slot = a.slot.slot_id
                if not _needs_repair():
                    return assignments
                break

    if not _needs_repair():
        return assignments

    # Step 2: search full candidate pool
    if all_candidates:
        used_ids = {a.selected.candidate_id for a in assignments if a.selected}
        # find best premise candidate not yet used
        premise_pool = [
            c for c in all_candidates
            if c.candidate_id not in used_ids and c.premise_value >= 0.40
        ]
        premise_pool.sort(key=lambda c: c.premise_value, reverse=True)
        if premise_pool:
            best_premise = premise_pool[0]
            # inject into the slot with lowest selection_priority (weakest slot)
            weakest_a = min(
                [a for a in assignments
                 if a.selected and a.slot.role not in ("tease_end", "open_hook")],
                key=lambda a: a.selected.selection_priority if a.selected else 1.0,
                default=None,
            )
            if weakest_a:
                logger.warning(
                    f"preview_coherence [step2]: force inject "
                    f"{best_premise.candidate_id} (premise={best_premise.premise_value:.2f}) "
                    f"into slot '{weakest_a.slot.slot_id}'"
                )
                weakest_a.alternatives.insert(0, weakest_a.selected)
                weakest_a.selected = best_premise
                best_premise.assigned_slot = weakest_a.slot.slot_id

    return assignments


def ending_sting_pass(
    assignments: List[SlotAssignment],
    cfg: TrailerModeConfig,
) -> List[SlotAssignment]:
    """
    Ensure the last clip in chronological order has high ending_strength.
    If not, swap with the best ending-capable alternative.
    """
    filled = sorted(_selected(assignments), key=lambda x: x[1].start)
    if not filled: return assignments

    last_a, last_c = filled[-1]
    if last_c.ending_strength >= 0.55: return assignments

    for alt in last_a.alternatives:
        if alt.ending_strength > last_c.ending_strength + 0.10:
            logger.info(
                f"ending_sting: slot '{last_a.slot.slot_id}' "
                f"ending {last_c.ending_strength:.2f}→{alt.ending_strength:.2f}"
            )
            last_a.alternatives.insert(0, last_c)
            last_a.selected = alt
            alt.assigned_slot = last_a.slot.slot_id
            break
    return assignments


_EXP_TYPES_OVER = frozenset({"explanation", "step_by_step", "definition", "tutorial", "demo"})


def _adjacent_redundancy_score(
    prev: Optional[TrailerCandidate],
    cur: TrailerCandidate,
    op: OptimizationPolicy,
) -> float:
    """
    0–1: how redundant `cur` feels vs `prev` (multi-signal, weights from OptimizationPolicy).
    """
    if prev is None:
        return 0.0
    wv = op.redundancy_w_visual
    wl = op.redundancy_w_lexical
    ws = op.redundancy_w_scene
    wt = op.redundancy_w_theme
    wsp = op.redundancy_w_speaker
    wm = op.redundancy_w_mode
    wsum = wv + wl + ws + wt + wsp + wm
    if wsum <= 0:
        return 0.0
    vis = _jaccard(prev.visual_signature, cur.visual_signature)
    lex = _jaccard(prev.semantic_markers, cur.semantic_markers)
    sce = _jaccard(prev.scene_signature, cur.scene_signature)
    same_theme = (
        1.0 if (prev.theme_id and cur.theme_id and prev.theme_id == cur.theme_id) else 0.0
    )
    spk = (
        1.0 if (
            prev.speaker_signature
            and cur.speaker_signature
            and prev.speaker_signature == cur.speaker_signature
            and prev.speaker_signature not in ("unknown",)
        ) else 0.0
    )
    mode_same = 1.0 if prev.source_mode == cur.source_mode else 0.0
    return float(
        (wv * vis + wl * lex + ws * sce + wt * same_theme + wsp * spk + wm * mode_same) / wsum
    )


def _explanation_burden(assign: SlotAssignment, op: OptimizationPolicy, cfg: TrailerModeConfig) -> float:
    """Graded 0–~1.2: how much this slot reads as “lecture / mini-summary”, not binary."""
    if not assign.selected:
        return 0.0
    c = assign.selected
    st = (c.source_type or "").lower()
    b = 0.0
    if c.source_mode == "educational":
        b += 0.38
    if st in _EXP_TYPES_OVER:
        b += 0.32
    b += 0.22 * float(np.clip(c.explanation_spoiler_risk, 0, 1))
    b += 0.12 * float(np.clip(c.premise_value, 0, 1))
    b += 0.10 * float(np.clip((c.duration - 6.0) / 18.0, 0, 1))
    if assign.slot.role == "premise" or assign.slot.preview_intent == "context":
        b *= op.over_explanation_context_slot_discount
    if assign.slot.role == "tease_end":
        b *= 0.42
    if cfg.assembly_mode != "preview":
        b *= 0.88
    return float(np.clip(b, 0, 1.35))


def _chronological_neighbors_for_assignment(
    assignments: List[SlotAssignment],
    target: SlotAssignment,
) -> Tuple[Optional[TrailerCandidate], Optional[TrailerCandidate]]:
    filled = sorted(_selected(assignments), key=lambda x: x[1].start)
    idx = next((i for i, (aa, _) in enumerate(filled) if aa is target), -1)
    if idx < 0:
        return None, None
    prev_c = filled[idx - 1][1] if idx > 0 else None
    next_c = filled[idx + 1][1] if idx + 1 < len(filled) else None
    return prev_c, next_c


def _min_premise_floor_for_replacement(assign: SlotAssignment) -> float:
    """Mixed videos: do not strip all context from premise/context slots."""
    if not assign.selected:
        return 0.0
    if assign.slot.role != "premise" and assign.slot.preview_intent != "context":
        return 0.0
    pv = assign.selected.premise_value
    if pv < 0.42:
        return 0.0
    return 0.22 + 0.35 * float(np.clip(pv, 0, 1))


def over_explanation_pass(
    assignments: List[SlotAssignment],
    cfg: TrailerModeConfig,
    all_candidates: Optional[List[TrailerCandidate]] = None,
) -> List[SlotAssignment]:
    """
    Prevent mini-summary overload. v3.2: graded explanation *burden* (not binary edu flag),
    swap order by composite priority so mixed-format videos lose the most “lecture-like”
    beats first; premise/context slots are discounted and replacements must preserve a
    minimum premise_value when the original carried real context.
    """
    op = cfg.optimization

    def _is_explanation_slot(a: SlotAssignment) -> bool:
        if not a.selected:
            return False
        return (
            a.selected.source_mode == "educational"
            or (a.selected.source_type or "").lower() in _EXP_TYPES_OVER
        )

    explanation_slots = [a for a in assignments if _is_explanation_slot(a)]
    total_dur = sum(a.selected.duration for a in assignments if a.selected)
    exp_dur = sum(a.selected.duration for a in explanation_slots)
    max_exp = cfg.optimization.max_explanation_slots
    max_share = cfg.optimization.max_explanation_duration_share

    if cfg.assembly_mode == "preview":
        max_exp = min(max_exp, 1)
        max_share = min(max_share, 0.35)

    trigger_count = len(explanation_slots) > max_exp
    trigger_dur = total_dur > 0 and (exp_dur / total_dur) > max_share

    if not trigger_count and not trigger_dur:
        return assignments

    n_over_by_count = max(0, len(explanation_slots) - max_exp)
    n_over_by_dur = 0
    if trigger_dur:
        avg_exp_dur = exp_dur / max(len(explanation_slots), 1)
        excess = exp_dur - max_share * total_dur
        n_over_by_dur = max(1, int(np.ceil(excess / max(avg_exp_dur, 1.0))))

    n_to_reduce = max(n_over_by_count, n_over_by_dur)
    logger.warning(
        f"over_explanation: {len(explanation_slots)} explanatory slots, "
        f"share={exp_dur/max(total_dur,1):.0%} — reducing {n_to_reduce} (graded burden)"
    )

    used_ids = {a.selected.candidate_id for a in assignments if a.selected}

    def _swap_priority(a: SlotAssignment) -> float:
        burden = _explanation_burden(a, op, cfg)
        prev_c, next_c = _chronological_neighbors_for_assignment(assignments, a)
        r1 = _adjacent_redundancy_score(prev_c, a.selected, op) if a.selected else 0.0
        r2 = _adjacent_redundancy_score(a.selected, next_c, op) if a.selected and next_c else 0.0
        redund = max(r1, r2)
        dur_w = 1.0 + 0.28 * float(np.clip(a.selected.duration / 14.0, 0, 1.5))
        curiosity_rel = a.selected.curiosity_gap_score if a.selected else 0.0
        # high curiosity explanatory clip is *less* urgent to remove
        return burden * dur_w * (1.0 + 0.45 * redund) - 0.18 * curiosity_rel

    if op.over_explanation_use_graded_burden:
        explanation_slots.sort(key=_swap_priority, reverse=True)
    else:
        explanation_slots.sort(
            key=lambda a: a.selected.trailer_score_final if a.selected else 0
        )

    def _replacement_ok(alt: TrailerCandidate, slot_assign: SlotAssignment) -> bool:
        floor = _min_premise_floor_for_replacement(slot_assign)
        if floor <= 0:
            return True
        return alt.premise_value >= floor

    for a in explanation_slots[:n_to_reduce]:
        swapped = False
        floor = _min_premise_floor_for_replacement(a)
        alts_ranked = sorted(
            a.alternatives,
            key=lambda x: (
                x.curiosity_gap_score * 0.45 + x.preview_value * 0.35 + x.trailer_score_final * 0.20
            ),
            reverse=True,
        )
        for alt in alts_ranked:
            if alt.source_mode == "educational":
                continue
            if (alt.source_type or "").lower() in _EXP_TYPES_OVER:
                continue
            if not _replacement_ok(alt, a):
                continue
            logger.info(
                f"over_explanation [alt]: slot '{a.slot.slot_id}' "
                f"{a.selected.source_mode}→{alt.source_mode} "
                f"(burden={_explanation_burden(a, op, cfg):.2f})"
            )
            a.alternatives.insert(0, a.selected)
            a.selected = alt
            alt.assigned_slot = a.slot.slot_id
            used_ids.add(alt.candidate_id)
            swapped = True
            break

        if not swapped and all_candidates:
            pool = sorted(
                all_candidates,
                key=lambda c: (
                    c.curiosity_gap_score * 0.45 + c.preview_value * 0.35 + c.trailer_score_final * 0.20
                ),
                reverse=True,
            )
            for cand in pool:
                if cand.candidate_id in used_ids:
                    continue
                if cand.source_mode == "educational":
                    continue
                if (cand.source_type or "").lower() in _EXP_TYPES_OVER:
                    continue
                if not _replacement_ok(cand, a):
                    continue
                fit = _score_for_slot(cand, a.slot)
                if fit >= 0:
                    logger.warning(
                        f"over_explanation [pool]: slot '{a.slot.slot_id}' "
                        f"inject {cand.candidate_id} ({cand.source_mode})"
                    )
                    a.alternatives.insert(0, a.selected)
                    a.selected = cand
                    cand.assigned_slot = a.slot.slot_id
                    used_ids.add(cand.candidate_id)
                    break

    return assignments


def scene_redundancy_pass(
    assignments: List[SlotAssignment],
    cfg: TrailerModeConfig,
) -> List[SlotAssignment]:
    """
    Reduce adjacent clips that feel like duplicates. v3.2: same multi-signal redundancy
    as over_explanation (visual, lexical, scene tags, theme, speaker, mode).
    """
    op = cfg.optimization
    thresh = op.scene_redundancy_similarity_threshold
    filled = sorted(_selected(assignments), key=lambda x: x[1].start)
    for i in range(1, len(filled)):
        prev_a, prev_c = filled[i - 1]
        cur_a, cur_c = filled[i]
        combined_sim = _adjacent_redundancy_score(prev_c, cur_c, op)
        if combined_sim < thresh or not cur_a.alternatives:
            continue
        alts_scored = sorted(
            cur_a.alternatives,
            key=lambda alt: _adjacent_redundancy_score(prev_c, alt, op),
        )
        for alt in alts_scored:
            alt_sim = _adjacent_redundancy_score(prev_c, alt, op)
            if alt_sim < combined_sim - 0.08:
                logger.info(
                    f"scene_redundancy: slot '{cur_a.slot.slot_id}' "
                    f"redund {combined_sim:.2f}→{alt_sim:.2f}"
                )
                cur_a.alternatives.insert(0, cur_c)
                cur_a.selected = alt
                alt.assigned_slot = cur_a.slot.slot_id
                break
    return assignments


def optimize_trailer_plan(
    assignments: List[SlotAssignment],
    candidates: List[TrailerCandidate],
    cfg: TrailerModeConfig,
    transition_edges: Optional[List[TransitionEdge]] = None,
    base_analysis: Optional[Dict] = None,
    video_dur: float = 0.0,
) -> List[SlotAssignment]:
    op = cfg.optimization

    sel = [a.selected for a in assignments if a.selected]
    if not sel: return assignments

    # 1. Spoiler budget
    if op.enable_spoiler_swap:
        budget = cfg.spoiler.max_trailer_spoiler_budget
        agg = float(np.mean([c.spoiler_risk for c in sel]))
        if agg > budget:
            for a in sorted(assignments,
                            key=lambda x: x.selected.spoiler_risk if x.selected else 0,
                            reverse=True):
                if not a.selected: continue
                for alt in a.alternatives:
                    if alt.spoiler_risk < a.selected.spoiler_risk*0.75:
                        a.alternatives.insert(0, a.selected)
                        a.selected = alt; alt.assigned_slot=a.slot.slot_id; break
                if float(np.mean([x.selected.spoiler_risk for x in assignments if x.selected])) <= budget:
                    break

    # 2. Monotony guard
    if op.enable_monotony_guard:
        filled = sorted([(a,a.selected) for a in assignments if a.selected],
                        key=lambda x: x[1].start)
        for i in range(1, len(filled)):
            pa,pc = filled[i-1]; ca,cc = filled[i]
            if pc.source_mode == cc.source_mode and ca.alternatives:
                for alt in ca.alternatives:
                    if alt.source_mode != pc.source_mode:
                        ca.alternatives=[cc]+[x for x in ca.alternatives if x.candidate_id!=alt.candidate_id]
                        ca.selected=alt; alt.assigned_slot=ca.slot.slot_id; break

    # 3. Theme diversity
    if op.enable_theme_diversity:
        used_t: set = set()
        for a in assignments:
            if not a.selected: continue
            tid = a.selected.theme_id
            if tid and tid in used_t and a.alternatives:
                for alt in a.alternatives:
                    if alt.theme_id not in used_t:
                        a.alternatives.insert(0,a.selected)
                        a.selected=alt; alt.assigned_slot=a.slot.slot_id; tid=alt.theme_id; break
            if tid: used_t.add(tid)

    # 4. Transition quality — run multiple passes (rebuild graph each pass; no stale edges)
    if op.enable_transition_optimization:
        floor = op.transition_quality_floor
        for _pass_n in range(op.transition_optimizer_passes):
            live_edges = build_transition_graph(assignments, base_analysis, video_dur)
            improved_any = False
            for edge in [e for e in live_edges if e.score < floor]:
                to_a = next((a for a in assignments
                             if a.selected and a.selected.candidate_id == edge.to_candidate_id),
                            None)
                if not to_a or not to_a.alternatives: continue
                from_c = next((a.selected for a in assignments
                               if a.selected and a.selected.candidate_id == edge.from_candidate_id),
                              None)
                if not from_c: continue
                best_alt = None
                best_alt_score = edge.score
                for alt in to_a.alternatives:
                    alt_edge = compute_transition_score(
                        from_c, alt, edge.from_slot_id, to_a.slot.slot_id,
                        base_analysis, video_dur,
                    )
                    # v3.1: lower improvement threshold 0.10 → 0.06
                    if alt_edge.score > best_alt_score + 0.06:
                        best_alt_score = alt_edge.score
                        best_alt = alt
                if best_alt is not None:
                    to_a.alternatives.insert(0, to_a.selected)
                    to_a.selected = best_alt
                    best_alt.assigned_slot = to_a.slot.slot_id
                    improved_any = True
            if not improved_any:
                break  # early exit if no swaps happened

    # v3 passes (all_candidates fed in so passes can draw from full pool)
    if op.enable_preview_coherence:
        assignments = preview_coherence_pass(assignments, candidates)
    if op.enable_ending_sting:
        assignments = ending_sting_pass(assignments, cfg)
    if op.enable_over_explanation:
        assignments = over_explanation_pass(assignments, cfg, candidates)
    if op.enable_scene_redundancy:
        assignments = scene_redundancy_pass(assignments, cfg)

    return assignments


# =============================================================================
# STAGE 9 — UI PAYLOAD (extended)
# =============================================================================

def _c_dict(c: Optional[TrailerCandidate]) -> Optional[Dict]:
    if c is None: return None
    return {
        "candidate_id": c.candidate_id,
        "start": round(c.start if c.refined_start is None else c.refined_start, 2),
        "end": round(c.end if c.refined_end is None else c.refined_end, 2),
        "duration": round(c.duration, 2),
        "source_mode": c.source_mode, "source_type": c.source_type,
        "title": c.title, "score": round(c.trailer_score_final, 4),
        "suitability_score": round(c.suitability_score, 4),
        "risk_penalty": round(c.risk_penalty, 4),
        "spoiler_risk": round(c.spoiler_risk, 3),
        "spoiler_category": c.spoiler_category,
        "curiosity_gap": round(c.curiosity_gap_score, 3),
        "hook_strength": round(c.hook_strength, 3),
        "preview_value": round(c.preview_value, 3),
        "premise_value": round(c.premise_value, 3),
        "ending_strength": round(c.ending_strength, 3),
        "self_containedness": round(c.self_containedness, 3),
        "cut_naturalness": round(c.cut_naturalness_score, 3),
        "theme_id": c.theme_id, "theme_label": c.theme_label, "theme_role": c.theme_role,
        "narrative_role_hint": c.narrative_role_hint,
        "is_fallback_pick": c.is_fallback_pick,
        "semantic_markers": c.semantic_markers[:5],
    }


def _swap_reason(sel: TrailerCandidate, alt: TrailerCandidate) -> str:
    parts = []
    if alt.spoiler_risk < sel.spoiler_risk: parts.append("менее спойлерный")
    elif alt.spoiler_risk > sel.spoiler_risk: parts.append("более спойлерный")
    if alt.curiosity_gap_score > sel.curiosity_gap_score: parts.append("больше интриги")
    if alt.source_mode != sel.source_mode: parts.append(f"другой режим ({alt.source_mode})")
    if alt.duration < sel.duration: parts.append("короче")
    elif alt.duration > sel.duration: parts.append("длиннее")
    if alt.theme_id != sel.theme_id: parts.append("другая тема")
    if alt.preview_value > sel.preview_value: parts.append("лучше для превью")
    if alt.premise_value > sel.premise_value: parts.append("больше контекста")
    if alt.trailer_score_final < sel.trailer_score_final: parts.append("чуть слабее по score")
    return ", ".join(parts) if parts else "альтернативный вариант"


def _edit_commands_v3() -> List[Dict]:
    cmds = [
        TrailerEditCommand("replace_clip", description="Заменить выбранный клип в слоте"),
        TrailerEditCommand("replace_theme", description="Заменить весь смысловой блок"),
        TrailerEditCommand("replace_with_same_theme", description="Взять другой клип из той же темы"),
        TrailerEditCommand("replace_with_more_visual", description="Взять клип с более сильным визуалом"),
        TrailerEditCommand("replace_with_more_context", description="Взять клип с большей explanatory ценностью"),
        TrailerEditCommand("make_less_spoilery", description="Заменить на менее спойлерный вариант"),
        TrailerEditCommand("make_faster", description="Взять более короткий вариант"),
        TrailerEditCommand("make_clearer", description="Взять клип с более высоким self-containedness"),
        TrailerEditCommand("reduce_spoilers", description="Пересобрать с пониженным spoiler budget"),
        TrailerEditCommand("increase_pace", description="Сократить длину — только primary + короткие слоты"),
        TrailerEditCommand("more_intrigue", description="Поднять curiosity_target"),
        TrailerEditCommand("shorter", description="Обрезать до target * 0.7"),
        TrailerEditCommand("more_educational", description="Повысить вес educational mode"),
        TrailerEditCommand("swap_slot", description="Поменять содержимое двух слотов"),
        TrailerEditCommand("reorder_blocks", description="Поменять хронологию слотов"),
        TrailerEditCommand("add_more_from_theme", description="Добавить второй клип из той же темы"),
    ]
    return [asdict(c) for c in cmds]


def build_ui_payload(
    assignments: List[SlotAssignment],
    theme_blocks: List[ThemeBlock],
    transition_edges: List[TransitionEdge],
    all_candidates: List[TrailerCandidate],
    cfg: TrailerModeConfig,
) -> Dict[str, Any]:
    cand_map = {c.candidate_id: c for c in all_candidates}

    assembly_plan = []
    for a in assignments:
        entry: Dict[str,Any] = {
            "slot_id": a.slot.slot_id, "role": a.slot.role,
            "slot_group": a.slot.slot_group, "required": a.slot.required,
            "preview_intent": a.slot.preview_intent,
            "description": a.slot.description,
            "duration_target": {"min": a.slot.min_duration, "max": a.slot.max_duration},
            "selected": _c_dict(a.selected), "filled": a.selected is not None,
            "used_fallback": a.used_fallback,
            "eligibility_fail_count": a.eligibility_fail_count,
            "selection_reason": a.selection_reason,
            # NEW v3
            "spoiler_meter": round(a.selected.spoiler_risk, 3) if a.selected else 0.0,
            "preview_intent_score": round(a.selected.preview_value, 3) if a.selected else 0.0,
        }
        assembly_plan.append(entry)

    # clip alternatives with extended reasons
    alternatives: Dict[str, List[Dict]] = {}
    for a in assignments:
        alts = []
        for alt in a.alternatives:
            d = _c_dict(alt)
            if d:
                d["swap_reason"] = _swap_reason(a.selected, alt) if a.selected else "альтернативный вариант"
                # why_not: specific rejection reason from eligibility
                er = alt.eligibility_results.get(a.slot.slot_id)
                d["why_not_primary"] = (
                    "; ".join(er.hard_fail_reasons) if er and er.hard_fail_reasons
                    else "score below selected"
                )
                alts.append(d)
        alternatives[a.slot.slot_id] = alts

    # theme-level alternatives
    theme_alternatives: Dict[str, List[Dict]] = {}
    for block in theme_blocks:
        themed = [cand_map[cid] for cid in block.best_candidate_ids if cid in cand_map]
        theme_alternatives[block.theme_id] = [_c_dict(c) for c in themed if _c_dict(c)]

    # transition graph with why_this_transition explanations
    transition_graph = []
    for edge in transition_edges:
        entry_e = {
            "from_slot": edge.from_slot_id, "to_slot": edge.to_slot_id,
            "score": edge.score, "reasons": edge.reasons,
            "why_this_transition": _transition_explanation(edge),
            "sub_scores": {
                "scene_cut": edge.scene_cut_score,
                "audio_carry": edge.audio_carry_score,
                "narrative_handoff": edge.narrative_handoff_score,
                "visual_match": edge.visual_match_score,
                "boundary_handoff": edge.boundary_handoff_score,
                "temporal_contrast": edge.temporal_contrast_score,
                "mode_contrast": edge.mode_contrast_score,
                "rhythm": edge.rhythm_compatibility,
                "spoiler_safe": edge.spoiler_escalation_safe,
                "hard_break_penalty": edge.hard_break_penalty,
            },
        }
        transition_graph.append(entry_e)

    # theme map
    theme_map = [
        {
            "theme_id": b.theme_id, "label": b.label, "constructor_label": b.constructor_label,
            "start": round(b.start,2), "end": round(b.end,2), "role": b.role,
            "importance": round(b.importance,3), "spoiler_level": round(b.spoiler_level,3),
            "semantic_signature": b.semantic_signature[:5],
            "scene_signature": b.scene_signature[:3],
            "dominant_modes": b.dominant_source_modes,
            "speaker_signature": b.speaker_signature,
            "replaceable": b.replaceable,
            "alternative_theme_ids": b.alternative_theme_ids,
            "topic_confidence": round(b.topic_confidence, 3),
            "num_candidates_in_theme": len(b.candidate_ids),
            "calibration_hint": (
                "tight_cluster" if b.topic_confidence >= 0.55
                else "loose_cluster_review_on_real_video"
            ),
            "coherence_with_neighbors": round(b.coherence_with_neighbors,3),
        }
        for b in theme_blocks
    ]

    # explanations with spoiler_meter and preview_intent
    explanations = []
    for a in assignments:
        if a.selected:
            txt = (
                f"Слот '{a.slot.slot_id}' ({a.slot.role} / {a.slot.preview_intent}): "
                f"[{a.selected.source_mode}/{a.selected.source_type}] "
                f"{a.selected.start:.1f}–{a.selected.end:.1f}s "
                f"(score={a.selected.trailer_score_final:.3f}, "
                f"suitability={a.selected.suitability_score:.3f}, "
                f"risk={a.selected.risk_penalty:.3f}, "
                f"spoiler={a.selected.spoiler_risk:.2f}, "
                f"preview_value={a.selected.preview_value:.2f}, "
                f"cut_naturalness={a.selected.cut_naturalness_score:.2f}"
                + (" [FALLBACK]" if a.used_fallback else "")
                + f"). {a.selection_reason}"
            )
        else:
            txt = (
                f"Слот '{a.slot.slot_id}' ({a.slot.role}): не заполнен. "
                f"{a.eligibility_fail_count} ineligible кандидатов."
            )
        explanations.append({
            "slot_id": a.slot.slot_id,
            "preview_intent": a.slot.preview_intent,
            "spoiler_meter": round(a.selected.spoiler_risk, 3) if a.selected else 0.0,
            "text": txt,
        })

    edit_contract = {
        "supported_commands": _edit_commands_v3(),
        "reassembly_strategy": "slot_aware_v3.0",
        "assembly_mode": cfg.assembly_mode,
        "slot_ids": [a.slot.slot_id for a in assignments],
        "theme_ids": [b.theme_id for b in theme_blocks],
    }
    editable_actions = {
        "replace_clip": True, "replace_theme": True,
        "add_more_from_theme": True, "reduce_spoilers": True,
        "increase_pace": True, "make_clearer": True, "swap_slot": True,
        "reorder_blocks": True, "shorter": True, "more_dynamic": True,
        "more_intrigue": True, "replace_with_same_theme": True,
        "replace_with_more_visual": True, "replace_with_more_context": True,
        "make_less_spoilery": True, "make_faster": True,
    }

    return {
        "assembly_plan": assembly_plan,
        "alternatives": alternatives,
        "theme_alternatives": theme_alternatives,
        "theme_map": theme_map,
        "transition_graph": transition_graph,
        "explanations": explanations,
        "edit_contract": edit_contract,
        "editable_actions": editable_actions,
    }


def _transition_explanation(edge: TransitionEdge) -> str:
    parts = []
    if edge.scene_cut_score > 0.65: parts.append("чистый визуальный переход")
    elif edge.scene_cut_score < 0.35: parts.append("резкий визуальный разрыв")
    if edge.audio_carry_score > 0.65: parts.append("плавный аудио-переход")
    elif edge.audio_carry_score < 0.35: parts.append("аудио-разрыв")
    if edge.narrative_handoff_score > 0.7: parts.append("естественный нарративный поток")
    if edge.visual_match_score > 0.55: parts.append("визуальная преемственность")
    elif edge.visual_match_score < 0.12: parts.append("сильная смена картинки")
    if edge.boundary_handoff_score > 0.72: parts.append("аккуратная точка входа во второй клип")
    if edge.mode_contrast_score > 0.5: parts.append("смена режима — разнообразие")
    if edge.hard_break_penalty > 0.15: parts.append("внимание: жёсткий обрыв речи/действия")
    return "; ".join(parts) if parts else f"score={edge.score:.2f}"


# =============================================================================
# FINAL CLIP BUILDER + EMPTY RESULT
# =============================================================================

def _make_clip(c: TrailerCandidate) -> Dict[str, Any]:
    score = round(c.trailer_score_final, 3)
    s = c.refined_start if c.refined_start is not None else c.start
    e = c.refined_end if c.refined_end is not None else c.end
    return {
        "start": round(s,2), "end": round(e,2), "duration": round(e-s,2),
        "score": score, "source_mode": c.source_mode, "source_type": c.source_type,
        "title": c.title, "summary": c.summary,
        "export_title": (f"Trailer_{c.source_mode}_{c.source_type}_"
                         f"{s:.1f}-{e:.1f}s_score{score:.2f}"),
        "reasons": c.reasons, "priority": "trailer", "priority_score": 1.0,
        # v3 extras
        "candidate_id": c.candidate_id,
        "theme_id": c.theme_id, "theme_label": c.theme_label,
        "assigned_slot": c.assigned_slot, "is_fallback_pick": c.is_fallback_pick,
        "spoiler_risk": round(c.spoiler_risk,3), "spoiler_category": c.spoiler_category,
        "curiosity_gap_score": round(c.curiosity_gap_score,3),
        "hook_strength": round(c.hook_strength,3),
        "preview_value": round(c.preview_value,3),
        "premise_value": round(c.premise_value,3),
        "ending_strength": round(c.ending_strength,3),
        "self_containedness": round(c.self_containedness,3),
        "cut_naturalness_score": round(c.cut_naturalness_score,3),
        "narrative_role_hint": c.narrative_role_hint,
        "scene_safe_start": round(c.scene_safe_start or s, 2),
        "scene_safe_end": round(c.scene_safe_end or e, 2),
        "score_breakdown": {
            "suitability": round(c.suitability_score,4),
            "risk_penalty": round(c.risk_penalty,4),
            "slot_fit": round(c.slot_fit_score,4),
            "raw": round(c.trailer_score_raw,4),
            "spoiler_subtypes": {
                "outcome": round(c.outcome_spoiler_risk,3),
                "reveal": round(c.reveal_spoiler_risk,3),
                "payoff": round(c.payoff_spoiler_risk,3),
                "explanation": round(c.explanation_spoiler_risk,3),
            },
        },
    }


def _empty(video_dur: float, error: str, cfg: Optional[TrailerModeConfig]=None) -> Dict:
    _c = cfg or TrailerModeConfig()
    return {
        "mode": "trailer", "assembly_mode": _c.assembly_mode, "error": error,
        "trailer_clips":[], "render_instructions":[], "theme_blocks":[],
        "assembly_plan":[], "alternatives":{}, "theme_alternatives":{},
        "transition_graph":[], "explanations":[], "edit_contract":{},
        "editable_actions":{}, "theme_map":[], "boundary_diagnostics":{},
        "stats": {"total_duration":video_dur, "target_duration":_c.target_trailer_duration,
                  "trailer_duration":0.0, "num_clips":0},
    }


# =============================================================================
# ORCHESTRATION PIPELINES
# =============================================================================

def run_candidate_pipeline(hook_result, story_result, viral_result, educational_result,
                            video_dur, cfg, base_analysis, slots) -> List[TrailerCandidate]:
    candidates = collect_candidates(hook_result, story_result, viral_result, educational_result)
    n_raw = len(candidates)
    candidates = [c for c in candidates if c.duration >= cfg.min_clip_duration]
    candidates = [c for c in candidates if c.base_score >= cfg.min_base_score]
    logger.info(f"Candidates: {n_raw} raw → {len(candidates)} after quality filter")
    if not candidates: return candidates
    candidates = enrich_candidates(candidates, video_dur, cfg, base_analysis, slots)
    for c in candidates:
        compute_trailer_score(c, cfg, video_dur)
    candidates.sort(key=lambda c: c.trailer_score_final, reverse=True)
    return candidates


def run_theme_pipeline(candidates, video_dur, cfg,
                       base_analysis=None, asr_segments=None) -> List[ThemeBlock]:
    return build_theme_blocks(candidates, video_dur, cfg, base_analysis, asr_segments)


def run_slot_pipeline(nms_pool, slots, cfg, base_analysis=None,
                      transition_matrix=None, video_dur: float = 0.0):
    assignments = assign_candidates_to_slots(
        nms_pool, slots, cfg, base_analysis, transition_matrix, video_dur
    )
    if cfg.slot.tease_end_required:
        assignments = force_tease_end(assignments, nms_pool, cfg)
    return assignments


def run_optimizer_pipeline(
    assignments, nms_pool, cfg, base_analysis=None, video_dur: float = 0.0,
) -> Tuple[List[SlotAssignment], List[TransitionEdge]]:
    edges: List[TransitionEdge] = []
    if cfg.enable_transition_graph:
        edges = build_transition_graph(assignments, base_analysis, video_dur)
    assignments = optimize_trailer_plan(
        assignments, nms_pool, cfg, edges, base_analysis, video_dur
    )
    if cfg.enable_transition_graph:
        edges = build_transition_graph(assignments, base_analysis, video_dur)
    return assignments, edges


def run_ui_payload_builder(assignments, theme_blocks, edges, all_candidates, cfg):
    if not cfg.enable_ui_payload: return {}
    return build_ui_payload(assignments, theme_blocks, edges, all_candidates, cfg)


# =============================================================================
# v3.2 — ADAPTIVE DURATION + SECOND-PASS NMS + DEGRADED PREVIEW
# =============================================================================

def _resolve_effective_target(
    cfg: TrailerModeConfig,
    video_dur: float,
) -> Tuple[float, Dict[str, Any]]:
    """
    Адаптивный target_duration:
      max_total_duration = min(target_trailer_duration, video_duration * 0.45)

    Для коротких роликов (< 3 мин) понижаем дефолтные 90с до 30-45с:
      - video < 60s: 25% от длительности (но не менее 10с, не более 20с)
      - 60 ≤ video < 180s: 30-45с
      - 180 ≤ video < 600s: target как есть, но не более 45% длительности
      - video ≥ 600s: target как есть

    Возвращает (effective_target, trace).
    """
    original = cfg.target_trailer_duration
    hard_cap = video_dur * 0.45

    if video_dur < 60.0:
        target = max(10.0, min(20.0, video_dur * 0.25))
        reason = "short_video_auto_preview"
    elif video_dur < 180.0:
        target = min(max(30.0, original * 0.50), 45.0)
        reason = "medium_video_preview_30_45s"
    elif video_dur < 600.0:
        target = min(original, hard_cap)
        reason = "cap_at_45pct_of_video"
    else:
        target = original
        reason = "long_video_use_original"

    # Абсолютный потолок — не больше 45% длительности видео
    target = min(target, hard_cap)
    target = max(target, 5.0)  # sanity floor

    trace = {
        "original_target": original,
        "video_duration_sec": video_dur,
        "max_hard_cap_45pct": hard_cap,
        "effective_target": round(target, 2),
        "reason": reason,
    }
    return float(target), trace


def _second_pass_nms_and_dedupe(
    final_candidates: List["TrailerCandidate"],
    assignments: List["SlotAssignment"],
    overlap_iou_thresh: float = 0.35,
    containment_thresh: float = 0.60,
    min_overlap_sec: float = 2.0,
) -> Tuple[List["TrailerCandidate"], Dict[str, Any]]:
    """
    Второй pass NMS после slot assignment:
      1. Дубли по candidate_id убираются.
      2. Overlap/containment: конфликт если overlap > min_overlap_sec AND
         (iou > overlap_iou_thresh OR containment > containment_thresh).
         Это ловит вложенные клипы, которые чистый IoU пропускает.
      3. При конфликте: удаляем слабейший клип (по trailer_score_final).
      4. Возвращает (kept_candidates, trace_dict).
    """
    trace: Dict[str, Any] = {
        "overlap_iou_thresh": overlap_iou_thresh,
        "containment_thresh": containment_thresh,
        "min_overlap_sec": min_overlap_sec,
        "input_count": len(final_candidates),
        "duplicate_removed": [],
        "overlap_removed": [],
    }

    # 1. Убираем дубли по candidate_id
    seen_ids: set = set()
    dedup: List["TrailerCandidate"] = []
    for c in final_candidates:
        cid = getattr(c, "candidate_id", None)
        if cid in seen_ids:
            trace["duplicate_removed"].append({
                "candidate_id": cid, "start": c.start, "end": c.end,
                "reason": "already_assigned_to_another_slot",
            })
            continue
        seen_ids.add(cid)
        dedup.append(c)

    # 2. Conflict NMS — сортируем по score desc
    sorted_by_score = sorted(dedup, key=lambda x: getattr(x, "trailer_score_final", 0.0), reverse=True)
    kept: List["TrailerCandidate"] = []
    for c in sorted_by_score:
        conflict_with = None
        for k in kept:
            if _is_conflict(
                (c.start, c.end), (k.start, k.end),
                min_overlap_sec=min_overlap_sec,
                iou_thresh=overlap_iou_thresh,
                containment_thresh=containment_thresh,
            ):
                conflict_with = k
                break
        if conflict_with is None:
            kept.append(c)
        else:
            c_score = getattr(c, "trailer_score_final", 0.0)
            k_score = getattr(conflict_with, "trailer_score_final", 0.0)
            c_dur = c.end - c.start
            k_dur = conflict_with.end - conflict_with.start
            overlap = max(0.0, min(c.end, conflict_with.end) - max(c.start, conflict_with.start))
            iou_val = round(_iou((c.start, c.end), (conflict_with.start, conflict_with.end)), 3)
            ct_val = round(_containment((c.start, c.end), (conflict_with.start, conflict_with.end)), 3)
            # Keep higher-score; on tie keep longer (more content)
            if c_score > k_score * 1.05 or (c_score >= k_score * 0.95 and c_dur < k_dur):
                kept.remove(conflict_with)
                kept.append(c)
                trace["overlap_removed"].append({
                    "candidate_id": getattr(conflict_with, "candidate_id", None),
                    "start": conflict_with.start, "end": conflict_with.end,
                    "score": round(k_score, 3),
                    "replaced_by": getattr(c, "candidate_id", None),
                    "reason": "lower_score_conflict",
                    "iou": iou_val, "containment": ct_val, "overlap_sec": round(overlap, 2),
                })
            else:
                trace["overlap_removed"].append({
                    "candidate_id": getattr(c, "candidate_id", None),
                    "start": c.start, "end": c.end,
                    "score": round(c_score, 3),
                    "conflict_with": getattr(conflict_with, "candidate_id", None),
                    "reason": "conflict_lower_score_removed",
                    "iou": iou_val, "containment": ct_val, "overlap_sec": round(overlap, 2),
                })

    trace["output_count"] = len(kept)
    return sorted(kept, key=lambda c: c.start), trace


_SLOT_PRIORITY: Dict[str, int] = {
    "open_hook":           0,
    "premise":             1,
    "proof_moment":        2,
    "tease_end":           3,
    "escalation":          4,
    "supporting_escalation": 5,
    "backup_proof":        6,
    "micro_tease":         7,
}


def _repair_sequence_hard(
    candidates: List["TrailerCandidate"],
    effective_target: float,
    min_overlap_sec: float = 2.0,
    iou_thresh: float = 0.25,
    containment_thresh: float = 0.60,
    trim_tolerance_sec: float = 1.5,
    max_iter: int = 5,
) -> Tuple[List["TrailerCandidate"], Dict[str, Any]]:
    """
    Hard repair loop: runs after second-pass NMS to guarantee
    sequence_valid=True.

    Each iteration:
      1. Overlap repair  — remove weaker clip in any conflict pair.
      2. Duration repair — if total_dur > effective_target:
           a. Try trimming the last clip's end (if overshoot <= trim_tolerance_sec).
           b. Otherwise remove the lowest-priority / lowest-score clip.

    Returns (repaired_candidates, repair_trace).
    repair_trace contains:
      - overlap_repair_trace: list of removed overlap events
      - duration_trimmed:     list of trim/remove events
      - iterations:           how many passes were run
      - sequence_valid_after: bool
    """
    repair_trace: Dict[str, Any] = {
        "overlap_repair_trace": [],
        "duration_trimmed": [],
        "iterations": 0,
        "input_count": len(candidates),
        "sequence_valid_after": False,
    }

    work = list(candidates)

    for _iter in range(max_iter):
        repair_trace["iterations"] = _iter + 1
        changed = False

        # ── 1. Overlap repair ───────────────────────────────────────────────
        work_sorted = sorted(work, key=lambda c: getattr(c, "trailer_score_final", 0.0), reverse=True)
        clean: List["TrailerCandidate"] = []
        for c in work_sorted:
            conflict_with = None
            for k in clean:
                if _is_conflict(
                    (c.start, c.end), (k.start, k.end),
                    min_overlap_sec=min_overlap_sec,
                    iou_thresh=iou_thresh,
                    containment_thresh=containment_thresh,
                ):
                    conflict_with = k
                    break
            if conflict_with is None:
                clean.append(c)
            else:
                # Remove weaker (c is already weaker because sorted desc)
                overlap = max(0.0, min(c.end, conflict_with.end) - max(c.start, conflict_with.start))
                ct = round(_containment((c.start, c.end), (conflict_with.start, conflict_with.end)), 3)
                iou_v = round(_iou((c.start, c.end), (conflict_with.start, conflict_with.end)), 3)
                repair_trace["overlap_repair_trace"].append({
                    "removed": getattr(c, "candidate_id", None),
                    "start": c.start, "end": c.end,
                    "score": round(getattr(c, "trailer_score_final", 0.0), 3),
                    "kept": getattr(conflict_with, "candidate_id", None),
                    "overlap_sec": round(overlap, 2),
                    "iou": iou_v, "containment": ct,
                    "iteration": _iter + 1,
                })
                changed = True

        work = sorted(clean, key=lambda c: c.start)

        # ── 2. Duration repair ──────────────────────────────────────────────
        total_dur = sum(c.end - c.start for c in work)
        if total_dur > effective_target:
            overshoot = total_dur - effective_target

            if overshoot <= trim_tolerance_sec and work:
                # Trim last clip's end
                last = work[-1]
                new_end = round(last.end - overshoot, 2)
                if new_end > last.start + 3.0:  # keep at least 3s
                    repair_trace["duration_trimmed"].append({
                        "action": "trim_end",
                        "candidate_id": getattr(last, "candidate_id", None),
                        "old_end": last.end, "new_end": new_end,
                        "overshoot_sec": round(overshoot, 2),
                        "iteration": _iter + 1,
                    })
                    last.end = new_end
                    last.duration = last.end - last.start
                    changed = True
            elif work:
                # Remove lowest-priority / lowest-score clip
                def _priority(c: "TrailerCandidate") -> Tuple[int, float]:
                    slot = getattr(c, "assigned_slot", "") or ""
                    prio = _SLOT_PRIORITY.get(slot, 99)
                    score = getattr(c, "trailer_score_final", 0.0)
                    return (prio, -score)  # highest prio number = remove first

                to_remove = max(work, key=_priority)
                repair_trace["duration_trimmed"].append({
                    "action": "remove_clip",
                    "candidate_id": getattr(to_remove, "candidate_id", None),
                    "start": to_remove.start, "end": to_remove.end,
                    "score": round(getattr(to_remove, "trailer_score_final", 0.0), 3),
                    "slot": getattr(to_remove, "assigned_slot", None),
                    "overshoot_sec": round(overshoot, 2),
                    "iteration": _iter + 1,
                })
                work.remove(to_remove)
                changed = True

        # ── 3. Recheck ──────────────────────────────────────────────────────
        total_dur = sum(c.end - c.start for c in work)
        has_overlap = any(
            _is_conflict(
                (work[i].start, work[i].end), (work[j].start, work[j].end),
                min_overlap_sec=min_overlap_sec,
                iou_thresh=iou_thresh,
                containment_thresh=containment_thresh,
            )
            for i in range(len(work))
            for j in range(i + 1, len(work))
        )
        if not has_overlap and total_dur <= effective_target * 1.15:
            repair_trace["sequence_valid_after"] = True
            break
        if not changed:
            break  # can't improve further

    repair_trace["output_count"] = len(work)
    repair_trace["total_duration_after"] = round(sum(c.end - c.start for c in work), 2)
    return work, repair_trace


def _classify_trailer_export_decision(
    final_clips: List[Dict],
    total_duration: float,
    effective_target: float,
    num_themes: int,
    curiosity_agg: float,
    spoiler_agg: float,
    boundary_ok_avg: float,
    is_degraded_preview: bool,
    overlap_removed_count: int,
) -> Tuple[str, List[str]]:
    """
    Единая логика export_decision для trailer:
      auto_export    — всё хорошо + есть diversity + нет overlap + duration OK
      manual_review  — есть сигнал, но есть одна из проблем
      reject         — нет сигнала вообще

    Возвращает (decision, reasons).
    """
    reasons: List[str] = []

    if not final_clips:
        return "reject", ["no_final_clips"]

    # degraded_preview (нет hook/story) никогда auto_export
    if is_degraded_preview:
        reasons.append("degraded_preview_no_hook_or_story")

    # Duration guard: если общая длительность > target * 1.15 → manual_review;
    # если > target * 1.5 → reject
    if total_duration > effective_target * 1.5:
        reasons.append(f"duration_exceeds_1.5x_target ({total_duration:.1f}s > {effective_target*1.5:.1f}s)")
        return "reject", reasons
    if total_duration > effective_target * 1.15:
        reasons.append(f"duration_exceeds_target ({total_duration:.1f}s > {effective_target*1.15:.1f}s)")

    if num_themes <= 1:
        reasons.append(f"low_theme_diversity (themes={num_themes})")
    if curiosity_agg <= 0.05:
        reasons.append(f"low_curiosity_gap ({curiosity_agg:.2f})")
    if spoiler_agg > 0.6:
        reasons.append(f"high_spoiler_risk ({spoiler_agg:.2f})")
    if boundary_ok_avg < 0.55:
        reasons.append(f"weak_boundaries (avg={boundary_ok_avg:.2f})")
    if overlap_removed_count > 2:
        reasons.append(f"many_overlaps_had_to_be_removed ({overlap_removed_count})")

    # Есть клипы, но есть проблемы → manual_review
    if reasons:
        return "manual_review", reasons

    return "auto_export", ["all_quality_gates_passed"]


def _build_input_candidate_library(
    hook_result, story_result, viral_result, educational_result,
) -> List[Dict]:
    """Собирает все входящие кандидаты от всех режимов в один список для debug."""
    library: List[Dict] = []
    for src_mode, result_dict, key in [
        ("hook",        hook_result,        "hook_moments"),
        ("story",       story_result,       "story_moments"),
        ("viral",       viral_result,       "viral_moments"),
        ("educational", educational_result, "educational_moments"),
    ]:
        if not result_dict:
            continue
        moments = result_dict.get(key, []) or result_dict.get("moments", [])
        for m in moments:
            library.append({
                "source_mode": src_mode,
                "start": m.get("start") or m.get("start_sec", 0.0),
                "end": m.get("end") or m.get("end_sec", 0.0),
                "score": m.get("score") or m.get("virality_score") or m.get("hook_score", 0.0),
                "type": m.get("hook_type") or m.get("story_type") or m.get("viral_type")
                        or m.get("segment_type") or "",
                "title": m.get("title", ""),
            })
    return library


# =============================================================================
# PUBLIC API
# =============================================================================

def find_trailer_clips(
    video_path: str,
    video_duration_sec: float,
    hook_result: Optional[Dict] = None,
    story_result: Optional[Dict] = None,
    viral_result: Optional[Dict] = None,
    educational_result: Optional[Dict] = None,
    base_analysis: Optional[Dict] = None,
    asr_segments: Optional[List[Dict]] = None,
    config: Optional[TrailerModeConfig] = None,
) -> Dict:
    """
    Trailer Mode v3.0 — Smart Auto-Preview Director.
    Backward-compatible with v1, v2, v2.1.
    """
    logger.info("="*70)
    logger.info(f"TRAILER MODE v3.0 — {('PREVIEW' if (config or TrailerModeConfig()).assembly_mode == 'preview' else 'TRAILER')} Director")
    logger.info("="*70)

    cfg = config or TrailerModeConfig()
    video_dur = float(video_duration_sec)

    # v3.2: input candidate library для debug (все кандидаты до фильтрации)
    input_candidate_library = _build_input_candidate_library(
        hook_result, story_result, viral_result, educational_result,
    )

    # v3.2: degraded_preview — если нет hook И story, то trailer не может
    # нормально собрать preview. Работаем, но помечаем результат.
    has_hook = bool(hook_result and hook_result.get("hook_moments"))
    has_story = bool(story_result and story_result.get("story_moments"))
    is_degraded_preview = not (has_hook or has_story)

    # v3.2: адаптивный effective target_duration с учётом длительности видео
    effective_target, duration_cap_trace = _resolve_effective_target(cfg, video_dur)
    original_target = cfg.target_trailer_duration
    cfg.target_trailer_duration = effective_target
    logger.info(
        f"Effective target_duration: {effective_target:.1f}s "
        f"(original={original_target:.1f}s, video={video_dur:.1f}s, "
        f"reason={duration_cap_trace['reason']})"
    )
    if is_degraded_preview:
        logger.warning(
            "degraded_preview: hook and story results are both empty — "
            "result will be marked as needing manual review"
        )

    if all(r is None for r in [hook_result, story_result, viral_result, educational_result]):
        return _empty(video_dur, "no_mode_results", cfg)

    slots = build_trailer_plan(cfg)

    candidates = run_candidate_pipeline(
        hook_result, story_result, viral_result, educational_result,
        video_dur, cfg, base_analysis, slots,
    )
    if not candidates:
        return _empty(video_dur, "all_clips_filtered", cfg)

    # boundary refinement before scoring
    if cfg.enable_boundary_refinement:
        candidates = refine_preview_boundaries(candidates, asr_segments, base_analysis, cfg)

    n_before_nms = len(candidates)
    nms_pool = _nms(candidates, cfg.nms_iou_thresh)
    logger.info(f"NMS: {n_before_nms} → {len(nms_pool)}")

    theme_blocks = run_theme_pipeline(nms_pool, video_dur, cfg, base_analysis, asr_segments)

    # re-score after theme_value updated
    for c in nms_pool:
        compute_trailer_score(c, cfg, video_dur)
    nms_pool.sort(key=lambda c: c.trailer_score_final, reverse=True)

    # v3.1: precompute transition matrix so slot assignment uses O(1) lookup
    transition_matrix = precompute_transition_matrix(
        nms_pool, base_analysis, video_dur=video_dur
    )

    assignments = run_slot_pipeline(
        nms_pool, slots, cfg, base_analysis, transition_matrix, video_dur
    )
    assignments, edges = run_optimizer_pipeline(
        assignments, nms_pool, cfg, base_analysis, video_dur
    )

    # collect final clips (slot-selected first, then secondary fill)
    slot_selected = [a.selected for a in assignments if a.selected is not None]
    slot_assignment_before_nms = [
        {
            "slot": a.slot.slot_id,
            "candidate_id": getattr(a.selected, "candidate_id", None) if a.selected else None,
            "start": a.selected.start if a.selected else None,
            "end": a.selected.end if a.selected else None,
            "score": round(getattr(a.selected, "trailer_score_final", 0.0), 3) if a.selected else None,
            "source_mode": getattr(a.selected, "source_mode", None) if a.selected else None,
        }
        for a in assignments
    ]

    used_ids = {c.candidate_id for c in slot_selected}
    current_dur = sum(c.duration for c in slot_selected)

    for c in [x for x in nms_pool if x.candidate_id not in used_ids]:
        if current_dur >= cfg.target_trailer_duration: break
        if current_dur+c.duration > cfg.target_trailer_duration*cfg.fill_target_tolerance: continue
        if any(_iou((c.start,c.end),(k.start,k.end)) > cfg.nms_iou_thresh for k in slot_selected): continue
        if cfg.min_gap_between_clips > 0 and not _min_gap_ok(c, slot_selected, cfg.min_gap_between_clips): continue
        slot_selected.append(c); used_ids.add(c.candidate_id); current_dur+=c.duration

    # v3.2: SECOND-PASS NMS — IoU(0.35) + containment(0.60) + dedupe по candidate_id
    slot_selected, second_pass_trace = _second_pass_nms_and_dedupe(
        slot_selected, assignments,
        overlap_iou_thresh=0.35, containment_thresh=0.60, min_overlap_sec=2.0,
    )
    logger.info(
        f"Second-pass NMS: {second_pass_trace['input_count']} → {second_pass_trace['output_count']} "
        f"(dup_removed={len(second_pass_trace['duplicate_removed'])}, "
        f"overlap_removed={len(second_pass_trace['overlap_removed'])})"
    )

    # v3.3: HARD REPAIR — loop until sequence_valid or max_iter
    slot_selected, _hard_repair_trace = _repair_sequence_hard(
        slot_selected,
        effective_target=effective_target,
        min_overlap_sec=2.0,
        iou_thresh=0.25,
        containment_thresh=0.60,
        trim_tolerance_sec=1.5,
        max_iter=5,
    )
    logger.info(
        f"Hard repair: {_hard_repair_trace['input_count']} → {_hard_repair_trace['output_count']} "
        f"(overlap_removed={len(_hard_repair_trace['overlap_repair_trace'])}, "
        f"duration_ops={len(_hard_repair_trace['duration_trimmed'])}, "
        f"valid={_hard_repair_trace['sequence_valid_after']})"
    )

    final_candidates = sorted(slot_selected, key=lambda c: c.start)

    # hard caps
    hard_cap = cfg.target_trailer_duration * cfg.fill_target_tolerance
    if sum(c.duration for c in final_candidates) > hard_cap:
        by_s = sorted(final_candidates, key=lambda c: c.trailer_score_final, reverse=True)
        kept, acc = [], 0.0
        for c in by_s:
            if acc+c.duration<=hard_cap: kept.append(c); acc+=c.duration
        final_candidates = sorted(kept, key=lambda c: c.start)

    if cfg.max_clips_total > 0 and len(final_candidates) > cfg.max_clips_total:
        final_candidates = sorted(
            sorted(final_candidates, key=lambda c: c.trailer_score_final, reverse=True)[:cfg.max_clips_total],
            key=lambda c: c.start,
        )

    if not final_candidates:
        return _empty(video_dur, "no_clips_selected", cfg)

    ui = run_ui_payload_builder(assignments, theme_blocks, edges, nms_pool, cfg)
    trailer_clips = [_make_clip(c) for c in final_candidates]

    for i, tc in enumerate(trailer_clips):
        logger.info(
            f"  Clip #{i+1}: [{tc['source_mode']}/{tc['source_type']}] "
            f"{tc['start']:.1f}-{tc['end']:.1f}s  score={tc['score']:.3f}  "
            f"slot={tc['assigned_slot']}  spoiler={tc['spoiler_risk']:.2f}  "
            f"cut={tc['cut_naturalness_score']:.2f}  preview={tc['preview_value']:.2f}"
        )

    total_dur = round(sum(c["duration"] for c in trailer_clips), 1)
    scores = [c.trailer_score_final for c in final_candidates]
    spoiler_agg = round(float(np.mean([c.spoiler_risk for c in final_candidates])),3)
    curiosity_agg = round(float(np.mean([c.curiosity_gap_score for c in final_candidates])),3)
    tease_a = next((a for a in assignments if "tease" in a.slot.slot_id and a.slot.must_fill_if_possible), None)
    has_tease = tease_a is not None and tease_a.selected is not None
    tease_str = round(tease_a.selected.ending_strength,3) if has_tease and tease_a.selected else 0.0
    mode_counts = dict(Counter(c.source_mode for c in final_candidates))
    zone_counts = dict(Counter(_zone(c.start,c.end,video_dur) for c in final_candidates))

    edge_scores = [e.score for e in edges] if edges else []
    avg_tr = float(np.mean(edge_scores)) if edge_scores else 0.0
    min_tr = float(np.min(edge_scores)) if edge_scores else 0.0
    tr_ok = avg_tr >= cfg.optimization.min_avg_transition_score if edge_scores else True
    if edge_scores and not tr_ok:
        logger.warning(
            f"Transition chain below target: avg={avg_tr:.3f} "
            f"(min_floor={cfg.optimization.min_avg_transition_score:.2f}) — "
            f"review cuts / ASR / shot data on this asset"
        )

    render_instructions = [{"start":c["start"],"end":c["end"],"type":c["source_mode"],
                             "score":c["score"],"reasons":c["reasons"]} for c in trailer_clips]

    # v3.2: slot_assignment_after_nms — что осталось в финальной раскладке
    slot_assignment_after_nms = [
        {
            "candidate_id": getattr(c, "candidate_id", None),
            "start": c.start, "end": c.end,
            "score": round(getattr(c, "trailer_score_final", 0.0), 3),
            "source_mode": getattr(c, "source_mode", None),
            "assigned_slot": getattr(c, "assigned_slot", None),
        }
        for c in final_candidates
    ]

    # v3.2: final_sequence_validation — набор проверок финальной последовательности
    boundary_scores = [c.get("cut_naturalness_score", 0.0) for c in trailer_clips]
    boundary_ok_avg = float(np.mean(boundary_scores)) if boundary_scores else 0.0
    has_overlap = any(
        _iou((trailer_clips[i]["start"], trailer_clips[i]["end"]),
             (trailer_clips[j]["start"], trailer_clips[j]["end"])) > 0.0
        for i in range(len(trailer_clips))
        for j in range(i + 1, len(trailer_clips))
    )
    candidate_ids_final = [c.get("candidate_id") for c in trailer_clips]
    has_duplicates = len(candidate_ids_final) != len(set(candidate_ids_final))

    final_sequence_validation = {
        "num_clips": len(trailer_clips),
        "total_duration_sec": total_dur,
        "effective_target_sec": effective_target,
        "target_fill_pct": round(total_dur / max(effective_target, 1e-6) * 100, 1),
        "duration_exceeds_target": total_dur > effective_target * 1.15,
        "duration_exceeds_hard_cap": total_dur > effective_target * 1.5,
        "has_time_overlap": has_overlap,
        "has_duplicate_candidates": has_duplicates,
        "num_themes": len(theme_blocks),
        "avg_boundary_quality": round(boundary_ok_avg, 3),
        "is_degraded_preview": is_degraded_preview,
        "sequence_valid": (
            not has_overlap
            and not has_duplicates
            and total_dur <= effective_target * 1.15
            and len(trailer_clips) > 0
        ),
    }

    # v3.2: export_decision — единая логика по спеке F
    export_decision, export_reasons = _classify_trailer_export_decision(
        trailer_clips,
        total_duration=total_dur,
        effective_target=effective_target,
        num_themes=len(theme_blocks),
        curiosity_agg=curiosity_agg,
        spoiler_agg=spoiler_agg,
        boundary_ok_avg=boundary_ok_avg,
        is_degraded_preview=is_degraded_preview,
        overlap_removed_count=len(second_pass_trace["overlap_removed"]),
    )

    result: Dict[str,Any] = {
        "mode": "trailer",
        "assembly_mode": cfg.assembly_mode,
        "trailer_clips": trailer_clips,
        "render_instructions": render_instructions,
        # v3.2: debug artifacts
        "input_candidate_library": input_candidate_library,
        "slot_assignment_before_nms": slot_assignment_before_nms,
        "slot_assignment_after_nms": slot_assignment_after_nms,
        "duplicate_removed": second_pass_trace["duplicate_removed"],
        "overlap_removed": second_pass_trace["overlap_removed"],
        "duration_cap_trace": duration_cap_trace,
        "final_sequence_validation": final_sequence_validation,
        "final_repair_trace": _hard_repair_trace,
        "overlap_repair_trace": _hard_repair_trace["overlap_repair_trace"],
        "duration_trimmed": _hard_repair_trace["duration_trimmed"],
        "is_degraded_preview": is_degraded_preview,
        "export_decision": export_decision,
        "export_decision_reasons": export_reasons,
        "theme_blocks": [
            {"theme_id":b.theme_id,"label":b.label,"constructor_label":b.constructor_label,
             "start":round(b.start,2),"end":round(b.end,2),"role":b.role,
             "importance":round(b.importance,3),"spoiler_level":round(b.spoiler_level,3),
             "semantic_signature":b.semantic_signature[:5],
             "scene_signature":b.scene_signature[:3],
             "dominant_modes":b.dominant_source_modes,
             "speaker_signature":b.speaker_signature,
             "replaceable":b.replaceable,"alternative_theme_ids":b.alternative_theme_ids,
             "best_candidate_ids":b.best_candidate_ids,
             "topic_confidence":round(b.topic_confidence,3),
             "coherence_with_neighbors":round(b.coherence_with_neighbors,3)}
            for b in theme_blocks
        ],
        "assembly_plan":    ui.get("assembly_plan",[]),
        "alternatives":     ui.get("alternatives",{}),
        "theme_alternatives": ui.get("theme_alternatives",{}),
        "transition_graph": ui.get("transition_graph",[]),
        "explanations":     ui.get("explanations",[]),
        "edit_contract":    ui.get("edit_contract",{}),
        "editable_actions": ui.get("editable_actions",{}),
        "theme_map":        ui.get("theme_map",[]),
        "boundary_diagnostics": {
            c["candidate_id"]: {
                "refined_start": c["scene_safe_start"],
                "refined_end": c["scene_safe_end"],
                "cut_naturalness": c["cut_naturalness_score"],
            }
            for c in trailer_clips
        },
        "stats": {
            "total_duration": video_dur, "target_duration": cfg.target_trailer_duration,
            "trailer_duration": total_dur, "num_clips": len(trailer_clips),
            "fill_pct": round(total_dur/max(cfg.target_trailer_duration,1e-6)*100,1),
            "max_trailer_score": round(max(scores),3) if scores else 0.0,
            "avg_trailer_score": round(float(np.mean(scores)),3) if scores else 0.0,
            "n_candidates_before_nms": n_before_nms,
            "num_themes": len(theme_blocks),
            "aggregate_spoiler_risk": spoiler_agg,
            "aggregate_curiosity_gap": curiosity_agg,
            "has_tease_end": has_tease,
            "tease_end_strength": tease_str,
            "ending_open_loop_score": tease_str,
            "mode_counts": mode_counts, "zone_counts": zone_counts,
            "slots_filled": sum(1 for a in assignments if a.selected),
            "slots_total": len(assignments),
            "fallback_slots": sum(1 for a in assignments if a.used_fallback and a.selected),
            "assembly_mode": cfg.assembly_mode,
            "slot_template": cfg.slot.template,
            "profile_name": f"{cfg.mode_name} {cfg.profile_version}",
            "avg_transition_score": round(avg_tr, 3),
            "min_transition_score": round(min_tr, 3),
            "transition_edge_count": len(edge_scores),
            "transition_quality_ok": tr_ok,
            # v3.2: debug / export
            "is_degraded_preview": is_degraded_preview,
            "export_decision": export_decision,
            "export_decision_reasons": export_reasons,
            "effective_target_duration": effective_target,
            "original_target_duration": original_target,
            "n_input_candidates": len(input_candidate_library),
            "n_duplicates_removed": len(second_pass_trace["duplicate_removed"]),
            "n_overlaps_removed": len(second_pass_trace["overlap_removed"]),
            "sequence_valid": final_sequence_validation["sequence_valid"],
        },
    }

    logger.info(
        f"Assembled: {len(trailer_clips)} clips, {total_dur:.1f}s "
        f"(effective_target={effective_target:.1f}s) | "
        f"mode={cfg.assembly_mode} spoiler={spoiler_agg:.2f} curiosity={curiosity_agg:.2f} "
        f"themes={len(theme_blocks)} tease={has_tease} "
        f"degraded={is_degraded_preview} export={export_decision}"
    )
    logger.info("="*70)
    return result


# =============================================================================
# UI BADGES
# =============================================================================

UI_TRAILER_BADGES: Dict[str,Any] = {
    "source_mode_labels": {
        "hook":{"text":"Хук","color":"#3B82F6"},
        "story":{"text":"История","color":"#8B5CF6"},
        "viral":{"text":"Вирал","color":"#EF4444"},
        "educational":{"text":"Обучение","color":"#10B981"},
    },
    "slot_role_labels": {
        "open_hook":{"text":"Открытие","color":"#F59E0B"},
        "premise":{"text":"Контекст","color":"#6366F1"},
        "escalation":{"text":"Нагнетание","color":"#EC4899"},
        "proof_moment":{"text":"Доказат.","color":"#14B8A6"},
        "tease_end":{"text":"Тизер","color":"#F97316"},
    },
    "preview_intent_labels": {
        "hook":{"text":"Захват","color":"#F59E0B"},
        "context":{"text":"Смысл","color":"#6366F1"},
        "escalation":{"text":"Ценность","color":"#EC4899"},
        "tease":{"text":"Интрига","color":"#F97316"},
        "depth":{"text":"Глубина","color":"#84CC16"},
    },
    "assembly_mode_labels": {
        "trailer":{"text":"Trailer","color":"#DC2626"},
        "preview":{"text":"Preview","color":"#2563EB"},
    },
}


# =============================================================================
# SMOKE TEST
# =============================================================================

if __name__ == "__main__":
    _trailer_cfg = TrailerModeConfig(
        mode_name="smoke_trailer",
        target_trailer_duration=60.0,
    )
    _preview_cfg = TrailerModeConfig.for_preview(
        mode_name="smoke_preview",
        target_trailer_duration=45.0,
    )

    _hook = {"hook_moments": [
        {"start":0.5,"end":5.0,"duration":4.5,"score":0.78,
         "hook_type":"question_hook","title":"А вы знали?",
         "summary":"Что если 70% делают это неправильно?",
         "reasons":[{"code":"strong_question","message":"Сильный вопрос","weight":0.85}]},
        {"start":8.0,"end":13.0,"duration":5.0,"score":0.63,
         "hook_type":"curiosity_hook","title":"Секрет",
         "summary":"Это скрывают от большинства","reasons":[]},
    ]}
    _story = {"story_moments": [
        {"start":18.0,"end":36.0,"duration":18.0,"score":0.70,
         "story_type":"man_in_hole","title":"Кризис",
         "summary":"Герой сталкивается с проблемой","reasons":[]},
        {"start":60.0,"end":78.0,"duration":18.0,"score":0.55,
         "story_type":"resolution","title":"В итоге всё решилось",
         "summary":"Разгадка найдена и вот ответ","reasons":[]},
    ]}
    _viral = {"viral_moments": [
        {"start":40.0,"end":52.0,"duration":12.0,"score":0.76,
         "viral_type":"wow_moment","title":"Неожиданный поворот",
         "summary":"Никто не ожидал такого","reasons":[]},
    ]}
    _edu = {"educational_moments": [
        {"start":24.0,"end":38.0,"duration":14.0,"score":0.64,
         "segment_type":"explanation","title":"Объяснение механизма",
         "summary":"Как именно это работает шаг за шагом","reasons":[]},
    ]}

    for mode_label, cfg in [("TRAILER", _trailer_cfg), ("PREVIEW", _preview_cfg)]:
        print(f"\n{'='*60}")
        print(f"Testing {mode_label} mode")
        r = find_trailer_clips(
            video_path="test.mp4", video_duration_sec=120.0,
            hook_result=_hook, story_result=_story,
            viral_result=_viral, educational_result=_edu,
            config=cfg,
        )
        print(f"Clips: {len(r['trailer_clips'])}, themes: {r['stats']['num_themes']}")
        for c in r["trailer_clips"]:
            print(
                f"  [{c['source_mode']}/{c['source_type']}] "
                f"{c['start']:.1f}-{c['end']:.1f}s  score={c['score']:.3f}  "
                f"slot={c['assigned_slot']}  spoiler={c['spoiler_risk']:.2f}  "
                f"cut={c['cut_naturalness_score']:.2f}  "
                f"preview={c['preview_value']:.2f}  ending={c['ending_strength']:.2f}  "
                f"spoiler_cat={c['spoiler_category']}"
            )
        print(f"Transitions: {len(r['transition_graph'])}")
        for t in r["transition_graph"]:
            print(f"  {t['from_slot']}→{t['to_slot']} score={t['score']:.3f}  {t['why_this_transition']}")
        stats = r["stats"]
        for k in ("has_tease_end","tease_end_strength","aggregate_spoiler_risk",
                  "aggregate_curiosity_gap","slots_filled","slots_total","fallback_slots"):
            print(f"  {k}: {stats[k]}")

    print("\nALL OK")
