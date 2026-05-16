"""
non_content_detector.py — promo avoidance generation v1.

Detects non-content zones (intro / promo / sponsor / outro / filler) using:
  1. Structural weak signals (always available).
  2. Optional CLIP visual classification (requires open_clip_torch).

Usage:
    from non_content_detector import build_non_content_segments, NonContentConfig, save_non_content_artifacts
    cfg = NonContentConfig(detector="auto")
    segs = build_non_content_segments(video_path, asr_segments, video_duration_sec, cfg)
    save_non_content_artifacts(shared_dir, segs, cfg)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

NON_CONTENT_LABELS = {"intro_animation", "channel_promo", "sponsor_ad", "outro_endcard", "filler_or_music"}
ALL_LABELS = {"main_content", "unknown"} | NON_CONTENT_LABELS

# ── CLIP text prompts per label ───────────────────────────────────────────────

_CLIP_PROMPTS: Dict[str, List[str]] = {
    "main_content": [
        "main content of a YouTube video, interview, lecture, explanation",
        "person speaking about the main topic",
        "educational or story content, normal video segment",
    ],
    "intro_animation": [
        "channel intro animation, logo title screen, opening sequence",
        "YouTube intro screen with logo graphics",
        "animated intro or title card",
    ],
    "channel_promo": [
        "subscribe screen, social media handles, channel promotion",
        "YouTube channel promo graphic with subscribe button",
        "creator self promotion screen",
    ],
    "sponsor_ad": [
        "sponsored advertisement, product promotion, brand logo, promo code",
        "person promoting a product or app",
        "advertisement segment in a video",
    ],
    "outro_endcard": [
        "YouTube end screen with video cards and subscribe button",
        "outro screen, end card, final screen",
        "end of video with suggested videos",
    ],
    "filler_or_music": [
        "b-roll montage, music interlude, filler footage",
        "cinematic filler without main speech",
        "transition montage",
    ],
    "unknown": [
        "unclear video frame",
    ],
}


@dataclass
class NonContentConfig:
    detector: str = "auto"           # auto | structural | clip | off
    window_sec: float = 8.0
    stride_sec: float = 4.0
    frames_per_window: int = 3
    confidence_threshold: float = 0.30
    strict_real: bool = False
    # Structural thresholds
    intro_risk_duration_min: float = 180.0   # video length to trigger intro risk
    intro_risk_window_sec: float = 45.0
    outro_risk_window_sec: float = 60.0
    low_speech_window_sec: float = 10.0
    low_speech_coverage_threshold: float = 0.15   # ASR coverage fraction below which = filler


def _open_clip_available() -> bool:
    try:
        import open_clip  # noqa: F401
        return True
    except ImportError:
        return False


# ── Structural weak detector ─────────────────────────────────────────────────

def detect_structural_non_content(
    asr_segments: List[Dict[str, Any]],
    video_duration_sec: float,
    cfg: NonContentConfig,
) -> List[Dict[str, Any]]:
    """Return weak structural risk segments (no visual analysis)."""
    results: List[Dict[str, Any]] = []
    dur = max(video_duration_sec, 0.0)

    # Intro risk: early window of longer videos
    if dur >= cfg.intro_risk_duration_min:
        intro_end = min(cfg.intro_risk_window_sec, dur)
        if intro_end > 0:
            results.append({
                "start_sec": 0.0,
                "end_sec": round(intro_end, 2),
                "label": "intro_animation",
                "confidence": 0.45,
                "source": "structural_weak",
                "reason": "early_video_intro_risk",
                "metadata": {"video_duration_sec": dur},
            })

    # Outro risk: late window of longer videos
    if dur >= cfg.intro_risk_duration_min:
        outro_start = max(dur - cfg.outro_risk_window_sec, 0.0)
        if outro_start < dur:
            results.append({
                "start_sec": round(outro_start, 2),
                "end_sec": round(dur, 2),
                "label": "outro_endcard",
                "confidence": 0.45,
                "source": "structural_weak",
                "reason": "late_video_outro_risk",
                "metadata": {"video_duration_sec": dur},
            })

    # Low speech density: scan windows
    if dur > 0:
        pos = 0.0
        w = cfg.low_speech_window_sec
        while pos + w <= dur + 1e-6:
            win_end = min(pos + w, dur)
            win_start = pos

            # Compute ASR coverage in this window
            coverage = 0.0
            for seg in asr_segments:
                try:
                    seg_s = float(seg.get("start", 0.0))
                    seg_e = float(seg.get("end", 0.0))
                except (TypeError, ValueError):
                    continue
                overlap = max(0.0, min(seg_e, win_end) - max(seg_s, win_start))
                coverage += overlap

            win_len = win_end - win_start
            coverage_ratio = coverage / win_len if win_len > 0 else 0.0

            if coverage_ratio < cfg.low_speech_coverage_threshold:
                results.append({
                    "start_sec": round(win_start, 2),
                    "end_sec": round(win_end, 2),
                    "label": "filler_or_music",
                    "confidence": 0.50,
                    "source": "structural_weak",
                    "reason": "low_speech_density",
                    "metadata": {
                        "asr_coverage_ratio": round(coverage_ratio, 4),
                        "window_sec": w,
                    },
                })
            pos += cfg.stride_sec

    return results


# ── CLIP visual detector ──────────────────────────────────────────────────────

def detect_visual_clip_non_content(
    video_path: "str | Path",
    video_duration_sec: float,
    cfg: NonContentConfig,
) -> List[Dict[str, Any]]:
    """
    Optional CLIP-based visual classification. Returns [] if CLIP unavailable.
    Raises RuntimeError only if cfg.strict_real=True and clip mode was requested.
    """
    if not _open_clip_available():
        if cfg.strict_real and cfg.detector == "clip":
            raise RuntimeError(
                "open_clip_torch is not installed. Run: pip install open_clip_torch"
            )
        logger.info("[non_content] open_clip_torch not available, skipping visual classification")
        return []

    try:
        import open_clip
        import torch
        import cv2 as _cv2
    except ImportError as exc:
        logger.warning("[non_content] CLIP dependencies missing (%s), skipping", exc)
        return []

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai", device=device
        )
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        model.eval()
    except Exception as exc:
        logger.warning("[non_content] CLIP model load failed (%s), skipping", exc)
        return []

    # Pre-encode all text prompts
    label_order = list(_CLIP_PROMPTS.keys())
    all_texts: List[str] = []
    label_indices: List[Tuple[str, int, int]] = []  # (label, start_idx, end_idx)
    for label in label_order:
        start = len(all_texts)
        all_texts.extend(_CLIP_PROMPTS[label])
        label_indices.append((label, start, len(all_texts)))

    with torch.no_grad():
        tokens = tokenizer(all_texts).to(device)
        text_features = model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    results: List[Dict[str, Any]] = []
    dur = max(video_duration_sec, 0.0)
    cap = _cv2.VideoCapture(str(video_path))
    fps = cap.get(_cv2.CAP_PROP_FPS) or 25.0

    import numpy as np
    from PIL import Image

    pos = 0.0
    while pos < dur:
        win_end = min(pos + cfg.window_sec, dur)
        frame_times = [
            pos + (win_end - pos) * i / max(cfg.frames_per_window - 1, 1)
            for i in range(cfg.frames_per_window)
        ]

        frame_features_list: List[Any] = []
        for ft in frame_times:
            cap.set(_cv2.CAP_PROP_POS_MSEC, ft * 1000)
            ret, frame = cap.read()
            if not ret:
                continue
            pil_img = Image.fromarray(_cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB))
            img_t = preprocess(pil_img).unsqueeze(0).to(device)
            with torch.no_grad():
                feats = model.encode_image(img_t)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            frame_features_list.append(feats)

        if not frame_features_list:
            pos += cfg.stride_sec
            continue

        with torch.no_grad():
            img_feat = torch.cat(frame_features_list, dim=0).mean(dim=0, keepdim=True)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            sims = (img_feat @ text_features.T).squeeze(0).cpu().float().numpy()

        # Aggregate per label (mean of prompt sims)
        label_scores: Dict[str, float] = {}
        for label, si, ei in label_indices:
            label_scores[label] = float(np.mean(sims[si:ei]))

        # Softmax-like normalisation over all labels
        raw_scores = np.array([label_scores[l] for l in label_order])
        exp_s = np.exp(raw_scores - raw_scores.max())
        probs = exp_s / exp_s.sum()
        best_idx = int(np.argmax(probs))
        best_label = label_order[best_idx]
        best_conf = float(probs[best_idx])

        if best_label in NON_CONTENT_LABELS and best_conf >= cfg.confidence_threshold:
            results.append({
                "start_sec": round(pos, 2),
                "end_sec": round(win_end, 2),
                "label": best_label,
                "confidence": round(best_conf, 4),
                "source": "visual_clip",
                "reason": "visual_window_classification",
                "metadata": {
                    "window_sec": cfg.window_sec,
                    "stride_sec": cfg.stride_sec,
                    "frames_per_window": cfg.frames_per_window,
                    "label_probs": {l: round(float(p), 4) for l, p in zip(label_order, probs)},
                },
            })

        pos += cfg.stride_sec

    cap.release()
    return results


# ── Merge overlapping segments ────────────────────────────────────────────────

def merge_non_content_segments(
    segments: List[Dict[str, Any]],
    gap_merge_sec: float = 1.0,
) -> List[Dict[str, Any]]:
    """Merge adjacent/overlapping segments with the same label."""
    if not segments:
        return []

    by_label: Dict[str, List[Dict[str, Any]]] = {}
    for seg in segments:
        lbl = seg.get("label", "unknown")
        by_label.setdefault(lbl, []).append(seg)

    merged: List[Dict[str, Any]] = []
    for label, segs in by_label.items():
        ordered = sorted(segs, key=lambda s: float(s.get("start_sec", 0.0)))
        cur = dict(ordered[0])
        for nxt in ordered[1:]:
            nxt_s = float(nxt.get("start_sec", 0.0))
            nxt_e = float(nxt.get("end_sec", 0.0))
            cur_e = float(cur.get("end_sec", 0.0))
            cur_s = float(cur.get("start_sec", 0.0))
            if nxt_s <= cur_e + gap_merge_sec:
                # Merge: extend end, take max confidence, combine sources
                cur["end_sec"] = max(cur_e, nxt_e)
                cur["confidence"] = max(float(cur.get("confidence", 0.0)),
                                        float(nxt.get("confidence", 0.0)))
                existing_src = cur.get("source", "")
                new_src = nxt.get("source", "")
                if new_src and new_src not in existing_src:
                    cur["source"] = f"{existing_src}|{new_src}" if existing_src else new_src
            else:
                merged.append(cur)
                cur = dict(nxt)
        merged.append(cur)

    return sorted(merged, key=lambda s: float(s.get("start_sec", 0.0)))


# ── Top-level build function ──────────────────────────────────────────────────

def build_non_content_segments(
    video_path: "str | Path",
    asr_segments: List[Dict[str, Any]],
    video_duration_sec: float,
    cfg: NonContentConfig,
) -> List[Dict[str, Any]]:
    """Build merged list of non-content segments using the configured detector."""
    if cfg.detector == "off":
        return []

    structural = detect_structural_non_content(asr_segments, video_duration_sec, cfg)

    clip_segs: List[Dict[str, Any]] = []
    use_clip = False
    if cfg.detector in ("clip", "auto"):
        if _open_clip_available():
            use_clip = True
        elif cfg.detector == "clip" and cfg.strict_real:
            raise RuntimeError(
                "open_clip_torch not installed; --non-content-detector clip requires it. "
                "Run: pip install open_clip_torch"
            )
        else:
            logger.info("[non_content] CLIP requested but not available, using structural only")

    if use_clip:
        try:
            clip_segs = detect_visual_clip_non_content(video_path, video_duration_sec, cfg)
        except Exception as exc:
            if cfg.strict_real:
                raise
            logger.warning("[non_content] CLIP detection failed (%s), falling back to structural", exc)

    all_segs = structural + clip_segs
    return merge_non_content_segments(all_segs)


# ── Artifact save ─────────────────────────────────────────────────────────────

def save_non_content_artifacts(
    shared_dir: "str | Path",
    segments: List[Dict[str, Any]],
    cfg: NonContentConfig,
    used_for_filtering: bool = True,
) -> None:
    """Write shared/non_content_segments.json and shared/non_content_manifest.json."""
    out = Path(shared_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "non_content_segments.json", "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)

    labels_used = sorted({s.get("label", "unknown") for s in segments})
    sources_used = sorted({
        src
        for s in segments
        for src in s.get("source", "").split("|")
        if src
    })

    manifest = {
        "policy": "promo_avoidance_generation_v1",
        "detector": cfg.detector,
        "segments_count": len(segments),
        "sources": sources_used,
        "labels": labels_used,
        "used_for_candidate_filtering": used_for_filtering,
    }
    with open(out / "non_content_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
