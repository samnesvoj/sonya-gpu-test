# SONYA GPU Test — Audit of Old SONYA-DATASET Modules

**Date:** 2026-04-28  
**Auditor:** Cursor (automated review)  
**Source dir:** `SONYA-DATASET/scripts/`  
**Target repo:** `sonya-gpu-test-upload/`

---

## Summary Table

| File | Useful content | Used in current repo? | Action | Integration risk |
|------|---------------|----------------------|--------|-----------------|
| `asr.py` | `transcribe_video`, `segments_to_windows`, `get_first_n_seconds_text` — unified ASR with Whisper cache | **YES** — already in `scripts/asr.py` (same file, slightly newer) | Already integrated ✓ | Low |
| `asr_production.py` | `transcribe_video_segment`, `transcribe_full_video`, `get_asr_segments_production` — stable ffmpeg→wav→Whisper (no pipe issues) | NO — not in current repo | Optional: use if Whisper pipe fails on Vast. Replace `run_asr()` in benchmark_runner. | Low |
| `asr_transcribe.py` | `transcribe_video`, `segments_to_windows`, `get_first_n_seconds_text` — older version, subset of current `asr.py` | NO — superseded by `asr.py` | Keep as archive. Functions already in `asr.py`. | None |
| `llm_segment_analysis.py` | `analyze_educational`, `analyze_story_arc`, `analyze_hook` — OpenRouter LLM for semantic scoring | **INTEGRATED** (2026-04-28) — copied to `scripts/` | Done ✓ | Low (requires OPENROUTER_API_KEY in .env) |
| `real_visual_metrics.py` | `compute_all_frame_metrics` — optical flow (action_intensity), Haar face cascade (emotional_peaks), rule-of-thirds (composition_score) | **INTEGRATED** (2026-04-28) — copied to `scripts/`, connected in `benchmark_runner.run_yolo_analysis()` | Done ✓ | Low — graceful fallback if cv2 missing. `has_real_visual_metrics` flag in output. |
| `sonya_yolo_analyzer.py` | `SonyaYOLOAnalyzer` — full YOLO track(), velocities, viral moments, head tracking | NO — requires `head_tracker_kalman.py`, `head_tracker_detectors.py` (not in repo) | Do NOT copy without dependencies. Integrate after head_tracker files are ported. | HIGH — 8+ dependency files missing |
| `modes_scoring_v2.py` | Visual-only fallback (no ASR): scene changes, optical flow, face detection for all 5 modes | NO — not in current repo | Optional: useful for silent/music videos. Import `find_viral_moments`, `find_hooks` as no-ASR fallback. | Medium — depends on `modes_scoring.py` functions |
| `fast_modes_scoring_v2.py` | Optimized version of modes_scoring_v2 (skip-frame scene detection, low-res optical flow) | NO | Archive. Use only if modes_scoring_v2 is too slow on long videos. | Medium |
| `cut_clips_from_result.py` | `cut_clips_programmatically` — ffmpeg clip cutting from JSON result, social crop, parallel workers | NO | Future integration: post-benchmark clip export step. Not needed for GPU test. | Low |
| `trailer_ab_test.py` | A/B test harness for trailer variants | NO — file not present in SONYA-DATASET (name mentioned, not found) | Skip | N/A |
| `trailer_health_check.py` | Batch sanity-check for Trailer Mode: reads JSON results, computes fill%, overfill, clip counts | NO — standalone CLI tool | Keep as archive. Run manually: `python trailer_health_check.py outputs/` | None |
| `audio_topic_segmentation.py` | Audio-based topic segmentation (VAD + silence gaps) | NOT FOUND in SONYA-DATASET/scripts | Likely integrated into `asr.py` or `modes_scoring.py` already | N/A |

---

## Bugs Fixed in This Session

### 1. `hook_mode_v1.py` — KeyError: `'intensity'`
**Root cause:** `weak_hook_fallback_used` path created `weak_moment` dicts at line 4149 without the `"intensity"` key, but line 4263 accessed `m["intensity"]` for ALL hook_candidates including these.  
**Fix:** Added `"intensity": round(float(wk.get("intensity", 0.0)), 3)` to weak_moment dict. Changed line 4263 to `.get("intensity", 0.0)` as safety net.

### 2. `educational_mode_v5.py` — 0 candidates when threshold kills all
**Root cause:** Step 5 adaptive threshold (75th percentile × 0.90) can still be higher than all window scores. Step 6 then returns empty `above_threshold`.  
**Fix:** After step 6, if `above_threshold` is empty AND there are eligible windows, promote top-3 as `manual_review` fallback (`_weak_edu_fallback_used = True`). Logged as warning.

### 3. `llm_segment_analysis.py` — "not available" in viral/modes_scoring
**Root cause:** File was in SONYA-DATASET but not in the GPU test repo's `scripts/`.  
**Fix:** Copied to `sonya-gpu-test-upload/scripts/llm_segment_analysis.py`. `modes_scoring.py` already has correct try/import logic.

### 4. `real_visual_metrics.py` — placeholder visual features
**Root cause:** `compute_all_frame_metrics` existed in SONYA-DATASET but wasn't in the GPU repo or called anywhere.  
**Fix:** Copied to `scripts/`. Integrated into `benchmark_runner.run_yolo_analysis()` — now computes optical flow, face detection, and rule-of-thirds per sampled frame. Results stored as `action_intensity`, `emotional_peaks`, `composition_score` in each detection entry.

---

## What Stays in Archive (SONYA-DATASET, NOT copied to GPU repo)

- `sonya_yolo_analyzer.py` — needs head_tracker deps
- `modes_scoring_v2.py`, `fast_modes_scoring_v2.py` — useful for silent video fallback (future)
- `cut_clips_from_result.py` — clip export tool (post-benchmark step)
- `trailer_health_check.py` — manual QA tool
- `asr_transcribe.py` — superseded by `asr.py`
- `asr_production.py` — optional if pipe issues arise on Vast

---

## What Still Needs Work (next iteration)

| Issue | File | Priority |
|-------|------|----------|
| `viral` mode: llm_segment_analysis now available but verify no other fallback | `modes_scoring.py` | P1 |
| `sonya_yolo_analyzer.py` full integration | needs `head_tracker_kalman.py` | P2 |
| Add `modes_scoring_v2` as ASR-less fallback for silent videos | `modes_scoring_v2.py` | P2 |
| `trailer_mode_v3` input quality depends on hook/story/educational scores | verify after P1 fixes | P1 |
| Verify `real_visual_metrics` improves educational scores on Vast GPU run | `benchmark_runner.py` | P1 |
