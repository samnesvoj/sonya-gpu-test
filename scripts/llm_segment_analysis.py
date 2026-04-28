"""
LLM-анализ текста сегментов для режимов educational, story, hook.
Использует OpenRouter (текстовая модель), ключ из .env.local.
"""
import os
import json
import re
from pathlib import Path
from typing import Dict, Optional

# загрузка .env.local (опционально)
try:
    _root = Path(__file__).resolve().parent.parent
    from dotenv import load_dotenv
    load_dotenv(_root / ".env.local")
    load_dotenv(_root / ".env")
except ImportError:
    pass

OPENROUTER_API = "https://openrouter.ai/api/v1/chat/completions"
# дешёвая модель для текста (JSON)
DEFAULT_MODEL = "openai/gpt-4o-mini"


def _get_api_key() -> str:
    return (os.getenv("OPENROUTER_API_KEY") or "").strip()


def _call_llm(prompt: str, max_tokens: int = 500, model: str = DEFAULT_MODEL) -> Optional[str]:
    key = _get_api_key()
    if not key:
        return None
    try:
        import requests
        r = requests.post(
            OPENROUTER_API,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/sonya-dataset",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.2,
            },
            timeout=30,
        )
        if r.status_code != 200:
            return None
        return r.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    except Exception:
        return None


def _parse_json_from_response(text: str) -> Optional[Dict]:
    if not text:
        return None
    # вытаскиваем первый JSON-объект
    m = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return None


def analyze_educational(segment_text: str) -> Dict:
    """
    Анализ образовательной ценности сегмента.
    Возвращает has_insight, key_takeaway, insight_quality, explanation_clarity, practical_value, structure_score, content_type.
    """
    prompt = f'''Analyze this educational content segment. Reply with ONLY a JSON object, no other text.

Text: "{segment_text[:2000]}"

JSON format:
{{
  "has_insight": true or false,
  "key_takeaway": "one sentence summary or empty string",
  "insight_quality": 0.0 to 1.0,
  "explanation_clarity": 0.0 to 1.0,
  "practical_value": 0.0 to 1.0,
  "structure_score": 0.0 to 1.0,
  "content_type": "insight" or "example" or "explanation" or "definition" or "none"
}}

Be strict: set has_insight=true only if there is a clear, actionable takeaway.'''

    raw = _call_llm(prompt)
    data = _parse_json_from_response(raw) if raw else None
    if not data:
        return {"has_insight": False, "key_takeaway": "", "insight_quality": 0.5, "explanation_clarity": 0.5, "practical_value": 0.5, "structure_score": 0.5, "content_type": "none"}
    return {
        "has_insight": bool(data.get("has_insight", False)),
        "key_takeaway": str(data.get("key_takeaway", ""))[:500],
        "insight_quality": float(data.get("insight_quality", 0.5)),
        "explanation_clarity": float(data.get("explanation_clarity", 0.5)),
        "practical_value": float(data.get("practical_value", 0.5)),
        "structure_score": float(data.get("structure_score", 0.5)),
        "content_type": str(data.get("content_type", "none")),
    }


def analyze_story_arc(window_text: str) -> Dict:
    """
    Проверка наличия полного story arc (setup → conflict → resolution).
    """
    prompt = f'''Analyze if this is a complete story. Reply with ONLY a JSON object.

Text: "{window_text[:3000]}"

A complete story has: 1) SETUP (who/what/where), 2) CONFLICT (problem arises), 3) RESOLUTION (outcome).

JSON format:
{{
  "is_complete_story": true or false,
  "setup_quality": 0.0 to 1.0,
  "conflict_intensity": 0.0 to 1.0,
  "resolution_quality": 0.0 to 1.0,
  "has_twist": true or false,
  "arc_type": "hero" or "tragedy" or "comedy" or "revelation" or "none",
  "story_summary": "max 2 sentences",
  "emotional_arc_score": 0.0 to 1.0
}}'''

    raw = _call_llm(prompt)
    data = _parse_json_from_response(raw) if raw else None
    if not data:
        return {"is_complete_story": False, "setup_quality": 0.5, "conflict_intensity": 0.5, "resolution_quality": 0.5, "has_twist": False, "arc_type": "none", "story_summary": "", "emotional_arc_score": 0.5}
    return {
        "is_complete_story": bool(data.get("is_complete_story", False)),
        "setup_quality": float(data.get("setup_quality", 0.5)),
        "conflict_intensity": float(data.get("conflict_intensity", 0.5)),
        "resolution_quality": float(data.get("resolution_quality", 0.5)),
        "has_twist": bool(data.get("has_twist", False)),
        "arc_type": str(data.get("arc_type", "none")),
        "story_summary": str(data.get("story_summary", ""))[:300],
        "emotional_arc_score": float(data.get("emotional_arc_score", 0.5)),
    }


def analyze_hook(first_seconds_text: str) -> Dict:
    """
    Анализ силы хука (первые 5 сек): intrigue, emotion, hook_type.
    """
    prompt = f'''Rate the hook quality of this video opening (first few seconds). Reply with ONLY a JSON object.

Text: "{first_seconds_text[:500]}"

JSON format:
{{
  "hook_type": "question" or "provocation" or "statement" or "emotion" or "mystery" or "none",
  "intrigue_score": 0.0 to 1.0,
  "emotional_intensity": 0.0 to 1.0,
  "curiosity_gap": 0.0 to 1.0
}}

Good hooks: "You won't believe...", "Biggest mistake...", "Watch till end...". Bad: "Hey guys", "Today I want to talk about..."'''

    raw = _call_llm(prompt)
    data = _parse_json_from_response(raw) if raw else None
    if not data:
        return {"hook_type": "none", "intrigue_score": 0.5, "emotional_intensity": 0.5, "curiosity_gap": 0.5}
    return {
        "hook_type": str(data.get("hook_type", "none")),
        "intrigue_score": float(data.get("intrigue_score", 0.5)),
        "emotional_intensity": float(data.get("emotional_intensity", 0.5)),
        "curiosity_gap": float(data.get("curiosity_gap", 0.5)),
    }


if __name__ == "__main__":
    t = "In this video I explain why most people fail at learning languages. The key is consistency."
    print("Educational:", analyze_educational(t))
    print("Story:", analyze_story_arc(t + " So I struggled for years. Then I found a method. Now I speak three languages."))
    print("Hook:", analyze_hook("You won't believe what happened next."))
