"""
Translation wrapper for languages that don't reliably get native script
from the LLM directly: Hindi, Tamil, Punjabi, Bengali.

These route through live machine translation at the boundary:
patient text -> English (existing pipeline, unchanged) -> back to patient's language.

English and Hinglish are NOT touched by this module — they keep using the
existing native-prompting path in llm.py exactly as before. Hindi used to be
grouped with them; it isn't anymore (see WRAP_LANGUAGES below) — llm.py's own
"hi" branches (deterministic resistance responses, reflective prefixes) were
producing Romanized Hindi, not Devanagari, since that's what those hand-
written strings actually contain. Routing Hindi through this module instead
guarantees real Devanagari output via translation, at the cost of losing that
hand-tuned native phrasing — llm.py's "hi" branches are now unreachable dead
code (left in place, not deleted, since llm.py stays untouched by this
project's convention).

Translation runs on the SAME local vLLM server that generates the replies — no
API key, no third party, nothing leaving the machine. This previously used
deep-translator's GoogleTranslator, an unofficial scrape of the consumer
translate.google.com endpoint, which sent both sides of every hi/ta/pa/bn
conversation to Google under consumer terms with no data processing agreement.

Nothing outside this file needs to change to swap the backend again; keep it
that way.
"""

import os

from utils import get_logger

logger = get_logger("translation")

# Languages that get the live-translation treatment.
# en / hinglish deliberately excluded — Latin script, the LLM produces them
# natively and reliably. hi is included precisely because the LLM does NOT
# reliably produce Devanagari for it — see module docstring.
WRAP_LANGUAGES = {"ta", "pa", "bn", "hi"}


_LANG_NAMES = {
    "en": "English", "hi": "Hindi", "ta": "Tamil",
    "pa": "Punjabi", "bn": "Bengali", "hinglish": "Hinglish",
}

# Kept deliberately terse. A chatty instruction invites the model to explain
# itself, and anything it adds ends up shown to the patient as if the clinician
# had said it.
_SYSTEM = (
    "You are a translation engine for a medical interview. Translate the user's "
    "message into {target}. Preserve clinical meaning exactly — symptoms, "
    "durations, dosages and negations must survive intact. Keep the same tone and "
    "register. Output ONLY the translation: no preamble, no explanation, no "
    "quotes, no notes."
)


async def _translate(text: str, source: str, target: str) -> str:
    """Translate via the local model, falling back to the original text.

    WHY LOCAL: this used to call deep_translator's GoogleTranslator, an
    unofficial scrape of the consumer translate.google.com endpoint. That sent
    BOTH sides of every hi/ta/pa/bn conversation — chief complaints, symptoms,
    suicidal disclosures — to Google, with no API key and therefore no contract,
    no data processing agreement and no deletion guarantee. Every other part of
    this system runs on our own hardware; that one dependency undid the whole
    privacy posture.

    Now it goes to the same vLLM server that generates the replies. No download,
    no extra GPU, no new dependency, and nothing leaves the machine.

    ASYNC, unlike the version it replaces. The old one made a blocking network
    call inline on the event loop, so one hi/ta/pa/bn turn stalled every other
    session for two round-trips. Awaiting the model call fixes that as a side
    effect of fixing the privacy problem.

    On any failure the original text is returned unchanged and the error is
    logged — a turn must never die because translation hiccuped.
    """
    if not text or not text.strip():
        return text

    target_name = _LANG_NAMES.get(target, target)
    try:
        import llm
        # llm.call_llm is async; this function is sync and is called from async
        # code, so it is offloaded by the caller (see translate_* below).
        out = await llm.call_llm(
            prompt=text,
            system_prompt=_SYSTEM.format(target=target_name),
            max_tokens=_max_tokens_for(text),
            think=False,
        )
        return out.strip() or text
    except Exception as e:
        logger.error(
            f"[TRANSLATE] {source}->{target} failed, falling back to original text: {e}"
        )
        return text


def _max_tokens_for(text: str) -> int:
    """Generous headroom: Indic scripts tokenize far worse than English, so a
    budget sized on the English length truncates the middle of a reply."""
    return max(256, min(2048, len(text.split()) * 8 + 128))


async def translate_to_english(text: str, source_lang: str) -> str:
    """Patient's message (in source_lang) -> English, for the internal pipeline."""
    return await _translate(text, source=source_lang, target="en")


async def translate_from_english(text: str, target_lang: str) -> str:
    """Internal pipeline's English output -> patient's language, for display."""
    return await _translate(text, source="en", target=target_lang)