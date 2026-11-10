"""
Utility functions for markdown escaping, language detection, and text processing.
"""
import logging
import os

def get_logger(name: str) -> logging.Logger:
    os.makedirs("logs", exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # ← add this
    fh = logging.FileHandler(f"logs/{name}.log")
    fh.setLevel(logging.DEBUG)      # ← add this
    fh.setFormatter(logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)s — %(message)s",
        datefmt="%H:%M:%S",
    ))
    logger.addHandler(fh)
    return logger


def _make_test_logger(name: str, filename: str) -> logging.Logger:
    """
    Build a dedicated logger writing to logs/test_logs/<filename>.

    Used for the LLM audit trail. Unlike get_logger, this guards against
    duplicate handlers (each logger is fetched from more than one module)
    and writes UTF-8 so non-Latin scripts (Hindi/Tamil/Punjabi/Bengali)
    are recorded intact.
    """
    log_dir = os.path.join("logs", "test_logs")
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(os.path.join(log_dir, filename), encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s.%(msecs)03d — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(fh)
        logger.propagate = False
    return logger


def get_reply_prompt_logger() -> logging.Logger:
    """
    Reply-generation audit trail → logs/test_logs/reply_prompts.log

    Records, per conversational turn: the exact prompt sent to the model,
    the model's raw response, and the actual reply shown to the patient.
    """
    return _make_test_logger("test_reply_prompts", "reply_prompts.log")


def get_extraction_prompt_logger() -> logging.Logger:
    """
    Field-extraction audit trail → logs/test_logs/extraction_prompts.log

    Records, per extraction call: the exact extraction prompt sent to the
    model, the model's raw response, and the structured fields extracted.
    """
    return _make_test_logger("test_extraction_prompts", "extraction_prompts.log")


def get_planner_logger() -> logging.Logger:
    """
    Turn-planner audit trail → logs/test_logs/planner.log

    One block per turn: what the patient said, the strategy the planner chose
    (and why), the safety flag, what was still unfilled at decision time, and
    the reply that strategy produced. This is the window into the Phase-2
    planner — read this file when the bot's behaviour feels wrong, since the
    strategy explains WHY it asked what it asked.
    """
    return _make_test_logger("test_planner", "planner.log")


def get_followup_advisor_logger() -> logging.Logger:
    """
    Follow-up advisor audit trail → logs/test_logs/followup_advisor.log

    Experimental, log-only: per turn, records what the bot asked, what the
    patient replied, and what the LLM suggests as a clarifying/follow-up
    question (plus the full prompt). The suggestion is NOT shown to the
    patient yet — it's captured so we can decide what to actually implement.
    """
    return _make_test_logger("test_followup_advisor", "followup_advisor.log")

import re
from typing import Optional, List, Dict

from data_model import FIELD_HUMANIZE


# ── Optimistic write-ahead cache ──────────────────────────────────
# The main loop writes here immediately when the LLM asks about a topic,
# and reads from here (merged with DB) on the next turn — closing the 5-15s
# window between "patient answers" and "extraction finishes saving to DB".

_FILLED_FIELDS_CACHE: dict[str, set[str]] = {}
_PREV_SECTION: dict[str, str] = {}


def _cache_key(chat_id) -> str:
    """Normalise a session identifier to a cache key.

    These caches were written for Telegram, where chat_id is always an integer,
    and every accessor called int(chat_id) on the way in. Over REST the session
    id is whatever the caller chose — "smoke-1", a UUID, an email — so int()
    raised ValueError and the API returned 500 on the first real turn of any
    session whose id was not all digits. Keying on str() accepts both, and
    normalises 12345 and "12345" to the same entry rather than splitting them.
    """
    return str(chat_id)


def mark_topic_as_filled(chat_id, topic_id: str, section: str) -> None:
    """DEPRECATED — now a no-op. Do not call.

    This used to mark every field of a topic as FILLED as soon as the bot ASKED
    about that topic. Asking is not answering. The consequences were severe:

      - The interview marched past questions the patient never answered
        (patient says "yes" to "what's your education?" -> education was already
        flagged filled, so the bot moved on to the next topic).
      - The reply prompt was told "ALREADY ANSWERED — do NOT ask about these
        again: Education", so the model was FORBIDDEN from clarifying.
      - With nothing new to reflect, replies recycled the last real fact
        ("You're 21. That helps me understand your background...") every turn.
      - filled_fields_set diverged wildly from the DB (logs showed DB: set()
        with a 12-field cache, filled=0 pct=0%), which also corrupted the
        planner's view of what still needed asking.

    Real extraction results are cached by mark_fields_filled() instead, which
    solves the ONLY legitimate problem this was aimed at: extraction runs in a
    background task, so the DB write may not have landed before the next turn.
    """
    logging.getLogger("handlers").warning(
        f"OPTIMISTIC — ignoring deprecated mark_topic_as_filled({topic_id!r}); "
        f"asking a topic no longer marks its fields answered"
    )
    return


def mark_fields_filled(chat_id, field_names) -> None:
    """Cache fields extraction ACTUALLY captured, to bridge DB-write latency.

    Extraction runs as a background task after the reply is sent, so the next
    turn can read section_fields before that write lands. This cache covers that
    gap only — it must never contain a field the patient did not answer.
    """
    names = {f for f in (field_names or []) if f}
    if not names:
        return
    cache = _FILLED_FIELDS_CACHE.setdefault(_cache_key(chat_id), set())
    cache.update(names)
    logging.getLogger("handlers").info(
        f"OPTIMISTIC — cached extracted fields: {sorted(names)}"
    )


def get_optimistic_filled_fields(chat_id, db_fields: dict) -> set[str]:
    db_filled = {f for f, v in db_fields.items() if v}
    cache = _FILLED_FIELDS_CACHE.get(_cache_key(chat_id), set())
    if cache:
        merged = db_filled | cache
        logger = logging.getLogger("handlers")
        logger.info(f"OPTIMISTIC — DB: {db_filled} | cache: {cache} -> merged: {merged}")
        return merged
    return db_filled


def clear_filled_fields_cache(chat_id) -> None:
    _FILLED_FIELDS_CACHE.pop(_cache_key(chat_id), None)
    _PREV_SECTION.pop(_cache_key(chat_id), None)
    logger = logging.getLogger("handlers")
    logger.info(f"OPTIMISTIC — cache cleared for chat {chat_id}")


def get_prev_section(chat_id) -> str | None:
    return _PREV_SECTION.get(_cache_key(chat_id))


def set_prev_section(chat_id, section: str) -> None:
    _PREV_SECTION[_cache_key(chat_id)] = section


# ── Utilities ──────────────────────────────────────────────────────


def escape_md(text: str) -> str:
    """
    Escape special characters for Telegram MarkdownV2.
    These characters need to be escaped: . ! ( ) - = + # | { } [ ] ~ _ * ` \
    """
    special = r'\_*[]()~`>#+-=|{}.!\\`'
    return re.sub(f'([{re.escape(special)}])', r'\\\1', str(text))


# Unicode script blocks for the 4 non-Latin scripts we can tell apart
# reliably. Mutually exclusive ranges — no ML model or network call needed.
SCRIPT_RANGES = {
    "hi": r'[\u0900-\u097F]',   # Devanagari
    "bn": r'[\u0980-\u09FF]',   # Bengali
    "pa": r'[\u0A00-\u0A7F]',   # Gurmukhi (Punjabi)
    "ta": r'[\u0B80-\u0BFF]',   # Tamil
}

# Languages a mid-session switch can confidently land on via SCRIPT alone.
# Deliberately excludes 'en' and 'hinglish' — both Latin script, indistin-
# guishable from a bare number, a name, or a single loanword by script alone.
# Switching INTO English specifically is handled separately, by word content
# rather than script — see is_confidently_english below. Switching INTO
# Hinglish isn't attempted at all: there's no script-based OR word-based
# signal that reliably tells "this is Hinglish" apart from "this is English"
# or "this is Hindi typed in Roman letters" — genuinely ambiguous, left alone.
SWITCHABLE_LANGUAGES = {"hi", "bn", "pa", "ta"}

# Function words used to detect a confident switch INTO English specifically.
# Weighted toward grammatical "glue" words rather than vocabulary on purpose:
# content words like "doctor", "office", "checkup" get borrowed into Hinglish
# constantly, but a Hindi speaker writing Hinglish essentially never swaps
# "hai"/"hoon"/"tha" for "is"/"am"/"was" — those stay Hindi even in a heavily
# English-mixed sentence. A high ratio of THESE specific words is a much
# cleaner English signal than counting any recognized English word.
ENGLISH_FUNCTION_WORDS = {
    "the", "a", "an", "i", "you", "he", "she", "it", "we", "they", "me", "him",
    "her", "us", "them", "my", "your", "his", "its", "our", "their", "mine",
    "yours", "hers", "ours", "theirs", "am", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "can", "could", "shall", "should", "may", "might", "must", "this", "that",
    "these", "those", "and", "or", "but", "because", "although", "though", "so",
    "if", "unless", "while", "what", "when", "where", "why", "how", "who",
    "whom", "whose", "which", "in", "on", "at", "to", "from", "with", "for",
    "of", "about", "into", "onto", "over", "under", "between", "during",
    "after", "before", "since", "until", "not", "no", "never", "none",
    "nothing", "nobody", "nowhere", "there", "here", "very", "also", "just",
    "only", "even", "still", "already",
}


def detect_language(text: str) -> str:
    """
    Detect the language of the input text by Unicode script block.
    Returns one of: 'hi', 'bn', 'pa', 'ta', defaults to 'en'.

    Simple heuristic, run fresh on every message (no caching, no network call —
    this is a plain regex scan, microseconds, near-zero cost):
    - Devanagari script  -> Hindi ('hi')
    - Bengali script     -> Bengali ('bn')
    - Gurmukhi script    -> Punjabi ('pa')
    - Tamil script       -> Tamil ('ta')
    - Otherwise          -> English ('en') — covers Latin script, digits,
                            punctuation, and Hinglish (which is Hindi content
                            in Latin script, so it falls through here too;
                            Hinglish sessions don't use this function for
                            translation routing, only for the old native path)

    Any-match — returns as soon as ANY character of a script appears. That's
    the right bar for "does this message need a translate call" (a single
    Tamil word mixed into the message still means there's something to
    translate). It is NOT the right bar for "has the patient switched
    languages" — see detect_dominant_language below for that.
    """
    for lang_code, pattern in SCRIPT_RANGES.items():
        if re.search(pattern, text):
            return lang_code

    return "en"


def detect_dominant_language(text: str, threshold: float = 0.5) -> str:
    """
    Like detect_language, but requires the script to be the MAJORITY of the
    message's alphabetic characters, not just present somewhere in it.

    This is the guard against loanwords and proper nouns falsely triggering
    a language switch: "நான் ஒரு software engineer" is still mostly Tamil
    (the English term doesn't dominate), and "Kaushik Deb referred me" is
    still mostly English (one Latin-script name doesn't make it Hindi).
    A script needs to clear `threshold` (default: strict majority) of all
    alphabetic characters in the message before it counts as a real switch.

    Returns one of 'hi'/'bn'/'pa'/'ta' if one of them dominates, else 'en'
    — 'en' here covers genuine English, Hinglish, bare numbers, names, or
    anything too short/mixed to call confidently. Callers should treat 'en'
    as "no confident switch signal", not as "the patient switched to
    English" — see SWITCHABLE_LANGUAGES for why that distinction matters.
    """
    script_counts = {lang: len(re.findall(pattern, text)) for lang, pattern in SCRIPT_RANGES.items()}
    latin_count = len(re.findall(r'[A-Za-z]', text))
    total_alpha = sum(script_counts.values()) + latin_count

    if total_alpha == 0:
        return "en"

    top_lang, top_count = max(script_counts.items(), key=lambda kv: kv[1])
    if top_count / total_alpha > threshold:
        return top_lang

    return "en"


def is_confidently_english(text: str, min_words: int = 4, min_ratio: float = 0.3) -> bool:
    """
    Detects a confident switch INTO English specifically — a different
    problem from detect_dominant_language above, which deliberately can't
    tell English apart from Hinglish (both pure Latin script) and doesn't
    try. This doesn't solve that either; it only tries to tell "real
    English" apart from "Hinglish, or a stray loanword/name", which script
    alone genuinely cannot do.

    Two guards, both need to pass:
    - min_words: message needs enough words to be meaningful. Stops a bare
      "yes"/"ok"/"25"/"male" — all valid English words — from flipping the
      session on a single-word factual answer.
    - min_ratio: a high enough fraction of words must be recognized English
      function words. Stops a Hinglish sentence that happens to borrow a
      few English nouns ("office", "doctor") from counting as English —
      those get borrowed constantly; "is"/"am"/"the" essentially don't.

    Not a real language-ID model — a word list and two thresholds. Cheap,
    transparent, no new dependency, no network call. Will misclassify some
    code-mixed messages in both directions; the design goal is "doesn't
    false-trigger on a short Hindi/Tamil/Punjabi/Bengali reply", not
    "perfect classification" — same philosophy as detect_dominant_language's
    own threshold guard against loanwords.
    """
    words = re.findall(r"[a-zA-Z]+", text.lower())
    if len(words) < min_words:
        return False
    matches = sum(1 for w in words if w in ENGLISH_FUNCTION_WORDS)
    return (matches / len(words)) >= min_ratio


def format_field_for_display(field_name: str) -> str:
    """Convert field_name (snake_case) to human-readable title."""
    return field_name.replace("_", " ").title()


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to max_length, adding suffix if truncated."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def build_filled_fields_summary(fields: dict) -> str:
    """
    Build a human-readable summary of already-filled fields for the LLM's context.
    
    Example output:
    "You've already told me: age (28), education (Bachelor's degree), occupation (Software Engineer)"
    """
    if not fields:
        return ""
    
    filled = []
    for field_name, value in fields.items():
        if value:
            display_name = format_field_for_display(field_name)
            filled.append(f"{display_name}: {truncate_text(str(value), 50)}")
    
    if not filled:
        return ""
    
    return "You've already told me: " + ", ".join(filled)


def build_missing_fields_hint(fields: dict, section_fields: list) -> str:
    """
    Build a hint list of missing fields (for LLM system prompt).
    Show which fields are still needed without disclosing exact field names.
    
    Example:
    "Still need: general background info, occupational details, family structure, ..."
    """
    filled_set = set(f for f, v in fields.items() if v)
    missing = [f for f in section_fields if f not in filled_set]
    
    if not missing:
        return ""
    
    # Generic hints based on section field patterns
    hints = []
    for field in missing[:5]:  # Show first 5 missing
        # Convert to human-readable hint
        hint = format_field_for_display(field)
        hints.append(hint)
    
    if len(missing) > 5:
        return ", ".join(hints) + ", and a few more details"
    
    return ", ".join(hints)


# def build_topic_based_hint(topics: List[Dict], filled_fields: set, language: str = "en") -> str:
#     """
#     Build a conversational hint based on remaining topics to explore.
#     Instead of listing field names, suggest natural next conversation topics.
    
#     Args:
#         topics: List of topic dicts from SECTION_CONVERSATION_TOPICS
#         filled_fields: Set of field names that have been filled
#         language: Language code ('en', 'hi', or 'hinglish')
    
#     Returns:
#         A natural conversational hint about what to cover next
#     """
#     remaining_topics = []
    
#     for topic in topics:
#         topic_fields = topic.get("fields", [])
#         # A topic is "complete" if all its required fields are filled
#         required_count = topic.get("min_required", 1)
#         filled_in_topic = sum(1 for f in topic_fields if f in filled_fields)
        
#         if filled_in_topic < required_count:
#             remaining_topics.append(topic)
    
#     if not remaining_topics:
#         return ""
    
#     # Return first 3-4 topic names as natural conversational hints
#     topic_names = [t.get("topic_name", "") for t in remaining_topics[:3]]
    
#     if language in ["hi", "hinglish"]:
#         return "Topics to explore: " + ", ".join(topic_names)
    
#     return "Topics to explore: " + ", ".join(topic_names)

# def build_topic_based_hint(topics, filled_fields, language):
#     lang = language if language in FIELD_HUMANIZE else "en"
#     humanize = FIELD_HUMANIZE[lang]

#     remaining = []
#     for topic in topics:
#         filled_in = sum(1 for f in topic["fields"] if f in filled_fields)
#         if filled_in < topic.get("min_required", 1):
#             remaining.append(topic)

#     if not remaining:
#         return ""

#     parts = []
#     for t in remaining[:3]:
#         human_fields = [humanize.get(f, f) for f in t["fields"]]
#         fields_str = " and ".join(human_fields)
#         parts.append(f"{t['topic_name']}({fields_str})")

#     return "Topics to explore: " + ", ".join(parts)

# def build_topic_based_hint(topics, filled_fields, language):
#     lang = language if language in FIELD_HUMANIZE else "en"
#     humanize = FIELD_HUMANIZE[lang]

#     remaining = []
#     for topic in topics:
#         filled_in = sum(1 for f in topic["fields"] if f in filled_fields)
#         if filled_in < topic.get("min_required", 1):
#             remaining.append(topic)

#     if not remaining:
#         return ""

#     parts = []
#     for t in remaining[:3]:
#         human_fields = [humanize.get(f, f) for f in t["fields"]]
#         fields_str = " and ".join(human_fields)
#         parts.append(f"{t['topic_name']}({fields_str})")

#     return "Topics to explore: " + ", ".join(parts)

def build_topic_based_hint(topics, filled_fields, language, last_topic_id=None):
    lang = language if language in FIELD_HUMANIZE else "en"
    humanize = FIELD_HUMANIZE[lang]

    last_idx = next(
        (i for i, t in enumerate(topics) if t["topic_id"] == last_topic_id),
        -1
    )
    ordered = topics[last_idx+1:] + topics[:last_idx+1]

    remaining = []
    for topic in ordered:
        filled_in = sum(1 for f in topic["fields"] if f in filled_fields)
        if filled_in < topic.get("min_required", 1):
            remaining.append(topic)

    if not remaining:
        return ""

    parts = []
    for t in remaining[:3]:
        # Show only the fields still missing, not all fields in the topic.
        # This gives the response LLM a precise picture of what's left.
        unfilled = [f for f in t["fields"] if f not in filled_fields]
        display_fields = unfilled if unfilled else t["fields"]
        human_fields = [humanize.get(f, f) for f in display_fields]
        fields_str = " and ".join(human_fields)
        parts.append(f"{t['topic_name']}({fields_str})")

    return "Topics to explore: " + ", ".join(parts)


def get_next_natural_question(
    topics: List[Dict],
    filled_fields: set,
    language: str = "en",
    last_assistant_question: Optional[str] = None,
) -> Optional[str]:
    """
    Get the next natural question from remaining topics.
    
    Args:
        topics: List of topic dicts
        filled_fields: Set of field names that have been filled
        language: Language code ('en', 'hi', or 'hinglish')
    
    Returns:
        A natural conversation question, or None if all topics are covered.
    """
    return get_next_natural_question_with_history(
        topics,
        filled_fields,
        language,
        last_assistant_question,
    )


def get_next_natural_question_with_history(
    topics: List[Dict],
    filled_fields: set,
    language: str = "en",
    last_assistant_question: Optional[str] = None,
) -> Optional[str]:
    """
    Get the next natural question, skipping a question that was just asked.
    This prevents the bot from repeating itself after a refusal or non-answer.
    """
    last_norm = _normalize_question(last_assistant_question or "")

    for topic in topics:
        topic_fields = topic.get("fields", [])
        required_count = topic.get("min_required", 1)
        filled_in_topic = sum(1 for f in topic_fields if f in filled_fields)

        if filled_in_topic < required_count:
            # PARTIALLY answered: only ever ask about a field that is still
            # missing. Never fall back to the broad question here — the broad
            # question covers the WHOLE topic, so it re-asks fields the patient
            # has already answered.
            #
            # This is why a patient who had just said "Mom dad sister" was asked
            # "Who all are there in your family?" a second time: the
            # family_structure topic needs 2 fields, living_arrangement hadn't
            # been extracted yet (extraction runs in a background task), its
            # targeted question was skipped as a just-asked duplicate, and
            # control fell through to natural_questions_en[0] — which is
            # family_members_list's question, already answered.
            if filled_in_topic > 0:
                unfilled = [f for f in topic_fields if f not in filled_fields]
                fq_key = f"field_questions_{language}"
                field_qs = topic.get(fq_key) or topic.get("field_questions_en", {})
                for field in unfilled:
                    q = field_qs.get(field)
                    if q and _normalize_question(q) != last_norm:
                        return q
                # Nothing askable for what's actually missing (no targeted
                # question, or we just asked it). Move to the next topic rather
                # than re-asking something we already have.
                continue

            # Fully unanswered topic: the broad question is the right opener.
            question_key = f"natural_questions_{language}"
            questions = topic.get(question_key, []) or topic.get("natural_questions_en", [])
            for question in questions:
                if _normalize_question(question) != last_norm:
                    return question

    return None


def get_next_topic_after(
    topics: List[Dict],
    filled_fields: set,
    last_topic_id: Optional[str],
    language: str = "en",
    last_assistant_question: Optional[str] = None,
) -> tuple[Optional[str], Optional[str]]:
    """
    Return first unfilled topic AFTER last_topic_id.
    Wraps around if needed.
    Returns (topic_id, question) or (None, None) if all complete.
    """
    last_norm = _normalize_question(last_assistant_question or "")

    last_idx = next(
        (i for i, t in enumerate(topics) if t["topic_id"] == last_topic_id),
        -1
    )

    ordered = topics[last_idx+1:] + topics[:last_idx+1]

    for topic in ordered:
        filled_in = sum(1 for f in topic["fields"] if f in filled_fields)
        if filled_in < topic.get("min_required", 1):
            # PARTIALLY answered: only ask about fields still missing. The broad
            # question covers the whole topic and would re-ask what the patient
            # already told us — see the matching comment in
            # get_next_natural_question_with_history.
            if filled_in > 0:
                unfilled = [f for f in topic["fields"] if f not in filled_fields]
                fq_key = f"field_questions_{language}"
                field_qs = topic.get(fq_key) or topic.get("field_questions_en", {})
                for field in unfilled:
                    q = field_qs.get(field)
                    if q and _normalize_question(q) != last_norm:
                        return topic["topic_id"], q
                # Nothing askable for what's missing — move on rather than
                # re-asking an answered field.
                continue

            # Fully unanswered topic: the broad question is the right opener.
            question_key = f"natural_questions_{language}"
            questions = topic.get(question_key) or topic.get("natural_questions_en", [])
            for q in questions:
                if _normalize_question(q) != last_norm:
                    return topic["topic_id"], q

    return None, None


def _normalize_question(text: str) -> str:
    """Normalize a question for repeat detection."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", text.lower())).strip()



def escape_json_string(text: str) -> str:
    """Escape a string for safe JSON inclusion."""
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")