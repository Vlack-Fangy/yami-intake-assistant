"""
LLM interactions for psychiatric history-taking bot.
Loads meta-llama/Meta-Llama-3.2-3B locally and uses the best available
device automatically: CUDA, MPS, then CPU.
"""
from datetime import datetime
def _ts():
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]

from utils import (
    get_logger,
    get_reply_prompt_logger,
    get_extraction_prompt_logger,
    get_followup_advisor_logger,
)
logger = get_logger("llm")
reply_log = get_reply_prompt_logger()
extract_log = get_extraction_prompt_logger()
advisor_log = get_followup_advisor_logger()

import asyncio
import ast
import hashlib
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = os.getenv("LLM_MODEL_NAME")
LOCAL_FILES_ONLY = os.getenv("LLM_LOCAL_FILES_ONLY", "1").lower() not in {"0", "false", "no"}
MAX_CONTEXT_TOKENS = int(os.getenv("LLM_MAX_CONTEXT_TOKENS", "4096"))
hf_token = os.getenv("hf_token")

# ── Inference backend selection ─────────────────────────────────────────────
# "hf"   → in-process HuggingFace transformers.generate (original path).
# "vllm" → talk to a local vLLM OpenAI-compatible server. This unlocks
#          continuous batching, automatic prefix caching (huge: the ~1,100-word
#          system prompt is identical every turn), and guaranteed structured
#          output via guided decoding — none of which the in-process path has.
# Default stays "hf" so nothing changes until the vLLM server is up and the flag
# is flipped. Launch the server with scripts/serve_vllm.sh.
LLM_BACKEND = os.getenv("LLM_BACKEND", "hf").strip().lower()
VLLM_BASE_URL = os.getenv("LLM_VLLM_BASE_URL", "http://127.0.0.1:8100/v1").rstrip("/")
# Name the model was served under (vllm serve --served-model-name). Defaults to
# the local path in MODEL_NAME, which is what vLLM uses if the flag is omitted.
VLLM_MODEL = os.getenv("LLM_VLLM_MODEL", MODEL_NAME or "qwen3-14b")
VLLM_API_KEY = os.getenv("LLM_VLLM_API_KEY", "EMPTY")  # vLLM ignores it; client requires a value
VLLM_TIMEOUT_S = float(os.getenv("LLM_VLLM_TIMEOUT_S", "120"))

# Whether to let Qwen3 / reasoning models think before answering.
# OFF by default for speed. The re-asking-already-answered-fields bug is now
# handled deterministically by the filled-field guard + the negative "do NOT
# ask" list in the prompt, so thinking is no longer needed for correctness.
# Set LLM_ENABLE_THINKING=1 to opt back in for slightly more natural REPLY
# phrasing (capped by LLM_MAX_THINKING_TOKENS). Extraction never thinks
# regardless — it's a mechanical JSON task that gains nothing from reasoning.
ENABLE_THINKING = os.getenv("LLM_ENABLE_THINKING", "0").lower() not in {"0", "false", "no"}

TOKENIZER = None
MODEL = None
DEVICE: Optional[torch.device] = None
MODEL_LOCK = asyncio.Lock()


def _select_device() -> Tuple[torch.device, torch.dtype]:
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Selected device:", device)
        return device, torch.float16

    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if mps_available:
        device = torch.device("mps")
        print("Selected device:", device)
        return device, torch.float16

    device = torch.device("cpu")
    print("Selected device:", device)
    return device, torch.float32


def _language_label(language_code: str) -> str:
    return {
        "en": "English",
        "hi": "Hindi",
        "hinglish": "Hinglish (Hindi written in English)",
    }.get(language_code, language_code)


def _needs_reflection(text: str) -> bool:
    """Return True when the patient message suggests emotional distress.

    Keeps routine turns concise and reserves reflective language for moments
    where empathy is clinically useful.
    """
    lowered = text.lower()
    distress_cues = [
        "anxious", "anxiety", "panic", "depressed", "depression", "hopeless",
        "overwhelmed", "scared", "afraid", "fear", "stressed", "stress",
        "can't sleep", "cannot sleep", "insomnia", "trauma", "abuse",
        "hurt", "cry", "crying", "suicidal", "self harm", "self-harm",
        "lonely", "worthless", "helpless", "grief", "loss",
    ]
    return any(cue in lowered for cue in distress_cues)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", text.lower())).strip()


def _has_stop_intent(text: str) -> bool:
    """Patient wants to stop, pause, or continue another time.

    Defers to safety.SOFT_STOP_PHRASES — the canonical list. This used to keep
    its OWN copy, and the two drifted: neither contained "pause", which is the
    word the bot itself offers ("Let's take a pause if you'd like"). A patient
    replied "yes lets pause" and was asked the next interview question instead.
    Two lists meant two places to forget the same word; now there is one.
    """
    import safety  # local import: safety -> db -> data_model, no cycle with llm

    return safety.is_soft_stop_message(text or "")


def _is_uncooperative_turn(text: str) -> bool:
    lowered = _normalize_text(text)
    resistant_cues = [
        "idk",
        "dont know",
        "do not know",
        "whatever",
        "nope",
        "boring",
        "not interested",
        "leave it",
        "nothing",
        "no stress",
        "when did i say",
        "you are boring",
        "wtf",
    ]
    return any(cue in lowered for cue in resistant_cues)


def _has_explicit_or_offtopic_sexual_content(text: str) -> bool:
    lowered = _normalize_text(text)
    cues = [
        "blowjob",
        "sex",
        "porn",
        "hookup",
        "nudes",
        "horny",
    ]
    return any(cue in lowered for cue in cues)


def _deterministic_resistance_response(
    detected_language: str,
    next_natural_question: str,
    patient_text: str,
    last_assistant_question: str = "",
) -> str:
    if _has_stop_intent(patient_text):
        if detected_language == "hi":
            return "Theek hai, hum yahin pause karte hain. Jab aap ready hon, /resume type kijiye."
        if detected_language == "hinglish":
            return "Theek hai, hum yahin pause karte hain. Jab ready ho, /resume type kar dena."
        return "No problem. We can pause here. Whenever you want to continue, just type /resume."

    if _has_explicit_or_offtopic_sexual_content(patient_text):
        if detected_language == "hi":
            return "Samjha. Main baat ko aapki sehat par rakhoonga. Kya aajkal neend, mood, ya tanav mein koi badlav hai?"
        if detected_language == "hinglish":
            return "Samjha. Chalo focus health par rakhte hain. Sleep, mood, ya stress mein koi change notice hua?"
        return "Understood. I will keep this focused on your wellbeing. Any recent change in sleep, mood, or stress?"

    if next_natural_question and _normalize_text(next_natural_question) != _normalize_text(last_assistant_question):
        return f"Okay. {next_natural_question}"

    if detected_language == "hi":
        return "Theek hai. Hum aaram se chalenge. Abhi aapko sabse zyada kis baat ki pareshani hai?"
    if detected_language == "hinglish":
        return "Theek hai. Hum slowly chalenge. Abhi aapko sabse zyada kis cheez se dikkat hai?"
    return "Okay. We can go at your pace. What has been bothering you most lately?"


_THANKS_CUES = [
    "thank", "thanks", "thankyou", "thnx", "thx", "ty",
    "shukriya", "shukria", "dhanyavaad", "dhanyawad", "dhanyvaad",
]


def _patient_thanked(text: str) -> bool:
    """True if the patient's message expresses thanks (en / hi / hinglish)."""
    lowered = _normalize_text(text)              # strips apostrophes/punctuation
    compact = lowered.replace(" ", "")           # catches "thank you" -> "thankyou"
    return any(cue in lowered or cue in compact for cue in _THANKS_CUES)


def _strip_unprompted_welcome(response: str, patient_text: str) -> str:
    """
    Remove a 'You're welcome' (and close variants) the model volunteered when
    the patient never actually thanked it.

    If the reply was NOTHING but the unprompted welcome, return "" so the caller
    treats it as a bad response and falls back to a real question. This is what
    stops the "Bot: are you thinking of harming yourself? / Patient: yes /
    Bot: you're welcome" failure — an empty result routes to a safety follow-up
    instead of sending the tone-deaf filler.
    """
    if _patient_thanked(patient_text):
        return response

    cleaned = re.sub(
        r"\b(you'?re\s+welcome|you\s+are\s+welcome|my\s+pleasure|"
        r"koi\s+baat\s+nahi|koi\s+baat\s+nahin)\b[\s,.!?:;–—-]*",
        "",
        response,
        flags=re.IGNORECASE,
    ).strip()

    if not cleaned:
        return ""  # whole reply was an unprompted welcome — signal caller to fall back
    # Re-capitalise if stripping left a lowercase lead.
    return cleaned[0].upper() + cleaned[1:] if cleaned[0].islower() else cleaned


# Gratitude fillers the model habitually prepends every turn ("Thank you for
# sharing that.", "Thanks for telling me.", "I appreciate you opening up.").
# Overused, they make the conversation feel scripted — see the anti-filler rule
# in the reply system prompt. We strip such an opener when real content follows,
# so the paraphrase/question the model was told to lead with becomes the opener
# instead. A reply that is ONLY a gratitude line is left untouched here and gets
# caught by the normal validator/fallback path.
_GRATITUDE_OPENER_RE = re.compile(
    r"^\s*(?:i\s+)?(?:thank\s*(?:s|\s*you|\s*u)?|thankyou|appreciate)\b"
    r"[^.!?\n]*?"
    r"\b(?:shar\w*|tell\w*|told|open\w*\s+up|let\w*\s+me\s+know|trust\w*|"
    r"for\s+(?:that|this|sharing|telling))\b"
    r"[^.!?\n]*[.!?]+[\s\"']*",
    re.IGNORECASE,
)


def _strip_scripted_gratitude_opener(response: str) -> str:
    """Drop a leading 'Thank you for sharing'-style filler when substance follows."""
    cleaned = _GRATITUDE_OPENER_RE.sub("", response, count=1).strip()
    if not cleaned or cleaned == response.strip():
        return response  # nothing left, or nothing matched — leave as-is
    return cleaned[0].upper() + cleaned[1:] if cleaned[0].islower() else cleaned


def _extract_question(text: str) -> str:
    match = re.findall(r"[^?]*\?", text)
    if match:
        return match[-1].strip()
    return text.strip()


def _extract_last_assistant_question(history: List[Dict]) -> str:
    for msg in reversed(history):
        if msg.get("role") == "assistant":
            return _extract_question(msg.get("content", ""))
    return ""


def _parse_missing_labels(missing_fields_hint: str) -> List[str]:
    labels = [part.strip(" .") for part in missing_fields_hint.split(",") if part.strip()]
    cleaned = [label for label in labels if "few more details" not in label.lower()]
    return cleaned


LABEL_REWRITE = {
    "ses": "socioeconomic background",
    "sex": "gender",
    "residence": "current city or town",
    "source of referral": "how you came for help",
}


def _humanize_label(label: str) -> str:
    return LABEL_REWRITE.get(label.lower().strip(), label.lower().strip())


# def _choose_prefix(detected_language: str, reflection_needed: bool, patient_text: str) -> str:
#     if reflection_needed:
#         reflective = {
#             "en": ["I hear you.", "I understand.", "That sounds hard."],
#             "hi": ["Main samajh raha hoon.", "Main aapki baat samajh raha hoon."],
#             "hinglish": ["I understand.", "Samajh raha hoon."],
#         }
#         options = reflective.get(detected_language, reflective["en"])
#     else:
#         neutral = {
#             "en": ["", "", "Okay.", ""],
#             "hi": ["", "", "Theek hai.", ""],
#             "hinglish": ["", "", "Theek hai.", ""],
#         }
#         options = neutral.get(detected_language, neutral["en"])

#     index = len(patient_text.strip()) % len(options)
#     return options[index]



def _choose_prefix(detected_language: str, reflection_needed: bool, patient_text: str) -> str:
    if reflection_needed:
        reflective = {
            "en": [
                "I hear you.", "I understand.", "That sounds hard.", "That sounds really difficult.",
                "I can see how that would be tough.", "That must be a lot to deal with.",
                "I get what you're saying.", "I'm really sorry you're going through this.",
                "That sounds overwhelming.", "That makes sense.", "I can imagine that wasn't easy.",
                "That must feel heavy.", "I hear how much this is affecting you.",
                "It sounds like this has been weighing on you.", "I can see why you'd feel that way.",
                "That sounds frustrating.", "That sounds painful.", "I hear the effort you're putting in.",
                "That must be exhausting.", "I understand why this matters to you.",
                "I can sense how important this is for you.", "That sounds confusing to deal with.",
                "I hear your concern.", "That must feel really intense.",
                "It sounds like you're going through a lot right now.",
                "I can see how that could feel upsetting.",
                "That sounds like a difficult situation to be in.",
                "I hear the struggle in what you're saying.",
                "It sounds like this has been building up for you.",
                "That must have taken a lot out of you.",
                "I can imagine how challenging that is.",
                "That sounds like it really hurt.",
                "I hear you trying to make sense of it.",
                "That seems really tough to carry.",
                "I can understand why you'd feel stuck.",
                "It sounds like you're dealing with a lot of pressure.",
                "That must have been really uncomfortable.",
                "I can see how this could be draining.",
                "That sounds emotionally heavy.",
                "I hear how much you're trying.",
                "That must be difficult to process.",
                "I can see why that would be upsetting.",
                "That sounds like a lot to handle alone.",
                "I hear your frustration.",
                "That must feel confusing and tiring.",
                "I can sense how much this matters to you.",
                "That sounds like a really hard experience.",
                "I understand why you'd feel overwhelmed.",
                "That must be weighing on your mind.",
                "I hear how challenging this has been.",
                "That sounds like a lot to sit with.",
                "I can imagine that felt intense."
            ],
            "hi": [
                "Main samajh raha hoon.", "Main aapki baat samajh raha hoon.",
                "Yeh mushkil lag raha hai.", "Yeh kaafi kathin lag raha hai.",
                "Samajh aa raha hai ki yeh aasan nahi hai.",
                "Aap jo keh rahe hain woh samajh mein aa raha hai.",
                "Yeh sun kar lagta hai ki aap par kaafi pressure hai.",
                "Yeh kaafi overwhelming lag raha hai.",
                "Main mehsoos kar sakta hoon ki yeh aapke liye tough hai.",
                "Yeh situation mushkil lag rahi hai.",
                "Aapki baat mein dard samajh aa raha hai.",
                "Yeh thoda heavy lag raha hai.",
                "Lagta hai yeh aapko kaafi affect kar raha hai.",
                "Main samajh sakta hoon ki aap aisa kyun feel kar rahe hain.",
                "Yeh kaafi frustrating lag raha hai.",
                "Aap kaafi kuch handle kar rahe hain.",
                "Yeh emotionally heavy lag raha hai.",
                "Yeh situation samajhna bhi mushkil ho sakta hai.",
                "Lagta hai aap kaafi kuch se guzar rahe hain.",
                "Yeh kaafi exhausting lag raha hai.",
                "Main dekh sakta hoon ki yeh important hai aapke liye.",
                "Yeh thoda confusing bhi lag raha hai.",
                "Aapki baat sun kar lagta hai yeh aasan nahi tha.",
                "Yeh kaafi intense lag raha hai.",
                "Lagta hai yeh aapko kaafi pareshan kar raha hai.",
                "Main mehsoos kar raha hoon ki yeh situation tough hai.",
                "Yeh aapke liye emotionally challenging lag raha hai.",
                "Lagta hai yeh aap par bojh daal raha hai.",
                "Yeh sun kar lagta hai aap struggle kar rahe hain.",
                "Aapki baat samajh aa rahi hai.",
                "Yeh kaafi bhaari lag raha hai.",
                "Aap kaafi himmat dikha rahe hain.",
                "Yeh situation handle karna mushkil ho sakta hai."
            ],
            "hinglish": [
                "I understand.", "Samajh raha hoon.", "Yeh tough lag raha hai.",
                "That sounds hard yaar.", "I get what you're saying.",
                "Yeh kaafi overwhelming lag raha hai.",
                "Samajh aa raha hai yeh easy nahi hai.",
                "That must be difficult.", "Yeh situation thodi heavy lag rahi hai.",
                "I can see why you'd feel like this.",
                "Yeh kaafi frustrating lag raha hai.",
                "Samajh raha hoon aap kya keh rahe ho.",
                "That sounds exhausting.", "Yeh kaafi pressure lag raha hai.",
                "I hear you.", "Yeh thoda confusing bhi lag raha hai.",
                "Samajh aa raha hai ki yeh tough hai.",
                "That sounds painful.", "Yeh emotionally heavy lag raha hai.",
                "I can imagine that wasn't easy.",
                "Yeh kaafi intense lag raha hai.",
                "Samajh raha hoon yeh important hai aapke liye.",
                "That sounds like a lot.", "Yeh kaafi draining lag raha hai.",
                "I can see this is affecting you.",
                "Yeh kaafi struggle lag raha hai.",
                "Samajh aa raha hai ki yeh aasan nahi tha.",
                "That sounds upsetting.",
                "Yeh kaafi heavy feel ho raha hai.",
                "I hear your frustration.",
                "Yeh thoda overwhelming lag raha hai honestly."
            ],
        }
        options = reflective.get(detected_language, reflective["en"])
    else:
        neutral = {
            "en": [
                "", "", "Okay.", "", "Alright.", "", "Got it.", "",
                "Understood.", "", "I see.", "", "Hmm.", "", "Right.", "",
                "Makes sense.", "", "Alright then.", "", "Noted.", "",
                "Okay, I see.", "", "I understand.", "", "Sure.", "",
                "Alright, got it.", "", "Okay, understood.", ""
            ],
            "hi": [
                "", "", "Theek hai.", "", "Samajh gaya.", "", "Achha.", "",
                "Ji.", "", "Theek.", "", "Samajh aaya.", "", "Haan.", "",
                "Theek hai, samajh gaya.", "", "Achha, theek hai.", "",
                "Ji samajh gaya.", "", "Theek hai ji.", "", ""
            ],
            "hinglish": [
                "", "", "Theek hai.", "", "Got it.", "", "Okay.", "",
                "Samajh gaya.", "", "Alright.", "", "Haan theek hai.", "",
                "I see.", "", "Okay got it.", "", "Samajh aa gaya.", "",
                "Right.", "", "Okay then.", "", ""
            ],
        }
        options = neutral.get(detected_language, neutral["en"])

    return _stable_choice(options, patient_text)


def _stable_choice(options: List[str], patient_text: str) -> str:
    """Deterministically choose an option based on patient text content."""
    if not options:
        return ""

    h = hashlib.md5(patient_text.encode("utf-8")).hexdigest()
    return options[int(h, 16) % len(options)]

def _deterministic_next_question(
    missing_fields_hint: str,
    detected_language: str,
    last_assistant_question: str,
    reflection_needed: bool,
    patient_text: str,
) -> str:
    labels = _parse_missing_labels(missing_fields_hint)
    last_q_norm = _normalize_text(last_assistant_question)

    if not labels:
        # All topics covered — nothing left to ask in this section.
        # Acknowledge neutrally; the background extraction task will
        # transition to the next section when ready.
        return "I understand. Let's continue when you're ready."

    target_label = labels[0]
    found_label = False
    for label in labels:
        if _normalize_text(label) not in last_q_norm:
            target_label = label
            found_label = True
            break
    if not found_label:
        target_label = labels[0]

    human_label = _humanize_label(target_label)
    prefix = _choose_prefix(detected_language, reflection_needed, patient_text)

    if prefix:
        prefix_text = f"{prefix} "
    else:
        prefix_text = ""

    if detected_language == "hi":
        return f"{prefix_text}Kya aap apna {human_label} bata sakte hain?"

    if detected_language == "hinglish":
        return f"{prefix_text}Kya aap apna {human_label} bata sakte hain?"

    return f"{prefix_text}Could you tell me your {human_label}?"


# def _is_bad_or_repetitive_response(
#     response: str,
#     patient_text: str,
#     history: List[Dict],
#     section_display_name: str,
#     missing_fields_hint: str,
# ) -> bool:
    
#     if not response.strip():
#         return True

#     lowered = response.lower()
#     banned_phrases = [
#         "it seems like",
#         "looks like",
#         "getting close",
#         "we're missing",
#         "you've shared",
#         "you've confirmed",
#         "let's move on to",
#         "how did you feel about sharing",
#         "i can't fulfill this request",
#         "i can't help with that",
#         "is there something else i can assist you with",
#     ]

#     if any(phrase in lowered for phrase in banned_phrases):
#         return True

#     if "section" in lowered or section_display_name.lower() in lowered:
#         return True

#     # Don't mirror back the patient's raw short answer in follow-up questions.
#     patient_norm = _normalize_text(patient_text)
#     response_norm = _normalize_text(response)
#     if patient_norm and len(patient_norm.split()) <= 4 and patient_norm in response_norm and "?" in response:
#         return True

#     # Prevent repeating the same question turn after turn.
#     last_q = _extract_last_assistant_question(history)
#     current_q = _extract_question(response)
#     if _normalize_text(last_q) and _normalize_text(last_q) == _normalize_text(current_q):
#         return True

#     # If there are still missing fields, force a concrete question.
#     if missing_fields_hint.strip() and "?" not in response:
#         return True

#     return False

# Keywords that indicate a question is *about* a given field. Used to detect
# when the model re-asks something already collected — the prompt's "already
# collected" list alone doesn't reliably stop a model from re-asking. Keywords
# are matched case-insensitively against the model's response; a field is only
# checked when it is ALREADY FILLED, so questions about unfilled fields are
# never affected.
_FIELD_QUESTION_KEYWORDS = {
    "name":                ["your name", "call you"],
    "age":                 ["how old", "your age"],
    "sex":                 ["your gender", "your sex", "male/female", "male or female"],
    "education":           ["education", "degree", "qualification", "studied", "schooling", "academic background"],
    "occupation":          ["occupation", "your job", "work as", "do for work", "for a living", "current work", "profession"],
    "occupational_history":["previous job", "past job", "work history", "occupational history", "previous role"],
    "ses":                 ["income", "socioeconomic", "socio-economic", "salary", "earn", "financial status"],
    "residence":           ["where do you live", "where you live", "city you", "rural area", "which city", "town you"],
    "address":             ["your address", "general location", "where are you located", "locality", "neighbourhood", "neighborhood"],
    "family_type":         ["nuclear", "joint family", "family setup", "family structure"],
    "marital_status":      ["marital status", "are you married", "relationship status"],
    "marital_history":     ["marital history", "relationship history"],
    "religion":            ["religion", "spiritual practice", "your faith"],
    "mother_tongue":       ["mother tongue", "first language", "native language"],
    "source_of_referral":  ["referred", "referral", "who sent you", "seek help today", "self-refer"],
    "informant":           ["who provided", "informant", "accompanied you"],
}


def _question_targets_filled_field(response: str, filled_field_names) -> Optional[str]:
    """
    If the response asks about a field that is already filled, return that
    field name; otherwise None. Only filled fields are checked, so this never
    blocks a legitimate question about something still missing.
    """
    if not filled_field_names:
        return None
    lowered = response.lower()
    for field in filled_field_names:
        for kw in _FIELD_QUESTION_KEYWORDS.get(field, []):
            if kw in lowered:
                return field
    return None


def _is_bad_or_repetitive_response(
    response: str,
    patient_text: str,
    history: List[Dict],
    section_display_name: str,
    missing_fields_hint: str,
    filled_field_names=None,
) -> bool:
    logger.info(f"\n[VALIDATOR] checking response: {response!r}")

    if not response.strip():
        logger.warning(f"[VALIDATOR] FAIL — empty response")
        return True

    # Re-asking an already-collected field — the most common failure mode.
    refilled = _question_targets_filled_field(response, filled_field_names)
    if refilled:
        logger.warning(f"[VALIDATOR] FAIL — re-asks already-filled field: {refilled!r}")
        return True

    lowered = response.lower()
    banned_phrases = [
        "it seems like",
        "looks like",
        "getting close",
        "we're missing",
        # "you've shared",
        "you've confirmed",
        "let's move on to",
        "how did you feel about sharing",
        "i can't fulfill this request",
        "i can't help with that",
        "is there something else i can assist you with",
        "topic:",
        "assistant",
        "patient"
    ]
    for phrase in banned_phrases:
        if phrase.lower() in lowered:
            logger.warning(f"[VALIDATOR] FAIL — banned phrase: {phrase!r}")
            return True

    import re
    if "section" in lowered:
        logger.warning(f"[VALIDATOR] FAIL — word 'section' found")
        return True
    section_name_pattern = re.escape(section_display_name.lower()) + r'(?!\w)'
    if re.search(section_name_pattern, lowered):
        logger.warning(f"[VALIDATOR] FAIL — section display name {section_display_name.lower()!r} found")
        return True

    patient_norm = _normalize_text(patient_text)
    response_norm = _normalize_text(response)
    if patient_norm and len(patient_norm.split()) <= 4 and "?" in response:
        # only flag if response question directly asks for what patient just said
        # not if it merely acknowledges it and moves on
        sentences = re.split(r'(?<=[.!])\s+', response)
        question_sentences = [s for s in sentences if '?' in s]     
        question_part = question_sentences[-1] if question_sentences else response
        question_norm = _normalize_text(question_part)
        if re.search(rf"\b{re.escape(patient_norm)}\b", question_norm):  # word boundary, not substring
            logger.warning(f"[VALIDATOR] FAIL — mirrors patient text in question: {patient_norm!r} in {question_norm!r}")
            return True

    last_q = _extract_last_assistant_question(history)
    current_q = _extract_question(response)
    logger.info(f"[VALIDATOR] last_q:    {last_q!r}")
    logger.info(f"[VALIDATOR] current_q: {current_q!r}")
    logger.info(f"[VALIDATOR] normalized last_q:    {_normalize_text(last_q)!r}")
    logger.info(f"[VALIDATOR] normalized current_q: {_normalize_text(current_q)!r}")
    logger.info(f"[VALIDATOR] exact match: {_normalize_text(last_q) == _normalize_text(current_q)}")

    if _normalize_text(last_q) and _normalize_text(last_q) == _normalize_text(current_q):
        logger.warning(f"[VALIDATOR] FAIL — exact repeat of last question")
        return True

    if missing_fields_hint.strip() and "?" not in response:
        logger.warning(f"[VALIDATOR] FAIL — no question mark but fields missing")
        return True

    logger.info(f"[VALIDATOR] PASS — response accepted")
    return False


async def _load_model() -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    global TOKENIZER, MODEL, DEVICE

    if LLM_BACKEND == "vllm":
        # Hard guard. On the vLLM backend, generation happens in the server;
        # loading the HF model here would pin ANOTHER ~28GB of the same GPU on
        # top of the server's allocation (this is what put GPU 3 at ~72GB).
        # Note the prompt-audit logging in get_next_response / extract_info /
        # harvest_info calls this purely to render a chat template for the log —
        # each of those wraps it in try/except and degrades to plain-text
        # logging, which is exactly what we want here.
        raise RuntimeError(
            "_load_model() called while LLM_BACKEND=vllm — refusing to load a "
            "second in-process copy of the model."
        )

    if TOKENIZER is not None and MODEL is not None and DEVICE is not None:
        return TOKENIZER, MODEL

    async with MODEL_LOCK:
        if TOKENIZER is not None and MODEL is not None and DEVICE is not None:
            return TOKENIZER, MODEL

        # NOTE: this used to be `device, torch.dtype = _select_device()`, which
        # assigned the dtype INTO the torch module's own `dtype` attribute,
        # clobbering it process-wide. Keep it in a local.
        device, model_dtype = _select_device()
        DEVICE = device

        def load_in_thread() -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
            tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAME,
                local_files_only=LOCAL_FILES_ONLY,
                use_fast=True,
            )

            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token

            model_kwargs = {
                "local_files_only": LOCAL_FILES_ONLY,
                # transformers 4.x takes `torch_dtype`; only 5.x renamed it to
                # `dtype`. We pin 4.51.3 (vLLM 0.8.5's requirement), so passing
                # `dtype` here raised: Qwen3ForCausalLM.__init__() got an
                # unexpected keyword argument 'dtype'.
                "torch_dtype": model_dtype,
                "low_cpu_mem_usage": True,
            }

            if device.type == "cuda":
                    model_kwargs["device_map"] = "auto"
            model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)
            
            print("Model device:", next(model.parameters()).device)
            
            if device.type in {"cpu", "mps"}:
                model = model.to(device)

            model.eval()
            return tokenizer, model

        TOKENIZER, MODEL = await asyncio.to_thread(load_in_thread)
        return TOKENIZER, MODEL


async def warmup_model() -> None:
    """
    Prepare the configured backend at startup so the first user turn is fast.

    On the vllm backend this process must NOT load the model: generation happens
    in the vLLM server, and loading a second copy here would pin another ~28GB of
    the SAME GPU on top of the server's own allocation (that's what pushed GPU 3
    to ~72GB). We just check the server is reachable so failures surface here
    instead of as a "technical issue" on every patient turn.
    """
    if LLM_BACKEND == "vllm":
        print(f"LLM_BACKEND=vllm — no in-process model load; using {VLLM_BASE_URL}")
        try:
            import httpx

            # trust_env=False: never route a localhost call through the box's
            # http_proxy (see _call_vllm) — it's a Squid proxy that 503s.
            async with httpx.AsyncClient(timeout=10, trust_env=False) as client:
                resp = await client.get(f"{VLLM_BASE_URL}/models")
                resp.raise_for_status()
            print(f"vLLM server reachable; serving model {VLLM_MODEL!r}.")
        except Exception as error:
            # Non-fatal: the server may still be loading. Warn loudly instead of
            # crashing the bot at boot.
            print(f"WARNING: vLLM server not reachable at {VLLM_BASE_URL} ({error}). "
                  f"Start it with scripts/serve_vllm.sh, or set LLM_BACKEND=hf.")
        return

    print("Warming up language model...")
    await _load_model()
    print("Language model ready.")


def _is_qwen3_model() -> bool:
    return bool(MODEL_NAME and "qwen3" in MODEL_NAME.lower())


def _build_chat_prompt(tokenizer, system_prompt: str, prompt: str, think: Optional[bool] = None) -> str:
    # think=None → use the global ENABLE_THINKING default; otherwise per-call override.
    think_enabled = ENABLE_THINKING if think is None else think
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if prompt:
        messages.append({"role": "user", "content": prompt})

    if messages and hasattr(tokenizer, "apply_chat_template"):
        if _is_qwen3_model():
            # Qwen3 thinking mode is gated by think_enabled.
            # enable_thinking= requires transformers >= 4.51.0; on older
            # versions fall back to a /no_think prefix only when thinking is OFF
            # (Qwen3 defaults to thinking ON, so nothing to prepend when ON).
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=think_enabled,
                )
            except TypeError:
                if not think_enabled and messages and messages[-1]["role"] == "user":
                    messages[-1] = {**messages[-1], "content": "/no_think " + messages[-1]["content"]}

        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    parts = []
    if system_prompt:
        parts.append(system_prompt)
    if prompt:
        parts.append(prompt)
    return "\n\n".join(parts)


def _parse_json_response(response_text: str) -> Optional[Dict]:
    def try_parse(candidate: str) -> Optional[Dict]:
        candidate = candidate.strip()
        if not candidate:
            return None

        # Remove markdown fences if present.
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\s*```$", "", candidate)

        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass

        # Handle common model outputs that are Python-dict-like (single quotes,
        # True/False/None) instead of strict JSON.
        try:
            parsed = ast.literal_eval(candidate)
            return parsed if isinstance(parsed, dict) else None
        except (ValueError, SyntaxError):
            return None

    # 1) Try raw response first.
    parsed = try_parse(response_text)
    if parsed is not None:
        return parsed

    # 2) Try fenced JSON block.
    fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", response_text, flags=re.IGNORECASE)
    for block in fenced:
        parsed = try_parse(block)
        if parsed is not None:
            return parsed

    # 3) Extract balanced JSON-like object candidates and parse each.
    candidates: List[str] = []
    stack = 0
    start = -1
    for index, char in enumerate(response_text):
        if char == "{":
            if stack == 0:
                start = index
            stack += 1
        elif char == "}" and stack > 0:
            stack -= 1
            if stack == 0 and start != -1:
                candidates.append(response_text[start:index + 1])
                start = -1

    # Prefer longer candidates first (usually full payload over nested object).
    for candidate in sorted(candidates, key=len, reverse=True):
        parsed = try_parse(candidate)
        if parsed is not None:
            return parsed

    return None


def _normalize_extraction_result(parsed: Optional[Dict]) -> Dict:
    """Return extraction payload with a stable schema and safe types."""
    default_payload = {"updates": {}, "free_notes": "", "section_complete_signal": False}
    if not isinstance(parsed, dict):
        return default_payload

    updates = parsed.get("updates", {})
    if not isinstance(updates, dict):
        updates = {}

    cleaned_updates: Dict[str, str] = {}
    for key, value in updates.items():
        if isinstance(key, str) and value is not None:
            cleaned_updates[key] = str(value).strip()

    free_notes = parsed.get("free_notes", "")
    if free_notes is None:
        free_notes = ""

    section_complete_signal = parsed.get("section_complete_signal", False)
    if isinstance(section_complete_signal, str):
        section_complete_signal = section_complete_signal.strip().lower() in {"true", "1", "yes"}

    return {
        "updates": cleaned_updates,
        "free_notes": str(free_notes).strip(),
        "section_complete_signal": bool(section_complete_signal),
    }


def _extract_topic_id(response: str) -> tuple[str, str]:
    """
    Extract topic_id from LLM response.
    LLM appends '[TOPIC: <topic_id>]' anywhere in response.
    Returns (clean_response, topic_id).
    """
    import re
    match = re.search(r'\[TOPIC:\s*([^\]]+)\]', response, re.IGNORECASE)
    if match:
        topic_id = match.group(1).strip().lower()
        # remove the [TOPIC:...] line including surrounding newlines
        clean = re.sub(r'\n*\[TOPIC:\s*[^\]]+\]\n*', '', response, flags=re.IGNORECASE).strip()
        return clean, topic_id
    return response.strip(), "unknown"


def _strip_think_tokens(text: str) -> str:
    """
    Remove <think>...</think> blocks emitted by reasoning models
    (Qwen3, DeepSeek-R1, etc.) before their actual answer.
    Handles multiline blocks and multiple occurrences.

    Guardrail: if an opening <think> is present but no closing </think>
    (i.e. the block was truncated mid-thought), the entire response is
    considered corrupt and an empty string is returned — callers treat
    that as a fallback trigger.
    """
    # Check for unclosed <think> block — truncated mid-thought
    open_count = len(re.findall(r"<think>", text, flags=re.IGNORECASE))
    close_count = len(re.findall(r"</think>", text, flags=re.IGNORECASE))
    if open_count > close_count:
        logger.warning("[LLM] Unclosed <think> block detected — triggering fallback")
        return ""

    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


# How many tokens the model may spend on <think>...</think> reasoning.
# Uses a LogitsProcessor to force </think> once the budget is exhausted —
# the model then continues generating its actual answer within max_tokens.
# Set to 0 to disable thinking entirely.
MAX_THINKING_TOKENS = int(os.getenv("LLM_MAX_THINKING_TOKENS", "512"))


class ThinkingTokenBudgetProcessor:
    """
    Logits processor that caps Qwen3 / DeepSeek-R1 style thinking tokens.

    Once MAX_THINKING_TOKENS tokens have been generated inside a <think> block,
    this processor forces the model to emit </think> and start its real answer.

    Adapted from: https://muellerzr.github.io/til/end_thinking.html
    Works with any model that uses <think>...</think> delimiters.
    """

    def __init__(self, tokenizer, max_thinking_tokens: int):
        self.max_thinking_tokens = max_thinking_tokens
        self.think_end_token = tokenizer.encode("</think>", add_special_tokens=False)[0]
        self.nl_token = tokenizer.encode("\n", add_special_tokens=False)[0]
        self.tokens_generated = 0
        self.stopped_thinking = False
        self.neg_inf = float("-inf")

    def __call__(self, input_ids: "torch.LongTensor", scores: "torch.FloatTensor") -> "torch.FloatTensor":
        self.tokens_generated += 1

        # max_thinking_tokens=0 → skip thinking entirely on the very first token
        if self.max_thinking_tokens == 0 and not self.stopped_thinking:
            scores[:] = self.neg_inf
            scores[0][self.nl_token] = 0
            scores[0][self.think_end_token] = 0
            self.stopped_thinking = True
            return scores

        if self.max_thinking_tokens and not self.stopped_thinking:
            ratio = self.tokens_generated / self.max_thinking_tokens

            # At 95 % of budget, start nudging logits toward </think>
            if ratio > 0.95:
                boost = 1 + ratio
                scores[0][self.nl_token] = scores[0][self.think_end_token] * boost
                scores[0][self.think_end_token] = scores[0][self.think_end_token] * boost

            # At 100 % of budget, force the closing sequence
            if self.tokens_generated >= self.max_thinking_tokens - 1:
                scores[:] = self.neg_inf
                if self.tokens_generated == self.max_thinking_tokens - 1:
                    scores[0][self.nl_token] = 0       # emit \n first
                else:
                    scores[0][self.think_end_token] = 0  # then </think>
                    self.stopped_thinking = True

        return scores


async def _call_vllm(
    prompt: str,
    system_prompt: str = "",
    max_tokens: int = 200,
    think: Optional[bool] = None,
    guided_json: Optional[Dict] = None,
) -> str:
    """
    Call a local vLLM OpenAI-compatible server (/v1/chat/completions).

    We send raw chat messages and let vLLM apply the model's chat template
    server-side (so prefix caching sees the identical rendered system prompt
    every turn). Qwen3 thinking is toggled via chat_template_kwargs, matching
    _build_chat_prompt's enable_thinking. guided_json, when given, forces the
    output to conform to that JSON schema via vLLM's guided decoding (XGrammar).
    """
    think_enabled = ENABLE_THINKING if think is None else think
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if prompt:
        messages.append({"role": "user", "content": prompt})

    thinking_budget = MAX_THINKING_TOKENS if think_enabled else 0
    payload: Dict = {
        "model": VLLM_MODEL,
        "messages": messages,
        "max_tokens": max_tokens + thinking_budget,
        "temperature": 0.7,
        "top_p": 0.9,
        # Qwen3: disable the reasoning span unless this call opted into thinking.
        "chat_template_kwargs": {"enable_thinking": think_enabled},
    }
    if guided_json is not None:
        # vLLM guided decoding — guarantees schema-valid JSON, no parse retries.
        payload["guided_json"] = guided_json

    headers = {"Authorization": f"Bearer {VLLM_API_KEY}"}

    # Tell the GPU manager a request is in flight, so its idle timer can never
    # pull the server out from under a generation that is already running. This
    # is the authoritative in-flight signal — handler-level activity tracking
    # only sees inbound messages, not the work they kick off.
    import gpu_manager
    gpu_manager.begin_turn()
    try:
        return await _call_vllm_inner(payload, headers)
    finally:
        gpu_manager.end_turn()


async def _call_vllm_inner(payload: Dict, headers: Dict) -> str:
    import httpx

    # trust_env=False is REQUIRED here. This box sets http_proxy/https_proxy
    # (Squid), and httpx honours those by default — so a request to
    # localhost:8100 got sent to the proxy, which returned its own HTML error
    # page as a 503. curl bypasses the proxy for localhost, which is why curl
    # worked while the bot silently fell back to canned questions on every turn.
    # We only ever talk to a local vLLM server, so a proxy is never correct.
    async with httpx.AsyncClient(timeout=VLLM_TIMEOUT_S, trust_env=False) as client:
        resp = await client.post(
            f"{VLLM_BASE_URL}/chat/completions", json=payload, headers=headers
        )
        if resp.status_code >= 400:
            # vLLM puts the ACTUAL reason (bad model name, chat-template error,
            # unsupported param) in the body. raise_for_status() drops it, which
            # turns every failure into a silent fallback to canned questions.
            raise RuntimeError(
                f"vLLM {resp.status_code} for model={VLLM_MODEL!r} at "
                f"{VLLM_BASE_URL}: {resp.text[:800]}"
            )
        data = resp.json()
    raw = (data["choices"][0]["message"]["content"] or "").strip()
    # Strip any residual <think> block (same contract as the HF path).
    return _strip_think_tokens(raw)


async def call_llm(
    prompt: str,
    system_prompt: str = "",
    max_tokens: int = 200,
    think: Optional[bool] = None,
    guided_json: Optional[Dict] = None,
) -> str:
    """
    Call the local LLM. Dispatches to the configured backend (LLM_BACKEND):
    "vllm" → local vLLM OpenAI server; "hf" (default) → in-process transformers.

    think: per-call thinking override. None → use the global ENABLE_THINKING
    default. Pass think=False to force a fast no-think call (e.g. extraction).

    guided_json: optional JSON Schema. On the vLLM backend it forces
    schema-valid output (guided decoding); on the HF backend it is ignored
    (callers still parse best-effort), so structured callers must tolerate both.

    For thinking models (Qwen3, DeepSeek-R1):
    - ThinkingTokenBudgetProcessor caps reasoning to MAX_THINKING_TOKENS
    - _strip_think_tokens() removes any residual <think> blocks from output
    - Unclosed <think> blocks (truncated mid-thought) return "" → caller fallback

    max_tokens controls the answer budget only; thinking tokens are separate.
    """
    if LLM_BACKEND == "vllm":
        try:
            return await _call_vllm(prompt, system_prompt, max_tokens, think, guided_json)
        except Exception as error:
            print(f"vLLM call failed: {error}")
            raise

    think_enabled = ENABLE_THINKING if think is None else think
    tokenizer, model = await _load_model()
    device = DEVICE or torch.device("cpu")

    def run_generation() -> str:
        full_prompt = _build_chat_prompt(tokenizer, system_prompt, prompt, think=think_enabled)
        inputs = tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_CONTEXT_TOKENS,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # Only reserve a thinking budget / install the budget processor when
        # thinking is actually enabled. When OFF, the chat template already
        # closed the <think> block, so there's nothing to cap and no extra
        # tokens to reserve (original fast path).
        thinking_budget = MAX_THINKING_TOKENS if think_enabled else 0

        logits_processors = None
        if think_enabled and MAX_THINKING_TOKENS:
            from transformers import LogitsProcessorList
            processor = ThinkingTokenBudgetProcessor(tokenizer, MAX_THINKING_TOKENS)
            logits_processors = LogitsProcessorList([processor])

        generation_kwargs = {
            "max_new_tokens": max_tokens + thinking_budget,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if logits_processors:
            generation_kwargs["logits_processor"] = logits_processors

        with torch.inference_mode():
            output_ids = model.generate(**inputs, **generation_kwargs)

        input_length = inputs["input_ids"].shape[-1]
        new_tokens = output_ids[0][input_length:]
        raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Strip any residual think blocks; returns "" if block was unclosed
        return _strip_think_tokens(raw)

    try:
        return await asyncio.to_thread(run_generation)
    except Exception as error:
        print(f"LLM call failed: {error}")
        raise


# Deterministic filler removal.
#
# Banning openers by STRING does not work: the model simply rotates synonyms —
# "I see" -> "That's helpful" -> "I understand" -> "That's good to know" ->
# "Thank you for sharing" — while the underlying template ("<filler> — can I
# ask <question>?") survives untouched on every single turn. Prompt rules have
# now failed at this four times, so we no longer ask: we delete it after
# generation. A separator is REQUIRED after the filler, which is what keeps this
# from eating genuine empathy ("I hear that panic attacks bring up..." has no
# separator and is left alone).
_FILLER_OPENER_RE = re.compile(
    r"^\s*(?:"
    r"i see|i understand|i hear (?:you|that)|got it|okay|ok|alright|all right|"
    r"sure|right|thanks|thank you(?: for sharing| for telling me)?|noted|"
    r"understood|good to know|fair enough|of course|certainly|"
    r"that(?:'|’)?s (?:helpful|great|good(?: to know)?|interesting|important|"
    r"useful|a good [\w\s]{1,20}|a solid [\w\s]{1,20}|a great [\w\s]{1,20})|"
    r"that (?:makes sense|helps(?: me understand)?[\w\s]{0,20})|makes sense"
    r")\s*[—–\-,.:!]+\s*",
    re.IGNORECASE,
)


# The model's habitual joint: "<acknowledgement> — <the actual question>".
# Catches evaluative openers the fixed list can't enumerate ("That's a good age
# to be exploring and growing — ...").
_ACK_BRIDGE_RE = re.compile(r"^\s*([^—–?!]{1,90}?)\s*[—–]\s*(.+)$", re.DOTALL)


def _strip_leading_filler(text: str) -> str:
    """Remove a leading throat-clearing clause ("I see — ", "That's helpful — ").

    Returns text unchanged if stripping would leave nothing useful.
    """
    if not text:
        return text
    stripped = _FILLER_OPENER_RE.sub("", text, count=1).lstrip()
    if len(stripped) < 8:
        return text
    # Fixed list missed it — fall back to the dash bridge: if everything before
    # the dash is a non-question clause and a question follows, the clause is
    # throat-clearing regardless of how it's worded.
    if stripped == text:
        m = _ACK_BRIDGE_RE.match(text)
        if m:
            head, rest = m.group(1), m.group(2).strip()
            if "?" not in head and "?" in rest and len(rest) >= 8:
                stripped = rest
    if stripped == text or len(stripped) < 8:
        return text
    return stripped[0].upper() + stripped[1:]


# "Can I ask about X?" — the model wraps every question in a permission clause.
# Two harms, and the second is not cosmetic:
#   1. It opens every single turn identically, which reads as a form.
#   2. It makes yes/no answers AMBIGUOUS. "Can I ask about any joint or muscle
#      pain?" -> "no" was recorded as `musculoskeletal_negative: denied` — but
#      "no" there literally declines permission. The whole review-of-systems
#      block in one real session was "no / no problems / no", every one banked as
#      a clinical negative.
# It is also hollow: the bot asks permission it does not honour.
# The data_model questions are already well-formed and direct, so we just unwrap.
_CAN_I_ASK_RE = re.compile(
    r"^\s*(?:and\s+|so\s+)?(?:can|could|may)\s+i\s+ask\s+(?:you\s+)?(?:about\s+)?",
    re.IGNORECASE,
)
_WH_WORDS = ("what", "how", "when", "where", "why", "whether", "if", "who", "which")
# A wh-question is already direct when an auxiliary/verb follows the wh-word
# ("what MAKES it worse", "how ARE you sleeping"). If a subject follows instead
# ("how long THIS has been going on") it's an embedded clause and unwrapping it
# would produce broken grammar, so we leave those wrapped.
_DIRECT_Q_VERBS = frozenset({
    "is", "are", "was", "were", "do", "does", "did", "has", "have", "had",
    "can", "could", "will", "would", "should", "makes", "made", "make",
    "brings", "brought", "happens", "happened", "causes", "caused", "feels",
})

# "Can I ask IF you've ever been diagnosed...?" is the model's most common
# wrapper and the one a bare unwrap can't handle: dropping "can I ask" leaves
# "If you've ever been diagnosed...?", which is not a sentence. These invert the
# embedded clause back into a real question ("Have you ever been diagnosed...?").
# Ordered longest-first so "if you have" wins before "if you".
_IF_CLAUSE_INVERSIONS = (
    ("if you have ever", "Have you ever"),
    ("if you've ever", "Have you ever"),
    ("if you have", "Have you"),
    ("if you've", "Have you"),
    ("if you had", "Had you"),
    ("if you are", "Are you"),
    ("if you're", "Are you"),
    ("if you were", "Were you"),
    ("if you can", "Can you"),
    ("if you do", "Do you"),
    ("if you ever", "Do you ever"),
    ("if there are", "Are there"),
    ("if there is", "Is there"),
    ("if there's", "Is there"),
    ("if there has been", "Has there been"),
    ("if anything", "Does anything"),
    ("if that", "Does that"),
    ("if this", "Does this"),
    ("if it", "Does it"),
)


def _unwrap_permission_question(text: str) -> str:
    """Turn "Can I ask about any joint pain?" into "Any joint pain?" etc."""
    if not text:
        return text
    m = _CAN_I_ASK_RE.match(text)
    if not m:
        return text
    rest = text[m.end():].strip()
    if len(rest) < 4:
        return text
    tokens = rest.split()
    first = tokens[0].lower().strip(",")
    second = tokens[1].lower().strip(",") if len(tokens) > 1 else ""
    lowered = rest.lower()
    inverted = next(
        ((pre, sub) for pre, sub in _IF_CLAUSE_INVERSIONS if lowered.startswith(pre)),
        None,
    )
    if inverted:
        # "can I ask if you've ever been diagnosed...?"
        #   -> "Have you ever been diagnosed...?"
        pre, sub = inverted
        out = sub + rest[len(pre):]
    elif first in _WH_WORDS:
        # Guarded on the verb: an EMBEDDED clause ("...how long this has been
        # going on") needs auxiliary inversion a regex can't do, and unwrapping
        # it blindly yields "How long this has been going on?". When the shape
        # isn't already a direct question, leave the reply alone — a wrapped
        # question is bad, a broken one is worse.
        if second not in _DIRECT_Q_VERBS:
            return text
        # "can I ask what makes it worse?" -> "What makes it worse?"
        out = rest
    elif first == "any":
        # "can I ask about any joint pain?" -> "Do you have any joint pain?"
        out = "Do you have " + rest
    elif first in ("your", "the", "his", "her", "their"):
        # "can I ask about your sleep patterns?" -> "Tell me about your sleep patterns."
        out = "Tell me about " + rest.rstrip("?").rstrip() + "."
    else:
        out = rest
    return out[0].upper() + out[1:]


def _has_filler_opener(text: str) -> bool:
    """True if this reply opens with an acknowledgement rather than substance."""
    return bool(text) and _strip_leading_filler(text) != text


def _opening_clause(text: str, max_words: int = 9) -> str:
    """First clause of a reply — the part that kept repeating verbatim.

    Cuts at the first sentence/clause boundary so we compare "You're 21. That
    helps me understand your background" style openers rather than whole replies.
    """
    if not text:
        return ""
    cleaned = re.split(r"\[TOPIC:", text)[0].strip()
    # Stop at the first sentence end or em-dash bridge, whichever comes first.
    m = re.split(r"(?<=[.!?])\s+|\s+—\s+", cleaned)
    head = (m[0] if m else cleaned).strip()
    words = head.split()
    if len(words) > max_words:
        head = " ".join(words[:max_words])
    return head.strip(" -—,")


# ── Phase 2: the planner ────────────────────────────────────────────────────
# The original design (clinical_state.py:6-8) deliberately reduced the model to
# NLG: keyword lists and counters decided everything, and the model only
# rephrased those decisions. That's why the interview marched one-question-at-a-
# time, re-asked known facts, couldn't follow the patient's lead, and ignored
# meta-questions ("do you think it is okay?"). The planner gives that judgment
# back to the model in a SMALL, INSPECTABLE, constrained step: it picks a
# response strategy (and a target) BEFORE generation, and the reply is then
# conditioned on it. The old rule-based decide_next() stays as the backstop —
# if planning fails we fall through to exactly the previous behaviour.
_PLAN_STRATEGIES = (
    "ask",               # routine fact — just ask, do NOT reflect it back
    "reflect_then_ask",  # only when what they said carries weight
    "go_deeper",         # thread is clinically rich; stay on it another turn
    "clarify",           # answer was ambiguous/too brief to use
    "answer_meta",       # patient asked ABOUT the process — answer, then continue
    "skip",              # already known or irrelevant — ask something else
    "transition",        # this area is covered — move on
    "de_escalate",       # safety concern surfaced — address it directly
    "respect_refusal",   # patient declined this topic — drop it, do not re-ask
    "reorient",          # reply isn't a usable answer — flag it, don't interpret
    "wrap_up",           # enough gathered; close out warmly
)

_STRATEGY_GUIDANCE = {
    "ask": (
        "They gave a routine factual answer. Keep it warm but light — this "
        "should feel like a person talking, not a form.\n"
        "- You MAY use a short, natural acknowledgement, but VARY it every time "
        "and never reuse the one from your previous reply. Never lean on one "
        "stock phrase (\"Got it\" every turn is exactly as robotic as no "
        "acknowledgement at all).\n"
        "- Do NOT repeat their answer back to them (\"You're 21...\", \"You're "
        "currently living in...\").\n"
        "- Do NOT comment on what their answer tells you (\"That helps me "
        "understand...\", \"That's helpful\").\n"
        "- NEVER evaluate, judge, or editorialise a demographic fact. Banned: "
        "\"That's a great age to be exploring and growing\", \"That's a good "
        "start\" (about income), \"That's a solid foundation\", \"It sounds "
        "like you're in a role that keeps you busy\". Their age, income, job, "
        "religion and marital status are facts to record, not things to have an "
        "opinion about — and you know far too little to have one.\n"
        "- Do NOT invent detail they did not give you. \"I am working as a RA\" "
        "tells you nothing about whether they are busy.\n"
        "Then ask the next question."
    ),
    "reflect_then_ask": (
        "What they said carries some weight. Reflect the MEANING of it in your "
        "own words in one short clause — never by restating their answer back — "
        "then ask the next question."
    ),
    "go_deeper": (
        "This thread matters clinically and is NOT exhausted. Stay with it: ask one "
        "concrete follow-up that deepens THIS topic (what happened next, when it "
        "started, how it affected them). Do not change subject."
    ),
    "clarify": (
        "Their answer was too brief or ambiguous to use. Ask one specific, easy "
        "question that pins down what they meant. Do not re-ask the whole question."
    ),
    "answer_meta": (
        "They asked about YOU or about this process (why you're asking, how long "
        "it takes, whether something is normal), not about their history.\n"
        "- Answer THEIR question first — directly, warmly, briefly, in one or "
        "two sentences. Never ignore it.\n"
        "- Then RE-ASK THE QUESTION YOU JUST ASKED, the one they were querying. "
        "Do NOT move on to a different topic. You explained why you needed to "
        "know about their school memories and then asked about their job "
        "instead; they replied \"so do you want to know about school memories "
        "or not\" — and they were right. Explaining why you need something and "
        "then not asking for it is worse than never explaining.\n"
        "- If they push back a second time, do not explain again and do not "
        "argue — treat it as a refusal and drop the topic for good."
    ),
    "skip": (
        "You already know this, or it does not apply. Do NOT ask it again. "
        "Acknowledge briefly and ask about something genuinely new instead."
    ),
    "transition": (
        "This area is covered. Acknowledge what they said in one clause, then move "
        "to the next area in the same message."
    ),
    "de_escalate": (
        "They may have expressed hopelessness, self-harm, or not wanting to be "
        "alive — possibly indirectly. Do NOT continue the routine interview. "
        "Respond with care to what they actually said and ask one gentle, direct "
        "question about their safety. Do not thank them or use filler.\n"
        "TENSE MATTERS: establish WHEN, plainly — are these thoughts here RIGHT "
        "NOW, or did they happen in the past / only during an episode? Ask it "
        "directly (\"Are you having those thoughts right now, or was that during "
        "the panic attack?\"). If they have already told you it is not happening "
        "now, ACCEPT it — do not keep re-asking the same thing in new words."
    ),
    "respect_refusal": (
        "They have declined to discuss this topic. Accept it — genuinely, not as "
        "a transition trick.\n"
        "- Acknowledge the refusal in ONE short line, without arguing, "
        "bargaining, or explaining why you need it.\n"
        "- Do NOT ask anything about this topic again, in any form. Rewording it "
        "is still asking it.\n"
        "- Do NOT say \"let's move to another topic\" and then ask another "
        "question about the SAME topic — that is what you did to a patient who "
        "refused nine times, until she gave in.\n"
        "- Move to a genuinely DIFFERENT area, or if they asked to stop or "
        "pause, stop. If they said they want to continue another day, say yes "
        "and ask nothing further.\n"
        "It is always acceptable to leave a field uncollected. It is never "
        "acceptable to pressure a patient into answering."
    ),
    "reorient": (
        "Their message cannot be taken as a genuine answer — nonsense, joking, "
        "role-play, or testing you. Do NOT play along, and do NOT interpret it.\n"
        "- Never validate it. Real replies you gave a patient who said she was "
        "a cat: \"That's a lovely detail.\" / \"That's a charming "
        "introduction!\" Both are wrong.\n"
        "- Never invent a feeling from it. You answered \"Meow meow\" with "
        "\"It sounds like you're feeling a bit overwhelmed.\" She told you "
        "nothing of the kind.\n"
        "- Never diagnose, speculate, or comment on WHY they said it. You "
        "cannot tell from a chat whether something odd is a joke or a genuine "
        "symptom, and guessing wrong either way harms them.\n"
        "- Say warmly and plainly that you want to be sure you understand, and "
        "invite them to answer in their own words. Ask ONCE — do not repeat the "
        "same question at them.\n"
        "- If they carry on, don't argue and don't lecture. Offer to pick this "
        "up when they're ready."
    ),
    "wrap_up": (
        "You have enough for now. Acknowledge warmly and confirm rather than "
        "opening a new topic."
    ),
}

# Human-readable targets for the planner's chosen field. FIELD_HUMANIZE has no
# entries for these, and the raw name is useless to the model ("ASK ABOUT THIS:
# complaint 1"). These labels also disambiguate fields the patient experiences
# as identical — residence vs address both surfaced as "where do you live?",
# so she answered "city", then "delhi", then "i told u".
_TARGET_LABELS = {
    "complaint_1": "the main problem that brought them here today — what is actually troubling them",
    "complaint_2": "any other problem they are having, besides the main one",
    "complaint_3": "any further problem they are having",
    "duration_each": "how long each of those problems has been going on",
    "impact_on_functioning": "how these problems affect their day-to-day life",
    "residence": "whether they live in a city/town or a rural area (NOT their address)",
    "address": "which area or locality they live in (NOT city-vs-rural)",
    "ses": "their monthly household income",
    "source_of_referral": "who sent them here",
    "informant": "who came with them or gave the information",
    "family_type": "whether they live in a nuclear or joint family",
    "mother_tongue": "the language they are most comfortable speaking",
}


def _target_label(field_name: str) -> str:
    """Readable description of a planner target for the reply prompt."""
    if field_name in _TARGET_LABELS:
        return _TARGET_LABELS[field_name]
    try:
        from data_model import FIELD_HUMANIZE
        humanized = FIELD_HUMANIZE.get(field_name)
        if humanized:
            return humanized
    except Exception:
        pass
    return field_name.replace("_", " ")


_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "strategy": {"type": "string", "enum": list(_PLAN_STRATEGIES)},
        "target_field": {"type": "string"},
        "safety_concern": {"type": "boolean"},
        "rationale": {"type": "string"},
    },
    "required": ["strategy", "target_field", "safety_concern", "rationale"],
    "additionalProperties": False,
}


async def plan_turn(
    patient_text: str,
    history: List[Dict],
    section_display_name: str,
    filled_fields_summary: str,
    unfilled_fields: List[str],
    detected_language: str = "en",
) -> Optional[Dict]:
    """
    Decide the response STRATEGY for this turn before generating.

    Small, cheap call: tiny constrained output, and on the vllm backend the long
    conversation prefix is prefix-cached, so this costs far less than a reply.
    Returns None on any failure so the caller falls back to the old rule-based
    routing rather than blocking the turn.
    """
    conversation = ""
    for msg in history[-8:]:
        role = "Patient" if msg.get("role") == "user" else "Assistant"
        conversation += f"{role}: {msg.get('content', '')}\n"

    unfilled_str = ", ".join(unfilled_fields[:20]) if unfilled_fields else "(none left)"

    system_prompt = f"""You are the planner for a psychiatric intake interview. You do NOT write the reply. You decide HOW the interviewer should respond to the patient's latest message.

Current area: {section_display_name}
Already known: {filled_fields_summary or "nothing yet"}
Still needed in this area: {unfilled_str}

Choose exactly one strategy:
- ask: DEFAULT for routine factual answers (age, gender, education, job, address, income, mother tongue, a plain yes/no to a symptom). Just ask the next thing — no reflection. Most demographic turns are this.
- reflect_then_ask: ONLY when what they said carries emotional or clinical weight (a symptom, a loss, a fear, an impact on their life). Never for a plain fact like an address or a salary.
- go_deeper: they opened something clinically important and it is NOT yet understood. Stay on it.
- clarify: their answer was too vague/brief to record.
- answer_meta: they asked about the process or about you ("why do you ask?", "is that normal?", "how long is this?", "do you think it is okay?") rather than answering. This ALWAYS takes priority over asking the next question.
- skip: the information is already known from the conversation, or does not apply. Never re-ask it.
- transition: this area is sufficiently covered; move to the next.
- de_escalate: they hinted at self-harm, hopelessness, or not wanting to be alive — INCLUDING indirect phrasing. Overrides everything else.
- reorient: their message is not a usable answer — nonsense, joking, role-play, or testing the system ("Meow meow", "I identify as a cat, call me Dr. Meowsy", "Test", random characters). Choose this INSTEAD of trying to interpret it. Important: do NOT conclude they are joking — bizarre statements can equally be a genuine symptom. You are only saying "this cannot be used as an answer", not saying why. A human clinician will judge it.
- respect_refusal: they declined this topic, asked you to stop, asked to continue another time, or objected to being asked ("I don't want to", "don't ask me these questions", "can we discuss this later", "let's wrap up", "please don't ask more"). Choose this the FIRST time they say it — do not wait for them to repeat themselves. A refusal is an answer, not an obstacle.
- wrap_up: enough has been gathered overall.

Rules:
- FIRST decide whether their message actually ANSWERED the question you just asked. A bare "yes", "ok", "sure", "hmm", or "yeah" to an open question (education, occupation, living situation) answers NOTHING — choose clarify and stay on that question. NEVER move to a new topic after a non-answer.
- Judge from the CONVERSATION, not just the field list. If they already told you something, choose skip rather than re-asking it.
- TENSE: distinguish what is happening NOW from what happened in the PAST or only during an episode. "I felt like killing myself during a panic attack last week" and "I want to die right now" are completely different. If the timeframe is unclear, choose clarify and ask directly whether it is happening right now. If they have ALREADY told you it is not happening now, do not keep asking — that is what skip is for.
- Prefer go_deeper over moving on when the patient just disclosed something significant that is not yet understood.
- Set safety_concern=true if there is ANY hint of self-harm/hopelessness, even indirect. Be conservative: when unsure, true.
- target_field: the field name you intend to address next, copied exactly from "Still needed", or "" if none applies. For respect_refusal, put the field they just REFUSED here so it is never asked again.
- rationale: at most 12 words."""

    prompt = f"""Conversation so far:
{conversation}
Patient's latest message: "{patient_text}"

Decide the strategy. Return ONLY the JSON object."""

    try:
        raw = await call_llm(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=120,
            think=False,
            guided_json=_PLAN_SCHEMA,
        )
    except Exception as e:
        logger.warning(f"[PLANNER] call failed: {e}")
        return None

    parsed = _parse_json_response(raw)
    if not parsed or parsed.get("strategy") not in _PLAN_STRATEGIES:
        logger.warning(f"[PLANNER] unusable plan: {raw[:200]!r}")
        return None
    return parsed


async def get_next_response(
    patient_text: str,
    session,
    history: List[Dict],
    section: str,
    section_display_name: str,
    topic_hint: str = "",
    next_natural_question: str = "",
    filled_fields_summary: str = "",
    detected_language: str = "en",
    # Keep for backward compatibility during transition
    missing_fields_hint: str = "",
    valid_topics_with_fields: str = "",
    filled_field_names: Optional[List[str]] = None,
    needs_clarification: str = "",
    cross_section_summary: str = "",
    target_domain: str = "",
    target_lead: str = "",
    transition_hint: str = "",
    strategy: str = "",
    strategy_target: str = "",
) -> tuple[str, str]:
    """
    Generate next conversational response and question for the patient.
    
    This is the main conversational LLM call.
    Asks ONE question at a time, warm and non-clinical.
    Uses topic-based guidance for natural conversation flow.
    
    When target_domain is set, the model is constrained to ask about
    that specific clinical domain instead of choosing freely.
    """
    
    conversation_context = ""
    for msg in history[-10:]:
        role = "Patient" if msg["role"] == "user" else "Assistant"
        conversation_context += f"{role}: {msg['content']}\n"

    reflection_needed = _needs_reflection(patient_text)
    uncooperative_mode = _is_uncooperative_turn(patient_text)
    explicit_mode = _has_explicit_or_offtopic_sexual_content(patient_text)
    stop_mode = _has_stop_intent(patient_text)

    last_assistant_question = _extract_last_assistant_question(history)

    # Build system prompt with topic guidance
    topic_context = f"Conversational areas left to explore: {topic_hint}" if topic_hint else ""
    print(f"==========> Old topic context: {topic_context}")
    
    # topic_context = f"ALLOWED CONVERSATIONAL topics for this section ONLY — do not discuss anything else:\n{topic_hint}" if topic_hint else ""
    # print(f"==========> New topic context: {topic_context}")
    
    from data_model import SECTION_CONVERSATION_TOPICS
    valid_topic_ids = [
        t["topic_id"] 
        for t in SECTION_CONVERSATION_TOPICS.get(section, [])
    ]
    
    from data_model import SECTION_CONVERSATION_TOPICS
    topics = SECTION_CONVERSATION_TOPICS.get(section, [])
    valid_topic_lines = []
    for t in topics:
        fields_str = ", ".join(t["fields"])
        valid_topic_lines.append(f"  {t['topic_id']}: covers {fields_str}")

    # use pre-filtered unfilled topics from handlers
    if not valid_topics_with_fields:
        # fallback — compute all (shouldn't happen)
        print("# fallback — compute all (shouldn't happen)")
        valid_topics_with_fields = "\n".join(
            f"  {t['topic_id']}: covers {', '.join(t['fields'])}"
            for t in topics
        )
    
    example_topic = topics[0] if topics else None
    example_topic_id = example_topic["topic_id"] if example_topic else "background_basics"
    example_question = (
        example_topic.get("natural_questions_en", ["How are you feeling today?"])[0]
        if example_topic else "How are you feeling today?"
    )
    
    topic_id_instruction = (
        f"\n\nYour response MUST follow this exact format:\n"
        f"<your response here>\n"
        f"[TOPIC: <topic_id>]\n\n"
        f"Example:\n"
        f"{example_question}\n"
        f"[TOPIC: {example_topic_id}]\n\n"
        f"Valid topic IDs for this section:\n"
        f"{valid_topics_with_fields}\n\n"
        f"Always include the [TOPIC:] tag at the end. Pick the id matching what you just discussed."
    )

    print(f"==========> topic_id_instruction: {topic_id_instruction}")
    
    natural_question_hint = f"\n\nYou may consider asking: {next_natural_question}" if next_natural_question else ""

    # When we've just moved to a new area, fold the bridge into THIS reply as a
    # single short clause — never as a separate message and never as canned
    # "thanks, a few more questions" filler (that used to come from a second
    # message and contradicted the anti-filler rules below).
    transition_bridge = ""
    if transition_hint:
        transition_bridge = f"""

YOU HAVE JUST MOVED TO A NEW AREA: {transition_hint}.
Open THIS reply by acknowledging the shift in ONE short, warm clause that
connects to what the patient just said, then ask your question — all in a
SINGLE message (e.g. "That helps me understand things at home — can I ask about
your sleep now?"). Do NOT thank the patient, do NOT say "let's continue" or "a
few more questions", and do NOT send this as a separate message."""

    target_section = ""
    if target_domain:
        if target_domain == "demographics":
            target_section = """
TARGET FOR THIS REPLY:
Domain: demographics
Ask about the patient's background — family, living situation, occupation,
or socioeconomic status. Do NOT ask clinical or symptom questions.
"""
        else:
            target_section = f"""
TARGET FOR THIS REPLY:
Domain: {target_domain}
Goal: learn about "{target_domain.replace('_', ' ').title()}"
"""
            if target_lead:
                target_section += f"Lead to follow: {target_lead}\n"
            # The hard "exactly ONE question about this target, nothing else"
            # clamp is what made the interview feel like a form being read out.
            # Keep it only for the default flow; when the planner picked a
            # different strategy (go deeper, answer their question, skip what we
            # already know, address safety), that strategy governs instead.
            if strategy in ("", "ask", "reflect_then_ask"):
                target_section += (
                    "Generate exactly ONE natural question about this target. "
                    "Do not ask about anything else. "
                    "The question must uncover a NEW clinical fact — ask WHAT changed, "
                    "HOW OFTEN it happens, WHEN it started, or how it AFFECTED "
                    "something else. Do NOT ask how the patient feels about what "
                    "they already told you."
                )
            else:
                target_section += (
                    "Treat this as the DEFAULT topic only. The response strategy "
                    "below takes priority over it.\n"
                )

    # Deterministic anti-repetition. "NEVER start a reply with the same words as
    # your previous reply" is a plea the 14B ignores — it happily emitted
    # "You're 21. That helps me understand your background —" on three
    # consecutive turns. The validator never caught it because it only compares
    # the QUESTION, and the question changed each time. So instead of asking,
    # we show it exactly which openers are already burned.
    recent_openers = []
    for msg in reversed(history):
        if msg.get("role") != "assistant":
            continue
        opener = _opening_clause(msg.get("content", ""))
        if opener and opener not in recent_openers:
            recent_openers.append(opener)
        if len(recent_openers) >= 3:
            break
    banned_openers_block = ""
    if recent_openers:
        listed = "\n".join(f'  - "{o}"' for o in recent_openers)
        banned_openers_block = (
            "\n\nYOU HAVE ALREADY USED THESE OPENERS — do NOT begin with any of "
            f"them, or any near-paraphrase:\n{listed}\n"
            "Open differently. If the patient's last message added nothing new "
            "(e.g. just \"yes\"), do NOT re-state an old fact back at them — "
            "address what they actually said instead."
        )

    # Planner-chosen strategy for this turn (see llm.plan_turn). Empty when the
    # planner failed — then the old rule-based behaviour above applies unchanged.
    strategy_block = ""
    if strategy and strategy in _STRATEGY_GUIDANCE:
        strategy_block = (
            f"\n\nRESPONSE STRATEGY FOR THIS TURN — {strategy}:\n"
            f"{_STRATEGY_GUIDANCE[strategy]}\n"
            "This strategy overrides the default topic/question rules above."
        )
        # The planner's chosen target used to be logged and then thrown away, so
        # the question still came from the rule-based topic walk. That's how the
        # bot asked "how long have you been feeling this way?" while the planner
        # was pointing at complaint_1 — it asked the duration of a complaint that
        # had never been collected.
        if strategy_target:
            label = _target_label(strategy_target)
            strategy_block += (
                f"\nASK ABOUT THIS, NOT ANYTHING ELSE: {label}\n"
                "If any suggested question above is about a different subject, "
                "ignore it — this target wins."
            )

    # SAFETY: telling the model a strategy "overrides" the section machinery is
    # not enough — a concrete canned question beats an abstract instruction. A
    # patient disclosed self-harm thoughts and the model still asked the next
    # review-of-systems question ("any chest pain?") because
    # next_natural_question/target_section were still in the prompt. When the
    # planner raises a safety concern we REMOVE the interview scaffolding
    # entirely so there is nothing else for the model to follow.
    if strategy == "de_escalate":
        natural_question_hint = ""
        target_section = ""
        topic_context = ""
        # Also drop the [TOPIC:] checklist — it lists the very symptom topics
        # that pulled the model back to "any chest pain?". A missing tag just
        # yields topic_id="unknown" and the caller falls back safely.
        topic_id_instruction = ""
        strategy_block += (
            "\n\nThe routine interview is SUSPENDED for this reply. There is no "
            "next topic and no field to collect. Ignore any interview checklist. "
            "Respond ONLY to what the patient just disclosed about their safety."
        )

    system_prompt = f"""You are an experienced psychiatrist conducting an intake interview. Your objective is to build a complete, coherent clinical picture by gradually uncovering new relevant information while keeping the conversation natural.

You're currently discussing: {section_display_name}
{topic_context}
{transition_bridge}
{strategy_block}
{banned_openers_block}

Detected language: {_language_label(detected_language)}
Respond only in {_language_label(detected_language)}.
Reflection mode: {"ON" if reflection_needed else "OFF"}
Uncooperative mode: {"ON" if uncooperative_mode else "OFF"}
Boundary mode (sexual/off-topic content): {"ON" if explicit_mode else "OFF"}

CONVERSATION FLOW:
- Think like an interviewer, not a questionnaire. Reflect briefly, follow up on the most important thread, clarify, then ask.
- When a patient shares something emotionally significant, reflect first then probe that thread. Stay with it for 2-3 turns at most.
- Periodically summarize the timeline: "So [X], then [Y], then [Z] — is that right?" This confirms understanding and signals transitions.

THREAD SATURATION:
Do not continue exploring the same event indefinitely. After 2-3 follow-ups on the same topic (e.g., guilt about one event, a traumatic memory, a conflict), assume it is sufficiently established unless critical clinical information is still missing. Summarize briefly and transition to a consequence: how it affected daily life, functioning, relationships, or what happened next.

STAY WITH AN OPEN THREAD (do not defer-and-revisit):
- When the patient opens something clinically or emotionally significant ("I had a bad relationship with my mother", "I lost my job", "I couldn't stop crying"), FINISH that thread now with one or two CONCRETE follow-ups before moving elsewhere. Do not park it and circle back later — that makes the conversation feel disjointed and repetitive.
- Do NOT respond with a bare yes/no invitation like "Do you want to elaborate on that?". Instead ask a specific question that deepens it: "What was it like with her growing up?", "What would happen between you two?", "How did that affect you?". Then, once you understand it, acknowledge and move on — don't ask about it again.
- Never re-open a thread you already explored earlier unless the patient brings genuinely new information.

CONTINUITY:
- Every reply must connect to what the patient just said. Briefly reflect their last answer, then ask the next question so it feels like one conversation, not a list of unrelated questions.
- When you shift to a new area, name the bridge in one short clause ("That helps me understand things at home — can I ask about your sleep now?") rather than jumping cold.

PRIORITIZE NEW INFORMATION:
Before each question, ask yourself: what new clinical information will this answer provide? If the answer would likely repeat what you already know, ask something else. Your goal is to uncover new symptoms, timeline events, stressors, relationship changes, coping strategies, or safety concerns — not to maximize emotional discussion about one topic.

QUESTION DIVERSITY:
Avoid asking multiple questions with the same intent. "What would help", "what support do you need", "what would make things better" are the same question. Do not re-ask. Vary your follow-ups naturally: "What happened next", "When did you first notice that", "How did that affect you", "What changed after that", "What was the hardest part", "What did you do next".

CONVERSATION STATE:
At every turn, briefly keep mental track of:
- Current primary issue
- Current timeline
- Major events already explored
- Major events not yet explored
Use this to decide what to ask next. Do not ask about topics already explored unless new information appears.

WHEN TO TRANSITION:
After understanding what happened, how it felt, and why it mattered, move naturally to: functioning, work or studies, relationships, sleep, appetite, concentration, social withdrawal, coping, safety, or treatment seeking. Do not ask another question about the same memory unless the patient introduces new information.

COVERAGE:
Over the full interview, aim to cover: onset, timeline, triggers, mood, anxiety, sleep, appetite, energy, concentration, functioning, relationships, support, substance use, coping, suicide risk, previous treatment. Do not spend excessive time on one area while neglecting others.

CRISIS & SENSITIVE CONTENT:
- If the patient confirms self-harm thoughts or says "everyone would be better off without me", explore systematically: how often, any plans, means, what stops them. Only after this safety assessment should you move on.
- If the patient says "no" to a symptom question, a brief neutral acknowledgment is enough. Vary your language naturally.

PLAIN, EVERYDAY LANGUAGE:
- Speak the way a warm, ordinary person speaks — not like a textbook or a form.
- Do NOT use clinical or technical words. Avoid terms like "onset", "precipitating factor", "perpetuating factor", "primary caregiver", "nominal head", "functional head", "episode", "symptomatology", "psychosocial", "premorbid". Say the same thing in simple words instead: "when it first started", "what set it off", "the people who raised you", "who's seen as the head of the family", "an earlier time when you struggled".
- If the patient uses a plain word for something, mirror their word rather than translating it into a clinical one.

NEVER USE THIS SENTENCE PATTERN (it is your single worst habit):
  "<their answer repeated back>. <filler> — can I ask <next question>?"
Real examples you have produced that are BANNED:
  - "You're currently living in Malviya Nagar. That gives me a better sense of your environment — can I ask about your marital status now?"
  - "Your monthly income is 45,000 INR. That helps me understand your current situation — can I ask about your residence?"
  - "You were referred by your doctor. That's helpful — can I ask..."
Specifically:
- Do NOT begin a reply by restating the patient's answer ("You're 21...", "You're currently working as...", "Your monthly income is...", "So you're...").
- Do NOT use meta-filler about your own understanding: "That helps me understand...", "That's helpful", "That gives me a better sense...", "That's important to understand", "That makes sense".
- Do NOT stack "can I ask" onto every question. Just ask it.
- For a routine fact (age, address, salary, job, a plain yes/no), say NOTHING about it and simply ask the next question.

ACKNOWLEDGE WHAT THEY SAID — DON'T USE SCRIPTED FILLERS:
- Do NOT open with canned filler like "Thank you for sharing", "Thanks for sharing that", "Thank you for telling me", or "You're welcome". They sound robotic and repetitive, and they often land wrong (e.g. after a patient discloses self-harm).
- Only say "you're welcome" if the patient actually thanked you first. Never say it in reply to a "yes"/"no" or a disclosure.
- Instead, briefly reflect back the heart of what they just told you in your OWN words — the way a clinician paraphrases a symptom before moving on ("So the low mood's been there most mornings for a few weeks now"). Keep it to one short clause, don't parrot their exact wording, then ask your next question.
- When a patient answers "yes"/"no" to a sensitive question (like thoughts of self-harm), respond to the meaning with care and follow up — never with thanks or filler.

CONVERSATION TONE:
- Be warm but efficient. NEVER start a reply with the same words as your previous reply, and vary how you acknowledge each turn.
- Keep responses concise (under 40 words) unless reflecting on something emotionally serious.
- You are gathering information, not diagnosing or giving advice. Do NOT mention sections, progress, fields, or internal structure.
- If the patient is resistant, terse, or dismissive, do not argue. Acknowledge and ask one easy, concrete question.
- If the patient wants to stop, respect it and offer to pause.
- ONLY ask about the topics listed for this section. If the patient brings up something else, acknowledge briefly and redirect.
{target_section}
{natural_question_hint}
{topic_id_instruction}
"""
# 11. If you've gathered enough about a topic, smoothly transition to the next one.{natural_question_hint}

    # Explicit negative list — for a structured intake, a "do NOT ask" list of
    # already-answered topics steers the model more reliably than the positive
    # "already collected" summary alone.
    from data_model import FIELD_HUMANIZE
    do_not_ask_labels = sorted({
        FIELD_HUMANIZE.get(f, f.replace("_", " ").title())
        for f in (filled_field_names or [])
    })
    do_not_ask_line = (
        f"\n\nALREADY ANSWERED — do NOT ask about these again: {', '.join(do_not_ask_labels)}."
        if do_not_ask_labels else ""
    )

    prompt = f"""Already collected this session:
{filled_fields_summary}{do_not_ask_line}
{needs_clarification}{cross_section_summary}

Recent conversation:
{conversation_context}

Patient's latest message: "{patient_text}"

Write your reply now."""

    # ── Reply-prompt audit trail (logs/test_logs/reply_prompts.log) ──
    # Capture the exact prompt and raw model output for this conversational
    # turn. Wrapped so a logging failure can never break the turn.
    sid = getattr(session, "session_id", "?")
    try:
        try:
            tokenizer, _ = await _load_model()
            exact_prompt = _build_chat_prompt(tokenizer, system_prompt, prompt)
        except Exception:
            # Model not ready — fall back to the readable system/user components.
            exact_prompt = f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{prompt}"
        reply_log.info("=" * 100)
        reply_log.info(f"[TURN] session={sid} section={section} lang={detected_language}")
        reply_log.info("----- 1. EXACT PROMPT SENT TO LLM -----\n%s", exact_prompt)
    except Exception as log_err:
        logger.warning(f"[REPLY_IO] prompt logging failed: {log_err}")

    try:
        response = await call_llm(prompt=prompt, system_prompt=system_prompt, max_tokens=150)
        try:
            reply_log.info("----- 2. LLM RAW RESPONSE -----\n%r", response)
        except Exception as log_err:
            logger.warning(f"[REPLY_IO] response logging failed: {log_err}")
        logger.info(f"[LLM RAW FULL] {response!r}")
        
        cleaned_response = response.strip()
        
        logger.info(f"[{_ts()}] LLM RESPONSE: {cleaned_response!r}")
        
        clean_response, topic_id = _extract_topic_id(cleaned_response)
        cleaned_response = clean_response
        logger.info(f"[LLM RAW TOPIC] topic_id: {topic_id!r}")
        logger.info(f"[LLM CLEAN] {cleaned_response!r}")

        # Drop an unprompted "You're welcome" — the model emits it even when the patient never thanked it, and the prompt rule alone doesn't stop it.
        before_strip = cleaned_response
        cleaned_response = _strip_unprompted_welcome(cleaned_response, patient_text)
        if cleaned_response != before_strip:
            logger.info(f"[LLM WELCOME-STRIP] {before_strip!r} -> {cleaned_response!r}")

        # Drop a habitual "Thank you for sharing"-style filler opener when real
        # content follows, so the reflection/question leads instead. Overused,
        # these openers make every turn feel scripted.
        before_grat = cleaned_response
        cleaned_response = _strip_scripted_gratitude_opener(cleaned_response)
        if cleaned_response != before_grat:
            logger.info(f"[LLM GRATITUDE-STRIP] {before_grat!r} -> {cleaned_response!r}")

        # NEVER TWO ACKNOWLEDGEMENTS IN A ROW.
        #
        # Banning openers by string failed: the model just rotates synonyms
        # ("I see" -> "That's helpful" -> "I understand" -> "That's good to
        # know" -> "Thank you for sharing"), so every single turn still opened
        # with throat-clearing. But stripping them ALL made the interview read
        # like a form. A person does neither — they acknowledge sometimes.
        # So we enforce the rhythm in code: if the previous reply already opened
        # with an acknowledgement, this one loses its own. The model keeps its
        # freedom to be warm; it just can't do it twice running.
        # de_escalate is exempt — opening with warmth there is correct.
        if strategy != "de_escalate":
            prev_assistant = next(
                (m.get("content", "") for m in reversed(history)
                 if m.get("role") == "assistant"),
                "",
            )
            if _has_filler_opener(prev_assistant):
                before_filler = cleaned_response
                cleaned_response = _strip_leading_filler(cleaned_response)
                if cleaned_response != before_filler:
                    logger.info(
                        f"[LLM FILLER-STRIP] previous reply already acknowledged; "
                        f"{before_filler!r} -> {cleaned_response!r}"
                    )

            # Unwrap "Can I ask about X?" -> a direct question, so a "no" answers
            # the CLINICAL question rather than the permission request.
            before_perm = cleaned_response
            cleaned_response = _unwrap_permission_question(cleaned_response)
            if cleaned_response != before_perm:
                logger.info(f"[LLM PERMISSION-UNWRAP] {before_perm!r} -> {cleaned_response!r}")

        if stop_mode:
            return _deterministic_resistance_response(
                detected_language=detected_language,
                next_natural_question=next_natural_question,
                patient_text=patient_text,
                last_assistant_question=last_assistant_question,
            ), "unknown"

        if _is_bad_or_repetitive_response(
            response=cleaned_response,
            patient_text=patient_text,
            history=history,
            section_display_name=section_display_name,
            missing_fields_hint=missing_fields_hint or topic_hint,
            filled_field_names=filled_field_names,
        ):
            if uncooperative_mode or explicit_mode:
                return _deterministic_resistance_response(
                    detected_language=detected_language,
                    next_natural_question=next_natural_question,
                    patient_text=patient_text,
                    last_assistant_question=last_assistant_question,
                ), "unknown"

            # TODO: Here get the topic id from next_natural qn
            if next_natural_question:
                return next_natural_question, "unknown"

            return _deterministic_next_question(
                missing_fields_hint=next_natural_question or topic_hint or missing_fields_hint,
                detected_language=detected_language,
                last_assistant_question=last_assistant_question,
                reflection_needed=reflection_needed,
                patient_text=patient_text,
            ), "unknown"

        return cleaned_response, topic_id
    except Exception as e:
        print(f"Failed to get next response: {e}")
        # Fallback response - resistance-aware fallback before generic prompting
        if stop_mode or uncooperative_mode or explicit_mode:
            return _deterministic_resistance_response(
                detected_language=detected_language,
                next_natural_question=next_natural_question,
                patient_text=patient_text,
                last_assistant_question=last_assistant_question,
            ), "unknown"

        if next_natural_question:
            return next_natural_question, "unknown"
        return _deterministic_next_question(
            missing_fields_hint=topic_hint or missing_fields_hint,
            detected_language=detected_language,
            last_assistant_question=last_assistant_question,
            reflection_needed=reflection_needed,
            patient_text=patient_text,
        ), "unknown"


# Per-field interpretation notes for the extractor. Only the lines for fields
# actually available in the current section (+ unfilled previous-section fields)
# are injected into the prompt — dumping every section's fields tempts the model
# to emit field names that don't belong here, which then get dropped on save.
_FIELD_GUIDELINES = {
    "name": '- "name": Patient\'s name',
    "age": '- "age": Patient\'s age in years (or birth date)',
    "sex": '- "sex": Patient\'s sex/gender ("male", "female", "other") — any answer to a question about sex or gender maps here',
    "education": '- "education": Highest education level or current enrolment',
    "occupation": '- "occupation": CURRENT job or work status (full-time, part-time, studying, unemployed)',
    "occupational_history": '- "occupational_history": PAST jobs and work trajectory (previous roles, how long, why they left) — NOT the current job',
    "ses": '- "ses": Monthly income or total family income — an amount (e.g. in INR) or any financial indication is a valid value',
    "residence": '- "residence": Where the patient lives (city or rural area)',
    "address": '- "address": current address or general location — do NOT put family structure here',
    "family_type": '- "family_type": Nuclear or joint family setup, number of family members. A household member count alone is a valid value; a large household can reasonably be inferred as joint',
    "marital_status": '- "marital_status": CURRENT status (single, married, divorced, separated, widowed)',
    "marital_history": '- "marital_history": relationship/marriage history and quality over time (past partners, how long, how it went)',
    "religion": '- "religion": Religion or spiritual practice',
    "mother_tongue": '- "mother_tongue": first language',
    "duration_each": '- "duration_each": How long the symptoms have been present',
    "impact_on_functioning": '- "impact_on_functioning": The EFFECT of the problem on daily life ("can\'t sleep, missing work, can\'t concentrate"). NOT the events that caused the distress',
    "mode_of_onset": '- "mode_of_onset": sudden vs gradual start',
    "course_of_illness": '- "course_of_illness": how symptoms changed since onset',
    "progress_of_illness": '- "progress_of_illness": whether they became more intense/frequent over time',
    "time_course": '- "time_course": remissions and relapses (periods of improvement, return after improvement)',
    "hospitalizations": '- "hospitalizations": admissions for the CURRENT illness',
    "hospitalizations_past": '- "hospitalizations_past": previous admissions (reason, duration, outcome)',
    "functional_head": '- "functional_head": who actually runs the family',
    "nominal_head": '- "nominal_head": who is formally considered the head',
    # Personal-history fields (past-oriented life history)
    "birth_and_early_development": '- "birth_and_early_development": birth and infancy — full-term vs complications, early development',
    "developmental_milestones": '- "developmental_milestones": when they started walking, talking, going to school',
    "childhood_environment": '- "childhood_environment": what home/childhood was like — atmosphere, upbringing, early relationships with caregivers',
    "school_history": '- "school_history": schooling experience and academic performance',
    "peer_relationships_childhood": '- "peer_relationships_childhood": friendships and peer relationships growing up',
    "sexual_development": '- "sexual_development": puberty and sexual development history',
    "reproductive_history": '- "reproductive_history": pregnancies, children, menstrual/reproductive history',
    "substance_use_history": '- "substance_use_history": alcohol, tobacco, or drug use (past and present); write "denied" if none',
    "forensic_history": '- "forensic_history": legal/criminal history (arrests, charges); write "denied" if none',
    "premorbid_personality": '- "premorbid_personality": personality/temperament before the illness began',
    # Family-history fields — capture WHO and WHAT, never a bare yes/no
    "family_members_list": '- "family_members_list": who is in the family — members and their relation (e.g. "father, mother, one younger sister")',
    "education_occupation_members": '- "education_occupation_members": education and occupation of family members, tied to who (e.g. "father: graduate, businessman; mother: homemaker")',
    "living_arrangement": '- "living_arrangement": who the patient lives with — e.g. "lives alone", "with parents", "with flatmates". Never store a bare "yes"/"no"',
    "relationship_quality": '- "relationship_quality": quality of relationships with parents/siblings (close, strained, distant)',
    "family_dynamics": '- "family_dynamics": how family members interact — conflict, avoidance, support patterns',
    "support_system": '- "support_system": whom the patient turns to for emotional support, and whether they feel supported',
    "psychiatric_illness_in_family": '- "psychiatric_illness_in_family": any mental-health condition in the family (who, what); "denied" if none',
    "genetic_conditions": '- "genetic_conditions": hereditary/genetic conditions in the family; "denied" if none',
}


def _build_field_guidelines(field_names: List[str]) -> str:
    """Build a FIELD INTERPRETATION GUIDELINES block scoped to the given fields."""
    lines = []
    seen = set()
    for f in field_names:
        line = _FIELD_GUIDELINES.get(f)
        if line and line not in seen:
            lines.append(line)
            seen.add(line)
    # Group/pattern-based fields that don't have one entry per name.
    if any(f.startswith("complaint_") for f in field_names):
        lines.append('- "complaint_1/2/3": Patient\'s main concerns or symptoms mentioned naturally')
    if any(f.endswith("_negative") for f in field_names):
        lines.append('- "*_negative" fields: store a short clinical phrase, never bare Yes/No — if symptoms are reported, list them ("POSITIVE: <symptoms>"); if denied, write "denied"')
    if any(f.endswith(("_biological", "_psychological", "_social")) for f in field_names):
        lines.append(
            '- Factors of illness — classify by TYPE of factor:\n'
            '  - "*_biological": physical health, genetics, family history of mental illness, medical illness, head injury, substance use\n'
            '  - "*_psychological": personality, coping style, thoughts, self-esteem, stress tolerance\n'
            '  - "*_social": relationships, work, finances, living environment, social support\n'
            '  (predisposing = long-standing vulnerability; precipitating = recent trigger; perpetuating = what keeps it going)'
        )
    if not lines:
        return ""
    return "FIELD INTERPRETATION GUIDELINES:\n" + "\n".join(lines)


def _build_extraction_schema(field_names: List[str]) -> Dict:
    """
    JSON Schema for the cognition pass.

    On the vllm backend this is enforced by guided decoding (XGrammar), so the
    model CANNOT emit malformed JSON or invent a field name outside this
    section. That removes two whole failure classes at once: the JSON-parse
    retry pass, and the "dropped unknown field" extraction loss. On the hf
    backend it's ignored and _parse_json_response still parses best-effort.
    """
    lead_schema = {
        "type": "object",
        "properties": {
            "present": {"type": "boolean"},
            "phrase": {"type": "string"},
        },
        "required": ["present"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "updates": {
                "type": "object",
                # Constraining to exactly this section's field names is what
                # makes invented/out-of-section names impossible.
                "properties": {name: {"type": "string"} for name in field_names},
                "additionalProperties": False,
            },
            "free_notes": {"type": "string"},
            "section_complete_signal": {"type": "boolean"},
            "leads": {
                "type": "object",
                "properties": {
                    "safety": lead_schema,
                    "guilt": lead_schema,
                    "contradiction": lead_schema,
                },
                "additionalProperties": False,
            },
        },
        "required": ["updates", "free_notes", "section_complete_signal", "leads"],
        "additionalProperties": False,
    }


def _leads_from_extraction(parsed: Dict) -> List[Dict]:
    """
    Pull clinical leads out of the merged cognition result.

    Emits the same {"type", "priority"} shape the keyword pass and the old
    async_extract_leads() produced, so ClinicalState.store_llm_leads is
    unchanged. Safety outranks the rest.
    """
    leads: List[Dict] = []
    raw_leads = parsed.get("leads") or {}
    if not isinstance(raw_leads, dict):
        return leads
    for lead_type in ("safety", "guilt", "contradiction"):
        entry = raw_leads.get(lead_type) or {}
        if isinstance(entry, dict) and entry.get("present"):
            leads.append({
                "type": lead_type,
                "priority": 1 if lead_type == "safety" else 2,
            })
    return leads


async def extract_info(
    patient_text: str,
    preceding_question: str,  # the assistant question the patient was answering
    # bot_response: str,      # OLD: received the bot's NEXT reply (generated after the answer), so the model never saw the actual question context
    section: str,
    field_list: List[str],
    already_filled: Dict[str, str],
    prev_field_list: Optional[List[str]] = None
) -> Dict:
    """
    Silent structured extraction from latest exchange.
    Returns JSON with extracted fields and completion signal.

    Called non-blocking after each patient message.
    """

    already_filled_str = ", ".join([f"{k}: {v}" for k, v in already_filled.items() if v]) or "Nothing filled yet"

    prev_fields_line = ""
    if prev_field_list:
        prev_fields_line = (
            f"\nPrevious-section fields (fill ONLY if the patient clearly provides that specific information — never repurpose the current answer to fill these): "
            f"{', '.join(prev_field_list)}"
        )

    # Scope the interpretation guidelines to only the fields valid in THIS
    # exchange (current section + unfilled previous-section fields). Showing
    # every section's fields tempts the model to emit out-of-section names.
    field_guidelines = _build_field_guidelines(list(field_list) + list(prev_field_list or []))

    system_prompt = f"""You are a clinical data extraction engine for a psychiatric intake interview.
CRITICAL: Extract information contextually, not literally by field name.

Current section: {section}
Fields available for extraction: {', '.join(field_list)}{prev_fields_line}

{field_guidelines}

EXTRACTION RULES:
1. Extract based on MEANING, not literal field names
2. For short replies (a number, a single word), use the assistant's preceding question to identify the field — bot asked about age → "21" → age="21"; bot asked a 1-10 scale → "7" → severity="7/10"
3. Infer contextually when reasonable: "PhD student" → occupation: PhD student, education: Doctoral degree
4. If patient mentions a concern naturally, put it in the first available complaint_X field. Duration/impact may come up in the same turn as complaints
5. Omit fields the patient didn't address. Never write "not specified", "N/A", or similar placeholders — leave the field out entirely
6. A clear negative answer IS information: if the assistant asked about a listed field and the patient denies it, record the denial — e.g. substance_use_history: "denies any substance use"
7. If patient explicitly refused ("I'd rather not say", "that's private"): return the value as exactly "[declined]"
8. Extract values from the patient's words. The assistant's message provides question context only — do not extract values from it
9. Use ONLY field names from the lists above, copied exactly. If information does not fit any listed field, put it in free_notes instead of inventing a field name
10. Always use free_notes for clinically relevant context that doesn't fit a field — especially life transitions (new job, recent move, relationship change, bereavement), emotional tone, or anything the patient mentions in passing. Do this even in routine sections like sociodemographic
11. Never store a bare "yes" or "no" as the value of a DESCRIPTIVE field (names, lists, who-lives-where, relationships, environment). Capture the actual content (who/what). If a yes/no reply carries no content, either resolve it from the preceding question (bot asked "do you live alone?" + "yes" → living_arrangement: "lives alone") or omit the field.
12. BUT a "no" to a SYMPTOM-SCREENING question is a real clinical finding and MUST be recorded — write it as a short phrase, never as a bare "no". Asked "are there thoughts that keep coming back to you?" + "no" → thought_content: "denies recurring or intrusive thoughts". Asked "have you heard voices or seen things others don't?" + "no" → perception: "denies hallucinations". Asked "how is your mood?" + "okay" → mood_subjective: "reports mood as okay".
   This matters: when you drop these, the field stays empty and the interview ASKS THE SAME QUESTION AGAIN. A real patient was asked about recurring thoughts, hallucinations and mood twice each in one session because their clear answers were discarded here.

CLINICAL LEADS — separately from the fields, classify the patient's answer for these three signals:
- "safety": ANY hint of self-harm, suicide, hopelessness, or not wanting to be alive — INCLUDING paraphrases that avoid explicit words ("everyone would be better off without me", "I don't see the point anymore", "I just want it to stop", "what's the use").
- "guilt": self-blame, worthlessness, or shame ("it was my fault", "I ruined everything").
- "contradiction": this answer conflicts with something the patient stated earlier.
Set present=true and copy the exact triggering phrase into "phrase" ONLY if the message clearly contains one. Be conservative — if unsure, present=false.

Return ONLY valid JSON:
{{
  "updates": {{"<field_name>": "<extracted_value>"}},
  "free_notes": "<context or clinical notes that don't fit standard fields>",
  "section_complete_signal": false,
  "leads": {{
    "safety": {{"present": false, "phrase": ""}},
    "guilt": {{"present": false, "phrase": ""}},
    "contradiction": {{"present": false, "phrase": ""}}
  }}
}}

Return empty updates if nothing fits the fields: {{"updates": {{}}, "free_notes": "", "section_complete_signal": false, "leads": {{"safety": {{"present": false}}, "guilt": {{"present": false}}, "contradiction": {{"present": false}}}}}}

Set "section_complete_signal" to true only if patient has provided sufficient information for this section.
"""

    # OLD prompt — paired the patient's answer with the bot's NEXT reply, so the model had to guess which question the answer belonged to:
    # prompt = f"""Already filled fields (do not re-extract): {already_filled_str}
    #
    # Exchange:
    # Patient said: "{patient_text}"
    # Assistant said: "{bot_response}"
    #
    # Extract information from the patient's natural language response. Return ONLY the JSON object."""
    prompt = f"""Already filled fields (do not re-extract): {already_filled_str}

Exchange:
Assistant asked: "{preceding_question}"
Patient answered: "{patient_text}"

Extract information from the patient's answer. Return ONLY the JSON object."""

    # ── Extraction-prompt audit trail (logs/test_logs/extraction_prompts.log) ──
    # Wrapped so a logging failure can never break extraction.
    try:
        try:
            tokenizer, _ = await _load_model()
            exact_prompt = _build_chat_prompt(tokenizer, system_prompt, prompt, think=False)
        except Exception:
            exact_prompt = f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{prompt}"
        extract_log.info("=" * 100)
        extract_log.info(f"[EXTRACT] section={section}")
        extract_log.info("----- 1. EXACT EXTRACTION PROMPT SENT TO LLM -----\n%s", exact_prompt)
    except Exception as log_err:
        logger.warning(f"[EXTRACT_IO] prompt logging failed: {log_err}")

    try:
        # Extraction never thinks — mechanical JSON task, keep it fast.
        # guided_json is enforced on the vllm backend (schema-valid output, so
        # the retry below never fires) and ignored on hf.
        extraction_schema = _build_extraction_schema(
            list(field_list) + list(prev_field_list or [])
        )
        response = await call_llm(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=300,
            think=False,
            guided_json=extraction_schema,
        )
        try:
            extract_log.info("----- 2. LLM RAW RESPONSE -----\n%r", response)
        except Exception as log_err:
            logger.warning(f"[EXTRACT_IO] response logging failed: {log_err}")

        parsed_response = _parse_json_response(response)
        if parsed_response is not None:
            result = _normalize_extraction_result(parsed_response)
            # Leads ride along on the same pass now — this is what replaced the
            # separate async_extract_leads() forward pass.
            result["leads"] = _leads_from_extraction(parsed_response)
            try:
                extract_log.info("----- 3. EXTRACTED RESULT -----\n%r", result)
            except Exception:
                pass
            return result

        second_response = await call_llm(
            prompt=f"Return only the JSON object.\n\n{prompt}",
            system_prompt="You are a JSON extraction tool for psychiatric interviews. Extract information from natural language into the provided field structure.",
            max_tokens=200,
            think=False,
        )
        parsed_second_response = _parse_json_response(second_response)
        if parsed_second_response is not None:
            result = _normalize_extraction_result(parsed_second_response)
            result["leads"] = _leads_from_extraction(parsed_second_response)
            try:
                extract_log.info("----- 2b. RETRY LLM RAW RESPONSE -----\n%r", second_response)
                extract_log.info("----- 3. EXTRACTED RESULT (retry) -----\n%r", result)
            except Exception:
                pass
            return result

        logger.info(f"JSON parse failed twice for section {section}")
        logger.info(f"First response excerpt: {response[:300]!r}")
        try:
            extract_log.info("----- 3. EXTRACTED RESULT -----\n(JSON parse failed twice — nothing extracted)")
        except Exception:
            pass
        return {"updates": {}, "free_notes": "", "section_complete_signal": False}
    
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return {"updates": {}, "free_notes": "", "section_complete_signal": False}


async def suggest_followup(
    preceding_question: str,
    patient_text: str,
    section: str = "",
) -> Optional[str]:
    """
    EXPERIMENTAL, log-only follow-up advisor.

    Given the question the bot asked and what the patient replied, ask the LLM
    whether the answer is vague/incomplete and what ONE clarifying or follow-up
    question it would ask next. The result is written to
    logs/test_logs/followup_advisor.log (with the full prompt) and is NOT shown
    to the patient — it's data for deciding what to implement later.

    Runs as a background task; never raises into the caller.
    """
    system_prompt = (
        "You are a clinical interview supervisor reviewing a mental-health intake chat. "
        "You are given the question the assistant asked and the patient's reply. "
        "Judge ONLY whether the reply is vague, ambiguous, or incomplete, and if so, "
        "what single clarifying or follow-up question would best elaborate it.\n\n"
        "Examples of vague replies worth clarifying:\n"
        '- "a few" / "sometimes" / "a lot" → ask for a concrete number or frequency.\n'
        '- "45k" to a question about income → ask whether that is personal or total family income.\n'
        '- "feeling low" → ask about sleep, appetite, energy, and how long.\n'
        "If the reply is already specific and complete, say so.\n\n"
        "Respond in strict JSON:\n"
        '{"is_vague": true/false, "reason": "<short>", "followup_question": "<one question, or empty if not needed>"}'
    )
    prompt = (
        f"Section: {section or 'unknown'}\n"
        f"Assistant asked: \"{preceding_question}\"\n"
        f"Patient replied: \"{patient_text}\"\n\n"
        "Return ONLY the JSON object."
    )

    try:
        try:
            tokenizer, _ = await _load_model()
            exact_prompt = _build_chat_prompt(tokenizer, system_prompt, prompt, think=False)
        except Exception:
            exact_prompt = f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{prompt}"
        advisor_log.info("=" * 100)
        advisor_log.info(f"[ADVISOR] section={section}")
        advisor_log.info("----- 1. EXACT ADVISOR PROMPT SENT TO LLM -----\n%s", exact_prompt)
    except Exception as log_err:
        logger.warning(f"[ADVISOR] prompt logging failed: {log_err}")

    try:
        response = await call_llm(prompt=prompt, system_prompt=system_prompt, max_tokens=200, think=False)
        try:
            advisor_log.info("----- 2. LLM RAW RESPONSE -----\n%r", response)
            parsed = _parse_json_response(response)
            if parsed is not None:
                advisor_log.info("----- 3. PARSED SUGGESTION -----\n%r", parsed)
        except Exception:
            pass
        return response
    except Exception as e:
        logger.warning(f"[ADVISOR] suggest_followup failed: {e}")
        return None


def _build_harvest_schema(field_names: List[str]) -> Dict:
    """Schema forcing every harvested field to carry a verbatim evidence quote."""
    return {
        "type": "object",
        "properties": {
            "updates": {
                "type": "object",
                "properties": {
                    name: {
                        "type": "object",
                        "properties": {
                            "value": {"type": "string"},
                            "evidence": {"type": "string"},
                        },
                        "required": ["value", "evidence"],
                        "additionalProperties": False,
                    }
                    for name in field_names
                },
                "additionalProperties": False,
            }
        },
        "required": ["updates"],
        "additionalProperties": False,
    }


def _verify_harvest_evidence(parsed: Dict, conversation_text: str, section: str) -> Dict:
    """
    Keep only harvested fields whose evidence quote REALLY appears in the transcript.

    Harvest's job is to avoid re-asking things the patient already volunteered.
    But it was inventing them: entering factors_of_illness it filled
    course_of_illness / progress_of_illness from a conversation that only
    mentioned panic attacks in crowds, and filled onset_timing for a patient who
    had just said "first i dont remember". Two harms:
      1. Fabricated findings land in the DB and then in the clinician's report.
      2. Those phantom fields drive completion to 100%, which fires the next
         transition, which harvests again — the cascade that blasted through six
         sections in ten turns.
    Rule 4 already told the model not to guess and it guessed anyway, so we no
    longer trust it: the model must quote the patient verbatim, and we check the
    quote against the transcript here. No quote, no field.
    """
    haystack = _normalize_text(conversation_text or "")
    kept: Dict[str, str] = {}
    for field_name, entry in (parsed.get("updates") or {}).items():
        if not isinstance(entry, dict):
            continue
        value = (entry.get("value") or "").strip()
        evidence = (entry.get("evidence") or "").strip()
        if not value or not evidence:
            continue
        needle = _normalize_text(evidence)
        # Very short quotes ("no", "yes") prove nothing — they'd match anywhere.
        if len(needle) < 8 or needle not in haystack:
            logger.warning(
                f"[HARVEST] section={section} DROPPED {field_name}={value!r} — "
                f"evidence not found verbatim in transcript: {evidence!r}"
            )
            continue
        kept[field_name] = value
    if kept:
        logger.info(f"[HARVEST] section={section} verified fields: {sorted(kept)}")
    return kept


async def harvest_info(
    conversation_text: str,
    section: str,
    field_list: List[str],
    already_filled: Dict[str, str]
) -> Dict:
    """
    One-time harvest pass when a section transition fires: re-read the recent conversation and pre-fill the NEW section's fields from information the patient already volunteered in earlier answers. Prevents re-asking things a rich narrative answer already covered.
    """

    already_filled_str = ", ".join([f"{k}: {v}" for k, v in already_filled.items() if v]) or "Nothing filled yet"

    system_prompt = f"""You are a clinical data extraction engine reviewing a psychiatric intake conversation.
The interview has just moved to a new section. Review the conversation so far and extract any information the patient has ALREADY provided that fits the new section's fields.

New section: {section}
Fields to fill: {', '.join(field_list)}

RULES:
1. Extract based on MEANING, not literal field names
2. Use ONLY field names from the list above, copied exactly — if information fits no listed field, leave it out
3. Extract values only from the patient's words, not from the assistant's questions
4. Fill a field ONLY if the patient clearly provided that specific information — do not force-fit or guess
5. Omit fields not addressed. Never write "not specified", "N/A", or similar placeholders
6. For EVERY field you fill you must supply "evidence": the patient's own words, copied EXACTLY and word-for-word from the conversation above. Do not paraphrase, do not summarise, do not reconstruct. If you cannot copy an exact sentence the patient actually typed, then you do not have the information — omit the field entirely.
   This goes into a medical record a doctor will read. A field you inferred but the patient never said is a fabricated clinical finding.

Return ONLY valid JSON:
{{
  "updates": {{"<field_name>": {{"value": "<extracted_value>", "evidence": "<exact words the patient typed>"}}}}
}}

Return empty updates if nothing fits: {{"updates": {{}}}}
"""

    prompt = f"""Already filled fields (do not re-extract): {already_filled_str}

Conversation so far:
{conversation_text}

Extract information for the new section's fields. Return ONLY the JSON object."""

    try:
        response = await call_llm(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=400,
            guided_json=_build_harvest_schema(field_list),
        )
        parsed = _parse_json_response(response)
        if parsed is None:
            logger.info(f"Harvest JSON parse failed for section {section}: {response[:200]!r}")
            return {"updates": {}, "free_notes": "", "section_complete_signal": False}
        verified = _verify_harvest_evidence(parsed, conversation_text, section)
        return _normalize_extraction_result({"updates": verified})
    except Exception as e:
        logger.error(f"Harvest extraction failed: {e}")
        return {"updates": {}, "free_notes": "", "section_complete_signal": False}


# Domains where an unsupported claim in the clinician's report is dangerous.
#
# If NONE of a domain's fields hold a value, the summary must make NO claim
# about that domain — not an absence ("no indication of suicidal ideation"), and
# not a normal finding ("affect and cognition appear intact"). Both were found in
# real IIT Kalyani reports: one for a patient who answered "Yess" to having
# thoughts of harming herself, one for a patient who answered "Not yet", and
# three for patients who were never asked at all. An MSE was asserted over a text
# chat where no affect can be observed. The report generator cannot distinguish
# "asked and denied" from "never asked", so we enforce that distinction here.
_GUARDED_CLAIMS = {
    "Suicide/self-harm risk": {
        "fields": ("suicidality_assessed", "safety_risk", "suicidality", "self_harm"),
        "pattern": re.compile(
            r"suicid|self[-\s]?harm|ideation|harm(?:ing)?\s+(?:them|him|her)self",
            re.IGNORECASE,
        ),
    },
    "Psychotic symptoms": {
        "fields": ("hallucinations", "delusions", "perception", "thought_content",
                   "psychosis", "thought_form"),
        "pattern": re.compile(r"psychotic|psychosis|hallucinat|delusion", re.IGNORECASE),
    },
    "Mental status examination": {
        "fields": ("affect", "mood_objective", "cognition", "attention_concentration",
                   "insight", "judgement", "judgment", "psychomotor",
                   "appearance_behaviour", "orientation", "speech"),
        "pattern": re.compile(
            r"\baffect\b|\bcognition\b|cognitive functioning|psychomotor|\binsight\b|"
            r"\bjudge?ment\b|\borientation\b|mental status",
            re.IGNORECASE,
        ),
    },
}


def _enforce_evidence_in_summary(summary: str, fields: Dict, section: str) -> str:
    """
    Delete any sentence that makes a claim about a domain we never assessed, and
    say so explicitly instead.

    "Not assessed" and "assessed and negative" are completely different facts to
    a clinician triaging risk, and the generator conflates them every time. This
    is the last line of defence before the report reaches a doctor.
    """
    if not summary:
        return summary

    unassessed = {
        domain: spec for domain, spec in _GUARDED_CLAIMS.items()
        if not any((fields or {}).get(f) for f in spec["fields"])
    }
    if not unassessed:
        return summary

    kept, dropped = [], []
    for sentence in re.split(r"(?<=[.!?])\s+", summary):
        hit = next(
            (d for d, spec in unassessed.items() if spec["pattern"].search(sentence)),
            None,
        )
        if hit:
            dropped.append((hit, sentence))
            logger.warning(
                f"[SUMMARY-GUARD] section={section} REMOVED unsupported {hit!r} claim "
                f"(no field assessed): {sentence!r}"
            )
        else:
            kept.append(sentence)

    out = " ".join(s for s in kept if s.strip()).strip()
    # Replace the deleted reassurance with the truth: we did not ask.
    for domain in dict.fromkeys(d for d, _ in dropped):
        out = (out + f"\n\n{domain}: NOT ASSESSED during this interview.").strip()
    return out


async def generate_section_summary(section: str, section_data: Dict) -> str:
    """
    Generate clinical summary for completed section.
    Called when section transitions.
    """
    
    fields_text = ""
    for field_name, value in section_data.get("fields", {}).items():
        if value:
            fields_text += f"- {field_name}: {value}\n"
    
    notes_text = ""
    for note in section_data.get("notes", []):
        notes_text += f"- {note}\n"
    
    system_prompt = f"""You are a clinical documentation assistant. Write a concise, formal clinical summary paragraph (3-6 sentences) for the "{section}" section based on the collected data below.

Use formal psychiatric clinical language. Include all relevant clinical information. Do not speculate. Write flowing prose without listing field names.
"""

    prompt = f"""Data:
{fields_text}

Additional notes:
{notes_text}

Write the summary in 3-6 sentences."""
    
    try:
        summary = await call_llm(prompt=prompt, system_prompt=system_prompt, max_tokens=250)
        return _enforce_evidence_in_summary(
            summary.strip(), section_data.get("fields", {}), section
        )
    except Exception as e:
        print(f"Failed to generate section summary: {e}")
        return f"Clinical summary for {section} could not be generated."


async def generate_transition_message(
    completed_section: str,
    next_section: str,
    next_section_display_name: str,
    language: str
) -> str:
    """
    Generate warm transitional message between sections.
    Brief acknowledgement + introduction to next section.
    """
    
    system_prompt = f"""You are a compassionate mental health intake assistant.

Detected language code: {language}
Respond only in {_language_label(language)}.

Write a 2-sentence transitional message that:
1. Briefly acknowledges the patient's effort warmly
2. Gently asks to continue with a few more important questions

Keep it under 30 words. No clinical jargon. Do not mention section names, progress, or internal workflow."""
    
    try:
        message = await call_llm(prompt="", system_prompt=system_prompt, max_tokens=80)
        return message.strip()
    except Exception as e:
        print(f"Failed to generate transition message: {e}")
        if language == "hi":
            return "Shukriya. Aaiye kuch aur zaroori sawaalon par aage badhte hain."
        if language == "hinglish":
            return "Shukriya. Chaliye kuch aur zaroori questions continue karte hain."
        return "Thanks for sharing. Let's continue with a few more important questions."


async def generate_closing_message(language: str) -> str:
    """
    Generate warm closing message when session completes.
    """
    
    closing_prompts = {
        "en": (
            "Thank you so much for taking the time to share all of this. Your doctor "
            "will now have a much better understanding of your situation before your appointment. "
            "Take care of yourself. 💙"
        ),
        "hi": (
            "इतना कुछ साझा करने के लिए धन्यवाद। आपके डॉक्टर को आपकी स्थिति की बहुत बेहतर समझ होगी। "
            "अपना ख्याल रखें। 💙"
        )
    }
    
    return closing_prompts.get(language, closing_prompts["en"])