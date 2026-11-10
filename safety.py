"""
Safety detection for psychiatric history bot.
Pure detection functions — no Telegram, no REST, no side effects.
Imported by api.py (and any future interface) for safety checks.
"""

import os
import re

import db  # used by the legacy handle_crisis* helpers below

CRISIS_HELPLINE = os.getenv("CRISIS_HELPLINE", "iCall: 9152987821")

# ── Keyword lists ─────────────────────────────────────────────

CRISIS_KEYWORDS = [
    # English
    "suicide", "suicidal", "kill myself", "end my life", "ending my life",
    "want to die", "can't go on", "harm myself", "self-harm", "self harm",
    "hurt myself", "no reason to live", "better off dead",
    "killing myself", "going to kill", "will kill myself",
    "should die", "deserve to die", "want to disappear", "end it all", "ending it all",
    # Colloquial / contracted forms. "I wanna die" did NOT match "want to die"
    # and so never entered crisis mode: the patient got a safety-probing reply
    # from the planner, said "right now", and was asked where she lived. No
    # helpline, no clinician alert, no triage. False positives here are cheap
    # (an unnecessary safety check); false negatives are not.
    "wanna die", "wanna kill myself", "wanna end it", "wanna end my life",
    "wanna be dead", "wanna disappear", "wanna sleep forever",
    "gonna kill myself", "gonna end it", "gonna end my life",
    "dont wanna live", "don't wanna live", "dont want to live",
    "do not want to live", "not want to live", "no longer want to live",
    "dont wanna be here", "don't wanna be here", "dont want to be here",
    "wish i was dead", "wish i were dead", "wish i wasnt here",
    "wish i wasn't here", "rather be dead", "want to be dead",
    "no point in living", "no point living", "nothing to live for",
    "tired of living", "done with life", "end myself",
    # Indirect phrasings clinicians flagged — these carry real intent and were
    # previously invisible ("I have lost the willing to live" reached the bot
    # and got "You're welcome.").
    "better off without me", "lost the will to live", "lost the willing to live",
    # Method-based phrasings (self-harm intent expressed via a method)
    "jump from", "jump off", "jump in front", "jumping from", "jumping off",
    "hang myself", "hanging myself", "slit", "slitting",
    "cut myself", "cutting myself", "overdose", "over dose",
    "poison myself", "drown myself", "set myself on fire", "in front of a train",
    "harming myself", "hurting myself", "cut my wrist", "cutting my wrist",
    "scratch myself", "scratching myself",
    # Hindi transliterations
    "khud ko marna", "aatma hatya", "jeena nahi chahta", "marna chahta",
    "marr jana", "mar jana", "khod ko takleef", "apne aap ko chot",
    "chhat se kud", "neeche kud", "phaansi", "khudkushi", "jaan dena",
    "jaan de dunga", "jaan de dungi",
]

TRAUMA_KEYWORDS = [
    "abuse", "abusive", "assault", "rape", "raping", "rapist", "violence", "violent",
    "beaten", "beating", "hit",
    "sexual assault", "sexual abuse", "molestation", "molested", "trauma",
    "attacked", "attacking", "harassed", "harassment", "harassing",
    "tortured", "torture", "domestic violence",
]

# The patient asking to stop, pause, or continue later. THE canonical list —
# llm._has_stop_intent defers to this so the two can't drift apart again.
#
# Every phrase below marked (real) was said by an actual patient and IGNORED,
# because the list didn't contain it. Most damning: the bot itself offers
# "Let's take a pause if you'd like... How would you prefer to proceed?" — and
# the word "pause" was in NO stop-detector, so "yes lets pause" fell through to
# the next interview question. We were asking a question we could not hear the
# answer to.
SOFT_STOP_PHRASES = [
    # Stopping outright
    "i want to end", "end this", "stop this", "want to stop",
    "let's stop", "lets stop", "done for now",
    "dont ask more", "don't ask more", "dont ask me more", "stop asking",  # (real)
    "can we not do this",                                                   # (real)
    # Pausing — the bot's OWN offered word, previously unrecognised
    "pause", "take a break", "have a break", "need a break",                # (real)
    # Deferring to another time
    "another time", "next session", "some other time",                      # (real)
    "resume tomorrow", "resume later", "continue later", "discuss later",   # (real)
    "discuss this later", "talk later", "do this later", "later maybe",
    "wrap up", "wrap this up",                                              # (real)
    # Leaving
    "goodnight", "good night", "goodbye", "bye", "i gotta sleep",
    "i have to sleep", "i need to sleep",
]

CRISIS_SAFE_SIGNALS = [
    "ok", "okay", "i'm fine", "im fine", "safe", "better",
    "continue", "yes", "sure", "ready", "let's continue",
    "thik hai", "theek hai", "bilkul", "haan",
]


# ── Detection functions ───────────────────────────────────────

def _keyword_pattern(keywords):
    r"""Compile keywords with a LEADING word boundary only.

    Plain substring matching (`kw in text`) matched keywords INSIDE unrelated
    words, with real consequences:
      - "hit"  matched "Tris-hit-a", "Ro-hit", "Mo-hit", "Nik-hit-a" — common
        Indian names in an Indian deployment. A patient who typed her own name
        was told "Please know that what happened was not your fault."
      - "rape" matched "the-rap-ist" and "g-rape-s".

    The boundary is leading-only on purpose. Anchoring the END too (\b...\b)
    would stop "overdose" matching "overdosed" and "suicide" matching
    "suicidal" — i.e. it would introduce FALSE NEGATIVES on crisis detection,
    which is the one direction we must never fail in. A leading \b removes the
    mid-word false positives while keeping every suffixed form.
    """
    return re.compile(
        r"\b(?:" + "|".join(re.escape(k) for k in keywords) + r")",
        re.IGNORECASE,
    )


_CRISIS_RE = _keyword_pattern(CRISIS_KEYWORDS)
_TRAUMA_RE = _keyword_pattern(TRAUMA_KEYWORDS)


def is_crisis_message(text: str) -> bool:
    """True if message contains crisis/suicidal keywords."""
    return bool(_CRISIS_RE.search(text or ""))


def is_trauma_message(text: str) -> bool:
    """True if message contains trauma-related keywords."""
    return bool(_TRAUMA_RE.search(text or ""))


# NOTE: both boundaries here, unlike _keyword_pattern's leading-only.
# Crisis needs trailing suffixes to match ("overdose" -> "overdosed"), but a
# stop-phrase must NOT: "pause" is a request, "paused" is not — "I paused my
# medication once" is medication history and must not end the interview.
# Leading-only would also let "pause" fire on "menoPAUSE", a word that
# genuinely comes up in a psychiatric intake.
_SOFT_STOP_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(p) for p in SOFT_STOP_PHRASES) + r")\b",
    re.IGNORECASE,
)


def is_soft_stop_message(text: str) -> bool:
    """True if patient wants to end/pause the session."""
    return bool(_SOFT_STOP_RE.search(text or ""))


def is_safe_signal(text: str) -> bool:
    """True if patient signals they are safe (used after crisis mode)."""
    lowered = text.lower()
    return any(sig in lowered for sig in CRISIS_SAFE_SIGNALS)


async def handle_crisis(update, context, session) -> str:
    """
    Handle crisis message: send compassionate response and crisis helpline.
    Returns CRISIS_MODE state.
    """
    from data_model import CRISIS_MODE

    crisis_message = (
        "I hear that you're going through something very difficult right now. 💙\n\n"
        "Please know that you are not alone, and support is available right now.\n\n"
        f"🆘 Crisis Helpline: {CRISIS_HELPLINE}\n\n"
        "Please reach out to them — they are trained to help and available 24/7.\n\n"
        "Once you feel safe and ready, we can continue at your own pace. "
        "You don't have to do anything you're not comfortable with."
    )

    await update.message.reply_text(crisis_message, parse_mode=None)
    await db.set_session_status(update.effective_chat.id, "crisis")

    clinician_id = os.getenv("CLINICIAN_TELEGRAM_ID")
    if clinician_id:
        try:
            log_text = update.message.text[:100].replace("\\", "").replace("\"", "")
            alert_msg = (
                f"⚠️ CRISIS FLAG: Patient in session {update.effective_chat.id} "
                f"may need immediate attention.\n\n"
                f"Last message: \"{log_text}\""
            )
            await context.bot.send_message(clinician_id, alert_msg, parse_mode=None)
        except Exception as e:
            print(f"Failed to notify clinician: {e}")

    return CRISIS_MODE


async def handle_trauma_alert(update, context, section: str) -> None:
    """Send explicit care message after trauma disclosures."""
    care_message = (
        "Thank you for trusting me with that. I recognize that sharing this takes courage. 💙\n\n"
        "Please know that what happened was not your fault. Your feelings are valid.\n\n"
        "If you ever feel overwhelmed, it's okay to take a break. "
        "You can pause anytime by typing /pause."
    )
    await update.message.reply_text(care_message, parse_mode=None)


async def handle_crisis_response(update, context) -> str:
    """
    Handle patient's response after crisis message.
    Returns CRISIS_MODE if still not safe, COLLECTING_HISTORY if safe.
    """
    from data_model import CRISIS_MODE, COLLECTING_HISTORY

    patient_text = update.message.text.lower()
    is_safe = any(signal in patient_text for signal in CRISIS_SAFE_SIGNALS)

    if is_safe:
        await db.set_session_status(update.effective_chat.id, "active")
        resume_message = (
            "I'm glad you're okay. Whenever you feel ready, we can continue. "
            "Just send me any message to pick up where we left off. 💙"
        )
        await update.message.reply_text(resume_message, parse_mode=None)
        return COLLECTING_HISTORY
    else:
        persistent_message = (
            f"I'm still here with you. 💙\n\n"
            f"If you need immediate support, please reach out to:\n"
            f"🆘 {CRISIS_HELPLINE}\n\n"
            f"They're available 24/7 and trained to help."
        )
        await update.message.reply_text(persistent_message, parse_mode=None)
        return CRISIS_MODE