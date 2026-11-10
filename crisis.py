"""
Crisis brief-triage engine for the psychiatric history-taking bot.

Replaces the old static crisis handling (a single canned helpline message
resent verbatim on every turn) with a short, stateful, *adaptive* safety
triage:

    acknowledge  →  intent  →  plan  →  immediacy  →  safety-plan / hand-off

Design goals (from clinician feedback):
  * NEVER repeat the same message verbatim turn after turn.
  * Actually process what the patient says — reflect it back, and adapt the
    next question to their answer instead of ignoring context.
  * Ask a brief risk hierarchy (intent → plan → immediacy) one gentle question
    at a time, rather than jumping straight to a phone number and stopping.
  * Escalate a clinician alert when intent + plan / immediacy indicate high or
    imminent risk.
  * Handle "I want to talk about something else" / off-topic without replaying
    the helpline — acknowledge, do a soft safety check, and adapt.
  * Provide the helpline once up front and again at hand-off, woven in — not as
    the entire reply every single turn.
  * Let the patient return to the interview once they signal safety, with a
    warm bridge rather than an abrupt snap back to questions.

This module is interface-agnostic (no Telegram / REST imports). The caller
sends `message` to the patient and, when `stay_in_crisis` is False, resumes the
normal interview. `clinician_alert` (when set) is a short string the caller may
forward to the on-call clinician.

Risk classification is deterministic (keyword/pattern based) so escalation never
depends on the LLM returning parseable JSON. The LLM is used only to phrase the
empathetic reply naturally; a deterministic fallback covers any LLM failure.
"""

import os
import re
from typing import Optional

from utils import get_logger

logger = get_logger("crisis")

CRISIS_HELPLINE = os.getenv("CRISIS_HELPLINE", "iCall: 9152987821")

# Consecutive turns carrying a risk signal before we hand off regardless of
# tense. Clinician direction: even if it's all in the past, three suicidal or
# urgent messages in a row means this is no longer a history-taking
# conversation.
RISK_TURNS_BEFORE_HANDOFF = int(os.getenv("RISK_TURNS_BEFORE_HANDOFF", "3"))


# ── Triage stages ──────────────────────────────────────────────
# Brief hierarchy. "closure" is the supportive hand-off / continue-or-pause
# offer once the key dimensions are covered or the patient has de-escalated.
STAGES = ["intent", "plan", "immediacy", "closure"]


class CrisisTurn:
    """Result of one crisis turn."""

    __slots__ = ("message", "stay_in_crisis", "clinician_alert", "risk")

    def __init__(self, message, stay_in_crisis, clinician_alert=None, risk="unknown"):
        self.message = message
        self.stay_in_crisis = stay_in_crisis
        self.clinician_alert = clinician_alert
        self.risk = risk


class CrisisAssessment:
    """Per-session crisis triage state. Kept in-memory across CRISIS_MODE turns."""

    __slots__ = ("stage_idx", "findings", "risk", "turns", "helpline_given",
                 "resolved", "alerted_at", "risk_turns", "urgent_turns",
                 "interview_locked", "awaiting_confirmation")

    def __init__(self):
        self.stage_idx = 0
        self.findings = {}      # dimension -> short human string
        self.risk = "unknown"   # unknown | low | moderate | high | imminent
        self.turns = 0
        self.helpline_given = False
        self.resolved = False
        self.alerted_at = ""    # highest risk level already alerted on (dedupes clinician pings)
        # Consecutive turns carrying a risk signal. Clinician direction: even if
        # the patient is talking about the PAST, three suicidal/urgent messages
        # in a row is no longer a history-taking conversation — end it and hand
        # off. Any clear denial / safety statement resets this.
        self.risk_turns = 0
        # Urgent (high/imminent) turns so far. Drives which hand-off text goes
        # out: the clinician-approved message must land on the FIRST urgent
        # turn, whenever that happens — not on crisis turn 1. Keying the
        # variant off assess.turns meant escalating on turn 2 sent the
        # follow-up text and the approved message was never seen.
        self.urgent_turns = 0
        # Once the patient confirms the suicidal thoughts are ACTIVE, the
        # history-taking interview is over. Not paused — over. This is a
        # one-way door: de-escalation later ("I'm ok now") still must NOT drop
        # them back into "what is your highest level of education?". A patient
        # who has just told us they actively want to die needs a clinician, and
        # resuming an admin form around that is the exact "assessment takes
        # priority over the disclosure" failure clinicians flagged.
        self.interview_locked = False
        # De-escalation handshake. When the patient gives a genuine safety /
        # denial signal we do NOT snap straight back to the interview — we ask
        # them to confirm they're okay in their own words and set this flag.
        # Their NEXT message, if it confirms, resumes the interview. This is
        # what breaks the "urgent message forever" loop: previously, once risk
        # reached high/imminent, nothing the patient said could register as
        # de-escalation because the urgent branch returned first and risk never
        # fell.
        self.awaiting_confirmation = False

    def current_stage(self) -> str:
        return STAGES[min(self.stage_idx, len(STAGES) - 1)]

    def advance(self) -> None:
        if self.stage_idx < len(STAGES) - 1:
            self.stage_idx += 1


_STATES: dict[str, CrisisAssessment] = {}


def get_assessment(session_key: str) -> CrisisAssessment:
    if session_key not in _STATES:
        _STATES[session_key] = CrisisAssessment()
    return _STATES[session_key]


def clear(session_key: str) -> None:
    _STATES.pop(session_key, None)


def in_crisis(session_key: str) -> bool:
    return session_key in _STATES and not _STATES[session_key].resolved


# ── Deterministic classification ───────────────────────────────

def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", text.lower())).strip()


# Denial of intent / de-escalation
_DENY_INTENT = [
    "no", "nope", "not really", "i wouldnt", "i would not", "i wont", "i will not",
    "not going to", "dont want to die", "do not want to die", "just thoughts",
    "only thoughts", "wouldnt act", "would never", "not planning", "no plan",
    "dont have a plan", "do not have a plan", "not right now", "i can wait",
    "im not going to", "i am not going to",
    # Explicit denial of current suicidality. "I don't have any suicidal
    # thoughts at this moment" previously matched nothing here, so a clear
    # denial registered no de-escalation at all.
    "not suicidal", "no suicidal thoughts", "no longer suicidal",
    "dont have suicidal", "do not have suicidal", "dont have any suicidal",
    "do not have any suicidal", "not having those thoughts",
    "those thoughts are gone", "no such thoughts", "no thoughts of hurting",
]

# Explicit confirmation of safety — consulted ONLY at the de-escalation
# handshake step, after we have asked "can you tell me you're okay?". A bare
# "yes" / "ok" is a valid answer to THAT question, so this list is broader than
# _SAFE; the handshake only trusts it on a turn that carries no fresh risk cue
# (see _fresh_risk), so "yes" here can never confirm safety in the same breath
# as a plan or present intent.
_CONFIRM_SAFE = [
    "yes", "yeah", "yep", "yup", "ya", "okay", "ok", "okk", "kk",
    "im okay", "i am okay", "im ok", "i am ok", "im fine", "i am fine",
    "im safe", "i am safe", "im good", "i am good", "im better",
    "i feel better", "feeling better", "im alright", "i am alright",
    "i confirm", "confirmed", "i promise", "sure", "definitely", "absolutely",
    "im calm", "i am calm", "all good",
    # Hindi / Hinglish
    "haan", "haan ji", "thik hai", "theek hai", "bilkul", "thik hoon",
    "main thik hoon", "main theek hoon", "sab thik hai", "main safe hoon",
]
# Affirmation of intent to act
_YES_INTENT = [
    "yes", "yeah", "yep", "i do", "i want to", "i am going to", "im going to",
    "i will", "i intend", "planning to", "i might do it", "i want to die",
    # Colloquial forms. classify("I wanna die") previously returned {} — the
    # entry keyword list knew "want to die" and this one didn't, so a patient
    # entered crisis mode with NO intent registered and the risk ladder never
    # left "unknown".
    "wanna die", "wanna kill myself", "wanna end it", "wanna be dead",
    "gonna kill myself", "gonna end it", "want to be dead",
    "dont wanna live", "don't wanna live", "dont want to live",
    "wish i was dead", "wish i were dead", "no point in living",
    "tired of living", "better off without me", "lost the will to live",
    "lost the willing to live",
    # PAST-tense forms. "i wanted to die last year" matched nothing here
    # ("want to die" != "wanted to die"), so historical suicidality registered
    # no intent at all and the consecutive-risk counter never moved. These are
    # scored as intent; the tense logic in _update_risk decides the LEVEL.
    "wanted to die", "wanted to kill myself", "wanted to end it",
    "felt like dying", "felt like killing myself", "felt like ending it",
    "thought about dying", "thought about killing myself", "thought of dying",
    "tried to kill myself", "tried to die", "attempted suicide",
    # Acting on the thoughts. "i wanna act on it" — a direct answer to our own
    # intent question — matched NOTHING here, so the turn registered no intent
    # and produced no clinician alert.
    "wanna act", "want to act", "act on it", "acting on it", "act on them",
    "acting on them", "act on these", "might act", "could act", "close to acting",
]
# Unambiguous statements of intent — used ONLY at the closure stage.
# At closure the question we asked is "continue, or take a break?", so a bare
# "yes"/"yeah"/"i will" from _YES_INTENT is answering THAT, not declaring intent
# to act. Reading it as intent kept patients trapped in the helpline loop. These
# phrases mean the same thing regardless of what was asked.
_EXPLICIT_INTENT = [
    "i want to die", "i am going to", "im going to", "i intend",
    "planning to", "i might do it", "kill myself", "end it", "end my life",
]
# Concrete plan / means
_PLAN_MEANS = [
    "plan", "planned", "how i would", "how i will", "thought about how",
    "pills", "tablets", "overdose", "over dose", "hang", "hanging", "rope",
    "jump", "jumping", "cut", "cutting", "blade", "knife", "gun", "poison",
    "bridge", "train", "sleeping pills", "stockpiled", "bought",
]
# Immediacy
_IMMEDIATE = [
    "right now", "tonight", "today", "this moment", "about to", "gonna do it",
    "going to do it now", "as soon as", "in a few", "any minute", "cant wait",
]
# Safety / willingness to stay safe
_SAFE = [
    "safe", "im okay", "i am okay", "im ok", "i am ok", "feel better", "feeling better",
    "i can stay safe", "i will be okay", "ill be okay", "i can hold on", "i can wait",
    "im not in danger", "no im safe", "not going to hurt", "ready to continue",
    "lets continue", "we can continue", "im fine", "i am fine",
]
# Protective factors mentioned
_PROTECTIVE = [
    "my family", "my kids", "my children", "my parents", "my mother", "my father",
    "couldnt do that to", "my faith", "my god", "my friends", "someone needs me",
    "my partner", "my wife", "my husband",
]
# ── Tense ──────────────────────────────────────────────────────
# Intent is only downgraded to "moderate" when the patient is clearly talking
# about the PAST. Present-tense or unclear intent is treated as active.
# "I felt like killing myself during a panic attack last week" and "I want to
# die" are completely different clinical facts, and the bot previously scored
# both the same.
_PAST_TENSE = [
    "used to", "back then", "at that time", "in the past", "at the time",
    "years ago", "year ago", "months ago", "month ago", "weeks ago",
    "week ago", "days ago", "day ago", "last year", "last month", "last week",
    "when i was", "as a child", "as a teenager", "in school", "in college",
    "long ago", "a while ago", "some time ago", "previously", "earlier",
    "before", "that was", "it was", "i was", "i had", "i felt",
    "not anymore", "not any more", "no longer", "has passed", "it passed",
    "got better", "went away",
]
# Explicit present markers override the past ones: "I used to feel this way and
# I still do" is present.
_PRESENT_TENSE = [
    "right now", "at the moment", "currently", "today", "tonight",
    "these days", "lately", "still", "even now", "at present",
    "i am", "i m", "im", "i feel like", "i want", "i wanna", "i keep",
    "keeps happening", "happening now", "going on now", "present",
]

# Wants to change subject
_DEFLECT = [
    "talk about something else", "change the subject", "dont want to talk about this",
    "do not want to talk about this", "something else", "different topic",
    "can we talk about", "not this", "lets not talk about",
]


_CUE_CACHE: dict = {}


def _has(text_norm: str, cues) -> bool:
    """Word-boundary cue match.

    This was `any(c in text_norm for c in cues)` — plain substring — and it
    read the most dangerous answer in the triage backwards:

        "Are you having those thoughts right now?"  ->  "right now"
        _has("right now", _DENY_INTENT) -> True     because "no" is inside "NOw"

    So a patient confirming IMMINENT risk was classified denies_intent=True,
    which also suppressed affirms_intent (it is `and not denies`). Same bug
    family as "hit" matching inside "Tris-hit-a" in safety.py.

    Boundaries on BOTH sides here: these cues are whole words/phrases ("no",
    "yes", "plan"), not stems needing suffixes, so "no" must not match "now"
    or "nothing", and "plan" must not match "planning to move house".
    """
    key = id(cues)
    pattern = _CUE_CACHE.get(key)
    if pattern is None:
        pattern = re.compile(
            r"\b(?:" + "|".join(re.escape(c) for c in cues) + r")\b",
            re.IGNORECASE,
        )
        _CUE_CACHE[key] = pattern
    return bool(pattern.search(text_norm or ""))


def classify(text: str) -> dict:
    """Deterministically read risk features from the patient's message."""
    n = _norm(text)
    # Order matters: an explicit denial of intent outweighs a bare "die" keyword.
    denies = _has(n, _DENY_INTENT)
    affirms_intent = _has(n, _YES_INTENT) and not denies
    plan = _has(n, _PLAN_MEANS)
    immediate = _has(n, _IMMEDIATE)
    safe = _has(n, _SAFE)
    protective = _has(n, _PROTECTIVE)
    deflect = _has(n, _DEFLECT)

    past = _has(n, _PAST_TENSE)
    present = _has(n, _PRESENT_TENSE)

    return {
        "denies_intent": denies,
        "affirms_intent": affirms_intent,
        "plan": plan,
        "immediate": immediate,
        "safe": safe,
        "protective": protective,
        "deflect": deflect,
        # past_only == "they are clearly describing history, not now".
        # An explicit present marker beats a past one: "I used to feel this way
        # and I still do" is present.
        "past_only": past and not present,
        "present_tense": present,
    }


def _update_risk(assess: CrisisAssessment, f: dict) -> None:
    """Fold this turn's features into the running risk level.

    Risk is computed from ACCUMULATED findings across the whole crisis
    conversation, not just this message — so intent stated three turns ago plus
    a plan mentioned now still escalates to high. Risk only rises; a later "I'm
    safe" is handled by the separate de-escalation path so we never under-react
    mid-triage.
    """
    # Record human-readable findings for the clinician hand-off first, so the
    # level below reflects everything disclosed so far.
    if f["affirms_intent"]:
        assess.findings["intent"] = "expressed intent to act"
    elif f["denies_intent"]:
        assess.findings.setdefault("intent", "denies active intent")
    if f["plan"]:
        assess.findings["plan"] = "mentioned a method/plan"
    if f["immediate"]:
        assess.findings["immediacy"] = "immediate timeframe"
    if f["protective"]:
        assess.findings["protective"] = "protective factor mentioned"

    has_intent = assess.findings.get("intent") == "expressed intent to act" or f["affirms_intent"]
    has_plan = "plan" in assess.findings or f["plan"]
    has_immediate = "immediacy" in assess.findings or f["immediate"]

    # ── Tense decides everything ──────────────────────────────────────────
    # Clinician direction: intent is "moderate" ONLY when the patient is
    # clearly describing the PAST. Present-tense or unclear intent is an
    # immediate case — we hand off and do NOT go on to ask about method.
    #
    #   "I felt like killing myself during a panic attack last week"  -> moderate
    #   "I wanna die" / "yes, right now"                              -> imminent
    #
    # Previously both scored moderate (high required intent + plan), so a
    # patient who confirmed present intent was answered with "Have you thought
    # about how you might do it?" — still triaging someone who had just said
    # she meant it.
    # Gate on THIS turn's affirms_intent, never the accumulated has_intent:
    # tense is a property of the message being spoken. Using has_intent meant a
    # later present-tense message re-dated an OLD disclosure — "no not anymore,
    # im fine" contains "im", so past_only went False, the intent from two
    # turns ago was reclassified as present, and a DENIAL escalated to imminent.
    past_only = f.get("past_only", False)
    present_intent = f["affirms_intent"] and not past_only
    if present_intent:
        assess.findings["intent"] = "CONFIRMED active/present intent to act"
        # One-way door: the interview does not resume after this.
        assess.interview_locked = True
    elif f["affirms_intent"] and past_only:
        assess.findings.setdefault("intent", "intent described in the past")

    # Consecutive risk turns. Even about the past, three suicidal/urgent
    # messages in a row is no longer history-taking — end it and hand off.
    # Counts anything that SEEMS suicidal, not just affirmed intent: a patient
    # dwelling on "I wanted to die", "I had a plan back then", "I tried once"
    # is circling it, and that is the signal regardless of grammar.
    if f["affirms_intent"] or f["plan"] or f["immediate"]:
        assess.risk_turns += 1
    elif f["denies_intent"] or f["safe"]:
        # A clear denial or safety statement means they are not circling it.
        assess.risk_turns = 0
    repeated_risk = assess.risk_turns >= RISK_TURNS_BEFORE_HANDOFF
    if repeated_risk:
        assess.findings["repetition"] = (
            f"{assess.risk_turns} consecutive turns expressing risk"
        )

    if has_immediate and (has_plan or has_intent):
        level = "imminent"
    elif present_intent:
        # Intent stated in the present (or with no clear tense) — treat as active.
        level = "imminent"
    elif repeated_risk:
        # Past-tense, but they keep returning to it. Hand off anyway.
        level = "imminent"
    elif has_intent and has_plan:
        level = "high"
    elif has_plan or has_intent:
        level = "moderate"
    elif (f["denies_intent"] or f["safe"]) and assess.risk in ("unknown", "low"):
        level = "low"
    else:
        level = assess.risk if assess.risk != "unknown" else "moderate"

    order = {"unknown": 0, "low": 1, "moderate": 2, "high": 3, "imminent": 4}
    if order[level] >= order[assess.risk]:
        assess.risk = level


def _deescalated(assess: CrisisAssessment, f: dict) -> bool:
    """Patient looks safe enough to gently offer returning to the interview."""
    return (
        assess.risk in ("low", "unknown")
        and (f["safe"] or (f["denies_intent"] and f["protective"]))
        and assess.turns >= 1
    )


def _fresh_risk(text_norm: str, f: dict, explicit_only: bool = False) -> bool:
    """Risk expressed in THIS message, not accumulated across the session.

    Used by the de-escalation handshake to decide whether a safety statement is
    trustworthy this turn. Immediacy ALONE ("right now") is deliberately not
    fresh risk: "I want to talk right now" and "I'm fine now" are the patient
    trying to de-escalate, and reading them as acute is exactly what re-trapped
    them. Immediacy only counts alongside a plan or intent — mirroring
    _update_risk's own escalation rule.

    `explicit_only` is for the CONFIRMATION step. There we asked "can you tell
    me you're okay?", so a bare "yes"/"yeah" answers THAT question — it is not a
    declaration of intent, even though "yes" lives in _YES_INTENT (this is the
    same trap the closure stage documents). With explicit_only we count intent
    only from the unambiguous _EXPLICIT_INTENT phrases.
    """
    if explicit_only:
        intent = _has(text_norm, _EXPLICIT_INTENT)
    else:
        intent = _has(text_norm, _EXPLICIT_INTENT) or (
            f["affirms_intent"] and not f.get("past_only", False)
        )
    return intent or f["plan"] or (f["immediate"] and (f["plan"] or intent))


# ── Language-specific fallback wording ─────────────────────────
# Used verbatim only when the LLM call fails; otherwise the LLM phrases the
# reply and these guide its content.

def _helpline_line(lang: str) -> str:
    if lang == "hi":
        return f"🆘 Madad ke liye: {CRISIS_HELPLINE} (24/7 available hai)."
    if lang == "hinglish":
        return f"🆘 Help ke liye: {CRISIS_HELPLINE} (24/7 available)."
    return f"🆘 Help is available any time: {CRISIS_HELPLINE} (24/7)."


_STAGE_QUESTION = {
    "intent": {
        "en": "When these thoughts come, do you find yourself wanting to act on them, or are they more thoughts that pass through?",
        "hi": "Jab ye khayal aate hain, kya aapka man unhe karne ka hota hai, ya wo bas aa kar chale jaate hain?",
        "hinglish": "Jab ye thoughts aate hain, kya aapka man unhe act karne ka hota hai, ya wo bas pass ho jaate hain?",
    },
    "plan": {
        "en": "Have you thought about how you might do it, or made any kind of plan?",
        "hi": "Kya aapne socha hai ki aap ye kaise karenge, ya koi plan banaya hai?",
        "hinglish": "Kya aapne socha hai ki aap ye kaise karoge, ya koi plan banaya hai?",
    },
    "immediacy": {
        "en": "Are you feeling like you might act on this right now, or is it something that comes and goes?",
        "hi": "Kya aapko lag raha hai ki aap abhi kuch kar sakte hain, ya ye aata-jaata rehta hai?",
        "hinglish": "Kya aapko lagta hai ki aap abhi kuch kar sakte ho, ya ye aata-jaata rehta hai?",
    },
}


_ACK_ROTATION = {
    "en": [
        "I can hear how much pain you're in, and it took courage to say it. I'm here with you.",
        "That sounds incredibly heavy to be carrying. Thank you for letting me stay with you in it.",
        "I'm really glad you told me. You don't have to hold this on your own right now.",
        "What you're feeling matters, and I'm not going anywhere. Let's take this gently.",
    ],
    "hi": [
        "Aap itni takleef mein hain — ye batane ke liye himmat chahiye. Main yahin hoon.",
        "Ye bojh uthana bahut mushkil hai. Aapne mujhe bataya, iske liye shukriya. Main saath hoon.",
        "Achha hua aapne mujhe bataya. Abhi aapko ye akele nahi jhelna hai.",
        "Aapki feelings mayne rakhti hain, aur main kahin nahi ja raha. Dheere dheere chalte hain.",
    ],
    "hinglish": [
        "Aap itni pain mein ho — ye batane ke liye himmat lagti hai. Main yahin hoon aapke saath.",
        "Ye carry karna bahut heavy hai. Thank you ki aapne mujhe bataya. Main saath hoon.",
        "Achha kiya jo aapne bataya. Abhi ye akele face karne ki zaroorat nahi hai.",
        "Aapki feelings matter karti hain, aur main kahin nahi ja raha. Slowly chalte hain.",
    ],
}


def _acknowledge(lang: str, turn: int = 0) -> str:
    options = _ACK_ROTATION.get(lang, _ACK_ROTATION["en"])
    return options[turn % len(options)]


def _deterministic_message(assess: CrisisAssessment, lang: str, stage: str, include_helpline: bool) -> str:
    parts = [_acknowledge(lang, assess.turns)]
    q = _STAGE_QUESTION.get(stage, {}).get(lang) or _STAGE_QUESTION.get(stage, {}).get("en")
    if q:
        parts.append(q)
    if include_helpline:
        parts.append(_helpline_line(lang))
    return "\n\n".join(parts)


def _urgent_message(lang: str, turn: int = 0) -> str:
    hl = _helpline_line(lang)
    if lang == "hi":
        variants = [
            ("Mujhe aapki safety ki bahut chinta ho rahi hai abhi. Please turant kisi se baat karein.\n\n"
             f"{hl}\n\n"
             "Agar aap turant khatre mein hain, to kripya emergency services ya kisi bharose ke insaan ko abhi bulaayein. "
             "Kya aapke paas koi hai jise aap abhi bula sakte hain?"),
            ("Main abhi bhi aapke saath hoon aur aapki fikar kar raha hoon. Aapko akele nahi rehna hai.\n\n"
             f"{hl}\n\n"
             "Kya aap abhi kisi bharose ke insaan ke paas ja sakte hain, ya unhe bula sakte hain?"),
        ]
    elif lang == "hinglish":
        variants = [
            ("Mujhe abhi aapki safety ki bahut fikar ho rahi hai. Please turant kisi se baat karo.\n\n"
             f"{hl}\n\n"
             "Agar aap immediate danger mein ho, to please emergency services ya kisi trusted person ko abhi call karo. "
             "Kya koi aapke paas hai jise aap abhi reach kar sakte ho?"),
            ("Main abhi bhi aapke saath hoon aur aapki care karta hoon. Aapko ye akele face nahi karna.\n\n"
             f"{hl}\n\n"
             "Kya aap abhi kisi trusted person ke paas ja sakte ho, ya unhe bula sakte ho?"),
        ]
    else:
        # First urgent turn: the clinician-approved hand-off, sent verbatim the
        # moment active intent is confirmed. The interview stops here — we do
        # NOT go on to ask about method.
        # (No markdown asterisks: the caller sends with parse_mode=None, so
        # "*Crisis Helpline:*" would reach the patient with the asterisks in it.)
        variants = [
            ("I hear that you're going through something very difficult right now. 💙\n\n"
             "Please know that you are not alone, and support is available right now.\n\n"
             f"🆘 Crisis Helpline: {CRISIS_HELPLINE}\n\n"
             "Please reach out to them — they are trained to help and available 24/7.\n\n"
             "Once you feel safe and ready, we can continue at your own pace. "
             "You don't have to do anything you're not comfortable with."),
            # Subsequent urgent turns. The message above is never repeated —
            # that was the original bug. These stay with the patient, keep the
            # resource visible, and each one asks something different so it
            # doesn't read as a loop.
            ("I'm still right here with you, and I care about what happens to you. "
             "You shouldn't have to face this alone.\n\n"
             f"🆘 Crisis Helpline: {CRISIS_HELPLINE}\n\n"
             "Are you able to get to someone you trust right now, or call them to be with you?"),
            ("I'm not going anywhere. You don't have to explain or justify any of "
             "this to me.\n\n"
             f"🆘 Crisis Helpline: {CRISIS_HELPLINE}\n\n"
             "Is there anyone at home or nearby who could sit with you for a while?"),
            ("Thank you for staying with me. What you're carrying is serious, and "
             "you deserve someone with you for it.\n\n"
             f"🆘 Crisis Helpline: {CRISIS_HELPLINE}\n\n"
             "Would you be willing to call that number now, while I'm here with you?"),
        ]
    # `turn` is assess.urgent_turns, 1-based.
    #
    # variants[0] is the clinician-approved hand-off and is sent ONCE, on the
    # first urgent turn. After that we never come back to it: `% len(variants)`
    # cycled it back every other turn, which is the "resend the identical
    # crisis message forever" bug this engine exists to replace, just with a
    # period of 2. Subsequent turns rotate through the follow-ups only.
    if turn <= 1:
        return variants[0]
    follow_ups = variants[1:] or variants
    return follow_ups[(turn - 2) % len(follow_ups)]


def _locked_handoff_message(lang: str) -> str:
    """Sent when the patient has confirmed ACTIVE intent and later de-escalates.

    We do not resume the interview. We stay with them and point at real help.
    """
    hl = _helpline_line(lang)
    if lang == "hi":
        return (
            "Mujhe khushi hai ki abhi aap thoda behtar mehsoos kar rahe hain. "
            "Lekin aapne jo bataya, woh bahut important hai — aur uske baad "
            "main aapse form wale sawaal poochhna theek nahi samajhta.\n\n"
            f"{hl}\n\n"
            "Aapki baat aapke clinician tak pahunch chuki hai. Jab aap "
            "chaahein, hum baad mein baaki baatein kar sakte hain — abhi sabse "
            "zaroori yeh hai ki aap surakshit rahein aur kisi insaan se baat "
            "karein."
        )
    if lang == "hinglish":
        return (
            "Achha laga ki abhi aap thoda better feel kar rahe ho. Lekin jo "
            "aapne bataya woh bahut important hai — uske baad form wale "
            "questions poochhna sahi nahi lagta.\n\n"
            f"{hl}\n\n"
            "Aapki baat aapke clinician tak pahunch gayi hai. Baaki sawaal hum "
            "baad mein kar sakte hain — abhi sabse zaroori hai ki aap safe raho "
            "aur kisi se baat karo."
        )
    return (
        "I'm glad you're feeling steadier right now. But what you told me "
        "matters, and I'm not going to move you straight back to routine "
        "questions after that.\n\n"
        f"{hl}\n\n"
        "What you shared has been passed to your clinician. We can pick up the "
        "rest another time — right now the only thing that matters is that "
        "you're safe and talking to a person."
    )


def _closure_message(lang: str, safe: bool) -> str:
    if lang == "hi":
        base = (
            "Ye baat karne ke liye shukriya. Aapki takleef mayne rakhti hai, aur aap akele nahi hain.\n\n"
            f"{_helpline_line(lang)}\n\n"
        )
        tail = (
            "Jab aap tayyar ho, hum aaram se aage baat kar sakte hain — ya aap /pause type karke break le sakte hain. "
            "Aap kya karna chahenge?"
        )
        return base + tail
    if lang == "hinglish":
        base = (
            "Ye share karne ke liye thank you. Aapki feelings matter karti hain, aur aap akele nahi ho.\n\n"
            f"{_helpline_line(lang)}\n\n"
        )
        tail = (
            "Jab aap ready ho, hum slowly aage baat kar sakte hain — ya aap /pause type karke break le sakte ho. "
            "Aap kya karna chahoge?"
        )
        return base + tail
    base = (
        "Thank you for staying with me through that. What you're feeling matters, and you don't have to carry it alone.\n\n"
        f"{_helpline_line(lang)}\n\n"
    )
    tail = (
        "When you feel ready, we can continue gently — or you can take a break any time by typing /pause. "
        "What would you like to do?"
    )
    return base + tail


def _bridge_back(lang: str) -> str:
    if lang == "hi":
        return "Aapka shukriya ki aapne mujhe bataya. Jab aap kahen, hum dheere dheere aage badhte hain."
    if lang == "hinglish":
        return "Thank you ki aapne mujhe bataya. Jab aap kaho, hum slowly aage badhte hain."
    return "Thank you for trusting me with that. Whenever you're ready, we'll take the next bit slowly."


def _confirm_prompt(lang: str) -> str:
    """De-escalation step 1: acknowledge the shift and ask them to confirm they
    are safe, in their own words, before we resume. Keeps the helpline visible.
    """
    hl = _helpline_line(lang)
    if lang == "hi":
        return (
            "Sun kar thoda sukoon mila ki abhi aap pehle se behtar mehsoos kar "
            "rahe hain. 💙 Aage badhne se pehle main bas ye pakka karna chahta "
            "hoon — kya aap abhi surakshit mehsoos kar rahe hain? Agar haan, to "
            "apne shabdon mein bata dijiye ki aap thik hain, phir hum aaram se "
            "aage badhte hain.\n\n"
            f"{hl}"
        )
    if lang == "hinglish":
        return (
            "Sun kar accha laga ki abhi aap thoda better feel kar rahe ho. 💙 "
            "Aage badhne se pehle bas ye confirm kar do — kya aap abhi safe feel "
            "kar rahe ho? Agar haan, to apne words mein bata do ki aap thik ho, "
            "phir hum slowly aage badhte hain.\n\n"
            f"{hl}"
        )
    return (
        "I'm really glad to hear you're feeling a bit steadier. 💙 Before we pick "
        "things back up, I just want to make sure you're safe right now. If you "
        "are, could you tell me in your own words that you're okay — then we'll "
        "continue gently.\n\n"
        f"{hl}"
    )


def _resume_bridge(lang: str) -> str:
    """De-escalation step 2: the patient confirmed safety. Warm bridge back into
    the interview (the caller sends the next question right after this)."""
    if lang == "hi":
        return (
            "Mujhe khushi hai ki aap ab zyada sthir mehsoos kar rahe hain. "
            "Shukriya ki aap mere saath bane rahe. Hum agla hissa dheere dheere "
            "aage badhaate hain."
        )
    if lang == "hinglish":
        return (
            "Mujhe khushi hai ki aap ab thoda steady feel kar rahe ho. Thank you "
            "ki aap mere saath rahe. Hum agla part slowly aage badhate hain."
        )
    return (
        "I'm glad you're steadier. Thank you for staying with me through that — "
        "we'll take the next part slowly."
    )


# ── LLM phrasing ───────────────────────────────────────────────

_CRISIS_SYSTEM_PROMPT = """You are a warm, calm mental-health support worker doing a brief safety check with someone who has expressed thoughts of suicide or self-harm. This is NOT the routine interview — right now the ONLY priority is their safety and feeling heard.

Rules:
- First, briefly reflect back what they just said, in your own warm words, so they feel heard. Do NOT dismiss, argue, or rush.
- Then ask exactly ONE gentle question about the specific safety point given to you (their intent, whether they have a plan, or how immediate it feels). Ask nothing else.
- Never say "thank you for sharing" or any scripted filler. Never sound like a form.
- Do not repeat wording you have used before. Vary how you speak.
- Keep it under 55 words. Plain, human language — no clinical jargon.
- If a helpline is provided to you, include it naturally on its own line. If none is provided, do not invent one.
- Respond only in the requested language."""


def _language_label(lang: str) -> str:
    return {"en": "English", "hi": "Hindi", "hinglish": "Hinglish (Hindi written in English)"}.get(lang, "English")


# A crisis reply must NEVER congratulate, reassure, or express pleasure.
#
# Real failure: "Are you having those thoughts right now, Harshal?" -> "Yes"
# -> "I'm glad to hear that." A patient confirmed ACTIVE suicidal ideation and
# was congratulated for it.
#
# Note what did and did not fail. The deterministic layer was correct
# throughout: crisis mode entered on "I die", the tense question fired,
# classify("Yes") read affirms_intent. Only the LLM PHRASING inverted the
# meaning — it pattern-matched a bare "Yes" as good news. This is the same
# family as "You're welcome." in reply to "Yess" about self-harm.
#
# So we do not ask the model to be careful — we check its output and throw it
# away if it carries any of these. _llm_reply returning None makes the caller
# send the deterministic message, which is always safe. A blunt, correct reply
# beats a warm, catastrophic one.
_UNSAFE_CRISIS_PHRASING = re.compile(
    r"\b(?:"
    r"i(?:'|’)?m\s+glad|glad\s+to\s+hear|good\s+to\s+hear|nice\s+to\s+hear|"
    r"happy\s+to\s+hear|i(?:'|’)?m\s+happy|i(?:'|’)?m\s+relieved|"
    r"that(?:'|’)?s\s+(?:good|great|wonderful|nice|reassuring|a\s+relief)|"
    r"you(?:'|’)?re\s+welcome|congratulations|well\s+done|"
    r"that(?:'|’)?s\s+the\s+spirit|keep\s+it\s+up"
    r")\b",
    re.IGNORECASE,
)


# Minimising / toxic positivity. Telling a suicidal patient "it'll pass" or
# "stay strong" is dismissive at the exact moment they disclosed.
_CRISIS_MINIMISING = re.compile(
    r"\b(?:"
    r"it(?:'|’)?ll\s+pass|it\s+will\s+pass|this\s+too\s+shall\s+pass|"
    r"things\s+will\s+get\s+better|it\s+gets\s+better|"
    r"stay\s+strong|be\s+strong|stay\s+positive|think\s+positive|cheer\s+up|"
    r"everyone\s+feels\s+(?:this|that)\s+way|it(?:'|’)?s\s+not\s+that\s+bad|"
    r"don(?:'|’)?t\s+worry|nothing\s+to\s+worry|try\s+to\s+relax|calm\s+down|"
    r"look\s+on\s+the\s+bright\s+side|others\s+have\s+it\s+worse|"
    r"you(?:'|’)?ll\s+be\s+fine|snap\s+out\s+of\s+it"
    r")\b",
    re.IGNORECASE,
)

# The bot is not a therapist and must never offer treatment or advice.
_CRISIS_ADVICE = re.compile(
    r"\b(?:"
    r"you\s+should\s+try|i\s+recommend|i\s+suggest\s+you|i\s+advise|"
    r"have\s+you\s+(?:tried|considered)\s+(?:therapy|meditation|exercise|yoga)|"
    r"take\s+(?:some\s+)?medication|i\s+can\s+(?:help\s+you\s+get|treat|cure|fix)|"
    r"what\s+you\s+need\s+(?:is|to\s+do)"
    r")\b",
    re.IGNORECASE,
)

# The interview checklist leaking into a crisis turn. THIS is the
# "Yess -> any chest pain, palpitations, or breathlessness?" failure: the
# routine interview must be fully suspended during triage.
_CRISIS_OFFTOPIC = re.compile(
    r"\b(?:"
    r"chest\s+pain|palpitation|breathless|stomach\s+problem|nausea|"
    r"bowel|urination|urinary|joint\s+pain|muscle\s+pain|allerg|"
    r"thyroid|appetite|"
    r"highest\s+level\s+of\s+education|your\s+occupation|monthly\s+(?:household\s+)?income|"
    r"marital\s+status|mother\s+tongue|spiritual\s+practice|your\s+address|"
    r"socioeconomic|who\s+referred\s+you"
    r")\b",
    re.IGNORECASE,
)


def _crisis_reject_reason(text: str) -> Optional[str]:
    """Why this crisis reply must NOT be sent, or None if it's safe.

    Everything crisis._llm_reply produces goes through here. The model's output
    was previously sent unvalidated, and it has failed in this exact spot three
    separate ways — congratulating a suicide disclosure ("I'm glad to hear
    that" after "Yes"), answering one with "You're welcome.", and continuing
    the review-of-systems checklist ("any chest pain?"). All three were correct
    at the deterministic layer and wrong at the phrasing layer.

    Rejecting returns None from _llm_reply, so the caller sends
    _deterministic_message — always safe, if blunt. During a suicide triage a
    blunt correct reply beats a warm catastrophic one every time.
    """
    if not text or not text.strip():
        return "empty"
    if _UNSAFE_CRISIS_PHRASING.search(text):
        return "expresses pleasure/reassurance about a crisis disclosure"
    if _CRISIS_MINIMISING.search(text):
        return "minimises / toxic positivity"
    if _CRISIS_ADVICE.search(text):
        return "offers treatment or advice (we are not a therapist)"
    if _CRISIS_OFFTOPIC.search(text):
        return "leaks the routine interview checklist into crisis triage"
    if "?" not in text:
        # Every triage stage must ask something; a reply with no question is a
        # dead end that leaves the patient holding a disclosure alone.
        return "asks no question — triage cannot continue"
    return None


def _is_unsafe_crisis_reply(text: str) -> bool:
    """True if a crisis reply must not be sent."""
    return _crisis_reject_reason(text) is not None


async def _llm_reply(patient_text: str, stage: str, lang: str, include_helpline: bool, recent: str) -> Optional[str]:
    try:
        from llm import call_llm
    except Exception:
        return None

    focus = {
        "intent": "whether they actually want to act on the thoughts, or the thoughts pass",
        "plan": "whether they have thought about how they might do it / have a plan",
        "immediacy": "whether they feel they might act right now, or it comes and goes",
    }.get(stage, "how they are feeling right now and whether they feel safe")

    helpline = _helpline_line(lang) if include_helpline else "(no helpline this turn)"
    prompt = (
        f"Language: {_language_label(lang)}\n"
        f"Recent exchange:\n{recent}\n\n"
        f"Patient just said: \"{patient_text}\"\n\n"
        f"Safety point to gently ask about this turn: {focus}\n"
        f"Helpline to include (only if not '(no helpline this turn)'): {helpline}\n\n"
        "Write your reply now: one short reflection of what they said, then ONE gentle question about that safety point."
    )
    try:
        reply = await call_llm(prompt=prompt, system_prompt=_CRISIS_SYSTEM_PROMPT, max_tokens=120, think=False)
        reply = (reply or "").strip()
        reason = _crisis_reject_reason(reply)
        if reason:
            # Returning None -> caller sends _deterministic_message instead.
            logger.error(
                f"[CRISIS] REJECTED model reply — {reason}. stage={stage!r} "
                f"reply={reply!r} — sending deterministic reply instead"
            )
            return None
        return reply or None
    except Exception as e:
        logger.warning(f"[CRISIS] LLM phrasing failed: {e}")
        return None


# ── Main entry ─────────────────────────────────────────────────

async def handle_turn(
    session_key: str,
    patient_text: str,
    language: str = "en",
    recent_context: str = "",
) -> CrisisTurn:
    """Advance the crisis triage by one turn.

    Returns a CrisisTurn. When `stay_in_crisis` is False the caller should
    resume the normal interview (state already cleared here).
    """
    lang = language if language in ("en", "hi", "hinglish") else "en"
    assess = get_assessment(session_key)
    assess.turns += 1

    f = classify(patient_text)
    _update_risk(assess, f)

    logger.info(
        f"[CRISIS] session={session_key} turn={assess.turns} stage={assess.current_stage()} "
        f"risk={assess.risk} features={ {k: v for k, v in f.items() if v} }"
    )

    n = _norm(patient_text)
    fresh_risk = _fresh_risk(n, f)

    # 0. De-escalation handshake. This runs BEFORE the urgent branch on purpose:
    #    once risk hits high/imminent it never falls, and section 1 returns an
    #    urgent message every turn, so without an early exit here a patient
    #    repeating "I'm fine" is answered with rotating urgent prompts forever —
    #    the exact loop this block fixes. A genuine safety/denial signal (with no
    #    fresh risk cue this turn) is met with a request to confirm; the next
    #    message, if it confirms, resumes the interview.
    if assess.awaiting_confirmation:
        # Stricter gate here: a bare "yes" answers OUR "are you okay?" question,
        # so only an unambiguous intent phrase counts as fresh risk at this step.
        confirm_fresh_risk = _fresh_risk(n, f, explicit_only=True)
        if not confirm_fresh_risk and _has(n, _CONFIRM_SAFE):
            # Confirmed. Resume the interview — for EVERY session, including one
            # that earlier tripped the active-intent lock (product decision:
            # confirm-then-resume applies to all crisis sessions).
            assess.resolved = True
            alert = None
            if assess.interview_locked or assess.risk in ("high", "imminent"):
                findings_str = (
                    ", ".join(f"{k}: {v}" for k, v in assess.findings.items())
                    or "expressed suicidal ideation earlier this session"
                )
                alert = (
                    f"ℹ️ CRISIS DE-ESCALATED — session {session_key}. Patient "
                    f"confirmed safety after prior {assess.risk.upper()} risk"
                    f"{' (had active intent)' if assess.interview_locked else ''} "
                    f"and is resuming the interview. Findings: {findings_str}. "
                    f"Last message: \"{patient_text[:120]}\""
                )
                logger.error(
                    f"[CRISIS] de-escalated & resuming session={session_key} "
                    f"turn={assess.turns}"
                )
            clear(session_key)
            return CrisisTurn(
                _resume_bridge(lang),
                stay_in_crisis=False,
                clinician_alert=alert,
                risk=assess.risk,
            )
        # Not a clean confirmation (or a fresh risk cue appeared) — abandon the
        # handshake and let the normal triage / escalation paths below handle
        # this turn. A renewed risk signal re-escalates immediately.
        assess.awaiting_confirmation = False

    # Fresh safety/denial signal (not the entry turn) → ask them to confirm.
    deescalation_signal = (f["safe"] or f["denies_intent"]) and not fresh_risk
    if deescalation_signal and assess.turns >= 2:
        assess.awaiting_confirmation = True
        assess.helpline_given = True
        return CrisisTurn(_confirm_prompt(lang), stay_in_crisis=True, risk=assess.risk)

    # 1. Imminent / high risk → urgent hand-off, keep engaging, alert clinician.
    if assess.risk in ("imminent", "high"):
        assess.helpline_given = True
        # Alert on EVERY high/imminent turn, not only when the level first
        # rises. The dedup was written to avoid spamming the on-call clinician,
        # but at high/imminent that reasoning is inverted: a patient who is
        # still actively suicidal three turns later is not old news, and the
        # clinician has no way to tell "escalated once" from "still going" if
        # we go silent. The dedup still applies below high (see the triage
        # path), where passive ideation shouldn't page anyone repeatedly.
        order = {"": 0, "low": 1, "moderate": 2, "high": 3, "imminent": 4}
        if order.get(assess.risk, 0) > order.get(assess.alerted_at, 0):
            assess.alerted_at = assess.risk
        findings_str = ", ".join(f"{k}: {v}" for k, v in assess.findings.items()) or "expressed suicidal intent"
        alert = (
            f"⚠️ {assess.risk.upper()} SUICIDE RISK — session {session_key}. "
            f"Turn {assess.turns}"
            f"{' — INTERVIEW LOCKED (active intent)' if assess.interview_locked else ''}. "
            f"Findings: {findings_str}. Last message: \"{patient_text[:120]}\""
        )
        logger.error(f"[CRISIS] clinician alert ({assess.risk}) turn={assess.turns}")
        # Count urgent turns separately from crisis turns so the approved
        # hand-off text lands on the FIRST urgent turn, whenever it occurs.
        assess.urgent_turns += 1
        return CrisisTurn(
            _urgent_message(lang, assess.urgent_turns),
            stay_in_crisis=True,
            clinician_alert=alert,
            risk=assess.risk,
        )

    # 2. De-escalated and safe → offer to continue or pause (closure), then exit
    #    on the NEXT turn once they confirm. Here we present the choice.
    if _deescalated(assess, f) or assess.current_stage() == "closure":
        # Closure exists to hand control BACK to the patient. Its message asks
        # "continue, or take a break?" — so anything that isn't a fresh risk
        # signal is an answer to that question and must let them out.
        #
        # This used to require f["safe"] or f["denies_intent"], i.e. a phrase
        # from the _SAFE list. A patient answered "ready", then "continue" —
        # neither matches ("ready to continue" and "lets continue" are in the
        # list, the bare words are not) — so we re-sent the identical helpline
        # message on every turn and she could not get out of crisis mode at all.
        # Defaulting to "trapped" when we fail to parse a reply is the wrong way
        # round; acute risk is already handled above (section 1 returns early),
        # and any renewed intent/plan/immediacy re-escalates on the next turn.
        if assess.current_stage() == "closure":
            # Use _EXPLICIT_INTENT, not f["affirms_intent"]: the latter fires on
            # a bare "yes", which at closure is the answer to OUR "continue or
            # pause?" question, not a declaration of intent. If they DO restate
            # real risk here, the normal keyword detector re-enters crisis on the
            # very next message, so exiting on an ambiguous reply is recoverable
            # — leaving them stuck in a helpline loop is not.
            fresh_risk = (
                _has(_norm(patient_text), _EXPLICIT_INTENT)
                or f["plan"]
                or f["immediate"]
            )
            if fresh_risk:
                pass  # falls through to the closure message below
            elif assess.interview_locked:
                # They confirmed ACTIVE suicidal intent earlier in this session.
                # The interview is over — a later "I'm ok now" does not buy it
                # back. Stay with them, keep the resource in front of them, and
                # leave them for a clinician rather than resuming demographics.
                logger.error(
                    f"[CRISIS] session={session_key} interview LOCKED "
                    f"(active intent confirmed) — not resuming history-taking"
                )
                findings_str = (
                    ", ".join(f"{k}: {v}" for k, v in assess.findings.items())
                    or "confirmed active suicidal intent"
                )
                return CrisisTurn(
                    _locked_handoff_message(lang),
                    stay_in_crisis=True,
                    clinician_alert=(
                        f"⚠️ {assess.risk.upper()} SUICIDE RISK — session {session_key}. "
                        f"Turn {assess.turns} — INTERVIEW LOCKED (active intent); patient "
                        f"now reports feeling steadier but is NOT being returned to the "
                        f"interview. Findings: {findings_str}. "
                        f"Last message: \"{patient_text[:120]}\""
                    ),
                    risk=assess.risk,
                )
            else:
                assess.resolved = True
                clear(session_key)
                return CrisisTurn(_bridge_back(lang), stay_in_crisis=False, risk=assess.risk)
        assess.stage_idx = len(STAGES) - 1  # closure
        include_help = True
        assess.helpline_given = True
        return CrisisTurn(_closure_message(lang, safe=f["safe"]), stay_in_crisis=True, risk=assess.risk)

    # 3. Patient wants to change the subject mid-triage → acknowledge + soft check,
    #    do NOT replay the helpline verbatim.
    if f["deflect"]:
        if lang == "hi":
            msg = ("Bilkul, hum baat kar sakte hain. Bas pehle main ye jaanna chahta hoon — kya aap abhi surakshit "
                   "mehsoos kar rahe hain?")
        elif lang == "hinglish":
            msg = ("Bilkul, hum baat kar sakte hain. Bas pehle ye batao — kya aap abhi safe feel kar rahe ho?")
        else:
            msg = ("Of course, we can talk about whatever you'd like. Before we do, I just want to check — "
                   "are you feeling safe right now?")
        return CrisisTurn(msg, stay_in_crisis=True, risk=assess.risk)

    # 4. Normal brief triage: reflect + ask the current stage's question, then advance.
    stage = assess.current_stage()

    # Helpline: on the first crisis turn, and on EVERY turn the patient is
    # actively expressing intent. It used to appear once and then vanish, so a
    # patient who said "i wanna die" and then "i wanna act on it" got the
    # resource on turn 1 only — exactly as their intent was escalating. The
    # original rule existed to stop the old "resend the identical helpline
    # forever" bug, but that bug was about repeating the WHOLE message; the
    # reply body still changes every turn, only the resource line repeats.
    intent_live = f["affirms_intent"] or assess.risk in ("high", "imminent")
    include_help = (not assess.helpline_given) or intent_live

    reply = await _llm_reply(patient_text, stage, lang, include_help, recent_context)
    if not reply:
        reply = _deterministic_message(assess, lang, stage, include_help)
    if include_help and _norm(CRISIS_HELPLINE) not in _norm(reply):
        # Guarantee the resource is present even if the LLM omitted it.
        reply = f"{reply}\n\n{_helpline_line(lang)}"
    assess.helpline_given = assess.helpline_given or include_help

    # Clinician alert. This path previously emitted NO alert at all — alerting
    # lived only in the urgent branch (high/imminent). So a patient expressing
    # intent to die with no method disclosed scores "moderate" and reached a
    # clinician NEVER. Expressed intent must always be flagged, and must KEEP
    # being flagged while it stays live: a clinician needs to know it is
    # ongoing, not merely that it once began.
    alert = None
    order = {"": 0, "low": 1, "moderate": 2, "high": 3, "imminent": 4}
    escalated = order.get(assess.risk, 0) > order.get(assess.alerted_at, 0)
    if escalated or f["affirms_intent"]:
        if order.get(assess.risk, 0) > order.get(assess.alerted_at, 0):
            assess.alerted_at = assess.risk
        findings_str = (
            ", ".join(f"{k}: {v}" for k, v in assess.findings.items())
            or "expressed suicidal ideation"
        )
        alert = (
            f"⚠️ {assess.risk.upper()} SUICIDE RISK — session {session_key}. "
            f"Findings: {findings_str}. Stage: {stage}. "
            f"Last message: \"{patient_text[:120]}\""
        )
        logger.error(f"[CRISIS] clinician alert ({assess.risk}): {patient_text[:80]!r}")

    assess.advance()
    return CrisisTurn(reply, stay_in_crisis=True, clinician_alert=alert, risk=assess.risk)
