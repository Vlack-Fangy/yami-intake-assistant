"""
Clinical state machine for structured psychiatric interview progression.
Extracts facts from patient messages, tracks domain coverage, and
deterministically decides what to ask next.

Replaces LLM-based planning with a lightweight state machine so the
model only handles NLG (phrasing one question naturally) instead of
NLU + planning + generation in a single pass.

Lead extraction uses a two-pass approach:
  Pass 1 — keyword/pattern matching (synchronous, zero latency)
  Pass 2 — lightweight LLM call for safety + guilt + contradiction
            (async, catches paraphrased versions keywords miss)
"""

import os
import re
import heapq
import json
from typing import Optional


# ── Domain definitions ──────────────────────────────────────────
# Priority: lower number = more urgent. Max_turns: consecutive
# questions allowed before forced switch.
#
# Demographics (priority 0) is a special domain active only when
# the current section is "sociodemographic" — it keeps the LLM
# focused on background information until enough fields are filled
# to trigger a section transition.
DOMAIN_DEFS = [
    {"id": "demographics",         "label": "Background — family, living situation, socioeconomic status", "priority": 0,  "max_turns": 3, "section": "sociodemographic"},
    {"id": "safety",               "label": "Safety risk",                     "priority": 1,  "max_turns": 3},
    {"id": "symptom_detail",       "label": "Symptom details",                "priority": 2,  "max_turns": 1},
    {"id": "post_event_consequences", "label": "Impact on daily life",         "priority": 3,  "max_turns": 1},
    {"id": "functioning_daily",    "label": "How things are now",             "priority": 4,  "max_turns": 1},
    {"id": "mood_anhedonia",       "label": "Mood and interest",              "priority": 5,  "max_turns": 1},
    {"id": "sleep_appetite_energy","label": "Sleep, appetite, energy",        "priority": 6,  "max_turns": 1},
    {"id": "relationships_support","label": "Relationships and support",      "priority": 7,  "max_turns": 1},
    {"id": "presenting_complaint", "label": "Main concern",                   "priority": 8,  "max_turns": 1},
    {"id": "trigger",              "label": "Trigger",                        "priority": 9,  "max_turns": 1},
    {"id": "timeline",             "label": "Timeline",                       "priority": 10, "max_turns": 1},
]

DOMAIN_IDS = [d["id"] for d in DOMAIN_DEFS]


# ── Keyword extraction maps ─────────────────────────────────────
# Maps patient message keywords to clinical domains touched.
SYMPTOM_KEYWORDS = {
    "panic attack": "symptom_detail", "panic": "symptom_detail",
    "racing heart": "symptom_detail", "heart racing": "symptom_detail",
    "can't breathe": "symptom_detail", "couldn't breathe": "symptom_detail",
    "short of breath": "symptom_detail", "hyperventilating": "symptom_detail",
    "anxiety": "symptom_detail", "anxious": "symptom_detail",
    "fear": "symptom_detail", "scared": "symptom_detail",
    "worry": "symptom_detail", "worried": "symptom_detail",
    "nervous": "symptom_detail", "shaking": "symptom_detail",
    "trembling": "symptom_detail", "dizziness": "symptom_detail",
    "nausea": "symptom_detail",
    "cry": "mood_anhedonia", "crying": "mood_anhedonia",
    "cried": "mood_anhedonia", "tears": "mood_anhedonia",
    "sad": "mood_anhedonia", "depressed": "mood_anhedonia",
    "hopeless": "mood_anhedonia", "worthless": "mood_anhedonia",
    "nothing matters": "mood_anhedonia", "empty": "mood_anhedonia",
    "numb": "mood_anhedonia", "no interest": "mood_anhedonia",
    "don't care": "mood_anhedonia", "no motivation": "mood_anhedonia",
    "enjoy": "mood_anhedonia", "enjoyed": "mood_anhedonia",
    "happy": "mood_anhedonia", "pleasure": "mood_anhedonia",
    "sleep": "sleep_appetite_energy", "couldn't sleep": "sleep_appetite_energy",
    "insomnia": "sleep_appetite_energy", "waking up": "sleep_appetite_energy",
    "nightmare": "sleep_appetite_energy", "nightmares": "sleep_appetite_energy",
    "tired": "sleep_appetite_energy", "exhausted": "sleep_appetite_energy",
    "energy": "sleep_appetite_energy", "fatigue": "sleep_appetite_energy",
    "appetite": "sleep_appetite_energy", "eating": "sleep_appetite_energy",
    "ate": "sleep_appetite_energy", "food": "sleep_appetite_energy",
    "weight": "sleep_appetite_energy", "lost weight": "sleep_appetite_energy",
    "overeating": "sleep_appetite_energy",
    "quit": "post_event_consequences", "left": "post_event_consequences",
    "stopped": "post_event_consequences", "gave up": "post_event_consequences",
    "lost my job": "post_event_consequences", "resigned": "post_event_consequences",
    "dropped out": "post_event_consequences", "fired": "post_event_consequences",
    "function": "functioning_daily", "daily": "functioning_daily",
    "morning": "functioning_daily", "couldn't focus": "functioning_daily",
    "concentrate": "functioning_daily", "stay in bed": "functioning_daily",
    "couldn't get out": "functioning_daily", "routine": "functioning_daily",
    "shower": "functioning_daily", "hygiene": "functioning_daily",
    "chores": "functioning_daily",
    "friend": "relationships_support", "family": "relationships_support",
    "alone": "relationships_support", "isolated": "relationships_support",
    "nobody": "relationships_support", "support": "relationships_support",
    "brother": "relationships_support", "sister": "relationships_support",
    "mother": "relationships_support", "father": "relationships_support",
    "parent": "relationships_support", "parents": "relationships_support",
    "spouse": "relationships_support", "husband": "relationships_support",
    "wife": "relationships_support", "boyfriend": "relationships_support",
    "girlfriend": "relationships_support", "partner": "relationships_support",
    "started when": "timeline", "started after": "timeline",
    "began": "timeline", "months ago": "timeline", "years ago": "timeline",
    "weeks ago": "timeline", "since": "timeline",
    "trigger": "trigger", "because of": "trigger",
    "after that": "trigger", "it started": "trigger",
}

# High-priority lead patterns — these indicate clinically rich
# material that the planner should prioritise.
LEAD_PATTERNS = [
    (r"blame myself", "guilt"), (r"my fault", "guilt"),
    (r"should have", "guilt"), (r"could have", "guilt"),
    (r"if only", "guilt"), (r"feel guilty", "guilt"),
    (r"hate myself", "guilt"), (r"failed", "guilt"),
    (r"let them down", "guilt"), (r"disappointment", "guilt"),
    (r"regret", "guilt"), (r"mistake", "guilt"),
    (r"better off without me", "safety"),
    (r"want to die", "safety"), (r"end it", "safety"),
    (r"kill myself", "safety"), (r"hurt myself", "safety"),
    (r"suicide", "safety"), (r"suicidal", "safety"),
    (r"don't want to live", "safety"), (r"can't go on", "safety"),
    (r"no reason to live", "safety"), (r"end my life", "safety"),
    (r"won't be here", "safety"),
]

# Contradiction patterns — signal identity loss / major life shift.
CONTRADICTION_PATTERNS = [
    (r"loved\s+\w+\s+but", "contradiction"),
    (r"enjoyed\s+\w+\s+but", "contradiction"),
    (r"was\s+(so|really)\s+\w+\s+(and|but)\s+then", "contradiction"),
    (r"used to love", "contradiction"),
    (r"at first\s+\w+\s+but", "contradiction"),
    (r"initially\s+\w+\s+but", "contradiction"),
    (r"was\s+happy\s+(and|but)", "contradiction"),
]

LEAD_TO_DOMAIN = {
    # Guilt and contradiction route to "how this affects daily life" rather
    # than mood exploration — keeps the interview intake-oriented instead
    # of drifting into psychotherapy-style emotional reflection.
    "guilt": "post_event_consequences",
    "safety": "safety",
    "contradiction": "post_event_consequences",
}


# ── ClinicalState ──────────────────────────────────────────────

class ClinicalState:
    """Per-session state tracking clinical domain coverage.

    Usage:
        state = get_state(session_id)
        extraction = state.process_patient_message(text)
        domain, label, lead = state.decide_next(extraction, section="sociodemographic")
    """

    __slots__ = (
        "domains", "turn_count", "last_domain", "lead_queue",
        "_lead_counter", "_pending_llm_leads", "section_progress",
        "pending_transition", "no_complaint_turns", "orientation_sent",
    )

    def __init__(self):
        # Per-section progress tracking, used to break loops: how many turns
        # we've spent in a section and how many consecutive turns produced no
        # new filled field. {section: {"turns", "last_filled", "stall"}}
        self.section_progress: dict = {}
        self.domains = {
            d["id"]: {
                "status": "unexplored",  # unexplored | partial | explored
                "label": d["label"],
                "priority": d["priority"],
                "max_turns": d["max_turns"],
                "consecutive": 0,
                "total_asked": 0,
                "section": d.get("section"),  # None = always active, string = required section
            }
            for d in DOMAIN_DEFS
        }
        self.turn_count = 0
        self.last_domain: Optional[str] = None
        # Set to (from_section, to_section) by transitions._advance_to_next_section
        # when a section changes; consumed and cleared by the reply path so the
        # bridge is folded into the next single reply instead of a 2nd message.
        self.pending_transition: Optional[tuple] = None
        # Consecutive turns in chief_complaints with no complaint disclosed.
        # A patient said "nothing", "i am alright" and was asked "what brings
        # you here today?" four times, then "can you tell me when this feeling
        # of being okay first started?" — the schema needs a complaint, so it
        # invented one to attach a duration to. After a few of these we explain
        # ourselves and offer a break instead of asking again.
        self.no_complaint_turns: int = 0
        self.orientation_sent: bool = False
        self.lead_queue: list = []        # heap of (-priority, counter, lead_dict)
        self._lead_counter = 0
        self._pending_llm_leads: list[dict] = []  # LLM leads from PREVIOUS turn

    # ── Extraction ──────────────────────────────────────────

    def process_patient_message(self, text: str) -> dict:
        """Analyse patient message and update internal state.

        Returns an extraction result dict consumed by decide_next().
        """
        text_lower = text.lower()
        domains_touched: set[str] = set()
        leads: list[dict] = []

        # 1. Keyword symptom detection
        for keyword, domain in SYMPTOM_KEYWORDS.items():
            if keyword in text_lower:
                domains_touched.add(domain)

        # 2. Lead patterns (guilt, safety)
        for pattern, lead_type in LEAD_PATTERNS:
            if re.search(pattern, text_lower):
                priority = 1 if lead_type == "safety" else 2
                leads.append({"type": lead_type, "priority": priority})

        # 3. Contradiction patterns
        for pattern, _ in CONTRADICTION_PATTERNS:
            if re.search(pattern, text_lower):
                leads.append({"type": "contradiction", "priority": 2})

        safety_triggers = any(l["type"] == "safety" for l in leads)

        # Update domain statuses
        for domain in domains_touched:
            d = self.domains[domain]
            if d["status"] == "unexplored":
                d["status"] = "partial"

        # Push leads into priority queue
        for lead in leads:
            heapq.heappush(
                self.lead_queue,
                (-lead["priority"], self._lead_counter, lead),
            )
            self._lead_counter += 1

        # Include any LLM leads from the previous turn's extraction
        # Push them into the heap so decide_next() actually reads them.
        for llm_lead in self._pending_llm_leads:
            heapq.heappush(
                self.lead_queue,
                (-llm_lead["priority"], self._lead_counter, llm_lead),
            )
            self._lead_counter += 1
        llm_count = len(self._pending_llm_leads)
        all_leads = leads + self._pending_llm_leads
        # Clear consumed pending leads so they aren't replayed next turn
        self._pending_llm_leads = []

        return {
            "domains_touched": sorted(domains_touched),
            "leads": all_leads,
            "safety_triggers": safety_triggers or any(
                l["type"] == "safety" for l in all_leads
            ),
            "_keyword_lead_count": len(leads),
            "_llm_lead_count": llm_count,
        }

    # ── Planning ────────────────────────────────────────────

    def decide_next(self, extraction: dict, section: str = "") -> tuple[str, Optional[str], Optional[str]]:
        """Decide what domain to target next.

        `section` filters domains: when "sociodemographic", only the
        demographics domain is active. Other sections exclude demographics
        and use the full clinical domain set.

        Returns (domain_id, domain_label, specific_lead_text_or_None).
        domain_id == "" means all domains explored — let LLM behave normally.
        """
        self.turn_count += 1
        _section = section

        # 1. Safety override.
        #
        #    A genuine safety lead — the patient actually said something —
        #    fires in ANY section, immediately. That must never wait on a
        #    counter and is deliberately left unconditional.
        #
        #    The turn-count CHECKPOINT is only a backstop so safety gets asked
        #    eventually. It must NOT fire during sociodemographic: on turn ~10
        #    of pure demographics it produced
        #        "And what's your gender?"
        #        -> "have you experienced any thoughts of self-harm or suicide?"
        #    sandwiched between income and marital status. That is clinically
        #    wrong and alarming to a patient who has not even named a complaint
        #    yet. Suicidality belongs in the clinical sections — `hopi` already
        #    carries a suicidality_assessed field — so the checkpoint waits
        #    until we are out of demographics.
        if self.domains["safety"]["status"] != "explored":
            has_safety_lead = bool(extraction.get("safety_triggers"))
            checkpoint_due = (
                self.turn_count >= SAFETY_CHECKPOINT_TURN
                and section not in NO_SAFETY_CHECKPOINT_SECTIONS
            )
            if has_safety_lead or checkpoint_due:
                return self._target("safety")

        # 2. Force transition check — same domain hit its max consecutive turns
        force_switch = (
            self.last_domain is not None
            and self.domains[self.last_domain]["consecutive"]
            >= self.domains[self.last_domain]["max_turns"]
        )

        if force_switch:
            # Lead from a DIFFERENT domain is ok during force switch
            popped = self._pop_lead()
            if popped is not None:
                domain = LEAD_TO_DOMAIN.get(popped["type"])
                if domain is not None and domain != self.last_domain and self.domains[domain]["status"] != "explored":
                    if domain != "safety":  # safety only via override below
                        return self._target(domain, popped["type"] + " mentioned")
            # Otherwise pick next domain
            next_d = self._pick_next_domain(section=_section)
            if next_d is not None:
                return self._target(next_d)
            # No other domain available — stay (edge case, shouldn't happen)
            return ("", None, None)

        # 3. Saturation check — if the patient's latest answer only touched the
        # current domain (no new domains introduced), mark it saturated and move
        # on. This prevents squeezing a domain that's already well covered.
        touched = extraction.get("domains_touched", [])
        if (
            self.last_domain is not None
            and len(touched) == 1
            and touched[0] == self.last_domain
            and self.last_domain != "safety"
        ):
            self.domains[self.last_domain]["status"] = "explored"

        # 4. Lead queue (normal, non-forced turn) — safety leads only via override
        popped = self._pop_lead()
        if popped is not None:
            domain = LEAD_TO_DOMAIN.get(popped["type"])
            if domain is not None and self.domains[domain]["status"] != "explored":
                if popped["type"] != "safety":  # safety only via override
                    return self._target(domain, popped["type"] + " mentioned")

        # 5. Stay on current domain if still within limit
        if self.last_domain is not None and self.domains[self.last_domain]["status"] != "explored":
            if self.last_domain != "safety" and self.domains[self.last_domain]["consecutive"] < self.domains[self.last_domain]["max_turns"]:
                return self._target(self.last_domain)

        # 6. Pick best new domain
        next_d = self._pick_next_domain(section=_section)
        if next_d is not None:
            return self._target(next_d)

        # 7. Everything explored
        return ("", None, None)

    # ── Internals ───────────────────────────────────────────

    def _pick_next_domain(self, section: str = "") -> Optional[str]:
        """Highest-priority domain that isn't explored and isn't the current one.

        Safety is excluded from normal selection — it is handled
        exclusively by the safety override in decide_next() so it
        only fires at the right checkpoint (turn >= 10 or triggers).

        Domains with a `section` restriction are only available when
        the current section matches. Domains without a restriction
        are always available (except in sociodemographic, where only
        demographics is active).
        """
        candidates = [
            d_id
            for d_id in DOMAIN_IDS
            if d_id != "safety"
            and self.domains[d_id]["status"] != "explored"
            and d_id != self.last_domain
            and self._domain_active(d_id, section)
        ]
        # Sort: priority asc, then partial before unexplored
        def sort_key(d_id: str):
            d = self.domains[d_id]
            return (d["priority"], 0 if d["status"] == "partial" else 1)
        candidates.sort(key=sort_key)
        return candidates[0] if candidates else None

    @staticmethod
    def _domain_active(domain_id: str, section: str) -> bool:
        """Check if a domain is active for the given section."""
        # demographics only active during the sociodemographic section
        if domain_id == "demographics":
            return section == "sociodemographic"
        # clinical domains inactive during sociodemographic
        if section == "sociodemographic":
            return False
        return True

    def _target(self, domain_id: str, lead_text: Optional[str] = None) -> tuple[str, Optional[str], Optional[str]]:
        """Mark a domain as targeted and return the decision tuple."""
        # Reset consecutive counter for previous domain
        if self.last_domain is not None and self.last_domain != domain_id:
            self.domains[self.last_domain]["consecutive"] = 0

        d = self.domains[domain_id]
        d["consecutive"] += 1
        d["total_asked"] += 1
        if d["consecutive"] >= d["max_turns"]:
            d["status"] = "explored"
        self.last_domain = domain_id
        return (domain_id, d["label"], lead_text)

    def _pop_lead(self) -> Optional[dict]:
        """Pop highest-priority lead from queue."""
        if not self.lead_queue:
            return None
        _, _, lead = heapq.heappop(self.lead_queue)
        return lead

    # ── Helpers ─────────────────────────────────────────────

    def mark_explored(self, domain_id: str) -> None:
        """Manually mark a domain as explored (e.g. from filled fields)."""
        if domain_id in self.domains:
            self.domains[domain_id]["status"] = "explored"

    def summary(self) -> str:
        """One-line status of all domains (for logging / debugging)."""
        parts = []
        for d_id in DOMAIN_IDS:
            d = self.domains[d_id]
            icon = "[X]" if d["status"] == "explored" else "[~]" if d["status"] == "partial" else "[ ]"
            parts.append(f"{icon} {d['label']}")
        return " | ".join(parts)

    def store_llm_leads(self, leads: list[dict]) -> None:
        """Store LLM extraction results for the NEXT turn's planner."""
        self._pending_llm_leads = leads

    def record_section_progress(self, section: str, filled_count: int) -> int:
        """Track progress within a section and return the current stall count.

        `stall` is the number of consecutive turns in this section that added no
        new filled field. The caller uses it to force a graceful section advance
        so the interview never loops forever on an unanswerable / refused field.
        """
        sp = self.section_progress.get(section)
        if sp is None:
            sp = {"turns": 0, "last_filled": filled_count, "stall": 0}
            self.section_progress[section] = sp
        sp["turns"] += 1
        if filled_count > sp["last_filled"]:
            sp["last_filled"] = filled_count
            sp["stall"] = 0
        else:
            sp["stall"] += 1
        return sp["stall"]

    def reset(self) -> None:
        """Reset to initial state (used on session restart)."""
        self.__init__()


# ── Global state registry ──────────────────────────────────────

# MSE items that require in-person OBSERVATION and cannot be self-reported over
# a text chat. Asking "can I ask about your general appearance and behaviour?"
# is meaningless — the patient cannot observe herself, and whatever she types
# then becomes an MSE finding in the chart. This is how real reports asserted
# "the patient's affect and cognition appear intact" about someone nobody had
# ever seen. These must be examined in person; the bot must never ask them and
# never fill them.
#
# Deliberately NOT included (these genuinely are self-reportable or testable
# over text, and matter): mood_subjective, thought_content, thought_possession,
# perception (hallucinations), orientation, memory, attention_concentration,
# general_knowledge, abstract_ability, judgment, insight.
NOT_ASSESSABLE_OVER_TEXT = frozenset({
    "general_appearance_behaviour",
    "psychomotor_activity",
    "speech_characteristics",
    "mood_objective",
    "affect_quality",
    "affect_intensity",
    "affect_mobility",
    "thought_stream",
    "thought_form",
    "consciousness",
    "intelligence",
})

# After this many consecutive turns in chief_complaints with nothing disclosed,
# stop re-asking "what brings you here today?" and explain ourselves instead.
NO_COMPLAINT_ORIENT_AFTER = int(os.getenv("NO_COMPLAINT_ORIENT_AFTER", "3"))

# Sent ONCE, verbatim — deliberately not LLM-generated. It states what we are
# and where the information goes, and a model must never improvise those. Note
# it does NOT promise confidentiality beyond what is true: the compiled history
# is sent to the clinician (CLINICIAN_TELEGRAM_ID), so we say exactly that.
#
# It has to serve three different people at once, without accusing any of them:
# someone testing the bot, someone who genuinely has nothing wrong, and someone
# who has something but can't say it yet.
ORIENTATION_MESSAGE = {
    "en": (
        "Before we go further, let me be clear about what this is.\n\n"
        "I'm an assistant that takes a mental health history before you see a "
        "clinician. What you tell me is compiled into a history for the "
        "clinician who'll be seeing you — it isn't shared more widely than that.\n\n"
        "If nothing is troubling you right now, that's completely fine — you "
        "don't need to find something for my sake. And if there is something "
        "but it doesn't feel easy to put into words yet, that's alright too; we "
        "don't have to do this today.\n\n"
        "You can type /pause at any point and pick this up whenever you're "
        "ready. Otherwise, just tell me whatever's on your mind and we'll go "
        "from there."
    ),
    "hi": (
        "Isse pehle ki hum aage badhein, main saaf kar doon ki yeh kya hai.\n\n"
        "Main ek assistant hoon jo clinician se milne se pehle aapki mental "
        "health history leta hai. Aap jo batayenge woh us clinician ke liye "
        "history mein jaata hai jo aapko dekhenge — usse aage share nahi hota.\n\n"
        "Agar abhi aapko kuch pareshan nahi kar raha, toh bilkul theek hai — "
        "mere liye kuch dhoondhne ki zaroorat nahi. Aur agar kuch hai lekin "
        "abhi kehna aasaan nahi lag raha, woh bhi theek hai; yeh aaj hi karna "
        "zaroori nahi.\n\n"
        "Aap kabhi bhi /pause type kar sakte hain aur jab taiyaar hon tab jaari "
        "rakh sakte hain."
    ),
}
ORIENTATION_MESSAGE["hinglish"] = ORIENTATION_MESSAGE["hi"]

# Turn at which the safety domain is asked as a backstop if it hasn't come up
# naturally. A real safety lead always fires immediately regardless of this.
SAFETY_CHECKPOINT_TURN = int(os.getenv("SAFETY_CHECKPOINT_TURN", "10"))

# Sections where the blind turn-count safety checkpoint must never fire.
# Asking about suicide between "what's your gender?" and "what's your income?"
# is clinically wrong and alarming. A genuine safety lead still fires anywhere.
NO_SAFETY_CHECKPOINT_SECTIONS = {"sociodemographic"}

_STATES: dict[str, ClinicalState] = {}


def get_state(session_key: str) -> ClinicalState:
    """Get or create a ClinicalState for the given session key."""
    if session_key not in _STATES:
        _STATES[session_key] = ClinicalState()
    return _STATES[session_key]


def clear_state(session_key: str) -> None:
    """Remove a session's clinical state (on session end / reset)."""
    _STATES.pop(session_key, None)


# ── LLM-based lead extraction ──────────────────────────────────
# Falls back to keyword results on failure — always safe to call.

# Rolling failure counters for observability
_LLM_EXTRACTION_CALLS = 0
_LLM_EXTRACTION_FAILURES = 0


def llm_extraction_failure_rate() -> float:
    """Fraction of LLM extraction calls that failed (0.0–1.0)."""
    total = _LLM_EXTRACTION_CALLS
    return _LLM_EXTRACTION_FAILURES / total if total else 0.0


_LLM_EXTRACTION_SYSTEM_PROMPT = """You are a clinical text analyzer.
Patient message. Return ONLY valid JSON:

{
  "safety": {"present": false, "quote": ""},
  "guilt": {"present": false, "quote": ""},
  "contradiction": {"present": false, "quote": ""}
}

Set present=true + the exact triggering phrase only if the message clearly contains one of these.
Be conservative — if unsure, present=false."""


async def async_extract_leads(text: str) -> list[dict]:
    """Lightweight LLM pass for safety, guilt, and contradiction leads.

    Catches paraphrased versions keywords miss.
    Returns a list of lead dicts (same format as keyword extraction),
    empty list on failure or no findings.
    """
    global _LLM_EXTRACTION_CALLS, _LLM_EXTRACTION_FAILURES
    from llm import call_llm

    # Truncate very long messages to avoid wasting tokens; the LLM only
    # needs to classify, not understand every word.
    truncated = text[:1500]
    prompt = f"Patient message: \"{truncated}\""
    _LLM_EXTRACTION_CALLS += 1
    try:
        raw = await call_llm(
            prompt=prompt,
            system_prompt=_LLM_EXTRACTION_SYSTEM_PROMPT,
            max_tokens=120,
            think=False,
        )
    except Exception:
        _LLM_EXTRACTION_FAILURES += 1
        return []

    try:
        # Extract JSON from the response (handle markdown-wrapped output)
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            # Remove markdown code fences
            cleaned = cleaned.split("\n", 1)[-1]
            cleaned = cleaned.rsplit("```", 1)[0]
        parsed = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        _LLM_EXTRACTION_FAILURES += 1
        return []

    leads: list[dict] = []
    for lead_type in ("safety", "guilt", "contradiction"):
        entry = parsed.get(lead_type, {})
        if entry.get("present"):
            priority = 1 if lead_type == "safety" else 2
            leads.append({"type": lead_type, "priority": priority})

    return leads
