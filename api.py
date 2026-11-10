"""
Yami REST API
FastAPI app for psychiatric history-taking bot.
Replaces Telegram bot interface with HTTP endpoints.
"""
from dotenv import load_dotenv
load_dotenv()

from utils import get_logger
logger = get_logger("api")

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
from collections import defaultdict
import asyncio
import hashlib
import hmac
import os

import db
import llm
from data_model import SECTION_CONFIG, SECTION_CONVERSATION_TOPICS, SECTIONS_ORDER
from utils import (
    build_topic_based_hint,
    build_filled_fields_summary,
    get_next_topic_after,
    get_next_natural_question,
    detect_language,
    detect_dominant_language,
    is_confidently_english,
    SWITCHABLE_LANGUAGES,
    get_optimistic_filled_fields,
    mark_topic_as_filled,
    clear_filled_fields_cache,
    get_prev_section,
    set_prev_section,
)
from translation import WRAP_LANGUAGES, translate_to_english, translate_from_english
from transitions import check_section_transition
from progress import (
    compute_completion_percentage,
    check_milestone_and_update,
    update_progress_message,
    build_progress_summary,
)
from report import generate_report, generate_chat_transcript
import safety
import clinical_state
import crisis
import stt
import turn_pipeline
from rest_adapter import RestContext


# ── App ───────────────────────────────────────────────────────

app = FastAPI(
    title="Yami API",
    version="1.0.0",
    root_path=os.getenv("API_ROOT_PATH", ""),
)


# Require a deployment-specific key; there is no shared public fallback.
API_KEY = os.getenv("API_KEY", "").strip()

# Paths that must stay reachable without the key.
#   /health  — monitoring; exposes three booleans and no patient data.
#   docs     — the schema is not the secret; the key is.
_OPEN_PATHS = ("/health", "/docs", "/redoc", "/openapi.json")


@app.middleware("http")
async def require_api_key(request, call_next):
    """Gate every request on the shared key.

    Middleware rather than a per-route dependency deliberately: it covers routes
    added later without anyone remembering to annotate them. A forgotten
    dependency on a new endpoint is exactly the kind of hole that goes unnoticed.
    """
    path = request.url.path
    if request.method == "OPTIONS" or path in _OPEN_PATHS:
        return await call_next(request)
    if not API_KEY:
        return JSONResponse(status_code=503, content={"detail": "API_KEY is not configured."})

    auth = request.headers.get("authorization", "")
    scheme, _, token = auth.partition(" ")
    # compare_digest, not ==, so a wrong key cannot be discovered a character at
    # a time by timing the response.
    if scheme.lower() != "bearer" or not hmac.compare_digest(token.strip(), API_KEY):
        logger.warning(
            f"[AUTH] rejected {request.method} {path} from "
            f"{request.client.host if request.client else 'unknown'}"
        )
        return JSONResponse(
            status_code=401,
            content={"detail": "Missing or invalid API key."},
            headers={"WWW-Authenticate": "Bearer"},
        )

    return await call_next(request)


EXTRACTION_LOCKS: dict = defaultdict(asyncio.Lock)
RESPONSE_LOCKS:   dict = defaultdict(asyncio.Lock)


# ── Pydantic models ───────────────────────────────────────────

class CreateSessionRequest(BaseModel):
    user_id: str


class MessageRequest(BaseModel):
    message: str


class MCQOption(BaseModel):
    value: str
    label: str


class BotResponse(BaseModel):
    bot_message: str
    input_type: str          # "text" | "mcq"
    options: Optional[List[MCQOption]] = None
    section: Optional[str] = None
    session_status: Optional[str] = None
    # Handle for this reply, so a client can attach feedback to the exact
    # message it is rendering without first re-reading /history. Null when the
    # reply was not persisted (the language-selection prompt, for instance).
    message_id: Optional[int] = None
    # Suicide-risk level while a crisis triage is open: low | moderate |
    # high | imminent. Null otherwise. Lets a client keep the helpline in
    # front of the patient at high risk instead of inferring everything
    # from session_status alone.
    risk: Optional[str] = None


class FeedbackRequest(BaseModel):
    rating: str                          # "up" | "down"
    review: Optional[str] = None         # optional — a thumb alone is valid


class FeedbackUpdate(BaseModel):
    """Both optional: change the verdict, add the words later, or do both."""
    rating: Optional[str] = None
    review: Optional[str] = None


class FeedbackResponse(BaseModel):
    review_id: int
    session_id: str
    message_id: int
    rating: str
    review: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class TranscriptResponse(BaseModel):
    """Dictation result. No session state changes when this is returned."""
    transcript: str
    language: Optional[str] = None


class VoiceResponse(BotResponse):
    """A normal turn plus what we heard.

    The transcript is returned so the caller can show it back — a patient who
    sees "I heard: …" can correct a misheard symptom before it is recorded as
    clinical fact.
    """
    transcript: str


class SessionCreatedResponse(BaseModel):
    session_id: str
    bot_message: str
    input_type: str
    options: Optional[List[MCQOption]] = None


# ── Constants ─────────────────────────────────────────────────

LANGUAGE_OPTIONS = [
    MCQOption(value="en",        label="English"),
    MCQOption(value="hi",        label="Hindi"),
    MCQOption(value="hinglish",  label="Hinglish"),
    MCQOption(value="ta",        label="Tamil"),
    MCQOption(value="pa",        label="Punjabi"),
    MCQOption(value="bn",        label="Bengali"),
]

VALID_LANGUAGES = {"en", "hi", "hinglish", "ta", "pa", "bn"}

WELCOME_MESSAGE = (
    "Hello! 👋 I'm here to help your doctor understand you better before your appointment.\n\n"
    "I'll ask you some questions about yourself and how you've been feeling. "
    "There are no right or wrong answers — just share whatever feels comfortable.\n\n"
    "This will take about 60–75 minutes. You can pause anytime.\n\n"
    "First, which language would you prefer?"
)

FIRST_QUESTION = {
    "en":        "Thank you! Let's get started.\n\nTo begin with, could you tell me your name?",
    "hinglish":  "Thank you! Chaliye shuru karte hain.\n\nSabse pehle, kya aap apna naam bata sakte hain?",
    # No "hi" entry on purpose — falls through to "en" below via .get()'s
    # default, then gets machine-translated into real Devanagari by the
    # wrap mechanism. The old "hi" entry here was Romanized text ("Dhanyavaad!
    # Chaliye..."), not English — translate_from_english on that would have
    # treated Romanized Hindi as if it were English source text and produced
    # garbage. ta/pa/bn never had their own entries either; this just makes
    # hi consistent with how they were already handled.
}


# ── Safety response constants (REST-specific) ─────────────────

CRISIS_MESSAGE = (
    "I hear that you're going through something very difficult right now. 💙\n\n"
    "Please know that you are not alone, and support is available right now.\n\n"
    "🆘 Crisis Helpline: {helpline}\n\n"
    "Please reach out to them — they are trained to help and available 24/7.\n\n"
    "Once you feel safe and ready, we can continue at your own pace."
)

TRAUMA_CARE_MESSAGE = (
    "Thank you for trusting me with that. I recognize that sharing this takes courage. 💙\n\n"
    "Please know that what happened was not your fault. Your feelings are valid.\n\n"
    "If you ever feel overwhelmed, it's okay to take a break. You can pause anytime.\n\n"
)

# _handle_crisis_response lived here. It answered every crisis message with
# one canned string that never assessed risk, never advanced, and never
# reached a clinician. crisis.py's staged triage replaces it for both
# transports; see turn_pipeline.run_crisis_turn.

# ── Core extraction (background task) ─────────────────────────

# Extraction and lead-classification used to be duplicated here. Both now
# live in turn_pipeline, which is what the Telegram path has always run, so
# the two transports can no longer disagree about what a turn extracted.


# ── Core turn processing ──────────────────────────────────────

async def _process_turn(
    session_id: str,
    patient_text: str,
    background_tasks: BackgroundTasks,
) -> BotResponse:
    """
    Core turn: fetch context → translate in (ta/pa/bn only) → safety checks
    → call LLM → translate out (ta/pa/bn only) → save → fire background extraction.
    """
    session = await db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # ── Inter-sentence language switch ──
    # If the patient's CURRENT message is dominantly in one of the 4
    # script-distinguishable languages and it differs from the session's
    # active language, switch the session over — same persisted field used
    # at initial selection, just updated mid-conversation. This is
    # inter-sentence only: one message fully in language X moves the whole
    # session to X. No intra-sentence mixing support — a sentence that
    # mixes two languages doesn't get partially processed in each.
    #
    # Switching INTO en is handled separately below, by word content rather
    # than script (is_confidently_english) — script can't tell English
    # apart from Hinglish, both Latin. Switching INTO Hinglish specifically
    # isn't attempted: no script-based or word-based signal reliably
    # separates "this is Hinglish" from "this is English" or "this is Hindi
    # in Roman letters" — genuinely ambiguous, left alone, known gap.
    dominant_script = detect_dominant_language(patient_text)
    new_lang = None

    if dominant_script in SWITCHABLE_LANGUAGES and dominant_script != session.preferred_language:
        new_lang = dominant_script
    elif (
        dominant_script == "en"
        and session.preferred_language != "en"
        and is_confidently_english(patient_text)
    ):
        new_lang = "en"

    if new_lang:
        logger.info(
            f"[TURN] language switch for session={session_id}: "
            f"{session.preferred_language} -> {new_lang}"
        )
        await db.set_preferred_language(session_id, new_lang)
        session.preferred_language = new_lang  # keep in-memory session in sync for the rest of this turn

    # ── Language boundary: incoming ──
    # ta/pa/bn/hi live-translate. en/hinglish pass straight through, exactly
    # like before this feature existed.
    #
    # This MUST happen before the safety checks below — CRISIS_KEYWORDS and
    # TRAUMA_KEYWORDS are English/Hindi-transliteration strings, so a crisis
    # message typed in Devanagari/Tamil/Punjabi/Bengali script would never
    # match if we checked the raw text. Translating first is what makes
    # detection work for these languages at all. (Romanized Hindi typed
    # while session language is "hi" still falls through untranslated below
    # and still gets caught by the transliteration entries directly, same
    # as before hi moved into this group — see translation.py's docstring.)
    is_wrap = session.preferred_language in WRAP_LANGUAGES
    original_patient_text = patient_text

    if is_wrap:
        detected_script = detect_language(patient_text)
        if detected_script == session.preferred_language:
            patient_text = await translate_to_english(patient_text, source_lang=session.preferred_language)
        # else: message has no target-script characters (bare digits, a name
        # typed in Latin script, etc) — nothing to translate, skip the call.

    # Internal pipeline only ever sees en / hi / hinglish. Wrap-language
    # sessions present as 'en' here — llm.py never needs to know ta/pa/bn
    # exist, translation happens at the edges only.
    llm_language = "en" if is_wrap else session.preferred_language

    # ── Crisis triage ──
    # crisis.py's staged safety check, the same one Telegram runs. It replaces a
    # single canned string that never advanced, never assessed risk and never
    # reached a clinician. Entry mirrors handlers: a keyword hit OR an already
    # open triage.
    #
    # THIS MUST COME BEFORE THE SOFT-STOP CHECK. The two keyword sets overlap:
    # SOFT_STOP_PHRASES contains "i want to end" (meaning "end the session") and
    # CRISIS_KEYWORDS contains "end my life". "I want to end my life" matches
    # BOTH. Checked in the other order — which is what this file did — the
    # sentence was read as a request to pause, the session was paused, and a
    # patient disclosing suicidal intent got "No problem. Your progress is
    # saved." with no triage and no clinician alert. handlers.py:483 has always
    # had this order; only the REST path had it inverted.
    context = RestContext(session_id)

    if safety.is_crisis_message(patient_text) or crisis.in_crisis(str(session_id)):
        await db.set_session_status(session_id, "crisis")
        stay, risk = await turn_pipeline.run_crisis_turn(
            chat_id=session_id,
            patient_text=patient_text,
            session=session,
            context=context,
        )
        if not stay:
            # Safety confirmed. Telegram asks the next interview question in the
            # same turn; do the same here so the patient is not left on a bridge
            # message with nothing to answer.
            await db.set_session_status(session_id, "active")
            session = await db.get_session(session_id)
            try:
                await turn_pipeline.run_turn(
                    chat_id=session_id, patient_text=patient_text,
                    session=session, context=context, llm_language=llm_language,
                )
            except Exception as e:
                logger.error(f"[TURN] post-crisis resume failed: {e}")

        await _record_alerts(session_id, context, risk)
        return await _reply_from(session_id, session, context, is_wrap,
                                 risk=risk,
                                 status="crisis" if stay else "active")

    # ── Soft stop ──
    # Only reached once crisis has been ruled out above, so "I want to end my
    # life" can never be mistaken for "I want to end this session".
    if safety.is_soft_stop_message(patient_text):
        await db.set_session_status(session_id, "paused")
        logger.info(f"[TURN] soft stop detected for session={session_id}")
        return BotResponse(
            bot_message="No problem. Your progress is saved. 💾\n\nSend any message whenever you're ready to continue.",
            input_type="text",
            section=session.current_section,
            session_status="paused",
        )

    # ── Normal turn ──
    # One call. Everything the interview does — planner, clinical state machine,
    # section transitions, orientation, extraction — lives in turn_pipeline and is
    # shared with Telegram, so the two can no longer drift.
    try:
        await turn_pipeline.run_turn(
            chat_id=session_id,
            patient_text=patient_text,
            session=session,
            context=context,
            llm_language=llm_language,
        )
    except Exception as e:
        logger.error(f"[TURN] pipeline failed for session={session_id}: {e}")
        raise HTTPException(status_code=503, detail="The assistant had a technical issue. Please send that again.")

    await _record_alerts(session_id, context, None)
    return await _reply_from(session_id, session, context, is_wrap)


# ── Reading the turn back off the connector ───────────────────

async def _record_alerts(session_id: str, context, risk) -> None:
    """Persist anything the pipeline addressed to the clinician.

    On Telegram these are DMs. Over REST there is no chat, so they land in
    clinician_alerts and are served from /admin/alerts. Never let a storage
    failure break the patient's turn — the reply matters more than the record,
    and the alert is already in the log either way.
    """
    for text in context.alerts:
        try:
            await db.save_clinician_alert(session_id, text, risk)
        except Exception as e:
            logger.error(f"[ALERT] failed to store for session={session_id}: {e}")


async def _reply_from(session_id, session, context, is_wrap, risk=None, status=None) -> BotResponse:
    """Turn what the pipeline sent into one HTTP response.

    A turn can produce several messages — the crisis bridge then the next
    question, or a section-closing message — because that is how it behaves on
    Telegram. They are joined in order rather than dropped: reading the bridge
    before the question is the whole point of sending both.
    """
    reply = context.joined_reply()
    if not reply:
        # The pipeline finished without addressing the patient. Better a plain
        # nudge than an empty bubble.
        reply = "Sorry — could you say that again?"

    display = (await translate_from_english(reply, session.preferred_language)) if is_wrap else reply

    fresh = await db.get_session(session_id) or session
    last_id = None
    try:
        rows = await db.get_full_conversation_history(session_id)
        for m in reversed(rows):
            if m["role"] == "assistant":
                last_id = m["message_id"]
                break
    except Exception:
        pass

    return BotResponse(
        bot_message=display,
        input_type="text",
        section=fresh.current_section,
        session_status=status or fresh.status,
        message_id=last_id,
        risk=risk if risk and risk != "unknown" else None,
    )



def _reset_runtime_state(session_id: str) -> None:
    """Drop the in-memory state that outlives the database row.

    clinical_state and crisis both keep per-session dicts at module level. The
    Telegram path has always cleared them (handlers._reset_runtime_chat_state);
    the REST path cleared only clinical_state, so crisis.in_crisis() stayed True
    for an id that had been deleted. Recreating a session with the same id then
    routed every message straight into a triage that no longer had any history
    behind it — the interview could not start at all.
    """
    clinical_state.clear_state(str(session_id))
    crisis.clear(str(session_id))
    clear_filled_fields_cache(session_id)


# ── Startup ───────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    await db.init_db()
    import os
    if os.getenv("SKIP_MODEL_WARMUP", "0") == "1":
        logger.info("[STARTUP] Skipping model warmup (SKIP_MODEL_WARMUP=1)")
        return
    try:
        await llm.warmup_model()
    except Exception as e:
        logger.warning(f"[STARTUP] Model warmup failed: {e}")
        logger.warning("[STARTUP] API started without model — LLM endpoints will fail until model loads")


# ── Endpoints ─────────────────────────────────────────────────

@app.post("/sessions", response_model=SessionCreatedResponse, status_code=201)
async def create_session(request: CreateSessionRequest):
    """
    Create or reset a session for a user.
    Returns welcome message with language selection MCQ.
    """
    await db.create_session(request.user_id)
    await db.set_preferred_language(request.user_id, None)  # NULL = pending language selection
    _reset_runtime_state(request.user_id)
    return SessionCreatedResponse(
        session_id=request.user_id,
        bot_message=WELCOME_MESSAGE,
        input_type="mcq",
        options=LANGUAGE_OPTIONS,
    )


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session info and current state."""
    session = await db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": session.session_id,
        "status": session.status,
        "current_section": session.current_section,
        "preferred_language": session.preferred_language,
        "started_at": session.started_at,
        "last_active": session.last_active,
    }


@app.post("/sessions/{session_id}/messages", response_model=BotResponse)
async def send_message(
    session_id: str,
    request: MessageRequest,
    background_tasks: BackgroundTasks,
):
    """
    Send a patient message.
    Handles language selection on first message, then normal turns.
    """
    session = await db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Crisis sessions fall through to _process_turn, which routes them into
    # crisis.py's triage. There is no separate canned handler any more.

    # ── Language selection state ──
    if session.preferred_language is None:
        lang = request.message.strip().lower()
        if lang not in VALID_LANGUAGES:
            return BotResponse(
                bot_message=f"Please choose one of: {', '.join(VALID_LANGUAGES)}",
                input_type="mcq",
                options=LANGUAGE_OPTIONS,
            )
        await db.set_preferred_language(session_id, lang)
        first_q_en = FIRST_QUESTION.get(lang, FIRST_QUESTION["en"])
        lang_is_wrap = lang in WRAP_LANGUAGES
        first_q_display = (await translate_from_english(first_q_en, lang)) if lang_is_wrap else first_q_en
        first_q_mid = await db.save_message(
            session_id, "assistant", first_q_en, "sociodemographic",
            original_content=first_q_display if lang_is_wrap else None,
        )
        return BotResponse(
            bot_message=first_q_display,
            input_type="text",
            section="sociodemographic",
            session_status=session.status,
            message_id=first_q_mid,
        )

    # ── Normal turn ──
    return await _process_turn(session_id, request.message, background_tasks)


@app.post("/sessions/{session_id}/transcribe", response_model=TranscriptResponse)
async def transcribe_only(
    session_id: str,
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
):
    """
    Speech to text, and nothing else. Does not advance the interview.

    This is dictation: the caller records, gets words back, and the patient
    reviews them before anything is sent. Distinct from
    POST /messages/audio, which commits the recording as a turn immediately.

    Dictation is the safer default for a clinical intake. ASR misreads symptoms,
    dosages and durations, and a turn cannot be taken back once extraction has
    run on it — whereas a transcript sitting in a text box can simply be
    corrected. Nothing here writes to the session.

    `language` overrides the session language in response metadata only.
    SraVaani recognizes speech without language conditioning.

    Unlike /messages/audio this does NOT require a language to have been chosen.
    Nothing here is persisted, so a missing language has nothing to corrupt —
    SraVaani needs no language hint. The other endpoint refuses (409) because it
    commits a turn, and the turn pipeline needs a language.
    """
    session = await db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    audio = await file.read()
    if not audio:
        raise HTTPException(status_code=400, detail="Empty audio upload")

    hint = language or session.preferred_language
    try:
        transcript = await stt.transcribe_voice(
            audio, language=hint, filename=file.filename
        )
    except Exception as e:
        logger.error(f"[DICTATE] transcription failed for session={session_id}: {e}")
        raise HTTPException(status_code=503, detail="Transcription failed") from e

    if not transcript or not transcript.strip():
        raise HTTPException(
            status_code=422,
            detail="Could not make out any speech in that recording.",
        )

    logger.info(f"[DICTATE] session={session_id} transcript={transcript!r}")
    return TranscriptResponse(transcript=transcript, language=hint)


@app.post("/sessions/{session_id}/messages/audio", response_model=VoiceResponse)
async def send_voice_message(
    session_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Send a spoken patient message. Transcribes locally, then runs the ordinary turn.

    The REST counterpart of the Telegram voice handler. Voice had only ever been
    wired into the Telegram transport — stt.py and handlers.handle_voice_message
    existed, but nothing here could accept audio — so a REST client had no way to
    reach it at all.

    Transcription happens locally via the SraVaani-1.0 benchmark pipeline.
    The transcript is returned alongside the reply so the caller can show the
    patient what was heard and let them correct it, which matters because a
    misheard clinical detail is worse than no detail.
    """
    session = await db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Language selection is an MCQ, not something to answer by voice — asking a
    # patient to say "hinglish" clearly enough for an ASR model to pick it out of
    # six options is a worse experience than tapping a button.
    if session.preferred_language is None:
        raise HTTPException(
            status_code=409,
            detail="Select a language first (POST /sessions/{id}/messages).",
        )

    audio = await file.read()
    if not audio:
        raise HTTPException(status_code=400, detail="Empty audio upload")

    try:
        transcript = await stt.transcribe_voice(
            audio,
            language=session.preferred_language,
            filename=file.filename,
        )
    except Exception as e:
        logger.error(f"[VOICE] transcription failed for session={session_id}: {e}")
        raise HTTPException(status_code=503, detail="Transcription failed") from e

    if not transcript or not transcript.strip():
        # Not an error: silence, background noise, or a mis-tap. Tell the caller
        # plainly instead of feeding an empty string into the interview.
        raise HTTPException(
            status_code=422,
            detail="Could not make out any speech in that recording.",
        )

    logger.info(f"[VOICE] session={session_id} transcript={transcript!r}")

    # Crisis state is checked here rather than before transcription: we cannot
    # know whether a recording is a crisis disclosure until we have the words.
    # _process_turn handles crisis routing itself now, for voice as for text.
    reply = await _process_turn(session_id, transcript, background_tasks)

    return VoiceResponse(transcript=transcript, **reply.model_dump())


# ── Feedback on AI messages ───────────────────────────────────
# Thumbs up/down plus an optional written review, attached to one generated
# reply. Append-only: each submission is its own review with its own review_id,
# so multiple reviewers can rate the same message without overwriting one
# another. Deciding which review is "current" is the caller's business.

_VALID_RATINGS = {"up", "down"}


async def _require_ratable_message(session_id: str, message_id: int) -> dict:
    """Fetch a message and refuse anything that must not be rated.

    Scoped by session, so one session's id cannot be used to read or annotate
    another's transcript. Assistant-only, because the point of this data is to
    judge what the model said — quietly accepting a rating on the patient's own
    words would poison the dataset the reviews exist to build.
    """
    session = await db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    message = await db.get_message(session_id, message_id)
    if not message:
        raise HTTPException(
            status_code=404,
            detail=f"Message {message_id} not found in this session",
        )
    if message["role"] != "assistant":
        raise HTTPException(
            status_code=422,
            detail="Only AI-generated messages can be rated.",
        )
    return message


@app.post(
    "/sessions/{session_id}/messages/{message_id}/feedback",
    response_model=FeedbackResponse,
    status_code=201,
)
async def create_feedback(session_id: str, message_id: int, request: FeedbackRequest):
    """Rate an AI message. Returns the new review_id."""
    if request.rating not in _VALID_RATINGS:
        raise HTTPException(
            status_code=422,
            detail=f"rating must be one of: {', '.join(sorted(_VALID_RATINGS))}",
        )
    await _require_ratable_message(session_id, message_id)

    review_id = await db.save_feedback(
        session_id, message_id, request.rating, request.review
    )
    logger.info(
        f"[FEEDBACK] session={session_id} message={message_id} "
        f"rating={request.rating} review_id={review_id} "
        f"has_review={bool(request.review)}"
    )
    saved = await db.get_feedback(session_id, review_id)
    return FeedbackResponse(**saved)


@app.get("/sessions/{session_id}/feedback", response_model=List[FeedbackResponse])
async def list_session_feedback(session_id: str):
    """Every review left in this session, oldest first."""
    session = await db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return [FeedbackResponse(**f) for f in await db.get_feedback_for_session(session_id)]


@app.get(
    "/sessions/{session_id}/messages/{message_id}/feedback",
    response_model=List[FeedbackResponse],
)
async def list_message_feedback(session_id: str, message_id: int):
    """Every review left on one message."""
    await _require_ratable_message(session_id, message_id)
    return [
        FeedbackResponse(**f)
        for f in await db.get_feedback_for_message(session_id, message_id)
    ]


@app.patch("/sessions/{session_id}/feedback/{review_id}", response_model=FeedbackResponse)
async def edit_feedback(session_id: str, review_id: int, request: FeedbackUpdate):
    """Change a verdict, or add the written review after the fact."""
    if request.rating is not None and request.rating not in _VALID_RATINGS:
        raise HTTPException(
            status_code=422,
            detail=f"rating must be one of: {', '.join(sorted(_VALID_RATINGS))}",
        )
    if not await db.update_feedback(session_id, review_id, request.rating, request.review):
        raise HTTPException(status_code=404, detail="Review not found in this session")
    return FeedbackResponse(**await db.get_feedback(session_id, review_id))


@app.delete("/sessions/{session_id}/feedback/{review_id}")
async def remove_feedback(session_id: str, review_id: int):
    """Withdraw a review."""
    if not await db.delete_feedback(session_id, review_id):
        raise HTTPException(status_code=404, detail="Review not found in this session")
    return {"deleted": True, "review_id": review_id}


@app.get("/sessions/{session_id}/progress")
async def get_progress(session_id: str):
    """Get current progress — per-section completion and time estimates."""
    session = await db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    info_store = await db.get_full_info_store(session_id)
    summary = build_progress_summary(info_store, session.current_section)
    snapshot = await db.get_progress_snapshot(session_id)
    return {**summary, "snapshot": snapshot}


@app.get("/sessions/{session_id}/fields")
async def get_fields(session_id: str):
    """Get all collected fields across all sections."""
    session = await db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return await db.get_full_info_store(session_id)


@app.get("/sessions/{session_id}/history")
async def get_history(session_id: str):
    """
    Get full conversation history as JSON.
    Each message has: role, content, section, timestamp, and original_content
    (the latter is null except for ta/pa/bn sessions, where it holds the
    native-script text actually typed/shown).
    """
    session = await db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    messages = await db.get_full_conversation_history(session_id)
    return {
        "session_id": session_id,
        "message_count": len(messages),
        "messages": messages,
    }


@app.get("/sessions/{session_id}/report")
async def get_report(session_id: str):
    """Generate and return clinical case history report."""
    session = await db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"report": await generate_report(session_id)}


@app.get("/sessions/{session_id}/transcript")
async def get_transcript(session_id: str):
    """Get full patient-bot conversation transcript."""
    session = await db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"transcript": await generate_chat_transcript(session_id)}


@app.post("/sessions/{session_id}/pause")
async def pause_session(session_id: str):
    """Pause an active session."""
    session = await db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.status == "completed":
        raise HTTPException(status_code=400, detail="Session already completed")
    await db.set_session_status(session_id, "paused")
    clinician_id = os.getenv("CLINICIAN_TELEGRAM_ID")
    if clinician_id:
        logger.info(f"[PAUSE] session={session_id} — clinician {clinician_id} should be notified (REST: no Telegram)")
    return {"status": "paused", "message": "Session paused. Resume anytime."}


@app.post("/sessions/{session_id}/resume")
async def resume_session(session_id: str):
    """Resume a paused session."""
    session = await db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.status != "paused":
        raise HTTPException(status_code=400, detail="Session is not paused")
    await db.set_session_status(session_id, "active")
    return {"status": "active", "message": "Session resumed. Continue where you left off."}


# ── Useful additions ──────────────────────────────────────────

@app.get("/sessions/{session_id}/fields/{section}")
async def get_section_fields(session_id: str, section: str):
    """Get collected fields for one specific section."""
    session = await db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    from data_model import SECTION_CONFIG
    if section not in SECTION_CONFIG:
        raise HTTPException(status_code=400, detail=f"Unknown section: {section}")
    fields = await db.get_section_fields(session_id, section)
    return {
        "session_id": session_id,
        "section": section,
        "fields": fields,
    }


@app.get("/sessions/{session_id}/summaries")
async def get_summaries(session_id: str):
    """Get LLM-generated summaries for completed sections."""
    session = await db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    summaries = await db.get_all_summaries(session_id)
    return {
        "session_id": session_id,
        "summaries": summaries,
    }


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Hard delete all data for a session."""
    session = await db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    await db.delete_session(session_id)
    _reset_runtime_state(session_id)
    return {"deleted": True, "session_id": session_id}


@app.patch("/sessions/{session_id}/language")
async def update_language(session_id: str, request: MessageRequest):
    """Change preferred language mid-session."""
    session = await db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    lang = request.message.strip().lower()
    if lang not in VALID_LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Invalid language. Choose: {', '.join(VALID_LANGUAGES)}")
    await db.set_preferred_language(session_id, lang)
    return {"session_id": session_id, "preferred_language": lang}


@app.get("/sessions/{session_id}/progress/text")
async def get_progress_text(session_id: str):
    """Get progress as markdown string for display."""
    session = await db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    from progress import build_progress_text
    info_store = await db.get_full_info_store(session_id)
    text = build_progress_text(info_store, session.current_section)
    return {"session_id": session_id, "progress_text": text}


# ── Admin ─────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check — model status, DB reachable, API version."""
    from llm import TOKENIZER, MODEL
    db_ok = False
    try:
        await db.init_db()
        db_ok = True
    except Exception:
        pass
    return {
        "status": "ok" if db_ok else "degraded",
        "model_loaded": TOKENIZER is not None and MODEL is not None,
        "db_reachable": db_ok,
        "api_version": "1.0.0",
    }


class ClinicianAlert(BaseModel):
    alert_id: int
    session_id: str
    risk: Optional[str] = None
    message: str
    created_at: Optional[str] = None
    acknowledged_at: Optional[str] = None


@app.get("/admin/alerts", response_model=List[ClinicianAlert])
async def list_alerts(acknowledged: bool = False, limit: int = 100):
    """Suicide-risk alerts raised during crisis triage.

    Unacknowledged first, newest first within each group. Over Telegram these
    reach a clinician as a DM; over REST there is no chat, so they land here and
    something has to be watching — a dashboard polling this, or a person.

    `?acknowledged=true` includes ones already handled.
    """
    return [ClinicianAlert(**a) for a in await db.list_clinician_alerts(
        include_acknowledged=acknowledged, limit=limit
    )]


@app.post("/admin/alerts/{alert_id}/acknowledge", response_model=ClinicianAlert)
async def acknowledge_alert(alert_id: int):
    """Mark an alert as handled so it drops out of the open list."""
    alert = await db.acknowledge_clinician_alert(alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    logger.info(f"[ALERT] acknowledged alert_id={alert_id}")
    return ClinicianAlert(**alert)


@app.get("/admin/sessions")
async def list_sessions(
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
):
    """
    List all sessions, paginated.
    Optional filter: ?status=active|paused|completed|crisis
    """
    sessions = await db.list_sessions(status=status, limit=limit, offset=offset)
    return {
        "count": len(sessions),
        "limit": limit,
        "offset": offset,
        "status_filter": status,
        "sessions": sessions,
    }
