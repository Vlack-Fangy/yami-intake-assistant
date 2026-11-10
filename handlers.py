"""
Telegram event handlers for psychiatric history-taking bot.
Handles start, language selection, patient messages, and commands.
"""


from utils import get_logger
logger = get_logger("handlers")  # → logs/handlers.log

from datetime import datetime

def _ts():
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]

from utils import build_topic_based_hint, build_filled_fields_summary, get_next_natural_question, get_next_topic_after, get_optimistic_filled_fields, mark_fields_filled, clear_filled_fields_cache, get_prev_section, set_prev_section, get_planner_logger

planner_log = get_planner_logger()

import asyncio
import os
from collections import defaultdict
import db
from data_model import LANGUAGE_SELECTION, COLLECTING_HISTORY, CRISIS_MODE, SUPPORTED_LANGUAGES, SECTION_CONFIG, SECTION_CONVERSATION_TOPICS, SECTIONS_ORDER
import llm
import safety
import crisis
import stt
from progress import send_initial_progress_message, update_progress_message, compute_completion_percentage, check_milestone_and_update
from transitions import check_section_transition
import clinical_state
import turn_pipeline
from report import send_report_handler, send_clinician_pause_summary
from telegram import Update
from telegram.ext import ContextTypes
from telegram.constants import ChatAction, ParseMode


EXTRACTION_LOCKS = defaultdict(asyncio.Lock)
RESPONSE_LOCKS = defaultdict(asyncio.Lock)
MESSAGE_BUFFERS = defaultdict(list)
MESSAGE_TIMER_TASKS = {}
MESSAGE_DEBOUNCE_SECONDS = float(os.getenv("MESSAGE_DEBOUNCE_SECONDS", "1.8"))


# ── On-demand GPU / model server ────────────────────────────────────────────
# The bot polls Telegram 24x7 but the model server only runs while someone is
# actually talking, so the A100 is free for everyone else the rest of the time.
# See gpu_manager for the lifecycle and the "never disturb another job" rules.
import gpu_manager

STARTUP_NOTICE = (
    "I keep the server's GPU free when nobody's using me, so I need to boot up first. "
    "That takes about 7–8 minutes — I'll message you the moment I'm ready.\n\n"
    "(I also switch myself off after 15 minutes of quiet. If that happens, just send "
    "a message and I'll wake up again — nothing is lost.)"
)
BUSY_NOTICE = (
    "Every GPU on the server is busy with other work right now, so I can't start up. "
    "Nothing is wrong — please try /start again in a few minutes."
)
OFFLINE_NOTICE = (
    "I'm offline at the moment because of a technical problem on the server. "
    "Please contact the administrator, and try again later."
)


async def _model_offline_reply(chat_id, context, status) -> None:
    text = BUSY_NOTICE if status is gpu_manager.Status.NO_FREE_GPU else OFFLINE_NOTICE
    await context.bot.send_message(chat_id, text, parse_mode=None)


async def ensure_model_ready(chat_id, context, lead: str = "") -> bool:
    """
    Make sure the model server is up, keeping the patient informed while it
    boots. Returns False if it could not be started (caller should stop).

    Returns almost instantly in the common case where the server is already up.
    """
    if not gpu_manager.ENABLED or await gpu_manager.is_healthy():
        gpu_manager.note_activity(chat_id)
        return True

    # Don't repeat the full explanation to someone who just read it on /start —
    # they get a short progress line instead.
    if context.user_data.get("startup_notice_shown"):
        text = "Still getting ready — thanks for your patience."
    else:
        text = (lead + "\n\n" if lead else "") + STARTUP_NOTICE
        context.user_data["startup_notice_shown"] = True
    try:
        notice = await context.bot.send_message(chat_id, text, parse_mode=None)
    except Exception:
        notice = None

    # Drive the progress display from here rather than through ensure_ready's
    # callback, because a second patient arriving mid-boot waits on the manager's
    # lock and would otherwise sit in silence until the first one's boot finished.
    import time as _time
    started = _time.monotonic()
    task = asyncio.ensure_future(gpu_manager.ensure_ready())
    while True:
        done, _pending = await asyncio.wait({task}, timeout=10)
        if done:
            break
        if notice is None:
            continue
        elapsed = int(_time.monotonic() - started)
        try:
            # Editing one message rather than sending a stream of them: an
            # 8-minute silence looks like a hang, but 40 new messages is worse.
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=notice.message_id,
                text=f"{text}\n\nStarting up… {elapsed // 60}m {elapsed % 60}s elapsed.",
                parse_mode=None,
            )
        except Exception:
            pass    # a failed edit (rate limit, "not modified") is not fatal

    status, detail = task.result()
    logger.info(f"[{_ts()}] GPU — ensure_ready for {chat_id}: {status.value} ({detail})")

    if status is gpu_manager.Status.READY:
        gpu_manager.note_activity(chat_id)
        if notice is not None:
            try:
                await context.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=notice.message_id,
                    text="All set — thanks for waiting. 🙂",
                    parse_mode=None,
                )
            except Exception:
                pass
        return True

    await _model_offline_reply(chat_id, context, status)
    return False


TOO_EARLY_NOTICE = (
    "Sorry — I was still starting up when you sent that, so it hasn't reached me. "
    "I'm ready now: go ahead and reply to the question above."
)


def _sent_before_conversation_started(update, context) -> bool:
    """
    Was this message typed before we were actually ready to receive answers?

    Two cases, both of which used to get fed to the model as if they were
    answers:

      1. Typed while the model was booting ("is it ready?").
      2. Typed after the model came up but BEFORE the first question was sent.
         Telegram updates are dispatched one at a time, so a message sent during
         the last moments of language_selected sits in the queue and arrives as
         the patient's "answer" to a question they had not yet been asked. That
         is what produced a duplicated "Could you tell me your name?".

    Compared against Telegram's own send timestamp, not arrival time, because
    arrival order is exactly what misleads us here.
    """
    message = getattr(update, "message", None)
    if message is None or message.date is None:
        return False
    sent_at = message.date.timestamp()

    started_at = context.user_data.get("conversation_started_at")
    if started_at is not None and sent_at < started_at:
        return True
    return gpu_manager.was_sent_during_boot(sent_at)


def with_model(handler, fallback_state):
    """
    Wrap a message handler so the model server is warm before it runs, and so
    every inbound message counts as activity against the idle timer.

    `fallback_state` is the conversation state to stay in when the model cannot
    be started — the session is left intact so the patient can simply try again.
    """
    async def wrapped(update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id = update.effective_chat.id
        gpu_manager.note_activity(chat_id)

        if _sent_before_conversation_started(update, context):
            logger.info(f"[{_ts()}] GPU — dropping pre-start message from {chat_id}")
            await update.message.reply_text(TOO_EARLY_NOTICE, parse_mode=None)
            return fallback_state

        if not await ensure_model_ready(chat_id, context):
            return fallback_state
        return await handler(update, context)

    wrapped.__name__ = f"with_model({getattr(handler, '__name__', 'handler')})"
    return wrapped


def _reset_runtime_chat_state(chat_id: int) -> None:
    MESSAGE_BUFFERS.pop(int(chat_id), None)
    task = MESSAGE_TIMER_TASKS.pop(int(chat_id), None)
    if task and not task.done():
        task.cancel()
    clear_filled_fields_cache(chat_id)
    clinical_state.clear_state(str(chat_id))
    crisis.clear(str(chat_id))


def _cancel_pending_turn(chat_id: int) -> None:
    """Cancel any debounced/buffered normal turn without wiping interview state.

    Used when a crisis message arrives: we must respond immediately and must not
    also fire the normal interview reply, but we keep clinical_state so the
    interview can resume cleanly afterwards.
    """
    MESSAGE_BUFFERS.pop(int(chat_id), None)
    task = MESSAGE_TIMER_TASKS.pop(int(chat_id), None)
    if task and not task.done():
        task.cancel()


async def _run_crisis_turn(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    session,
    patient_text: str,
) -> str:
    """Telegram wrapper: map the shared crisis turn onto conversation states.

    All the triage logic lives in turn_pipeline.run_crisis_turn so the REST
    transport runs exactly the same code. What stays here is Telegram-shaped:
    the FSM state to return, and re-driving the buffered pipeline so the next
    interview question arrives in the same turn as the de-escalation bridge.
    """
    chat_id = update.effective_chat.id
    stay, _risk = await turn_pipeline.run_crisis_turn(
        chat_id=chat_id,
        patient_text=patient_text,
        session=session,
        context=context,
    )
    if stay:
        return CRISIS_MODE
    _resume_interview_with_next_question(chat_id, patient_text, context)
    return COLLECTING_HISTORY

def _resume_interview_with_next_question(
    chat_id: int, patient_text: str, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """After a crisis de-escalation, ask the next interview question in the same
    turn by driving the normal buffered-turn pipeline with the patient's
    confirmation message — reusing all of the existing question / extraction
    logic instead of reimplementing it here."""
    MESSAGE_BUFFERS[int(chat_id)].append(patient_text)
    existing = MESSAGE_TIMER_TASKS.get(int(chat_id))
    if existing and not existing.done():
        existing.cancel()
    MESSAGE_TIMER_TASKS[int(chat_id)] = asyncio.create_task(
        _process_buffered_turn(chat_id=chat_id, context=context)
    )


async def enter_crisis(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    session,
    patient_text: str,
) -> str:
    """First crisis turn: pause the normal interview and start the safety check."""
    chat_id = update.effective_chat.id
    _cancel_pending_turn(chat_id)
    await db.set_session_status(chat_id, "crisis")
    logger.info(f"[{_ts()}] CRISIS — entering crisis mode for chat {chat_id}")
    return await _run_crisis_turn(update, context, session, patient_text)


async def crisis_followup_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
    """Handles every patient message while in CRISIS_MODE (replaces the old
    static helpline-repeat handler)."""
    chat_id = update.effective_chat.id
    patient_text = (update.message.text or "").strip()
    session = await db.get_session(chat_id)
    if not session:
        await update.message.reply_text(
            "Session not found. Please use /start to begin.", parse_mode=None
        )
        return COLLECTING_HISTORY
    if not patient_text:
        return CRISIS_MODE
    return await _run_crisis_turn(update, context, session, patient_text)


LANGUAGE_ACK = {
    "en": "Great! Language set to {language}.",
    "hi": "Bahut badhiya! Bhasha {language} set ho gayi hai.",
    "hinglish": "Great! Language set ho gayi: {language}.",
}


FIRST_QUESTION_BY_LANGUAGE = {
    "en": (
        "Thank you! Let's get started.\n\n"
        "To begin with, could you tell me your name?"
    ),
    "hi": (
        "Dhanyavaad! Chaliye shuru karte hain.\n\n"
        "Sabse pehle, kya aap apna naam bata sakte hain?"
    ),
    "hinglish": (
        "Thank you! Chaliye shuru karte hain.\n\n"
        "Sabse pehle, kya aap apna naam bata sakte hain?"
    ),
}


async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
    """
    /start command handler.
    Creates new session and shows language selection.
    """
    
    chat_id = update.effective_chat.id

    _reset_runtime_chat_state(chat_id)

    # Preflight BEFORE creating a session: is there a GPU we're allowed to take?
    # This is a couple of nvidia-smi calls, so the patient finds out immediately
    # rather than after picking a language and waiting eight minutes for nothing.
    gpu_manager.note_activity(chat_id)
    status, detail = await gpu_manager.preflight()
    logger.info(f"[{_ts()}] GPU — preflight for {chat_id}: {status.value} ({detail})")
    if status is not gpu_manager.Status.READY:
        await _model_offline_reply(chat_id, context, status)
        return -1  # End conversation

    # Start the model loading NOW, in the background, so it overlaps with the
    # welcome text and language choice instead of running after them.
    booting = gpu_manager.ENABLED and not await gpu_manager.is_healthy()
    if booting:
        context.application.create_task(gpu_manager.ensure_ready())

    # Create new session in DB
    await db.create_session(chat_id)

    # Send warm welcome message
    welcome_msg = (
        "Hello! 👋 I'm here to help your doctor understand you better before your appointment.\n\n"
        "I'll ask you some questions about yourself and how you've been feeling. "
        "There are no right or wrong answers — just share whatever feels comfortable.\n\n"
        "This will take about 60–75 minutes in total. You can pause anytime by typing /pause "
        "and come back later.\n\n"
        "First, which language would you prefer?"
    )
    
    await update.message.reply_text(welcome_msg, parse_mode=None)

    if booting:
        await update.message.reply_text(STARTUP_NOTICE, parse_mode=None)
        context.user_data["startup_notice_shown"] = True

    # Build language buttons
    buttons = []
    for lang_code, lang_name in SUPPORTED_LANGUAGES.items():
        buttons.append({
            "text": lang_name,
            "callback_data": f"lang_{lang_code}"
        })
    
    # Send language selection keyboard
    from telegram import InlineKeyboardMarkup, InlineKeyboardButton
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton(name, callback_data=f"lang_{code}")]
        for code, name in SUPPORTED_LANGUAGES.items()
    ])
    
    await context.bot.send_message(
        chat_id,
        "Choose your language / अपनी भाषा चुनें:",
        reply_markup=keyboard,
        parse_mode=None
    )
    
    return LANGUAGE_SELECTION


async def language_selected(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
    """
    Language selection callback handler.
    Saves language, sends initial progress message, starts history collection.
    """
    
    query = update.callback_query
    await query.answer()
    
    chat_id = update.effective_chat.id
    _reset_runtime_chat_state(chat_id)
    lang_code = query.data.split("_")[1]  # Extract "en" or "hi" from "lang_en"
    
    # Save language to DB
    await db.set_preferred_language(chat_id, lang_code)
    
    # Edit the language selection message
    ack_template = LANGUAGE_ACK.get(lang_code, LANGUAGE_ACK["en"])
    await query.edit_message_text(
        f"{ack_template.format(language=SUPPORTED_LANGUAGES[lang_code])} 🎯",
        parse_mode=None
    )
    
    # Wait for the model here rather than after the first question: otherwise the
    # patient answers it and *then* discovers there's an eight-minute wait.
    if not await ensure_model_ready(chat_id, context):
        return -1  # End conversation

    # Send and pin the progress message
    try:
        await send_initial_progress_message(chat_id, context)
    except Exception as e:
        print(f"Failed to send initial progress message: {e}")
        logger.error(f"[{_ts()}] EXTRACT — Failed to send initial progress message: {e}")
    
    # Send first question for sociodemographic section
    first_question = FIRST_QUESTION_BY_LANGUAGE.get(lang_code, FIRST_QUESTION_BY_LANGUAGE["en"])
    
    await context.bot.send_message(chat_id, first_question, parse_mode=None)

    # Save the first question to history so extraction can see which question the patient's first answer is responding to.
    await db.save_message(chat_id, "assistant", first_question, "sociodemographic")

    # The interview starts HERE. Anything the patient typed before this moment
    # was not an answer to anything — see _sent_before_conversation_started.
    import time as _time
    context.user_data["conversation_started_at"] = _time.time()

    return COLLECTING_HISTORY


async def awaiting_language_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
    """
    Text typed while we're still waiting on the language buttons.

    Previously nothing was registered for this, so these messages vanished
    silently — which during a 7-minute boot looks exactly like the bot being
    broken. Answer them instead.
    """
    if gpu_manager.ENABLED and not await gpu_manager.is_healthy():
        text = ("I'm still starting up — this takes about 7–8 minutes. "
                "Please pick a language with the buttons above; "
                "I'll begin as soon as I'm ready.")
    else:
        text = "Please choose your language using the buttons above to begin."
    await update.message.reply_text(text, parse_mode=None)
    return LANGUAGE_SELECTION


async def handle_patient_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
    """
    Main message handler. Processes every patient message:
    1. Check for crisis keywords
    2. Generate next question via LLM
    3. Extract structured info
    4. Check for section transition
    5. Update progress tracker
    """
    
    chat_id = update.effective_chat.id
    patient_text = (update.message.text or "").strip()
    if not patient_text:
        return COLLECTING_HISTORY
    
    # Get session
    session = await db.get_session(chat_id)
    if not session:
        await update.message.reply_text(
            "Session not found. Please use /start to begin.",
            parse_mode=None
        )
        return COLLECTING_HISTORY
    
    # 1. Safety check — crisis keyword detection.
    # Crisis handling now runs a brief adaptive safety triage (see crisis.py)
    # instead of resetting state and sending a single canned helpline message.
    if safety.is_crisis_message(patient_text) or crisis.in_crisis(str(chat_id)):
        return await enter_crisis(update, context, session, patient_text)

    if safety.is_soft_stop_message(patient_text):
        _reset_runtime_chat_state(chat_id)
        await db.set_session_status(chat_id, "paused")
        await update.message.reply_text(
            "No problem. We can pause here. Type /resume whenever you want to continue.",
            parse_mode=None,
        )
        return COLLECTING_HISTORY
    
    # Trauma detection — send care message
    if safety.is_trauma_message(patient_text):
        await safety.handle_trauma_alert(update, context, session.current_section)

    # Debounce and batch fragmented user messages into one model turn.
    MESSAGE_BUFFERS[int(chat_id)].append(patient_text)
    existing_task = MESSAGE_TIMER_TASKS.get(int(chat_id))
    if existing_task and not existing_task.done():
        existing_task.cancel()

    MESSAGE_TIMER_TASKS[int(chat_id)] = asyncio.create_task(
        _process_buffered_turn(chat_id=chat_id, context=context)
    )
    
    return COLLECTING_HISTORY


async def _process_buffered_turn(chat_id: int, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        await asyncio.sleep(MESSAGE_DEBOUNCE_SECONDS)
    except asyncio.CancelledError:
        return

    try:
        lock = RESPONSE_LOCKS[int(chat_id)]
        async with lock:
            chunks = MESSAGE_BUFFERS.pop(int(chat_id), [])
            if not chunks:
                return

            patient_text = "\n".join([chunk.strip() for chunk in chunks if chunk.strip()]).strip()
            if not patient_text:
                return
            
            logger.info(f"\n{'='*60}")
            logger.info(f"[{_ts()}] NEW TURN — patient: {patient_text!r}")
            logger.info(f"{'='*60}")

            session = await db.get_session(chat_id)

            if not session:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text="Session not found. Please use /start to begin.",
                    parse_mode=None,
                )
                return
            
            await turn_pipeline.run_turn(
                chat_id=chat_id,
                patient_text=patient_text,
                session=session,
                context=context,
            )

    except Exception as error:
        
        logger.error(f"[{_ts()}] BUFFERED TURN FAILED: {error}")
        print(f"Buffered turn processing failed: {error}")
        try:
            await context.bot.send_message(
                chat_id=chat_id,
                text="I had a technical issue. Please send that once again.",
                parse_mode=None,
            )
        except Exception:
            pass
    finally:
        current_task = MESSAGE_TIMER_TASKS.get(int(chat_id))
        if current_task is asyncio.current_task():
            MESSAGE_TIMER_TASKS.pop(int(chat_id), None)


async def pause_session(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
    """
    /pause command handler.
    Saves session state and pauses history collection.
    """
    
    chat_id = update.effective_chat.id
    _reset_runtime_chat_state(chat_id)
    
    # Update session status
    await db.set_session_status(chat_id, "paused")

    # Best-effort clinician summary for paused sessions.
    await send_clinician_pause_summary(chat_id, context)
    
    pause_msg = (
        "No problem. Your progress is saved. 💾\n\n"
        "Type /resume whenever you're ready to continue — even days from now.\n"
        "Everything will be exactly where you left off."
    )
    
    await update.message.reply_text(pause_msg, parse_mode=None)
    
    return COLLECTING_HISTORY


async def resume_session(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
    """
    /resume command handler.
    Resumes paused session, re-pins progress message.
    """
    
    chat_id = update.effective_chat.id
    session = await db.get_session(chat_id)
    
    if not session:
        await update.message.reply_text(
            "No active session found. Use /start to begin.",
            parse_mode=None
        )
        return COLLECTING_HISTORY
    
    if session.status != "paused":
        await update.message.reply_text(
            "Your session is already active!",
            parse_mode=None
        )
        return COLLECTING_HISTORY
    
    # Resume: update status
    await db.set_session_status(chat_id, "active")
    
    # Re-pin progress message
    if session.progress_message_id:
        try:
            await context.bot.pin_chat_message(
                chat_id=chat_id,
                message_id=session.progress_message_id,
                disable_notification=True
            )
        except Exception as e:
            print(f"Failed to re-pin progress message: {e}")
            logger.error(f"[{_ts()}] EXTRACT — Failed to re-pin progress message: {e}")
    
    resume_msg = (
        "Welcome back! 👋\n\n"
        "Your progress has been restored. Let's continue where we left off.\n"
        "Just send me any message to pick up."
    )
    
    await update.message.reply_text(resume_msg, parse_mode=None)
    
    return COLLECTING_HISTORY


async def cancel_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
    """
    /cancel command handler.
    Cancels the conversation.
    """
    
    chat_id = update.effective_chat.id
    _reset_runtime_chat_state(chat_id)

    await update.message.reply_text(
        "Session cancelled. Use /start to begin a new session.",
        parse_mode=None
    )
    
    return -1  # End conversation


async def report_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
    """
    /report command handler.
    Generates and sends current report.
    """
    
    await send_report_handler(update, context)
    return COLLECTING_HISTORY


async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /help command handler.
    Shows available commands.
    """
    
    help_text = (
        "Available commands:\n\n"
        "/pause — Save progress and pause the session\n"
        "/resume — Continue a paused session\n"
        "/report — Download your compiled history so far\n"
        "/cancel — Cancel the session\n"
        "/help — Show this message"
    )
    
    await update.message.reply_text(help_text, parse_mode=None)


# ─── handlers.py — voice message support ────────────────────────────────────
#
# STEP 1: Add this import near the top of handlers.py (after "import safety"):
#
#     import stt
#
# STEP 2: Paste the function below into handlers.py
#         (anywhere after handle_patient_message).
#
# STEP 3: See bot_voice_patch.py for the bot.py changes.
# ────────────────────────────────────────────────────────────────────────────

async def handle_voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
    """
    Voice message handler.

    Downloads the Telegram OGG/Opus voice note, transcribes it locally via
    SraVaani-1.0, then feeds the transcript directly into _process_buffered_turn
    (bypassing the read-only update.message.text field).
    """
    chat_id = update.effective_chat.id

    voice = update.message.voice or update.message.audio
    if not voice:
        await update.message.reply_text(
            "I couldn't read that audio. Could you type it instead?",
            parse_mode=None,
        )
        return COLLECTING_HISTORY

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    ack = await update.message.reply_text(
        "🎙️ Got your voice message, transcribing...",
        parse_mode=None,
    )

    try:
        tg_file = await context.bot.get_file(voice.file_id)
        file_bytes = await tg_file.download_as_bytearray()

        session = await db.get_session(chat_id)
        lang_hint = session.preferred_language if session else None

        transcript = await stt.transcribe_voice(bytes(file_bytes), language=lang_hint)

    except Exception as e:
        logger.error(f"[VOICE] download/transcribe error: {e}")
        transcript = None

    # Remove the "transcribing..." message regardless of outcome
    try:
        await context.bot.delete_message(chat_id=chat_id, message_id=ack.message_id)
    except Exception:
        pass

    if not transcript:
        await update.message.reply_text(
            "Sorry, I couldn't make out that voice message. "
            "Could you type it instead?",
            parse_mode=None,
        )
        return COLLECTING_HISTORY

    # Show the patient what was heard so they can correct it if needed
    await update.message.reply_text(
        f'🎙️ I heard: "{transcript}"',
        parse_mode=None,
    )

    # ── FIX: don't mutate update.message.text (it's read-only in PTB) ──
    # Instead run the same crisis + safety checks then feed directly into
    # the buffered-turn pipeline using the transcript string.

    session = await db.get_session(chat_id)
    if not session:
        await update.message.reply_text(
            "Session not found. Please use /start to begin.",
            parse_mode=None,
        )
        return COLLECTING_HISTORY

    if safety.is_crisis_message(transcript) or crisis.in_crisis(str(chat_id)):
        return await enter_crisis(update, context, session, transcript)

    if safety.is_soft_stop_message(transcript):
        _reset_runtime_chat_state(chat_id)
        await db.set_session_status(chat_id, "paused")
        await update.message.reply_text(
            "No problem. We can pause here. Type /resume whenever you want to continue.",
            parse_mode=None,
        )
        return COLLECTING_HISTORY

    if safety.is_trauma_message(transcript):
        await safety.handle_trauma_alert(update, context, session.current_section)

    # Buffer the transcript and trigger the debounced processing pipeline
    MESSAGE_BUFFERS[int(chat_id)].append(transcript)
    existing_task = MESSAGE_TIMER_TASKS.get(int(chat_id))
    if existing_task and not existing_task.done():
        existing_task.cancel()

    MESSAGE_TIMER_TASKS[int(chat_id)] = asyncio.create_task(
        _process_buffered_turn(chat_id=chat_id, context=context)
    )

    return COLLECTING_HISTORY