"""
Main entry point for psychiatric history-taking Telegram bot.
Initializes the Application and registers all handlers.
"""

import os
import asyncio
from dotenv import load_dotenv
from telegram.ext import (
    Application,
    ConversationHandler,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
)

# Load environment variables
load_dotenv()

import logging

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(name)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[]  # terminal only at root level
)
logging.basicConfig(
    level=logging.WARNING,  # ← only warnings/errors to terminal
    format="%(asctime)s.%(msecs)03d [%(name)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler()]
)

# Import handlers
from handlers import (
    start_handler,
    language_selected,
    handle_patient_message,
    pause_session,
    resume_session,
    report_handler,
    cancel_handler,
    help_handler,
    crisis_followup_handler,
    with_model,
    awaiting_language_handler,
    handle_voice_message,
)

# Import model warmup
import llm

# On-demand GPU lifecycle: the bot runs 24x7, the model server does not.
import gpu_manager

# Import states
from data_model import LANGUAGE_SELECTION, COLLECTING_HISTORY, CRISIS_MODE

# Import database initialization
import db

# Get bot token
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

if not TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN environment variable not set")


async def _idle_check_job(context):
    """Timer callback — see gpu_manager.idle_check."""
    await gpu_manager.idle_check(context.bot)


async def _release_gpu_on_shutdown(_app):
    """Never leave the GPU pinned when the bot stops or is restarted."""
    await gpu_manager.shutdown("bot shutting down")


def main():
    """Initialize and run the bot."""
    
    # Initialize database
    print("Initializing database...")
    asyncio.run(db.init_db())
    print("Database initialized.")

    # Warm up the model before the first patient message arrives.
    # This avoids the appearance that the bot is stuck after the initial name prompt.
    #
    # Skipped in lazy-GPU mode: there is deliberately no model server at boot —
    # /start brings one up. Calling warmup here would just print a scary
    # "vLLM server not reachable" warning describing the intended state.
    # Refuse to run lazy-GPU mode on the in-process backend. LLM_BACKEND defaults
    # to "hf" (llm.py:46), and hf loads ~28GB of Qwen3-14B into THIS process on
    # whatever card CUDA picks — bypassing every free-GPU check in gpu_manager and
    # very possibly landing on someone else's work. Under watchdog.sh this never
    # showed up because the script exports LLM_BACKEND=vllm itself; running
    # `python bot.py` by hand does not. Fail loudly rather than quietly grab a GPU.
    if gpu_manager.ENABLED and llm.LLM_BACKEND != "vllm":
        raise SystemExit(
            f"LLM_BACKEND is {llm.LLM_BACKEND!r}, but on-demand GPU mode requires 'vllm'.\n"
            f"The '{llm.LLM_BACKEND}' backend would load the model into this process on an\n"
            f"unchecked GPU. Add LLM_BACKEND=vllm to .env, or set LLM_LAZY_GPU=0 to opt out\n"
            f"of on-demand GPU management."
        )

    if gpu_manager.ENABLED:
        where = (
            f"pinned to GPU {gpu_manager.GPU_DEDICATED_INDEX}"
            if gpu_manager.DEDICATED
            else f"GPU priority {gpu_manager.GPU_PRIORITY}"
        )
        when = (
            "stays warm once started"
            if gpu_manager.WARM
            else f"idle shutdown after {int(gpu_manager.IDLE_TIMEOUT_S / 60)} min"
        )
        print(f"Lazy GPU mode: model server starts on demand ({where}, {when}).")
    else:
        asyncio.run(llm.warmup_model())
    
    # asyncio.run() closes the event loop, so we need to create a new one
    # for app.run_polling()
    asyncio.set_event_loop(asyncio.new_event_loop())
    
    # Create Application
    print("Starting bot...")
    # app = Application.builder().token(TOKEN).build()
    
    from telegram.request import HTTPXRequest
    request = HTTPXRequest(read_timeout=60, connect_timeout=60)
    builder = Application.builder().token(TOKEN).request(request)
    if gpu_manager.ENABLED:
        builder = builder.post_shutdown(_release_gpu_on_shutdown)
    app = builder.build()
    
    # Create conversation handler
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start_handler)],
        states={
            LANGUAGE_SELECTION: [
                CallbackQueryHandler(language_selected),
                # Text here used to vanish silently, which during a 7-8 minute
                # boot is indistinguishable from the bot being broken.
                MessageHandler(
                    filters.TEXT & ~filters.COMMAND,
                    awaiting_language_handler
                ),
            ],
            COLLECTING_HISTORY: [
                # with_model: bring the GPU back up if the idle timer released it,
                # and count the message against the idle clock. Transparent to the
                # patient apart from a "starting up" notice on a cold turn.
                MessageHandler(
                    filters.TEXT & ~filters.COMMAND,
                    with_model(handle_patient_message, COLLECTING_HISTORY)
                ),
                # Voice notes take the same path as text: handle_voice_message
                # transcribes, then feeds the transcript into the very same
                # buffered turn pipeline. with_model for the same reason as
                # above — transcription is local, but the reply still needs the
                # model server up.
                MessageHandler(
                    filters.VOICE | filters.AUDIO,
                    with_model(handle_voice_message, COLLECTING_HISTORY)
                ),
                CommandHandler("pause", pause_session),
                CommandHandler("resume", resume_session),
                CommandHandler("report", report_handler),
            ],
            CRISIS_MODE: [
                MessageHandler(
                    filters.TEXT & ~filters.COMMAND,
                    with_model(crisis_followup_handler, CRISIS_MODE)
                ),
                # Someone in crisis may well prefer to speak than type, and
                # dropping their voice note on the floor is the worst possible
                # moment to go silent. handle_voice_message checks
                # crisis.in_crisis itself and routes to enter_crisis, so the
                # transcript reaches _run_crisis_turn rather than the interview.
                MessageHandler(
                    filters.VOICE | filters.AUDIO,
                    with_model(handle_voice_message, CRISIS_MODE)
                ),
                # Allow the patient to pause/resume/cancel from within crisis mode.
                CommandHandler("pause", pause_session),
                CommandHandler("resume", resume_session),
                CommandHandler("cancel", cancel_handler),
            ]
        },
        fallbacks=[CommandHandler("cancel", cancel_handler)],
    )
    
    # Add handlers to application
    app.add_handler(conv_handler)
    app.add_handler(CommandHandler("help", help_handler))

    # Idle watcher: warns, then releases the GPU after a quiet stretch. Runs every
    # minute — cheap, and it never fires while a generation is in flight.
    # Not scheduled at all in warm mode, where idle_check would return immediately
    # every time — no point waking the job queue once a minute forever to do nothing.
    if gpu_manager.ENABLED and not gpu_manager.WARM:
        app.job_queue.run_repeating(
            _idle_check_job, interval=60, first=60, name="gpu_idle_check",
        )

    # Start the bot
    print("Bot is running. Press Ctrl+C to stop.")
    app.run_polling(allowed_updates=["message", "callback_query"])


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBot stopped.")
