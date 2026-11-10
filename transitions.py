"""
Section transition logic for psychiatric history bot.
Detects when sections are complete and moves to the next section.
"""

import db
from data_model import SECTIONS_ORDER, SECTION_CONFIG
from progress import update_progress_message, compute_completion_percentage
import llm
import clinical_state
import os
import io
import asyncio

from utils import get_logger
logger = get_logger("transitions")

# Consecutive no-progress turns in a section before we force a graceful advance.
# Breaks the loop where a section can never reach min_required (patient refuses,
# answers "no", or the answer isn't extractable) and the bot re-asks forever.
SECTION_STALL_LIMIT = int(os.getenv("SECTION_STALL_LIMIT", "4"))

def get_next_section(current_section: str) -> str:
    """Get the next section after the current one, or None if at end."""
    try:
        idx = SECTIONS_ORDER.index(current_section)
        if idx + 1 < len(SECTIONS_ORDER):
            return SECTIONS_ORDER[idx + 1]
    except ValueError:
        pass
    return None


async def check_section_transition(
    chat_id: str,
    session,
    context
) -> bool:

    try:
        chat_id_int = int(chat_id)
    except (ValueError, TypeError):
        chat_id_int = chat_id
    section = session.current_section

    fields = await db.get_section_fields(chat_id_int, section)
    config = SECTION_CONFIG[section]

    filled = sum(1 for v in fields.values() if v)
    completion_pct = compute_completion_percentage(section, fields)

    # Track stall so a section that can't reach min_required (refusals,
    # un-extractable answers, "no/never") doesn't loop forever.
    state = clinical_state.get_state(str(chat_id))
    stall = state.record_section_progress(section, filled)

    logger.info(f"[TRANSITION] section={section} filled={filled} pct={completion_pct}% stall={stall}")

    complete = completion_pct >= 100
    forced = (not complete) and stall >= SECTION_STALL_LIMIT

    if not (complete or forced):
        return False

    # A name is worth a few extra asks: don't leave the first section without one
    # until the stall is much deeper.
    if section == "sociodemographic" and not fields.get("name"):
        if not forced or stall < SECTION_STALL_LIMIT * 2:
            return False

    if forced:
        logger.info(f"[TRANSITION] section={section} STALLED (stall={stall}) — forcing graceful advance")
        uncovered = [f for f in config["fields"] if not fields.get(f)]
        if uncovered:
            note = (
                "Section advanced without full coverage (patient declined, could not answer, "
                "or the conversation stalled). Not captured: " + ", ".join(uncovered)
            )
            try:
                await db.save_section_note(chat_id_int, section, note)
            except Exception as e:
                logger.warning(f"[TRANSITION] failed to save uncovered note: {e}")
    else:
        logger.info(f"[TRANSITION] section complete — firing summary in background")

    return await _advance_to_next_section(chat_id_int, session, context, section)


async def _advance_to_next_section(chat_id_int, session, context, section) -> bool:
    """Close the current section, move to the next (or finish), and send a warm
    bridging message so the patient isn't jolted between areas."""
    asyncio.create_task(close_section(chat_id_int, section, context))

    next_section = get_next_section(section)
    if next_section is None:
        logger.info(f"[TRANSITION] all sections done — finishing session")
        await finish_session(chat_id_int, context)
        return True

    await db.set_current_section(chat_id_int, next_section)
    logger.info(f"[TRANSITION] section set: {section} → {next_section}")

    await db.set_last_topic_asked(chat_id_int, None)
    logger.info(f"[TRANSITION] LAST_TOPIC_ASKED reset for chat {chat_id_int}")

    # Record the transition so the NEXT reply can open the new area warmly
    # inside a SINGLE message, instead of sending a separate bridge message.
    #
    # The old behaviour sent generate_transition_message() as its own message
    # here. Two problems: (1) this runs inside a background extraction task, so
    # the bridge arrived out of order — often after the patient had already
    # replied — and (2) its prompt generated exactly the "thanks… a few more
    # questions" filler that the main reply prompt explicitly bans, producing a
    # canned second message. We now hand the transition to the reply path (see
    # handlers._process_buffered_turn + llm.get_next_response transition_hint),
    # which folds the acknowledgement into the one patient-facing reply.
    try:
        state = clinical_state.get_state(str(chat_id_int))
        state.pending_transition = (section, next_section)
        logger.info(f"[TRANSITION] recorded pending transition {section} → {next_section}")
    except Exception as e:
        logger.warning(f"[TRANSITION] failed to record pending transition: {e}")

    info_store = await db.get_full_info_store(chat_id_int)
    await update_progress_message(
        chat_id_int, context, info_store, next_section, force=True
    )

    return True


async def close_section(chat_id: int, section: str, context) -> None:
    logger.info(f"[CLOSE] generating summary for section={section}")
    section_data = await db.get_section_data(chat_id, section)
    summary = await llm.generate_section_summary(section, section_data)
    await db.save_section_summary(chat_id, section, summary)
    logger.info(f"[CLOSE] summary saved for section={section}")


async def finish_session(
    chat_id: int,
    context
) -> None:
    """
    Complete the entire session.
    Generate final report and send to patient and clinician.
    """
    from report import generate_report, generate_chat_transcript
    
    session = await db.get_session(chat_id)
    
    # Generate final report
    report_text = await generate_report(chat_id)
    transcript_text = await generate_chat_transcript(chat_id)
    
    # Send to patient
    file = io.BytesIO(report_text.encode("utf-8"))
    file.name = f"case_history_{chat_id}.txt"
    
    try:
        await context.bot.send_document(
            chat_id,
            document=file,
            caption="Here is your compiled history. Please share this with your doctor. 📋",
            parse_mode=None
        )
    except Exception as e:
        print(f"Failed to send report to patient: {e}")
    
    # Send to clinician if configured
    clinician_id = os.getenv("CLINICIAN_TELEGRAM_ID")
    if clinician_id:
        try:
            file.seek(0)
            await context.bot.send_document(
                clinician_id,
                document=file,
                caption=f"Case history completed for session {chat_id}",
                parse_mode=None
            )

            transcript_file = io.BytesIO(transcript_text.encode("utf-8"))
            transcript_file.name = f"conversation_transcript_{chat_id}.txt"
            await context.bot.send_document(
                clinician_id,
                document=transcript_file,
                caption=f"Full conversation transcript for session {chat_id}",
                parse_mode=None
            )
        except Exception as e:
            print(f"Failed to send report to clinician: {e}")
    
    # Send warm closing message
    closing_msg = await llm.generate_closing_message(session.preferred_language)
    await context.bot.send_message(chat_id, closing_msg, parse_mode=None)
    
    # Update session status
    await db.set_session_status(chat_id, "completed")