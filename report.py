"""
Report generation for psychiatric history-taking bot.
Produces clinical case history documents for patients and clinicians.
"""

from datetime import datetime
import io
import os
import db
from data_model import SECTIONS_ORDER, SECTION_CONFIG


async def generate_report(chat_id: int) -> str:
    """
    Generate comprehensive psychiatric case history report.
    Called when session completes or via /report command.
    
    Uses clinical language and formatting.
    """
    
    session = await db.get_session(chat_id)
    if not session:
        return "Session not found."
    
    summaries = await db.get_all_summaries(chat_id)
    fields = await db.get_full_info_store(chat_id)
    
    report_lines = [
        "=" * 60,
        "PSYCHIATRIC CASE HISTORY REPORT",
        "=" * 60,
        f"Session ID: {chat_id}",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Language: {session.preferred_language}",
        "-" * 60,
        ""
    ]
    
    # Add each section
    for section in SECTIONS_ORDER:
        config = SECTION_CONFIG[section]
        section_title = config["display_name"].upper()
        
        report_lines.append(section_title)
        report_lines.append("-" * 40)
        
        # Use summary if available, else raw fields
        summary = summaries.get(section)
        if summary:
            report_lines.append(summary)
        else:
            # Fallback to raw fields
            section_fields = fields.get(section, {})
            if section_fields:
                for field_name, value in section_fields.items():
                    if value:
                        display_name = field_name.replace("_", " ").title()
                        report_lines.append(f"{display_name}: {value}")
            else:
                report_lines.append("(No data collected)")
        
        report_lines.append("")
    
    # Footer
    report_lines.append("=" * 60)
    report_lines.append(f"Report generated: {datetime.now().isoformat()}")
    report_lines.append("End of Report")
    report_lines.append("=" * 60)
    
    return "\n".join(report_lines)


async def generate_chat_transcript(chat_id: int) -> str:
    """Generate a plain-text transcript of the full patient-bot conversation."""
    session = await db.get_session(chat_id)
    if not session:
        return "Session not found."

    messages = await db.get_full_conversation_history(chat_id)

    lines = [
        "=" * 80,
        "PATIENT-BOT CONVERSATION TRANSCRIPT",
        "=" * 80,
        f"Session ID: {chat_id}",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Language: {session.preferred_language}",
        "-" * 80,
        "",
    ]

    if not messages:
        lines.append("(No conversation history found)")
    else:
        for msg in messages:
            role = "Patient" if msg["role"] == "user" else "Assistant"
            timestamp = msg.get("timestamp") or ""
            section = msg.get("section") or ""
            header_parts = [role]
            if timestamp:
                header_parts.append(timestamp)
            if section:
                header_parts.append(f"section={section}")

            header = " | ".join(header_parts)
            lines.append(f"[{header}]")
            original = msg.get("original_content")
            if original:
                lines.append(f"  Original ({session.preferred_language}): {original}")
                lines.append(f"  English: {msg['content']}")
            else:
                lines.append(msg["content"])
            lines.append("")

    lines.append("=" * 80)
    lines.append("End of Transcript")
    lines.append("=" * 80)
    return "\n".join(lines)


async def send_report_handler(update, context):
    """
    /report command handler.
    Generates and sends current report to patient.
    """
    import io
    
    chat_id = update.effective_chat.id
    
    try:
        report_text = await generate_report(chat_id)
        
        # Send as file (often too long for single message)
        file = io.BytesIO(report_text.encode("utf-8"))
        file.name = f"case_history_{chat_id}.txt"
        
        await context.bot.send_document(
            chat_id,
            document=file,
            caption="Here is your compiled history so far. 📋",
            parse_mode=None
        )
    
    except Exception as e:
        print(f"Failed to generate/send report: {e}")
        await update.message.reply_text(
            "Sorry, I couldn't generate your report right now. Please try again later.",
            parse_mode=None
        )


async def send_clinician_pause_summary(chat_id: int, context) -> bool:
    """Send the current compiled summary to the clinician when a session is paused."""
    clinician_id = os.getenv("CLINICIAN_TELEGRAM_ID")
    if not clinician_id:
        return False

    try:
        session = await db.get_session(chat_id)
        if not session:
            return False

        report_text = await generate_report(chat_id)
        summary_file = io.BytesIO(report_text.encode("utf-8"))
        summary_file.name = f"case_history_pause_summary_{chat_id}.txt"

        await context.bot.send_document(
            clinician_id,
            document=summary_file,
            caption=f"Session paused: current summary for session {chat_id}",
            parse_mode=None,
        )
        return True
    except Exception as error:
        print(f"Failed to send pause summary to clinician: {error}")
        return False