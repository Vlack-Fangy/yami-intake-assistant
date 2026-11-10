"""
Progress tracker module for psychiatric history bot.
DB-backed progress snapshots — no Telegram dependency.
"""

from datetime import datetime
import db
from data_model import SECTION_CONFIG, SECTIONS_ORDER, MILESTONE_THRESHOLDS, PROGRESS_UPDATE_MIN_INTERVAL_SECONDS


def compute_time_estimate(section: str, fields: dict) -> int:
    """
    Compute estimated minutes remaining for a section.
    Formula: est_remaining = baseline * (1 - completion_fraction)
    """
    config = SECTION_CONFIG[section]
    baseline = config["baseline_minutes"]
    min_required = config["min_required"]

    filled = sum(1 for v in fields.values() if v)
    completion = min(filled / min_required, 1.0)
    remaining = baseline * (1.0 - completion)

    return max(1, round(remaining))


def compute_completion_percentage(section: str, fields: dict) -> int:
    """Compute completion percentage for a section (0-100)."""
    config = SECTION_CONFIG[section]
    min_required = config["min_required"]

    filled = sum(1 for v in fields.values() if v)
    pct = min(int((filled / min_required) * 100), 100)

    return pct


def build_progress_text(info_store: dict, current_section: str) -> str:
    """
    Build markdown-formatted progress for frontend display.
    Standard markdown — works across most renderers.
    """
    lines = ["📋 **Your Progress**", ""]
    total_remaining_mins = 0

    for section in SECTIONS_ORDER:
        config = SECTION_CONFIG[section]
        display_name = config["display_name"]
        section_fields = info_store.get(section, {})
        filled = sum(1 for v in section_fields.values() if v)
        min_required = config["min_required"]
        completion_pct = min(int((filled / min_required) * 100), 100)

        if section == current_section:
            remaining_mins = compute_time_estimate(section, section_fields)
            total_remaining_mins += remaining_mins
            if remaining_mins < 1:
                time_str = "Almost done!"
            else:
                time_str = f"about {remaining_mins} min left"
            lines.append(f"⏳ **{display_name}** — {time_str}")

        elif completion_pct >= 100:
            lines.append(f"✅ {display_name} — done")

        else:
            baseline = config["baseline_minutes"]
            total_remaining_mins += baseline
            lines.append(f"🔜 {display_name} — about {baseline} min")

    lines.append("")
    if total_remaining_mins < 1:
        total_str = "Almost done!"
    else:
        total_str = f"About {total_remaining_mins} minutes remaining in total"
    lines.append(f"⏱ {total_str}")

    return "\n".join(lines)


def build_progress_summary(info_store: dict, current_section: str) -> dict:
    """
    Build structured progress dict for API JSON responses.
    """
    sections = []
    total_remaining_mins = 0

    for section in SECTIONS_ORDER:
        config = SECTION_CONFIG[section]
        section_fields = info_store.get(section, {})
        filled = sum(1 for v in section_fields.values() if v)
        min_required = config["min_required"]
        completion_pct = min(int((filled / min_required) * 100), 100)

        if section == current_section:
            status = "in_progress"
            remaining_mins = compute_time_estimate(section, section_fields)
            total_remaining_mins += remaining_mins
        elif completion_pct >= 100:
            status = "complete"
            remaining_mins = 0
        else:
            status = "pending"
            remaining_mins = config["baseline_minutes"]
            total_remaining_mins += remaining_mins

        sections.append({
            "section": section,
            "display_name": config["display_name"],
            "status": status,
            "completion_pct": completion_pct,
            "filled_count": filled,
            "min_required": min_required,
            "estimated_mins_remaining": remaining_mins,
        })

    return {
        "current_section": current_section,
        "sections": sections,
        "total_estimated_mins_remaining": total_remaining_mins,
    }


async def should_update_progress(session_id: str) -> bool:
    """
    Rate limit guard — check if enough time passed since last update.
    PROGRESS_UPDATE_MIN_INTERVAL_SECONDS = 0 means always passes for now.
    """
    last_update = await db.get_last_progress_update(session_id)

    if last_update is None:
        return True

    elapsed = (datetime.now() - last_update).total_seconds()
    return elapsed >= PROGRESS_UPDATE_MIN_INTERVAL_SECONDS


async def update_progress_message(
    chat_id: int,
    context,           # kept for signature compatibility — unused in REST
    info_store: dict,
    current_section: str,
    force: bool = False,
) -> None:
    """
    Write progress snapshot to DB for current section.
    Rate-limited unless force=True.
    context param retained so callers need no changes.
    """
    session_id = str(chat_id)

    if not force:
        if not await should_update_progress(session_id):
            return

    section_fields = info_store.get(current_section, {})
    pct = compute_completion_percentage(current_section, section_fields)
    filled = sum(1 for v in section_fields.values() if v)
    await db.update_progress_snapshot(chat_id, current_section, pct, filled)
    await db.set_last_progress_update(session_id)


async def send_initial_progress_message(chat_id: int, context) -> int:
    """
    Send the initial (empty) progress message and pin it.
    Telegram-specific. Returns the message ID of the pinned message.
    """
    empty_store = {section: {} for section in SECTIONS_ORDER}
    text = build_progress_text(empty_store, SECTIONS_ORDER[0])

    message = await context.bot.send_message(
        chat_id=chat_id,
        text=text,
        parse_mode=None,
    )

    message_id = message.message_id

    try:
        await context.bot.pin_chat_message(
            chat_id=chat_id,
            message_id=message_id,
            disable_notification=True,
        )
    except Exception as e:
        print(f"Failed to pin progress message: {e}")

    await db.set_progress_message_id(chat_id, message_id)
    return message_id


async def check_milestone_and_update(
    chat_id: int,
    section: str,
    old_pct: int,
    new_pct: int,
    context,
    info_store: dict,
) -> bool:
    """
    Check if a milestone threshold was crossed.
    If so, update progress snapshot in DB.
    Returns True if an update was made.
    """
    for threshold in MILESTONE_THRESHOLDS:
        if old_pct < threshold <= new_pct:
            await update_progress_message(chat_id, context, info_store, section, force=True)
            return True

    return False