"""
BotContext adapter for REST API.
Replaces Telegram's ContextTypes.DEFAULT_TYPE with no-op implementations.

All context.bot.* calls (send_message, send_document, etc.) become no-ops.
Session data is read from DB; reports and transcripts served via REST endpoints.

Usage:
    context = BotContext()
    await check_section_transition(chat_id, session, context)
    await extract_and_store(chat_id, patient_text, bot_response, session, context)
"""


class _NoOpBot:
    """No-op implementation of Telegram bot methods used in core pipeline."""

    async def send_message(self, chat_id, text, **kwargs):
        pass

    async def send_document(self, chat_id, document, **kwargs):
        pass

    async def send_chat_action(self, *args, **kwargs):
        pass

    async def pin_chat_message(self, *args, **kwargs):
        pass

    async def edit_message_text(self, *args, **kwargs):
        pass


class BotContext:
    """
    REST API context adapter.

    Drop-in replacement for Telegram context wherever context is passed
    through the core pipeline (transitions, extraction, progress).

    Telegram-specific actions (pinning messages, sending documents) are
    silently ignored. All persistent state goes through the DB instead.
    """

    def __init__(self):
        self.bot = _NoOpBot()