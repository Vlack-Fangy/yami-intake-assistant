"""
The connector: lets the REST API drive the Telegram interview pipeline.

`turn_pipeline` still talks by SENDING — `context.bot.send_message(...)` — because
that is how it was written and moving it unchanged is the whole point. Over HTTP
there is nobody to send to, so this supplies a `context` that writes the messages
down instead of transmitting them. The API then returns what was recorded.

The consequence worth stating: REST is not a reimplementation that has to be kept
in step. It runs the same code, so it inherits every future pipeline improvement
and every fix, automatically.

Replaces the old `bot_context.BotContext`, which discarded messages rather than
capturing them — fine when the API had its own pipeline, useless now that the
reply itself arrives this way.
"""

from typing import Optional

from utils import get_logger

logger = get_logger("api")


class RecordingBot:
    """Stands in for `context.bot`, keeping what would have been sent.

    Patient replies and clinician alerts are told apart by their destination:
    the pipeline addresses the patient with the session's own chat_id and the
    clinician with CLINICIAN_TELEGRAM_ID. That is a property of the code as
    written, not a convention invented here, so no extra flag is needed.
    """

    def __init__(self, session_id: str):
        self.session_id = str(session_id)
        self.replies: list[str] = []      # patient-facing, in order
        self.alerts: list[str] = []       # clinician notifications
        self.documents: int = 0           # reports; REST serves these at /report

    async def send_message(self, chat_id=None, text=None, **kwargs):
        # The pipeline calls this both positionally and by keyword.
        if text is None:
            text = kwargs.get("text")
        if not text:
            return
        if str(chat_id) == self.session_id:
            self.replies.append(str(text))
        else:
            logger.error(f"[ALERT] session={self.session_id}: {text}")
            self.alerts.append(str(text))

    async def send_document(self, *args, **kwargs):
        # The report is a REST endpoint, not a push. Count it so a caller can
        # tell the interview finished, and drop the payload.
        self.documents += 1

    # Telegram chrome with no REST equivalent. No-ops, as in the adapter this
    # replaces — a typing indicator and a pinned progress message have no
    # meaning in a request/response exchange.
    async def send_chat_action(self, *args, **kwargs):
        pass

    async def pin_chat_message(self, *args, **kwargs):
        pass

    async def edit_message_text(self, *args, **kwargs):
        pass

    async def delete_message(self, *args, **kwargs):
        pass

    async def get_file(self, *args, **kwargs):
        raise NotImplementedError("REST uploads audio directly; see api.transcribe_only")


class RestContext:
    """The `context` object the pipeline expects."""

    # Where clinician alerts go when there is no Telegram to DM. The pipeline
    # reads this as a fallback for CLINICIAN_TELEGRAM_ID; any value that is not
    # the session's own id works, since that is how RecordingBot tells a
    # clinician notification from a patient reply.
    clinician_channel = "__rest_clinician__"

    def __init__(self, session_id: str):
        self.bot = RecordingBot(session_id)

    # ── what the API reads back ──

    @property
    def replies(self) -> list[str]:
        return self.bot.replies

    @property
    def alerts(self) -> list[str]:
        return self.bot.alerts

    def joined_reply(self, separator: str = "\n\n") -> Optional[str]:
        """Every patient-facing message of this turn, as one string.

        A turn can legitimately produce more than one: the crisis de-escalation
        bridge followed by the next interview question, or a section-closing
        message. Telegram shows them as separate bubbles; HTTP has one response,
        so they are joined. Order is preserved — reading the bridge before the
        question is the point of sending both.
        """
        parts = [r for r in self.bot.replies if r and r.strip()]
        return separator.join(parts) if parts else None


class _RestMessage:
    """The `update.message` surface the crisis path touches."""

    def __init__(self, text: str, bot: RecordingBot, chat_id: str):
        self.text = text
        self._bot = bot
        self._chat_id = chat_id

    async def reply_text(self, text, **kwargs):
        await self._bot.send_message(self._chat_id, text)


class _RestChat:
    def __init__(self, chat_id):
        self.id = chat_id


class RestUpdate:
    """A stand-in for telegram.Update.

    Deliberately minimal: an AST scan of the turn and crisis paths shows only
    `effective_chat.id`, `message.text` and `message.reply_text` are ever
    reached. Anything else would be a Telegram-only path that REST should not
    be calling in the first place, so leaving it absent makes that a loud
    AttributeError rather than a silent no-op.
    """

    def __init__(self, session_id: str, text: str, context: RestContext):
        self.effective_chat = _RestChat(session_id)
        self.message = _RestMessage(text, context.bot, session_id)
