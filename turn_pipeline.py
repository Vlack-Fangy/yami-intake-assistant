"""
The interview turn pipeline — shared by both transports.

This is the Telegram pipeline, moved here unchanged. It still SENDS through
`context.bot.send_message`; what differs is who is listening. bot.py passes the
real Telegram context and the words go to Telegram; api.py passes a recording
adapter (rest_adapter.RecordingBot) and the same words become the HTTP response.

Keeping the send-based shape rather than returning a reply string is deliberate:
a turn legitimately produces more than one message (the crisis de-escalation
bridge followed by the next interview question, a section-closing message), and
rewriting that into a return value would have changed behaviour. The point of
this move is that behaviour does not change.

Nothing in here may import `telegram` — that is what keeps it transport-neutral.
"""

from utils import get_logger
logger = get_logger("handlers")   # same log file as before the move

from datetime import datetime


def _ts():
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


from utils import (
    build_topic_based_hint, build_filled_fields_summary, get_next_natural_question,
    get_next_topic_after, get_optimistic_filled_fields, mark_fields_filled,
    clear_filled_fields_cache, get_prev_section, set_prev_section, get_planner_logger,
)

planner_log = get_planner_logger()

import asyncio
import os
from collections import defaultdict

import db
import llm
import safety
import crisis
import clinical_state
from data_model import SECTION_CONFIG, SECTION_CONVERSATION_TOPICS, SECTIONS_ORDER
from progress import update_progress_message, compute_completion_percentage, check_milestone_and_update
from transitions import check_section_transition

# Extraction is serialised per session so two turns cannot interleave writes.
EXTRACTION_LOCKS = defaultdict(asyncio.Lock)

# Strong references to in-flight background work.
#
# asyncio only holds a WEAK reference to a task, so a bare create_task() whose
# result nobody keeps can be garbage-collected before it ever runs. Under
# Telegram this was masked: the enclosing buffered-turn task outlived the work.
# Over HTTP the request handler's frame dies as soon as the response is written,
# and extraction silently never happened — fields stayed empty and the interview
# re-asked what the patient had already answered. Holding the reference until
# the task completes is the documented fix, and it protects both transports.
_BACKGROUND: set = set()


def _done(task: asyncio.Task) -> None:
    """Drop the reference, and never let a failure pass in silence.

    Nobody awaits these tasks, so an exception inside one is swallowed until
    the interpreter garbage-collects it — if it is reported at all. That is
    exactly how extraction came to be silently dead over HTTP. Logging here
    turns the next such failure into a line in the log instead of a mystery.
    """
    _BACKGROUND.discard(task)
    if task.cancelled():
        return
    error = task.exception()
    if error is not None:
        logger.error(f"[{_ts()}] BACKGROUND TASK FAILED: {error!r}", exc_info=error)


def _spawn(coro) -> asyncio.Task:
    """create_task, but the task survives until it finishes and reports failures."""
    task = asyncio.create_task(coro)
    _BACKGROUND.add(task)
    task.add_done_callback(_done)
    return task


async def build_recent_context(chat_id: int, limit: int = 6) -> str:
    """Short plain-text transcript of the last few turns for crisis phrasing."""
    try:
        history = await db.get_recent_history(chat_id, limit=limit)
    except Exception:
        return ""
    lines = []
    for m in history:
        role = "Patient" if m.get("role") == "user" else "Assistant"
        lines.append(f"{role}: {m.get('content', '')}")
    return "\n".join(lines)




async def send_clinician_alert(context, text: str) -> None:
    """Notify the on-call clinician of a suicide-risk finding.

    The destination falls back to one the transport supplies. Telegram sets
    CLINICIAN_TELEGRAM_ID and this DMs them; a transport with no chat — REST —
    offers its own channel instead, and the recording adapter routes it to the
    clinician_alerts table. Without that fallback an unset env var silently
    dropped every alert, which is the one failure mode this must not have.
    """
    clinician_id = os.getenv("CLINICIAN_TELEGRAM_ID") or getattr(
        context, "clinician_channel", None
    )
    if not clinician_id or not text:
        logger.error(f"[CRISIS] alert with nowhere to go: {text[:120]}")
        return
    try:
        await context.bot.send_message(clinician_id, text, parse_mode=None)
    except Exception as e:
        logger.warning(f"[CRISIS] failed to notify clinician: {e}")



def looks_like_denial(value) -> bool:
    """True only for a genuine NO — never for silence, refusal or not-knowing.

    The distinction is load-bearing. Real records coded "I didn't notice" and
    "I dont know" as `denied`, and a report then told a clinician there were
    "no significant birth complications or early developmental concerns" for a
    patient who had simply never been told. Not knowing is not denying, and
    declining to answer is not denying either.
    """
    if not value:
        return False
    v = str(value).strip().lower()
    if "declin" in v or "not specified" in v or "dont know" in v or "don't know" in v:
        return False
    if v in {"no", "none", "nil", "denied", "negative", "n/a", "na"}:
        return True
    return v.startswith(("deni", "no ", "none", "never", "nil "))



async def run_crisis_turn(chat_id, patient_text: str, session, context) -> tuple:
    """Drive one crisis-triage turn. Returns True while the safety check continues.

    Moved from handlers._run_crisis_turn. The Telegram caller maps the boolean
    onto its CRISIS_MODE / COLLECTING_HISTORY conversation states; the REST
    caller maps it onto session status. Sends through `context.bot`, so a
    recording adapter captures the same messages a patient would have seen.

    Returns (stay_in_crisis, risk). The risk has to come back with the flag:
    handle_turn clears the assessment when triage ends, so a caller cannot
    look it up afterwards.
    """
    language = getattr(session, "preferred_language", "en") or "en"
    recent = await build_recent_context(chat_id)

    try:
        result = await crisis.handle_turn(
            session_key=str(chat_id),
            patient_text=patient_text,
            language=language,
            recent_context=recent,
        )
    except Exception as e:
        logger.error(f"[CRISIS] handle_turn failed: {e}")
        # Fail safe: never drop a suicidal patient. Send resource + stay in crisis.
        await context.bot.send_message(chat_id, f"I'm here with you. If you need someone right now, please reach out: {safety.CRISIS_HELPLINE}",
            parse_mode=None,
        )
        return True, "unknown"

    await context.bot.send_message(chat_id, result.message, parse_mode=None)

    if result.clinician_alert:
        await send_clinician_alert(context, result.clinician_alert)

    if result.stay_in_crisis:
        # Persist the exchange so the interview transcript/report keeps context.
        try:
            await db.save_message(chat_id, "user", patient_text, session.current_section)
            await db.save_message(chat_id, "assistant", result.message, session.current_section)
        except Exception as e:
            logger.warning(f"[CRISIS] failed to save crisis messages: {e}")
        return True, result.risk

    # Patient confirmed they're safe — resume the interview and ask the next
    # question in the SAME turn (a second message). The warm de-escalation
    # bridge was just sent above as its own assertive message; save only that
    # assistant bridge here — the normal turn pipeline (scheduled below) saves
    # the patient's confirmation and generates the next interview question from
    # it, so we don't double-save the user turn.
    try:
        await db.save_message(chat_id, "assistant", result.message, session.current_section)
    except Exception as e:
        logger.warning(f"[CRISIS] failed to save crisis bridge message: {e}")
    await db.set_session_status(chat_id, "active")
    return False, result.risk



async def run_turn(
    chat_id,
    patient_text: str,
    session,
    context,
    *,
    llm_language: str = None,
) -> None:
    """One interview turn. Sends its reply (or replies) through `context.bot`.

    `llm_language` is the language the MODEL works in, which is not always the
    patient's. Tamil/Punjabi/Bengali sessions run the pipeline in English and
    translate at the transport edge, so the conversation history stays
    English-stable across turns. Defaults to the session's own language, which
    is what Telegram passes.
    """
    llm_language = llm_language or session.preferred_language

    logger.info(f"[{_ts()}] DB — fetching section_fields")

    section = session.current_section

    # Clear optimistic cache if section changed since last turn
    prev_section = get_prev_section(chat_id)
    if prev_section is not None and prev_section != section:
        clear_filled_fields_cache(chat_id)
    set_prev_section(chat_id, section)

    config = SECTION_CONFIG[section]
    section_fields = await db.get_section_fields(chat_id, section)
    
    logger.info(f"[{_ts()}] DB — section_fields done: {section_fields}")
    
    logger.info(f"[{_ts()}] DB — fetching history")
    
    history = await db.get_recent_history(chat_id, limit=14)

    logger.info(f"[{_ts()}] DB — history done: {len(history)} messages")

    last_assistant_question = None
    for message in reversed(history):
        if message.get("role") == "assistant":
            last_assistant_question = message.get("content", "")
            break

    filled_fields_set = get_optimistic_filled_fields(chat_id, section_fields)
    section_topics = SECTION_CONVERSATION_TOPICS.get(section, [])
    last_topic_id = await db.get_last_topic_asked(chat_id)
    topic_hint = build_topic_based_hint(
        section_topics,
        filled_fields_set,
        llm_language,
        last_topic_id=last_topic_id,
    )
    

    fallback_topic_id, next_natural_question = get_next_topic_after(
        section_topics,
        filled_fields_set,
        last_topic_id,
        llm_language,
        last_assistant_question=last_assistant_question,
    )
    
    logger.info(f"[{_ts()}] FALLBACK TOPIC: {fallback_topic_id!r}")
    logger.info(f"[{_ts()}] NEXT Q: {next_natural_question!r}")
    logger.info(f"[{_ts()}] TOPIC — last_topic_id before: {last_topic_id!r}")
    
    if next_natural_question is None:
        next_natural_question = get_next_natural_question(
            section_topics,
            filled_fields_set,
            llm_language,
            last_assistant_question=last_assistant_question,
        )
        fallback_topic_id = None
    

    # If name is still missing in sociodemographic but enough other fields are collected, override to specifically target name so the section doesn't keep collecting optional fields indefinitely.
    if section == "sociodemographic" and "name" not in filled_fields_set:
        other_filled = len([f for f in filled_fields_set if f != "name"])
        min_req = config.get("min_required", 9)
        if other_filled >= min_req - 1:
            next_natural_question = "I don't think I caught your name — could you tell me what to call you?"
            topic_hint = "Basic Background (name still needed)"

    filled_fields_summary = build_filled_fields_summary(section_fields)

    # Split filled fields into "clear" (detailed) and "brief" (needs probe) so the
    # LLM doesn't get conflicting instructions (do_not_ask_labels vs. needs_clarification).
    brief_field_names = set()
    for fname in filled_fields_set:
        fval = section_fields.get(fname)
        if fval and len(fval.strip().split()) <= 4 and fval.strip().lower() not in ("yes", "no", "yeah", "nah", "nope", "yep", "no.", "yes."):
            brief_field_names.add(fname)
    clear_field_names = filled_fields_set - brief_field_names

    needs_clarification_str = ""
    if brief_field_names:
        brief_items = [(f, section_fields.get(f, "")) for f in brief_field_names if section_fields.get(f)]
        parts = [f"{f.replace('_', ' ').title()} (\"{v}\")" for f, v in brief_items[:5]]
        needs_clarification_str = "\nNeeds clarification (brief answer — probe if relevant): " + ", ".join(parts)

    # Build cross-section summary: what's been covered in other sections
    cross_section_str = ""
    try:
        full_store = await db.get_full_info_store(chat_id)
        other_summaries = []
        for s_name, s_fields in full_store.items():
            if s_name == section:
                continue
            filled_names = [f.replace("_", " ").title() for f, v in s_fields.items() if v]
            if filled_names:
                # Only show first 2 fields per section to keep it compact
                other_summaries.append(f"{s_name}: {', '.join(filled_names[:2])}")
        if other_summaries:
            cross_section_str = "\nPreviously covered: " + " | ".join(other_summaries)
    except Exception:
        pass
    
    logger.info(f"[{_ts()}] CONTEXT — filled_fields_set: {filled_fields_set}")
    logger.info(f"[{_ts()}] CONTEXT — topic_hint: {topic_hint}")
    logger.info(f"[{_ts()}] CONTEXT — next_natural_question: {next_natural_question!r}")
    logger.info(f"[{_ts()}] CONTEXT — filled_fields_summary: {filled_fields_summary!r}")
    logger.info(f"[{_ts()}] CONTEXT — needs_clarification: {needs_clarification_str!r}")
    logger.info(f"[{_ts()}] CONTEXT — cross_section_summary: {cross_section_str!r}")

    # ── Clinical state machine ──────────────────────────────
    # Runs for ALL sections. The planner automatically picks the
    # right domains based on section: demographics for sociodemographic,
    # clinical domains for everything else.
    target_domain = ""
    target_lead = ""
    state = clinical_state.get_state(str(chat_id))
    extraction = state.process_patient_message(patient_text)
    target_domain, _, target_lead = state.decide_next(extraction, section=section)
    logger.info(
        f"[{_ts()}] CLINICAL_STATE — turn={state.turn_count} "
        f"section={section} "
        f"domains={state.summary()} "
        f"target={target_domain!r} lead={target_lead!r} "
        f"keyword_leads={extraction['_keyword_lead_count']} "
        f"pending_llm_leads={extraction['_llm_lead_count']}"
    )

    # If a section transition was recorded (by transitions._advance_to_
    # next_section) on a previous turn, fold the acknowledgement into
    # THIS single reply rather than sending a separate bridge message.
    transition_hint = ""
    pending_transition = getattr(state, "pending_transition", None)
    if pending_transition:
        _from_sec, _to_sec = pending_transition
        transition_hint = (
            SECTION_CONFIG.get(_to_sec, {}).get("display_name", "") or _to_sec
        )

    # ── No complaint disclosed: explain ourselves, don't keep asking ──
    # A patient answered "nothing" / "i am alright" and was asked "what
    # brings you here today?" four times, then "can you tell me when
    # this feeling of being okay first started?" — the schema demands a
    # complaint, so it invented one to hang a duration on. Instead we
    # say what we are, where the history goes, and that they're free to
    # have nothing or to stop. Sent ONCE, verbatim.
    # Use the STALL counter, not a counter of our own. Extraction runs in
    # a background task AFTER the reply, so "complaint_1 not in
    # filled_fields_set" is true on every turn including the one where
    # the patient just disclosed — the field simply hasn't been written
    # yet. Counting that fired the orientation at a patient who had just
    # said "I have been procrastinating".
    # stall is computed post-extraction (record_section_progress) and
    # resets to 0 the moment any field lands, so it means what we need:
    # "the patient has given us nothing for N turns".
    state.no_complaint_turns = (
        (state.section_progress.get(section) or {}).get("stall", 0)
    )

    if (
        section == "chief_complaints"
        and "complaint_1" not in filled_fields_set
        and state.no_complaint_turns >= clinical_state.NO_COMPLAINT_ORIENT_AFTER
        and not state.orientation_sent
    ):
        state.orientation_sent = True
        lang = getattr(session, "preferred_language", "en") or "en"
        orientation = clinical_state.ORIENTATION_MESSAGE.get(
            lang, clinical_state.ORIENTATION_MESSAGE["en"]
        )
        logger.warning(
            f"[{_ts()}] ORIENTATION — no complaint after "
            f"{state.no_complaint_turns} turns; explaining instead of re-asking"
        )
        await context.bot.send_message(chat_id, orientation, parse_mode=None)
        await db.save_message(chat_id, "user", patient_text, section)
        await db.save_message(chat_id, "assistant", orientation, section)
        await db.save_section_note(
            chat_id, section,
            "Patient reported no presenting complaint across "
            f"{state.no_complaint_turns} consecutive turns. Purpose and "
            "pause option were explained. Exact words: "
            f"\"{patient_text}\".",
        )
        return

    # ── Planner ────────────────────────────────────────────────────
    # Decide the response STRATEGY before generating. Returns None on
    # any failure, in which case we fall through to exactly the old
    # rule-based behaviour (target_domain/target_lead above).
    strategy = ""
    plan = None
    # Bound outside the try so the planner audit log below can always
    # reference it, even if planning raises.
    # Never offer the planner an MSE item that requires seeing the
    # patient. Over text there is no appearance, no psychomotor
    # activity, no audible speech and no observable affect — asking the
    # patient to self-describe them produces a fabricated exam.
    unfilled_fields = [
        f for f in config["fields"]
        if f not in filled_fields_set
        and f not in clinical_state.NOT_ASSESSABLE_OVER_TEXT
    ]
    try:
        plan = await llm.plan_turn(
            patient_text=patient_text,
            history=history,
            section_display_name=config["display_name"],
            filled_fields_summary=filled_fields_summary,
            unfilled_fields=unfilled_fields,
            detected_language=llm_language,
        )
    except Exception as plan_error:
        logger.warning(f"[{_ts()}] PLANNER — failed, using rule fallback: {plan_error}")

    # Dedicated audit trail → logs/test_logs/planner.log
    try:
        planner_log.info("=" * 90)
        planner_log.info(f"[TURN] chat={chat_id} section={section} turn={state.turn_count}")
        planner_log.info(f"  patient said : {patient_text!r}")
        planner_log.info(f"  still unfilled: {unfilled_fields}")
        planner_log.info(f"  filled so far : {sorted(filled_fields_set)}")
        if plan:
            planner_log.info(
                f"  -> STRATEGY  : {plan.get('strategy')!r}  "
                f"(safety_concern={plan.get('safety_concern')})"
            )
            planner_log.info(f"     target_field: {plan.get('target_field')!r}")
            planner_log.info(f"     rationale   : {plan.get('rationale')!r}")
        else:
            planner_log.info("  -> STRATEGY  : (planner unavailable — rule-based fallback)")
    except Exception as log_err:
        logger.warning(f"[PLANNER_LOG] failed: {log_err}")

    strategy_target = ""
    if plan:
        strategy = plan.get("strategy", "")
        # Only honour a target the planner is actually allowed to ask.
        candidate = (plan.get("target_field") or "").strip()
        if candidate in unfilled_fields:
            strategy_target = candidate

        # Deterministic dependency: you cannot ask how long a complaint
        # has lasted, or how it affects daily life, before the complaint
        # itself exists. The planner asked duration_each while
        # complaint_1 was empty, producing "how long have you been
        # feeling this way?" to a patient who had never named a problem
        # — she replied "which way", correctly.
        # past_history has the SAME dependency shape. treatment,
        # medications and hospitalisations are all consequences of
        # having had a psychiatric episode — asking for their details
        # first invents a history. A real patient answered "no" to
        # diagnoses and "no" to medications and was then asked "What
        # kind of treatments did you receive?"; another gave "Yes" to a
        # compound version and her report told a clinician she "has
        # received prior psychiatric treatment". She had not.
        #
        # Gate on previous_psychiatric_episodes, NOT on diagnoses:
        # a patient can be treated without ever being diagnosed
        # ("Yes but undiagnosed" is a real answer we've seen).
        _PH_DEPENDENTS = (
            "treatment_history", "medications_past", "hospitalizations_past",
        )
        if section == "past_history" and strategy_target in _PH_DEPENDENTS:
            episodes = section_fields.get("previous_psychiatric_episodes")
            if not episodes:
                logger.info(
                    f"[{_ts()}] PLANNER — overriding target {strategy_target!r} "
                    f"-> 'previous_psychiatric_episodes' (no psychiatric history "
                    f"established yet)"
                )
                strategy_target = "previous_psychiatric_episodes"
                if strategy in ("transition", "skip", "wrap_up"):
                    strategy = "ask"
            elif looks_like_denial(episodes):
                # They told us there is no psychiatric history. Its
                # treatment/medication/hospitalisation details are moot —
                # record that plainly instead of interrogating for them.
                logger.info(
                    f"[{_ts()}] PLANNER — previous_psychiatric_episodes denied; "
                    f"marking {_PH_DEPENDENTS} not applicable"
                )
                for dep in _PH_DEPENDENTS:
                    if not section_fields.get(dep):
                        try:
                            await db.save_field(
                                chat_id, section, dep,
                                "not applicable — no previous psychiatric "
                                "history reported",
                            )
                        except Exception as dep_error:
                            logger.warning(f"[{_ts()}] dep mark failed: {dep_error}")
                mark_fields_filled(chat_id, list(_PH_DEPENDENTS))
                strategy_target = ""

        if (
            section == "chief_complaints"
            and "complaint_1" not in filled_fields_set
            and strategy_target != "complaint_1"
        ):
            logger.info(
                f"[{_ts()}] PLANNER — overriding target {strategy_target!r} "
                f"-> 'complaint_1' (no complaint established yet)"
            )
            strategy_target = "complaint_1"
            if strategy in ("transition", "skip", "wrap_up"):
                strategy = "ask"
        # Deterministic safety floor: the planner may RAISE concern (it
        # catches paraphrased self-harm the keyword list misses), but it
        # can never lower it. Hard CRISIS_MODE entry stays keyword-driven
        # by design (crisis.py:30-32) — this only steers the reply.
        if plan.get("safety_concern") and strategy != "de_escalate":
            logger.warning(
                f"[{_ts()}] PLANNER — safety_concern raised; forcing de_escalate "
                f"(was {strategy!r})"
            )
            strategy = "de_escalate"
        # Honouring a refusal in prose is not enough: the topic walk
        # would return to the same field next turn. A real patient
        # refused family questions NINE times and was asked nine more,
        # until she capitulated — consent ground out by attrition.
        # Record the refused field as declined so it is never asked
        # again and the section can advance without it. Leaving a field
        # uncollected is always acceptable; pressuring a patient is not.
        if strategy == "respect_refusal" and strategy_target:
            try:
                await db.save_field(
                    chat_id, section, strategy_target,
                    "patient declined to answer",
                )
                mark_fields_filled(chat_id, [strategy_target])
                logger.warning(
                    f"[{_ts()}] PLANNER — refusal honoured; "
                    f"{strategy_target!r} recorded as declined, will not be re-asked"
                )
            except Exception as refusal_error:
                logger.warning(
                    f"[{_ts()}] PLANNER — failed to record refusal: {refusal_error}"
                )

        logger.info(
            f"[{_ts()}] PLANNER — strategy={strategy!r} "
            f"target_field={plan.get('target_field')!r} "
            f"safety={plan.get('safety_concern')} "
            f"why={plan.get('rationale')!r}"
        )

    try:
        # The literal rather than telegram.constants.ChatAction.TYPING — that
        # enum IS the string "typing" (it subclasses str), so Telegram behaves
        # identically while this module stays free of a telegram import. A
        # non-Telegram transport just no-ops it.
        await context.bot.send_chat_action(chat_id, "typing")
        
        logger.info(f"[{_ts()}] LLM RESPONSE — call start")
        
        unfilled_topic_ids = {
            t["topic_id"] for t in section_topics
            if sum(1 for f in t["fields"] if f in filled_fields_set) < t.get("min_required", 1)
        }
        
        # compute unfilled topics BEFORE LLM call
        unfilled_section_topics = [
            t for t in section_topics
            if sum(1 for f in t["fields"] if f in filled_fields_set) < t.get("min_required", 1)
        ]
        valid_topics_str = "\n".join(
            f"  {t['topic_id']}: covers {', '.join(t['fields'])}"
            for t in unfilled_section_topics
        ) if unfilled_section_topics else "  (all topics covered)"

        bot_response, topic_id = await llm.get_next_response(
            patient_text=patient_text,
            session=session,
            history=history,
            section=section,
            section_display_name=config["display_name"],
            topic_hint=topic_hint,
            next_natural_question=next_natural_question,
            filled_fields_summary=filled_fields_summary,
            detected_language=llm_language,
            valid_topics_with_fields=valid_topics_str,
            filled_field_names=list(clear_field_names),
            needs_clarification=needs_clarification_str,
            cross_section_summary=cross_section_str,
            target_domain=target_domain,
            target_lead=target_lead,
            transition_hint=transition_hint,
            strategy=strategy,
            strategy_target=strategy_target,
        )

        # The transition has now been acknowledged inside this reply;
        # clear the flag so we don't bridge again next turn.
        if pending_transition:
            state.pending_transition = None


        stored_topic = topic_id if (topic_id != "unknown" and topic_id in unfilled_topic_ids) else fallback_topic_id
        await db.set_last_topic_asked(chat_id, stored_topic)

        # NOTE: we deliberately do NOT mark this topic's fields as filled
        # here. Asking is not answering — doing so made the interview
        # march past unanswered questions and forbade the model from
        # clarifying them. Only real extraction results are cached now
        # (see mark_fields_filled in extract_and_store).

        logger.info(f"[{_ts()}] TOPIC — stored_topic: {stored_topic!r}")
        
        logger.info(f"[{_ts()}] Topic: {topic_id!r}")
        logger.info(f"[{_ts()}] RESPONSE: {bot_response!r}")

        # Close the planner trail: strategy -> the reply it produced.
        try:
            planner_log.info(f"  reply sent   : {bot_response!r}")
        except Exception:
            pass
    except Exception as error:
        logger.error(f"LLM call failed: {error}")
        bot_response = "I had a technical issue. Could you please rephrase that?"
        
    logger.info(f"[{_ts()}] TELEGRAM — sending response")

    await context.bot.send_message(chat_id, bot_response, parse_mode=None)
    
    logger.info(f"[{_ts()}] TELEGRAM — sent")
    
    logger.info(f"[{_ts()}] DB — saving messages")
    
    await db.save_message(chat_id, "user", patient_text, section)
    await db.save_message(chat_id, "assistant", bot_response, section)
    
    logger.info(f"[{_ts()}] DB — messages saved")
    
    if strategy == "reorient":
        # Do NOT mine clinical fields out of a turn we could not read as
        # a genuine answer. A real session banked
        #   family_members_list: "Meowma, Meowpa, Meowbro and me"
        # as the patient's actual family, and the report then wrote up a
        # 23-year-old's joke as "possible communication challenges...
        # further clarification warranted regarding the patient's
        # conceptualization of their identity" — a thought-disorder
        # differential manufactured from a gag.
        #
        # We record the words VERBATIM and refuse to interpret them.
        # That cuts both ways on purpose: if this is actually psychosis,
        # a clinician sees the raw utterance and decides. Neither we nor
        # a 14B can tell a joke from a symptom over a text chat, and
        # being wrong in either direction hurts the patient.
        logger.warning(
            f"[{_ts()}] REORIENT — not extracting from uninterpretable turn: "
            f"{patient_text!r}"
        )
        try:
            await db.save_section_note(
                chat_id,
                section,
                "Response could not be taken as a genuine answer and was NOT "
                "clinically interpreted (no fields recorded from it). "
                f"Patient's exact words: \"{patient_text}\". "
                "Flagged for clinician review — not assessed as either "
                "intentional or symptomatic.",
            )
        except Exception as note_error:
            logger.warning(f"[{_ts()}] REORIENT — note failed: {note_error}")
    else:
        logger.info(f"[{_ts()}] EXTRACTION — firing background task")

        _spawn(
            extract_and_store(chat_id, patient_text, last_assistant_question or "", session, context)
        )

        logger.info(f"[{_ts()}] EXTRACTION — task created (running in background)")

    # NOTE: the follow-up advisor pass was removed here. It ran a full
    # LLM generation every turn whose output was written only to a log
    # file and never used by the bot — pure GPU contention on the
    # single shared model, for zero user-facing benefit.

    # NOTE: the standalone LLM lead-extraction pass was removed here.
    # Safety/guilt/contradiction leads are now classified by the SAME
    # cognition pass that extracts the fields (see extract_and_store),
    # so they no longer cost a second forward pass over the identical
    # patient_text. Results still feed the NEXT turn's planner.


async def extract_and_store(
    chat_id: int,
    patient_text: str,
    preceding_question: str,  # the assistant question the patient was answering
    # bot_response: str,      # OLD: was the bot's reply generated AFTER the answer
    session,
    context
) -> None:
    """
    Non-blocking task: extract structured info and check for transitions.
    Called via asyncio.create_task() after patient message.
    """
    
    # str(), not int(). This is shared code now: Telegram chat ids are integers
    # but a REST session id is whatever the caller chose, and int("unittest_user")
    # raises. Because this runs as a fire-and-forget task nobody awaits, that
    # ValueError was swallowed whole — extraction simply never happened over
    # HTTP, fields stayed empty, and the interview re-asked what the patient had
    # already answered. str() accepts both and keys 12345 and "12345" together.
    lock = EXTRACTION_LOCKS[str(chat_id)]
    async with lock:
        try:
            logger.info(f"[{_ts()}] EXTRACT — start for: {patient_text!r}")
            
            section = session.current_section
            config = SECTION_CONFIG[section]
            
            logger.info(f"[{_ts()}] EXTRACT — fetching current fields from DB")
            
            # Get already-filled fields
            section_fields = await db.get_section_fields(chat_id, section)
            already_filled = {k: v for k, v in section_fields.items() if v}
            
            logger.info(f"[{_ts()}] EXTRACT — already_filled: {already_filled}")

            prev_section = None
            prev_unfilled_fields = []
            try:
                idx = SECTIONS_ORDER.index(section)
                if idx > 0:
                    prev_section = SECTIONS_ORDER[idx - 1]
                    prev_fields = await db.get_section_fields(chat_id, prev_section)
                    prev_unfilled_fields = [
                        f for f in SECTION_CONFIG[prev_section]["fields"]
                        if not prev_fields.get(f)
                    ]
            except ValueError:
                pass

            field_list = list(config["fields"])
            current_fields = set(config["fields"])
            prev_fields_set = set(prev_unfilled_fields)

            logger.info(f"[{_ts()}] EXTRACT — LLM extraction call start")
            
            # Extract via LLM
            extraction_result = await llm.extract_info(
                patient_text=patient_text,
                # bot_response=bot_response,  # OLD: the bot's NEXT question (useless context)
                preceding_question=preceding_question,
                section=section,
                field_list=field_list,
                already_filled=already_filled,
                prev_field_list=prev_unfilled_fields,
            )

            logger.info(f"[{_ts()}] EXTRACT — LLM extraction done: {extraction_result}")

            # Save extracted fields, routing each to whichever section it belongs to
            updates = extraction_result.get("updates", {})
            # Cache what was ACTUALLY extracted so the next turn sees it even if
            # this background task's DB write hasn't landed yet. This is the
            # legitimate race the old (harmful) topic pre-marking was aimed at.
            mark_fields_filled(chat_id, [f for f, v in updates.items() if v])
            for field_name, value in updates.items():
                if not value:
                    continue
                if field_name in current_fields:
                    target_section = section
                elif field_name in prev_fields_set:
                    target_section = prev_section
                else:
                    logger.warning(f"[{_ts()}] EXTRACT — dropped unknown field {field_name!r}={value!r} (not in {section} or {prev_section})")
                    continue
                if value == "[declined]":
                    # Patient explicitly refused — save a human-readable note so the clinician sees it, and count it toward completion so the bot moves on.
                    logger.info(f"[{_ts()}] EXTRACT — patient declined: {field_name} (section={target_section})")
                    await db.save_field(chat_id, target_section, field_name, "patient declined to answer")
                    logger.info(f"[{_ts()}] EXTRACT — field saved (declined): {field_name}")
                else:
                    # CLINICAL PRECEDENCE: never let a later negative erase a
                    # positive already on record. Compound review-of-systems
                    # questions cause this constantly:
                    #   "Any stomach problems - nausea, appetite changes...?" -> "yes"
                    #   "Any changes in your bowel habits?"                   -> "no"
                    # and the "no" overwrote the "yes", leaving gi_negative:
                    # denied. In a real session an appetite change — a
                    # depression criterion — was erased exactly this way, and a
                    # "Yes" to lifetime substance use became
                    # substance_use_history: denied. A denial is only the truth
                    # for a field nobody has answered positively yet.
                    existing = (await db.get_section_fields(chat_id, target_section)).get(field_name)
                    if existing and looks_like_denial(value) and not looks_like_denial(existing):
                        logger.warning(
                            f"[{_ts()}] EXTRACT — REFUSED to overwrite positive "
                            f"{field_name}={existing!r} with denial {value!r}"
                        )
                        continue
                    logger.info(f"[{_ts()}] EXTRACT — saving field: {field_name}={value!r} (section={target_section})")
                    await db.save_field(chat_id, target_section, field_name, value)
                    logger.info(f"[{_ts()}] EXTRACT — field saved: {field_name}")
            
            # Save free-text notes
            free_notes = extraction_result.get("free_notes", "")
            if free_notes:
                await db.save_section_note(chat_id, section, free_notes)

            # Clinical leads (safety/guilt/contradiction) now ride along on the
            # SAME cognition pass that extracted the fields. This replaced
            # _extract_leads_for_next_turn(), which spent an extra LLM forward
            # pass re-reading the identical patient_text every turn.
            llm_leads = extraction_result.get("leads") or []
            if llm_leads:
                logger.info(f"[{_ts()}] COGNITION — llm_leads: {llm_leads}")
                clinical_state.get_state(str(chat_id)).store_llm_leads(llm_leads)
            
            # Check for section transition
            logger.info(f"[{_ts()}] EXTRACT — checking section transition")
            section_before = session.current_section
            await check_section_transition(chat_id, session, context)
            session_after = await db.get_session(chat_id)
            section_after = session_after.current_section if session_after else section_before
            if section_after != section_before:
                logger.info(f"[{_ts()}] EXTRACT — SECTION CHANGED: {section_before} → {section_after}")

                try:
                    new_config = SECTION_CONFIG[section_after]
                    new_section_fields = await db.get_section_fields(chat_id, section_after)
                    new_already = {k: v for k, v in new_section_fields.items() if v}
                    unfilled_new = [f for f in new_config["fields"] if f not in new_already]
                    if unfilled_new:
                        recent = await db.get_recent_history(chat_id, limit=14)
                        convo_lines = [
                            f"{'Assistant' if m.get('role') == 'assistant' else 'Patient'}: {m.get('content', '')}"
                            for m in recent
                        ]
                        logger.info(f"[{_ts()}] HARVEST — start for {section_after}, targeting: {unfilled_new}")
                        harvest_result = await llm.harvest_info(
                            conversation_text="\n".join(convo_lines),
                            section=section_after,
                            field_list=unfilled_new,
                            already_filled=new_already,
                        )
                        harvest_updates = harvest_result.get("updates", {})
                        logger.info(f"[{_ts()}] HARVEST — extraction done: {harvest_updates}")
                        allowed_new = set(unfilled_new)
                        for h_field, h_value in harvest_updates.items():
                            if not h_value or h_value == "[declined]":
                                continue
                            if h_field in allowed_new:
                                logger.info(f"[{_ts()}] HARVEST — saving field: {h_field}={h_value!r} (section={section_after})")
                                await db.save_field(chat_id, section_after, h_field, h_value)
                            else:
                                logger.warning(f"[{_ts()}] HARVEST — dropped unknown field {h_field!r}={h_value!r} (not unfilled in {section_after})")
                except Exception as harvest_error:
                    logger.error(f"[{_ts()}] HARVEST — failed: {harvest_error}")
            else:
                logger.info(f"[{_ts()}] EXTRACT — section unchanged: {section_before}")
            logger.info(f"[{_ts()}] EXTRACT — done")
            
            # Update progress tracker
            info_store = await db.get_full_info_store(chat_id)
            
            # Check for milestone
            old_pct = compute_completion_percentage(section, section_fields)
            new_fields = await db.get_section_fields(chat_id, section)
            new_pct = compute_completion_percentage(section, new_fields)
            
            if old_pct < new_pct:
                # Progress changed
                await check_milestone_and_update(
                    chat_id, section, old_pct, new_pct, context, info_store
                )
            else:
                # Regular periodic update (every 5 min)
                await update_progress_message(chat_id, context, info_store, section)
        
        except Exception as e:
            logger.error(f"[{_ts()}] EXTRACT — ERROR: {e}")
            print(f"Error in extract_and_store: {e}")


