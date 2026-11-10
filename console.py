from dotenv import load_dotenv
load_dotenv()
import asyncio
import db
from data_model import SECTION_CONFIG, SECTION_CONVERSATION_TOPICS
from utils import build_topic_based_hint, build_filled_fields_summary, get_next_natural_question
import llm
from transitions import check_section_transition



CHAT_ID = 99999  # fake chat_id

class FakeContext:
    class bot:
        @staticmethod
        async def send_message(chat_id, text, **kwargs):
            print(f"\nYami: {text}\n")
        @staticmethod
        async def send_chat_action(*args, **kwargs):
            pass
        @staticmethod
        async def pin_chat_message(*args, **kwargs):
            pass
        @staticmethod
        async def edit_message_text(*args, **kwargs):
            pass

async def main():
    await db.init_db()
    await db.create_session(CHAT_ID)
    await db.set_preferred_language(CHAT_ID, "en")
    
    # test_fields = {
    #     'age': '23', 'sex': 'male', 'education': 'PhD',
    #     'occupation': 'researcher', 'ses': 'stable',
    #     'marital_status': 'single', 'religion': 'hindu',
    #     'residence': 'delhi', 'mother_tongue': 'hindi',
    #     'address': 'vasant kunj', 'source_of_referral': 'friend',
    #     'informant': 'self'
    # }
    # for field, value in test_fields.items():
    #     await db.save_field(CHAT_ID, 'sociodemographic', field, value)
    # print("[TEST] Seeded all sociodemographic fields — topic_hint should be empty on first message")

    await llm.warmup_model()

    context = FakeContext()
    print("Yami: Hello! What's your name?")

    while True:
        patient_text = await asyncio.to_thread(input, "You: ")
        patient_text = patient_text.strip()
        if patient_text.lower() in ["quit", "exit"]:
            break

        session = await db.get_session(CHAT_ID)
        section = session.current_section
        config = SECTION_CONFIG[section]
        section_fields = await db.get_section_fields(CHAT_ID, section)
        history = await db.get_recent_history(CHAT_ID, limit=14)

        filled_fields_set = {f for f, v in section_fields.items() if v}
        section_topics = SECTION_CONVERSATION_TOPICS.get(section, [])
        topic_hint = build_topic_based_hint(section_topics, filled_fields_set, "en")
        next_natural_question = get_next_natural_question(section_topics, filled_fields_set, "en")
        filled_fields_summary = build_filled_fields_summary(section_fields)

        bot_response, topic_id = await llm.get_next_response(
            patient_text=patient_text,
            session=session,
            history=history,
            section=section,
            section_display_name=config["display_name"],
            topic_hint=topic_hint,
            next_natural_question=next_natural_question,
            filled_fields_summary=filled_fields_summary,
            detected_language="en",
        )

        print(f"\nYami: {bot_response}\n")
        print(f"[topic: {topic_id}]")

        await db.save_message(CHAT_ID, "user", patient_text, section)
        await db.save_message(CHAT_ID, "assistant", bot_response, section)

        # extraction in background
        asyncio.create_task(
            _extract(CHAT_ID, patient_text, bot_response, session, context)
        )

async def _extract(chat_id, patient_text, bot_response, session, context):
    from handlers import extract_and_store
    await extract_and_store(chat_id, patient_text, bot_response, session, context)

if __name__ == "__main__":
    asyncio.run(main())