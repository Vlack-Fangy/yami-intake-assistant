"""
Yami REST API — Unit Test Suite
================================
Usage:
  python test_api.py                    # all tests (requires LLM + waits 60s)
  python test_api.py --no-llm           # skip LLM-dependent tests
  python test_api.py --no-wait          # skip extraction wait
  python -m unittest test_api -v        # verbose unittest output
"""



import hashlib
import io
import unittest
import requests as _requests          # raw, unauthenticated — used by TestAuth
import time
import sys
import os
import wave
os.environ["no_proxy"] = "localhost,127.0.0.1"
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

BASE       = os.getenv("API_BASE", "http://localhost:5000")
NO_LLM     = "--no-llm"  in sys.argv
NO_WAIT    = "--no-wait" in sys.argv
EXTRACTION_WAIT = 60

# The API requires `Authorization: Bearer <key>`. Rather than touch all ~60 call
# sites below, rebind the module name to a Session carrying the header — every
# existing requests.get/post/... then authenticates with no change. TestAuth uses
# _requests to reach the API without a key on purpose.
API_KEY = os.getenv("API_KEY", "")
_client = _requests.Session()
if API_KEY:
    _client.headers["Authorization"] = f"Bearer {API_KEY}"
requests = _client


def silent_wav(seconds: float = 0.4) -> bytes:
    """A valid WAV containing nothing but silence.

    Enough to prove the audio endpoints parse an upload and reach the model
    without shipping a speech fixture into the repo. The VAD finds no speech, so
    the expected answer is 422 — which is itself the behaviour worth pinning.
    """
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * int(16000 * seconds))
    return buf.getvalue()


# ── Base ───────────────────────────────────────────────────────

class YamiTestCase(unittest.TestCase):
    """Base — fresh session before each test, deleted after."""

    USER_ID = "unittest_user"

    def setUp(self):
        r = requests.post(f"{BASE}/sessions", json={"user_id": self.USER_ID})
        self.assertEqual(r.status_code, 201, f"setUp failed: {r.text}")

    def tearDown(self):
        requests.delete(f"{BASE}/sessions/{self.USER_ID}")

    def url(self, path: str = "") -> str:
        return f"{BASE}/sessions/{self.USER_ID}{path}"

    def send(self, message: str) -> dict:
        r = requests.post(self.url("/messages"), json={"message": message})
        self.assertEqual(r.status_code, 200, f"send failed: {r.text}")
        return r.json()

    def select_language(self, lang: str = "en") -> dict:
        d = self.send(lang)
        self.assertEqual(d["input_type"], "text")
        return d


# ── 1. Session Lifecycle ───────────────────────────────────────

class TestSessionLifecycle(YamiTestCase):

    def test_create_returns_201(self):
        r = requests.post(f"{BASE}/sessions", json={"user_id": self.USER_ID})
        self.assertEqual(r.status_code, 201)

    def test_create_returns_mcq(self):
        r = requests.post(f"{BASE}/sessions", json={"user_id": self.USER_ID})
        d = r.json()
        self.assertEqual(d["input_type"], "mcq")
        # 6, not 3: Tamil, Punjabi and Bengali joined en/hi/hinglish.
        self.assertEqual(len(d["options"]), 6)
        self.assertIn("session_id", d)
        self.assertIn("bot_message", d)

    def test_create_language_null(self):
        r = requests.get(self.url())
        self.assertIsNone(r.json()["preferred_language"])

    def test_create_status_active(self):
        r = requests.get(self.url())
        self.assertEqual(r.json()["status"], "active")

    def test_get_session_fields(self):
        d = requests.get(self.url()).json()
        for key in ["session_id", "status", "current_section",
                    "preferred_language", "started_at", "last_active"]:
            self.assertIn(key, d)

    def test_create_resets_existing(self):
        self.select_language("en")
        requests.post(f"{BASE}/sessions", json={"user_id": self.USER_ID})
        self.assertIsNone(requests.get(self.url()).json()["preferred_language"])

    def test_delete_removes_session(self):
        requests.delete(self.url())
        self.assertEqual(requests.get(self.url()).status_code, 404)

    def test_delete_returns_deleted_true(self):
        r = requests.delete(self.url())
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["deleted"])

    def test_404_all_endpoints(self):
        endpoints = [
            ("GET",    "/sessions/ghost_xyz"),
            ("GET",    "/sessions/ghost_xyz/history"),
            ("GET",    "/sessions/ghost_xyz/fields"),
            ("GET",    "/sessions/ghost_xyz/fields/sociodemographic"),
            ("GET",    "/sessions/ghost_xyz/progress"),
            ("GET",    "/sessions/ghost_xyz/progress/text"),
            ("GET",    "/sessions/ghost_xyz/report"),
            ("GET",    "/sessions/ghost_xyz/transcript"),
            ("GET",    "/sessions/ghost_xyz/summaries"),
            ("POST",   "/sessions/ghost_xyz/messages"),
            ("POST",   "/sessions/ghost_xyz/pause"),
            ("POST",   "/sessions/ghost_xyz/resume"),
            ("DELETE", "/sessions/ghost_xyz"),
        ]
        for method, path in endpoints:
            r = requests.request(
                method, f"{BASE}{path}",
                json={"message": "x"} if method == "POST" else None
            )
            self.assertEqual(r.status_code, 404, f"Expected 404: {method} {path}")


# ── 2. Language Selection ──────────────────────────────────────

class TestLanguageSelection(YamiTestCase):

    def test_invalid_language_returns_mcq(self):
        d = self.send("klingon")
        self.assertEqual(d["input_type"], "mcq")
        # 6, not 3: Tamil, Punjabi and Bengali joined en/hi/hinglish.
        self.assertEqual(len(d["options"]), 6)

    def test_valid_english(self):
        d = self.send("en")
        self.assertEqual(d["input_type"], "text")
        self.assertEqual(d["section"], "sociodemographic")
        self.assertIn("name", d["bot_message"].lower())

    def test_valid_hindi(self):
        d = self.send("hi")
        self.assertEqual(d["input_type"], "text")

    def test_valid_hinglish(self):
        d = self.send("hinglish")
        self.assertEqual(d["input_type"], "text")

    def test_case_insensitive(self):
        d = self.send("EN")
        self.assertEqual(d["input_type"], "text")

    def test_language_persisted(self):
        self.send("en")
        self.assertEqual(requests.get(self.url()).json()["preferred_language"], "en")

    def test_update_language_valid(self):
        self.select_language("en")
        r = requests.patch(self.url("/language"), json={"message": "hi"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["preferred_language"], "hi")
        self.assertEqual(requests.get(self.url()).json()["preferred_language"], "hi")

    def test_update_language_invalid_returns_400(self):
        self.select_language("en")
        r = requests.patch(self.url("/language"), json={"message": "latin"})
        self.assertEqual(r.status_code, 400)


# ── 3. Pause / Resume ─────────────────────────────────────────

class TestPauseResume(YamiTestCase):

    def test_pause_returns_paused(self):
        r = requests.post(self.url("/pause"))
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "paused")

    def test_pause_reflected_in_get(self):
        requests.post(self.url("/pause"))
        self.assertEqual(requests.get(self.url()).json()["status"], "paused")

    def test_resume_returns_active(self):
        requests.post(self.url("/pause"))
        r = requests.post(self.url("/resume"))
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "active")

    def test_resume_without_pause_returns_400(self):
        r = requests.post(self.url("/resume"))
        self.assertEqual(r.status_code, 400)

    def test_message_while_paused_shows_status(self):
        self.select_language("en")
        requests.post(self.url("/pause"))
        d = self.send("hello")
        self.assertEqual(d["session_status"], "paused")

    def test_pause_resume_cycle(self):
        requests.post(self.url("/pause"))
        self.assertEqual(requests.get(self.url()).json()["status"], "paused")
        requests.post(self.url("/resume"))
        self.assertEqual(requests.get(self.url()).json()["status"], "active")


# ── 4. Data Endpoints ─────────────────────────────────────────

class TestDataEndpoints(YamiTestCase):

    def setUp(self):
        super().setUp()
        self.select_language("en")

    def test_fields_initially_empty(self):
        self.assertIsInstance(requests.get(self.url("/fields")).json(), dict)

    def test_section_fields_valid(self):
        r = requests.get(self.url("/fields/sociodemographic"))
        self.assertEqual(r.status_code, 200)
        d = r.json()
        self.assertEqual(d["section"], "sociodemographic")
        self.assertIn("fields", d)

    def test_section_fields_invalid_returns_400(self):
        r = requests.get(self.url("/fields/made_up"))
        self.assertEqual(r.status_code, 400)

    def test_progress_has_8_sections(self):
        # 8, not 9. mse is deliberately disabled (data_model.py): 11 of its 22
        # fields need the patient to be SEEN, and the bot was inventing them.
        self.assertEqual(len(requests.get(self.url("/progress")).json()["sections"]), 8)

    def test_mse_section_is_disabled(self):
        names = [s["section"] for s in requests.get(self.url("/progress")).json()["sections"]]
        self.assertNotIn("mse", names, "mse must stay disabled — it cannot be done over text")

    def test_progress_required_keys(self):
        d = requests.get(self.url("/progress")).json()
        for key in ["current_section", "sections", "total_estimated_mins_remaining", "snapshot"]:
            self.assertIn(key, d)

    def test_progress_section_keys(self):
        sections = requests.get(self.url("/progress")).json()["sections"]
        for s in sections:
            for key in ["section", "status", "completion_pct",
                        "filled_count", "min_required", "estimated_mins_remaining"]:
                self.assertIn(key, s)

    def test_progress_valid_status_values(self):
        sections = requests.get(self.url("/progress")).json()["sections"]
        for s in sections:
            self.assertIn(s["status"], ["in_progress", "pending", "complete"])

    def test_progress_current_section_in_progress(self):
        d = requests.get(self.url("/progress")).json()
        current = next(s for s in d["sections"] if s["section"] == d["current_section"])
        self.assertEqual(current["status"], "in_progress")

    def test_progress_text_returns_markdown(self):
        r = requests.get(self.url("/progress/text"))
        self.assertEqual(r.status_code, 200)
        text = r.json()["progress_text"]
        self.assertIn("Your Progress", text)
        self.assertIn("⏳", text)

    def test_history_structure(self):
        d = requests.get(self.url("/history")).json()
        self.assertIn("session_id", d)
        self.assertIn("message_count", d)
        self.assertIn("messages", d)
        self.assertEqual(d["session_id"], self.USER_ID)
        self.assertEqual(d["message_count"], len(d["messages"]))

    def test_history_message_fields(self):
        d = requests.get(self.url("/history")).json()
        for msg in d["messages"]:
            for key in ["role", "content", "section", "timestamp"]:
                self.assertIn(key, msg)
            self.assertIn(msg["role"], ["user", "assistant"])

    def test_history_nonempty_after_language(self):
        count = requests.get(self.url("/history")).json()["message_count"]
        self.assertGreater(count, 0)

    def test_summaries_empty_initially(self):
        d = requests.get(self.url("/summaries")).json()
        self.assertIn("summaries", d)
        self.assertIsInstance(d["summaries"], dict)

    def test_report_contains_header(self):
        report = requests.get(self.url("/report")).json()["report"]
        self.assertIn("PSYCHIATRIC CASE HISTORY REPORT", report)
        self.assertIn(self.USER_ID, report)

    def test_transcript_contains_header(self):
        t = requests.get(self.url("/transcript")).json()["transcript"]
        self.assertIn("PATIENT-BOT CONVERSATION TRANSCRIPT", t)
        self.assertIn(self.USER_ID, t)


# ── 5. Admin ──────────────────────────────────────────────────

class TestAdmin(YamiTestCase):

    def test_health_status_ok(self):
        r = requests.get(f"{BASE}/health")
        self.assertEqual(r.status_code, 200)

    def test_health_fields(self):
        d = requests.get(f"{BASE}/health").json()
        for key in ["status", "model_loaded", "db_reachable", "api_version"]:
            self.assertIn(key, d)

    def test_health_db_reachable(self):
        self.assertTrue(requests.get(f"{BASE}/health").json()["db_reachable"])

    def test_list_sessions_structure(self):
        r = requests.get(f"{BASE}/admin/sessions")
        self.assertEqual(r.status_code, 200)
        d = r.json()
        self.assertIn("sessions", d)
        self.assertIn("count", d)
        self.assertIsInstance(d["sessions"], list)

    def test_list_sessions_contains_current(self):
        ids = [s["session_id"] for s in
               requests.get(f"{BASE}/admin/sessions").json()["sessions"]]
        self.assertIn(self.USER_ID, ids)

    def test_list_sessions_status_filter(self):
        requests.post(self.url("/pause"))
        d = requests.get(f"{BASE}/admin/sessions?status=paused").json()
        self.assertEqual(d["status_filter"], "paused")
        for s in d["sessions"]:
            self.assertEqual(s["status"], "paused")

    def test_list_sessions_pagination(self):
        d = requests.get(f"{BASE}/admin/sessions?limit=1&offset=0").json()
        self.assertLessEqual(len(d["sessions"]), 1)

    def test_list_sessions_session_fields(self):
        sessions = requests.get(f"{BASE}/admin/sessions").json()["sessions"]
        for s in sessions:
            for key in ["session_id", "status", "current_section",
                        "preferred_language", "started_at", "last_active"]:
                self.assertIn(key, s)


# ── 6. LLM Flow ───────────────────────────────────────────────

@unittest.skipIf(NO_LLM, "Skipping LLM tests (--no-llm)")
class TestLLMFlow(YamiTestCase):

    def setUp(self):
        super().setUp()
        self.select_language("en")

    def test_message_returns_text_type(self):
        d = self.send("Harshal Rathi")
        self.assertEqual(d["input_type"], "text")

    def test_response_nonempty(self):
        d = self.send("Harshal")
        self.assertTrue(len(d["bot_message"]) > 0)

    def test_response_has_session_status(self):
        d = self.send("Harshal")
        self.assertEqual(d["session_status"], "active")

    def test_history_grows_after_send(self):
        before = requests.get(self.url("/history")).json()["message_count"]
        self.send("Harshal")
        after = requests.get(self.url("/history")).json()["message_count"]
        self.assertGreater(after, before)

    def test_no_repeated_questions(self):
        messages = ["Harshal Rathi", "23 Male", "Researcher at IIT Delhi",
                    "Single", "Hindu, Hindi speaking"]
        responses = [self.send(m)["bot_message"] for m in messages]
        self.assertEqual(len(set(responses)), len(responses),
                         "Repeated question detected — chaining broken")

    def test_topics_advance(self):
        r1 = self.send("Harshal Rathi")
        r2 = self.send("23 Male")
        r3 = self.send("Researcher")
        self.assertNotEqual(r1["bot_message"], r2["bot_message"])
        self.assertNotEqual(r2["bot_message"], r3["bot_message"])

    def test_section_stays_consistent(self):
        messages = ["Harshal Rathi", "23 Male", "Researcher"]
        for msg in messages:
            d = self.send(msg)
            self.assertEqual(d["section"], "sociodemographic")


# ── 7. Extraction ─────────────────────────────────────────────

@unittest.skipIf(NO_LLM, "Skipping extraction tests (--no-llm)")
class TestExtraction(YamiTestCase):
    """
    Sends messages once in setUpClass, waits for extraction,
    then runs all assertion tests against the results.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        requests.post(f"{BASE}/sessions", json={"user_id": cls.USER_ID})
        requests.post(f"{BASE}/sessions/{cls.USER_ID}/messages", json={"message": "en"})

        for msg in ["Harshal Rathi", "23 Male", "Researcher at IIT Delhi",
                    "Single", "Hindu, Hindi"]:
            requests.post(f"{BASE}/sessions/{cls.USER_ID}/messages",
                          json={"message": msg})

        wait = 3 if NO_WAIT else EXTRACTION_WAIT
        print(f"\n  ⏳ Waiting {wait}s for extraction...", flush=True)
        time.sleep(wait)

    @classmethod
    def tearDownClass(cls):
        requests.delete(f"{BASE}/sessions/{cls.USER_ID}")

    def setUp(self):
        pass  # setUpClass owns the session — don't reset between tests

    def tearDown(self):
        pass  # session lives until tearDownClass

    def _socio(self) -> dict:
        r = requests.get(f"{BASE}/sessions/{self.USER_ID}/fields/sociodemographic")
        return r.json().get("fields", {})

    def test_age_extracted(self):
        self.assertTrue(bool(self._socio().get("age")),
                        "Field 'age' not extracted")

    def test_sex_extracted(self):
        self.assertTrue(bool(self._socio().get("sex")),
                        "Field 'sex' not extracted")

    def test_filled_count_nonzero(self):
        d = requests.get(f"{BASE}/sessions/{self.USER_ID}/progress").json()
        socio = next(s for s in d["sections"] if s["section"] == "sociodemographic")
        self.assertGreater(socio["filled_count"], 0)

    def test_completion_pct_nonzero(self):
        d = requests.get(f"{BASE}/sessions/{self.USER_ID}/progress").json()
        socio = next(s for s in d["sections"] if s["section"] == "sociodemographic")
        self.assertGreater(socio["completion_pct"], 0)

    def test_snapshot_updated(self):
        snap = requests.get(
            f"{BASE}/sessions/{self.USER_ID}/progress"
        ).json()["snapshot"]["sociodemographic"]
        self.assertGreater(snap.get("filled_count", 0), 0)

    def test_fields_endpoint_nonempty(self):
        fields = requests.get(
            f"{BASE}/sessions/{self.USER_ID}/fields"
        ).json()
        self.assertGreater(len(fields.get("sociodemographic", {})), 0,
                           "No fields extracted")


# ── 9. Authentication ──────────────────────────────────────────

@unittest.skipUnless(API_KEY, "auth disabled (API_KEY empty)")
class TestAuth(unittest.TestCase):
    """The key gate. Uses _requests directly so nothing carries the header."""

    def test_no_key_rejected(self):
        r = _requests.post(f"{BASE}/sessions", json={"user_id": "authtest"})
        self.assertEqual(r.status_code, 401)

    def test_wrong_key_rejected(self):
        r = _requests.get(f"{BASE}/admin/sessions",
                          headers={"Authorization": "Bearer nope"})
        self.assertEqual(r.status_code, 401)

    def test_wrong_scheme_rejected(self):
        """Basic with the right value is still wrong — the scheme is part of it."""
        r = _requests.get(f"{BASE}/admin/sessions",
                          headers={"Authorization": f"Basic {API_KEY}"})
        self.assertEqual(r.status_code, 401)

    def test_bare_key_without_scheme_rejected(self):
        r = _requests.get(f"{BASE}/admin/sessions", headers={"Authorization": API_KEY})
        self.assertEqual(r.status_code, 401)

    def test_correct_key_accepted(self):
        r = _requests.get(f"{BASE}/admin/sessions",
                          headers={"Authorization": f"Bearer {API_KEY}"})
        self.assertEqual(r.status_code, 200)

    def test_challenge_header_present(self):
        """401 must say how to authenticate, not just refuse."""
        r = _requests.get(f"{BASE}/admin/sessions")
        self.assertEqual(r.headers.get("WWW-Authenticate"), "Bearer")

    def test_health_is_exempt(self):
        self.assertEqual(_requests.get(f"{BASE}/health").status_code, 200)

    def test_ui_route_is_removed(self):
        self.assertEqual(_requests.get(
            f"{BASE}/ui", headers={"Authorization": f"Bearer {API_KEY}"}
        ).status_code, 404)

    def test_openapi_is_exempt(self):
        self.assertEqual(_requests.get(f"{BASE}/openapi.json").status_code, 200)

    def test_patient_data_is_not_exempt(self):
        """The endpoints that return clinical content must never be open."""
        for path in ("/sessions/x/report", "/sessions/x/history",
                     "/sessions/x/transcript", "/sessions/x/fields"):
            self.assertEqual(_requests.get(f"{BASE}{path}").status_code, 401, path)


# ── 10. Message IDs ────────────────────────────────────────────

class TestMessageIds(YamiTestCase):
    """message_id is the handle feedback attaches to."""

    def test_language_reply_has_message_id(self):
        self.assertIsInstance(self.select_language("en").get("message_id"), int)

    def test_history_exposes_message_ids(self):
        self.select_language("en")
        msgs = requests.get(self.url("/history")).json()["messages"]
        self.assertTrue(msgs)
        for m in msgs:
            self.assertIsInstance(m.get("message_id"), int)

    def test_history_ids_ascend(self):
        """Ordered by id, not timestamp — both messages of a turn tie on time."""
        self.select_language("en")
        ids = [m["message_id"] for m in requests.get(self.url("/history")).json()["messages"]]
        self.assertEqual(ids, sorted(ids))

    def test_ids_unique(self):
        self.select_language("en")
        ids = [m["message_id"] for m in requests.get(self.url("/history")).json()["messages"]]
        self.assertEqual(len(ids), len(set(ids)))


# ── 11. Feedback ───────────────────────────────────────────────

class TestFeedback(YamiTestCase):

    def setUp(self):
        super().setUp()
        self.mid = self.select_language("en")["message_id"]

    def rate(self, rating="up", review=None, mid=None):
        return requests.post(
            self.url(f"/messages/{mid or self.mid}/feedback"),
            json={"rating": rating, "review": review},
        )

    def test_rate_returns_201_and_review_id(self):
        r = self.rate("up")
        self.assertEqual(r.status_code, 201)
        self.assertIsInstance(r.json()["review_id"], int)

    def test_thumb_alone_is_valid(self):
        """Review text is optional — requiring it would cost most ratings."""
        self.assertIsNone(self.rate("down").json()["review"])

    def test_review_stored(self):
        self.assertEqual(self.rate("down", "asked two things").json()["review"],
                         "asked two things")

    def test_reviews_append_not_replace(self):
        first  = self.rate("up").json()["review_id"]
        second = self.rate("down").json()["review_id"]
        self.assertNotEqual(first, second)
        self.assertEqual(len(requests.get(self.url("/feedback")).json()), 2)

    def test_patch_adds_review_to_existing(self):
        rid = self.rate("up").json()["review_id"]
        d = requests.patch(self.url(f"/feedback/{rid}"),
                           json={"review": "clear and kind"}).json()
        self.assertEqual(d["review"], "clear and kind")
        self.assertEqual(d["rating"], "up", "adding a review must not blank the rating")
        self.assertIsNotNone(d["updated_at"])

    def test_patch_can_flip_rating(self):
        rid = self.rate("up").json()["review_id"]
        self.assertEqual(
            requests.patch(self.url(f"/feedback/{rid}"), json={"rating": "down"}).json()["rating"],
            "down")

    def test_delete_removes_review(self):
        rid = self.rate("up").json()["review_id"]
        self.assertEqual(requests.delete(self.url(f"/feedback/{rid}")).status_code, 200)
        self.assertEqual(len(requests.get(self.url("/feedback")).json()), 0)

    def test_per_message_listing(self):
        self.rate("up"); self.rate("down")
        self.assertEqual(len(requests.get(self.url(f"/messages/{self.mid}/feedback")).json()), 2)

    # ── refusals ──

    def test_cannot_rate_a_patient_message(self):
        """Rating the patient's own words would poison the dataset."""
        self.send("My name is Test")
        user_ids = [m["message_id"]
                    for m in requests.get(self.url("/history")).json()["messages"]
                    if m["role"] == "user"]
        self.assertTrue(user_ids, "no user message to test against")
        self.assertEqual(self.rate("up", mid=user_ids[0]).status_code, 422)

    def test_invalid_rating_rejected(self):
        self.assertEqual(self.rate("maybe").status_code, 422)

    def test_unknown_message_404(self):
        self.assertEqual(self.rate("up", mid=999999).status_code, 404)

    def test_unknown_review_404(self):
        self.assertEqual(requests.patch(self.url("/feedback/999999"),
                                        json={"review": "x"}).status_code, 404)
        self.assertEqual(requests.delete(self.url("/feedback/999999")).status_code, 404)

    def test_other_session_cannot_touch_our_feedback(self):
        """Session scoping is what stops one id reaching into another transcript."""
        rid = self.rate("up").json()["review_id"]
        other = "unittest_other"
        requests.post(f"{BASE}/sessions", json={"user_id": other})
        try:
            self.assertEqual(requests.post(
                f"{BASE}/sessions/{other}/messages/{self.mid}/feedback",
                json={"rating": "up"}).status_code, 404)
            self.assertEqual(requests.patch(
                f"{BASE}/sessions/{other}/feedback/{rid}",
                json={"review": "hijack"}).status_code, 404)
            self.assertEqual(requests.delete(
                f"{BASE}/sessions/{other}/feedback/{rid}").status_code, 404)
            self.assertEqual(len(requests.get(self.url("/feedback")).json()), 1)
        finally:
            requests.delete(f"{BASE}/sessions/{other}")

    def test_feedback_dies_with_session(self):
        self.rate("up")
        requests.delete(self.url())
        requests.post(f"{BASE}/sessions", json={"user_id": self.USER_ID})
        self.assertEqual(requests.get(self.url("/feedback")).json(), [])


# ── 12. Voice ──────────────────────────────────────────────────

class TestVoice(YamiTestCase):
    """Contract and refusals. Silence is used so no speech fixture is needed."""

    def post_audio(self, path, data=b"", name="clip.wav"):
        return requests.post(self.url(path), files={"file": (name, data, "audio/wav")})

    def test_transcribe_works_before_language_is_chosen(self):
        """Dictation needs no language; the send-a-turn endpoint does.

        /transcribe changes nothing, so there is nothing for a missing language
        to corrupt — SraVaani needs no language hint. /messages/audio commits a turn,
        and the turn pipeline needs a language, so it refuses with 409. The two
        differ on purpose; 422 here is "no speech in the silence", not a refusal.
        """
        self.assertEqual(self.post_audio("/transcribe", silent_wav()).status_code, 422)

    def test_audio_turn_requires_language_first(self):
        # Choosing between six languages by voice asks ASR to do the thing it is
        # worst at; the MCQ exists for a reason.
        self.assertEqual(
            self.post_audio("/messages/audio", silent_wav()).status_code, 409)

    def test_empty_upload_rejected(self):
        self.select_language("en")
        self.assertEqual(self.post_audio("/transcribe", b"").status_code, 400)

    def test_silence_is_422_not_an_error(self):
        """Silence and mis-taps are normal; "" must never enter the interview."""
        self.select_language("en")
        self.assertEqual(self.post_audio("/transcribe", silent_wav()).status_code, 422)

    def test_unknown_session_404(self):
        r = requests.post(f"{BASE}/sessions/ghost_xyz/transcribe",
                          files={"file": ("c.wav", silent_wav(), "audio/wav")})
        self.assertEqual(r.status_code, 404)

    def test_dictation_does_not_advance_the_interview(self):
        """The whole point of /transcribe: it changes nothing."""
        self.select_language("en")
        before = requests.get(self.url("/history")).json()["message_count"]
        self.post_audio("/transcribe", silent_wav())
        self.assertEqual(requests.get(self.url("/history")).json()["message_count"], before)


# ── 13. Crisis triage ──────────────────────────────────────────

@unittest.skipIf(NO_LLM, "crisis triage needs the model")
class TestCrisisTriage(YamiTestCase):
    """The REST path used to answer every crisis message with one canned string
    that never assessed risk and never reached a clinician. It now runs the same
    staged triage as Telegram."""

    # The old canned reply, so a regression back to it is caught by name.
    CANNED = "I hear that you're going through something very difficult right now"

    def setUp(self):
        super().setUp()
        self.select_language("en")

    def test_disclosure_enters_crisis(self):
        d = self.send("I have been thinking about killing myself")
        self.assertEqual(d["session_status"], "crisis")

    def test_disclosure_reports_a_risk_level(self):
        d = self.send("I have been thinking about killing myself")
        self.assertIn(d["risk"], ("low", "moderate", "high", "imminent"))

    def test_reply_is_triage_not_the_old_canned_string(self):
        d = self.send("I have been thinking about killing myself")
        self.assertNotIn(self.CANNED, d["bot_message"],
                         "regressed to the pre-triage canned response")

    def test_triage_advances_rather_than_repeating(self):
        first = self.send("I have been thinking about killing myself")["bot_message"]
        second = self.send("yes, I have thought about how")["bot_message"]
        self.assertNotEqual(first, second,
                            "triage repeated itself instead of advancing a stage")

    def test_end_my_life_is_crisis_not_a_pause_request(self):
        """The two keyword sets overlap and the wrong order is dangerous.

        SOFT_STOP_PHRASES has "i want to end" (end the session); CRISIS_KEYWORDS
        has "end my life". "I want to end my life" matches BOTH. Checked in the
        wrong order the API paused the session and replied "No problem. Your
        progress is saved." to a suicidal disclosure — no triage, no alert.
        """
        for phrase in ("I want to end my life", "I want to end it all"):
            with self.subTest(phrase=phrase):
                requests.post(f"{BASE}/sessions", json={"user_id": self.USER_ID})
                self.select_language("en")
                d = self.send(phrase)
                self.assertEqual(d["session_status"], "crisis",
                                 f"{phrase!r} must trigger triage, not a pause")

    def test_genuine_pause_requests_still_pause(self):
        """The fix must not swallow real stop requests into crisis."""
        for phrase in ("can we take a break", "I want to end this for today"):
            with self.subTest(phrase=phrase):
                requests.post(f"{BASE}/sessions", json={"user_id": self.USER_ID})
                self.select_language("en")
                self.send(phrase)
                self.assertEqual(requests.get(self.url()).json()["status"], "paused")

    def test_risk_is_null_outside_crisis(self):
        self.assertIsNone(self.send("I am 30 and I work as a teacher")["risk"])

    def test_crisis_exchange_is_persisted(self):
        self.send("I have been thinking about killing myself")
        roles = [m["role"] for m in requests.get(self.url("/history")).json()["messages"]]
        self.assertIn("user", roles)
        self.assertIn("assistant", roles)


@unittest.skipIf(NO_LLM, "alerts are raised by crisis triage")
class TestClinicianAlerts(YamiTestCase):

    def setUp(self):
        super().setUp()
        self.select_language("en")

    def open_alerts(self):
        return [a for a in requests.get(f"{BASE}/admin/alerts").json()
                if a["session_id"] == self.USER_ID]

    def test_crisis_raises_an_alert(self):
        """Telegram DMs these. Over REST they must land somewhere a human looks."""
        self.send("I want to kill myself tonight")
        self.assertTrue(self.open_alerts(), "no clinician alert was recorded")

    def test_alert_carries_risk_and_session(self):
        self.send("I want to kill myself tonight")
        a = self.open_alerts()[0]
        self.assertEqual(a["session_id"], self.USER_ID)
        self.assertIsNotNone(a["message"])
        self.assertIsNone(a["acknowledged_at"])

    def test_acknowledge_clears_it_from_the_open_list(self):
        self.send("I want to kill myself tonight")
        alert_id = self.open_alerts()[0]["alert_id"]
        r = requests.post(f"{BASE}/admin/alerts/{alert_id}/acknowledge")
        self.assertEqual(r.status_code, 200)
        self.assertIsNotNone(r.json()["acknowledged_at"])
        self.assertNotIn(alert_id, [a["alert_id"] for a in self.open_alerts()])
        # still retrievable when explicitly asked for
        every = requests.get(f"{BASE}/admin/alerts?acknowledged=true").json()
        self.assertIn(alert_id, [a["alert_id"] for a in every])

    def test_acknowledge_unknown_returns_404(self):
        self.assertEqual(
            requests.post(f"{BASE}/admin/alerts/999999/acknowledge").status_code, 404)

    def test_alerts_die_with_the_session(self):
        """They quote the patient, so a hard delete has to take them too."""
        self.send("I want to kill myself tonight")
        self.assertTrue(self.open_alerts())
        requests.delete(self.url())
        self.assertEqual(self.open_alerts(), [])


class TestSoftStopShared(YamiTestCase):
    """Both transports now use safety.is_soft_stop_message. Telegram's local
    copy was a stale 10-phrase list missing most of these."""

    def setUp(self):
        super().setUp()
        self.select_language("en")

    def test_phrases_the_old_telegram_list_missed(self):
        for phrase in ("can we take a break", "let's wrap up", "some other time"):
            with self.subTest(phrase=phrase):
                requests.post(f"{BASE}/sessions", json={"user_id": self.USER_ID})
                self.select_language("en")
                self.send(phrase)
                self.assertEqual(
                    requests.get(self.url()).json()["status"], "paused",
                    f"{phrase!r} should pause the session")

    def test_menopause_does_not_pause(self):
        """The word-boundary fix: 'pause' must not fire inside 'menopause'."""
        self.send("my mother went through menopause early")
        self.assertNotEqual(requests.get(self.url()).json()["status"], "paused")


# ── Runner ─────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.argv = [sys.argv[0]] + [
        a for a in sys.argv[1:]
        if a not in ("--no-llm", "--no-wait")
    ]
    unittest.main(verbosity=2)
