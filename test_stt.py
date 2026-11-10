"""Offline SraVaani integration regressions; no model download or GPU required.

Run: python -m unittest test_stt -v
"""
import asyncio
import io
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import AsyncMock, Mock, patch
import wave

import numpy as np

import stt
from sravaani_stt import Pipeline, POLICY
from sravaani_stt.base import Row
from sravaani_stt.segment import Window
from sravaani_stt.stitch import stitch


def silent_wav():
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\0\0" * 16000)
    return output.getvalue()


class TestPipeline(unittest.TestCase):
    def test_silence_does_not_load_asr(self):
        pipeline = Pipeline("/missing/model")
        with patch.object(pipeline, "load", side_effect=AssertionError("ASR loaded")):
            self.assertIsNone(pipeline.transcribe(np.zeros(16000, dtype=np.float32)))
            self.assertIsNone(pipeline.transcribe(np.zeros(0, dtype=np.float32)))

    def test_long_recording_is_bounded_and_reaches_the_end(self):
        pipeline = Pipeline("/unused")
        pipeline.vad = lambda x: np.ones(len(x) // 512, dtype=np.float32)
        backend = Mock()
        backend.transcribe_windows.side_effect = lambda wav, windows, iso: [
            Row(text=f"part{i}", decode_s=0) for i in range(len(windows))
        ]
        with patch.object(pipeline, "load", return_value=backend):
            text = pipeline.transcribe(np.ones(90 * 16000, dtype=np.float32))
        windows = backend.transcribe_windows.call_args.args[1]
        self.assertEqual(POLICY.key(), "29cd002f")
        self.assertGreater(len(windows), 9)
        self.assertTrue(all(b - a <= 10 + 1e-6 for a, b in windows))
        self.assertEqual(windows[0][0], 0)
        self.assertEqual(windows[-1][1], 90)
        self.assertEqual(text.split()[-1], f"part{len(windows) - 1}")
        self.assertIsNone(backend.transcribe_windows.call_args.kwargs["iso"])

    def test_only_overlapping_windows_have_words_deduplicated(self):
        texts = ["नमस्ते आज हम बात करेंगे", "बात करेंगे मन के बारे में"]
        overlapping = [Window(0, 10, forced=True), Window(9.5, 15)]
        self.assertEqual(stitch(overlapping, texts)[0], "नमस्ते आज हम बात करेंगे मन के बारे में")
        separate = [Window(0, 10), Window(11, 15)]
        self.assertEqual(stitch(separate, texts)[0], " ".join(texts))

    def test_incomplete_model_fails_before_loading(self):
        pipeline = Pipeline("/missing/sravaani-model")
        with self.assertRaisesRegex(FileNotFoundError, "Incomplete local SraVaani"):
            pipeline.load()


class TestVoiceAdapter(unittest.IsolatedAsyncioTestCase):
    async def test_same_pipeline_for_english_hinglish_and_no_hint(self):
        pipeline = Mock()
        pipeline.transcribe.return_value = "spoken words"
        with patch.object(stt, "_pipeline", pipeline):
            for lang in ("en", "hinglish", "ta", None):
                result = await stt.transcribe_voice(silent_wav(), lang, "clip.wav")
                self.assertEqual(result, "spoken words")
                self.assertEqual(pipeline.transcribe.call_args.args[0].shape, (16000,))
                self.assertEqual(pipeline.transcribe.call_args.kwargs, {})

    async def test_decoder_failure_removes_upload(self):
        seen = []

        def fail(path, sampling_rate):
            seen.append(path)
            self.assertTrue(Path(path).exists())
            raise ValueError("invalid audio")

        with patch("faster_whisper.audio.decode_audio", side_effect=fail):
            self.assertIsNone(await stt.transcribe_voice(b"bad", filename="recording.WEBM"))
        self.assertEqual(Path(seen[0]).suffix, ".webm")
        self.assertFalse(Path(seen[0]).exists())

    async def test_concurrent_cold_requests_create_one_pipeline_and_serialize(self):
        active = 0
        peak = 0
        guard = threading.Lock()

        def transcribe(wav):
            nonlocal active, peak
            with guard:
                active += 1
                peak = max(peak, active)
            time.sleep(0.02)
            with guard:
                active -= 1
            return "test"

        pipeline = Mock()
        pipeline.transcribe.side_effect = transcribe
        with patch.object(stt, "_pipeline", None), \
             patch.object(stt, "_resolve_device", return_value="cpu"), \
             patch("sravaani_stt.Pipeline", return_value=pipeline) as factory:
            result = await asyncio.gather(*[
                stt.transcribe_voice(silent_wav(), filename="clip.wav") for _ in range(4)
            ])
        self.assertEqual(result, ["test"] * 4)
        factory.assert_called_once()
        self.assertEqual(peak, 1)

    def test_unpinned_gpu_is_not_claimed(self):
        with patch.object(stt, "_DEVICE", "cuda"), \
             patch.object(stt, "_DEVICE_INDEX", ""), \
             patch.dict(os.environ, {}, clear=True):
            self.assertEqual(stt._resolve_device(), "cpu")


class TestVoiceEndpoints(unittest.TestCase):
    """Exercise real routes and SQLite state without generating clinical replies."""

    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        os.environ.setdefault("API_KEY", "synthetic-offline-test-key")
        import api

        cls.api = api
        cls.temp = tempfile.TemporaryDirectory()
        cls.env = patch.dict(os.environ, {"SKIP_MODEL_WARMUP": "1"})
        cls.database = patch.object(api.db, "DATABASE_URL", str(Path(cls.temp.name) / "test.db"))
        cls.env.start()
        cls.database.start()
        headers = {"Authorization": f"Bearer {api.API_KEY}"} if api.API_KEY else {}
        cls.client = TestClient(api.app, headers=headers)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)
        cls.database.stop()
        cls.env.stop()
        cls.temp.cleanup()

    def setUp(self):
        self.sid = "sravaani-unit-test"
        response = self.client.post("/sessions", json={"user_id": self.sid})
        self.assertEqual(response.status_code, 201)
        self.base = f"/sessions/{self.sid}"

    def tearDown(self):
        self.client.delete(self.base)

    def upload(self, path):
        return self.client.post(self.base + path, files={"file": ("clip.wav", silent_wav(), "audio/wav")})

    def test_dictation_returns_text_without_advancing_or_selecting_language(self):
        before = self.client.get(self.base + "/history").json()
        with patch.object(self.api.stt, "transcribe_voice", AsyncMock(return_value="नमस्ते")) as transcribe:
            response = self.upload("/transcribe")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"transcript": "नमस्ते", "language": None})
        self.assertIsNone(transcribe.call_args.kwargs["language"])
        self.assertEqual(self.client.get(self.base + "/history").json(), before)

    def test_voice_message_passes_transcript_into_turn(self):
        self.client.post(self.base + "/messages", json={"message": "en"})
        reply = self.api.BotResponse(bot_message="test reply", input_type="text")
        with patch.object(self.api.stt, "transcribe_voice", AsyncMock(return_value="test words")), \
             patch.object(self.api, "_process_turn", AsyncMock(return_value=reply)) as process:
            response = self.upload("/messages/audio")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["transcript"], "test words")
        self.assertEqual(process.call_args.args[:2], (self.sid, "test words"))

    def test_silence_and_language_precondition_keep_existing_status_codes(self):
        with patch.object(self.api.stt, "transcribe_voice", AsyncMock(return_value=None)) as transcribe:
            self.assertEqual(self.upload("/messages/audio").status_code, 409)
            transcribe.assert_not_called()
            self.assertEqual(self.upload("/transcribe").status_code, 422)


if __name__ == "__main__":
    unittest.main()
