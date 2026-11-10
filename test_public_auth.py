"""Offline checks for the public release's authentication defaults."""
import os
os.environ.setdefault("API_KEY", "synthetic-offline-test-key")
os.environ["LLM_LAZY_GPU"] = "0"
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
import api

class PublicAuthTests(unittest.IsolatedAsyncioTestCase):
    async def call(self, path, key, token=""):
        req = SimpleNamespace(url=SimpleNamespace(path=path), method="GET",
                              headers={"authorization": token}, client=None)
        next_call = AsyncMock(return_value="accepted")
        with patch.object(api, "API_KEY", key):
            result = await api.require_api_key(req, next_call)
        return result, next_call

    async def test_missing_server_key_fails_closed(self):
        result, downstream = await self.call("/history", "")
        self.assertEqual(result.status_code, 503)
        downstream.assert_not_called()

    async def test_wrong_key_is_rejected(self):
        result, downstream = await self.call("/history", "expected", "Bearer wrong")
        self.assertEqual(result.status_code, 401)
        downstream.assert_not_called()

    async def test_valid_key_is_accepted(self):
        result, downstream = await self.call("/history", "expected", "Bearer expected")
        self.assertEqual(result, "accepted")
        downstream.assert_awaited_once()

    async def test_health_is_public_but_prefix_is_not(self):
        result, _ = await self.call("/health", "")
        self.assertEqual(result, "accepted")
        result, _ = await self.call("/health/private", "")
        self.assertEqual(result.status_code, 503)
