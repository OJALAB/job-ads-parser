from __future__ import annotations

import json
import socket
import unittest
import urllib.error
from unittest.mock import patch

from esco_skill_batch.extractors import OllamaExtractor, PassthroughExtractor


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class ExtractorTests(unittest.TestCase):
    def test_passthrough_extractor_deduplicates_pipe_separated_values(self) -> None:
        extractor = PassthroughExtractor("skills_raw")

        mentions = extractor.extract({"skills_raw": "Python | SQL|Python| communication "}, "")

        self.assertEqual([item.text for item in mentions], ["Python", "SQL", "communication"])

    def test_passthrough_extractor_handles_lists(self) -> None:
        extractor = PassthroughExtractor("skills_raw")

        mentions = extractor.extract({"skills_raw": ["Python", "Python", "SQL", ""]}, "")

        self.assertEqual([item.text for item in mentions], ["Python", "SQL"])

    def test_ollama_extractor_parses_json_response(self) -> None:
        extractor = OllamaExtractor(
            model="qwen3:14b",
            base_url="http://127.0.0.1:11434",
            timeout_seconds=10,
            temperature=0.0,
        )
        payload = {
            "message": {
                "content": json.dumps(
                    {
                        "skills": [
                            {"mention": "Python", "label": "skill"},
                            {"mention": "SQL", "label": "skill"},
                            {"mention": "Python", "label": "skill"},
                        ]
                    }
                )
            }
        }

        with patch("urllib.request.urlopen", return_value=FakeResponse(payload)):
            mentions = extractor.extract({}, "Need Python and SQL")

        self.assertEqual([item.text for item in mentions], ["Python", "SQL"])

    def test_ollama_extractor_wraps_connection_errors(self) -> None:
        extractor = OllamaExtractor(
            model="qwen3:14b",
            base_url="http://127.0.0.1:11434",
            timeout_seconds=10,
            temperature=0.0,
        )

        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("connection refused")):
            with self.assertRaisesRegex(RuntimeError, "Could not reach Ollama"):
                extractor.extract({}, "Need Python")

    def test_ollama_extractor_wraps_timeouts(self) -> None:
        extractor = OllamaExtractor(
            model="qwen3:14b",
            base_url="http://127.0.0.1:11434",
            timeout_seconds=10,
            temperature=0.0,
        )

        with patch("urllib.request.urlopen", side_effect=socket.timeout("timed out")):
            with self.assertRaisesRegex(RuntimeError, "timed out"):
                extractor.extract({}, "Need Python")


if __name__ == "__main__":
    unittest.main()
