from __future__ import annotations

import json
import socket
import unittest
import urllib.error
from unittest.mock import patch

from esco_skill_batch.extractors import OllamaExtractor, PassthroughExtractor
from esco_skill_batch.normalization import normalize_extracted_skill_mention
from esco_skill_batch.prompt_presets import BIELIK_PL_OLLAMA_PROMPT


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class RequestRecorder:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.request_bodies: list[dict] = []

    def __call__(self, request, timeout):
        body = request.data.decode("utf-8")
        self.request_bodies.append(json.loads(body))
        return FakeResponse(self.payload)


class ExtractorTests(unittest.TestCase):
    def test_passthrough_extractor_deduplicates_pipe_separated_values(self) -> None:
        extractor = PassthroughExtractor("skills_raw")

        mentions = extractor.extract({"skills_raw": "Python | SQL|Python| communication "}, "")

        self.assertEqual([item.text for item in mentions], ["Python", "SQL", "communication"])

    def test_passthrough_extractor_handles_lists(self) -> None:
        extractor = PassthroughExtractor("skills_raw")

        mentions = extractor.extract({"skills_raw": ["Python", "Python", "SQL", ""]}, "")

        self.assertEqual([item.text for item in mentions], ["Python", "SQL"])

    def test_passthrough_extractor_handles_list_of_objects(self) -> None:
        extractor = PassthroughExtractor("gold_skills")

        mentions = extractor.extract(
            {
                "gold_skills": [
                    {"mention": "Python", "esco_uri": "uri:python"},
                    {"mention": "SQL", "esco_uri": "uri:sql"},
                    {"mention": "Python", "esco_uri": "uri:python"},
                ]
            },
            "",
        )

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

    def test_ollama_extractor_uses_custom_system_prompt(self) -> None:
        extractor = OllamaExtractor(
            model="bielik-pl:4.5b",
            base_url="http://127.0.0.1:11434",
            timeout_seconds=10,
            temperature=0.0,
            system_prompt=BIELIK_PL_OLLAMA_PROMPT,
        )
        payload = {"message": {"content": json.dumps({"skills": []})}}
        recorder = RequestRecorder(payload)

        with patch("urllib.request.urlopen", side_effect=recorder):
            extractor.extract({}, "Szukamy osoby z Python i SQL")

        self.assertEqual(
            recorder.request_bodies[0]["messages"][0]["content"],
            BIELIK_PL_OLLAMA_PROMPT,
        )

    def test_ollama_extractor_normalizes_polish_mentions(self) -> None:
        extractor = OllamaExtractor(
            model="qwen2.5:7b",
            base_url="http://127.0.0.1:11434",
            timeout_seconds=10,
            temperature=0.0,
            system_prompt=BIELIK_PL_OLLAMA_PROMPT,
        )
        payload = {
            "message": {
                "content": json.dumps(
                    {
                        "skills": [
                            {"mention": "zna Python", "label": "skill"},
                            {"mention": "programowanie w SQL", "label": "skill"},
                            {"mention": "komunikacyjne", "label": "skill"},
                        ]
                    }
                )
            }
        }

        with patch("urllib.request.urlopen", return_value=FakeResponse(payload)):
            mentions = extractor.extract({"language": "pl"}, "tekst")

        self.assertEqual([item.text for item in mentions], ["Python", "SQL", "umiejetnosci komunikacyjne"])
        self.assertEqual([item.raw_text for item in mentions], ["zna Python", "programowanie w SQL", "komunikacyjne"])

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

    def test_ollama_extractor_reports_missing_model_on_404(self) -> None:
        extractor = OllamaExtractor(
            model="bielik-pl-7b",
            base_url="http://127.0.0.1:11434",
            timeout_seconds=10,
            temperature=0.0,
        )

        error = urllib.error.HTTPError(
            url="http://127.0.0.1:11434/api/chat",
            code=404,
            msg="Not Found",
            hdrs=None,
            fp=None,
        )
        error.read = lambda: json.dumps({"error": "model 'bielik-pl-7b' not found"}).encode("utf-8")

        with patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaisesRegex(RuntimeError, "Check `ollama list`"):
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

    def test_normalize_extracted_skill_mention(self) -> None:
        self.assertEqual(normalize_extracted_skill_mention("zna Python", language="pl"), "Python")
        self.assertEqual(normalize_extracted_skill_mention("programowanie w SQL", language="pl"), "SQL")
        self.assertEqual(normalize_extracted_skill_mention("programming in Python", language="pl"), "programowanie w Pythonie")


if __name__ == "__main__":
    unittest.main()
