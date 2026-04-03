from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from dataclasses import asdict

from esco_skill_batch.normalization import normalize_extracted_skill_mention
from esco_skill_batch.prompt_presets import DEFAULT_OLLAMA_PROMPT
from esco_skill_batch.text_utils import unique_preserve_order
from esco_skill_batch.types import SkillMention


class PassthroughExtractor:
    def __init__(self, mentions_field: str) -> None:
        self.mentions_field = mentions_field

    def extract(self, record: dict, _: str) -> list[SkillMention]:
        raw = record.get(self.mentions_field, [])
        if raw is None:
            return []
        if isinstance(raw, str):
            values = [part.strip() for part in raw.split("|") if part.strip()]
        elif isinstance(raw, list):
            values = []
            for item in raw:
                if isinstance(item, dict):
                    candidate = str(item.get("mention", "")).strip()
                else:
                    candidate = str(item).strip()
                if candidate:
                    values.append(candidate)
        else:
            raise ValueError(f"Unsupported mentions field type: {type(raw).__name__}")
        return [SkillMention(text=value, label="skill") for value in unique_preserve_order(values)]


class OllamaExtractor:
    def __init__(
        self,
        model: str,
        base_url: str,
        timeout_seconds: int,
        temperature: float,
        system_prompt: str | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature
        self.system_prompt = system_prompt or DEFAULT_OLLAMA_PROMPT

    def extract(self, record: dict, text: str) -> list[SkillMention]:
        schema = {
            "type": "object",
            "properties": {
                "skills": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "mention": {"type": "string"},
                            "label": {"type": "string"},
                        },
                        "required": ["mention"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["skills"],
            "additionalProperties": False,
        }

        payload = {
            "model": self.model,
            "stream": False,
            "format": schema,
            "options": {"temperature": self.temperature},
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": text},
            ],
        }
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url=f"{self.base_url}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except (TimeoutError, socket.timeout) as exc:
            raise RuntimeError(
                "Ollama request timed out. Increase --ollama-timeout-seconds or use a smaller model."
            ) from exc
        except urllib.error.HTTPError as exc:
            error_detail = ""
            try:
                payload = json.loads(exc.read().decode("utf-8"))
                error_detail = str(payload.get("error", "")).strip()
            except Exception:
                error_detail = ""

            if exc.code == 404:
                detail = f" Details: {error_detail}" if error_detail else ""
                raise RuntimeError(
                    f"Ollama returned 404 for model `{self.model}` at {self.base_url}. "
                    f"Check `ollama list` and set the exact model name with OLLAMA_MODEL.{detail}"
                ) from exc

            detail = f" Details: {error_detail}" if error_detail else ""
            raise RuntimeError(
                f"Ollama request failed with HTTP {exc.code} for model `{self.model}` at {self.base_url}.{detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Could not reach Ollama at {self.base_url}: {exc}") from exc

        try:
            message_content = raw["message"]["content"]
            parsed = json.loads(message_content)
        except (KeyError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Ollama response is not valid JSON: {raw}") from exc

        skills = parsed.get("skills", [])
        mentions: list[SkillMention] = []
        seen: set[str] = set()
        language = str(record.get("language", "") or "").strip().lower() or None
        for item in skills:
            raw_mention = str(item.get("mention", "")).strip()
            if not raw_mention:
                continue
            mention = normalize_extracted_skill_mention(raw_mention, language=language)
            normalized = mention.casefold()
            if normalized in seen:
                continue
            seen.add(normalized)
            mentions.append(
                SkillMention(
                    text=mention,
                    raw_text=raw_mention if raw_mention != mention else None,
                    label=str(item.get("label", "skill")),
                )
            )
        return mentions


class GLiNERExtractor:
    def __init__(self, model_name: str, threshold: float, labels: list[str] | None = None) -> None:
        try:
            from gliner import GLiNER
        except ImportError as exc:
            raise RuntimeError("GLiNER extractor requires `gliner`. Install with `.[gliner]`.") from exc

        self.model = GLiNER.from_pretrained(model_name)
        self.threshold = threshold
        self.labels = labels or ["skill", "transversal skill", "tool", "technology", "framework"]

    def extract(self, record: dict, text: str) -> list[SkillMention]:
        raw_entities = self.model.predict_entities(text, labels=self.labels, threshold=self.threshold)
        mentions: list[SkillMention] = []
        seen: set[str] = set()
        for entity in raw_entities:
            mention = str(entity.get("text", "")).strip()
            if not mention:
                continue
            normalized = mention.casefold()
            if normalized in seen:
                continue
            seen.add(normalized)
            mentions.append(
                SkillMention(
                    text=mention,
                    label=str(entity.get("label", "skill")),
                    score=float(entity["score"]) if "score" in entity else None,
                    start=int(entity["start"]) if "start" in entity else None,
                    end=int(entity["end"]) if "end" in entity else None,
                )
            )
        return mentions


def mentions_to_json(mentions: list[SkillMention]) -> list[dict]:
    return [asdict(mention) for mention in mentions]
