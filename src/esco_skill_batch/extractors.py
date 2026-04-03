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


class HFTokenClassificationExtractor:
    def __init__(
        self,
        model_name: str,
        aggregation_strategy: str = "simple",
        entity_labels: list[str] | None = None,
        device: int = -1,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForTokenClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "HF token classification extractor requires `transformers` and `torch`. Install with `.[hf]`."
            ) from exc

        self.torch = torch
        self.model_name = model_name
        self.aggregation_strategy = aggregation_strategy
        self.entity_labels = {label.casefold() for label in (entity_labels or [])}
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.id2label = {
            int(key): str(value)
            for key, value in getattr(self.model.config, "id2label", {}).items()
        }
        max_length = int(getattr(self.model.config, "max_position_embeddings", 512))
        self.max_length = max_length if max_length > 0 else 512

        if device >= 0 and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{device}")
        else:
            self.device = torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()

        if not getattr(self.tokenizer, "is_fast", False):
            raise RuntimeError(
                f"HF token classification extractor requires a fast tokenizer with offsets. Model `{model_name}` does not provide one."
            )

    def extract(self, record: dict, text: str) -> list[SkillMention]:
        encoded = self.tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        offset_mapping = encoded.pop("offset_mapping")[0].tolist()
        attention_mask = encoded["attention_mask"][0].tolist()
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        with self.torch.no_grad():
            outputs = self.model(**encoded)
            logits = outputs.logits[0]
            probabilities = self.torch.softmax(logits, dim=-1)
            predicted_ids = self.torch.argmax(logits, dim=-1).tolist()
            token_scores = probabilities.max(dim=-1).values.tolist()

        language = str(record.get("language", "") or "").strip().lower() or None
        token_offsets: list[tuple[int, int]] = []
        token_labels: list[str] = []
        filtered_scores: list[float] = []
        for mask, offset, predicted_id, score in zip(attention_mask, offset_mapping, predicted_ids, token_scores):
            if not mask:
                continue
            start, end = int(offset[0]), int(offset[1])
            token_offsets.append((start, end))
            token_labels.append(self.id2label.get(int(predicted_id), "O"))
            filtered_scores.append(float(score))

        return _decode_hf_token_predictions(
            text=text,
            token_offsets=token_offsets,
            token_labels=token_labels,
            token_scores=filtered_scores,
            language=language,
            entity_labels=self.entity_labels,
        )


def _normalize_hf_entity_label(value: object) -> str:
    label = str(value or "").strip()
    if not label:
        return ""
    if label.startswith(("B-", "I-")):
        label = label[2:]
    return label.casefold()


def _clean_hf_mention_text(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = text.replace(" ##", "")
    text = text.replace("##", "")
    text = text.replace("▁", " ")
    return " ".join(text.split())


def _split_hf_bio_label(value: object) -> tuple[str, str]:
    label = _normalize_hf_entity_label(value)
    if label in {"b", "i", "o"}:
        return label, "skill"
    if label.startswith("b-"):
        return "b", label[2:]
    if label.startswith("i-"):
        return "i", label[2:]
    if label == "o":
        return "o", ""
    return "b", label


def _decode_hf_token_predictions(
    text: str,
    token_offsets: list[tuple[int, int]],
    token_labels: list[str],
    token_scores: list[float],
    language: str | None,
    entity_labels: set[str] | None = None,
) -> list[SkillMention]:
    mentions: list[SkillMention] = []
    seen: set[str] = set()
    allowed = entity_labels or set()

    current_start: int | None = None
    current_end: int | None = None
    current_type = "skill"
    current_scores: list[float] = []

    def flush() -> None:
        nonlocal current_start, current_end, current_type, current_scores
        if current_start is None or current_end is None or current_end <= current_start:
            current_start = None
            current_end = None
            current_type = "skill"
            current_scores = []
            return

        raw_text = text[current_start:current_end].strip()
        if not raw_text:
            current_start = None
            current_end = None
            current_type = "skill"
            current_scores = []
            return

        mention = normalize_extracted_skill_mention(raw_text, language=language)
        normalized = mention.casefold()
        if normalized not in seen:
            seen.add(normalized)
            mentions.append(
                SkillMention(
                    text=mention,
                    raw_text=raw_text if raw_text != mention else None,
                    label=current_type or "skill",
                    score=(sum(current_scores) / len(current_scores)) if current_scores else None,
                    start=current_start,
                    end=current_end,
                )
            )

        current_start = None
        current_end = None
        current_type = "skill"
        current_scores = []

    for (start, end), raw_label, score in zip(token_offsets, token_labels, token_scores):
        if start == end:
            flush()
            continue

        prefix, entity_type = _split_hf_bio_label(raw_label)
        if prefix == "o":
            flush()
            continue

        entity_type = entity_type or "skill"
        if allowed and entity_type.casefold() not in allowed:
            flush()
            continue

        should_start_new = (
            current_start is None
            or prefix == "b"
            or entity_type != current_type
            or start > (current_end or start) + 1
        )

        if should_start_new:
            flush()
            current_start = start
            current_end = end
            current_type = entity_type
            current_scores = [score]
            continue

        current_end = max(current_end or end, end)
        current_scores.append(score)

    flush()
    return mentions


def mentions_to_json(mentions: list[SkillMention]) -> list[dict]:
    return [asdict(mention) for mention in mentions]
