from __future__ import annotations

import json
import random
import re
from pathlib import Path

from esco_skill_batch.io_utils import read_records
from esco_skill_batch.text_utils import strip_accents


TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def _coerce_skill_items(raw: object) -> list[dict]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [{"mention": part.strip()} for part in raw.split("|") if part.strip()]
    if isinstance(raw, list):
        items: list[dict] = []
        for item in raw:
            if isinstance(item, dict):
                mention = str(item.get("mention", "")).strip()
                if mention:
                    items.append(dict(item))
                continue
            mention = str(item).strip()
            if mention:
                items.append({"mention": mention})
        return items
    raise ValueError(f"Unsupported skills field type: {type(raw).__name__}")


def tokenize_with_offsets(text: str) -> list[tuple[str, int, int]]:
    return [(match.group(0), match.start(), match.end()) for match in TOKEN_PATTERN.finditer(text)]


def _find_all_occurrences(text: str, needle: str) -> list[tuple[int, int]]:
    if not needle:
        return []
    positions: list[tuple[int, int]] = []
    start = 0
    while True:
        index = text.find(needle, start)
        if index < 0:
            return positions
        positions.append((index, index + len(needle)))
        start = index + 1


def _find_mention_occurrences(text: str, mention: str) -> list[tuple[int, int]]:
    exact = _find_all_occurrences(text, mention)
    if exact:
        return exact

    lowered = _find_all_occurrences(text.lower(), mention.lower())
    if lowered:
        return lowered

    stripped = _find_all_occurrences(strip_accents(text).lower(), strip_accents(mention).lower())
    if stripped:
        return stripped

    return []


def _resolve_skill_char_span(
    text: str,
    item: dict,
    used_spans: set[tuple[int, int]],
) -> tuple[int, int] | None:
    raw_start = item.get("start")
    raw_end = item.get("end")
    if raw_start is not None and raw_end is not None:
        start = int(raw_start)
        end = int(raw_end)
        if 0 <= start < end <= len(text):
            used_spans.add((start, end))
            return start, end

    mention = str(item.get("mention", "")).strip()
    for candidate in _find_mention_occurrences(text, mention):
        if candidate in used_spans:
            continue
        used_spans.add(candidate)
        return candidate
    return None


def _char_span_to_token_span(
    token_rows: list[tuple[str, int, int]],
    start_char: int,
    end_char: int,
) -> tuple[int, int] | None:
    start_token: int | None = None
    end_token: int | None = None

    for index, (_, token_start, token_end) in enumerate(token_rows):
        overlaps = token_end > start_char and token_start < end_char
        if not overlaps:
            continue
        if start_token is None:
            start_token = index
        end_token = index

    if start_token is None or end_token is None:
        return None
    return start_token, end_token


def _deduplicate_spans(spans: list[tuple[int, int, str]]) -> list[list[object]]:
    seen: set[tuple[int, int, str]] = set()
    output: list[list[object]] = []
    for start, end, label in sorted(spans, key=lambda item: (item[0], item[1], item[2])):
        key = (start, end, label)
        if key in seen:
            continue
        seen.add(key)
        output.append([start, end, label])
    return output


def prepare_gliner_record(
    record: dict,
    text_field: str,
    skills_field: str,
    label: str,
    keep_empty: bool,
    max_tokens: int | None,
    window_stride: int | None,
) -> tuple[list[dict], list[str], str | None]:
    record_id = str(record.get("id", "") or "").strip() or None
    text = str(record.get(text_field, "") or "")
    if not text.strip():
        return [], [], record_id

    token_rows = tokenize_with_offsets(text)
    if not token_rows:
        return [], [], record_id

    raw_skills = _coerce_skill_items(record.get(skills_field, []))
    used_spans: set[tuple[int, int]] = set()
    aligned_spans: list[tuple[int, int, str]] = []
    unmatched_mentions: list[str] = []

    for item in raw_skills:
        mention = str(item.get("mention", "")).strip()
        if not mention:
            continue
        char_span = _resolve_skill_char_span(text, item, used_spans)
        if char_span is None:
            unmatched_mentions.append(mention)
            continue
        token_span = _char_span_to_token_span(token_rows, char_span[0], char_span[1])
        if token_span is None:
            unmatched_mentions.append(mention)
            continue
        aligned_spans.append((token_span[0], token_span[1], label))

    if raw_skills and not aligned_spans:
        return [], unmatched_mentions, record_id

    tokens = [token for token, _, _ in token_rows]
    if max_tokens is None or max_tokens <= 0 or len(tokens) <= max_tokens:
        if aligned_spans or keep_empty:
            example_id = record_id or "record"
            return [
                {
                    "id": example_id,
                    "record_id": record_id or example_id,
                    "tokenized_text": tokens,
                    "ner": _deduplicate_spans(aligned_spans),
                }
            ], unmatched_mentions, record_id
        return [], unmatched_mentions, record_id

    stride = window_stride if window_stride and window_stride > 0 else max_tokens
    examples: list[dict] = []
    chunk_index = 0
    window_start = 0
    while window_start < len(tokens):
        window_end = min(window_start + max_tokens, len(tokens))
        window_spans = [
            (start - window_start, end - window_start, span_label)
            for start, end, span_label in aligned_spans
            if start >= window_start and end < window_end
        ]
        if window_spans or keep_empty:
            example_id = f"{record_id or 'record'}#{chunk_index}"
            examples.append(
                {
                    "id": example_id,
                    "record_id": record_id or "record",
                    "tokenized_text": tokens[window_start:window_end],
                    "ner": _deduplicate_spans(window_spans),
                }
            )
        if window_end >= len(tokens):
            break
        window_start += stride
        chunk_index += 1

    return examples, unmatched_mentions, record_id


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare_gliner_datasets(
    input_paths: list[Path],
    output_dir: Path,
    text_field: str,
    skills_field: str,
    label: str,
    dev_ratio: float,
    seed: int,
    keep_empty: bool,
    max_tokens: int | None,
    window_stride: int | None,
    max_records: int | None = None,
    fail_on_unmatched: bool = False,
) -> dict:
    grouped_examples: list[tuple[str, list[dict]]] = []
    unmatched_records: list[dict] = []
    input_records = 0
    skipped_no_text = 0
    skipped_unmatched = 0

    for input_path in input_paths:
        for record in read_records(input_path):
            if max_records is not None and input_records >= max_records:
                break
            input_records += 1
            examples, unmatched_mentions, record_id = prepare_gliner_record(
                record=record,
                text_field=text_field,
                skills_field=skills_field,
                label=label,
                keep_empty=keep_empty,
                max_tokens=max_tokens,
                window_stride=window_stride,
            )

            if unmatched_mentions:
                unmatched_records.append(
                    {
                        "id": record_id or str(input_records),
                        "mentions": unmatched_mentions,
                    }
                )
                if fail_on_unmatched:
                    raise ValueError(
                        f"Could not align all mentions for record {record_id or input_records}: {', '.join(unmatched_mentions)}"
                    )

            if not str(record.get(text_field, "") or "").strip():
                skipped_no_text += 1
                continue

            raw_skills = _coerce_skill_items(record.get(skills_field, []))
            if raw_skills and not examples:
                skipped_unmatched += 1
                continue

            if examples:
                group_id = examples[0]["record_id"]
                grouped_examples.append((str(group_id), examples))

        if max_records is not None and input_records >= max_records:
            break

    record_ids = [record_id for record_id, _ in grouped_examples]
    rng = random.Random(seed)
    indices = list(range(len(record_ids)))
    rng.shuffle(indices)

    if dev_ratio <= 0 or len(indices) <= 1:
        dev_count = 0
    else:
        dev_count = max(1, int(round(len(indices) * dev_ratio)))
        dev_count = min(dev_count, max(len(indices) - 1, 0))

    dev_index_set = set(indices[:dev_count])

    train_examples: list[dict] = []
    dev_examples: list[dict] = []
    train_records = 0
    dev_records = 0

    for index, (_, examples) in enumerate(grouped_examples):
        if index in dev_index_set:
            dev_records += 1
            dev_examples.extend(examples)
        else:
            train_records += 1
            train_examples.extend(examples)

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.json"
    dev_path = output_dir / "dev.json"
    manifest_path = output_dir / "manifest.json"

    _write_json(train_path, train_examples)
    _write_json(dev_path, dev_examples)
    _write_json(
        manifest_path,
        {
            "status": "ok",
            "input_records": input_records,
            "prepared_records": len(grouped_examples),
            "train_records": train_records,
            "dev_records": dev_records,
            "train_examples": len(train_examples),
            "dev_examples": len(dev_examples),
            "skipped_no_text": skipped_no_text,
            "skipped_unmatched": skipped_unmatched,
            "unmatched_records": unmatched_records,
            "text_field": text_field,
            "skills_field": skills_field,
            "label": label,
            "keep_empty": keep_empty,
            "max_tokens": max_tokens,
            "window_stride": window_stride,
            "seed": seed,
            "dev_ratio": dev_ratio,
            "train_path": str(train_path),
            "dev_path": str(dev_path),
        },
    )

    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _load_json_dataset(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"GLiNER training data at {path} must be a JSON array.")
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError(f"GLiNER training data at {path} must contain JSON objects.")
    return payload


def train_gliner_model(
    train_data: Path,
    dev_data: Path | None,
    model_name: str,
    output_dir: Path,
    learning_rate: float,
    others_learning_rate: float,
    weight_decay: float,
    others_weight_decay: float,
    warmup_ratio: float,
    train_batch_size: int,
    eval_batch_size: int,
    max_steps: int,
    save_steps: int,
    logging_steps: int,
    save_total_limit: int,
    max_grad_norm: float,
    negatives: float,
    loss_alpha: float,
    loss_gamma: float,
    loss_prob_margin: float,
    loss_reduction: str,
    masking: str,
    scheduler_type: str,
    gradient_accumulation_steps: int,
    dataloader_num_workers: int,
    freeze_components: list[str] | None,
    use_cpu: bool,
    bf16: bool,
    compile_model: bool,
    seed: int,
) -> dict:
    try:
        import torch
        import transformers
        from gliner import GLiNER
    except ImportError as exc:
        raise RuntimeError("GLiNER fine-tuning requires `gliner`, `torch` and `transformers`. Install with `.[gliner]`.") from exc

    transformers.set_seed(seed)

    train_dataset = _load_json_dataset(train_data)
    dev_dataset = _load_json_dataset(dev_data) if dev_data is not None and dev_data.exists() else []

    model = GLiNER.from_pretrained(model_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer = model.train_model(
        train_dataset=train_dataset,
        eval_dataset=dev_dataset or None,
        output_dir=str(output_dir),
        learning_rate=learning_rate,
        others_lr=others_learning_rate,
        weight_decay=weight_decay,
        others_weight_decay=others_weight_decay,
        warmup_ratio=warmup_ratio,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        max_steps=max_steps,
        save_steps=save_steps,
        logging_steps=logging_steps,
        save_total_limit=save_total_limit,
        max_grad_norm=max_grad_norm,
        negatives=negatives,
        focal_loss_alpha=loss_alpha,
        focal_loss_gamma=loss_gamma,
        focal_loss_prob_margin=loss_prob_margin,
        loss_reduction=loss_reduction,
        masking=masking,
        lr_scheduler_type=scheduler_type,
        dataloader_num_workers=dataloader_num_workers,
        gradient_accumulation_steps=gradient_accumulation_steps,
        use_cpu=use_cpu or not torch.cuda.is_available(),
        bf16=bf16,
        report_to="none",
        freeze_components=freeze_components,
        compile_model=compile_model,
    )

    final_model_dir = output_dir / "final-model"
    model.save_pretrained(final_model_dir)

    summary = {
        "status": "ok",
        "base_model": model_name,
        "train_records": len(train_dataset),
        "dev_records": len(dev_dataset),
        "output_dir": str(output_dir),
        "final_model_dir": str(final_model_dir),
        "max_steps": max_steps,
        "train_batch_size": train_batch_size,
        "eval_batch_size": eval_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "others_learning_rate": others_learning_rate,
        "used_cpu": bool(use_cpu or not torch.cuda.is_available()),
        "bf16": bool(bf16),
        "freeze_components": freeze_components or [],
        "best_checkpoint": str(getattr(trainer.state, "best_model_checkpoint", "") or ""),
    }
    _write_json(output_dir / "training-summary.json", summary)
    return summary
