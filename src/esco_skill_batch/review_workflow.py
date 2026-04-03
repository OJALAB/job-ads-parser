from __future__ import annotations

import csv
import random
import re
from collections import Counter
from pathlib import Path

from esco_skill_batch.io_utils import read_records, write_jsonl
from esco_skill_batch.normalization import normalize_extracted_skill_mention
from esco_skill_batch.text_utils import normalize_text, strip_accents, tokenize
from esco_skill_batch.types import SkillMention


DEFAULT_DECISION = ""
VALID_DECISIONS = {"", "accept_esco", "no_match", "reject"}
CONTEXT_WINDOW_CHARS = 90
GENERIC_NOISE = {
    "benefity",
    "benefit",
    "benefits",
    "doswiadczenie",
    "experience",
    "hybrydowa",
    "hybrid",
    "karta sportowa",
    "luxmed",
    "mile widziane",
    "multisport",
    "praca hybrydowa",
    "praca zdalna",
    "remote",
    "umowa o prace",
    "warszawa",
    "wynagrodzenie",
}
CSV_CONTEXT_COLUMNS = 5
CSV_CANDIDATE_COLUMNS = 3
MULTISPACE_RE = re.compile(r"\s+")


def _coerce_decision(value: object) -> str:
    raw = normalize_text(str(value or "")).replace("-", "_").replace(" ", "_")
    if raw not in VALID_DECISIONS:
        raise ValueError(
            f"Unsupported review decision `{value}`. Expected one of: accept_esco, no_match, reject."
        )
    return raw


def _candidate_id(index: int) -> str:
    return f"cand-{index:06d}"


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


def _resolve_offsets(text: str, mention: SkillMention, fallback_mention: str) -> tuple[int | None, int | None]:
    if mention.start is not None and mention.end is not None:
        start = int(mention.start)
        end = int(mention.end)
        if 0 <= start < end <= len(text):
            return start, end

    candidates = _find_mention_occurrences(text, mention.text)
    if len(candidates) == 1:
        return candidates[0]

    candidates = _find_mention_occurrences(text, fallback_mention)
    if len(candidates) == 1:
        return candidates[0]

    return None, None


def _build_context(text: str, start: int | None, end: int | None) -> str:
    compact = MULTISPACE_RE.sub(" ", text).strip()
    if start is None or end is None or start < 0 or end > len(text) or start >= end:
        return compact[: CONTEXT_WINDOW_CHARS * 2].strip()

    left = max(0, start - CONTEXT_WINDOW_CHARS)
    right = min(len(text), end + CONTEXT_WINDOW_CHARS)
    excerpt = MULTISPACE_RE.sub(" ", text[left:right]).strip()
    if left > 0:
        excerpt = f"... {excerpt}"
    if right < len(text):
        excerpt = f"{excerpt} ..."
    return excerpt


def _auto_status(mention_normalized: str, candidates: list[dict]) -> str:
    if not mention_normalized:
        return "reject_noise"

    tokens = tokenize(mention_normalized)
    if not tokens:
        return "reject_noise"

    if mention_normalized in GENERIC_NOISE:
        return "reject_noise"

    if len(tokens) == 1 and tokens[0] in {"benefity", "benefits", "experience", "doswiadczenie"}:
        return "reject_noise"

    if candidates:
        top = candidates[0]
        top_score = float(top.get("score", 0.0) or 0.0)
        matched_on = str(top.get("matched_on", ""))
        if matched_on in {"review_alias", "exact_label"} and top_score >= 0.99:
            return "high_confidence"

    return "needs_review"


def _best_representative(counter: Counter[str], fallback: str) -> str:
    if counter:
        return counter.most_common(1)[0][0]
    return fallback


def _candidate_summary(candidates: list) -> list[dict]:
    output: list[dict] = []
    for candidate in candidates:
        output.append(
            {
                "concept_uri": candidate.concept_uri,
                "preferred_label": candidate.preferred_label,
                "score": candidate.score,
                "matched_on": candidate.matched_on,
                "category": candidate.category,
                "skill_type": candidate.skill_type,
                "reuse_level": candidate.reuse_level,
            }
        )
    return output


def prepare_review_queue(
    input_paths: list[Path],
    output_path: Path,
    extractor,
    matcher,
    *,
    text_field: str,
    id_field: str,
    top_k: int,
    score_threshold: float,
    max_records: int | None,
    max_contexts: int = 5,
) -> dict:
    grouped: dict[tuple[str, str], dict] = {}
    processed_records = 0
    total_mentions = 0

    for input_path in input_paths:
        for record_number, record in enumerate(read_records(input_path), start=1):
            if max_records is not None and processed_records >= max_records:
                break

            text = str(record.get(text_field, "") or "")
            if not text.strip():
                processed_records += 1
                continue

            processed_records += 1
            record_id = str(record.get(id_field, str(record_number)))
            language = str(record.get("language", "") or "").strip().lower() or "und"
            title = str(record.get("title", "") or "").strip()
            source_url = str(record.get("source_url", "") or "").strip()

            mentions = extractor.extract(record, text)
            total_mentions += len(mentions)
            for mention in mentions:
                representative = normalize_extracted_skill_mention(mention.text, language=language)
                mention_normalized = normalize_text(representative)
                if not mention_normalized:
                    continue

                start, end = _resolve_offsets(text, mention, representative)
                occurrence = {
                    "record_id": record_id,
                    "mention_text": mention.text,
                    "raw_text": mention.raw_text or mention.text,
                    "start": start,
                    "end": end,
                    "context": _build_context(text, start, end),
                    "title": title,
                    "source_url": source_url,
                    "score": mention.score,
                }

                key = (language, mention_normalized)
                entry = grouped.setdefault(
                    key,
                    {
                        "language": language,
                        "mention_normalized": mention_normalized,
                        "mention_variants": Counter(),
                        "raw_variants": Counter(),
                        "occurrences": [],
                        "source_record_ids": [],
                    },
                )
                entry["mention_variants"][representative] += 1
                entry["raw_variants"][mention.text] += 1
                entry["occurrences"].append(occurrence)
                if record_id not in entry["source_record_ids"]:
                    entry["source_record_ids"].append(record_id)
        if max_records is not None and processed_records >= max_records:
            break

    rows: list[dict] = []
    for index, key in enumerate(sorted(grouped), start=1):
        entry = grouped[key]
        representative = _best_representative(entry["mention_variants"], entry["mention_normalized"])
        raw_variant = _best_representative(entry["raw_variants"], representative)
        matches = matcher.match(
            SkillMention(text=representative, label="skill"),
            top_k=top_k,
            score_threshold=score_threshold,
        )
        candidate_row = {
            "candidate_id": _candidate_id(index),
            "mention_raw": raw_variant,
            "mention_normalized": entry["mention_normalized"],
            "language": entry["language"],
            "occurrence_count": len(entry["occurrences"]),
            "source_record_ids": entry["source_record_ids"],
            "example_contexts": entry["occurrences"][:max_contexts],
            "occurrences": entry["occurrences"],
            "extractor_votes": {"gliner": len(entry["occurrences"])},
            "top_k_esco_candidates": _candidate_summary(matches),
            "auto_status": _auto_status(entry["mention_normalized"], _candidate_summary(matches)),
            "decision": DEFAULT_DECISION,
            "selected_esco_uri": "",
            "canonical_mention": representative,
            "notes": "",
        }
        rows.append(candidate_row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, rows)
    return {
        "status": "ok",
        "processed_records": processed_records,
        "total_mentions": total_mentions,
        "queue_size": len(rows),
        "output": str(output_path),
        "text_field": text_field,
        "id_field": id_field,
    }


def export_review_csv(queue_path: Path, output_path: Path) -> dict:
    rows = list(read_records(queue_path))
    fieldnames = [
        "candidate_id",
        "language",
        "mention_raw",
        "mention_normalized",
        "canonical_mention",
        "occurrence_count",
        "auto_status",
    ]
    for index in range(1, CSV_CONTEXT_COLUMNS + 1):
        fieldnames.append(f"example_context_{index}")
    for index in range(1, CSV_CANDIDATE_COLUMNS + 1):
        fieldnames.extend(
            [
                f"candidate_{index}_label",
                f"candidate_{index}_uri",
                f"candidate_{index}_score",
                f"candidate_{index}_matched_on",
            ]
        )
    fieldnames.extend(["decision", "selected_esco_uri", "notes"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            csv_row = {
                "candidate_id": row.get("candidate_id", ""),
                "language": row.get("language", ""),
                "mention_raw": row.get("mention_raw", ""),
                "mention_normalized": row.get("mention_normalized", ""),
                "canonical_mention": row.get("canonical_mention", ""),
                "occurrence_count": row.get("occurrence_count", 0),
                "auto_status": row.get("auto_status", ""),
                "decision": row.get("decision", ""),
                "selected_esco_uri": row.get("selected_esco_uri", ""),
                "notes": row.get("notes", ""),
            }
            for index, context in enumerate(row.get("example_contexts", [])[:CSV_CONTEXT_COLUMNS], start=1):
                csv_row[f"example_context_{index}"] = str(context.get("context", "")).strip()
            for index, candidate in enumerate(row.get("top_k_esco_candidates", [])[:CSV_CANDIDATE_COLUMNS], start=1):
                csv_row[f"candidate_{index}_label"] = candidate.get("preferred_label", "")
                csv_row[f"candidate_{index}_uri"] = candidate.get("concept_uri", "")
                csv_row[f"candidate_{index}_score"] = candidate.get("score", "")
                csv_row[f"candidate_{index}_matched_on"] = candidate.get("matched_on", "")
            writer.writerow(csv_row)

    return {
        "status": "ok",
        "rows": len(rows),
        "output": str(output_path),
    }


def import_review_csv(queue_path: Path, csv_path: Path, output_path: Path) -> dict:
    queue_rows = list(read_records(queue_path))
    queue_by_id = {str(row["candidate_id"]): dict(row) for row in queue_rows}
    reviewed_ids: set[str] = set()

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            candidate_id = str(row.get("candidate_id", "")).strip()
            if not candidate_id:
                continue
            if candidate_id not in queue_by_id:
                raise ValueError(f"Unknown candidate_id in review CSV: {candidate_id}")

            decision = _coerce_decision(row.get("decision", ""))
            selected_esco_uri = str(row.get("selected_esco_uri", "") or "").strip()
            canonical_mention = str(row.get("canonical_mention", "") or "").strip()
            notes = str(row.get("notes", "") or "").strip()

            if decision == "accept_esco" and not selected_esco_uri:
                raise ValueError(f"Candidate {candidate_id} is accept_esco but selected_esco_uri is empty.")

            queue_row = queue_by_id[candidate_id]
            queue_row["decision"] = decision
            queue_row["selected_esco_uri"] = selected_esco_uri
            if canonical_mention:
                queue_row["canonical_mention"] = canonical_mention
            queue_row["notes"] = notes
            reviewed_ids.add(candidate_id)

    ordered_rows = [queue_by_id[str(row["candidate_id"])] for row in queue_rows]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, ordered_rows)
    return {
        "status": "ok",
        "reviewed_candidates": len(reviewed_ids),
        "total_candidates": len(queue_rows),
        "output": str(output_path),
    }


def _resolve_occurrence_char_span(text: str, occurrence: dict, canonical_mention: str) -> tuple[tuple[int, int] | None, str | None]:
    start = occurrence.get("start")
    end = occurrence.get("end")
    if start is not None and end is not None:
        start_int = int(start)
        end_int = int(end)
        if 0 <= start_int < end_int <= len(text):
            return (start_int, end_int), None

    for candidate in [
        str(occurrence.get("mention_text", "") or "").strip(),
        str(occurrence.get("raw_text", "") or "").strip(),
        canonical_mention,
    ]:
        if not candidate:
            continue
        hits = _find_mention_occurrences(text, candidate)
        if len(hits) == 1:
            return hits[0], None
        if len(hits) > 1:
            return None, "multiple_occurrences"

    return None, "mention_not_found"


def build_finetune_corpus(
    input_paths: list[Path],
    reviewed_queue_path: Path,
    output_dir: Path,
    *,
    text_field: str,
    holdout_ratio: float,
    seed: int,
) -> dict:
    records_by_id: dict[str, dict] = {}
    ordered_record_ids: list[str] = []
    for input_path in input_paths:
        for record in read_records(input_path):
            record_id = str(record.get("id", "") or "").strip()
            if not record_id:
                continue
            if record_id not in records_by_id:
                ordered_record_ids.append(record_id)
            records_by_id[record_id] = dict(record)

    accepted_by_record: dict[str, list[dict]] = {}
    ambiguous_rows: list[dict] = []
    alias_rows: list[dict] = []

    for candidate in read_records(reviewed_queue_path):
        decision = _coerce_decision(candidate.get("decision", ""))
        if decision != "accept_esco":
            continue

        selected_esco_uri = str(candidate.get("selected_esco_uri", "") or "").strip()
        canonical_mention = str(candidate.get("canonical_mention", "") or "").strip() or str(
            candidate.get("mention_normalized", "") or ""
        ).strip()
        mention_normalized = normalize_text(canonical_mention or str(candidate.get("mention_normalized", "")))
        top_candidates = candidate.get("top_k_esco_candidates", [])
        preferred_label = ""
        for match in top_candidates:
            if str(match.get("concept_uri", "")) == selected_esco_uri:
                preferred_label = str(match.get("preferred_label", "")).strip()
                break
        alias_rows.append(
            {
                "candidate_id": candidate.get("candidate_id", ""),
                "canonical_mention": canonical_mention,
                "mention_normalized": mention_normalized,
                "concept_uri": selected_esco_uri,
                "preferred_label": preferred_label,
            }
        )

        for occurrence in candidate.get("occurrences", []):
            record_id = str(occurrence.get("record_id", "") or "").strip()
            if not record_id:
                ambiguous_rows.append(
                    {
                        "candidate_id": candidate.get("candidate_id", ""),
                        "reason": "missing_record_id",
                        "occurrence": occurrence,
                    }
                )
                continue

            record = records_by_id.get(record_id)
            if record is None:
                ambiguous_rows.append(
                    {
                        "candidate_id": candidate.get("candidate_id", ""),
                        "record_id": record_id,
                        "reason": "record_not_found",
                        "occurrence": occurrence,
                    }
                )
                continue

            text = str(record.get(text_field, "") or "")
            if not text.strip():
                ambiguous_rows.append(
                    {
                        "candidate_id": candidate.get("candidate_id", ""),
                        "record_id": record_id,
                        "reason": "missing_text",
                        "occurrence": occurrence,
                    }
                )
                continue

            span, error_reason = _resolve_occurrence_char_span(text, occurrence, canonical_mention)
            if span is None:
                ambiguous_rows.append(
                    {
                        "candidate_id": candidate.get("candidate_id", ""),
                        "record_id": record_id,
                        "reason": error_reason or "unresolved",
                        "occurrence": occurrence,
                    }
                )
                continue

            start, end = span
            accepted_by_record.setdefault(record_id, []).append(
                {
                    "mention": text[start:end],
                    "start": start,
                    "end": end,
                    "esco_uri": selected_esco_uri,
                }
            )

    positive_record_ids = [record_id for record_id in ordered_record_ids if accepted_by_record.get(record_id)]
    rng = random.Random(seed)
    shuffled_positive_ids = list(positive_record_ids)
    rng.shuffle(shuffled_positive_ids)

    holdout_count = 0
    if shuffled_positive_ids and holdout_ratio > 0:
        holdout_count = int(round(len(shuffled_positive_ids) * holdout_ratio))
        if len(shuffled_positive_ids) > 1:
            holdout_count = max(1, min(holdout_count, len(shuffled_positive_ids) - 1))
    holdout_ids = set(shuffled_positive_ids[:holdout_count])

    silver_rows: list[dict] = []
    holdout_rows: list[dict] = []
    for record_id in positive_record_ids:
        record = dict(records_by_id[record_id])
        deduplicated = {
            (int(item["start"]), int(item["end"]), str(item["esco_uri"])): item
            for item in accepted_by_record[record_id]
        }
        record["gold_skills"] = [
            deduplicated[key]
            for key in sorted(deduplicated, key=lambda item: (item[0], item[1], item[2]))
        ]
        if record_id in holdout_ids:
            holdout_rows.append(record)
        else:
            silver_rows.append(record)

    output_dir.mkdir(parents=True, exist_ok=True)
    silver_path = output_dir / "silver_train.jsonl"
    holdout_path = output_dir / "manual_gold_holdout.jsonl"
    ambiguous_path = output_dir / "ambiguous_occurrences.jsonl"
    aliases_path = output_dir / "review_aliases.jsonl"
    write_jsonl(silver_path, silver_rows)
    write_jsonl(holdout_path, holdout_rows)
    write_jsonl(ambiguous_path, ambiguous_rows)
    write_jsonl(aliases_path, alias_rows)

    return {
        "status": "ok",
        "input_records": len(ordered_record_ids),
        "reviewed_candidates": sum(1 for row in read_records(reviewed_queue_path) if _coerce_decision(row.get("decision", ""))),
        "accepted_candidates": len(alias_rows),
        "silver_train_records": len(silver_rows),
        "manual_gold_holdout_records": len(holdout_rows),
        "ambiguous_occurrences": len(ambiguous_rows),
        "silver_train_path": str(silver_path),
        "manual_gold_holdout_path": str(holdout_path),
        "ambiguous_occurrences_path": str(ambiguous_path),
        "review_aliases_path": str(aliases_path),
        "text_field": text_field,
        "holdout_ratio": holdout_ratio,
        "seed": seed,
    }
