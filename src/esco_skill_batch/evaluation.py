from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from esco_skill_batch.text_utils import normalize_text


@dataclass(slots=True)
class EvalCounts:
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0

    def precision(self) -> float:
        denominator = self.true_positive + self.false_positive
        return self.true_positive / denominator if denominator else 0.0

    def recall(self) -> float:
        denominator = self.true_positive + self.false_negative
        return self.true_positive / denominator if denominator else 0.0

    def f1(self) -> float:
        precision = self.precision()
        recall = self.recall()
        denominator = precision + recall
        return 2 * precision * recall / denominator if denominator else 0.0


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL line {line_number} must be an object.")
            rows.append(payload)
    return rows


def _gold_items_by_record(gold_rows: list[dict]) -> dict[str, list[dict]]:
    output: dict[str, list[dict]] = {}
    for row in gold_rows:
        record_id = str(row["id"])
        items = row.get("gold_skills", [])
        if not isinstance(items, list):
            raise ValueError(f"`gold_skills` must be a list for record {record_id}.")
        output[record_id] = items
    return output


def _prediction_items_by_record(prediction_rows: list[dict]) -> dict[str, list[dict]]:
    output: dict[str, list[dict]] = {}
    for row in prediction_rows:
        record_id = str(row["id"])
        items = row.get("matches", [])
        if not isinstance(items, list):
            raise ValueError(f"`matches` must be a list for record {record_id}.")
        output[record_id] = items
    return output


def _prediction_mentions_by_record(prediction_rows: list[dict]) -> dict[str, dict[str, dict]]:
    output: dict[str, dict[str, dict]] = {}
    for row in prediction_rows:
        record_id = str(row["id"])
        mentions: dict[str, dict] = {}
        for item in row.get("matches", []):
            mention_text = str(item.get("mention", {}).get("text", "")).strip()
            if not mention_text:
                continue
            mentions[normalize_text(mention_text)] = item
        output[record_id] = mentions
    return output


def evaluate_predictions(gold_path: Path, predictions_path: Path, top_k: int = 5) -> dict:
    gold_rows = _load_jsonl(gold_path)
    prediction_rows = _load_jsonl(predictions_path)

    gold_by_record = _gold_items_by_record(gold_rows)
    predictions_by_record = _prediction_items_by_record(prediction_rows)

    mention_counts = EvalCounts()
    mapping_top1_correct = 0
    mapping_topk_correct = 0
    mapping_total = 0
    exact_mention_match_records = 0
    exact_top1_uri_match_records = 0
    mapping_mismatches: list[dict] = []

    missing_predictions = sorted(set(gold_by_record) - set(predictions_by_record))

    for record_id, gold_items in gold_by_record.items():
        prediction_items = predictions_by_record.get(record_id, [])

        gold_mentions = {normalize_text(item["mention"]) for item in gold_items}
        predicted_mentions = {
            normalize_text(item.get("mention", {}).get("text", ""))
            for item in prediction_items
            if item.get("mention", {}).get("text")
        }

        mention_counts.true_positive += len(gold_mentions & predicted_mentions)
        mention_counts.false_positive += len(predicted_mentions - gold_mentions)
        mention_counts.false_negative += len(gold_mentions - predicted_mentions)

        if gold_mentions == predicted_mentions:
            exact_mention_match_records += 1

        gold_uri_set = set()
        predicted_top1_uri_set = set()
        gold_by_mention = {
            normalize_text(item["mention"]): str(item["esco_uri"])
            for item in gold_items
        }
        predictions_by_mention = {
            normalize_text(item.get("mention", {}).get("text", "")): item.get("esco_matches", [])
            for item in prediction_items
            if item.get("mention", {}).get("text")
        }

        for normalized_mention, gold_uri in gold_by_mention.items():
            mapping_total += 1
            gold_uri_set.add(gold_uri)
            predicted_matches = predictions_by_mention.get(normalized_mention, [])
            candidate_uris = [str(match.get("concept_uri")) for match in predicted_matches[:top_k]]
            if candidate_uris:
                predicted_top1_uri_set.add(candidate_uris[0])
            if candidate_uris and candidate_uris[0] == gold_uri:
                mapping_top1_correct += 1
            else:
                mapping_mismatches.append(
                    {
                        "id": record_id,
                        "mention": normalized_mention,
                        "expected_uri": gold_uri,
                        "predicted_top1_uri": candidate_uris[0] if candidate_uris else None,
                        "predicted_candidate_uris": candidate_uris,
                    }
                )
            if gold_uri in candidate_uris:
                mapping_topk_correct += 1

        if gold_uri_set == predicted_top1_uri_set:
            exact_top1_uri_match_records += 1

    total_records = len(gold_by_record)
    mention_precision = mention_counts.precision()
    mention_recall = mention_counts.recall()
    mention_f1 = mention_counts.f1()

    return {
        "status": "ok",
        "gold_records": total_records,
        "prediction_records": len(predictions_by_record),
        "missing_prediction_ids": missing_predictions,
        "mention_precision": mention_precision,
        "mention_recall": mention_recall,
        "mention_f1": mention_f1,
        "mention_true_positive": mention_counts.true_positive,
        "mention_false_positive": mention_counts.false_positive,
        "mention_false_negative": mention_counts.false_negative,
        "mapping_top1_accuracy": mapping_top1_correct / mapping_total if mapping_total else 0.0,
        "mapping_topk_recall": mapping_topk_correct / mapping_total if mapping_total else 0.0,
        "mapping_total": mapping_total,
        "mapping_mismatches": mapping_mismatches,
        "exact_mention_match_rate": exact_mention_match_records / total_records if total_records else 0.0,
        "exact_top1_uri_match_rate": exact_top1_uri_match_records / total_records if total_records else 0.0,
        "top_k": top_k,
    }


def build_record_report(gold_path: Path, predictions_path: Path, top_k: int = 5) -> dict:
    gold_rows = _load_jsonl(gold_path)
    prediction_rows = _load_jsonl(predictions_path)

    gold_by_record = {str(row["id"]): row for row in gold_rows}
    predictions_by_record = {str(row["id"]): row for row in prediction_rows}
    prediction_mentions_by_record = _prediction_mentions_by_record(prediction_rows)

    records: list[dict] = []
    for record_id, gold_row in gold_by_record.items():
        gold_items = gold_row.get("gold_skills", [])
        gold_mentions_by_norm = {
            normalize_text(str(item["mention"])): item for item in gold_items
        }

        prediction_row = predictions_by_record.get(record_id, {"matches": []})
        predicted_mentions = prediction_mentions_by_record.get(record_id, {})

        gold_norms = set(gold_mentions_by_norm)
        predicted_norms = set(predicted_mentions)
        missing_norms = sorted(gold_norms - predicted_norms)
        extra_norms = sorted(predicted_norms - gold_norms)

        mapping_errors: list[dict] = []
        for normalized_mention, gold_item in gold_mentions_by_norm.items():
            expected_uri = str(gold_item["esco_uri"])
            predicted_item = predicted_mentions.get(normalized_mention)
            predicted_candidates = []
            if predicted_item is not None:
                predicted_candidates = [
                    {
                        "concept_uri": str(match.get("concept_uri")),
                        "preferred_label": str(match.get("preferred_label", "")),
                        "score": match.get("score"),
                    }
                    for match in predicted_item.get("esco_matches", [])[:top_k]
                ]
            predicted_top1_uri = predicted_candidates[0]["concept_uri"] if predicted_candidates else None
            if predicted_top1_uri != expected_uri:
                mapping_errors.append(
                    {
                        "mention": str(gold_item["mention"]),
                        "expected_uri": expected_uri,
                        "predicted_top1_uri": predicted_top1_uri,
                        "predicted_candidates": predicted_candidates,
                    }
                )

        records.append(
            {
                "id": record_id,
                "title": gold_row.get("title"),
                "language": gold_row.get("language"),
                "description": gold_row.get("description"),
                "gold_mentions": [str(item["mention"]) for item in gold_items],
                "predicted_mentions": [
                    str(item.get("mention", {}).get("text", ""))
                    for item in prediction_row.get("matches", [])
                    if item.get("mention", {}).get("text")
                ],
                "missing_mentions": [str(gold_mentions_by_norm[item]["mention"]) for item in missing_norms],
                "extra_mentions": [
                    str(predicted_mentions[item].get("mention", {}).get("text", ""))
                    for item in extra_norms
                ],
                "mapping_errors": mapping_errors,
                "is_exact_mention_match": not missing_norms and not extra_norms,
                "is_exact_top1_uri_match": not mapping_errors and not missing_norms and not extra_norms,
            }
        )

    return {
        "status": "ok",
        "top_k": top_k,
        "records": records,
    }


def render_record_report_markdown(report: dict, metrics: dict) -> str:
    lines: list[str] = []
    lines.append("# Evaluation Report")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Gold records: {metrics['gold_records']}")
    lines.append(f"- Mention F1: {metrics['mention_f1']:.3f}")
    lines.append(f"- Mapping top-1 accuracy: {metrics['mapping_top1_accuracy']:.3f}")
    lines.append(f"- Mapping top-k recall: {metrics['mapping_topk_recall']:.3f}")
    lines.append(f"- Exact mention match rate: {metrics['exact_mention_match_rate']:.3f}")
    lines.append(f"- Exact top-1 URI match rate: {metrics['exact_top1_uri_match_rate']:.3f}")
    lines.append("")

    for record in report["records"]:
        lines.append(f"## {record['id']}")
        if record.get("title"):
            lines.append(f"- Title: {record['title']}")
        if record.get("language"):
            lines.append(f"- Language: {record['language']}")
        if record.get("description"):
            lines.append(f"- Description: {record['description']}")
        lines.append(f"- Gold mentions: {', '.join(record['gold_mentions']) or '(none)'}")
        lines.append(f"- Predicted mentions: {', '.join(record['predicted_mentions']) or '(none)'}")
        lines.append(f"- Missing mentions: {', '.join(record['missing_mentions']) or '(none)'}")
        lines.append(f"- Extra mentions: {', '.join(record['extra_mentions']) or '(none)'}")
        if record["mapping_errors"]:
            lines.append("- Mapping errors:")
            for error in record["mapping_errors"]:
                candidate_uris = ", ".join(
                    candidate["concept_uri"] for candidate in error["predicted_candidates"]
                ) or "(none)"
                lines.append(
                    f"  - {error['mention']}: expected {error['expected_uri']}, "
                    f"predicted {error['predicted_top1_uri'] or '(none)'}, candidates {candidate_uris}"
                )
        else:
            lines.append("- Mapping errors: (none)")
        lines.append("")

    return "\n".join(lines).strip() + "\n"
