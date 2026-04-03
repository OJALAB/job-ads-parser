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
