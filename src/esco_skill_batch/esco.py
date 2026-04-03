from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from esco_skill_batch.text_utils import normalize_text, tokenize, unique_preserve_order
from esco_skill_batch.types import EscoSkill


FIELD_ALIASES = {
    "concept_uri": ["concepturi", "uri", "id", "identifier"],
    "preferred_label": ["preferredlabel"],
    "alt_labels": ["altlabels", "nonpreferredlabels"],
    "hidden_labels": ["hiddenlabels"],
    "description": ["description"],
    "definition": ["definition"],
    "scope_note": ["scopenote"],
    "skill_type": ["skilltype"],
    "reuse_level": ["reuselevel"],
    "concept_type": ["concepttype"],
    "in_scheme": ["inscheme"],
    "broader_concepts": ["broaderconcepts", "broaderhierarchyconcepts"],
}


def _normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _match_key(fieldnames: list[str], aliases: list[str], language: str | None) -> str | None:
    normalized = {name: _normalize_key(name) for name in fieldnames}
    if language:
        suffix = language.lower()
        for field, normalized_name in normalized.items():
            for alias in aliases:
                if normalized_name == f"{alias}{suffix}" or normalized_name.endswith(f"{alias}{suffix}"):
                    return field
    for field, normalized_name in normalized.items():
        for alias in aliases:
            if normalized_name == alias:
                return field
    if language:
        for field, normalized_name in normalized.items():
            for alias in aliases:
                if normalized_name.startswith(alias) and normalized_name.endswith(language.lower()):
                    return field
    return None


def _split_labels(value: str) -> list[str]:
    if not value:
        return []
    parts = re.split(r"[;\n|]+", value)
    return unique_preserve_order([part.strip() for part in parts if part.strip()])


def _contains_meta_token(value: str, token: str) -> bool:
    return token in normalize_text(value)


def infer_category(
    skill_type: str,
    reuse_level: str,
    in_scheme: str,
    broader_concepts: str,
) -> str:
    meta = " ".join([skill_type, reuse_level, in_scheme, broader_concepts])
    meta_normalized = normalize_text(meta)

    if "knowledge" in meta_normalized:
        if "language" in meta_normalized:
            return "language"
        return "knowledge"

    if "language" in meta_normalized:
        return "language"

    if "transversal" in meta_normalized:
        return "transversal"

    return "skill"


def load_esco_skills(
    csv_path: Path,
    language: str | None,
    include_knowledge: bool,
    include_language_skills: bool,
) -> list[EscoSkill]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("ESCO CSV has no header.")

        fields = {
            logical_name: _match_key(reader.fieldnames, aliases, language)
            for logical_name, aliases in FIELD_ALIASES.items()
        }

        if fields["concept_uri"] is None or fields["preferred_label"] is None:
            raise ValueError(
                "Could not infer required ESCO columns. Need at least conceptUri/uri and preferredLabel."
            )

        output: list[EscoSkill] = []
        for row in reader:
            concept_uri = (row.get(fields["concept_uri"] or "", "") or "").strip()
            preferred_label = (row.get(fields["preferred_label"] or "", "") or "").strip()
            if not concept_uri or not preferred_label:
                continue

            alt_labels = _split_labels(row.get(fields["alt_labels"] or "", "") or "")
            hidden_labels = _split_labels(row.get(fields["hidden_labels"] or "", "") or "")
            description = (row.get(fields["description"] or "", "") or "").strip()
            definition = (row.get(fields["definition"] or "", "") or "").strip()
            scope_note = (row.get(fields["scope_note"] or "", "") or "").strip()
            skill_type = (row.get(fields["skill_type"] or "", "") or "").strip()
            reuse_level = (row.get(fields["reuse_level"] or "", "") or "").strip()
            concept_type = (row.get(fields["concept_type"] or "", "") or "").strip()
            in_scheme = (row.get(fields["in_scheme"] or "", "") or "").strip()
            broader_concepts = (row.get(fields["broader_concepts"] or "", "") or "").strip()

            category = infer_category(skill_type, reuse_level, in_scheme, broader_concepts)
            if category == "knowledge" and not include_knowledge:
                continue
            if category == "language" and not include_language_skills:
                continue

            labels = unique_preserve_order([preferred_label, *alt_labels, *hidden_labels])
            search_text = " | ".join(
                part for part in [preferred_label, *alt_labels, description, definition, scope_note] if part
            )
            labels_normalized = [normalize_text(label) for label in labels if normalize_text(label)]

            output.append(
                EscoSkill(
                    concept_uri=concept_uri,
                    preferred_label=preferred_label,
                    alt_labels=alt_labels,
                    hidden_labels=hidden_labels,
                    description=description,
                    definition=definition,
                    scope_note=scope_note,
                    skill_type=skill_type,
                    reuse_level=reuse_level,
                    concept_type=concept_type,
                    category=category,
                    search_text=search_text,
                    labels_normalized=labels_normalized,
                )
            )
        return output


def save_index(
    output_dir: Path,
    skills: list[EscoSkill],
    source_csv: Path,
    language: str | None,
    include_knowledge: bool,
    include_language_skills: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [asdict(skill) for skill in skills]
    with (output_dir / "skills.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    token_index: dict[str, list[int]] = defaultdict(list)
    exact_label_index: dict[str, list[int]] = defaultdict(list)
    for idx, skill in enumerate(skills):
        labels = [skill.preferred_label, *skill.alt_labels, *skill.hidden_labels]
        for label in labels:
            normalized = normalize_text(label)
            if normalized:
                exact_label_index[normalized].append(idx)
            for token in set(tokenize(label)):
                token_index[token].append(idx)

    manifest = {
        "source_csv": str(source_csv),
        "language": language,
        "include_knowledge": include_knowledge,
        "include_language_skills": include_language_skills,
        "size": len(skills),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "token_index.json").write_text(
        json.dumps(token_index, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "exact_label_index.json").write_text(
        json.dumps(exact_label_index, ensure_ascii=False),
        encoding="utf-8",
    )


def load_index(index_dir: Path) -> tuple[list[EscoSkill], dict, dict, dict]:
    skills: list[EscoSkill] = []
    with (index_dir / "skills.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            skills.append(EscoSkill(**payload))

    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    token_index = json.loads((index_dir / "token_index.json").read_text(encoding="utf-8"))
    exact_label_index = json.loads((index_dir / "exact_label_index.json").read_text(encoding="utf-8"))
    return skills, manifest, token_index, exact_label_index
