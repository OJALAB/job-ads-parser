from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class EscoSkill:
    concept_uri: str
    preferred_label: str
    alt_labels: list[str] = field(default_factory=list)
    hidden_labels: list[str] = field(default_factory=list)
    description: str = ""
    definition: str = ""
    scope_note: str = ""
    skill_type: str = ""
    reuse_level: str = ""
    concept_type: str = ""
    category: str = "skill"
    search_text: str = ""
    labels_normalized: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SkillMention:
    text: str
    raw_text: str | None = None
    label: str = "skill"
    score: float | None = None
    start: int | None = None
    end: int | None = None


@dataclass(slots=True)
class SkillMatch:
    concept_uri: str
    preferred_label: str
    category: str
    score: float
    matched_on: str
    skill_type: str = ""
    reuse_level: str = ""
