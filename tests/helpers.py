from __future__ import annotations

import csv
from pathlib import Path


def write_esco_csv(path: Path) -> None:
    rows = [
        {
            "conceptType": "Skill/competence",
            "conceptUri": "http://data.europa.eu/esco/skill/python",
            "skillType": "skill/competence",
            "reuseLevel": "sector-specific",
            "preferredLabel": "Python",
            "altLabels": "Python programming|programming in Python",
            "hiddenLabels": "",
            "inScheme": "member-skills",
            "description": "Develop software in Python.",
            "definition": "",
            "scopeNote": "",
        },
        {
            "conceptType": "Skill/competence",
            "conceptUri": "http://data.europa.eu/esco/skill/sql",
            "skillType": "skill/competence",
            "reuseLevel": "sector-specific",
            "preferredLabel": "SQL",
            "altLabels": "Structured Query Language",
            "hiddenLabels": "",
            "inScheme": "member-skills",
            "description": "Query and manipulate relational databases.",
            "definition": "",
            "scopeNote": "",
        },
        {
            "conceptType": "Skill/competence",
            "conceptUri": "http://data.europa.eu/esco/skill/communication",
            "skillType": "skill/competence",
            "reuseLevel": "transversal",
            "preferredLabel": "communication",
            "altLabels": "communication skills|communicate effectively",
            "hiddenLabels": "",
            "inScheme": "transversal-skills",
            "description": "Communicate clearly with stakeholders.",
            "definition": "",
            "scopeNote": "",
        },
        {
            "conceptType": "Skill/competence",
            "conceptUri": "http://data.europa.eu/esco/skill/problem-solving",
            "skillType": "skill/competence",
            "reuseLevel": "transversal",
            "preferredLabel": "problem solving",
            "altLabels": "solving problems|problem-solving skills",
            "hiddenLabels": "",
            "inScheme": "transversal-skills",
            "description": "Solve problems and identify workable solutions.",
            "definition": "",
            "scopeNote": "",
        },
        {
            "conceptType": "Knowledge",
            "conceptUri": "http://data.europa.eu/esco/skill/statistics-knowledge",
            "skillType": "knowledge",
            "reuseLevel": "sector-specific",
            "preferredLabel": "statistics",
            "altLabels": "statistical theory",
            "hiddenLabels": "",
            "inScheme": "member-knowledge",
            "description": "Knowledge of statistics.",
            "definition": "",
            "scopeNote": "",
        },
        {
            "conceptType": "Knowledge",
            "conceptUri": "http://data.europa.eu/esco/skill/english-language",
            "skillType": "knowledge",
            "reuseLevel": "sector-specific",
            "preferredLabel": "English",
            "altLabels": "English language",
            "hiddenLabels": "",
            "inScheme": "language-skills-and-knowledge",
            "description": "Knowledge of English.",
            "definition": "",
            "scopeNote": "",
        },
    ]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
