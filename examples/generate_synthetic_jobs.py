from __future__ import annotations

import json
from pathlib import Path


RECORDS = [
    {
        "id": "syn-001",
        "language": "en",
        "title": "Junior Data Analyst",
        "description": "We are looking for a junior analyst with Python, SQL and communication skills.",
        "skills_raw": ["Python", "SQL", "communication skills"],
    },
    {
        "id": "syn-002",
        "language": "pl",
        "title": "Analityk Danych",
        "description": "Szukamy osoby, ktora zna Python, zapytania SQL oraz umiejetnosci komunikacyjne.",
        "skills_raw": ["Python", "zapytania SQL", "umiejetnosci komunikacyjne"],
    },
    {
        "id": "syn-003",
        "language": "en",
        "title": "BI Specialist",
        "description": "The role requires Structured Query Language, problem-solving skills and Python programming.",
        "skills_raw": ["Structured Query Language", "problem-solving skills", "Python programming"],
    },
    {
        "id": "syn-004",
        "language": "pl",
        "title": "Specjalista BI",
        "description": "W tej roli potrzebne sa rozwiazywanie problemow, programowanie w Pythonie i SQL.",
        "skills_raw": ["rozwiazywanie problemow", "programowanie w Pythonie", "SQL"],
    },
    {
        "id": "syn-005",
        "language": "de",
        "title": "Data Engineer",
        "description": "Wir suchen eine Person mit Python, SQL-Abfragen und Kommunikationsfahigkeiten.",
        "skills_raw": ["Python", "SQL-Abfragen", "Kommunikationsfahigkeiten"],
    },
    {
        "id": "syn-006",
        "language": "es",
        "title": "Ingeniero de Datos",
        "description": "El puesto requiere programacion en Python, consultas SQL y resolucion de problemas.",
        "skills_raw": ["programacion en Python", "consultas SQL", "resolucion de problemas"],
    },
    {
        "id": "syn-007",
        "language": "fr",
        "title": "Analyste Donnees",
        "description": "Le poste demande Python, requetes SQL et competences en communication.",
        "skills_raw": ["Python", "requetes SQL", "competences en communication"],
    },
    {
        "id": "syn-008",
        "language": "en",
        "title": "Product Analyst",
        "description": "Candidates should communicate effectively and use SQL plus Python for reporting.",
        "skills_raw": ["communicate effectively", "SQL", "Python"],
    },
    {
        "id": "syn-009",
        "language": "pl",
        "title": "Mlodszy Developer",
        "description": "Oferta dla osoby z Python, kompetencje komunikacyjne i umiejetnosc rozwiazywania problemow.",
        "skills_raw": ["Python", "kompetencje komunikacyjne", "umiejetnosc rozwiazywania problemow"],
    },
    {
        "id": "syn-010",
        "language": "de",
        "title": "Reporting Specialist",
        "description": "Gesucht werden SQL, Problemlosung und programming in Python in einem internationalen Team.",
        "skills_raw": ["SQL", "Problemlosung", "programming in Python"],
    },
    {
        "id": "syn-011",
        "language": "es",
        "title": "Analista Junior",
        "description": "Buscamos a alguien con Python, habilidades de comunicacion y Structured Query Language.",
        "skills_raw": ["Python", "habilidades de comunicacion", "Structured Query Language"],
    },
    {
        "id": "syn-012",
        "language": "fr",
        "title": "Consultant Data",
        "description": "Il faut resolution de problemes, SQL et programmation Python pour ce poste.",
        "skills_raw": ["resolution de problemes", "SQL", "programmation Python"],
    },
]


def write_jsonl(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in RECORDS:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    output_path = Path(__file__).with_name("generated_jobs_multilingual.jsonl")
    write_jsonl(output_path)
    print(f"Wrote {len(RECORDS)} records to {output_path}")


if __name__ == "__main__":
    main()
