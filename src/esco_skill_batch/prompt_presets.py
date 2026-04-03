from __future__ import annotations


DEFAULT_OLLAMA_PROMPT = """Extract only explicit skill mentions from the job ad.

Rules:
- Return only skills and transversal skills.
- Do not return occupations, company names, benefits, locations, salaries, teams or departments.
- Keep mentions verbatim from the source text whenever possible.
- Do not invent normalized names.
- Deduplicate while preserving order.
- If no skills are present, return an empty list.
"""


BIELIK_PL_OLLAMA_PROMPT = """Wyodrebnij z ogloszenia o prace wyłącznie jawnie wymienione umiejetnosci.

Zasady:
- Zwracaj tylko skills i transversal skills.
- Nie zwracaj stanowisk, nazw firm, benefitow, lokalizacji, wynagrodzen, nazw zespolow ani dzialow.
- Zachowuj brzmienie fraz z tekstu zrodlowego, jesli to mozliwe.
- Nie normalizuj nazw i niczego nie dopowiadaj.
- Usun duplikaty, zachowujac kolejnosc pierwszego wystapienia.
- Jesli w tekscie nie ma umiejetnosci, zwroc pusta liste.
"""


OLLAMA_PROMPT_PRESETS = {
    "default_en": DEFAULT_OLLAMA_PROMPT,
    "bielik_pl": BIELIK_PL_OLLAMA_PROMPT,
}


def resolve_ollama_system_prompt(
    preset: str = "default_en",
    custom_prompt: str | None = None,
) -> str:
    if custom_prompt is not None:
        return custom_prompt
    try:
        return OLLAMA_PROMPT_PRESETS[preset]
    except KeyError as exc:
        available = ", ".join(sorted(OLLAMA_PROMPT_PRESETS))
        raise ValueError(f"Unknown Ollama prompt preset: {preset}. Available: {available}") from exc
