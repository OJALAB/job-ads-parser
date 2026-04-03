from __future__ import annotations

import json
from pathlib import Path

from esco_skill_batch.esco import load_index
from esco_skill_batch.io_utils import read_records
from esco_skill_batch.text_utils import normalize_text, tokenize
from esco_skill_batch.types import EscoSkill, SkillMatch, SkillMention


class LexicalMatcher:
    def __init__(self, index_dir: Path) -> None:
        self.skills, self.manifest, self.token_index, self.exact_label_index = load_index(index_dir)

    def match(self, mention: SkillMention, top_k: int, score_threshold: float) -> list[SkillMatch]:
        normalized_mention = normalize_text(mention.text)
        mention_tokens = set(tokenize(mention.text))
        candidate_ids: set[int] = set()

        if normalized_mention in self.exact_label_index:
            candidate_ids.update(self.exact_label_index[normalized_mention])
        for token in mention_tokens:
            for idx in self.token_index.get(token, []):
                candidate_ids.add(idx)
        if not candidate_ids:
            candidate_ids = set(range(len(self.skills)))

        scored: list[SkillMatch] = []
        for idx in candidate_ids:
            skill = self.skills[idx]
            score, matched_on = self._score(skill, normalized_mention, mention_tokens)
            if score >= score_threshold:
                scored.append(
                    SkillMatch(
                        concept_uri=skill.concept_uri,
                        preferred_label=skill.preferred_label,
                        category=skill.category,
                        score=score,
                        matched_on=matched_on,
                        skill_type=skill.skill_type,
                        reuse_level=skill.reuse_level,
                    )
                )

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    def _score(self, skill: EscoSkill, normalized_mention: str, mention_tokens: set[str]) -> tuple[float, str]:
        best_score = 0.0
        best_source = "search_text"

        for label in skill.labels_normalized:
            label_tokens = set(tokenize(label))
            if not label_tokens:
                continue
            if normalized_mention == label:
                return 1.0, "exact_label"

            overlap = len(mention_tokens & label_tokens)
            containment = overlap / max(len(mention_tokens), 1)
            jaccard = overlap / max(len(mention_tokens | label_tokens), 1)
            substring_bonus = 0.15 if normalized_mention and normalized_mention in label else 0.0
            score = min(0.95, 0.65 * containment + 0.2 * jaccard + substring_bonus)
            if score > best_score:
                best_score = score
                best_source = "label_overlap"

        if best_score == 0.0 and skill.description:
            description = normalize_text(skill.description)
            if normalized_mention and normalized_mention in description:
                best_score = 0.35
                best_source = "description_substring"

        return best_score, best_source


class ReviewAliasMatcher:
    def __init__(self, base_matcher, index_dir: Path, aliases_path: Path) -> None:
        self.base_matcher = base_matcher
        self.skills, _, _, _ = load_index(index_dir)
        self.skill_by_uri = {skill.concept_uri: skill for skill in self.skills}
        self.aliases = self._load_aliases(aliases_path)

    def _load_aliases(self, aliases_path: Path) -> dict[str, dict]:
        aliases: dict[str, dict] = {}
        for row in read_records(aliases_path):
            concept_uri = str(row.get("concept_uri", "")).strip()
            if not concept_uri or concept_uri not in self.skill_by_uri:
                continue
            for candidate in [
                str(row.get("canonical_mention", "")).strip(),
                str(row.get("mention_normalized", "")).strip(),
            ]:
                normalized = normalize_text(candidate)
                if not normalized:
                    continue
                aliases[normalized] = {
                    "concept_uri": concept_uri,
                    "preferred_label": str(row.get("preferred_label", "")).strip() or self.skill_by_uri[concept_uri].preferred_label,
                }
        return aliases

    def match(self, mention: SkillMention, top_k: int, score_threshold: float) -> list[SkillMatch]:
        normalized_mention = normalize_text(mention.text)
        base_matches = self.base_matcher.match(mention, top_k=top_k, score_threshold=score_threshold)

        alias = self.aliases.get(normalized_mention)
        if alias is None:
            return base_matches

        skill = self.skill_by_uri[alias["concept_uri"]]
        alias_match = SkillMatch(
            concept_uri=skill.concept_uri,
            preferred_label=skill.preferred_label,
            category=skill.category,
            score=1.0,
            matched_on="review_alias",
            skill_type=skill.skill_type,
            reuse_level=skill.reuse_level,
        )

        merged = [alias_match]
        seen = {alias_match.concept_uri}
        for match in base_matches:
            if match.concept_uri in seen:
                continue
            merged.append(match)
            seen.add(match.concept_uri)
            if len(merged) >= top_k:
                break
        return merged[:top_k]


class EmbeddingMatcher:
    def __init__(self, index_dir: Path, model_name: str | None = None) -> None:
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Embedding matcher requires `numpy` and `sentence-transformers`. Install with `.[embedding]`."
            ) from exc

        self.np = np
        self.skills, self.manifest, self.token_index, self.exact_label_index = load_index(index_dir)
        self.embeddings_path = index_dir / "embeddings.npy"
        if not self.embeddings_path.exists():
            raise RuntimeError(
                f"Embedding matrix not found at {self.embeddings_path}. Rebuild index with --embedding-model."
            )
        self.embeddings = np.load(self.embeddings_path)
        self.model_name = model_name or self.manifest.get("embedding_model")
        if not self.model_name:
            raise RuntimeError("No embedding model recorded in the index manifest.")
        self.model = SentenceTransformer(self.model_name)

    def match(self, mention: SkillMention, top_k: int, score_threshold: float) -> list[SkillMatch]:
        vector = self.model.encode([mention.text], normalize_embeddings=True, show_progress_bar=False)
        scores = self.embeddings @ vector[0]
        top_indices = self.np.argsort(scores)[::-1][:top_k]
        results: list[SkillMatch] = []
        for idx in top_indices:
            score = float(scores[idx])
            if score < score_threshold:
                continue
            skill = self.skills[int(idx)]
            results.append(
                SkillMatch(
                    concept_uri=skill.concept_uri,
                    preferred_label=skill.preferred_label,
                    category=skill.category,
                    score=score,
                    matched_on="embedding",
                    skill_type=skill.skill_type,
                    reuse_level=skill.reuse_level,
                )
            )
        return results


class HybridMatcher:
    def __init__(self, index_dir: Path, model_name: str | None = None) -> None:
        self.lexical = LexicalMatcher(index_dir)
        self.embedding = EmbeddingMatcher(index_dir, model_name=model_name)

    def match(self, mention: SkillMention, top_k: int, score_threshold: float) -> list[SkillMatch]:
        lexical_matches = self.lexical.match(mention, top_k=top_k * 3, score_threshold=0.0)
        lexical_by_uri = {item.concept_uri: item for item in lexical_matches}
        embedding_matches = self.embedding.match(mention, top_k=top_k * 3, score_threshold=0.0)

        merged: dict[str, SkillMatch] = {}
        for match in embedding_matches:
            lexical_score = lexical_by_uri.get(match.concept_uri).score if match.concept_uri in lexical_by_uri else 0.0
            combined_score = 0.75 * match.score + 0.25 * lexical_score
            if combined_score < score_threshold:
                continue
            merged[match.concept_uri] = SkillMatch(
                concept_uri=match.concept_uri,
                preferred_label=match.preferred_label,
                category=match.category,
                score=combined_score,
                matched_on="hybrid",
                skill_type=match.skill_type,
                reuse_level=match.reuse_level,
            )

        if not merged:
            for match in lexical_matches:
                if match.score >= score_threshold:
                    merged[match.concept_uri] = match

        results = sorted(merged.values(), key=lambda item: item.score, reverse=True)
        return results[:top_k]


def build_embeddings(index_dir: Path, model_name: str, batch_size: int) -> None:
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Building embeddings requires `numpy` and `sentence-transformers`. Install with `.[embedding]`."
        ) from exc

    skills, manifest, _, _ = load_index(index_dir)
    model = SentenceTransformer(model_name)
    texts = [skill.search_text for skill in skills]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    np.save(index_dir / "embeddings.npy", embeddings)
    manifest["embedding_model"] = model_name
    (index_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
