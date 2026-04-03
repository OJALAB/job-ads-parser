from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from esco_skill_batch.esco import load_esco_skills, save_index
from esco_skill_batch.extractors import GLiNERExtractor, OllamaExtractor, PassthroughExtractor, mentions_to_json
from esco_skill_batch.io_utils import count_records, read_records
from esco_skill_batch.matching import EmbeddingMatcher, HybridMatcher, LexicalMatcher, build_embeddings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch extraction of job-ad skills mapped to ESCO.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_index = subparsers.add_parser("build-index", help="Build a local ESCO skill index from CSV.")
    build_index.add_argument("--esco-csv", required=True, type=Path, help="Path to official ESCO skills CSV.")
    build_index.add_argument("--output-dir", required=True, type=Path, help="Where the local index should be written.")
    build_index.add_argument("--language", default=None, help="Prefer language-specific columns, e.g. en or pl.")
    build_index.add_argument("--include-knowledge", action="store_true", help="Include ESCO knowledge concepts.")
    build_index.add_argument(
        "--include-language-skills",
        action="store_true",
        help="Include ESCO language skills and knowledge concepts.",
    )
    build_index.add_argument(
        "--embedding-model",
        default=None,
        help="Optional embedding model for semantic matching, e.g. BAAI/bge-m3.",
    )
    build_index.add_argument("--embedding-batch-size", type=int, default=64)

    extract_batch = subparsers.add_parser("extract-batch", help="Extract skills from job ads in batch.")
    extract_batch.add_argument("--input", required=True, type=Path)
    extract_batch.add_argument("--output", required=True, type=Path)
    extract_batch.add_argument("--index-dir", required=True, type=Path)
    extract_batch.add_argument("--text-field", default="text")
    extract_batch.add_argument("--id-field", default="id")
    extract_batch.add_argument(
        "--extractor",
        choices=["ollama", "gliner", "passthrough"],
        default="ollama",
    )
    extract_batch.add_argument(
        "--mapping-backend",
        choices=["lexical", "embedding", "hybrid"],
        default="lexical",
    )
    extract_batch.add_argument("--mentions-field", default="skills_raw")
    extract_batch.add_argument("--ollama-model", default="qwen3:14b")
    extract_batch.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    extract_batch.add_argument("--ollama-timeout-seconds", type=int, default=120)
    extract_batch.add_argument("--ollama-temperature", type=float, default=0.0)
    extract_batch.add_argument("--gliner-model", default="urchade/gliner_multi-v2.1")
    extract_batch.add_argument("--gliner-threshold", type=float, default=0.35)
    extract_batch.add_argument("--top-k", type=int, default=5)
    extract_batch.add_argument("--score-threshold", type=float, default=0.35)
    extract_batch.add_argument("--max-records", type=int, default=None)
    extract_batch.add_argument("--keep-text", action="store_true")
    extract_batch.add_argument("--no-progress", action="store_true", help="Disable progress output on stderr.")

    return parser


class ProgressReporter:
    def __init__(self, total: int | None, enabled: bool, stream=None) -> None:
        self.total = total
        self.enabled = enabled
        self.stream = stream or sys.stderr
        self.is_tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self.started_at = time.monotonic()
        self.last_render_width = 0

    def start_record(self, completed: int, record_id: str) -> None:
        if not self.enabled:
            return
        self._render(completed=completed, record_id=record_id, phase="running")

    def complete_record(self, completed: int, record_id: str) -> None:
        if not self.enabled:
            return
        self._render(completed=completed, record_id=record_id, phase="done")

    def finish(self) -> None:
        if not self.enabled:
            return
        if self.is_tty:
            self.stream.write("\n")
            self.stream.flush()

    def _render(self, completed: int, record_id: str, phase: str) -> None:
        elapsed = time.monotonic() - self.started_at
        if self.total:
            ratio = min(max(completed / self.total, 0.0), 1.0)
            filled = int(24 * ratio)
            bar = "#" * filled + "-" * (24 - filled)
            line = (
                f"[{bar}] {completed}/{self.total} {ratio * 100:5.1f}% "
                f"elapsed {elapsed:6.1f}s {phase}: {record_id}"
            )
        else:
            line = f"{completed} records elapsed {elapsed:6.1f}s {phase}: {record_id}"

        if self.is_tty:
            padding = " " * max(self.last_render_width - len(line), 0)
            self.stream.write("\r" + line + padding)
            self.stream.flush()
            self.last_render_width = len(line)
            return

        self.stream.write(line + "\n")
        self.stream.flush()


def _make_extractor(args: argparse.Namespace):
    if args.extractor == "passthrough":
        return PassthroughExtractor(mentions_field=args.mentions_field)
    if args.extractor == "ollama":
        return OllamaExtractor(
            model=args.ollama_model,
            base_url=args.ollama_url,
            timeout_seconds=args.ollama_timeout_seconds,
            temperature=args.ollama_temperature,
        )
    return GLiNERExtractor(
        model_name=args.gliner_model,
        threshold=args.gliner_threshold,
    )


def _make_matcher(args: argparse.Namespace):
    if args.mapping_backend == "lexical":
        return LexicalMatcher(args.index_dir)
    if args.mapping_backend == "embedding":
        return EmbeddingMatcher(args.index_dir)
    return HybridMatcher(args.index_dir)


def run_build_index(args: argparse.Namespace) -> None:
    skills = load_esco_skills(
        csv_path=args.esco_csv,
        language=args.language,
        include_knowledge=args.include_knowledge,
        include_language_skills=args.include_language_skills,
    )
    save_index(
        output_dir=args.output_dir,
        skills=skills,
        source_csv=args.esco_csv,
        language=args.language,
        include_knowledge=args.include_knowledge,
        include_language_skills=args.include_language_skills,
    )
    if args.embedding_model:
        build_embeddings(
            index_dir=args.output_dir,
            model_name=args.embedding_model,
            batch_size=args.embedding_batch_size,
        )
    print(
        json.dumps(
            {
                "status": "ok",
                "indexed_skills": len(skills),
                "output_dir": str(args.output_dir),
                "embedding_model": args.embedding_model,
            },
            ensure_ascii=False,
        )
    )


def run_extract_batch(args: argparse.Namespace) -> None:
    extractor = _make_extractor(args)
    matcher = _make_matcher(args)
    total_records = count_records(args.input)
    if args.max_records is not None:
        total_records = min(total_records, args.max_records)
    progress = ProgressReporter(total=total_records, enabled=not args.no_progress)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    processed_records = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for record_number, record in enumerate(read_records(args.input), start=1):
            if args.max_records is not None and record_number > args.max_records:
                break

            record_id = str(record.get(args.id_field, str(record_number)))
            progress.start_record(completed=processed_records, record_id=record_id)

            text = str(record.get(args.text_field, "") or "")
            mentions = extractor.extract(record, text)
            matches = []
            for mention in mentions:
                candidates = matcher.match(mention, top_k=args.top_k, score_threshold=args.score_threshold)
                matches.append(
                    {
                        "mention": asdict(mention),
                        "esco_matches": [asdict(candidate) for candidate in candidates],
                    }
                )

            output_row = {
                "id": record_id,
                "skill_mentions": mentions_to_json(mentions),
                "matches": matches,
            }
            if args.keep_text:
                output_row["text"] = text

            handle.write(json.dumps(output_row, ensure_ascii=False) + "\n")
            processed_records += 1
            progress.complete_record(completed=processed_records, record_id=record_id)

    progress.finish()
    print(
        json.dumps(
            {
                "status": "ok",
                "processed_records": processed_records,
                "output": str(args.output),
                "extractor": args.extractor,
                "mapping_backend": args.mapping_backend,
            },
            ensure_ascii=False,
        )
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "build-index":
        run_build_index(args)
        return
    if args.command == "extract-batch":
        run_extract_batch(args)
        return
    parser.error(f"Unknown command: {args.command}")
