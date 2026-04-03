from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from esco_skill_batch.esco import load_esco_skills, save_index
from esco_skill_batch.evaluation import build_record_report, evaluate_predictions, render_record_report_markdown
from esco_skill_batch.extractors import (
    GLiNERExtractor,
    HFTokenClassificationExtractor,
    OllamaExtractor,
    PassthroughExtractor,
    mentions_to_json,
)
from esco_skill_batch.gliner_training import prepare_gliner_datasets, train_gliner_model
from esco_skill_batch.io_utils import count_records, read_records
from esco_skill_batch.matching import EmbeddingMatcher, HybridMatcher, LexicalMatcher, build_embeddings
from esco_skill_batch.prompt_presets import OLLAMA_PROMPT_PRESETS, resolve_ollama_system_prompt


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
        choices=["ollama", "gliner", "hf_token_classifier", "passthrough"],
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
    extract_batch.add_argument(
        "--ollama-prompt-preset",
        choices=sorted(OLLAMA_PROMPT_PRESETS),
        default="default_en",
        help="Named system prompt preset for Ollama extraction.",
    )
    extract_batch.add_argument(
        "--ollama-system-prompt-file",
        type=Path,
        default=None,
        help="Optional path to a custom system prompt file for Ollama extraction.",
    )
    extract_batch.add_argument("--gliner-model", default="urchade/gliner_multi-v2.1")
    extract_batch.add_argument("--gliner-threshold", type=float, default=0.35)
    extract_batch.add_argument("--hf-model", default="jjzha/escoxlmr_skill_extraction")
    extract_batch.add_argument(
        "--hf-aggregation-strategy",
        choices=["none", "simple", "first", "average", "max"],
        default="simple",
    )
    extract_batch.add_argument(
        "--hf-entity-labels",
        default="",
        help="Comma-separated HF token classification labels to keep. Empty means keep all non-O entities.",
    )
    extract_batch.add_argument("--hf-device", type=int, default=-1, help="HF pipeline device. Use -1 for CPU.")
    extract_batch.add_argument("--top-k", type=int, default=5)
    extract_batch.add_argument("--score-threshold", type=float, default=0.35)
    extract_batch.add_argument("--max-records", type=int, default=None)
    extract_batch.add_argument("--keep-text", action="store_true")
    extract_batch.add_argument("--no-progress", action="store_true", help="Disable progress output on stderr.")

    evaluate = subparsers.add_parser("evaluate", help="Evaluate predictions against a gold JSONL file.")
    evaluate.add_argument("--gold", required=True, type=Path)
    evaluate.add_argument("--predictions", required=True, type=Path)
    evaluate.add_argument("--top-k", type=int, default=5)

    report = subparsers.add_parser("report", help="Generate a per-record evaluation report.")
    report.add_argument("--gold", required=True, type=Path)
    report.add_argument("--predictions", required=True, type=Path)
    report.add_argument("--output", required=True, type=Path)
    report.add_argument("--top-k", type=int, default=5)
    report.add_argument("--format", choices=["markdown", "json"], default="markdown")

    prepare_gliner = subparsers.add_parser(
        "prepare-gliner-data",
        help="Convert labeled JSONL/JSON data into GLiNER training files.",
    )
    prepare_gliner.add_argument("--input", required=True, nargs="+", type=Path)
    prepare_gliner.add_argument("--output-dir", required=True, type=Path)
    prepare_gliner.add_argument("--text-field", default="description")
    prepare_gliner.add_argument("--skills-field", default="gold_skills")
    prepare_gliner.add_argument("--label", default="skill")
    prepare_gliner.add_argument("--dev-ratio", type=float, default=0.2)
    prepare_gliner.add_argument("--seed", type=int, default=42)
    prepare_gliner.add_argument(
        "--keep-empty",
        action="store_true",
        help="Keep windows without any skills as negative training examples.",
    )
    prepare_gliner.add_argument(
        "--max-tokens",
        type=int,
        default=300,
        help="Split long records into windows of at most this many tokens. Use 0 to disable chunking.",
    )
    prepare_gliner.add_argument(
        "--window-stride",
        type=int,
        default=300,
        help="Stride between token windows when chunking long records.",
    )
    prepare_gliner.add_argument("--max-records", type=int, default=None)
    prepare_gliner.add_argument(
        "--fail-on-unmatched",
        action="store_true",
        help="Stop if a labeled mention cannot be aligned back into the source text.",
    )

    train_gliner = subparsers.add_parser("train-gliner", help="Fine-tune a GLiNER model on prepared JSON data.")
    train_gliner.add_argument("--train-data", required=True, type=Path)
    train_gliner.add_argument("--dev-data", type=Path, default=None)
    train_gliner.add_argument("--model-name", default="urchade/gliner_large-v2.1")
    train_gliner.add_argument("--output-dir", required=True, type=Path)
    train_gliner.add_argument("--learning-rate", type=float, default=1e-5)
    train_gliner.add_argument("--others-learning-rate", type=float, default=3e-5)
    train_gliner.add_argument("--weight-decay", type=float, default=0.1)
    train_gliner.add_argument("--others-weight-decay", type=float, default=0.01)
    train_gliner.add_argument("--warmup-ratio", type=float, default=0.05)
    train_gliner.add_argument("--train-batch-size", type=int, default=2)
    train_gliner.add_argument("--eval-batch-size", type=int, default=2)
    train_gliner.add_argument("--max-steps", type=int, default=250)
    train_gliner.add_argument("--save-steps", type=int, default=50)
    train_gliner.add_argument("--logging-steps", type=int, default=10)
    train_gliner.add_argument("--save-total-limit", type=int, default=2)
    train_gliner.add_argument("--max-grad-norm", type=float, default=10.0)
    train_gliner.add_argument("--negatives", type=float, default=1.0)
    train_gliner.add_argument("--loss-alpha", type=float, default=0.75)
    train_gliner.add_argument("--loss-gamma", type=float, default=0.0)
    train_gliner.add_argument("--loss-prob-margin", type=float, default=0.0)
    train_gliner.add_argument("--loss-reduction", default="sum")
    train_gliner.add_argument("--masking", default="none")
    train_gliner.add_argument("--scheduler-type", default="cosine")
    train_gliner.add_argument("--gradient-accumulation-steps", type=int, default=1)
    train_gliner.add_argument("--dataloader-num-workers", type=int, default=0)
    train_gliner.add_argument("--freeze-components", default="")
    train_gliner.add_argument("--seed", type=int, default=42)
    train_gliner.add_argument("--use-cpu", action="store_true")
    train_gliner.add_argument("--bf16", action="store_true")
    train_gliner.add_argument("--compile-model", action="store_true")

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
        custom_prompt = None
        if args.ollama_system_prompt_file is not None:
            custom_prompt = args.ollama_system_prompt_file.read_text(encoding="utf-8")
        return OllamaExtractor(
            model=args.ollama_model,
            base_url=args.ollama_url,
            timeout_seconds=args.ollama_timeout_seconds,
            temperature=args.ollama_temperature,
            system_prompt=resolve_ollama_system_prompt(
                preset=args.ollama_prompt_preset,
                custom_prompt=custom_prompt,
            ),
        )
    if args.extractor == "hf_token_classifier":
        entity_labels = [item.strip() for item in args.hf_entity_labels.split(",") if item.strip()]
        return HFTokenClassificationExtractor(
            model_name=args.hf_model,
            aggregation_strategy=args.hf_aggregation_strategy,
            entity_labels=entity_labels,
            device=args.hf_device,
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


def run_prepare_gliner_data(args: argparse.Namespace) -> None:
    manifest = prepare_gliner_datasets(
        input_paths=args.input,
        output_dir=args.output_dir,
        text_field=args.text_field,
        skills_field=args.skills_field,
        label=args.label,
        dev_ratio=args.dev_ratio,
        seed=args.seed,
        keep_empty=args.keep_empty,
        max_tokens=None if args.max_tokens == 0 else args.max_tokens,
        window_stride=None if args.window_stride == 0 else args.window_stride,
        max_records=args.max_records,
        fail_on_unmatched=args.fail_on_unmatched,
    )
    print(json.dumps(manifest, ensure_ascii=False))


def run_train_gliner(args: argparse.Namespace) -> None:
    freeze_components = [item.strip() for item in args.freeze_components.split(",") if item.strip()]
    summary = train_gliner_model(
        train_data=args.train_data,
        dev_data=args.dev_data,
        model_name=args.model_name,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        others_learning_rate=args.others_learning_rate,
        weight_decay=args.weight_decay,
        others_weight_decay=args.others_weight_decay,
        warmup_ratio=args.warmup_ratio,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        max_grad_norm=args.max_grad_norm,
        negatives=args.negatives,
        loss_alpha=args.loss_alpha,
        loss_gamma=args.loss_gamma,
        loss_prob_margin=args.loss_prob_margin,
        loss_reduction=args.loss_reduction,
        masking=args.masking,
        scheduler_type=args.scheduler_type,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_num_workers=args.dataloader_num_workers,
        freeze_components=freeze_components or None,
        use_cpu=args.use_cpu,
        bf16=args.bf16,
        compile_model=args.compile_model,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "build-index":
        run_build_index(args)
        return
    if args.command == "extract-batch":
        run_extract_batch(args)
        return
    if args.command == "evaluate":
        print(json.dumps(evaluate_predictions(args.gold, args.predictions, top_k=args.top_k), ensure_ascii=False))
        return
    if args.command == "report":
        metrics = evaluate_predictions(args.gold, args.predictions, top_k=args.top_k)
        report = build_record_report(args.gold, args.predictions, top_k=args.top_k)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.format == "json":
            args.output.write_text(
                json.dumps({"metrics": metrics, "report": report}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        else:
            args.output.write_text(
                render_record_report_markdown(report, metrics),
                encoding="utf-8",
            )
        print(json.dumps({"status": "ok", "output": str(args.output), "format": args.format}, ensure_ascii=False))
        return
    if args.command == "prepare-gliner-data":
        run_prepare_gliner_data(args)
        return
    if args.command == "train-gliner":
        run_train_gliner(args)
        return
    parser.error(f"Unknown command: {args.command}")
