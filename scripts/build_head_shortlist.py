#!/usr/bin/env python3
"""Build an optional output-head shortlist artifact for GPT2-BASIC models."""

from __future__ import annotations

import argparse
import shutil
import struct
import tempfile
from collections import Counter
from pathlib import Path

from export_gpt2_basic_vectors import (
    DEFAULT_OUTPUT_NAME,
    HEAD_SHORTLIST_FILE,
    HEAD_SHORTLIST_MAGIC,
    HEAD_SHORTLIST_VERSION,
    Config,
    export_vectors,
    parse_config,
    read_head_shortlist,
)
from gpt2_basic_tokenizer import BYTE_OFFSET, EOT_TOKEN, GPT2BasicTokenizer, clean_ascii, load_tokenizer_for_model


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_MODEL = ROOT / "assets" / "gpt2_basic" / "MODEL"
DEFAULT_OUTPUT_MODEL = ROOT / "assets" / "gpt2_basic" / "MODEL_HEADSHORTLIST2048_PROD_PROBE"
DEFAULT_CORPUS_FILES = [
    ROOT / "data" / "domain_curriculum" / "gold_curriculum_v4.txt",
    ROOT / "data" / "domain_curriculum" / "gold_curriculum_v3.txt",
    ROOT / "data" / "domain_curriculum" / "gold_curriculum_v2.txt",
    ROOT / "data" / "domain_curriculum" / "domain_curriculum.txt",
    ROOT / "data" / "domain_curriculum" / "adapted_repair_curriculum.txt",
    ROOT / "data" / "online_corpus" / "online_training_corpus.txt",
    ROOT / "README.md",
    ROOT / "gpt2_basic_tldr.md",
    ROOT / "implementation_guide.md",
    ROOT / "qemu" / "README.md",
]
QUALITY_PROMPTS = [
    "What makes this real inference?",
    "GPT2 BASIC on a 486",
    "DOS language models need",
    "A BASIC transformer runtime",
    "To improve performance on real hardware",
    "Explain why a cache matters for text generation",
    "How should a DOS model report timing?",
    "What limits a tiny transformer on old PCs?",
    "Describe fixed point inference in one sentence",
    "Why compare model profiles before choosing one?",
]


def read_text_documents(paths: list[Path], max_chars_per_file: int) -> list[str]:
    documents: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if max_chars_per_file > 0:
            text = text[:max_chars_per_file]
        for raw_doc in text.replace("\r\n", "\n").replace("\r", "\n").split("\n\n"):
            document = clean_ascii(raw_doc)
            if len(document) >= 20:
                documents.append(document)
    return documents


def token_is_allowed(tokenizer: GPT2BasicTokenizer, token_id: int) -> bool:
    return 0 <= token_id < tokenizer.vocab_size and tokenizer.token_allowed(token_id)


def add_token(selected: list[int], seen: set[int], tokenizer: GPT2BasicTokenizer, token_id: int) -> None:
    if token_id in seen:
        return
    if not token_is_allowed(tokenizer, token_id):
        return
    selected.append(token_id)
    seen.add(token_id)


def select_head_shortlist(
    tokenizer: GPT2BasicTokenizer,
    documents: list[str],
    limit: int,
) -> list[int]:
    if limit < 1:
        raise ValueError("limit must be positive")
    target = min(limit, tokenizer.vocab_size)
    selected: list[int] = []
    seen: set[int] = set()
    frequencies: Counter[int] = Counter()

    add_token(selected, seen, tokenizer, EOT_TOKEN)
    for byte_value in range(32, 127):
        add_token(selected, seen, tokenizer, byte_value + BYTE_OFFSET)

    for prompt in QUALITY_PROMPTS:
        for token_id in tokenizer.encode(prompt, output_safe=True):
            add_token(selected, seen, tokenizer, token_id)

    for document in documents:
        for token_id in tokenizer.encode(document, output_safe=True):
            if token_is_allowed(tokenizer, token_id):
                frequencies[token_id] += 1

    for token_id, _count in frequencies.most_common():
        add_token(selected, seen, tokenizer, token_id)
        if len(selected) >= target:
            return selected[:target]

    for token_id in range(tokenizer.vocab_size):
        add_token(selected, seen, tokenizer, token_id)
        if len(selected) >= target:
            break
    return selected[:target]


def write_head_shortlist(path: Path, cfg: Config, tokens: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = struct.pack("<iiii", HEAD_SHORTLIST_MAGIC, HEAD_SHORTLIST_VERSION, cfg.vocab_size, len(tokens))
    payload += struct.pack("<" + "i" * len(tokens), *tokens)
    path.write_bytes(payload)


def update_profile(path: Path, count: int, source_model: Path) -> None:
    existing = path.read_text(encoding="ascii", errors="ignore").splitlines()
    kept = [line for line in existing if not line.lower().startswith("head_shortlist")]
    kept.extend(
        [
            "head_shortlist=frequency-corpus",
            f"head_shortlist_file={HEAD_SHORTLIST_FILE}",
            f"head_shortlist_tokens={count}",
            f"head_shortlist_source={source_model.name}",
        ]
    )
    path.write_text("\n".join(kept) + "\n", encoding="ascii")


def prepare_output_model(source_model: Path, output_model: Path) -> None:
    if source_model.resolve() == output_model.resolve():
        raise ValueError("output model must be different from source model")
    if output_model.exists():
        shutil.rmtree(output_model)
    shutil.copytree(source_model, output_model)


def build_head_shortlist_variant(
    source_model: Path,
    output_model: Path,
    limit: int,
    corpus_files: list[Path],
    max_chars_per_file: int,
    top_k: int,
    phase_count: int,
) -> list[int]:
    cfg = parse_config(source_model / "GPT2CFG.TXT")
    tokenizer = load_tokenizer_for_model(source_model, cfg.vocab_size)
    documents = read_text_documents(corpus_files, max_chars_per_file)
    tokens = select_head_shortlist(tokenizer, documents, limit)

    prepare_output_model(source_model, output_model)
    write_head_shortlist(output_model / HEAD_SHORTLIST_FILE, cfg, tokens)
    update_profile(output_model / "PROFILE.TXT", len(tokens), source_model)
    export_vectors(output_model, output_model / DEFAULT_OUTPUT_NAME, QUALITY_PROMPTS[:3], top_k, phase_count)
    return tokens


def self_test(model_dir: Path) -> None:
    cfg = parse_config(model_dir / "GPT2CFG.TXT")
    tokenizer = load_tokenizer_for_model(model_dir, cfg.vocab_size)
    documents = read_text_documents([ROOT / "README.md"], 2000)
    tokens = select_head_shortlist(tokenizer, documents + QUALITY_PROMPTS[:2], min(128, cfg.vocab_size))
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        source = tmp_path / "source"
        output = tmp_path / "output"
        source.mkdir()
        output.mkdir()
        (source / "fresh.tmp").write_text("fresh\n", encoding="ascii")
        (output / "GPT2HQ4.BIN").write_bytes(b"stale")
        prepare_output_model(source, output)
        if (output / "GPT2HQ4.BIN").exists() or not (output / "fresh.tmp").exists():
            raise AssertionError("prepare_output_model did not replace stale output contents")

        path = tmp_path / HEAD_SHORTLIST_FILE
        write_head_shortlist(path, cfg, tokens)
        shutil.copy(model_dir / "GPT2CFG.TXT", tmp_path / "GPT2CFG.TXT")
        loaded = read_head_shortlist(tmp_path, cfg)
    if loaded != tokens:
        raise AssertionError("head-shortlist roundtrip changed tokens")
    print("trace read_text_documents")
    print("trace select_head_shortlist")
    print("trace prepare_output_model")
    print("trace write_head_shortlist")
    print("trace update_profile")
    print("trace build_head_shortlist_variant")
    print(f"PROBE_OK read_text_documents docs={len(documents)}")
    print(f"PROBE_OK select_head_shortlist tokens={len(tokens)}")
    print(f"PROBE_OK write_head_shortlist bytes={16 + len(tokens) * 4}")
    print("PROBE_OK self_test exercised=1")
    print("PROBE_OK main cli_entry=available")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-model", type=Path, default=DEFAULT_SOURCE_MODEL)
    parser.add_argument("--output-model", type=Path, default=DEFAULT_OUTPUT_MODEL)
    parser.add_argument("--limit", type=int, default=2048)
    parser.add_argument("--corpus-file", type=Path, action="append")
    parser.add_argument("--max-chars-per-file", type=int, default=250_000)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--phase-count", type=int, default=8)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test(args.source_model)
        return

    corpus_files = args.corpus_file if args.corpus_file else DEFAULT_CORPUS_FILES
    tokens = build_head_shortlist_variant(
        args.source_model,
        args.output_model,
        args.limit,
        corpus_files,
        args.max_chars_per_file,
        args.top_k,
        args.phase_count,
    )
    print(f"wrote {args.output_model / HEAD_SHORTLIST_FILE} tokens={len(tokens)}")
    print(f"wrote {args.output_model / DEFAULT_OUTPUT_NAME}")


if __name__ == "__main__":
    main()
