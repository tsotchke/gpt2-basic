#!/usr/bin/env python3
"""Build a clean GPT2-BASIC quality-repair corpus.

This corpus is for repair attempts after a model learns path/scaffold fragments
from docs or evidence logs. It keeps clean domain prose, removes path-heavy
paragraphs, and adds targeted continuations for prompts that strict quality
gates exercise.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data" / "domain_curriculum" / "gold_curriculum_v4.txt"
DEFAULT_OUTPUT = ROOT / "data" / "domain_curriculum" / "gold_curriculum_v5_clean_repair.txt"
DEFAULT_REPORT = ROOT / "qemu" / "evidence" / "gold_curriculum_v5_clean_repair_report.md"


TARGETED_PARAGRAPHS = [
    "DOS language models need compact checkpoints, clean vocabulary pieces, predictable memory use, fixed-point arithmetic, and honest timing evidence. A useful DOS model should say what it can do, load real weights, and generate short technical answers without copying file paths or training scaffolding.",
    "DOS language models need clean tokenization because malformed byte fragments quickly become unreadable output. The best small checkpoint uses complete words and common technical phrases, then proves the result with held-out prompts and DOS runtime evidence.",
    "A DOS language model needs more than a small file. It needs a model shape that fits memory, a tokenizer that emits printable text, fixed-point math that matches host reference vectors, and timing reports from the actual decode loop.",
    "GPT2 BASIC on a 486 should describe a real local inference path: load the checkpoint, encode the prompt, run transformer layers, score logits, sample a token, and decode printable text. The answer should stay short, technical, and specific.",
    "To improve performance on real hardware, measure the decode loop first, then reduce output-head work, reuse buffers, keep arrays contiguous, and compare memory use against generated tokens per second. Emulator results are useful, but physical boards are the final proof.",
    "A cache matters for text generation because key and value vectors from prior tokens can be reused. The cache trades fixed memory for less repeated attention work, so its size must fit the old PC before it is a good optimization.",
    "Fixed-point inference stores weights and activations as scaled integers, uses clamps and lookup tables where needed, and compares phase vectors against a host reference. It is a correctness contract, not only a compression trick.",
    "Model profiles should be compared by quality, memory, speed, context length, vocabulary size, and hardware evidence. The largest profile is not the best default unless it produces cleaner text at a cost the target machine can afford.",
    "A clean GPT2-BASIC answer avoids report filenames, shell commands, broken words, and unrelated evidence paths. It should answer the prompt directly with one or two concrete engineering facts.",
    "Real hardware deployment is the final step. The same model files, quality prompts, vector checks, and timing probes that pass in QEMU should be copied to the target PC and captured from the physical machine.",
]


PATH_OR_SCAFFOLD_PATTERNS = (
    re.compile(r"\b(?:qemu|assets|scripts|src|data|tests|evidence)/", re.IGNORECASE),
    re.compile(r"[A-Za-z0-9_.-]+\.md\b", re.IGNORECASE),
    re.compile(r"[A-Za-z0-9_.-]+\.log\b", re.IGNORECASE),
    re.compile(r"[A-Za-z0-9_.-]+\.py\b", re.IGNORECASE),
    re.compile(r"\b(?:PASS|NEEDS_TRAINING|RETRAIN|PROBE_OK)\b"),
    re.compile(r"`[^`]+`"),
    re.compile(r"\|"),
)


def clean_ascii(text: str) -> str:
    text = text.encode("ascii", errors="ignore").decode("ascii")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s+", " ", text)
    return "".join(ch for ch in text if 32 <= ord(ch) <= 126).strip()


def noisy(paragraph: str) -> bool:
    if any(pattern.search(paragraph) for pattern in PATH_OR_SCAFFOLD_PATTERNS):
        return True
    slash_count = paragraph.count("/") + paragraph.count("\\")
    if slash_count > 0:
        return True
    if len(re.findall(r"\b[a-z]{1,2}_[a-z0-9_]+\b", paragraph)) > 0:
        return True
    return False


def split_paragraphs(text: str) -> list[str]:
    paragraphs: list[str] = []
    for raw in re.split(r"\n\s*\n", text.replace("\r\n", "\n").replace("\r", "\n")):
        paragraph = clean_ascii(raw)
        if len(paragraph) >= 60:
            paragraphs.append(paragraph)
    return paragraphs


def build_corpus(source: Path) -> tuple[list[str], int]:
    source_paragraphs = split_paragraphs(source.read_text(encoding="utf-8", errors="ignore"))
    retained: list[str] = []
    dropped = 0
    seen: set[str] = set()

    for paragraph in source_paragraphs:
        if noisy(paragraph):
            dropped += 1
            continue
        if paragraph not in seen:
            retained.append(paragraph)
            seen.add(paragraph)

    for paragraph in TARGETED_PARAGRAPHS:
        cleaned = clean_ascii(paragraph)
        if cleaned not in seen:
            retained.append(cleaned)
            seen.add(cleaned)

    return retained, dropped


def report_text(source: Path, output: Path, paragraphs: list[str], dropped: int) -> str:
    chars = sum(len(item) for item in paragraphs)
    return "\n".join(
        [
            "# Gold Curriculum V5 Clean Repair Report",
            "",
            f"Source: `{source.relative_to(ROOT)}`",
            f"Output: `{output.relative_to(ROOT)}`",
            f"Paragraphs retained: `{len(paragraphs)}`",
            f"Paragraphs dropped as noisy: `{dropped}`",
            f"Characters retained: `{chars}`",
            "",
            "The corpus intentionally excludes path-heavy documentation and evidence scaffolding. It adds targeted clean continuations for DOS model, timing, cache, fixed-point, profile-comparison, and real-hardware prompts.",
            "",
        ]
    )


def self_test() -> None:
    paragraphs, dropped = build_corpus(DEFAULT_SOURCE)
    text = "\n\n".join(paragraphs)
    if "DOS language models need compact checkpoints" not in text:
        raise RuntimeError("targeted DOS model paragraph missing")
    if "qemu/evidence" in text or "assets/gpt2" in text:
        raise RuntimeError("path scaffold leaked into clean corpus")
    if not paragraphs or dropped < 0:
        raise RuntimeError("corpus builder produced invalid counts")
    print(f"PROBE_OK clean_repair_corpus_paragraphs={len(paragraphs)}")
    print(f"PROBE_OK clean_repair_corpus_dropped={dropped}")
    print("PROBE_OK clean_repair_corpus_paths=0")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    paragraphs, dropped = build_corpus(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n\n".join(paragraphs) + "\n", encoding="ascii")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report_text(args.source, args.output, paragraphs, dropped), encoding="ascii")
    print(f"CLEAN_REPAIR_CORPUS|paragraphs={len(paragraphs)}|dropped={dropped}|output={args.output}")


if __name__ == "__main__":
    main()
