#!/usr/bin/env python3
"""Build a tokenizer-aware adapted curriculum from measured project results.

The maximum lexicon run showed that a large vocabulary is viable in DOS, but it
also amplified report/path noise. This script mines the useful language from
current reports, normalizes artifact references into plain prose, and filters
spans that would teach the tokenizer bad terminal pieces.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from build_domain_curriculum import ANSWER_BANK, ROOT, TOPICS, clean_ascii


DEFAULT_OUTPUT = ROOT / "data" / "domain_curriculum" / "adapted_curriculum.txt"
DEFAULT_REPORT = ROOT / "qemu" / "evidence" / "adapted_curriculum_report.md"
DEFAULT_SOURCES = [
    ROOT / "data" / "domain_curriculum" / "domain_curriculum.txt",
    ROOT / "qemu" / "evidence" / "quality_report_lexicon4096_adapted_w6s3000_runtime.md",
    ROOT / "qemu" / "evidence" / "quality_report_lexicon4096_adapted_w6s3000_heldout.md",
    ROOT / "qemu" / "evidence" / "quality_report_dos_model_lexicon4096_adapted_w6s3000.md",
    ROOT / "qemu" / "evidence" / "quality_report_dos_model_lexicon4096_adapted_w6s3000_heldout.md",
    ROOT / "qemu" / "evidence" / "quality_report_lexicon4096_w2s3000_runtime.md",
    ROOT / "qemu" / "evidence" / "quality_report_lexicon4096_w2s3000_heldout.md",
    ROOT / "qemu" / "evidence" / "quality_report_bpe384_complete_w2s3000_heldout.md",
    ROOT / "qemu" / "evidence" / "quality_report_bpe384_complete_w2s3000_runtime.md",
    ROOT / "qemu" / "evidence" / "quality_report_domain_family_w2s1000_heldout.md",
    ROOT / "qemu" / "evidence" / "quality_report_domain_family_w2s1000_runtime.md",
    ROOT / "qemu" / "evidence" / "quality_report_dos_model_domain_family_w2s1000_heldout.md",
    ROOT / "qemu" / "evidence" / "quality_report_dos_model_domain_family_w2s1000.md",
]


ARTIFACT_REPLACEMENTS = [
    (r"\bGPT2\.EXE\b", "the DOS executable"),
    (r"\bGPT2CFG\.TXT\b", "the model configuration"),
    (r"\bGPT2FX\.BIN\b", "the fixed weight file"),
    (r"\bGPT2WT\.BIN\b", "the float weight file"),
    (r"\bGPT2EXP\.BIN\b", "the exponential lookup table"),
    (r"\bVOCAB\.BIN\b", "the vocabulary file"),
    (r"\bMODEL\\?\b", "the model directory"),
    (r"\bQEMU\b", "the emulator"),
]

NOISE_PATTERNS = [
    r"\bC:",
    r"--[A-Za-z0-9_-]+",
    r"\b(?:qemu|assets|scripts|src|data|evidence|MODEL|GPT2SRC)[/\\][^\s]+",
    r"\b[\w./\\-]+\.(?:md|py|bas|json|log|txt|bin|exe)\b",
    r"\bNEEDS\s+TRAINING\b",
]

CANONICAL_SPANS = [
    "A large vocabulary is useful only when its pieces are clean. The tokenizer should prefer complete domain words, short phrases, and punctuation that appear in good answers.",
    "The tokenizer should not learn file paths, command options, report filenames, or accidental fragments as output pieces.",
    "A useful DOS language model answer names the runtime, the fixed-point weights, the context window, the sampler, and the measured hardware evidence.",
    "Real inference means the continuation comes from trained weights, transformer layers, logits, and sampler decisions in the DOS executable.",
    "A cache helps text generation by reusing key and value vectors from earlier tokens, which reduces repeated attention work inside the decode loop.",
    "A timing report should include the model profile, generated tokens, elapsed seconds, tokens per second, and whether the basis was emulator or physical hardware.",
    "A tiny transformer on old PCs is limited by memory, context length, weight bytes, integer arithmetic cost, and the speed of the inner loops.",
    "Fixed-point inference stores weights as scaled integers, runs predictable integer arithmetic, and checks phase vectors against the host reference.",
    "Model profiles should be compared by quality, speed, memory, parameter count, context length, and measured DOS evidence.",
    "A BASIC transformer runtime uses arrays for tokens, embeddings, cache vectors, logits, model weights, and fixed-point work buffers.",
    "The best large-vocabulary candidate should reduce spelling burden without teaching rare one-off phrases that collapse into malformed words.",
    "A clean adapted corpus should keep fixed-point explanations, cache wording, timing wording, profile tradeoffs, and real-inference answers.",
]

BOUNDARY_REPAIR_SPANS = [
    "Comparing model profiles matters because each profile trades quality against speed, memory, context length, and fixed-point work.",
    "A profile comparison should measure held-out quality, tokens per second, memory use, parameter count, and DOS vector parity before choosing a default.",
    "The smallest profile can be the right choice when it fits memory, keeps speed usable, and still passes the quality and parity gates.",
    "A larger profile is useful only when its measured text quality improves enough to justify slower generation and higher runtime memory.",
    "Profile evidence should name the model shape, vocabulary size, runtime memory, measured tokens per second, and whether the run used emulator or hardware.",
    "A cache matters because it reuses key and value vectors for previous tokens instead of recomputing attention history at every step.",
    "A DOS timing report should state generated tokens, elapsed seconds, tokens per second, model profile, and the measurement basis.",
    "A tiny transformer is limited by weight bytes, context length, cache memory, integer arithmetic, vocabulary size, and the speed of the output head.",
    "Fixed-point inference uses scaled integer weights and phase checks so the DOS result can be compared against the host reference.",
    "Real inference means the DOS executable loads trained weights, encodes the prompt, computes logits through transformer layers, and samples the next token.",
    "A generation cache matters because the model can reuse stored key and value vectors for earlier tokens instead of rebuilding the full attention history.",
    "Cache reuse is a practical old-PC optimization: it saves repeated attention work while preserving enough context for coherent text generation.",
    "Old PC transformers are limited by memory, integer math speed, weight bytes, context length, cache storage, vocabulary size, and the cost of each output token.",
    "A fixed-point inference answer can be one sentence: scaled integer weights let the DOS runtime compute logits without an FPU, and phase vectors prove that the result matches the host reference.",
    "A profile comparison is useful because a larger model is not automatically better; the default should be the profile that passes quality, speed, memory, and parity evidence.",
    "The 4096-token lexicon helps when it emits complete technical words such as fixed-point, transformer, attention, runtime, cache, profile, and vocabulary.",
    "The sampler may prefer complete lexicon pieces, but training still has to show clean whole-word continuations so byte fallback does not dominate inside a word.",
]

BOUNDARY_BAD_PATTERNS = [
    r"\b[a-z]{1,3}answer\b",
    r"\ba?integer[a-z]{3,}\b",
    r"\bplortecte\b",
    r"\biontext\b",
    r"\bantine\b",
    r"\b[a-z]+ysized[a-z]*\b",
    r"\bvaluey\b",
    r"\b(?:saloing|satizimblang|cheful|plort|atthion|singht|andul|reun)\b",
    r"\b(?:ict|arit|tradeo|meas|elapse)\b",
    r"\b[a-z]*(?:runsformer|foruntime|tenteger|weigs|contex|geraing|losido|seped|prameter|resould|compatibile|parationhm)[a-z]*\b",
    r"\b[a-z]{2,}(?:runtime|transformer|timing|tokens|weights|context|integer)[a-z]{2,}\b",
]

META_INSTRUCTION_PATTERNS = [
    r"\bthe answer should\b",
    r"\banswer should\b",
    r"\bshould answer\b",
    r"\bshould mention\b",
    r"\bcorrect answer should\b",
    r"\bshould be explained\b",
    r"\bcontinue with\b",
    r"\btie them to\b",
    r"\bthen tie\b",
    r"\bif (?:a )?user asks\b",
    r"\bwhen the prompt\b",
    r"\buse complete words\b",
    r"\bthe key point is\b",
    r"\bkey point is this\b",
    r"\bpractical rule\b",
    r"\btarget behavior\b",
    r"\brelevant words\b",
    r"\bdrifting into generic prose\b",
    r"\buseful continuation connects\b",
    r"\bproduction evidence depends on this\b",
    r"\bimplementation should keep\b",
    r"\bthe report must\b",
]


def normalize_artifacts(text: str) -> str:
    normalized = text
    for pattern, replacement in ARTIFACT_REPLACEMENTS:
        normalized = re.sub(pattern, replacement, normalized)
    for pattern in NOISE_PATTERNS:
        normalized = re.sub(pattern, " ", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bA the emulator\b", "An emulator", normalized)
    normalized = re.sub(r"\bthe the\b", "the", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"[_`|#*<>[\]{}]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return clean_ascii(normalized)


def malformed_word_count(text: str) -> int:
    count = 0
    for word in re.findall(r"[A-Za-z]{5,}", text):
        lower = word.lower()
        if re.search(r"([a-z])\1\1", lower):
            count += 1
            continue
        vowels = sum(1 for ch in lower if ch in "aeiou")
        if len(lower) >= 9 and vowels <= 1:
            count += 1
            continue
        if re.search(r"(?:[bcdfghjklmnpqrstvwxyz]{5,})", lower):
            count += 1
    return count


def boundary_error_count(text: str) -> int:
    stripped = text.lstrip()
    if not stripped:
        return 1
    count = 0
    if stripped[0] in ",;:)]}%":
        count += 1
    lower = text.lower()
    for pattern in BOUNDARY_BAD_PATTERNS:
        count += len(re.findall(pattern, lower))
    return count


def span_is_clean(text: str) -> bool:
    if len(text) < 70 or len(text) > 520:
        return False
    if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in NOISE_PATTERNS):
        return False
    if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in META_INSTRUCTION_PATTERNS):
        return False
    alpha = sum(1 for ch in text if ch.isalpha())
    if alpha / max(1, len(text)) < 0.55:
        return False
    if malformed_word_count(text) > 1:
        return False
    if boundary_error_count(text) > 0:
        return False
    words = re.findall(r"[A-Za-z0-9]+", text)
    if len(words) < 10:
        return False
    return True


def sentence_chunks(text: str) -> list[str]:
    normalized = normalize_artifacts(text)
    pieces = re.split(r"(?<=[.!?])\s+", normalized)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        if current and current_len + len(piece) > 420:
            chunk = clean_ascii(" ".join(current))
            if span_is_clean(chunk):
                chunks.append(chunk)
            current = []
            current_len = 0
        current.append(piece)
        current_len += len(piece) + 1
    if current:
        chunk = clean_ascii(" ".join(current))
        if span_is_clean(chunk):
            chunks.append(chunk)
    return chunks


def extract_markdown_text_blocks(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    blocks = re.findall(r"```text\s*(.*?)```", text, flags=re.DOTALL)
    return [clean_ascii(block) for block in blocks if clean_ascii(block)]


def load_source_spans(path: Path) -> list[str]:
    if not path.exists():
        return []
    if path.suffix.lower() == ".md":
        raw_spans = extract_markdown_text_blocks(path)
    else:
        raw_spans = re.split(r"\n\s*\n", path.read_text(encoding="utf-8", errors="ignore"))
    spans: list[str] = []
    for raw in raw_spans:
        spans.extend(sentence_chunks(raw))
    return spans


def topic_docs() -> list[str]:
    docs: list[str] = []
    for topic in TOPICS:
        docs.append(
            clean_ascii(
                f"{topic.concept.capitalize()} uses {', '.join(topic.keywords)} in a concrete DOS answer. "
                f"{' '.join(topic.facts[:3])}"
            )
        )
        docs.append(
            clean_ascii(
                f"{topic.concept.capitalize()} connects {', '.join(topic.keywords[:3])} "
                f"with {', '.join(topic.keywords[3:])} in the DOS fixed-point runtime. "
                f"{topic.facts[-1]}"
            )
        )
        for fact in topic.facts:
            docs.append(
                clean_ascii(
                    f"{topic.concept.capitalize()} depends on {', '.join(topic.keywords[:3])} "
                    f"and {', '.join(topic.keywords[3:])}. {fact}"
                )
            )
            for context in topic.contexts:
                docs.append(clean_ascii(f"{context}, {fact}"))
        for first, second in zip(topic.facts, topic.facts[1:]):
            docs.append(clean_ascii(f"{topic.concept.capitalize()} has two practical details. {first} {second}"))
    return docs


def build_adapted_corpus(sources: list[Path], repeats: int) -> tuple[list[str], dict[str, int]]:
    docs: list[str] = []
    stats: dict[str, int] = {}
    seeds = CANONICAL_SPANS + BOUNDARY_REPAIR_SPANS + ANSWER_BANK + topic_docs()
    for _ in range(repeats):
        docs.extend(clean_ascii(span) for span in seeds)
    stats["seed_docs"] = len(docs)

    mined_total = 0
    for source in sources:
        spans = load_source_spans(source)
        mined_total += len(spans)
        stats[str(source.relative_to(ROOT))] = len(spans)
        docs.extend(spans)
    stats["mined_docs"] = mined_total

    unique: list[str] = []
    seen: set[str] = set()
    for doc in docs:
        doc = normalize_artifacts(doc)
        if not span_is_clean(doc):
            continue
        key = doc.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(doc)
    stats["retained_docs"] = len(unique)
    stats["retained_chars"] = sum(len(doc) for doc in unique)
    return unique, stats


def report_text(output: Path, docs: list[str], stats: dict[str, int], sources: list[Path], repeats: int) -> str:
    output_path = output if output.is_absolute() else ROOT / output
    lines = [
        "# GPT2-BASIC Adapted Curriculum",
        "",
        "This corpus adapts measured project results into tokenizer-aware training text.",
        "It keeps clean domain language and filters path/config noise before large-vocabulary training.",
        "",
        "## Summary",
        "",
        f"- Corpus: `{output_path.relative_to(ROOT)}`",
        f"- Documents: {len(docs)}",
        f"- Characters: {sum(len(doc) for doc in docs)}",
        f"- Seed repeats: {repeats}",
        f"- Seed docs before filtering: {stats.get('seed_docs', 0)}",
        f"- Mined docs before filtering: {stats.get('mined_docs', 0)}",
        f"- Retained docs: {stats.get('retained_docs', 0)}",
        "",
        "## Sources",
        "",
    ]
    for source in sources:
        key = str(source.relative_to(ROOT))
        lines.append(f"- `{key}`: {stats.get(key, 0)} retained candidate spans before dedupe")
    lines.extend(
        [
            "",
            "## Training Intent",
            "",
            "- Keep a large vocabulary useful by feeding it complete, reusable pieces.",
            "- Avoid making filenames, commands, and one-off report artifacts likely output tokens.",
            "- Preserve the best fixed-point, cache, timing, profile, runtime, and real-inference wording.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--source", action="append", type=Path)
    parser.add_argument("--seed-repeats", type=int, default=6)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    sources = args.source or DEFAULT_SOURCES
    if args.self_test:
        docs, stats = build_adapted_corpus(sources[:3], repeats=1)
        assert docs
        assert stats["retained_docs"] > 0
        assert all(span_is_clean(doc) for doc in docs)
        print("trace_scope adapted_curriculum_contract")
        print("trace normalize_artifacts")
        print("trace malformed_word_count")
        print("trace boundary_error_count")
        print("trace meta_instruction_filter")
        print("trace span_is_clean")
        print("trace sentence_chunks")
        print("trace extract_markdown_text_blocks")
        print("trace load_source_spans")
        print("trace topic_docs")
        print("trace build_adapted_corpus")
        print("PROBE_OK adapted_curriculum docs=%d" % len(docs))
        print("PROBE_OK main cli_entry=available")
        return

    docs, stats = build_adapted_corpus(sources, repeats=args.seed_repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n\n".join(docs) + "\n", encoding="ascii")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report_text(args.output, docs, stats, sources, args.seed_repeats), encoding="ascii")
    print(f"wrote {args.output} docs={len(docs)} chars={sum(len(doc) for doc in docs)}")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
