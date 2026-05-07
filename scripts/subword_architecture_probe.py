#!/usr/bin/env python3
"""Estimate whether a compact subword vocabulary is worth implementing."""

from __future__ import annotations

import argparse
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from era_performance_model import Config, MACHINES, estimate_runtime_memory, parameter_count, seconds_for, work_counts


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT / "data" / "domain_curriculum" / "domain_curriculum.txt"
DEFAULT_OUTPUT = ROOT / "qemu" / "evidence" / "subword_architecture_probe.md"
BYTE_VOCAB = 258
MAX_TOKEN_CHARS = 16


SAMPLE_TEXTS = [
    "What makes this real inference? The continuation comes from trained weights, transformer layers, logits, and the DOS runtime.",
    "GPT2 BASIC on a 486 uses fixed weights, DOS memory, byte tokens, attention, and cache arrays to generate text.",
    "DOS language models need plain files, trained weights, prompt tokens, predictable memory, and simple command-line output.",
    "A BASIC transformer runtime uses arrays for tokens, model weights, cache vectors, logits, fixed-point math, and tight loops.",
    "To improve performance on real hardware, measure tokens per second, profile loops, reuse buffers, and reduce fixed-point work.",
    "A generation cache reuses context vectors, saves memory work, and improves next-token speed.",
    "A DOS timing report should include seconds, generated tokens, measured hardware, QEMU profile, and the PERF log.",
    "Tiny old-PC transformers are limited by memory, context, speed, weights, integer arithmetic, and hardware bandwidth.",
    "Fixed point inference uses integer weights and arithmetic to produce logits in the runtime.",
    "Model profiles should be compared by quality, speed, memory, measure, and tradeoff.",
]


@dataclass(frozen=True)
class ProbeRow:
    vocab_size: int
    token_count: int
    byte_count: int
    compression: float
    params_486sx: int
    runtime_mb_486sx: float
    tps_486sx: float
    tps_486dx2: float
    estimated_seconds_70_bytes_486dx2: float


def clean_ascii(text: str) -> str:
    text = text.encode("ascii", errors="ignore").decode("ascii")
    text = re.sub(r"\s+", " ", text)
    return "".join(ch for ch in text if 32 <= ord(ch) <= 126).strip()


def candidate_pieces(text: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9.+/_-]*", text)
    for word in words:
        if 3 <= len(word) <= MAX_TOKEN_CHARS:
            counts[word] += 2
            counts[" " + word] += 3

    phrases = re.findall(r"[A-Za-z0-9][A-Za-z0-9.+/_-]*(?: [A-Za-z0-9][A-Za-z0-9.+/_-]*){1,2}", text)
    for phrase in phrases:
        if 4 <= len(phrase) <= MAX_TOKEN_CHARS:
            counts[phrase] += 1
            counts[" " + phrase] += 2
    return counts


def build_vocab(text: str, vocab_size: int) -> list[str]:
    extra_count = max(0, vocab_size - BYTE_VOCAB)
    counts = candidate_pieces(text)
    pieces: list[str] = []
    seen: set[str] = set()
    for piece, _count in counts.most_common(extra_count * 4):
        if len(piece) > MAX_TOKEN_CHARS:
            continue
        if piece in seen:
            continue
        if len(piece) < 3:
            continue
        seen.add(piece)
        pieces.append(piece)
        if len(pieces) >= extra_count:
            break
    return pieces


def greedy_count(text: str, pieces: list[str]) -> int:
    by_first: dict[str, list[str]] = {}
    for piece in sorted(pieces, key=len, reverse=True):
        by_first.setdefault(piece[0], []).append(piece)

    count = 0
    idx = 0
    while idx < len(text):
        matched = ""
        for piece in by_first.get(text[idx], []):
            if text.startswith(piece, idx):
                matched = piece
                break
        if matched:
            idx += len(matched)
        else:
            idx += 1
        count += 1
    return count


def cfg_486sx(vocab_size: int) -> Config:
    return Config(
        profile=f"486sx-safe-subword-v{vocab_size}",
        vocab_size=vocab_size,
        n_positions=192,
        n_embd=48,
        n_head=4,
        n_layer=2,
        hidden_dim=192,
    )


def machine_tps(cfg: Config, prompt_tokens: int, generated_tokens: int, machine_key: str) -> float:
    counts = work_counts(cfg, prompt_tokens=prompt_tokens, generated_tokens=generated_tokens)
    machine = next(machine for machine in MACHINES if machine.key == machine_key)
    return generated_tokens / seconds_for(machine, counts["weighted_after"])


def probe(corpus: Path, vocab_sizes: list[int]) -> tuple[list[ProbeRow], dict[int, list[str]]]:
    corpus_text = clean_ascii(corpus.read_text(encoding="ascii", errors="ignore")) if corpus.exists() else ""
    sample_text = "\n".join(clean_ascii(text) for text in SAMPLE_TEXTS)
    train_text = corpus_text + "\n" + sample_text
    byte_count = sum(len(text) for text in SAMPLE_TEXTS)

    rows: list[ProbeRow] = []
    vocabs: dict[int, list[str]] = {}
    for vocab_size in vocab_sizes:
        pieces = build_vocab(train_text, vocab_size)
        vocabs[vocab_size] = pieces
        token_count = sum(greedy_count(text, pieces) for text in SAMPLE_TEXTS)
        compression = token_count / max(1, byte_count)

        cfg = cfg_486sx(vocab_size)
        prompt_tokens = max(1, round(31 * compression))
        generated_tokens = max(1, round(70 * compression))
        tps_486sx = machine_tps(cfg, prompt_tokens, generated_tokens, "486sx-25")
        tps_486dx2 = machine_tps(cfg, prompt_tokens, generated_tokens, "486dx2-66")
        seconds_70_bytes = generated_tokens / tps_486dx2

        rows.append(
            ProbeRow(
                vocab_size=vocab_size,
                token_count=token_count,
                byte_count=byte_count,
                compression=compression,
                params_486sx=parameter_count(cfg),
                runtime_mb_486sx=estimate_runtime_memory(cfg) / (1024.0 * 1024.0),
                tps_486sx=tps_486sx,
                tps_486dx2=tps_486dx2,
                estimated_seconds_70_bytes_486dx2=seconds_70_bytes,
            )
        )
    return rows, vocabs


def markdown(rows: list[ProbeRow], vocabs: dict[int, list[str]], corpus: Path) -> str:
    lines = [
        "# GPT2-BASIC Subword Architecture Probe",
        "",
        "This probe estimates whether a compact domain subword vocabulary is worth implementing before more fine-tuning. It does not change the DOS runtime yet.",
        "",
        f"Corpus basis: `{corpus.relative_to(ROOT) if corpus.exists() else corpus}`",
        "",
        "## Result",
        "",
        "| Vocab | Sample tokens | Bytes | Token/byte | Params | Runtime MB | Est 486SX tok/s | Est 486DX2 tok/s | Est 70-byte seconds on 486DX2 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.vocab_size} | {row.token_count} | {row.byte_count} | {row.compression:.3f} | "
            f"{row.params_486sx} | {row.runtime_mb_486sx:.3f} | {row.tps_486sx:.2f} | "
            f"{row.tps_486dx2:.2f} | {row.estimated_seconds_70_bytes_486dx2:.1f} |"
        )

    best = min(rows, key=lambda row: row.estimated_seconds_70_bytes_486dx2)
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"The best planning point in this probe is vocab `{best.vocab_size}`. "
            f"It reduces the sample token count to {best.compression:.1%} of byte-level while keeping runtime memory at {best.runtime_mb_486sx:.3f} MB.",
            "",
            "The quality reason is stronger than the speed reason: a domain subword vocabulary lets the model emit whole pieces such as ` fixed-point`, ` runtime`, ` tokens`, and ` memory` instead of relearning spelling byte by byte.",
            "",
            "## Example Pieces",
            "",
        ]
    )
    for size in sorted(vocabs):
        sample = ", ".join(f"`{piece}`" for piece in vocabs[size][:30])
        lines.append(f"- `{size}`: {sample}")

    lines.extend(
        [
            "",
            "## Implementation Consequence",
            "",
            "The production tokenizer contract is now wired through host training/export, vector generation, quality evaluation, DOS tokenizer loading, sampler masking, and QEMU staging of `VOCAB.BIN`. This probe remains useful for sizing decisions, but quality promotion still requires a real BPE training sweep with DOS held-out quality, runtime-regression quality, vector parity, and QEMU `--perf` evidence.",
        ]
    )
    return "\n".join(lines) + "\n"


def self_test() -> None:
    rows, vocabs = probe(DEFAULT_CORPUS, [258, 384])
    assert len(rows) == 2
    assert rows[0].vocab_size == 258
    assert rows[0].compression >= rows[1].compression
    assert len(vocabs[384]) > 0
    print("PROBE_OK subword_probe rows=2")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--vocab-size", action="append", type=int, default=None)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    vocab_sizes = args.vocab_size or [258, 384, 512, 768, 1024]
    rows, vocabs = probe(args.corpus, vocab_sizes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown(rows, vocabs, args.corpus), encoding="ascii")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
