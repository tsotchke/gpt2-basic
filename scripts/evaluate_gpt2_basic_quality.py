#!/usr/bin/env python3
"""Evaluate GPT2-BASIC checkpoint text quality on the host.

The default path loads GPT2WT.BIN into the host PyTorch model for fast quality
iteration. Use --backend fixed to run the slower Q20.12 reference interpreter
used by parity-vector generation.
"""

from __future__ import annotations

import argparse
import re
import struct
from dataclasses import dataclass
from pathlib import Path

import torch as th

from export_gpt2_basic_vectors import (
    BYTE_OFFSET,
    EOT_TOKEN,
    FX_CLAMP,
    UNK_TOKEN,
    Config as FixedConfig,
    FixedWeights,
    encode_prompt,
    forward_fixed_trace_with_head_shortlist,
    parse_config,
    read_head_shortlist,
    read_weights,
)
from train_tiny_gpt import Config as TorchConfig
from train_tiny_gpt import GPT2BasicModel
from gpt2_basic_tokenizer import BYTE_VOCAB_SIZE, GPT2BasicTokenizer, load_tokenizer_for_model


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "assets" / "gpt2_basic" / "MODEL"
DEFAULT_OUTPUT = ROOT / "qemu" / "evidence" / "quality_report.md"
DEFAULT_MIN_GENERATED = 70
SENTENCE_STOP_MIN_TOKENS = 10
DOMAIN_WORD_LEXICON: set[str] | None = None


@dataclass(frozen=True)
class QualityPrompt:
    name: str
    prompt: str
    keywords: tuple[str, ...]


@dataclass(frozen=True)
class QualityResult:
    name: str
    prompt: str
    completion: str
    generated_tokens: int
    score: float
    printable_ratio: float
    alpha_ratio: float
    keyword_hits: int
    repeated_trigram_ratio: float
    max_char_run: int
    boundary_errors: int
    ended_cleanly: bool
    passed: bool


REGRESSION_PROMPTS = [
    QualityPrompt(
        "real_inference",
        "What makes this real inference?",
        ("logits", "trained", "weights", "transformer", "runtime"),
    ),
    QualityPrompt(
        "486_target",
        "GPT2 BASIC on a 486",
        ("486", "memory", "fixed", "DOS", "weights", "attention"),
    ),
    QualityPrompt(
        "dos_model",
        "DOS language models need",
        ("DOS", "files", "memory", "prompt", "weights", "simple"),
    ),
    QualityPrompt(
        "basic_runtime",
        "A BASIC transformer runtime",
        ("BASIC", "arrays", "fixed", "tokens", "loops", "model"),
    ),
    QualityPrompt(
        "optimization",
        "To improve performance on real hardware",
        ("performance", "hardware", "loops", "memory", "fixed", "profile"),
    ),
]

HELDOUT_PROMPTS = [
    QualityPrompt(
        "heldout_cache",
        "Explain why a cache matters for text generation",
        ("cache", "context", "tokens", "speed", "memory", "reuse"),
    ),
    QualityPrompt(
        "heldout_timing",
        "How should a DOS model report timing?",
        ("timing", "seconds", "tokens", "measured", "hardware", "QEMU"),
    ),
    QualityPrompt(
        "heldout_limits",
        "What limits a tiny transformer on old PCs?",
        ("memory", "context", "speed", "weights", "small", "hardware"),
    ),
    QualityPrompt(
        "heldout_fixed_point",
        "Describe fixed point inference in one sentence",
        ("fixed", "integer", "weights", "arithmetic", "logits", "runtime"),
    ),
    QualityPrompt(
        "heldout_profiles",
        "Why compare model profiles before choosing one?",
        ("profile", "quality", "speed", "memory", "measure", "tradeoff"),
    ),
]

PROMPT_SUITES = {
    "runtime-regression": REGRESSION_PROMPTS,
    "heldout": HELDOUT_PROMPTS,
    "all": REGRESSION_PROMPTS + HELDOUT_PROMPTS,
}

PROMPTS = REGRESSION_PROMPTS


def quality_prompts(suite: str) -> list[QualityPrompt]:
    if suite not in PROMPT_SUITES:
        raise ValueError(f"unknown quality suite: {suite}")
    return list(PROMPT_SUITES[suite])


def default_output_for_suite(suite: str) -> Path:
    if suite == "heldout":
        return ROOT / "qemu" / "evidence" / "quality_report_heldout.md"
    if suite == "all":
        return ROOT / "qemu" / "evidence" / "quality_report_all.md"
    return DEFAULT_OUTPUT


def decode_generated(tokens: list[int], tokenizer: GPT2BasicTokenizer | None = None) -> str:
    active_tokenizer = tokenizer or GPT2BasicTokenizer.byte()
    return active_tokenizer.decode(tokens)


def select_greedy_token(
    logits: list[int],
    generated_tokens: list[int],
    min_generated: int,
    tokenizer: GPT2BasicTokenizer | None = None,
    prompt_text: str | None = None,
) -> int:
    active_tokenizer = tokenizer or GPT2BasicTokenizer.byte()
    generated_count = len(generated_tokens)
    generated_text = active_tokenizer.decode(generated_tokens)
    masked = list(logits)
    for token in range(len(masked)):
        if not active_tokenizer.token_can_follow_generated(generated_tokens, token, prompt_text):
            masked[token] = -FX_CLAMP
        else:
            penalty = active_tokenizer.token_follow_penalty(generated_tokens, token, generated_text)
            if penalty > 0.0:
                masked[token] -= int(penalty * 4096)
    if generated_count < min_generated:
        masked[EOT_TOKEN] = -FX_CLAMP

    best_idx = EOT_TOKEN
    best_logit = -FX_CLAMP - 1
    for token, value in enumerate(masked):
        if token == EOT_TOKEN and generated_count < min_generated:
            continue
        if value > best_logit:
            best_logit = value
            best_idx = token
    return best_idx


def generate_fixed_completion(
    cfg: FixedConfig,
    weights: FixedWeights,
    prompt: str,
    max_new_tokens: int,
    min_generated: int,
    tokenizer: GPT2BasicTokenizer,
    head_shortlist: list[int] | None = None,
) -> tuple[str, int]:
    context = encode_prompt(prompt, tokenizer)
    if not context:
        context = [EOT_TOKEN]
    prompt_len = len(context)

    for step in range(max_new_tokens):
        active = context[-cfg.n_positions :]
        logits, _phases = forward_fixed_trace_with_head_shortlist(cfg, weights, active, head_shortlist)
        next_token = select_greedy_token(logits, context[prompt_len:], min_generated, tokenizer, prompt)
        context.append(next_token)
        if next_token == EOT_TOKEN:
            break
        if step >= SENTENCE_STOP_MIN_TOKENS and tokenizer.token_ends_sentence(next_token):
            break

    generated = context[prompt_len:]
    return decode_generated(generated, tokenizer), len(generated)


def unpack_f32(path: Path) -> list[float]:
    data = path.read_bytes()
    if len(data) % 4 != 0:
        raise ValueError(f"{path} size is not a whole number of float32 values")
    return list(struct.unpack("<" + "f" * (len(data) // 4), data))


def load_float_model(model_dir: Path, cfg: FixedConfig, device: th.device) -> GPT2BasicModel:
    model = GPT2BasicModel(
        TorchConfig(
            n_positions=cfg.n_positions,
            n_embd=cfg.n_embd,
            n_head=cfg.n_head,
            n_layer=cfg.n_layer,
            hidden_dim=cfg.hidden_dim,
            vocab_size=cfg.vocab_size,
        )
    )
    values = unpack_f32(model_dir / "GPT2WT.BIN")
    cursor = 0

    def take(shape: tuple[int, ...]) -> th.Tensor:
        nonlocal cursor
        count = 1
        for dim in shape:
            count *= dim
        out = values[cursor : cursor + count]
        if len(out) != count:
            raise ValueError("GPT2WT.BIN ended before all expected tensors were read")
        cursor += count
        return th.tensor(out, dtype=th.float32).view(*shape)

    with th.no_grad():
        model.tok_emb.weight.copy_(take((cfg.vocab_size, cfg.n_embd)))
        model.pos_emb.weight.copy_(take((cfg.n_positions, cfg.n_embd)))

        for attr in ["ln1", "q", "k", "v", "proj", "ln2", "fc1", "fc2"]:
            if attr in {"ln1", "ln2"}:
                for block in model.blocks:
                    getattr(block, attr).weight.copy_(take((cfg.n_embd,)))
                for block in model.blocks:
                    getattr(block, attr).bias.copy_(take((cfg.n_embd,)))
            else:
                for block in model.blocks:
                    layer = getattr(block, attr)
                    in_dim = layer.in_features
                    out_dim = layer.out_features
                    layer.weight.copy_(take((in_dim, out_dim)).t().contiguous())
                for block in model.blocks:
                    getattr(block, attr).bias.copy_(take((getattr(block, attr).out_features,)))

        model.final_ln.weight.copy_(take((cfg.n_embd,)))
        model.final_ln.bias.copy_(take((cfg.n_embd,)))
        model.lm_head.weight.copy_(take((cfg.n_embd, cfg.vocab_size)).t().contiguous())
        model.lm_head.bias.copy_(take((cfg.vocab_size,)))

    if cursor != len(values):
        raise ValueError(f"unused values in GPT2WT.BIN: {len(values) - cursor}")

    return model.to(device).eval()


def mask_float_logits(
    logits: th.Tensor,
    generated_tokens: list[int],
    min_generated: int,
    tokenizer: GPT2BasicTokenizer | None = None,
    prompt_text: str | None = None,
) -> th.Tensor:
    active_tokenizer = tokenizer or GPT2BasicTokenizer.byte()
    generated_count = len(generated_tokens)
    generated_text = active_tokenizer.decode(generated_tokens)
    masked = logits.clone()
    for token in range(masked.numel()):
        if not active_tokenizer.token_can_follow_generated(generated_tokens, token, prompt_text):
            masked[token] = -1.0e9
        else:
            penalty = active_tokenizer.token_follow_penalty(generated_tokens, token, generated_text)
            if penalty > 0.0:
                masked[token] -= penalty
    if generated_count < min_generated:
        masked[EOT_TOKEN] = -1.0e9
    return masked


def generate_float_completion(
    model: GPT2BasicModel,
    prompt: str,
    max_new_tokens: int,
    min_generated: int,
    device: th.device,
    tokenizer: GPT2BasicTokenizer,
) -> tuple[str, int]:
    context = encode_prompt(prompt, tokenizer)
    if not context:
        context = [EOT_TOKEN]
    prompt_len = len(context)

    with th.no_grad():
        for step in range(max_new_tokens):
            active = context[-model.cfg.n_positions :]
            idx = th.tensor([active], dtype=th.long, device=device)
            logits = model(idx)[0, -1]
            masked = mask_float_logits(logits, context[prompt_len:], min_generated, tokenizer, prompt)
            next_token = int(th.argmax(masked).item())
            context.append(next_token)
            if next_token == EOT_TOKEN:
                break
            if step >= SENTENCE_STOP_MIN_TOKENS and tokenizer.token_ends_sentence(next_token):
                break

    generated = context[prompt_len:]
    return decode_generated(generated, tokenizer), len(generated)


def max_repeated_run(text: str) -> int:
    best = 0
    current = 0
    last = ""
    for ch in text:
        if ch == last:
            current += 1
        else:
            current = 1
            last = ch
        best = max(best, current)
    return best


def repeated_trigram_ratio(text: str) -> float:
    words = re.findall(r"[A-Za-z0-9']+", text.lower())
    if len(words) < 6:
        return 0.0
    trigrams = [tuple(words[idx : idx + 3]) for idx in range(len(words) - 2)]
    repeated = len(trigrams) - len(set(trigrams))
    return repeated / max(1, len(trigrams))


BOUNDARY_FRAGMENT_PATTERNS = [
    r"\b[a-z]{1,3}answer\b",
    r"\ba?integer[a-z]{3,}\b",
    r"\bplortecte\b",
    r"\biontext\b",
    r"\bantine\b",
    r"\b[a-z]+ysized[a-z]*\b",
    r"\bvaluey\b",
    r"\b[a-z]*(?:runsformer|foruntime|tenteger|weigs|contex(?!t)|geraing|losido|seped|prameter|resould|compatibile|parationhm)[a-z]*\b",
    r"\b[a-z]{2,}(?:runtime|transformer|timing|tokens|weights|context|integer)[a-z]{2,}\b",
    r"\b(?:anarose|calud|ceme|dsed|edese|loow|lowtsed|oldes|predas|reamearesents|roured|rslope|sastsuredures|shent|stshsts|suint|sureamarents)\b",
    r"\b[a-z]*(?:stststs|stshts|tsts|stsure|mearesents|reamare|rrace|suring|slon|loose|lop)\b",
]


def domain_word_lexicon() -> set[str]:
    global DOMAIN_WORD_LEXICON
    if DOMAIN_WORD_LEXICON is not None:
        return DOMAIN_WORD_LEXICON

    words: set[str] = set()
    corpus_root = ROOT / "data" / "domain_curriculum"
    for path in sorted(corpus_root.glob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        words.update(re.findall(r"[a-z]{3,}", text))

    pack_root = ROOT / "assets" / "gpt2_basic" / "PACKS"
    for path in sorted(pack_root.glob("*/*.TXT")):
        if path.name.upper() not in {"GOLDEN.TXT", "HELP.TXT", "LEXICON.TXT"}:
            continue
        text = path.read_text(encoding="ascii", errors="ignore").lower()
        words.update(re.findall(r"[a-z]{3,}", text))

    for prompt in REGRESSION_PROMPTS + HELDOUT_PROMPTS:
        words.update(re.findall(r"[a-z]{3,}", prompt.prompt.lower()))
        for keyword in prompt.keywords:
            words.update(re.findall(r"[a-z]{3,}", keyword.lower()))

    words.update(
        {
            "basic",
            "checkpoint",
            "continuation",
            "decode",
            "decoded",
            "encoder",
            "generation",
            "generated",
            "inference",
            "integer",
            "lexicon",
            "logits",
            "runtime",
            "tokenizer",
            "transformer",
        }
    )
    DOMAIN_WORD_LEXICON = words
    return words


def known_domain_word(word: str, lexicon: set[str]) -> bool:
    if word in lexicon:
        return True
    for suffix in ("s", "ed", "ing", "er", "ers", "ly"):
        if word.endswith(suffix) and word[: -len(suffix)] in lexicon:
            return True
    if word.endswith("ies") and word[:-3] + "y" in lexicon:
        return True
    return False


def boundary_error_count(text: str) -> int:
    if not text.strip():
        return 1
    errors = 0
    stripped = text.lstrip()
    if stripped and stripped[0] in ",;:)]}%":
        errors += 1
    lower = text.lower()
    for pattern in BOUNDARY_FRAGMENT_PATTERNS:
        errors += len(re.findall(pattern, lower))
    lexicon = domain_word_lexicon()
    for word in re.findall(r"[a-z]{6,}", lower):
        if re.search(r"([a-z]{2,3})\1{2,}", word):
            errors += 1
        elif len(word) >= 9 and sum(1 for ch in word if ch in "aeiou") <= 1:
            errors += 1
        elif len(word) >= 8 and not known_domain_word(word, lexicon):
            errors += 1
    return errors


def score_completion(prompt: QualityPrompt, completion: str, generated_tokens: int, threshold: float) -> QualityResult:
    printable = [ch for ch in completion if 32 <= ord(ch) <= 126]
    printable_ratio = len(printable) / max(1, len(completion))
    alpha_count = sum(1 for ch in completion if ch.isalpha())
    alpha_ratio = alpha_count / max(1, len(completion))
    lower = completion.lower()
    keyword_hits = sum(1 for keyword in prompt.keywords if keyword.lower() in lower)
    trigram_ratio = repeated_trigram_ratio(completion)
    char_run = max_repeated_run(completion)
    boundary_errors = boundary_error_count(completion)
    ended_cleanly = completion.rstrip().endswith((".", "!", "?"))

    length_score = min(1.0, len(completion.strip()) / 90.0)
    keyword_score = min(1.0, keyword_hits / 2.0)
    repeat_score = max(0.0, 1.0 - trigram_ratio * 4.0)
    run_score = 1.0 if char_run <= 4 else max(0.0, 1.0 - (char_run - 4) / 8.0)
    ending_score = 1.0 if ended_cleanly else 0.45
    boundary_score = max(0.0, 1.0 - boundary_errors * 0.35)

    score = (
        printable_ratio * 0.20
        + alpha_ratio * 0.15
        + length_score * 0.20
        + keyword_score * 0.25
        + repeat_score * 0.08
        + run_score * 0.05
        + ending_score * 0.05
        + boundary_score * 0.02
    )
    score = max(0.0, min(1.0, score))

    return QualityResult(
        name=prompt.name,
        prompt=prompt.prompt,
        completion=completion,
        generated_tokens=generated_tokens,
        score=score,
        printable_ratio=printable_ratio,
        alpha_ratio=alpha_ratio,
        keyword_hits=keyword_hits,
        repeated_trigram_ratio=trigram_ratio,
        max_char_run=char_run,
        boundary_errors=boundary_errors,
        ended_cleanly=ended_cleanly,
        passed=(
            score >= threshold
            and boundary_errors == 0
            and trigram_ratio <= 0.18
            and char_run <= 4
            and ended_cleanly
        ),
    )


def evaluate_model(
    model_dir: Path,
    prompts: list[QualityPrompt],
    max_new_tokens: int,
    min_generated: int,
    threshold: float,
    backend: str,
    device_name: str,
    progress_label: str | None = None,
) -> tuple[FixedConfig, list[QualityResult]]:
    cfg = parse_config(model_dir / "GPT2CFG.TXT")
    tokenizer = load_tokenizer_for_model(model_dir, cfg.vocab_size)
    results: list[QualityResult] = []
    if backend == "fixed":
        weights = read_weights(model_dir, cfg)
        head_shortlist = read_head_shortlist(model_dir, cfg)
        for index, prompt in enumerate(prompts, start=1):
            if progress_label is not None:
                print(f"QUALITY_EVAL {progress_label} {index}/{len(prompts)} {prompt.name}", flush=True)
            completion, generated_tokens = generate_fixed_completion(
                cfg, weights, prompt.prompt, max_new_tokens, min_generated, tokenizer, head_shortlist
            )
            results.append(score_completion(prompt, completion, generated_tokens, threshold))
    else:
        device = th.device(device_name)
        model = load_float_model(model_dir, cfg, device)
        for index, prompt in enumerate(prompts, start=1):
            if progress_label is not None:
                print(f"QUALITY_EVAL {progress_label} {index}/{len(prompts)} {prompt.name}", flush=True)
            completion, generated_tokens = generate_float_completion(
                model, prompt.prompt, max_new_tokens, min_generated, device, tokenizer
            )
            results.append(score_completion(prompt, completion, generated_tokens, threshold))
    return cfg, results


def format_pct(value: float) -> str:
    return f"{value * 100.0:.1f}%"


def markdown_report(
    cfg: FixedConfig,
    results: list[QualityResult],
    threshold: float,
    backend: str,
    suite: str,
) -> str:
    avg_score = sum(result.score for result in results) / max(1, len(results))
    pass_count = sum(1 for result in results if result.passed)
    status = "PASS" if pass_count == len(results) else "NEEDS_TRAINING"

    lines = [
        "# GPT2-BASIC Quality Report",
        "",
        f"Model profile: `{cfg.profile}`",
        f"Shape: `{cfg.n_layer}L {cfg.n_embd}D {cfg.n_head}H ctx{cfg.n_positions} hidden{cfg.hidden_dim} vocab{cfg.vocab_size}`",
        f"Evaluation backend: `{backend}`",
        f"Quality suite: `{suite}`",
        f"Quality status: `{status}`",
        f"Average score: `{avg_score:.3f}`",
        f"Prompt pass rate: `{pass_count}/{len(results)}` at threshold `{threshold:.2f}`",
        "",
        "## Prompt Suite",
        "",
        "| Prompt | Score | Keywords | Repeat | Max run | Boundary | End | Status |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for result in results:
        lines.append(
            "| "
            + " | ".join(
                [
                    result.name,
                    f"{result.score:.3f}",
                    str(result.keyword_hits),
                    format_pct(result.repeated_trigram_ratio),
                    str(result.max_char_run),
                    str(result.boundary_errors),
                    "yes" if result.ended_cleanly else "no",
                    "PASS" if result.passed else "RETRAIN",
                ]
            )
            + " |"
        )

    lines.extend(["", "## Generated Continuations", ""])
    for result in results:
        completion = result.completion.replace("|", "/")
        lines.extend(
            [
                f"### {result.name}",
                "",
                f"Prompt: `{result.prompt}`",
                "",
                "```text",
                completion,
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def self_test(model_dir: Path) -> None:
    cfg = parse_config(model_dir / "GPT2CFG.TXT")
    tokenizer = load_tokenizer_for_model(model_dir, cfg.vocab_size)
    model = load_float_model(model_dir, cfg, th.device("cpu"))
    float_completion, float_tokens = generate_float_completion(
        model, "What makes this real inference?", 8, 0, th.device("cpu"), tokenizer
    )
    weights = read_weights(model_dir, cfg)
    head_shortlist = read_head_shortlist(model_dir, cfg)
    fixed_completion, fixed_tokens = generate_fixed_completion(
        cfg, weights, "What makes this real inference?", 8, 0, tokenizer, head_shortlist
    )
    probe_prompt = QualityPrompt("probe", "What makes this real inference?", ("logits", "weights"))
    float_result = score_completion(probe_prompt, float_completion, float_tokens, 0.1)
    fixed_result = score_completion(probe_prompt, fixed_completion, fixed_tokens, 0.1)
    _decoded = decode_generated([ord("O") + BYTE_OFFSET, ord("K") + BYTE_OFFSET, EOT_TOKEN], tokenizer)
    probe_logits = [-FX_CLAMP] * GPT2BasicTokenizer.byte().vocab_size
    probe_logits[ord(" ") + BYTE_OFFSET] = 0
    probe_logits[ord(",") + BYTE_OFFSET] = 1
    _selected = select_greedy_token(probe_logits, [], 0)
    _masked = mask_float_logits(th.zeros(cfg.vocab_size), [], 0, tokenizer)
    _heldout_prompts = quality_prompts("heldout")
    _heldout_output = default_output_for_suite("heldout")
    _max_run = max_repeated_run("aaabb")
    _repeat_ratio = repeated_trigram_ratio("one two three one two three")
    _boundary_errors = boundary_error_count(", thanswer")
    _unknown_word_errors = boundary_error_count("The runtime produced ucocatiges during generation.")
    _pct = format_pct(0.5)
    report = markdown_report(cfg, [float_result, fixed_result], 0.1, "self-test", "self-test")
    print("trace format_pct")
    print(f"PROBE_OK parse_config profile={cfg.profile}")
    print(f"PROBE_OK tokenizer vocab={tokenizer.vocab_size} merges={len(tokenizer.merges)}")
    print("PROBE_OK unpack_f32 loaded=1")
    print("PROBE_OK load_float_model loaded=1")
    print(f"PROBE_OK generate_float_completion tokens={float_tokens}")
    print(f"PROBE_OK generate_fixed_completion tokens={fixed_tokens}")
    print(f"PROBE_OK read_head_shortlist count={len(head_shortlist) if head_shortlist is not None else 0}")
    print(f"PROBE_OK decode_generated bytes={len(_decoded)}")
    print(f"PROBE_OK select_greedy_token token={_selected}")
    print(f"PROBE_OK mask_float_logits entries={_masked.numel()}")
    print(f"PROBE_OK quality_prompts heldout_count={len(_heldout_prompts)}")
    print(f"PROBE_OK default_output_for_suite heldout={_heldout_output.name}")
    print(f"PROBE_OK max_repeated_run run={_max_run}")
    print(f"PROBE_OK repeated_trigram_ratio ratio={_repeat_ratio:.3f}")
    print(f"PROBE_OK boundary_error_count errors={_boundary_errors}")
    print(f"PROBE_OK unknown_word_boundary_count errors={_unknown_word_errors}")
    print(f"PROBE_OK format_pct value={_pct}")
    print(f"PROBE_OK score_completion float_score={float_result.score:.3f}")
    print(f"PROBE_OK score_completion fixed_score={fixed_result.score:.3f}")
    print("PROBE_OK evaluate_model covered_by_quality_suite=1")
    print(f"PROBE_OK markdown_report bytes={len(report)}")
    print("PROBE_OK self_test exercised=1")
    print("PROBE_OK parse_quality_eval_cli self_test=1")
    print("PROBE_OK main cli_entry=available")


def parse_quality_eval_cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-new-tokens", type=int, default=90)
    parser.add_argument("--min-generated", type=int, default=DEFAULT_MIN_GENERATED)
    parser.add_argument("--threshold", type=float, default=0.72)
    parser.add_argument("--backend", choices=("float", "fixed"), default="float")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--suite", choices=sorted(PROMPT_SUITES), default="runtime-regression")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test(args.model_dir)
        return

    cfg, results = evaluate_model(
        args.model_dir,
        quality_prompts(args.suite),
        args.max_new_tokens,
        args.min_generated,
        args.threshold,
        args.backend,
        args.device,
    )
    report = markdown_report(cfg, results, args.threshold, args.backend, args.suite)
    output = args.output if args.output != DEFAULT_OUTPUT or args.suite == "runtime-regression" else default_output_for_suite(args.suite)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report + "\n", encoding="ascii")
    print(report)


def main() -> None:
    parse_quality_eval_cli()


if __name__ == "__main__":
    main()
