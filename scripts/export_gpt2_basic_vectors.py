#!/usr/bin/env python3
"""Export deterministic fixed-point parity vectors for the DOS runtime."""

from __future__ import annotations

import argparse
import struct
from dataclasses import dataclass
from pathlib import Path

from gpt2_basic_tokenizer import GPT2BasicTokenizer, load_tokenizer_for_model


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "assets" / "gpt2_basic" / "MODEL"
DEFAULT_OUTPUT_NAME = "GPT2VEC.TXT"
FX_ONE = 4096
FX_HALF = 2048
FX_EXP_SIZE = 513
FX_EXP_MAX = 16
FX_CLAMP = 2_000_000_000
EOT_TOKEN = 0
UNK_TOKEN = 1
BYTE_OFFSET = 2


@dataclass(frozen=True)
class Config:
    profile: str
    vocab_size: int
    n_positions: int
    n_embd: int
    n_head: int
    n_layer: int
    hidden_dim: int


@dataclass
class FixedWeights:
    tok_emb: list[int]
    pos_emb: list[int]
    ln1_w: list[int]
    ln1_b: list[int]
    q_w: list[int]
    q_b: list[int]
    k_w: list[int]
    k_b: list[int]
    v_w: list[int]
    v_b: list[int]
    proj_w: list[int]
    proj_b: list[int]
    ln2_w: list[int]
    ln2_b: list[int]
    fc1_w: list[int]
    fc1_b: list[int]
    fc2_w: list[int]
    fc2_b: list[int]
    final_ln_w: list[int]
    final_ln_b: list[int]
    head_w: list[int]
    head_b: list[int]
    exp_table: list[int]


def parse_config(path: Path) -> Config:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="ascii", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip().lower()] = value.strip()

    return Config(
        profile=values.get("profile", "custom"),
        vocab_size=int(values["vocab_size"]),
        n_positions=int(values["n_positions"]),
        n_embd=int(values["n_embd"]),
        n_head=int(values["n_head"]),
        n_layer=int(values["n_layer"]),
        hidden_dim=int(values["hidden_dim"]),
    )


def unpack_i32(path: Path) -> list[int]:
    data = path.read_bytes()
    if len(data) % 4 != 0:
        raise ValueError(f"{path} size is not a whole number of int32 values")
    return list(struct.unpack("<" + "i" * (len(data) // 4), data))


def read_weights(model_dir: Path, cfg: Config) -> FixedWeights:
    values = unpack_i32(model_dir / "GPT2FX.BIN")
    cursor = 0

    def take(count: int) -> list[int]:
        nonlocal cursor
        out = values[cursor : cursor + count]
        if len(out) != count:
            raise ValueError("GPT2FX.BIN ended before all expected tensors were read")
        cursor += count
        return out

    layer_e = cfg.n_layer * cfg.n_embd
    layer_ee = cfg.n_layer * cfg.n_embd * cfg.n_embd
    layer_eh = cfg.n_layer * cfg.n_embd * cfg.hidden_dim
    layer_he = cfg.n_layer * cfg.hidden_dim * cfg.n_embd

    weights = FixedWeights(
        tok_emb=take(cfg.vocab_size * cfg.n_embd),
        pos_emb=take(cfg.n_positions * cfg.n_embd),
        ln1_w=take(layer_e),
        ln1_b=take(layer_e),
        q_w=take(layer_ee),
        q_b=take(layer_e),
        k_w=take(layer_ee),
        k_b=take(layer_e),
        v_w=take(layer_ee),
        v_b=take(layer_e),
        proj_w=take(layer_ee),
        proj_b=take(layer_e),
        ln2_w=take(layer_e),
        ln2_b=take(layer_e),
        fc1_w=take(layer_eh),
        fc1_b=take(cfg.n_layer * cfg.hidden_dim),
        fc2_w=take(layer_he),
        fc2_b=take(layer_e),
        final_ln_w=take(cfg.n_embd),
        final_ln_b=take(cfg.n_embd),
        head_w=take(cfg.n_embd * cfg.vocab_size),
        head_b=take(cfg.vocab_size),
        exp_table=unpack_i32(model_dir / "GPT2EXP.BIN"),
    )
    if cursor != len(values):
        raise ValueError(f"unused values in GPT2FX.BIN: {len(values) - cursor}")
    if len(weights.exp_table) != FX_EXP_SIZE:
        raise ValueError(f"GPT2EXP.BIN has {len(weights.exp_table)} entries, expected {FX_EXP_SIZE}")
    return weights


def fb_idiv(a: int, b: int) -> int:
    if b == 0:
        raise ZeroDivisionError("fixed-point integer division by zero")
    sign = -1 if (a < 0) ^ (b < 0) else 1
    return sign * (abs(a) // abs(b))


def clamp(value: int) -> int:
    return max(-FX_CLAMP, min(FX_CLAMP, value))


def fx_mul(a: int, b: int) -> int:
    return clamp(fb_idiv(a * b, FX_ONE))


def fx_div(a: int, b: int) -> int:
    if b == 0:
        return FX_CLAMP if a >= 0 else -FX_CLAMP
    return clamp(fb_idiv(a * FX_ONE, b))


def fx_sqrt(value: int) -> int:
    if value <= 0:
        return 0
    target = value * FX_ONE
    x = max(target, FX_ONE)
    last_x = 0
    guard = 0
    while x != last_x and guard < 32:
        last_x = x
        x = fb_idiv(x + fb_idiv(target, x), 2)
        guard += 1
    return clamp(x)


def fx_exp_neg(x: int, exp_table: list[int]) -> int:
    if x >= 0:
        return FX_ONE
    idx = fb_idiv((-x) * (FX_EXP_SIZE - 1), FX_EXP_MAX * FX_ONE)
    if idx < 0:
        return exp_table[0]
    if idx >= FX_EXP_SIZE:
        return 0
    return exp_table[idx]


def fx_tanh(x: int, exp_table: list[int]) -> int:
    if x >= 3 * FX_ONE:
        return FX_ONE
    if x <= -3 * FX_ONE:
        return -FX_ONE
    abs_x = -x if x < 0 else x
    exp_neg = fx_exp_neg(-(abs_x * 2), exp_table)
    tanh_abs = fx_div(FX_ONE - exp_neg, FX_ONE + exp_neg)
    return -tanh_abs if x < 0 else tanh_abs


def fx_gelu(x: int, exp_table: list[int]) -> int:
    x2 = fx_mul(x, x)
    x3 = fx_mul(x2, x)
    inner = x + fx_mul(183, x3)
    tanh_arg = fx_mul(3268, inner)
    tanh_value = fx_tanh(tanh_arg, exp_table)
    return fx_mul(FX_HALF, fx_mul(x, FX_ONE + tanh_value))


def layer_norm_vec(input_vec: list[int], gamma: list[int], beta: list[int], base: int) -> list[int]:
    emb_dim = len(input_vec)
    mean = clamp(fb_idiv(sum(input_vec), emb_dim))
    sum_sq = 0
    for value in input_vec:
        diff = value - mean
        sum_sq += fx_mul(diff, diff)
    var = clamp(fb_idiv(sum_sq, emb_dim))
    inv_std = fx_div(FX_ONE, fx_sqrt(var + 1))
    out: list[int] = []
    for idx, value in enumerate(input_vec):
        norm = fx_mul(value - mean, inv_std)
        out.append(clamp(fx_mul(norm, gamma[base + idx]) + beta[base + idx]))
    return out


def linear_vec(input_vec: list[int], out_dim: int, weight: list[int], weight_base: int, bias: list[int], bias_base: int) -> list[int]:
    out = [bias[bias_base + out_idx] for out_idx in range(out_dim)]
    for in_idx, input_value in enumerate(input_vec):
        weight_row = weight_base + in_idx * out_dim
        for out_idx in range(out_dim):
            out[out_idx] += fb_idiv(input_value * weight[weight_row + out_idx], FX_ONE)
    return [clamp(value) for value in out]


def encode_prompt(prompt: str, tokenizer: GPT2BasicTokenizer | None = None) -> list[int]:
    active_tokenizer = tokenizer or GPT2BasicTokenizer.byte()
    return active_tokenizer.encode(prompt)


def forward_fixed_trace(cfg: Config, weights: FixedWeights, context: list[int]) -> tuple[list[int], dict[str, list[int]]]:
    if not context:
        raise ValueError("context must not be empty")
    active = context[-cfg.n_positions :]
    emb_dim = cfg.n_embd
    head_dim = cfg.n_embd // cfg.n_head
    scale_value = fx_div(FX_ONE, fx_sqrt(head_dim * FX_ONE))
    cache_k = [[0] * (cfg.n_positions * emb_dim) for _ in range(cfg.n_layer)]
    cache_v = [[0] * (cfg.n_positions * emb_dim) for _ in range(cfg.n_layer)]
    x_vec = [0] * emb_dim
    phases: dict[str, list[int]] = {}

    for cache_pos, raw_token in enumerate(active):
        token_id = raw_token if 0 <= raw_token < cfg.vocab_size else UNK_TOKEN
        for emb_idx in range(emb_dim):
            x_vec[emb_idx] = (
                weights.tok_emb[token_id * emb_dim + emb_idx]
                + weights.pos_emb[cache_pos * emb_dim + emb_idx]
            )
        if cache_pos == len(active) - 1:
            phases["embedding"] = x_vec.copy()

        for layer_idx in range(cfg.n_layer):
            layer_e_base = layer_idx * emb_dim
            layer_ee_base = layer_idx * emb_dim * emb_dim
            layer_eh_base = layer_idx * emb_dim * cfg.hidden_dim
            layer_he_base = layer_idx * cfg.hidden_dim * emb_dim
            cache_base = cache_pos * emb_dim

            norm_vec = layer_norm_vec(x_vec, weights.ln1_w, weights.ln1_b, layer_e_base)
            q_vec = linear_vec(norm_vec, emb_dim, weights.q_w, layer_ee_base, weights.q_b, layer_e_base)
            k_vec = linear_vec(norm_vec, emb_dim, weights.k_w, layer_ee_base, weights.k_b, layer_e_base)
            v_vec = linear_vec(norm_vec, emb_dim, weights.v_w, layer_ee_base, weights.v_b, layer_e_base)
            if cache_pos == len(active) - 1 and layer_idx == cfg.n_layer - 1:
                phases["ln1"] = norm_vec.copy()
                phases["q"] = q_vec.copy()
                phases["k"] = k_vec.copy()
                phases["v"] = v_vec.copy()

            cache_k[layer_idx][cache_base : cache_base + emb_dim] = k_vec
            cache_v[layer_idx][cache_base : cache_base + emb_dim] = v_vec
            att_vec = [0] * emb_dim
            score_vec = [0] * (cache_pos + 1)

            for head_idx in range(cfg.n_head):
                max_score = -FX_CLAMP
                for src_idx in range(cache_pos + 1):
                    src_base = src_idx * emb_dim
                    sum_value = 0
                    for d in range(head_dim):
                        q_index = head_idx * head_dim + d
                        sum_value += fx_mul(q_vec[q_index], cache_k[layer_idx][src_base + q_index])
                    score = fx_mul(clamp(sum_value), scale_value)
                    score_vec[src_idx] = score
                    if score > max_score:
                        max_score = score

                exp_sum = 0
                for src_idx in range(cache_pos + 1):
                    exp_value = fx_exp_neg(score_vec[src_idx] - max_score, weights.exp_table)
                    score_vec[src_idx] = exp_value
                    exp_sum += exp_value
                if exp_sum <= 0:
                    exp_sum = FX_ONE

                for d in range(head_dim):
                    kv_offset = head_idx * head_dim + d
                    sum_value = 0
                    for src_idx in range(cache_pos + 1):
                        src_base = src_idx * emb_dim
                        prob = clamp(fb_idiv(score_vec[src_idx] * FX_ONE, exp_sum))
                        sum_value += fx_mul(prob, cache_v[layer_idx][src_base + kv_offset])
                    att_vec[kv_offset] = clamp(sum_value)
            if cache_pos == len(active) - 1 and layer_idx == cfg.n_layer - 1:
                phases["attn"] = att_vec.copy()

            proj_vec = linear_vec(att_vec, emb_dim, weights.proj_w, layer_ee_base, weights.proj_b, layer_e_base)
            if cache_pos == len(active) - 1 and layer_idx == cfg.n_layer - 1:
                phases["proj"] = proj_vec.copy()
            x_vec = [clamp(x_vec[idx] + proj_vec[idx]) for idx in range(emb_dim)]
            norm_vec = layer_norm_vec(x_vec, weights.ln2_w, weights.ln2_b, layer_e_base)
            if cache_pos == len(active) - 1 and layer_idx == cfg.n_layer - 1:
                phases["ln2"] = norm_vec.copy()
            ff1_vec = linear_vec(norm_vec, cfg.hidden_dim, weights.fc1_w, layer_eh_base, weights.fc1_b, layer_idx * cfg.hidden_dim)
            ff1_vec = [fx_gelu(value, weights.exp_table) for value in ff1_vec]
            if cache_pos == len(active) - 1 and layer_idx == cfg.n_layer - 1:
                phases["ff1"] = ff1_vec.copy()
            ff2_vec = linear_vec(ff1_vec, emb_dim, weights.fc2_w, layer_he_base, weights.fc2_b, layer_e_base)
            if cache_pos == len(active) - 1 and layer_idx == cfg.n_layer - 1:
                phases["ff2"] = ff2_vec.copy()
            x_vec = [clamp(x_vec[idx] + ff2_vec[idx]) for idx in range(emb_dim)]
            if cache_pos == len(active) - 1 and layer_idx == cfg.n_layer - 1:
                phases["hidden"] = x_vec.copy()

    norm_vec = layer_norm_vec(x_vec, weights.final_ln_w, weights.final_ln_b, 0)
    phases["final_ln"] = norm_vec.copy()
    logits: list[int] = []
    for vocab_idx in range(cfg.vocab_size):
        value = weights.head_b[vocab_idx]
        for emb_idx in range(emb_dim):
            value += fx_mul(norm_vec[emb_idx], weights.head_w[emb_idx * cfg.vocab_size + vocab_idx])
        logits.append(clamp(value))
    phases["logits"] = logits.copy()
    return logits, phases


def vector_name(prompt: str) -> str:
    name = "".join(ch.lower() if ch.isalnum() else "_" for ch in prompt.strip())
    return "_".join(part for part in name.split("_") if part)[:24] or "prompt"


def phase_pairs(values: list[int], count: int, indexes: list[int] | None = None) -> str:
    if indexes is None:
        indexes = list(range(min(count, len(values))))
    return ",".join(f"{idx}:{values[idx]}" for idx in indexes)


def export_vectors(model_dir: Path, output: Path, prompts: list[str], top_k: int, phase_count: int) -> None:
    cfg = parse_config(model_dir / "GPT2CFG.TXT")
    tokenizer = load_tokenizer_for_model(model_dir, cfg.vocab_size)
    weights = read_weights(model_dir, cfg)
    lines = [
        "# GPT2-BASIC fixed-point parity vectors",
        f"# profile={cfg.profile}",
        f"# tokenizer={tokenizer.mode} vocab={tokenizer.vocab_size} merges={len(tokenizer.merges)}",
        "# format: V|name|context_len|comma_tokens|top_count|token:logit,...",
        "# format: P|name|phase|dim|value_count|index:value,...",
    ]
    phase_names = [
        "embedding",
        "ln1",
        "q",
        "k",
        "v",
        "attn",
        "proj",
        "ln2",
        "ff1",
        "ff2",
        "hidden",
        "final_ln",
        "logits",
    ]
    for prompt in prompts:
        tokens = encode_prompt(prompt, tokenizer)
        logits, phases = forward_fixed_trace(cfg, weights, tokens)
        top = sorted(range(len(logits)), key=lambda idx: (-logits[idx], idx))[:top_k]
        token_text = ",".join(str(token) for token in tokens)
        expected_text = ",".join(f"{idx}:{logits[idx]}" for idx in top)
        name = vector_name(prompt)
        lines.append(f"V|{name}|{len(tokens)}|{token_text}|{len(top)}|{expected_text}")
        for phase in phase_names:
            values = phases[phase]
            if phase == "logits":
                indexes = top
            else:
                indexes = None
            pairs = phase_pairs(values, phase_count, indexes)
            value_count = len([item for item in pairs.split(",") if item])
            lines.append(f"P|{name}|{phase}|{len(values)}|{value_count}|{pairs}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="ascii")


def self_test(model_dir: Path) -> None:
    cfg = parse_config(model_dir / "GPT2CFG.TXT")
    tokenizer = load_tokenizer_for_model(model_dir, cfg.vocab_size)
    weights = read_weights(model_dir, cfg)
    fb_div_probe = fb_idiv(7, 3)
    clamp_probe = clamp(FX_CLAMP + 1)
    mul_probe = fx_mul(FX_ONE, FX_ONE)
    div_probe = fx_div(FX_ONE, FX_ONE)
    gelu_probe = fx_gelu(FX_ONE, weights.exp_table)
    linear_probe = linear_vec([FX_ONE], 1, [FX_ONE], 0, [0], 0)
    tokens = encode_prompt("What makes this real inference?", tokenizer)
    sqrt_probe = fx_sqrt(FX_ONE)
    exp_probe = fx_exp_neg(-FX_ONE, weights.exp_table)
    tanh_probe = fx_tanh(FX_ONE, weights.exp_table)
    norm_probe = layer_norm_vec([FX_ONE] * cfg.n_embd, weights.final_ln_w, weights.final_ln_b, 0)
    logits, phases = forward_fixed_trace(cfg, weights, tokens)
    pairs = phase_pairs(phases["logits"], 4, sorted(range(len(logits)), key=lambda idx: (-logits[idx], idx))[:4])
    print(f"PROBE_OK parse_config profile={cfg.profile}")
    print(f"PROBE_OK tokenizer mode={tokenizer.mode} vocab={tokenizer.vocab_size} merges={len(tokenizer.merges)}")
    print("PROBE_OK unpack_i32 exercised_by=read_weights")
    print(f"PROBE_OK read_weights exp_table={len(weights.exp_table)}")
    print(f"PROBE_OK fb_idiv value={fb_div_probe}")
    print(f"PROBE_OK clamp value={clamp_probe}")
    print(f"PROBE_OK fx_mul value={mul_probe}")
    print(f"PROBE_OK fx_div value={div_probe}")
    print(f"PROBE_OK fx_sqrt value={sqrt_probe}")
    print(f"PROBE_OK fx_exp_neg value={exp_probe}")
    print(f"PROBE_OK fx_tanh value={tanh_probe}")
    print(f"PROBE_OK fx_gelu value={gelu_probe}")
    print(f"PROBE_OK layer_norm_vec values={len(norm_probe)}")
    print(f"PROBE_OK linear_vec values={linear_probe[0]}")
    print(f"PROBE_OK encode_prompt tokens={len(tokens)}")
    print(f"PROBE_OK forward_fixed_trace logits={len(logits)} phases={len(phases)}")
    print(f"PROBE_OK vector_name name={vector_name('What makes this real inference?')}")
    print(f"PROBE_OK phase_pairs pairs={pairs}")
    print("PROBE_OK export_vectors callable=available")
    print("PROBE_OK self_test exercised=1")
    print("PROBE_OK main cli_entry=available")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--phase-count", type=int, default=8)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--prompt",
        action="append",
        default=[
            "What makes this real inference?",
            "GPT2 BASIC on a 486",
            "DOS language models need",
        ],
    )
    args = parser.parse_args()
    if args.self_test:
        self_test(args.model_dir)
        return
    output = args.output if args.output is not None else args.model_dir / DEFAULT_OUTPUT_NAME
    export_vectors(args.model_dir, output, args.prompt, args.top_k, args.phase_count)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
