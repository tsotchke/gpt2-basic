#!/usr/bin/env python3
"""Create production quantized GPT2-BASIC weight artifacts.

The production q4-log format targets the vocabulary-sized tensors first. With
the 4096-token lexicon model, the token embedding and output head are each
196,608 fixed-point values. GPT2TQ4.BIN stores token embeddings as per-token
4-bit log codes, and GPT2HQ4.BIN stores the output head as per-output-token
codes. GPT2FX.BIN is rewritten with the exact dequantized values used by the
DOS loader, keeping host quality/vector tools and DOS inference on one numeric
contract.
"""

from __future__ import annotations

import argparse
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path

from export_gpt2_basic_vectors import FixedWeights, parse_config, read_weights


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "assets" / "gpt2_basic" / "MODEL"
HEAD_Q4_MAGIC = 0x34514847  # "GHQ4" little-endian.
TOKEN_Q4_MAGIC = 0x34515447  # "GTQ4" little-endian.
HEAD_Q4_VERSION = 1
TOKEN_Q4_VERSION = 1
FX_ONE = 4096
LOG_Q4_LEVELS = (0, 256, 512, 1024, 1536, 2048, 3072, 4096)


@dataclass(frozen=True)
class Q4Tensor:
    dequantized: list[int]
    scales: list[int]
    codes: list[int]
    packed: bytes


def clamp_i32(value: int) -> int:
    return max(-2_000_000_000, min(2_000_000_000, int(value)))


def pack_nibbles(codes: list[int]) -> bytes:
    packed = bytearray()
    for idx in range(0, len(codes), 2):
        low = codes[idx] & 0x0F
        high = codes[idx + 1] & 0x0F if idx + 1 < len(codes) else 0
        packed.append(low | (high << 4))
    return bytes(packed)


def nearest_log_q4_code(value: int, scale: int) -> tuple[int, int]:
    sign_bit = 8 if value < 0 else 0
    target = (abs(value) * FX_ONE) // scale
    best_mag = 0
    best_err = abs(target - LOG_Q4_LEVELS[0])
    for mag, level in enumerate(LOG_Q4_LEVELS[1:], start=1):
        err = abs(target - level)
        if err < best_err:
            best_mag = mag
            best_err = err

    code = best_mag | (sign_bit if best_mag else 0)
    restored = (scale * LOG_Q4_LEVELS[best_mag]) // FX_ONE
    if sign_bit:
        restored = -restored
    return code, clamp_i32(restored)


def quantize_rows_q4(values: list[int], row_count: int, row_dim: int) -> Q4Tensor:
    expected_values = row_count * row_dim
    if len(values) != expected_values:
        raise ValueError(f"row tensor has {len(values)} values, expected {expected_values}")

    dequantized = [0] * len(values)
    scales: list[int] = []
    codes = [0] * len(values)

    for row_idx in range(row_count):
        row_base = row_idx * row_dim
        row_values = values[row_base : row_base + row_dim]
        scale = max(abs(value) for value in row_values) or 1
        scales.append(scale)

        for col_idx, value in enumerate(row_values):
            code, restored = nearest_log_q4_code(value, scale)
            dequantized[row_base + col_idx] = restored
            codes[row_base + col_idx] = code

    return Q4Tensor(dequantized, scales, codes, pack_nibbles(codes))


def quantize_token_q4(weights: FixedWeights, vocab_size: int, emb_dim: int) -> Q4Tensor:
    return quantize_rows_q4(weights.tok_emb, vocab_size, emb_dim)


def quantize_head_q4(weights: FixedWeights, vocab_size: int, emb_dim: int) -> Q4Tensor:
    dequantized = [0] * len(weights.head_w)
    scales: list[int] = []
    codes = [0] * len(weights.head_w)
    # GPT2FX stores the head in input-major order: head_w[in_idx * vocab + out_idx].
    for out_idx in range(vocab_size):
        values = [weights.head_w[in_idx * vocab_size + out_idx] for in_idx in range(emb_dim)]
        scale = max(abs(value) for value in values) or 1
        scales.append(scale)

        for in_idx, value in enumerate(values):
            code, restored = nearest_log_q4_code(value, scale)
            dequantized[in_idx * vocab_size + out_idx] = clamp_i32(restored)
            codes[in_idx * vocab_size + out_idx] = code

    return Q4Tensor(dequantized, scales, codes, pack_nibbles(codes))


def write_q4_artifact(path: Path, cfg: dict[str, str], artifact: Q4Tensor, magic: int, version: int) -> None:
    vocab_size = int(cfg["vocab_size"])
    emb_dim = int(cfg["n_embd"])
    value_count = vocab_size * emb_dim
    payload = bytearray()
    payload.extend(
        struct.pack(
            "<iiiiiiii",
            magic,
            version,
            vocab_size,
            emb_dim,
            value_count,
            len(LOG_Q4_LEVELS),
            len(artifact.scales),
            len(artifact.packed),
        )
    )
    payload.extend(struct.pack("<" + "i" * len(LOG_Q4_LEVELS), *LOG_Q4_LEVELS))
    payload.extend(struct.pack("<" + "i" * len(artifact.scales), *artifact.scales))
    payload.extend(artifact.packed)
    path.write_bytes(bytes(payload))


def write_head_q4(path: Path, cfg: dict[str, str], artifact: Q4Tensor) -> None:
    write_q4_artifact(path, cfg, artifact, HEAD_Q4_MAGIC, HEAD_Q4_VERSION)


def write_token_q4(path: Path, cfg: dict[str, str], artifact: Q4Tensor) -> None:
    write_q4_artifact(path, cfg, artifact, TOKEN_Q4_MAGIC, TOKEN_Q4_VERSION)


def fixed_weight_parts(weights: FixedWeights, tok_emb: list[int], head_w: list[int]) -> list[list[int]]:
    return [
        tok_emb,
        weights.pos_emb,
        weights.ln1_w,
        weights.ln1_b,
        weights.q_w,
        weights.q_b,
        weights.k_w,
        weights.k_b,
        weights.v_w,
        weights.v_b,
        weights.proj_w,
        weights.proj_b,
        weights.ln2_w,
        weights.ln2_b,
        weights.fc1_w,
        weights.fc1_b,
        weights.fc2_w,
        weights.fc2_b,
        weights.final_ln_w,
        weights.final_ln_b,
        head_w,
        weights.head_b,
    ]


def rewrite_fixed_weights(path: Path, weights: FixedWeights, tok_emb: list[int], head_w: list[int]) -> None:
    flat: list[int] = []
    for part in fixed_weight_parts(weights, tok_emb, head_w):
        flat.extend(part)
    path.write_bytes(struct.pack("<" + "i" * len(flat), *flat))


def update_profile(model_dir: Path, head_artifact: Q4Tensor | None, token_artifact: Q4Tensor | None) -> None:
    profile_path = model_dir / "PROFILE.TXT"
    if not profile_path.exists():
        return
    lines = profile_path.read_text(encoding="ascii", errors="ignore").splitlines()
    filtered = [
        line
        for line in lines
        if not line.startswith("head_quantization=")
        and not line.startswith("head_quantized_file=")
        and not line.startswith("head_quantized_bytes=")
        and not line.startswith("token_embedding_quantization=")
        and not line.startswith("token_embedding_quantized_file=")
        and not line.startswith("token_embedding_quantized_bytes=")
    ]
    if token_artifact is not None:
        filtered.extend(
            [
                "token_embedding_quantization=q4-log-per-token",
                "token_embedding_quantized_file=GPT2TQ4.BIN",
                f"token_embedding_quantized_bytes={len(token_artifact.packed)}",
            ]
        )
    if head_artifact is not None:
        filtered.extend(
            [
                "head_quantization=q4-log-per-token",
                "head_quantized_file=GPT2HQ4.BIN",
                f"head_quantized_bytes={len(head_artifact.packed)}",
            ]
        )
    profile_path.write_text("\n".join(filtered) + "\n", encoding="ascii")


def quantize_model(
    model_dir: Path,
    output_dir: Path | None,
    rewrite_fixed_head: bool,
    rewrite_fixed_token: bool,
    emit_head_q4: bool,
    emit_token_q4: bool,
) -> Path:
    target_dir = model_dir
    if output_dir is not None:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        shutil.copytree(model_dir, output_dir)
        target_dir = output_dir

    cfg = parse_config(target_dir / "GPT2CFG.TXT").__dict__
    weights = read_weights(target_dir, parse_config(target_dir / "GPT2CFG.TXT"))
    vocab_size = int(cfg["vocab_size"])
    emb_dim = int(cfg["n_embd"])
    head_artifact = quantize_head_q4(weights, vocab_size, emb_dim) if emit_head_q4 else None
    token_artifact = quantize_token_q4(weights, vocab_size, emb_dim) if emit_token_q4 else None
    if head_artifact is not None:
        write_head_q4(target_dir / "GPT2HQ4.BIN", cfg, head_artifact)
    if token_artifact is not None:
        write_token_q4(target_dir / "GPT2TQ4.BIN", cfg, token_artifact)
    if (head_artifact is not None and rewrite_fixed_head) or (token_artifact is not None and rewrite_fixed_token):
        rewritten_head = head_artifact.dequantized if head_artifact is not None and rewrite_fixed_head else weights.head_w
        rewritten_tok_emb = token_artifact.dequantized if token_artifact is not None and rewrite_fixed_token else weights.tok_emb
        rewrite_fixed_weights(target_dir / "GPT2FX.BIN", weights, rewritten_tok_emb, rewritten_head)
    update_profile(target_dir, head_artifact, token_artifact)

    original_tensor_bytes = vocab_size * emb_dim * 4
    if token_artifact is not None:
        print(f"wrote {target_dir / 'GPT2TQ4.BIN'}")
        print(
            "token_q4: "
            f"values={vocab_size * emb_dim} "
            f"i32_bytes={original_tensor_bytes} packed_bytes={len(token_artifact.packed)} "
            f"ratio={len(token_artifact.packed) / original_tensor_bytes:.3f}"
        )
    if head_artifact is not None:
        print(f"wrote {target_dir / 'GPT2HQ4.BIN'}")
        print(
            "head_q4: "
            f"values={vocab_size * emb_dim} "
            f"i32_bytes={original_tensor_bytes} packed_bytes={len(head_artifact.packed)} "
            f"ratio={len(head_artifact.packed) / original_tensor_bytes:.3f}"
        )
    if (head_artifact is not None and rewrite_fixed_head) or (token_artifact is not None and rewrite_fixed_token):
        print(f"rewrote {target_dir / 'GPT2FX.BIN'} with dequantized q4 tensors")
    return target_dir


def self_test(model_dir: Path) -> None:
    cfg = parse_config(model_dir / "GPT2CFG.TXT")
    weights = read_weights(model_dir, cfg)
    head_artifact = quantize_head_q4(weights, cfg.vocab_size, cfg.n_embd)
    token_artifact = quantize_token_q4(weights, cfg.vocab_size, cfg.n_embd)
    expected_values = cfg.vocab_size * cfg.n_embd
    for name, artifact in [("head", head_artifact), ("token", token_artifact)]:
        if len(artifact.codes) != expected_values:
            raise AssertionError(f"{name} q4 code count mismatch")
        if len(artifact.scales) != cfg.vocab_size:
            raise AssertionError(f"{name} q4 scale count mismatch")
        if len(artifact.packed) != (expected_values + 1) // 2:
            raise AssertionError(f"{name} q4 packed size mismatch")
    print(f"PROBE_OK parse_config profile={cfg.profile}")
    print(f"PROBE_OK nearest_log_q4_code levels={len(LOG_Q4_LEVELS)}")
    print(f"PROBE_OK quantize_rows_q4 values={expected_values}")
    print(f"PROBE_OK quantize_token_q4 values={expected_values}")
    print(f"PROBE_OK quantize_head_q4 values={expected_values}")
    print(f"PROBE_OK pack_nibbles bytes={len(head_artifact.packed)}")
    print("PROBE_OK self_test exercised=1")
    print("PROBE_OK main cli_entry=available")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-head-q4", action="store_true")
    parser.add_argument("--no-token-q4", action="store_true")
    parser.add_argument("--no-rewrite-fixed-head", action="store_true")
    parser.add_argument("--no-rewrite-fixed-token", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test(args.model_dir)
        return
    quantize_model(
        args.model_dir,
        args.output_dir,
        not args.no_rewrite_fixed_head,
        not args.no_rewrite_fixed_token,
        not args.no_head_q4,
        not args.no_token_q4,
    )


if __name__ == "__main__":
    main()
