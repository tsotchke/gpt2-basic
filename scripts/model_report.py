#!/usr/bin/env python3
"""Validate and summarize a GPT2-BASIC DOS checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from gpt2_basic_tokenizer import BYTE_VOCAB_SIZE, GPT2BasicTokenizer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "assets" / "gpt2_basic" / "MODEL"
EXP_TABLE_SIZE = 513
BYTES_PER_VALUE = 4


def resolve_file(model_dir: Path, primary_name: str, legacy_name: str) -> Path:
    primary_path = model_dir / primary_name
    if primary_path.exists():
        return primary_path
    return model_dir / legacy_name


def parse_config(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="ascii", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip().lower()] = value.strip()
    return values


def optional_artifact(model_dir: Path, name: str) -> Path:
    return model_dir / name


def require_int(values: dict[str, str], key: str) -> int:
    if key not in values:
        raise ValueError(f"missing config key {key}")
    try:
        parsed = int(values[key])
    except ValueError as exc:
        raise ValueError(f"invalid integer for {key}: {values[key]!r}") from exc
    if parsed < 1:
        raise ValueError(f"{key} must be positive")
    return parsed


def expected_parameter_count(cfg: dict[str, str]) -> int:
    vocab_size = require_int(cfg, "vocab_size")
    n_positions = require_int(cfg, "n_positions")
    n_embd = require_int(cfg, "n_embd")
    n_head = require_int(cfg, "n_head")
    n_layer = require_int(cfg, "n_layer")
    hidden_dim = require_int(cfg, "hidden_dim")

    if n_embd % n_head != 0:
        raise ValueError("n_embd must be divisible by n_head")

    layer_e = n_layer * n_embd
    layer_ee = n_layer * n_embd * n_embd
    layer_eh = n_layer * n_embd * hidden_dim
    layer_he = n_layer * hidden_dim * n_embd

    total = vocab_size * n_embd
    total += n_positions * n_embd
    total += layer_e * 8
    total += layer_ee * 4
    total += layer_eh + layer_he
    total += n_layer * hidden_dim
    total += layer_e
    total += n_embd * 2
    total += n_embd * vocab_size
    total += vocab_size
    return total


def validate_file_size(path: Path, expected_bytes: int) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(f"{path} is {actual_bytes} bytes, expected {expected_bytes}")


def validate_profile(path: Path, cfg: dict[str, str]) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(path)

    profile = parse_config(path)
    required = {
        "profile",
        "vocab_size",
        "n_positions",
        "n_embd",
        "n_head",
        "n_layer",
        "hidden_dim",
        "weights",
        "fixed_weights",
        "exp_table",
    }
    missing = sorted(required - set(profile))
    if missing:
        raise ValueError(f"{path} missing profile keys: {', '.join(missing)}")

    for key in ["profile", "vocab_size", "n_positions", "n_embd", "n_head", "n_layer", "hidden_dim"]:
        if profile[key] != cfg[key]:
            raise ValueError(f"{path} {key}={profile[key]!r}, expected {cfg[key]!r}")

    expected_files = {
        "weights": "GPT2WT.BIN",
        "fixed_weights": "GPT2FX.BIN",
        "exp_table": "GPT2EXP.BIN",
    }
    for key, expected in expected_files.items():
        if profile[key].upper() != expected:
            raise ValueError(f"{path} {key}={profile[key]!r}, expected {expected!r}")

    return profile


def validate_optional_alias(model_dir: Path, primary_name: str, legacy_name: str, expected_bytes: int | None = None) -> str:
    primary = model_dir / primary_name
    legacy = model_dir / legacy_name

    if not legacy.exists():
        return "absent optional legacy alias"
    if not primary.exists():
        if expected_bytes is not None:
            validate_file_size(legacy, expected_bytes)
        return f"OK legacy source ({legacy.stat().st_size} bytes)"
    if expected_bytes is not None:
        validate_file_size(legacy, expected_bytes)
    if primary.read_bytes() != legacy.read_bytes():
        raise ValueError(f"{legacy_name} exists but does not match {primary_name}")
    return f"OK alias of {primary_name} ({legacy.stat().st_size} bytes)"


def validate_vector_file(path: Path, cfg: dict[str, str]) -> tuple[int, int, int, int]:
    if not path.exists():
        return 0, 0, 0, 0

    vocab_size = require_int(cfg, "vocab_size")
    n_positions = require_int(cfg, "n_positions")
    n_embd = require_int(cfg, "n_embd")
    hidden_dim = require_int(cfg, "hidden_dim")
    vector_count = 0
    expected_count = 0
    phase_count = 0
    phase_expected_count = 0
    phase_dims = {
        "embedding": n_embd,
        "ln1": n_embd,
        "q": n_embd,
        "k": n_embd,
        "v": n_embd,
        "attn": n_embd,
        "proj": n_embd,
        "ln2": n_embd,
        "ff1": hidden_dim,
        "ff2": n_embd,
        "hidden": n_embd,
        "final_ln": n_embd,
        "logits": vocab_size,
    }

    for line_no, raw_line in enumerate(path.read_text(encoding="ascii", errors="ignore").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split("|")
        if len(fields) != 6 or fields[0] not in {"V", "P"}:
            raise ValueError(f"{path}:{line_no} invalid vector record")
        if fields[0] == "P":
            phase_name = fields[2]
            if phase_name not in phase_dims:
                raise ValueError(f"{path}:{line_no} unknown phase {phase_name}")
            declared_dim = int(fields[3])
            value_count = int(fields[4])
            expected = [item for item in fields[5].split(",") if item.strip()]
            if declared_dim != phase_dims[phase_name]:
                raise ValueError(f"{path}:{line_no} {phase_name} dim {declared_dim}, expected {phase_dims[phase_name]}")
            if value_count != len(expected):
                raise ValueError(f"{path}:{line_no} declared {value_count} phase values but has {len(expected)}")
            for pair in expected:
                idx_text, value_text = pair.split(":", 1)
                idx = int(idx_text)
                int(value_text)
                if idx < 0 or idx >= declared_dim:
                    raise ValueError(f"{path}:{line_no} phase index outside dimension")
            phase_count += 1
            phase_expected_count += len(expected)
            continue

        declared_len = int(fields[2])
        tokens = [int(item) for item in fields[3].split(",") if item.strip()]
        top_count = int(fields[4])
        expected = [item for item in fields[5].split(",") if item.strip()]

        if declared_len != len(tokens):
            raise ValueError(f"{path}:{line_no} declared {declared_len} tokens but has {len(tokens)}")
        if declared_len < 1 or declared_len > n_positions:
            raise ValueError(f"{path}:{line_no} context length {declared_len} outside 1..{n_positions}")
        if any(token < 0 or token >= vocab_size for token in tokens):
            raise ValueError(f"{path}:{line_no} token outside vocabulary")
        if top_count != len(expected):
            raise ValueError(f"{path}:{line_no} declared {top_count} expected logits but has {len(expected)}")
        for pair in expected:
            token_text, logit_text = pair.split(":", 1)
            token_id = int(token_text)
            int(logit_text)
            if token_id < 0 or token_id >= vocab_size:
                raise ValueError(f"{path}:{line_no} expected token outside vocabulary")
        vector_count += 1
        expected_count += len(expected)

    if vector_count == 0:
        raise ValueError(f"{path} contains no vector records")
    return vector_count, expected_count, phase_count, phase_expected_count


def validate_tokenizer_file(path: Path, cfg: dict[str, str]) -> str:
    vocab_size = require_int(cfg, "vocab_size")
    if not path.exists():
        if vocab_size == BYTE_VOCAB_SIZE:
            return "absent optional byte-level tokenizer"
        raise FileNotFoundError(f"{path} required for vocab_size={vocab_size}")
    tokenizer = GPT2BasicTokenizer.read_vocab_bin(path)
    tokenizer.validate_for_vocab_size(vocab_size)
    return (
        f"OK mode={tokenizer.mode} vocab={tokenizer.vocab_size} merges={len(tokenizer.merges)} "
        f"output_allowed={sum(tokenizer.output_allowed)} ({path.stat().st_size} bytes)"
    )


def artifact_line(name: str, status: str) -> str:
    return f"artifact {name}: {status}"


def named_artifact_status(model_dir: Path, name: str, fallback_name: str | None = None) -> str:
    path = model_dir / name
    if path.exists():
        return f"OK ({path.stat().st_size} bytes)"
    if fallback_name:
        fallback = model_dir / fallback_name
        if fallback.exists():
            return f"absent; using {fallback_name} legacy source"
    return "absent"


def self_test(model_dir: Path, strict: bool) -> None:
    cfg_path = resolve_file(model_dir, "GPT2CFG.TXT", "TINYCFG.TXT")
    fixed_path = resolve_file(model_dir, "GPT2FX.BIN", "TINYFX.BIN")
    exp_path = resolve_file(model_dir, "GPT2EXP.BIN", "TINYEXP.BIN")
    float_path = resolve_file(model_dir, "GPT2WT.BIN", "TINYWT.BIN")
    profile_path = optional_artifact(model_dir, "PROFILE.TXT")
    vector_path = optional_artifact(model_dir, "GPT2VEC.TXT")
    vocab_path = optional_artifact(model_dir, "VOCAB.BIN")

    cfg = parse_config(cfg_path)
    params = expected_parameter_count(cfg)
    expected_weight_bytes = params * BYTES_PER_VALUE
    expected_exp_bytes = EXP_TABLE_SIZE * BYTES_PER_VALUE
    validate_file_size(fixed_path, expected_weight_bytes)
    validate_file_size(exp_path, expected_exp_bytes)
    if strict:
        validate_file_size(float_path, expected_weight_bytes)
    profile = validate_profile(profile_path, cfg)
    vectors = validate_vector_file(vector_path, cfg)
    tokenizer_status = validate_tokenizer_file(vocab_path, cfg)
    alias = validate_optional_alias(model_dir, "GPT2FX.BIN", "TINYFX.BIN", expected_weight_bytes)
    status = named_artifact_status(model_dir, "GPT2CFG.TXT", "TINYCFG.TXT")
    line = artifact_line("GPT2CFG.TXT", status)

    print(f"PROBE_OK resolve_file cfg={cfg_path.name}")
    print("PROBE_OK require_int exercised_by=expected_parameter_count")
    print(f"PROBE_OK expected_parameter_count params={params}")
    print(f"PROBE_OK validate_file_size fixed_bytes={expected_weight_bytes}")
    print(f"PROBE_OK validate_profile profile={profile['profile']}")
    print("trace validate_tokenizer_file")
    print(f"PROBE_OK validate_tokenizer_file status={tokenizer_status}")
    print(f"PROBE_OK validate_vector_file vectors={vectors[0]} phases={vectors[2]}")
    print(f"PROBE_OK validate_optional_alias status={alias}")
    print(f"PROBE_OK named_artifact_status status={status}")
    print(f"PROBE_OK artifact_line line={line}")
    print("PROBE_OK self_test exercised=1")
    print("PROBE_OK main cli_entry=available")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--require-legacy-names", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test(args.model_dir, args.strict)
        return

    model_dir = args.model_dir
    cfg_path = resolve_file(model_dir, "GPT2CFG.TXT", "TINYCFG.TXT")
    fixed_path = resolve_file(model_dir, "GPT2FX.BIN", "TINYFX.BIN")
    exp_path = resolve_file(model_dir, "GPT2EXP.BIN", "TINYEXP.BIN")
    float_path = resolve_file(model_dir, "GPT2WT.BIN", "TINYWT.BIN")
    profile_path = optional_artifact(model_dir, "PROFILE.TXT")
    vocab_path = optional_artifact(model_dir, "VOCAB.BIN")

    cfg = parse_config(cfg_path)
    params = expected_parameter_count(cfg)
    expected_weight_bytes = params * BYTES_PER_VALUE
    expected_exp_bytes = EXP_TABLE_SIZE * BYTES_PER_VALUE

    validate_file_size(fixed_path, expected_weight_bytes)
    validate_file_size(exp_path, expected_exp_bytes)
    if args.strict:
        validate_file_size(float_path, expected_weight_bytes)
    profile_metadata = validate_profile(profile_path, cfg)
    tokenizer_status = validate_tokenizer_file(vocab_path, cfg)
    vector_path = optional_artifact(model_dir, "GPT2VEC.TXT")
    vector_count, vector_expected_count, phase_count, phase_expected_count = validate_vector_file(vector_path, cfg)

    legacy_status = {
        "TINYCFG.TXT": validate_optional_alias(model_dir, "GPT2CFG.TXT", "TINYCFG.TXT"),
        "TINYWT.BIN": validate_optional_alias(model_dir, "GPT2WT.BIN", "TINYWT.BIN", expected_weight_bytes),
        "TINYFX.BIN": validate_optional_alias(model_dir, "GPT2FX.BIN", "TINYFX.BIN", expected_weight_bytes),
        "TINYEXP.BIN": validate_optional_alias(model_dir, "GPT2EXP.BIN", "TINYEXP.BIN", expected_exp_bytes),
    }
    if args.require_legacy_names:
        missing_legacy = [name for name, status in legacy_status.items() if status.startswith("absent")]
        if missing_legacy:
            raise FileNotFoundError(", ".join(missing_legacy))

    profile = cfg.get("profile", "custom")
    print(f"model_dir: {model_dir}")
    print(f"profile: {profile}")
    print(
        "shape: "
        f"layers={cfg['n_layer']} emb={cfg['n_embd']} heads={cfg['n_head']} "
        f"ctx={cfg['n_positions']} hidden={cfg['hidden_dim']} vocab={cfg['vocab_size']}"
    )
    print(f"parameters: {params}")
    print(f"fixed_weight_bytes: {expected_weight_bytes}")
    print(f"config: {cfg_path.name}")
    print(f"fixed_weights: {fixed_path.name}")
    print(f"exp_table: {exp_path.name}")
    if float_path.exists():
        print(f"float_reference: {float_path.name}")
    print(f"profile_metadata: {profile_path.name}")
    print(artifact_line("GPT2CFG.TXT", named_artifact_status(model_dir, "GPT2CFG.TXT", "TINYCFG.TXT")))
    print(artifact_line("GPT2WT.BIN", named_artifact_status(model_dir, "GPT2WT.BIN", "TINYWT.BIN")))
    print(artifact_line("GPT2FX.BIN", named_artifact_status(model_dir, "GPT2FX.BIN", "TINYFX.BIN")))
    print(artifact_line("GPT2EXP.BIN", named_artifact_status(model_dir, "GPT2EXP.BIN", "TINYEXP.BIN")))
    print(artifact_line("PROFILE.TXT", f"OK profile={profile_metadata['profile']}"))
    print(artifact_line("VOCAB.BIN", tokenizer_status))
    if vector_path.exists():
        print(
            artifact_line(
                "GPT2VEC.TXT",
                f"OK vectors={vector_count} expected_logits={vector_expected_count} "
                f"phases={phase_count} expected_phase_values={phase_expected_count}",
            )
        )
    for name, status in legacy_status.items():
        print(artifact_line(name, status))
    print("status: OK")


if __name__ == "__main__":
    main()
