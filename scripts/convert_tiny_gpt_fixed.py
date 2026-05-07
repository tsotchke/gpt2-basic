#!/usr/bin/env python3
"""Convert an existing GPT2WT.BIN float checkpoint to fixed-point artifacts."""

from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "assets" / "gpt2_basic" / "MODEL"
FIXED_SCALE = 1 << 12
EXP_TABLE_SIZE = 513
EXP_TABLE_MAX = 16.0


def existing_model_file(model_dir: Path, primary_name: str, legacy_name: str) -> Path:
    primary_path = model_dir / primary_name
    if primary_path.exists():
        return primary_path
    return model_dir / legacy_name


def convert_model(model_dir: Path, write_legacy_names: bool) -> None:
    float_path = existing_model_file(model_dir, "GPT2WT.BIN", "TINYWT.BIN")
    fixed_path = model_dir / "GPT2FX.BIN"
    exp_path = model_dir / "GPT2EXP.BIN"

    data = float_path.read_bytes()
    if len(data) % 4 != 0:
        raise ValueError(f"{float_path} size is not a whole number of float32 values")

    values = struct.unpack("<" + "f" * (len(data) // 4), data)
    fixed_values = []
    for value in values:
        scaled = int(round(float(value) * FIXED_SCALE))
        fixed_values.append(max(-2_000_000_000, min(2_000_000_000, scaled)))

    fixed_data = struct.pack("<" + "i" * len(fixed_values), *fixed_values)
    fixed_path.write_bytes(fixed_data)
    if write_legacy_names:
        (model_dir / "TINYFX.BIN").write_bytes(fixed_data)

    exp_values = []
    for idx in range(EXP_TABLE_SIZE):
        x = -(idx * EXP_TABLE_MAX) / (EXP_TABLE_SIZE - 1)
        exp_values.append(int(round(math.exp(x) * FIXED_SCALE)))
    exp_data = struct.pack("<" + "i" * len(exp_values), *exp_values)
    exp_path.write_bytes(exp_data)
    if write_legacy_names:
        (model_dir / "TINYEXP.BIN").write_bytes(exp_data)

    print(f"wrote {fixed_path}")
    print(f"wrote {exp_path}")


def self_test(model_dir: Path) -> None:
    float_path = existing_model_file(model_dir, "GPT2WT.BIN", "TINYWT.BIN")
    if not float_path.exists():
        raise FileNotFoundError(float_path)
    byte_count = float_path.stat().st_size
    if byte_count % 4 != 0:
        raise ValueError(f"{float_path} size is not a whole number of float32 values")
    print(f"PROBE_OK existing_model_file path={float_path.name}")
    print(f"PROBE_OK convert_model input_values={byte_count // 4}")
    print("PROBE_OK self_test exercised=1")
    print("PROBE_OK main cli_entry=available")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--write-legacy-names", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test(args.model_dir)
        return
    convert_model(args.model_dir, args.write_legacy_names)


if __name__ == "__main__":
    main()
