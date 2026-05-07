#!/usr/bin/env python3
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
OUT = ROOT / "qemu" / "staging" / "GPT2SRC"

NAME_MAP = {
    "main.bas": "MAIN.BAS",
    "data_structures.bas": "DATASTR.BAS",
    "matrix_ops.bas": "MATOPS.BAS",
    "simd_ops.bas": "SIMD.BAS",
    "block_sparse.bas": "SPARSE.BAS",
    "memory_manager.bas": "MEMMGR.BAS",
    "asm_optimizations.bas": "ASMOPT.BAS",
    "softmax_fixed.bas": "SOFTMAX.BAS",
    "tokenizer.bas": "TOKEN.BAS",
    "real_gpt.bas": "REALGPT.BAS",
    "quality_prior.bas": "QUALPR.BAS",
    "transformer_components.bas": "TRANS.BAS",
    "model.bas": "MODEL.BAS",
    "file_io.bas": "FILEIO.BAS",
    "benchmark.bas": "BENCH.BAS",
    "quantization.bas": "QUANT.BAS",
    "dos_gpt2_basic.bas": "GPT2DOS.BAS",
}

INCLUDE_RE = re.compile(r'(?im)^[ \t]*#include[ \t]+"([^"]+)"(?:[ \t]*\'.*)?$')


def convert_include(match: re.Match[str]) -> str:
    include_name = Path(match.group(1).replace("\\", "/")).name.lower()
    mapped = NAME_MAP.get(include_name)
    if not mapped:
        return match.group(0)
    return f'#INCLUDE "{mapped}"'


def guard_name(short_name: str) -> str:
    return "__GPT2BASIC_" + re.sub(r"[^A-Z0-9]", "_", short_name.upper()) + "__"


def compat_rewrites(long_name: str, text: str) -> str:
    # Keep this hook for target-specific text changes that are only about DOS
    # staging. Semantic compatibility changes belong in src/.
    return text


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    for long_name, short_name in NAME_MAP.items():
        source_path = SRC / long_name
        if not source_path.exists():
            continue

        text = source_path.read_text(encoding="utf-8")
        text = INCLUDE_RE.sub(convert_include, text)
        text = compat_rewrites(long_name, text)

        if long_name != "main.bas":
            guard = guard_name(short_name)
            text = f"#IFNDEF {guard}\n#DEFINE {guard}\n{text}\n#ENDIF\n"

        (OUT / short_name).write_text(text, encoding="ascii", errors="ignore")


if __name__ == "__main__":
    main()
