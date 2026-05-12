#!/usr/bin/env python3
"""Verify the GPT2-BASIC assistant-pack surface and QEMU evidence."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACK_ROOT = ROOT / "assets" / "gpt2_basic" / "PACKS"
DEFAULT_ASSISTANT_LOG = ROOT / "qemu" / "evidence" / "assistant_486.log"
DEFAULT_COMPILE_LOG = ROOT / "qemu" / "evidence" / "assistant_compile_486.log"
DEFAULT_EVIDENCE_DIR = ROOT / "qemu" / "evidence"


@dataclass(frozen=True)
class PackInfo:
    pack_id: str
    model_path: str
    pack_dir: Path
    usage_path: Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"ASSISTANT_PROBE_FAILED {message}")


def read(path: Path) -> str:
    require(path.exists(), f"missing={path}")
    return path.read_text(encoding="ascii", errors="ignore")


def pack_ids(pack_root: Path) -> list[str]:
    list_path = pack_root / "PACKS.TXT"
    text = read(list_path)
    ids: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith(";"):
            continue
        ids.append(line.upper())
    return ids


def parse_ini(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith(";") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip().upper()] = value.strip()
    return values


def model_host_path(pack_root: Path, model_value: str) -> Path:
    value = model_value.replace("\\", "/")
    if value.upper().startswith("PACKS/"):
        return pack_root.parent / value
    return ROOT / "assets" / "gpt2_basic" / value


def verify_pack_model(pack_root: Path, pack_id: str, model_value: str) -> None:
    model_dir = model_host_path(pack_root, model_value)
    for name in ("GPT2CFG.TXT", "GPT2WT.BIN", "GPT2FX.BIN", "GPT2EXP.BIN", "PROFILE.TXT"):
        require((model_dir / name).exists(), f"missing_pack_model_{name}={pack_id}")


def verify_pack_files(pack_root: Path) -> list[PackInfo]:
    ids = pack_ids(pack_root)
    require(len(ids) >= 3, "pack_count_lt_3")
    require(ids[0] == "CHAT", "chat_pack_not_default")
    for expected in ("CHAT", "DOSHELP", "OFFICE"):
        require(expected in ids, f"missing_pack={expected}")
    packs: list[PackInfo] = []
    for pack_id in ids:
        pack_dir = pack_root / pack_id
        ini = read(pack_dir / "PACK.INI")
        values = parse_ini(ini)
        help_text = read(pack_dir / "HELP.TXT")
        require(f"ID={pack_id}" in ini.upper(), f"pack_id_mismatch={pack_id}")
        for key in ("TITLE=", "MODEL=", "PERSONA=", "HELP=", "USAGE=", "SPRITE=", "ICONS=", "ACTIONS="):
            require(key in ini.upper(), f"missing_{key.rstrip('=')}={pack_id}")
        require("|" in help_text, f"missing_retrieval_rows={pack_id}")
        usage_text = read(pack_dir / values["USAGE"])
        for marker in ("Purpose:", "How it works:", "How to use it:", "Good prompts:", "Actions:"):
            require(marker in usage_text, f"missing_usage_marker_{marker.rstrip(':').lower().replace(' ', '_')}={pack_id}")
        verify_pack_model(pack_root, pack_id, values["MODEL"])
        packs.append(PackInfo(pack_id, values["MODEL"], pack_dir, pack_dir / values["USAGE"]))
    return packs


def verify_source() -> None:
    source = read(ROOT / "src" / "assistant.bas")
    for needle in (
        "ASSIST_PACK_LIST",
        "AssistLoadPackList",
        "AssistInitializeModel",
        "GPT2BasicLoadModel",
        "ASSIST_PACK",
        "ASSIST_MODEL",
        "ASSIST_REPLY",
        "USAGE",
        "AssistPrintPackUsage",
        'LCASE$(command_text) = "/about"',
        "SPRITE",
        "ICONS",
    ):
        require(needle in source, f"source_missing={needle}")


def verify_pack_quality(packs: list[PackInfo], evidence_dir: Path) -> None:
    pack_by_id = {pack.pack_id: pack for pack in packs}
    for pack_id in ("DOSHELP", "OFFICE"):
        pack = pack_by_id[pack_id]
        report = read(evidence_dir / f"quality_report_assistant_{pack.pack_id.lower()}.md")
        require("Quality status: `PASS`" in report, f"pack_quality_not_pass={pack.pack_id}")
        require("Prompt pass rate: `4/4`" in report, f"pack_quality_pass_rate={pack.pack_id}")


def verify_qemu_logs(packs: list[PackInfo], assistant_log: Path, compile_log: Path) -> None:
    assist = read(assistant_log)
    compile_text = read(compile_log)
    require("ASSIST_COMPILE_OK" in compile_text, "compile_marker_missing")
    require("ASSIST_BEGIN|suite=pack-shell|version=1" in assist, "begin_marker_missing")
    require("ASSIST_END|packs=" in assist, "end_marker_missing")
    for pack in packs:
        require(f"ASSIST_PACK|id={pack.pack_id}" in assist, f"pack_log_missing={pack.pack_id}")
        require(f"ASSIST_MODEL|pack={pack.pack_id}" in assist, f"model_log_missing={pack.pack_id}")
        require(
            f"path={pack.model_path}" in assist,
            f"model_path_log_missing={pack.pack_id}",
        )
    require("CHAT pack" in assist, "chat_usage_missing")
    require("DOSHELP pack" in assist, "doshelp_usage_missing")
    require("OFFICE pack" in assist, "office_usage_missing")
    require("ASSIST_REPLY|pack=CHAT|intent=general_chat" in assist, "chat_reply_missing")
    require("ASSIST_REPLY|pack=DOSHELP|intent=dos_memory" in assist, "doshelp_reply_missing")
    require("ASSIST_REPLY|pack=OFFICE|intent=office_rewrite" in assist, "office_reply_missing")
    require("status=model_unavailable" not in assist, "model_unavailable_in_assistant_log")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack-root", type=Path, default=DEFAULT_PACK_ROOT)
    parser.add_argument("--assistant-log", type=Path, default=DEFAULT_ASSISTANT_LOG)
    parser.add_argument("--compile-log", type=Path, default=DEFAULT_COMPILE_LOG)
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE_DIR)
    args = parser.parse_args()

    packs = verify_pack_files(args.pack_root)
    verify_source()
    verify_pack_quality(packs, args.evidence_dir)
    verify_qemu_logs(packs, args.assistant_log, args.compile_log)
    print(f"PROBE_OK assistant_pack_count={len(packs)}")
    print("PROBE_OK assistant_pack_loader=1")
    print("PROBE_OK assistant_pack_models=1")
    print("PROBE_OK assistant_pack_quality=1")
    print("PROBE_OK assistant_model_switch=1")
    print("PROBE_OK assistant_structured_reply=1")
    print("PROBE_OK assistant_art_slots=1")
    print("PROBE_OK assistant_qemu_evidence=1")


if __name__ == "__main__":
    main()
