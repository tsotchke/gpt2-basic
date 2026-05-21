#!/usr/bin/env python3
"""Shared assistant-pack metadata contract for DOS, Windows, and OS/2 shells.

The DOS runtime remains the source compatibility target. This host module
defines the subset of PACKS.TXT and PACK.INI behavior that every future shell
must preserve before it is treated as release scope.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACK_ROOT = ROOT / "assets" / "gpt2_basic" / "PACKS"

PACK_ID_RE = re.compile(r"^[A-Z0-9_]{1,8}$")
ACTION_RE = re.compile(r"^[a-z0-9_]{1,24}$")

REQUIRED_INI_KEYS = (
    "ID",
    "TITLE",
    "MODEL",
    "PERSONA",
    "HELP",
    "KNOW",
    "KDB",
    "KDBIDX",
    "KDBBIN",
    "KDBBIDX",
    "USER",
    "USAGE",
    "SPRITE",
    "ICONS",
    "ACTIONS",
)

KDB2_HEADER_BYTES = 64
KDB2_RECORD_BYTES = 160 + 64 + 192

REQUIRED_MODEL_FILES = (
    "GPT2CFG.TXT",
    "GPT2WT.BIN",
    "GPT2FX.BIN",
    "GPT2EXP.BIN",
    "PROFILE.TXT",
    "VOCAB.BIN",
)

REQUIRED_USAGE_MARKERS = (
    "Purpose:",
    "How it works:",
    "How to use it:",
    "Good prompts:",
    "Actions:",
)


class PackContractError(ValueError):
    """Raised when a pack violates the shared shell contract."""


@dataclass(frozen=True)
class HelpRow:
    key: str
    title: str
    text: str


@dataclass(frozen=True)
class KdbIndexRow:
    bucket: str
    entries: int
    filename: str


@dataclass(frozen=True)
class KdbBinaryIndexRow:
    bucket: str
    entries: int
    filename: str
    byte_count: int


@dataclass(frozen=True)
class PackContract:
    pack_id: str
    title: str
    model_value: str
    persona: str
    help_path: Path
    knowledge_path: Path
    kdb_path: Path
    kdb_index_path: Path
    kdb_bin_path: Path
    kdb_bin_index_path: Path
    user_path: Path
    usage_path: Path
    sprite_path: Path
    icons_path: Path
    actions: tuple[str, ...]
    help_rows: tuple[HelpRow, ...]
    knowledge_rows: tuple[HelpRow, ...]
    kdb_rows: tuple[HelpRow, ...]
    kdb_index_rows: tuple[KdbIndexRow, ...]
    kdb_bin_index_rows: tuple[KdbBinaryIndexRow, ...]
    ini_values: dict[str, str]


def read_ascii(path: Path) -> str:
    try:
        return path.read_text(encoding="ascii")
    except FileNotFoundError as exc:
        raise PackContractError(f"missing file: {path}") from exc
    except UnicodeDecodeError as exc:
        raise PackContractError(f"non-ascii file: {path}") from exc


def parse_ini_text(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith(";"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().upper()
        value = value.strip()
        if not key:
            continue
        values[key] = value
    return values


def parse_pack_ids(pack_root: Path) -> tuple[str, ...]:
    ids: list[str] = []
    for raw in read_ascii(pack_root / "PACKS.TXT").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith(";"):
            continue
        pack_id = line.upper()
        if not PACK_ID_RE.fullmatch(pack_id):
            raise PackContractError(f"invalid pack id in PACKS.TXT: {line!r}")
        ids.append(pack_id)
    if not ids:
        raise PackContractError("PACKS.TXT contains no packs")
    if len(set(ids)) != len(ids):
        raise PackContractError("PACKS.TXT contains duplicate pack ids")
    return tuple(ids)


def _contains_path_escape(value: str) -> bool:
    return any(part == ".." for part in value.replace("\\", "/").split("/"))


def resolve_pack_value(pack_root: Path, pack_dir: Path, value: str, default_name: str = "") -> Path:
    value = value.strip() or default_name
    if not value:
        raise PackContractError("empty path value")
    if _contains_path_escape(value):
        raise PackContractError(f"path traversal is not allowed: {value!r}")
    normalized = value.replace("\\", "/")
    if normalized.upper().startswith("PACKS/"):
        return pack_root.parent / normalized
    if "/" in normalized:
        return pack_dir / normalized
    return pack_dir / value


def parse_actions(text: str) -> tuple[str, ...]:
    actions = tuple(action.strip().lower() for action in text.split(",") if action.strip())
    if not actions:
        raise PackContractError("ACTIONS must list at least one action")
    for action in actions:
        if not ACTION_RE.fullmatch(action):
            raise PackContractError(f"invalid action name: {action!r}")
    if "cancel" not in actions:
        raise PackContractError("ACTIONS must include cancel")
    return actions


def parse_help_rows(path: Path) -> tuple[HelpRow, ...]:
    rows: list[HelpRow] = []
    for raw in read_ascii(path).splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith(";"):
            continue
        parts = [part.strip() for part in line.split("|", 2)]
        if len(parts) != 3 or not all(parts):
            raise PackContractError(f"invalid HELP.TXT row in {path}: {raw!r}")
        rows.append(HelpRow(parts[0], parts[1], parts[2]))
    if not rows:
        raise PackContractError(f"HELP.TXT has no retrieval rows: {path}")
    return tuple(rows)


def parse_kdb_index_rows(path: Path) -> tuple[KdbIndexRow, ...]:
    rows: list[KdbIndexRow] = []
    for raw in read_ascii(path).splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith(";"):
            continue
        parts = [part.strip() for part in line.split("|", 2)]
        if len(parts) != 3 or not all(parts):
            raise PackContractError(f"invalid KDBIDX.TXT row in {path}: {raw!r}")
        bucket, entries_text, filename = parts
        if len(bucket) != 1 or not re.fullmatch(r"[0-9A-Z]", bucket):
            raise PackContractError(f"invalid KDBIDX bucket in {path}: {raw!r}")
        if not entries_text.isdigit() or int(entries_text) <= 0:
            raise PackContractError(f"invalid KDBIDX entry count in {path}: {raw!r}")
        if not re.fullmatch(r"KDB[0-9A-Z]\.TXT", filename.upper()):
            raise PackContractError(f"invalid KDBIDX filename in {path}: {raw!r}")
        rows.append(KdbIndexRow(bucket, int(entries_text), filename.upper()))
    if not rows:
        raise PackContractError(f"KDBIDX.TXT has no bucket rows: {path}")
    return tuple(rows)


def parse_kdb_bin_index_rows(path: Path) -> tuple[KdbBinaryIndexRow, ...]:
    rows: list[KdbBinaryIndexRow] = []
    for raw in read_ascii(path).splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith(";"):
            continue
        parts = [part.strip() for part in line.split("|", 3)]
        if len(parts) != 4 or not all(parts):
            raise PackContractError(f"invalid KB2IDX.TXT row in {path}: {raw!r}")
        bucket, entries_text, filename, bytes_text = parts
        if len(bucket) != 1 or not re.fullmatch(r"[0-9A-Z]", bucket):
            raise PackContractError(f"invalid KB2IDX bucket in {path}: {raw!r}")
        if not entries_text.isdigit() or int(entries_text) <= 0:
            raise PackContractError(f"invalid KB2IDX entry count in {path}: {raw!r}")
        if not re.fullmatch(r"KB2[0-9A-Z]\.BIN", filename.upper()):
            raise PackContractError(f"invalid KB2IDX filename in {path}: {raw!r}")
        if not bytes_text.isdigit() or int(bytes_text) <= KDB2_HEADER_BYTES:
            raise PackContractError(f"invalid KB2IDX byte count in {path}: {raw!r}")
        rows.append(KdbBinaryIndexRow(bucket, int(entries_text), filename.upper(), int(bytes_text)))
    if not rows:
        raise PackContractError(f"KB2IDX.TXT has no bucket rows: {path}")
    return tuple(rows)


def validate_kdb2_binary(path: Path, expected_rows: int | None = None) -> None:
    try:
        payload = path.read_bytes()
    except FileNotFoundError as exc:
        raise PackContractError(f"missing KDB2 binary: {path}") from exc
    if len(payload) < KDB2_HEADER_BYTES:
        raise PackContractError(f"KDB2 binary too short: {path}")
    if not payload.startswith(b"KDB2|V=1|"):
        raise PackContractError(f"KDB2 binary missing header: {path}")
    body_bytes = len(payload) - KDB2_HEADER_BYTES
    if body_bytes % KDB2_RECORD_BYTES != 0:
        raise PackContractError(f"KDB2 binary has partial record: {path}")
    row_count = body_bytes // KDB2_RECORD_BYTES
    if expected_rows is not None and row_count != expected_rows:
        raise PackContractError(f"KDB2 binary row count mismatch: {path}")


def validate_usage_file(path: Path) -> None:
    text = read_ascii(path)
    for marker in REQUIRED_USAGE_MARKERS:
        if marker not in text:
            raise PackContractError(f"USAGE.TXT missing marker {marker!r}: {path}")


def validate_model_dir(path: Path) -> None:
    for name in REQUIRED_MODEL_FILES:
        if not (path / name).is_file():
            raise PackContractError(f"model directory missing {name}: {path}")


def load_pack_contract(pack_root: Path, pack_id: str) -> PackContract:
    pack_id = pack_id.upper()
    if not PACK_ID_RE.fullmatch(pack_id):
        raise PackContractError(f"invalid pack id: {pack_id!r}")
    pack_dir = pack_root / pack_id
    ini_path = pack_dir / "PACK.INI"
    values = parse_ini_text(read_ascii(ini_path))
    for key in REQUIRED_INI_KEYS:
        if key not in values or not values[key].strip():
            raise PackContractError(f"{pack_id} missing PACK.INI key {key}")
    if values["ID"].upper() != pack_id:
        raise PackContractError(f"{pack_id} PACK.INI ID mismatch: {values['ID']!r}")

    model_path = resolve_pack_value(pack_root, pack_dir, values["MODEL"])
    help_path = resolve_pack_value(pack_root, pack_dir, values["HELP"], "HELP.TXT")
    knowledge_path = resolve_pack_value(pack_root, pack_dir, values["KNOW"], "KNOW.TXT")
    kdb_path = resolve_pack_value(pack_root, pack_dir, values["KDB"], "KDB.TXT")
    kdb_index_path = resolve_pack_value(pack_root, pack_dir, values["KDBIDX"], "KDBIDX.TXT")
    kdb_bin_path = resolve_pack_value(pack_root, pack_dir, values["KDBBIN"], "KB2ALL.BIN")
    kdb_bin_index_path = resolve_pack_value(pack_root, pack_dir, values["KDBBIDX"], "KB2IDX.TXT")
    user_path = resolve_pack_value(pack_root, pack_dir, values["USER"], "USER.TXT")
    usage_path = resolve_pack_value(pack_root, pack_dir, values["USAGE"], "USAGE.TXT")
    sprite_path = resolve_pack_value(pack_root, pack_dir, values["SPRITE"])
    icons_path = resolve_pack_value(pack_root, pack_dir, values["ICONS"])

    validate_model_dir(model_path)
    help_rows = parse_help_rows(help_path)
    knowledge_rows = parse_help_rows(knowledge_path)
    kdb_rows = parse_help_rows(kdb_path)
    kdb_index_rows = parse_kdb_index_rows(kdb_index_path) if kdb_index_path.exists() else tuple()
    kdb_bin_index_rows = parse_kdb_bin_index_rows(kdb_bin_index_path) if kdb_bin_index_path.exists() else tuple()
    if kdb_bin_path.exists():
        validate_kdb2_binary(kdb_bin_path, len(kdb_rows))
    for row in kdb_bin_index_rows:
        bucket_path = kdb_bin_path.parent / row.filename
        if bucket_path.exists():
            validate_kdb2_binary(bucket_path, row.entries)
    validate_usage_file(usage_path)
    if not user_path.is_file():
        raise PackContractError(f"user note template missing: {user_path}")
    for artifact in (sprite_path, icons_path):
        if not artifact.is_file():
            raise PackContractError(f"pack art asset missing: {artifact}")

    return PackContract(
        pack_id=pack_id,
        title=values["TITLE"],
        model_value=values["MODEL"],
        persona=values["PERSONA"],
        help_path=help_path,
        knowledge_path=knowledge_path,
        kdb_path=kdb_path,
        kdb_index_path=kdb_index_path,
        kdb_bin_path=kdb_bin_path,
        kdb_bin_index_path=kdb_bin_index_path,
        user_path=user_path,
        usage_path=usage_path,
        sprite_path=sprite_path,
        icons_path=icons_path,
        actions=parse_actions(values["ACTIONS"]),
        help_rows=help_rows,
        knowledge_rows=knowledge_rows,
        kdb_rows=kdb_rows,
        kdb_index_rows=kdb_index_rows,
        kdb_bin_index_rows=kdb_bin_index_rows,
        ini_values=values,
    )


def load_all_pack_contracts(pack_root: Path = DEFAULT_PACK_ROOT) -> tuple[PackContract, ...]:
    ids = parse_pack_ids(pack_root)
    packs = tuple(load_pack_contract(pack_root, pack_id) for pack_id in ids)
    if packs[0].pack_id != "CHAT":
        raise PackContractError("CHAT must remain the default pack")
    return packs


def self_test() -> None:
    packs = load_all_pack_contracts(DEFAULT_PACK_ROOT)
    by_id = {pack.pack_id: pack for pack in packs}
    for expected in ("CHAT", "DOSHELP", "OFFICE", "DEV", "PORTABLE"):
        if expected not in by_id:
            raise PackContractError(f"missing expected pack: {expected}")
    chat = by_id["CHAT"]
    if chat.model_value != r"PACKS\CHAT\MODEL":
        raise PackContractError("CHAT model path is not pack-local")
    if len(chat.help_rows) < 3:
        raise PackContractError("CHAT help rows are unexpectedly sparse")
    if len(chat.knowledge_rows) < 3:
        raise PackContractError("CHAT knowledge rows are unexpectedly sparse")
    if len(chat.kdb_rows) < len(chat.help_rows):
        raise PackContractError("CHAT KDB rows are unexpectedly sparse")
    if len(chat.kdb_index_rows) < 3:
        raise PackContractError("CHAT KDB index rows are unexpectedly sparse")
    if len(chat.kdb_bin_index_rows) < 3:
        raise PackContractError("CHAT KDB2 index rows are unexpectedly sparse")
    print(f"PROBE_OK assistant_pack_contract_count={len(packs)}")
    print("PROBE_OK assistant_pack_contract_parser=1")
    print("PROBE_OK assistant_pack_contract_artifacts=1")
    print("PROBE_OK assistant_pack_contract_usage=1")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack-root", type=Path, default=DEFAULT_PACK_ROOT)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    packs = load_all_pack_contracts(args.pack_root)
    for pack in packs:
        print(
            "ASSISTANT_PACK_CONTRACT|"
            f"id={pack.pack_id}|"
            f"title={pack.title}|"
            f"model={pack.model_value}|"
            f"actions={','.join(pack.actions)}|"
            f"help_rows={len(pack.help_rows)}|"
            f"knowledge_rows={len(pack.knowledge_rows)}|"
            f"kdb_rows={len(pack.kdb_rows)}"
        )
    if args.self_test:
        self_test()


if __name__ == "__main__":
    main()
