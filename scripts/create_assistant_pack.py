#!/usr/bin/env python3
"""Create a complete GPT2-BASIC assistant pack from a folder of notes."""

from __future__ import annotations

import argparse
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from assistant_pack_contract import (
    KdbBinaryIndexRow,
    KdbIndexRow,
    HelpRow,
    PACK_ID_RE,
    PackContract,
    PackContractError,
    REQUIRED_MODEL_FILES,
    load_pack_contract,
    parse_actions,
    resolve_pack_value,
)
from build_assistant_kdb import (
    DEFAULT_PACK_ROOT,
    KDB2_FULL_NAME,
    KDB2_INDEX_NAME,
    render_bucket_files,
    render_index,
    render_kdb,
    render_kdb2_bucket_files,
    render_kdb2_full,
    render_kdb2_index,
    render_kdb2_term_index_files,
)
from import_assistant_notes import ImportedRow, rows_from_file


NOTE_SUFFIXES = {".txt", ".md", ".note", ".notes"}
DEFAULT_ACTIONS = "ask,explain,search,cancel"
DEFAULT_MODEL = r"PACKS\CHAT\MODEL"
DEFAULT_SPRITE = r"PACKS\CHAT\CHAT.SPR"
DEFAULT_ICONS = r"PACKS\CHAT\CHAT.ICN"


@dataclass(frozen=True)
class PackPlan:
    pack_id: str
    pack_dir: Path
    title: str
    persona: str
    model: str
    sprite: str
    icons: str
    actions: tuple[str, ...]
    help_rows: tuple[HelpRow, ...]
    knowledge_rows: tuple[HelpRow, ...]
    note_files: tuple[Path, ...]


def clean_field(value: str, limit: int) -> str:
    cleaned = " ".join(value.replace("|", "/").split()).strip()
    return cleaned[:limit].strip()


def default_title(pack_id: str) -> str:
    return " ".join(part.capitalize() for part in pack_id.replace("_", " ").split()) + " Pack"


def collect_note_files(notes_dir: Path) -> tuple[Path, ...]:
    if not notes_dir.is_dir():
        raise PackContractError(f"notes directory is missing: {notes_dir}")
    files = tuple(
        sorted(
            path
            for path in notes_dir.rglob("*")
            if path.is_file() and not path.name.startswith(".") and path.suffix.lower() in NOTE_SUFFIXES
        )
    )
    if not files:
        raise PackContractError(f"notes directory contains no note files: {notes_dir}")
    return files


def load_note_rows(note_files: tuple[Path, ...]) -> tuple[HelpRow, ...]:
    imported: list[ImportedRow] = []
    for path in note_files:
        imported.extend(rows_from_file(path))
    rows: list[HelpRow] = []
    seen: set[tuple[str, str, str]] = set()
    for row in imported:
        key = clean_field(row.key, 48)
        title = clean_field(row.title, 48)
        body = clean_field(row.body, 180)
        identity = (key.lower(), title.lower(), body.lower())
        if not key or not title or not body or identity in seen:
            continue
        rows.append(HelpRow(key, title, body))
        seen.add(identity)
    if not rows:
        raise PackContractError("notes produced no assistant rows")
    return tuple(rows)


def build_help_rows(pack_id: str, title: str) -> tuple[HelpRow, ...]:
    title_short = clean_field(title, 36)
    key_base = clean_field(title.lower(), 32) or pack_id.lower()
    return (
        HelpRow(
            f"about {key_base}",
            f"{title_short} overview",
            clean_field(f"{title} answers from local notes, USER.TXT, and generated KB2 recall.", 180),
        ),
        HelpRow(
            f"use {key_base}",
            f"Using {title_short}",
            "Ask one short question. The pack searches local recall first, then gives a concise answer.",
        ),
        HelpRow(
            f"limits {key_base}",
            f"{title_short} limits",
            "Answers are offline and local. Add machine-specific facts to USER.TXT.",
        ),
    )


def render_rows(rows: tuple[HelpRow, ...], header: str) -> str:
    lines = ["# key|title|body", header]
    lines.extend(f"{row.key}|{row.title}|{row.text}" for row in rows)
    return "\n".join(lines) + "\n"


def render_pack_ini(plan: PackPlan) -> str:
    return "\n".join(
        [
            f"ID={plan.pack_id}",
            f"TITLE={plan.title}",
            f"MODEL={plan.model}",
            f"PERSONA={plan.persona}",
            "HELP=HELP.TXT",
            "KNOW=KNOW.TXT",
            "KDB=KDB.TXT",
            "KDBIDX=KDBIDX.TXT",
            f"KDBBIN={KDB2_FULL_NAME}",
            f"KDBBIDX={KDB2_INDEX_NAME}",
            "USER=USER.TXT",
            "USAGE=USAGE.TXT",
            f"SPRITE={plan.sprite}",
            f"ICONS={plan.icons}",
            f"ACTIONS={','.join(plan.actions)}",
            "",
        ]
    )


def render_user_file() -> str:
    return "\n".join(
        [
            "# key|title|body",
            "# Machine-local notes for this pack. Keep rows ASCII and concise.",
            "",
        ]
    )


def render_usage(plan: PackPlan) -> str:
    examples = [row.key for row in plan.knowledge_rows[:3]]
    while len(examples) < 3:
        examples.append(f"ask about {plan.title.lower()}")
    return "\n".join(
        [
            f"{plan.pack_id} pack",
            "",
            "Purpose:",
            f"  Use this pack for {plan.title.lower()} questions backed by local notes.",
            "",
            "How it works:",
            "  The pack shares a small model by default, then retrieves concise local",
            "  rows from KB2TERM.TXT and KB2T?.TXT term indexes, compiled KB2*.BIN pages, generated",
            "  KDB.TXT buckets, bundled KNOW.TXT notes, and editable USER.TXT notes.",
            "",
            "How to use it:",
            f"  Type /pack {plan.pack_id}, then ask one short question about the notes.",
            "  Edit USER.TXT for machine-local overrides, or rebuild the pack from",
            "  source notes when bundled knowledge changes.",
            "",
            "Good prompts:",
            *(f"  {example}" for example in examples),
            "",
            "Actions:",
            f"  {', '.join(plan.actions)}",
            "",
        ]
    )


def pack_contract_for_generation(plan: PackPlan) -> PackContract:
    pack_dir = plan.pack_dir
    return PackContract(
        pack_id=plan.pack_id,
        title=plan.title,
        model_value=plan.model,
        persona=plan.persona,
        help_path=pack_dir / "HELP.TXT",
        knowledge_path=pack_dir / "KNOW.TXT",
        kdb_path=pack_dir / "KDB.TXT",
        kdb_index_path=pack_dir / "KDBIDX.TXT",
        kdb_bin_path=pack_dir / KDB2_FULL_NAME,
        kdb_bin_index_path=pack_dir / KDB2_INDEX_NAME,
        user_path=pack_dir / "USER.TXT",
        usage_path=pack_dir / "USAGE.TXT",
        sprite_path=resolve_pack_value(pack_dir.parent, pack_dir, plan.sprite),
        icons_path=resolve_pack_value(pack_dir.parent, pack_dir, plan.icons),
        actions=plan.actions,
        help_rows=plan.help_rows,
        knowledge_rows=plan.knowledge_rows,
        kdb_rows=tuple(),
        kdb_index_rows=tuple(),
        kdb_bin_index_rows=tuple(),
        ini_values={},
    )


def write_generated_kdb(plan: PackPlan) -> None:
    pack = pack_contract_for_generation(plan)
    (plan.pack_dir / "KDB.TXT").write_text(render_kdb(pack), encoding="ascii")
    (plan.pack_dir / "KDBIDX.TXT").write_text(render_index(pack), encoding="ascii")
    for name, text in render_bucket_files(pack).items():
        (plan.pack_dir / name).write_text(text, encoding="ascii")
    (plan.pack_dir / KDB2_FULL_NAME).write_bytes(render_kdb2_full(pack))
    (plan.pack_dir / KDB2_INDEX_NAME).write_text(render_kdb2_index(pack), encoding="ascii")
    for name, payload in render_kdb2_bucket_files(pack).items():
        (plan.pack_dir / name).write_bytes(payload)
    for name, text in render_kdb2_term_index_files(pack).items():
        (plan.pack_dir / name).write_text(text, encoding="ascii")


def update_pack_list(pack_root: Path, pack_id: str) -> bool:
    list_path = pack_root / "PACKS.TXT"
    existing = list_path.read_text(encoding="ascii") if list_path.exists() else ""
    lines = [line.strip().upper() for line in existing.splitlines() if line.strip() and not line.strip().startswith(("#", ";"))]
    if pack_id in lines:
        return False
    with list_path.open("a", encoding="ascii") as handle:
        if existing and not existing.endswith("\n"):
            handle.write("\n")
        handle.write(f"{pack_id}\n")
    return True


def build_plan(args: argparse.Namespace) -> PackPlan:
    pack_id = args.pack.upper()
    if not PACK_ID_RE.fullmatch(pack_id):
        raise PackContractError(f"pack id must be 1-8 uppercase DOS-safe characters: {args.pack!r}")
    title = clean_field(args.title or default_title(pack_id), 48)
    persona = clean_field(args.persona or f"You are a concise {title.lower()} assistant.", 120)
    note_files = collect_note_files(args.notes_dir)
    knowledge_rows = load_note_rows(note_files)
    actions = parse_actions(args.actions)
    return PackPlan(
        pack_id=pack_id,
        pack_dir=args.pack_root / pack_id,
        title=title,
        persona=persona,
        model=args.model,
        sprite=args.sprite,
        icons=args.icons,
        actions=actions,
        help_rows=build_help_rows(pack_id, title),
        knowledge_rows=knowledge_rows,
        note_files=note_files,
    )


def write_pack(plan: PackPlan, pack_root: Path, force: bool, register: bool) -> bool:
    if plan.pack_dir.exists():
        if not force:
            raise PackContractError(f"pack already exists, use --force to replace it: {plan.pack_dir}")
        shutil.rmtree(plan.pack_dir)
    plan.pack_dir.mkdir(parents=True)
    (plan.pack_dir / "PACK.INI").write_text(render_pack_ini(plan), encoding="ascii")
    (plan.pack_dir / "HELP.TXT").write_text(
        render_rows(plan.help_rows, "# Generated help rows from scripts/create_assistant_pack.py."),
        encoding="ascii",
    )
    (plan.pack_dir / "KNOW.TXT").write_text(
        render_rows(plan.knowledge_rows, "# Generated knowledge rows from source notes."),
        encoding="ascii",
    )
    (plan.pack_dir / "USER.TXT").write_text(render_user_file(), encoding="ascii")
    (plan.pack_dir / "USAGE.TXT").write_text(render_usage(plan), encoding="ascii")
    write_generated_kdb(plan)
    registered = update_pack_list(pack_root, plan.pack_id) if register else False
    load_pack_contract(pack_root, plan.pack_id)
    return registered


def create_pack(args: argparse.Namespace) -> int:
    plan = build_plan(args)
    print(
        "ASSISTANT_PACK_CREATE|"
        f"pack={plan.pack_id}|title={plan.title}|notes={len(plan.note_files)}|"
        f"know_rows={len(plan.knowledge_rows)}|write={int(args.write)}|register={int(args.register)}"
    )
    if not args.write:
        for row in plan.knowledge_rows:
            print(f"{row.key}|{row.title}|{row.text}")
        print("PROBE_OK assistant_pack_create_dry_run=1")
        return 0
    registered = write_pack(plan, args.pack_root, args.force, args.register)
    print(
        "ASSISTANT_PACK_CREATED|"
        f"pack={plan.pack_id}|path={plan.pack_dir}|registered={int(registered)}|"
        f"rows={len(plan.help_rows) + len(plan.knowledge_rows)}"
    )
    print("PROBE_OK assistant_pack_create=1")
    return 0


def create_minimal_shared_assets(pack_root: Path) -> None:
    chat = pack_root / "CHAT"
    model = chat / "MODEL"
    model.mkdir(parents=True)
    for name in REQUIRED_MODEL_FILES:
        (model / name).write_bytes(b"probe\n")
    (chat / "CHAT.SPR").write_text("probe sprite\n", encoding="ascii")
    (chat / "CHAT.ICN").write_text("probe icon\n", encoding="ascii")
    (pack_root / "PACKS.TXT").write_text("CHAT\n", encoding="ascii")


def self_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        pack_root = root / "PACKS"
        notes = root / "notes"
        notes.mkdir(parents=True)
        create_minimal_shared_assets(pack_root)
        (notes / "hardware repair.txt").write_text(
            "Check the power supply, reseat cards, and record the first failing step.\n\n"
            "Keep a boot disk and note which CONFIG.SYS change fixed the machine.",
            encoding="ascii",
        )
        args = argparse.Namespace(
            pack_root=pack_root,
            pack="HWREPAIR",
            notes_dir=notes,
            title="Hardware Repair",
            persona=None,
            actions=DEFAULT_ACTIONS,
            model=DEFAULT_MODEL,
            sprite=DEFAULT_SPRITE,
            icons=DEFAULT_ICONS,
            write=True,
            register=True,
            force=False,
        )
        assert create_pack(args) == 0
        pack = load_pack_contract(pack_root, "HWREPAIR")
        assert pack.model_value == DEFAULT_MODEL
        assert len(pack.knowledge_rows) == 2
        assert (pack.kdb_bin_path.parent / "KB2TERM.TXT").is_file()
        assert any(path.name.startswith("KB2T") and path.name != "KB2TERM.TXT" for path in pack.kdb_bin_path.parent.glob("KB2T*.TXT"))
        assert "HWREPAIR" in (pack_root / "PACKS.TXT").read_text(encoding="ascii")
    print("PROBE_OK assistant_pack_create_self_test=1")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack-root", type=Path, default=DEFAULT_PACK_ROOT)
    parser.add_argument("--pack", required=False)
    parser.add_argument("--notes-dir", type=Path)
    parser.add_argument("--title")
    parser.add_argument("--persona")
    parser.add_argument("--actions", default=DEFAULT_ACTIONS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--sprite", default=DEFAULT_SPRITE)
    parser.add_argument("--icons", default=DEFAULT_ICONS)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test()
            return
        if not args.pack or args.notes_dir is None:
            raise PackContractError("--pack and --notes-dir are required")
        raise SystemExit(create_pack(args))
    except (PackContractError, UnicodeDecodeError) as exc:
        raise SystemExit(f"ASSISTANT_PACK_CREATE_FAILED {exc}") from exc


if __name__ == "__main__":
    main()
