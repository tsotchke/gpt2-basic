#!/usr/bin/env python3
"""Validate editable assistant packs for release and local authoring."""

from __future__ import annotations

import argparse
from pathlib import Path

from assistant_pack_contract import PackContractError, parse_help_rows, load_all_pack_contracts
from build_assistant_kdb import DEFAULT_PACK_ROOT, render_bucket_files, render_index, render_kdb


def validate_user_file(path: Path) -> int:
    text = path.read_text(encoding="ascii")
    rows = 0
    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith(";"):
            continue
        parts = [part.strip() for part in line.split("|", 2)]
        if len(parts) != 3 or not all(parts):
            raise PackContractError(f"invalid USER.TXT row {path}:{line_no}: {raw!r}")
        if len(parts[0]) > 48:
            raise PackContractError(f"USER.TXT key too long {path}:{line_no}")
        if len(parts[1]) > 48:
            raise PackContractError(f"USER.TXT title too long {path}:{line_no}")
        if len(parts[2]) > 220:
            raise PackContractError(f"USER.TXT body too long {path}:{line_no}")
        rows += 1
    return rows


def validate_pack_authoring(pack_root: Path) -> None:
    packs = load_all_pack_contracts(pack_root)
    for pack in packs:
        expected_kdb = render_kdb(pack)
        current_kdb = pack.kdb_path.read_text(encoding="ascii")
        if current_kdb != expected_kdb:
            raise PackContractError(f"KDB.TXT is stale for {pack.pack_id}; run scripts/build_assistant_kdb.py --write")
        expected_index = render_index(pack)
        current_index = pack.kdb_index_path.read_text(encoding="ascii")
        if current_index != expected_index:
            raise PackContractError(f"KDBIDX.TXT is stale for {pack.pack_id}; run scripts/build_assistant_kdb.py --write")
        expected_buckets = render_bucket_files(pack)
        for name, expected_text in expected_buckets.items():
            bucket_path = pack.kdb_path.parent / name
            if not bucket_path.exists() or bucket_path.read_text(encoding="ascii") != expected_text:
                raise PackContractError(f"{name} is stale for {pack.pack_id}; run scripts/build_assistant_kdb.py --write")
        parse_help_rows(pack.help_path)
        parse_help_rows(pack.knowledge_path)
        parse_help_rows(pack.kdb_path)
        user_rows = validate_user_file(pack.user_path)
        print(
            "ASSISTANT_PACK_AUTHORING|"
            f"pack={pack.pack_id}|"
            f"help={len(pack.help_rows)}|"
            f"know={len(pack.knowledge_rows)}|"
            f"kdb={len(pack.kdb_rows)}|"
            f"idx={len(pack.kdb_index_rows)}|"
            f"bucket_entries={sum(len(text.splitlines()) - 2 for text in expected_buckets.values())}|"
            f"user={user_rows}"
        )
    print("PROBE_OK assistant_pack_authoring=1")


def self_test() -> None:
    validate_pack_authoring(DEFAULT_PACK_ROOT)
    print("PROBE_OK assistant_pack_authoring_self_test=1")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack-root", type=Path, default=DEFAULT_PACK_ROOT)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test()
        else:
            validate_pack_authoring(args.pack_root)
    except PackContractError as exc:
        raise SystemExit(f"ASSISTANT_PACK_AUTHORING_FAILED {exc}") from exc


if __name__ == "__main__":
    main()
