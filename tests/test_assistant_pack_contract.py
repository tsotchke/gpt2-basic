from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from assistant_pack_contract import (  # noqa: E402
    PackContractError,
    load_all_pack_contracts,
    parse_actions,
    parse_ini_text,
    parse_pack_ids,
    resolve_pack_value,
)


class AssistantPackContractTests(unittest.TestCase):
    def test_shipped_packs_satisfy_shared_contract(self) -> None:
        packs = load_all_pack_contracts(ROOT / "assets" / "gpt2_basic" / "PACKS")
        by_id = {pack.pack_id: pack for pack in packs}

        self.assertEqual([pack.pack_id for pack in packs], ["CHAT", "DOSHELP", "OFFICE", "DEV"])
        self.assertEqual(by_id["CHAT"].model_value, r"PACKS\CHAT\MODEL")
        self.assertIn("cancel", by_id["CHAT"].actions)
        self.assertGreaterEqual(len(by_id["CHAT"].help_rows), 3)
        self.assertGreaterEqual(len(by_id["CHAT"].knowledge_rows), 10)
        self.assertGreaterEqual(len(by_id["CHAT"].kdb_rows), len(by_id["CHAT"].help_rows))
        self.assertGreaterEqual(len(by_id["CHAT"].kdb_index_rows), 3)
        self.assertTrue(by_id["CHAT"].knowledge_path.name.endswith("KNOW.TXT"))
        self.assertTrue(by_id["CHAT"].kdb_path.name.endswith("KDB.TXT"))
        self.assertTrue(by_id["CHAT"].kdb_index_path.name.endswith("KDBIDX.TXT"))
        self.assertTrue(by_id["CHAT"].user_path.name.endswith("USER.TXT"))
        self.assertTrue(by_id["CHAT"].usage_path.name.endswith("USAGE.TXT"))
        self.assertTrue(by_id["CHAT"].sprite_path.name.endswith("CHAT.SPR"))
        self.assertTrue(by_id["CHAT"].icons_path.name.endswith("CHAT.ICN"))

    def test_ini_parser_matches_dos_case_insensitive_keys(self) -> None:
        values = parse_ini_text(
            """
            ; comment
            title = Demo
            MODEL=PACKS\\DEMO\\MODEL
            Help = HELP.TXT
            """
        )

        self.assertEqual(values["TITLE"], "Demo")
        self.assertEqual(values["MODEL"], r"PACKS\DEMO\MODEL")
        self.assertEqual(values["HELP"], "HELP.TXT")

    def test_pack_list_ignores_comments_and_requires_unique_8_3_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "PACKS.TXT").write_text("# comment\nCHAT\n; skip\nDOSHELP\n", encoding="ascii")
            self.assertEqual(parse_pack_ids(root), ("CHAT", "DOSHELP"))

            (root / "PACKS.TXT").write_text("CHAT\nCHAT\n", encoding="ascii")
            with self.assertRaises(PackContractError):
                parse_pack_ids(root)

            (root / "PACKS.TXT").write_text("LONGPACKID\n", encoding="ascii")
            with self.assertRaises(PackContractError):
                parse_pack_ids(root)

    def test_path_resolution_preserves_pack_local_and_pack_root_model_paths(self) -> None:
        pack_root = ROOT / "assets" / "gpt2_basic" / "PACKS"
        pack_dir = pack_root / "CHAT"

        self.assertEqual(resolve_pack_value(pack_root, pack_dir, "HELP.TXT"), pack_dir / "HELP.TXT")
        self.assertEqual(
            resolve_pack_value(pack_root, pack_dir, r"PACKS\CHAT\MODEL"),
            ROOT / "assets" / "gpt2_basic" / "PACKS" / "CHAT" / "MODEL",
        )
        with self.assertRaises(PackContractError):
            resolve_pack_value(pack_root, pack_dir, r"..\SECRET.TXT")

    def test_actions_are_lowercase_shell_tokens_and_include_cancel(self) -> None:
        self.assertEqual(parse_actions("chat, ask,idea,cancel"), ("chat", "ask", "idea", "cancel"))
        with self.assertRaises(PackContractError):
            parse_actions("chat,open-window,cancel")
        with self.assertRaises(PackContractError):
            parse_actions("chat,more")


if __name__ == "__main__":
    unittest.main()
