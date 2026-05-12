from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class InteractiveAssistantDemoTests(unittest.TestCase):
    def test_interactive_batch_starts_real_assistant_session(self) -> None:
        text = (ROOT / "qemu" / "fdauto_assist_interactive.bat").read_text(encoding="ascii")

        self.assertIn("ASSIST.EXE", text)
        self.assertNotIn("ASSIST.EXE --scripted", text)
        self.assertNotIn("> ASSIST.LOG", text)
        self.assertNotIn("FDAPM.COM POWEROFF", text)
        self.assertIn("/about", text)
        self.assertIn("/pack CHAT", text)
        self.assertIn("/up, /down, /history", text)

    def test_interactive_launcher_uses_windowed_qemu_display(self) -> None:
        text = (ROOT / "qemu" / "run_assistant_interactive_486.sh").read_text(encoding="ascii")

        self.assertIn("fdauto_assist_interactive.bat", text)
        self.assertIn('-display "$QEMU_DISPLAY"', text)
        self.assertIn('QEMU_DISPLAY="cocoa"', text)
        self.assertNotIn("-display curses", text)
        self.assertNotIn("--get ASSIST.LOG", text)

    def test_assistant_source_has_transcript_paging_commands(self) -> None:
        text = (ROOT / "src" / "assistant.bas").read_text(encoding="ascii")

        self.assertIn("ASSIST_HISTORY_MAX", text)
        self.assertIn("SUB AssistRenderHistory()", text)
        self.assertIn('LCASE$(command_text) = "/up"', text)
        self.assertIn('LCASE$(command_text) = "/down"', text)
        self.assertIn('LCASE$(command_text) = "/history"', text)
        self.assertIn('LCASE$(command_text) = "/about"', text)

    def test_chat_pack_is_first_and_has_usage_instructions(self) -> None:
        pack_list = (ROOT / "assets" / "gpt2_basic" / "PACKS" / "PACKS.TXT").read_text(encoding="ascii")
        first_pack = next(line.strip() for line in pack_list.splitlines() if line.strip() and not line.startswith("#"))
        self.assertEqual(first_pack, "CHAT")

        chat_ini = (ROOT / "assets" / "gpt2_basic" / "PACKS" / "CHAT" / "PACK.INI").read_text(encoding="ascii")
        chat_usage = (ROOT / "assets" / "gpt2_basic" / "PACKS" / "CHAT" / "USAGE.TXT").read_text(encoding="ascii")
        self.assertIn("TITLE=Conversation Pack", chat_ini)
        self.assertIn("MODEL=MODEL", chat_ini)
        self.assertIn("USAGE=USAGE.TXT", chat_ini)
        self.assertIn("Purpose:", chat_usage)
        self.assertIn("How it works:", chat_usage)
        self.assertIn("How to use it:", chat_usage)


if __name__ == "__main__":
    unittest.main()
