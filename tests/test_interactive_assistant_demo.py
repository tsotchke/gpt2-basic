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

    def test_assistant_streams_generated_tokens_without_prompt_echo(self) -> None:
        text = (ROOT / "src" / "assistant.bas").read_text(encoding="ascii")

        self.assertIn("FUNCTION AssistStreamGenerate", text)
        self.assertIn("SUB AssistPrepareGenerationPrompt", text)
        self.assertIn("SUB AssistPrefillPrompt", text)
        self.assertIn("FUNCTION AssistCleanGeneratedText", text)
        self.assertIn("GPT2BasicPrefillToken", text)
        self.assertIn("PRINT \"Thinking: \";", text)
        self.assertIn("PRINT \"Answer: \";", text)
        self.assertIn("PRINT \".\";", text)
        self.assertIn("PRINT CHR$(8); \" \"; CHR$(8);", text)
        self.assertIn("PRINT piece;", text)
        self.assertIn("Decode(generated_tokens(), generated_count)", text)
        self.assertIn("bubble = generated", text)
        self.assertIn("IF input_tokens(input_count - 1) = 0 THEN input_count = input_count - 1", text)
        self.assertIn("keep_count = context_limit - reserve_count", text)
        self.assertIn('INSTR(raw_text, ". ")', text)
        self.assertIn('INSTR(lower_text, " user:")', text)
        self.assertNotIn('INSTR(lower_text, " prompt")', text)
        self.assertNotIn('INSTR(lower_text, " assistant")', text)
        self.assertNotIn("MID$(decoded_text, LEN(prompt) + 1)", text)

    def test_assistant_retrieval_prefers_specific_pack_rows(self) -> None:
        text = (ROOT / "src" / "assistant.bas").read_text(encoding="ascii")

        self.assertIn("best_score", text)
        self.assertIn("row_score = LEN(key_text)", text)
        self.assertIn("IF best_text <> \"\" THEN RETURN best_text", text)

    def test_chat_pack_is_first_and_has_usage_instructions(self) -> None:
        pack_list = (ROOT / "assets" / "gpt2_basic" / "PACKS" / "PACKS.TXT").read_text(encoding="ascii")
        first_pack = next(line.strip() for line in pack_list.splitlines() if line.strip() and not line.startswith("#"))
        self.assertEqual(first_pack, "CHAT")

        chat_ini = (ROOT / "assets" / "gpt2_basic" / "PACKS" / "CHAT" / "PACK.INI").read_text(encoding="ascii")
        chat_usage = (ROOT / "assets" / "gpt2_basic" / "PACKS" / "CHAT" / "USAGE.TXT").read_text(encoding="ascii")
        self.assertIn("TITLE=Conversation Pack", chat_ini)
        self.assertIn("MODEL=PACKS\\CHAT\\MODEL", chat_ini)
        self.assertIn("USAGE=USAGE.TXT", chat_ini)
        self.assertIn("Purpose:", chat_usage)
        self.assertIn("How it works:", chat_usage)
        self.assertIn("How to use it:", chat_usage)


if __name__ == "__main__":
    unittest.main()
