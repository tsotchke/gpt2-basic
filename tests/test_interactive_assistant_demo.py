from __future__ import annotations

from pathlib import Path
import subprocess
import sys
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
        self.assertIn("/u, /d, /h", text)

    def test_interactive_launcher_uses_windowed_qemu_display(self) -> None:
        text = (ROOT / "qemu" / "run_assistant_interactive_486.sh").read_text(encoding="ascii")

        self.assertIn("fdauto_assist_interactive.bat", text)
        self.assertIn('-display "$QEMU_DISPLAY"', text)
        self.assertIn('QEMU_DISPLAY="cocoa"', text)
        self.assertIn("/u, /d, /h", text)
        self.assertNotIn("-display curses", text)
        self.assertNotIn("--get ASSIST.LOG", text)

    def test_scripted_qemu_logs_are_extracted_as_normalized_text(self) -> None:
        for script in (
            "compile_main_486.sh",
            "run_assistant_486.sh",
            "run_hardware_capture_486.sh",
            "run_main_486.sh",
            "run_perf_486.sh",
            "run_quality_486.sh",
            "run_sampling_486.sh",
            "run_trace_486.sh",
            "run_vectors_486.sh",
            "run_visual_trace_486.sh",
        ):
            text = (ROOT / "qemu" / script).read_text(encoding="ascii")
            self.assertNotIn("--get ", text.replace("--get GPT2.EXE", ""))
            self.assertIn("--get-text", text)

    def test_assistant_source_has_transcript_paging_commands(self) -> None:
        text = (ROOT / "src" / "assistant.bas").read_text(encoding="ascii")

        self.assertIn("ASSIST_HISTORY_MAX", text)
        self.assertIn("SUB AssistRenderHistory()", text)
        self.assertIn('LCASE$(command_text) = "/up"', text)
        self.assertIn('LCASE$(command_text) = "/down"', text)
        self.assertIn('LCASE$(command_text) = "/history"', text)
        self.assertIn('LCASE$(command_text) = "/u"', text)
        self.assertIn('LCASE$(command_text) = "/d"', text)
        self.assertIn('LCASE$(command_text) = "/h"', text)
        self.assertIn('LCASE$(command_text) = "/about"', text)
        self.assertIn('LCASE$(command_text) = "/capabilities"', text)
        self.assertIn('LCASE$(command_text) = "/limits"', text)
        self.assertIn('LCASE$(command_text) = "/sources"', text)
        self.assertIn('LCASE$(command_text) = "/memory"', text)
        self.assertIn('LCASE$(command_text) = "/forget"', text)
        self.assertIn('LEFT$(LCASE$(command_text), 10) = "/remember "', text)
        self.assertIn("ASSIST_MEMORY_FILE", text)
        self.assertIn("AssistLoadMemoryFacts", text)
        self.assertIn("AssistSaveMemoryFacts", text)
        self.assertIn("AssistPrintCapabilities", text)
        self.assertIn("AssistPrintLimits", text)
        self.assertIn("AssistPrintSources", text)

    def test_interactive_assistant_preloads_active_pack_before_prompt(self) -> None:
        text = (ROOT / "src" / "assistant.bas").read_text(encoding="ascii")

        self.assertIn("SUB AssistPreloadActivePackModel()", text)
        self.assertIn('PRINT "Loading "; pack_id; " model before prompt..."', text)
        self.assertIn("AssistInitializeModel(g_assist_active_pack)", text)

        commands_pos = text.index('PRINT "Commands: /about')
        preload_pos = text.index("AssistPreloadActivePackModel", commands_pos)
        prompt_pos = text.index('PRINT "> ";', commands_pos)
        self.assertLess(preload_pos, prompt_pos)

        pack_branch_pos = text.index('ELSEIF LEFT$(LCASE$(command_text), 6) = "/pack "')
        pack_preload_pos = text.index("AssistPreloadActivePackModel", pack_branch_pos)
        unknown_pack_pos = text.index('PRINT "Unknown pack."', pack_branch_pos)
        self.assertLess(pack_preload_pos, unknown_pack_pos)

    def test_assistant_streams_generated_tokens_without_prompt_echo(self) -> None:
        text = (ROOT / "src" / "assistant.bas").read_text(encoding="ascii")

        self.assertIn("FUNCTION AssistStreamGenerate", text)
        self.assertIn("SUB AssistPrepareGenerationPrompt", text)
        self.assertIn("SUB AssistPrefillPrompt", text)
        self.assertIn("FUNCTION AssistVisibleToken", text)
        self.assertIn("FUNCTION AssistCleanGeneratedText", text)
        self.assertIn("GPT2BasicPrefillToken", text)
        self.assertIn("PRINT \"Thinking: \";", text)
        self.assertIn('PRINT "  ctx"; i + 1; ": ";', text)
        self.assertIn("PRINT AssistVisibleToken(input_tokens(i));", text)
        self.assertIn('PRINT "Thinking: sampling output tokens"', text)
        self.assertIn('progress_text = "<t" + LTRIM$(STR$(i + 1)) + ">"', text)
        self.assertIn("PRINT \"Answer: \";", text)
        self.assertIn("FOR erase_idx = 1 TO LEN(progress_text)", text)
        self.assertIn('PRINT CHR$(8); " "; CHR$(8);', text)
        self.assertIn("PRINT piece;", text)
        self.assertIn("Decode(generated_tokens(), generated_count)", text)
        self.assertIn("bubble = generated", text)
        self.assertIn("IF input_tokens(input_count - 1) = 0 THEN input_count = input_count - 1", text)
        self.assertIn("keep_count = context_limit - reserve_count", text)
        self.assertIn("CONST ASSIST_MAX_REPLY_TOKENS = 64", text)
        self.assertIn("CONST ASSIST_SENTENCE_STOP_MIN_TOKENS = 10", text)
        self.assertIn('INSTR(lower_text, " user:")', text)
        self.assertNotIn('INSTR(lower_text, " prompt")', text)
        self.assertNotIn('INSTR(lower_text, " assistant")', text)
        self.assertNotIn("MID$(decoded_text, LEN(prompt) + 1)", text)

    def test_interactive_assistant_guards_bad_model_output(self) -> None:
        text = (ROOT / "src" / "assistant.bas").read_text(encoding="ascii")

        self.assertIn("FUNCTION AssistGeneratedLooksBad", text)
        self.assertIn("FUNCTION AssistTextHasRepeatedChunk", text)
        self.assertIn("FUNCTION AssistFallbackReply", text)
        self.assertIn("FUNCTION AssistGoldenReply", text)
        self.assertIn("SUB AssistGuardProbe()", text)
        self.assertIn('command_line = "--guard-probe"', text)
        self.assertIn('INSTR(lower_text, "use two brief sentences")', text)
        self.assertIn('INSTR(lower_text, "use to brief sentences")', text)
        self.assertIn('INSTR(lower_text, "small friendly dos chat assistant")', text)
        self.assertIn("AssistGeneratedLooksBad(generated, query) = 0", text)
        self.assertIn('reply_source = "golden"', text)
        self.assertIn('reply_source = "retrieval"', text)
        self.assertIn('reply_source = "memory"', text)
        self.assertIn('reply_source = "model"', text)
        self.assertIn('reply_source = "fallback"', text)
        self.assertIn("FUNCTION AssistCanonicalQuery", text)
        self.assertIn('prompt = "User: " + canonical_query', text)
        self.assertIn('IF memory_context <> "" THEN prompt = memory_context + " " + prompt', text)
        self.assertNotIn('prompt = AssistTrimFixed(g_assist_packs(pack_index).persona) + " User: " + query', text)

        guard_pos = text.index("AssistGeneratedLooksBad(generated, query) = 0")
        print_pos = text.index('PRINT "Answer: "; bubble', guard_pos)
        self.assertLess(guard_pos, print_pos)

    def test_interactive_assistant_has_stress_probe(self) -> None:
        text = (ROOT / "src" / "assistant.bas").read_text(encoding="ascii")

        self.assertIn("SUB AssistStressProbe()", text)
        self.assertIn('command_line = "--stress-probe"', text)
        self.assertIn("ASSIST_BEGIN|suite=stress-probe|version=1", text)
        self.assertIn("why did my answer repeat itself", text)
        self.assertIn("make a tiny plan for fixing a bug", text)
        self.assertIn("can you browse the internet from dos", text)
        self.assertIn("i feel lonely", text)
        self.assertIn("my name is Tyr", text)
        self.assertIn("what is my name", text)
        self.assertIn("what are we working on", text)
        self.assertIn("how should you answer me", text)
        self.assertIn("what did i just ask", text)
        self.assertIn("what do you remember", text)
        self.assertIn("why does protected mode need a dpmi host", text)
        self.assertIn("how should i clean autoexec.bat", text)
        self.assertIn("summarize this: tests passed but the tag was stale", text)
        self.assertIn("make this clearer: the artifact uploaded but the tag was stale", text)
        self.assertIn('AssistSelectPack "DEV"', text)
        self.assertIn("how can this feel modern on a 486", text)
        self.assertIn("what does retrieval first mean", text)
        self.assertIn("how do i author a pack", text)
        self.assertIn('"|query=" + AssistSafeText(query)', text)
        self.assertIn('"|canonical=" + AssistSafeText(canonical_query)', text)
        self.assertIn('"|memory=" + AssistSafeText(memory_context)', text)
        self.assertIn('"|answer=" + AssistSafeText(bubble)', text)

    def test_assistant_has_session_memory_commands_and_reply_source(self) -> None:
        text = (ROOT / "src" / "assistant.bas").read_text(encoding="ascii")

        self.assertIn("FUNCTION AssistMemoryReply", text)
        self.assertIn("FUNCTION AssistMemoryContext", text)
        self.assertIn("SUB AssistLoadMemoryFacts", text)
        self.assertIn("SUB AssistSaveMemoryFacts", text)
        self.assertIn("SUB AssistRememberFact", text)
        self.assertIn("SUB AssistRememberTurn", text)
        self.assertIn("SUB AssistClearMemoryFacts", text)
        self.assertIn("my name is ", text)
        self.assertIn("what are we working on", text)
        self.assertIn("what did i just ask", text)
        self.assertIn('reply_source = "memory"', text)
        self.assertIn("/remember KEY=VALUE", text)
        self.assertIn("Memory persists in ", text)

    def test_assistant_stress_script_rejects_bad_visible_text(self) -> None:
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "stress_assistant_behavior.py"), "--self-test"],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("PROBE_OK assistant_stress_self_test=1", result.stdout)

    def test_qemu_assistant_stress_launcher_is_headless(self) -> None:
        launcher = (ROOT / "qemu" / "run_assistant_stress_486.sh").read_text(encoding="ascii")
        batch = (ROOT / "qemu" / "fdauto_assist_stress.bat").read_text(encoding="ascii")

        self.assertIn("-display curses", launcher)
        self.assertIn("fdauto_assist_stress.bat", launcher)
        self.assertIn("--get-text ASTRESS.LOG", launcher)
        self.assertIn("QEMU_TIMEOUT_SECONDS", launcher)
        self.assertIn("QEMU stress timeout reached", launcher)
        self.assertIn("scripts/stress_assistant_behavior.py", launcher)
        self.assertNotIn("-display cocoa", launcher)
        self.assertIn("ASSIST.EXE --stress-probe", batch)

    def test_tokenizer_has_bucketed_lexicon_path(self) -> None:
        text = (ROOT / "src" / "tokenizer.bas").read_text(encoding="ascii")

        self.assertIn("lexicon_bucket_start", text)
        self.assertIn("lexicon_bucket_count", text)
        self.assertIn("lexicon_length_present", text)
        self.assertIn("lexicon_order", text)
        self.assertIn("SUB TokenizerBuildLexiconBuckets", text)
        self.assertIn("FindToken(tokenizer, candidate)", text)
        self.assertIn("FOR piece_len = max_len TO 1 STEP -1", text)

    def test_assistant_retrieval_prefers_specific_pack_rows(self) -> None:
        text = (ROOT / "src" / "assistant.bas").read_text(encoding="ascii")

        self.assertIn("best_score", text)
        self.assertIn("AssistRetrievalScore", text)
        self.assertIn("AssistScanBucketedKdbV2 kdb_bin_path", text)
        self.assertIn("AssistScanBinaryKdbFile kdb_bin_path", text)
        self.assertIn("AssistScanBucketedKdb kdb_path", text)
        self.assertIn("AssistKdbBucketPath", text)
        self.assertIn("AssistScanRetrievalFile kdb_path", text)
        self.assertIn("AssistScanRetrievalFile user_path", text)
        self.assertIn("score_bonus", text)
        self.assertIn("IF best_score >= 8 THEN RETURN best_text", text)

    def test_chat_pack_is_first_and_has_usage_instructions(self) -> None:
        pack_list = (ROOT / "assets" / "gpt2_basic" / "PACKS" / "PACKS.TXT").read_text(encoding="ascii")
        first_pack = next(line.strip() for line in pack_list.splitlines() if line.strip() and not line.startswith("#"))
        self.assertEqual(first_pack, "CHAT")

        chat_ini = (ROOT / "assets" / "gpt2_basic" / "PACKS" / "CHAT" / "PACK.INI").read_text(encoding="ascii")
        chat_usage = (ROOT / "assets" / "gpt2_basic" / "PACKS" / "CHAT" / "USAGE.TXT").read_text(encoding="ascii")
        self.assertIn("TITLE=Conversation Pack", chat_ini)
        self.assertIn("MODEL=PACKS\\CHAT\\MODEL", chat_ini)
        self.assertIn("KDB=KDB.TXT", chat_ini)
        self.assertIn("KDBIDX=KDBIDX.TXT", chat_ini)
        self.assertIn("KDBBIN=KB2ALL.BIN", chat_ini)
        self.assertIn("KDBBIDX=KB2IDX.TXT", chat_ini)
        self.assertIn("USER=USER.TXT", chat_ini)
        self.assertIn("USAGE=USAGE.TXT", chat_ini)
        self.assertIn("Purpose:", chat_usage)
        self.assertIn("How it works:", chat_usage)
        self.assertIn("How to use it:", chat_usage)


if __name__ == "__main__":
    unittest.main()
